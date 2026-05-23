"""JAX-native rollout driver for end-to-end GPU PPO.

`collect_rollout_jax(model, key, episodes_per_iter, opponent='noop',
horizon=500, seed=0)` resets N JAX env instances in parallel via vmap,
then runs `horizon` steps of (featurize → policy forward → sample
action → env.step), accumulating per-step transitions for PPO.

Phase W4 spike scope:
  - Opponent: `noop` only (always empty actions). Real `baseline_v1`
    integration is a follow-up (it requires either porting the
    rule-based agent to JAX or paying the device↔host round-trip per
    step, defeating the purpose).
  - PPO update is still PyTorch (W4-c). The rollout returns JAX arrays;
    the train.py glue copies them to torch and runs the existing
    `ppo_update`.
  - Reward: ±1 terminal win/loss + shaping = `shaping_coef * Δ(my_ships
    − enemy_ships)` per turn, matching `env.OrbitWarsEpisode` so PPO
    sees the same advantage signal.

Returns a `JaxRolloutBatch` containing all per-step JAX arrays plus the
episode outcomes for logging.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from jax_env.reset import reset
from jax_env.state import EnvState
from jax_env.step import MAX_LAUNCHES_PER_AGENT
from jax_env.step import step as jax_env_step
from pipeline.rulebase.case1.baseline_jax import (
    compute_actions_jax as _baseline_jax_actions,
)

from ..policy.featurizer_jax import (
    HistoryStateJax,
    featurize_jax_w1,
    init_history_jax,
    update_history_jax,
)
from ..policy.model_jax import ActorCriticJax
from ..policy.sampling_jax import (
    sample_action_jax,
    sampled_action_to_env_actions,
)

NUM_AGENTS_MAX = 4  # jax_env's NUM_AGENTS_MAX

# opponent_mode constants — kept int to stay scan/vmap friendly.
OPPONENT_NOOP: int = 0
OPPONENT_BASELINE_JAX_LITE: int = 1

OPPONENT_NAME_TO_MODE: dict[str, int] = {
    "noop": OPPONENT_NOOP,
    "baseline_jax_lite": OPPONENT_BASELINE_JAX_LITE,
}


class JaxRolloutBatch(NamedTuple):
    """Per-step transitions collected by a (vmap'd) JAX rollout.

    Leading axes are (B, T) where B = episodes_per_iter and T = horizon.
    Episodes that terminate before T continue to advance the env but
    `done_mask` marks the steps that should be ignored by the consumer.
    """

    planet_feats: jax.Array  # (B, T, MAX_PLANETS, PLANET_FEAT_DIM) f32
    planet_mask: jax.Array  # (B, T, MAX_PLANETS) bool
    my_planet_mask: jax.Array  # (B, T, MAX_PLANETS) bool
    target_mask: jax.Array  # (B, T, MAX_PLANETS) bool
    global_feats: jax.Array  # (B, T, GLOBAL_FEAT_DIM) f32
    template_ctx: jax.Array  # (B, T, MAX_PLANETS, TEMPLATE_CTX_DIM) f32
    candidate_feats: jax.Array  # (B, T, MAX_PLANETS, CAND_K, CAND_FEAT_DIM)
    candidate_mask: jax.Array  # (B, T, MAX_PLANETS, CAND_K) bool
    candidate_pid: jax.Array  # (B, T, MAX_PLANETS, CAND_K) int32
    target_slot: jax.Array  # (B, T, MAX_PLANETS) int32
    log1p_ships: jax.Array  # (B, T, MAX_PLANETS) f32
    log_probs: jax.Array  # (B, T) f32
    values: jax.Array  # (B, T) f32
    rewards: jax.Array  # (B, T) f32
    done_mask: jax.Array  # (B, T) bool — step counted iff valid (pre-terminal)
    episode_outcomes: jax.Array  # (B,) f32 — final reward sign per ep


def _empty_jax_actions() -> jax.Array:
    """Action tensor for a single env where no agent fires."""
    return jnp.full(
        (NUM_AGENTS_MAX, MAX_LAUNCHES_PER_AGENT, 3), -1.0, dtype=jnp.float32
    )


def _ship_totals(state: EnvState, player: int) -> tuple[jax.Array, jax.Array]:
    """Sum (own, enemy) ships across the current planet table."""
    owner = state.planet_owner
    valid = state.planet_valid
    ships = state.planet_ships.astype(jnp.float32)
    is_mine = valid & (owner == player)
    is_enemy = valid & (owner != player) & (owner != -1)
    return jnp.sum(jnp.where(is_mine, ships, 0.0)), jnp.sum(
        jnp.where(is_enemy, ships, 0.0)
    )


def _planet_count_totals(state: EnvState, player: int) -> tuple[jax.Array, jax.Array]:
    """Count (own, enemy) planets in the current planet table.

    Alternative shaping signal: ship totals grow with production so the
    diff is dominated by "who has more rotation" rather than territorial
    control. Planet count diff is a cleaner proxy of strategic state —
    flips only when a planet changes hands.
    """
    owner = state.planet_owner
    valid = state.planet_valid
    is_mine = valid & (owner == player)
    is_enemy = valid & (owner != player) & (owner != -1)
    return jnp.sum(is_mine.astype(jnp.float32)), jnp.sum(is_enemy.astype(jnp.float32))


SHAPING_MODE_SHIPS: int = 0
SHAPING_MODE_PLANETS: int = 1
SHAPING_MODE_NAME_TO_INT: dict[str, int] = {
    "ships": SHAPING_MODE_SHIPS,
    "planets": SHAPING_MODE_PLANETS,
}


def _shaping_diff(state: EnvState, seat: int, shaping_mode: int) -> jax.Array:
    """Return Δ(mine - enemy) for the configured shaping signal."""
    ship_my, ship_en = _ship_totals(state, seat)
    plt_my, plt_en = _planet_count_totals(state, seat)
    # jax.lax.cond keeps the trace single.
    out: jax.Array = jax.lax.cond(
        shaping_mode == SHAPING_MODE_PLANETS,
        lambda: plt_my - plt_en,
        lambda: ship_my - ship_en,
    )
    return out


def _rollout_one_env(
    model: ActorCriticJax,
    key: jax.Array,
    init_state: EnvState,
    horizon: int,
    shaping_coef: float,
    seat: int,
    opponent_mode: int,
    shaping_mode: int,
) -> JaxRolloutBatch:
    """Single-env rollout via `lax.scan`.

    `init_state` is a JAX `EnvState` that the caller already built (via
    `reset(seed=...)` on host). Threading it as a JAX array — instead
    of taking `seed: int` and calling `reset` inside the body — keeps
    the jit cache to ONE trace regardless of how many distinct seeds
    we run. The earlier `seed: int` arg caused per-seed recompilation
    and ate >3 GB of cache on RunPod (root cause of the
    `bench_rollout_gpu` SIGABRT during the 16-ep variant).
    """
    init_history = init_history_jax()

    # Initial shaping diff (ship-based or planet-count-based per shaping_mode).
    init_diff = _shaping_diff(init_state, seat, shaping_mode)

    def step_fn(
        carry: tuple[
            EnvState,
            HistoryStateJax,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
        ],
        _t: jax.Array,
    ) -> tuple[
        tuple[
            EnvState,
            HistoryStateJax,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
        ],
        tuple[
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
        ],
    ]:
        state, history, prev_diff, done_already, key, ep_outcome = carry
        key, k_sample = jax.random.split(key)
        batch = featurize_jax_w1(state, player=seat, history=history)
        output = model(batch)
        action = sample_action_jax(output, batch.my_planet_mask, k_sample)

        # Build env action for seat=0 (we control seat 0); opponent is
        # noop so other rows stay -1.
        target_slot_2d = action.target_slot[0]  # (P,) since batch=1
        log1p_ships_2d = action.log1p_ships[0]
        my_planet_mask_2d = batch.my_planet_mask[0]
        env_actions = sampled_action_to_env_actions(
            target_slot_2d,
            log1p_ships_2d,
            state.planet_xy,
            state.planet_ships,
            state.planet_id,
            state.planet_valid,
            my_planet_mask_2d,
            seat,
            NUM_AGENTS_MAX,
        )

        # Opponent row (seat 1-seat). For noop we leave the -1 sentinels
        # already written by sampled_action_to_env_actions. For
        # baseline_jax_lite we compute its (L, 3) row and splice it in.
        opp_seat = jnp.int32(1) - jnp.int32(seat)
        opp_baseline_actions = _baseline_jax_actions(state, 1 - seat)  # (L, 3)
        env_actions = jax.lax.cond(
            opponent_mode == OPPONENT_BASELINE_JAX_LITE,
            lambda ea: ea.at[opp_seat].set(opp_baseline_actions),
            lambda ea: ea,
            env_actions,
        )

        # Step env. Either advance, or hold if already done (so shapes
        # stay constant during scan).
        new_state, _r, term = jax_env_step(state, env_actions)
        # If we're already past terminal, freeze.
        state_for_next = jax.tree.map(
            lambda new, old: jnp.where(done_already, old, new),
            new_state,
            state,
        )

        # Shaping reward: shaping_coef * Δ(my − enemy) using either ship
        # totals (legacy) or planet count (F: cleaner territorial signal).
        diff = _shaping_diff(state_for_next, seat, shaping_mode)
        shaping = shaping_coef * (diff - prev_diff)

        # Terminal reward: env emits rewards on termination; turn into
        # ±1/0 by comparing the env's rewards for both seats.
        rewards_arr = state_for_next.rewards
        r_self = rewards_arr[seat]
        r_opp = rewards_arr[1 - seat]
        terminal_reward = jnp.where(
            term, jnp.sign(r_self - r_opp).astype(jnp.float32), jnp.float32(0.0)
        )
        step_reward = jnp.where(
            done_already, jnp.float32(0.0), shaping + terminal_reward
        )
        new_done = done_already | term
        # Outcome: cumulative reward (= terminal sign + shaping, just
        # like the torch path).
        new_outcome = ep_outcome + step_reward

        # Update history AFTER featurize, mirroring the torch loop's
        # `update_history(history, obs, actions=actions_list)` call at the
        # end of the step. The history pytree needs the planet snapshot
        # at the current state and any launch events.
        # For the W4 spike, we pass empty actions to history (no launches
        # tracked); a follow-up will track real launches.
        empty_pid = jnp.full((4,), -1, dtype=jnp.int32)
        empty_ships = jnp.zeros((4,), dtype=jnp.int32)
        empty_valid = jnp.zeros((4,), dtype=jnp.bool_)
        new_history = update_history_jax(
            history,
            state,
            empty_pid,
            empty_ships,
            empty_valid,
            player=seat,
        )

        # Pack per-step transition.
        per_step = (
            batch.planet_feats[0],
            batch.planet_mask[0],
            batch.my_planet_mask[0],
            batch.target_mask[0],
            batch.global_feats[0],
            batch.template_ctx[0],
            batch.candidate_feats[0],
            batch.candidate_mask[0],
            batch.candidate_pid[0],
            action.target_slot[0],
            action.log1p_ships[0],
            action.log_prob[0],
            output.value[0],
            step_reward,
            ~done_already,  # done_mask = step is valid (pre-terminal)
        )
        new_carry = (
            state_for_next,
            new_history,
            diff,
            new_done,
            key,
            new_outcome,
        )
        return new_carry, per_step

    init_carry = (
        init_state,
        init_history,
        init_diff,
        jnp.bool_(False),
        key,
        jnp.float32(0.0),
    )
    final_carry, traj = jax.lax.scan(
        step_fn, init_carry, jnp.arange(horizon, dtype=jnp.int32)
    )
    (
        planet_feats_t,
        planet_mask_t,
        my_planet_mask_t,
        target_mask_t,
        global_feats_t,
        template_ctx_t,
        candidate_feats_t,
        candidate_mask_t,
        candidate_pid_t,
        target_slot_t,
        log1p_ships_t,
        log_probs_t,
        values_t,
        rewards_t,
        done_mask_t,
    ) = traj
    ep_outcome = final_carry[5]
    return JaxRolloutBatch(
        planet_feats=planet_feats_t,
        planet_mask=planet_mask_t,
        my_planet_mask=my_planet_mask_t,
        target_mask=target_mask_t,
        global_feats=global_feats_t,
        template_ctx=template_ctx_t,
        candidate_feats=candidate_feats_t,
        candidate_mask=candidate_mask_t,
        candidate_pid=candidate_pid_t,
        target_slot=target_slot_t,
        log1p_ships=log1p_ships_t,
        log_probs=log_probs_t,
        values=values_t,
        rewards=rewards_t,
        done_mask=done_mask_t,
        episode_outcomes=ep_outcome,
    )


def collect_rollout_jax(
    model: ActorCriticJax,
    key: jax.Array,
    episodes_per_iter: int,
    *,
    horizon: int = 500,
    shaping_coef: float = 0.001,
    seat: int = 0,
    seed: int = 0,
    opponent: str = "noop",
    shaping_mode: str = "ships",
) -> JaxRolloutBatch:
    """Run N parallel single-seat rollouts.

    Per-episode keys are derived from `key` via `jax.random.split`; per-
    episode env seeds are `seed, seed + 1, ..., seed + N - 1`. `opponent`
    selects the rule used for the other seat:
      - "noop": always emit empty actions (default, fastest)
      - "baseline_jax_lite": JAX rule-based agent under
        `pipeline/rulebase/case1/baseline_jax/` (intended PPO curriculum
        target — roughly mirrors the Python `baseline_v1` strategy with
        a single-pass priority score).
    """
    if opponent not in OPPONENT_NAME_TO_MODE:
        raise ValueError(
            f"unknown opponent={opponent!r}; supported: {sorted(OPPONENT_NAME_TO_MODE)}"
        )
    if shaping_mode not in SHAPING_MODE_NAME_TO_INT:
        raise ValueError(
            f"unknown shaping_mode={shaping_mode!r}; "
            f"supported: {sorted(SHAPING_MODE_NAME_TO_INT)}"
        )
    opponent_mode = jnp.int32(OPPONENT_NAME_TO_MODE[opponent])
    shaping_mode_int = jnp.int32(SHAPING_MODE_NAME_TO_INT[shaping_mode])

    keys = jax.random.split(key, episodes_per_iter)
    # Build all N initial states on host (reset is Python-side), then
    # stack into a single batched EnvState so vmap can broadcast model
    # parameters across the episode axis. This keeps the jit cache to
    # ONE compilation regardless of episode count.
    init_states = [reset(seed=seed + i, num_agents=2) for i in range(episodes_per_iter)]
    batched_state = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *init_states)

    # vmap over (key, init_state); model + scalar args broadcast.
    vmapped = jax.vmap(
        _rollout_one_env, in_axes=(None, 0, 0, None, None, None, None, None)
    )
    return vmapped(
        model,
        keys,
        batched_state,
        horizon,
        shaping_coef,
        seat,
        opponent_mode,
        shaping_mode_int,
    )


__all__ = [
    "JaxRolloutBatch",
    "collect_rollout_jax",
]
