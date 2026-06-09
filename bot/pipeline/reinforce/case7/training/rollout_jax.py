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
import numpy as _np
from orbit_wars_jax.reset import reset
from orbit_wars_jax.state import EnvState
from orbit_wars_jax.step import MAX_LAUNCHES_PER_AGENT
from orbit_wars_jax.step import step as jax_env_step

from pipeline.rulebase.case1.baseline_jax import (
    compute_actions_jax as _baseline_jax_actions,
)
from pipeline.rulebase.case1.baseline_jax_full import (
    compute_actions_jax as _baseline_jax_full_actions,
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
OPPONENT_BASELINE_JAX_FULL: int = 2
OPPONENT_SELF_SNAPSHOT: int = 3
OPPONENT_PYTHON_V1: int = 4
OPPONENT_PYTHON_V4: int = 5
OPPONENT_PYTHON_V8: int = 6

OPPONENT_NAME_TO_MODE: dict[str, int] = {
    "noop": OPPONENT_NOOP,
    "baseline_jax_lite": OPPONENT_BASELINE_JAX_LITE,
    "baseline_jax_full": OPPONENT_BASELINE_JAX_FULL,
    "self_snapshot": OPPONENT_SELF_SNAPSHOT,
    # Real Python baseline_v{1,4,8} via pure_callback host roundtrip. Closes the
    # train/eval gap (memory: case6 PFSP was 0/30 vs live v1 because the in-JAX
    # baseline_jax_lite/full approximate but don't match the real rule). Cost:
    # vmap forces sequential per-game host calls so rollout throughput drops.
    # v1=sigmaborov LB897, v4=phase C champion (winrate 70.3%), v8=case8 predict
    # cache + LB1224 strategy.
    "python_v1": OPPONENT_PYTHON_V1,
    "python_v4": OPPONENT_PYTHON_V4,
    "python_v8": OPPONENT_PYTHON_V8,
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


def _production_totals(state: EnvState, player: int) -> tuple[jax.Array, jax.Array]:
    """Sum (own, enemy) planet production across the current planet table.

    Production-weighted territorial signal (H5): instead of counting planets
    equally, weight each by its production so holding a high-prod planet is
    rewarded more than holding a low-prod one.
    """
    owner = state.planet_owner
    valid = state.planet_valid
    prod = state.planet_prod.astype(jnp.float32)
    is_mine = valid & (owner == player)
    is_enemy = valid & (owner != player) & (owner != -1)
    return jnp.sum(jnp.where(is_mine, prod, 0.0)), jnp.sum(
        jnp.where(is_enemy, prod, 0.0)
    )


SHAPING_MODE_SHIPS: int = 0
SHAPING_MODE_PLANETS: int = 1
SHAPING_MODE_COMBINED: int = 2  # H1: ship Δ と planet Δ を個別係数で同時加算
SHAPING_MODE_RATIO: int = 3  # H2: 保持割合 mine/(mine+enemy) の Δ を shaping
SHAPING_MODE_RATIO_PROD: int = 4  # H5: planet 信号を production 加重保持割合に
SHAPING_MODE_NAME_TO_INT: dict[str, int] = {
    "ships": SHAPING_MODE_SHIPS,
    "planets": SHAPING_MODE_PLANETS,
    "combined": SHAPING_MODE_COMBINED,
    "ratio": SHAPING_MODE_RATIO,
    "ratio_prod": SHAPING_MODE_RATIO_PROD,
}

_RATIO_EPS: float = 1e-6


def _shaping_potentials(
    state: EnvState, seat: int, shaping_mode: int
) -> tuple[jax.Array, jax.Array]:
    """Return (Φ_ship, Φ_planet) potentials for the configured mode.

    - ships / planets / combined → diff potential (mine - enemy)
    - ratio (H2)                 → retention ratio mine/(mine+enemy) (planet = count)
    - ratio_prod (H5)            → ship retention ratio + production-weighted
                                   planet ratio prod_mine/(prod_mine+prod_en)

    Both signals are computed unconditionally (cheap) and carried
    independently so each mode keeps two separate potentials → the
    per-turn shaping reward = c_ship·ΔΦ_ship + c_planet·ΔΦ_planet stays
    potential-based (PBRS) for every mode. Legacy modes are bit-identical
    because `_shaping_coefs` zeroes the unused coefficient.
    """
    ship_my, ship_en = _ship_totals(state, seat)
    plt_my, plt_en = _planet_count_totals(state, seat)
    prod_my, prod_en = _production_totals(state, seat)
    diff_ship = ship_my - ship_en
    diff_plt = plt_my - plt_en
    ratio_ship = ship_my / (ship_my + ship_en + _RATIO_EPS)
    ratio_plt = plt_my / (plt_my + plt_en + _RATIO_EPS)
    ratio_prod = prod_my / (prod_my + prod_en + _RATIO_EPS)
    # branch index: modes 0-2 (ships/planets/combined) → 0 (diff),
    # mode 3 (ratio) → 1, mode 4 (ratio_prod) → 2.
    out: tuple[jax.Array, jax.Array] = jax.lax.switch(
        jnp.clip(shaping_mode - (SHAPING_MODE_RATIO - 1), 0, 2),
        [
            lambda: (diff_ship, diff_plt),  # ships/planets/combined → diff
            lambda: (ratio_ship, ratio_plt),  # ratio (count-based planet)
            lambda: (ratio_ship, ratio_prod),  # ratio_prod (production-weighted)
        ],
    )
    return out


def _shaping_coefs(
    shaping_mode: int,
    shaping_coef: float,
    coef_ship: float,
    coef_planet: float,
) -> tuple[jax.Array, jax.Array]:
    """Resolve (c_ship, c_planet) so reward = c_ship·ΔΦ_ship + c_planet·ΔΦ_planet.

    - ships      → (shaping_coef, 0)            (legacy: only ship diff)
    - planets    → (0, shaping_coef)            (legacy: only planet diff)
    - combined   → (coef_ship, coef_planet)     (H1: both diff signals)
    - ratio      → (shaping_coef, shaping_coef) (H2: both ratio signals, equal weight)
    - ratio_prod → (shaping_coef, shaping_coef) (H5: ship ratio + prod-weighted ratio)

    Unifying all modes into one additive form keeps the scan body a single
    trace and makes the legacy modes provably equivalent.
    """
    c_ship = jnp.float32(coef_ship)
    c_planet = jnp.float32(coef_planet)
    c_coef = jnp.float32(shaping_coef)
    zero = jnp.float32(0.0)
    out: tuple[jax.Array, jax.Array] = jax.lax.switch(
        jnp.clip(shaping_mode, 0, 4),
        [
            lambda: (c_coef, zero),  # ships
            lambda: (zero, c_coef),  # planets
            lambda: (c_ship, c_planet),  # combined
            lambda: (c_coef, c_coef),  # ratio
            lambda: (c_coef, c_coef),  # ratio_prod
        ],
    )
    return out


def _self_snapshot_opponent_actions(
    opp_model: ActorCriticJax, state: EnvState, player: int
) -> jax.Array:
    """Deterministic (L, 3) opponent action row from a frozen self snapshot.

    Featurizes the opponent seat with an empty history (the rule opponents
    use no history either, so this keeps the snapshot opponent stateless and
    scan-friendly), runs the frozen `opp_model` forward, and takes the
    argmax target slot + mean ship count (no sampling noise). H1 fixes the
    snapshot at the start of training; pool / sampling is H2+.
    """
    history = init_history_jax()
    batch = featurize_jax_w1(state, player=player, history=history)
    output = opp_model(batch)
    target_slot = jnp.argmax(output.per_planet_logits[0], axis=-1).astype(jnp.int32)
    my_planet_mask = batch.my_planet_mask[0]
    target_slot = jnp.where(my_planet_mask, target_slot, jnp.zeros_like(target_slot))
    log1p_ships = jnp.where(
        my_planet_mask, output.ship_mean[0], jnp.zeros_like(output.ship_mean[0])
    )
    return sampled_action_to_env_actions(
        target_slot,
        log1p_ships,
        state.planet_xy,
        state.planet_ships,
        state.planet_id,
        state.planet_valid,
        my_planet_mask,
        player,
        NUM_AGENTS_MAX,
    )[player]


# ---- Real Python baseline_v1 opponent via host callback. ---------------------
# Lazy import so importing this module on a vendored Kaggle runtime (which has
# no `pipeline.rulebase.case1.baseline`) doesn't crash; the callback only fires
# when the opponent is actually selected.
def _host_python_v1_action(state: EnvState, seat: int) -> "_np.ndarray":
    from orbit_wars_jax.observation import state_to_obs as _state_to_obs

    from pipeline.rulebase.case1.baseline.agent import agent as _v1_agent

    # case1 baseline has no module-level predict cache (it's a port of the
    # sigmaborov LB897 notebook, the predict-position cache is a case8 addition).
    # So nothing to reset here.
    obs = _state_to_obs(state, player=int(seat))
    moves = _v1_agent(obs)
    out = _np.full((MAX_LAUNCHES_PER_AGENT, 3), -1.0, dtype=_np.float32)
    out[:, 1:] = 0.0
    for j, mv in enumerate(moves[:MAX_LAUNCHES_PER_AGENT]):
        out[j] = [float(mv[0]), float(mv[1]), float(mv[2])]
    return out


# Prototype tells pure_callback the result's shape/dtype without an untyped
# `jax.ShapeDtypeStruct` constructor.
_PYTHON_V1_PROTO = jnp.zeros((MAX_LAUNCHES_PER_AGENT, 3), dtype=jnp.float32)


def _python_v1_opponent_actions(state: EnvState, player: int) -> jax.Array:
    """Call the real Python baseline_v1 agent for one seat via host callback.

    `vmap_method='sequential'` so that when the rollout is vmapped over a batch
    of games, each game's seat-1 obs is dispatched to the host one at a time
    (the Python agent is not vectorizable). This is the gold-standard opponent:
    closes the train/eval gap that the in-JAX `baseline_jax_lite/full`
    approximations leave open (case6 was 0/30 vs live v1 with them).
    """
    return jnp.asarray(
        jax.pure_callback(
            _host_python_v1_action,
            _PYTHON_V1_PROTO,
            state,
            player,
            vmap_method="sequential",
        )
    )


# v4 / v8: case4 / case8 baseline. PFSP pool diversity. v8 has predict cache
# (case8 addition) — reset before each oracle call to avoid `id(initial_by_id)`
# collision in tight loops (project_case_jax_phase0 memory).
def _host_python_v4_action(state: EnvState, seat: int) -> "_np.ndarray":
    from orbit_wars_jax.observation import state_to_obs as _state_to_obs

    from pipeline.rulebase.case4.baseline.agent import agent as _v4_agent

    obs = _state_to_obs(state, player=int(seat))
    moves = _v4_agent(obs)
    out = _np.full((MAX_LAUNCHES_PER_AGENT, 3), -1.0, dtype=_np.float32)
    out[:, 1:] = 0.0
    for j, mv in enumerate(moves[:MAX_LAUNCHES_PER_AGENT]):
        out[j] = [float(mv[0]), float(mv[1]), float(mv[2])]
    return out


def _host_python_v8_action(state: EnvState, seat: int) -> "_np.ndarray":
    from orbit_wars_jax.observation import state_to_obs as _state_to_obs

    from pipeline.rulebase.case8.baseline.agent import agent as _v8_agent
    from pipeline.rulebase.case8.baseline.core.physics import (
        reset_predict_cache as _reset_v8_cache,
    )

    _reset_v8_cache()
    obs = _state_to_obs(state, player=int(seat))
    moves = _v8_agent(obs)
    out = _np.full((MAX_LAUNCHES_PER_AGENT, 3), -1.0, dtype=_np.float32)
    out[:, 1:] = 0.0
    for j, mv in enumerate(moves[:MAX_LAUNCHES_PER_AGENT]):
        out[j] = [float(mv[0]), float(mv[1]), float(mv[2])]
    return out


def _python_v4_opponent_actions(state: EnvState, player: int) -> jax.Array:
    return jnp.asarray(
        jax.pure_callback(
            _host_python_v4_action,
            _PYTHON_V1_PROTO,
            state,
            player,
            vmap_method="sequential",
        )
    )


def _python_v8_opponent_actions(state: EnvState, player: int) -> jax.Array:
    return jnp.asarray(
        jax.pure_callback(
            _host_python_v8_action,
            _PYTHON_V1_PROTO,
            state,
            player,
            vmap_method="sequential",
        )
    )


def _rollout_one_env(
    model: ActorCriticJax,
    key: jax.Array,
    init_state: EnvState,
    horizon: int,
    shaping_coef: float,
    seat: int,
    opponent_mode: int,
    shaping_mode: int,
    coef_ship: float,
    coef_planet: float,
    shaping_clip: float,
    dense_coef_ship: float,
    dense_coef_planet: float,
    time_bonus_coef: float,
    time_penalty_coef: float,
    opp_model: ActorCriticJax,
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

    # Initial shaping diffs (ship and planet carried independently so the
    # combined mode keeps two separate potentials → potential-based).
    init_ship_diff, init_plt_diff = _shaping_potentials(init_state, seat, shaping_mode)
    c_ship, c_planet = _shaping_coefs(
        shaping_mode, shaping_coef, coef_ship, coef_planet
    )

    def step_fn(
        carry: tuple[
            EnvState,
            HistoryStateJax,
            jax.Array,
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
        (
            state,
            history,
            prev_ship_diff,
            prev_plt_diff,
            done_already,
            key,
            ep_outcome,
        ) = carry
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
        # baseline_jax_{lite,full} we compute its (L, 3) row and splice it in.
        # For self_snapshot we featurize the opponent seat, run the frozen
        # opp_model forward, and take the deterministic (argmax) action.
        opp_seat = jnp.int32(1) - jnp.int32(seat)
        # Compute ONLY the selected opponent's actions. Each python_v* branch is
        # a `pure_callback` host hop (vmap_method='sequential'): computing it
        # eagerly (outside the switch) fires the host round-trip every step even
        # for a noop/pool run, which dominated rollout (~390s/iter, GPU idle).
        # Wrapping every branch in a lambda so only the chosen one executes keeps
        # the in-JAX opponents (noop/lite/full/self_snapshot) fully on-device.
        # Dispatch: 0 → noop, 1 → lite, 2 → full, 3 → self_snapshot,
        # 4 → python_v1, 5 → python_v4, 6 → python_v8.
        opp_actions = jax.lax.switch(
            jnp.clip(opponent_mode, 0, 6),
            [
                lambda: jnp.full((MAX_LAUNCHES_PER_AGENT, 3), -1.0, dtype=jnp.float32),
                lambda: _baseline_jax_actions(state, 1 - seat),
                lambda: _baseline_jax_full_actions(state, 1 - seat),
                lambda: _self_snapshot_opponent_actions(opp_model, state, 1 - seat),
                lambda: _python_v1_opponent_actions(state, 1 - seat),
                lambda: _python_v4_opponent_actions(state, 1 - seat),
                lambda: _python_v8_opponent_actions(state, 1 - seat),
            ],
        )
        # Splice into env_actions row for opponent seat only when not noop.
        env_actions = jax.lax.cond(
            opponent_mode != OPPONENT_NOOP,
            lambda ea: ea.at[opp_seat].set(opp_actions),
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

        # Shaping reward: c_ship·ΔΦ_ship + c_planet·ΔΦ_planet, where Φ is the
        # mode-specific potential (diff for ships/planets/combined, ratio for
        # H2). Both potentials are tracked independently so every mode stays
        # potential-based per signal. Legacy modes zero one coefficient
        # (see _shaping_coefs) → bit-identical.
        ship_diff, plt_diff = _shaping_potentials(state_for_next, seat, shaping_mode)
        shaping = c_ship * (ship_diff - prev_ship_diff) + c_planet * (
            plt_diff - prev_plt_diff
        )
        # H2 (dense differential, 2026-06-08): non-PBRS dense reward on the
        # NORMALIZED material *advantage* `c·(mine-enemy)/(mine+enemy)` per turn,
        # bounded in [-c, +c]. Unlike the old `c·mine_count` (which biased toward
        # hoarding — Ng et al. 1999), the differential rewards out-positioning
        # the opponent, the behavior that actually beats a rulebase. It is dense
        # (fires every turn, not just terminal) so it gives gradient where the
        # sparse win/loss signal dies vs a strong rulebase. dense_coef=0 is a
        # no-op so every existing mode stays bit-identical.
        ship_mine, ship_enemy = _ship_totals(state_for_next, seat)
        plt_mine, plt_enemy = _planet_count_totals(state_for_next, seat)
        ship_adv = (ship_mine - ship_enemy) / (ship_mine + ship_enemy + 1.0)
        plt_adv = (plt_mine - plt_enemy) / (plt_mine + plt_enemy + 1.0)
        dense_addition = dense_coef_ship * ship_adv + dense_coef_planet * plt_adv
        shaping = shaping + dense_addition
        # H7: band-clip the per-turn shaping reward to [-clip, +clip] so the
        # early-game ratio spikes (when totals are tiny, ΔΦ swings ~±1) are
        # capped while the steady-state signal passes through. clip<=0 is a
        # no-op so every existing mode stays bit-identical.
        shaping = jnp.where(
            shaping_clip > 0.0,
            jnp.clip(shaping, -shaping_clip, shaping_clip),
            shaping,
        )
        # H6: per-turn time penalty to discourage stalling. coef=0 is no-op so
        # every existing mode stays bit-identical. Applied AFTER clip so the
        # clip bounds only the PBRS+dense signal, not the linear time pressure.
        shaping = shaping - time_penalty_coef

        # Terminal reward: env emits rewards on termination; turn into
        # ±1/0 by comparing the env's rewards for both seats.
        rewards_arr = state_for_next.rewards
        r_self = rewards_arr[seat]
        r_opp = rewards_arr[1 - seat]
        terminal_sign = jnp.sign(r_self - r_opp).astype(jnp.float32)
        # H6: time bonus on winning terminal — reward early victories. coef=0
        # is no-op. Win → `coef · (1 - t/horizon)`; loss/draw → 0 (only positive
        # sign adds bonus). `_t` is the scan index, horizon is captured above.
        win_mask = (terminal_sign > 0.0).astype(jnp.float32)
        time_bonus = (
            time_bonus_coef
            * win_mask
            * (1.0 - _t.astype(jnp.float32) / jnp.float32(horizon))
        )
        terminal_reward = jnp.where(term, terminal_sign + time_bonus, jnp.float32(0.0))
        step_reward = jnp.where(
            done_already, jnp.float32(0.0), shaping + terminal_reward
        )
        new_done = done_already | term
        # Outcome: cumulative reward (= terminal sign + shaping, just
        # like the torch path).
        new_outcome = ep_outcome + step_reward

        # Update history AFTER featurize, mirroring the torch loop's
        # `update_history(history, obs, actions=actions_list)` call at the
        # end of the step. Record the agent's REAL launches this turn so the
        # history's memory/timeline features match the eval path (closes the
        # train/eval gap; "memory features" per arXiv 2507.06825). The agent
        # controls `seat`; its env_actions row is (MAX_LAUNCHES, 3) =
        # [from_pid, angle, ships] with from_pid == -1 for non-fire slots.
        my_launch_row = env_actions[seat]  # (MAX_LAUNCHES, 3)
        launch_pid = my_launch_row[:, 0].astype(jnp.int32)
        launch_ships = my_launch_row[:, 2].astype(jnp.int32)
        launch_valid = launch_pid >= 0
        new_history = update_history_jax(
            history,
            state,
            launch_pid,
            launch_ships,
            launch_valid,
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
            ship_diff,
            plt_diff,
            new_done,
            key,
            new_outcome,
        )
        return new_carry, per_step

    init_carry = (
        init_state,
        init_history,
        init_ship_diff,
        init_plt_diff,
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
    ep_outcome = final_carry[6]
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
    coef_ship: float = 0.0,
    coef_planet: float = 0.0,
    shaping_clip: float = 0.0,
    dense_coef_ship: float = 0.0,
    dense_coef_planet: float = 0.0,
    time_bonus_coef: float = 0.0,
    time_penalty_coef: float = 0.0,
    opp_model: ActorCriticJax | None = None,
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
      - "self_snapshot": frozen self-snapshot agent (`opp_model`) plays the
        opponent seat deterministically. `opp_model` must be supplied for
        this mode; for all other modes it is unused and defaults to `model`
        so the pytree structure threaded into vmap stays valid (H1: PFSP
        self-play foundation).
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
    if opponent == "self_snapshot" and opp_model is None:
        raise ValueError("opponent='self_snapshot' requires opp_model to be supplied")
    opponent_mode = jnp.int32(OPPONENT_NAME_TO_MODE[opponent])
    shaping_mode_int = jnp.int32(SHAPING_MODE_NAME_TO_INT[shaping_mode])
    # Unused for non-snapshot opponents, but threaded so the vmap pytree
    # structure stays valid regardless of mode.
    effective_opp_model = model if opp_model is None else opp_model

    keys = jax.random.split(key, episodes_per_iter)
    # Build all N initial states on host (reset is Python-side), then
    # stack into a single batched EnvState so vmap can broadcast model
    # parameters across the episode axis. This keeps the jit cache to
    # ONE compilation regardless of episode count.
    init_states = [reset(seed=seed + i, num_agents=2) for i in range(episodes_per_iter)]
    batched_state = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *init_states)

    # vmap over (key, init_state); model + scalar args + opp_model broadcast.
    # in_axes mirrors _rollout_one_env's positional signature: model(None),
    # key(0), init_state(0), then every scalar shaping arg(None), and the
    # broadcast opp_model(None) last.
    vmapped = jax.vmap(
        _rollout_one_env,
        in_axes=(
            None,  # model
            0,  # key
            0,  # init_state
            None,  # horizon
            None,  # shaping_coef
            None,  # seat
            None,  # opponent_mode
            None,  # shaping_mode
            None,  # coef_ship
            None,  # coef_planet
            None,  # shaping_clip
            None,  # dense_coef_ship
            None,  # dense_coef_planet
            None,  # time_bonus_coef
            None,  # time_penalty_coef
            None,  # opp_model
        ),
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
        coef_ship,
        coef_planet,
        shaping_clip,
        dense_coef_ship,
        dense_coef_planet,
        time_bonus_coef,
        time_penalty_coef,
        effective_opp_model,
    )


__all__ = [
    "JaxRolloutBatch",
    "collect_rollout_jax",
]
