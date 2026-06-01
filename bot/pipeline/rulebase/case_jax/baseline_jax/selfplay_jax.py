# Adapted from "Orbit Wars 2026 - Reinforce" by sigmaborov
# https://www.kaggle.com/code/sigmaborov/lb-897-orbit-wars-2026-reinforce
# Licensed under Apache License 2.0
"""Vmapped batched JAX self-play harness for the case_jax rulebase agent.

This is BOTH the win-rate evaluator and the GPU speed bench: it pits two
fully-JAX `compute_actions` policies against each other inside a single
`lax.scan` game loop, vmapped across N parallel games, all on-device with no
Python per-turn round-trip.

Pipeline mirrors `pipeline/reinforce/case1/training/rollout_jax.py`:

    host: reset(seed, num_agents=2) per game -> stack into a batched EnvState
    device: jax.vmap(_play_one) over the batch
        _play_one: lax.scan over `horizon` turns; each turn
            act0 = agent0_fn(state)   # (L, 3)
            act1 = agent1_fn(state)   # (L, 3)
            env_actions = empty.at[0].set(act0).at[1].set(act1)
            state, _r, term = step(state, env_actions)
            carry a sticky `done` + the frozen final rewards (the env writes
            +1/-1 into `rewards` on termination; freeze them so post-terminal
            steps cannot change the outcome).

The default action fn for BOTH seats is the jax_v4 `compute_actions`, so
`selfplay_winrate` is a jax_v4-vs-jax_v4 symmetry sanity check (~50%). The two
fns are parameterizable so a future jax_v4-vs-OTHER eval just swaps `agent1_fn`.

Outcome decode (from the frozen final `rewards`):
    rewards[0] > rewards[1] -> seat0 win
    rewards[1] > rewards[0] -> seat1 win
    otherwise               -> draw
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from orbit_wars_jax.constants import EPISODE_STEPS
from orbit_wars_jax.reset import reset
from orbit_wars_jax.state import EnvState
from orbit_wars_jax.step import MAX_LAUNCHES_PER_AGENT, NUM_AGENTS_MAX
from orbit_wars_jax.step import step as jax_env_step

from .agent_jax import _modes_from_features, compute_actions
from .world_features import build_world_features_from_state

# Action fn signature: EnvState -> (MAX_LAUNCHES_PER_AGENT, 3) float32 launch row.
ActionFn = Callable[[EnvState], jax.Array]

# Outcome codes (int32, scan/vmap-friendly — no Python branch on traced values).
OUTCOME_SEAT0_WIN: int = 0
OUTCOME_SEAT1_WIN: int = 1
OUTCOME_DRAW: int = 2


class SelfPlayResult(NamedTuple):
    """Per-game outcomes from a vmapped self-play batch.

    Leading axis B = len(seeds). All arrays are device arrays returned by the
    single vmapped call (no host loop over games).

    `seat_*_planets/ships/prod` summarize the final board state for seat 0 and
    seat 1. When `horizon` cuts off before termination (`terminated == False`,
    `final_rewards` is all zeros), these board summaries are the only signal
    available for "is jax_v4 losing degenerate" — compare seat0 vs seat1 totals.
    """

    outcome: jax.Array  # (B,) int32 in {OUTCOME_SEAT0_WIN, SEAT1_WIN, DRAW}
    final_rewards: jax.Array  # (B, NUM_AGENTS_MAX) float32 frozen terminal rewards
    terminated: jax.Array  # (B,) bool — game reached env termination within horizon
    turns_played: jax.Array  # (B,) int32 — turns until termination (== horizon if not)
    seat0_planets: jax.Array  # (B,) int32 — # planets owned by seat 0 at end
    seat1_planets: jax.Array  # (B,) int32
    seat0_ships: jax.Array  # (B,) int32 — total ships on seat-0 planets at end
    seat1_ships: jax.Array  # (B,) int32
    seat0_prod: jax.Array  # (B,) int32 — total production on seat-0 planets
    seat1_prod: jax.Array  # (B,) int32


def _seat_action(state: EnvState, seat: int) -> jax.Array:
    """jax_v4 `compute_actions` for one static seat -> (MAX_LAUNCHES_PER_AGENT, 3)."""
    features = build_world_features_from_state(state, seat)
    modes = _modes_from_features(features)
    return compute_actions(features, modes)


def jax_v4_action_fn(seat: int) -> ActionFn:
    """Bind the static `seat` and return an `ActionFn` for the jax_v4 policy."""

    def _fn(state: EnvState) -> jax.Array:
        return _seat_action(state, seat)

    return _fn


def python_v8_action_fn(seat: int) -> ActionFn:
    """ActionFn that calls the Python case8 `baseline_v8.agent` via host callback.

    Converts the JAX `EnvState` into a kaggle obs dict (via `state_to_obs`) on
    the host, invokes the Python `baseline.agent.agent(obs)`, packs the move list
    into a `(MAX_LAUNCHES_PER_AGENT, 3)` tensor, returns it as a device array.
    The callback runs on host (CPU) so vmap parallelism is reduced to a host
    loop for this seat — acceptable for win-rate eval, not for throughput.
    """
    from orbit_wars_jax.observation import state_to_obs as _state_to_obs

    from pipeline.rulebase.case8.baseline.agent import agent as _v8_agent
    from pipeline.rulebase.case8.baseline.core.physics import (
        reset_predict_cache as _reset_cache,
    )

    seat_i = int(seat)

    def _host_v8(state: EnvState) -> np.ndarray:
        # State arrives as numpy via pure_callback. Build the obs and call v8.
        obs = _state_to_obs(state, player=seat_i)
        _reset_cache()
        moves = _v8_agent(obs)
        out = np.full((MAX_LAUNCHES_PER_AGENT, 3), -1.0, dtype=np.float32)
        out[:, 1:] = 0.0
        for j, mv in enumerate(moves[:MAX_LAUNCHES_PER_AGENT]):
            out[j] = [float(mv[0]), float(mv[1]), float(mv[2])]
        return out

    # Prototype array tells pure_callback the result's shape and dtype without
    # constructing an untyped jax.ShapeDtypeStruct (mypy-safe).
    _proto = jnp.zeros((MAX_LAUNCHES_PER_AGENT, 3), dtype=jnp.float32)

    def _fn(state: EnvState) -> jax.Array:
        # vmap_method='sequential' = call _host_v8 once per game in the batch,
        # then stack results. Host callbacks can't truly parallelize on GPU; this
        # makes the (otherwise unsupported) vmap legal. Throughput drops to per-
        # call rate, but win-rate eval just needs each game's outcome.
        return jnp.asarray(
            jax.pure_callback(_host_v8, _proto, state, vmap_method="sequential")
        )

    return _fn


def batched_turn_actions(batched_state: EnvState, seat: int = 0) -> jax.Array:
    """`vmap(_seat_action)` over a batched EnvState → `(B, MAX_LAUNCHES, 3)`.

    The single-turn throughput unit: one `compute_actions` per game, vmapped
    across the batch. This is what the GPU speed bench measures — it compiles
    once (the per-turn graph, ~82s CPU) and avoids the intractable game-loop
    `lax.scan` compile. `seat` is static.
    """

    def turn(state: EnvState) -> jax.Array:
        return _seat_action(state, seat)

    return jax.vmap(turn)(batched_state)


def _empty_env_actions() -> jax.Array:
    """No-op action tensor (all seats suppressed via -1 sentinel)."""
    return jnp.full(
        (NUM_AGENTS_MAX, MAX_LAUNCHES_PER_AGENT, 3), -1.0, dtype=jnp.float32
    )


class _Carry(NamedTuple):
    """lax.scan carry for one self-play game."""

    state: EnvState
    done: jax.Array  # bool — sticky termination flag
    final_rewards: jax.Array  # (NUM_AGENTS_MAX,) float32 — frozen at first termination
    turns: jax.Array  # int32 — turns played before termination (clamped at done)


def _play_one(
    init_state: EnvState,
    horizon: int,
    agent0_fn: ActionFn,
    agent1_fn: ActionFn,
) -> tuple[
    jax.Array, jax.Array, jax.Array, jax.Array,
    jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array,
]:
    """Run one self-play game via `lax.scan` over `horizon` turns.

    Returns `(outcome, final_rewards, terminated, turns_played)`. Both seats run
    their respective `ActionFn`; the env is advanced every turn but the carry
    freezes `final_rewards` / `turns` once `done` so post-terminal steps cannot
    change the outcome (shapes stay constant — no Python branch on traced data).
    """

    def step_fn(carry: _Carry, _t: jax.Array) -> tuple[_Carry, None]:
        act0 = agent0_fn(carry.state)
        act1 = agent1_fn(carry.state)
        env_actions = _empty_env_actions().at[0].set(act0).at[1].set(act1)

        new_state, _r, term = jax_env_step(carry.state, env_actions)
        # Hold the state once already done so the trace stays single-shape.
        state_for_next = jax.tree.map(
            lambda new, old: jnp.where(carry.done, old, new), new_state, carry.state
        )

        just_terminated = (~carry.done) & term
        final_rewards = jnp.where(
            just_terminated, state_for_next.rewards, carry.final_rewards
        )
        turns = jnp.where(carry.done, carry.turns, carry.turns + 1)
        new_done = carry.done | term
        return (
            _Carry(
                state=state_for_next,
                done=new_done,
                final_rewards=final_rewards,
                turns=turns,
            ),
            None,
        )

    init_carry = _Carry(
        state=init_state,
        done=jnp.bool_(False),
        final_rewards=jnp.zeros(NUM_AGENTS_MAX, dtype=jnp.float32),
        turns=jnp.int32(0),
    )
    final_carry, _ = jax.lax.scan(
        step_fn, init_carry, jnp.arange(horizon, dtype=jnp.int32)
    )

    r0 = final_carry.final_rewards[0]
    r1 = final_carry.final_rewards[1]
    outcome = jnp.where(
        r0 > r1,
        jnp.int32(OUTCOME_SEAT0_WIN),
        jnp.where(r1 > r0, jnp.int32(OUTCOME_SEAT1_WIN), jnp.int32(OUTCOME_DRAW)),
    )

    # Per-seat board summary at the end of the scan (whether terminated or not).
    # Lets the bench compare seat0 vs seat1 totals when reward is 0 (horizon cut).
    fs = final_carry.state
    seat0_mask = fs.planet_valid & (fs.planet_owner == jnp.int32(0))
    seat1_mask = fs.planet_valid & (fs.planet_owner == jnp.int32(1))
    seat0_planets = jnp.sum(seat0_mask.astype(jnp.int32))
    seat1_planets = jnp.sum(seat1_mask.astype(jnp.int32))
    seat0_ships = jnp.sum(jnp.where(seat0_mask, fs.planet_ships, 0))
    seat1_ships = jnp.sum(jnp.where(seat1_mask, fs.planet_ships, 0))
    seat0_prod = jnp.sum(jnp.where(seat0_mask, fs.planet_prod, 0))
    seat1_prod = jnp.sum(jnp.where(seat1_mask, fs.planet_prod, 0))

    return (
        outcome,
        final_carry.final_rewards,
        final_carry.done,
        final_carry.turns,
        seat0_planets,
        seat1_planets,
        seat0_ships,
        seat1_ships,
        seat0_prod,
        seat1_prod,
    )


def _stack_states(states: list[EnvState]) -> EnvState:
    """Stack per-game `EnvState`s into a single batched pytree (leading axis B)."""
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *states)


def run_selfplay_batch(
    seeds: list[int],
    horizon: int = EPISODE_STEPS,
    *,
    agent0_fn: ActionFn | None = None,
    agent1_fn: ActionFn | None = None,
) -> SelfPlayResult:
    """Run N parallel self-play games (one per seed) via `jax.vmap(_play_one)`.

    Host resets each seed (`num_agents=2`) and stacks the states into a batched
    `EnvState`; a single vmapped device call plays all games. Both action fns
    default to the jax_v4 `compute_actions` (seat 0 / seat 1), i.e. jax_v4
    self-play. Pass a different `agent1_fn` for jax_v4-vs-OTHER evaluation.
    """
    if not seeds:
        raise ValueError("seeds must be a non-empty list of ints")
    a0 = agent0_fn if agent0_fn is not None else jax_v4_action_fn(0)
    a1 = agent1_fn if agent1_fn is not None else jax_v4_action_fn(1)

    init_states = [reset(seed=s, num_agents=2) for s in seeds]
    batched_state = _stack_states(init_states)

    def play(state: EnvState) -> tuple[
        jax.Array, jax.Array, jax.Array, jax.Array,
        jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array,
    ]:
        return _play_one(state, horizon, a0, a1)

    (
        outcome,
        final_rewards,
        terminated,
        turns,
        seat0_planets,
        seat1_planets,
        seat0_ships,
        seat1_ships,
        seat0_prod,
        seat1_prod,
    ) = jax.vmap(play)(batched_state)
    return SelfPlayResult(
        outcome=outcome,
        final_rewards=final_rewards,
        terminated=terminated,
        turns_played=turns,
        seat0_planets=seat0_planets,
        seat1_planets=seat1_planets,
        seat0_ships=seat0_ships,
        seat1_ships=seat1_ships,
        seat0_prod=seat0_prod,
        seat1_prod=seat1_prod,
    )


def selfplay_winrate(
    seeds: list[int], horizon: int = EPISODE_STEPS
) -> dict[str, float | int]:
    """Aggregate seat0 win-rate over the self-play batch (symmetry sanity check).

    jax_v4-vs-jax_v4 self-play should land near 50% seat0 win-rate by symmetry
    (modulo the env's seat-0/seat-1 start asymmetry). Returns counts plus the
    seat0 win-rate (decisive games and overall).
    """
    res = run_selfplay_batch(seeds, horizon)
    outcome = res.outcome
    n = int(outcome.shape[0])
    seat0_wins = int(jnp.sum(outcome == OUTCOME_SEAT0_WIN))
    seat1_wins = int(jnp.sum(outcome == OUTCOME_SEAT1_WIN))
    draws = int(jnp.sum(outcome == OUTCOME_DRAW))
    decisive = seat0_wins + seat1_wins
    return {
        "games": n,
        "seat0_wins": seat0_wins,
        "seat1_wins": seat1_wins,
        "draws": draws,
        "seat0_winrate": (seat0_wins / n) if n > 0 else 0.0,
        "seat0_winrate_decisive": (seat0_wins / decisive) if decisive > 0 else 0.0,
        "terminated": int(jnp.sum(res.terminated)),
    }


__all__ = [
    "OUTCOME_DRAW",
    "OUTCOME_SEAT0_WIN",
    "OUTCOME_SEAT1_WIN",
    "ActionFn",
    "SelfPlayResult",
    "jax_v4_action_fn",
    "run_selfplay_batch",
    "selfplay_winrate",
]
