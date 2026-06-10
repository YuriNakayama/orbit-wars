"""Host-driven batch self-play between two JAX agents (compile-feasible).

``selfplay_jax.run_selfplay_batch`` wraps a 500-turn ``lax.scan`` of
``compute_actions`` inside ``vmap`` — one monolithic graph. The case8 XLA
investigation (case8_xla_investigation.md) showed the per-TURN
``compute_actions`` graph alone is huge (2304 cells × HORIZON scan): ~28 s to
compile on GPU, and intractable on CPU. Folding 500 turns into one scan makes it
worse (CPU smoke ran 29 min without finishing).

This module instead compiles ONLY the per-turn step — ``vmap`` of (agent0 action,
agent1 action, env step) over the B games — ONCE, then drives the turn loop on
the host in Python. Same outcome semantics (sticky ``done`` + frozen terminal
``rewards``), but the compile is the validated per-turn graph and the loop is
cheap host overhead. It also yields per-turn progress for live observability.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from orbit_wars_jax.constants import EPISODE_STEPS
from orbit_wars_jax.reset import reset
from orbit_wars_jax.state import EnvState
from orbit_wars_jax.step import MAX_LAUNCHES_PER_AGENT, NUM_AGENTS_MAX
from orbit_wars_jax.step import step as jax_env_step

ActionFn = Callable[[EnvState], jax.Array]

OUTCOME_SEAT0_WIN: int = 0
OUTCOME_SEAT1_WIN: int = 1
OUTCOME_DRAW: int = 2


def _empty_env_actions() -> jax.Array:
    return jnp.full(
        (NUM_AGENTS_MAX, MAX_LAUNCHES_PER_AGENT, 3), -1.0, dtype=jnp.float32
    )


def _make_turn_step(
    agent0_fn: ActionFn, agent1_fn: ActionFn
) -> Callable[
    [EnvState, jax.Array, jax.Array, jax.Array],
    tuple[EnvState, jax.Array, jax.Array, jax.Array],
]:
    """Build the jitted, vmapped single-turn transition for a batch of games.

    Per game: compute both seats' actions, splice into env actions, env step;
    freeze the state once ``done`` so terminated games stop evolving (single
    shape, no host branch on traced data). Returns
    ``(new_state, new_done, new_final_rewards, new_turns)`` per game.
    """

    def one_game(
        state: EnvState,
        done: jax.Array,
        final_rewards: jax.Array,
        turns: jax.Array,
    ) -> tuple[EnvState, jax.Array, jax.Array, jax.Array]:
        act0 = agent0_fn(state)
        act1 = agent1_fn(state)
        env_actions = _empty_env_actions().at[0].set(act0).at[1].set(act1)
        new_state, _r, term = jax_env_step(state, env_actions)
        state_next = jax.tree.map(
            lambda new, old: jnp.where(done, old, new), new_state, state
        )
        just_term = (~done) & term
        new_final = jnp.where(just_term, state_next.rewards, final_rewards)
        new_turns = jnp.where(done, turns, turns + 1)
        new_done = done | term
        return state_next, new_done, new_final, new_turns

    # G2 (donate_argnums): the host loop overwrites all 4 carry buffers each turn,
    # so XLA may reuse them in-place — eliminates per-turn carry copies. Donation
    # changes only memory layout, never numerics (results are identical).
    return jax.jit(jax.vmap(one_game), donate_argnums=(0, 1, 2, 3))


def run_host_batch(
    seeds: list[int],
    agent0_fn: ActionFn,
    agent1_fn: ActionFn,
    horizon: int = EPISODE_STEPS,
    *,
    on_turn: Callable[[int, jax.Array], None] | None = None,
    sync_every: int = 16,
) -> dict[str, jax.Array]:
    """Play one game per seed, host-driven. Returns batched outcome arrays.

    Speed: a large ``seeds`` batch amortizes the per-turn ``compute_actions``
    grid+allocator cost (case8 bench: 2092 ms/state at batch 8 → 283 ms/state at
    batch 64, a 7.4x win — fill the batch). The all-done early-break (#3b) and the
    per-turn host sync (#2) are throttled to every ``sync_every`` turns: checking
    ``jnp.all(done)`` each turn forces a device→host sync that serializes the
    async dispatch pipeline. Throttling lets ``sync_every`` turns queue on the GPU
    before one sync — terminated games are frozen by the step fn, so running a few
    extra turns is harmless to the outcome (results are identical).

    ``on_turn(turn_idx, done_array)`` is invoked at each sync point for live
    progress (e.g. log how many games have finished).
    """
    if not seeds:
        raise ValueError("seeds must be a non-empty list of ints")
    states = [reset(seed=s, num_agents=2) for s in seeds]
    batched = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *states)
    b = len(seeds)
    done = jnp.zeros((b,), dtype=jnp.bool_)
    final_rewards = jnp.zeros((b, NUM_AGENTS_MAX), dtype=jnp.float32)
    turns = jnp.zeros((b,), dtype=jnp.int32)

    turn_step = _make_turn_step(agent0_fn, agent1_fn)

    if sync_every <= 0:
        # G1 (full async enqueue): queue ALL horizon turns with NO mid-loop host
        # sync, then block once. Per JAX async-dispatch docs, the host enqueues
        # faster than the device runs, so the accelerator never waits on Python.
        # No early-break (terminated games are frozen by the step fn, so running
        # the full horizon costs only cheap no-op turns) — results are identical.
        for _t in range(horizon):
            batched, done, final_rewards, turns = turn_step(
                batched, done, final_rewards, turns
            )
        done.block_until_ready()
    else:
        # Throttled sync: check the all-done early-break every `sync_every` turns.
        for t in range(horizon):
            batched, done, final_rewards, turns = turn_step(
                batched, done, final_rewards, turns
            )
            if (t + 1) % sync_every == 0 or t == horizon - 1:
                done.block_until_ready()
                if on_turn is not None:
                    on_turn(t, done)
                if bool(jnp.all(done)):
                    break

    r0 = final_rewards[:, 0]
    r1 = final_rewards[:, 1]
    outcome = jnp.where(
        r0 > r1,
        jnp.int32(OUTCOME_SEAT0_WIN),
        jnp.where(r1 > r0, jnp.int32(OUTCOME_SEAT1_WIN), jnp.int32(OUTCOME_DRAW)),
    )
    return {
        "outcome": outcome,
        "final_rewards": final_rewards,
        "done": done,
        "turns": turns,
    }
