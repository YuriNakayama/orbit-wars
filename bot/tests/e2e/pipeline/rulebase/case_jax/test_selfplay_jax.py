"""Self-play harness sanity for the vmapped batched JAX rulebase self-play.

`run_selfplay_batch(seeds, horizon)` plays jax_v4-vs-jax_v4 games inside a single
`lax.scan` loop, vmapped across the batch, entirely on-device. This test asserts
the harness is shape-correct, vmaps, freezes outcomes once a game terminates, and
decodes every result to a valid outcome code.

CPU REALITY (measured on this dev box, 2026-05-29):
    A single game, horizon=8, compiled + ran in ~422s; a vmapped batch of 2 at
    horizon=8 is comparable. The scan body calls the deep `compute_actions`
    graph TWICE per turn (seat 0 + seat 1), and that graph alone is ~24s to
    compile on CPU, so XLA compile dominates regardless of horizon. The full
    500-turn / 300-game win-rate + speed bench is therefore GPU-only (RunPod);
    on CPU we only verify the harness COMPILES, VMAPS, and decodes correctly at
    the smallest meaningful scale.

    Because the env cannot decide a winner in a handful of turns, short-horizon
    self-play games end as draws (frozen rewards all 0). The jax_v4-vs-jax_v4
    symmetry win-rate check (~50% seat0) only becomes meaningful at full horizon,
    so it lives behind `RUN_JAX_SELFPLAY_FULL=1` and is GPU-intended.

These tests are marked `slow`; the default scan smoke uses HORIZON=8 / batch=2,
the smallest scale that still exercises compile + vmap + outcome decode.
"""

from __future__ import annotations

import os

import jax.numpy as jnp
import pytest

from pipeline.rulebase.case_jax.baseline_jax.selfplay_jax import (
    OUTCOME_DRAW,
    OUTCOME_SEAT0_WIN,
    OUTCOME_SEAT1_WIN,
    SelfPlayResult,
    run_selfplay_batch,
    selfplay_winrate,
)

pytestmark = pytest.mark.slow

# Smallest scale that compiles + vmaps + decodes on CPU (see module docstring).
HORIZON = 8
SMOKE_BATCH = 2

# Full-scale symmetry win-rate is GPU-intended (needs full horizon to decide
# games). Opt in with RUN_JAX_SELFPLAY_FULL=1 (RunPod GPU).
_RUN_FULL = os.environ.get("RUN_JAX_SELFPLAY_FULL") == "1"
FULL_HORIZON = 500
FULL_SEEDS = list(range(8))

_VALID_CODES = {OUTCOME_SEAT0_WIN, OUTCOME_SEAT1_WIN, OUTCOME_DRAW}


def test_run_selfplay_batch_vmaps_and_decodes() -> None:
    """Vmapped batch compiles, runs, and decodes to valid outcome codes/shapes."""
    seeds = list(range(SMOKE_BATCH))
    res = run_selfplay_batch(seeds, horizon=HORIZON)

    assert isinstance(res, SelfPlayResult)
    n = len(seeds)
    assert res.outcome.shape == (n,)
    assert res.final_rewards.shape[0] == n
    assert res.terminated.shape == (n,)
    assert res.turns_played.shape == (n,)

    # Every game decodes to a valid outcome code.
    assert set(int(o) for o in res.outcome) <= _VALID_CODES
    # final_rewards are finite (frozen terminal rewards or zeros pre-terminal).
    assert bool(jnp.all(jnp.isfinite(res.final_rewards)))

    # turns_played is in [1, HORIZON]; a non-terminated game ran the full horizon.
    turns = res.turns_played
    assert bool(jnp.all((turns >= 1) & (turns <= HORIZON)))
    not_term = ~res.terminated
    assert bool(jnp.all(jnp.where(not_term, turns == HORIZON, True)))


def test_outcome_matches_final_rewards() -> None:
    """Outcome decode is consistent with the frozen seat-0/seat-1 rewards."""
    res = run_selfplay_batch(list(range(SMOKE_BATCH)), horizon=HORIZON)
    for i in range(SMOKE_BATCH):
        r0 = float(res.final_rewards[i, 0])
        r1 = float(res.final_rewards[i, 1])
        code = int(res.outcome[i])
        if r0 > r1:
            assert code == OUTCOME_SEAT0_WIN
        elif r1 > r0:
            assert code == OUTCOME_SEAT1_WIN
        else:
            assert code == OUTCOME_DRAW


@pytest.mark.skipif(
    not _RUN_FULL,
    reason="full-horizon symmetry win-rate is GPU-only; set RUN_JAX_SELFPLAY_FULL=1",
)
def test_selfplay_winrate_is_unbiased_full_horizon() -> None:
    """jax_v4 self-play seat0 win-rate stays in [0.30, 0.70] (symmetry sanity).

    GPU-intended: at full horizon games decide, so seat0 win-rate should land
    near 50% by symmetry (modulo the env's seat-0/seat-1 start asymmetry).
    """
    stats = selfplay_winrate(FULL_SEEDS, horizon=FULL_HORIZON)
    assert stats["games"] == len(FULL_SEEDS)

    winrate = float(stats["seat0_winrate"])
    print(  # noqa: T201 — surfaced intentionally as the requested measurement
        f"jax_v4 self-play seat0 win-rate (horizon={FULL_HORIZON}, "
        f"{len(FULL_SEEDS)} seeds) = {winrate:.3f} "
        f"(seat0={stats['seat0_wins']} seat1={stats['seat1_wins']} "
        f"draws={stats['draws']} terminated={stats['terminated']})"
    )
    assert 0.30 <= winrate <= 0.70, f"lopsided self-play win-rate: {winrate}"
