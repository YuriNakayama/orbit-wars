"""Binary strength comparison between two JAX rule agents: "is A > B?".

The ranking driver (``ranking.py``) exploits three facts the user established:
strength is a strict TOTAL ORDER (transitive, no rock-paper-scissors), some
agents are near-equal, and case number ≈ strength (a strong prior). Under a
total order, ranking is SORTING, so each step only needs a single binary verdict
``A > B`` / ``B > A`` / ``tie`` — far cheaper than a win-rate leaderboard.

This module is that primitive. It plays A-vs-B over ``seeds`` in BOTH seat
assignments (cancelling the env's seat-0/seat-1 start asymmetry, exactly like
``run_tournament._play_pair``), tallies decisive wins, and classifies the
result by the Wilson 95% CI of A's decisive win-rate:

  * CI lower bound > 0.5  → A_WINS  (A decisively stronger)
  * CI upper bound < 0.5  → B_WINS  (B decisively stronger)
  * CI straddles 0.5      → TIE     (near-equal; do not burn more games)

Games per comparison are FIXED (no sequential early-stop): ``seeds`` per seat
half → ``2 * len(seeds)`` games total. Draws are excluded from the win-rate
denominator (they inform neither direction of the order).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp

from evaluate.vs_baseline import wilson_ci
from pipeline.rulebase._bench.tournament.agents_jax import action_fn
from pipeline.rulebase._bench.tournament.selfplay_host import (
    OUTCOME_DRAW,
    OUTCOME_SEAT0_WIN,
    OUTCOME_SEAT1_WIN,
    run_host_batch,
)

A_WINS: str = "a_wins"
B_WINS: str = "b_wins"
TIE: str = "tie"


@dataclass(frozen=True)
class ComparisonResult:
    """Outcome of one A-vs-B comparison (both seat halves aggregated)."""

    a: str
    b: str
    a_decisive_wins: int
    b_decisive_wins: int
    draws: int
    a_win_rate: float  # a / (a + b), draws excluded
    ci95_lo: float
    ci95_hi: float
    decision: str  # A_WINS | B_WINS | TIE
    half_secs: tuple[float, float]


def _decide(a_wins: int, b_wins: int) -> tuple[float, float, float, str]:
    """Classify by the Wilson 95% CI of A's decisive win-rate vs 0.5."""
    decisive = a_wins + b_wins
    if decisive == 0:
        return 0.5, 0.0, 1.0, TIE
    rate = a_wins / decisive
    lo, hi = wilson_ci(a_wins, decisive)
    if lo > 0.5:
        decision = A_WINS
    elif hi < 0.5:
        decision = B_WINS
    else:
        decision = TIE
    return rate, lo, hi, decision


def compare_pair(
    a: str,
    b: str,
    seeds: list[int],
    horizon: int,
    *,
    run_batch: Callable[..., dict[str, jax.Array]] = run_host_batch,
) -> ComparisonResult:
    """Play A-vs-B over ``seeds`` in both seat halves and return the verdict.

    ``run_batch`` is injected for testing (defaults to the real GPU self-play).
    The seat-swap (half 1: a=seat0; half 2: a=seat1) cancels start asymmetry, so
    A's aggregated decisive win-rate is an unbiased strength estimate.
    """

    a_wins = 0
    b_wins = 0
    draws = 0
    half_secs: list[float] = []

    for s0_name, s1_name in ((a, b), (b, a)):
        t0 = time.perf_counter()
        res = run_batch(
            seeds,
            action_fn(s0_name, 0),
            action_fn(s1_name, 1),
            horizon=horizon,
        )
        outcome = res["outcome"]
        outcome.block_until_ready()
        half_secs.append(round(time.perf_counter() - t0, 2))

        s0 = int(jnp.sum(outcome == OUTCOME_SEAT0_WIN))
        s1 = int(jnp.sum(outcome == OUTCOME_SEAT1_WIN))
        draws += int(jnp.sum(outcome == OUTCOME_DRAW))
        # Map seat wins back to A/B: whoever sat in seat0 this half got s0 wins.
        if s0_name == a:
            a_wins += s0
            b_wins += s1
        else:
            a_wins += s1
            b_wins += s0

    rate, lo, hi, decision = _decide(a_wins, b_wins)
    return ComparisonResult(
        a=a,
        b=b,
        a_decisive_wins=a_wins,
        b_decisive_wins=b_wins,
        draws=draws,
        a_win_rate=rate,
        ci95_lo=lo,
        ci95_hi=hi,
        decision=decision,
        half_secs=(half_secs[0], half_secs[1]),
    )
