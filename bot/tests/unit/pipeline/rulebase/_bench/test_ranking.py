"""Unit tests for the strength-ranking insertion-sort (no GPU / no env).

``rank_agents`` is pure given a comparator: it drives ``compare_pair`` to verify
a prior order into rank buckets. We inject a synthetic comparator (strength
lookup) so these tests exercise the sort/tie-merge/inversion-repair logic
without running any self-play.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.rulebase._bench.tournament import ranking
from pipeline.rulebase._bench.tournament.compare import (
    A_WINS,
    B_WINS,
    TIE,
    ComparisonResult,
)

PRIOR = ["jax_v8", "jax_v6", "jax_v4", "jax_v3", "jax_v2", "jax_v1"]


def _make_comparator(strength: dict[str, float], eps: float = 0.05):
    """Synthetic compare_pair: A>B iff strength gap exceeds eps, else TIE."""

    def cp(a: str, b: str, seeds: list[int], horizon: int) -> ComparisonResult:
        da, db = strength[a], strength[b]
        if abs(da - db) < eps:
            decision, aw, bw = TIE, 100, 100
        elif da > db:
            decision, aw, bw = A_WINS, 170, 30
        else:
            decision, aw, bw = B_WINS, 30, 170
        rate = aw / (aw + bw)
        return ComparisonResult(a, b, aw, bw, 0, rate, 0.4, 0.6, decision, (1.0, 1.0))

    return cp


@pytest.fixture
def bind_constants(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bind the verdict constants ranking.py reads as module globals."""
    monkeypatch.setattr(ranking, "A_WINS", A_WINS)
    monkeypatch.setattr(ranking, "B_WINS", B_WINS)
    monkeypatch.setattr(ranking, "TIE", TIE)


def _rank(monkeypatch: pytest.MonkeyPatch, strength: dict[str, float]):
    monkeypatch.setattr(ranking, "compare_pair", _make_comparator(strength))
    return ranking.rank_agents(PRIOR, [0, 1, 2, 3], horizon=5, out_dir=Path("/tmp"))


def test_prior_correct_needs_only_n_minus_1(bind_constants, monkeypatch):
    strength = {f"jax_v{n}": float(n) for n in (8, 6, 4, 3, 2, 1)}
    buckets, comps = _rank(monkeypatch, strength)
    assert buckets == [[a] for a in PRIOR]
    assert len(comps) == len(PRIOR) - 1  # optimal: 5 comparisons


def test_near_equal_pair_merges_into_one_bucket(bind_constants, monkeypatch):
    strength = {"jax_v8": 8, "jax_v6": 6, "jax_v4": 4.0, "jax_v3": 4.02, "jax_v2": 2, "jax_v1": 1}
    buckets, comps = _rank(monkeypatch, strength)
    assert ["jax_v4", "jax_v3"] in buckets
    assert len(buckets) == 5  # one bucket holds the tied pair


def test_adjacent_inversion_is_repaired(bind_constants, monkeypatch):
    # Prior says v4 > v3, but v3 is actually stronger.
    strength = {"jax_v8": 8, "jax_v6": 6, "jax_v4": 4, "jax_v3": 5, "jax_v2": 2, "jax_v1": 1}
    buckets, comps = _rank(monkeypatch, strength)
    flat = [a for bucket in buckets for a in bucket]
    assert flat.index("jax_v3") < flat.index("jax_v4")
    assert len(comps) == len(PRIOR)  # one extra comparison for the repair


def test_weakest_prior_bubbles_to_top(bind_constants, monkeypatch):
    strength = {"jax_v8": 8, "jax_v6": 6, "jax_v4": 4, "jax_v3": 3, "jax_v2": 2, "jax_v1": 99}
    buckets, _ = _rank(monkeypatch, strength)
    assert buckets[0] == ["jax_v1"]


def test_all_tied_collapse_to_single_bucket(bind_constants, monkeypatch):
    strength = {a: 5.0 for a in PRIOR}
    buckets, _ = _rank(monkeypatch, strength)
    assert buckets == [PRIOR]


def test_fully_reversed_prior_still_sorts_correctly(bind_constants, monkeypatch):
    # A fully REVERSED prior is the worst case: each agent bubbles to the top.
    # With 6 agents the deepest bubble is 5 hops (== MAX_REPAIR_HOPS), so it
    # still completes — just expensively — and yields the true reversed order.
    # Strength INCREASES along the prior (prior is fully backwards), so each new
    # agent beats every predecessor and bubbles to the top.
    strength = {name: float(i) for i, name in enumerate(PRIOR)}
    buckets, _ = _rank(monkeypatch, strength)
    assert buckets == [[a] for a in reversed(PRIOR)]


def test_runaway_inversion_aborts(bind_constants, monkeypatch):
    # A reversed prior LONGER than MAX_REPAIR_HOPS+1 forces a bubble deeper than
    # the guard allows — must raise rather than silently O(n^2) on a bad prior.
    long_prior = [f"a{i}" for i in range(ranking.MAX_REPAIR_HOPS + 3)]
    # Strength increasing along the prior → fully backwards → deepest bubble.
    strength = {name: float(i) for i, name in enumerate(long_prior)}
    monkeypatch.setattr(ranking, "compare_pair", _make_comparator(strength))
    with pytest.raises(RuntimeError, match="repair hops"):
        ranking.rank_agents(long_prior, [0, 1], horizon=5, out_dir=Path("/tmp"))
