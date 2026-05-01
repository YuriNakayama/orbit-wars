"""Self-play vs baseline summarization helpers shared across cases.

The case-specific `evaluation/eval_vs_baseline.py` wrappers wire the
challenger / baseline AGENT_REGISTRY keys, run episodes via
`dataset.selfplay.runner.run_episodes`, and call `summarize_records` here
to compute Wilson 95 % CI over the wins/losses/draws split.
"""

from __future__ import annotations

import math
from typing import Any

from dataset.schema import MatchRecord


def wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — stable for small successes / small n."""
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = (z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarize_records(
    records: list[MatchRecord], challenger_idx: int
) -> dict[str, Any]:
    """Aggregate `records` from the challenger's perspective."""
    n = len(records)
    wins = sum(1 for r in records if r.winner == challenger_idx and not r.draw)
    losses = sum(
        1
        for r in records
        if r.winner != challenger_idx and r.winner >= 0 and not r.draw
    )
    draws = sum(1 for r in records if r.draw)
    ci_lo, ci_hi = wilson_ci(wins, n)
    return {
        "episodes": float(n),
        "wins": float(wins),
        "losses": float(losses),
        "draws": float(draws),
        "win_rate": wins / n if n else 0.0,
        "non_draw_win_rate": wins / (wins + losses) if (wins + losses) else 0.0,
        "win_rate_ci95_lo": ci_lo,
        "win_rate_ci95_hi": ci_hi,
    }
