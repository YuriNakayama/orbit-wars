"""Multi-step beam search optimizer for case8.

`run_beam` searches over orderings of the top-K candidate missions and returns
the one whose resulting commitments project to the highest scored future state
under `simulate_planet_timeline`. When `BEAM_ENABLED=False`, callers fall back
to the case7 greedy ordering (score-desc) and skip this module entirely.
"""

from __future__ import annotations

from .beam import run_beam

__all__ = ["run_beam"]
