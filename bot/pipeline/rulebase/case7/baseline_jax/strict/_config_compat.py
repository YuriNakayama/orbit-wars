"""Config shim: re-export case7 constants for the strict JAX modules.

case7 = case8 (engine-replay aim, grid+allocator) + STAY_BURST (case6) +
ACCUMULATE (case7-only multi-turn ship hoarding). case7's config is a superset
of case8's, so this is a straight re-export — no neutralization needed (FULL_COMMIT
etc are all present).

This strict base ports the case8 single-source mission family WITHOUT the
cross-turn STAY/ACCUMULATE state machine (build_stay_holds / build_accumulate),
which need host-side consecutive-hold counters; see agent_jax for the wiring
status. case7's submission config is left untouched (cross-case independence).
"""

from __future__ import annotations

from ...baseline.core.config import *  # noqa: F401,F403
