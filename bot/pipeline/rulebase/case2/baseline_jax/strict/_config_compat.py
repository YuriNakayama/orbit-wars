"""Config shim: case2 constants + neutralized case8-only FULL_COMMIT flags.

The case2 strict JAX modules are the case8 strategy/allocator port with case1's
GEOMETRIC aim (case2 is case1-lineage on aim, case8-lineage on strategy). case2's
config carries every constant the case8 modules read EXCEPT the FULL_COMMIT pair,
which case8 added to `preferred_send` and case2 never had (case2's preferred_send
ends at ``return min(src_available, total)`` with no full-commit escalation).

Neutralize FULL_COMMIT so the case8 `preferred_send_jax` branch
``src_available >= THRESHOLD and total >= src*FRACTION`` can never fire,
reducing it to case2's behavior. Everything else comes straight from case2's
config (HARASS_ENABLED=True, SAFE_INTERCEPT_HALF_STEP=True, FINISHING_TIE_GUARD=
False with the OM/LOOKAHEAD/COMET_NPV feature flags all off).

case2's submission config is left untouched (cross-case independence rule).
"""

from __future__ import annotations

# Re-export every case2 constant (the source of truth for this port).
from ...baseline.core.config import *  # noqa: F401,F403

# --- case8-only FULL_COMMIT, neutralized to case2 behavior --------------------
# case2's preferred_send has no full-commit escalation; THRESHOLD huge +
# FRACTION > 1 make the case8 branch unreachable.
FULL_COMMIT_THRESHOLD_SHIPS: int = 1_000_000
FULL_COMMIT_FRACTION: float = 2.0
