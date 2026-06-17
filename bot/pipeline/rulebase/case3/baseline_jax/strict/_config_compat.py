"""Config shim: case3 constants + neutralized case8-only FULL_COMMIT flags.

case3 = case2 + rollout. The strict JAX modules are copied from case2's strict
port (case8 strategy + case1 geometric aim); case3's config (with ROLLOUT_* etc)
is the source of truth. Like case2, case3's preferred_send has no FULL_COMMIT
escalation, so those two case8-only constants are neutralized here.

NOTE: this strict base does NOT yet apply case3's rollout_reorder (the shallow
true2p mission re-ordering); see _config_compat usage and the agent docstring.
case3's submission config is left untouched (cross-case independence rule).
"""

from __future__ import annotations

# Re-export every case3 constant (resolves to case3/baseline/core/config.py).
from ...baseline.core.config import *  # noqa: F401,F403

# case8-only FULL_COMMIT, neutralized (case3's preferred_send never full-commits).
FULL_COMMIT_THRESHOLD_SHIPS: int = 1_000_000
FULL_COMMIT_FRACTION: float = 2.0
