"""Config shim: case1 constants + neutralized case8-only flags.

The strict JAX modules were copied from case8/baseline_jax, which import config
constants from ``baseline.core.config``. case1's config is the source of truth
for this port, but it lacks a handful of case8-only constants (HARASS, FULL_COMMIT,
the half-step intercept flag, the dynamic-horizon flag). Rather than pollute
case1's submission-bound ``baseline/core/config.py`` (cross-case independence rule),
this module re-exports every case1 constant and defines the missing case8-only
names with values that **reduce the case8 JAX code back to case1 behavior**:

* ``HARASS_ENABLED = False`` — case1 has no harass family; the harass grid emits
  nothing. The remaining ``HARASS_*`` values are dummy placeholders (never read
  when disabled, but required for the import to resolve).
* ``FULL_COMMIT_THRESHOLD_SHIPS`` huge + ``FULL_COMMIT_FRACTION > 1`` — case1's
  ``preferred_send`` has no full-commit override; both guards make the case8
  branch ``src_available >= THRESHOLD and total >= src*FRACTION`` never fire.
* ``FINISHING_TIE_GUARD = False`` — case1 has no ``_avoid_tie_total`` logic.
* ``SAFE_INTERCEPT_HALF_STEP = False`` — case1's ``search_safe_intercept`` walks
  integer turn steps (case8 added half steps). This MUST be False for angle/turns
  parity.
* ``DYNAMIC_PROACTIVE_HORIZON_ENABLED = False`` — case1 uses a fixed horizon.

The strict modules import config exclusively through this module.
"""

from __future__ import annotations

# Re-export every case1 constant (the real source of truth for this port).
from ...baseline.core.config import *  # noqa: F401,F403

# --- case8-only constants, neutralized to case1 behavior ----------------------

# HARASS family: case1 has none. Disable so the harass grid produces no cells.
HARASS_ENABLED: bool = False
HARASS_MIN_TARGET_PRODUCTION: int = 2
HARASS_MIN_TARGET_SHIPS: int = 1
HARASS_MAX_TRAVEL_TURNS: int = 20
HARASS_MIN_SRC_RESERVE: int = 10
HARASS_COST_TURN_WEIGHT: float = 0.5
HARASS_VALUE_MULT: float = 1.0
HARASS_PRODUCTION_STEAL_TURNS: int = 5

# FULL_COMMIT override: case1's preferred_send never escalates to full inventory.
# THRESHOLD huge + FRACTION > 1 → the case8 branch can never trigger.
FULL_COMMIT_THRESHOLD_SHIPS: int = 1_000_000
FULL_COMMIT_FRACTION: float = 2.0

# FINISHING_TIE_GUARD: case1 has no _avoid_tie_total path.
FINISHING_TIE_GUARD: bool = False

# Intercept search: case1 walks integer turn steps (no half-step refinement).
SAFE_INTERCEPT_HALF_STEP: bool = False

# Proactive horizon: case1 uses a fixed horizon, not the dynamic case8 branch.
DYNAMIC_PROACTIVE_HORIZON_ENABLED: bool = False
DYNAMIC_PROACTIVE_HORIZON_MAX: int = 24
DYNAMIC_PROACTIVE_HORIZON_PROD_STEP: int = 3
DYNAMIC_PROACTIVE_HORIZON_PROD_CAP: int = 5
