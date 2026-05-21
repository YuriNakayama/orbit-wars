"""Constants for the JAX env. Mirror the vendor sim where applicable.

`MAX_*` are fixed-shape ceilings dictated by jit/vmap (the vendor sim is
unbounded). Overflow handling is documented in `docs/plans/jax-env/01-design.md`.
"""

from __future__ import annotations

# Physical / game constants — must match the vendor sim byte-for-byte.
BOARD_SIZE: float = 100.0
CENTER: float = BOARD_SIZE / 2.0
SUN_RADIUS: float = 10.0
ROTATION_RADIUS_LIMIT: float = 50.0
COMET_RADIUS: float = 1.0
COMET_PRODUCTION: int = 1
PLANET_CLEARANCE: int = 7
MIN_PLANET_GROUPS: int = 5
MAX_PLANET_GROUPS: int = 10
MIN_STATIC_GROUPS: int = 3
COMET_SPAWN_STEPS: tuple[int, ...] = (50, 150, 250, 350, 450)

# Default configuration values from the vendor sim's orbit_wars.json.
EPISODE_STEPS: int = 500
SHIP_SPEED_MAX: float = 6.0
COMET_SPEED: float = 4.0

# Fixed-shape ceilings — see design doc Phase 1.
MAX_PLANETS: int = 48
MAX_FLEETS: int = 512
MAX_COMETS: int = 5
MAX_COMET_PATH_LEN: int = 40
NUM_AGENTS_MAX: int = 4

# Sentinels for invalid slots inside the fixed-shape arrays.
INVALID_ID: int = -1
INVALID_OWNER: int = -1  # neutral and unused both use -1
