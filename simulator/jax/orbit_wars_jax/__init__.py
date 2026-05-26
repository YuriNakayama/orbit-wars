"""JAX-native Orbit Wars environment for high-throughput PPO rollouts.

Mirrors the official `simulator/python/orbit_wars_vendor/orbit_wars.py` behavior.
Reset uses CPU + Python `random` to keep RNG byte-equal with the vendor sim;
step is pure JAX (jit + vmap friendly).

See `docs/plans/jax-env/01-design.md`.
"""

from __future__ import annotations

from .constants import (
    BOARD_SIZE,
    CENTER,
    COMET_RADIUS,
    EPISODE_STEPS,
    MAX_COMETS,
    MAX_COMET_PATH_LEN,
    MAX_FLEETS,
    MAX_PLANETS,
    NUM_AGENTS_MAX,
    ROTATION_RADIUS_LIMIT,
    SUN_RADIUS,
)
from .state import EnvState

__all__ = [
    "BOARD_SIZE",
    "CENTER",
    "COMET_RADIUS",
    "EPISODE_STEPS",
    "EnvState",
    "MAX_COMET_PATH_LEN",
    "MAX_COMETS",
    "MAX_FLEETS",
    "MAX_PLANETS",
    "NUM_AGENTS_MAX",
    "ROTATION_RADIUS_LIMIT",
    "SUN_RADIUS",
]
