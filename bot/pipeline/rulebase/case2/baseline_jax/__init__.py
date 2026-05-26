# Adapted from "Orbit Wars 2026 - Reinforce" by sigmaborov
# https://www.kaggle.com/code/sigmaborov/lb-897-orbit-wars-2026-reinforce
# Licensed under Apache License 2.0
"""JAX port of the rulebase/case2 core numeric modules (geometry / physics).

Mirrors `baseline/core/{geometry,physics}.py`. Pure, jit/vmap-friendly
helpers; non-numeric / variable-length control flow stays on the host (see
each module docstring for the deferred-feature list).
"""

from __future__ import annotations

from .geometry_jax import (
    actual_path_geometry_jax,
    dist_jax,
    launch_point_jax,
    point_to_segment_distance_jax,
    safe_angle_and_distance_jax,
    segment_hits_sun_jax,
)
from .physics_jax import (
    BIG_TURNS,
    estimate_arrival_jax,
    fleet_speed_jax,
    is_static_planet_jax,
    predict_planet_position_jax,
    travel_time_jax,
)

__all__ = [
    "BIG_TURNS",
    "actual_path_geometry_jax",
    "dist_jax",
    "estimate_arrival_jax",
    "fleet_speed_jax",
    "is_static_planet_jax",
    "launch_point_jax",
    "point_to_segment_distance_jax",
    "predict_planet_position_jax",
    "safe_angle_and_distance_jax",
    "segment_hits_sun_jax",
    "travel_time_jax",
]
