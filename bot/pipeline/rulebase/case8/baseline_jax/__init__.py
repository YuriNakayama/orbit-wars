# Adapted from "Orbit Wars 2026 - Reinforce" by sigmaborov
# https://www.kaggle.com/code/sigmaborov/lb-897-orbit-wars-2026-reinforce
# Licensed under Apache License 2.0
"""Fully-JAX rule-based agent (case8): faithful port of the case8 pipeline.

Unlike the per-source-argmax approximations in `case1/baseline_jax*`, this case
re-implements the *real* decision pipeline — mission scoring, `argsort(-score)`,
and the sequential greedy global allocator — as fixed-shape `lax.scan` so it
keeps the Python baseline's strength while being `jax.vmap`-able across games for
parallel GPU self-play.

This module bootstraps the numeric foundation (geometry / physics / aim), copied
from `case2/baseline_jax` (cross-case duplication per the case-independence rule)
and parity-tested against `baseline/core/{geometry,physics}.py`. Higher layers
(scoring / missions / allocator / movements) build on top in sibling modules.
"""

from __future__ import annotations

from .aim_jax import (
    MAX_COMET_PATH_LEN,
    aim_with_prediction_jax,
    resolve_comet_path,
)
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
    "MAX_COMET_PATH_LEN",
    "actual_path_geometry_jax",
    "aim_with_prediction_jax",
    "dist_jax",
    "estimate_arrival_jax",
    "fleet_speed_jax",
    "is_static_planet_jax",
    "launch_point_jax",
    "point_to_segment_distance_jax",
    "predict_planet_position_jax",
    "resolve_comet_path",
    "safe_angle_and_distance_jax",
    "segment_hits_sun_jax",
    "travel_time_jax",
]
