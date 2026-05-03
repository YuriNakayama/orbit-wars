"""Vendored copy of `kaggle_environments/envs/orbit_wars` (Apache-2.0).

See `simulator/python/LICENSE` and `simulator/python/NOTICE` for provenance.
"""

from __future__ import annotations

from .orbit_wars import (
    BOARD_SIZE,
    CENTER,
    COMET_PRODUCTION,
    COMET_RADIUS,
    COMET_SPAWN_STEPS,
    MAX_PLANET_GROUPS,
    MIN_PLANET_GROUPS,
    MIN_STATIC_GROUPS,
    PLANET_CLEARANCE,
    ROTATION_RADIUS_LIMIT,
    SUN_RADIUS,
    Fleet,
    Planet,
    agents,
    distance,
    generate_comet_paths,
    generate_planets,
    html_renderer,
    interpreter,
    point_to_segment_distance,
    random_agent,
    renderer,
    specification,
    starter_agent,
)

__all__ = [
    "BOARD_SIZE",
    "CENTER",
    "COMET_PRODUCTION",
    "COMET_RADIUS",
    "COMET_SPAWN_STEPS",
    "Fleet",
    "MAX_PLANET_GROUPS",
    "MIN_PLANET_GROUPS",
    "MIN_STATIC_GROUPS",
    "PLANET_CLEARANCE",
    "Planet",
    "ROTATION_RADIUS_LIMIT",
    "SUN_RADIUS",
    "agents",
    "distance",
    "generate_comet_paths",
    "generate_planets",
    "html_renderer",
    "interpreter",
    "point_to_segment_distance",
    "random_agent",
    "renderer",
    "specification",
    "starter_agent",
]
