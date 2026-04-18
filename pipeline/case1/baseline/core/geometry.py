# Adapted from "Orbit Wars 2026 - Reinforce" by sigmaborov
# https://www.kaggle.com/code/sigmaborov/lb-897-orbit-wars-2026-reinforce
# Licensed under Apache License 2.0
"""2D geometry helpers (distance, sun-avoidance, launch point)."""

from __future__ import annotations

import math

from .config import (
    CENTER_X,
    CENTER_Y,
    LAUNCH_CLEARANCE,
    SUN_R,
    SUN_SAFETY,
)
from .types import Planet


def dist(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def orbital_radius(planet: Planet) -> float:
    return dist(planet.x, planet.y, CENTER_X, CENTER_Y)


def point_to_segment_distance(
    px: float, py: float, x1: float, y1: float, x2: float, y2: float
) -> float:
    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq <= 1e-9:
        return dist(px, py, x1, y1)
    t = ((px - x1) * dx + (py - y1) * dy) / seg_len_sq
    t = max(0.0, min(1.0, t))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return dist(px, py, proj_x, proj_y)


def segment_hits_sun(
    x1: float, y1: float, x2: float, y2: float, safety: float = SUN_SAFETY
) -> bool:
    return (
        point_to_segment_distance(CENTER_X, CENTER_Y, x1, y1, x2, y2) < SUN_R + safety
    )


def launch_point(sx: float, sy: float, sr: float, angle: float) -> tuple[float, float]:
    clearance = sr + LAUNCH_CLEARANCE
    return sx + math.cos(angle) * clearance, sy + math.sin(angle) * clearance


def actual_path_geometry(
    sx: float, sy: float, sr: float, tx: float, ty: float, tr: float
) -> tuple[float, float, float, float, float, float]:
    angle = math.atan2(ty - sy, tx - sx)
    start_x, start_y = launch_point(sx, sy, sr, angle)
    hit_distance = max(0.0, dist(sx, sy, tx, ty) - (sr + LAUNCH_CLEARANCE) - tr)
    end_x = start_x + math.cos(angle) * hit_distance
    end_y = start_y + math.sin(angle) * hit_distance
    return angle, start_x, start_y, end_x, end_y, hit_distance


def safe_angle_and_distance(
    sx: float, sy: float, sr: float, tx: float, ty: float, tr: float
) -> tuple[float, float] | None:
    angle, start_x, start_y, end_x, end_y, hit_distance = actual_path_geometry(
        sx, sy, sr, tx, ty, tr
    )
    if segment_hits_sun(start_x, start_y, end_x, end_y):
        return None
    return angle, hit_distance
