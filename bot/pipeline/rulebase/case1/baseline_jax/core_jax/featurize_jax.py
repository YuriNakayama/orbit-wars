"""Per-target features for the mission score chain (JAX-native).

These mirror WorldModel scalars that target_value/opening_filter/preferred_send
read: reaction_times (min travel_time from my/enemy planets), is_static, and the
safe/contested-neutral classifications. Fixed-shape over the padded planet
arrays; the mission scorer composes these.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .geometry_jax import safe_angle_and_distance
from .physics_jax import fleet_speed

Arr = jax.Array

SAFE_NEUTRAL_MARGIN = 2
CONTESTED_NEUTRAL_MARGIN = 2
BIG_T = 10**9


def _travel_time(
    sx: Arr, sy: Arr, sr: Arr, tx: Arr, ty: Arr, tr: Arr, ships: Arr
) -> Arr:
    """travel_time = estimate_arrival(...).turns, or BIG_T if unsafe (sun)."""
    _angle, total_d, safe = safe_angle_and_distance(sx, sy, sr, tx, ty, tr)
    speed = fleet_speed(jnp.maximum(ships, 1))
    turns = jnp.maximum(1, jnp.ceil(total_d / speed)).astype(jnp.int32)
    return jnp.where(safe, turns, BIG_T)


def reaction_times(
    target_x: Arr,
    target_y: Arr,
    target_r: Arr,
    planet_x: Arr,
    planet_y: Arr,
    planet_r: Arr,
    planet_ships: Arr,
    is_mine: Arr,
    is_enemy: Arr,
) -> tuple[Arr, Arr]:
    """(my_t, enemy_t) = min travel_time from my / enemy planets to the target.

    planet_* are the full padded planet arrays [P]; is_mine/is_enemy are bool
    masks. Non-eligible sources are masked to BIG_T before the min. Mirrors
    WorldModel.reaction_times default=10**9 when a side has no planets.
    """
    tt = jax.vmap(
        lambda px, py, pr, ps: _travel_time(
            px, py, pr, target_x, target_y, target_r, jnp.maximum(ps, 1)
        )
    )(planet_x, planet_y, planet_r, planet_ships)
    my_t = jnp.min(jnp.where(is_mine, tt, BIG_T))
    enemy_t = jnp.min(jnp.where(is_enemy, tt, BIG_T))
    return my_t, enemy_t


def is_safe_neutral(owner: Arr, my_t: Arr, enemy_t: Arr) -> Arr:
    """owner==-1 and my_t <= enemy_t - SAFE_NEUTRAL_MARGIN."""
    return (owner == -1) & (my_t <= enemy_t - SAFE_NEUTRAL_MARGIN)


def is_contested_neutral(owner: Arr, my_t: Arr, enemy_t: Arr) -> Arr:
    """owner==-1 and |my_t - enemy_t| <= CONTESTED_NEUTRAL_MARGIN."""
    return (owner == -1) & (jnp.abs(my_t - enemy_t) <= CONTESTED_NEUTRAL_MARGIN)
