"""Convert an `EnvState` back to vendor-compatible obs dict.

Called outside jit by rollout code. Produces Python lists in the same row
layout the vendor sim emits.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .state import EnvState


def state_to_obs(state: EnvState, player: int) -> dict[str, Any]:
    """Return ``{'planets', 'fleets', 'player', 'step', 'angular_velocity'}``.

    Planet rows: ``[id, owner, y, x, radius, ships, prod]`` (vendor row 105).
    Fleet rows:  ``[id, owner, x, y, angle, from_planet_id, ships]``.
    """
    planet_valid = np.asarray(state.planet_valid)
    planet_id = np.asarray(state.planet_id)
    planet_owner = np.asarray(state.planet_owner)
    planet_xy = np.asarray(state.planet_xy)
    planet_radius = np.asarray(state.planet_radius)
    planet_ships = np.asarray(state.planet_ships)
    planet_prod = np.asarray(state.planet_prod)

    planets: list[list[float | int]] = []
    for i in range(planet_valid.shape[0]):
        if not bool(planet_valid[i]):
            continue
        planets.append(
            [
                int(planet_id[i]),
                int(planet_owner[i]),
                float(planet_xy[i, 0]),
                float(planet_xy[i, 1]),
                float(planet_radius[i]),
                int(planet_ships[i]),
                int(planet_prod[i]),
            ]
        )

    fleet_valid = np.asarray(state.fleet_valid)
    fleet_owner = np.asarray(state.fleet_owner)
    fleet_xy = np.asarray(state.fleet_xy)
    fleet_angle = np.asarray(state.fleet_angle)
    fleet_from_pid = np.asarray(state.fleet_from_pid)
    fleet_ships = np.asarray(state.fleet_ships)

    fleets: list[list[float | int]] = []
    for i in range(fleet_valid.shape[0]):
        if not bool(fleet_valid[i]):
            continue
        fleets.append(
            [
                i,  # fleet id is slot index (vendor monotonic id not preserved)
                int(fleet_owner[i]),
                float(fleet_xy[i, 0]),
                float(fleet_xy[i, 1]),
                float(fleet_angle[i]),
                int(fleet_from_pid[i]),
                int(fleet_ships[i]),
            ]
        )

    return {
        "planets": planets,
        "fleets": fleets,
        "player": int(player),
        "step": int(state.step),
        "angular_velocity": float(state.angular_velocity),
    }


__all__ = ["state_to_obs"]
