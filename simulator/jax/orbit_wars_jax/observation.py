"""Convert an `EnvState` back to vendor-compatible obs dict.

Called outside jit by rollout code. Produces Python lists in the same row
layout the vendor sim emits. Mirrors the obs schema produced by
`simulator/python/orbit_wars_vendor/orbit_wars.py:interpreter()`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .state import EnvState


def state_to_obs(state: EnvState, player: int) -> dict[str, Any]:
    """Return a vendor-compatible obs dict for `player`.

    Keys emitted (matching what the reinforce/case1 featurizer reads):
      planets             list of [id, owner, x, y, radius, ships, prod]
      fleets              list of [id, owner, x, y, angle, from_pid, ships]
      initial_planets     list of [id, owner, x_init, y_init, radius, ships, prod]
      comet_planet_ids    list of int (subset of planet ids that are comets)
      comets              list of group dicts
                          {"planet_ids": [int, ...],
                           "paths":      [[(x,y), ...], ...],
                           "path_index": int}
      player, step, angular_velocity
    """
    planet_valid = np.asarray(state.planet_valid)
    planet_id = np.asarray(state.planet_id)
    planet_owner = np.asarray(state.planet_owner)
    planet_xy = np.asarray(state.planet_xy)
    planet_initial_xy = np.asarray(state.planet_initial_xy)
    planet_radius = np.asarray(state.planet_radius)
    planet_ships = np.asarray(state.planet_ships)
    planet_prod = np.asarray(state.planet_prod)
    planet_is_comet = np.asarray(state.planet_is_comet)

    planets: list[list[float | int]] = []
    initial_planets: list[list[float | int]] = []
    comet_planet_ids: list[int] = []
    for i in range(planet_valid.shape[0]):
        if not bool(planet_valid[i]):
            continue
        pid = int(planet_id[i])
        radius = float(planet_radius[i])
        ships = int(planet_ships[i])
        prod = int(planet_prod[i])
        planets.append(
            [
                pid,
                int(planet_owner[i]),
                float(planet_xy[i, 0]),
                float(planet_xy[i, 1]),
                radius,
                ships,
                prod,
            ]
        )
        # Vendor `initial_planets` is a snapshot of planets at reset, except
        # for comet planets (added later). The initial owner/ships/prod for
        # comet planets is -1/0/0 in vendor; for non-comet planets the row
        # at reset matches `planets` with original xy.
        initial_planets.append(
            [
                pid,
                int(planet_owner[i]) if not bool(planet_is_comet[i]) else -1,
                float(planet_initial_xy[i, 0]),
                float(planet_initial_xy[i, 1]),
                radius,
                ships if not bool(planet_is_comet[i]) else 0,
                prod if not bool(planet_is_comet[i]) else 0,
            ]
        )
        if bool(planet_is_comet[i]):
            comet_planet_ids.append(pid)

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

    # comets is a list of groups, one per activated comet. Each group has
    #   planet_ids: the planet ids (one per quadrant) the comet occupies
    #   paths:      list of per-planet path lists [(x, y), ...]
    #   path_index: current index into each path
    comet_paths = np.asarray(state.comet_paths)
    comet_path_index = np.asarray(state.comet_path_index)
    comet_planet_slot = np.asarray(state.comet_planet_slot)
    comet_active = np.asarray(state.comet_active)
    comet_path_len = np.asarray(state.comet_path_len)

    comets: list[dict[str, Any]] = []
    for c in range(comet_paths.shape[0]):
        if not bool(comet_active[c]):
            continue
        path_len = int(comet_path_len[c])
        if path_len <= 0:
            continue
        group_pids: list[int] = []
        group_paths: list[list[tuple[float, float]]] = []
        for q in range(comet_paths.shape[1]):
            slot = int(comet_planet_slot[c, q])
            if slot < 0:
                continue
            pid = int(planet_id[slot])
            if pid < 0:
                continue
            path_xy = comet_paths[c, q, :path_len]
            group_pids.append(pid)
            group_paths.append([(float(x), float(y)) for x, y in path_xy])
        if not group_pids:
            continue
        comets.append(
            {
                "planet_ids": group_pids,
                "paths": group_paths,
                "path_index": int(comet_path_index[c]),
            }
        )

    return {
        "planets": planets,
        "fleets": fleets,
        "initial_planets": initial_planets,
        "comet_planet_ids": comet_planet_ids,
        "comets": comets,
        "player": int(player),
        "step": int(state.step),
        "angular_velocity": float(state.angular_velocity),
    }


__all__ = ["state_to_obs"]
