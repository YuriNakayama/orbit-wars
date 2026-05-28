# Adapted from "Orbit Wars 2026 - Reinforce" by sigmaborov
# https://www.kaggle.com/code/sigmaborov/lb-897-orbit-wars-2026-reinforce
# Licensed under Apache License 2.0
"""Host-side resolver that mirrors `WorldModel.__init__` as fixed-shape arrays.

`build_world_features(obs) -> WorldFeatures` parses the kaggle observation dict
(the same one `baseline/agent.build_world` consumes) and assembles every
per-planet field and scalar that `WorldModel.__init__` + `build_modes` compute,
as `MAX_PLANETS`-padded JAX arrays with a `planet_valid` mask. The output is the
fixed-shape input downstream grid ops (`scoring_jax`) and `vmap` over
`simulate_planet_timeline_jax` consume.

Resolution strategy
-------------------
The ragged obs lists, the arrival ledger (`build_arrival_ledger` /
`fleet_target_planet`), and comet bucketing are computed on the HOST in plain
Python/NumPy reusing the Python oracle semantics verbatim, then packed into JAX
arrays. The numeric helpers themselves (`is_static_planet_jax`,
`indirect_wealth_jax`, `travel_time_jax`) run on-device so they stay in lockstep
with the rest of the JAX port.

Arrival-table representation
----------------------------
Per planet we store the full per-turn arrival ledger as
`arr_owner / arr_ships / arr_valid` of shape
`(MAX_PLANETS, HORIZON + 1, MAX_ARRIVALS_PER_TURN)`. Each planet's slice
`arr_*[i]` is exactly the `(HORIZON+1, MAX_ARRIVALS_PER_TURN)` table that
`simulate_planet_timeline_jax` consumes, so `vmap`-ing the timeline over planets
reproduces `WorldModel.base_timeline`. Bucketing follows `normalize_arrivals`:
drop `ships <= 0` and `eta > HORIZON`, `eta = max(1, ceil(turns))`; turn 0 unused.

Dead branches (NOT ported, verified in `core/config.py`)
-------------------------------------------------------
* `OPPONENT_MODEL_ENABLED == False`  -> `predicted_arrivals` empty: the ledger is
  built from real fleets only (`build_arrival_ledger`), no prediction merge.
* `LOOKAHEAD_ENABLED == False`       -> no lookahead probe arrivals.
Both are confirmed disabled, so `build_world` passes empty dicts and the merge in
`WorldModel.__init__` reduces to `self.arrivals_by_planet[planet.id]`.

`reaction_times` is ported as a `(P_src, P_tgt)` `travel_time_jax` grid masked by
owner, taking the per-target min (sentinel `BIG_REACTION == 1e9` when no planet
of that owner exists), matching the Python `default=10**9`.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ..baseline.core.config import (
    AHEAD_ATTACK_MARGIN_BONUS,
    AHEAD_DOMINATION,
    BEHIND_ATTACK_MARGIN_PENALTY,
    BEHIND_DOMINATION,
    EARLY_TURN_LIMIT,
    FINISHING_ATTACK_MARGIN_BONUS,
    FINISHING_DOMINATION,
    FINISHING_PROD_RATIO,
    HORIZON,
    LATE_REMAINING_TURNS,
    OPENING_TURN_LIMIT,
    TOTAL_STEPS,
    VERY_LATE_REMAINING_TURNS,
)
from ..baseline.core.physics import comet_remaining_life
from ..baseline.core.types import Fleet, Planet
from ..baseline.core.world_model import build_arrival_ledger, normalize_arrivals
from .physics_jax import is_static_planet_jax, travel_time_jax
from .timeline_jax import (
    MAX_ARRIVALS_PER_TURN,
    MAX_PLANETS,
    indirect_wealth_jax,
)

# Sentinel for "no planet of this owner" reaction time. Mirrors the Python
# `default=10**9` in `WorldModel.reaction_times`.
BIG_REACTION: float = 1e9

# Owners are exactly {0, 1, 2, 3}; owner_strength is keyed on these.
NUM_OWNERS: int = 4


class WorldFeatures(NamedTuple):
    """Fixed-shape JAX mirror of `WorldModel.__init__` + `build_modes`.

    Per-planet arrays are `MAX_PLANETS`-padded; `planet_valid` masks real slots.
    Planets keep obs order in slots `0..len(planets)-1`.
    """

    # Per-planet (MAX_PLANETS)
    planet_id: jax.Array  # int32
    owner: jax.Array  # int32 (-1 neutral)
    xy: jax.Array  # float32[MAX_PLANETS, 2]
    radius: jax.Array  # float32
    ships: jax.Array  # int32
    prod: jax.Array  # int32
    planet_valid: jax.Array  # bool
    is_static: jax.Array  # bool
    is_comet: jax.Array  # bool
    comet_life: jax.Array  # int32
    indirect_wealth: jax.Array  # float32
    initial_xy: jax.Array  # float32[MAX_PLANETS, 2]
    initial_radius: jax.Array  # float32
    reaction_my_t: jax.Array  # float32 (BIG_REACTION sentinel)
    reaction_enemy_t: jax.Array  # float32 (BIG_REACTION sentinel)

    # Per-planet arrival ledger (MAX_PLANETS, HORIZON+1, MAX_ARRIVALS_PER_TURN)
    arr_owner: jax.Array  # int32
    arr_ships: jax.Array  # int32
    arr_valid: jax.Array  # bool

    # Scalars
    step: jax.Array  # int32
    player: jax.Array  # int32
    ang_vel: jax.Array  # float32
    num_players: jax.Array  # int32
    remaining_steps: jax.Array  # int32
    is_early: jax.Array  # bool
    is_opening: jax.Array  # bool
    is_late: jax.Array  # bool
    is_very_late: jax.Array  # bool
    is_four_player: jax.Array  # bool
    my_total: jax.Array  # int32
    enemy_total: jax.Array  # int32
    max_enemy_strength: jax.Array  # int32
    my_prod: jax.Array  # int32
    enemy_prod: jax.Array  # int32
    static_neutral_count: jax.Array  # int32
    owner_strength: jax.Array  # int32[NUM_OWNERS]
    horizon: jax.Array  # int32

    # modes (build_modes)
    domination: jax.Array  # float32
    is_behind: jax.Array  # bool
    is_ahead: jax.Array  # bool
    is_dominating: jax.Array  # bool
    is_finishing: jax.Array  # bool
    attack_margin_mult: jax.Array  # float32


def _read(obs: Any, key: str, default: Any) -> Any:
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _parse_obs(
    obs: Any,
) -> tuple[
    int,
    int,
    list[Planet],
    list[Fleet],
    float,
    dict[int, Planet],
    list[dict[str, Any]],
    set[int],
]:
    player = int(_read(obs, "player", 0) or 0)
    step = int(_read(obs, "step", 0) or 0)
    raw_planets = _read(obs, "planets", []) or []
    raw_fleets = _read(obs, "fleets", []) or []
    ang_vel = float(_read(obs, "angular_velocity", 0.0) or 0.0)
    raw_init = _read(obs, "initial_planets", []) or []
    comets: list[dict[str, Any]] = _read(obs, "comets", []) or []
    comet_ids = {int(cid) for cid in (_read(obs, "comet_planet_ids", []) or [])}

    planets = [Planet(*planet) for planet in raw_planets]
    fleets = [Fleet(*fleet) for fleet in raw_fleets]
    initial_by_id = {planet.id: planet for planet in (Planet(*p) for p in raw_init)}
    return player, step, planets, fleets, ang_vel, initial_by_id, comets, comet_ids


def _bucket_planet_arrivals(
    arrivals: list[tuple[int, int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bucket one planet's ledger into (HORIZON+1, K) owner/ship/valid tables.

    Mirrors `simulate_planet_timeline`'s `normalize_arrivals` + `by_turn`.
    """
    owner = np.full((HORIZON + 1, MAX_ARRIVALS_PER_TURN), -1, dtype=np.int32)
    ships = np.zeros((HORIZON + 1, MAX_ARRIVALS_PER_TURN), dtype=np.int32)
    valid = np.zeros((HORIZON + 1, MAX_ARRIVALS_PER_TURN), dtype=bool)
    counts = np.zeros(HORIZON + 1, dtype=np.int32)
    for eta, own, shp in normalize_arrivals(arrivals, HORIZON):
        slot = int(counts[eta])
        if slot >= MAX_ARRIVALS_PER_TURN:
            continue
        owner[eta, slot] = own
        ships[eta, slot] = int(shp)
        valid[eta, slot] = True
        counts[eta] += 1
    return owner, ships, valid


def _count_players(planets: list[Planet], fleets: list[Fleet]) -> int:
    owners: set[int] = {p.owner for p in planets if p.owner != -1}
    owners.update(f.owner for f in fleets)
    return max(2, len(owners))


def build_world_features(obs: Any) -> WorldFeatures:
    """Resolve the kaggle obs into fixed-shape `WorldFeatures` JAX arrays."""
    (
        player,
        step,
        planets,
        fleets,
        ang_vel,
        initial_by_id,
        comets,
        comet_ids,
    ) = _parse_obs(obs)

    n = len(planets)

    # --- padded per-planet host arrays ---
    pid = np.zeros(MAX_PLANETS, dtype=np.int32)
    owner = np.full(MAX_PLANETS, -1, dtype=np.int32)
    xs = np.zeros(MAX_PLANETS, dtype=np.float32)
    ys = np.zeros(MAX_PLANETS, dtype=np.float32)
    radius = np.zeros(MAX_PLANETS, dtype=np.float32)
    ships = np.zeros(MAX_PLANETS, dtype=np.int32)
    prod = np.zeros(MAX_PLANETS, dtype=np.int32)
    valid = np.zeros(MAX_PLANETS, dtype=bool)
    is_comet = np.zeros(MAX_PLANETS, dtype=bool)
    comet_life = np.zeros(MAX_PLANETS, dtype=np.int32)
    init_x = np.zeros(MAX_PLANETS, dtype=np.float32)
    init_y = np.zeros(MAX_PLANETS, dtype=np.float32)
    init_r = np.zeros(MAX_PLANETS, dtype=np.float32)

    for i, planet in enumerate(planets):
        pid[i] = planet.id
        owner[i] = planet.owner
        xs[i] = planet.x
        ys[i] = planet.y
        radius[i] = planet.radius
        ships[i] = int(planet.ships)
        prod[i] = int(planet.production)
        valid[i] = True
        is_comet[i] = planet.id in comet_ids
        comet_life[i] = comet_remaining_life(planet.id, comets)
        init = initial_by_id.get(planet.id)
        init_x[i] = init.x if init is not None else planet.x
        init_y[i] = init.y if init is not None else planet.y
        init_r[i] = init.radius if init is not None else planet.radius

    # --- arrival ledger -> per-planet bucketed tables ---
    ledger = build_arrival_ledger(fleets, planets)
    arr_owner = np.full(
        (MAX_PLANETS, HORIZON + 1, MAX_ARRIVALS_PER_TURN), -1, dtype=np.int32
    )
    arr_ships = np.zeros(
        (MAX_PLANETS, HORIZON + 1, MAX_ARRIVALS_PER_TURN), dtype=np.int32
    )
    arr_valid = np.zeros((MAX_PLANETS, HORIZON + 1, MAX_ARRIVALS_PER_TURN), dtype=bool)
    for i, planet in enumerate(planets):
        a_owner, a_ships, a_valid = _bucket_planet_arrivals(ledger[planet.id])
        arr_owner[i] = a_owner
        arr_ships[i] = a_ships
        arr_valid[i] = a_valid

    # --- owner strength / production (owners 0..3) ---
    owner_strength = np.zeros(NUM_OWNERS, dtype=np.int64)
    owner_production = np.zeros(NUM_OWNERS, dtype=np.int64)
    for planet in planets:
        if planet.owner != -1:
            owner_strength[planet.owner] += int(planet.ships)
            owner_production[planet.owner] += int(planet.production)
    for fleet in fleets:
        owner_strength[fleet.owner] += int(fleet.ships)

    my_total = int(owner_strength[player])
    enemy_total = int(owner_strength.sum() - owner_strength[player])
    enemy_mask = np.arange(NUM_OWNERS) != player
    max_enemy_strength = int(owner_strength[enemy_mask].max(initial=0))
    my_prod = int(owner_production[player])
    enemy_prod = int(owner_production.sum() - owner_production[player])

    num_players = _count_players(planets, fleets)
    remaining_steps = max(1, TOTAL_STEPS - step)

    # --- on-device numeric features ---
    xs_j = jnp.asarray(xs)
    ys_j = jnp.asarray(ys)
    radius_j = jnp.asarray(radius)
    owner_j = jnp.asarray(owner)
    prod_f_j = jnp.asarray(prod.astype(np.float32))
    valid_j = jnp.asarray(valid)

    is_static_j = jnp.where(valid_j, is_static_planet_jax(xs_j, ys_j, radius_j), False)

    wealth_j = jax.vmap(
        indirect_wealth_jax, in_axes=(0, None, None, None, None, None, None)
    )(
        jnp.arange(MAX_PLANETS, dtype=jnp.int32),
        xs_j,
        ys_j,
        owner_j,
        prod_f_j,
        valid_j,
        jnp.int32(player),
    )
    wealth_j = jnp.where(valid_j, wealth_j, 0.0)

    reaction_my, reaction_enemy = _reaction_times(
        xs_j, ys_j, radius_j, owner_j, ships, valid_j, player
    )

    static_neutral_count = int(np.sum(valid & (owner == -1) & np.asarray(is_static_j)))

    # --- scalars / modes (host floats; matches build_modes exactly) ---
    is_early = step < EARLY_TURN_LIMIT
    is_opening = step < OPENING_TURN_LIMIT
    is_late = remaining_steps < LATE_REMAINING_TURNS
    is_very_late = remaining_steps < VERY_LATE_REMAINING_TURNS
    is_four_player = num_players >= 4

    domination = (my_total - enemy_total) / max(1, my_total + enemy_total)
    is_behind = domination < BEHIND_DOMINATION
    is_ahead = domination > AHEAD_DOMINATION
    is_dominating = is_ahead or (
        max_enemy_strength > 0 and my_total > max_enemy_strength * 1.25
    )
    is_finishing = (
        domination > FINISHING_DOMINATION
        and my_prod > enemy_prod * FINISHING_PROD_RATIO
        and step > 100
    )
    attack_margin_mult = 1.0
    if is_ahead:
        attack_margin_mult += AHEAD_ATTACK_MARGIN_BONUS
    if is_behind:
        attack_margin_mult -= BEHIND_ATTACK_MARGIN_PENALTY
    if is_finishing:
        attack_margin_mult += FINISHING_ATTACK_MARGIN_BONUS

    _ = n  # planets fill slots 0..n-1; n kept for readability of the contract.

    return WorldFeatures(
        planet_id=jnp.asarray(pid),
        owner=owner_j,
        xy=jnp.stack([xs_j, ys_j], axis=1),
        radius=radius_j,
        ships=jnp.asarray(ships),
        prod=jnp.asarray(prod),
        planet_valid=valid_j,
        is_static=is_static_j,
        is_comet=jnp.asarray(is_comet),
        comet_life=jnp.asarray(comet_life),
        indirect_wealth=wealth_j.astype(jnp.float32),
        initial_xy=jnp.stack([jnp.asarray(init_x), jnp.asarray(init_y)], axis=1),
        initial_radius=jnp.asarray(init_r),
        reaction_my_t=reaction_my,
        reaction_enemy_t=reaction_enemy,
        arr_owner=jnp.asarray(arr_owner),
        arr_ships=jnp.asarray(arr_ships),
        arr_valid=jnp.asarray(arr_valid),
        step=jnp.int32(step),
        player=jnp.int32(player),
        ang_vel=jnp.float32(ang_vel),
        num_players=jnp.int32(num_players),
        remaining_steps=jnp.int32(remaining_steps),
        is_early=jnp.bool_(is_early),
        is_opening=jnp.bool_(is_opening),
        is_late=jnp.bool_(is_late),
        is_very_late=jnp.bool_(is_very_late),
        is_four_player=jnp.bool_(is_four_player),
        my_total=jnp.int32(my_total),
        enemy_total=jnp.int32(enemy_total),
        max_enemy_strength=jnp.int32(max_enemy_strength),
        my_prod=jnp.int32(my_prod),
        enemy_prod=jnp.int32(enemy_prod),
        static_neutral_count=jnp.int32(static_neutral_count),
        owner_strength=jnp.asarray(owner_strength.astype(np.int32)),
        horizon=jnp.int32(HORIZON),
        domination=jnp.float32(domination),
        is_behind=jnp.bool_(is_behind),
        is_ahead=jnp.bool_(is_ahead),
        is_dominating=jnp.bool_(is_dominating),
        is_finishing=jnp.bool_(is_finishing),
        attack_margin_mult=jnp.float32(attack_margin_mult),
    )


def _reaction_times(
    xs: jax.Array,
    ys: jax.Array,
    radius: jax.Array,
    owner: jax.Array,
    ships_np: np.ndarray,
    valid: jax.Array,
    player: int,
) -> tuple[jax.Array, jax.Array]:
    """Per-target min travel_time from my / enemy planets (BIG_REACTION sentinel).

    Mirrors `WorldModel.reaction_times`: for each target, min over my (resp.
    enemy) planets of `travel_time(src->target, ships=max(1, src.ships))`. A
    target with no qualifying source planet gets `BIG_REACTION` (Python 10**9).
    """
    ships_clamped = jnp.asarray(np.maximum(1, ships_np).astype(np.int32))

    def src_to_targets(
        sx: jax.Array,
        sy: jax.Array,
        sr: jax.Array,
        s_ships: jax.Array,
        s_valid: jax.Array,
    ) -> jax.Array:
        # travel_time from one source to all targets (vector over targets).
        tt = travel_time_jax(sx, sy, sr, xs, ys, radius, s_ships).astype(jnp.float32)
        return jnp.where(s_valid, tt, BIG_REACTION)

    # (P_src, P_tgt) grid: each row is one source's travel_time to every target.
    grid = jax.vmap(src_to_targets, in_axes=(0, 0, 0, 0, 0))(
        xs, ys, radius, ships_clamped, valid
    )

    is_mine = valid & (owner == player)
    is_enemy = valid & (owner != -1) & (owner != player)

    my_grid = jnp.where(is_mine[:, None], grid, BIG_REACTION)
    enemy_grid = jnp.where(is_enemy[:, None], grid, BIG_REACTION)

    reaction_my = jnp.min(my_grid, axis=0)
    reaction_enemy = jnp.min(enemy_grid, axis=0)
    return reaction_my.astype(jnp.float32), reaction_enemy.astype(jnp.float32)


__all__ = [
    "BIG_REACTION",
    "NUM_OWNERS",
    "WorldFeatures",
    "build_world_features",
]
