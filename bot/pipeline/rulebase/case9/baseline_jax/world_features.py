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
from orbit_wars_jax.state import EnvState

from ..baseline.core.config import (
    AHEAD_ATTACK_MARGIN_BONUS,
    AHEAD_DOMINATION,
    BEHIND_ATTACK_MARGIN_PENALTY,
    BEHIND_DOMINATION,
    DYNAMIC_PROACTIVE_HORIZON_ENABLED,
    EARLY_TURN_LIMIT,
    FINISHING_ATTACK_MARGIN_BONUS,
    FINISHING_DOMINATION,
    FINISHING_PROD_RATIO,
    HORIZON,
    LATE_REMAINING_TURNS,
    MULTI_ENEMY_PROACTIVE_HORIZON,
    MULTI_ENEMY_PROACTIVE_RATIO,
    MULTI_ENEMY_STACK_WINDOW,
    OPENING_TURN_LIMIT,
    PROACTIVE_DEFENSE_HORIZON,
    PROACTIVE_DEFENSE_RATIO,
    TOTAL_STEPS,
    VERY_LATE_REMAINING_TURNS,
)
from ..baseline.core.config import HORIZON as _HORIZON_INT
from ..baseline.core.physics import comet_remaining_life
from ..baseline.core.types import Fleet, Planet
from ..baseline.core.world_model import build_arrival_ledger, normalize_arrivals
from .aim_jax import MAX_COMET_PATH_LEN, resolve_comet_path
from .physics_jax import fleet_speed_jax, is_static_planet_jax, travel_time_jax
from .plan_shot_jax import MAX_OTHER_COMETS, _COMET_RADIUS_DEFAULT
from .timeline_jax import (
    MAX_ARRIVALS_PER_TURN,
    MAX_PLANETS,
    indirect_wealth_jax,
    simulate_planet_timeline_jax,
)

# Sentinel for "no planet of this owner" reaction time. Mirrors the Python
# `default=10**9` in `WorldModel.reaction_times`.
BIG_REACTION: float = 1e9

# Owners are exactly {0, 1, 2, 3}; owner_strength is keyed on these.
NUM_OWNERS: int = 4

# Defense-buffer port relies on the static-horizon branch of
# `_dynamic_proactive_horizon`. The dynamic branch is NOT ported; assert the
# flag is disabled at import so re-enabling it fails loud instead of silently
# diverging from the Python oracle.
assert not DYNAMIC_PROACTIVE_HORIZON_ENABLED, (
    "DYNAMIC_PROACTIVE_HORIZON_ENABLED is True; the dynamic-horizon branch of "
    "_multi_enemy_proactive_keep is not ported to baseline_jax/world_features.py"
)


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

    # Defense buffers (WorldModel._compute_defense_buffers). My planets only;
    # non-mine slots are 0.
    reserve: jax.Array  # int32 min(ships, max(keep_needed, proactive_keep))
    available: jax.Array  # int32 max(0, ships - reserve)

    # Per-planet arrival ledger (MAX_PLANETS, HORIZON+1, MAX_ARRIVALS_PER_TURN)
    arr_owner: jax.Array  # int32
    arr_ships: jax.Array  # int32
    arr_valid: jax.Array  # bool

    # Host-resolved comet-path inputs for plan_shot_jax (the §12c pattern).
    # Per-planet target comet path + the shared other-comet sweep arrays.
    comet_path: jax.Array  # float32[MAX_PLANETS, MAX_COMET_PATH_LEN, 2]
    comet_path_index: jax.Array  # int32[MAX_PLANETS]
    comet_path_len: jax.Array  # int32[MAX_PLANETS] (0 == non-comet)
    plan_max_turns: jax.Array  # int32[MAX_PLANETS] (HORIZON, comet-life capped)
    other_paths: jax.Array  # float32[MAX_OTHER_COMETS, MAX_COMET_PATH_LEN, 2]
    other_path_index: jax.Array  # int32[MAX_OTHER_COMETS]
    other_path_len: jax.Array  # int32[MAX_OTHER_COMETS]
    other_planet_id: jax.Array  # int32[MAX_OTHER_COMETS]
    comet_radius: jax.Array  # float32 (scalar, _COMET_RADIUS_DEFAULT)

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


def _resolve_target_comet_paths(
    planets: list[Planet],
    comets: list[dict[str, Any]],
    comet_ids: set[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-planet target comet path + the plan_shot `max_turns` cap.

    Mirrors `aim_with_prediction`'s comet bracket: `max_turns = HORIZON`, and for
    a comet `max_turns = min(HORIZON, max(0, comet_life - 1))`. Non-comet slots
    get a zero path with `path_len == 0` (the orbital branch).
    """
    paths = np.zeros((MAX_PLANETS, MAX_COMET_PATH_LEN, 2), dtype=np.float32)
    path_index = np.zeros(MAX_PLANETS, dtype=np.int32)
    path_len = np.zeros(MAX_PLANETS, dtype=np.int32)
    max_turns = np.full(MAX_PLANETS, _HORIZON_INT, dtype=np.int32)
    for i, planet in enumerate(planets):
        path, pidx, plen = resolve_comet_path(planet.id, comets, comet_ids)
        paths[i] = np.asarray(path, dtype=np.float32)
        path_index[i] = pidx
        path_len[i] = plen
        if planet.id in comet_ids:
            life = comet_remaining_life(planet.id, comets)
            max_turns[i] = min(_HORIZON_INT, max(0, life - 1))
    return paths, path_index, path_len, max_turns


def _resolve_other_comets(
    comets: list[dict[str, Any]],
    comet_ids: set[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Host-resolve every active comet planet into fixed MAX_OTHER_COMETS slots.

    Mirrors `fleet_crosses_other_comet`'s sweep set; empty slots get
    `path_len == 0` and `planet_id == -1` (matches the test assembly pattern).
    """
    paths = np.zeros((MAX_OTHER_COMETS, MAX_COMET_PATH_LEN, 2), dtype=np.float32)
    path_index = np.zeros(MAX_OTHER_COMETS, dtype=np.int32)
    path_len = np.zeros(MAX_OTHER_COMETS, dtype=np.int32)
    planet_id = np.full(MAX_OTHER_COMETS, -1, dtype=np.int32)
    for slot, cid in enumerate(sorted(comet_ids)):
        if slot >= MAX_OTHER_COMETS:
            break
        path, pidx, plen = resolve_comet_path(cid, comets, comet_ids)
        paths[slot] = np.asarray(path, dtype=np.float32)
        path_index[slot] = pidx
        path_len[slot] = plen
        planet_id[slot] = cid
    return paths, path_index, path_len, planet_id


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

    # --- comet-path inputs for plan_shot_jax (host-resolved §12c pattern) ---
    (
        comet_path,
        comet_path_index,
        comet_path_len,
        plan_max_turns,
    ) = _resolve_target_comet_paths(planets, comets, comet_ids)
    (
        other_paths,
        other_path_index,
        other_path_len,
        other_planet_id,
    ) = _resolve_other_comets(comets, comet_ids)

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

    # --- defense buffers (WorldModel._compute_defense_buffers) ---
    ships_j = jnp.asarray(ships)
    arr_owner_j = jnp.asarray(arr_owner)
    arr_ships_j = jnp.asarray(arr_ships)
    arr_valid_j = jnp.asarray(arr_valid)
    keep_needed = _keep_needed_grid(
        owner_j,
        ships_j,
        jnp.asarray(prod),
        arr_owner_j,
        arr_ships_j,
        arr_valid_j,
        player,
    )
    proactive_keep = _proactive_keep_grid(
        xs_j, ys_j, radius_j, owner_j, ships, valid_j, player
    )
    is_mine_j = valid_j & (owner_j == player)
    reserve = jnp.where(
        is_mine_j,
        jnp.minimum(ships_j, jnp.maximum(keep_needed, proactive_keep)),
        jnp.int32(0),
    ).astype(jnp.int32)
    available = jnp.where(
        is_mine_j, jnp.maximum(0, ships_j - reserve), jnp.int32(0)
    ).astype(jnp.int32)

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
        ships=ships_j,
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
        reserve=reserve,
        available=available,
        arr_owner=arr_owner_j,
        arr_ships=arr_ships_j,
        arr_valid=arr_valid_j,
        comet_path=jnp.asarray(comet_path, dtype=jnp.float32),
        comet_path_index=jnp.asarray(comet_path_index, dtype=jnp.int32),
        comet_path_len=jnp.asarray(comet_path_len, dtype=jnp.int32),
        plan_max_turns=jnp.asarray(plan_max_turns, dtype=jnp.int32),
        other_paths=jnp.asarray(other_paths, dtype=jnp.float32),
        other_path_index=jnp.asarray(other_path_index, dtype=jnp.int32),
        other_path_len=jnp.asarray(other_path_len, dtype=jnp.int32),
        other_planet_id=jnp.asarray(other_planet_id, dtype=jnp.int32),
        comet_radius=jnp.float32(_COMET_RADIUS_DEFAULT),
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


def _keep_needed_grid(
    owner: jax.Array,
    ships: jax.Array,
    prod: jax.Array,
    arr_owner: jax.Array,
    arr_ships: jax.Array,
    arr_valid: jax.Array,
    player: int,
) -> jax.Array:
    """Per-planet `keep_needed` via `simulate_planet_timeline_jax` vmap.

    Mirrors the `timeline["keep_needed"]` the Python `_compute_defense_buffers`
    reads from `base_timeline[planet.id]`. Returns int32[MAX_PLANETS]; non-owned
    planets get 0 (the timeline already zeroes them), padded slots included.
    """

    def one(
        p_owner: jax.Array,
        p_ships: jax.Array,
        p_prod: jax.Array,
        t_owner: jax.Array,
        t_ships: jax.Array,
        t_valid: jax.Array,
    ) -> jax.Array:
        tl = simulate_planet_timeline_jax(
            p_owner,
            p_ships.astype(jnp.float32),
            p_prod.astype(jnp.float32),
            t_owner,
            t_ships,
            t_valid,
            jnp.int32(player),
            HORIZON,
        )
        return tl.keep_needed

    return jax.vmap(one, in_axes=(0, 0, 0, 0, 0, 0))(
        owner, ships, prod, arr_owner, arr_ships, arr_valid
    ).astype(jnp.int32)


def _proactive_keep_one(
    eta_row: jax.Array,
    enemy_ships: jax.Array,
    enemy_is_threat: jax.Array,
) -> jax.Array:
    """Mirror `_multi_enemy_proactive_keep` for a single my-planet.

    `eta_row[P]` is `travel_time(enemy->this_planet)` for every enemy planet,
    `enemy_ships[P]` the raw `int(enemy.ships)` contribution, and
    `enemy_is_threat[P]` selects valid enemy planets whose `eta <= stack_horizon`
    (== MULTI_ENEMY_PROACTIVE_HORIZON, the static branch). Returns
    `max(proactive, legacy)` as an int32.

    The Python two-pointer keeps a contiguous eta-window of width
    `MULTI_ENEMY_STACK_WINDOW` on the sorted threat list, tracking the max
    running ship-sum. Because every maximal window is anchored at some left edge
    present in the list, the max equals the per-anchor sum
    `sum{ships[j] : 0 <= eta[j]-eta[i] <= WINDOW}` maximised over anchors `i`.
    We compute that O(P^2) masked form directly (P<=48), so no sort is needed.
    """
    stack_horizon = MULTI_ENEMY_PROACTIVE_HORIZON
    legacy_horizon = PROACTIVE_DEFENSE_HORIZON

    threat = enemy_is_threat & (eta_row <= stack_horizon)
    ships_i = jnp.where(threat, enemy_ships, 0).astype(jnp.int32)
    eta_i = eta_row.astype(jnp.float32)

    # window[a, j]: threat j lies in the eta-window anchored at threat a, i.e.
    # both are threats and 0 <= eta[j]-eta[a] <= WINDOW.
    delta = eta_i[None, :] - eta_i[:, None]
    in_window = (delta >= 0) & (delta <= MULTI_ENEMY_STACK_WINDOW)
    pair = in_window & threat[:, None] & threat[None, :]
    anchor_sums = jnp.sum(jnp.where(pair, ships_i[None, :], 0), axis=1)
    best_stacked = jnp.max(jnp.where(threat, anchor_sums, 0))
    proactive = jnp.floor(best_stacked * MULTI_ENEMY_PROACTIVE_RATIO).astype(jnp.int32)

    legacy_threat = threat & (eta_row <= legacy_horizon)
    legacy_each = jnp.floor(ships_i * PROACTIVE_DEFENSE_RATIO).astype(jnp.int32)
    legacy = jnp.max(jnp.where(legacy_threat, legacy_each, 0))

    has_threat = jnp.any(threat)
    return jnp.where(has_threat, jnp.maximum(proactive, legacy), jnp.int32(0))


def _proactive_keep_grid(
    xs: jax.Array,
    ys: jax.Array,
    radius: jax.Array,
    owner: jax.Array,
    ships_np: np.ndarray,
    valid: jax.Array,
    player: int,
) -> jax.Array:
    """Per-MY-planet `_multi_enemy_proactive_keep` over the enemy-planet grid.

    Builds the `(P_my, P_enemy)` eta grid `travel_time(enemy->my, ships=max(1,
    enemy.ships))` then reduces each row via `_proactive_keep_one`. Enemy planets
    are `owner not in (-1, player)`; sun-blocked etas come back as `BIG_TURNS`
    (1e9) and are filtered by the `eta <= 14` threat test.
    """
    enemy_ships_raw = jnp.asarray(ships_np.astype(np.int32))
    enemy_ships_tt = jnp.asarray(np.maximum(1, ships_np).astype(np.int32))
    is_enemy = valid & (owner != -1) & (owner != player)

    def enemy_to_targets(
        ex: jax.Array,
        ey: jax.Array,
        er: jax.Array,
        e_ships: jax.Array,
    ) -> jax.Array:
        # travel_time from one enemy to every my-planet (vector over targets).
        return travel_time_jax(ex, ey, er, xs, ys, radius, e_ships).astype(jnp.float32)

    # (P_enemy, P_my): row e = enemy e's eta to every planet.
    grid = jax.vmap(enemy_to_targets, in_axes=(0, 0, 0, 0))(
        xs, ys, radius, enemy_ships_tt
    )
    # Transpose to (P_my, P_enemy): row m = every enemy's eta to my-planet m.
    eta_my_by_enemy = grid.T

    def per_my_planet(eta_row: jax.Array) -> jax.Array:
        return _proactive_keep_one(eta_row, enemy_ships_raw, is_enemy)

    return jax.vmap(per_my_planet)(eta_my_by_enemy)


# ---------------------------------------------------------------------------
# Pure-JAX path: build_world_features_from_state(state, seat)
#
# Everything below computes the SAME WorldFeatures from a fixed-shape EnvState
# with no host-side Python list parsing or arrival loop, so the whole function
# is jax.jit-able and jax.vmap-able over a batch of EnvStates. The host path
# above (`build_world_features(obs)`) is left untouched.
#
# Coordinate convention: `state_to_obs` emits a planet obs row as
# `[id, owner, x=planet_xy[:, 0], y=planet_xy[:, 1], ...]`, so on the JAX side
# `xs = state.planet_xy[:, 0]` and `ys = state.planet_xy[:, 1]`, matching what
# the host featurizer reads as `planet.x` / `planet.y`.
# ---------------------------------------------------------------------------

# +inf-like sentinel for "no candidate planet" in the fleet-target argmin. Must
# exceed HORIZON so masked planets never win the argmin, and stay finite so the
# downstream `best_time <= HORIZON` test is exact.
_NO_TARGET_TURNS: float = 1e18


def _fleet_target_jax(
    fx: jax.Array,
    fy: jax.Array,
    f_angle: jax.Array,
    f_ships: jax.Array,
    f_valid: jax.Array,
    px: jax.Array,
    py: jax.Array,
    pradius: jax.Array,
    p_valid: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Pure-JAX `fleet_target_planet` for one fleet over all planets.

    Returns `(target_slot, eta, hit)`. `hit == False` mirrors the Python
    `(None, None)`. The ray-vs-circle nearest-hit logic and the `turns <=
    HORIZON` / strict-min tie-break (first planet in slot order) match
    `world_model.fleet_target_planet` exactly; `argmin` returns the first
    minimal index, matching the Python `turns < best_time` strict comparison.
    """
    dir_x = jnp.cos(f_angle)
    dir_y = jnp.sin(f_angle)
    speed = fleet_speed_jax(f_ships)

    dx = px - fx
    dy = py - fy
    proj = dx * dir_x + dy * dir_y
    perp_sq = dx * dx + dy * dy - proj * proj
    radius_sq = pradius * pradius
    hit_d = jnp.maximum(0.0, proj - jnp.sqrt(jnp.maximum(0.0, radius_sq - perp_sq)))
    turns = hit_d / speed

    eligible = (
        p_valid
        & f_valid
        & (proj >= 0.0)
        & (perp_sq < radius_sq)
        & (turns <= float(_HORIZON_INT))
    )
    scored = jnp.where(eligible, turns, _NO_TARGET_TURNS)
    best_slot = jnp.argmin(scored).astype(jnp.int32)
    best_time = scored[best_slot]
    hit = jnp.any(eligible)
    eta = jnp.maximum(1, jnp.ceil(best_time)).astype(jnp.int32)
    return best_slot, eta, hit


def _arrival_ledger_jax(
    fleet_xy: jax.Array,
    fleet_angle: jax.Array,
    fleet_ships: jax.Array,
    fleet_owner: jax.Array,
    fleet_valid: jax.Array,
    px: jax.Array,
    py: jax.Array,
    pradius: jax.Array,
    p_valid: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Build the per-(planet, turn) arrival tables in pure JAX.

    Returns `(arr_owner, arr_ships, arr_valid)` of shape
    `(MAX_PLANETS, HORIZON+1, MAX_ARRIVALS_PER_TURN)`, identical to the host
    `_bucket_planet_arrivals` output. Fleets are scattered in slot order (which
    equals obs order) with a per-(planet, turn) slot counter; overflow past
    `MAX_ARRIVALS_PER_TURN` is dropped exactly as the host does.

    Within-turn slot order: the host buckets `normalize_arrivals` (a stable sort
    by eta), so within a fixed eta fleets keep obs/slot order. Scattering fleets
    in slot order with a running counter reproduces that order bit-for-bit,
    including which arrivals overflow.
    """
    horizon_plus = _HORIZON_INT + 1

    # Per-fleet target slot + eta + hit (vmap over fleets).
    target_slot, eta, hit = jax.vmap(
        _fleet_target_jax, in_axes=(0, 0, 0, 0, 0, None, None, None, None)
    )(
        fleet_xy[:, 0],
        fleet_xy[:, 1],
        fleet_angle,
        fleet_ships,
        fleet_valid,
        px,
        py,
        pradius,
        p_valid,
    )
    # normalize_arrivals also drops ships <= 0; eta is already clamped >= 1 and
    # only kept when turns <= HORIZON (so eta <= HORIZON). Fleet validity and a
    # real hit gate the contribution.
    contributes = hit & fleet_valid & (fleet_ships > 0)

    init_owner = jnp.full(
        (MAX_PLANETS, horizon_plus, MAX_ARRIVALS_PER_TURN), -1, dtype=jnp.int32
    )
    init_ships = jnp.zeros(
        (MAX_PLANETS, horizon_plus, MAX_ARRIVALS_PER_TURN), dtype=jnp.int32
    )
    init_valid = jnp.zeros(
        (MAX_PLANETS, horizon_plus, MAX_ARRIVALS_PER_TURN), dtype=jnp.bool_
    )
    init_counts = jnp.zeros((MAX_PLANETS, horizon_plus), dtype=jnp.int32)

    Carry = tuple[jax.Array, jax.Array, jax.Array, jax.Array]
    Xs = tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]

    def body(carry: Carry, xs: Xs) -> tuple[Carry, None]:
        owner_t, ships_t, valid_t, counts = carry
        tgt, e, own, shp, ok = xs
        slot = counts[tgt, e]
        do = ok & (slot < MAX_ARRIVALS_PER_TURN)
        s_slot = jnp.where(do, slot, jnp.int32(0))
        new_owner = owner_t.at[tgt, e, s_slot].set(
            jnp.where(do, own, owner_t[tgt, e, s_slot])
        )
        new_ships = ships_t.at[tgt, e, s_slot].set(
            jnp.where(do, shp, ships_t[tgt, e, s_slot])
        )
        new_valid = valid_t.at[tgt, e, s_slot].set(
            jnp.where(do, True, valid_t[tgt, e, s_slot])
        )
        new_counts = counts.at[tgt, e].add(jnp.where(do, jnp.int32(1), jnp.int32(0)))
        return (new_owner, new_ships, new_valid, new_counts), None

    (arr_owner, arr_ships, arr_valid, _counts), _ = jax.lax.scan(
        body,
        (init_owner, init_ships, init_valid, init_counts),
        (target_slot, eta, fleet_owner, fleet_ships, contributes),
    )
    return arr_owner, arr_ships, arr_valid


def _comet_fields_from_state(
    state: EnvState,
) -> tuple[
    jax.Array,  # is_comet[P]
    jax.Array,  # comet_life[P]
    jax.Array,  # comet_path[P, L, 2]
    jax.Array,  # comet_path_index[P]
    jax.Array,  # comet_path_len[P]
    jax.Array,  # plan_max_turns[P]
    jax.Array,  # other_paths[MAX_OTHER_COMETS, L, 2]
    jax.Array,  # other_path_index[MAX_OTHER_COMETS]
    jax.Array,  # other_path_len[MAX_OTHER_COMETS]
    jax.Array,  # other_planet_id[MAX_OTHER_COMETS]
]:
    """Resolve every comet-path field from EnvState's fixed comet arrays.

    Replaces the host `resolve_comet_path` / `comet_remaining_life`: for each
    `(comet c, quadrant q)` pair the planet slot is `comet_planet_slot[c, q]`,
    its path is `comet_paths[c, q]` (already `(MAX_COMET_PATH_LEN, 2)`),
    `path_index = comet_path_index[c]`, `path_len = comet_path_len[c]`. The
    paths are emitted unswapped, matching `state_to_obs`'s comet `paths`.
    """
    max_comets = state.comet_planet_slot.shape[0]  # MAX_COMETS
    quads = state.comet_planet_slot.shape[1]  # 4
    flat_slot = state.comet_planet_slot.reshape(-1)  # (C*4,)
    flat_path = state.comet_paths.reshape(
        max_comets * quads, MAX_COMET_PATH_LEN, 2
    )  # (C*4, L, 2)
    comet_idx = jnp.repeat(jnp.arange(max_comets, dtype=jnp.int32), quads)  # (C*4,)
    flat_index = state.comet_path_index[comet_idx]  # (C*4,)
    flat_len = state.comet_path_len[comet_idx]  # (C*4,)
    flat_active = state.comet_active[comet_idx]  # (C*4,)
    flat_pid = state.planet_id[jnp.clip(flat_slot, 0, MAX_PLANETS - 1)]  # (C*4,)

    # A comet planet is one whose slot is valid, the comet is active, and the
    # path is non-empty. This mirrors `comet_planet_ids` in `state_to_obs`
    # (planet_is_comet & valid), bridged through comet bookkeeping.
    flat_is_comet = (flat_slot >= 0) & flat_active & (flat_len > 0)

    planet_idx = jnp.arange(MAX_PLANETS, dtype=jnp.int32)
    # (P, C*4): does (c, q) entry k reference planet slot p?
    match = (planet_idx[:, None] == flat_slot[None, :]) & flat_is_comet[None, :]
    is_comet = jnp.any(match, axis=1)  # (P,)
    pick = jnp.argmax(match.astype(jnp.int32), axis=1)  # (P,) first matching entry

    comet_path = jnp.where(is_comet[:, None, None], flat_path[pick], 0.0)
    comet_index_p = jnp.where(is_comet, flat_index[pick], jnp.int32(0))
    comet_len_p = jnp.where(is_comet, flat_len[pick], jnp.int32(0))
    # comet_remaining_life = max(0, path_len - path_index).
    raw_life = flat_len[pick] - flat_index[pick]
    comet_life = jnp.where(is_comet, jnp.maximum(0, raw_life), jnp.int32(0)).astype(
        jnp.int32
    )
    # plan_max_turns: non-comet -> HORIZON; comet -> min(HORIZON, max(0, life-1)).
    comet_cap = jnp.minimum(_HORIZON_INT, jnp.maximum(0, comet_life - 1))
    plan_max_turns = jnp.where(is_comet, comet_cap, jnp.int32(_HORIZON_INT)).astype(
        jnp.int32
    )

    # --- other comets: every active comet planet, by ascending planet id. ---
    # Host `_resolve_other_comets` iterates `sorted(comet_ids)`; we mirror that
    # ordering by sorting active comet-planet entries by their planet id.
    cap = MAX_OTHER_COMETS
    sort_key = jnp.where(flat_is_comet, flat_pid, jnp.int32(2**30))
    order = jnp.argsort(sort_key)  # ascending pid; empties pushed to the back
    sel = order[:cap]
    sel_is_comet = flat_is_comet[sel]
    other_paths = jnp.where(sel_is_comet[:, None, None], flat_path[sel], 0.0)
    other_path_index = jnp.where(sel_is_comet, flat_index[sel], jnp.int32(0))
    other_path_len = jnp.where(sel_is_comet, flat_len[sel], jnp.int32(0))
    other_planet_id = jnp.where(sel_is_comet, flat_pid[sel], jnp.int32(-1))

    return (
        is_comet,
        comet_life,
        comet_path,
        comet_index_p,
        comet_len_p,
        plan_max_turns,
        other_paths,
        other_path_index,
        other_path_len,
        other_planet_id,
    )


def _reaction_times_state(
    xs: jax.Array,
    ys: jax.Array,
    radius: jax.Array,
    owner: jax.Array,
    ships: jax.Array,
    valid: jax.Array,
    player: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Pure-JAX `_reaction_times` (per-target min travel_time, BIG sentinel)."""
    ships_clamped = jnp.maximum(1, ships).astype(jnp.int32)

    def src_to_targets(
        sx: jax.Array,
        sy: jax.Array,
        sr: jax.Array,
        s_ships: jax.Array,
        s_valid: jax.Array,
    ) -> jax.Array:
        tt = travel_time_jax(sx, sy, sr, xs, ys, radius, s_ships).astype(jnp.float32)
        return jnp.where(s_valid, tt, BIG_REACTION)

    grid = jax.vmap(src_to_targets, in_axes=(0, 0, 0, 0, 0))(
        xs, ys, radius, ships_clamped, valid
    )
    is_mine = valid & (owner == player)
    is_enemy = valid & (owner != -1) & (owner != player)
    my_grid = jnp.where(is_mine[:, None], grid, BIG_REACTION)
    enemy_grid = jnp.where(is_enemy[:, None], grid, BIG_REACTION)
    return (
        jnp.min(my_grid, axis=0).astype(jnp.float32),
        jnp.min(enemy_grid, axis=0).astype(jnp.float32),
    )


def _proactive_keep_grid_state(
    xs: jax.Array,
    ys: jax.Array,
    radius: jax.Array,
    owner: jax.Array,
    ships: jax.Array,
    valid: jax.Array,
    player: jax.Array,
) -> jax.Array:
    """Pure-JAX `_proactive_keep_grid` (takes a jnp ships array, not numpy)."""
    enemy_ships_raw = ships.astype(jnp.int32)
    enemy_ships_tt = jnp.maximum(1, ships).astype(jnp.int32)
    is_enemy = valid & (owner != -1) & (owner != player)

    def enemy_to_targets(
        ex: jax.Array,
        ey: jax.Array,
        er: jax.Array,
        e_ships: jax.Array,
    ) -> jax.Array:
        return travel_time_jax(ex, ey, er, xs, ys, radius, e_ships).astype(jnp.float32)

    grid = jax.vmap(enemy_to_targets, in_axes=(0, 0, 0, 0))(
        xs, ys, radius, enemy_ships_tt
    )
    eta_my_by_enemy = grid.T

    def per_my_planet(eta_row: jax.Array) -> jax.Array:
        return _proactive_keep_one(eta_row, enemy_ships_raw, is_enemy)

    return jax.vmap(per_my_planet)(eta_my_by_enemy)


def _count_players_jax(
    owner: jax.Array,
    valid: jax.Array,
    fleet_owner: jax.Array,
    fleet_valid: jax.Array,
) -> jax.Array:
    """Pure-JAX `count_players`: distinct owners over planets (!= -1) + fleets."""
    owners = jnp.arange(NUM_OWNERS, dtype=jnp.int32)
    planet_present = jax.vmap(lambda o: jnp.any(valid & (owner != -1) & (owner == o)))(
        owners
    )
    fleet_present = jax.vmap(lambda o: jnp.any(fleet_valid & (fleet_owner == o)))(
        owners
    )
    present = planet_present | fleet_present
    return jnp.maximum(2, jnp.sum(present.astype(jnp.int32))).astype(jnp.int32)


def build_world_features_from_state(state: EnvState, seat: int) -> WorldFeatures:
    """Pure-JAX `WorldFeatures` from an `EnvState` (jit- and vmap-able).

    Same output contract as `build_world_features(obs)`, computed entirely from
    fixed-shape EnvState arrays with no host-side parsing. `seat` is the player
    perspective (a static Python int; do not trace over it).
    """
    player = jnp.int32(seat)
    xs = state.planet_xy[:, 0]
    ys = state.planet_xy[:, 1]
    radius = state.planet_radius
    owner = state.planet_owner.astype(jnp.int32)
    valid = state.planet_valid
    ships = state.planet_ships.astype(jnp.int32)
    prod = state.planet_prod.astype(jnp.int32)
    pid = jnp.where(valid, state.planet_id, jnp.int32(0)).astype(jnp.int32)
    owner = jnp.where(valid, owner, jnp.int32(-1))

    # initial position: `state_to_obs` feeds `initial_planets` from
    # `planet_initial_xy` (x = [:, 0], y = [:, 1]); the host featurizer reads
    # init.x / init.y from that. For comet planets these are -99.0, matching.
    init_x = state.planet_initial_xy[:, 0]
    init_y = state.planet_initial_xy[:, 1]
    init_r = jnp.where(valid, radius, 0.0)

    is_static = jnp.where(valid, is_static_planet_jax(xs, ys, radius), False)

    wealth = jax.vmap(
        indirect_wealth_jax, in_axes=(0, None, None, None, None, None, None)
    )(
        jnp.arange(MAX_PLANETS, dtype=jnp.int32),
        xs,
        ys,
        owner,
        prod.astype(jnp.float32),
        valid,
        player,
    )
    wealth = jnp.where(valid, wealth, 0.0).astype(jnp.float32)

    reaction_my, reaction_enemy = _reaction_times_state(
        xs, ys, radius, owner, ships, valid, player
    )

    arr_owner, arr_ships, arr_valid = _arrival_ledger_jax(
        state.fleet_xy,
        state.fleet_angle,
        state.fleet_ships.astype(jnp.int32),
        state.fleet_owner.astype(jnp.int32),
        state.fleet_valid,
        xs,
        ys,
        radius,
        valid,
    )

    keep_needed = _keep_needed_grid(
        owner, ships, prod, arr_owner, arr_ships, arr_valid, seat
    )
    proactive_keep = _proactive_keep_grid_state(
        xs, ys, radius, owner, ships, valid, player
    )
    is_mine = valid & (owner == player)
    reserve = jnp.where(
        is_mine, jnp.minimum(ships, jnp.maximum(keep_needed, proactive_keep)), 0
    ).astype(jnp.int32)
    available = jnp.where(is_mine, jnp.maximum(0, ships - reserve), 0).astype(jnp.int32)

    (
        is_comet,
        comet_life,
        comet_path,
        comet_path_index,
        comet_path_len,
        plan_max_turns,
        other_paths,
        other_path_index,
        other_path_len,
        other_planet_id,
    ) = _comet_fields_from_state(state)

    # --- owner strength / production (owners 0..3) ---
    owners = jnp.arange(NUM_OWNERS, dtype=jnp.int32)
    planet_owned = valid & (owner != -1)

    def strength_for(o: jax.Array) -> jax.Array:
        p_ships = jnp.sum(jnp.where(planet_owned & (owner == o), ships, 0))
        f_ships = jnp.sum(
            jnp.where(
                state.fleet_valid & (state.fleet_owner == o), state.fleet_ships, 0
            )
        )
        return (p_ships + f_ships).astype(jnp.int32)

    def production_for(o: jax.Array) -> jax.Array:
        return jnp.sum(jnp.where(planet_owned & (owner == o), prod, 0)).astype(
            jnp.int32
        )

    owner_strength = jax.vmap(strength_for)(owners)
    owner_production = jax.vmap(production_for)(owners)

    my_total = owner_strength[player]
    enemy_total = jnp.sum(owner_strength) - my_total
    enemy_mask = owners != player
    max_enemy_strength = jnp.max(jnp.where(enemy_mask, owner_strength, 0))
    my_prod = owner_production[player]
    enemy_prod = jnp.sum(owner_production) - my_prod

    num_players = _count_players_jax(
        owner, valid, state.fleet_owner.astype(jnp.int32), state.fleet_valid
    )
    step = state.step.astype(jnp.int32)
    remaining_steps = jnp.maximum(1, TOTAL_STEPS - step)
    static_neutral_count = jnp.sum(
        (valid & (owner == -1) & is_static).astype(jnp.int32)
    ).astype(jnp.int32)

    is_early = step < EARLY_TURN_LIMIT
    is_opening = step < OPENING_TURN_LIMIT
    is_late = remaining_steps < LATE_REMAINING_TURNS
    is_very_late = remaining_steps < VERY_LATE_REMAINING_TURNS
    is_four_player = num_players >= 4

    # --- modes (build_modes), pure JAX ---
    denom = jnp.maximum(1, my_total + enemy_total).astype(jnp.float32)
    domination = (my_total - enemy_total).astype(jnp.float32) / denom
    is_behind = domination < BEHIND_DOMINATION
    is_ahead = domination > AHEAD_DOMINATION
    is_dominating = is_ahead | (
        (max_enemy_strength > 0)
        & (my_total > max_enemy_strength.astype(jnp.float32) * 1.25)
    )
    is_finishing = (
        (domination > FINISHING_DOMINATION)
        & (
            my_prod.astype(jnp.float32)
            > enemy_prod.astype(jnp.float32) * FINISHING_PROD_RATIO
        )
        & (step > 100)
    )
    attack_margin_mult = (
        jnp.float32(1.0)
        + jnp.where(is_ahead, jnp.float32(AHEAD_ATTACK_MARGIN_BONUS), 0.0)
        - jnp.where(is_behind, jnp.float32(BEHIND_ATTACK_MARGIN_PENALTY), 0.0)
        + jnp.where(is_finishing, jnp.float32(FINISHING_ATTACK_MARGIN_BONUS), 0.0)
    )

    return WorldFeatures(
        planet_id=pid,
        owner=owner,
        xy=jnp.stack([xs, ys], axis=1),
        radius=radius,
        ships=ships,
        prod=prod,
        planet_valid=valid,
        is_static=is_static,
        is_comet=is_comet,
        comet_life=comet_life,
        indirect_wealth=wealth,
        initial_xy=jnp.stack([init_x, init_y], axis=1),
        initial_radius=init_r,
        reaction_my_t=reaction_my,
        reaction_enemy_t=reaction_enemy,
        reserve=reserve,
        available=available,
        arr_owner=arr_owner,
        arr_ships=arr_ships,
        arr_valid=arr_valid,
        comet_path=comet_path,
        comet_path_index=comet_path_index,
        comet_path_len=comet_path_len,
        plan_max_turns=plan_max_turns,
        other_paths=other_paths,
        other_path_index=other_path_index,
        other_path_len=other_path_len,
        other_planet_id=other_planet_id,
        comet_radius=jnp.float32(_COMET_RADIUS_DEFAULT),
        step=step,
        player=player,
        ang_vel=state.angular_velocity.astype(jnp.float32),
        num_players=num_players,
        remaining_steps=remaining_steps,
        is_early=is_early,
        is_opening=is_opening,
        is_late=is_late,
        is_very_late=is_very_late,
        is_four_player=is_four_player,
        my_total=my_total.astype(jnp.int32),
        enemy_total=enemy_total.astype(jnp.int32),
        max_enemy_strength=max_enemy_strength.astype(jnp.int32),
        my_prod=my_prod.astype(jnp.int32),
        enemy_prod=enemy_prod.astype(jnp.int32),
        static_neutral_count=static_neutral_count,
        owner_strength=owner_strength.astype(jnp.int32),
        horizon=jnp.int32(HORIZON),
        domination=domination.astype(jnp.float32),
        is_behind=is_behind,
        is_ahead=is_ahead,
        is_dominating=is_dominating,
        is_finishing=is_finishing,
        attack_margin_mult=attack_margin_mult.astype(jnp.float32),
    )


__all__ = [
    "BIG_REACTION",
    "MAX_COMET_PATH_LEN",
    "MAX_OTHER_COMETS",
    "NUM_OWNERS",
    "WorldFeatures",
    "build_world_features",
    "build_world_features_from_state",
]
