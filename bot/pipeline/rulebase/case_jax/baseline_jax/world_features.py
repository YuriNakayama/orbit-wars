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
from .physics_jax import is_static_planet_jax, travel_time_jax
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


__all__ = [
    "BIG_REACTION",
    "MAX_COMET_PATH_LEN",
    "MAX_OTHER_COMETS",
    "NUM_OWNERS",
    "WorldFeatures",
    "build_world_features",
]
