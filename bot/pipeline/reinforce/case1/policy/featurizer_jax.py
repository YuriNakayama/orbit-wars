"""JAX-native featurizer for reinforce/case1 (W1: global + simple planet cols).

End-to-end JAX rollout requires the featurizer to be vmap'able on GPU.
Phase W1 implements the columns of `BatchFeatures` that depend only on
per-planet scalars (no fleet ETA, no orbit prediction, no template
resolution). Remaining columns are zero-filled and will be filled in
W2-W3:

- W1 (this module):
  - global_feats (20-dim) — all columns, no-launch path (history=None)
  - planet_feats indices 0..8 (position, log1p ships/prod, owner one-hot,
    is_comet)
  - planet_feats index 11 (sun_dist)
  - planet_feats index 12 (is_static)
  - planet_feats indices 22-27 (orbit predictions — placeholder zeros)
  - planet_feats indices 35-40 (timeline — placeholder zeros)
  - planet_mask, my_planet_mask, target_mask

- W2 (next):
  - planet_feats fleet-dependent columns (eta_norm, incoming_*_ships,
    nearest_*_dist, support_density, threat_pressure, ally/enemy_eta_min,
    delta_t1/t2, owner_changed)
  - template_ctx, candidate_feats, candidate_mask, candidate_pid
  - planet_feats orbit prediction columns (21-28)
  - planet_feats timeline columns (35-40)

Input contract: the JAX env's `EnvState` plus the viewing player. No
host-side obs dict needed.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

from jax_env.constants import (
    BOARD_SIZE,
    CENTER,
    ROTATION_RADIUS_LIMIT,
)
from jax_env.state import EnvState

# Output dimensions — matches the PyTorch featurizer in featurizer.py.
PLANET_FEAT_DIM = 41
GLOBAL_FEAT_DIM = 20
MAX_PLANETS = 36
MAX_FLEETS = 512
COMET_WAVES = (50, 150, 250, 350, 450)
COMET_WINDOW = 30
LOG_NORM_DENOM = 6.0
LAUNCH_COUNT_NORM = 10.0
DIAG = math.sqrt(2.0) * BOARD_SIZE
NEIGHBOR_RADIUS_SHORT = 8.0
NEIGHBOR_RADIUS_LONG = 25.0
HORIZON_TURNS = 30
ORBIT_HORIZONS = (1, 2, 4, 8)
TEMPLATE_CTX_DIM = 40  # NUM_TEMPLATES (8) * PER_TEMPLATE_FEATS (5)
CAND_K = 8
CAND_FEAT_DIM = 14


class BatchFeaturesJax(NamedTuple):
    """JAX-native counterpart of `policy/types.py:BatchFeatures`.

    All leaves are `jax.Array` so the whole struct is a pytree and can be
    vmap'd / jit'd. Shapes match the PyTorch BatchFeatures with a leading
    batch dimension (typically size 1 outside vmap, or the env batch axis
    inside vmap).
    """

    planet_feats: jax.Array  # (B, MAX_PLANETS, PLANET_FEAT_DIM) float32
    planet_mask: jax.Array  # (B, MAX_PLANETS) bool
    my_planet_mask: jax.Array  # (B, MAX_PLANETS) bool
    target_mask: jax.Array  # (B, MAX_PLANETS) bool
    global_feats: jax.Array  # (B, GLOBAL_FEAT_DIM) float32
    template_ctx: jax.Array  # (B, MAX_PLANETS, TEMPLATE_CTX_DIM) float32
    candidate_feats: jax.Array  # (B, MAX_PLANETS, CAND_K, CAND_FEAT_DIM) float32
    candidate_mask: jax.Array  # (B, MAX_PLANETS, CAND_K) bool
    candidate_pid: jax.Array  # (B, MAX_PLANETS, CAND_K) int32


def _comet_active(step: jax.Array) -> jax.Array:
    """Vendor: any(0 <= step - w < COMET_WINDOW for w in COMET_WAVES)."""
    waves = jnp.array(COMET_WAVES, dtype=jnp.int32)
    diffs = step - waves
    in_window = (diffs >= 0) & (diffs < COMET_WINDOW)
    return jnp.any(in_window)


def _next_comet_eta(step: jax.Array) -> jax.Array:
    """Vendor: min(w - step for w in COMET_WAVES if w >= step), 100 fallback.

    Returns int32. When comet is active or no upcoming wave, vendor returns
    0 and 100 respectively; we replicate.
    """
    waves = jnp.array(COMET_WAVES, dtype=jnp.int32)
    active = _comet_active(step)
    # Mask out past waves with a large value; remaining waves' diff to step.
    diff = waves - step
    is_upcoming = diff >= 0
    large = jnp.int32(10_000)
    masked = jnp.where(is_upcoming, diff, large)
    min_upcoming = jnp.min(masked)
    has_upcoming = jnp.any(is_upcoming)
    # If active, return 0; else if has upcoming, that diff; else 100.
    return jnp.where(
        active,
        jnp.int32(0),
        jnp.where(has_upcoming, min_upcoming, jnp.int32(100)),
    )


def featurize_jax_w1(
    state: EnvState,
    player: int,
) -> BatchFeaturesJax:
    """W1 implementation: global_feats + simple per-planet columns.

    `player` is a static int (not vmap'd). For multi-agent rollouts where
    each env has the same player viewpoint we keep it as a Python int so
    jit doesn't recompile on changes.

    Returns BatchFeaturesJax with batch dim 1. Caller can vmap over env
    states to get (B, ...) for B envs.

    Columns NOT yet implemented (zero-filled, will be filled in W2-W3):
      planet_feats: 9, 10 (incoming_* / eta_norm), 13 (prod_per_ship),
                    14..20 (nearest dist, support, threat, net_signed),
                    21..28 (orbit predictions), 29..34 (eta/incoming
                    / delta), 35..40 (timeline)
      template_ctx, candidate_feats, candidate_mask, candidate_pid:
                    all zeros / -1 for pid
      my_planet_mask: implemented (owner == player and valid)
      target_mask: implemented (valid and not owner==player)
    """
    # Slice / truncate to MAX_PLANETS (= 36, smaller than jax_env's 48).
    # In practice vendor generates at most 24-32 planets at reset and the
    # game adds 4 per comet activation, capped well below 36 unless many
    # comets are simultaneously active.
    owner = state.planet_owner[:MAX_PLANETS]
    xy = state.planet_xy[:MAX_PLANETS]  # (MAX_PLANETS, 2)
    radius = state.planet_radius[:MAX_PLANETS]
    ships = state.planet_ships[:MAX_PLANETS]
    prod = state.planet_prod[:MAX_PLANETS]
    valid_full = state.planet_valid[:MAX_PLANETS]
    is_comet = state.planet_is_comet[:MAX_PLANETS]

    # Truncate to MAX_PLANETS — same semantics as `n = min(len, MAX_PLANETS)`.
    # `planet_valid` already encodes which slots are in use.

    is_mine = valid_full & (owner == player)
    is_enemy = valid_full & (owner != player) & (owner != -1)
    is_neutral = valid_full & (owner == -1)

    # Per-planet simple features. Shape (MAX_PLANETS, PLANET_FEAT_DIM).
    feats = jnp.zeros((MAX_PLANETS, PLANET_FEAT_DIM), dtype=jnp.float32)

    px = xy[:, 0].astype(jnp.float32)
    py = xy[:, 1].astype(jnp.float32)
    radius_f = radius.astype(jnp.float32)
    ships_f = ships.astype(jnp.float32)
    prod_f = prod.astype(jnp.float32)
    ang_vel = state.angular_velocity.astype(jnp.float32)

    feats = feats.at[:, 0].set(px / BOARD_SIZE)
    feats = feats.at[:, 1].set(py / BOARD_SIZE)
    feats = feats.at[:, 2].set(radius_f / 5.0)
    feats = feats.at[:, 3].set(jnp.log1p(jnp.maximum(0.0, ships_f)))
    feats = feats.at[:, 4].set(jnp.log1p(jnp.maximum(0.0, prod_f)))
    feats = feats.at[:, 5].set(is_mine.astype(jnp.float32))
    feats = feats.at[:, 6].set(is_enemy.astype(jnp.float32))
    feats = feats.at[:, 7].set(is_neutral.astype(jnp.float32))
    feats = feats.at[:, 8].set(is_comet.astype(jnp.float32))
    # idx 9: log1p(incoming_enemy) - log1p(incoming_ally) — W2
    # idx 10: eta_norm — W2
    sun_dist = jnp.sqrt((px - CENTER) ** 2 + (py - CENTER) ** 2)
    feats = feats.at[:, 11].set(sun_dist / DIAG)
    is_static = (sun_dist + radius_f) >= ROTATION_RADIUS_LIMIT
    feats = feats.at[:, 12].set(is_static.astype(jnp.float32))
    feats = feats.at[:, 13].set(
        jnp.minimum(prod_f / jnp.maximum(1.0, ships_f), 5.0) / 5.0
    )
    # idx 14, 15, 18, 19, 16 (partial): planet×planet pairwise distance
    # block. Build a (P, P) distance matrix once and reduce.
    #
    # Vendor (featurizer.py:259-289): for each planet i, iterate j != i,
    # compute hypot(xi-xj, yi-yj), maintain nearest_{ally,enemy,neutral}_dist
    # (init = DIAG) and accumulate support_density (own planets within
    # NEIGHBOR_RADIUS_LONG) and threat_pressure_short (enemy planets within
    # NEIGHBOR_RADIUS_SHORT). The fleet half of threat_pressure_short is
    # added later by the W2-fleet sub-phase; we leave idx 16 = 0 here.
    dx = px[:, None] - px[None, :]
    dy = py[:, None] - py[None, :]
    dist_mat = jnp.sqrt(dx * dx + dy * dy)  # (P, P), float32
    self_mask = jnp.eye(MAX_PLANETS, dtype=jnp.bool_)
    # j_valid: planet j is a usable counterparty (not self, valid slot).
    j_valid = valid_full[None, :] & ~self_mask  # (P, P)

    # Owner-typed neighbor masks (planet-vs-planet, broadcasting over i).
    j_owner = owner[None, :]
    j_is_ally = j_valid & (j_owner == player)
    j_is_neutral = j_valid & (j_owner == -1)
    j_is_enemy = j_valid & (j_owner != -1) & (j_owner != player)

    # Vendor initializes nearest_*_dist[i] = DIAG, then takes min over
    # matching j. Replicate by masking non-matching to +inf and taking min.
    large = jnp.float32(jnp.inf)
    diag_f = jnp.float32(DIAG)
    nearest_ally = jnp.min(
        jnp.where(j_is_ally, dist_mat, large), axis=1
    )
    nearest_ally = jnp.where(jnp.isfinite(nearest_ally), nearest_ally, diag_f)
    nearest_enemy = jnp.min(
        jnp.where(j_is_enemy, dist_mat, large), axis=1
    )
    nearest_enemy = jnp.where(jnp.isfinite(nearest_enemy), nearest_enemy, diag_f)
    nearest_neutral = jnp.min(
        jnp.where(j_is_neutral, dist_mat, large), axis=1
    )
    nearest_neutral = jnp.where(
        jnp.isfinite(nearest_neutral), nearest_neutral, diag_f
    )

    j_ships = ships.astype(jnp.float32)[None, :]
    support_density = jnp.sum(
        jnp.where(j_is_ally & (dist_mat <= NEIGHBOR_RADIUS_LONG), j_ships, 0.0),
        axis=1,
    )
    threat_pressure_planet = jnp.sum(
        jnp.where(
            j_is_enemy & (dist_mat <= NEIGHBOR_RADIUS_SHORT), j_ships, 0.0
        ),
        axis=1,
    )

    feats = feats.at[:, 14].set(nearest_enemy / diag_f)
    feats = feats.at[:, 15].set(jnp.log1p(support_density) / LOG_NORM_DENOM)
    # idx 16: planet half only — vendor adds fleet contributions in W2-fleet.
    # Leave as the planet half so subsequent W2-fleet only adds the fleet
    # term. The PyTorch reference includes fleet contributions which are
    # zero in the no-fleet parity fixture (W1 path).
    feats = feats.at[:, 16].set(
        jnp.log1p(threat_pressure_planet) / LOG_NORM_DENOM
    )
    # idx 17: net_signed — W2-fleet (needs incoming_*_ships)
    feats = feats.at[:, 18].set(nearest_ally / diag_f)
    feats = feats.at[:, 19].set(nearest_neutral / diag_f)
    # idx 20: unused in PyTorch featurizer (always 0)

    # ------------------------------------------------------------------
    # W2b: fleet×planet ETA matrix.
    # Vendor (featurizer.py:236-262) iterates each fleet × each planet,
    # computing in-cone ETA via _fleet_target_eta; accumulates
    # incoming_{ally,enemy,neutral}_ships, *_eta_min, nearest_eta, and
    # threat_pressure_short's fleet contribution. We vectorize over the
    # fixed-shape fleet table (MAX_FLEETS = 512 in jax_env).
    fleet_owner = state.fleet_owner  # (F,)
    fleet_xy_arr = state.fleet_xy  # (F, 2)
    fleet_angle = state.fleet_angle  # (F,)
    fleet_ships_arr = state.fleet_ships  # (F,)
    fleet_valid = state.fleet_valid  # (F,)

    fx = fleet_xy_arr[:, 0].astype(jnp.float32)
    fy = fleet_xy_arr[:, 1].astype(jnp.float32)
    fang = fleet_angle.astype(jnp.float32)
    fships_f = fleet_ships_arr.astype(jnp.float32)
    f_speed = jnp.maximum(
        0.5, 2.0 - 0.05 * jnp.sqrt(jnp.maximum(1.0, fships_f))
    )

    dx_fp = px[None, :] - fx[:, None]  # (F, P)
    dy_fp = py[None, :] - fy[:, None]  # (F, P)
    dir_x = jnp.cos(fang)
    dir_y = jnp.sin(fang)
    proj = dx_fp * dir_x[:, None] + dy_fp * dir_y[:, None]  # (F, P)
    perp_sq = dx_fp * dx_fp + dy_fp * dy_fp - proj * proj
    radius_sq = (radius_f * radius_f)[None, :]
    in_cone = (proj >= 0) & (perp_sq < radius_sq)
    hit_d = jnp.maximum(
        0.0, proj - jnp.sqrt(jnp.maximum(0.0, radius_sq - perp_sq))
    )
    eta_mat = hit_d / f_speed[:, None]  # (F, P)
    # vendor: skip if eta > HORIZON_TURNS or out of cone, also require
    # planet valid and fleet valid.
    in_horizon = (
        in_cone
        & (eta_mat <= HORIZON_TURNS)
        & fleet_valid[:, None]
        & valid_full[None, :]
    )
    eta_masked = jnp.where(in_horizon, eta_mat, jnp.float32(jnp.inf))

    fl_is_ally = fleet_owner[:, None] == player
    fl_is_neutral_only = fleet_owner[:, None] == -1
    fl_is_enemy = ~fl_is_ally & ~fl_is_neutral_only

    incoming_ally = jnp.sum(
        jnp.where(in_horizon & fl_is_ally, fships_f[:, None], 0.0), axis=0
    )
    incoming_enemy = jnp.sum(
        jnp.where(in_horizon & fl_is_enemy, fships_f[:, None], 0.0), axis=0
    )
    # incoming_neutral is used by the timeline (W2e), not the planet feats
    # directly. Skip until W2e.

    horizon_default = jnp.float32(HORIZON_TURNS + 1.0)
    nearest_eta = jnp.min(eta_masked, axis=0)
    nearest_eta = jnp.where(
        jnp.isfinite(nearest_eta), nearest_eta, horizon_default
    )
    ally_eta_min = jnp.min(
        jnp.where(fl_is_ally, eta_masked, jnp.float32(jnp.inf)), axis=0
    )
    ally_eta_min = jnp.where(
        jnp.isfinite(ally_eta_min), ally_eta_min, horizon_default
    )
    enemy_eta_min = jnp.min(
        jnp.where(fl_is_enemy, eta_masked, jnp.float32(jnp.inf)), axis=0
    )
    enemy_eta_min = jnp.where(
        jnp.isfinite(enemy_eta_min), enemy_eta_min, horizon_default
    )

    # Fleet contribution to threat_pressure_short (idx 16): enemy fleets
    # within NEIGHBOR_RADIUS_SHORT of each planet (raw Euclidean, not ETA).
    fleet_planet_dist = jnp.sqrt(dx_fp * dx_fp + dy_fp * dy_fp)  # (F, P)
    fl_threat_mask = (
        fl_is_enemy
        & fleet_valid[:, None]
        & valid_full[None, :]
        & (fleet_planet_dist <= NEIGHBOR_RADIUS_SHORT)
    )
    threat_pressure_fleet = jnp.sum(
        jnp.where(fl_threat_mask, fships_f[:, None], 0.0), axis=0
    )
    # Replace idx 16 with planet half + fleet half summed *before* log1p.
    feats = feats.at[:, 16].set(
        jnp.log1p(threat_pressure_planet + threat_pressure_fleet)
        / LOG_NORM_DENOM
    )

    # idx 9: log1p(incoming_enemy) - log1p(incoming_ally)
    feats = feats.at[:, 9].set(
        jnp.log1p(incoming_enemy) - jnp.log1p(incoming_ally)
    )
    # idx 10: eta_norm = min(nearest_eta, HORIZON+1) / (HORIZON+1)
    feats = feats.at[:, 10].set(
        jnp.minimum(nearest_eta, horizon_default) / horizon_default
    )
    # idx 17: net_signed = (incoming_enemy - incoming_ally) / max(1, ships)
    # then clamp to [-3, 3] and divide by 3
    ships_safe = jnp.maximum(1.0, ships_f)
    net_signed = (incoming_enemy - incoming_ally) / ships_safe
    feats = feats.at[:, 17].set(
        jnp.maximum(-3.0, jnp.minimum(3.0, net_signed)) / 3.0
    )
    # idx 28: ally_eta_norm
    feats = feats.at[:, 28].set(
        jnp.minimum(ally_eta_min, horizon_default) / horizon_default
    )
    # idx 29: enemy_eta_norm
    feats = feats.at[:, 29].set(
        jnp.minimum(enemy_eta_min, horizon_default) / horizon_default
    )
    # idx 30: log1p(incoming_ally) / LOG_NORM_DENOM
    feats = feats.at[:, 30].set(jnp.log1p(incoming_ally) / LOG_NORM_DENOM)
    # idx 31: log1p(incoming_enemy) / LOG_NORM_DENOM
    feats = feats.at[:, 31].set(jnp.log1p(incoming_enemy) / LOG_NORM_DENOM)
    # idx 32, 33: delta_t1, delta_t2 — W2-history (require prior snapshots)
    # idx 34: owner_changed — W2-history

    # ------------------------------------------------------------------
    # W2c: orbit predictions (cols 20..27).
    # For each planet, predict its position at +1/+2/+4/+8 turns and emit
    # (dx, dy)/BOARD_SIZE pairs.
    #
    # Vendor (featurizer.py:_orbit_predictions): non-comet planets use
    # closed-form rotation around CENTER; comet planets look up
    # comet_paths[c, q, path_index + turns]. Both paths emit
    # ((nx - x)/B, (ny - y)/B). Static planets (init_orbital + r >= limit)
    # return (current x, current y), so delta is zero.
    init_xy = state.planet_initial_xy[:MAX_PLANETS].astype(jnp.float32)
    init_x = init_xy[:, 0]
    init_y = init_xy[:, 1]
    init_orbital = jnp.sqrt((init_x - CENTER) ** 2 + (init_y - CENTER) ** 2)
    # is_rotating mirrors reset.py's pre-computed flag but recomputed from
    # initial xy + radius so it stays correct under comet planets (which
    # have their own initial_xy after activation).
    is_rotating = (init_orbital + radius_f) < ROTATION_RADIUS_LIMIT

    # Current angle of each rotating planet from CENTER.
    # NOTE: cur_ang must use (current xy - CENTER), matching vendor
    # geometry.py line 105 (`math.atan2(planet.y - CENTER_Y, planet.x -
    # CENTER_X)`). For invalid slots cur_ang is garbage but it's masked
    # later.
    cur_ang = jnp.arctan2(py - CENTER, px - CENTER)
    # r for the orbit is computed from initial_xy → CENTER, matching
    # vendor (`r = dist(init.x, init.y, CENTER_X, CENTER_Y)`).
    r = init_orbital

    # Build comet path lookup: for each planet slot, the (c, q) it
    # belongs to (or -1 if not a comet). Reuse the same eq_mask trick
    # as in step.py:_compute_planet_new_xy.
    flat_slots = state.comet_planet_slot.reshape(-1)  # (C*4,)
    planet_idx = jnp.arange(MAX_PLANETS, dtype=jnp.int32)
    eq_mask = (planet_idx[:, None] == flat_slots[None, :]) & (
        flat_slots[None, :] >= 0
    )
    has_comet_entry = jnp.any(eq_mask, axis=-1)  # (P,)
    pick_idx = jnp.argmax(eq_mask.astype(jnp.int32), axis=-1)  # (P,) into C*4
    c_of_p = pick_idx // 4
    q_of_p = pick_idx % 4
    # path_index of the comet that owns this planet (0 if no comet).
    path_index_p = jnp.where(
        has_comet_entry, state.comet_path_index[c_of_p], jnp.int32(0)
    )
    path_len_p = jnp.where(
        has_comet_entry, state.comet_path_len[c_of_p], jnp.int32(0)
    )

    # Compute predictions for each horizon and write to feats cols 20..27.
    for h_idx, turns in enumerate(ORBIT_HORIZONS):
        # Rotating planets — closed form.
        new_ang = cur_ang + ang_vel * turns
        rot_nx = CENTER + r * jnp.cos(new_ang)
        rot_ny = CENTER + r * jnp.sin(new_ang)
        rot_dx = (rot_nx - px) / BOARD_SIZE
        rot_dy = (rot_ny - py) / BOARD_SIZE
        # Static or unknown → (0, 0)
        rot_dx = jnp.where(is_rotating, rot_dx, 0.0)
        rot_dy = jnp.where(is_rotating, rot_dy, 0.0)

        # Comet planets — lookup comet_paths[c, q, path_index + turns].
        # NOTE: vendor's `predict_comet_position` reads `paths[idx]`
        # as raw `(nx, ny)` *without* the per-quadrant swap that
        # `_compute_planet_new_xy` applies when writing planet_xy. This
        # means torch's orbit prediction delta is mathematically
        # inconsistent with the actual planet motion at q=0 and q=3, but
        # the BC weights were trained against this exact featurizer
        # output, so we must replicate the bug.
        future_idx = path_index_p + jnp.int32(turns)
        in_range = (future_idx >= 0) & (future_idx < path_len_p)
        safe_idx = jnp.clip(future_idx, 0, state.comet_paths.shape[2] - 1)
        comet_xy = state.comet_paths[c_of_p, q_of_p, safe_idx]
        comet_dx = (comet_xy[:, 0] - px) / BOARD_SIZE
        comet_dy = (comet_xy[:, 1] - py) / BOARD_SIZE
        # Vendor returns (0, 0) if future_idx out of range.
        comet_dx = jnp.where(in_range, comet_dx, 0.0)
        comet_dy = jnp.where(in_range, comet_dy, 0.0)

        is_comet_planet = has_comet_entry & is_comet
        dx_h = jnp.where(is_comet_planet, comet_dx, rot_dx)
        dy_h = jnp.where(is_comet_planet, comet_dy, rot_dy)
        feats = feats.at[:, 20 + 2 * h_idx].set(dx_h)
        feats = feats.at[:, 21 + 2 * h_idx].set(dy_h)

    # idx 35..40: timeline columns — W2e

    # Mask invalid slots — zero out all columns for them.
    feats = feats * valid_full[:, None].astype(jnp.float32)

    # Global features.
    step = state.step.astype(jnp.int32)

    my_ships_total = jnp.sum(jnp.where(is_mine, ships_f, 0.0))
    enemy_ships_total = jnp.sum(jnp.where(is_enemy, ships_f, 0.0))
    neutral_ships_total = jnp.sum(jnp.where(is_neutral, ships_f, 0.0))
    my_prod_total = jnp.sum(jnp.where(is_mine, prod_f, 0.0))
    enemy_prod_total = jnp.sum(jnp.where(is_enemy, prod_f, 0.0))
    my_count = jnp.sum(is_mine.astype(jnp.int32))
    enemy_count = jnp.sum(is_enemy.astype(jnp.int32))
    n_used = jnp.sum(valid_full.astype(jnp.int32))
    total_planets = jnp.maximum(jnp.int32(1), n_used)
    total_ships = my_ships_total + enemy_ships_total + neutral_ships_total
    total_prod = my_prod_total + enemy_prod_total

    phase_mid = jnp.where((step >= 100) & (step < 300), 1.0, 0.0)
    phase_late = jnp.where(step >= 300, 1.0, 0.0)
    score_diff = jnp.log1p(my_ships_total) - jnp.log1p(enemy_ships_total)
    next_eta = _next_comet_eta(step).astype(jnp.float32) / 100.0

    # History-dependent globals (idx 16..19) require recent_launches —
    # zero-fill in W1. Will be added in a later sub-phase as a history pytree.

    g = jnp.zeros((GLOBAL_FEAT_DIM,), dtype=jnp.float32)
    g = g.at[0].set(step.astype(jnp.float32) / 500.0)
    g = g.at[1].set(ang_vel * 10.0)
    g = g.at[2].set(jnp.log1p(my_ships_total))
    g = g.at[3].set(jnp.log1p(enemy_ships_total))
    g = g.at[4].set(jnp.log1p(neutral_ships_total))
    g = g.at[5].set(jnp.log1p(my_prod_total) - jnp.log1p(enemy_prod_total))
    g = g.at[6].set(my_count.astype(jnp.float32) / total_planets.astype(jnp.float32))
    g = g.at[7].set(
        enemy_count.astype(jnp.float32) / total_planets.astype(jnp.float32)
    )
    g = g.at[8].set(_comet_active(step).astype(jnp.float32))
    g = g.at[9].set(phase_mid)
    g = g.at[10].set(phase_late)
    g = g.at[11].set(jnp.minimum(1.0, next_eta))
    g = g.at[12].set(
        jnp.where(total_ships > 0, my_ships_total / total_ships, 0.0)
    )
    g = g.at[13].set(
        jnp.where(total_ships > 0, enemy_ships_total / total_ships, 0.0)
    )
    g = g.at[14].set(
        jnp.where(total_prod > 0, my_prod_total / total_prod, 0.0)
    )
    g = g.at[15].set(jnp.maximum(-3.0, jnp.minimum(3.0, score_diff)) / 3.0)
    # idx 16..19: launch counts — W1 leaves as 0

    # Add batch dimension. Outside vmap caller gets B=1; inside vmap the
    # caller wraps featurize_jax_w1 in jax.vmap(..., in_axes=(0, None)).
    batch_planet_feats = feats[None, ...]
    batch_planet_mask = valid_full[None, ...]
    batch_my_planet_mask = is_mine[None, ...]
    batch_target_mask = (valid_full & ~is_mine)[None, ...]
    batch_global_feats = g[None, ...]
    # W2 placeholders.
    batch_template_ctx = jnp.zeros(
        (1, MAX_PLANETS, TEMPLATE_CTX_DIM), dtype=jnp.float32
    )
    batch_candidate_feats = jnp.zeros(
        (1, MAX_PLANETS, CAND_K, CAND_FEAT_DIM), dtype=jnp.float32
    )
    batch_candidate_mask = jnp.zeros(
        (1, MAX_PLANETS, CAND_K), dtype=jnp.bool_
    )
    batch_candidate_pid = jnp.full(
        (1, MAX_PLANETS, CAND_K), -1, dtype=jnp.int32
    )

    return BatchFeaturesJax(
        planet_feats=batch_planet_feats,
        planet_mask=batch_planet_mask,
        my_planet_mask=batch_my_planet_mask,
        target_mask=batch_target_mask,
        global_feats=batch_global_feats,
        template_ctx=batch_template_ctx,
        candidate_feats=batch_candidate_feats,
        candidate_mask=batch_candidate_mask,
        candidate_pid=batch_candidate_pid,
    )


__all__ = [
    "BatchFeaturesJax",
    "featurize_jax_w1",
    "PLANET_FEAT_DIM",
    "GLOBAL_FEAT_DIM",
    "MAX_PLANETS",
    "TEMPLATE_CTX_DIM",
    "CAND_K",
    "CAND_FEAT_DIM",
]
