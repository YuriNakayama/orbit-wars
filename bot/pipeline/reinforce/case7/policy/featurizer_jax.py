"""JAX-native featurizer for reinforce/case7 (W1: global + simple planet cols).

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
from orbit_wars_jax.constants import (
    BOARD_SIZE,
    CENTER,
    ROTATION_RADIUS_LIMIT,
)
from orbit_wars_jax.state import EnvState

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
SUN_X = CENTER
SUN_Y = CENTER
SUN_R = 10.0
LAUNCH_CLEARANCE = 0.1
ROTATION_LIMIT = 50.0
MAX_SHIPS = 400.0
MAX_PRODUCTION = 5.0
SHIPS_NEEDED = 5  # candidates.fixed_ship_count default when env var unset
NEIGHBOR_RADIUS_SHORT = 8.0
NEIGHBOR_RADIUS_LONG = 25.0
HORIZON_TURNS = 30
ORBIT_HORIZONS = (1, 2, 4, 8)
TIMELINE_HORIZON = 30
SHORT_WINDOW = 3  # `loss_3turn` window in summarize_timeline
KEEP_BSEARCH_ITERS = 10  # log2(max ships ~= 400) rounded up
OWNER_ALLY = 0
OWNER_ENEMY = 1
OWNER_NEUTRAL = 2
HISTORY_TURNS = 4
# Fixed-size buffers for the JAX history pytree. We retain the last
# `N_PREV_SNAPSHOTS` planet snapshots (the featurizer reads positions -2
# and -3, so we need 3 prior snapshots indexed by ring offset). Launch
# events use a fixed-size ring; at 1 launch / side / step over
# HISTORY_TURNS=4 the high-water mark is ~8 launches, well below 64.
N_PREV_SNAPSHOTS = 3
LAUNCH_BUFFER = 64
TEMPLATE_CTX_DIM = 40  # NUM_TEMPLATES (8) * PER_TEMPLATE_FEATS (5)
CAND_K = 8
CAND_FEAT_DIM = 14


class HistoryStateJax(NamedTuple):
    """JAX-native history pytree mirroring PyTorch HistoryState.

    Holds the last N_PREV_SNAPSHOTS planet snapshots in a ring buffer and
    a fixed-size launch-event ring buffer. All shapes are fixed so the
    state is jit/vmap-friendly.

    Ring buffer semantics:
      `snap_count` is the number of snapshots ever pushed (saturates at
      N_PREV_SNAPSHOTS for indexing). `snap_head` is the index of the
      next write slot. The most-recent snapshot is at
      `(snap_head - 1) mod N`; the one before that is at
      `(snap_head - 2) mod N`; etc. featurizer reads positions -2 and -3
      mapping to `(snap_head - 2) mod N` and `(snap_head - 3) mod N`.
    """

    # Snapshot ring (3 prior states).
    snap_ships: jax.Array  # (N_PREV_SNAPSHOTS, MAX_PLANETS) int32
    snap_owner: jax.Array  # (N_PREV_SNAPSHOTS, MAX_PLANETS) int32
    snap_pid: jax.Array  # (N_PREV_SNAPSHOTS, MAX_PLANETS) int32; -1 = empty slot
    snap_valid: jax.Array  # (N_PREV_SNAPSHOTS, MAX_PLANETS) bool
    snap_count: jax.Array  # int32 scalar (clamped at N_PREV_SNAPSHOTS)
    snap_head: jax.Array  # int32 scalar (mod N_PREV_SNAPSHOTS)

    # Launch event ring.
    launch_step: jax.Array  # (LAUNCH_BUFFER,) int32; -1 = empty
    launch_owner: jax.Array  # (LAUNCH_BUFFER,) int32
    launch_ships: jax.Array  # (LAUNCH_BUFFER,) int32
    launch_valid: jax.Array  # (LAUNCH_BUFFER,) bool


def init_history_jax() -> HistoryStateJax:
    """Empty history (matches PyTorch `HistoryState()` semantics)."""
    return HistoryStateJax(
        snap_ships=jnp.zeros((N_PREV_SNAPSHOTS, MAX_PLANETS), dtype=jnp.int32),
        snap_owner=jnp.full((N_PREV_SNAPSHOTS, MAX_PLANETS), -1, dtype=jnp.int32),
        snap_pid=jnp.full((N_PREV_SNAPSHOTS, MAX_PLANETS), -1, dtype=jnp.int32),
        snap_valid=jnp.zeros((N_PREV_SNAPSHOTS, MAX_PLANETS), dtype=jnp.bool_),
        snap_count=jnp.int32(0),
        snap_head=jnp.int32(0),
        launch_step=jnp.full((LAUNCH_BUFFER,), -1, dtype=jnp.int32),
        launch_owner=jnp.full((LAUNCH_BUFFER,), -1, dtype=jnp.int32),
        launch_ships=jnp.zeros((LAUNCH_BUFFER,), dtype=jnp.int32),
        launch_valid=jnp.zeros((LAUNCH_BUFFER,), dtype=jnp.bool_),
    )


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


def _build_template_ctx(
    px: jax.Array,  # (P,) float32
    py: jax.Array,  # (P,) float32
    ships_f: jax.Array,  # (P,) float32
    prod_f: jax.Array,  # (P,) float32
    owner: jax.Array,  # (P,) int32
    valid_full: jax.Array,  # (P,) bool
    is_mine: jax.Array,  # (P,) bool
    player: int,
) -> jax.Array:
    """JAX equivalent of `templates.template_context_features_parsed`.

    Returns (P, TEMPLATE_CTX_DIM=40) array of per-src per-template feats.
    Only own (is_mine) sources are populated; other slots stay zero,
    mirroring PyTorch which only calls `template_context_features` from
    inside the `if is_mine` branch of the planet loop.
    """
    P = MAX_PLANETS
    diag = jnp.float32(DIAG)

    # Build pairwise (P_src, P_tgt) distance with src != tgt mask.
    dx = px[:, None] - px[None, :]
    dy = py[:, None] - py[None, :]
    dist_st = jnp.sqrt(dx * dx + dy * dy)  # (P, P)
    self_mask = jnp.eye(P, dtype=jnp.bool_)
    tgt_valid = valid_full[None, :] & ~self_mask  # (P_src, P_tgt)

    # Per-target owner classifications (broadcast to (P_src, P_tgt)).
    t_owner = owner[None, :]  # (1, P)
    t_is_neutral = tgt_valid & (t_owner == -1)
    t_is_enemy = tgt_valid & (t_owner != player) & (t_owner != -1)
    # mine_other (== ally other) — used by templates 4 and 5; computed
    # later as `mine_other_mask_2d`.

    INF = jnp.float32(jnp.inf)
    NEG_INF = jnp.float32(-jnp.inf)

    def masked_argmin(
        values: jax.Array, mask: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Return (idx, has_any) per row. idx is 0 when has_any=False."""
        masked = jnp.where(mask, values, INF)
        idx = jnp.argmin(masked, axis=-1)
        has_any = jnp.any(mask, axis=-1)
        return idx, has_any

    def masked_argmax_prod(
        prod: jax.Array, dist: jax.Array, mask: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """argmax by (prod, -dist) lexicographic. Implemented as argmax
        over a composite key prod * BIG - dist where BIG >> max(dist).
        """
        BIG = jnp.float32(1.0e6)
        key = prod[None, :] * BIG - dist
        key = jnp.where(mask, key, NEG_INF)
        idx = jnp.argmax(key, axis=-1)
        has_any = jnp.any(mask, axis=-1)
        return idx, has_any

    # T0 NEAREST_NEUTRAL_LOW: nearest neutral with ships <= max(1, src.ships)
    # Fallback to nearest neutral if no low-ships candidate.
    cap_per_src = jnp.maximum(1.0, ships_f)  # (P,)
    low_neutral_mask = t_is_neutral & (ships_f[None, :] <= cap_per_src[:, None])
    idx_low, has_low = masked_argmin(dist_st, low_neutral_mask)
    idx_any_neutral, has_neutral = masked_argmin(dist_st, t_is_neutral)
    t0_idx = jnp.where(has_low, idx_low, idx_any_neutral)
    t0_has = has_low | has_neutral

    # T1 NEAREST_ENEMY
    t1_idx, t1_has = masked_argmin(dist_st, t_is_enemy)

    # T2 HIGH_PROD_NEUTRAL: argmax by (prod, -dist)
    t2_idx, t2_has = masked_argmax_prod(prod_f, dist_st, t_is_neutral)

    # T3 HIGH_PROD_ENEMY
    t3_idx, t3_has = masked_argmax_prod(prod_f, dist_st, t_is_enemy)

    # T4 REINFORCE_FRONTLINE: enemy centroid → nearest mine_other to it.
    # If no mine_other OR no enemy, fall back to nearest mine_other.
    mine_other_mask_2d = tgt_valid & (t_owner == player)
    enemy_mask_2d = t_is_enemy
    enemy_count_global = jnp.sum(
        (valid_full & (owner != player) & (owner != -1)).astype(jnp.float32)
    )
    enemy_x_sum = jnp.sum(
        jnp.where(valid_full & (owner != player) & (owner != -1), px, 0.0)
    )
    enemy_y_sum = jnp.sum(
        jnp.where(valid_full & (owner != player) & (owner != -1), py, 0.0)
    )
    safe_enemy_count = jnp.maximum(1.0, enemy_count_global)
    ex = enemy_x_sum / safe_enemy_count
    ey = enemy_y_sum / safe_enemy_count
    # Per-src per-tgt distance from each tgt's xy to (ex, ey).
    # The distance only depends on tgt, but we need shape (P_src, P_tgt).
    tgt_to_centroid = jnp.sqrt((px - ex) ** 2 + (py - ey) ** 2)  # (P,)
    t4_centroid_dist = jnp.broadcast_to(tgt_to_centroid[None, :], (P, P))
    # Per-src branching: if no enemy OR no mine_other, use nearest
    # mine_other by src-distance (matches PyTorch fallback).
    nearest_mine_idx, has_mine_other = masked_argmin(dist_st, mine_other_mask_2d)
    frontline_idx, _ = masked_argmin(t4_centroid_dist, mine_other_mask_2d)
    # Has enemy globally? has_any_enemy across all sources is the same
    # value, but we compute per-row via reduction over the tgt mask.
    has_any_enemy_per_src = jnp.any(enemy_mask_2d, axis=-1)
    use_centroid = has_any_enemy_per_src & has_mine_other
    t4_idx = jnp.where(use_centroid, frontline_idx, nearest_mine_idx)
    t4_has = has_mine_other

    # T5 REINFORCE_WEAKEST: min ships among mine_other.
    t5_idx, t5_has = masked_argmin(
        jnp.broadcast_to(ships_f[None, :], (P, P)), mine_other_mask_2d
    )

    # T6 WEAKEST_ENEMY: min ships among enemy.
    t6_idx, t6_has = masked_argmin(
        jnp.broadcast_to(ships_f[None, :], (P, P)), enemy_mask_2d
    )

    # Pack template results: (T, P_src) of (target_idx, has_any).
    target_idx = jnp.stack(
        [t0_idx, t1_idx, t2_idx, t3_idx, t4_idx, t5_idx, t6_idx], axis=0
    )  # (7, P)
    target_has = jnp.stack(
        [t0_has, t1_has, t2_has, t3_has, t4_has, t5_has, t6_has], axis=0
    )  # (7, P)

    # Compute per-template per-src features.
    # Build per-(t, src) target info by gather.
    src_idx_grid = jnp.arange(P, dtype=jnp.int32)[None, :]  # (1, P)
    tgt_x = px[target_idx]  # (7, P)
    tgt_y = py[target_idx]
    tgt_ships = ships_f[target_idx]
    tgt_prod = prod_f[target_idx]
    tgt_owner = owner[target_idx]

    d = jnp.sqrt((px[None, :] - tgt_x) ** 2 + (py[None, :] - tgt_y) ** 2)
    prox = jnp.maximum(0.0, 1.0 - d / diag)
    src_ships_b = ships_f[None, :]
    ratio = (src_ships_b + 1.0) / (tgt_ships + 1.0)
    ship_adv = ratio / (1.0 + ratio)
    prod_norm = jnp.minimum(1.0, tgt_prod / 10.0)
    score = 0.5 * prox + 0.3 * ship_adv + 0.2 * prod_norm
    tgt_is_enemy_per = (tgt_owner != player) & (tgt_owner != -1)
    tgt_is_mine_per = tgt_owner == player

    # Filter out cases where target == src (PyTorch skips with `if rid
    # is None or rid == src.id`). target_idx points to a planet index;
    # src is the row index, both in same P-space, so equality of indices.
    skip_self = target_idx == src_idx_grid
    write_mask = target_has & ~skip_self  # (7, P)

    # Assemble template_ctx by writing per-template 5-tuple at offset
    # (t * 5 .. (t+1) * 5). Vectorize by building a (T, P, 5) tensor,
    # masking, then reshape/transpose.
    per_t_feats = jnp.stack(
        [
            score,
            prox,
            ship_adv,
            tgt_is_enemy_per.astype(jnp.float32),
            tgt_is_mine_per.astype(jnp.float32),
        ],
        axis=-1,
    )  # (7, P, 5)
    per_t_feats = per_t_feats * write_mask[..., None].astype(jnp.float32)

    # Reshape: (7, P, 5) -> (P, 7, 5) -> (P, 35). Concat NO_OP slot.
    per_p = jnp.transpose(per_t_feats, (1, 0, 2)).reshape(P, 7 * 5)
    any_candidate = jnp.any(write_mask, axis=0)  # (P,)
    noop = jnp.zeros((P, 5), dtype=jnp.float32)
    noop = noop.at[:, 0].set(jnp.where(any_candidate, 0.0, 1.0).astype(jnp.float32))
    full = jnp.concatenate([per_p, noop], axis=1)  # (P, 40)

    # Only own planets get template_ctx (PyTorch only writes for is_mine).
    full = jnp.where(is_mine[:, None], full, 0.0)
    return full


def _build_candidate_block(
    px: jax.Array,
    py: jax.Array,
    pid_arr: jax.Array,
    radius_f: jax.Array,
    ships_arr: jax.Array,  # int32 (P,) — for ships_needed comparison
    ships_f: jax.Array,  # float32 (P,) — for normalization
    prod_f: jax.Array,
    owner: jax.Array,
    valid_full: jax.Array,
    is_mine: jax.Array,
    player: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """JAX equivalent of `candidates.build_candidate_block` for all own srcs.

    Returns:
      cand_feats: (P, CAND_K, CAND_FEAT_DIM) float32
      cand_mask:  (P, CAND_K) bool
      cand_pid:   (P, CAND_K) int32 (-1 = no-op slot or unfilled)
    """
    P = MAX_PLANETS
    F = CAND_FEAT_DIM

    # Per-(src, tgt) basics.
    dx = px[None, :] - px[:, None]  # (P_src, P_tgt)
    dy = py[None, :] - py[:, None]
    dist_st = jnp.sqrt(dx * dx + dy * dy)
    self_mask = jnp.eye(P, dtype=jnp.bool_)
    tgt_valid = valid_full[None, :] & ~self_mask

    t_owner = owner[None, :]
    is_enemy_t = tgt_valid & (t_owner != player) & (t_owner != -1)
    is_neutral_t = tgt_valid & (t_owner == -1)
    is_friendly_t = tgt_valid & (t_owner == player)

    # Vendor sort key: (dist, id) ascending. Composite as
    # `dist * BIG_ID_MULT + pid` so id breaks ties without changing
    # dist ordering. BIG must exceed max-id (< 100).
    BIG_ID_MULT = jnp.float32(1.0e6)
    pid_f = pid_arr[None, :].astype(jnp.float32)  # (1, P_tgt)
    sort_key = dist_st * BIG_ID_MULT + pid_f  # (P_src, P_tgt)

    INF = jnp.float32(jnp.inf)

    def topk_indices(mask: jax.Array, k: int) -> jax.Array:
        """Top-k smallest sort_key per src, masked. Returns (P, k) int32.

        Invalid slots (mask all False at the corresponding rank) get
        target_idx of an arbitrary slot; caller must consult per-slot
        validity via the consumed mask.
        """
        keys = jnp.where(mask, sort_key, INF)
        idx = jnp.argsort(keys, axis=-1)  # ascending
        return idx[:, :k]

    enemy_top = topk_indices(is_enemy_t, 2)  # (P, 2)
    neutral_top = topk_indices(is_neutral_t, 2)
    friendly_top = topk_indices(is_friendly_t, 3)

    # Per-slot validity flag from the bucket mask (top-k returns
    # arbitrary indices when fewer than k valid targets exist).
    def picked_valid(idx_block: jax.Array, mask: jax.Array) -> jax.Array:
        row_idx = jnp.arange(P, dtype=jnp.int32)[:, None]
        return mask[row_idx, idx_block]

    enemy_valid = picked_valid(enemy_top, is_enemy_t)  # (P, 2)
    neutral_valid = picked_valid(neutral_top, is_neutral_t)
    friendly_valid = picked_valid(friendly_top, is_friendly_t)

    # Replicate PyTorch's `chosen = enemies + neutrals + friendlies`
    # then `chosen.extend(fallback[: 7 - len(chosen)])`. Variable-length
    # concatenation in fixed-shape JAX = sequential append driven by
    # bucket validity flags.
    final_idx = jnp.full((P, 7), -1, dtype=jnp.int32)
    final_valid = jnp.zeros((P, 7), dtype=jnp.bool_)
    write_pos = jnp.zeros((P,), dtype=jnp.int32)

    def _append(
        final_idx: jax.Array,
        final_valid: jax.Array,
        write_pos: jax.Array,
        cand_idx_col: jax.Array,
        cand_valid_col: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Append (cand_idx_col, cand_valid_col) to final_*, advancing
        write_pos per-row when valid. cand_*_col are (P,)."""
        wp = write_pos
        # Only write when both the candidate is valid AND we still have
        # room (wp < 7). PyTorch caps the chosen list at 7 by
        # `fallback[: target_count - len(chosen)]`, dropping later
        # fallback entries.
        room = wp < 7
        effective_valid = cand_valid_col & room
        wp_safe = jnp.minimum(wp, 6)
        rows = jnp.arange(P, dtype=jnp.int32)
        new_idx = final_idx.at[rows, wp_safe].set(
            jnp.where(effective_valid, cand_idx_col, final_idx[rows, wp_safe])
        )
        new_valid = final_valid.at[rows, wp_safe].set(
            final_valid[rows, wp_safe] | effective_valid
        )
        new_wp = wp + effective_valid.astype(jnp.int32)
        return new_idx, new_valid, new_wp

    # Phase 1: enemies (2 slots).
    for s in range(2):
        final_idx, final_valid, write_pos = _append(
            final_idx,
            final_valid,
            write_pos,
            enemy_top[:, s],
            enemy_valid[:, s],
        )
    # Phase 2: neutrals (2 slots).
    for s in range(2):
        final_idx, final_valid, write_pos = _append(
            final_idx,
            final_valid,
            write_pos,
            neutral_top[:, s],
            neutral_valid[:, s],
        )
    # Phase 3: friendlies (3 slots).
    for s in range(3):
        final_idx, final_valid, write_pos = _append(
            final_idx,
            final_valid,
            write_pos,
            friendly_top[:, s],
            friendly_valid[:, s],
        )

    # Phase 4: fallback. Build the "remaining" pool by excluding everything
    # written so far (final_idx where valid). Then top-k by (dist, id).
    picked_mask = jnp.zeros((P, P), dtype=jnp.bool_)
    rows = jnp.arange(P, dtype=jnp.int32)
    for s in range(7):
        was_valid = final_valid[:, s]
        col = final_idx[:, s]
        # Clamp invalid (col=-1) to 0 to avoid OOB; result is gated by where.
        col_safe = jnp.maximum(col, 0)
        new_picked = picked_mask.at[rows, col_safe].set(True)
        picked_mask = jnp.where(was_valid[:, None], new_picked, picked_mask)
    fallback_pool = tgt_valid & ~picked_mask
    fallback_top = topk_indices(fallback_pool, 7)
    fallback_top_valid = picked_valid(fallback_top, fallback_pool)
    for s in range(7):
        final_idx, final_valid, write_pos = _append(
            final_idx,
            final_valid,
            write_pos,
            fallback_top[:, s],
            fallback_top_valid[:, s],
        )

    # Gather per-(src, slot) target attributes.
    tgt_x = px[final_idx]  # (P, 7)
    tgt_y = py[final_idx]
    tgt_radius = radius_f[final_idx]
    tgt_ships_f = ships_f[final_idx]
    tgt_prod = prod_f[final_idx]
    tgt_owner_g = owner[final_idx]
    tgt_pid_g = pid_arr[final_idx]

    src_x = px[:, None]  # (P, 1)
    src_y = py[:, None]
    src_ships_f = ships_f[:, None]
    src_ships_i = ships_arr[:, None]
    src_radius_b = radius_f[:, None]

    d_x = tgt_x - src_x
    d_y = tgt_y - src_y
    dist_full = jnp.sqrt(d_x * d_x + d_y * d_y)
    angle = jnp.arctan2(d_y, d_x)

    # Per-target is_neutral/is_mine/is_enemy/is_rotating.
    is_neutral_g = tgt_owner_g == -1
    is_mine_g = tgt_owner_g == player
    is_enemy_g = (~is_neutral_g) & (~is_mine_g)
    tgt_to_sun = jnp.sqrt((tgt_x - SUN_X) ** 2 + (tgt_y - SUN_Y) ** 2)
    is_rotating_t = (tgt_to_sun + tgt_radius) < ROTATION_LIMIT

    # _shot_crosses_sun: point-to-segment distance from sun to the line
    # from (start_x, start_y) to (tgt_x, tgt_y). start is src + radius +
    # LAUNCH_CLEARANCE along the angle.
    start_x = src_x + jnp.cos(angle) * (src_radius_b + LAUNCH_CLEARANCE)
    start_y = src_y + jnp.sin(angle) * (src_radius_b + LAUNCH_CLEARANCE)
    sx_dx = tgt_x - start_x
    sy_dy = tgt_y - start_y
    seg_len_sq = sx_dx * sx_dx + sy_dy * sy_dy
    seg_safe = jnp.maximum(seg_len_sq, 1e-9)
    t_param = ((SUN_X - start_x) * sx_dx + (SUN_Y - start_y) * sy_dy) / seg_safe
    t_param = jnp.maximum(0.0, jnp.minimum(1.0, t_param))
    cx = start_x + t_param * sx_dx
    cy = start_y + t_param * sy_dy
    dist_sun_seg = jnp.sqrt((SUN_X - cx) ** 2 + (SUN_Y - cy) ** 2)
    # When seg_len_sq is tiny, vendor falls back to point distance
    # hypot(px - x1, py - y1). Replicate that branch.
    fallback_d = jnp.sqrt((SUN_X - start_x) ** 2 + (SUN_Y - start_y) ** 2)
    dist_sun = jnp.where(seg_len_sq <= 1e-9, fallback_d, dist_sun_seg)
    crosses = dist_sun < SUN_R

    # Build the 14-dim feature row per (src, slot).
    feats_block = jnp.zeros((P, 7, F), dtype=jnp.float32)
    feats_block = feats_block.at[:, :, 0].set(final_valid.astype(jnp.float32))
    feats_block = feats_block.at[:, :, 1].set(is_neutral_g.astype(jnp.float32))
    feats_block = feats_block.at[:, :, 2].set(is_mine_g.astype(jnp.float32))
    feats_block = feats_block.at[:, :, 3].set(is_enemy_g.astype(jnp.float32))
    feats_block = feats_block.at[:, :, 4].set(tgt_x / BOARD_SIZE)
    feats_block = feats_block.at[:, :, 5].set(tgt_y / BOARD_SIZE)
    feats_block = feats_block.at[:, :, 6].set(d_x / BOARD_SIZE)
    feats_block = feats_block.at[:, :, 7].set(d_y / BOARD_SIZE)
    feats_block = feats_block.at[:, :, 8].set(dist_full / BOARD_SIZE)
    feats_block = feats_block.at[:, :, 9].set(
        jnp.minimum(tgt_ships_f, MAX_SHIPS) / MAX_SHIPS
    )
    feats_block = feats_block.at[:, :, 10].set(tgt_prod / MAX_PRODUCTION)
    feats_block = feats_block.at[:, :, 11].set(is_rotating_t.astype(jnp.float32))
    feats_block = feats_block.at[:, :, 12].set(crosses.astype(jnp.float32))
    feats_block = feats_block.at[:, :, 13].set(
        jnp.minimum(src_ships_f, MAX_SHIPS) / MAX_SHIPS
    )
    # Zero out invalid slots (PyTorch leaves them at 0 because they were
    # never populated).
    feats_block = feats_block * final_valid[..., None].astype(jnp.float32)

    # Slot 0 = no-op. is_valid=1, mask=True, pid=-1.
    noop_feats = jnp.zeros((P, 1, F), dtype=jnp.float32).at[:, 0, 0].set(1.0)
    cand_feats = jnp.concatenate([noop_feats, feats_block], axis=1)  # (P, 8, F)

    # cand_mask: slot 0 always True; slot 1..7 = ships_needed > 0 AND not
    # crosses AND src.ships >= ships_needed.
    can_fire = (
        (jnp.int32(SHIPS_NEEDED) > 0)
        & ~crosses
        & (src_ships_i >= jnp.int32(SHIPS_NEEDED))
    )
    slot_mask = final_valid & can_fire
    noop_mask = jnp.ones((P, 1), dtype=jnp.bool_)
    cand_mask = jnp.concatenate([noop_mask, slot_mask], axis=1)  # (P, 8)

    # cand_pid: slot 0 = -1, slot 1..7 = tgt_pid if valid else -1.
    noop_pid = jnp.full((P, 1), -1, dtype=jnp.int32)
    slot_pid = jnp.where(final_valid, tgt_pid_g, jnp.int32(-1))
    cand_pid = jnp.concatenate([noop_pid, slot_pid], axis=1)

    # Only own planets actually get populated. PyTorch sets to zero for
    # non-mine slots (never populated).
    mine_b = is_mine[:, None, None]
    cand_feats = jnp.where(mine_b, cand_feats, 0.0)
    mine_b_2d = is_mine[:, None]
    cand_mask = jnp.where(mine_b_2d, cand_mask, jnp.bool_(False))
    cand_pid = jnp.where(mine_b_2d, cand_pid, jnp.int32(-1))

    return cand_feats, cand_mask, cand_pid


def _resolve_arrival_step(
    owner_cls: jax.Array,  # int32 scalar in {0, 1, 2}
    garrison: jax.Array,  # float32 scalar
    arrivals_3: jax.Array,  # float32 (3,) — ships by owner class
) -> tuple[jax.Array, jax.Array]:
    """JAX equivalent of `resolve_arrival_event` collapsed to 3 owner classes.

    Mirrors the vendor logic: aggregate ships by attacker class, find top
    and second-by-ships, compute survivor = (top_owner, top - second) with
    tie → (-1, 0). Combine survivor with the existing garrison: same
    owner → garrison + survivor_ships; different owner → garrison -=
    survivor_ships, flipping owner if garrison goes negative.

    NEUTRAL acts as a "no survivor" sentinel here; survivor_ships <= 0
    keeps the existing (owner_cls, garrison) untouched (after the
    `max(0, garrison)` floor that PyTorch applies on the empty path).
    """
    # Total ships per class (ally, enemy, neutral).
    # Sort to find top, second. Use jnp.sort descending.
    sorted_vals = jnp.sort(arrivals_3)[::-1]  # (3,) descending
    top_val = sorted_vals[0]
    second_val = sorted_vals[1]
    # top_idx = argmax (ties: first occurrence — matches PyTorch's sorted
    # stable order when ships are equal).
    top_idx = jnp.argmax(arrivals_3).astype(jnp.int32)
    nonzero_any = top_val > 0
    # any-other-nonzero = (sum of 3 - top) > 0 → second_val > 0 since
    # sorted descending. Vendor uses len(sorted_players) > 1 which is True
    # iff at least 2 classes have nonzero arrivals.
    has_second = second_val > 0
    tie = has_second & (top_val == second_val)
    survivor_ships = jnp.where(
        tie,
        jnp.float32(0.0),
        jnp.where(has_second, top_val - second_val, top_val),
    )
    # When tie, survivor_owner becomes "neutral" (-1 in vendor → no
    # ownership change). When no attackers, the carry is untouched. We
    # use OWNER_NEUTRAL (2) as the tie sentinel; the consumer below
    # ignores survivor when survivor_ships <= 0.
    survivor_owner = jnp.where(tie, jnp.int32(OWNER_NEUTRAL), top_idx)

    # Apply to current (owner_cls, garrison).
    no_change = (~nonzero_any) | (survivor_ships <= 0)
    same_owner = survivor_owner == owner_cls
    # garrison if same_owner: garrison + survivor_ships
    # else: garrison - survivor_ships; if negative, flip owner & take |neg|
    g_same = garrison + survivor_ships
    g_minus = garrison - survivor_ships
    flip = g_minus < 0
    new_owner_flip = survivor_owner
    new_garrison_flip = -g_minus
    new_owner_no_flip = owner_cls
    new_garrison_no_flip = g_minus

    new_owner_attack = jnp.where(flip, new_owner_flip, new_owner_no_flip)
    new_garrison_attack = jnp.where(flip, new_garrison_flip, new_garrison_no_flip)

    new_owner = jnp.where(
        no_change,
        owner_cls,
        jnp.where(same_owner, owner_cls, new_owner_attack),
    )
    new_garrison = jnp.where(
        no_change,
        jnp.maximum(0.0, garrison),
        jnp.where(same_owner, g_same, new_garrison_attack),
    )
    return new_owner, new_garrison


def _simulate_one_timeline(
    init_owner_cls: jax.Array,  # int32 scalar
    init_ships: jax.Array,  # float32 scalar
    production: jax.Array,  # float32 scalar
    arrivals_per_turn: jax.Array,  # float32 (HORIZON+1, 3)
    initial_garrison: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Simulate per-planet timeline; return ships_at, owner_at,
    min_owned (if planet was ally at t=0), fall_seen, fall_turn.

    Returns:
      ships_at:        (HORIZON+1,) float32
      owner_at:        (HORIZON+1,) int32
      min_owned_final: float32 (scalar) — min ally garrison across times
                       when owner == ALLY; 0 if planet was never ALLY.
      fall_seen:       bool — True iff planet flipped from ALLY to non-ALLY
      fall_turn:       int32 — first turn of the fall (HORIZON+1 if never)
    """
    horizon = TIMELINE_HORIZON

    if initial_garrison is None:
        initial_garrison = jnp.maximum(0.0, init_ships)

    init_min_owned = jnp.where(
        init_owner_cls == OWNER_ALLY, initial_garrison, jnp.float32(0.0)
    )

    def step(
        carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
        x: tuple[jax.Array, jax.Array],  # (arrivals_3, turn_idx)
    ) -> tuple[
        tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
        tuple[jax.Array, jax.Array],
    ]:
        owner_cls, garrison, min_owned, fall_seen, fall_turn = carry
        arrivals_3, turn_idx = x

        # Production: vendor adds production if owner != -1 (neutral). In
        # our 3-class encoding, NEUTRAL is class 2.
        garrison = garrison + jnp.where(
            owner_cls != OWNER_NEUTRAL, production, jnp.float32(0.0)
        )

        prev_owner = owner_cls
        new_owner, new_garrison = _resolve_arrival_step(owner_cls, garrison, arrivals_3)

        # Track fall: was ally, now not ally.
        was_ally = prev_owner == OWNER_ALLY
        now_not_ally = new_owner != OWNER_ALLY
        new_fall_this_turn = was_ally & now_not_ally
        new_fall_seen = fall_seen | new_fall_this_turn
        # First fall turn: keep first occurrence (only update if we haven't
        # seen one yet).
        new_fall_turn = jnp.where(new_fall_seen & ~fall_seen, turn_idx, fall_turn)

        # min_owned: min garrison while owner is ALLY.
        new_min_owned = jnp.where(
            new_owner == OWNER_ALLY,
            jnp.minimum(min_owned, new_garrison),
            min_owned,
        )

        new_carry = (
            new_owner,
            new_garrison,
            new_min_owned,
            new_fall_seen,
            new_fall_turn,
        )
        return new_carry, (new_garrison, new_owner)

    turns = jnp.arange(1, horizon + 1, dtype=jnp.int32)
    # arrivals_per_turn includes turn 0 (no arrivals applied at t=0).
    # Scan over turns 1..horizon → use arrivals_per_turn[1..horizon].
    init_carry = (
        init_owner_cls,
        initial_garrison,
        init_min_owned,
        jnp.bool_(False),
        jnp.int32(horizon + 1),
    )
    (
        (
            final_owner,
            final_garrison,
            final_min_owned,
            final_fall_seen,
            final_fall_turn,
        ),
        (ships_seq, owner_seq),
    ) = jax.lax.scan(step, init_carry, (arrivals_per_turn[1:], turns))

    # Prepend t=0 frame.
    ships_at = jnp.concatenate([initial_garrison[None], ships_seq], axis=0)
    owner_at = jnp.concatenate([init_owner_cls[None], owner_seq], axis=0)
    return ships_at, owner_at, final_min_owned, final_fall_seen, final_fall_turn


def _build_timeline_cols(
    px: jax.Array,
    py: jax.Array,
    owner: jax.Array,  # int32 (P,) raw player id (0/1/.../-1)
    ships_f: jax.Array,  # float32 (P,)
    prod_f: jax.Array,  # float32 (P,) production
    valid_full: jax.Array,
    player: int,
    eta_mat: jax.Array,  # float32 (F, P)
    in_horizon: jax.Array,  # bool (F, P)
    fl_is_ally: jax.Array,  # bool (F, 1)
    fl_is_enemy: jax.Array,  # bool (F, 1)
    fl_is_neutral_only: jax.Array,  # bool (F, 1)
    fships_f: jax.Array,  # float32 (F,)
) -> jax.Array:
    """Return (P, 6) timeline summary cols: loss_3turn, ttf_norm, min_owned,
    surplus, fall_predicted, keep_needed (already log1p'd / normalized where
    applicable to match the planet_feats spec).
    """
    P = MAX_PLANETS
    H = TIMELINE_HORIZON

    # Per-planet 3-class initial owner.
    init_owner_cls = jnp.where(
        owner == player,
        jnp.int32(OWNER_ALLY),
        jnp.where(owner == -1, jnp.int32(OWNER_NEUTRAL), jnp.int32(OWNER_ENEMY)),
    )  # (P,)

    # Per-fleet integer eta: max(1, ceil(eta)). Vendor's normalize_arrivals
    # uses ceil. Mask to horizon.
    eta_int = jnp.maximum(1, jnp.ceil(eta_mat)).astype(jnp.int32)
    valid_arr = in_horizon & (eta_int <= H)  # (F, P)

    # Owner class per (F, P): broadcast fleet owner; planet doesn't matter.
    # fl_is_* are (F, 1) so broadcast to (F, P).
    fl_cls = jnp.where(
        fl_is_ally,
        jnp.int32(OWNER_ALLY),
        jnp.where(fl_is_neutral_only, jnp.int32(OWNER_NEUTRAL), jnp.int32(OWNER_ENEMY)),
    )  # (F, 1)
    fl_cls_b = jnp.broadcast_to(fl_cls, (eta_mat.shape[0], P))

    # Build (P, H+1, 3) arrivals: for each (planet, turn, owner_class),
    # sum fleet ships matching that bucket.
    # Approach: scatter-add via segment_sum or vectorized where.
    # The (F, P) eta_int can be encoded as a one-hot turn mask, multiplied
    # by ships and class mask, then reduced over fleets.
    # Memory: F * P * (H+1) = 512 * 36 * 31 = ~570k entries; acceptable.
    turn_onehot = eta_int[..., None] == jnp.arange(
        0, H + 1, dtype=jnp.int32
    )  # (F, P, H+1) bool
    cls_onehot = fl_cls_b[..., None] == jnp.arange(
        0, 3, dtype=jnp.int32
    )  # (F, P, 3) bool

    # We want arrivals[p, t, c] = sum_f ships_f[f] * valid_arr[f, p] *
    #                                turn_onehot[f, p, t] *
    #                                cls_onehot[f, p, c]
    # Combine first: (F, P, H+1, 3)
    ships_fp = (
        fships_f[:, None, None, None] * valid_arr[..., None, None]
    )  # (F, P, 1, 1)
    selectors = turn_onehot[..., None] & cls_onehot[..., None, :]  # (F, P, H+1, 3)
    arrivals_ptc = jnp.sum(
        ships_fp * selectors.astype(jnp.float32), axis=0
    )  # (P, H+1, 3)

    # Run the per-planet scan via vmap over P.
    def _run(
        owner_cls: jax.Array,
        ships: jax.Array,
        prod: jax.Array,
        arrivals: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        return _simulate_one_timeline(owner_cls, ships, prod, arrivals)

    ships_at, owner_at, min_owned, fall_seen, fall_turn = jax.vmap(_run)(
        init_owner_cls, ships_f, prod_f, arrivals_ptc
    )
    # ships_at: (P, H+1), owner_at: (P, H+1).

    # Summarize:
    s0 = ships_at[:, 0]  # = init garrison
    s_short = ships_at[:, jnp.minimum(SHORT_WINDOW, H)]  # (P,)
    owner_short = owner_at[:, jnp.minimum(SHORT_WINDOW, H)]
    same_owner_short = owner_at[:, 0] == owner_short
    loss_3turn = jnp.where(
        same_owner_short,
        jnp.maximum(0.0, s0 - s_short),
        jnp.maximum(0.0, s0),
    )

    ttf_norm = jnp.where(
        fall_seen,
        jnp.maximum(
            0.0,
            jnp.minimum(
                1.0,
                fall_turn.astype(jnp.float32) / jnp.float32(jnp.maximum(1, H)),
            ),
        ),
        jnp.float32(1.0),
    )

    s_h = ships_at[:, H]
    own_h = owner_at[:, H]
    surplus = jnp.where(own_h == owner_at[:, 0], s_h, jnp.float32(0.0))
    fall_predicted = fall_seen.astype(jnp.float32)

    # min_owned: floor & clamp (vendor: max(0, int(floor(min_owned))))
    min_owned_clamped = jnp.maximum(jnp.float32(0.0), jnp.floor(min_owned))
    # Vendor returns 0 if planet.owner != player (init_owner_cls != ALLY).
    min_owned_clamped = jnp.where(
        init_owner_cls == OWNER_ALLY, min_owned_clamped, jnp.float32(0.0)
    )

    # keep_needed binary search: only for ally planets. Run KEEP_BSEARCH_ITERS
    # iterations of bsearch; track whether full ships works.
    def _survives_with_keep(
        owner_cls: jax.Array,
        keep: jax.Array,
        prod: jax.Array,
        arrivals: jax.Array,
    ) -> jax.Array:
        # Replicate vendor `survives_with_keep`: simulate with garrison=keep
        # and check if owner stays ALLY across the full horizon AND ends
        # ALLY. Vendor's `survives_with_keep` returns True iff owner remains
        # ALLY throughout (including final).
        # We use _simulate_one_timeline with initial_garrison = keep, then
        # check fall_seen == False AND owner_at[H] == ALLY.
        ships_at_k, owner_at_k, _mo, fs, _ft = _simulate_one_timeline(
            owner_cls,
            jnp.float32(keep),
            prod,
            arrivals,
            initial_garrison=jnp.float32(keep),
        )
        return (~fs) & (owner_at_k[H] == OWNER_ALLY)

    def _keep_needed_one(
        owner_cls: jax.Array,
        ships: jax.Array,
        prod: jax.Array,
        arrivals: jax.Array,
    ) -> jax.Array:
        # Only valid for ally planets; non-ally returns 0.
        is_ally = owner_cls == OWNER_ALLY
        full_ships = jnp.float32(ships)
        survives_full = _survives_with_keep(owner_cls, full_ships, prod, arrivals)

        # Binary search [0, ships]; result is min keep that survives.
        # We use fixed-iter binary search.
        def cond(_state: tuple[jax.Array, jax.Array]) -> bool:
            return True  # unused; using lax.fori_loop fixed iters

        def body(
            _i: jax.Array, state: tuple[jax.Array, jax.Array]
        ) -> tuple[jax.Array, jax.Array]:
            lo, hi = state
            mid = (lo + hi) // 2
            ok = _survives_with_keep(owner_cls, mid, prod, arrivals)
            new_lo = jnp.where(ok, lo, mid + 1)
            new_hi = jnp.where(ok, mid, hi)
            return (new_lo, new_hi)

        lo_init = jnp.int32(0)
        hi_init = jnp.maximum(jnp.int32(0), ships.astype(jnp.int32))
        lo_f, _hi_f = jax.lax.fori_loop(0, KEEP_BSEARCH_ITERS, body, (lo_init, hi_init))
        keep_needed = jnp.where(survives_full, lo_f.astype(jnp.float32), full_ships)
        return jnp.where(is_ally, keep_needed, jnp.float32(0.0))

    keep_needed = jax.vmap(_keep_needed_one)(
        init_owner_cls, ships_f, prod_f, arrivals_ptc
    )

    # Assemble (P, 6) according to featurizer.py:374-413 order:
    # idx 35: log1p(loss_3turn)
    # idx 36: ttf_norm
    # idx 37: log1p(min_owned)
    # idx 38: log1p(surplus)
    # idx 39: fall_predicted
    # idx 40: log1p(keep_needed)
    out = jnp.stack(
        [
            jnp.log1p(loss_3turn),
            ttf_norm,
            jnp.log1p(min_owned_clamped),
            jnp.log1p(surplus),
            fall_predicted,
            jnp.log1p(keep_needed),
        ],
        axis=-1,
    )
    # Zero out invalid planets (PyTorch only fills for valid slots).
    out = out * valid_full[:, None].astype(jnp.float32)
    return out


def featurize_jax_w1(
    state: EnvState,
    player: int,
    history: HistoryStateJax | None = None,
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
    if history is None:
        history = init_history_jax()

    # Slice / truncate to MAX_PLANETS (= 36, smaller than jax_env's 48).
    # In practice vendor generates at most 24-32 planets at reset and the
    # game adds 4 per comet activation, capped well below 36 unless many
    # comets are simultaneously active.
    pid_arr = state.planet_id[:MAX_PLANETS]
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
    nearest_ally = jnp.min(jnp.where(j_is_ally, dist_mat, large), axis=1)
    nearest_ally = jnp.where(jnp.isfinite(nearest_ally), nearest_ally, diag_f)
    nearest_enemy = jnp.min(jnp.where(j_is_enemy, dist_mat, large), axis=1)
    nearest_enemy = jnp.where(jnp.isfinite(nearest_enemy), nearest_enemy, diag_f)
    nearest_neutral = jnp.min(jnp.where(j_is_neutral, dist_mat, large), axis=1)
    nearest_neutral = jnp.where(jnp.isfinite(nearest_neutral), nearest_neutral, diag_f)

    j_ships = ships.astype(jnp.float32)[None, :]
    support_density = jnp.sum(
        jnp.where(j_is_ally & (dist_mat <= NEIGHBOR_RADIUS_LONG), j_ships, 0.0),
        axis=1,
    )
    threat_pressure_planet = jnp.sum(
        jnp.where(j_is_enemy & (dist_mat <= NEIGHBOR_RADIUS_SHORT), j_ships, 0.0),
        axis=1,
    )

    feats = feats.at[:, 14].set(nearest_enemy / diag_f)
    feats = feats.at[:, 15].set(jnp.log1p(support_density) / LOG_NORM_DENOM)
    # idx 16: planet half only — vendor adds fleet contributions in W2-fleet.
    # Leave as the planet half so subsequent W2-fleet only adds the fleet
    # term. The PyTorch reference includes fleet contributions which are
    # zero in the no-fleet parity fixture (W1 path).
    feats = feats.at[:, 16].set(jnp.log1p(threat_pressure_planet) / LOG_NORM_DENOM)
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
    f_speed = jnp.maximum(0.5, 2.0 - 0.05 * jnp.sqrt(jnp.maximum(1.0, fships_f)))

    dx_fp = px[None, :] - fx[:, None]  # (F, P)
    dy_fp = py[None, :] - fy[:, None]  # (F, P)
    dir_x = jnp.cos(fang)
    dir_y = jnp.sin(fang)
    proj = dx_fp * dir_x[:, None] + dy_fp * dir_y[:, None]  # (F, P)
    perp_sq = dx_fp * dx_fp + dy_fp * dy_fp - proj * proj
    radius_sq = (radius_f * radius_f)[None, :]
    in_cone = (proj >= 0) & (perp_sq < radius_sq)
    hit_d = jnp.maximum(0.0, proj - jnp.sqrt(jnp.maximum(0.0, radius_sq - perp_sq)))
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
    nearest_eta = jnp.where(jnp.isfinite(nearest_eta), nearest_eta, horizon_default)
    ally_eta_min = jnp.min(
        jnp.where(fl_is_ally, eta_masked, jnp.float32(jnp.inf)), axis=0
    )
    ally_eta_min = jnp.where(jnp.isfinite(ally_eta_min), ally_eta_min, horizon_default)
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
        jnp.log1p(threat_pressure_planet + threat_pressure_fleet) / LOG_NORM_DENOM
    )

    # idx 9: log1p(incoming_enemy) - log1p(incoming_ally)
    feats = feats.at[:, 9].set(jnp.log1p(incoming_enemy) - jnp.log1p(incoming_ally))
    # idx 10: eta_norm = min(nearest_eta, HORIZON+1) / (HORIZON+1)
    feats = feats.at[:, 10].set(
        jnp.minimum(nearest_eta, horizon_default) / horizon_default
    )
    # idx 17: net_signed = (incoming_enemy - incoming_ally) / max(1, ships)
    # then clamp to [-3, 3] and divide by 3
    ships_safe = jnp.maximum(1.0, ships_f)
    net_signed = (incoming_enemy - incoming_ally) / ships_safe
    feats = feats.at[:, 17].set(jnp.maximum(-3.0, jnp.minimum(3.0, net_signed)) / 3.0)
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
    # W2e: timeline cols 35..40.
    timeline_cols = _build_timeline_cols(
        px,
        py,
        owner,
        ships_f,
        prod_f,
        valid_full,
        player,
        eta_mat,
        in_horizon,
        fl_is_ally,
        fl_is_enemy,
        fl_is_neutral_only,
        fships_f,
    )
    feats = feats.at[:, 35:41].set(timeline_cols)

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
    eq_mask = (planet_idx[:, None] == flat_slots[None, :]) & (flat_slots[None, :] >= 0)
    has_comet_entry = jnp.any(eq_mask, axis=-1)  # (P,)
    pick_idx = jnp.argmax(eq_mask.astype(jnp.int32), axis=-1)  # (P,) into C*4
    c_of_p = pick_idx // 4
    q_of_p = pick_idx % 4
    # path_index of the comet that owns this planet (0 if no comet).
    path_index_p = jnp.where(
        has_comet_entry, state.comet_path_index[c_of_p], jnp.int32(0)
    )
    path_len_p = jnp.where(has_comet_entry, state.comet_path_len[c_of_p], jnp.int32(0))

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

    # ------------------------------------------------------------------
    # W2d: history-dependent per-planet cols 32, 33, 34.
    #
    # PyTorch reads `prev_planet_snapshots[-2]` (snap_t1) and `[-3]`
    # (snap_t2) by planet id. For each current planet, look up its id in
    # both snapshots (if present); compute
    #   delta = (current_ships - prev_ships) / max(1, current_ships)
    #   clamped to [-3, 3] / 3
    # delta_t1 → col 32, delta_t2 → col 33.
    # owner_changed (col 34) = 1.0 if snap_t1 had a different owner; uses
    # snap_t1 only.
    snap_t1_pos = (history.snap_head - 2) % N_PREV_SNAPSHOTS
    snap_t2_pos = (history.snap_head - 3) % N_PREV_SNAPSHOTS
    snap_t1_available = history.snap_count >= 2
    snap_t2_available = history.snap_count >= 3

    snap_t1_pid = history.snap_pid[snap_t1_pos]  # (MAX_PLANETS,)
    snap_t1_ships = history.snap_ships[snap_t1_pos]
    snap_t1_owner = history.snap_owner[snap_t1_pos]
    snap_t1_valid_mask = history.snap_valid[snap_t1_pos]
    snap_t2_pid = history.snap_pid[snap_t2_pos]
    snap_t2_ships = history.snap_ships[snap_t2_pos]
    snap_t2_valid_mask = history.snap_valid[snap_t2_pos]

    # For each current planet i, find row j in snap_t1 where pid matches.
    # eq_mat[i, j] = True iff current_pid[i] == snap_t1_pid[j] AND both
    # valid.
    eq_t1 = (
        (pid_arr[:, None] == snap_t1_pid[None, :])
        & snap_t1_valid_mask[None, :]
        & valid_full[:, None]
    )
    has_match_t1 = jnp.any(eq_t1, axis=1) & snap_t1_available
    match_idx_t1 = jnp.argmax(eq_t1.astype(jnp.int32), axis=1)
    prev_ships_t1 = snap_t1_ships[match_idx_t1].astype(jnp.float32)
    prev_owner_t1 = snap_t1_owner[match_idx_t1]
    ships_safe_p = jnp.maximum(1.0, ships_f)
    raw_delta_t1 = (ships_f - prev_ships_t1) / ships_safe_p
    delta_t1 = jnp.maximum(-3.0, jnp.minimum(3.0, raw_delta_t1)) / 3.0
    delta_t1 = jnp.where(has_match_t1, delta_t1, 0.0)
    owner_changed = jnp.where(
        has_match_t1, (prev_owner_t1 != owner).astype(jnp.float32), 0.0
    )

    eq_t2 = (
        (pid_arr[:, None] == snap_t2_pid[None, :])
        & snap_t2_valid_mask[None, :]
        & valid_full[:, None]
    )
    has_match_t2 = jnp.any(eq_t2, axis=1) & snap_t2_available
    match_idx_t2 = jnp.argmax(eq_t2.astype(jnp.int32), axis=1)
    prev_ships_t2 = snap_t2_ships[match_idx_t2].astype(jnp.float32)
    raw_delta_t2 = (ships_f - prev_ships_t2) / ships_safe_p
    delta_t2 = jnp.maximum(-3.0, jnp.minimum(3.0, raw_delta_t2)) / 3.0
    delta_t2 = jnp.where(has_match_t2, delta_t2, 0.0)

    feats = feats.at[:, 32].set(delta_t1)
    feats = feats.at[:, 33].set(delta_t2)
    feats = feats.at[:, 34].set(owner_changed)

    # idx 35..40: timeline columns — W2e (deferred, complex per-turn
    # multi-attacker resolution + binary search for keep_needed). The
    # PyTorch path emits 6 cols here; we leave them zero for now and will
    # come back if BC accuracy degrades.

    # ------------------------------------------------------------------
    # W2f: per-source template-context block (template_ctx, MAX_PLANETS
    # × 40 = 8 templates × 5 feats).
    # For each "own" planet i (src), resolve 7 template targets (id 0-6)
    # against all other planets and emit per-template
    # (score, prox, ship_adv, tgt_is_enemy, tgt_is_mine). NO_OP slot
    # (template id 7) emits 1.0 in its score column when no template
    # found a target, else 0.0.
    template_ctx_arr = _build_template_ctx(
        px,
        py,
        ships_f,
        prod_f,
        owner,
        valid_full,
        is_mine,
        player,
    )

    # ------------------------------------------------------------------
    # W2-final: candidate block (CAND_K=8 per own src × 14 feats each).
    cand_feats_arr, cand_mask_arr, cand_pid_arr = _build_candidate_block(
        px,
        py,
        pid_arr,
        radius_f,
        ships,
        ships_f,
        prod_f,
        owner,
        valid_full,
        is_mine,
        player,
    )

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

    # Launch-history aggregates over the last HISTORY_TURNS steps (W2d).
    # PyTorch (featurizer.py:465-478) filters launches by
    # `ev.step >= step - HISTORY_TURNS` and bins by owner.
    launch_threshold = step - HISTORY_TURNS
    launch_in_window = history.launch_valid & (history.launch_step >= launch_threshold)
    launch_is_ally = history.launch_owner == player
    launch_is_neutral = history.launch_owner == -1
    launch_is_enemy = ~launch_is_ally & ~launch_is_neutral
    ally_count_lh = jnp.sum(
        (launch_in_window & launch_is_ally).astype(jnp.int32)
    ).astype(jnp.float32)
    ally_ships_lh = jnp.sum(
        jnp.where(launch_in_window & launch_is_ally, history.launch_ships, 0)
    ).astype(jnp.float32)
    enemy_count_lh = jnp.sum(
        (launch_in_window & launch_is_enemy).astype(jnp.int32)
    ).astype(jnp.float32)
    enemy_ships_lh = jnp.sum(
        jnp.where(launch_in_window & launch_is_enemy, history.launch_ships, 0)
    ).astype(jnp.float32)

    g = jnp.zeros((GLOBAL_FEAT_DIM,), dtype=jnp.float32)
    g = g.at[0].set(step.astype(jnp.float32) / 500.0)
    g = g.at[1].set(ang_vel * 10.0)
    g = g.at[2].set(jnp.log1p(my_ships_total))
    g = g.at[3].set(jnp.log1p(enemy_ships_total))
    g = g.at[4].set(jnp.log1p(neutral_ships_total))
    g = g.at[5].set(jnp.log1p(my_prod_total) - jnp.log1p(enemy_prod_total))
    g = g.at[6].set(my_count.astype(jnp.float32) / total_planets.astype(jnp.float32))
    g = g.at[7].set(enemy_count.astype(jnp.float32) / total_planets.astype(jnp.float32))
    g = g.at[8].set(_comet_active(step).astype(jnp.float32))
    g = g.at[9].set(phase_mid)
    g = g.at[10].set(phase_late)
    g = g.at[11].set(jnp.minimum(1.0, next_eta))
    g = g.at[12].set(jnp.where(total_ships > 0, my_ships_total / total_ships, 0.0))
    g = g.at[13].set(jnp.where(total_ships > 0, enemy_ships_total / total_ships, 0.0))
    g = g.at[14].set(jnp.where(total_prod > 0, my_prod_total / total_prod, 0.0))
    g = g.at[15].set(jnp.maximum(-3.0, jnp.minimum(3.0, score_diff)) / 3.0)
    g = g.at[16].set(jnp.minimum(1.0, enemy_count_lh / LAUNCH_COUNT_NORM))
    g = g.at[17].set(jnp.log1p(enemy_ships_lh) / LOG_NORM_DENOM)
    g = g.at[18].set(jnp.minimum(1.0, ally_count_lh / LAUNCH_COUNT_NORM))
    g = g.at[19].set(jnp.log1p(ally_ships_lh) / LOG_NORM_DENOM)

    # Add batch dimension. Outside vmap caller gets B=1; inside vmap the
    # caller wraps featurize_jax_w1 in jax.vmap(..., in_axes=(0, None)).
    batch_planet_feats = feats[None, ...]
    batch_planet_mask = valid_full[None, ...]
    batch_my_planet_mask = is_mine[None, ...]
    batch_target_mask = (valid_full & ~is_mine)[None, ...]
    batch_global_feats = g[None, ...]
    batch_template_ctx = template_ctx_arr[None, ...]
    batch_candidate_feats = cand_feats_arr[None, ...]
    batch_candidate_mask = cand_mask_arr[None, ...]
    batch_candidate_pid = cand_pid_arr[None, ...]

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


def update_history_jax(
    history: HistoryStateJax,
    state: EnvState,
    actions_pid: jax.Array,
    actions_ships: jax.Array,
    actions_valid: jax.Array,
    player: int,
) -> HistoryStateJax:
    """Update the history pytree mirroring PyTorch `update_history`.

    Arguments:
      history: previous history pytree.
      state: current EnvState (provides planet snapshot + step).
      actions_pid, actions_ships, actions_valid: the launcher player's
        actions for this turn. Each is shape (L,) where L is the number
        of launch slots. Invalid slots have actions_valid[i] == False.
      player: viewing player (used as `owner` for the recorded launches,
        matching PyTorch which records `history.recent_launches[i].owner
        = player`).

    Returns:
      New HistoryStateJax with the current snapshot appended and any
      launches inserted into the ring. Launches whose step has aged out
      (step < current_step - HISTORY_TURNS) are *not* explicitly
      evicted; instead the featurizer filters by step in-window when
      reading.
    """
    # Push current state's planet snapshot into the ring at snap_head.
    pos = history.snap_head
    new_snap_ships = history.snap_ships.at[pos].set(state.planet_ships[:MAX_PLANETS])
    new_snap_owner = history.snap_owner.at[pos].set(state.planet_owner[:MAX_PLANETS])
    new_snap_pid = history.snap_pid.at[pos].set(state.planet_id[:MAX_PLANETS])
    new_snap_valid = history.snap_valid.at[pos].set(state.planet_valid[:MAX_PLANETS])
    new_snap_count = jnp.minimum(jnp.int32(N_PREV_SNAPSHOTS), history.snap_count + 1)
    new_snap_head = (history.snap_head + 1) % N_PREV_SNAPSHOTS

    # Append launches to the ring buffer. The number of new launch slots
    # is L = actions_pid.shape[0]. We find the next free slots in
    # launch_valid (sorted in any order); for simplicity, overwrite the
    # oldest slots when full.
    cur_step = state.step.astype(jnp.int32)

    # Strategy: maintain a head pointer in launch_step's len, modulo
    # LAUNCH_BUFFER. We'll track head implicitly by writing into the
    # next L slots starting from `launch_head` which we store as the
    # number of writes so far modulo LAUNCH_BUFFER. To avoid adding
    # another scalar, we use the position of the first launch_valid =
    # False as the head; if all valid, wrap to 0. This is approximate
    # but works since old entries are filtered by step in-window anyway.
    L = actions_pid.shape[0]
    # Find the first free slot index.
    first_free = jnp.argmax((~history.launch_valid).astype(jnp.int32))
    # If launch_valid is all-True, argmax returns 0 (and free will be
    # False); we wrap by always assuming first_free is a valid slot to
    # overwrite — the featurizer's in-window step filter handles old
    # entries.
    indices = (first_free + jnp.arange(L, dtype=jnp.int32)) % LAUNCH_BUFFER

    # Mask out invalid action slots.
    write_mask = actions_valid
    # Use scatter via .at[].set with where-style guard.
    new_launch_step = history.launch_step
    new_launch_owner = history.launch_owner
    new_launch_ships = history.launch_ships
    new_launch_valid = history.launch_valid
    # We need a scatter that respects write_mask. JAX has no
    # straightforward masked scatter, so do it as a for-loop over L
    # (unrolled at trace time since L is static).
    for i in range(L):
        idx = indices[i]
        write_now = write_mask[i]
        new_launch_step = new_launch_step.at[idx].set(
            jnp.where(write_now, cur_step, new_launch_step[idx])
        )
        new_launch_owner = new_launch_owner.at[idx].set(
            jnp.where(write_now, jnp.int32(player), new_launch_owner[idx])
        )
        new_launch_ships = new_launch_ships.at[idx].set(
            jnp.where(write_now, actions_ships[i], new_launch_ships[idx])
        )
        new_launch_valid = new_launch_valid.at[idx].set(
            write_now | new_launch_valid[idx]
        )

    return HistoryStateJax(
        snap_ships=new_snap_ships,
        snap_owner=new_snap_owner,
        snap_pid=new_snap_pid,
        snap_valid=new_snap_valid,
        snap_count=new_snap_count,
        snap_head=new_snap_head,
        launch_step=new_launch_step,
        launch_owner=new_launch_owner,
        launch_ships=new_launch_ships,
        launch_valid=new_launch_valid,
    )


__all__ = [
    "BatchFeaturesJax",
    "HistoryStateJax",
    "init_history_jax",
    "update_history_jax",
    "featurize_jax_w1",
    "PLANET_FEAT_DIM",
    "GLOBAL_FEAT_DIM",
    "MAX_PLANETS",
    "TEMPLATE_CTX_DIM",
    "CAND_K",
    "CAND_FEAT_DIM",
]
