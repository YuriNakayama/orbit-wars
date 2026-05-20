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
    # idx 14..20: nearest_enemy_dist, support_density, threat_pressure_short,
    # net_signed, nearest_ally_dist, nearest_neutral_dist — W2 (require
    # planet×planet pairwise distances + fleet ETA accumulation)
    # idx 21..28: orbit predictions — W2 (calls geometry.py predict_*)
    # idx 29..34: ally/enemy_eta_norm, incoming_*_ships_log, delta_t1/t2,
    # owner_changed — W2
    # idx 35..40: timeline columns — W2

    # Mask invalid slots — zero out all columns for them.
    feats = feats * valid_full[:, None].astype(jnp.float32)

    # Global features.
    step = state.step.astype(jnp.int32)
    ang_vel = state.angular_velocity.astype(jnp.float32)

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
