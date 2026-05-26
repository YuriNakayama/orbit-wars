"""JAX rule-based agent (full variant) — stronger opponent for PPO training.

vs baseline_jax_lite: adds orbit prediction with 2-step Banach iteration
on travel_time so we aim at the target's *future* xy at arrival, not its
current xy. This is Phase 1; Phases 2-4 add snipe, swarm, defense, cap.

Strategy outline (Phase 1):
    1. Compute per-source surplus = ships - reserve  (reserve = lite version)
    2. (P, P) pair distance via predicted target xy at expected arrival
       (fixed-point: t0 = dist_now/speed; t1 = dist(predict(t0))/speed)
    3. need_to_capture uses target ships + travel*prod + buffer
    4. pair_score = (prod * W - need) / (dist + 1)
    5. per-source argmax → action tensor (MAX_LAUNCHES_PER_AGENT, 3)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jax_env.constants import CENTER, MAX_PLANETS
from jax_env.state import EnvState
from jax_env.step import MAX_LAUNCHES_PER_AGENT

from .geometry_jax import predict_planet_xy

# Vendor constants — replicated to keep this module self-contained.
SUN_RADIUS: float = 10.0
PRODUCTION_TURNS_RESERVE: int = 3
SCORE_PROD_WEIGHT: float = 8.0
NEED_BUFFER: int = 1

# Phase 2 (crash_exploit): planets near the sun lose ships every turn
# (vendor: sun damage on overlap). We bias toward capturing them with
# fewer ships because the sun will help finish them off after we land.
CRASH_EXPLOIT_RADIUS_MARGIN: float = 5.0  # planet_radius + margin from SUN_RADIUS
CRASH_EXPLOIT_NEED_DISCOUNT: float = 0.5  # need * 0.5 for crash-prone targets
CRASH_EXPLOIT_SCORE_BONUS: float = 4.0  # add to base_score

# Phase 3 (pair swarm): when a single source can't afford a high-value
# target alone, two sources can coordinate. We allow the *second-best*
# source per target to also fire if (1) the primary argmax source for
# that target lacks surplus, and (2) the combined surplus covers `need`.
PAIR_SWARM_MIN_TARGET_SCORE: float = -10.0  # only pair on positive-score targets
PAIR_SWARM_NEED_RATIO: float = 0.5  # each pair src contributes >= 0.5 * need


def _fleet_speed(ships: jax.Array) -> jax.Array:
    """Vendor: speed = max(0.5, 2.0 - 0.05 * sqrt(ships))."""
    return jnp.maximum(0.5, 2.0 - 0.05 * jnp.sqrt(jnp.maximum(1.0, ships)))


def compute_actions_jax(state: EnvState, seat: int) -> jax.Array:
    """Return (MAX_LAUNCHES_PER_AGENT, 3) launch tensor for one seat."""
    pid = state.planet_id  # (P,)
    owner = state.planet_owner  # (P,)
    xy = state.planet_xy  # (P, 2) — current (y, x)
    init_xy = state.planet_initial_xy  # (P, 2) — used by orbit pred
    radius = state.planet_radius  # (P,)
    ships = state.planet_ships.astype(jnp.float32)  # (P,)
    prod = state.planet_prod.astype(jnp.float32)  # (P,)
    valid = state.planet_valid  # (P,)
    cur_step = state.step  # () int

    seat_i32 = jnp.int32(seat)
    is_mine = valid & (owner == seat_i32)
    is_target = valid & (owner != seat_i32)

    # ---- reserve: lite-compatible flat 3-turn (Phase 4 ETA-based was
    # over-reserving, causing full to lose to lite in self-play. Keep
    # lite parity for now; revisit in Phase 6 tuning.).
    reserve = jnp.ceil(prod * jnp.float32(PRODUCTION_TURNS_RESERVE)).astype(jnp.float32)
    reserve = jnp.minimum(reserve, ships)
    surplus = jnp.where(is_mine, jnp.maximum(0.0, ships - reserve), 0.0)  # (P,)

    # ---- pair distance with 2-step orbit-aware fixed-point -------------
    # Step 0: distance from src current xy to target current xy.
    src_xy = xy[:, None, :]
    tgt_xy_now = xy[None, :, :]
    dxdy0 = tgt_xy_now - src_xy
    dist0 = jnp.sqrt(dxdy0[..., 0] ** 2 + dxdy0[..., 1] ** 2 + 1e-8)  # (P, P)

    # Speed: depends on surplus[src] (proxy for ships sent).
    src_ships_proxy = jnp.maximum(1.0, surplus)[:, None]  # (P, 1)
    speed = _fleet_speed(src_ships_proxy)  # (P, 1)

    t0 = dist0 / speed  # (P, P) — turns to reach target's current xy

    # Step 1: predict target xy after t0 turns (use mean t0 per target as
    # a scalar advance per target — keep cheap; per-pair predict would be
    # (P, P, 2) memory). Mean across src for each target column.
    mean_t0_per_target = jnp.mean(t0, axis=0)  # (P,)

    # predict_planet_xy expects scalar dt; we vmap over per-target dt.
    # To avoid that, just use per-target dt as a (P,) scalar broadcast.
    def _predict_per_target(dt_scalar: jax.Array) -> jax.Array:
        return predict_planet_xy(
            init_xy,
            radius,
            state.angular_velocity,
            game_step=cur_step,
            dt=dt_scalar,
        )  # (P, 2)

    # Vmap over per-target dt to get (P_target, P_planet, 2). We only need
    # the diagonal: predicted xy of target i at dt_i.
    pred_all = jax.vmap(_predict_per_target)(mean_t0_per_target)  # (P, P, 2)
    pred_per_target = jnp.take_along_axis(
        pred_all,
        jnp.arange(MAX_PLANETS)[:, None, None].repeat(2, axis=-1),
        axis=1,
    ).squeeze(1)  # (P, 2) — predicted xy at target's own arrival dt

    tgt_xy_pred = pred_per_target[None, :, :]  # (1, P, 2)
    dxdy1 = tgt_xy_pred - src_xy
    dx = dxdy1[..., 0]
    dy = dxdy1[..., 1]
    dist = jnp.sqrt(dx * dx + dy * dy + 1e-8)  # (P, P) — refined distance
    travel_turns = dist / speed  # (P, P)

    # ---- need to capture -----------------------------------------------
    tgt_ships = ships[None, :]
    tgt_prod = prod[None, :]
    raw_need = jnp.ceil(tgt_ships + travel_turns * tgt_prod).astype(jnp.float32)
    need = raw_need + jnp.float32(NEED_BUFFER)

    # ---- Phase 2 crash_exploit: discount need + score boost for
    # planets near the sun. They lose ships to sun damage each turn so
    # we can take them with fewer ships and earlier than headline `need`
    # would suggest. Detection: distance(target, sun=CENTER) < SUN_RADIUS
    # + planet_radius + margin.
    tgt_dx_sun = xy[:, 0] - jnp.float32(CENTER)
    tgt_dy_sun = xy[:, 1] - jnp.float32(CENTER)
    tgt_sun_dist = jnp.sqrt(tgt_dx_sun**2 + tgt_dy_sun**2 + 1e-8)  # (P,)
    crash_threshold = (
        jnp.float32(SUN_RADIUS) + radius + jnp.float32(CRASH_EXPLOIT_RADIUS_MARGIN)
    )  # (P,)
    is_crash_prone = (tgt_sun_dist < crash_threshold) & is_target  # (P,)
    crash_mask_b = is_crash_prone[None, :]  # (1, P)
    need = jnp.where(
        crash_mask_b, need * jnp.float32(CRASH_EXPLOIT_NEED_DISCOUNT), need
    )
    need = jnp.maximum(need, jnp.float32(1.0))  # min 1 ship

    # ---- pair score ----------------------------------------------------
    base_score = (tgt_prod * jnp.float32(SCORE_PROD_WEIGHT) - need) / (dist + 1.0)
    base_score = base_score + crash_mask_b.astype(jnp.float32) * jnp.float32(
        CRASH_EXPLOIT_SCORE_BONUS
    )

    not_self = ~jnp.eye(MAX_PLANETS, dtype=jnp.bool_)
    pair_basic_mask = is_mine[:, None] & is_target[None, :] & not_self

    # Phase 3 pair swarm: a src is allowed even if surplus alone < need,
    # provided the *total* surplus over my planets that pick this target
    # covers need. We approximate this in a fixed-shape way: count, per
    # target, the top-2 src by score; if their combined surplus >= need,
    # both are allowed (each ships `need * PAIR_SWARM_NEED_RATIO`).

    # Affordable in 1 shot.
    solo_afford = pair_basic_mask & (need <= surplus[:, None])
    # Affordable in 2 shots (each src contributes >= need * ratio).
    # Phase 3 swarm: temporarily disabled — initial broadcast logic was
    # buggy and we couldn't separate its effect from orbit/crash. Will
    # re-enable in Phase 6 tuning.
    half_threshold = need * jnp.float32(PAIR_SWARM_NEED_RATIO)
    pair_afford = jnp.zeros_like(pair_basic_mask)

    # pair is valid only if score positive enough (avoid griefing self).
    score_floor = jnp.float32(PAIR_SWARM_MIN_TARGET_SCORE)
    pair_afford = pair_afford & (base_score > score_floor)

    # For pair-only targets we also need to confirm the *2nd-best* src
    # for that target also has half-need surplus. Use a fixed top-2
    # check: per target, look at the 2nd-largest surplus in pair_afford
    # rows. Use jnp.where mask then take top-2 of the surplus column.
    # (For runtime: shape (P, P); jnp.sort O(P log P) per col; P=36, OK.)
    masked_surplus_for_tgt = jnp.where(
        pair_afford, surplus[:, None], jnp.float32(-1.0)
    )  # (P, P)
    # top-2 surplus along src axis for each target.
    sorted_desc = -jnp.sort(-masked_surplus_for_tgt, axis=0)  # (P, P)
    top2_surplus = sorted_desc[1, :]  # (P,) — 2nd largest per target
    # pair viable only where top-2 surplus >= half_threshold for that target.
    half_t = (
        half_threshold[0, :]
        if half_threshold.ndim == 2
        else jnp.broadcast_to(half_threshold, (MAX_PLANETS,))
    )
    # Recompute half_t cleanly: need is (1, P); ratio scalar; thresh (1, P)→(P,)
    half_t = need[0, :] * jnp.float32(PAIR_SWARM_NEED_RATIO)
    pair_viable_target = top2_surplus >= half_t  # (P,)
    pair_afford = pair_afford & pair_viable_target[None, :]

    valid_mask = solo_afford | pair_afford
    pair_score = jnp.where(valid_mask, base_score, jnp.float32(-1e9))

    # ---- per-source argmax target --------------------------------------
    tgt_idx = jnp.argmax(pair_score, axis=1)  # (P,)
    row_has_valid = jnp.any(valid_mask, axis=1)  # (P,)

    chosen_need = jnp.take_along_axis(need, tgt_idx[:, None], axis=1).squeeze(-1)
    chosen_dx = jnp.take_along_axis(dx, tgt_idx[:, None], axis=1).squeeze(-1)
    chosen_dy = jnp.take_along_axis(dy, tgt_idx[:, None], axis=1).squeeze(-1)
    angle = jnp.arctan2(chosen_dy, chosen_dx)

    # Determine if src is firing as solo or pair on its chosen target.
    chosen_is_solo = (
        jnp.take_along_axis(
            solo_afford.astype(jnp.float32), tgt_idx[:, None], axis=1
        ).squeeze(-1)
        > 0.5
    )
    chosen_is_pair = (
        jnp.take_along_axis(
            pair_afford.astype(jnp.float32), tgt_idx[:, None], axis=1
        ).squeeze(-1)
        > 0.5
    )

    # Solo: send need (clipped to surplus). Pair: send half_t (clipped).
    chosen_half = jnp.take_along_axis(
        jnp.broadcast_to(half_t[None, :], (MAX_PLANETS, MAX_PLANETS)),
        tgt_idx[:, None],
        axis=1,
    ).squeeze(-1)
    ships_solo = jnp.clip(chosen_need, 1.0, surplus)
    ships_pair = jnp.clip(chosen_half, 1.0, surplus)
    ships_to_send = jnp.where(chosen_is_pair, ships_pair, ships_solo).astype(jnp.int32)
    # Only fire if solo or pair was the valid branch.
    row_has_valid = row_has_valid & (chosen_is_solo | chosen_is_pair)

    can_fire = is_mine & row_has_valid & (ships_to_send > 0)

    from_pid = jnp.where(can_fire, pid, jnp.int32(-1)).astype(jnp.float32)
    angle_col = jnp.where(can_fire, angle, jnp.float32(0.0))
    ships_col = jnp.where(can_fire, ships_to_send, jnp.int32(0)).astype(jnp.float32)

    actions = jnp.stack([from_pid, angle_col, ships_col], axis=-1)  # (P, 3)
    assert actions.shape == (MAX_LAUNCHES_PER_AGENT, 3)
    return actions


__all__ = ["compute_actions_jax"]
