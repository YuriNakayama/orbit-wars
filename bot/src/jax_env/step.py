"""JAX-pure step function and sub-helpers.

Top-level `step` is jit-able. Each helper takes / returns an EnvState (or
small ancillary arrays). Mirrors the vendor `interpreter` (rows 397-697)
ordering: expire → activate → launch → produce → move → combat → terminate.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .combat import resolve_combat
from .constants import (
    BOARD_SIZE,
    CENTER,
    COMET_PRODUCTION,
    COMET_RADIUS,
    EPISODE_STEPS,
    MAX_COMETS,
    MAX_FLEETS,
    MAX_PLANETS,
    NUM_AGENTS_MAX,
    SHIP_SPEED_MAX,
    SUN_RADIUS,
)
from .geometry import point_to_segment_distance, swept_pair_hit
from .state import EnvState

# At most one launch per owned planet per agent per turn.
MAX_LAUNCHES_PER_AGENT: int = MAX_PLANETS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _expire_comets(state: EnvState) -> EnvState:
    """Drop comet planets whose path index passed the end. Vendor rows 401-420.

    Note: vendor expires *before* fleet launch using the previous step's
    path_index; in our model `comet_path_index` is already the previous value
    (gets incremented during _move_fleets_and_detect_hits).
    """
    # next index after step's planet-move would be path_index + 1.
    # vendor uses: idx >= len(path) → expired. We mirror that using the *next*
    # index value comparison so expired entries are pruned at the start.
    # Since the increment happens during planet movement, here we check
    # `path_index >= path_len - 1` to mark slots that already produced the
    # final visible position last step and are now past.
    will_expire = (
        state.comet_active
        & (state.comet_path_index >= (state.comet_path_len - 1))
        & (state.comet_path_index >= 0)
    )

    # For each comet that will expire, invalidate its 4 planet slots.
    # comet_planet_slot: (MAX_COMETS, 4). Build a (MAX_COMETS, 4, MAX_PLANETS)
    # one-hot then OR over comets/quadrants.
    slot_idx = state.comet_planet_slot  # (C, 4) int32, -1 for unused
    valid_slot = (slot_idx >= 0) & will_expire[:, None]  # (C, 4) bool

    planet_idx = jnp.arange(MAX_PLANETS, dtype=jnp.int32)
    drop_mask = jnp.any(
        valid_slot[:, :, None] & (slot_idx[:, :, None] == planet_idx[None, None, :]),
        axis=(0, 1),
    )  # (P,)

    new_planet_valid = state.planet_valid & ~drop_mask
    # Reset slot bookkeeping for expired comets.
    new_comet_active = jnp.where(will_expire, False, state.comet_active)
    new_comet_planet_slot = jnp.where(
        will_expire[:, None],
        jnp.full_like(slot_idx, -1),
        slot_idx,
    )
    # Also clear is_comet flag on dropped planets.
    new_planet_is_comet = state.planet_is_comet & ~drop_mask

    return state._replace(
        planet_valid=new_planet_valid,
        planet_is_comet=new_planet_is_comet,
        comet_active=new_comet_active,
        comet_planet_slot=new_comet_planet_slot,
    )


def _activate_comets(state: EnvState) -> EnvState:
    """Spawn the comet whose spawn_step matches state.step + 1. Vendor rows 422-462."""
    # Identify the unique comet (at most one per step) to activate.
    want = (
        (state.comet_spawn_step == (state.step + 1))
        & (state.comet_path_len > 0)
        & (~state.comet_active)
    )  # (C,)

    # Activation index: argmax gives first True; valid only if want.any().
    activate_idx = jnp.argmax(want.astype(jnp.int32))
    any_activate = jnp.any(want)

    # Find first 4 free planet slots.
    free = ~state.planet_valid  # (P,)
    # Use `argsort` on negated to push True (free) to the front, then take first 4.
    # Stable sort preserves slot order.
    order = jnp.argsort(~free, stable=True)  # (P,) — free slots first
    chosen = order[:4]  # (4,)
    enough_free = jnp.sum(free) >= 4
    do_activate = any_activate & enough_free

    # Build the 4 new planet rows.
    comet_ships = state.comet_initial_ships[activate_idx]
    new_id_base = state.next_planet_id

    # Update planet arrays at slots `chosen` (when do_activate).
    def _scatter_planet(arr: jax.Array, values: jax.Array) -> jax.Array:
        """Scatter `values[k]` into arr[chosen[k]] for k in 0..3 if do_activate."""
        if arr.ndim == 1:
            updated = arr.at[chosen].set(jnp.where(do_activate, values, arr[chosen]))
        else:
            updated = arr.at[chosen].set(jnp.where(do_activate, values, arr[chosen]))
        return updated

    new_pids = new_id_base + jnp.arange(4, dtype=jnp.int32)
    new_planet_id = _scatter_planet(state.planet_id, new_pids)
    new_planet_owner = _scatter_planet(
        state.planet_owner, jnp.full((4,), -1, dtype=jnp.int32)
    )
    new_planet_xy = state.planet_xy.at[chosen].set(
        jnp.where(
            do_activate,
            jnp.full((4, 2), -99.0, dtype=jnp.float32),
            state.planet_xy[chosen],
        )
    )
    new_planet_radius = _scatter_planet(
        state.planet_radius, jnp.full((4,), COMET_RADIUS, dtype=jnp.float32)
    )
    new_planet_ships = _scatter_planet(
        state.planet_ships, jnp.full((4,), comet_ships, dtype=jnp.int32)
    )
    new_planet_prod = _scatter_planet(
        state.planet_prod, jnp.full((4,), COMET_PRODUCTION, dtype=jnp.int32)
    )
    new_planet_initial_xy = state.planet_initial_xy.at[chosen].set(
        jnp.where(
            do_activate,
            jnp.full((4, 2), -99.0, dtype=jnp.float32),
            state.planet_initial_xy[chosen],
        )
    )
    new_planet_valid = state.planet_valid.at[chosen].set(
        jnp.where(do_activate, jnp.full((4,), True), state.planet_valid[chosen])
    )
    new_planet_is_comet = state.planet_is_comet.at[chosen].set(
        jnp.where(do_activate, jnp.full((4,), True), state.planet_is_comet[chosen])
    )
    new_planet_is_rotating = state.planet_is_rotating.at[chosen].set(
        jnp.where(do_activate, jnp.full((4,), False), state.planet_is_rotating[chosen])
    )

    # Update comet bookkeeping.
    new_comet_active = state.comet_active.at[activate_idx].set(
        jnp.where(do_activate, True, state.comet_active[activate_idx])
    )
    new_comet_path_index = state.comet_path_index.at[activate_idx].set(
        jnp.where(do_activate, -1, state.comet_path_index[activate_idx])
    )
    new_comet_planet_slot = state.comet_planet_slot.at[activate_idx].set(
        jnp.where(
            do_activate,
            chosen.astype(jnp.int32),
            state.comet_planet_slot[activate_idx],
        )
    )

    new_next_planet_id = state.next_planet_id + jnp.where(do_activate, 4, 0)

    return state._replace(
        planet_id=new_planet_id,
        planet_owner=new_planet_owner,
        planet_xy=new_planet_xy,
        planet_radius=new_planet_radius,
        planet_ships=new_planet_ships,
        planet_prod=new_planet_prod,
        planet_initial_xy=new_planet_initial_xy,
        planet_valid=new_planet_valid,
        planet_is_comet=new_planet_is_comet,
        planet_is_rotating=new_planet_is_rotating,
        comet_active=new_comet_active,
        comet_path_index=new_comet_path_index,
        comet_planet_slot=new_comet_planet_slot,
        next_planet_id=new_next_planet_id,
    )


def _launch_fleets(state: EnvState, actions: jax.Array) -> EnvState:
    """Apply per-agent launch actions.

    actions: int32-ish[NUM_AGENTS_MAX, MAX_LAUNCHES_PER_AGENT, 3] with fields
        (from_planet_id, angle_q, ships). To stay int-typed, `angle` is passed
        as float32; we accept ``actions`` as a struct of two arrays in practice
        but here use a single float32 tensor: positions 0 and 2 are integral
        but stored as float32. Sentinel: from_planet_id == -1 means no-op.
    """
    # Flatten (A, L, 3) -> (A*L, 3) and scan in deterministic order.
    A = NUM_AGENTS_MAX
    L = MAX_LAUNCHES_PER_AGENT
    flat = actions.reshape(A * L, 3)
    agent_ids = jnp.repeat(jnp.arange(A, dtype=jnp.int32), L)  # (A*L,)

    def body(
        carry: tuple[EnvState, jax.Array], inputs: tuple[jax.Array, jax.Array]
    ) -> tuple[tuple[EnvState, jax.Array], None]:
        st, fleet_cursor = carry
        action, agent_id = inputs
        from_pid = action[0].astype(jnp.int32)
        angle = action[1].astype(jnp.float32)
        ships_req = action[2].astype(jnp.int32)

        # Find planet slot matching from_pid (and valid).
        match = (st.planet_id == from_pid) & st.planet_valid  # (P,)
        any_match = jnp.any(match)
        slot = jnp.argmax(match.astype(jnp.int32))  # 0 if no match
        slot = jnp.where(any_match, slot, jnp.int32(-1))

        owner_ok = jnp.where(slot >= 0, st.planet_owner[slot] == agent_id, False)
        ships_avail = jnp.where(slot >= 0, st.planet_ships[slot], jnp.int32(0))
        ships_ok = (ships_req > 0) & (ships_avail >= ships_req)
        fleet_slot_ok = fleet_cursor < MAX_FLEETS
        sentinel_ok = from_pid >= 0
        do_launch = sentinel_ok & any_match & owner_ok & ships_ok & fleet_slot_ok

        # Compute start position.
        planet_y = jnp.where(slot >= 0, st.planet_xy[slot, 0], jnp.float32(0.0))
        planet_x = jnp.where(slot >= 0, st.planet_xy[slot, 1], jnp.float32(0.0))
        radius = jnp.where(slot >= 0, st.planet_radius[slot], jnp.float32(0.0))
        # Vendor row 481-482:
        #   start_x = from_planet[2] + cos(angle) * (r + 0.1)   ← planet[2] is y
        #   start_y = from_planet[3] + sin(angle) * (r + 0.1)   ← planet[3] is x
        # So vendor's "fleet[2]" = start_x is derived from planet's y. We
        # preserve this quirk exactly. In our state.fleet_xy we store the same
        # ordering as vendor's fleet[2], fleet[3]: (start_x, start_y).
        offset = radius + jnp.float32(0.1)
        start_x = planet_y + jnp.cos(angle) * offset
        start_y = planet_x + jnp.sin(angle) * offset

        # Decrement planet ships.
        # When do_launch is False, leave ships unchanged.
        slot_safe = jnp.maximum(slot, 0)
        new_planet_ships = st.planet_ships.at[slot_safe].set(
            jnp.where(do_launch, ships_avail - ships_req, st.planet_ships[slot_safe])
        )

        # Append fleet at fleet_cursor (if do_launch).
        cur = jnp.minimum(fleet_cursor, jnp.int32(MAX_FLEETS - 1))
        new_fleet_owner = st.fleet_owner.at[cur].set(
            jnp.where(do_launch, agent_id, st.fleet_owner[cur])
        )
        new_fleet_xy = st.fleet_xy.at[cur].set(
            jnp.where(
                do_launch,
                jnp.stack([start_x, start_y]),
                st.fleet_xy[cur],
            )
        )
        new_fleet_angle = st.fleet_angle.at[cur].set(
            jnp.where(do_launch, angle, st.fleet_angle[cur])
        )
        new_fleet_ships = st.fleet_ships.at[cur].set(
            jnp.where(do_launch, ships_req, st.fleet_ships[cur])
        )
        new_fleet_from_pid = st.fleet_from_pid.at[cur].set(
            jnp.where(do_launch, from_pid, st.fleet_from_pid[cur])
        )
        new_fleet_valid = st.fleet_valid.at[cur].set(
            jnp.where(do_launch, True, st.fleet_valid[cur])
        )

        new_next_fleet_id = st.next_fleet_id + jnp.where(do_launch, 1, 0)

        new_state = st._replace(
            planet_ships=new_planet_ships,
            fleet_owner=new_fleet_owner,
            fleet_xy=new_fleet_xy,
            fleet_angle=new_fleet_angle,
            fleet_ships=new_fleet_ships,
            fleet_from_pid=new_fleet_from_pid,
            fleet_valid=new_fleet_valid,
            next_fleet_id=new_next_fleet_id,
        )
        new_cursor = fleet_cursor + jnp.where(do_launch, 1, 0)
        return (new_state, new_cursor), None

    # Find next free fleet slot to start from. Use argmin on ~valid.
    free = ~state.fleet_valid
    init_cursor = jnp.argmax(free.astype(jnp.int32))
    # If no free slot, all valid → cursor=MAX_FLEETS (overflow).
    any_free = jnp.any(free)
    init_cursor = jnp.where(any_free, init_cursor, jnp.int32(MAX_FLEETS))

    (final_state, _), _ = jax.lax.scan(body, (state, init_cursor), (flat, agent_ids))
    return final_state


def _produce_ships(state: EnvState) -> EnvState:
    """Owned planets get +production. Vendor rows 499-502."""
    bonus = jnp.where(
        state.planet_valid & (state.planet_owner != -1),
        state.planet_prod,
        jnp.int32(0),
    )
    return state._replace(planet_ships=state.planet_ships + bonus)


def _compute_planet_new_xy(state: EnvState) -> jax.Array:
    """Per-planet end-of-tick (y, x). Vendor rows 504-548.

    - Static planets: stay put.
    - Rotating planets: rotate around CENTER from initial_xy.
    - Comet planets: jump to comet_paths[c, q, path_index + 1].
    """
    # Rotating planets.
    init_y = state.planet_initial_xy[:, 0]
    init_x = state.planet_initial_xy[:, 1]
    dx = init_y - CENTER  # vendor: dx = initial_p[2] - CENTER → that's the y-coord
    dy = init_x - CENTER  # vendor: dy = initial_p[3] - CENTER → that's the x-coord
    r = jnp.sqrt(dx * dx + dy * dy)
    init_angle = jnp.arctan2(dy, dx)
    # Vendor uses obs0.step (= current step number entering this tick) for
    # the rotation argument; see vendor orbit_wars.py:508. state.step is
    # incremented *after* this function runs, so state.step here equals
    # obs0.step on the vendor side.
    current_angle = init_angle + state.angular_velocity * state.step.astype(
        jnp.float32
    )
    rot_y = CENTER + r * jnp.cos(current_angle)
    rot_x = CENTER + r * jnp.sin(current_angle)
    rot_xy = jnp.stack([rot_y, rot_x], axis=-1)  # (P, 2)

    # Comet planets — advance by 1 along their path.
    # Map planet slot -> (c, q) via comet_planet_slot. Build a lookup of next xy.
    planet_idx = jnp.arange(MAX_PLANETS, dtype=jnp.int32)

    # For each (c, q): planet slot = comet_planet_slot[c, q]; new position =
    # comet_paths[c, q, comet_path_index[c] + 1].
    next_idx = jnp.clip(
        state.comet_path_index + 1, 0, state.comet_paths.shape[2] - 1
    )  # (C,)
    # gather comet_paths[c, q, next_idx[c]]
    c_idx = jnp.arange(MAX_COMETS)
    # shape (C, 4, 2)
    comet_next_xy = state.comet_paths[
        c_idx[:, None], jnp.arange(4)[None, :], next_idx[:, None]
    ]
    # comet_paths stores per-quadrant 4-fold rotational symmetries per
    # vendor/Rust convention (comet_gen.py:91-95):
    #   q=0: [y, x]                  ← needs swap to (x, y)
    #   q=1: [BOARD_SIZE - x, y]     ← already in row order
    #   q=2: [x, BOARD_SIZE - y]     ← already in row order
    #   q=3: [BOARD_SIZE - y, BS-x]  ← needs swap
    # Rust emits these unchanged into obs row cols (2, 3). For JAX planet_xy
    # to match Rust's row layout, swap only q=0 and q=3. (BC weights expect
    # Rust's layout because all training data was generated by the Rust sim.)
    swap_q = jnp.array([True, False, False, True])
    swapped = comet_next_xy[..., ::-1]
    comet_next_xy = jnp.where(swap_q[None, :, None], swapped, comet_next_xy)
    # Flatten (C, 4) for scatter.
    flat_slots = state.comet_planet_slot.reshape(-1)  # (C*4,)
    flat_xy = comet_next_xy.reshape(-1, 2)  # (C*4, 2)
    # We need to map slot index -> new xy. For each planet, find if any (c,q)
    # references it. The mapping is at most 1:1.
    # Build (C*4, P) mask = flat_slots[k] == p AND flat_slots[k] >= 0
    valid_assign = flat_slots >= 0
    # For each planet slot, find the matching (c,q) row (or none).
    # eq_mask shape (P, C*4)
    eq_mask = (planet_idx[:, None] == flat_slots[None, :]) & valid_assign[None, :]
    has_comet = jnp.any(eq_mask, axis=-1)  # (P,)
    # Pick first matching row.
    pick_idx = jnp.argmax(eq_mask.astype(jnp.int32), axis=-1)  # (P,)
    comet_xy = flat_xy[pick_idx]  # (P, 2)

    new_xy = jnp.where(
        has_comet[:, None],
        comet_xy,
        jnp.where(state.planet_is_rotating[:, None], rot_xy, state.planet_xy),
    )
    # Invalid slots keep their current xy (doesn't matter).
    new_xy = jnp.where(state.planet_valid[:, None], new_xy, state.planet_xy)
    return new_xy


def _move_fleets_and_detect_hits(
    state: EnvState, new_planet_xy: jax.Array
) -> tuple[EnvState, jax.Array, jax.Array]:
    """Move all fleets, detect hits / OOB / sun-cross. Vendor rows 550-590.

    Also bumps comet `path_index` (which is "applied" during planet movement).
    """
    # Fleet speed (vendor row 559).
    ships = state.fleet_ships.astype(jnp.float32)
    safe_ships = jnp.maximum(ships, 1.0)  # avoid log(0)
    speed = (
        1.0 + (SHIP_SPEED_MAX - 1.0) * (jnp.log(safe_ships) / jnp.log(1000.0)) ** 1.5
    )
    speed = jnp.minimum(speed, SHIP_SPEED_MAX)

    old_xy = state.fleet_xy
    angle = state.fleet_angle
    # Vendor: fleet[2] += cos(angle) * speed; fleet[3] += sin(angle) * speed.
    # fleet[2] / [3] correspond to our fleet_xy[:, 0] / [:, 1].
    new_x = old_xy[:, 0] + jnp.cos(angle) * speed
    new_y = old_xy[:, 1] + jnp.sin(angle) * speed
    new_xy = jnp.stack([new_x, new_y], axis=-1)

    # Per-planet swept-pair hit: shape (F, P).
    # Planet old/new xy: state.planet_xy / new_planet_xy. (P, 2)
    p_old = state.planet_xy  # (P, 2)
    p_new = new_planet_xy  # (P, 2)
    radius = state.planet_radius  # (P,)
    # Comet planets with old_pos[0] < 0 must NOT trigger hits (vendor row 547).
    p_check = state.planet_is_comet & (p_old[:, 0] < 0)
    skip_planet = p_check & state.planet_is_comet  # planet hit check disabled
    # vendor sets planet_paths[pid] = (old, old, True) when old_pos was off-board.
    # Actually vendor row 547: `check = old_pos[0] >= 0`. So when old_pos was
    # off-board (e.g. comet just activated), skip hit-check entirely.
    skip_planet = state.planet_is_comet & (p_old[:, 0] < 0)

    hit = jax.vmap(
        lambda fa, fb: jax.vmap(
            lambda pa, pb, r: swept_pair_hit(fa, fb, pa, pb, r),
            in_axes=(0, 0, 0),
        )(p_old, p_new, radius)
    )(old_xy, new_xy)  # (F, P)

    # Mask out invalid planets / fleets / skipped planets.
    valid_mask = (
        state.fleet_valid[:, None]
        & state.planet_valid[None, :]
        & (~skip_planet[None, :])
    )
    hit = hit & valid_mask

    # First planet hit per fleet (lowest planet index).
    any_hit = jnp.any(hit, axis=-1)  # (F,)
    hit_slot = jnp.argmax(hit.astype(jnp.int32), axis=-1)  # (F,) → first True idx
    hit_slot = jnp.where(any_hit, hit_slot, jnp.int32(-1))

    # OOB & sun checks (only for fleets that did NOT hit a planet).
    oob = (new_x < 0) | (new_x > BOARD_SIZE) | (new_y < 0) | (new_y > BOARD_SIZE)
    sun_cross = (
        jax.vmap(
            lambda a, b: point_to_segment_distance(jnp.array([CENTER, CENTER]), a, b)
        )(old_xy, new_xy)
        < SUN_RADIUS
    )
    remove_oob = state.fleet_valid & ~any_hit & oob
    remove_sun = state.fleet_valid & ~any_hit & ~oob & sun_cross

    # Combined hit_mask used by resolve_combat (will also clear valid flag).
    hit_planet_mask = state.fleet_valid & any_hit
    fleet_remove_mask = hit_planet_mask | remove_oob | remove_sun

    # Update fleet xy (only for fleets that survived AND are still valid).
    new_fleet_xy = jnp.where(
        state.fleet_valid[:, None] & ~fleet_remove_mask[:, None],
        new_xy,
        state.fleet_xy,
    )

    # Bump comet path_index for any active comet.
    new_comet_path_index = jnp.where(
        state.comet_active,
        state.comet_path_index + 1,
        state.comet_path_index,
    )

    new_state = state._replace(
        fleet_xy=new_fleet_xy,
        comet_path_index=new_comet_path_index,
    )
    return new_state, hit_slot, fleet_remove_mask


def _check_termination(state: EnvState) -> EnvState:
    """Step-based + alive-count termination + reward. Vendor rows 665-697."""
    step_done = state.step >= (EPISODE_STEPS - 2)

    agent_idx = jnp.arange(NUM_AGENTS_MAX, dtype=jnp.int32)
    has_planet = jnp.any(
        state.planet_valid[None, :]
        & (state.planet_owner[None, :] == agent_idx[:, None]),
        axis=-1,
    )  # (A,)
    has_fleet = jnp.any(
        state.fleet_valid[None, :] & (state.fleet_owner[None, :] == agent_idx[:, None]),
        axis=-1,
    )  # (A,)
    alive = has_planet | has_fleet
    alive_count = jnp.sum(alive.astype(jnp.int32))

    terminated = step_done | (alive_count <= 1)

    # Score: per-agent sum of planet_ships (if owned) + fleet_ships.
    planet_score = jnp.sum(
        jnp.where(
            state.planet_valid[None, :]
            & (state.planet_owner[None, :] == agent_idx[:, None]),
            state.planet_ships[None, :],
            0,
        ),
        axis=-1,
    )
    fleet_score = jnp.sum(
        jnp.where(
            state.fleet_valid[None, :]
            & (state.fleet_owner[None, :] == agent_idx[:, None]),
            state.fleet_ships[None, :],
            0,
        ),
        axis=-1,
    )
    scores = (planet_score + fleet_score).astype(jnp.float32)
    max_score = jnp.max(scores)
    is_winner = (scores == max_score) & (max_score > 0)

    # Only agents [0, num_agents) get rewards; the rest stay 0.
    is_agent = agent_idx < state.num_agents
    rewards_when_done = jnp.where(
        is_agent,
        jnp.where(is_winner, jnp.float32(1.0), jnp.float32(-1.0)),
        jnp.float32(0.0),
    )
    rewards = jnp.where(terminated, rewards_when_done, state.rewards)

    return state._replace(terminated=terminated, rewards=rewards)


# ---------------------------------------------------------------------------
# Top-level step
# ---------------------------------------------------------------------------
@jax.jit
def step(state: EnvState, actions: jax.Array) -> tuple[EnvState, jax.Array, jax.Array]:
    """Advance one tick. Returns (new_state, rewards, terminated).

    actions: float32[NUM_AGENTS_MAX, MAX_LAUNCHES_PER_AGENT, 3] with columns
        (from_planet_id, angle, ships). Sentinel: from_planet_id == -1.
    """
    state = _expire_comets(state)
    state = _activate_comets(state)
    state = _launch_fleets(state, actions)
    state = _produce_ships(state)
    new_planet_xy = _compute_planet_new_xy(state)
    state, hit_slot, hit_mask = _move_fleets_and_detect_hits(state, new_planet_xy)
    state = state._replace(planet_xy=new_planet_xy)
    state = resolve_combat(state, hit_slot, hit_mask)
    state = state._replace(step=state.step + 1)
    state = _check_termination(state)
    return state, state.rewards, state.terminated


__all__ = [
    "step",
    "MAX_LAUNCHES_PER_AGENT",
    "_expire_comets",
    "_activate_comets",
    "_launch_fleets",
    "_produce_ships",
    "_compute_planet_new_xy",
    "_move_fleets_and_detect_hits",
    "_check_termination",
]


def empty_actions() -> jax.Array:
    """Helper: build a no-op action tensor with the right shape and sentinel."""
    actions = jnp.full(
        (NUM_AGENTS_MAX, MAX_LAUNCHES_PER_AGENT, 3),
        -1.0,
        dtype=jnp.float32,
    )
    return actions
