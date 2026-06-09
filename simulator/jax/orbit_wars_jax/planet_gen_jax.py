"""JAX-native planet generation (jit/vmap-friendly).

Mirrors the STRATEGY of `planet_gen.py` (4-fold symmetric groups, distance-
clearance rejection, rotating/non-rotating ring-collision checks, has_orbiting
termination) but is built for `jax.jit` + `jax.vmap`:

  * `random.Random` → `jax.random` (vendor byte-stream parity intentionally
    abandoned — the reinforce/case8 PFSP rollout only needs self-consistent
    valid layouts, not exact vendor placement);
  * the unbounded Python `for/while` rejection loops → two `lax.while_loop`s
    over a pre-allocated fixed-shape buffer (`MAX_PLANETS`), scattering accepted
    groups via `.at[idx].set(...)` and incrementing the count only on accept;
  * all-pairs distance / ring-collision checks vectorized over the buffer +
    valid mask (no Python inner loops).

Output: a `PlanetBuf` of fixed-shape arrays (length MAX_PLANETS) + a valid mask,
consumed by `reset_jax.py`. Slot layout matches `reset.py`'s planet arrays:
xy stored as (y, x).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .constants import (
    BOARD_SIZE,
    CENTER,
    MAX_PLANETS,
    MAX_PLANET_GROUPS,
    MIN_PLANET_GROUPS,
    MIN_STATIC_GROUPS,
    PLANET_CLEARANCE,
    ROTATION_RADIUS_LIMIT,
    SUN_RADIUS,
)

_HALF_PI = jnp.pi / 2.0
# Match the Python loop bounds (planet_gen.py): 5000 phase-1 tries, 5000 phase-2.
_PHASE1_MAX_TRIES = 5000
_PHASE2_MAX_TRIES = 5000


class PlanetBuf(NamedTuple):
    """Fixed-shape planet buffer (length MAX_PLANETS). xy stored as (y, x)."""

    xy: jax.Array  # float32[MAX_PLANETS, 2] — (y, x)
    radius: jax.Array  # float32[MAX_PLANETS]
    ships: jax.Array  # int32[MAX_PLANETS]
    prod: jax.Array  # int32[MAX_PLANETS]
    valid: jax.Array  # bool[MAX_PLANETS]
    n_valid: jax.Array  # int32 scalar — number of valid slots


def _empty_buf() -> PlanetBuf:
    return PlanetBuf(
        xy=jnp.zeros((MAX_PLANETS, 2), dtype=jnp.float32),
        radius=jnp.zeros(MAX_PLANETS, dtype=jnp.float32),
        ships=jnp.zeros(MAX_PLANETS, dtype=jnp.int32),
        prod=jnp.zeros(MAX_PLANETS, dtype=jnp.int32),
        valid=jnp.zeros(MAX_PLANETS, dtype=jnp.bool_),
        n_valid=jnp.int32(0),
    )


def _orbital(y: jax.Array, x: jax.Array) -> jax.Array:
    return jnp.sqrt((y - CENTER) ** 2 + (x - CENTER) ** 2)


def _quad_group(y: jax.Array, x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """The 4-fold symmetric (y, x) copies of a base point, as (4,) arrays.

    Vendor row layout temp_planets uses (y, x):
        [y, x], [BOARD-x, y], [x, BOARD-y], [BOARD-y, BOARD-x]
    """
    ys = jnp.stack([y, BOARD_SIZE - x, x, BOARD_SIZE - y])
    xs = jnp.stack([x, y, BOARD_SIZE - y, BOARD_SIZE - x])
    return ys, xs


def _group_fits(
    buf: PlanetBuf,
    gy: jax.Array,  # (4,)
    gx: jax.Array,  # (4,)
    gr: jax.Array,  # scalar radius (same for all 4)
    *,
    check_ring: bool,
) -> jax.Array:
    """True iff the 4-planet group can be placed without overlap.

    - distance-clearance vs every existing valid planet (vectorized);
    - vs the other 3 members of the same group;
    - if `check_ring`: also reject when a rotating member is too close (in
      orbital radius) to a non-rotating existing planet (phase-2 rule).
    """
    ex_y = buf.xy[:, 0]  # (MAX_PLANETS,)
    ex_x = buf.xy[:, 1]
    ex_r = buf.radius
    ex_valid = buf.valid
    ex_orb = _orbital(ex_y, ex_x)
    ex_rot = (ex_orb + ex_r) < ROTATION_RADIUS_LIMIT

    g_orb = _orbital(gy, gx)  # (4,)
    g_rot = (g_orb + gr) < ROTATION_RADIUS_LIMIT  # (4,)

    # (4, MAX_PLANETS) pairwise distances between group members and existing.
    dy = gy[:, None] - ex_y[None, :]
    dx = gx[:, None] - ex_x[None, :]
    dist = jnp.sqrt(dy * dy + dx * dx)
    clearance = gr + ex_r[None, :] + PLANET_CLEARANCE
    overlap = (dist < clearance) & ex_valid[None, :]

    if check_ring:
        # rotating-vs-static ring collision: when tp_is_rotating != p_is_rotating
        # and |orb_tp - orb_p| < r_tp + r_p + clearance.
        diff_rot = g_rot[:, None] != ex_rot[None, :]
        orb_close = jnp.abs(g_orb[:, None] - ex_orb[None, :]) < (
            gr + ex_r[None, :] + PLANET_CLEARANCE
        )
        ring_bad = diff_rot & orb_close & ex_valid[None, :]
        overlap = overlap | ring_bad

    overlap_existing = jnp.any(overlap)

    # intra-group: members vs each other (i<j). (4,4) distance matrix.
    iy = gy[:, None] - gy[None, :]
    ix = gx[:, None] - gx[None, :]
    idist = jnp.sqrt(iy * iy + ix * ix)
    iu = jnp.triu(jnp.ones((4, 4), dtype=jnp.bool_), k=1)
    intra_bad = jnp.any((idist < (2.0 * gr + PLANET_CLEARANCE)) & iu)

    return ~(overlap_existing | intra_bad)


def _scatter_group(
    buf: PlanetBuf,
    gy: jax.Array,
    gx: jax.Array,
    gr: jax.Array,
    ships: jax.Array,
    prod: jax.Array,
) -> PlanetBuf:
    """Write the 4 group members at slots [count, count+1, count+2, count+3].

    Assumes count + 4 <= MAX_PLANETS (the loop guard ensures this).
    """
    base = buf.n_valid
    idx = base + jnp.arange(4)
    xy = buf.xy.at[idx, 0].set(gy).at[idx, 1].set(gx)
    radius = buf.radius.at[idx].set(gr)
    ships_a = buf.ships.at[idx].set(ships.astype(jnp.int32))
    prod_a = buf.prod.at[idx].set(prod.astype(jnp.int32))
    valid = buf.valid.at[idx].set(True)
    return buf._replace(
        xy=xy,
        radius=radius,
        ships=ships_a,
        prod=prod_a,
        valid=valid,
        n_valid=base + 4,
    )


class _P1Carry(NamedTuple):
    key: jax.Array
    buf: PlanetBuf
    groups: jax.Array  # int32 — static groups placed so far
    tries: jax.Array  # int32


def _phase1(key: jax.Array) -> tuple[jax.Array, PlanetBuf]:
    """Place MIN_STATIC_GROUPS guaranteed static (non-rotating) groups."""

    def cond(c: _P1Carry) -> jax.Array:
        return (c.groups < MIN_STATIC_GROUPS) & (c.tries < _PHASE1_MAX_TRIES)

    def body(c: _P1Carry) -> _P1Carry:
        key, k_prod, k_ang, k_orb, k_s1, k_s2 = jax.random.split(c.key, 6)
        prod = jax.random.randint(k_prod, (), 1, 6)  # 1..5
        r = 1.0 + jnp.log(prod.astype(jnp.float32))
        angle = jax.random.uniform(k_ang, (), minval=0.0, maxval=_HALF_PI)
        min_orbital = ROTATION_RADIUS_LIMIT - r
        denom = jnp.maximum(jnp.cos(angle), jnp.sin(angle))
        max_orbital = (BOARD_SIZE - CENTER - r) / denom
        # rng draw happens regardless (uniform over [min,max]); if min>max the
        # vendor `continue`s — we mimic by treating the group as not-fitting.
        orbital_r = jax.random.uniform(
            k_orb, (), minval=min_orbital, maxval=jnp.maximum(max_orbital, min_orbital)
        )
        x = CENTER + orbital_r * jnp.cos(angle)
        y = CENTER + orbital_r * jnp.sin(angle)

        in_band = min_orbital <= max_orbital
        # bounds rejections (vendor rows 55-60):
        bounds_ok = (
            (x + r <= BOARD_SIZE)
            & (x - r >= 0)
            & (y + r <= BOARD_SIZE)
            & (y - r >= 0)
            & ((BOARD_SIZE - x) - r >= 0)
            & ((BOARD_SIZE - y) - r >= 0)
            & ((x - CENTER) >= r + 5)
            & ((y - CENTER) >= r + 5)
        )

        ships = jnp.minimum(
            jax.random.randint(k_s1, (), 5, 100), jax.random.randint(k_s2, (), 5, 100)
        )
        gy, gx = _quad_group(y, x)
        fits = _group_fits(c.buf, gy, gx, r, check_ring=False)
        room = (c.buf.n_valid + 4) <= MAX_PLANETS
        accept = in_band & bounds_ok & fits & room

        new_buf = jax.lax.cond(
            accept,
            lambda: _scatter_group(c.buf, gy, gx, r, ships, prod),
            lambda: c.buf,
        )
        return _P1Carry(
            key=key,
            buf=new_buf,
            groups=c.groups + accept.astype(jnp.int32),
            tries=c.tries + 1,
        )

    init = _P1Carry(key=key, buf=_empty_buf(), groups=jnp.int32(0), tries=jnp.int32(0))
    out = jax.lax.while_loop(cond, body, init)
    return out.key, out.buf


class _P2Carry(NamedTuple):
    key: jax.Array
    buf: PlanetBuf
    has_orbiting: jax.Array  # bool
    tries: jax.Array  # int32
    target: jax.Array  # int32 — num_q1 * 4


def _phase2(
    key: jax.Array, buf: PlanetBuf, target: jax.Array
) -> tuple[jax.Array, PlanetBuf]:
    """Fill remaining groups with the normal random loop until target planets +
    at least one orbiting planet exist (vendor phase 2)."""

    def cond(c: _P2Carry) -> jax.Array:
        need_more = c.buf.n_valid < c.target
        need_orbit = (~c.has_orbiting) & (c.tries < _PHASE2_MAX_TRIES)
        return (need_more | need_orbit) & (c.tries < _PHASE2_MAX_TRIES)

    def body(c: _P2Carry) -> _P2Carry:
        key, k_prod, k_x, k_y, k_s = jax.random.split(c.key, 5)
        prod = jax.random.randint(k_prod, (), 1, 6)
        r = 1.0 + jnp.log(prod.astype(jnp.float32))
        x = jax.random.uniform(k_x, (), minval=CENTER + 15, maxval=BOARD_SIZE - r - 5)
        y = jax.random.uniform(k_y, (), minval=CENTER + 15, maxval=BOARD_SIZE - r - 5)
        ships = jax.random.randint(k_s, (), 5, 31)  # 5..30

        orbital_radius = _orbital(y, x)
        # vendor: continue if orbital_radius < SUN_RADIUS + r + 10
        sun_ok = orbital_radius >= (SUN_RADIUS + r + 10)
        # if orbital + r >= ROTATION_LIMIT: must satisfy board bounds.
        is_far = (orbital_radius + r) >= ROTATION_RADIUS_LIMIT
        bounds_ok = (
            (x + r <= BOARD_SIZE) & (x - r >= 0) & (y + r <= BOARD_SIZE) & (y - r >= 0)
        )
        far_ok = jnp.where(is_far, bounds_ok, True)

        gy, gx = _quad_group(y, x)
        fits = _group_fits(c.buf, gy, gx, r, check_ring=True)
        room = (c.buf.n_valid + 4) <= MAX_PLANETS
        accept = sun_ok & far_ok & fits & room

        # has_orbiting flips when an accepted group is rotating (orb + r < limit).
        this_orbiting = (orbital_radius + r) < ROTATION_RADIUS_LIMIT
        new_buf = jax.lax.cond(
            accept,
            lambda: _scatter_group(c.buf, gy, gx, r, ships, prod),
            lambda: c.buf,
        )
        new_orbit = c.has_orbiting | (accept & this_orbiting)
        return _P2Carry(
            key=key,
            buf=new_buf,
            has_orbiting=new_orbit,
            tries=c.tries + 1,
            target=c.target,
        )

    init = _P2Carry(
        key=key,
        buf=buf,
        has_orbiting=jnp.bool_(False),
        tries=jnp.int32(0),
        target=target,
    )
    out = jax.lax.while_loop(cond, body, init)
    return out.key, out.buf


def generate_planets_jax(key: jax.Array) -> PlanetBuf:
    """JAX-native `generate_planets`. Returns a fixed-shape PlanetBuf.

    num_q1 (group target) ~ randint(MIN_PLANET_GROUPS, MAX_PLANET_GROUPS); phase 1
    places MIN_STATIC_GROUPS static groups, phase 2 fills to num_q1*4 with at
    least one orbiting planet.
    """
    key, k_groups = jax.random.split(key)
    num_q1 = jax.random.randint(k_groups, (), MIN_PLANET_GROUPS, MAX_PLANET_GROUPS + 1)
    target = num_q1 * 4
    key, buf = _phase1(key)
    _, buf = _phase2(key, buf, target)
    return buf
