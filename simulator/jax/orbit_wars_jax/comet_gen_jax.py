"""JAX-native comet generation (jit/vmap-friendly).

Mirrors the STRATEGY of `comet_gen.py` (ellipse param rejection, dense sampling +
arc-length resample, in-board visible-window extraction, sun + planet overlap
checks) but built for `jax.jit` + `jax.vmap`:

  * `random.Random` → `jax.random` (vendor byte-stream parity intentionally
    abandoned — see reset_jax/planet_gen_jax docstrings);
  * the 300-try Python rejection loop → `lax.while_loop` per comet, accepting the
    first ellipse whose resampled in-board window satisfies 5 <= len <= 40 and the
    sun / static-planet / time-evolved-orbiting-planet overlap checks;
  * dense sampling (vendor 5000 pts) + arc-length resample → fixed-size arrays;
  * the variable-length visible window → a fixed `MAX_COMET_PATH_LEN` buffer + a
    `path_len` count (the EnvState comet representation).

Key simplification (verified against comet_gen.py): the vendor grows a running
planet set across spawns, but the appended rows are off-board (y=x=-99) no-ops
AND excluded again via comet_pid_set, so each comet's overlap check is
effectively against the original planet buffer only. The 5 comets are therefore
INDEPENDENT and generated with the same planet buffer (no sequential carry).

Output: (paths, path_len, initial_ships) matching EnvState comet fields:
  paths[c, q, t, :] = (y, x) of visible point t for quadrant q (vendor layout).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .constants import (
    BOARD_SIZE,
    CENTER,
    COMET_RADIUS,
    COMET_SPAWN_STEPS,
    COMET_SPEED,
    MAX_COMETS,
    MAX_COMET_PATH_LEN,
    ROTATION_RADIUS_LIMIT,
    SUN_RADIUS,
)
from .planet_gen_jax import PlanetBuf

# Vendor draws t over [0.3π, 1.7π] with 5000 dense points, then resamples by
# arc length at spacing COMET_SPEED. We keep the dense count and resample into a
# fixed buffer of MAX_RESAMPLE points (>= any achievable in-board window).
_DENSE = 5000
_MAX_RESAMPLE = 256  # fixed resampled-path length ceiling (>> 40 visible cap)
_MAX_TRIES = 300
_T0 = 0.3 * jnp.pi
_T_SPAN = 1.4 * jnp.pi

# while_loop carry: (key, tries, found, vx, vy, vlen).
_Carry = tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]


def _ellipse_dense(
    e: jax.Array, a: jax.Array, phi: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Dense (x, y) sampling of the rotated ellipse arc. Returns x, y of len _DENSE."""
    b = a * jnp.sqrt(1.0 - e * e)
    c_val = a * e
    i = jnp.arange(_DENSE, dtype=jnp.float32)
    t = _T0 + _T_SPAN * i / (_DENSE - 1)
    ex = c_val + a * jnp.cos(t)
    ey = b * jnp.sin(t)
    x = CENTER + ex * jnp.cos(phi) - ey * jnp.sin(phi)
    y = CENTER + ex * jnp.sin(phi) + ey * jnp.cos(phi)
    return x, y


def _resample_by_arclen(
    x: jax.Array, y: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Arc-length resample at spacing COMET_SPEED into a fixed _MAX_RESAMPLE buffer.

    Vendor keeps dense[0], then each dense point where cumulative arc length first
    crosses k*speed. We replicate by: cum arc length over dense points, then for
    output slot j (j=0..MAX_RESAMPLE-1) pick the FIRST dense index whose cum >=
    j*speed (j=0 → dense[0]). Returns (rx, ry, rvalid) where rvalid[j] is True
    while j*speed <= total_arc (i.e. the resampled point actually exists).
    """
    dx = jnp.diff(x)
    dy = jnp.diff(y)
    seg = jnp.sqrt(dx * dx + dy * dy)
    cum = jnp.concatenate([jnp.zeros((1,), x.dtype), jnp.cumsum(seg)])  # (_DENSE,)
    total = cum[-1]

    targets = jnp.arange(_MAX_RESAMPLE, dtype=x.dtype) * COMET_SPEED  # (_MAX_RESAMPLE,)
    # first dense index with cum >= target (searchsorted left).
    idx = jnp.searchsorted(cum, targets, side="left")
    idx = jnp.clip(idx, 0, _DENSE - 1)
    rx = x[idx]
    ry = y[idx]
    rvalid = targets <= total
    return rx, ry, rvalid


def _inboard(x: jax.Array, y: jax.Array) -> jax.Array:
    return (x >= 0) & (x <= BOARD_SIZE) & (y >= 0) & (y <= BOARD_SIZE)


def _extract_visible(
    rx: jax.Array, ry: jax.Array, rvalid: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Contiguous in-board window [board_start, board_end] of the resampled path.

    Vendor: board_start = first in-board index, board_end = last in-board index,
    visible = path[board_start:board_end+1] (CONTIGUOUS slice, may include a few
    out-of-board points between the first and last in-board ones — matches vendor).
    Returns (vx[MAX_COMET_PATH_LEN], vy[MAX_COMET_PATH_LEN], vlen) where vlen is the
    window length clamped to MAX_COMET_PATH_LEN (vendor rejects len>40, so a window
    >40 just yields vlen>40 → caller rejects).
    """
    onboard = _inboard(rx, ry) & rvalid  # (_MAX_RESAMPLE,)
    pos = jnp.arange(_MAX_RESAMPLE, dtype=jnp.int32)
    any_on = jnp.any(onboard)
    start = jnp.min(jnp.where(onboard, pos, _MAX_RESAMPLE))
    end = jnp.max(jnp.where(onboard, pos, -1))
    raw_len = jnp.where(any_on, end - start + 1, 0)

    # gather window [start, start+MAX_COMET_PATH_LEN) (clamped); slots >= raw_len
    # are padding. (If raw_len > MAX_COMET_PATH_LEN the caller rejects on length.)
    offs = jnp.arange(MAX_COMET_PATH_LEN, dtype=jnp.int32)
    gather = jnp.clip(start + offs, 0, _MAX_RESAMPLE - 1)
    vx = rx[gather]
    vy = ry[gather]
    return vx, vy, raw_len


def _window_ok(
    vx: jax.Array,
    vy: jax.Array,
    vlen: jax.Array,
    pbuf: PlanetBuf,
    angular_velocity: jax.Array,
    spawn_step: jax.Array,
) -> jax.Array:
    """Vendor validity of a visible window: sun clearance + static/orbiting overlap.

    Checked only over the first `vlen` slots (mask). Mirrors comet_gen rows 109-148.
    """
    slot = jnp.arange(MAX_COMET_PATH_LEN, dtype=jnp.int32)
    active = (slot < vlen) & (vlen >= 5) & (vlen <= MAX_COMET_PATH_LEN)

    # vendor stores `visible` as cartesian (x, y); here vx/vy ARE (x, y).
    # sun clearance: dist((cx,cy),(CENTER,CENTER)) >= SUN+COMET_R.
    d_sun = jnp.sqrt((vx - CENTER) ** 2 + (vy - CENTER) ** 2)
    sun_bad = (d_sun < (SUN_RADIUS + COMET_RADIUS)) & active

    # symmetric points for each visible point (vendor rows 116-121):
    # (cy, cx), (BOARD-cx, cy), (cx, BOARD-cy), (BOARD-cy, BOARD-cx)
    sym_y = jnp.stack([vy, BOARD_SIZE - vx, vx, BOARD_SIZE - vy], axis=-1)  # (L,4)
    sym_x = jnp.stack([vx, vy, BOARD_SIZE - vy, BOARD_SIZE - vx], axis=-1)  # (L,4)

    py = pbuf.xy[:, 0]  # (MAX_PLANETS,) as (y)
    px = pbuf.xy[:, 1]
    pr = pbuf.radius
    pvalid = pbuf.valid
    p_orb = jnp.sqrt((py - CENTER) ** 2 + (px - CENTER) ** 2)
    p_rot = (p_orb + pr) < ROTATION_RADIUS_LIMIT
    is_static = pvalid & (~p_rot)
    is_orbiting = pvalid & p_rot
    buf_clear = COMET_RADIUS + 0.5

    # static overlap: any visible point's any sym point within planet[r]+buf.
    # shapes: sym (L,4), planets (P,) → (L,4,P)
    sdy = sym_y[:, :, None] - py[None, None, :]
    sdx = sym_x[:, :, None] - px[None, None, :]
    sdist = jnp.sqrt(sdy * sdy + sdx * sdx)
    static_hit = (sdist < (pr[None, None, :] + buf_clear)) & is_static[None, None, :]
    static_bad = jnp.any(static_hit, axis=(1, 2)) & active  # (L,)

    # orbiting overlap: planet position at game_step = spawn_step-1+k advances by
    # angular_velocity*game_step around CENTER. (vendor rows 132-146)
    k_idx = slot.astype(jnp.float32)  # per-visible index k
    game_step = (spawn_step - 1) + k_idx  # (L,)
    dxp = py - CENTER  # vendor uses planet[2]-CENTER, planet[3]-CENTER = (y-,x-)
    dyp = px - CENTER
    orb_r = jnp.sqrt(dxp * dxp + dyp * dyp)  # (P,)
    init_angle = jnp.arctan2(dyp, dxp)  # (P,)
    # cur_angle (L,P)
    cur_angle = init_angle[None, :] + angular_velocity * game_step[:, None]
    opx = CENTER + orb_r[None, :] * jnp.cos(cur_angle)  # (L,P)
    opy = CENTER + orb_r[None, :] * jnp.sin(cur_angle)
    # check each sym point of visible k vs orbiting planet at step k: (L,4,P)
    ody = sym_y[:, :, None] - opy[:, None, :]
    odx = sym_x[:, :, None] - opx[:, None, :]
    odist = jnp.sqrt(ody * ody + odx * odx)
    orb_hit = (odist < (pr[None, None, :] + COMET_RADIUS)) & is_orbiting[None, None, :]
    orb_bad = jnp.any(orb_hit, axis=(1, 2)) & active  # (L,)

    len_ok = (vlen >= 5) & (vlen <= MAX_COMET_PATH_LEN)
    return len_ok & (~jnp.any(sun_bad)) & (~jnp.any(static_bad)) & (~jnp.any(orb_bad))


def _gen_one_comet(
    key: jax.Array,
    pbuf: PlanetBuf,
    angular_velocity: jax.Array,
    spawn_step: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Generate ONE comet via <=300-try rejection. Returns (paths[4,L,2], plen, ships).

    plen=0 means generation failed (no valid ellipse in 300 tries).
    """
    init = (
        key,
        jnp.int32(0),  # tries
        jnp.bool_(False),  # found
        jnp.zeros((MAX_COMET_PATH_LEN,), jnp.float32),  # vx (cartesian x)
        jnp.zeros((MAX_COMET_PATH_LEN,), jnp.float32),  # vy (cartesian y)
        jnp.int32(0),  # vlen
    )

    def cond(c: _Carry) -> jax.Array:
        _key, tries, found, _vx, _vy, _vlen = c
        return (~found) & (tries < _MAX_TRIES)

    def body(c: _Carry) -> _Carry:
        key, tries, _found, vx_p, vy_p, vlen_p = c
        key, k_e, k_a, k_phi = jax.random.split(key, 4)
        e = jax.random.uniform(k_e, (), minval=0.75, maxval=0.93)
        a = jax.random.uniform(k_a, (), minval=60.0, maxval=150.0)
        phi = jax.random.uniform(k_phi, (), minval=jnp.pi / 6, maxval=jnp.pi / 3)
        perihelion = a * (1.0 - e)
        peri_ok = perihelion >= (SUN_RADIUS + COMET_RADIUS)

        x, y = _ellipse_dense(e, a, phi)
        rx, ry, rvalid = _resample_by_arclen(x, y)
        vx, vy, vlen = _extract_visible(rx, ry, rvalid)
        ok = peri_ok & _window_ok(vx, vy, vlen, pbuf, angular_velocity, spawn_step)

        new_found = ok
        vx_o = jnp.where(ok, vx, vx_p)
        vy_o = jnp.where(ok, vy, vy_p)
        vlen_o = jnp.where(ok, vlen, vlen_p)
        return (key, tries + 1, new_found, vx_o, vy_o, vlen_o)

    _key, _tries, found, vx, vy, vlen = jax.lax.while_loop(cond, body, init)
    plen = jnp.where(found, vlen, 0).astype(jnp.int32)

    # build the 4 quadrant copies, stored as (y, x) per vendor layout.
    # vendor: q0=[y,x], q1=[BOARD-x,y], q2=[x,BOARD-y], q3=[BOARD-y,BOARD-x]
    # where (x,y) = cartesian (vx, vy).
    q0 = jnp.stack([vy, vx], axis=-1)
    q1 = jnp.stack([BOARD_SIZE - vx, vy], axis=-1)
    q2 = jnp.stack([vx, BOARD_SIZE - vy], axis=-1)
    q3 = jnp.stack([BOARD_SIZE - vy, BOARD_SIZE - vx], axis=-1)
    paths = jnp.stack([q0, q1, q2, q3], axis=0)  # (4, MAX_COMET_PATH_LEN, 2)
    # zero out padding (slots >= plen) for cleanliness.
    slot = jnp.arange(MAX_COMET_PATH_LEN)
    mask = (slot < plen)[None, :, None]
    paths = jnp.where(mask, paths, 0.0).astype(jnp.float32)

    # ships = min of 4 randint(1,99) (vendor rows 195-200). Use a fresh key split.
    key, ks = jax.random.split(_key)
    ship_draws = jax.random.randint(ks, (4,), 1, 100)
    ships = jnp.where(found, jnp.min(ship_draws), 0).astype(jnp.int32)
    return paths, plen, ships


def generate_comets_jax(
    key: jax.Array, pbuf: PlanetBuf, angular_velocity: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Generate all MAX_COMETS comets independently (vmap over spawn slots).

    Returns (paths[MAX_COMETS,4,L,2], path_len[MAX_COMETS], initial_ships[MAX_COMETS]).
    """
    keys = jax.random.split(key, MAX_COMETS)
    spawns = jnp.asarray(COMET_SPAWN_STEPS, dtype=jnp.int32)

    def one(
        k: jax.Array, spawn: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        # spawn_step is traced here; _gen_one_comet uses it only in arithmetic
        # (game_step = spawn-1+k), which is fine traced.
        return _gen_one_comet(k, pbuf, angular_velocity, spawn)

    paths, plen, ships = jax.vmap(one, in_axes=(0, 0))(keys, spawns)
    return paths, plen, ships
