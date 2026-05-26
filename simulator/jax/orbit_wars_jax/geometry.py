"""Pure JAX geometry helpers. Vendor parity is unit-tested.

All functions are jit/vmap friendly. No control flow on data; conditions are
expressed with `jnp.where`.
"""

from __future__ import annotations

import jax.numpy as jnp


def distance_squared(p1: jnp.ndarray, p2: jnp.ndarray) -> jnp.ndarray:
    """Squared Euclidean distance. Shape: (..., 2) -> (...,)."""
    diff = p1 - p2
    return jnp.sum(diff * diff, axis=-1)


def distance(p1: jnp.ndarray, p2: jnp.ndarray) -> jnp.ndarray:
    """Euclidean distance, matching vendor `distance`."""
    return jnp.sqrt(distance_squared(p1, p2))


def point_to_segment_distance(
    p: jnp.ndarray, v: jnp.ndarray, w: jnp.ndarray
) -> jnp.ndarray:
    """Min distance from point `p` to segment `vw`. Vendor parity.

    Degenerate case (v == w) is handled with where(l2 == 0, dist(p, v), …).
    """
    l2 = jnp.sum((v - w) ** 2, axis=-1)
    # t = clip(((p - v) · (w - v)) / l2, 0, 1); guard against l2 == 0.
    t_raw = jnp.sum((p - v) * (w - v), axis=-1) / jnp.where(l2 == 0, 1.0, l2)
    t = jnp.clip(t_raw, 0.0, 1.0)
    projection = v + t[..., None] * (w - v)
    seg_dist = distance(p, projection)
    return jnp.where(l2 == 0, distance(p, v), seg_dist)


def swept_pair_hit(
    A: jnp.ndarray, B: jnp.ndarray, P0: jnp.ndarray, P1: jnp.ndarray, r: jnp.ndarray
) -> jnp.ndarray:
    """True iff moving points A->B and P0->P1 come within `r` during t in [0, 1].

    Matches the vendor `swept_pair_hit` quadratic / discriminant approach.

    Shapes:
      A, B, P0, P1: (..., 2)
      r: (...,) or scalar
    Returns:
      bool (...,)
    """
    d0 = A - P0  # (..., 2)
    dv = (B - A) - (P1 - P0)  # (..., 2)
    a = jnp.sum(dv * dv, axis=-1)
    b = 2.0 * jnp.sum(d0 * dv, axis=-1)
    c = jnp.sum(d0 * d0, axis=-1) - r * r

    # Degenerate (no relative motion): hit iff c <= 0.
    linear_hit = c <= 0.0

    disc = b * b - 4.0 * a * c
    # Avoid sqrt of negative; mask via disc_ok.
    disc_ok = disc >= 0.0
    sq = jnp.sqrt(jnp.maximum(disc, 0.0))
    # Solutions: t1 = (-b - sq) / (2a), t2 = (-b + sq) / (2a).
    # We want any t in [0, 1] -> t2 >= 0 AND t1 <= 1.
    denom = 2.0 * jnp.where(a < 1e-12, 1.0, a)  # guard div0; result masked out
    t1 = (-b - sq) / denom
    t2 = (-b + sq) / denom
    quad_hit = disc_ok & (t2 >= 0.0) & (t1 <= 1.0)

    return jnp.where(a < 1e-12, linear_hit, quad_hit)
