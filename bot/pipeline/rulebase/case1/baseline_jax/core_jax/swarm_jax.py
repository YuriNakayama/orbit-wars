"""JAX port of process_multi_source_mission's ordered allocation (2-source).

mission_resolver.py:108-118: sort options by (turns, -limit, src_id), then
allocate send = min(limit, max(0, remaining - sum_of_later_limits)) in order.
Returns per-source sends (0 if the swarm can't cover `need`).

Parity-tested standalone before wiring into the agent (a prior naive split
regressed the win-rate). Fixed 2-source here (the dominant case); 3-source later.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

Arr = jax.Array


def allocate_2(
    need: Arr,
    limit_a: Arr,
    turns_a: Arr,
    src_a: Arr,
    limit_b: Arr,
    turns_b: Arr,
    src_b: Arr,
) -> tuple[Arr, Arr, Arr]:
    """Return (send_a, send_b, ok) for a 2-source swarm on one target.

    Mirrors the ordered allocation: order by (turns, -limit, src_id); first gets
    min(limit, max(0, need - other_limit)); second gets the remainder. ok=False
    (sends 0) if combined can't cover need.
    """
    # determine order: a before b iff (turns_a, -limit_a, src_a) < (...b)
    a_first = (turns_a < turns_b) | (
        (turns_a == turns_b)
        & ((limit_a > limit_b) | ((limit_a == limit_b) & (src_a < src_b)))
    )
    # first/second limits in allocation order
    lim1 = jnp.where(a_first, limit_a, limit_b)
    lim2 = jnp.where(a_first, limit_b, limit_a)
    # idx0: remaining_other = lim2; send1 = min(lim1, max(0, need - lim2))
    send1 = jnp.minimum(lim1, jnp.maximum(0, need - lim2))
    remaining = need - send1
    # idx1: remaining_other = 0; send2 = min(lim2, max(0, remaining))
    send2 = jnp.minimum(lim2, jnp.maximum(0, remaining))
    covered = (send1 + send2) >= need
    ok = covered & (need > 0)
    # map back to a/b
    send_a = jnp.where(a_first, send1, send2)
    send_b = jnp.where(a_first, send2, send1)
    send_a = jnp.where(ok, send_a, 0)
    send_b = jnp.where(ok, send_b, 0)
    return send_a.astype(jnp.int32), send_b.astype(jnp.int32), ok
