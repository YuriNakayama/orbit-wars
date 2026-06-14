"""Self-Imitation Learning (SIL) winning-episode replay buffer.

Oh et al. 2018 (https://arxiv.org/abs/1806.05635): re-train on the agent's own
past good decisions to amplify scarce successful trajectories. Here it targets
the rare strict_v1 wins (~1/64) that the curriculum produces but the V-MPO
update otherwise dilutes within a single iteration's batch.

A fixed-capacity circular buffer holds flattened *winning* timesteps
(`FlatRollout` rows whose source episode had `episode_outcome > 0`). Each iter,
a clipped-advantage-prioritised minibatch is sampled and concatenated onto the
current rollout's `FlatRollout` before the policy update — so good past steps
keep getting reinforced across iterations, not just the iter they appeared in.

All ops are fixed-shape and jit/scan-friendly: the buffer is a `FlatRollout`
pytree of capacity-C leaves plus a scalar `count` and `cursor`. Disabled
(`enabled=False` in the caller) it is never touched, so the run stays
bit-identical to the no-SIL baseline.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .ppo_jax import FlatRollout


def sil_buffer_init(template: FlatRollout, capacity: int) -> FlatRollout:
    """Allocate a capacity-C buffer shaped like a single `FlatRollout` row set.

    Every leaf becomes `(capacity, *leaf.shape[1:])`, zero-filled. The buffer is
    itself a `FlatRollout`, so add/sample reuse the same pytree structure.
    """

    def alloc(x: jax.Array) -> jax.Array:
        return jnp.zeros((capacity,) + x.shape[1:], dtype=x.dtype)

    out: FlatRollout = jax.tree.map(alloc, template)
    return out


def sil_buffer_add(
    buffer: FlatRollout,
    cursor: jax.Array,
    count: jax.Array,
    new_steps: FlatRollout,
    keep_mask: jax.Array,
) -> tuple[FlatRollout, jax.Array, jax.Array]:
    """Scatter the `keep_mask`-selected rows of `new_steps` into the circular
    buffer starting at `cursor`.

    `keep_mask` (N,) marks the winning + valid steps to persist. Selection is
    done with a stable argsort on `~keep_mask` (kept rows float to the front)
    so the scatter count is data-dependent only through `n_add`, keeping the
    shape static. Capacity overflow wraps (FIFO overwrite).
    """
    capacity = jax.tree.leaves(buffer)[0].shape[0]
    n = keep_mask.shape[0]
    # Stable order: kept rows first, preserving their original order.
    order = jnp.argsort(~keep_mask, stable=True)
    n_add = jnp.sum(keep_mask.astype(jnp.int32))
    # Destination slots for the first n_add ordered rows (wrap around capacity).
    dest = (cursor + jnp.arange(n)) % capacity
    write = jnp.arange(n) < n_add  # only the first n_add are real adds

    def scatter(buf_leaf: jax.Array, new_leaf: jax.Array) -> jax.Array:
        src = new_leaf[order]  # kept rows up front
        # Mask non-writes by pointing them at slot 0 with a no-op value: we use
        # a where on the gathered-then-scattered result instead, to stay pure.
        updated = buf_leaf.at[dest].set(
            jnp.where(
                write.reshape((-1,) + (1,) * (new_leaf.ndim - 1)),
                src,
                buf_leaf[dest],
            )
        )
        return updated

    new_buffer = jax.tree.map(scatter, buffer, new_steps)
    new_cursor = (cursor + n_add) % capacity
    new_count = jnp.minimum(count + n_add, capacity)
    return new_buffer, new_cursor, new_count


def sil_buffer_sample(
    buffer: FlatRollout,
    count: jax.Array,
    key: jax.Array,
    sample_size: int,
) -> FlatRollout:
    """Draw `sample_size` rows prioritised by clipped advantage `(R - V)+`.

    Only the first `count` slots are valid; invalid slots get priority 0 so
    they are never drawn once the buffer has any content. Sampling is
    with-replacement via `jax.random.choice` over a normalised priority vector
    (fixed shape = capacity), which is jit-friendly.
    """
    capacity = jax.tree.leaves(buffer)[0].shape[0]
    valid = jnp.arange(capacity) < count
    # Clipped advantage as priority (SIL: (R - V)+). buffer.advantages already
    # is the GAE advantage at store time; clip at 0 so only "good" steps weigh.
    prio = jnp.clip(buffer.advantages, min=0.0) * valid.astype(jnp.float32)
    # Fall back to uniform-over-valid if all priorities are zero.
    total = jnp.sum(prio)
    uniform = valid.astype(jnp.float32) / jnp.maximum(jnp.sum(valid), 1)
    probs = jnp.where(total > 0.0, prio / jnp.maximum(total, 1e-8), uniform)
    idx = jax.random.choice(
        key, capacity, shape=(sample_size,), replace=True, p=probs
    )
    out: FlatRollout = jax.tree.map(lambda leaf: leaf[idx], buffer)
    return out


def concat_flat(a: FlatRollout, b: FlatRollout) -> FlatRollout:
    """Concatenate two `FlatRollout`s along the leading (N) axis."""
    out: FlatRollout = jax.tree.map(
        lambda x, y: jnp.concatenate([x, y], axis=0), a, b
    )
    return out
