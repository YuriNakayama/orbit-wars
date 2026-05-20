"""JAX action sampling + log-prob for ActorCriticJax.

Mirrors `sampling.py` but produces JAX arrays so the whole rollout step
can sit inside `jit(scan(env_step))`. Action factorization per source
planet `s`:
  - target_slot[s] ~ Categorical(per_planet_logits[s])     # P+1 slots
  - log1p_ships[s] ~ Normal(ship_mean[s], exp(log_std))    # state-indep σ

Returns the env action tensor directly so the rollout loop doesn't have
to round-trip through Python lists.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .featurizer_jax import MAX_PLANETS
from .model_jax import PolicyOutputJax

NO_OP_INDEX = MAX_PLANETS  # last slot in per_planet_logits


class SampledActionJax(NamedTuple):
    target_slot: jax.Array  # (B, P) int32
    log1p_ships: jax.Array  # (B, P) float32
    log_prob: jax.Array  # (B,) float32
    entropy: jax.Array  # (B,) float32


def _categorical_sample(
    logits: jax.Array, key: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Sample from Categorical(logits) along the last axis.

    Returns (sample, log_prob_of_sample, entropy_per_row).
    """
    log_softmax = jax.nn.log_softmax(logits, axis=-1)
    # Sample via Gumbel-max trick.
    g = -jnp.log(
        -jnp.log(jax.random.uniform(key, logits.shape, minval=1e-9, maxval=1.0))
    )
    sample = jnp.argmax(log_softmax + g, axis=-1)  # (..., )
    # log_prob: log_softmax[..., sample]
    lp = jnp.take_along_axis(log_softmax, sample[..., None], axis=-1).squeeze(-1)
    probs = jnp.exp(log_softmax)
    entropy = -jnp.sum(probs * log_softmax, axis=-1)
    return sample, lp, entropy


def _normal_sample(
    mean: jax.Array, log_std: jax.Array, key: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    std = jnp.maximum(1e-3, jnp.exp(log_std))
    std = jnp.broadcast_to(std, mean.shape)
    eps = jax.random.normal(key, mean.shape)
    sample = mean + std * eps
    var = std**2
    lp = -0.5 * jnp.log(2 * jnp.pi * var) - 0.5 * (sample - mean) ** 2 / var
    # Differential entropy of Normal = 0.5 * log(2πeσ²).
    entropy = 0.5 * (jnp.log(2 * jnp.pi * var) + 1.0)
    return sample, lp, entropy


def sample_action_jax(
    output: PolicyOutputJax,
    my_planet_mask: jax.Array,  # (B, P) bool
    key: jax.Array,
) -> SampledActionJax:
    """Mirror sampling.sample_action.

    Non-source rows (~my_planet_mask) get forced to NO_OP / zero ships
    after sampling, and contribute zero to log_prob / entropy.
    """
    k_cat, k_norm = jax.random.split(key, 2)
    pp_logits = output.per_planet_logits  # (B, P, P+1)

    safe_logits = jnp.where(
        my_planet_mask[..., None], pp_logits, jnp.zeros_like(pp_logits)
    )
    target_sample, t_lp, t_ent = _categorical_sample(safe_logits, k_cat)

    ship_mean = output.ship_mean
    log_std = output.ship_log_std
    ship_sample, s_lp, s_ent = _normal_sample(ship_mean, log_std, k_norm)

    # Sum log_prob / entropy over my_planet_mask rows.
    mask_f = my_planet_mask.astype(jnp.float32)
    log_prob = jnp.sum(t_lp * mask_f, axis=-1) + jnp.sum(s_lp * mask_f, axis=-1)
    entropy = jnp.sum(t_ent * mask_f, axis=-1) + jnp.sum(s_ent * mask_f, axis=-1)

    target_slot = jnp.where(
        my_planet_mask,
        target_sample,
        jnp.full_like(target_sample, NO_OP_INDEX),
    )
    log1p_ships = jnp.where(my_planet_mask, ship_sample, jnp.zeros_like(ship_sample))

    return SampledActionJax(
        target_slot=target_slot.astype(jnp.int32),
        log1p_ships=log1p_ships,
        log_prob=log_prob,
        entropy=entropy,
    )


def sampled_action_to_env_actions(
    target_slot: jax.Array,  # (P,) int32, in [0, P]; P = no-op
    log1p_ships: jax.Array,  # (P,) float32
    planet_xy: jax.Array,  # (P_total, 2) float32 (full state.planet_xy)
    planet_ships: jax.Array,  # (P_total,) int32 (full state.planet_ships)
    planet_id: jax.Array,  # (P_total,) int32
    valid_full: jax.Array,  # (P_total,) bool
    is_mine: jax.Array,  # (MAX_PLANETS,) bool
    seat: int,
    num_agents_max: int,
) -> jax.Array:
    """Build a JAX env action tensor (NUM_AGENTS, MAX_LAUNCHES, 3) for one env.

    For each own (is_mine) planet slot `s` with `target_slot[s] != NO_OP`:
      - from_planet_id = planet_id[s]
      - angle = atan2(target_y - src_y, target_x - src_x) using full
                state.planet_xy positions (so the target_slot index is
                interpreted as a planet slot index in the env state).
      - ships = clamp(round(expm1(log1p_ships[s])), 1, src.ships)

    Note that target_slot indexes into [0, MAX_PLANETS], not into the full
    state.planet_xy (which has length JAX_MAX_PLANETS). Since the
    featurizer truncates to MAX_PLANETS the index is well-defined for
    valid targets.

    Other seats get all-no-op rows.
    """
    P = MAX_PLANETS  # number of own-mask slots
    src_xy = planet_xy[:P]  # (P, 2)
    src_ships_i = planet_ships[:P]  # (P,)
    src_pid = planet_id[:P]  # (P,)

    is_fire = (target_slot != NO_OP_INDEX) & is_mine  # (P,)
    safe_tgt = jnp.clip(target_slot, 0, P - 1)
    tgt_xy = planet_xy[safe_tgt]  # (P, 2)
    dx = tgt_xy[:, 0] - src_xy[:, 0]
    dy = tgt_xy[:, 1] - src_xy[:, 1]
    angle = jnp.arctan2(dy, dx)

    raw_ships = jnp.maximum(0.0, jnp.expm1(log1p_ships))
    int_ships = jnp.round(raw_ships).astype(jnp.int32)
    int_ships = jnp.clip(int_ships, 1, src_ships_i)
    # When not firing, set from_pid=-1 (no-op sentinel for the env step).
    from_pid = jnp.where(is_fire, src_pid, jnp.int32(-1))
    final_angle = jnp.where(is_fire, angle, jnp.float32(0.0))
    final_ships = jnp.where(is_fire, int_ships, jnp.int32(0))

    # Build seat's row: (P, 3). Then pad to (MAX_LAUNCHES_PER_AGENT, 3).
    # JAX env's `actions` shape is (NUM_AGENTS_MAX, MAX_LAUNCHES_PER_AGENT, 3).
    # MAX_LAUNCHES_PER_AGENT == MAX_PLANETS (= 48 in jax_env) > our P=36.
    # We pad with -1 from_pid sentinels.
    # First col is from_pid, second angle, third ships.
    seat_actions = jnp.stack(
        [
            from_pid.astype(jnp.float32),
            final_angle,
            final_ships.astype(jnp.float32),
        ],
        axis=-1,
    )  # (P, 3)
    # Pad to MAX_LAUNCHES_PER_AGENT (= jax_env.MAX_PLANETS = 48).
    from jax_env.step import MAX_LAUNCHES_PER_AGENT

    pad_len = MAX_LAUNCHES_PER_AGENT - P
    pad = jnp.full((pad_len, 3), -1.0, dtype=jnp.float32)
    seat_full = jnp.concatenate([seat_actions, pad], axis=0)  # (L, 3)

    # Build the (NUM_AGENTS_MAX, L, 3) tensor with only `seat` populated.
    actions = jnp.full(
        (num_agents_max, MAX_LAUNCHES_PER_AGENT, 3), -1.0, dtype=jnp.float32
    )
    actions = actions.at[seat].set(seat_full)
    # `actions` is float32 per jax_env step contract.
    del valid_full  # unused (kept in signature for future per-planet filters)
    return actions


__all__ = [
    "NO_OP_INDEX",
    "SampledActionJax",
    "sample_action_jax",
    "sampled_action_to_env_actions",
]
