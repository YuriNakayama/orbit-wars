"""Unit tests for reinforce/case5 combined shaping mode (support_reward H1).

Covers:
  - `_shaping_coefs` resolves (c_ship, c_planet) per mode so that
    reward = c_ship·Δship + c_planet·Δplanet:
      * ships    → (shaping_coef, 0)   (legacy, ship-only)
      * planets  → (0, shaping_coef)   (legacy, planet-only)
      * combined → (coef_ship, coef_planet)  (H1)
  - `collect_rollout_jax(shaping_mode="combined")` runs end-to-end and
    produces finite (non-NaN) rewards — the smoke guard before GPU spend.
  - combined with coef_ship=0 reproduces the planets-mode reward exactly
    (potential-based equivalence of the legacy path).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from pipeline.reinforce.case5.policy.model_jax import ActorCriticJax, ModelConfigJax
from pipeline.reinforce.case5.training.rollout_jax import (
    SHAPING_MODE_COMBINED,
    SHAPING_MODE_PLANETS,
    SHAPING_MODE_RATIO,
    SHAPING_MODE_RATIO_PROD,
    SHAPING_MODE_SHIPS,
    _shaping_coefs,
    collect_rollout_jax,
)


def _coefs(
    mode: int, shaping_coef: float, coef_ship: float, coef_planet: float
) -> tuple[float, float]:
    c_ship, c_planet = _shaping_coefs(
        jnp.int32(mode), shaping_coef, coef_ship, coef_planet
    )
    return float(c_ship), float(c_planet)


def test_shaping_coefs_ships_mode_uses_shaping_coef_for_ship_only() -> None:
    c_ship, c_planet = _coefs(SHAPING_MODE_SHIPS, 0.5, 0.001, 0.5)
    np.testing.assert_allclose(c_ship, 0.5, rtol=1e-6)
    assert c_planet == 0.0


def test_shaping_coefs_planets_mode_uses_shaping_coef_for_planet_only() -> None:
    c_ship, c_planet = _coefs(SHAPING_MODE_PLANETS, 0.5, 0.001, 0.5)
    assert c_ship == 0.0
    np.testing.assert_allclose(c_planet, 0.5, rtol=1e-6)


def test_shaping_coefs_combined_mode_uses_both_explicit_coefs() -> None:
    c_ship, c_planet = _coefs(SHAPING_MODE_COMBINED, 0.5, 0.001, 0.5)
    np.testing.assert_allclose(c_ship, 0.001, rtol=1e-5)
    np.testing.assert_allclose(c_planet, 0.5, rtol=1e-6)


def test_shaping_coefs_ratio_mode_uses_shaping_coef_for_both() -> None:
    """H2: ratio mode applies shaping_coef equally to ship and planet ratio."""
    c_ship, c_planet = _coefs(SHAPING_MODE_RATIO, 0.5, 0.001, 0.5)
    np.testing.assert_allclose(c_ship, 0.5, rtol=1e-6)
    np.testing.assert_allclose(c_planet, 0.5, rtol=1e-6)


def test_shaping_coefs_ratio_prod_mode_uses_shaping_coef_for_both() -> None:
    """H5: ratio_prod applies shaping_coef equally (ship ratio + prod-weighted ratio)."""
    c_ship, c_planet = _coefs(SHAPING_MODE_RATIO_PROD, 1.0, 0.001, 0.5)
    np.testing.assert_allclose(c_ship, 1.0, rtol=1e-6)
    np.testing.assert_allclose(c_planet, 1.0, rtol=1e-6)


def _tiny_model() -> ActorCriticJax:
    cfg = ModelConfigJax(
        hidden=32,
        attn_heads=4,
        inducing_points=4,
        encoder_layers=1,
    )
    return ActorCriticJax.from_init(jax.random.PRNGKey(0), cfg)


def test_combined_rollout_rewards_are_finite() -> None:
    """End-to-end smoke: combined mode must not emit NaN/inf rewards."""
    model = _tiny_model()
    batch = collect_rollout_jax(
        model,
        jax.random.PRNGKey(1),
        episodes_per_iter=2,
        horizon=8,
        seed=0,
        opponent="noop",
        shaping_mode="combined",
        coef_ship=0.001,
        coef_planet=0.5,
    )
    rewards = np.asarray(batch.rewards)
    assert np.all(np.isfinite(rewards)), "combined shaping produced non-finite reward"


def test_ratio_rollout_rewards_are_finite() -> None:
    """H2 end-to-end smoke: ratio mode must not emit NaN/inf rewards.

    Ratio potentials are in [0,1] so per-turn shaping ΔΦ ∈ [-1,1]; with
    shaping_coef=0.5 the shaping term stays bounded and finite.
    """
    model = _tiny_model()
    batch = collect_rollout_jax(
        model,
        jax.random.PRNGKey(3),
        episodes_per_iter=2,
        horizon=8,
        seed=0,
        opponent="noop",
        shaping_mode="ratio",
        shaping_coef=0.5,
    )
    rewards = np.asarray(batch.rewards)
    assert np.all(np.isfinite(rewards)), "ratio shaping produced non-finite reward"


def test_ratio_prod_rollout_rewards_are_finite() -> None:
    """H5 end-to-end smoke: ratio_prod (production-weighted) must stay finite.

    Both ship ratio and production-weighted planet ratio are in [0,1], so the
    per-turn shaping term is bounded; the rollout must not emit NaN/inf.
    """
    model = _tiny_model()
    batch = collect_rollout_jax(
        model,
        jax.random.PRNGKey(4),
        episodes_per_iter=2,
        horizon=8,
        seed=0,
        opponent="noop",
        shaping_mode="ratio_prod",
        shaping_coef=1.0,
    )
    rewards = np.asarray(batch.rewards)
    assert np.all(np.isfinite(rewards)), "ratio_prod shaping produced non-finite reward"


def test_ratio_clip_bounds_per_turn_shaping_reward() -> None:
    """H7: shaping_clip caps the per-turn shaping reward to [-clip, +clip].

    The terminal ±1 is added after the clip, so a finished episode's final
    step can exceed clip; we therefore bound only the pre-terminal steps
    (done_mask True) which carry shaping alone. With coef=1.0 and ratio∈[0,1]
    the unclipped per-turn ΔΦ can reach ~±2 (ship+planet), so a 0.1 clip is
    a strict bound that must hold.
    """
    model = _tiny_model()
    clip = 0.1
    batch = collect_rollout_jax(
        model,
        jax.random.PRNGKey(7),
        episodes_per_iter=2,
        horizon=8,
        seed=0,
        opponent="noop",
        shaping_mode="ratio",
        shaping_coef=1.0,
        shaping_clip=clip,
    )
    rewards = np.asarray(batch.rewards)
    done_mask = np.asarray(batch.done_mask)
    # Pre-terminal steps with no terminal reward must lie within [-clip, clip].
    pre_terminal = rewards[done_mask]
    assert np.all(np.isfinite(pre_terminal))
    assert np.all(np.abs(pre_terminal) <= clip + 1e-5), (
        "ratio clip failed to bound the per-turn shaping reward"
    )


def test_dense_zero_matches_plain_ratio() -> None:
    """H3 control: dense_coef=0 must reproduce plain ratio reward bit-for-bit."""
    model = _tiny_model()
    plain = collect_rollout_jax(
        model,
        jax.random.PRNGKey(11),
        episodes_per_iter=2,
        horizon=8,
        seed=0,
        opponent="noop",
        shaping_mode="ratio",
        shaping_coef=1.0,
    )
    dense_zero = collect_rollout_jax(
        model,
        jax.random.PRNGKey(11),
        episodes_per_iter=2,
        horizon=8,
        seed=0,
        opponent="noop",
        shaping_mode="ratio",
        shaping_coef=1.0,
        dense_coef_ship=0.0,
        dense_coef_planet=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(plain.rewards),
        np.asarray(dense_zero.rewards),
        rtol=1e-6,
        atol=1e-6,
    )


def test_dense_positive_produces_higher_cumulative_reward() -> None:
    """H3 control: dense>0 inflates per-turn reward (mine_count is non-negative).

    The dense addition is `c_ship·ship_mine + c_planet·plt_mine`, both of which
    are ≥ 0 for an active player. So the cumulative reward over pre-terminal
    steps must be strictly higher with dense>0 than with dense=0.
    """
    model = _tiny_model()
    plain = collect_rollout_jax(
        model,
        jax.random.PRNGKey(12),
        episodes_per_iter=2,
        horizon=8,
        seed=0,
        opponent="noop",
        shaping_mode="ratio",
        shaping_coef=1.0,
    )
    dense = collect_rollout_jax(
        model,
        jax.random.PRNGKey(12),
        episodes_per_iter=2,
        horizon=8,
        seed=0,
        opponent="noop",
        shaping_mode="ratio",
        shaping_coef=1.0,
        dense_coef_ship=0.01,
        dense_coef_planet=0.1,
    )
    # Compare cumulative shaping (pre-terminal steps via done_mask).
    plain_sum = float(
        jnp.sum(jnp.where(plain.done_mask, plain.rewards, jnp.float32(0.0)))
    )
    dense_sum = float(
        jnp.sum(jnp.where(dense.done_mask, dense.rewards, jnp.float32(0.0)))
    )
    assert dense_sum > plain_sum, (
        f"dense addition expected to inflate reward (mine_count ≥ 0): "
        f"dense_sum={dense_sum} vs plain_sum={plain_sum}"
    )


def test_ratio_clip_zero_matches_unclipped_ratio() -> None:
    """shaping_clip=0 must reproduce the plain ratio reward bit-for-bit."""
    model = _tiny_model()
    unclipped = collect_rollout_jax(
        model,
        jax.random.PRNGKey(8),
        episodes_per_iter=2,
        horizon=8,
        seed=0,
        opponent="noop",
        shaping_mode="ratio",
        shaping_coef=1.0,
    )
    clip_zero = collect_rollout_jax(
        model,
        jax.random.PRNGKey(8),
        episodes_per_iter=2,
        horizon=8,
        seed=0,
        opponent="noop",
        shaping_mode="ratio",
        shaping_coef=1.0,
        shaping_clip=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(unclipped.rewards),
        np.asarray(clip_zero.rewards),
        rtol=1e-6,
        atol=1e-6,
    )


def test_combined_with_zero_ship_coef_matches_planets_mode() -> None:
    """coef_ship=0 + coef_planet=c must equal planets mode (shaping_coef=c)."""
    model = _tiny_model()
    planets = collect_rollout_jax(
        model,
        jax.random.PRNGKey(2),
        episodes_per_iter=2,
        horizon=8,
        seed=0,
        opponent="noop",
        shaping_mode="planets",
        shaping_coef=0.5,
    )
    combined = collect_rollout_jax(
        model,
        jax.random.PRNGKey(2),
        episodes_per_iter=2,
        horizon=8,
        seed=0,
        opponent="noop",
        shaping_mode="combined",
        coef_ship=0.0,
        coef_planet=0.5,
    )
    np.testing.assert_allclose(
        np.asarray(planets.rewards), np.asarray(combined.rewards), rtol=1e-6, atol=1e-6
    )
