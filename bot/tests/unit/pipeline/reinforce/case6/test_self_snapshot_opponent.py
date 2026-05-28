"""H1: self_snapshot opponent path in reinforce/case6 rollout_jax.

Verifies the PFSP foundation:
  - `self_snapshot` requires `opp_model` (raises otherwise).
  - rollout against a frozen self snapshot runs and returns shape-correct,
    finite tensors.
  - the existing noop / baseline opponents still work (no regression from
    threading the extra opp_model arg).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from pipeline.reinforce.case6.policy.featurizer_jax import (
    GLOBAL_FEAT_DIM,
    PLANET_FEAT_DIM,
)
from pipeline.reinforce.case6.policy.model_jax import ActorCriticJax, ModelConfigJax
from pipeline.reinforce.case6.training.rollout_jax import (
    OPPONENT_NAME_TO_MODE,
    OPPONENT_SELF_SNAPSHOT,
    collect_rollout_jax,
)

_EPISODES = 2
_HORIZON = 30


def _tiny_model() -> ActorCriticJax:
    cfg = ModelConfigJax(
        planet_in_dim=PLANET_FEAT_DIM,
        global_in_dim=GLOBAL_FEAT_DIM,
        hidden=32,
        attn_heads=4,
        inducing_points=4,
        encoder_layers=1,
    )
    return ActorCriticJax.from_init(jax.random.PRNGKey(0), cfg)


def test_self_snapshot_registered() -> None:
    assert OPPONENT_NAME_TO_MODE["self_snapshot"] == OPPONENT_SELF_SNAPSHOT


def test_self_snapshot_requires_opp_model() -> None:
    model = _tiny_model()
    with pytest.raises(ValueError, match="self_snapshot"):
        collect_rollout_jax(
            model,
            jax.random.PRNGKey(0),
            episodes_per_iter=_EPISODES,
            horizon=_HORIZON,
            opponent="self_snapshot",
        )


def test_self_snapshot_rollout_runs_and_is_finite() -> None:
    model = _tiny_model()
    opp_model = _tiny_model()  # distinct frozen snapshot
    batch = collect_rollout_jax(
        model,
        jax.random.PRNGKey(1),
        episodes_per_iter=_EPISODES,
        horizon=_HORIZON,
        opponent="self_snapshot",
        opp_model=opp_model,
    )
    assert batch.rewards.shape == (_EPISODES, _HORIZON)
    assert batch.episode_outcomes.shape == (_EPISODES,)
    assert bool(jnp.all(jnp.isfinite(batch.rewards)))
    assert bool(jnp.all(jnp.isfinite(batch.episode_outcomes)))


@pytest.mark.parametrize("opponent", ["noop", "baseline_jax_lite"])
def test_non_snapshot_opponents_still_run(opponent: str) -> None:
    """Regression: threading opp_model must not break the existing modes."""
    model = _tiny_model()
    batch = collect_rollout_jax(
        model,
        jax.random.PRNGKey(2),
        episodes_per_iter=_EPISODES,
        horizon=_HORIZON,
        opponent=opponent,
    )
    assert batch.rewards.shape == (_EPISODES, _HORIZON)
    assert bool(jnp.all(jnp.isfinite(batch.rewards)))


# --- H2: opponent pool ---


def test_pool_fifo_cap_and_sample() -> None:
    import numpy as np

    from pipeline.reinforce.case6.training.train_jax import _OpponentPool

    pool = _OpponentPool(cap=3)
    assert len(pool) == 0
    models = [_tiny_model() for _ in range(5)]
    for m in models:
        pool.push(m)
    # FIFO: only the last 3 survive.
    assert len(pool) == 3
    # sample returns a usable model (forward runs).
    rng = np.random.default_rng(0)
    sampled = pool.sample(rng)
    out = collect_rollout_jax(
        _tiny_model(),
        jax.random.PRNGKey(3),
        episodes_per_iter=_EPISODES,
        horizon=_HORIZON,
        opponent="self_snapshot",
        opp_model=sampled,
    )
    assert out.rewards.shape == (_EPISODES, _HORIZON)
    assert bool(jnp.all(jnp.isfinite(out.rewards)))


def test_pool_cap_one() -> None:
    import numpy as np

    from pipeline.reinforce.case6.training.train_jax import _OpponentPool

    pool = _OpponentPool(cap=1)
    pool.push(_tiny_model())
    pool.push(_tiny_model())
    assert len(pool) == 1
    assert pool.sample(np.random.default_rng(0)) is not None


# --- H4: prioritized opponent selector (f_hard) ---


def test_prioritized_selector_favors_hard_opponents() -> None:
    import numpy as np

    from pipeline.reinforce.case6.training.train_jax import (
        _PrioritizedOpponentSelector,
    )

    sel = _PrioritizedOpponentSelector(p=2.0, ema=0.7)
    sel.set_entries([_tiny_model(), _tiny_model()], include_full=True)
    assert len(sel) == 3  # full + 2 snapshots
    # Make entry 0 (full) a hard opponent (low win), entry 1 easy (high win).
    sel.update(0, 0.0)  # win_ema -> 0.7*0.5 + 0.3*0.0 = 0.35
    sel.update(1, 1.0)  # win_ema -> 0.7*0.5 + 0.3*1.0 = 0.65
    rng = np.random.default_rng(0)
    counts = [0, 0, 0]
    for _ in range(2000):
        idx, _entry = sel.sample(rng)
        counts[idx] += 1
    # f_hard: (1-0.35)^2=0.42 vs (1-0.65)^2=0.12 → hard entry 0 sampled more.
    assert counts[0] > counts[1]


def test_prioritized_selector_ema_update() -> None:
    from pipeline.reinforce.case6.training.train_jax import (
        _PrioritizedOpponentSelector,
    )

    sel = _PrioritizedOpponentSelector(p=2.0, ema=0.7)
    sel.set_entries([], include_full=True)
    sel.update(0, 1.0)
    # 0.7*0.5 + 0.3*1.0 = 0.65
    assert abs(sel._entries[0].win_ema - 0.65) < 1e-9


def test_prioritized_selector_carries_ema_on_pool_refresh() -> None:
    from pipeline.reinforce.case6.training.train_jax import (
        _PrioritizedOpponentSelector,
    )

    sel = _PrioritizedOpponentSelector(p=2.0, ema=0.7)
    m1, m2 = _tiny_model(), _tiny_model()
    sel.set_entries([m1], include_full=True)
    sel.update(0, 0.0)  # full hard
    full_ema = sel._entries[0].win_ema
    # Pool refresh adds a new snapshot; full entry keeps its EMA.
    sel.set_entries([m1, m2], include_full=True)
    assert abs(sel._entries[0].win_ema - full_ema) < 1e-9
