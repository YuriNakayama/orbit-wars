"""Forward-pass parity between PyTorch ActorCritic and JAX ActorCriticJax.

Initializes a PyTorch ActorCritic with a fixed seed, saves its state_dict
to a temp .pt, loads into ActorCriticJax via `load_bc_weights_jax`, and
asserts that forward outputs match within float32 tolerance.

Pre-requisites for the parity:
  - timeline cols 35-40 are zero in the JAX featurizer (W2e deferred);
    we zero them in the torch input too so the inputs are identical.
"""

from __future__ import annotations

import os
import tempfile

import jax
import numpy as np
import pytest
import torch
from orbit_wars_jax.observation import state_to_obs
from orbit_wars_jax.reset import reset

from pipeline.reinforce.case1.policy.featurizer import (
    HistoryState,
    featurize,
)
from pipeline.reinforce.case1.policy.featurizer_jax import featurize_jax_w1
from pipeline.reinforce.case1.policy.model import ActorCritic, ModelConfig
from pipeline.reinforce.case1.policy.model_jax import (
    ActorCriticJax,
    load_bc_weights_jax,
)

PARITY_TOL = 1e-4


def _torch_to_jax_models(torch_seed: int) -> tuple[ActorCritic, ActorCriticJax]:
    torch.manual_seed(torch_seed)
    torch_model = ActorCritic(ModelConfig())
    torch_model.eval()
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        weights_path = f.name
        torch.save(torch_model.state_dict(), weights_path)
    try:
        jax_model = ActorCriticJax.from_init(jax.random.PRNGKey(0))
        jax_model, loaded, missing = load_bc_weights_jax(jax_model, weights_path)
        assert missing == 0, (
            f"BC load left {missing} torch keys unmapped; loaded={loaded}"
        )
        return torch_model, jax_model
    finally:
        os.unlink(weights_path)


@pytest.mark.parametrize("seed", [0, 7, 42])
@pytest.mark.parametrize("torch_seed", [1, 13])
def test_forward_parity(seed: int, torch_seed: int) -> None:
    """ActorCriticJax forward must match PyTorch's within float32 tol."""
    torch_model, jax_model = _torch_to_jax_models(torch_seed)
    state = reset(seed=seed, num_agents=2)
    obs = state_to_obs(state, player=0)
    torch_batch, _ = featurize(obs, history=HistoryState())
    jax_batch = featurize_jax_w1(state, player=0)

    with torch.no_grad():
        torch_out = torch_model(torch_batch)
    jax_out = jax_model(jax_batch)

    diff_pl = np.abs(
        torch_out.per_planet_logits[0].cpu().numpy()
        - np.asarray(jax_out.per_planet_logits[0])
    )
    diff_ship = np.abs(
        torch_out.ship_mean[0].cpu().numpy() - np.asarray(jax_out.ship_mean[0])
    )
    diff_value = abs(torch_out.value.item() - float(jax_out.value[0]))

    assert diff_pl.max() < PARITY_TOL, (
        f"seed={seed} torch_seed={torch_seed} "
        f"per_planet_logits max diff {diff_pl.max():.3e}"
    )
    assert diff_ship.max() < PARITY_TOL, (
        f"seed={seed} torch_seed={torch_seed} ship_mean max diff {diff_ship.max():.3e}"
    )
    assert diff_value < PARITY_TOL, (
        f"seed={seed} torch_seed={torch_seed} value diff {diff_value:.3e}"
    )


def test_bc_weight_count_matches() -> None:
    """JAX model param count matches PyTorch (3.1M)."""
    import equinox as eqx

    torch.manual_seed(0)
    torch_model = ActorCritic(ModelConfig())
    torch_n = sum(p.numel() for p in torch_model.parameters() if p.requires_grad)

    jax_model = ActorCriticJax.from_init(jax.random.PRNGKey(0))
    params = eqx.filter(jax_model, eqx.is_array)
    leaves = jax.tree.leaves(params)
    jax_n = sum(int(np.prod(np.asarray(leaf.shape))) for leaf in leaves)

    assert torch_n == jax_n, f"param count diff: torch={torch_n} jax={jax_n}"
