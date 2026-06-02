"""Convert a JAX-trained best.pt (npz of Equinox leaves) into a PyTorch
`weights.pt` state_dict loadable by `policy/agent.py`'s ActorCritic.

The JAX npz stores `eqx.filter(model, eqx.is_array)` leaves in tree order
keyed `leaf_0..leaf_N`. We rebuild the ActorCriticJax from those leaves,
then read each named JAX field and write the correspondingly-named PyTorch
state_dict key, inverting the mapping documented in
`model_jax.load_bc_weights_jax` (torch <-> JAX field names).

Note on head MLPs: the PyTorch nn.Sequential has a Dropout at index 2, so
the second Linear sits at index 3 (`.3.weight`) while the JAX field is
`*_2_w`. backbone.psi / value_head.net have no dropout (linears at 0/2).

Usage:
    uv run python -m pipeline.reinforce.case6.training.jax_to_torch \
        --best <run_dir>/best.pt --out pipeline/reinforce/case6/policy/weights.pt
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import numpy as np
import torch
import typer

from ..policy.featurizer_jax import GLOBAL_FEAT_DIM, PLANET_FEAT_DIM
from ..policy.model_jax import ActorCriticJax, ModelConfigJax


def _load_jax_model_from_npz(best_path: Path) -> ActorCriticJax:
    """Rebuild ActorCriticJax by unflattening saved leaves into a fresh model."""
    template = ActorCriticJax.from_init(
        jax.random.PRNGKey(0),
        ModelConfigJax(planet_in_dim=PLANET_FEAT_DIM, global_in_dim=GLOBAL_FEAT_DIM),
    )
    params, static = eqx.partition(template, eqx.is_array)
    flat, treedef = jax.tree.flatten(params)
    npz = np.load(best_path, allow_pickle=False)
    loaded = [jax.numpy.asarray(npz[f"leaf_{i}"]) for i in range(len(flat))]
    if len(loaded) != len(flat):
        raise ValueError(f"leaf count mismatch: npz {len(loaded)} vs model {len(flat)}")
    new_params = jax.tree.unflatten(treedef, loaded)
    model: ActorCriticJax = eqx.combine(new_params, static)
    return model


def _t(arr: Any) -> torch.Tensor:
    return torch.from_numpy(np.asarray(arr)).float()


def _state_dict(m: ActorCriticJax) -> dict[str, torch.Tensor]:
    sd: dict[str, torch.Tensor] = {}
    bb = m.backbone
    sd["backbone.in_proj.weight"] = _t(bb.in_proj_w)
    sd["backbone.in_proj.bias"] = _t(bb.in_proj_b)

    def mab(prefix: str, b: Any) -> None:
        sd[f"{prefix}.mha.in_proj_weight"] = _t(b.mha_in_proj_weight)
        sd[f"{prefix}.mha.in_proj_bias"] = _t(b.mha_in_proj_bias)
        sd[f"{prefix}.mha.out_proj.weight"] = _t(b.mha_out_proj_weight)
        sd[f"{prefix}.mha.out_proj.bias"] = _t(b.mha_out_proj_bias)
        sd[f"{prefix}.ln1.weight"] = _t(b.ln1_w)
        sd[f"{prefix}.ln1.bias"] = _t(b.ln1_b)
        sd[f"{prefix}.ln2.weight"] = _t(b.ln2_w)
        sd[f"{prefix}.ln2.bias"] = _t(b.ln2_b)
        sd[f"{prefix}.ff.0.weight"] = _t(b.ff0_w)
        sd[f"{prefix}.ff.0.bias"] = _t(b.ff0_b)
        sd[f"{prefix}.ff.2.weight"] = _t(b.ff1_w)
        sd[f"{prefix}.ff.2.bias"] = _t(b.ff1_b)

    for i, block in enumerate(bb.encoder):
        sd[f"backbone.encoder.{i}.inducing"] = _t(block.inducing)
        mab(f"backbone.encoder.{i}.mab1", block.mab1)
        mab(f"backbone.encoder.{i}.mab2", block.mab2)

    sd["backbone.pool.seeds"] = _t(bb.pool.seeds)
    mab("backbone.pool.mab", bb.pool.mab)
    sd["backbone.psi.0.weight"] = _t(bb.psi_0_w)
    sd["backbone.psi.0.bias"] = _t(bb.psi_0_b)
    sd["backbone.psi.2.weight"] = _t(bb.psi_2_w)
    sd["backbone.psi.2.bias"] = _t(bb.psi_2_b)

    h = m.head_per_planet
    sd["head_per_planet.src_proj.0.weight"] = _t(h.src_proj_0_w)
    sd["head_per_planet.src_proj.0.bias"] = _t(h.src_proj_0_b)
    sd["head_per_planet.src_proj.3.weight"] = _t(h.src_proj_2_w)
    sd["head_per_planet.src_proj.3.bias"] = _t(h.src_proj_2_b)
    sd["head_per_planet.tgt_proj.0.weight"] = _t(h.tgt_proj_0_w)
    sd["head_per_planet.tgt_proj.0.bias"] = _t(h.tgt_proj_0_b)
    sd["head_per_planet.tgt_proj.3.weight"] = _t(h.tgt_proj_2_w)
    sd["head_per_planet.tgt_proj.3.bias"] = _t(h.tgt_proj_2_b)
    sd["head_per_planet.pair_proj.weight"] = _t(h.pair_proj_w)
    sd["head_per_planet.noop_query"] = _t(h.noop_query)
    sd["head_per_planet.ship_head.0.weight"] = _t(h.ship_head_0_w)
    sd["head_per_planet.ship_head.0.bias"] = _t(h.ship_head_0_b)
    sd["head_per_planet.ship_head.3.weight"] = _t(h.ship_head_2_w)
    sd["head_per_planet.ship_head.3.bias"] = _t(h.ship_head_2_b)

    v = m.value_head
    sd["value_head.net.0.weight"] = _t(v.net_0_w)
    sd["value_head.net.0.bias"] = _t(v.net_0_b)
    sd["value_head.net.2.weight"] = _t(v.net_2_w)
    sd["value_head.net.2.bias"] = _t(v.net_2_b)
    sd["ship_log_std.log_std"] = _t(m.ship_log_std)
    return sd


app = typer.Typer(add_completion=False)

_BEST_OPT = typer.Option(..., "--best", help="JAX best.pt (npz) path.")
_OUT_OPT = typer.Option(..., "--out", help="Output PyTorch weights.pt path.")


@app.command()
def main(
    best: Path = _BEST_OPT,
    out: Path = _OUT_OPT,
) -> None:
    jm = _load_jax_model_from_npz(best)
    sd = _state_dict(jm)
    # Validate against the PyTorch model's expected keys + shapes.
    from ..policy.model import ActorCritic, ModelConfig

    ref = ActorCritic(ModelConfig()).state_dict()
    missing = set(ref) - set(sd)
    extra = set(sd) - set(ref)
    if missing or extra:
        raise ValueError(
            f"key mismatch: missing={sorted(missing)} extra={sorted(extra)}"
        )
    for k, t in sd.items():
        if tuple(t.shape) != tuple(ref[k].shape):
            raise ValueError(
                f"shape mismatch {k}: "
                f"jax {tuple(t.shape)} vs torch {tuple(ref[k].shape)}"
            )
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sd, out)
    print(f"wrote {out} ({len(sd)} tensors)")


if __name__ == "__main__":
    app()
