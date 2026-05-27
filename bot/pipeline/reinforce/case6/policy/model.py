"""Actor-Critic model for reinforce/case6 PPO baseline.

Architecture (case9 per_planet variant + value head, BC warm-start friendly):

  - `backbone`         Set Transformer (ISAB×4, hidden=192, attn=8, m=24)
                       — keys are identical to case9 so BC weights load cleanly
  - `head_per_planet`  Pointer-style (P+1) categorical + ship_pred regression
                       — same names as case9 so BC keys map 1:1
  - `value_head`       New scalar V head on `ctx` (random init on warm-start)
  - `ship_log_std`     New learnable scalar log-σ for the Gaussian ships head
                       (state-independent; ship_pred is the mean)

The forward pass returns `PolicyOutput` that wraps:
  per_planet_logits  (B, P, P+1)
  ship_mean          (B, P)            # = ship_pred in log1p space
  ship_log_std       scalar
  value              (B,)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .featurizer import GLOBAL_FEAT_DIM, MAX_PLANETS, PLANET_FEAT_DIM
from .heads.backbone import BackboneConfig, SetTransformerBackbone
from .heads.per_planet import PerPlanetHead
from .heads.value import ShipLogStd, ValueHead
from .types import BatchFeatures


@dataclass(frozen=True)
class ModelConfig:
    planet_in_dim: int = PLANET_FEAT_DIM
    global_in_dim: int = GLOBAL_FEAT_DIM
    hidden: int = 192
    attn_heads: int = 8
    inducing_points: int = 24
    encoder_layers: int = 4
    head_dropout: float = 0.0
    ship_log_std_init: float = 0.5
    # Subtracted from per_planet_logits[:, :, NO_OP_INDEX] before sampling.
    # BC is heavily biased to no-op (logit +4.9 vs best fire −5.3), so without
    # this offset Categorical samples no-op with probability ≈ 1.0 → PPO never
    # explores fire actions. case9's inference uses IL_CASE9_PP_NOOP_BIAS; we
    # bake the same idea into the model so sample / greedy / evaluate stay
    # consistent.
    no_op_bias: float = 8.0


@dataclass(frozen=True)
class PolicyOutput:
    per_planet_logits: torch.Tensor  # (B, P, P+1)
    ship_mean: torch.Tensor  # (B, P) — log1p(ships) space
    ship_log_std: torch.Tensor  # scalar
    value: torch.Tensor  # (B,)


class ActorCritic(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ModelConfig()
        bb_cfg = BackboneConfig(
            planet_in_dim=self.cfg.planet_in_dim,
            global_in_dim=self.cfg.global_in_dim,
            hidden=self.cfg.hidden,
            attn_heads=self.cfg.attn_heads,
            inducing_points=self.cfg.inducing_points,
            encoder_layers=self.cfg.encoder_layers,
        )
        self.backbone = SetTransformerBackbone(bb_cfg)
        self.head_per_planet = PerPlanetHead(
            hidden=self.cfg.hidden,
            head_dropout=self.cfg.head_dropout,
        )
        self.value_head = ValueHead(self.cfg.hidden)
        self.ship_log_std = ShipLogStd(self.cfg.ship_log_std_init)

    def forward(self, batch: BatchFeatures) -> PolicyOutput:
        h_with_ctx, ctx, h = self.backbone(
            batch.planet_feats, batch.planet_mask, batch.global_feats
        )
        per_planet_logits, ship_mean = self.head_per_planet(
            h_with_ctx, ctx, h, batch.planet_mask
        )
        if self.cfg.no_op_bias != 0.0:
            # Index NO_OP_INDEX = MAX_PLANETS (the last slot in the (P+1) dim).
            offset = torch.zeros_like(per_planet_logits)
            offset[..., -1] = -float(self.cfg.no_op_bias)
            per_planet_logits = per_planet_logits + offset
        value = self.value_head(ctx)
        return PolicyOutput(
            per_planet_logits=per_planet_logits,
            ship_mean=ship_mean,
            ship_log_std=self.ship_log_std(),
            value=value,
        )


def load_bc_weights(model: ActorCritic, weights_path: str) -> tuple[int, int]:
    """Load BC (case9 per_planet) weights, ignoring missing/extra keys.

    Returns (loaded, missing). `missing` should equal the number of params in
    `value_head` + `ship_log_std` on a clean warm-start.
    """
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    own = model.state_dict()
    loaded = 0
    for k, v in state.items():
        if k in own and own[k].shape == v.shape:
            own[k] = v
            loaded += 1
    model.load_state_dict(own, strict=True)
    missing = len(own) - loaded
    return loaded, missing


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(cfg: ModelConfig | None = None) -> ActorCritic:
    model = ActorCritic(cfg)
    model.eval()
    return model


__all__ = [
    "ActorCritic",
    "ModelConfig",
    "PolicyOutput",
    "MAX_PLANETS",
    "build_model",
    "count_parameters",
    "load_bc_weights",
]
