"""Set Transformer + per_planet head policy for imitation/case11.

case11 keeps only the per_planet variant of case9:
  per-source × (P+1) target categorical (planet_id 直接予測 + no-op sentinel) +
  log1p(ships) regression head.

Backbone: Set Transformer ISAB×L + PMA + global concat.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from .candidates import CAND_FEAT_DIM, CAND_K
from .featurizer import GLOBAL_FEAT_DIM, MAX_PLANETS, PLANET_FEAT_DIM
from .heads.backbone import BackboneConfig, SetTransformerBackbone
from .heads.per_planet import PerPlanetHead
from .types import BatchFeatures, PolicyOutput

SUPPORTED_HEAD_MODES = ("per_planet",)


@dataclass(frozen=True)
class ModelConfig:
    planet_in_dim: int = PLANET_FEAT_DIM
    global_in_dim: int = GLOBAL_FEAT_DIM
    cand_in_dim: int = CAND_FEAT_DIM
    cand_k: int = CAND_K
    hidden: int = 128
    attn_heads: int = 4
    inducing_points: int = 16
    encoder_layers: int = 3
    ships_buckets: int = 4
    head_dropout: float = 0.0
    head_mode: str = "per_planet"


class Case11Policy(nn.Module):
    """Set Transformer backbone + per_planet head."""

    def __init__(self, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ModelConfig()
        if self.cfg.head_mode not in SUPPORTED_HEAD_MODES:
            raise ValueError(
                f"unknown head_mode={self.cfg.head_mode!r}; "
                f"supported: {SUPPORTED_HEAD_MODES}"
            )
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

    def forward(self, batch: BatchFeatures) -> PolicyOutput:
        h_with_ctx, ctx, h = self.backbone(
            batch.planet_feats, batch.planet_mask, batch.global_feats
        )
        per_planet_logits, ship_pred = self.head_per_planet(
            h_with_ctx, ctx, h, batch.planet_mask
        )
        return PolicyOutput(per_planet_logits=per_planet_logits, ship_pred=ship_pred)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(cfg: ModelConfig | None = None) -> Case11Policy:
    """Convenience constructor used by training and inference."""
    model = Case11Policy(cfg)
    model.eval()
    return model


__all__ = [
    "Case11Policy",
    "MAX_PLANETS",
    "ModelConfig",
    "SUPPORTED_HEAD_MODES",
    "build_model",
    "count_parameters",
]
