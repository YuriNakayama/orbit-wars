"""Set Transformer policy for imitation/case10.

case10 keeps only the two non-3-head families:
  - "candidate_ships": candidate categorical + 4-bucket ships logits
  - "template_ships": template categorical incl no-op + ships bucket
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from .candidates import CAND_FEAT_DIM, CAND_K
from .featurizer import GLOBAL_FEAT_DIM, MAX_PLANETS, PLANET_FEAT_DIM
from .heads.backbone import BackboneConfig, SetTransformerBackbone
from .heads.candidate_ships import CandidateShipsHead
from .heads.template_ships import TemplateShipsHead
from .types import BatchFeatures, PolicyOutput

SUPPORTED_HEAD_MODES = (
    "candidate_ships",
    "template_ships",
)


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
    head_mode: str = "candidate_ships"


class Case10Policy(nn.Module):
    """Set Transformer backbone + head-mode-switchable head."""

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

        if self.cfg.head_mode == "candidate_ships":
            self.head_cand_ships = CandidateShipsHead(
                hidden=self.cfg.hidden,
                cand_in_dim=self.cfg.cand_in_dim,
                ships_buckets=self.cfg.ships_buckets,
                head_dropout=self.cfg.head_dropout,
            )
        elif self.cfg.head_mode == "template_ships":
            self.head_template_ships = TemplateShipsHead(
                hidden=self.cfg.hidden,
                attn_heads=self.cfg.attn_heads,
                ships_buckets=self.cfg.ships_buckets,
            )

    def forward(self, batch: BatchFeatures) -> PolicyOutput:
        h_with_ctx, ctx, h = self.backbone(
            batch.planet_feats, batch.planet_mask, batch.global_feats
        )

        if self.cfg.head_mode == "candidate_ships":
            cand_logits, ships_logits = self.head_cand_ships(
                h, ctx, batch.candidate_feats, batch.candidate_mask
            )
            return PolicyOutput(candidate_logits=cand_logits, ships_logits=ships_logits)
        if self.cfg.head_mode == "template_ships":
            target_logits, ships_logits = self.head_template_ships(
                h_with_ctx, h, batch.template_ctx, batch.planet_mask
            )
            return PolicyOutput(target_logits=target_logits, ships_logits=ships_logits)
        raise ValueError(f"unknown head_mode={self.cfg.head_mode!r}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(cfg: ModelConfig | None = None) -> Case10Policy:
    """Convenience constructor used by training and inference."""
    model = Case10Policy(cfg)
    model.eval()
    return model


__all__ = [
    "Case10Policy",
    "MAX_PLANETS",
    "ModelConfig",
    "SUPPORTED_HEAD_MODES",
    "build_model",
    "count_parameters",
]
