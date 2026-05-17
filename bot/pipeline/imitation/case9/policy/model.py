"""Set Transformer + head-mode-switchable policy for imitation/case9.

config の `head_mode` で 3 variant を切り替え:
  - "three_head":      from + target + ships (case7 流)
  - "candidate":       per-source × CAND_K categorical (case8 流) + SmoothL1 ship_pred
  - "candidate_ships": candidate categorical + 4-bucket ships_logits hybrid
  - "template_ships":  template categorical incl no-op + ships bucket
  - "dual":            3-head + candidate, trained with blended loss

Backbone (Set Transformer ISAB×3 + PMA + global concat) は共通。
Head 実装は `policy/heads/` 配下の別ファイルに分離している。
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from .candidates import CAND_FEAT_DIM, CAND_K
from .featurizer import GLOBAL_FEAT_DIM, MAX_PLANETS, PLANET_FEAT_DIM
from .heads.backbone import BackboneConfig, SetTransformerBackbone
from .heads.candidate import CandidateHead
from .heads.candidate_ships import CandidateShipsHead
from .heads.per_planet import PerPlanetHead
from .heads.template_ships import TemplateShipsHead
from .heads.three_head import ThreeHead
from .types import BatchFeatures, PolicyOutput

SUPPORTED_HEAD_MODES = (
    "three_head",
    "candidate",
    "candidate_ships",
    "template_ships",
    "dual",
    "per_planet",
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
    head_mode: str = "three_head"


class Case9Policy(nn.Module):
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

        if self.cfg.head_mode == "three_head":
            self.head_three = ThreeHead(
                hidden=self.cfg.hidden,
                attn_heads=self.cfg.attn_heads,
                ships_buckets=self.cfg.ships_buckets,
            )
        elif self.cfg.head_mode == "candidate":
            self.head_cand = CandidateHead(
                hidden=self.cfg.hidden,
                cand_in_dim=self.cfg.cand_in_dim,
                head_dropout=self.cfg.head_dropout,
            )
        elif self.cfg.head_mode == "candidate_ships":
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
        elif self.cfg.head_mode == "per_planet":
            self.head_per_planet = PerPlanetHead(
                hidden=self.cfg.hidden,
                head_dropout=self.cfg.head_dropout,
            )
        else:  # dual
            self.head_three = ThreeHead(
                hidden=self.cfg.hidden,
                attn_heads=self.cfg.attn_heads,
                ships_buckets=self.cfg.ships_buckets,
            )
            self.head_cand = CandidateHead(
                hidden=self.cfg.hidden,
                cand_in_dim=self.cfg.cand_in_dim,
                head_dropout=self.cfg.head_dropout,
            )

    def forward(self, batch: BatchFeatures) -> PolicyOutput:
        h_with_ctx, ctx, h = self.backbone(
            batch.planet_feats, batch.planet_mask, batch.global_feats
        )

        if self.cfg.head_mode == "three_head":
            from_logits, target_logits, ships_logits = self.head_three(
                h_with_ctx,
                ctx,
                h,
                batch.template_ctx,
                batch.my_planet_mask,
                batch.planet_mask,
            )
            return PolicyOutput(
                from_logits=from_logits,
                target_logits=target_logits,
                ships_logits=ships_logits,
            )
        if self.cfg.head_mode == "candidate":
            cand_logits, ship_pred = self.head_cand(
                h, ctx, batch.candidate_feats, batch.candidate_mask
            )
            return PolicyOutput(candidate_logits=cand_logits, ship_pred=ship_pred)
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
        if self.cfg.head_mode == "per_planet":
            per_planet_logits, ship_pred = self.head_per_planet(
                h_with_ctx, ctx, h, batch.planet_mask
            )
            return PolicyOutput(
                per_planet_logits=per_planet_logits, ship_pred=ship_pred
            )

        from_logits, target_logits, ships_logits = self.head_three(
            h_with_ctx,
            ctx,
            h,
            batch.template_ctx,
            batch.my_planet_mask,
            batch.planet_mask,
        )
        cand_logits, ship_pred = self.head_cand(
            h, ctx, batch.candidate_feats, batch.candidate_mask
        )
        return PolicyOutput(
            from_logits=from_logits,
            target_logits=target_logits,
            ships_logits=ships_logits,
            candidate_logits=cand_logits,
            ship_pred=ship_pred,
        )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(cfg: ModelConfig | None = None) -> Case9Policy:
    """Convenience constructor used by training and inference."""
    model = Case9Policy(cfg)
    model.eval()
    return model


__all__ = [
    "Case9Policy",
    "MAX_PLANETS",
    "ModelConfig",
    "SUPPORTED_HEAD_MODES",
    "build_model",
    "count_parameters",
]
