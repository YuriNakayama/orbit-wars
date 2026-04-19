"""DeepSets policy network for imitation/case1 IL baseline.

3 heads:
  - from: per-planet binary "send fleet from this planet" logit (B, MAX_PLANETS)
  - target: per-source × per-target logits (B, MAX_PLANETS, MAX_PLANETS+1)
           last column = no-op (don't fire from this src)
  - ships: per-source ships-bucket logits (B, MAX_PLANETS, ships_buckets)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .featurizer import GLOBAL_FEAT_DIM, MAX_PLANETS, PLANET_FEAT_DIM
from .types import BatchFeatures, PolicyOutput


@dataclass(frozen=True)
class ModelConfig:
    planet_in_dim: int = PLANET_FEAT_DIM
    global_in_dim: int = GLOBAL_FEAT_DIM
    hidden: int = 64
    ships_buckets: int = 5


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


class DeepSetsPolicy(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ModelConfig()
        h = self.cfg.hidden
        self.phi = _mlp(self.cfg.planet_in_dim, h, h)
        self.psi = _mlp(h + self.cfg.global_in_dim, h, h)
        # per-planet conditioning: phi(planet) ++ context
        self.from_head = nn.Linear(h + h, 1)
        # for target head: pairwise (src_phi, tgt_phi, context) → 1 logit
        self.target_pair = nn.Sequential(
            nn.Linear(h + h + h, h),
            nn.ReLU(),
            nn.Linear(h, 1),
        )
        # per-src ships head: (src_phi, context) → ships_buckets
        self.ships_head = nn.Linear(h + h, self.cfg.ships_buckets)
        # learned no-op vector for the (MAX_PLANETS+1)-th target slot
        self.noop_logit = nn.Parameter(torch.zeros(1))

    def forward(self, batch: BatchFeatures) -> PolicyOutput:
        x = batch.planet_feats  # (B, P, F)
        mask = batch.planet_mask  # (B, P)
        b, p, _ = x.shape
        h = self.phi(x)  # (B, P, H)

        # masked mean pool
        m = mask.unsqueeze(-1).float()  # (B, P, 1)
        denom = m.sum(dim=1).clamp_min(1.0)  # (B, 1)
        pooled = (h * m).sum(dim=1) / denom  # (B, H)
        ctx = self.psi(torch.cat([pooled, batch.global_feats], dim=-1))  # (B, H)

        ctx_exp = ctx.unsqueeze(1).expand(-1, p, -1)  # (B, P, H)
        h_with_ctx = torch.cat([h, ctx_exp], dim=-1)  # (B, P, 2H)

        from_logits = self.from_head(h_with_ctx).squeeze(-1)  # (B, P)
        # mask: only my planets can be "from". non-my → -inf.
        from_logits = from_logits.masked_fill(~batch.my_planet_mask, float("-inf"))

        # target head: for each src P_i × tgt P_j build (h_i, h_j, ctx)
        h_src = h.unsqueeze(2).expand(-1, -1, p, -1)  # (B, P, P, H)
        h_tgt = h.unsqueeze(1).expand(-1, p, -1, -1)  # (B, P, P, H)
        ctx_pair = ctx.unsqueeze(1).unsqueeze(2).expand(-1, p, p, -1)  # (B, P, P, H)
        pair_in = torch.cat([h_src, h_tgt, ctx_pair], dim=-1)  # (B, P, P, 3H)
        tgt_logits_planets = self.target_pair(pair_in).squeeze(-1)  # (B, P, P)
        # Target candidates = any valid (non-padding) planet. Reinforcing an
        # owned planet is legal. Only mask padding slots.
        pad_mask = batch.planet_mask.unsqueeze(1).expand(-1, p, -1)  # (B, P, P)
        tgt_logits_planets = tgt_logits_planets.masked_fill(~pad_mask, float("-inf"))
        # append no-op logit (broadcast learned scalar) as the last column
        noop = self.noop_logit.expand(b, p, 1)  # (B, P, 1)
        target_logits = torch.cat([tgt_logits_planets, noop], dim=-1)  # (B, P, P+1)

        ships_logits = self.ships_head(h_with_ctx)  # (B, P, ships_buckets)

        return PolicyOutput(
            from_logits=from_logits,
            target_logits=target_logits,
            ships_logits=ships_logits,
        )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(cfg: ModelConfig | None = None) -> DeepSetsPolicy:
    """Convenience constructor used by training and inference."""
    model = DeepSetsPolicy(cfg)
    model.eval()
    return model


__all__ = [
    "DeepSetsPolicy",
    "ModelConfig",
    "MAX_PLANETS",
    "build_model",
    "count_parameters",
]
