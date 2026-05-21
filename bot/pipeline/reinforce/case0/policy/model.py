"""Actor-Critic network for reinforce/case0 PPO baseline.

Architecture (intentionally lightweight, ~50k params):
  - per-planet MLP encoder
  - DeepSets-style mean pool → global context (concat with global_feats)
  - heads operating on (h_planet + ctx):
      * from_logits (B, P)
      * target_logits (B, P_src, P_tgt) via pairwise MLP
      * ships_logits (B, P, K)
      * value (B,) — pooled context → scalar

The three action heads are conditionally independent given state: from-head
samples Bernoulli per my-planet (multi-discrete), then per source we sample
a target and ships bucket. log_probs are summed across the heads.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .featurizer import GLOBAL_FEAT_DIM, PLANET_FEAT_DIM
from .types import BatchFeatures, PolicyOutput

SHIPS_BUCKETS = 4


@dataclass(frozen=True)
class ModelConfig:
    planet_in_dim: int = PLANET_FEAT_DIM
    global_in_dim: int = GLOBAL_FEAT_DIM
    hidden: int = 64
    ships_buckets: int = SHIPS_BUCKETS


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


class ActorCritic(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ModelConfig()
        h = self.cfg.hidden
        self.phi = _mlp(self.cfg.planet_in_dim, h, h)
        self.psi = _mlp(h + self.cfg.global_in_dim, h, h)
        self.from_head = nn.Linear(h + h, 1)
        # pairwise target head: input = [h_src, h_tgt, ctx, dx, dy, dist]
        self.target_head = nn.Sequential(
            nn.Linear(h + h + h + 3, h),
            nn.ReLU(),
            nn.Linear(h, 1),
        )
        self.ships_head = nn.Linear(h + h, self.cfg.ships_buckets)
        self.value_head = nn.Sequential(
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, 1),
        )

    def forward(self, batch: BatchFeatures) -> PolicyOutput:
        x = batch.planet_feats  # (B, P, F)
        mask = batch.planet_mask  # (B, P)
        b, p, _ = x.shape

        h = self.phi(x) * mask.unsqueeze(-1).float()  # (B, P, H)

        m = mask.unsqueeze(-1).float()
        denom = m.sum(dim=1).clamp_min(1.0)
        pooled = (h * m).sum(dim=1) / denom  # (B, H)
        ctx = self.psi(torch.cat([pooled, batch.global_feats], dim=-1))  # (B, H)

        ctx_exp = ctx.unsqueeze(1).expand(-1, p, -1)
        h_ctx = torch.cat([h, ctx_exp], dim=-1)  # (B, P, 2H)

        from_logits = self.from_head(h_ctx).squeeze(-1)
        from_logits = from_logits.masked_fill(~batch.my_planet_mask, float("-inf"))

        x_coord = x[..., 0]
        y_coord = x[..., 1]
        dx = x_coord.unsqueeze(2) - x_coord.unsqueeze(1)
        dy = y_coord.unsqueeze(2) - y_coord.unsqueeze(1)
        dist = torch.sqrt(dx * dx + dy * dy + 1e-8)
        pair_geo = torch.stack([dx, dy, dist], dim=-1)  # (B, P_src, P_tgt, 3)

        h_src = h.unsqueeze(2).expand(-1, -1, p, -1)
        h_tgt = h.unsqueeze(1).expand(-1, p, -1, -1)
        ctx_pair = ctx.unsqueeze(1).unsqueeze(1).expand(-1, p, p, -1)
        target_input = torch.cat([h_src, h_tgt, ctx_pair, pair_geo], dim=-1)
        target_logits = self.target_head(target_input).squeeze(-1)  # (B, P, P)

        eye = torch.eye(p, dtype=torch.bool, device=x.device).unsqueeze(0)
        target_logits = target_logits.masked_fill(eye, float("-inf"))
        target_logits = target_logits.masked_fill(
            ~batch.target_mask.unsqueeze(1), float("-inf")
        )

        ships_logits = self.ships_head(h_ctx)

        value = self.value_head(ctx).squeeze(-1)

        return PolicyOutput(
            from_logits=from_logits,
            target_logits=target_logits,
            ships_logits=ships_logits,
            value=value,
        )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(cfg: ModelConfig | None = None) -> ActorCritic:
    model = ActorCritic(cfg)
    model.eval()
    return model


__all__ = [
    "ActorCritic",
    "ModelConfig",
    "SHIPS_BUCKETS",
    "build_model",
    "count_parameters",
]
