"""Candidate head variant: per-source × CAND_K categorical (case8 流).

Inputs from backbone:
  h:        (B, P, H)        — encoded per-node feature (without ctx concat)
  ctx:      (B, H)            — global context vector
  candidate_feats: (B, P, K, CAND_FEAT_DIM)
  candidate_mask:  (B, P, K) bool

Outputs:
  candidate_logits: (B, P, K) — slot 0 reserved as no-op
  ship_pred:        (B, P)    — case8-style continuous ship-count regression
"""

from __future__ import annotations

import torch
from torch import nn

from ..candidates import CAND_FEAT_DIM


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


class CandidateHead(nn.Module):
    def __init__(
        self,
        hidden: int,
        cand_in_dim: int = CAND_FEAT_DIM,
        head_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        h = hidden
        self.self_proj = _mlp(h, h, h)
        self.global_proj = _mlp(h, h, h)
        self.cand_encoder = _mlp(cand_in_dim, h, h)
        p = float(head_dropout)
        self.cand_score = nn.Sequential(
            nn.Linear(h * 3, h),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(h, 1),
        )
        self.ship_head = nn.Sequential(
            nn.Linear(h * 2, h),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(h, 1),
        )

    def forward(
        self,
        h: torch.Tensor,
        ctx: torch.Tensor,
        candidate_feats: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, p, _ = h.shape
        self_h = self.self_proj(h)  # (B, P, H)
        global_h = self.global_proj(ctx)  # (B, H)
        cand_h = self.cand_encoder(candidate_feats)  # (B, P, K, H)
        k = cand_h.shape[2]
        self_exp = self_h.unsqueeze(2).expand(-1, -1, k, -1)
        global_exp = global_h.unsqueeze(1).unsqueeze(2).expand(-1, p, k, -1)
        joint = torch.cat([self_exp, global_exp, cand_h], dim=-1)
        cand_logits = self.cand_score(joint).squeeze(-1)
        cand_logits = cand_logits.masked_fill(~candidate_mask, -1e9)

        ship_in = torch.cat([self_h, global_h.unsqueeze(1).expand(-1, p, -1)], dim=-1)
        ship_pred = self.ship_head(ship_in).squeeze(-1)
        return cand_logits, ship_pred
