"""candidate + ships hybrid head variant.

Same candidate scoring path as `candidate.py`, but the per-source ship-count
prediction is a learned 4-bucket categorical (case7-style ships_head) instead
of the SmoothL1 regression `ship_pred` used by `candidate.py`.

Inputs are identical to `CandidateHead`. Output:
  candidate_logits: (B, P, CAND_K)
  ships_logits:     (B, P, ships_buckets)
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


class CandidateShipsHead(nn.Module):
    def __init__(
        self,
        hidden: int,
        cand_in_dim: int = CAND_FEAT_DIM,
        ships_buckets: int = 4,
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
        self.ships_head = nn.Sequential(
            nn.Linear(h * 2, h),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(h, ships_buckets),
        )

    def forward(
        self,
        h: torch.Tensor,
        ctx: torch.Tensor,
        candidate_feats: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, p, _ = h.shape
        self_h = self.self_proj(h)
        global_h = self.global_proj(ctx)
        cand_h = self.cand_encoder(candidate_feats)
        k = cand_h.shape[2]
        self_exp = self_h.unsqueeze(2).expand(-1, -1, k, -1)
        global_exp = global_h.unsqueeze(1).unsqueeze(2).expand(-1, p, k, -1)
        joint = torch.cat([self_exp, global_exp, cand_h], dim=-1)
        cand_logits = self.cand_score(joint).squeeze(-1)
        cand_logits = cand_logits.masked_fill(~candidate_mask, -1e9)

        ship_in = torch.cat([self_h, global_h.unsqueeze(1).expand(-1, p, -1)], dim=-1)
        ships_logits = self.ships_head(ship_in)
        return cand_logits, ships_logits
