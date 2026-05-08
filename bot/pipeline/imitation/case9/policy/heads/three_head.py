"""3-head variant: from + target + ships (case7 流).

Inputs from backbone:
  h_with_ctx: (B, P, 2H)
  ctx:        (B, H)
  h:          (B, P, H)
  template_ctx: (B, P, TEMPLATE_CTX_DIM)
  my_planet_mask: (B, P)
  planet_mask: (B, P)

Outputs:
  from_logits:    (B, P)
  target_logits:  (B, P, NUM_TEMPLATES)
  ships_logits:   (B, P, ships_buckets)
"""

from __future__ import annotations

import torch
from torch import nn

from ..templates import NUM_TEMPLATES, TEMPLATE_CTX_DIM
from .backbone import MultiheadAttentionBlock


class ThreeHead(nn.Module):
    def __init__(self, hidden: int, attn_heads: int, ships_buckets: int) -> None:
        super().__init__()
        h = hidden
        self.from_head = nn.Linear(h + h, 1)
        self.template_query = nn.Parameter(torch.randn(1, NUM_TEMPLATES, h) * 0.02)
        self.target_attn = MultiheadAttentionBlock(h, attn_heads)
        self.target_source_proj = nn.Linear(h + h + TEMPLATE_CTX_DIM, h)
        self.target_score = nn.Linear(h, 1)
        self.ships_head = nn.Linear(h + h, ships_buckets)

    def forward(
        self,
        h_with_ctx: torch.Tensor,
        ctx: torch.Tensor,
        h: torch.Tensor,
        template_ctx: torch.Tensor,
        my_planet_mask: torch.Tensor,
        planet_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, p, _ = h.shape

        from_logits = self.from_head(h_with_ctx).squeeze(-1)
        from_logits = from_logits.masked_fill(~my_planet_mask, float("-inf"))

        h_with_tctx = torch.cat([h_with_ctx, template_ctx], dim=-1)
        src_proj = self.target_source_proj(h_with_tctx)
        tq = self.template_query.expand(b, -1, -1)
        per_src_tq = src_proj.unsqueeze(2) + tq.unsqueeze(1)
        flat_q = per_src_tq.reshape(b, p * NUM_TEMPLATES, -1)
        attn_out = self.target_attn(flat_q, h, key_padding_mask=~planet_mask)
        attn_out = attn_out.reshape(b, p, NUM_TEMPLATES, -1)
        target_logits = self.target_score(attn_out).squeeze(-1)

        ships_logits = self.ships_head(h_with_ctx)

        return from_logits, target_logits, ships_logits
