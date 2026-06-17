"""per_planet head: per-source × (P+1) target categorical + ships regression.

仮説 H14 系の精緻化方針: target を `planet_id` 直接予測に戻し、no-op を専用
sentinel として 1 slot 追加した (P+1) クラス categorical で表現する。
ships は log1p(ships) の連続値 regression (SmoothL1) で量を直接学習する。

Inputs from backbone:
  h_with_ctx: (B, P, 2H)
  ctx:        (B, H)
  h:          (B, P, H)
  planet_mask: (B, P) bool — True for valid planet slots (used to mask invalid
               targets in the per-source softmax)

Outputs:
  per_planet_logits: (B, P, P + 1) — last index is the no-op sentinel
  ship_pred:         (B, P) — log1p(ships) regression scalar per source

設計詳細:
- src→tgt スコアは pointer attention 流: query = src 特徴 + ctx, key = tgt 特徴 + ctx。
  bilinear (Q · W · K^T) で per-source × per-target スコア行列を構築。
- 学習可能な no-op sentinel embedding を 1 つ持ち、各 src の (P+1) 番目 logit を
  per-src query との内積で生成する。
- 無効 target (planet_mask=False) の logit は -1e9 でマスク。no-op slot は常に有効。
- ships head は ctx + per-source 特徴の MLP 単一スカラ出力。

This head intentionally has high capacity: pointer attention の per-pair logit を
直接学習するため、planet 数が増えるほどパラメータと計算量は線形に増える。データ量
増加が前提のため敢えて精緻化方針を採る。
"""

from __future__ import annotations

import torch
from torch import nn


class PerPlanetHead(nn.Module):
    """Pointer-style per-source × per-target head + ships regression."""

    def __init__(
        self,
        hidden: int,
        head_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        h = hidden
        p = float(head_dropout)
        # src/tgt projections share the same hidden width so we can compute a
        # symmetric bilinear score (Q W K^T). W is a learnable square matrix
        # (`pair_proj`) acting in the projected space.
        self.src_proj = nn.Sequential(
            nn.Linear(h * 2, h),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(h, h),
        )
        self.tgt_proj = nn.Sequential(
            nn.Linear(h * 2, h),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(h, h),
        )
        self.pair_proj = nn.Linear(h, h, bias=False)
        # Learnable no-op token: 1 query-shaped embedding scored against each
        # source's projected query for the (P+1)-th logit.
        self.noop_query = nn.Parameter(torch.randn(1, 1, h) * 0.02)
        # Ships regression: source-specific scalar predictor on (h + ctx).
        self.ship_head = nn.Sequential(
            nn.Linear(h * 2, h),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(h, 1),
        )

    def forward(
        self,
        h_with_ctx: torch.Tensor,
        ctx: torch.Tensor,
        h: torch.Tensor,
        planet_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del ctx, h  # currently using h_with_ctx for both src and tgt projections
        # src_q, tgt_k: (B, P, H)
        src_q = self.src_proj(h_with_ctx)
        tgt_k = self.tgt_proj(h_with_ctx)
        tgt_kp = self.pair_proj(tgt_k)  # (B, P, H)
        # Pairwise bilinear score: (B, P_src, H) @ (B, H, P_tgt) -> (B, P_src, P_tgt)
        pair_logits = torch.bmm(src_q, tgt_kp.transpose(1, 2))
        # No-op logit: (B, P_src, 1) = src_q · noop_query
        noop_q = self.noop_query.expand(src_q.shape[0], -1, -1)  # (B, 1, H)
        noop_logits = torch.bmm(src_q, noop_q.transpose(1, 2))  # (B, P, 1)
        # Mask invalid targets (planet_mask=False); no-op slot stays valid.
        mask = planet_mask.unsqueeze(1).expand_as(pair_logits)  # (B, P_src, P_tgt)
        pair_logits = pair_logits.masked_fill(~mask, -1e9)
        per_planet_logits = torch.cat([pair_logits, noop_logits], dim=-1)
        # ships regression
        ship_pred = self.ship_head(h_with_ctx).squeeze(-1)
        return per_planet_logits, ship_pred
