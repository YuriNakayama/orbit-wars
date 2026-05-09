"""Set Transformer backbone shared by all 3 head variants of case10.

Encoder = ISAB×3 (m=16) + PMA bottleneck + global concat (case7 と同等)。
Output:
  h_with_ctx: (B, P, 2H) — per-node encoded feature concatenated with global ctx
  ctx:        (B, H)     — global context vector (used by ships head etc.)

case7 model.py からのコピー (cross-case import 禁止のため)。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..featurizer import GLOBAL_FEAT_DIM, PLANET_FEAT_DIM


@dataclass(frozen=True)
class BackboneConfig:
    planet_in_dim: int = PLANET_FEAT_DIM
    global_in_dim: int = GLOBAL_FEAT_DIM
    hidden: int = 128
    attn_heads: int = 4
    inducing_points: int = 16
    encoder_layers: int = 3


class MultiheadAttentionBlock(nn.Module):
    """MAB: Multihead Attention Block."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.mha = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attn_out, _ = self.mha(
            x, y, y, key_padding_mask=key_padding_mask, need_weights=False
        )
        h = self.ln1(x + attn_out)
        out: torch.Tensor = self.ln2(h + self.ff(h))
        return out


class InducedSetAttentionBlock(nn.Module):
    """ISAB: Induced Set Attention Block."""

    def __init__(self, dim: int, num_heads: int, num_inducing: int) -> None:
        super().__init__()
        self.inducing = nn.Parameter(torch.randn(1, num_inducing, dim) * 0.02)
        self.mab1 = MultiheadAttentionBlock(dim, num_heads)
        self.mab2 = MultiheadAttentionBlock(dim, num_heads)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        i = self.inducing.expand(b, -1, -1)
        kp_mask = ~mask
        h = self.mab1(i, x, key_padding_mask=kp_mask)
        out = self.mab2(x, h, key_padding_mask=None)
        result: torch.Tensor = out * mask.unsqueeze(-1).float()
        return result


class PMA(nn.Module):
    """Pooling by Multihead Attention."""

    def __init__(self, dim: int, num_heads: int, num_seeds: int) -> None:
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(1, num_seeds, dim) * 0.02)
        self.mab = MultiheadAttentionBlock(dim, num_heads)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        s = self.seeds.expand(b, -1, -1)
        kp_mask = ~mask
        out: torch.Tensor = self.mab(s, x, key_padding_mask=kp_mask)
        return out


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


class SetTransformerBackbone(nn.Module):
    """Encoder + Bottleneck shared by every head variant in case10."""

    def __init__(self, cfg: BackboneConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or BackboneConfig()
        h = self.cfg.hidden
        heads = self.cfg.attn_heads
        self.in_proj = nn.Linear(self.cfg.planet_in_dim, h)
        self.encoder = nn.ModuleList(
            [
                InducedSetAttentionBlock(h, heads, self.cfg.inducing_points)
                for _ in range(self.cfg.encoder_layers)
            ]
        )
        self.pool = PMA(h, heads, num_seeds=1)
        self.psi = _mlp(h + self.cfg.global_in_dim, h, h)

    def forward(
        self,
        planet_feats: torch.Tensor,
        planet_mask: torch.Tensor,
        global_feats: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (h_with_ctx (B,P,2H), ctx (B,H), h (B,P,H))."""
        h0 = self.in_proj(planet_feats) * planet_mask.unsqueeze(-1).float()
        h = h0
        for block in self.encoder:
            h = block(h, planet_mask)
        pooled = self.pool(h, planet_mask).squeeze(1)
        ctx = self.psi(torch.cat([pooled, global_feats], dim=-1))
        ctx_exp = ctx.unsqueeze(1).expand(-1, h.shape[1], -1)
        h_with_ctx = torch.cat([h, ctx_exp], dim=-1)
        return h_with_ctx, ctx, h
