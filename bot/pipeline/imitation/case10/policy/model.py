"""Set Transformer policy network for imitation/case7 IL baseline.

Replaces case6's Graph Attention U-Net with a Set Transformer (Lee et al.
2019) encoder + PMA-style cross-attention target head:

  - Encoder: 3 stacked ISAB (Induced Set Attention Block) over the planet set.
    ISAB uses learnable inducing points (m=16) to reduce O(P^2) attention to
    O(P*m), and is permutation-equivariant by construction. No graph / kNN /
    edge features are required — the model relies on attention over feature
    similarity instead.
  - Bottleneck: PMA (Pooling by Multihead Attention) with k=1 query + global
    feature concat for the per-batch context vector.
  - From head: per-source linear (case6-compatible).
  - Target head: PMA-style cross-attention with NUM_TEMPLATES learnable
    template queries, attending to the planet set conditioned on the source
    node. Output shape: (B, P, NUM_TEMPLATES) — same as case6.
  - Ships head: per-source linear (case6-compatible).

Class is aliased as `DeepSetsPolicy` so existing agent.py and weight loaders
continue to work after re-training.

References:
  Lee et al. 2019, "Set Transformer: A Framework for Attention-based
  Permutation-Invariant Neural Networks", ICML. arXiv:1810.00825.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .featurizer import GLOBAL_FEAT_DIM, MAX_PLANETS, PLANET_FEAT_DIM
from .templates import NUM_TEMPLATES, TEMPLATE_CTX_DIM
from .types import BatchFeatures, PolicyOutput

# featurizer column layout (PLANET_FEAT_DIM == 17 in case7: case6 と同一)
COL_X = 0
COL_Y = 1


@dataclass(frozen=True)
class ModelConfig:
    planet_in_dim: int = PLANET_FEAT_DIM
    global_in_dim: int = GLOBAL_FEAT_DIM
    hidden: int = 128
    ships_buckets: int = 4
    attn_heads: int = 4
    inducing_points: int = 16  # ISAB の inducing points 数 m
    encoder_layers: int = 3  # ISAB block 数


class MultiheadAttentionBlock(nn.Module):
    """MAB: Multihead Attention Block (Set Transformer の基本ブロック).

    MAB(X, Y) = LayerNorm(H + FF(H)), H = LayerNorm(X + MHA(X, Y, Y))
    """

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
        x: torch.Tensor,  # (B, Nx, D) query
        y: torch.Tensor,  # (B, Ny, D) key/value
        key_padding_mask: torch.Tensor | None = None,  # (B, Ny) bool, True=mask out
    ) -> torch.Tensor:
        # nn.MultiheadAttention の key_padding_mask は True で mask out
        attn_out, _ = self.mha(
            x, y, y, key_padding_mask=key_padding_mask, need_weights=False
        )
        h = self.ln1(x + attn_out)
        out: torch.Tensor = self.ln2(h + self.ff(h))
        return out


class InducedSetAttentionBlock(nn.Module):
    """ISAB: Induced Set Attention Block (Lee 2019).

    ISAB(X) = MAB(X, MAB(I, X)), I = learnable inducing points (m, D).
    O(N*m) instead of O(N^2). Permutation-equivariant.
    """

    def __init__(self, dim: int, num_heads: int, num_inducing: int) -> None:
        super().__init__()
        self.inducing = nn.Parameter(torch.randn(1, num_inducing, dim) * 0.02)
        self.mab1 = MultiheadAttentionBlock(dim, num_heads)
        self.mab2 = MultiheadAttentionBlock(dim, num_heads)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, P, D), mask: (B, P) bool, True=valid
        b = x.shape[0]
        i = self.inducing.expand(b, -1, -1)  # (B, m, D)
        kp_mask = ~mask  # nn.MultiheadAttention: True=mask out
        # MAB(I, X): inducing points が X に attend
        h = self.mab1(i, x, key_padding_mask=kp_mask)  # (B, m, D)
        # MAB(X, H): X が pooled inducing に attend
        out = self.mab2(x, h, key_padding_mask=None)  # (B, P, D), no mask on inducing
        # masked rows をゼロ化 (downstream で使われない保証)
        result: torch.Tensor = out * mask.unsqueeze(-1).float()
        return result


class PMA(nn.Module):
    """Pooling by Multihead Attention. Set → fixed-size summary."""

    def __init__(self, dim: int, num_heads: int, num_seeds: int) -> None:
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(1, num_seeds, dim) * 0.02)
        self.mab = MultiheadAttentionBlock(dim, num_heads)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        s = self.seeds.expand(b, -1, -1)
        kp_mask = ~mask
        out: torch.Tensor = self.mab(s, x, key_padding_mask=kp_mask)
        return out  # (B, num_seeds, D)


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


class SetTransformerPolicy(nn.Module):
    """Set Transformer encoder + PMA cross-attention target head.

    encoder/decoder/from_head/ships_head は case6 と等価な役割を持つが、
    全て attention ベース (graph/kNN/edge feature を使わない)。
    """

    def __init__(self, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ModelConfig()
        h = self.cfg.hidden
        heads = self.cfg.attn_heads

        # Input projection
        self.in_proj = nn.Linear(self.cfg.planet_in_dim, h)
        # Encoder: ISAB stack
        self.encoder = nn.ModuleList(
            [
                InducedSetAttentionBlock(h, heads, self.cfg.inducing_points)
                for _ in range(self.cfg.encoder_layers)
            ]
        )
        # Bottleneck: PMA k=1 + global feats
        self.pool = PMA(h, heads, num_seeds=1)
        self.psi = _mlp(h + self.cfg.global_in_dim, h, h)
        # From head: per-node binary
        self.from_head = nn.Linear(h + h, 1)
        # Target head: NUM_TEMPLATES learnable queries × cross-attention to planet set,
        #   conditioned on per-source planet feature + global ctx.
        # Implementation: per-source の (h_node, ctx) を query 拡張、template embeddings
        # を learnable bias、planet set を K/V とする cross-attention で
        # (B, P, NUM_TEMPLATES) ロジット を計算。
        self.template_query = nn.Parameter(torch.randn(1, NUM_TEMPLATES, h) * 0.02)
        self.target_attn = MultiheadAttentionBlock(h, heads)
        # Source node + ctx を template query に加算するための projection
        self.target_source_proj = nn.Linear(h + h + TEMPLATE_CTX_DIM, h)
        self.target_score = nn.Linear(h, 1)
        # Ships head: per-source linear
        self.ships_head = nn.Linear(h + h, self.cfg.ships_buckets)

    def forward(self, batch: BatchFeatures) -> PolicyOutput:
        x = batch.planet_feats  # (B, P, F)
        mask = batch.planet_mask  # (B, P) bool
        b, p, _ = x.shape

        # Input projection + masked
        h0 = self.in_proj(x) * mask.unsqueeze(-1).float()  # (B, P, H)

        # Encoder: ISAB stack
        h = h0
        for block in self.encoder:
            h = block(h, mask)  # (B, P, H)

        # Bottleneck: PMA pool + global feats
        pooled = self.pool(h, mask).squeeze(1)  # (B, H)
        ctx = self.psi(torch.cat([pooled, batch.global_feats], dim=-1))  # (B, H)

        # Per-node features with ctx
        ctx_exp = ctx.unsqueeze(1).expand(-1, p, -1)  # (B, P, H)
        h_with_ctx = torch.cat([h, ctx_exp], dim=-1)  # (B, P, 2H)

        # From head
        from_logits = self.from_head(h_with_ctx).squeeze(-1)  # (B, P)
        from_logits = from_logits.masked_fill(~batch.my_planet_mask, float("-inf"))

        # Target head (PMA-style cross-attention with template queries):
        #   per source planet i に対して、 template j (j=1..NUM_TEMPLATES) ごとに
        #   logit を計算する。
        #   1. q_ij = template_query[j] + W_s [h_i, ctx, tctx_ij]
        #   2. q_ij が planet set h に attend → 出力 (B, P, NUM_TEMPLATES, H)
        #   3. linear で score 化 → (B, P, NUM_TEMPLATES) logit
        # 計算量を抑えるため、 batched で全 source × all templates をまとめて処理。
        # Shape は (B, P*NUM_TEMPLATES, H) を query として MAB に通す。
        h_with_tctx = torch.cat(
            [h_with_ctx, batch.template_ctx], dim=-1
        )  # (B, P, 2H+TC)
        src_proj = self.target_source_proj(h_with_tctx)  # (B, P, H)
        # template query を per-source に broadcast
        tq = self.template_query.expand(b, -1, -1)  # (B, NUM_TEMPLATES, H)
        # (B, P, NUM_TEMPLATES, H): 各 source × 各 template の query
        per_src_tq = src_proj.unsqueeze(2) + tq.unsqueeze(1)  # (B, P, NUM_TEMPLATES, H)
        # flatten to (B, P*NUM_TEMPLATES, H) for batched cross-attention
        flat_q = per_src_tq.reshape(b, p * NUM_TEMPLATES, -1)
        # cross-attention to planet set (h, mask は P planets)
        attn_out = self.target_attn(flat_q, h, key_padding_mask=~mask)  # (B, P*NT, H)
        # reshape back + score
        attn_out = attn_out.reshape(b, p, NUM_TEMPLATES, -1)  # (B, P, NT, H)
        target_logits = self.target_score(attn_out).squeeze(-1)  # (B, P, NT)

        # Ships head
        ships_logits = self.ships_head(h_with_ctx)  # (B, P, ships_buckets)

        return PolicyOutput(
            from_logits=from_logits,
            target_logits=target_logits,
            ships_logits=ships_logits,
        )


# Backwards-compatible aliases
DeepSetsPolicy = SetTransformerPolicy
GraphUNetPolicy = SetTransformerPolicy  # case5 name compatibility
GraphAttentionUNetPolicy = SetTransformerPolicy  # case6 name compatibility


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(cfg: ModelConfig | None = None) -> SetTransformerPolicy:
    """Convenience constructor used by training and inference."""
    model = SetTransformerPolicy(cfg)
    model.eval()
    return model


__all__ = [
    "DeepSetsPolicy",
    "GraphAttentionUNetPolicy",
    "GraphUNetPolicy",
    "InducedSetAttentionBlock",
    "MultiheadAttentionBlock",
    "MAX_PLANETS",
    "ModelConfig",
    "PMA",
    "SetTransformerPolicy",
    "build_model",
    "count_parameters",
]
