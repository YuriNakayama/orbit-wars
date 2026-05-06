"""Graph Attention U-Net policy network for imitation/case6 IL baseline.

Variant of case5 backbone where the GraphConv (uniform-mean aggregation) is
replaced by a Graph Attention layer that consumes pairwise edge features:

  - kNN graph built per-frame from (x, y) coords (k=8, masked padding).
  - Encoder: 3 levels of GraphAttention → TopK pooling (P → P*2//3 → P//3).
  - Bottleneck: GraphAttention + global mean pool → context.
  - Decoder: 2 levels of unpool (scatter back) + skip-add + GraphAttention.
  - 3 heads identical to the previous model: from / target / ships.
  - `_pairwise_geometry` (dx, dy, dist, ship_log_diff, tgt_is_enemy,
    tgt_is_neutral) is computed once at the input level and gathered through
    pooling levels so each attention layer sees physically meaningful edges.

The class is aliased as `DeepSetsPolicy` so existing imports continue to load.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .featurizer import GLOBAL_FEAT_DIM, MAX_PLANETS, PLANET_FEAT_DIM
from .templates import NUM_TEMPLATES, TEMPLATE_CTX_DIM
from .types import BatchFeatures, PolicyOutput

# featurizer column layout (PLANET_FEAT_DIM == 17 in case6: 11 base + 6 ship-prediction)
COL_X = 0
COL_Y = 1
COL_SHIPS = 3
COL_IS_MINE = 5
COL_IS_ENEMY = 6
COL_IS_NEUTRAL = 7

PAIR_FEAT_DIM = 6  # dx, dy, dist, ship_log_diff, tgt_is_enemy, tgt_is_neutral
KNN_K = 8


@dataclass(frozen=True)
class ModelConfig:
    planet_in_dim: int = PLANET_FEAT_DIM
    global_in_dim: int = GLOBAL_FEAT_DIM
    hidden: int = 128
    ships_buckets: int = 4
    attn_heads: int = 4


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


def _pairwise_geometry(planet_feats: torch.Tensor) -> torch.Tensor:
    """Build (B, P, P, PAIR_FEAT_DIM) pairwise geometry, src=dim1, tgt=dim2."""
    x = planet_feats[..., COL_X]
    y = planet_feats[..., COL_Y]
    ships_log = planet_feats[..., COL_SHIPS]
    is_enemy = planet_feats[..., COL_IS_ENEMY]
    is_neutral = planet_feats[..., COL_IS_NEUTRAL]

    dx = x.unsqueeze(1) - x.unsqueeze(2)
    dy = y.unsqueeze(1) - y.unsqueeze(2)
    dist = torch.sqrt(dx * dx + dy * dy + 1e-8)
    ship_log_diff = ships_log.unsqueeze(1) - ships_log.unsqueeze(2)
    tgt_enemy = is_enemy.unsqueeze(1).expand_as(dist)
    tgt_neutral = is_neutral.unsqueeze(1).expand_as(dist)
    return torch.stack([dx, dy, dist, ship_log_diff, tgt_enemy, tgt_neutral], dim=-1)


def _knn_adjacency(
    coords: torch.Tensor,  # (B, P, 2)
    mask: torch.Tensor,  # (B, P) bool
    k: int,
) -> torch.Tensor:
    """Symmetric kNN adjacency from (x, y). Padding rows are isolated."""
    b, p, _ = coords.shape
    dx = coords.unsqueeze(2) - coords.unsqueeze(1)  # (B, P, P, 2)
    dist = (dx * dx).sum(-1)  # (B, P, P)
    big = torch.full_like(dist, float("inf"))
    valid_pair = mask.unsqueeze(1) & mask.unsqueeze(2)  # (B, P, P)
    dist = torch.where(valid_pair, dist, big)
    eye = torch.eye(p, dtype=torch.bool, device=coords.device).unsqueeze(0)
    dist = dist.masked_fill(eye, float("inf"))  # exclude self
    kk = min(k, p - 1)
    _, idx = dist.topk(kk, dim=-1, largest=False)  # (B, P, k)
    adj = torch.zeros(b, p, p, dtype=torch.bool, device=coords.device)
    adj.scatter_(2, idx, True)
    valid_neighbour = torch.isfinite(dist.gather(2, idx))
    adj.scatter_(2, idx, valid_neighbour)
    adj = adj | adj.transpose(1, 2)  # symmetric
    adj = adj & valid_pair
    return adj


def _gather_pair_feats(pair_feats: torch.Tensor, top_idx: torch.Tensor) -> torch.Tensor:
    """Gather (B, P, P, F) pairwise features down to (B, K, K, F) using TopK indices."""
    b, p, _, f = pair_feats.shape
    k = top_idx.shape[1]
    # rows: select K source nodes
    rows = pair_feats.gather(
        1, top_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, p, f)
    )  # (B, K, P, F)
    # cols: select K target nodes
    out = rows.gather(
        2, top_idx.unsqueeze(1).unsqueeze(-1).expand(-1, k, -1, f)
    )  # (B, K, K, F)
    return out


class GraphAttention(nn.Module):
    """Multi-head graph attention with edge features.

    For each kept neighbour j of node i, score:

        e_ij = LeakyReLU(a · [W_q h_i || W_k h_j || W_e e_ij])

    softmax over j ∈ N(i), aggregate value vectors, concat heads, project.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 4,
        edge_dim: int = PAIR_FEAT_DIM,
    ) -> None:
        super().__init__()
        if out_dim % num_heads != 0:
            raise ValueError(
                f"out_dim={out_dim} must be divisible by num_heads={num_heads}"
            )
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.q_proj = nn.Linear(in_dim, out_dim)
        self.k_proj = nn.Linear(in_dim, out_dim)
        self.v_proj = nn.Linear(in_dim, out_dim)
        self.edge_proj = nn.Linear(edge_dim, num_heads)  # one scalar per head
        self.out_proj = nn.Linear(out_dim, out_dim)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(
        self,
        h: torch.Tensor,  # (B, P, F_in)
        adj: torch.Tensor,  # (B, P, P) bool
        mask: torch.Tensor,  # (B, P) bool
        edge_feats: torch.Tensor,  # (B, P, P, edge_dim)
    ) -> torch.Tensor:
        b, p, _ = h.shape
        H = self.num_heads
        D = self.head_dim

        q = self.q_proj(h).view(b, p, H, D)  # (B, P, H, D)
        k = self.k_proj(h).view(b, p, H, D)
        v = self.v_proj(h).view(b, p, H, D)

        # Pairwise dot-product attention per head
        # scores[b, i, j, h] = q[b,i,h,:] · k[b,j,h,:] / sqrt(D)
        scores = torch.einsum("biHd,bjHd->bijH", q, k) / (D**0.5)  # (B, P, P, H)
        # Add edge feature contribution per head
        edge_bias = self.edge_proj(edge_feats)  # (B, P, P, H)
        scores = self.leaky(scores + edge_bias)

        # Mask: keep only edges in adj AND target node valid
        valid_edge = adj & mask.unsqueeze(1)  # (B, P, P)
        scores = scores.masked_fill(~valid_edge.unsqueeze(-1), float("-inf"))

        # If a row has no valid neighbours (isolated padding), softmax of all -inf
        # yields NaN. Replace such rows with uniform 0 weights — value sum is 0 anyway.
        any_valid = valid_edge.any(dim=-1, keepdim=True)  # (B, P, 1)
        scores = torch.where(
            any_valid.unsqueeze(-1).expand_as(scores),
            scores,
            torch.zeros_like(scores),
        )
        attn = torch.softmax(scores, dim=2)  # softmax over j

        # Aggregate values: out[b,i,h,d] = sum_j attn[b,i,j,h] * v[b,j,h,d]
        out = torch.einsum("bijH,bjHd->biHd", attn, v).reshape(b, p, H * D)
        out = self.out_proj(out)
        out = torch.relu(out)
        return out * mask.unsqueeze(-1).float()


class TopKPool(nn.Module):
    """Differentiable TopK pool (Gao & Ji 2019). Selects top ratio*P scored nodes."""

    def __init__(self, in_dim: int, ratio: float) -> None:
        super().__init__()
        self.score = nn.Linear(in_dim, 1)
        self.ratio = ratio

    def forward(
        self,
        h: torch.Tensor,  # (B, P, F)
        adj: torch.Tensor,  # (B, P, P)
        mask: torch.Tensor,  # (B, P)
        coords: torch.Tensor,  # (B, P, 2)
        edge_feats: torch.Tensor,  # (B, P, P, edge_dim)
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        b, p, f = h.shape
        score_logits = self.score(h).squeeze(-1)  # (B, P)
        score_logits = score_logits.masked_fill(~mask, float("-inf"))
        keep = max(1, int(round(p * self.ratio)))
        top_scores, top_idx = score_logits.topk(keep, dim=-1)  # (B, K)

        gate = torch.sigmoid(top_scores).unsqueeze(-1)  # (B, K, 1)
        h_pooled = h.gather(1, top_idx.unsqueeze(-1).expand(-1, -1, f)) * gate

        new_mask = torch.isfinite(top_scores)  # (B, K)
        new_coords = coords.gather(1, top_idx.unsqueeze(-1).expand(-1, -1, 2))

        adj_rows = adj.gather(1, top_idx.unsqueeze(-1).expand(-1, -1, p))  # (B, K, P)
        new_adj = adj_rows.gather(
            2, top_idx.unsqueeze(1).expand(-1, keep, -1)
        )  # (B, K, K)

        new_edge_feats = _gather_pair_feats(edge_feats, top_idx)  # (B, K, K, edge_dim)

        return h_pooled, new_adj, new_mask, new_coords, top_idx, new_edge_feats


class GraphAttentionUNetPolicy(nn.Module):
    """Graph Attention U-Net backbone + 3 prediction heads.

    Same structural shape as case5's GraphUNetPolicy but with attention layers
    instead of degree-normalized mean aggregation.
    """

    def __init__(self, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ModelConfig()
        h = self.cfg.hidden
        heads = self.cfg.attn_heads
        # Input projection
        self.in_proj = nn.Linear(self.cfg.planet_in_dim, h)
        # Encoder (3 GraphAttention + 2 TopK pools)
        self.enc0 = GraphAttention(h, h, num_heads=heads)
        self.pool0 = TopKPool(h, ratio=2.0 / 3.0)
        self.enc1 = GraphAttention(h, h, num_heads=heads)
        self.pool1 = TopKPool(h, ratio=0.5)
        self.enc2 = GraphAttention(h, h, num_heads=heads)
        # Bottleneck context (global pool + global feats)
        self.psi = _mlp(h + self.cfg.global_in_dim, h, h)
        # Decoder (2 GraphAttention after unpool + skip add)
        self.dec1 = GraphAttention(h, h, num_heads=heads)
        self.dec0 = GraphAttention(h, h, num_heads=heads)
        # Heads (input: per-node h with ctx concatenated)
        self.from_head = nn.Linear(h + h, 1)
        self.target_head = nn.Sequential(
            nn.Linear(h + h + TEMPLATE_CTX_DIM, h),
            nn.ReLU(),
            nn.Linear(h, NUM_TEMPLATES),
        )
        self.ships_head = nn.Linear(h + h, self.cfg.ships_buckets)

    @staticmethod
    def _unpool(
        h_low: torch.Tensor,
        idx: torch.Tensor,
        target_p: int,
    ) -> torch.Tensor:
        """Scatter pooled features back to a (B, target_p, F) zero-filled tensor."""
        b, _, f = h_low.shape
        out = h_low.new_zeros(b, target_p, f)
        out.scatter_(1, idx.unsqueeze(-1).expand(-1, -1, f), h_low)
        return out

    def forward(self, batch: BatchFeatures) -> PolicyOutput:
        x = batch.planet_feats  # (B, P, F)
        mask = batch.planet_mask  # (B, P)
        b, p, _ = x.shape

        coords = x[..., [COL_X, COL_Y]]
        adj0 = _knn_adjacency(coords, mask, KNN_K)
        edge0 = _pairwise_geometry(x)  # (B, P, P, PAIR_FEAT_DIM)

        h0 = self.in_proj(x) * mask.unsqueeze(-1).float()
        h0 = self.enc0(h0, adj0, mask, edge0)  # (B, P, H)

        h1, adj1, mask1, coords1, idx1, edge1 = self.pool0(
            h0, adj0, mask, coords, edge0
        )
        h1 = self.enc1(h1, adj1, mask1, edge1)

        h2, adj2, mask2, coords2, idx2, edge2 = self.pool1(
            h1, adj1, mask1, coords1, edge1
        )
        h2 = self.enc2(h2, adj2, mask2, edge2)

        # Bottleneck: global pool over the deepest level
        m2 = mask2.unsqueeze(-1).float()
        denom2 = m2.sum(dim=1).clamp_min(1.0)
        pooled = (h2 * m2).sum(dim=1) / denom2  # (B, H)
        ctx = self.psi(torch.cat([pooled, batch.global_feats], dim=-1))  # (B, H)

        # Decoder: unpool L2 → L1, add skip h1, GraphAttention
        h1_up = self._unpool(h2, idx2, h1.shape[1])
        h1_dec = self.dec1(h1_up + h1, adj1, mask1, edge1)

        # Decoder: unpool L1 → L0, add skip h0, GraphAttention
        h0_up = self._unpool(h1_dec, idx1, p)
        h0_dec = self.dec0(h0_up + h0, adj0, mask, edge0)  # (B, P, H)

        ctx_exp = ctx.unsqueeze(1).expand(-1, p, -1)
        h_with_ctx = torch.cat([h0_dec, ctx_exp], dim=-1)  # (B, P, 2H)

        from_logits = self.from_head(h_with_ctx).squeeze(-1)
        from_logits = from_logits.masked_fill(~batch.my_planet_mask, float("-inf"))

        target_input = torch.cat([h_with_ctx, batch.template_ctx], dim=-1)
        target_logits = self.target_head(target_input)  # (B, P, NUM_TEMPLATES)

        ships_logits = self.ships_head(h_with_ctx)

        return PolicyOutput(
            from_logits=from_logits,
            target_logits=target_logits,
            ships_logits=ships_logits,
        )


# Backwards-compatible alias: existing agent.py / tests import DeepSetsPolicy.
DeepSetsPolicy = GraphAttentionUNetPolicy
GraphUNetPolicy = GraphAttentionUNetPolicy  # case5 name compatibility


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(cfg: ModelConfig | None = None) -> GraphAttentionUNetPolicy:
    """Convenience constructor used by training and inference."""
    model = GraphAttentionUNetPolicy(cfg)
    model.eval()
    return model


__all__ = [
    "DeepSetsPolicy",
    "GraphAttentionUNetPolicy",
    "GraphUNetPolicy",
    "ModelConfig",
    "MAX_PLANETS",
    "build_model",
    "count_parameters",
]
