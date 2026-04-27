"""Graph U-Net policy network for imitation/case3 IL baseline.

Replaces the prior DeepSets backbone with a Graph U-Net (Gao & Ji 2019) over
the planet set:

  - kNN graph built per-frame from (x, y) coords (k=8, masked padding).
  - Encoder: 3 levels of GraphConv → TopK pooling (P → P*2//3 → P//3).
  - Bottleneck: GraphConv + global mean pool → context.
  - Decoder: 2 levels of unpool (scatter back) + skip-add + GraphConv.
  - 3 heads identical to the previous model: from / target / ships.

The class is aliased as `DeepSetsPolicy` so existing imports and weight files
continue to load when re-trained, but the architecture is fully different.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .featurizer_phase2 import GLOBAL_FEAT_DIM, MAX_PLANETS, PLANET_FEAT_DIM
from .templates import NUM_TEMPLATES, TEMPLATE_CTX_DIM
from .types import BatchFeatures, PolicyOutput

# featurizer column layout (PLANET_FEAT_DIM == 18 in case2; columns 0..7 unchanged)
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
    # Make any "infinite distance" neighbour fall away (padding rows).
    valid_neighbour = torch.isfinite(dist.gather(2, idx))
    adj.scatter_(2, idx, valid_neighbour)
    adj = adj | adj.transpose(1, 2)  # symmetric
    adj = adj & valid_pair
    return adj


class GraphConv(nn.Module):
    """1-hop mean aggregation + linear: h' = ReLU(W1 h + W2 mean_{j in N(i)} h_j)."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_neigh = nn.Linear(in_dim, out_dim)

    def forward(
        self, h: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        # h: (B, P, F), adj: (B, P, P) bool, mask: (B, P) bool
        a = adj.float()
        deg = a.sum(dim=-1, keepdim=True).clamp_min(1.0)  # (B, P, 1)
        agg = torch.bmm(a, h) / deg  # (B, P, F)
        out = self.lin_self(h) + self.lin_neigh(agg)
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
        coords: torch.Tensor,  # (B, P, 2) — pooled forward to next level
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b, p, f = h.shape
        score_logits = self.score(h).squeeze(-1)  # (B, P)
        score_logits = score_logits.masked_fill(~mask, float("-inf"))
        keep = max(1, int(round(p * self.ratio)))
        top_scores, top_idx = score_logits.topk(keep, dim=-1)  # (B, K)

        gate = torch.sigmoid(top_scores).unsqueeze(-1)  # (B, K, 1)
        h_pooled = h.gather(1, top_idx.unsqueeze(-1).expand(-1, -1, f)) * gate

        new_mask = torch.isfinite(top_scores)  # (B, K) — kept rows from valid mask
        new_coords = coords.gather(1, top_idx.unsqueeze(-1).expand(-1, -1, 2))

        # Sub-graph adjacency: rows/cols indexed by top_idx
        adj_rows = adj.gather(1, top_idx.unsqueeze(-1).expand(-1, -1, p))  # (B, K, P)
        new_adj = adj_rows.gather(
            2, top_idx.unsqueeze(1).expand(-1, keep, -1)
        )  # (B, K, K)

        return h_pooled, new_adj, new_mask, new_coords, top_idx


class GraphUNetPolicy(nn.Module):
    """Graph U-Net backbone + 3 prediction heads.

    Encoder/decoder operate on the planet set; final per-node features feed the
    same from/target/ships heads as the prior DeepSets model.
    """

    def __init__(self, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ModelConfig()
        h = self.cfg.hidden
        # Input projection
        self.in_proj = nn.Linear(self.cfg.planet_in_dim, h)
        # Encoder (3 GraphConv + 2 TopK pools)
        self.enc0 = GraphConv(h, h)
        self.pool0 = TopKPool(h, ratio=2.0 / 3.0)
        self.enc1 = GraphConv(h, h)
        self.pool1 = TopKPool(h, ratio=0.5)  # halves again ⇒ overall ~1/3
        self.enc2 = GraphConv(h, h)
        # Bottleneck context (global pool + global feats)
        self.psi = _mlp(h + self.cfg.global_in_dim, h, h)
        # Decoder (2 GraphConv after unpool + skip add)
        self.dec1 = GraphConv(h, h)
        self.dec0 = GraphConv(h, h)
        # Heads (input: per-node h with ctx concatenated)
        self.from_head = nn.Linear(h + h, 1)
        # target head additionally consumes per-source template context features
        # so the network can learn "which template's resolved planet is best for
        # this source in this state" rather than collapsing to a single argmax.
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

        h0 = self.in_proj(x) * mask.unsqueeze(-1).float()
        h0 = self.enc0(h0, adj0, mask)  # (B, P, H)

        h1, adj1, mask1, coords1, idx1 = self.pool0(h0, adj0, mask, coords)
        h1 = self.enc1(h1, adj1, mask1)

        h2, adj2, mask2, coords2, idx2 = self.pool1(h1, adj1, mask1, coords1)
        h2 = self.enc2(h2, adj2, mask2)

        # Bottleneck: global pool over the deepest level
        m2 = mask2.unsqueeze(-1).float()
        denom2 = m2.sum(dim=1).clamp_min(1.0)
        pooled = (h2 * m2).sum(dim=1) / denom2  # (B, H)
        ctx = self.psi(torch.cat([pooled, batch.global_feats], dim=-1))  # (B, H)

        # Decoder: unpool L2 → L1, add skip h1, GraphConv
        h1_up = self._unpool(h2, idx2, h1.shape[1])
        h1_dec = self.dec1(h1_up + h1, adj1, mask1)

        # Decoder: unpool L1 → L0, add skip h0, GraphConv
        h0_up = self._unpool(h1_dec, idx1, p)
        h0_dec = self.dec0(h0_up + h0, adj0, mask)  # (B, P, H)

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
DeepSetsPolicy = GraphUNetPolicy


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(cfg: ModelConfig | None = None) -> GraphUNetPolicy:
    """Convenience constructor used by training and inference."""
    model = GraphUNetPolicy(cfg)
    model.eval()
    return model


__all__ = [
    "DeepSetsPolicy",
    "GraphUNetPolicy",
    "ModelConfig",
    "MAX_PLANETS",
    "build_model",
    "count_parameters",
]
