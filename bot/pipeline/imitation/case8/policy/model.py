"""Graph U-Net + candidate head + ship-count head for imitation/case8.

Backbone (encoder / TopK pool / decoder / kNN) is identical to case4. Two heads:

  1. per-source x CAND_K candidate categorical (slot 0 = no-op) — same as case4.
  2. per-source ship-count regression (1 scalar per source planet) — NEW for case8.

Ship head input: `self_h ⊕ global_h` (no candidate dependency — produces a
single "if firing, fire this many" scalar per source, decoupled from which
slot is picked). Output is unbounded; decoder clamps to [rule_floor, src.ships].
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .candidates import CAND_FEAT_DIM, CAND_K
from .featurizer import GLOBAL_FEAT_DIM, MAX_PLANETS, PLANET_FEAT_DIM
from .types import BatchFeatures, PolicyOutput

COL_X = 0
COL_Y = 1
KNN_K = 8


@dataclass(frozen=True)
class ModelConfig:
    planet_in_dim: int = PLANET_FEAT_DIM
    global_in_dim: int = GLOBAL_FEAT_DIM
    cand_in_dim: int = CAND_FEAT_DIM
    cand_k: int = CAND_K
    hidden: int = 128


def _knn_adjacency(
    coords: torch.Tensor,  # (B, P, 2)
    mask: torch.Tensor,  # (B, P) bool
    k: int,
) -> torch.Tensor:
    """Symmetric kNN adjacency from (x, y). Padding rows are isolated."""
    b, p, _ = coords.shape
    dx = coords.unsqueeze(2) - coords.unsqueeze(1)
    dist = (dx * dx).sum(-1)
    big = torch.full_like(dist, float("inf"))
    valid_pair = mask.unsqueeze(1) & mask.unsqueeze(2)
    dist = torch.where(valid_pair, dist, big)
    eye = torch.eye(p, dtype=torch.bool, device=coords.device).unsqueeze(0)
    dist = dist.masked_fill(eye, float("inf"))
    kk = min(k, p - 1)
    _, idx = dist.topk(kk, dim=-1, largest=False)
    adj = torch.zeros(b, p, p, dtype=torch.bool, device=coords.device)
    adj.scatter_(2, idx, True)
    valid_neighbour = torch.isfinite(dist.gather(2, idx))
    adj.scatter_(2, idx, valid_neighbour)
    adj = adj | adj.transpose(1, 2)
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
        a = adj.float()
        deg = a.sum(dim=-1, keepdim=True).clamp_min(1.0)
        agg = torch.bmm(a, h) / deg
        out = self.lin_self(h) + self.lin_neigh(agg)
        out = torch.relu(out)
        return out * mask.unsqueeze(-1).float()


class TopKPool(nn.Module):
    """Differentiable TopK pool (Gao & Ji 2019)."""

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b, p, f = h.shape
        score_logits = self.score(h).squeeze(-1)
        score_logits = score_logits.masked_fill(~mask, float("-inf"))
        keep = max(1, int(round(p * self.ratio)))
        top_scores, top_idx = score_logits.topk(keep, dim=-1)

        gate = torch.sigmoid(top_scores).unsqueeze(-1)
        h_pooled = h.gather(1, top_idx.unsqueeze(-1).expand(-1, -1, f)) * gate

        new_mask = torch.isfinite(top_scores)
        new_coords = coords.gather(1, top_idx.unsqueeze(-1).expand(-1, -1, 2))

        adj_rows = adj.gather(1, top_idx.unsqueeze(-1).expand(-1, -1, p))
        new_adj = adj_rows.gather(2, top_idx.unsqueeze(1).expand(-1, keep, -1))

        return h_pooled, new_adj, new_mask, new_coords, top_idx


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
        nn.ReLU(),
    )


class CandidatePolicy(nn.Module):
    """Graph U-Net backbone + per-source candidate categorical head."""

    def __init__(self, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ModelConfig()
        h = self.cfg.hidden
        self.in_proj = nn.Linear(self.cfg.planet_in_dim, h)
        # Encoder (3 GraphConv + 2 TopK pools) — identical to case3
        self.enc0 = GraphConv(h, h)
        self.pool0 = TopKPool(h, ratio=2.0 / 3.0)
        self.enc1 = GraphConv(h, h)
        self.pool1 = TopKPool(h, ratio=0.5)
        self.enc2 = GraphConv(h, h)
        # Bottleneck context (global pool + global feats)
        self.psi = nn.Sequential(
            nn.Linear(h + self.cfg.global_in_dim, h),
            nn.ReLU(),
            nn.Linear(h, h),
        )
        # Decoder (2 GraphConv after unpool + skip add)
        self.dec1 = GraphConv(h, h)
        self.dec0 = GraphConv(h, h)
        # Heads (notebook-style 3-stream MLP fused per candidate)
        self.self_proj = _mlp(h, h, h)
        self.global_proj = _mlp(h, h, h)
        self.cand_encoder = _mlp(self.cfg.cand_in_dim, h, h)
        self.cand_score = nn.Sequential(
            nn.Linear(h * 3, h),
            nn.ReLU(),
            nn.Linear(h, 1),
        )
        self.ship_head = nn.Sequential(
            nn.Linear(h * 2, h),
            nn.ReLU(),
            nn.Linear(h, 1),
        )

    @staticmethod
    def _unpool(h_low: torch.Tensor, idx: torch.Tensor, target_p: int) -> torch.Tensor:
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
        h0 = self.enc0(h0, adj0, mask)

        h1, adj1, mask1, coords1, idx1 = self.pool0(h0, adj0, mask, coords)
        h1 = self.enc1(h1, adj1, mask1)

        h2, adj2, mask2, coords2, idx2 = self.pool1(h1, adj1, mask1, coords1)
        h2 = self.enc2(h2, adj2, mask2)

        m2 = mask2.unsqueeze(-1).float()
        denom2 = m2.sum(dim=1).clamp_min(1.0)
        pooled = (h2 * m2).sum(dim=1) / denom2  # (B, H)
        ctx = self.psi(torch.cat([pooled, batch.global_feats], dim=-1))  # (B, H)

        h1_up = self._unpool(h2, idx2, h1.shape[1])
        h1_dec = self.dec1(h1_up + h1, adj1, mask1)

        h0_up = self._unpool(h1_dec, idx1, p)
        h0_dec = self.dec0(h0_up + h0, adj0, mask)  # (B, P, H)

        # Per-source 3-stream candidate scoring.
        self_h = self.self_proj(h0_dec)  # (B, P, H)
        global_h = self.global_proj(ctx)  # (B, H)
        cand_h = self.cand_encoder(batch.candidate_feats)  # (B, P, K, H)

        # Broadcast self/global to (B, P, K, H)
        k = cand_h.shape[2]
        self_exp = self_h.unsqueeze(2).expand(-1, -1, k, -1)
        global_exp = global_h.unsqueeze(1).unsqueeze(2).expand(-1, p, k, -1)
        joint = torch.cat([self_exp, global_exp, cand_h], dim=-1)  # (B, P, K, 3H)
        cand_logits = self.cand_score(joint).squeeze(-1)  # (B, P, K)
        # Mask invalid slots: -1e9 で「実質ゼロ確率」を表現する。
        # `torch.finfo(float32).min` (= -3.4e38) を使うと `log_softmax` 内で
        # `x - max` が +inf に overflow し、loss が Inf に飛ぶ (commit 6e35f04
        # 後も再現)。1e9 は softmax 出力で exp(-1e9)≒0 を保証するのに十分大きく、
        # かつ log_softmax の差し引き演算で overflow しない。
        cand_logits = cand_logits.masked_fill(~batch.candidate_mask, -1e9)

        ship_in = torch.cat([self_h, global_h.unsqueeze(1).expand(-1, p, -1)], dim=-1)
        ship_pred = self.ship_head(ship_in).squeeze(-1)  # (B, P)
        return PolicyOutput(candidate_logits=cand_logits, ship_pred=ship_pred)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(cfg: ModelConfig | None = None) -> CandidatePolicy:
    """Convenience constructor used by training and inference."""
    model = CandidatePolicy(cfg)
    model.eval()
    return model


__all__ = [
    "CandidatePolicy",
    "ModelConfig",
    "MAX_PLANETS",
    "build_model",
    "count_parameters",
]
