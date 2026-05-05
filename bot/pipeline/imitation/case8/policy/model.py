"""Hierarchical Pointer Network policy network for imitation/case8 IL baseline.

Replaces case7's parallel 3-head design with an **autoregressive Pointer Network
decoder**: from → target → ships. The encoder (Set Transformer ISAB stack from
case7) is reused; only the head section becomes a sequential decoder where each
step's prediction is conditioned on the previous step.

Architecture:
  - Encoder: 3 ISAB stack from case7 (Set Transformer).
  - PMA bottleneck: k=1 query + global feats → ctx (B, H).
  - Decoder step 0 (from): per-source MLP score over my_planet_mask. (case6/7
    と同じ shape (B, P) を出すが、 decoder の起点として位置付け直し)
  - Decoder step 1 (target | from): for each source planet, the source's
    h_node + ctx are fed as query into a content-based attention over
    NUM_TEMPLATES learnable template embeddings. Output (B, P, NUM_TEMPLATES).
    Critical difference from case7: source の h_node を **GRU cell で 1 step
    update** してから template attention に入れる。これにより from を選んだ
    後の hidden state が target 選択に伝播する。
  - Decoder step 2 (ships | from, target): the GRU updated state + selected
    template embedding (forward では argmax cascade) を入力として、 ships
    バケット (4 classes) を予測。

References:
  Vinyals et al. 2015, "Pointer Networks", NeurIPS. arXiv:1506.03134.
  本実装は input set への pointer ではなく learnable template への pointer
  だが、 autoregressive な「前段の決定が後段を条件付ける」構造は同じ。

Class is aliased as `DeepSetsPolicy` so existing agent.py/loaders work.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .featurizer import GLOBAL_FEAT_DIM, MAX_PLANETS, PLANET_FEAT_DIM
from .templates import NUM_TEMPLATES, TEMPLATE_CTX_DIM
from .types import BatchFeatures, PolicyOutput

COL_X = 0
COL_Y = 1


@dataclass(frozen=True)
class ModelConfig:
    planet_in_dim: int = PLANET_FEAT_DIM
    global_in_dim: int = GLOBAL_FEAT_DIM
    hidden: int = 128
    ships_buckets: int = 4
    attn_heads: int = 4
    inducing_points: int = 16
    encoder_layers: int = 3


class MultiheadAttentionBlock(nn.Module):
    """MAB: Multihead Attention Block (case7 と同一)."""

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
    """ISAB: Induced Set Attention Block (case7 と同一)."""

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
    """Pooling by Multihead Attention (case7 と同一)."""

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


class HierarchicalPointerPolicy(nn.Module):
    """Set Transformer encoder + Hierarchical Pointer Network decoder.

    encoder は case7 と完全同一。 decoder のみ autoregressive 化:
    from の hidden state を GRU cell で 1 step update してから target を
    pointer attention で選び、 さらに 1 step update してから ships を
    出す。 forward は推論パス (argmax cascade)、 学習時の teacher
    forcing は losses.py で処理される。
    """

    def __init__(self, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ModelConfig()
        h = self.cfg.hidden
        heads = self.cfg.attn_heads

        # Input projection
        self.in_proj = nn.Linear(self.cfg.planet_in_dim, h)
        # Encoder: ISAB stack (case7 と同型)
        self.encoder = nn.ModuleList(
            [
                InducedSetAttentionBlock(h, heads, self.cfg.inducing_points)
                for _ in range(self.cfg.encoder_layers)
            ]
        )
        # Bottleneck: PMA k=1 + global feats
        self.pool = PMA(h, heads, num_seeds=1)
        self.psi = _mlp(h + self.cfg.global_in_dim, h, h)

        # ----- Decoder (autoregressive: from → target → ships) -----
        # Step 0 (from head): per-source MLP score
        self.from_head = nn.Linear(h + h, 1)  # input = [h_node, ctx]
        # Step 1 (target head): GRU cell で from の hidden state を 1 step update、
        #   その state を query として template embeddings に pointer attention。
        self.template_emb = nn.Parameter(torch.randn(1, NUM_TEMPLATES, h) * 0.02)
        self.from_to_target_gru = nn.GRUCell(
            h + TEMPLATE_CTX_DIM, h
        )  # input: [h_node + template_ctx], hidden: h
        # pointer attention: query (B*P, H) × keys (NUM_TEMPLATES, H)
        self.target_query_proj = nn.Linear(h, h)
        self.target_key_proj = nn.Linear(h, h)
        # Step 2 (ships head): GRU でさらに 1 step update、 linear で 4 buckets。
        self.target_to_ships_gru = nn.GRUCell(h, h)  # input: selected template_emb
        self.ships_head = nn.Linear(h + h, self.cfg.ships_buckets)

    def forward(self, batch: BatchFeatures) -> PolicyOutput:
        x = batch.planet_feats  # (B, P, F)
        mask = batch.planet_mask  # (B, P)
        b, p, _ = x.shape

        # Input projection + masked
        h0 = self.in_proj(x) * mask.unsqueeze(-1).float()

        # Encoder: ISAB stack
        h = h0
        for block in self.encoder:
            h = block(h, mask)

        # Bottleneck: PMA pool + global feats
        pooled = self.pool(h, mask).squeeze(1)  # (B, H)
        ctx = self.psi(torch.cat([pooled, batch.global_feats], dim=-1))  # (B, H)

        # Per-node features with ctx
        ctx_exp = ctx.unsqueeze(1).expand(-1, p, -1)
        h_with_ctx = torch.cat([h, ctx_exp], dim=-1)  # (B, P, 2H)

        # ----- Decoder step 0: from head -----
        from_logits = self.from_head(h_with_ctx).squeeze(-1)  # (B, P)
        from_logits = from_logits.masked_fill(~batch.my_planet_mask, float("-inf"))

        # ----- Decoder step 1: target | from (Pointer Network) -----
        # 各 source planet ごとに、 from の h_node + template_ctx を GRU cell で
        # 1 step update して state_after_from を得る。 state_after_from を query、
        # learnable template embeddings を keys として content-based attention
        # を行い、 (B, P, NUM_TEMPLATES) ロジット を出す。
        h_flat = h.reshape(b * p, -1)  # (B*P, H)
        tctx_flat = batch.template_ctx.reshape(b * p, -1)  # (B*P, TC)
        gru_in_target = torch.cat([h_flat, tctx_flat], dim=-1)  # (B*P, H+TC)
        # 初期 hidden state は ctx を broadcast (B*P, H)
        init_hidden = ctx.unsqueeze(1).expand(-1, p, -1).reshape(b * p, -1).contiguous()
        state_after_from = self.from_to_target_gru(
            gru_in_target, init_hidden
        )  # (B*P, H)

        # Pointer attention: query (B*P, H) × keys (NUM_TEMPLATES, H)
        q = self.target_query_proj(state_after_from)  # (B*P, H)
        k = self.target_key_proj(self.template_emb.squeeze(0))  # (NUM_TEMPLATES, H)
        target_scores = q @ k.t() / (self.cfg.hidden**0.5)  # (B*P, NUM_TEMPLATES)
        target_logits = target_scores.reshape(b, p, NUM_TEMPLATES)

        # ----- Decoder step 2: ships | from, target (argmax cascade in forward) -----
        target_argmax = target_logits.argmax(dim=-1)  # (B, P)
        # selected template embedding を抽出 (B, P, H)
        tmpl_exp = self.template_emb.expand(b, -1, -1)  # (B, NUM_TEMPLATES, H)
        sel_tmpl = tmpl_exp.gather(
            1, target_argmax.unsqueeze(-1).expand(-1, -1, self.cfg.hidden)
        )  # (B, P, H)
        # GRU step 2: state_after_from を hidden として selected template emb を入力
        gru_in_ships = sel_tmpl.reshape(b * p, -1)  # (B*P, H)
        state_after_target = self.target_to_ships_gru(gru_in_ships, state_after_from)
        state_after_target = state_after_target.reshape(b, p, -1)  # (B, P, H)
        # Ships head: [state_after_target, h_node]
        ships_input = torch.cat([state_after_target, h], dim=-1)  # (B, P, 2H)
        ships_logits = self.ships_head(ships_input)  # (B, P, ships_buckets)

        return PolicyOutput(
            from_logits=from_logits,
            target_logits=target_logits,
            ships_logits=ships_logits,
        )


# Backwards-compatible aliases
DeepSetsPolicy = HierarchicalPointerPolicy
GraphUNetPolicy = HierarchicalPointerPolicy
GraphAttentionUNetPolicy = HierarchicalPointerPolicy
SetTransformerPolicy = HierarchicalPointerPolicy


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(cfg: ModelConfig | None = None) -> HierarchicalPointerPolicy:
    """Convenience constructor used by training and inference."""
    model = HierarchicalPointerPolicy(cfg)
    model.eval()
    return model


__all__ = [
    "DeepSetsPolicy",
    "GraphAttentionUNetPolicy",
    "GraphUNetPolicy",
    "HierarchicalPointerPolicy",
    "InducedSetAttentionBlock",
    "MAX_PLANETS",
    "ModelConfig",
    "MultiheadAttentionBlock",
    "PMA",
    "SetTransformerPolicy",
    "build_model",
    "count_parameters",
]
