"""Frozen dataclasses for imitation/case11 (per_planet head only).

3-layer mask design (Lux3-derived):

  layer1 (invalid-action): target_mask — applied -1e9 on target axis of
      per_planet_logits at softmax. Physical-rule-only (no strategic prunes).
  layer2 (existence/ownership): effective_source_mask = my_planet & ships>0,
      restricts the source axis of per_planet_logits. encoder still sees all
      planets (enemy/neutral kept as observation context — never masked from
      attention).
  layer3 (conditional head): used inside the loss (should_learn_ship) — kept
      in batch-side tensors (dataset.py), not on BatchFeatures (decoder side).

Note: my_planet_mask は decoder/loss から完全に消えた。debug したい場合は
preprocess.py 段階で computed なので parquet から再生成可能。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BatchFeatures:
    planet_feats: torch.Tensor  # (B, MAX_PLANETS, PLANET_FEAT_DIM)
    planet_mask: torch.Tensor  # (B, MAX_PLANETS) bool — layer1 (existence)
    target_mask: torch.Tensor  # (B, MAX_PLANETS) bool — layer1 (physical-rule)
    effective_source_mask: torch.Tensor  # (B, MAX_PLANETS) bool — layer2
    global_feats: torch.Tensor  # (B, GLOBAL_FEAT_DIM)
    template_ctx: torch.Tensor  # (B, MAX_PLANETS, TEMPLATE_CTX_DIM)
    candidate_feats: torch.Tensor  # (B, MAX_PLANETS, CAND_K, CAND_FEAT_DIM)
    candidate_mask: torch.Tensor  # (B, MAX_PLANETS, CAND_K) bool
    candidate_pid: torch.Tensor  # (B, MAX_PLANETS, CAND_K) int64 — -1 for invalid


@dataclass(frozen=True)
class PolicyOutput:
    """per_planet head output: (P+1)-class CE + ships regression."""

    # (B, P, P+1) — last index is no-op sentinel
    per_planet_logits: torch.Tensor
    ship_pred: torch.Tensor  # (B, P) — log1p(ships) regression


@dataclass(frozen=True)
class WorldSnapshot:
    """Minimal obs view kept alongside features for decoding."""

    planet_ids: tuple[int, ...]
    my_planet_ids: tuple[int, ...]
    player: int
    step: int
