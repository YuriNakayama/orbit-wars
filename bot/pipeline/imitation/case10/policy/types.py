"""Frozen dataclasses for imitation/case10."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BatchFeatures:
    planet_feats: torch.Tensor  # (B, MAX_PLANETS, PLANET_FEAT_DIM)
    planet_mask: torch.Tensor  # (B, MAX_PLANETS) bool — True for valid slots
    my_planet_mask: torch.Tensor  # (B, MAX_PLANETS) bool — owner == player
    target_mask: torch.Tensor  # (B, MAX_PLANETS) bool — valid target candidates
    global_feats: torch.Tensor  # (B, GLOBAL_FEAT_DIM)
    template_ctx: torch.Tensor  # (B, MAX_PLANETS, TEMPLATE_CTX_DIM)
    candidate_feats: torch.Tensor  # (B, MAX_PLANETS, CAND_K, CAND_FEAT_DIM)
    candidate_mask: torch.Tensor  # (B, MAX_PLANETS, CAND_K) bool — slot 0 always True
    candidate_pid: torch.Tensor  # (B, MAX_PLANETS, CAND_K) int64 — -1 for invalid


@dataclass(frozen=True)
class PolicyOutput:
    """All variants populate the fields they use; others are None."""

    from_logits: torch.Tensor | None = None
    target_logits: torch.Tensor | None = None  # (B, P, NUM_TEMPLATES) — template_ships
    # (B, P, ships_buckets) — candidate_ships / template_ships
    ships_logits: torch.Tensor | None = None
    candidate_logits: torch.Tensor | None = None  # (B, P, CAND_K) — candidate_ships
    ship_pred: torch.Tensor | None = None


@dataclass(frozen=True)
class WorldSnapshot:
    """Minimal obs view kept alongside features for decoding."""

    planet_ids: tuple[int, ...]
    my_planet_ids: tuple[int, ...]
    player: int
    step: int
