"""Frozen dataclasses for imitation/case4.

Differences from case3:
  - drops `template_ctx` (templates are not used)
  - adds `candidate_feats` / `candidate_pid` / `candidate_mask`
  - PolicyOutput exposes `candidate_logits` only (single head)
"""

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
    candidate_feats: torch.Tensor  # (B, MAX_PLANETS, CAND_K, CAND_FEAT_DIM)
    candidate_mask: torch.Tensor  # (B, MAX_PLANETS, CAND_K) bool — slot 0 always True
    candidate_pid: torch.Tensor  # (B, MAX_PLANETS, CAND_K) int64 — -1 for invalid


@dataclass(frozen=True)
class PolicyOutput:
    candidate_logits: torch.Tensor  # (B, MAX_PLANETS, CAND_K) — slot 0 = no-op


@dataclass(frozen=True)
class WorldSnapshot:
    """Minimal obs view kept alongside features for decoding."""

    planet_ids: tuple[int, ...]  # length == valid planet count
    my_planet_ids: tuple[int, ...]
    player: int
    step: int
