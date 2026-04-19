"""Frozen dataclasses for case3 IL baseline data flow."""

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


@dataclass(frozen=True)
class PolicyOutput:
    from_logits: torch.Tensor  # (B, MAX_PLANETS) — sigmoid → from_prob
    target_logits: torch.Tensor  # (B, MAX_PLANETS, MAX_PLANETS + 1) — last slot = no-op
    ships_logits: torch.Tensor  # (B, MAX_PLANETS, SHIPS_BUCKETS)


@dataclass(frozen=True)
class WorldSnapshot:
    """Minimal obs view kept alongside features for decoding."""

    planet_ids: tuple[int, ...]  # length == valid planet count
    my_planet_ids: tuple[int, ...]
    player: int
    step: int
