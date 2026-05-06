"""Frozen dataclasses for imitation/case5 IL baseline data flow."""

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
    template_ctx: (
        torch.Tensor
    )  # (B, MAX_PLANETS, TEMPLATE_CTX_DIM) per-source template scores


@dataclass(frozen=True)
class PolicyOutput:
    from_logits: torch.Tensor  # (B, MAX_PLANETS) — sigmoid → from_prob
    target_logits: (
        torch.Tensor
    )  # (B, MAX_PLANETS, NUM_TEMPLATES) — argmax = template id
    ships_logits: torch.Tensor  # (B, MAX_PLANETS, SHIPS_BUCKETS)


@dataclass(frozen=True)
class WorldSnapshot:
    """Minimal obs view kept alongside features for decoding."""

    planet_ids: tuple[int, ...]  # length == valid planet count
    my_planet_ids: tuple[int, ...]
    player: int
    step: int
