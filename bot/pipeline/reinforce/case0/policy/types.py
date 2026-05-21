"""Tensor containers for the reinforce/case0 PPO baseline.

Kept tiny and immutable so they can be shared between rollout, training, and
inference without accidentally mutating tensors across module boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BatchFeatures:
    planet_feats: torch.Tensor  # (B, P, F)
    planet_mask: torch.Tensor  # (B, P) bool
    my_planet_mask: torch.Tensor  # (B, P) bool
    target_mask: torch.Tensor  # (B, P) bool (enemy + neutral)
    global_feats: torch.Tensor  # (B, G)


@dataclass(frozen=True)
class PolicyOutput:
    from_logits: torch.Tensor  # (B, P)
    target_logits: torch.Tensor  # (B, P, P)  src x tgt
    ships_logits: torch.Tensor  # (B, P, K)
    value: torch.Tensor  # (B,)


@dataclass(frozen=True)
class WorldSnapshot:
    planet_ids: tuple[int, ...]
    my_planet_ids: tuple[int, ...]
    player: int
    step: int


__all__ = ["BatchFeatures", "PolicyOutput", "WorldSnapshot"]
