"""Frozen dataclasses for reinforce/case6.

Carries the per_planet head's input tensors (planet/global feats + masks) plus
the world snapshot used by the decoder. `template_ctx` / `candidate_*` are
preserved because the case9 featurizer populates them and downstream changes
might need them, but the per_planet head ignores those fields.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BatchFeatures:
    planet_feats: torch.Tensor  # (B, MAX_PLANETS, PLANET_FEAT_DIM)
    planet_mask: torch.Tensor  # (B, MAX_PLANETS) bool
    my_planet_mask: torch.Tensor  # (B, MAX_PLANETS) bool
    target_mask: torch.Tensor  # (B, MAX_PLANETS) bool
    global_feats: torch.Tensor  # (B, GLOBAL_FEAT_DIM)
    template_ctx: torch.Tensor  # (B, MAX_PLANETS, TEMPLATE_CTX_DIM)
    candidate_feats: torch.Tensor  # (B, MAX_PLANETS, CAND_K, CAND_FEAT_DIM)
    candidate_mask: torch.Tensor  # (B, MAX_PLANETS, CAND_K) bool
    candidate_pid: torch.Tensor  # (B, MAX_PLANETS, CAND_K) int64


@dataclass(frozen=True)
class WorldSnapshot:
    planet_ids: tuple[int, ...]
    my_planet_ids: tuple[int, ...]
    player: int
    step: int
