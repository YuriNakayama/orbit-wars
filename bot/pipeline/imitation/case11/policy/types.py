"""Frozen dataclasses for imitation/case11.

case11 supports per_planet only in a single dir (head_mode flag):
  - three_head:       case3/case7 流の (from + target + ships) heads
  - candidate:        case4/case8 流の per-source candidate categorical
  - candidate_ships:  candidate + learned ships head の hybrid
  - template_ships:   template categorical incl no-op + learned ships head
  - dual:             3-head + candidate を同時に返す補助学習 variant

`BatchFeatures` carries both `template_ctx` (3-head 用) and `candidate_feats /
candidate_mask / candidate_pid` (candidate 用) なので、 head_mode に応じて
各 variant の forward が必要なフィールドだけ参照する。

`PolicyOutput` は head_mode で利用するフィールドだけが non-None になる。
dual mode では 3-head と candidate の両方のフィールドが non-None になる。
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
    template_ctx: torch.Tensor  # (B, MAX_PLANETS, TEMPLATE_CTX_DIM) — 3-head 用
    candidate_feats: torch.Tensor  # (B, MAX_PLANETS, CAND_K, CAND_FEAT_DIM)
    candidate_mask: torch.Tensor  # (B, MAX_PLANETS, CAND_K) bool — slot 0 always True
    candidate_pid: torch.Tensor  # (B, MAX_PLANETS, CAND_K) int64 — -1 for invalid


@dataclass(frozen=True)
class PolicyOutput:
    """All variants populate the fields they use; others are None."""

    from_logits: torch.Tensor | None = None  # (B, P) — three_head / dual
    target_logits: torch.Tensor | None = (
        None  # (B, P, NUM_TEMPLATES) — three_head / template_ships / dual
    )
    # (B, P, ships_buckets) — three_head / candidate_ships / template_ships / dual
    ships_logits: torch.Tensor | None = None
    candidate_logits: torch.Tensor | None = (
        None  # (B, P, CAND_K) — candidate / candidate_ships / dual
    )
    ship_pred: torch.Tensor | None = (
        None  # (B, P) — case8-style ship regression (optional)
    )
    # (B, P, P+1) — per_planet head: last index is no-op sentinel
    per_planet_logits: torch.Tensor | None = None


@dataclass(frozen=True)
class WorldSnapshot:
    """Minimal obs view kept alongside features for decoding."""

    planet_ids: tuple[int, ...]
    my_planet_ids: tuple[int, ...]
    player: int
    step: int
