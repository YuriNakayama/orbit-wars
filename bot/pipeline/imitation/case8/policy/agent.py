"""Kaggle entry: agent(obs) → action list using the trained DeepSets policy."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from .decoder import decode
from .featurizer import (
    HistoryState,
    featurize,
    fleet_snapshot_from_obs,
    planet_snapshot_from_obs,
)
from .model import DeepSetsPolicy, ModelConfig

_WEIGHTS_PATH = Path(__file__).resolve().parent / "weights.pt"

# Phase 1 inference knobs (tuned on val.parquet with canonical iter6 weights):
# - FROM_THRESHOLD=0.31 maximizes from_head F1 (sweep: 0.47 vs 0.20 at 0.5)
# - TARGET_TEMPERATURE=0.8 min val NLL on target_head, ECE 0.047 → 0.014
# - SHIPS_TEMPERATURE=1.1 min val NLL on ships_head, ECE 0.023 → 0.011
# - MIN_FIRE_TOPK=1 fallback for turns where no source clears FROM_THRESHOLD
# - MAX_FIRE_COUNT caps simultaneous fires to curb overfire at low thresholds
# Phase 3-D sweep hook: env var IL_MAX_FIRE_COUNT can override the constant
# ("none" disables the cap). Local-only path — Kaggle submits do not set it.
_TARGET_TEMPERATURE = 0.8
_SHIPS_TEMPERATURE = 1.1


def _resolve_from_threshold() -> float:
    # Phase 4-D sweep hook: env var IL_FROM_THRESHOLD overrides the F1-optimal
    # default 0.31 for local decode exploration. Kaggle submits leave it unset.
    raw = os.environ.get("IL_FROM_THRESHOLD")
    if raw is None:
        return 0.31
    return float(raw)


def _resolve_min_fire_topk() -> int:
    raw = os.environ.get("IL_MIN_FIRE_TOPK")
    if raw is None:
        return 1
    return int(raw)


def _resolve_max_fire_count() -> int | None:
    # Phase 3-D result (300-game sweep on iter9 weights): cap=4/8/12 all give
    # 0/300 wins; only cap=None returns 3/300 (CI [0.34%, 2.90%]), the first
    # reproducible non-zero win rate. Canonical default is now unbounded.
    # Env var IL_MAX_FIRE_COUNT is kept as an override for local sweeps.
    raw = os.environ.get("IL_MAX_FIRE_COUNT")
    if raw is None:
        return None
    s = raw.strip().lower()
    if s in {"none", "null", "off", "unbounded"}:
        return None
    return int(s)


_FROM_THRESHOLD = _resolve_from_threshold()
_MIN_FIRE_TOPK = _resolve_min_fire_topk()
_MAX_FIRE_COUNT = _resolve_max_fire_count()

_MODEL: DeepSetsPolicy | None = None
_HISTORY: HistoryState | None = None
_LAST_STEP: int = -1


def _load_model() -> DeepSetsPolicy:
    model = DeepSetsPolicy(ModelConfig())
    state = torch.load(_WEIGHTS_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def _get_model() -> DeepSetsPolicy:
    global _MODEL
    if _MODEL is None:
        _MODEL = _load_model()
    return _MODEL


def _get_history(step: int) -> HistoryState:
    """Return per-match HistoryState; reset when step regresses (new match)."""
    global _HISTORY, _LAST_STEP
    if _HISTORY is None or step < _LAST_STEP:
        _HISTORY = HistoryState()
    _LAST_STEP = step
    return _HISTORY


def agent(obs: Any) -> list[list[int | float]]:
    obs_dict = obs if isinstance(obs, dict) else dict(obs)
    model = _get_model()
    step = int(obs_dict.get("step", 0) or 0)
    history = _get_history(step)
    batch, snapshot = featurize(obs_dict, history=history)
    history.push(planet_snapshot_from_obs(obs_dict), fleet_snapshot_from_obs(obs_dict))
    with torch.no_grad():
        output = model(batch)
    return decode(
        output,
        snapshot,
        obs_dict,
        from_threshold=_FROM_THRESHOLD,
        min_fire_topk=_MIN_FIRE_TOPK,
        max_fire_count=_MAX_FIRE_COUNT,
        target_temperature=_TARGET_TEMPERATURE,
        ships_temperature=_SHIPS_TEMPERATURE,
    )
