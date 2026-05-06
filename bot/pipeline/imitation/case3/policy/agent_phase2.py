"""Kaggle entry: phase2 agent(obs) using extended featurizer + history.

Maintains a per-match `HistoryState` (planet snapshots + recent launches).
The buffer resets when `obs.step == 0` is observed (start of a new match).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from .decoder import decode
from .featurizer_phase2 import (
    GLOBAL_FEAT_DIM,
    PLANET_FEAT_DIM,
    HistoryState,
    featurize,
    update_history,
)
from .model import DeepSetsPolicy, ModelConfig

_WEIGHTS_PATH = Path(__file__).resolve().parent / "weights.pt"

_TARGET_TEMPERATURE = 0.8
_SHIPS_TEMPERATURE = 1.1


def _resolve_from_threshold() -> float:
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
_HISTORY: HistoryState = HistoryState()


def _load_model() -> DeepSetsPolicy:
    cfg = ModelConfig(planet_in_dim=PLANET_FEAT_DIM, global_in_dim=GLOBAL_FEAT_DIM)
    model = DeepSetsPolicy(cfg)
    state = torch.load(_WEIGHTS_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def _get_model() -> DeepSetsPolicy:
    global _MODEL
    if _MODEL is None:
        _MODEL = _load_model()
    return _MODEL


def _maybe_reset_history(obs: dict[str, Any]) -> None:
    """Reset history at episode start.

    Kaggle reuses the same Python process for sequential matches, so we must
    detect transitions: step==0 OR step regresses below previous last_step.
    """
    step = int(obs.get("step", 0) or 0)
    if step == 0:
        _HISTORY.clear()
    elif _HISTORY.last_step is not None and step <= _HISTORY.last_step:
        _HISTORY.clear()


def agent(obs: Any) -> list[list[int | float]]:
    obs_dict = obs if isinstance(obs, dict) else dict(obs)
    _maybe_reset_history(obs_dict)
    model = _get_model()
    batch, snapshot = featurize(obs_dict, _HISTORY)
    with torch.no_grad():
        output = model(batch)
    actions = decode(
        output,
        snapshot,
        obs_dict,
        from_threshold=_FROM_THRESHOLD,
        min_fire_topk=_MIN_FIRE_TOPK,
        max_fire_count=_MAX_FIRE_COUNT,
        target_temperature=_TARGET_TEMPERATURE,
        ships_temperature=_SHIPS_TEMPERATURE,
    )
    update_history(_HISTORY, obs_dict, actions)
    return actions
