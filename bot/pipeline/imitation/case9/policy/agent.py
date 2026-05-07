"""Kaggle entry: case9 agent(obs).

Per-match `HistoryState` is held module-level. Reset on `obs.step == 0` or
when step regresses below the recorded `last_step` (Kaggle re-uses the same
process across matches).

3 head variants supported via head_mode flag:
  - three_head      → weights_three_head.pt
  - candidate       → weights_candidate.pt
  - candidate_ships → weights_candidate_ships.pt
  - dual            → weights_dual.pt (candidate decode + 3-head auxiliary)
The canonical submission weight is `weights.pt` (a copy of the chosen variant).
The variant is read from the env var `IL_CASE9_HEAD_MODE` (default: candidate).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from .decoder import decode
from .featurizer import (
    GLOBAL_FEAT_DIM,
    PLANET_FEAT_DIM,
    HistoryState,
    featurize,
    update_history,
)
from .model import SUPPORTED_HEAD_MODES, Case9Policy, ModelConfig

_DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "weights.pt"


def _resolve_temperature() -> float:
    raw = os.environ.get("IL_CAND_TEMPERATURE")
    if raw is None:
        return 1.0
    try:
        return float(raw)
    except ValueError:
        return 1.0


def _resolve_head_mode() -> str:
    mode = os.environ.get("IL_CASE9_HEAD_MODE", "candidate")
    if mode not in SUPPORTED_HEAD_MODES:
        return "candidate"
    return mode


_TEMPERATURE = _resolve_temperature()
_HEAD_MODE = _resolve_head_mode()
_MODEL: Case9Policy | None = None
_HISTORY: HistoryState = HistoryState()


def _load_model() -> Case9Policy:
    cfg = ModelConfig(
        planet_in_dim=PLANET_FEAT_DIM,
        global_in_dim=GLOBAL_FEAT_DIM,
        head_mode=_HEAD_MODE,
    )
    model = Case9Policy(cfg)
    if _DEFAULT_WEIGHTS.exists():
        state = torch.load(_DEFAULT_WEIGHTS, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=False)
    model.eval()
    return model


def _get_model() -> Case9Policy:
    global _MODEL
    if _MODEL is None:
        _MODEL = _load_model()
    return _MODEL


def _maybe_reset_history(obs: dict[str, Any]) -> None:
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
        candidate_pid=batch.candidate_pid[0],
        candidate_mask=batch.candidate_mask[0],
        temperature=_TEMPERATURE,
        head_mode=_HEAD_MODE,
        template_ctx=batch.template_ctx[0],
    )
    update_history(_HISTORY, obs_dict, actions)
    return actions
