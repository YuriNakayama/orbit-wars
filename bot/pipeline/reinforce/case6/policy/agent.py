"""Kaggle entry: agent(obs) → action list using the PPO-trained policy."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .decoder import decode
from .featurizer import HistoryState, featurize
from .model import ActorCritic, ModelConfig, load_bc_weights
from .sampling import greedy_action

_WEIGHTS_PATH = Path(__file__).resolve().parent / "weights.pt"

_MODEL: ActorCritic | None = None
_HISTORY: HistoryState | None = None


def _load_model() -> ActorCritic:
    model = ActorCritic(ModelConfig())
    if _WEIGHTS_PATH.exists():
        # Reuse the strict=False loader: handles both PPO checkpoints (full
        # state_dict) and BC warm-start weights (no value_head / ship_log_std).
        load_bc_weights(model, str(_WEIGHTS_PATH))
    model.eval()
    return model


def _get_model() -> ActorCritic:
    global _MODEL
    if _MODEL is None:
        _MODEL = _load_model()
    return _MODEL


def _get_history() -> HistoryState:
    global _HISTORY
    if _HISTORY is None:
        _HISTORY = HistoryState()
    return _HISTORY


def agent(obs: Any) -> list[list[int | float]]:
    obs_dict = obs if isinstance(obs, dict) else dict(obs)
    if int(obs_dict.get("step", 0) or 0) == 0:
        # Reset history at the start of every episode.
        _get_history().clear()
    model = _get_model()
    batch, snapshot = featurize(obs_dict, history=_get_history())
    with torch.no_grad():
        output = model(batch)
    action = greedy_action(output, batch)
    return decode(action, snapshot, obs_dict)


__all__ = ["agent"]
