"""Kaggle entry: agent(obs) → action list using the PPO-trained policy."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from .decoder import decode
from .featurizer import HistoryState, featurize
from .model import ActorCritic, ModelConfig, load_bc_weights
from .sampling import greedy_action

_DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().parent / "weights.pt"

_MODEL: ActorCritic | None = None
_HISTORY: HistoryState | None = None


def _weights_path() -> Path:
    """Resolve the weights file. `ORBIT_WARS_CASE7_WEIGHTS` overrides the bundled
    `weights.pt` so an arbitrary per-iter `ckpt_i*.pt` can be evaluated vs a
    rulebase locally without copying over the canonical submit weights. The Kaggle
    submit runtime never sets this var, so production still uses `weights.pt`.
    """
    override = os.environ.get("ORBIT_WARS_CASE7_WEIGHTS", "").strip()
    return Path(override) if override else _DEFAULT_WEIGHTS_PATH


def _load_model() -> ActorCritic:
    model = ActorCritic(ModelConfig())
    weights = _weights_path()
    if weights.exists():
        # Reuse the strict=False loader: handles both PPO checkpoints (full
        # state_dict) and BC warm-start weights (no value_head / ship_log_std).
        load_bc_weights(model, str(weights))
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
