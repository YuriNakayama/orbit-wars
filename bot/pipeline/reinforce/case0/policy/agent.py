"""Kaggle entrypoint: agent(obs) → action list using the PPO-trained policy.

Loads weights from `policy/weights.pt` if present; otherwise falls back to
random-initialized parameters (lets the case be importable before training).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .decoder import decode
from .featurizer import featurize
from .model import ActorCritic, ModelConfig
from .sampling import greedy_action

_WEIGHTS_PATH = Path(__file__).resolve().parent / "weights.pt"

_MODEL: ActorCritic | None = None


def _load_model() -> ActorCritic:
    model = ActorCritic(ModelConfig())
    if _WEIGHTS_PATH.exists():
        state = torch.load(_WEIGHTS_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    model.eval()
    return model


def _get_model() -> ActorCritic:
    global _MODEL
    if _MODEL is None:
        _MODEL = _load_model()
    return _MODEL


def agent(obs: Any) -> list[list[int | float]]:
    obs_dict = obs if isinstance(obs, dict) else dict(obs)
    model = _get_model()
    batch, snapshot = featurize(obs_dict)
    with torch.no_grad():
        output = model(batch)
    action = greedy_action(output, batch, from_threshold=0.5)
    return decode(action, snapshot, obs_dict)


__all__ = ["agent"]
