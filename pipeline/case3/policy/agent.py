"""Kaggle entry: agent(obs) → action list using the trained DeepSets policy."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .decoder import decode
from .featurizer import featurize
from .model import DeepSetsPolicy, ModelConfig

_WEIGHTS_PATH = Path(__file__).resolve().parent / "weights.pt"
_FROM_THRESHOLD = 0.05

_MODEL: DeepSetsPolicy | None = None


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


def agent(obs: Any) -> list[list[int | float]]:
    obs_dict = obs if isinstance(obs, dict) else dict(obs)
    model = _get_model()
    batch, snapshot = featurize(obs_dict)
    with torch.no_grad():
        output = model(batch)
    return decode(output, snapshot, obs_dict, from_threshold=_FROM_THRESHOLD)
