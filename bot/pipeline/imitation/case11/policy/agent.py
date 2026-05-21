"""Kaggle entry: case11 agent(obs).

Per-match `HistoryState` is held module-level. Reset on `obs.step == 0` or
when step regresses below the recorded `last_step` (Kaggle re-uses the same
process across matches).

case11 supports only the per_planet head. Submission weights load from
`weights.pt`.
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
from .model import Case11Policy, ModelConfig

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
    return "per_planet"


_TEMPERATURE = _resolve_temperature()
_HEAD_MODE = _resolve_head_mode()
_MODEL: Case11Policy | None = None
_HISTORY: HistoryState = HistoryState()


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _infer_config_from_state(state: dict[str, torch.Tensor]) -> ModelConfig:
    in_proj = state.get("backbone.in_proj.weight")
    hidden = int(in_proj.shape[0]) if in_proj is not None else 192
    planet_in_dim = int(in_proj.shape[1]) if in_proj is not None else PLANET_FEAT_DIM
    inducing = state.get("backbone.encoder.0.inducing")
    inducing_points = int(inducing.shape[1]) if inducing is not None else 24
    encoder_layers = 1 + max(
        (
            int(k.split(".")[2])
            for k in state
            if k.startswith("backbone.encoder.") and k.split(".")[2].isdigit()
        ),
        default=3,
    )
    return ModelConfig(
        planet_in_dim=planet_in_dim,
        global_in_dim=GLOBAL_FEAT_DIM,
        hidden=hidden,
        attn_heads=_env_int("IL_CASE9_ATTN_HEADS", 8 if hidden >= 192 else 4),
        inducing_points=inducing_points,
        encoder_layers=encoder_layers,
        head_mode=_HEAD_MODE,
    )


def _load_model() -> Case11Policy:
    if _DEFAULT_WEIGHTS.exists():
        state = torch.load(_DEFAULT_WEIGHTS, map_location="cpu", weights_only=True)
        cfg = _infer_config_from_state(state)
        model = Case11Policy(cfg)
        model.load_state_dict(state, strict=False)
    else:
        cfg = ModelConfig(
            planet_in_dim=PLANET_FEAT_DIM,
            global_in_dim=GLOBAL_FEAT_DIM,
            hidden=192,
            attn_heads=8,
            inducing_points=24,
            encoder_layers=4,
            head_mode=_HEAD_MODE,
        )
        model = Case11Policy(cfg)
    model.eval()
    return model


def _get_model() -> Case11Policy:
    global _MODEL
    if _MODEL is None:
        _MODEL = _load_model()
    return _MODEL


def _maybe_reset_history(obs: dict[str, Any]) -> None:
    """Reset history at episode boundaries.

    2026-05-21 trap: the Rust simulator (`orbit_wars_rust.run_episode`) calls
    the agent **3 times at step=0** (a probe + two real calls) before
    advancing to step=1. The naive condition `step <= last_step` then
    clobbers history on every duplicate-step call, which silently degrades
    agents that depend on `from_multihot` / multi-step state. So we:
      * clear only when this is the very first call of the process
        (last_step is None), OR
      * the simulator restarted (step strictly less than last_step).
    Duplicate same-step calls are now idempotent w.r.t. history.
    """
    step = int(obs.get("step", 0) or 0)
    if _HISTORY.last_step is None:
        _HISTORY.clear()
    elif step < _HISTORY.last_step:
        _HISTORY.clear()


def agent(obs: Any, *_: Any, **__: Any) -> list[list[int | float]]:
    # `*_` / `**__` allow being called either as `agent(obs)` (Kaggle submit) or
    # `agent(obs, configuration)` (local selfplay simulator probe). Without
    # this, the rust simulator's probe raises TypeError, falls back to the
    # framework env.run() path, and the agent never produces a fire action.
    obs_dict = obs if isinstance(obs, dict) else dict(obs)
    _maybe_reset_history(obs_dict)
    # 2026-05-21 trap: same step can call us multiple times under the Rust
    # simulator (probe + real step calls). Skip the history.append in that
    # case so the launch_event deque doesn't accumulate duplicates.
    cur_step = int(obs_dict.get("step", 0) or 0)
    update_allowed = _HISTORY.last_step is None or cur_step > _HISTORY.last_step
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
    if update_allowed:
        update_history(_HISTORY, obs_dict, actions)
    return actions
