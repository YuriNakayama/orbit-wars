"""On-policy rollout collection for PPO with per_planet policy.

Mirrors case0's rollout.py but stores per_planet-shaped actions:
  target_slot  (P,) long  — including NO_OP_INDEX
  log1p_ships  (P,) float — Gaussian sample for ships head

Parallel mode: when `num_workers > 1`, episodes are dispatched across a
multiprocessing pool. The main process broadcasts `model.state_dict()` to
workers, each worker rebuilds an `ActorCritic` from the dict, rolls out its
assigned episodes, and ships transitions back. The order of episode results
is preserved so seat_swap semantics remain identical to the serial path.

The serial path is preserved as the reference implementation; setting
`num_workers <= 1` (the default) reproduces the original behaviour byte-for-byte
modulo Python's nondeterministic dict iteration (which the rollout doesn't
depend on).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from multiprocessing import get_context
from typing import Any

import torch

from ..policy.decoder import decode
from ..policy.featurizer import HistoryState, featurize
from ..policy.model import ActorCritic, ModelConfig
from ..policy.sampling import sample_action
from .env import OpponentAgent, OrbitWarsEpisode


@dataclass
class RolloutBatch:
    planet_feats: torch.Tensor
    planet_mask: torch.Tensor
    my_planet_mask: torch.Tensor
    target_mask: torch.Tensor
    global_feats: torch.Tensor
    template_ctx: torch.Tensor
    candidate_feats: torch.Tensor
    candidate_mask: torch.Tensor
    candidate_pid: torch.Tensor
    target_slot: torch.Tensor  # (N, P) long
    log1p_ships: torch.Tensor  # (N, P) float
    log_probs: torch.Tensor  # (N,)
    values: torch.Tensor  # (N,)
    rewards: torch.Tensor  # (N,)
    advantages: torch.Tensor  # (N,)
    returns: torch.Tensor  # (N,)
    episode_ends: torch.Tensor  # (N,) bool
    episode_outcomes: list[float]


@dataclass
class _EpisodeResult:
    """Single-episode rollout payload returned by a worker.

    All tensors are CPU and shape-aligned with the serial path; per-step lists
    are kept as Python lists to avoid repeated tensor concat in the worker.
    """

    planet_feats: list[torch.Tensor]
    planet_mask: list[torch.Tensor]
    my_planet_mask: list[torch.Tensor]
    target_mask: list[torch.Tensor]
    global_feats: list[torch.Tensor]
    template_ctx: list[torch.Tensor]
    candidate_feats: list[torch.Tensor]
    candidate_mask: list[torch.Tensor]
    candidate_pid: list[torch.Tensor]
    target_slot: list[torch.Tensor]
    log1p_ships: list[torch.Tensor]
    log_probs: list[float]
    values: list[float]
    rewards: list[float]
    ends: list[bool]
    outcome: float


def _compute_gae(
    rewards: list[float],
    values: list[float],
    ends: list[bool],
    gamma: float,
    lam: float,
) -> tuple[list[float], list[float]]:
    advantages = [0.0] * len(rewards)
    last_adv = 0.0
    next_value = 0.0
    for t in reversed(range(len(rewards))):
        if ends[t]:
            next_value = 0.0
            last_adv = 0.0
        delta = rewards[t] + gamma * next_value - values[t]
        last_adv = delta + gamma * lam * last_adv
        advantages[t] = last_adv
        next_value = values[t]
    returns = [a + v for a, v in zip(advantages, values, strict=True)]
    return advantages, returns


def _run_one_episode(
    model: ActorCritic,
    opponent: OpponentAgent,
    *,
    ep_seed: int,
    seat: int,
    shaping_coef: float,
) -> _EpisodeResult:
    """Roll out one episode against `opponent` and return all transitions.

    Pure function: no mutable state escapes. Called both from the serial
    fallback and from worker processes.
    """
    episode = OrbitWarsEpisode(
        opponent=opponent, seat=seat, seed=ep_seed, shaping_coef=shaping_coef
    )
    history = HistoryState()
    result = _EpisodeResult(
        planet_feats=[],
        planet_mask=[],
        my_planet_mask=[],
        target_mask=[],
        global_feats=[],
        template_ctx=[],
        candidate_feats=[],
        candidate_mask=[],
        candidate_pid=[],
        target_slot=[],
        log1p_ships=[],
        log_probs=[],
        values=[],
        rewards=[],
        ends=[],
        outcome=0.0,
    )
    while not episode.done:
        obs = episode.current_obs()
        batch, snapshot = featurize(obs, history=history)
        with torch.no_grad():
            output = model(batch)
            action = sample_action(output, batch)
        actions_list = decode(action, snapshot, obs)
        step = episode.step(actions_list)

        result.planet_feats.append(batch.planet_feats[0])
        result.planet_mask.append(batch.planet_mask[0])
        result.my_planet_mask.append(batch.my_planet_mask[0])
        result.target_mask.append(batch.target_mask[0])
        result.global_feats.append(batch.global_feats[0])
        result.template_ctx.append(batch.template_ctx[0])
        result.candidate_feats.append(batch.candidate_feats[0])
        result.candidate_mask.append(batch.candidate_mask[0])
        result.candidate_pid.append(batch.candidate_pid[0])
        result.target_slot.append(action.target_slot)
        result.log1p_ships.append(action.log1p_ships)
        result.log_probs.append(float(action.log_prob.item()))
        result.values.append(float(output.value[0].item()))
        result.rewards.append(float(step.reward))
        result.ends.append(bool(step.done))
        result.outcome += float(step.reward)
    return result


# --------------------------------------------------------------------------
# Worker-side: rebuild ActorCritic from state_dict, run an episode, return.
#
# We send (model_config, state_dict_bytes, opponent_name, ep_args) to the
# worker — passing the live ActorCritic instance via pickle works but copies
# tensors per task, so this path uses an explicit serialization handshake.
# --------------------------------------------------------------------------


_WORKER_MODEL: ActorCritic | None = None
_WORKER_MODEL_CFG: ModelConfig | None = None


def _worker_init(model_cfg_dict: dict[str, Any], state_dict_bytes: bytes) -> None:
    """One-time per-worker init: build the model and load weights."""
    import io

    global _WORKER_MODEL, _WORKER_MODEL_CFG
    _WORKER_MODEL_CFG = ModelConfig(**model_cfg_dict)
    _WORKER_MODEL = ActorCritic(_WORKER_MODEL_CFG)
    state_dict = torch.load(io.BytesIO(state_dict_bytes), map_location="cpu")
    _WORKER_MODEL.load_state_dict(state_dict)
    _WORKER_MODEL.eval()
    # Keep BLAS single-threaded inside workers so they don't fight each other.
    torch.set_num_threads(1)


def _worker_run(
    args: tuple[str, int, int, float],
) -> _EpisodeResult:
    """Worker entry: resolve opponent by name (avoid pickling closures) and roll out."""
    from dataset.selfplay.agents import resolve as _resolve_agent

    from .env import random_noop_agent

    opponent_name, ep_seed, seat, shaping_coef = args
    if opponent_name == "random_noop":
        opp = random_noop_agent
    else:
        obj = _resolve_agent(opponent_name)
        if not callable(obj):
            raise ValueError(f"opponent {opponent_name!r} is not callable")
        opp = obj  # type: ignore[assignment]
    assert _WORKER_MODEL is not None, "worker not initialized"
    return _run_one_episode(
        _WORKER_MODEL,
        opp,
        ep_seed=ep_seed,
        seat=seat,
        shaping_coef=shaping_coef,
    )


def _serialize_model(model: ActorCritic) -> tuple[dict[str, Any], bytes]:
    """Capture model config + state_dict bytes for worker init."""
    import io

    cfg = model.cfg
    cfg_dict = {
        "hidden": cfg.hidden,
        "attn_heads": cfg.attn_heads,
        "inducing_points": cfg.inducing_points,
        "encoder_layers": cfg.encoder_layers,
        "head_dropout": cfg.head_dropout,
        "ship_log_std_init": cfg.ship_log_std_init,
        "no_op_bias": cfg.no_op_bias,
    }
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return cfg_dict, buf.getvalue()


def _resolve_opponent_name(opponent: OpponentAgent) -> str:
    """Recover the agent name so workers can rebuild it without pickling closures."""
    from dataset.selfplay.agents import AGENT_REGISTRY

    from .env import random_noop_agent

    if opponent is random_noop_agent:
        return "random_noop"
    # Match against AGENT_REGISTRY by identity (resolve() returns the same object).
    from dataset.selfplay.agents import resolve as _resolve_agent

    for name in AGENT_REGISTRY:
        try:
            if _resolve_agent(name) is opponent:
                return name
        except Exception:  # noqa: BLE001
            continue
    raise ValueError(
        "opponent is not registered in AGENT_REGISTRY; parallel rollout requires "
        "a named agent (or random_noop). Pass num_workers=1 to use a closure."
    )


def collect_rollout(
    model: ActorCritic,
    opponent: OpponentAgent,
    num_episodes: int,
    *,
    seed: int,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    shaping_coef: float = 0.001,
    seat_swap: bool = True,
    num_workers: int = 1,
) -> RolloutBatch:
    """Collect on-policy rollouts.

    `num_workers > 1` enables a multiprocessing pool; episode order is preserved
    so seat_swap semantics match the serial path. `num_workers <= 0` is treated
    as `os.cpu_count() - 1` (capped at `num_episodes`).
    """
    model.eval()

    # Resolve effective worker count.
    if num_workers <= 0:
        cpu = os.cpu_count() or 1
        num_workers = max(1, cpu - 1)
    num_workers = min(num_workers, num_episodes)

    ep_args = [
        (
            seed + ep_idx,
            (ep_idx % 2) if seat_swap else 0,
            shaping_coef,
        )
        for ep_idx in range(num_episodes)
    ]

    if num_workers <= 1:
        results = [
            _run_one_episode(
                model,
                opponent,
                ep_seed=es,
                seat=st,
                shaping_coef=sc,
            )
            for es, st, sc in ep_args
        ]
    else:
        opponent_name = _resolve_opponent_name(opponent)
        cfg_dict, sd_bytes = _serialize_model(model)
        # `spawn` keeps workers independent of the parent's import state — safer
        # than `fork` when torch is involved on macOS/Linux.
        ctx = get_context("spawn")
        with ctx.Pool(
            processes=num_workers,
            initializer=_worker_init,
            initargs=(cfg_dict, sd_bytes),
        ) as pool:
            packed = [(opponent_name, es, st, sc) for es, st, sc in ep_args]
            results = pool.map(_worker_run, packed)

    # Aggregate in ep_idx order; this preserves seat_swap semantics.
    buf: dict[str, list[torch.Tensor]] = {
        "planet_feats": [],
        "planet_mask": [],
        "my_planet_mask": [],
        "target_mask": [],
        "global_feats": [],
        "template_ctx": [],
        "candidate_feats": [],
        "candidate_mask": [],
        "candidate_pid": [],
        "target_slot": [],
        "log1p_ships": [],
    }
    log_probs: list[float] = []
    values: list[float] = []
    rewards: list[float] = []
    ends: list[bool] = []
    outcomes: list[float] = []

    for r in results:
        buf["planet_feats"].extend(r.planet_feats)
        buf["planet_mask"].extend(r.planet_mask)
        buf["my_planet_mask"].extend(r.my_planet_mask)
        buf["target_mask"].extend(r.target_mask)
        buf["global_feats"].extend(r.global_feats)
        buf["template_ctx"].extend(r.template_ctx)
        buf["candidate_feats"].extend(r.candidate_feats)
        buf["candidate_mask"].extend(r.candidate_mask)
        buf["candidate_pid"].extend(r.candidate_pid)
        buf["target_slot"].extend(r.target_slot)
        buf["log1p_ships"].extend(r.log1p_ships)
        log_probs.extend(r.log_probs)
        values.extend(r.values)
        rewards.extend(r.rewards)
        ends.extend(r.ends)
        outcomes.append(r.outcome)

    advantages, returns = _compute_gae(rewards, values, ends, gamma, gae_lambda)

    def _stack(name: str) -> torch.Tensor:
        items = buf[name]
        if not items:
            raise RuntimeError(f"no transitions collected for {name}")
        return torch.stack(items, dim=0)

    return RolloutBatch(
        planet_feats=_stack("planet_feats"),
        planet_mask=_stack("planet_mask"),
        my_planet_mask=_stack("my_planet_mask"),
        target_mask=_stack("target_mask"),
        global_feats=_stack("global_feats"),
        template_ctx=_stack("template_ctx"),
        candidate_feats=_stack("candidate_feats"),
        candidate_mask=_stack("candidate_mask"),
        candidate_pid=_stack("candidate_pid"),
        target_slot=_stack("target_slot"),
        log1p_ships=_stack("log1p_ships"),
        log_probs=torch.tensor(log_probs, dtype=torch.float32),
        values=torch.tensor(values, dtype=torch.float32),
        rewards=torch.tensor(rewards, dtype=torch.float32),
        advantages=torch.tensor(advantages, dtype=torch.float32),
        returns=torch.tensor(returns, dtype=torch.float32),
        episode_ends=torch.tensor(ends, dtype=torch.bool),
        episode_outcomes=outcomes,
    )


__all__ = ["RolloutBatch", "collect_rollout"]
