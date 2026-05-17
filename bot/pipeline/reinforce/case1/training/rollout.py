"""On-policy rollout collection for PPO with per_planet policy.

Mirrors case0's rollout.py but stores per_planet-shaped actions:
  target_slot  (P,) long  — including NO_OP_INDEX
  log1p_ships  (P,) float — Gaussian sample for ships head
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..policy.decoder import decode
from ..policy.featurizer import HistoryState, featurize
from ..policy.model import ActorCritic
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
) -> RolloutBatch:
    model.eval()
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

    for ep_idx in range(num_episodes):
        ep_seed = seed + ep_idx
        seat = ep_idx % 2 if seat_swap else 0
        episode = OrbitWarsEpisode(
            opponent=opponent, seat=seat, seed=ep_seed, shaping_coef=shaping_coef
        )
        history = HistoryState()
        ep_reward = 0.0
        while not episode.done:
            obs = episode.current_obs()
            batch, snapshot = featurize(obs, history=history)
            with torch.no_grad():
                output = model(batch)
                action = sample_action(output, batch)
            actions_list = decode(action, snapshot, obs)
            step = episode.step(actions_list)

            buf["planet_feats"].append(batch.planet_feats[0])
            buf["planet_mask"].append(batch.planet_mask[0])
            buf["my_planet_mask"].append(batch.my_planet_mask[0])
            buf["target_mask"].append(batch.target_mask[0])
            buf["global_feats"].append(batch.global_feats[0])
            buf["template_ctx"].append(batch.template_ctx[0])
            buf["candidate_feats"].append(batch.candidate_feats[0])
            buf["candidate_mask"].append(batch.candidate_mask[0])
            buf["candidate_pid"].append(batch.candidate_pid[0])
            buf["target_slot"].append(action.target_slot)
            buf["log1p_ships"].append(action.log1p_ships)
            log_probs.append(float(action.log_prob.item()))
            values.append(float(output.value[0].item()))
            rewards.append(float(step.reward))
            ends.append(bool(step.done))
            ep_reward += float(step.reward)
        outcomes.append(ep_reward)

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
