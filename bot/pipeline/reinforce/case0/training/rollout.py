"""Collect on-policy rollouts for PPO.

One rollout = `num_episodes` complete games against the configured opponent.
We store transitions as flat tensors (one row per learner turn) and let PPO
operate on them in minibatches.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..policy.decoder import decode
from ..policy.featurizer import MAX_PLANETS, featurize
from ..policy.model import ActorCritic
from ..policy.sampling import sample_action
from .env import OpponentAgent, OrbitWarsEpisode


@dataclass
class RolloutBatch:
    planet_feats: torch.Tensor  # (N, P, F)
    planet_mask: torch.Tensor  # (N, P)
    my_planet_mask: torch.Tensor  # (N, P)
    target_mask: torch.Tensor  # (N, P)
    global_feats: torch.Tensor  # (N, G)
    from_choice: torch.Tensor  # (N, P) bool
    target_choice: torch.Tensor  # (N, P) long
    ships_choice: torch.Tensor  # (N, P) long
    log_probs: torch.Tensor  # (N,)
    values: torch.Tensor  # (N,)
    rewards: torch.Tensor  # (N,)
    advantages: torch.Tensor  # (N,)
    returns: torch.Tensor  # (N,)
    episode_ends: torch.Tensor  # (N,) bool — last transition of an episode
    episode_outcomes: list[float]  # one entry per finished episode: +1/0/-1


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
    feats_planet: list[torch.Tensor] = []
    feats_pmask: list[torch.Tensor] = []
    feats_mymask: list[torch.Tensor] = []
    feats_tmask: list[torch.Tensor] = []
    feats_global: list[torch.Tensor] = []
    from_choices: list[torch.Tensor] = []
    target_choices: list[torch.Tensor] = []
    ships_choices: list[torch.Tensor] = []
    log_probs: list[float] = []
    values: list[float] = []
    rewards: list[float] = []
    ends: list[bool] = []
    outcomes: list[float] = []

    for ep_idx in range(num_episodes):
        ep_seed = seed + ep_idx
        seat = ep_idx % 2 if seat_swap else 0
        episode = OrbitWarsEpisode(
            opponent=opponent,
            seat=seat,
            seed=ep_seed,
            shaping_coef=shaping_coef,
        )
        ep_reward_sum = 0.0
        while not episode.done:
            obs = episode.current_obs()
            batch, snapshot = featurize(obs)
            with torch.no_grad():
                output = model(batch)
                action = sample_action(output, batch)
            actions_list = decode(action, snapshot, obs)
            step = episode.step(actions_list)

            feats_planet.append(batch.planet_feats[0])
            feats_pmask.append(batch.planet_mask[0])
            feats_mymask.append(batch.my_planet_mask[0])
            feats_tmask.append(batch.target_mask[0])
            feats_global.append(batch.global_feats[0])
            from_choices.append(action.from_choice)
            target_choices.append(action.target_choice)
            ships_choices.append(action.ships_choice)
            log_probs.append(float(action.log_prob.item()))
            values.append(float(output.value[0].item()))
            rewards.append(float(step.reward))
            ends.append(bool(step.done))
            ep_reward_sum += float(step.reward)
        outcomes.append(ep_reward_sum)

    advantages, returns = _compute_gae(rewards, values, ends, gamma, gae_lambda)

    def _stack_or_empty(
        items: list[torch.Tensor], shape: tuple[int, ...]
    ) -> torch.Tensor:
        if not items:
            return torch.zeros(shape)
        return torch.stack(items, dim=0)

    p = MAX_PLANETS
    planet_feats = _stack_or_empty(feats_planet, (0, p, 0))
    planet_mask = _stack_or_empty(feats_pmask, (0, p))
    my_planet_mask = _stack_or_empty(feats_mymask, (0, p))
    target_mask = _stack_or_empty(feats_tmask, (0, p))
    global_feats = _stack_or_empty(feats_global, (0, 0))
    from_choice_t = _stack_or_empty(from_choices, (0, p))
    target_choice_t = _stack_or_empty(target_choices, (0, p))
    ships_choice_t = _stack_or_empty(ships_choices, (0, p))

    return RolloutBatch(
        planet_feats=planet_feats,
        planet_mask=planet_mask,
        my_planet_mask=my_planet_mask,
        target_mask=target_mask,
        global_feats=global_feats,
        from_choice=from_choice_t,
        target_choice=target_choice_t,
        ships_choice=ships_choice_t,
        log_probs=torch.tensor(log_probs, dtype=torch.float32),
        values=torch.tensor(values, dtype=torch.float32),
        rewards=torch.tensor(rewards, dtype=torch.float32),
        advantages=torch.tensor(advantages, dtype=torch.float32),
        returns=torch.tensor(returns, dtype=torch.float32),
        episode_ends=torch.tensor(ends, dtype=torch.bool),
        episode_outcomes=outcomes,
    )


__all__ = ["RolloutBatch", "collect_rollout"]
