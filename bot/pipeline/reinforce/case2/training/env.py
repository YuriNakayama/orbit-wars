"""Single-episode env wrapper around kaggle_environments orbit_wars.

Mirrors `reinforce/case0/training/env.py` so the rollout loop stays familiar.
Reward = ±1 terminal win/loss + small shaping on ship-count differential.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from env.orbit_wars import make_orbit_wars_env

OpponentAgent = Callable[[Any], list[list[int | float]]]


@dataclass
class EpisodeStep:
    obs: dict[str, Any]
    actions: list[list[int | float]]
    reward: float
    done: bool


def _ship_totals(obs: dict[str, Any], seat: int) -> tuple[float, float]:
    mine = 0.0
    enemy = 0.0
    for row in list(obs.get("planets") or []):
        _, owner, _, _, _, ships, _ = row
        owner_i = int(owner)
        if owner_i == seat:
            mine += float(ships)
        elif owner_i != -1:
            enemy += float(ships)
    return mine, enemy


def random_noop_agent(obs: Any) -> list[list[int | float]]:
    """Opponent that fires nothing (used as the smoke curriculum baseline)."""
    del obs
    return []


class OrbitWarsEpisode:
    """Iterator-style wrapper over kaggle_environments orbit_wars.

    Reward design:
      - terminal: +1 win, -1 loss, 0 tie (env reward comparison)
      - shaping: shaping_coef * Δ(my_ships − enemy_ships) per turn
    """

    def __init__(
        self,
        opponent: OpponentAgent,
        seat: int = 0,
        seed: int | None = None,
        shaping_coef: float = 0.001,
    ) -> None:
        if seat not in (0, 1):
            raise ValueError(f"seat must be 0 or 1, got {seat!r}")
        self.seat = seat
        self.opponent = opponent
        self.shaping_coef = shaping_coef
        self.env = make_orbit_wars_env(agents=2, seed=seed)
        self._prev_ships_diff: float | None = None

    @property
    def done(self) -> bool:
        return bool(self.env.done)

    def current_obs(self) -> dict[str, Any]:
        return dict(self.env.steps[-1][self.seat]["observation"])

    def opponent_obs(self) -> dict[str, Any]:
        return dict(self.env.steps[-1][1 - self.seat]["observation"])

    def step(self, learner_actions: list[list[int | float]]) -> EpisodeStep:
        if self.done:
            raise RuntimeError("step() called on a finished episode")
        opp_actions = self.opponent(self.opponent_obs()) or []
        if self.seat == 0:
            actions = [learner_actions, opp_actions]
        else:
            actions = [opp_actions, learner_actions]
        self.env.step(actions)
        new_obs = self.current_obs()
        mine, enemy = _ship_totals(new_obs, self.seat)
        diff = mine - enemy
        shaping = 0.0
        if self._prev_ships_diff is not None:
            shaping = self.shaping_coef * (diff - self._prev_ships_diff)
        self._prev_ships_diff = diff

        reward = shaping
        # bool() で property を local 化し、line 79 の type narrowing を断ち切る
        # (mypy が self.done を False に固定推論することで unreachable と誤判定)
        episode_done = bool(self.done)
        if episode_done:
            rewards = [s.get("reward", 0) or 0 for s in self.env.steps[-1]]
            r_self = float(rewards[self.seat])
            r_opp = float(rewards[1 - self.seat])
            if r_self > r_opp:
                reward += 1.0
            elif r_self < r_opp:
                reward -= 1.0
        return EpisodeStep(
            obs=new_obs,
            actions=learner_actions,
            reward=reward,
            done=self.done,
        )


__all__ = ["OrbitWarsEpisode", "EpisodeStep", "random_noop_agent", "OpponentAgent"]
