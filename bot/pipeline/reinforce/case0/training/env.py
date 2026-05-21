"""Single-episode env wrapper around kaggle_environments orbit_wars.

Surfaces a turn-by-turn loop where the learning agent observes from a fixed
seat, the opponent agent is supplied externally, and per-turn shaping +
terminal win/loss reward are returned.
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
    actions: list[list[int | float]]  # action emitted by the learner
    reward: float
    done: bool


def _ship_totals(obs: dict[str, Any], seat: int) -> tuple[float, float]:
    raw_planets = list(obs.get("planets") or [])
    mine = 0.0
    enemy = 0.0
    for row in raw_planets:
        _, owner, _, _, _, ships, _ = row
        owner_i = int(owner)
        if owner_i == seat:
            mine += float(ships)
        elif owner_i != -1:
            enemy += float(ships)
    return mine, enemy


def random_agent(obs: Any) -> list[list[int | float]]:
    """Opponent that fires nothing (worst-case curriculum baseline)."""
    del obs
    return []


class OrbitWarsEpisode:
    """Iterator-style wrapper. Each `step(actions)` returns the next observation
    for the learner along with shaping + done flag.

    Reward design (kept intentionally minimal for the baseline):
      - terminal: +1 win, -1 loss, 0 tie (based on env reward).
      - shaping: 0.001 * Δ(my_ships - enemy_ships) per turn — encourages
        accumulating production advantage without dominating the win signal.
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
        if self.done:
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


__all__ = ["OrbitWarsEpisode", "EpisodeStep", "random_agent", "OpponentAgent"]
