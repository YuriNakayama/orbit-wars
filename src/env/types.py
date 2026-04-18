"""Domain types for the evaluation framework.

All dataclasses are frozen to discourage mutation (backend.md: "NEVER mutate").
Flat dict serialization is used for Parquet compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MAX_PLAYERS = 4
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class AgentSpec:
    name: str
    version: str
    callable_path: str


@dataclass(frozen=True)
class MatchSpec:
    match_id: str
    run_id: str
    mode: str
    seed: int
    agents: tuple[AgentSpec, ...]
    save_replay: bool
    data_root: str


@dataclass(frozen=True)
class AgentTiming:
    timeouts: int
    p50: float
    p95: float
    max: float


@dataclass(frozen=True)
class MatchRecord:
    match_id: str
    run_id: str
    mode: str
    seed: int
    started_at: str
    elapsed_sec: float
    turns: int
    winner: int
    draw: bool
    agent_names: tuple[str, ...]
    agent_versions: tuple[str, ...]
    agent_scores: tuple[int, ...]
    agent_timings: tuple[AgentTiming, ...]
    replay_path: str
    git_sha: str

    def to_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "match_id": self.match_id,
            "run_id": self.run_id,
            "mode": self.mode,
            "seed": self.seed,
            "started_at": self.started_at,
            "elapsed_sec": self.elapsed_sec,
            "turns": self.turns,
            "winner": self.winner,
            "draw": self.draw,
            "replay_path": self.replay_path,
            "git_sha": self.git_sha,
        }
        for idx in range(MAX_PLAYERS):
            name = self.agent_names[idx] if idx < len(self.agent_names) else ""
            version = self.agent_versions[idx] if idx < len(self.agent_versions) else ""
            score = self.agent_scores[idx] if idx < len(self.agent_scores) else 0
            timing = (
                self.agent_timings[idx]
                if idx < len(self.agent_timings)
                else AgentTiming(timeouts=0, p50=0.0, p95=0.0, max=0.0)
            )
            row[f"agent_{idx}_name"] = name
            row[f"agent_{idx}_version"] = version
            row[f"agent_{idx}_score"] = score
            row[f"agent_{idx}_timeouts"] = timing.timeouts
            row[f"agent_{idx}_turn_p50"] = timing.p50
            row[f"agent_{idx}_turn_p95"] = timing.p95
            row[f"agent_{idx}_turn_max"] = timing.max
        return row

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> MatchRecord:
        names: list[str] = []
        versions: list[str] = []
        scores: list[int] = []
        timings: list[AgentTiming] = []
        for idx in range(MAX_PLAYERS):
            name = row.get(f"agent_{idx}_name", "")
            if not name:
                continue
            names.append(name)
            versions.append(row.get(f"agent_{idx}_version", ""))
            scores.append(int(row.get(f"agent_{idx}_score", 0)))
            timings.append(
                AgentTiming(
                    timeouts=int(row.get(f"agent_{idx}_timeouts", 0)),
                    p50=float(row.get(f"agent_{idx}_turn_p50", 0.0)),
                    p95=float(row.get(f"agent_{idx}_turn_p95", 0.0)),
                    max=float(row.get(f"agent_{idx}_turn_max", 0.0)),
                )
            )
        return cls(
            match_id=str(row["match_id"]),
            run_id=str(row["run_id"]),
            mode=str(row["mode"]),
            seed=int(row["seed"]),
            started_at=str(row["started_at"]),
            elapsed_sec=float(row["elapsed_sec"]),
            turns=int(row["turns"]),
            winner=int(row["winner"]),
            draw=bool(row["draw"]),
            agent_names=tuple(names),
            agent_versions=tuple(versions),
            agent_scores=tuple(scores),
            agent_timings=tuple(timings),
            replay_path=str(row.get("replay_path", "")),
            git_sha=str(row.get("git_sha", "")),
        )
