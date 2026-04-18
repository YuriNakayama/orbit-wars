"""Rich-table summary of a completed run."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from rich.console import Console
from rich.table import Table

from env.types import MatchRecord


@dataclass(frozen=True)
class AgentStats:
    name: str
    wins: int
    games: int
    draws: int
    timeouts: int
    turn_p95_max: float
    avg_turns: float

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0

    @property
    def timeout_rate(self) -> float:
        return self.timeouts / self.games if self.games else 0.0


def aggregate(records: list[MatchRecord]) -> list[AgentStats]:
    wins: dict[str, int] = defaultdict(int)
    games: dict[str, int] = defaultdict(int)
    draws: dict[str, int] = defaultdict(int)
    timeouts: dict[str, int] = defaultdict(int)
    p95_max: dict[str, float] = defaultdict(float)
    turns_sum: dict[str, int] = defaultdict(int)

    for record in records:
        for idx, name in enumerate(record.agent_names):
            games[name] += 1
            turns_sum[name] += record.turns
            if record.draw:
                draws[name] += 1
            elif record.winner == idx:
                wins[name] += 1
            timing = record.agent_timings[idx]
            timeouts[name] += timing.timeouts
            p95_max[name] = max(p95_max[name], timing.p95)

    stats = []
    for name in sorted(games):
        g = games[name]
        stats.append(
            AgentStats(
                name=name,
                wins=wins[name],
                games=g,
                draws=draws[name],
                timeouts=timeouts[name],
                turn_p95_max=p95_max[name],
                avg_turns=turns_sum[name] / g if g else 0.0,
            )
        )
    return stats


def build_table(records: list[MatchRecord], title: str = "Summary") -> Table:
    table = Table(title=title)
    table.add_column("agent")
    table.add_column("wins", justify="right")
    table.add_column("games", justify="right")
    table.add_column("win_rate", justify="right")
    table.add_column("draws", justify="right")
    table.add_column("avg_turns", justify="right")
    table.add_column("timeouts", justify="right")
    table.add_column("turn_p95", justify="right")

    for stat in aggregate(records):
        table.add_row(
            stat.name,
            str(stat.wins),
            str(stat.games),
            f"{stat.win_rate:.1%}",
            str(stat.draws),
            f"{stat.avg_turns:.1f}",
            str(stat.timeouts),
            f"{stat.turn_p95_max:.3f}s",
        )
    return table


def summarize(
    records: list[MatchRecord],
    console: Console | None = None,
    title: str = "Summary",
) -> None:
    target = console or Console()
    target.print(build_table(records, title=title))
