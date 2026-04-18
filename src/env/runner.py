"""Dispatch match specs to workers and persist the results.

Keeps IO (recorder) on the main process: workers return replay bytes
so the filesystem is only touched here.
"""

from __future__ import annotations

import datetime as dt
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.progress import Progress, TaskID

from env import recorder
from env.agents import agent_version
from env.executor import run_one_match
from env.types import AgentSpec, AgentTiming, MatchRecord, MatchSpec

MODE_NUM_AGENTS: dict[str, int] = {"1v1": 2, "ffa4": 4}


@dataclass(frozen=True)
class RunSpec:
    agents: tuple[str, ...]
    mode: str
    episodes: int
    seed: int
    parallel: int
    save_replay: bool
    data_root: Path


def _validate(spec: RunSpec) -> None:
    if spec.mode not in MODE_NUM_AGENTS:
        raise ValueError(f"unknown mode: {spec.mode!r}")
    expected = MODE_NUM_AGENTS[spec.mode]
    if len(spec.agents) != expected:
        raise ValueError(
            f"mode {spec.mode} needs {expected} agents, got {len(spec.agents)}"
        )
    if spec.episodes <= 0:
        raise ValueError(f"episodes must be positive, got {spec.episodes}")
    if spec.parallel <= 0:
        raise ValueError(f"parallel must be positive, got {spec.parallel}")


def _build_run_id() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _make_match_specs(spec: RunSpec, run_id: str) -> list[MatchSpec]:
    version = agent_version()
    agent_specs = tuple(
        AgentSpec(
            name=name,
            version=version,
            callable_path=name,
        )
        for name in spec.agents
    )
    specs: list[MatchSpec] = []
    for i in range(spec.episodes):
        match_id = f"{run_id}_{spec.mode}_seed{spec.seed + i}"
        specs.append(
            MatchSpec(
                match_id=match_id,
                run_id=run_id,
                mode=spec.mode,
                seed=spec.seed + i,
                agents=agent_specs,
                save_replay=spec.save_replay,
                data_root=str(spec.data_root),
            )
        )
    return specs


def _record_from_dict(row: dict[str, Any]) -> MatchRecord:
    timings = tuple(
        t if isinstance(t, AgentTiming) else AgentTiming(**t)
        for t in row["agent_timings"]
    )
    return MatchRecord(
        match_id=str(row["match_id"]),
        run_id=str(row["run_id"]),
        mode=str(row["mode"]),
        seed=int(row["seed"]),
        started_at=str(row["started_at"]),
        elapsed_sec=float(row["elapsed_sec"]),
        turns=int(row["turns"]),
        winner=int(row["winner"]),
        draw=bool(row["draw"]),
        agent_names=tuple(row["agent_names"]),
        agent_versions=tuple(row["agent_versions"]),
        agent_scores=tuple(row["agent_scores"]),
        agent_timings=timings,
        replay_path=str(row["replay_path"]),
        git_sha=str(row["git_sha"]),
    )


def _iter_results(
    specs: list[MatchSpec], parallel: int
) -> list[tuple[dict[str, Any], bytes | None]]:
    if parallel == 1:
        return [run_one_match(s) for s in specs]
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=parallel) as pool:
        return list(pool.imap_unordered(run_one_match, specs))


def run_episodes(spec: RunSpec, progress: Progress | None = None) -> list[MatchRecord]:
    _validate(spec)
    run_id = _build_run_id()
    match_specs = _make_match_specs(spec, run_id)

    task_id: TaskID | None = None
    if progress is not None:
        task_id = progress.add_task(f"run {spec.mode}", total=spec.episodes)

    results = _iter_results(match_specs, spec.parallel)

    records: list[MatchRecord] = []
    replay_bytes: dict[str, bytes] = {}
    for row, payload in results:
        record = _record_from_dict(row)
        records.append(record)
        if payload is not None:
            replay_bytes[record.match_id] = payload
        if progress is not None and task_id is not None:
            progress.advance(task_id)

    recorder.write_run(records, replay_bytes, spec.data_root)
    return records
