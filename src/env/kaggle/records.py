"""Kaggle EpisodeService レスポンス → MatchRecord 変換。

Kaggle `/api/i/competitions.EpisodeService/ListEpisodes` のレスポンス形式:

- `id`: episode_id
- `createTime` / `endTime`: ISO 8601
- `state`: "COMPLETED" / "ERRORED" など (episode レベル)
- `type`: "EPISODE_TYPE_PUBLIC" など
- `agents[]`:
    - `id`, `submissionId`, `index` (player index)
    - `reward`: float（勝者判定・スコア用）
    - `initialScore`, `updatedScore`: 提出者 rating（μ）

別途 `submissions[]` / `teams[]` を引いて teamId を補完する。
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from env.kaggle.types import EpisodeMeta
from env.types import (
    SOURCE_KAGGLE,
    AgentKaggleMeta,
    AgentTiming,
    MatchRecord,
)

MODE_BY_AGENT_COUNT: dict[int, str] = {2: "1v1", 4: "ffa4"}
FAILED_EPISODE_STATES: frozenset[str] = frozenset(
    {"ERRORED", "ERROR", "INVALID", "FAILED"}
)


def infer_mode(agent_count: int) -> str:
    if agent_count not in MODE_BY_AGENT_COUNT:
        raise ValueError(f"unsupported agent count: {agent_count}")
    return MODE_BY_AGENT_COUNT[agent_count]


def is_failed_episode(raw: dict[str, Any]) -> bool:
    """Episode レベルの state でフィルタ。"""

    return str(raw.get("state", "")).upper() in FAILED_EPISODE_STATES


def parse_episode_meta(raw: dict[str, Any]) -> EpisodeMeta:
    agents_raw = raw.get("agents") or []
    if not isinstance(agents_raw, list):
        raise ValueError("episode.agents is not a list")
    agents = tuple(a for a in agents_raw if isinstance(a, dict))
    mode = infer_mode(len(agents))
    return EpisodeMeta(
        episode_id=int(raw["id"]),
        mode=mode,
        create_time=str(raw.get("createTime") or ""),
        end_time=str(raw.get("endTime") or ""),
        agents=agents,
    )


def _elapsed_seconds(create_time: str, end_time: str) -> float:
    if not create_time or not end_time:
        return 0.0
    try:
        start = dt.datetime.fromisoformat(
            create_time.replace("Z", "+00:00").replace("+00:00+00:00", "+00:00")
        )
        end = dt.datetime.fromisoformat(
            end_time.replace("Z", "+00:00").replace("+00:00+00:00", "+00:00")
        )
    except ValueError:
        return 0.0
    return max((end - start).total_seconds(), 0.0)


def _agent_label(agent: dict[str, Any]) -> str:
    submission_id = agent.get("submissionId")
    if submission_id is not None:
        return f"kaggle_sub_{submission_id}"
    return "kaggle_unknown"


def _ordered_agents(
    agents: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    """`index` フィールドに沿って player 順序を正規化。"""

    return tuple(sorted(agents, key=lambda a: int(a.get("index") or 0)))


def _resolve_outcome(agents: tuple[dict[str, Any], ...]) -> tuple[int, bool]:
    """勝者 index と draw フラグを返す（reward の最大値で判定）。"""

    rewards = [
        float(a["reward"]) if a.get("reward") is not None else float("-inf")
        for a in agents
    ]
    if not rewards or all(r == float("-inf") for r in rewards):
        return -1, True
    top = max(rewards)
    leaders = [i for i, r in enumerate(rewards) if r == top and r != float("-inf")]
    if len(leaders) == 1:
        return leaders[0], False
    return -1, True


def _agent_score(agent: dict[str, Any]) -> int:
    for key in ("finalScore", "score", "reward"):
        value = agent.get(key)
        if value is None:
            continue
        try:
            return int(round(float(value)))
        except (TypeError, ValueError):
            continue
    return 0


def _kaggle_meta(
    agent: dict[str, Any],
    *,
    team_id_by_submission: dict[int, int] | None = None,
) -> AgentKaggleMeta:
    submission_id = int(agent.get("submissionId") or 0)
    team_id = 0
    if team_id_by_submission and submission_id in team_id_by_submission:
        team_id = team_id_by_submission[submission_id]
    return AgentKaggleMeta(
        submission_id=submission_id,
        team_id=team_id,
        rating_mu=float(agent.get("updatedScore") or 0.0),
        rating_sigma=0.0,
        state="",
    )


def build_match_record(
    meta: EpisodeMeta,
    *,
    run_id: str,
    scraped_at: str,
    team_id_by_submission: dict[int, int] | None = None,
) -> MatchRecord:
    agents = _ordered_agents(meta.agents)
    winner, draw = _resolve_outcome(agents)
    timings = tuple(AgentTiming(timeouts=0, p50=0.0, p95=0.0, max=0.0) for _ in agents)
    started_at = meta.create_time or scraped_at
    elapsed_sec = _elapsed_seconds(meta.create_time, meta.end_time)

    return MatchRecord(
        match_id=f"kaggle_ep_{meta.episode_id}",
        run_id=run_id,
        mode=meta.mode,
        seed=-1,
        started_at=started_at,
        elapsed_sec=elapsed_sec,
        turns=0,
        winner=winner,
        draw=draw,
        agent_names=tuple(_agent_label(a) for a in agents),
        agent_versions=tuple(str(a.get("submissionId") or "") for a in agents),
        agent_scores=tuple(_agent_score(a) for a in agents),
        agent_timings=timings,
        replay_path=f"replays/kaggle_ep_{meta.episode_id}.json.gz",
        git_sha="",
        source=SOURCE_KAGGLE,
        episode_id=meta.episode_id,
        scraped_at=scraped_at,
        agent_kaggle_meta=tuple(
            _kaggle_meta(a, team_id_by_submission=team_id_by_submission) for a in agents
        ),
    )


def extract_submission_team_map(listing: dict[str, Any]) -> dict[int, int]:
    """ListEpisodes レスポンスから submissionId → teamId マップを作る。"""

    submissions = listing.get("submissions") or []
    if not isinstance(submissions, list):
        return {}
    out: dict[int, int] = {}
    for s in submissions:
        if not isinstance(s, dict):
            continue
        sid = s.get("id")
        tid = s.get("teamId")
        if sid is not None and tid is not None:
            out[int(sid)] = int(tid)
    return out
