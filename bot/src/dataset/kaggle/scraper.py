"""Kaggle episode scraper orchestration.

leaderboard → team → publicLeaderboardSubmissionId → ListEpisodes → GetEpisodeReplay
→ records/replay 永続化の全体フロー。`ScrapeSpec` に設定を集約し、
`run(...)` が `ScrapeResult` を返す。`dry_run=True` では FS 書き込みを skip する。

実行は 2 phase 構成:

1. Phase 1 (plan): leaderboard / team / list_episodes を走査し、既取得 / mode 不一致 /
   失敗 episode を除外したうえで「実際に replay 取得すべき episode リスト」を確定する。
   replay は呼ばずレートも軽い (top×2 calls)。
2. Phase 2 (fetch): plan で確定した episode に対してのみ replay 取得 + 永続化。
   `[i/total]` 形式で進捗ログを出力。

差分ダウンロードは `state.existing_episode_ids` で読み込んだ `seen_ids` に含まれる
episode_id を Phase 1 段階で除外することで実現している。
"""

from __future__ import annotations

import gzip
import logging
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import requests

from dataset.kaggle import client, leaderboard, records, state
from dataset.kaggle.rate_limit import TokenBucket
from dataset.kaggle.types import ScrapeSpec, TeamRank
from dataset.schema import MatchRecord
from dataset.storage import recorder

logger = logging.getLogger(__name__)

LeaderboardFetcher = Callable[[int], list[TeamRank]]
TeamFetcher = Callable[[requests.Session, int], dict[str, Any]]
EpisodesLister = Callable[[requests.Session, int], dict[str, Any]]
ReplayFetcher = Callable[[requests.Session, int], dict[str, Any]]


@dataclass(frozen=True)
class ScrapeResult:
    run_id: str
    teams_scanned: int
    teams_without_submission: int
    episodes_considered: int
    episodes_skipped_existing: int
    episodes_skipped_failed: int
    episodes_skipped_mode: int
    episodes_planned: int
    episodes_fetched: int
    episodes_failed: int
    records_written: int
    replays_written: int
    dry_run: bool


@dataclass(frozen=True)
class _PlannedEpisode:
    episode_id: int
    raw: dict[str, Any]
    team_id_by_submission: dict[int, int]


@dataclass
class _Accumulator:
    buffered_records: list[MatchRecord] = field(default_factory=list)
    buffered_replays: dict[str, bytes] = field(default_factory=dict)
    teams_scanned: int = 0
    teams_without_submission: int = 0
    episodes_considered: int = 0
    episodes_skipped_existing: int = 0
    episodes_skipped_failed: int = 0
    episodes_skipped_mode: int = 0
    episodes_fetched: int = 0
    episodes_failed: int = 0


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def _default_rate_limit() -> TokenBucket:
    return TokenBucket(capacity=60, window_sec=60.0)


def _flush(acc: _Accumulator, spec: ScrapeSpec) -> tuple[int, int]:
    if spec.dry_run:
        return 0, 0
    if not acc.buffered_records and not acc.buffered_replays:
        return 0, 0
    recorder.write_records(acc.buffered_records, spec.data_root)
    for match_id, payload in acc.buffered_replays.items():
        recorder.write_replay(match_id, payload, spec.data_root)
    written_records = len(acc.buffered_records)
    written_replays = len(acc.buffered_replays)
    acc.buffered_records = []
    acc.buffered_replays = {}
    return written_records, written_replays


def _filter_episode(
    *,
    raw: dict[str, Any],
    spec: ScrapeSpec,
    seen_ids: set[int],
    acc: _Accumulator,
) -> int | None:
    """Phase 1 用フィルタ。replay 取得対象なら episode_id を返し、skip なら None。"""

    acc.episodes_considered += 1
    episode_id = int(raw.get("id") or 0)
    if episode_id <= 0:
        acc.episodes_failed += 1
        return None
    if episode_id in seen_ids:
        acc.episodes_skipped_existing += 1
        return None

    agents = raw.get("agents") or []
    if not isinstance(agents, list):
        acc.episodes_failed += 1
        return None

    if not spec.include_failed and records.is_failed_episode(raw):
        acc.episodes_skipped_failed += 1
        return None

    try:
        meta = records.parse_episode_meta(raw)
    except (ValueError, KeyError, TypeError) as exc:
        logger.warning("parse_episode_meta failed for %s: %s", episode_id, exc)
        acc.episodes_failed += 1
        return None

    if meta.mode not in spec.modes:
        acc.episodes_skipped_mode += 1
        return None

    return episode_id


@dataclass(frozen=True)
class _FetchedEpisode:
    """Phase 2 worker が main thread に返す結果。

    `record` / `replay_payload` が None の場合は何らかの理由でスキップ
    (parse 失敗 / API エラー)。`failed=True` のとき failed カウンタに加算。
    """

    episode_id: int
    record: MatchRecord | None
    replay_payload: bytes | None
    failed: bool


def _fetch_planned_episode_io(
    *,
    session: requests.Session,
    planned: _PlannedEpisode,
    spec: ScrapeSpec,
    run_id: str,
    rate_limit: TokenBucket,
    replay_fetcher: ReplayFetcher,
) -> _FetchedEpisode:
    """Pure I/O: replay 取得 + record 構築。スレッドから安全に呼べる。

    state mutation (acc / seen_ids) は呼び出し側 (main thread) で行う。
    """

    episode_id = planned.episode_id
    try:
        meta = records.parse_episode_meta(planned.raw)
    except (ValueError, KeyError, TypeError) as exc:
        logger.warning("parse_episode_meta failed for %s: %s", episode_id, exc)
        return _FetchedEpisode(episode_id, None, None, failed=True)

    with rate_limit.acquire():
        try:
            replay_resp = replay_fetcher(session, episode_id)
        except client.KaggleEpisodeError as exc:
            logger.warning("get_episode_replay failed for %s: %s", episode_id, exc)
            return _FetchedEpisode(episode_id, None, None, failed=True)

    record = records.build_match_record(
        meta,
        run_id=run_id,
        scraped_at=_utc_now_iso(),
        team_id_by_submission=planned.team_id_by_submission,
    )
    if spec.dry_run:
        return _FetchedEpisode(episode_id, record, None, failed=False)
    payload_str = client.extract_replay_json(replay_resp)
    payload = gzip.compress(payload_str.encode("utf-8"))
    return _FetchedEpisode(episode_id, record, payload, failed=False)


def _apply_fetched(
    fetched: _FetchedEpisode,
    *,
    spec: ScrapeSpec,
    acc: _Accumulator,
    seen_ids: set[int],
) -> None:
    """Worker から返った結果を main thread の buffer に反映する。"""

    if fetched.failed:
        acc.episodes_failed += 1
        return
    if fetched.record is None:
        return
    seen_ids.add(fetched.episode_id)
    acc.episodes_fetched += 1
    if spec.dry_run:
        return
    acc.buffered_records.append(fetched.record)
    if fetched.replay_payload is not None:
        acc.buffered_replays[fetched.record.match_id] = fetched.replay_payload


def _lookup_submission_id(
    session: requests.Session, team_id: int, team_fetcher: TeamFetcher
) -> int | None:
    try:
        team = team_fetcher(session, team_id)
    except client.KaggleEpisodeError as exc:
        logger.warning("get_team failed for %s: %s", team_id, exc)
        return None
    sid = team.get("publicLeaderboardSubmissionId")
    if sid is None:
        return None
    try:
        return int(sid)
    except (TypeError, ValueError):
        return None


def _plan(
    *,
    session: requests.Session,
    spec: ScrapeSpec,
    seen_ids: set[int],
    acc: _Accumulator,
    rate_limit: TokenBucket,
    leaderboard_fetcher: LeaderboardFetcher,
    team_fetcher: TeamFetcher,
    episodes_lister: EpisodesLister,
) -> list[_PlannedEpisode]:
    """Phase 1: 取得予定 episode を確定する (replay 取得は行わない)。"""

    teams = leaderboard_fetcher(spec.top)
    total_teams = len(teams)
    logger.info(
        "scrape phase=plan top=%d modes=%s limit_per_team=%s seen_existing=%d",
        spec.top,
        ",".join(spec.modes),
        spec.limit_per_team if spec.limit_per_team is not None else "all",
        len(seen_ids),
    )
    plan: list[_PlannedEpisode] = []
    started = time.monotonic()

    for team_idx, team in enumerate(teams, start=1):
        acc.teams_scanned += 1
        with rate_limit.acquire():
            submission_id = _lookup_submission_id(session, team.team_id, team_fetcher)
        if submission_id is None:
            acc.teams_without_submission += 1
            logger.info(
                "scrape phase=plan team=[%d/%d] team_id=%d no_submission",
                team_idx,
                total_teams,
                team.team_id,
            )
            continue
        with rate_limit.acquire():
            try:
                listing = episodes_lister(session, submission_id)
            except client.KaggleEpisodeError as exc:
                logger.warning(
                    "list_episodes_for_submission failed for %s: %s",
                    submission_id,
                    exc,
                )
                continue
        raw_episodes = listing.get("episodes") if isinstance(listing, dict) else None
        episodes: list[Any] = raw_episodes if isinstance(raw_episodes, list) else []
        team_id_by_submission = records.extract_submission_team_map(listing)

        planned_for_team = 0
        for raw in episodes:
            if not isinstance(raw, dict):
                continue
            if (
                spec.limit_per_team is not None
                and planned_for_team >= spec.limit_per_team
            ):
                break
            episode_id = _filter_episode(raw=raw, spec=spec, seen_ids=seen_ids, acc=acc)
            if episode_id is None:
                continue
            plan.append(
                _PlannedEpisode(
                    episode_id=episode_id,
                    raw=raw,
                    team_id_by_submission=team_id_by_submission,
                )
            )
            planned_for_team += 1

        elapsed = time.monotonic() - started
        pct = 100.0 * team_idx / total_teams if total_teams else 0.0
        eta_sec = elapsed * (total_teams - team_idx) / team_idx if team_idx > 0 else 0.0
        logger.info(
            "scrape phase=plan team=[%d/%d] %.1f%% team_id=%d "
            "listed=%d planned=%d cumulative=%d elapsed=%ds eta=%s",
            team_idx,
            total_teams,
            pct,
            team.team_id,
            len(episodes),
            planned_for_team,
            len(plan),
            int(elapsed),
            _format_eta(eta_sec),
        )

    logger.info(
        "scrape phase=plan done teams_scanned=%d planned=%d "
        "skipped_existing=%d skipped_failed=%d skipped_mode=%d",
        acc.teams_scanned,
        len(plan),
        acc.episodes_skipped_existing,
        acc.episodes_skipped_failed,
        acc.episodes_skipped_mode,
    )
    return plan


def _format_eta(seconds: float) -> str:
    if seconds <= 0 or seconds != seconds:  # NaN
        return "-"
    eta = datetime.now(tz=UTC) + timedelta(seconds=seconds)
    return f"{int(seconds)}s ({eta.strftime('%H:%M:%S')}Z)"


def _fetch_all(
    *,
    session: requests.Session,
    spec: ScrapeSpec,
    plan: list[_PlannedEpisode],
    seen_ids: set[int],
    acc: _Accumulator,
    run_id: str,
    rate_limit: TokenBucket,
    replay_fetcher: ReplayFetcher,
    progress_every: int,
    flush_every: int,
    workers: int,
) -> tuple[int, int]:
    """Phase 2 を ThreadPoolExecutor で並列実行し、(records, replays) を返す。

    - 各 worker は `_fetch_planned_episode_io` で replay を取得 (rate_limit
      で 60req/60s に制限される)。
    - 取得結果は main thread に集約され、buffer mutation は逐次実行される。
    - `flush_every` 件処理ごとに `_flush` を挟み、cancel/timeout 時の loss を抑える。
    - `workers=1` の場合は ThreadPool を作らず逐次実行 (テスト容易性のため)。
    """

    total_records = 0
    total_replays = 0
    total = len(plan)
    if total == 0:
        logger.info("scrape phase=fetch nothing to download")
        return total_records, total_replays
    logger.info(
        "scrape phase=fetch total=%d flush_every=%d workers=%d",
        total,
        flush_every,
        workers,
    )
    started = time.monotonic()

    def _io(planned: _PlannedEpisode) -> _FetchedEpisode:
        return _fetch_planned_episode_io(
            session=session,
            planned=planned,
            spec=spec,
            run_id=run_id,
            rate_limit=rate_limit,
            replay_fetcher=replay_fetcher,
        )

    def _on_processed(idx: int) -> None:
        nonlocal total_records, total_replays
        if idx == total or idx % progress_every == 0:
            elapsed = time.monotonic() - started
            pct = 100.0 * idx / total
            eta_sec = elapsed * (total - idx) / idx if idx > 0 else 0.0
            logger.info(
                "scrape phase=fetch [%d/%d] %.1f%% fetched=%d failed=%d "
                "elapsed=%ds eta=%s",
                idx,
                total,
                pct,
                acc.episodes_fetched,
                acc.episodes_failed,
                int(elapsed),
                _format_eta(eta_sec),
            )
        if idx % flush_every == 0 and acc.buffered_records:
            written_records, written_replays = _flush(acc, spec)
            total_records += written_records
            total_replays += written_replays
            logger.info(
                "scrape phase=fetch flush [%d/%d] persisted records=%d "
                "replays=%d (cumulative records=%d)",
                idx,
                total,
                written_records,
                written_replays,
                total_records,
            )

    if workers <= 1:
        for idx, planned in enumerate(plan, start=1):
            fetched = _io(planned)
            _apply_fetched(fetched, spec=spec, acc=acc, seen_ids=seen_ids)
            _on_processed(idx)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_io, p) for p in plan]
            for idx, future in enumerate(as_completed(futures), start=1):
                fetched = future.result()
                _apply_fetched(fetched, spec=spec, acc=acc, seen_ids=seen_ids)
                _on_processed(idx)
    return total_records, total_replays


def plan_only(
    spec: ScrapeSpec,
    *,
    session: requests.Session | None = None,
    rate_limit: TokenBucket | None = None,
    leaderboard_fetcher: LeaderboardFetcher | None = None,
    team_fetcher: TeamFetcher | None = None,
    episodes_lister: EpisodesLister | None = None,
    run_id: str | None = None,
) -> tuple[ScrapeResult, list[_PlannedEpisode]]:
    """Phase 1 のみ実行。確定 plan と途中 ScrapeResult (planned 件数まで) を返す。"""

    if spec.top <= 0:
        raise ValueError(f"spec.top must be positive, got {spec.top}")
    if not spec.modes:
        raise ValueError("spec.modes must be non-empty")

    owns_session = session is None
    session = session or client.build_session()
    rate_limit = rate_limit or _default_rate_limit()
    leaderboard_fetcher = leaderboard_fetcher or leaderboard.fetch
    team_fetcher = team_fetcher or client.get_team
    episodes_lister = episodes_lister or client.list_episodes_for_submission
    run_id = run_id or f"kaggle_{uuid.uuid4().hex[:8]}"

    seen_ids = state.existing_episode_ids(spec.data_root, modes=spec.modes)
    acc = _Accumulator()
    try:
        plan = _plan(
            session=session,
            spec=spec,
            seen_ids=seen_ids,
            acc=acc,
            rate_limit=rate_limit,
            leaderboard_fetcher=leaderboard_fetcher,
            team_fetcher=team_fetcher,
            episodes_lister=episodes_lister,
        )
    finally:
        if owns_session:
            session.close()

    result = ScrapeResult(
        run_id=run_id,
        teams_scanned=acc.teams_scanned,
        teams_without_submission=acc.teams_without_submission,
        episodes_considered=acc.episodes_considered,
        episodes_skipped_existing=acc.episodes_skipped_existing,
        episodes_skipped_failed=acc.episodes_skipped_failed,
        episodes_skipped_mode=acc.episodes_skipped_mode,
        episodes_planned=len(plan),
        episodes_fetched=0,
        episodes_failed=acc.episodes_failed,
        records_written=0,
        replays_written=0,
        dry_run=spec.dry_run,
    )
    return result, plan


def fetch_with_plan(
    spec: ScrapeSpec,
    plan: list[_PlannedEpisode],
    *,
    session: requests.Session | None = None,
    rate_limit: TokenBucket | None = None,
    replay_fetcher: ReplayFetcher | None = None,
    run_id: str,
    progress_every: int = 10,
    flush_every: int = 200,
    workers: int = 1,
) -> ScrapeResult:
    """Phase 1 で確定した plan を受け取り Phase 2 のみ実行する。"""

    if not spec.modes:
        raise ValueError("spec.modes must be non-empty")
    if progress_every <= 0:
        raise ValueError(f"progress_every must be positive, got {progress_every}")
    if flush_every <= 0:
        raise ValueError(f"flush_every must be positive, got {flush_every}")
    if workers <= 0:
        raise ValueError(f"workers must be positive, got {workers}")

    owns_session = session is None
    session = session or client.build_session()
    rate_limit = rate_limit or _default_rate_limit()
    replay_fetcher = replay_fetcher or client.get_episode_replay

    seen_ids = state.existing_episode_ids(spec.data_root, modes=spec.modes)
    acc = _Accumulator()
    records_written = 0
    replays_written = 0
    try:
        records_written, replays_written = _fetch_all(
            session=session,
            spec=spec,
            plan=plan,
            seen_ids=seen_ids,
            acc=acc,
            run_id=run_id,
            rate_limit=rate_limit,
            replay_fetcher=replay_fetcher,
            progress_every=progress_every,
            flush_every=flush_every,
            workers=workers,
        )
        tail_records, tail_replays = _flush(acc, spec)
        records_written += tail_records
        replays_written += tail_replays
    except KeyboardInterrupt:
        tail_records, tail_replays = _flush(acc, spec)
        records_written += tail_records
        replays_written += tail_replays
        raise
    finally:
        if owns_session:
            session.close()

    return ScrapeResult(
        run_id=run_id,
        teams_scanned=0,
        teams_without_submission=0,
        episodes_considered=len(plan),
        episodes_skipped_existing=0,
        episodes_skipped_failed=0,
        episodes_skipped_mode=0,
        episodes_planned=len(plan),
        episodes_fetched=acc.episodes_fetched,
        episodes_failed=acc.episodes_failed,
        records_written=records_written,
        replays_written=replays_written,
        dry_run=spec.dry_run,
    )


def serialize_plan(plan: list[_PlannedEpisode]) -> list[dict[str, Any]]:
    """JSON 化可能な plan 表現。`save_plan` / `load_plan` で永続化に使う。"""

    return [
        {
            "episode_id": p.episode_id,
            "raw": p.raw,
            "team_id_by_submission": {
                str(k): v for k, v in p.team_id_by_submission.items()
            },
        }
        for p in plan
    ]


def deserialize_plan(payload: list[dict[str, Any]]) -> list[_PlannedEpisode]:
    return [
        _PlannedEpisode(
            episode_id=int(item["episode_id"]),
            raw=item["raw"],
            team_id_by_submission={
                int(k): int(v) for k, v in item["team_id_by_submission"].items()
            },
        )
        for item in payload
    ]


def run(
    spec: ScrapeSpec,
    *,
    session: requests.Session | None = None,
    rate_limit: TokenBucket | None = None,
    leaderboard_fetcher: LeaderboardFetcher | None = None,
    team_fetcher: TeamFetcher | None = None,
    episodes_lister: EpisodesLister | None = None,
    replay_fetcher: ReplayFetcher | None = None,
    run_id: str | None = None,
    progress_every: int = 10,
    flush_every: int = 200,
    workers: int = 1,
) -> ScrapeResult:
    """Scrape Kaggle episodes according to `spec` and persist records."""

    if spec.top <= 0:
        raise ValueError(f"spec.top must be positive, got {spec.top}")
    if not spec.modes:
        raise ValueError("spec.modes must be non-empty")
    if progress_every <= 0:
        raise ValueError(f"progress_every must be positive, got {progress_every}")
    if flush_every <= 0:
        raise ValueError(f"flush_every must be positive, got {flush_every}")
    if workers <= 0:
        raise ValueError(f"workers must be positive, got {workers}")

    owns_session = session is None
    session = session or client.build_session()
    rate_limit = rate_limit or _default_rate_limit()
    leaderboard_fetcher = leaderboard_fetcher or leaderboard.fetch
    team_fetcher = team_fetcher or client.get_team
    episodes_lister = episodes_lister or client.list_episodes_for_submission
    replay_fetcher = replay_fetcher or client.get_episode_replay
    run_id = run_id or f"kaggle_{uuid.uuid4().hex[:8]}"

    seen_ids = state.existing_episode_ids(spec.data_root, modes=spec.modes)
    acc = _Accumulator()
    plan: list[_PlannedEpisode] = []
    records_written = 0
    replays_written = 0

    try:
        plan = _plan(
            session=session,
            spec=spec,
            seen_ids=seen_ids,
            acc=acc,
            rate_limit=rate_limit,
            leaderboard_fetcher=leaderboard_fetcher,
            team_fetcher=team_fetcher,
            episodes_lister=episodes_lister,
        )
        records_written, replays_written = _fetch_all(
            session=session,
            spec=spec,
            plan=plan,
            seen_ids=seen_ids,
            acc=acc,
            run_id=run_id,
            rate_limit=rate_limit,
            replay_fetcher=replay_fetcher,
            progress_every=progress_every,
            flush_every=flush_every,
            workers=workers,
        )
        tail_records, tail_replays = _flush(acc, spec)
        records_written += tail_records
        replays_written += tail_replays
    except KeyboardInterrupt:
        tail_records, tail_replays = _flush(acc, spec)
        records_written += tail_records
        replays_written += tail_replays
        raise
    finally:
        if owns_session:
            session.close()

    return ScrapeResult(
        run_id=run_id,
        teams_scanned=acc.teams_scanned,
        teams_without_submission=acc.teams_without_submission,
        episodes_considered=acc.episodes_considered,
        episodes_skipped_existing=acc.episodes_skipped_existing,
        episodes_skipped_failed=acc.episodes_skipped_failed,
        episodes_skipped_mode=acc.episodes_skipped_mode,
        episodes_planned=len(plan),
        episodes_fetched=acc.episodes_fetched,
        episodes_failed=acc.episodes_failed,
        records_written=records_written,
        replays_written=replays_written,
        dry_run=spec.dry_run,
    )
