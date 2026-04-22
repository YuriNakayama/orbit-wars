"""Kaggle episode scraper orchestration.

leaderboard → team → publicLeaderboardSubmissionId → ListEpisodes → GetEpisodeReplay
→ records/replay 永続化の全体フロー。`ScrapeSpec` に設定を集約し、
`run(...)` が `ScrapeResult` を返す。`dry_run=True` では FS 書き込みを skip する。
"""

from __future__ import annotations

import gzip
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
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
    episodes_fetched: int
    episodes_failed: int
    records_written: int
    replays_written: int
    dry_run: bool


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


def _process_episode(
    *,
    session: requests.Session,
    raw: dict[str, Any],
    spec: ScrapeSpec,
    seen_ids: set[int],
    acc: _Accumulator,
    run_id: str,
    rate_limit: TokenBucket,
    replay_fetcher: ReplayFetcher,
    team_id_by_submission: dict[int, int],
) -> None:
    acc.episodes_considered += 1
    episode_id = int(raw.get("id") or 0)
    if episode_id <= 0:
        acc.episodes_failed += 1
        return
    if episode_id in seen_ids:
        acc.episodes_skipped_existing += 1
        return

    agents = raw.get("agents") or []
    if not isinstance(agents, list):
        acc.episodes_failed += 1
        return

    if not spec.include_failed and records.is_failed_episode(raw):
        acc.episodes_skipped_failed += 1
        return

    try:
        meta = records.parse_episode_meta(raw)
    except (ValueError, KeyError, TypeError) as exc:
        logger.warning("parse_episode_meta failed for %s: %s", episode_id, exc)
        acc.episodes_failed += 1
        return

    if meta.mode not in spec.modes:
        acc.episodes_skipped_mode += 1
        return

    with rate_limit.acquire():
        try:
            replay_resp = replay_fetcher(session, episode_id)
        except client.KaggleEpisodeError as exc:
            logger.warning("get_episode_replay failed for %s: %s", episode_id, exc)
            acc.episodes_failed += 1
            return

    record = records.build_match_record(
        meta,
        run_id=run_id,
        scraped_at=_utc_now_iso(),
        team_id_by_submission=team_id_by_submission,
    )
    seen_ids.add(episode_id)
    acc.episodes_fetched += 1

    if spec.dry_run:
        return
    acc.buffered_records.append(record)
    payload_str = client.extract_replay_json(replay_resp)
    acc.buffered_replays[record.match_id] = gzip.compress(payload_str.encode("utf-8"))


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
) -> ScrapeResult:
    """Scrape Kaggle episodes according to `spec` and persist records."""

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
    replay_fetcher = replay_fetcher or client.get_episode_replay
    run_id = run_id or f"kaggle_{uuid.uuid4().hex[:8]}"

    seen_ids = state.existing_episode_ids(spec.data_root, modes=spec.modes)
    acc = _Accumulator()
    teams = leaderboard_fetcher(spec.top)

    try:
        for team in teams:
            acc.teams_scanned += 1
            with rate_limit.acquire():
                submission_id = _lookup_submission_id(
                    session, team.team_id, team_fetcher
                )
            if submission_id is None:
                acc.teams_without_submission += 1
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
            raw_episodes = (
                listing.get("episodes") if isinstance(listing, dict) else None
            )
            episodes: list[Any] = raw_episodes if isinstance(raw_episodes, list) else []
            team_id_by_submission = records.extract_submission_team_map(listing)
            fetched_for_team = 0
            for raw in episodes:
                if not isinstance(raw, dict):
                    continue
                if (
                    spec.limit_per_team is not None
                    and fetched_for_team >= spec.limit_per_team
                ):
                    break
                before_fetched = acc.episodes_fetched
                _process_episode(
                    session=session,
                    raw=raw,
                    spec=spec,
                    seen_ids=seen_ids,
                    acc=acc,
                    run_id=run_id,
                    rate_limit=rate_limit,
                    replay_fetcher=replay_fetcher,
                    team_id_by_submission=team_id_by_submission,
                )
                if acc.episodes_fetched > before_fetched:
                    fetched_for_team += 1
        records_written, replays_written = _flush(acc, spec)
    except KeyboardInterrupt:
        records_written, replays_written = _flush(acc, spec)
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
        episodes_fetched=acc.episodes_fetched,
        episodes_failed=acc.episodes_failed,
        records_written=records_written,
        replays_written=replays_written,
        dry_run=spec.dry_run,
    )
