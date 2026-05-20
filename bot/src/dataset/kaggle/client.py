"""Kaggle EpisodeService HTTP クライアント。

Kaggle 内部 API (`/api/i/competitions.*`) を叩く認証付きクライアント。
`~/.kaggle/kaggle.json` Basic auth + session cookie + `X-XSRF-TOKEN` を組み合わせる。
テスト時のモック注入を容易にするため `build_session()` と `_post` を独立させている。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

BASE_URL = "https://www.kaggle.com/api/i/competitions."
LEADERBOARD_BOOTSTRAP_URL = "https://www.kaggle.com/competitions/orbit-wars/leaderboard"


class KaggleEpisodeError(RuntimeError):
    """EpisodeService 呼び出しに失敗した場合に raise。"""


@dataclass(frozen=True)
class ClientConfig:
    user_agent: str = "orbit-wars-log-collector/0.1"
    timeout_sec: float = 30.0
    # 429 で Kaggle が返す `Retry-After: 60s` を待ち切れるように
    # total を増やし backoff_max を 60s に上げる。
    # backoff: 2,4,8,16,32,60,60,60s → 最悪 ~4分待ってから諦める。
    max_retries: int = 8
    backoff_factor: float = 2.0
    backoff_max: float = 60.0
    kaggle_config_path: str = "~/.kaggle/kaggle.json"
    pool_connections: int = 32
    pool_maxsize: int = 32


def _load_credentials(config_path: str) -> tuple[str, str]:
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if username and key:
        return username, key
    path = Path(config_path).expanduser()
    if not path.exists():
        raise KaggleEpisodeError(
            f"Kaggle 認証情報が見つかりません: {path} "
            "or KAGGLE_USERNAME/KAGGLE_KEY env vars"
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise KaggleEpisodeError(f"kaggle.json 読込失敗: {exc}") from exc
    username = data.get("username")
    key = data.get("key")
    if not username or not key:
        raise KaggleEpisodeError("kaggle.json に username/key が不足")
    return str(username), str(key)


def build_session(config: ClientConfig | None = None) -> requests.Session:
    """Auth + cookie + XSRF トークン付きの Session を構築する。"""

    cfg = config or ClientConfig()
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": cfg.user_agent,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
    )
    session.auth = _load_credentials(cfg.kaggle_config_path)
    retry = Retry(
        total=cfg.max_retries,
        backoff_factor=cfg.backoff_factor,
        backoff_max=cfg.backoff_max,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("POST", "GET"),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    session.mount(
        "https://",
        HTTPAdapter(
            max_retries=retry,
            pool_connections=cfg.pool_connections,
            pool_maxsize=cfg.pool_maxsize,
        ),
    )
    try:
        session.get(LEADERBOARD_BOOTSTRAP_URL, timeout=cfg.timeout_sec)
    except requests.exceptions.RequestException as exc:
        raise KaggleEpisodeError(f"cookie bootstrap failed: {exc}") from exc
    xsrf = session.cookies.get("XSRF-TOKEN")
    if xsrf:
        session.headers["X-XSRF-TOKEN"] = xsrf
    return session


def _post(
    session: requests.Session,
    path: str,
    body: dict[str, Any],
    *,
    timeout: float,
) -> dict[str, Any]:
    url = f"{BASE_URL}{path}"
    try:
        resp = session.post(url, json=body, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
    except requests.exceptions.RequestException as exc:
        raise KaggleEpisodeError(f"{path} failed: {exc}") from exc
    except ValueError as exc:
        raise KaggleEpisodeError(f"{path} returned non-JSON response") from exc
    if not isinstance(payload, dict):
        raise KaggleEpisodeError(
            f"{path} returned non-dict payload: {type(payload).__name__}"
        )
    return payload


def get_team(
    session: requests.Session,
    team_id: int,
    *,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Team 情報を取得（`publicLeaderboardSubmissionId` を含む）。"""

    return _post(session, "TeamService/GetTeam", {"teamId": team_id}, timeout=timeout)


def list_episodes_for_submission(
    session: requests.Session,
    submission_id: int,
    *,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Submission に紐づくエピソード（最大 30 件）のメタを取得。"""

    return _post(
        session,
        "EpisodeService/ListEpisodes",
        {"submissionId": submission_id},
        timeout=timeout,
    )


def list_episodes_by_ids(
    session: requests.Session,
    episode_ids: list[int],
    *,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Episode ID リストからメタを一括取得。"""

    return _post(
        session,
        "EpisodeService/ListEpisodes",
        {"ids": episode_ids},
        timeout=timeout,
    )


def get_episode_replay(
    session: requests.Session,
    episode_id: int,
    *,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Episode の完全な replay JSON (configuration / steps) を取得。"""

    return _post(
        session,
        "EpisodeService/GetEpisodeReplay",
        {"episodeId": episode_id},
        timeout=timeout,
    )


def extract_replay_json(response: dict[str, Any]) -> str:
    """Replay レスポンスを JSON 文字列として返す（gzip 化前の素データ）。"""

    return json.dumps(response, separators=(",", ":"))
