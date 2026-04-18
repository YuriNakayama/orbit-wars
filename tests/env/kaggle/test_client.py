"""Unit tests for src/env/kaggle/client.py."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import requests

from env.kaggle import client


class _FakeResponse:
    def __init__(self, payload: Any, *, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"status {self.status_code}")

    def json(self) -> Any:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


@pytest.fixture
def kaggle_config(tmp_path: Path) -> Path:
    cfg = tmp_path / "kaggle.json"
    cfg.write_text(json.dumps({"username": "u", "key": "k"}), encoding="utf-8")
    return cfg


def test_build_session_loads_creds_and_bootstraps_cookie(
    monkeypatch: pytest.MonkeyPatch, kaggle_config: Path
) -> None:
    calls: list[str] = []

    def fake_get(self: requests.Session, url: str, **_kw: Any) -> _FakeResponse:
        calls.append(url)
        self.cookies.set("XSRF-TOKEN", "abc123")
        return _FakeResponse({})

    monkeypatch.setattr(requests.Session, "get", fake_get, raising=False)

    cfg = client.ClientConfig(kaggle_config_path=str(kaggle_config))
    session = client.build_session(cfg)

    assert session.auth == ("u", "k")
    assert session.headers["X-XSRF-TOKEN"] == "abc123"
    assert "leaderboard" in calls[0]


def test_build_session_missing_creds_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    cfg = client.ClientConfig(kaggle_config_path=str(tmp_path / "missing.json"))
    with pytest.raises(client.KaggleEpisodeError, match="認証情報が見つかりません"):
        client.build_session(cfg)


def test_get_team_posts_expected_body() -> None:
    session = MagicMock(spec=requests.Session)
    session.post.return_value = _FakeResponse(
        {"id": 15637383, "publicLeaderboardSubmissionId": 99}
    )

    out = client.get_team(session, 15637383)

    assert out["publicLeaderboardSubmissionId"] == 99
    args, kwargs = session.post.call_args
    assert args[0].endswith(".TeamService/GetTeam")
    assert kwargs["json"] == {"teamId": 15637383}


def test_list_episodes_for_submission_posts_expected_body() -> None:
    session = MagicMock(spec=requests.Session)
    session.post.return_value = _FakeResponse({"episodes": []})

    out = client.list_episodes_for_submission(session, 42)

    assert out == {"episodes": []}
    args, kwargs = session.post.call_args
    assert args[0].endswith(".EpisodeService/ListEpisodes")
    assert kwargs["json"] == {"submissionId": 42}


def test_list_episodes_by_ids_posts_expected_body() -> None:
    session = MagicMock(spec=requests.Session)
    session.post.return_value = _FakeResponse({"episodes": []})

    client.list_episodes_by_ids(session, [1, 2, 3])

    _args, kwargs = session.post.call_args
    assert kwargs["json"] == {"ids": [1, 2, 3]}


def test_get_episode_replay_posts_expected_body() -> None:
    session = MagicMock(spec=requests.Session)
    session.post.return_value = _FakeResponse({"steps": [], "configuration": {}})

    out = client.get_episode_replay(session, 42)

    assert out["steps"] == []
    args, kwargs = session.post.call_args
    assert args[0].endswith(".EpisodeService/GetEpisodeReplay")
    assert kwargs["json"] == {"episodeId": 42}


def test_post_wraps_request_exception() -> None:
    session = MagicMock(spec=requests.Session)
    session.post.side_effect = requests.exceptions.ConnectionError("boom")

    with pytest.raises(client.KaggleEpisodeError, match="failed"):
        client.list_episodes_for_submission(session, 1)


def test_post_wraps_http_error() -> None:
    session = MagicMock(spec=requests.Session)
    session.post.return_value = _FakeResponse({}, status_code=503)

    with pytest.raises(client.KaggleEpisodeError, match="failed"):
        client.get_episode_replay(session, 1)


def test_post_wraps_non_json_response() -> None:
    session = MagicMock(spec=requests.Session)
    session.post.return_value = _FakeResponse(ValueError("not json"))

    with pytest.raises(client.KaggleEpisodeError, match="non-JSON"):
        client.list_episodes_for_submission(session, 1)


def test_post_wraps_non_dict_payload() -> None:
    session = MagicMock(spec=requests.Session)
    session.post.return_value = _FakeResponse([1, 2, 3])

    with pytest.raises(client.KaggleEpisodeError, match="non-dict"):
        client.list_episodes_for_submission(session, 1)


def test_extract_replay_json_serializes_payload() -> None:
    raw = client.extract_replay_json({"a": 1, "b": [2, 3]})
    assert json.loads(raw) == {"a": 1, "b": [2, 3]}
