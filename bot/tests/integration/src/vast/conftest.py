"""Shared fixtures for Vast.ai integration tests."""

from __future__ import annotations

import pytest

from vast.auth import AwsCreds
from vast.offers import Offer


@pytest.fixture
def mock_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "vast.cli.load_aws_creds",
        lambda profile=None: AwsCreds(
            access_key_id="AKIA",
            secret_access_key="secret",
            region="ap-northeast-1",
        ),
    )
    monkeypatch.setattr("vast.cli.load_vast_api_key", lambda: "VAST_KEY_FAKE")


@pytest.fixture
def mock_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "vast.cli._git_remote_url",
        lambda: "https://github.com/YuriNakayama/orbit-wars.git",
    )
    monkeypatch.setattr("vast.cli._git_current_branch", lambda: "feature-test")
    monkeypatch.setattr("vast.cli._verify_commit_pushed", lambda _sha: None)


@pytest.fixture
def mock_offer() -> Offer:
    return Offer(
        id=12345,
        gpu_name="RTX_3090",
        num_gpus=1,
        dph_total=0.13,
        reliability=0.995,
        geolocation="US",
        cuda_max_good=12.4,
        inet_down=500.0,
        verified=True,
    )
