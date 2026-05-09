"""Shared fixtures for RunPod integration tests."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from runpod_io.auth import AwsCreds
from runpod_io.runpod.offers import Offer


@pytest.fixture
def mock_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "runpod_io.cli.app.load_aws_creds",
        lambda profile=None: AwsCreds(
            access_key_id="AKIA",
            secret_access_key="secret",
            region="ap-northeast-1",
        ),
    )
    monkeypatch.setattr(
        "runpod_io.cli.app.load_runpod_api_key", lambda: "RUNPOD_KEY_FAKE"
    )


@pytest.fixture
def mock_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "runpod_io.cli.app._git_remote_url",
        lambda: "https://github.com/YuriNakayama/orbit-wars.git",
    )
    monkeypatch.setattr("runpod_io.cli.app._git_current_branch", lambda: "feature-test")
    monkeypatch.setattr("runpod_io.cli.app._verify_commit_pushed", lambda _sha: None)


@pytest.fixture
def mock_offer() -> Offer:
    return Offer(
        gpu_type_id="NVIDIA GeForce RTX 3090",
        display_name="RTX 3090",
        memory_gb=24,
        secure_cloud=True,
        community_cloud=True,
        secure_price=0.5,
        community_price=0.3,
        secure_spot_price=0.2,
        community_spot_price=0.1,
        cloud_type="SECURE",
        dph_total=0.5,
    )


@pytest.fixture
def write_launch() -> Callable[[Path], None]:
    def write(run_dir: Path, **overrides: object) -> None:
        payload: dict[str, object] = {
            "run_id": "run42",
            "pod_id": "pod-abc",
            "commit_sha": "deadbeef",
            "branch": "feature-x",
            "case": "case1",
            "cloud_type": "SECURE",
            "gpu_type_id": "RTX 3090",
            "dph_total": 0.5,
            "data_center_id": None,
            "launched_at": "2026-05-02T00:00:00Z",
        }
        payload.update(overrides)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "launch.json").write_text(json.dumps(payload), encoding="utf-8")

    return write
