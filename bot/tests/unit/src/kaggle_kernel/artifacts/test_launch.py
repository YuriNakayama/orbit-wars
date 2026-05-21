"""kaggle_kernel.artifacts.launch のユニットテスト。"""

from __future__ import annotations

from pathlib import Path

from kaggle_kernel.artifacts.launch import (
    LAUNCH_JSON_FILENAME,
    LaunchMeta,
    read_launch_json,
    write_launch_json,
)


def _sample_meta() -> LaunchMeta:
    return LaunchMeta(
        run_id="20260520-103000__feature-x__abc1234__seed0",
        case="case1",
        commit_sha="abc1234deadbeef",
        branch="feature-x",
        kernel_slug="yuri/orbit-wars-case1-run0",
        kernel_version=3,
        dataset_slug="yuri/orbit-wars-bot",
        dataset_version="v17",
        accelerator="gpu-t4x2",
        enable_internet=True,
        started_at="2026-05-20T10:30:00Z",
        seed=0,
    )


def test_write_read_roundtrip(tmp_path: Path) -> None:
    meta = _sample_meta()
    out = write_launch_json(tmp_path, meta)
    assert out.name == LAUNCH_JSON_FILENAME
    restored = read_launch_json(tmp_path)
    assert restored == meta


def test_write_creates_parent_dirs(tmp_path: Path) -> None:
    nested = tmp_path / "deep" / "run_dir"
    write_launch_json(nested, _sample_meta())
    assert (nested / LAUNCH_JSON_FILENAME).is_file()


def test_extra_field_roundtrips(tmp_path: Path) -> None:
    meta = LaunchMeta(
        run_id="r0",
        case="case1",
        commit_sha="abc1234",
        branch="main",
        kernel_slug="yuri/x",
        kernel_version=None,
        dataset_slug="yuri/d",
        dataset_version=None,
        accelerator="gpu-t4x2",
        enable_internet=True,
        started_at="2026-05-20T10:30:00Z",
        extra={"note": "first run"},
    )
    write_launch_json(tmp_path, meta)
    restored = read_launch_json(tmp_path)
    assert restored.extra == {"note": "first run"}
    assert restored.kernel_version is None
