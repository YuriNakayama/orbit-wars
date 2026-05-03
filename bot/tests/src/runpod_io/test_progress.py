"""runpod_io.progress: S3 marker 解析テスト (boto3 client は MagicMock)。"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock

from runpod_io import progress


def _build_response(keys: list[str]) -> dict[str, list[dict[str, str]]]:
    return {"Contents": [{"Key": k} for k in keys]}


def test_list_markers_orders_and_parses() -> None:
    client = MagicMock()
    client.list_objects_v2.return_value = _build_response(
        [
            f"{progress.PROGRESS_PREFIX}/run42/2026-05-02T01:00:00Z_30_before_uv_sync",
            f"{progress.PROGRESS_PREFIX}/run42/2026-05-02T00:00:01Z_00_container_started",
            f"{progress.PROGRESS_PREFIX}/run42/2026-05-02T00:30:00Z_10_before_clone",
        ]
    )

    markers = progress.list_markers("run42", s3_client=client)

    assert [m.step for m in markers] == [
        "00_container_started",
        "10_before_clone",
        "30_before_uv_sync",
    ]
    assert markers[0].timestamp == "2026-05-02T00:00:01Z"
    client.list_objects_v2.assert_called_once_with(
        Bucket=progress.PROGRESS_BUCKET,
        Prefix=f"{progress.PROGRESS_PREFIX}/run42/",
    )


def test_list_markers_skips_unparseable_keys() -> None:
    client = MagicMock()
    client.list_objects_v2.return_value = _build_response(
        [
            f"{progress.PROGRESS_PREFIX}/run42/garbage",
            f"{progress.PROGRESS_PREFIX}/run42/2026-05-02T00:00:00Z_99_done",
        ]
    )
    markers = progress.list_markers("run42", s3_client=client)
    assert len(markers) == 1
    assert markers[0].step == "99_done"


def test_list_markers_empty_when_no_contents() -> None:
    client = MagicMock()
    client.list_objects_v2.return_value = {}
    assert progress.list_markers("run42", s3_client=client) == []


def test_latest_step_returns_last() -> None:
    client = MagicMock()
    client.list_objects_v2.return_value = _build_response(
        [
            f"{progress.PROGRESS_PREFIX}/run42/2026-05-02T00:00:00Z_00_container_started",
            f"{progress.PROGRESS_PREFIX}/run42/2026-05-02T01:00:00Z_99_done",
        ]
    )
    markers = progress.list_markers("run42", s3_client=client)
    latest = progress.latest_step(markers)
    assert latest is not None
    assert latest.step == "99_done"


def test_latest_step_none_when_empty() -> None:
    assert progress.latest_step([]) is None


def test_fetch_onstart_log_prefers_run_dir(tmp_path: Path) -> None:
    """run_dir/onstart.log があればそれを返し、S3 は触らない。"""
    run_dir = tmp_path / "run42"
    run_dir.mkdir()
    (run_dir / "onstart.log").write_text("hello from run_dir\n", encoding="utf-8")
    client = MagicMock()  # S3 が呼ばれたら fail
    result = progress.fetch_onstart_log("run42", run_dir=run_dir, s3_client=client)
    assert result is not None
    assert result.text == "hello from run_dir\n"
    assert result.source == "run_dir"
    client.get_object.assert_not_called()


def test_fetch_onstart_log_falls_back_to_s3(tmp_path: Path) -> None:
    """run_dir/onstart.log が無ければ S3 から取得。"""
    run_dir = tmp_path / "missing_run"
    # run_dir.mkdir() しないので onstart.log も無い
    client = MagicMock()
    body_text = "step=00\nstep=99_done\n"
    client.get_object.return_value = {"Body": io.BytesIO(body_text.encode())}
    result = progress.fetch_onstart_log("run42", run_dir=run_dir, s3_client=client)
    assert result is not None
    assert result.text == body_text
    assert result.source == "s3"
    client.get_object.assert_called_once_with(
        Bucket=progress.PROGRESS_BUCKET,
        Key=f"{progress.PROGRESS_PREFIX}/run42/onstart.log",
    )


def test_fetch_onstart_log_returns_none_when_s3_missing() -> None:
    """S3 にも無いなら None。"""
    client = MagicMock()
    client.get_object.side_effect = RuntimeError("NoSuchKey")
    assert progress.fetch_onstart_log("ghost", s3_client=client) is None


def test_list_artifacts_returns_filenames() -> None:
    """`runpod_artifacts/<run>/` 直下のファイル名のみ返す (subdir は除外)。"""
    client = MagicMock()
    client.list_objects_v2.return_value = {
        "Contents": [
            {"Key": f"{progress.ARTIFACT_PREFIX}/run42/best.pt"},
            {"Key": f"{progress.ARTIFACT_PREFIX}/run42/metrics.json"},
            {"Key": f"{progress.ARTIFACT_PREFIX}/run42/run.json"},
            {"Key": f"{progress.ARTIFACT_PREFIX}/run42/onstart.log"},
            {
                "Key": f"{progress.ARTIFACT_PREFIX}/run42/sub/nested.txt"
            },  # subdir → 除外
        ]
    }
    files = progress.list_artifacts("run42", s3_client=client)
    assert files == ["best.pt", "metrics.json", "onstart.log", "run.json"]


def test_list_artifacts_empty() -> None:
    client = MagicMock()
    client.list_objects_v2.return_value = {}
    assert progress.list_artifacts("run42", s3_client=client) == []


def test_download_artifact_writes_bytes(tmp_path: Path) -> None:
    client = MagicMock()
    client.get_object.return_value = {"Body": io.BytesIO(b"binary-blob")}
    dest = tmp_path / "out" / "best.pt"
    ok = progress.download_artifact("run42", "best.pt", dest, s3_client=client)
    assert ok is True
    assert dest.read_bytes() == b"binary-blob"
    client.get_object.assert_called_once_with(
        Bucket=progress.PROGRESS_BUCKET,
        Key=f"{progress.ARTIFACT_PREFIX}/run42/best.pt",
    )


def test_download_artifact_returns_false_when_missing(tmp_path: Path) -> None:
    client = MagicMock()
    client.get_object.side_effect = RuntimeError("NoSuchKey")
    dest = tmp_path / "missing.pt"
    assert (
        progress.download_artifact("run42", "missing.pt", dest, s3_client=client)
        is False
    )
    assert not dest.exists()


def test_fetch_onstart_log_handles_bytes_body() -> None:
    """boto3 が bytes を返すケースをそのまま decode する。"""

    class FakeBody:
        def __init__(self, b: bytes) -> None:
            self._b = b

        def read(self) -> bytes:
            return self._b

    client = MagicMock()
    client.get_object.return_value = {"Body": FakeBody(b"raw bytes log")}
    result = progress.fetch_onstart_log("run42", s3_client=client)
    assert result is not None
    assert result.text == "raw bytes log"
    assert result.source == "s3"
