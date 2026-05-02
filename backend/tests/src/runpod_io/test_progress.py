"""runpod_io.progress: S3 marker 解析テスト (boto3 client は MagicMock)。"""

from __future__ import annotations

from unittest.mock import MagicMock

from runpod_io import progress


def _build_response(keys: list[str]) -> dict[str, list[dict[str, str]]]:
    return {"Contents": [{"Key": k} for k in keys]}


def test_list_markers_orders_and_parses() -> None:
    client = MagicMock()
    client.list_objects_v2.return_value = _build_response(
        [
            "runpod_progress/run42/2026-05-02T01:00:00Z_30_before_uv_sync",
            "runpod_progress/run42/2026-05-02T00:00:01Z_00_container_started",
            "runpod_progress/run42/2026-05-02T00:30:00Z_10_before_clone",
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
        Prefix="runpod_progress/run42/",
    )


def test_list_markers_skips_unparseable_keys() -> None:
    client = MagicMock()
    client.list_objects_v2.return_value = _build_response(
        [
            "runpod_progress/run42/garbage",
            "runpod_progress/run42/2026-05-02T00:00:00Z_99_done",
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
            "runpod_progress/run42/2026-05-02T00:00:00Z_00_container_started",
            "runpod_progress/run42/2026-05-02T01:00:00Z_99_done",
        ]
    )
    markers = progress.list_markers("run42", s3_client=client)
    latest = progress.latest_step(markers)
    assert latest is not None
    assert latest.step == "99_done"


def test_latest_step_none_when_empty() -> None:
    assert progress.latest_step([]) is None
