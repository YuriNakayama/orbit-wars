"""onstart の S3 marker (`runpod_progress/<RUN_ID>/`) を読んで進捗を返す。

`onstart.sh.tmpl` 側 (`mark()` 関数) が
`s3://orbit-wars-dvc-286854171013/runpod_progress/<RUN_ID>/<TIMESTAMP>_<STEP>`
形式の空オブジェクトを各ステップで書き出している。本モジュールはそれを
`boto3 s3.list_objects_v2` で列挙し、`ProgressMarker` のリストに整形する。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import boto3

PROGRESS_BUCKET = "orbit-wars-dvc-286854171013"
PROGRESS_PREFIX = "runpod_progress"
KEY_PATTERN = re.compile(
    r"^(?P<prefix>.+/)(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)_(?P<step>.+)$"
)


@dataclass(frozen=True)
class ProgressMarker:
    """S3 に書き出された 1 つの進捗マーカー。"""

    timestamp: str  # ISO8601 (Z 付き)
    step: str  # 例: "00_container_started", "60_before_train"


def _build_s3_client(profile: str | None = None) -> Any:
    if profile:
        session = boto3.Session(profile_name=profile)
        return session.client("s3")
    return boto3.client("s3")


def list_markers(
    run_id: str,
    *,
    profile: str | None = None,
    bucket: str = PROGRESS_BUCKET,
    prefix: str = PROGRESS_PREFIX,
    s3_client: Any | None = None,
) -> list[ProgressMarker]:
    """run_id に対する S3 マーカーを timestamp 昇順で返す。"""
    client = s3_client if s3_client is not None else _build_s3_client(profile)
    full_prefix = f"{prefix}/{run_id}/"
    response = client.list_objects_v2(Bucket=bucket, Prefix=full_prefix)
    contents = response.get("Contents") or []
    markers: list[ProgressMarker] = []
    for entry in contents:
        key = entry.get("Key", "")
        match = KEY_PATTERN.match(key)
        if match is None:
            continue
        markers.append(
            ProgressMarker(timestamp=match.group("ts"), step=match.group("step"))
        )
    markers.sort(key=lambda m: m.timestamp)
    return markers


def latest_step(markers: list[ProgressMarker]) -> ProgressMarker | None:
    if not markers:
        return None
    return markers[-1]


@dataclass(frozen=True)
class OnstartLog:
    """onstart の全文ログとその出処。"""

    text: str
    source: str  # "run_dir" / "s3"


def fetch_onstart_log(
    run_id: str,
    *,
    run_dir: Any | None = None,  # Path | None
    profile: str | None = None,
    bucket: str = PROGRESS_BUCKET,
    prefix: str = PROGRESS_PREFIX,
    s3_client: Any | None = None,
) -> OnstartLog | None:
    """onstart の全文ログを取得する。

    優先順:
    1. `run_dir/onstart.log` がローカルにあればそれを読む (DVC pull 後の経路)
    2. なければ `s3://<bucket>/<prefix>/<run_id>/onstart.log` から取得
       (失敗 pod の cleanup_destroy が直接 upload した snapshot)

    どちらも見つからなければ None。
    """
    if run_dir is not None:
        local_path = run_dir / "onstart.log"
        if local_path.is_file():
            return OnstartLog(
                text=local_path.read_text(encoding="utf-8", errors="replace"),
                source="run_dir",
            )

    client = s3_client if s3_client is not None else _build_s3_client(profile)
    key = f"{prefix}/{run_id}/onstart.log"
    try:
        obj = client.get_object(Bucket=bucket, Key=key)
    except Exception:  # noqa: BLE001 — boto3 ClientError 全部 catch
        return None
    body = obj["Body"].read()
    if isinstance(body, bytes):
        body = body.decode("utf-8", errors="replace")
    return OnstartLog(text=body, source="s3")
