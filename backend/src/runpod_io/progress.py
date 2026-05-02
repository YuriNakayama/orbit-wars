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
