"""onstart の S3 marker (`runpod_progress/<RUN_ID>/`) を読み書きする。

`onstart.sh.tmpl` 側 (`mark()` 関数) が
`s3://orbit-wars-dvc-286854171013/runpod_progress/<RUN_ID>/<TIMESTAMP>_<STEP>`
形式の空オブジェクトを各ステップで書き出している。本モジュールはそれを
`boto3 s3.list_objects_v2` で列挙し、`ProgressMarker` のリストに整形する。

加えて Python 学習プロセス側からも `mark_progress(run_id, step, payload?)` で
同形式のマーカーを書ける (例: epoch 単位の `train.epoch` ログ)。
"""

from __future__ import annotations

import datetime as _dt
import json as _json
import logging as _logging
import os as _os
import re
from dataclasses import dataclass
from typing import Any

import boto3

_logger = _logging.getLogger(__name__)

PROGRESS_BUCKET = "orbit-wars-dvc-286854171013"
# IAM の dvc-user は `remote/*` 配下にしか PutObject 権限を持たないので、
# progress / artifacts も `remote/runpod_progress/...` 配下に置く。
PROGRESS_PREFIX = "remote/runpod_progress"
ARTIFACT_PREFIX = "remote/runpod_artifacts"
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


def mark_progress(
    run_id: str,
    step: str,
    payload: dict[str, Any] | None = None,
    *,
    bucket: str = PROGRESS_BUCKET,
    prefix: str = PROGRESS_PREFIX,
    s3_client: Any | None = None,
) -> None:
    """Python 側から S3 progress marker を書き出す (onstart の `mark()` 互換)。

    Key 形式: `<prefix>/<run_id>/<ISO8601 UTC>_<step>`。本体は payload が
    あれば JSON、無ければ空 (bash 版と同じ)。AWS env / boto creds が無ければ
    sileent skip (local / test).
    """
    if not run_id or not step:
        return
    timestamp = _dt.datetime.now(tz=_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    key = f"{prefix}/{run_id}/{timestamp}_{step}"
    if s3_client is None:
        access_key = _os.environ.get("AWS_ACCESS_KEY_ID")
        if not access_key:
            _logger.debug("mark_progress skipped: no AWS_ACCESS_KEY_ID env")
            return
        try:
            s3_client = _build_s3_client()
        except Exception as exc:  # pragma: no cover - boto wiring varies
            _logger.warning("mark_progress: failed to build s3 client: %s", exc)
            return
    body = _json.dumps(payload).encode() if payload else b""
    try:
        s3_client.put_object(Bucket=bucket, Key=key, Body=body)
    except Exception as exc:  # pragma: no cover - non-fatal logging path
        _logger.warning("mark_progress put_object failed for %s: %s", key, exc)


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


# ===== artifact prefix (PR1: 成果物消失の止血) =====
# onstart.sh.tmpl が train 完了直後に s3://<bucket>/<ARTIFACT_PREFIX>/<RUN_ID>/
# 配下に best.pt / metrics.json / run.json / onstart.log を直接 upload する。
# DVC とは独立した経路で、dvc push が間に合わずに pod kill されたケースの保険。

ARTIFACT_FILES: tuple[str, ...] = (
    "best.pt",
    "metrics.json",
    "run.json",
    "onstart.log",
)


def list_artifacts(
    run_id: str,
    *,
    profile: str | None = None,
    bucket: str = PROGRESS_BUCKET,
    prefix: str = ARTIFACT_PREFIX,
    s3_client: Any | None = None,
) -> list[str]:
    """`runpod_artifacts/<RUN_ID>/` 直下にあるファイル名のリスト。"""
    client = s3_client if s3_client is not None else _build_s3_client(profile)
    full_prefix = f"{prefix}/{run_id}/"
    response = client.list_objects_v2(Bucket=bucket, Prefix=full_prefix)
    contents = response.get("Contents") or []
    files: list[str] = []
    for entry in contents:
        key = entry.get("Key", "")
        if key.startswith(full_prefix):
            tail = key[len(full_prefix) :]
            if tail and "/" not in tail:
                files.append(tail)
    files.sort()
    return files


def download_artifact(
    run_id: str,
    filename: str,
    dest: Any,  # Path
    *,
    profile: str | None = None,
    bucket: str = PROGRESS_BUCKET,
    prefix: str = ARTIFACT_PREFIX,
    s3_client: Any | None = None,
) -> bool:
    """`runpod_artifacts/<RUN_ID>/<filename>` を `dest` に保存する。

    成功時 True、見つからなければ False (例外は出さない)。
    """
    client = s3_client if s3_client is not None else _build_s3_client(profile)
    key = f"{prefix}/{run_id}/{filename}"
    try:
        obj = client.get_object(Bucket=bucket, Key=key)
    except Exception:  # noqa: BLE001
        return False
    body = obj["Body"].read()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(body, bytes):
        dest.write_bytes(body)
    else:
        dest.write_text(str(body), encoding="utf-8")
    return True
