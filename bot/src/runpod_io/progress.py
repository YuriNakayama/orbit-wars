"""onstart の S3 marker (`runpod_progress/<RUN_ID>/`) を読んで進捗を返す。

`onstart.sh.tmpl` 側 (`mark()` 関数) が
`s3://orbit-wars-dvc-286854171013/runpod_progress/<RUN_ID>/<TIMESTAMP>_<STEP>`
形式の空オブジェクトを各ステップで書き出している。本モジュールはそれを
`boto3 s3.list_objects_v2` で列挙し、`ProgressMarker` のリストに整形する。

Pod 内の Python (case5/case6 の preprocess.py / train.py) からは `mark_progress`
を呼ぶことで、同じ S3 prefix に JSON payload 付きの marker を書ける。
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import boto3

logger = logging.getLogger(__name__)

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
    """Pod 内 Python から S3 に進捗 marker を書く (best-effort、失敗は warning)。

    onstart.sh.tmpl の `mark()` と同じ key 命名規則
    (`<prefix>/<run_id>/<TS>_<step>`) で `<payload>` を JSON body として put。
    payload が None の場合は空 body (numeric marker と同等)。

    認証は IAM role / 環境変数 (RunPod pod 上では AWS_ACCESS_KEY_ID 等で渡す)。
    profile を渡したい場合は呼び出し側で `s3_client` を渡す。
    """
    if s3_client is None:
        s3_client = boto3.client("s3")
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    key = f"{prefix}/{run_id}/{timestamp}_{step}"
    body: bytes
    if payload is None:
        body = b""
    else:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    try:
        s3_client.put_object(Bucket=bucket, Key=key, Body=body)
    except Exception as exc:  # noqa: BLE001 — best-effort marker
        logger.warning("mark_progress failed run_id=%s step=%s: %s", run_id, step, exc)


def get_run_id() -> str | None:
    """RunPod pod 上で onstart が export した run_id を取得する補助。

    train.py は `ORBIT_WARS_RUN_ID` を env で受ける (onstart 側が
    `ORBIT_WARS_RUN_ID="<RUN_ID>"` 付きで `python -m ...train` を起動)。
    ローカル開発で env 未設定なら None。呼び出し側は None なら
    `mark_progress` を no-op にすればよい。
    """
    return os.environ.get("ORBIT_WARS_RUN_ID")


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
