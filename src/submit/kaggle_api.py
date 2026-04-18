"""Kaggle CLI を subprocess 経由で呼び出すラッパー。"""

from __future__ import annotations

import csv
import io
import re
import subprocess
import time
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

COMPETITION = "orbit-wars"


class KaggleCLIError(RuntimeError):
    """kaggle CLI 実行失敗。"""


def _run(args: list[str]) -> str:
    """kaggle CLI を実行し stdout を返す。認証情報は環境変数から継承される。"""

    cmd = ["kaggle", *args]
    try:
        proc = subprocess.run(  # noqa: S603 — trusted CLI
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise KaggleCLIError(
            "`kaggle` コマンドが見つかりません。"
            "`uv sync` 後に `uv run` 経由で実行してください。"
        ) from exc
    if proc.returncode != 0:
        raise KaggleCLIError(
            f"kaggle CLI 実行失敗 (args={args}):\nstderr: {proc.stderr.strip()}"
        )
    return proc.stdout


def submit(file_path: Path, message: str) -> str:
    """Kaggle に提出し、stdout/stderr を文字列で返す。

    kaggle CLI は submit 成功時にも非ゼロ終了することがあるため、
    アップロード完了を示す文字列を検出したら成功扱いにする。
    CLI 出力だけでは判断がつかない場合、呼び出し側で
    `confirm_submission()` を使って履歴 API で最終確認すること。
    """

    cmd = [
        "kaggle",
        "competitions",
        "submit",
        COMPETITION,
        "-f",
        str(file_path),
        "-m",
        message,
    ]
    try:
        proc = subprocess.run(  # noqa: S603 — trusted CLI
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise KaggleCLIError("`kaggle` コマンドが見つかりません。") from exc

    combined = f"{proc.stdout}\n{proc.stderr}".strip()
    upload_done = "100%" in proc.stderr or "Successfully submitted" in proc.stdout
    if proc.returncode == 0 or upload_done:
        return combined
    raise KaggleCLIError(f"提出失敗 (rc={proc.returncode}):\n{proc.stderr.strip()}")


def confirm_submission(
    message: str,
    *,
    timeout_s: int = 60,
    interval_s: int = 5,
) -> dict[str, str] | None:
    """履歴 API から提出が記録されたかを確認する。

    CLI のテキスト判定に頼らず、`list_submissions()` の description 一致で成否を
    判断する。タイムアウトまでに該当行が見つからなければ None を返す。
    """

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            rows = list_submissions()
        except KaggleCLIError:
            time.sleep(interval_s)
            continue
        for row in rows:
            description = row.get("description") or row.get("message") or ""
            if message in description:
                return row
        time.sleep(interval_s)
    return None


def list_submissions() -> list[dict[str, str]]:
    """`kaggle competitions submissions -v` の CSV 出力をパースして返す。"""

    out = _run(["competitions", "submissions", COMPETITION, "-v"])
    reader = csv.DictReader(io.StringIO(out))
    return [dict(row) for row in reader]


_DATE_PATTERNS: tuple[str, ...] = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
)


def _parse_utc_date(value: str) -> date | None:
    value = value.strip()
    if not value:
        return None
    for pat in _DATE_PATTERNS:
        try:
            return datetime.strptime(value, pat).replace(tzinfo=UTC).date()
        except ValueError:
            continue
    return None


def count_today(submissions: list[dict[str, str]] | None = None) -> int:
    """UTC 当日の提出件数を数える。パース不能時は保守的に 5 を返す。"""

    try:
        rows = submissions if submissions is not None else list_submissions()
    except KaggleCLIError:
        return 5
    today = datetime.now(UTC).date()
    count = 0
    for row in rows:
        date_str = row.get("date") or row.get("Date") or row.get("submittedAt") or ""
        parsed = _parse_utc_date(date_str)
        if parsed == today:
            count += 1
    return count


def _extract_submission_id(stdout: str) -> str | None:
    match = re.search(r"submission\s*id[:\s]+(\d+)", stdout, re.IGNORECASE)
    return match.group(1) if match else None


def poll(
    submission_ref: str | None,
    *,
    timeout_s: int = 300,
    interval_s: int = 30,
) -> dict[str, Any]:
    """最新提出の status を一定時間ポーリングする。

    Args:
        submission_ref: 提出IDやメッセージなど、`description` に含まれる一意な
            文字列。None の場合は最新行を対象にする。
    """

    deadline = time.time() + timeout_s
    last: dict[str, str] = {}
    while time.time() < deadline:
        rows = list_submissions()
        if not rows:
            time.sleep(interval_s)
            continue
        target = rows[0]
        if submission_ref:
            for row in rows:
                if submission_ref in (
                    row.get("description") or row.get("message") or ""
                ):
                    target = row
                    break
        last = target
        status = (target.get("status") or "").lower()
        if status not in {"pending", "running", ""}:
            return {"status": status, "row": target}
        time.sleep(interval_s)
    return {"status": "timeout", "row": last}


__all__ = [
    "COMPETITION",
    "KaggleCLIError",
    "_extract_submission_id",
    "confirm_submission",
    "count_today",
    "list_submissions",
    "poll",
    "submit",
]
