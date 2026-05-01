"""提出履歴を data/output/submit/<case>/submissions.jsonl に追記する。"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def record(
    *,
    base_dir: Path,
    case: str,
    message: str,
    archive: Path | None,
    dry_run: bool,
    result: dict[str, Any] | None,
) -> Path:
    """submissions.jsonl に 1 行追記する。

    Args:
        base_dir: `data/output/submit` のルート。
        case: case 名。
        message: 提出メッセージ。
        archive: 生成ファイル（ローカル）。
        dry_run: 実提出したかどうか。
        result: Kaggle 側からの結果サマリ。
    """

    log_dir = base_dir / case
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "submissions.jsonl"
    entry = {
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "case": case,
        "message": message,
        "archive": str(archive) if archive else None,
        "dry_run": dry_run,
        "result": result,
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return log_path
