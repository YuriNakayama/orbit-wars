"""history のユニットテスト。"""

from __future__ import annotations

import json
from pathlib import Path

from submit.history import record


def test_record_appends_jsonl(tmp_path: Path) -> None:
    log = record(
        base_dir=tmp_path,
        case="case0",
        message="test",
        archive=tmp_path / "a.tar.gz",
        dry_run=False,
        result={"ok": True},
    )
    assert log.exists()
    record(
        base_dir=tmp_path,
        case="case0",
        message="test2",
        archive=None,
        dry_run=True,
        result=None,
    )
    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    entries = [json.loads(line) for line in lines]
    assert entries[0]["case"] == "case0"
    assert entries[0]["message"] == "test"
    assert entries[1]["dry_run"] is True
    assert entries[1]["archive"] is None


def test_record_creates_missing_directory(tmp_path: Path) -> None:
    base = tmp_path / "deep" / "nest"
    record(
        base_dir=base,
        case="case1",
        message="m",
        archive=None,
        dry_run=True,
        result=None,
    )
    assert (base / "case1" / "submissions.jsonl").is_file()
