"""validator のスモークテスト。case0 を実際にロードして最低限のターンを実行する。"""

from __future__ import annotations

from pathlib import Path

import pytest

from submit.validator import ValidationError, dry_run

# this file: bot/tests/src/submit/test_validator.py → parents[3] = bot/
CASE0 = Path(__file__).resolve().parents[4] / "pipeline" / "rulebase" / "case0"


pytestmark = pytest.mark.slow


def test_dry_run_case0_completes() -> None:
    result = dry_run(CASE0, turns=20)
    assert result["steps"] >= 1
    assert result["final_statuses"]


def test_dry_run_missing_main_raises(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        dry_run(tmp_path, turns=5)
