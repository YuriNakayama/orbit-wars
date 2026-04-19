"""`kaggle competitions leaderboard` の subprocess ラッパー。

`src/submit/kaggle_api.py` と同じ `KaggleCLIError` 系譜で例外を揃える。
認証は `~/.kaggle/kaggle.json` or 環境変数 (KAGGLE_USERNAME / KAGGLE_KEY) を利用。
"""

from __future__ import annotations

import csv
import io
import subprocess
from pathlib import Path

from dataset.kaggle.types import TeamRank

COMPETITION = "orbit-wars"


class KaggleLeaderboardError(RuntimeError):
    """leaderboard CLI 呼び出しに失敗した場合に raise。"""


def _run_cli(args: list[str]) -> str:
    try:
        proc = subprocess.run(  # noqa: S603 — trusted CLI
            ["kaggle", *args],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise KaggleLeaderboardError(
            "`kaggle` コマンドが見つかりません。`uv sync` 後に実行してください。"
        ) from exc
    if proc.returncode != 0:
        raise KaggleLeaderboardError(
            f"kaggle CLI 実行失敗 (args={args}):\nstderr: {proc.stderr.strip()}"
        )
    return proc.stdout


def _parse_team_id(row: dict[str, str]) -> int:
    for key in ("teamId", "TeamId", "team_id"):
        raw = row.get(key)
        if raw:
            return int(raw)
    raise KaggleLeaderboardError(f"team id not found in row: {list(row.keys())}")


def _parse_score(row: dict[str, str]) -> float:
    for key in ("score", "Score", "publicScore"):
        raw = row.get(key)
        if raw:
            try:
                return float(raw)
            except ValueError:
                continue
    return 0.0


def _strip_preamble(csv_text: str) -> str:
    """CLI stdout の先頭に `Next Page Token = ...` 等のノイズが混じる場合に除去。"""

    lines = csv_text.splitlines()
    for i, line in enumerate(lines):
        if "teamId" in line or "TeamId" in line:
            return "\n".join(lines[i:])
    return csv_text


def _parse_ranks(csv_text: str, top: int) -> list[TeamRank]:
    reader = csv.DictReader(io.StringIO(_strip_preamble(csv_text)))
    ranks: list[TeamRank] = []
    for i, row in enumerate(reader, start=1):
        if i > top:
            break
        ranks.append(
            TeamRank(
                rank=i,
                team_id=_parse_team_id(row),
                team_name=row.get("teamName") or row.get("TeamName") or "",
                score=_parse_score(row),
                submission_date=row.get("submissionDate")
                or row.get("SubmissionDate")
                or "",
            )
        )
    return ranks


def fetch(top: int, *, raw_snapshot: Path | None = None) -> list[TeamRank]:
    """Kaggle CLI からリーダーボード上位 N team を取得する。"""

    if top <= 0:
        raise ValueError(f"top must be positive, got {top}")
    page_size = min(max(top, 20), 200)
    stdout = _run_cli(
        [
            "competitions",
            "leaderboard",
            "-c",
            COMPETITION,
            "--show",
            "--csv",
            "--page-size",
            str(page_size),
        ]
    )
    if raw_snapshot is not None:
        raw_snapshot.parent.mkdir(parents=True, exist_ok=True)
        raw_snapshot.write_text(stdout, encoding="utf-8")
    return _parse_ranks(stdout, top)
