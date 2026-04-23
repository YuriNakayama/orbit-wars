"""ensure_dvc_artifacts のユニットテスト。"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from submit.packager import ensure_dvc_artifacts


def _make_fake_repo(tmp_path: Path, outs: list[str]) -> Path:
    """dvc.yaml + .git ディレクトリを含むダミーリポジトリを作成する。"""
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / "backend").mkdir()
    outs_block = "\n".join(f"      - {p}" for p in outs)
    (repo / "dvc.yaml").write_text(
        f"stages:\n  train:\n    cmd: echo\n    outs:\n{outs_block}\n",
        encoding="utf-8",
    )
    return repo


def test_ensure_dvc_artifacts_no_repo_returns_empty(tmp_path: Path) -> None:
    case_dir = tmp_path / "not_a_repo_case"
    case_dir.mkdir()
    assert ensure_dvc_artifacts(case_dir) == []


def test_ensure_dvc_artifacts_skips_when_all_outs_exist(tmp_path: Path) -> None:
    repo = _make_fake_repo(tmp_path, outs=["backend/pipeline/case1/policy/weights.pt"])
    case_dir = repo / "backend/pipeline/case1"
    weights = case_dir / "policy/weights.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"dummy model bytes")

    with patch("submit.packager.subprocess.run") as run_mock:
        relevant = ensure_dvc_artifacts(case_dir)

    assert [p.name for p in relevant] == ["weights.pt"]
    run_mock.assert_not_called()


def test_ensure_dvc_artifacts_invokes_dvc_pull_for_missing(tmp_path: Path) -> None:
    repo = _make_fake_repo(tmp_path, outs=["backend/pipeline/case1/policy/weights.pt"])
    case_dir = repo / "backend/pipeline/case1"
    case_dir.mkdir(parents=True)
    # weights.pt は未生成

    with patch("submit.packager.subprocess.run") as run_mock:
        ensure_dvc_artifacts(case_dir)

    run_mock.assert_called_once()
    args, kwargs = run_mock.call_args
    cmd = args[0]
    assert cmd[:2] == ["uv", "run"]
    assert "dvc" in cmd and "pull" in cmd
    assert cmd[-1] == "backend/pipeline/case1/policy/weights.pt"
    assert kwargs["cwd"] == repo
    assert kwargs["check"] is True


def test_ensure_dvc_artifacts_ignores_outs_outside_case_dir(tmp_path: Path) -> None:
    repo = _make_fake_repo(
        tmp_path,
        outs=[
            "data/mart/case1/train.parquet",  # case_dir 外
            "backend/pipeline/case1/policy/weights.pt",
        ],
    )
    case_dir = repo / "backend/pipeline/case1"
    case_dir.mkdir(parents=True)

    with patch("submit.packager.subprocess.run") as run_mock:
        ensure_dvc_artifacts(case_dir)

    run_mock.assert_called_once()
    cmd = run_mock.call_args.args[0]
    # case_dir 配下のものだけ pull 対象
    assert "backend/pipeline/case1/policy/weights.pt" in cmd
    assert "data/mart/case1/train.parquet" not in cmd


def test_ensure_dvc_artifacts_no_dvc_yaml_returns_empty(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / "backend").mkdir()
    # dvc.yaml なし → repo root と認識されない
    case_dir = repo / "backend/pipeline/case1"
    case_dir.mkdir(parents=True)
    assert ensure_dvc_artifacts(case_dir) == []


@pytest.mark.parametrize("exists", [True, False])
def test_build_archive_calls_ensure_dvc_artifacts(tmp_path: Path, exists: bool) -> None:
    """build_archive が冒頭で ensure_dvc_artifacts を呼ぶことを確認。"""
    from submit.packager import build_archive

    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "main.py").write_text("def agent(obs): return []\n")
    if exists:
        (case_dir / "helper.py").write_text("x = 1\n")

    out_dir = tmp_path / "out"
    with patch("submit.packager.ensure_dvc_artifacts") as hook:
        hook.return_value = []
        build_archive(case_dir, out_dir)

    hook.assert_called_once_with(case_dir.resolve())
