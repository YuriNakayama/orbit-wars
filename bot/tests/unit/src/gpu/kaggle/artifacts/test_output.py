"""gpu.kaggle.artifacts.output のユニットテスト (KaggleApi / subprocess mock)。"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gpu.kaggle.artifacts.output import (
    dvc_add,
    place_into_run_dir,
    pull_kernel_output,
)


def test_pull_kernel_output_calls_api(tmp_path: Path) -> None:
    api = MagicMock()
    api.kernels_output_cli.return_value = None
    result = pull_kernel_output(api, "yuri/run0", tmp_path / "tmp")
    api.kernels_output_cli.assert_called_once_with(
        kernel="yuri/run0",
        path=str(tmp_path / "tmp"),
        force=True,
        quiet=False,
        file_pattern=None,
    )
    assert result == tmp_path / "tmp"
    assert result.is_dir()


def test_pull_kernel_output_passes_file_pattern(tmp_path: Path) -> None:
    api = MagicMock()
    api.kernels_output_cli.return_value = None
    pull_kernel_output(api, "yuri/run0", tmp_path / "tmp", file_pattern="^runs/")
    api.kernels_output_cli.assert_called_once_with(
        kernel="yuri/run0",
        path=str(tmp_path / "tmp"),
        force=True,
        quiet=False,
        file_pattern="^runs/",
    )


def test_place_into_run_dir_prefers_nested(tmp_path: Path) -> None:
    tmp = tmp_path / "tmp"
    (tmp / "runs" / "r0").mkdir(parents=True)
    (tmp / "runs" / "r0" / "best.pt").write_bytes(b"weights")
    (tmp / "runs" / "r0" / "run.json").write_text('{"a": 1}', encoding="utf-8")
    (tmp / "stray.txt").write_text("ignored", encoding="utf-8")
    run_dir = tmp_path / "run_dir"
    place_into_run_dir(tmp, run_dir, run_id="r0")
    assert (run_dir / "best.pt").is_file()
    assert (run_dir / "run.json").is_file()
    assert not (run_dir / "stray.txt").exists()


def test_place_into_run_dir_fallback_to_tmp_root(tmp_path: Path) -> None:
    """nested 'runs/<run_id>' が無い場合は tmp_dir 直下を取り込む。"""
    tmp = tmp_path / "tmp"
    tmp.mkdir()
    (tmp / "best.pt").write_bytes(b"x")
    run_dir = tmp_path / "run_dir"
    place_into_run_dir(tmp, run_dir, run_id="r0")
    assert (run_dir / "best.pt").is_file()


def test_place_into_run_dir_overwrites_existing_subdir(tmp_path: Path) -> None:
    tmp = tmp_path / "tmp"
    (tmp / "runs" / "r0" / "subdir").mkdir(parents=True)
    (tmp / "runs" / "r0" / "subdir" / "new.txt").write_text("new", encoding="utf-8")
    run_dir = tmp_path / "run_dir"
    (run_dir / "subdir").mkdir(parents=True)
    (run_dir / "subdir" / "old.txt").write_text("old", encoding="utf-8")
    place_into_run_dir(tmp, run_dir, run_id="r0")
    assert (run_dir / "subdir" / "new.txt").is_file()
    assert not (run_dir / "subdir" / "old.txt").exists()


def test_dvc_add_calls_subprocess(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "r0"
    run_dir.mkdir(parents=True)
    with patch("gpu.kaggle.artifacts.output.subprocess.run") as run_mock:
        run_mock.return_value = MagicMock(returncode=0)
        dvc_add(run_dir, tmp_path)
        run_mock.assert_called_once_with(
            ["dvc", "add", str(run_dir)],
            cwd=str(tmp_path),
            check=True,
        )


def test_dvc_add_propagates_failure(tmp_path: Path) -> None:
    import subprocess as sp_module

    run_dir = tmp_path / "runs" / "r0"
    run_dir.mkdir(parents=True)
    with patch("gpu.kaggle.artifacts.output.subprocess.run") as run_mock:
        run_mock.side_effect = sp_module.CalledProcessError(1, ["dvc", "add"])
        with pytest.raises(sp_module.CalledProcessError):
            dvc_add(run_dir, tmp_path)
