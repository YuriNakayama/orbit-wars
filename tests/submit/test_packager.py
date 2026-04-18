"""packager のユニットテスト。"""

from __future__ import annotations

import tarfile
from pathlib import Path

import pytest

from submit.packager import PackagingError, build_archive


def _write(path: Path, content: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_build_archive_creates_targz_with_main_at_root(tmp_path: Path) -> None:
    case_dir = tmp_path / "case" / "case0"
    _write(case_dir / "main.py", "def agent(obs):\n    return []\n")
    _write(case_dir / "helper.py", "x = 1\n")

    out_dir = tmp_path / "out"
    archive = build_archive(case_dir, out_dir)

    assert archive.parent == out_dir
    assert archive.suffix == ".gz"
    with tarfile.open(archive, "r:gz") as tar:
        names = sorted(tar.getnames())
    assert "main.py" in names
    assert "helper.py" in names


def test_build_archive_excludes_non_whitelisted_files(tmp_path: Path) -> None:
    """.md や docs など非ホワイトリスト拡張子は既定で除外される。"""

    case_dir = tmp_path / "case" / "caseN"
    _write(case_dir / "main.py", "def agent(obs):\n    return []\n")
    _write(case_dir / "README.md", "# readme")
    _write(case_dir / "agents.md", "doc")
    _write(case_dir / "__pycache__" / "main.cpython-314.pyc", "bin")
    _write(case_dir / "run.log", "log")

    archive = build_archive(case_dir, tmp_path / "out")
    with tarfile.open(archive, "r:gz") as tar:
        names = tar.getnames()
    assert "main.py" in names
    assert "README.md" not in names
    assert "agents.md" not in names
    assert "run.log" not in names
    assert not any("__pycache__" in n for n in names)


def test_build_archive_includes_model_weights(tmp_path: Path) -> None:
    """モデル重み（.pkl/.pt 等）はホワイトリストに含まれる。"""

    case_dir = tmp_path / "case" / "caseM"
    _write(case_dir / "main.py", "def agent(obs):\n    return []\n")
    _write(case_dir / "weights.pkl", "binary")
    _write(case_dir / "config.json", "{}")

    archive = build_archive(case_dir, tmp_path / "out")
    with tarfile.open(archive, "r:gz") as tar:
        names = sorted(tar.getnames())
    assert "main.py" in names
    assert "weights.pkl" in names
    assert "config.json" in names


def test_build_archive_uses_gnu_format_and_deterministic_meta(
    tmp_path: Path,
) -> None:
    """GNU_FORMAT で uid/gid/mtime が決定的であることを確認する。"""

    case_dir = tmp_path / "case" / "caseD"
    _write(case_dir / "main.py", "def agent(obs):\n    return []\n")

    archive = build_archive(case_dir, tmp_path / "out")
    with tarfile.open(archive, "r:gz") as tar:
        member = tar.getmember("main.py")
        assert member.uid == 0
        assert member.gid == 0
        assert member.uname == ""
        assert member.gname == ""
        assert member.mtime == 0
        assert member.mode == 0o644


def test_build_archive_single_file_returns_main_py(tmp_path: Path) -> None:
    case_dir = tmp_path / "case" / "case0"
    _write(case_dir / "main.py", "def agent(obs):\n    return []\n")

    result = build_archive(case_dir, tmp_path / "out", single_file=True)
    assert result.name == "main.py"
    assert result.parent == case_dir.resolve()


def test_build_archive_missing_main_raises(tmp_path: Path) -> None:
    case_dir = tmp_path / "case" / "broken"
    case_dir.mkdir(parents=True)
    _write(case_dir / "helper.py")

    with pytest.raises(PackagingError):
        build_archive(case_dir, tmp_path / "out")


def test_build_archive_missing_case_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(PackagingError):
        build_archive(tmp_path / "nope", tmp_path / "out")
