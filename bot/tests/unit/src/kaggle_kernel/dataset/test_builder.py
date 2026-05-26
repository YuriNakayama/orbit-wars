"""kaggle_kernel.dataset.builder のユニットテスト。"""

from __future__ import annotations

from pathlib import Path

import pytest

from kaggle_kernel.dataset.builder import (
    EXCLUDE_DIR_NAMES,
    INCLUDE_RELATIVE_PATHS,
    build_snapshot,
)


def _fake_repo(root: Path) -> None:
    """簡易な orbit-wars repo を作る。"""
    # included
    (root / "bot" / "src").mkdir(parents=True)
    (root / "bot" / "src" / "module_a.py").write_text("x = 1\n", encoding="utf-8")
    (root / "bot" / "pipeline").mkdir()
    (root / "bot" / "pipeline" / "case1.py").write_text("y = 2\n", encoding="utf-8")
    (root / "bot" / "pyproject.toml").write_text(
        "[project]\nname='bot'\n", encoding="utf-8"
    )
    (root / "bot" / "uv.lock").write_text("# lock\n", encoding="utf-8")
    (root / "params.yaml").write_text("seed: 0\n", encoding="utf-8")
    (root / "simulator" / "python").mkdir(parents=True)
    (root / "simulator" / "python" / "engine.py").write_text(
        "z = 3\n", encoding="utf-8"
    )
    (root / "simulator" / "rust" / "src").mkdir(parents=True)
    (root / "simulator" / "rust" / "src" / "lib.rs").write_text(
        "// rust\n", encoding="utf-8"
    )
    (root / "simulator" / "rust" / "Cargo.toml").write_text(
        "# cargo\n", encoding="utf-8"
    )
    # excluded
    (root / "data" / "lake").mkdir(parents=True)
    (root / "data" / "lake" / "huge.parquet").write_text("BIG", encoding="utf-8")
    (root / ".git").mkdir()
    (root / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (root / "docs").mkdir()
    (root / "docs" / "competition.md").write_text("# spec\n", encoding="utf-8")
    (root / "bot" / "src" / "__pycache__").mkdir()
    (root / "bot" / "src" / "__pycache__" / "x.cpython-313.pyc").write_text(
        "bytecode", encoding="utf-8"
    )


def test_build_snapshot_includes_expected_paths(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    dest = tmp_path / "snapshot"
    _fake_repo(repo)
    build_snapshot(repo, dest)
    assert (dest / "bot" / "src" / "module_a.py").is_file()
    assert (dest / "bot" / "pipeline" / "case1.py").is_file()
    assert (dest / "bot" / "pyproject.toml").is_file()
    assert (dest / "bot" / "uv.lock").is_file()
    assert (dest / "simulator" / "python" / "engine.py").is_file()
    assert (dest / "simulator" / "rust" / "src" / "lib.rs").is_file()


def test_build_snapshot_excludes_data_docs(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    dest = tmp_path / "snapshot"
    _fake_repo(repo)
    build_snapshot(repo, dest)
    assert not (dest / "data").exists()
    assert not (dest / "docs").exists()


def test_build_snapshot_creates_git_head_marker(tmp_path: Path) -> None:
    """find_repo_root が ``(bot/, .git/)`` を探すため、空の .git/HEAD を置く。"""
    repo = tmp_path / "repo"
    dest = tmp_path / "snapshot"
    _fake_repo(repo)
    build_snapshot(repo, dest)
    assert (dest / ".git" / "HEAD").is_file()
    assert (dest / "bot").is_dir()


def test_build_snapshot_excludes_pycache(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    dest = tmp_path / "snapshot"
    _fake_repo(repo)
    build_snapshot(repo, dest)
    pyc_files = list(dest.rglob("*.pyc"))
    pycache_dirs = list(dest.rglob("__pycache__"))
    assert pyc_files == []
    assert pycache_dirs == []


def test_build_snapshot_overwrites_existing_dest(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    dest = tmp_path / "snapshot"
    _fake_repo(repo)
    dest.mkdir()
    (dest / "stale.txt").write_text("old", encoding="utf-8")
    build_snapshot(repo, dest)
    assert not (dest / "stale.txt").exists()
    assert (dest / "bot" / "src" / "module_a.py").is_file()


def test_build_snapshot_includes_params_yaml(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    dest = tmp_path / "snapshot"
    _fake_repo(repo)
    build_snapshot(repo, dest)
    assert (dest / "params.yaml").is_file()


def test_build_snapshot_includes_mart_files(tmp_path: Path) -> None:
    """Kaggle dataset の top-level `data/` zip drop 回避のため、mart は
    `kmart__<case>__<split>.parquet` という flat なファイル名で配置される。
    """
    repo = tmp_path / "repo"
    dest = tmp_path / "snapshot"
    _fake_repo(repo)
    mart_dir = repo / "data" / "mart" / "imitation" / "case1"
    mart_dir.mkdir(parents=True)
    train_pq = mart_dir / "train.parquet"
    val_pq = mart_dir / "val.parquet"
    train_pq.write_bytes(b"PAR1-fake-train")
    val_pq.write_bytes(b"PAR1-fake-val")
    build_snapshot(repo, dest, include_mart_files=[train_pq, val_pq])
    assert (dest / "kmart__case1__train.parquet").is_file()
    assert (dest / "kmart__case1__val.parquet").is_file()


def test_build_snapshot_mart_file_outside_repo_accepted(tmp_path: Path) -> None:
    """worktree の data/ symlink が壊れた場合などに main repo の data/ を
    直接 --mart で渡せるよう、path に `data/` segment があれば repo_root の外でも
    `kmart__<case>__<split>.parquet` として flatten 配置する。
    """
    repo = tmp_path / "repo"
    dest = tmp_path / "snapshot"
    _fake_repo(repo)
    outside_mart = tmp_path / "main" / "data" / "mart" / "imitation" / "caseX"
    outside_mart.mkdir(parents=True)
    outside_train = outside_mart / "train.parquet"
    outside_train.write_bytes(b"x")
    # path 内に `data` segment があれば flatten path 経路で受理される
    build_snapshot(repo, dest, include_mart_files=[outside_train])
    assert (dest / "kmart__caseX__train.parquet").is_file()


def test_build_snapshot_mart_file_with_no_data_segment_rejected(
    tmp_path: Path,
) -> None:
    """path に `data/` segment も無く repo_root の外でもある場合は relative_to
    が失敗して ValueError が出る。
    """
    repo = tmp_path / "repo"
    dest = tmp_path / "snapshot"
    _fake_repo(repo)
    outside = tmp_path / "outside.parquet"
    outside.write_bytes(b"x")
    with pytest.raises(ValueError):
        build_snapshot(repo, dest, include_mart_files=[outside])


def test_build_snapshot_missing_mart_file_raises(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    dest = tmp_path / "snapshot"
    _fake_repo(repo)
    missing = repo / "data" / "mart" / "missing.parquet"
    with pytest.raises(FileNotFoundError):
        build_snapshot(repo, dest, include_mart_files=[missing])


def test_build_snapshot_excludes_dotenv_by_default(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    dest = tmp_path / "snapshot"
    _fake_repo(repo)
    (repo / "bot" / ".env").write_text("SECRET=value\n", encoding="utf-8")
    build_snapshot(repo, dest)
    assert not (dest / "bot" / ".env").exists()


def test_build_snapshot_includes_dotenv_when_opted_in(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    dest = tmp_path / "snapshot"
    _fake_repo(repo)
    (repo / "bot" / ".env").write_text(
        "AWS_ACCESS_KEY_ID=k\nAWS_SECRET_ACCESS_KEY=s\n", encoding="utf-8"
    )
    build_snapshot(repo, dest, include_dotenv=True)
    env_path = dest / "bot" / ".env"
    assert env_path.is_file()
    content = env_path.read_text()
    assert "AWS_ACCESS_KEY_ID=k" in content


def test_build_snapshot_dotenv_missing_raises(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    dest = tmp_path / "snapshot"
    _fake_repo(repo)
    # No bot/.env created
    with pytest.raises(FileNotFoundError, match="include_dotenv"):
        build_snapshot(repo, dest, include_dotenv=True)


def test_build_snapshot_includes_wheels(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    dest = tmp_path / "snapshot"
    _fake_repo(repo)
    wheel = tmp_path / "orbit_wars_rust-0.1.0-cp313-cp313-manylinux_2_17_x86_64.whl"
    wheel.write_bytes(b"PK\x03\x04fake-wheel-bytes")
    build_snapshot(repo, dest, include_wheels=[wheel])
    assert (dest / "wheels" / wheel.name).is_file()


def test_build_snapshot_non_wheel_rejected(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    dest = tmp_path / "snapshot"
    _fake_repo(repo)
    not_wheel = tmp_path / "x.tar.gz"
    not_wheel.write_bytes(b"tarball")
    with pytest.raises(ValueError, match="not a wheel"):
        build_snapshot(repo, dest, include_wheels=[not_wheel])


def test_build_snapshot_missing_wheel_raises(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    dest = tmp_path / "snapshot"
    _fake_repo(repo)
    missing = tmp_path / "missing-0.1.0.whl"
    with pytest.raises(FileNotFoundError):
        build_snapshot(repo, dest, include_wheels=[missing])


def test_exclude_dir_names_contains_critical() -> None:
    assert "data" in EXCLUDE_DIR_NAMES
    assert ".git" in EXCLUDE_DIR_NAMES
    assert "__pycache__" in EXCLUDE_DIR_NAMES


def test_include_paths_static() -> None:
    assert "bot/src" in INCLUDE_RELATIVE_PATHS
    assert "bot/pipeline" in INCLUDE_RELATIVE_PATHS
