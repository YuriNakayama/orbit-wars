"""bot/ から Kaggle Dataset 用 snapshot ディレクトリを作る。

除外規則:
    - data/, .venv/, __pycache__/, *.pyc, .dvc/, .git/, docs/, infra/,
      bot/tests/, node_modules/

包含:
    - bot/src/, bot/pipeline/, bot/pyproject.toml, bot/uv.lock,
      simulator/python/, simulator/rust/src/, params.yaml
    - include_wheels で渡された .whl は <dest>/wheels/ に配置
    - include_mart_files で渡された data/mart/... ファイルは元の相対 path で配置
    - Kaggle 側 ``find_repo_root()`` 解決のため空の <dest>/.git/HEAD を作成
"""

from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Iterable
from pathlib import Path

logger = logging.getLogger(__name__)

EXCLUDE_DIR_NAMES: frozenset[str] = frozenset(
    {
        ".git",
        ".venv",
        ".dvc",
        "__pycache__",
        "node_modules",
        "data",
        "docs",
        "infra",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "target",
    }
)

INCLUDE_RELATIVE_PATHS: tuple[str, ...] = (
    "bot/src",
    "bot/pipeline",
    "bot/pyproject.toml",
    "bot/uv.lock",
    "bot/.python-version",
    "simulator/python",
    "simulator/rust/src",
    "simulator/rust/Cargo.toml",
    "simulator/rust/pyproject.toml",
    "params.yaml",
)


def build_snapshot(
    repo_root: Path,
    dest_dir: Path,
    *,
    include_wheels: Iterable[Path] | None = None,
    include_mart_files: Iterable[Path] | None = None,
    include_dotenv: bool = False,
) -> Path:
    """``repo_root`` から ``dest_dir`` にコード snapshot を作成する。

    ``dest_dir`` が既存なら中身を削除して作り直す。
    ``include_wheels`` で渡された .whl は ``<dest_dir>/wheels/`` にコピー。
    ``include_mart_files`` で渡された ``data/mart/...`` 配下の絶対 path は
    repo_root からの相対 path を保ったまま dest_dir に配置する。
    Kaggle 側 ``find_repo_root`` が ``(bot/, .git/)`` ペアで repo root を
    認識するため、空の ``<dest_dir>/.git/HEAD`` を必ず作成する。
    ``include_dotenv`` を True にすると ``bot/.env`` を snapshot に同梱する
    (interactive mode の AWS creds 配送用、private dataset 前提)。
    戻り値は ``dest_dir`` (絶対 path)。
    """
    repo_root = repo_root.resolve()
    dest_dir = dest_dir.resolve()

    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True)

    for rel in INCLUDE_RELATIVE_PATHS:
        src = repo_root / rel
        if not src.exists():
            logger.debug("skip absent path: %s", src)
            continue
        target = dest_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if src.is_file():
            shutil.copy2(src, target)
        else:
            shutil.copytree(
                src, target, ignore=shutil.ignore_patterns(*_ignore_patterns())
            )

    if include_wheels:
        wheels_dir = dest_dir / "wheels"
        wheels_dir.mkdir(parents=True, exist_ok=True)
        for wheel in include_wheels:
            wheel = wheel.resolve()
            if not wheel.is_file():
                raise FileNotFoundError(f"wheel not found: {wheel}")
            if wheel.suffix != ".whl":
                raise ValueError(f"not a wheel file: {wheel}")
            shutil.copy2(wheel, wheels_dir / wheel.name)

    if include_mart_files:
        for src in include_mart_files:
            # worktree 配下では data/ が main repo data/ への symlink になる
            # ことがある。relative_to は symlink を follow しないため、
            # `..` の lexical 正規化のみ行い、symlink は follow しないように
            # normpath で重ねて解決する。コピー時のみ resolve() で実体を読む。
            src_norm = Path(os.path.normpath(str(src.absolute())))
            if not src_norm.is_file():
                raise FileNotFoundError(f"mart file not found: {src_norm}")
            try:
                rel_path = src_norm.relative_to(repo_root)
            except ValueError as e:
                raise ValueError(
                    f"mart file {src_norm} is not under repo_root {repo_root}"
                ) from e
            target = dest_dir / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_norm.resolve(), target)

    git_dir = dest_dir / ".git"
    git_dir.mkdir(exist_ok=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/kaggle-snapshot\n", encoding="utf-8")

    if include_dotenv:
        env_src = repo_root / "bot" / ".env"
        if not env_src.is_file():
            raise FileNotFoundError(
                f"include_dotenv requested but {env_src} does not exist"
            )
        env_dst = dest_dir / "bot" / ".env"
        env_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(env_src, env_dst)

    return dest_dir


def _ignore_patterns() -> tuple[str, ...]:
    return tuple(EXCLUDE_DIR_NAMES) + ("*.pyc", "*.pyo")
