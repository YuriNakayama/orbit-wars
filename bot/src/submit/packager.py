"""pipeline/<case>/ を Kaggle 提出用 tar.gz にパッケージする。

Kaggle の simulation runtime は `main.py` の実行に特化しており、非 .py の
付随ファイル（README.md 等）を同梱すると validation が ERROR になる事例が
ある。そのため既定ではホワイトリスト拡張子のみ同梱する。
"""

from __future__ import annotations

import fnmatch
import logging
import subprocess
import tarfile
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_INCLUDE_PATTERNS: tuple[str, ...] = (
    "*.py",
    "*.pkl",
    "*.npz",
    "*.npy",
    "*.pt",
    "*.pth",
    "*.bin",
    "*.json",
    "*.yaml",
    "*.yml",
    "*.txt",
)

EXCLUDE_DIR_NAMES: frozenset[str] = frozenset(
    {
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".git",
    }
)

SUBMITIGNORE_FILENAME = ".submitignore"


class PackagingError(RuntimeError):
    """パッケージング中に発生するエラー。"""


def _find_repo_root(case_dir: Path) -> Path | None:
    for p in case_dir.resolve().parents:
        if (p / ".git").exists() and (p / "bot").is_dir() and (p / "dvc.yaml").exists():
            return p
    return None


def _dvc_managed_outs(repo_root: Path) -> set[Path]:
    """`dvc.yaml` の outs に宣言された絶対パスを集める。"""
    dvc_yaml = repo_root / "dvc.yaml"
    if not dvc_yaml.is_file():
        return set()
    try:
        import yaml  # local import to keep non-DVC flows independent
    except ImportError:  # pragma: no cover
        return set()
    data = yaml.safe_load(dvc_yaml.read_text(encoding="utf-8")) or {}
    stages = data.get("stages") or {}
    outs: set[Path] = set()
    for stage in stages.values():
        for item in stage.get("outs") or []:
            raw = item if isinstance(item, str) else next(iter(item.keys()))
            outs.add((repo_root / raw).resolve())
    return outs


def ensure_dvc_artifacts(case_dir: Path) -> list[Path]:
    """`dvc.yaml` outs のうち case_dir 配下にある **欠損ファイル** を dvc pull する。

    存在するファイルには touch しない
    （cache.type=symlink の read-only 保護を壊さない）。
    """
    repo_root = _find_repo_root(case_dir)
    if repo_root is None:
        return []
    managed = _dvc_managed_outs(repo_root)
    if not managed:
        return []
    case_dir_resolved = case_dir.resolve()
    relevant = [p for p in managed if case_dir_resolved in p.parents]
    missing = [p for p in relevant if not p.exists()]
    if not missing:
        return relevant

    rels = [str(p.relative_to(repo_root)) for p in missing]
    logger.info("dvc pull for submission artifacts: %s", rels)
    subprocess.run(
        ["uv", "run", "--directory", "bot", "dvc", "pull", *rels],
        cwd=repo_root,
        check=True,
    )
    return relevant


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def _is_included(relative: Path, include_patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(relative.name, pat) for pat in include_patterns)


def _parse_submitignore(path: Path) -> list[str]:
    """`.submitignore` を読み、非コメント・非空行のパターンを返す。"""
    patterns: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        patterns.append(line)
    return patterns


def _matches_submitignore(relative: Path, patterns: list[str]) -> bool:
    """case_dir 相対パスが除外パターンに該当するか判定する。

    - 末尾 `/`: ディレクトリ指定。配下すべてにマッチ。
    - それ以外: fnmatch でパス全体とファイル名の両方に評価。
    """
    rel_posix = relative.as_posix()
    for pat in patterns:
        if pat.endswith("/"):
            prefix = pat.rstrip("/")
            if prefix in relative.parts:
                return True
            continue
        if fnmatch.fnmatch(rel_posix, pat):
            return True
        if fnmatch.fnmatch(relative.name, pat):
            return True
    return False


def _load_submitignore(case_dir: Path) -> list[str]:
    """`pipeline/.submitignore` (pipeline ルート直下) を読む。存在しなければ空。

    case_dir は `pipeline/<category>/<case>` の 2 階層構造を想定し、
    pipeline ルートは `case_dir.parent.parent`。
    """
    ignore_path = case_dir.parent.parent / SUBMITIGNORE_FILENAME
    if not ignore_path.is_file():
        return []
    return _parse_submitignore(ignore_path)


def _iter_case_files(case_dir: Path, include_patterns: tuple[str, ...]) -> list[Path]:
    ignore_patterns = _load_submitignore(case_dir)
    files: list[Path] = []
    for entry in sorted(case_dir.rglob("*")):
        if entry.is_dir():
            continue
        rel = entry.relative_to(case_dir)
        if any(part in EXCLUDE_DIR_NAMES for part in rel.parts):
            continue
        if ignore_patterns and _matches_submitignore(rel, ignore_patterns):
            continue
        if not _is_included(rel, include_patterns):
            continue
        files.append(entry)
    return files


def build_archive(
    case_dir: Path,
    out_dir: Path,
    *,
    single_file: bool = False,
    include_patterns: tuple[str, ...] = DEFAULT_INCLUDE_PATTERNS,
) -> Path:
    """case ディレクトリを提出用アーカイブに変換する。

    Args:
        case_dir: `pipeline/<case>` ディレクトリ。
        out_dir: 生成物を置くディレクトリ。
        single_file: True の場合は `main.py` をそのまま返す。
        include_patterns: tar.gz に含めるファイルのパターン（ファイル名に対して
            fnmatch マッチ）。

    Returns:
        生成された tar.gz、または single_file モードでは `main.py` のパス。
    """

    case_dir = case_dir.resolve()
    if not case_dir.is_dir():
        raise PackagingError(f"case ディレクトリが見つかりません: {case_dir}")
    main_py = case_dir / "main.py"
    if not main_py.is_file():
        raise PackagingError(f"main.py が見つかりません: {main_py}")

    ensure_dvc_artifacts(case_dir)

    if single_file:
        return main_py

    files = _iter_case_files(case_dir, include_patterns)
    if main_py not in files:
        raise PackagingError(
            "main.py が include_patterns に一致しません。設定を確認してください。"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    archive_path = out_dir / f"{case_dir.name}_{_timestamp()}.tar.gz"

    with tarfile.open(archive_path, "w:gz", format=tarfile.GNU_FORMAT) as tar:
        for path in files:
            arcname = path.relative_to(case_dir).as_posix()
            info = tar.gettarinfo(str(path), arcname=arcname)
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mode = 0o644
            info.mtime = 0
            with path.open("rb") as f:
                tar.addfile(info, fileobj=f)

    return archive_path
