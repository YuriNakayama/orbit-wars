"""Repository root discovery for development scripts.

Per-case `evaluation/*.py` and `training/*.py` need to resolve paths against
the repo root (e.g. to read `params.yaml` or write to `data/mart/...`). This
module owns the discovery logic so each script does not reinvent it.

The discovery walks parents of `__file__` looking for the `(backend/, .git)`
sentinel pair. Caches the result via a module-level constant computed at
import time so callers don't pay the I/O cost per call.
"""

from __future__ import annotations

from pathlib import Path


def find_repo_root(start: Path) -> Path:
    """Walk parents of `start` until a directory with `backend/` and `.git/` is found.

    Raises RuntimeError if no such directory is found.
    """
    here = start.resolve()
    for candidate in (here, *here.parents):
        if (candidate / "backend").is_dir() and (candidate / ".git").exists():
            return candidate
    raise RuntimeError(f"repo root not found from {start}")


def absolute_under_repo(rel: str | Path, *, start: Path) -> Path:
    """Resolve `rel` against the repo root located from `start`."""
    p = Path(rel)
    if p.is_absolute():
        return p
    return (find_repo_root(start) / p).resolve()
