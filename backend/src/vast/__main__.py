"""`uv run python -m vast` のエントリポイント。"""

from __future__ import annotations

from vast.cli import app

if __name__ == "__main__":
    app()
