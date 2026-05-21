"""`python -m kaggle_kernel` エントリポイント。"""

from __future__ import annotations

from kaggle_kernel.cli.app import app


def main() -> None:
    app()


if __name__ == "__main__":
    main()
