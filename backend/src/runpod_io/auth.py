"""AWS credentials と RUNPOD_API_KEY のローカル読み込みヘルパ。

AWS は vast.auth の `load_aws_creds()` をそのまま再利用する (重複実装を避ける)。
RunPod 用 `load_runpod_api_key()` のみ追加する。
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values

from vast.auth import (
    DEFAULT_AWS_PROFILE,
    DEFAULT_AWS_REGION,
    AwsCreds,
    CredentialsError,
    load_aws_creds,
)

__all__ = [
    "DEFAULT_AWS_PROFILE",
    "DEFAULT_AWS_REGION",
    "AwsCreds",
    "CredentialsError",
    "load_aws_creds",
    "load_runpod_api_key",
]


def load_runpod_api_key(
    *,
    env_path: Path | None = None,
) -> str:
    """`backend/.env` または環境変数から RUNPOD_API_KEY を読む。

    env file → process env の順に検査し、最初に値があれば返す。
    """
    if env_path is None:
        env_path = _default_env_path()
    if env_path.is_file():
        values = dotenv_values(env_path)
        api_key = values.get("RUNPOD_API_KEY")
        if api_key:
            return api_key.strip()
    fallback = os.environ.get("RUNPOD_API_KEY", "").strip()
    if fallback:
        return fallback
    raise CredentialsError(
        "RUNPOD_API_KEY not found. Add `RUNPOD_API_KEY=<your-key>` to "
        f"{env_path} or export it as an environment variable. "
        "Get a key from https://runpod.io/console/user/settings."
    )


def _default_env_path() -> Path:
    """backend/ 直下の .env を返す。"""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent / ".env"
    return Path(".env").resolve()
