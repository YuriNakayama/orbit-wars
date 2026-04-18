"""Kaggle 認証情報の有無を確認する。

認証情報そのものは絶対にログ・例外メッセージに含めない。
"""

from __future__ import annotations

import os
from pathlib import Path


class AuthError(RuntimeError):
    """認証情報が未設定のときに投げる例外。"""


def _has_env_credentials() -> bool:
    username = os.environ.get("KAGGLE_USERNAME", "").strip()
    key = os.environ.get("KAGGLE_KEY", "").strip()
    return bool(username) and bool(key)


def _has_config_file() -> bool:
    path = Path.home() / ".kaggle" / "kaggle.json"
    return path.exists()


def ensure_credentials() -> str:
    """認証情報が設定されていることを確認する。

    Returns:
        "env" または "config" — どちらの方式で認証されているか。
    Raises:
        AuthError: どちらも設定されていない場合。
    """

    if _has_env_credentials():
        return "env"
    if _has_config_file():
        return "config"
    raise AuthError(
        "Kaggle 認証が設定されていません。\n"
        "  環境変数 KAGGLE_USERNAME と KAGGLE_KEY を設定するか、\n"
        "  ~/.kaggle/kaggle.json を配置してください。"
    )
