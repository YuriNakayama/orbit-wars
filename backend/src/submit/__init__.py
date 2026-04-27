"""Kaggle Orbit Wars 提出自動化パッケージ。"""

from submit.auth import AuthError, ensure_credentials
from submit.history import record
from submit.kaggle_api import (
    COMPETITION,
    confirm_submission,
    list_submissions,
    poll,
)
from submit.kaggle_api import submit as kaggle_submit
from submit.packager import build_archive, ensure_dvc_artifacts
from submit.validator import ValidationError, dry_run

__all__ = [
    "COMPETITION",
    "AuthError",
    "ValidationError",
    "build_archive",
    "confirm_submission",
    "dry_run",
    "ensure_credentials",
    "ensure_dvc_artifacts",
    "kaggle_submit",
    "list_submissions",
    "poll",
    "record",
]
