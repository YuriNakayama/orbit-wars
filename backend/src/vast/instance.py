"""onstart テンプレ render と Vast.ai create_instance ラッパ。

placeholder 置換時に shell injection を防ぐため、値は厳格な regex でバリデーション
してから単純な文字列置換を行う。
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

# 置換可能な値の許容文字種。shell injection を防ぐため厳しめに固定。
# 英数 + . _ - / : の組合せのみ ( `:` は URL の scheme separator、`/` は path separator)
# branch slug 内に `/` は出ないが repo URL のために : と / は許容する。
_VALID_VALUE = re.compile(r"^[A-Za-z0-9._\-/:]+$")
_TEMPLATE_PLACEHOLDERS = (
    "<COMMIT_SHA>",
    "<RUN_ID>",
    "<STAGE>",
    "<BRANCH>",
    "<REPO_URL>",
)

DEFAULT_IMAGE = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime"
DEFAULT_DISK_GB = 40


class TemplateError(ValueError):
    """テンプレート置換で不正値を検出したときに投げる。"""


def _validate(name: str, value: str) -> None:
    if not value:
        raise TemplateError(f"empty value for {name!r}")
    if not _VALID_VALUE.match(value):
        raise TemplateError(
            f"invalid characters in {name!r}={value!r}; "
            "shell injection prevention rejected the value"
        )


def render_onstart(
    template_path: Path,
    *,
    commit_sha: str,
    run_id: str,
    stage: str,
    branch: str,
    repo_url: str,
) -> str:
    """テンプレを読み 5 placeholder を置換した script 文字列を返す。

    値は事前に regex でバリデーション。違反したら TemplateError。
    """
    _validate("commit_sha", commit_sha)
    _validate("run_id", run_id)
    _validate("stage", stage)
    _validate("branch", branch)
    _validate("repo_url", repo_url)
    text = template_path.read_text(encoding="utf-8")
    substitutions = {
        "<COMMIT_SHA>": commit_sha,
        "<RUN_ID>": run_id,
        "<STAGE>": stage,
        "<BRANCH>": branch,
        "<REPO_URL>": repo_url,
    }
    for placeholder, value in substitutions.items():
        text = text.replace(placeholder, value)
    # 念のため未置換の placeholder が残っていないか確認
    leftover = [p for p in _TEMPLATE_PLACEHOLDERS if p in text]
    if leftover:
        raise TemplateError(f"unsubstituted placeholders remain: {leftover}")
    return text


def build_env_dict(env: Mapping[str, str]) -> dict[str, str]:
    """`vastai SDK create_instance(env=...)` に渡す dict を組み立てる。

    SDK の低レベル API は `env` を dict として受け取り、JSON に直接埋める。
    変数名のバリデーションだけ行う (shell 注入経路は onstart 側で別途検証)。
    """
    result: dict[str, str] = {}
    for key, value in env.items():
        if not re.match(r"^[A-Z_][A-Z0-9_]*$", key):
            raise TemplateError(f"invalid env var name: {key!r}")
        result[key] = value
    return result


def create_instance(
    sdk: Any,
    offer_id: int,
    *,
    onstart_cmd: str,
    env: Mapping[str, str],
    image: str = DEFAULT_IMAGE,
    disk_gb: int = DEFAULT_DISK_GB,
    label: str,
    runtype: str = "ssh_direc ssh_proxy",
) -> int:
    """`VastAI().create_instance(...)` を呼び、生成された instance id を返す。

    SSH + direct connection 用の `runtype="ssh_direc ssh_proxy"` をデフォルトとする
    (vastai CLI の `--ssh --direct` と等価)。SDK の応答 dict から `new_contract`
    または `id` を抽出する。
    """
    response = sdk.create_instance(
        offer_id,
        image=image,
        disk=float(disk_gb),
        env=dict(env),
        label=label,
        onstart_cmd=onstart_cmd,
        runtype=runtype,
    )
    if not isinstance(response, Mapping):
        raise RuntimeError(
            f"unexpected create_instance response: {type(response).__name__}"
        )
    instance_id = response.get("new_contract") or response.get("id")
    if instance_id is None:
        raise RuntimeError(
            f"create_instance response missing instance id: keys={list(response)}"
        )
    return int(instance_id)
