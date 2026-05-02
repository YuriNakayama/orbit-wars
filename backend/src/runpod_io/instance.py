"""onstart テンプレ render と RunPod create_pod ラッパ。

placeholder 置換時に shell injection を防ぐため、値は厳格な regex でバリデート
してから単純な文字列置換を行う (vast.instance と同設計)。

RunPod の `docker_args` は GraphQL 文字列にそのまま埋め込まれる
(`f'dockerArgs: "{docker_args}"'`) ため、改行や `"` を含むスクリプトを直接渡すと
GraphQL Syntax Error になる。本基盤では onstart スクリプトを base64 エンコードして
1 行の bootstrap (`bash -c "echo <b64> | base64 -d | bash"`) として渡すことで
回避する。
"""

from __future__ import annotations

import base64
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

# 置換可能な値の許容文字種。shell injection を防ぐため厳しめに固定。
_VALID_VALUE = re.compile(r"^[A-Za-z0-9._\-/:]+$")
# CONFIG_ARG は空文字 (case1) または `--config <path>` の 2 トークンのみ許可。
_VALID_CONFIG_ARG = re.compile(r"^(|--config [A-Za-z0-9._\-/]+)$")
# PREPROCESS_CMD は空文字または `module.path[ --config <path>]` 形式のみ許可。
_VALID_PREPROCESS_CMD = re.compile(
    r"^(|[A-Za-z0-9._\-/]+( --config [A-Za-z0-9._\-/]+)?)$"
)
_TEMPLATE_PLACEHOLDERS = (
    "<COMMIT_SHA>",
    "<RUN_ID>",
    "<STAGE>",
    "<BRANCH>",
    "<REPO_URL>",
    "<CASE>",
    "<TRAIN_MODULE>",
    "<CONFIG_ARG>",
    "<PREPROCESS_CMD>",
)

# RunPod 公式 PyTorch image の最新フォーマット (~9.6GB)。Docker Hub に確実に
# 存在し、SSH/Jupyter/runpodctl が pre-install されている。古いタグ
# (2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04) や pytorch/pytorch image は
# pull 中に止まる挙動を観測したため、こちらを採用。
DEFAULT_IMAGE = os.environ.get(
    "ORBIT_WARS_RUNPOD_IMAGE",
    "runpod/pytorch:1.0.3-cu1300-torch291-ubuntu2404",
)
DEFAULT_DISK_GB = 40
DEFAULT_PORTS = "22/tcp,8888/http"


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


def _validate_config_arg(value: str) -> None:
    if not _VALID_CONFIG_ARG.match(value):
        raise TemplateError(
            f"invalid config_arg={value!r}; expected '' or '--config <path>' "
            "with safe characters only"
        )


def _validate_preprocess_cmd(value: str) -> None:
    if not _VALID_PREPROCESS_CMD.match(value):
        raise TemplateError(
            f"invalid preprocess_cmd={value!r}; expected '' or "
            "'module.path [--config <path>]' with safe characters only"
        )


def render_onstart(
    template_path: Path,
    *,
    commit_sha: str,
    run_id: str,
    stage: str,
    branch: str,
    repo_url: str,
    case: str = "case1",
    train_module: str = "pipeline.imitation.case1.training.train",
    config_arg: str = "",
    preprocess_cmd: str = "",
) -> str:
    """テンプレを読み placeholder を置換した script 文字列を返す。

    値は事前に regex でバリデーション。違反したら TemplateError。
    """
    _validate("commit_sha", commit_sha)
    _validate("run_id", run_id)
    _validate("stage", stage)
    _validate("branch", branch)
    _validate("repo_url", repo_url)
    _validate("case", case)
    _validate("train_module", train_module)
    _validate_config_arg(config_arg)
    _validate_preprocess_cmd(preprocess_cmd)
    text = template_path.read_text(encoding="utf-8")
    substitutions = {
        "<COMMIT_SHA>": commit_sha,
        "<RUN_ID>": run_id,
        "<STAGE>": stage,
        "<BRANCH>": branch,
        "<REPO_URL>": repo_url,
        "<CASE>": case,
        "<TRAIN_MODULE>": train_module,
        "<CONFIG_ARG>": config_arg,
        "<PREPROCESS_CMD>": preprocess_cmd,
    }
    for placeholder, value in substitutions.items():
        text = text.replace(placeholder, value)
    leftover = [p for p in _TEMPLATE_PLACEHOLDERS if p in text]
    if leftover:
        raise TemplateError(f"unsubstituted placeholders remain: {leftover}")
    return text


def build_env_dict(env: Mapping[str, str]) -> dict[str, str]:
    """`runpod.create_pod(env=...)` に渡す dict を組み立てる。

    変数名のバリデーションだけ行う。値の shell 注入経路は onstart 側で別途検証。
    """
    result: dict[str, str] = {}
    for key, value in env.items():
        if not re.match(r"^[A-Z_][A-Z0-9_]*$", key):
            raise TemplateError(f"invalid env var name: {key!r}")
        result[key] = value
    return result


def create_pod(
    sdk: Any,
    *,
    name: str,
    gpu_type_id: str,
    cloud_type: str,
    onstart_script: str,
    env: Mapping[str, str],
    image: str = DEFAULT_IMAGE,
    container_disk_gb: int = DEFAULT_DISK_GB,
    network_volume_id: str | None = None,
    volume_mount_path: str = "/persist",
    ports: str = DEFAULT_PORTS,
    data_center_id: str | None = None,
) -> str:
    """`runpod.create_pod(...)` を呼び、生成された pod id を返す。

    `docker_args` には `bash -c "<onstart_script>"` を渡す (RunPod は Vast の
    `onstart_cmd` 直接相当がなく、コンテナの ENTRYPOINT/CMD を上書きする経路)。
    onstart 側で末尾に `runpodctl stop pod $RUNPOD_POD_ID` を仕込んでおけば
    docker_args の exit による pod 再起動ループには入らない。
    """
    if cloud_type not in ("SECURE", "COMMUNITY", "ALL"):
        raise ValueError(f"cloud_type must be SECURE/COMMUNITY/ALL, got {cloud_type!r}")
    # RunPod の dockerArgs は GraphQL 文字列にそのまま挿入されるため、改行や引用符
    # を含む bash script を直接渡すと parse error になる。base64 化してから 1 行の
    # bootstrap で decode + bash 実行する。
    encoded = base64.b64encode(onstart_script.encode("utf-8")).decode("ascii")
    docker_args = f"bash -c 'echo {encoded} | base64 -d | bash'"
    kwargs: dict[str, Any] = {
        "name": name,
        "image_name": image,
        "gpu_type_id": gpu_type_id,
        "cloud_type": cloud_type,
        "gpu_count": 1,
        "container_disk_in_gb": container_disk_gb,
        "volume_in_gb": 0,
        "volume_mount_path": volume_mount_path,
        "docker_args": docker_args,
        "env": dict(env),
        "support_public_ip": True,
        "start_ssh": True,
        "ports": ports,
    }
    if network_volume_id is not None:
        kwargs["network_volume_id"] = network_volume_id
    if data_center_id is not None:
        kwargs["data_center_id"] = data_center_id
    response = sdk.create_pod(**kwargs)
    if not isinstance(response, Mapping):
        raise RuntimeError(f"unexpected create_pod response: {type(response).__name__}")
    pod_id = response.get("id") or response.get("podId")
    if pod_id is None:
        raise RuntimeError(f"create_pod response missing pod id: keys={list(response)}")
    return str(pod_id)
