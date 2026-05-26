"""gpu.runpod.runpod.instance のユニットテスト (vast.test_instance を mirror)。"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gpu.runpod.runpod.instance import (
    DEFAULT_IMAGE,
    TemplateError,
    build_env_dict,
    create_pod,
    render_onstart,
)

TEMPLATE_PATH = (
    Path(__file__).resolve().parents[6] / "src/gpu/runpod/execution/onstart.sh.tmpl"
)


def _valid_kwargs() -> dict[str, str]:
    return {
        "commit_sha": "abc1234deadbeef",
        "run_id": "20260502-143022__feature-runpod-basis__abc1234__seed0",
        "stage": "train_imitation_case1",
        "branch": "feature-runpod-basis",
        "repo_url": "https://github.com/YuriNakayama/orbit-wars.git",
    }


def test_render_onstart_replaces_all_placeholders() -> None:
    rendered = render_onstart(TEMPLATE_PATH, **_valid_kwargs())
    assert "abc1234deadbeef" in rendered
    assert "20260502-143022__feature-runpod-basis__abc1234__seed0" in rendered
    assert "train_imitation_case1" in rendered
    assert "feature-runpod-basis" in rendered
    assert "https://github.com/YuriNakayama/orbit-wars.git" in rendered
    for placeholder in (
        "<COMMIT_SHA>",
        "<RUN_ID>",
        "<STAGE>",
        "<BRANCH>",
        "<REPO_URL>",
        "<CASE>",
        "<CASE_FAMILY>",
        "<CASE_SUBDIR>",
        "<TRAIN_MODULE>",
        "__ONSTART_LIB_00_PRELUDE__",
        "__ONSTART_LIB_10_TOOLS__",
        "__ONSTART_LIB_20_CLEANUP__",
        "__ONSTART_LIB_30_PERSIST_AND_CLONE__",
        "__ONSTART_LIB_40_UV_AND_DVC_PULL__",
        "__ONSTART_LIB_50_PREPROCESS__",
        "__ONSTART_LIB_60_TRAIN__",
        "__ONSTART_LIB_70_ARTIFACTS_AND_DVC__",
    ):
        assert placeholder not in rendered


def test_render_onstart_with_case3_args() -> None:
    rendered = render_onstart(
        TEMPLATE_PATH,
        case="case3",
        train_module="pipeline.imitation.case3.training.train",
        config_arg="--config pipeline/imitation/case3/configs/il_phase2.yaml",
        preprocess_cmd=(
            "pipeline.imitation.case3.training.preprocess "
            "--config pipeline/imitation/case3/configs/il_phase2.yaml"
        ),
        **_valid_kwargs(),
    )
    assert "case3" in rendered
    assert "pipeline.imitation.case3.training.train" in rendered


def test_render_onstart_rejects_shell_injection() -> None:
    bad = _valid_kwargs()
    bad["commit_sha"] = "abc1234; rm -rf /"
    with pytest.raises(TemplateError):
        render_onstart(TEMPLATE_PATH, **bad)


def test_render_onstart_rejects_empty_value() -> None:
    bad = _valid_kwargs()
    bad["run_id"] = ""
    with pytest.raises(TemplateError):
        render_onstart(TEMPLATE_PATH, **bad)


def test_render_onstart_rejects_bad_config_arg() -> None:
    with pytest.raises(TemplateError):
        render_onstart(
            TEMPLATE_PATH,
            **_valid_kwargs(),
            config_arg="--config /etc/passwd; ls",
        )


def test_render_onstart_rejects_bad_preprocess_cmd() -> None:
    with pytest.raises(TemplateError):
        render_onstart(
            TEMPLATE_PATH,
            **_valid_kwargs(),
            preprocess_cmd="x; rm -rf /",
        )


def test_build_env_dict_validates_keys() -> None:
    env = build_env_dict({"FOO": "bar", "MY_KEY_123": "v"})
    assert env == {"FOO": "bar", "MY_KEY_123": "v"}


def test_build_env_dict_rejects_lowercase() -> None:
    with pytest.raises(TemplateError):
        build_env_dict({"lowercase": "x"})


def test_build_env_dict_rejects_starts_with_digit() -> None:
    with pytest.raises(TemplateError):
        build_env_dict({"1FOO": "x"})


def test_create_pod_passes_expected_kwargs() -> None:
    sdk = MagicMock()
    sdk.create_pod.return_value = {"id": "pod-abc123"}
    pod_id = create_pod(
        sdk,
        name="run-x",
        gpu_type_id="NVIDIA GeForce RTX 3090",
        cloud_type="SECURE",
        onstart_script="#!/bin/bash\necho hi\n",
        env={"FOO": "bar"},
        network_volume_id="vol-1",
    )
    assert pod_id == "pod-abc123"
    sdk.create_pod.assert_called_once()
    kwargs = sdk.create_pod.call_args.kwargs
    assert kwargs["name"] == "run-x"
    assert kwargs["image_name"] == DEFAULT_IMAGE
    assert kwargs["gpu_type_id"] == "NVIDIA GeForce RTX 3090"
    assert kwargs["cloud_type"] == "SECURE"
    assert kwargs["docker_args"].startswith("bash -c '")
    # /start.sh background 起動が含まれる (sshd 起動維持のため)
    assert "/start.sh" in kwargs["docker_args"]
    assert "base64 -d | gunzip | bash" in kwargs["docker_args"]
    # gzip+base64 で onstart_script が正しく埋め込まれていること
    import base64
    import gzip
    import re

    m = re.search(r"echo (\S+) \| base64 -d \| gunzip \| bash", kwargs["docker_args"])
    assert m is not None
    decoded = gzip.decompress(base64.b64decode(m.group(1))).decode("utf-8")
    assert decoded == "#!/bin/bash\necho hi\n"
    assert kwargs["env"] == {"FOO": "bar"}
    assert kwargs["network_volume_id"] == "vol-1"
    assert kwargs["volume_mount_path"] == "/persist"


def test_create_pod_response_alt_field() -> None:
    sdk = MagicMock()
    sdk.create_pod.return_value = {"podId": "alt-id"}
    pod_id = create_pod(
        sdk,
        name="x",
        gpu_type_id="X",
        cloud_type="COMMUNITY",
        onstart_script="x",
        env={},
    )
    assert pod_id == "alt-id"


def test_create_pod_invalid_cloud_type() -> None:
    sdk = MagicMock()
    with pytest.raises(ValueError):
        create_pod(
            sdk,
            name="x",
            gpu_type_id="X",
            cloud_type="WRONG",
            onstart_script="x",
            env={},
        )


def test_create_pod_response_missing_id() -> None:
    sdk = MagicMock()
    sdk.create_pod.return_value = {}
    with pytest.raises(RuntimeError):
        create_pod(
            sdk,
            name="x",
            gpu_type_id="X",
            cloud_type="SECURE",
            onstart_script="x",
            env={},
        )


def test_onstart_template_bash_syntax() -> None:
    rendered = render_onstart(
        TEMPLATE_PATH,
        case="case1",
        train_module="pipeline.imitation.case1.training.train",
        config_arg="",
        preprocess_cmd="",
        **_valid_kwargs(),
    )
    result = subprocess.run(
        ["bash", "-n"],
        input=rendered,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"bash -n failed: {result.stderr}"


def test_render_onstart_default_mode_is_oneshot() -> None:
    """既定 mode は oneshot で trap cleanup_destroy EXIT が残ること。"""
    rendered = render_onstart(TEMPLATE_PATH, **_valid_kwargs())
    assert 'RUNPOD_MODE="oneshot"' in rendered
    # oneshot 経路では cleanup trap セットが活きる
    assert "trap cleanup_destroy EXIT" in rendered


def test_render_onstart_interactive_mode_skips_train_and_trap() -> None:
    """mode=interactive で train を実行せず sleep infinity に入る分岐が見える。"""
    rendered = render_onstart(
        TEMPLATE_PATH,
        mode="interactive",
        **_valid_kwargs(),
    )
    assert 'RUNPOD_MODE="interactive"' in rendered
    # interactive 分岐の目印
    assert "interactive mode, awaiting SSH" in rendered
    assert "50_interactive_ready" in rendered
    # bash -n が通ること (構文破壊していない)
    result = subprocess.run(
        ["bash", "-n"],
        input=rendered,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"bash -n failed: {result.stderr}"


def test_render_onstart_rejects_unknown_mode() -> None:
    with pytest.raises(TemplateError):
        render_onstart(
            TEMPLATE_PATH,
            mode="debug",  # 未対応の値
            **_valid_kwargs(),
        )
