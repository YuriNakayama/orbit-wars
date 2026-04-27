"""instance.py のテンプレ render + create_instance ラッパのユニットテスト。"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from vast.instance import (
    DEFAULT_IMAGE,
    TemplateError,
    build_env_dict,
    create_instance,
    render_onstart,
)

# テンプレ実体は src/vast/onstart.sh.tmpl
TEMPLATE_PATH = Path(__file__).resolve().parents[2] / "src/vast/onstart.sh.tmpl"


def _valid_kwargs() -> dict[str, str]:
    return {
        "commit_sha": "abc1234deadbeef",
        "run_id": "20260425-143022__feature-vast-ai-basis__abc1234__seed0",
        "stage": "train_imitation_case1",
        "branch": "feature-vast-ai-basis",
        "repo_url": "https://github.com/YuriNakayama/orbit-wars.git",
    }


def test_render_onstart_replaces_all_placeholders() -> None:
    rendered = render_onstart(TEMPLATE_PATH, **_valid_kwargs())
    assert "abc1234deadbeef" in rendered
    assert "20260425-143022__feature-vast-ai-basis__abc1234__seed0" in rendered
    assert "train_imitation_case1" in rendered
    assert "feature-vast-ai-basis" in rendered
    assert "https://github.com/YuriNakayama/orbit-wars.git" in rendered
    # No placeholder remains
    for placeholder in (
        "<COMMIT_SHA>",
        "<RUN_ID>",
        "<STAGE>",
        "<BRANCH>",
        "<REPO_URL>",
    ):
        assert placeholder not in rendered


def test_render_onstart_rejects_shell_injection() -> None:
    bad = _valid_kwargs()
    bad["commit_sha"] = "abc1234; rm -rf /"
    with pytest.raises(TemplateError, match="invalid characters"):
        render_onstart(TEMPLATE_PATH, **bad)


def test_render_onstart_rejects_empty_value() -> None:
    bad = _valid_kwargs()
    bad["run_id"] = ""
    with pytest.raises(TemplateError, match="empty value"):
        render_onstart(TEMPLATE_PATH, **bad)


def test_render_onstart_rejects_quote() -> None:
    bad = _valid_kwargs()
    bad["branch"] = "feat'or'bad"
    with pytest.raises(TemplateError):
        render_onstart(TEMPLATE_PATH, **bad)


def test_rendered_template_has_valid_bash_syntax(tmp_path: Path) -> None:
    rendered = render_onstart(TEMPLATE_PATH, **_valid_kwargs())
    out = tmp_path / "rendered.sh"
    out.write_text(rendered, encoding="utf-8")
    result = subprocess.run(["bash", "-n", str(out)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_build_env_dict_passes_through() -> None:
    env = {
        "AWS_ACCESS_KEY_ID": "AKIA",
        "AWS_SECRET_ACCESS_KEY": "secret with space",
        "VAST_API_KEY": "abc123",
    }
    result = build_env_dict(env)
    assert result == env


def test_build_env_dict_rejects_bad_var_name() -> None:
    with pytest.raises(TemplateError, match="invalid env var name"):
        build_env_dict({"bad-name": "v"})


def test_create_instance_returns_new_contract_id() -> None:
    sdk = MagicMock()
    sdk.create_instance.return_value = {"success": True, "new_contract": 12345678}
    instance_id = create_instance(
        sdk,
        offer_id=99,
        onstart_cmd="echo hi",
        env={"A": "B"},
        label="test-run",
    )
    assert instance_id == 12345678
    kwargs = sdk.create_instance.call_args.kwargs
    assert kwargs["image"] == DEFAULT_IMAGE
    assert kwargs["disk"] == 40.0
    assert kwargs["label"] == "test-run"
    assert kwargs["onstart_cmd"] == "echo hi"
    assert kwargs["runtype"] == "ssh_direc ssh_proxy"
    assert kwargs["env"] == {"A": "B"}


def test_create_instance_missing_id_raises() -> None:
    sdk = MagicMock()
    sdk.create_instance.return_value = {"success": False, "msg": "out of stock"}
    with pytest.raises(RuntimeError, match="missing instance id"):
        create_instance(
            sdk,
            offer_id=99,
            onstart_cmd="echo hi",
            env={"A": "B"},
            label="test-run",
        )


def test_create_instance_unexpected_response_raises() -> None:
    sdk = MagicMock()
    sdk.create_instance.return_value = "not a dict"
    with pytest.raises(RuntimeError, match="unexpected"):
        create_instance(
            sdk,
            offer_id=99,
            onstart_cmd="echo hi",
            env={"A": "B"},
            label="test-run",
        )
