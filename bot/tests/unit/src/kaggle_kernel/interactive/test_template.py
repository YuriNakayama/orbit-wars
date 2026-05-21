"""kaggle_kernel.interactive.template のユニットテスト。"""

from __future__ import annotations

import nbformat
import pytest

from kaggle_kernel.interactive.template import (
    InteractiveContext,
    render_interactive_notebook,
)


def _ctx(**overrides: object) -> InteractiveContext:
    defaults: dict[str, object] = {
        "run_id": "20260521-000000__main__abc1234__seed0",
        "commit_sha": "abc1234deadbeef",
        "branch": "main",
        "case": "case1",
        "dataset_slug": "yuri/orbit-wars-bot",
        "accelerator": "gpu-t4x2",
        "kaggle_kernel_meta_initial": {
            "kernel_slug": "yuri/dev-case1-run0",
            "dataset_slug": "yuri/orbit-wars-bot",
            "accelerator": "gpu-t4x2",
        },
        "s3_bucket": "orbit-wars-dvc-286854171013",
        "max_idle_minutes": 480.0,
        "seed": 0,
    }
    defaults.update(overrides)
    return InteractiveContext(**defaults)  # type: ignore[arg-type]


def test_render_returns_valid_notebook() -> None:
    nb = render_interactive_notebook(_ctx())
    parsed = nbformat.from_dict(nb)
    nbformat.validate(parsed)
    assert nb["nbformat"] == 4


def test_render_has_four_cells() -> None:
    """interactive は 4 cell: env / mount / install / loop。train cell はない。"""
    nb = render_interactive_notebook(_ctx())
    assert len(nb["cells"]) == 4
    for cell in nb["cells"]:
        assert cell["cell_type"] == "code"


def test_cell_a_sets_interactive_flag() -> None:
    nb = render_interactive_notebook(_ctx())
    cell_a = "".join(nb["cells"][0]["source"])
    assert "ORBIT_WARS_KAGGLE_INTERACTIVE" in cell_a
    assert "ORBIT_WARS_RUN_ID" in cell_a


def test_cell_a_does_not_leak_secret_values() -> None:
    """Secret 値そのものは notebook に埋まらない (env name は出てきて OK)。"""
    nb = render_interactive_notebook(_ctx())
    cell_a = "".join(nb["cells"][0]["source"])
    # Kaggle 個人 API key は絶対に notebook に含めない
    assert "KAGGLE_KEY" not in cell_a


def test_cell_a_loads_aws_secrets_from_kaggle() -> None:
    """UserSecretsClient (priority 1) で AWS creds を読み込む経路がある。"""
    nb = render_interactive_notebook(_ctx())
    cell_a = "".join(nb["cells"][0]["source"])
    assert "UserSecretsClient" in cell_a
    assert "AWS_ACCESS_KEY_ID" in cell_a
    assert "AWS_SECRET_ACCESS_KEY" in cell_a


def test_cell_a_dotenv_fallback() -> None:
    """priority 2: dataset 同梱の bot/.env を os.walk で探索する fallback。"""
    nb = render_interactive_notebook(_ctx())
    cell_a = "".join(nb["cells"][0]["source"])
    # os.walk ベースで dotfile を見落とさない
    assert "os.walk('/kaggle/input')" in cell_a
    assert "_find_dotenv" in cell_a
    # actionable error メッセージ
    assert "dev/kaggle-kernel dev" in cell_a


def test_cell_a_defaults_aws_region() -> None:
    nb = render_interactive_notebook(_ctx())
    cell_a = "".join(nb["cells"][0]["source"])
    assert "ap-northeast-1" in cell_a


def test_cell_c_installs_boto3() -> None:
    nb = render_interactive_notebook(_ctx())
    cell_c = "".join(nb["cells"][2]["source"])
    assert "boto3" in cell_c


def test_cell_d_has_s3_loop() -> None:
    nb = render_interactive_notebook(_ctx())
    cell_d = "".join(nb["cells"][3]["source"])
    assert "kaggle_interactive" in cell_d
    assert "20260521-000000__main__abc1234__seed0" in cell_d
    assert "orbit-wars-dvc-286854171013" in cell_d
    # heartbeat + shutdown handlers
    assert "_heartbeat" in cell_d
    assert "__shutdown__" in cell_d
    # session start / max idle guard
    assert "MAX_IDLE_SEC" in cell_d


def test_cell_d_uses_inbox_outbox() -> None:
    nb = render_interactive_notebook(_ctx())
    cell_d = "".join(nb["cells"][3]["source"])
    assert "inbox/" in cell_d
    assert "outbox/" in cell_d


def test_invalid_run_id_rejected() -> None:
    with pytest.raises(ValueError, match="invalid run_id"):
        render_interactive_notebook(_ctx(run_id="bogus"))


def test_invalid_s3_bucket_rejected() -> None:
    with pytest.raises(ValueError, match="invalid s3_bucket"):
        render_interactive_notebook(_ctx(s3_bucket="bad bucket name"))


def test_non_positive_interval_rejected() -> None:
    with pytest.raises(ValueError, match="intervals must be positive"):
        render_interactive_notebook(_ctx(heartbeat_interval=0))


def test_max_idle_propagates_to_cell() -> None:
    nb = render_interactive_notebook(_ctx(max_idle_minutes=60.0))
    cell_d = "".join(nb["cells"][3]["source"])
    assert "MAX_IDLE_SEC = 3600.0" in cell_d
