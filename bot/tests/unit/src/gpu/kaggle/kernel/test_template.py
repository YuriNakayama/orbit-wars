"""gpu.kaggle.kernel.template のユニットテスト。"""

from __future__ import annotations

import nbformat
import pytest

from gpu.kaggle.kernel.template import RenderContext, render_notebook


def _ctx(**overrides: object) -> RenderContext:
    defaults: dict[str, object] = {
        "run_id": "20260520-103000__feature-x__abc1234__seed0",
        "commit_sha": "abc1234deadbeef",
        "branch": "feature-x",
        "case": "case1",
        "train_module": "pipeline.imitation.case1.training.train",
        "config_arg": "",
        "dataset_slug": "yuri/orbit-wars-bot",
        "accelerator": "gpu-t4x2",
        "kaggle_kernel_meta_initial": {
            "kernel_slug": "yuri/orbit-wars-case1-run0",
            "dataset_slug": "yuri/orbit-wars-bot",
            "accelerator": "gpu-t4x2",
        },
        "seed": 0,
    }
    defaults.update(overrides)
    return RenderContext(**defaults)  # type: ignore[arg-type]


def test_render_returns_valid_notebook() -> None:
    nb = render_notebook(_ctx())
    # nbformat の validate を pass
    nbformat.validate(nb)
    assert nb["nbformat"] == 4


def test_render_has_five_cells() -> None:
    nb = render_notebook(_ctx())
    assert len(nb["cells"]) == 5
    for cell in nb["cells"]:
        assert cell["cell_type"] == "code"


def test_cell_env_has_run_id_and_meta() -> None:
    nb = render_notebook(_ctx())
    cell_a = "".join(nb["cells"][0]["source"])
    assert "ORBIT_WARS_RUN_ID" in cell_a
    assert "20260520-103000__feature-x__abc1234__seed0" in cell_a
    assert "ORBIT_WARS_KAGGLE_KERNEL_META" in cell_a
    assert "ORBIT_WARS_KAGGLE_KERNEL_SLUG" in cell_a


def test_cell_env_does_not_leak_secrets() -> None:
    """KAGGLE_KEY / AWS_* / GIT_PAT が cell A に混入していないことを確認。"""
    nb = render_notebook(_ctx())
    cell_a = "".join(nb["cells"][0]["source"])
    assert "KAGGLE_KEY" not in cell_a
    assert "AWS_SECRET" not in cell_a
    assert "GIT_PAT" not in cell_a
    assert "RUNPOD_API_KEY" not in cell_a


def test_cell_install_uses_dataset_mount() -> None:
    nb = render_notebook(_ctx())
    cell_b = "".join(nb["cells"][1]["source"])
    cell_c = "".join(nb["cells"][2]["source"])
    # Cell B mirrors the dataset mount into REPO_ROOT
    assert "/kaggle/input/orbit-wars-bot" in cell_b
    assert "/tmp/orbit-wars-repo" in cell_b
    # Cell C makes bot importable via sys.path injection
    assert "sys.path" in cell_c
    assert "bot/src" in cell_c


def test_cell_train_invokes_module() -> None:
    nb = render_notebook(_ctx())
    cell_d = "".join(nb["cells"][3]["source"])
    assert "pipeline.imitation.case1.training.train" in cell_d
    assert "subprocess.run" in cell_d
    assert "train.log" in cell_d


def test_cell_train_with_config_arg() -> None:
    nb = render_notebook(_ctx(config_arg="--config configs/foo.yaml"))
    cell_d = "".join(nb["cells"][3]["source"])
    assert "--config" in cell_d
    assert "configs/foo.yaml" in cell_d


def test_cell_collect_writes_to_run_dir() -> None:
    nb = render_notebook(_ctx())
    cell_e = "".join(nb["cells"][4]["source"])
    assert "/kaggle/working/runs/20260520-103000__feature-x__abc1234__seed0" in cell_e


def test_invalid_run_id_rejected() -> None:
    with pytest.raises(ValueError, match="invalid run_id"):
        render_notebook(_ctx(run_id="bogus"))


def test_shell_injection_in_commit_sha_rejected() -> None:
    with pytest.raises(ValueError, match="invalid commit_sha"):
        render_notebook(_ctx(commit_sha="abc; rm -rf /"))


def test_invalid_dataset_slug_rejected() -> None:
    with pytest.raises(ValueError):
        render_notebook(_ctx(dataset_slug="no-slash"))


def test_invalid_config_arg_rejected() -> None:
    with pytest.raises(ValueError, match="invalid config_arg"):
        render_notebook(_ctx(config_arg="--config 'foo; rm -rf /'"))


def test_meta_initial_embedded_in_cell_a() -> None:
    """ORBIT_WARS_KAGGLE_KERNEL_META に渡される dict の中身が cell A に埋まる。"""
    nb = render_notebook(
        _ctx(
            kaggle_kernel_meta_initial={
                "kernel_slug": "yuri/x-run0",
                "dataset_slug": "yuri/orbit-wars-bot",
                "accelerator": "gpu-t4x2",
                "internet_enabled": True,
            }
        )
    )
    cell_a = "".join(nb["cells"][0]["source"])
    assert "kernel_slug" in cell_a
    assert "yuri/x-run0" in cell_a
    assert "internet_enabled" in cell_a
    # boolean は Python repr で True、JSON-in-string で true のどちらかで表現される
    assert "True" in cell_a or "true" in cell_a
