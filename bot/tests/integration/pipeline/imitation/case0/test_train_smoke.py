"""End-to-end smoke test for ``pipeline.imitation.case0.training.train``.

Imports the training entrypoint, invokes it on the bundled smoke YAML config
under a tmp run_dir, and asserts that:

  * Training reaches ``max_steps``
  * ``best.pt``, ``metrics.json``, ``run.json`` are created
  * Total runtime is well under the 90s ceiling we promise for the local
    smoke gate (we assert << 30s here so flaky CI doesn't drift)

Runs on CPU only — RunPod GPU coverage is provided by the live E2E run.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import yaml

from pipeline.imitation.case0.training.train import train

SMOKE_CONFIG = Path("pipeline/imitation/case0/configs/smoke.yaml")


def _load_cfg() -> dict[str, object]:
    with SMOKE_CONFIG.open() as fp:
        loaded = yaml.safe_load(fp)
    assert isinstance(loaded, dict)
    return loaded


def test_smoke_train_cpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _load_cfg()
    cfg["training"]["device"] = "cpu"  # type: ignore[index]
    monkeypatch.setenv("ORBIT_WARS_RUN_DIR", str(tmp_path))
    # Make absolutely sure we don't trigger an S3 write (we keep the env
    # unset so write_progress_marker short-circuits via missing run_id).
    monkeypatch.delenv("RUNPOD_POD_ID", raising=False)

    started = time.perf_counter()
    report = train(cfg)
    elapsed = time.perf_counter() - started

    assert elapsed < 30.0, f"smoke train took {elapsed:.1f}s — too slow"
    assert report.max_steps_hit is True
    assert report.steps_run == int(cfg["training"]["max_steps"])  # type: ignore[index]
    assert report.weights_path.is_file()

    metrics_path = tmp_path / "metrics.json"
    run_meta_path = tmp_path / "run.json"
    assert metrics_path.is_file()
    assert run_meta_path.is_file()

    metrics = json.loads(metrics_path.read_text())
    assert metrics["max_steps_hit"] is True
    assert metrics["device"] == "cpu"
    assert len(metrics["loss_history"]) == report.steps_run

    run_meta = json.loads(run_meta_path.read_text())
    assert run_meta["case"] == "imitation/case0"
    assert run_meta["purpose"] == "runpod_e2e_smoke"
