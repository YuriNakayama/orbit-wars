"""Unit tests for src.utils.gpu_bench.

Covers the two helpers with non-GPU, non-subprocess logic:
`run_dir` (env override vs timestamped fallback) and `reload_jax` (which
sys.modules prefixes get evicted before re-import). `install_cuda_jax` and
`diagnostic_log` are subprocess/GPU-bound and are exercised on the pod, not
here.
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest

from utils.gpu_bench import reload_jax, run_dir


def test_run_dir_uses_env_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = tmp_path / "run123"
    monkeypatch.setenv("ORBIT_WARS_RUN_DIR", str(target))

    assert run_dir() == target


def test_run_dir_fallback_is_created(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("ORBIT_WARS_RUN_DIR", raising=False)
    monkeypatch.chdir(tmp_path)

    result = run_dir()

    assert result.exists()
    assert result.parent.name == "bench_local"


def test_reload_jax_evicts_default_and_extra_prefixes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # We run reload_jax against a *fake* modules mapping (via the `modules`
    # seam) so the real jax / jaxlib / orbit_wars_jax modules loaded by sibling
    # tests on this worker are not evicted — re-initializing native jaxlib
    # mid-process corrupts their state and crashes the worker. import_module is
    # stubbed for the same reason. The eviction logic is what this test checks.
    sentinel = types.ModuleType("jax")
    fake_modules = {
        "jax": types.ModuleType("jax-old"),
        "jaxlib.foo": types.ModuleType("jaxlib.foo"),
        "orbit_wars_jax.step": types.ModuleType("orbit_wars_jax.step"),
        "my.featurizer_jax": types.ModuleType("my.featurizer_jax"),
        "keep_me": types.ModuleType("keep_me"),
    }
    monkeypatch.setattr(
        "utils.gpu_bench.importlib.import_module", lambda name: sentinel
    )

    reloaded = reload_jax(extra_prefixes=("my.featurizer_jax",), modules=fake_modules)

    # Default prefixes + the extra one are gone; the unrelated module stays.
    assert "jaxlib.foo" not in fake_modules
    assert "orbit_wars_jax.step" not in fake_modules
    assert "my.featurizer_jax" not in fake_modules
    assert "keep_me" in fake_modules
    # reload_jax returns whatever importlib.import_module("jax") yields.
    assert reloaded is sentinel
