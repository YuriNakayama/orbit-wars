"""Shared GPU / JAX plumbing for RunPod-hosted benchmarks.

Extracted from the per-bench `pipeline/reinforce/_bench/*_gpu/run_bench.py`
modules where these helpers were copy-pasted (and had drifted). They are
benchmark-agnostic: any GPU bench (or future GPU job) that needs to resolve a
run directory, (re)install the cuda12 JAX wheel, reload jax after install, or
emit device diagnostics can import them here.

Bench-specific pieces (`BenchResult`, `_bench_*`, `main`) stay in each bench
module — only this infrastructure is shared.
"""

from __future__ import annotations

import importlib
import logging
import os
import subprocess
import sys
from collections.abc import Iterable, MutableMapping
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

logger = logging.getLogger(__name__)


def run_dir() -> Path:
    """Resolve the artifact directory for a bench run.

    Uses `$ORBIT_WARS_RUN_DIR` (set by the RunPod onstart) when present, else a
    timestamped `bench_local/` fallback for local smoke runs.
    """
    env_dir = os.environ.get("ORBIT_WARS_RUN_DIR")
    if env_dir:
        return Path(env_dir)
    fallback = Path("bench_local") / datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def install_cuda_jax() -> None:
    """Reinstall jax + jaxlib with cuda12 plugin in the active venv.

    Pinning is intentionally loose (`jax[cuda12]`): jax/jaxlib has minor-version
    drift on the cuda plugin and we just need *any* GPU-capable build at bench
    time. The cpu wheel already in `.venv` is replaced.
    """
    logger.info("installing jax[cuda12] via uv pip…")
    proc = subprocess.run(
        ["uv", "pip", "install", "--reinstall", "jax[cuda12]"],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        logger.error(
            "uv pip install jax[cuda12] failed: %s", (proc.stderr or "")[-500:]
        )
        proc.check_returncode()
    logger.info("jax[cuda12] install OK")


def reload_jax(
    extra_prefixes: Iterable[str] = (),
    modules: MutableMapping[str, ModuleType] | None = None,
) -> Any:
    """Drop jax (and dependent) modules from sys.modules and re-import jax.

    After `install_cuda_jax()` the already-imported cpu jax must be evicted so
    the cuda plugin is picked up. `orbit_wars_jax` imports jax at module load,
    so it is always reloaded too. Pass `extra_prefixes` for additional modules
    that bind jax at import time (e.g. a bench's `featurizer_jax`). `modules`
    defaults to the live `sys.modules`; it exists as a seam for testing the
    eviction logic without touching the real interpreter state.
    """
    mods = sys.modules if modules is None else modules
    prefixes = ("jax.", "jaxlib", "orbit_wars_jax", *tuple(extra_prefixes))
    for mod in list(mods):
        if mod == "jax" or mod.startswith(prefixes):
            mods.pop(mod, None)
    return importlib.import_module("jax")


def diagnostic_log() -> None:
    """Emit JAX env / device diagnostics + a tiny GPU smoke.

    Intended as the very first action in a bench `main()`. Distinguishes among:
      (a) cuda12 plugin detected, GPU usable → bench should proceed
      (b) devices=[CpuDevice] only → cuda12 plugin install or load failed
      (c) `block_until_ready` hangs/crashes → cuda runtime/driver mismatch
    Each `print` uses flush=True so the line reaches train.log before any
    potential silent crash in subsequent JAX state.
    """
    print("=== JAX diagnostic start ===", flush=True)
    for k in ("RUNPOD_POD_ID", "CUDA_VISIBLE_DEVICES", "JAX_PLATFORMS"):
        print(f"env {k}={os.environ.get(k, '<unset>')}", flush=True)
    ldconfig_cmd = "ldconfig -p 2>/dev/null | grep libcuda.so | head -3"
    print(f"ldconfig libcuda: {os.popen(ldconfig_cmd).read().strip()}", flush=True)
    nvsmi_cmd = (
        "nvidia-smi --query-gpu=name,driver_version "
        "--format=csv,noheader 2>&1 | head -1"
    )
    print(f"nvidia-smi: {os.popen(nvsmi_cmd).read().strip()}", flush=True)
    print("pip jax-cuda12 packages:", flush=True)
    pip_cmd = 'uv pip list 2>&1 | grep -iE "^jax|^cuda" | head -10'
    print(os.popen(pip_cmd).read(), flush=True)

    import jax

    print(f"jax version: {jax.__version__}", flush=True)
    try:
        devs = jax.devices()
        print(f"jax devices: {devs}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"jax.devices() RAISED: {exc!r}", flush=True)
        return
    print(f"jax default_backend: {jax.default_backend()}", flush=True)

    import jax.numpy as jnp

    try:
        x = jnp.ones(1000)
        y = (x * 2.0).sum()
        val = float(y.block_until_ready())
        print(f"GPU smoke OK: y={val}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"GPU smoke RAISED: {exc!r}", flush=True)
    print("=== JAX diagnostic end ===", flush=True)
