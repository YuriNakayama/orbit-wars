"""Scale probe: strict_jax_v1 rollout — episodes upper limit + resource needs.

Sweeps episodes_per_iter at horizon=500 (the real training shape) against the
strict_jax_v1 opponent, measuring per size:
  * compile+first rollout secs, steady (2nd) rollout secs,
  * peak GPU util / GPU mem (nvidia-smi) and process CPU% / RSS during the run.
Continues past OOM failures so the limit is bracketed in one run.

Run ON the GPU pod:
    python -m pipeline.reinforce._bench.strict_rollout_scale_probe
"""

from __future__ import annotations

import subprocess
import threading
import time

from utils.gpu_bench import install_cuda_jax, reload_jax

EPISODE_SIZES = [16, 64, 96, 128]
HORIZON = 500


class _ResourceSampler:
    """Background sampler: peak GPU util/mem + process CPU%/RSS every 2s."""

    def __init__(self) -> None:
        self.peak_gpu_util = 0
        self.peak_gpu_mem = 0
        self.peak_cpu = 0.0
        self.peak_rss_mb = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        import os

        pid = str(os.getpid())
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                ).stdout.strip()
                util, mem = (int(x) for x in out.split(","))
                self.peak_gpu_util = max(self.peak_gpu_util, util)
                self.peak_gpu_mem = max(self.peak_gpu_mem, mem)
            except Exception:  # noqa: BLE001 — sampler must never kill the probe
                pass
            try:
                out = subprocess.run(
                    ["ps", "-o", "pcpu=,rss=", "-p", pid],
                    capture_output=True,
                    text=True,
                    timeout=5,
                ).stdout.split()
                self.peak_cpu = max(self.peak_cpu, float(out[0]))
                self.peak_rss_mb = max(self.peak_rss_mb, float(out[1]) / 1024)
            except Exception:  # noqa: BLE001
                pass
            time.sleep(2)

    def __enter__(self) -> "_ResourceSampler":
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)


def main() -> None:
    install_cuda_jax()
    reload_jax(extra_prefixes=("pipeline.reinforce", "pipeline.rulebase"))

    import jax

    from pipeline.reinforce.case7.training.rollout_jax import collect_rollout_jax
    from pipeline.reinforce.case7.training.train_jax import _build_model

    print(f"devices: {jax.devices()}", flush=True)
    model = _build_model({"model": {}})  # config defaults == timing configs
    key = jax.random.PRNGKey(0)

    results = []
    for eps in EPISODE_SIZES:
        try:
            with _ResourceSampler() as rs:
                t0 = time.perf_counter()
                batch = collect_rollout_jax(
                    model,
                    key,
                    eps,
                    horizon=HORIZON,
                    opponent="strict_jax_v1",
                    shaping_coef=0.5,
                    shaping_mode="planets",
                )
                jax.block_until_ready(batch.rewards)
                first = time.perf_counter() - t0
                t0 = time.perf_counter()
                batch = collect_rollout_jax(
                    model,
                    key,
                    eps,
                    horizon=HORIZON,
                    opponent="strict_jax_v1",
                    shaping_coef=0.5,
                    shaping_mode="planets",
                )
                jax.block_until_ready(batch.rewards)
                steady = time.perf_counter() - t0
            results.append((eps, first, steady, rs))
            print(
                f"eps={eps:4d} compile+1st={first:7.1f}s steady={steady:7.1f}s "
                f"per_ep={steady / eps:5.2f}s | peak GPU {rs.peak_gpu_util}% "
                f"{rs.peak_gpu_mem}MiB | CPU {rs.peak_cpu:.0f}% RSS {rs.peak_rss_mb:.0f}MB",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 — bracket the OOM limit, keep going
            msg = str(exc)[:140].replace("\n", " ")
            print(f"eps={eps:4d} FAILED: {msg}", flush=True)

    print("\n=== SUMMARY (horizon=500, opponent=strict_jax_v1) ===", flush=True)
    for eps, first, steady, rs in results:
        print(
            f"eps={eps:4d}: steady={steady:6.1f}s ({steady / 60:4.1f}min) "
            f"per_ep={steady / eps:5.2f}s gpu_mem={rs.peak_gpu_mem}MiB "
            f"cpu={rs.peak_cpu:.0f}% rss={rs.peak_rss_mb:.0f}MB",
            flush=True,
        )


if __name__ == "__main__":
    main()
