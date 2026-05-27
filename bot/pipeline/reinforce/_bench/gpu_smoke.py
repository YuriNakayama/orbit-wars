"""Minimal GPU smoke test for validating the RunPod GPU-acquisition flow.

Run as a module from ``bot/``::

    python -m pipeline.reinforce._bench.gpu_smoke

It prints CUDA availability, the GPU name and total VRAM, performs one tiny
CUDA tensor op, and exits 0. Designed to finish in seconds so the RunPod
basis itself can be validated end-to-end without incurring real GPU cost.
"""

from __future__ import annotations

import sys


def main() -> int:
    import torch

    available = torch.cuda.is_available()
    print(f"[gpu_smoke] torch.cuda.is_available() = {available}")
    print(f"[gpu_smoke] torch.__version__ = {torch.__version__}")

    if not available:
        print("[gpu_smoke] CUDA not available — failing smoke test", file=sys.stderr)
        return 1

    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    total_vram_gib = props.total_memory / (1024**3)
    print(f"[gpu_smoke] device_name = {name}")
    print(f"[gpu_smoke] total_vram = {total_vram_gib:.2f} GiB")

    # One tiny tensor op on the GPU.
    a = torch.tensor([2.0, 3.0, 4.0], device="cuda")
    b = torch.tensor([10.0, 10.0, 10.0], device="cuda")
    c = (a * b).cpu().tolist()
    print(f"[gpu_smoke] a * b = {c}")

    expected = [20.0, 30.0, 40.0]
    if c != expected:
        print(f"[gpu_smoke] unexpected result {c} != {expected}", file=sys.stderr)
        return 1

    print("[gpu_smoke] OK — GPU op succeeded, exiting 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
