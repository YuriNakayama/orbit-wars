# imitation/case0 — RunPod E2E smoke pipeline

Purpose: minimal end-to-end pipeline for verifying the RunPod GPU training basis (`dev/runpod`). NOT for model improvement — the model and dataset are deliberately tiny so the full cycle (commit → push → RunPod → train → S3 → DVC → pull) finishes within minutes.

## Scope

- Tiny synthetic dataset (32 samples, 8-feature vectors, 4-class label) generated deterministically from a seed.
- Tiny MLP (`Linear(8,16) → ReLU → Linear(16,4)`).
- 1 epoch, `max_steps=10`, batch size 4. CPU run completes in < 90s; GPU run completes in < 5min including container boot.
- Agent stub: `random` policy. Used only so `submit --dry-run` packaging works; not adopted as a Kaggle submission candidate.

## What this verifies

1. `dev/runpod train --case case0` end-to-end succeeds.
2. Structured progress markers reach S3 in order.
3. Training-time monitoring (CPU/RAM/GPU) is captured and tail-able.
4. Auto cleanup + retry behave correctly under simulated failures.

See `docs/experiment/imitation/20260505_case0_runpod_e2e/plan.md` for the full plan and `result.md` for measured timings.

## Local commands

```bash
# Generate dummy dataset
cd bot && uv run python -m pipeline.imitation.case0.training.dummy_data \
    --out /tmp/case0_train.parquet --num-samples 32 --seed 0

# CPU smoke train
cd bot && uv run python -m pipeline.imitation.case0.training.train \
    --config pipeline/imitation/case0/configs/smoke.yaml --device cpu

# Submit dry-run (sanity)
cd bot && uv run python -m submit submit imitation/case0 --dry-run --skip-validation -m smoke
```
