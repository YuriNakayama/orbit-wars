#!/usr/bin/env bash
# Post-training pipeline for case6 PFSP/pool_v1: convert JAX weights to torch
# and evaluate vs baseline_v1 over 30 episodes (local-only constraint).
#
# Usage:
#   cd bot
#   bash pipeline/reinforce/case6/training/post_train_eval.sh <run_id>
#
# Inputs:
#   $1 = run_id (e.g. 20260601-171440__feature-agent-pool-learning__8ea27b4__seed0)
#   $POST_TRAIN_CASE_SUBDIR (env, optional): case subdir under data/output/models/reinforce/
#     default: case6_kaggle_jax_train_pool_v1 (matches case_arg in cases.py)
#   $EPISODES (env, optional, default 30): number of eval episodes (local constraint)
#   $BASELINE (env, optional, default baseline_v1)
#
# Outputs:
#   pipeline/reinforce/case6/policy/weights.pt (updated)
#   data/mart/reinforce/case6/eval_metrics.json (eval result + Wilson CI)
#
# Note: this updates the CHALLENGER weights.pt globally. If you want to keep
# the previous weights for comparison, copy them aside before running.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <run_id>" >&2
  exit 64
fi

RUN_ID="$1"
CASE_SUBDIR="${POST_TRAIN_CASE_SUBDIR:-case6_kaggle_jax_train_pool_v1}"
EPISODES="${EPISODES:-30}"
BASELINE="${BASELINE:-baseline_v1}"

RUN_DIR="data/output/models/reinforce/${CASE_SUBDIR}/runs/${RUN_ID}"
BEST_PT="${RUN_DIR}/best.pt"
WEIGHTS_OUT="pipeline/reinforce/case6/policy/weights.pt"

if [[ ! -f "$BEST_PT" ]]; then
  echo "best.pt not found at $BEST_PT" >&2
  echo "Pull artifacts first: dev/kaggle pull $RUN_ID --case <case_arg>" >&2
  exit 65
fi

echo "==> jax_to_torch: $BEST_PT -> $WEIGHTS_OUT"
uv run python -m pipeline.reinforce.case6.training.jax_to_torch \
  --best "$BEST_PT" --out "$WEIGHTS_OUT"

echo "==> eval_vs_baseline: rl_v6 vs $BASELINE x $EPISODES eps"
uv run python -m pipeline.reinforce.case6.evaluation.eval_vs_baseline \
  --episodes "$EPISODES" --baseline "$BASELINE" \
  --label "${RUN_ID}-vs-${BASELINE}"

echo "==> done. metrics: data/mart/reinforce/case6/eval_metrics.json"
