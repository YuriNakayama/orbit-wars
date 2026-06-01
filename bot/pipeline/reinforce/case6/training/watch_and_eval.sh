#!/usr/bin/env bash
# Watch a Kaggle Kernel run until completion (COMPLETE / ERROR), then pull
# artifacts and run post_train_eval (jax_to_torch + live v1 30戦) on success.
#
# Usage:
#   cd bot
#   bash pipeline/reinforce/case6/training/watch_and_eval.sh <run_id> <case>
#
# Args:
#   $1 run_id (e.g. 20260601-191229__feature-agent-pool-learning__06b6faf__seed0)
#   $2 case (e.g. reinforce_case6_kaggle_jax_train_pool_v1)
# Env:
#   $POLL_INTERVAL_S (default 120): seconds between status polls
#   $MAX_WAIT_MIN (default 540 = 9h, Kaggle T4x2 cap): give up after this long
#   $EPISODES (default 30, local <=30 constraint): eval episode count
#   $BASELINE (default baseline_v1): eval opponent

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <run_id> <case>" >&2
  exit 64
fi

RUN_ID="$1"
CASE="$2"
POLL="${POLL_INTERVAL_S:-120}"
MAX_WAIT_MIN="${MAX_WAIT_MIN:-540}"

# Derive case_subdir for post_train_eval (e.g. reinforce_case6_kaggle_jax_train_pool_v1
# -> case6_kaggle_jax_train_pool_v1). The run dir mirrors the case suffix.
CASE_SUBDIR="${CASE#reinforce_}"

start=$(date +%s)
echo "==> watching $RUN_ID (case=$CASE, poll=${POLL}s, max=${MAX_WAIT_MIN}m)"

while :; do
  elapsed=$(( ($(date +%s) - start) / 60 ))
  if [[ $elapsed -ge $MAX_WAIT_MIN ]]; then
    echo "TIMEOUT: $MAX_WAIT_MIN min elapsed, giving up" >&2
    exit 75
  fi
  # dev/kaggle status returns text; grep its tail for status word.
  status_line=$(dev/kaggle status "$RUN_ID" --case "$CASE" 2>&1 | tail -3 | tr -d '\n')
  if echo "$status_line" | grep -q "COMPLETE"; then
    echo "==> COMPLETE after ${elapsed}min"
    break
  elif echo "$status_line" | grep -q "ERROR"; then
    echo "==> ERROR after ${elapsed}min" >&2
    echo "$status_line" >&2
    exit 70
  fi
  echo "[${elapsed}min] still RUNNING"
  sleep "$POLL"
done

echo "==> pulling artifacts"
dev/kaggle pull "$RUN_ID" --case "$CASE"

echo "==> running post_train_eval"
POST_TRAIN_CASE_SUBDIR="$CASE_SUBDIR" \
  bash pipeline/reinforce/case6/training/post_train_eval.sh "$RUN_ID"

echo "==> done. result: data/mart/reinforce/case6/eval_metrics.json"
