#!/bin/bash
# Helper for the case8 beat-rulebase PoC loop. Given an already-running RunPod
# interactive pod (run_id) and a config name, this: installs CUDA jaxlib if
# missing, writes a tmux launcher with the S3 crash-safe prefix, and starts
# training detached so it survives SSH disconnect.
#
# Usage (from repo root):
#   dev/runpod dev "$(git rev-parse HEAD)" --case case8      # launch pod, note run_id
#   bash bot/pipeline/reinforce/case8/training/launch_poc.sh <run_id> <config_name>
#
# config_name is the bare filename under configs/ (e.g. h1_handicap.yaml).
set -euo pipefail
RUN_ID="$1"
CONFIG="$2"
# Branch to deploy on the pod. Defaults to the current local branch so the helper
# works from any worktree (pass an explicit 3rd arg to override).
BRANCH="${3:-$(git rev-parse --abbrev-ref HEAD)}"
S3_PREFIX="s3://orbit-wars-dvc-286854171013/remote/runpod_artifacts/${RUN_ID}"
SSH="dev/runpod ssh ${RUN_ID} --case case8 --via direct --exec"

# 1) ensure GPU jaxlib (dev pod ships CPU-only jax)
$SSH "cd /workspace/orbit-wars/bot && .venv/bin/python -c 'import jax; print(jax.default_backend())' | grep -q gpu || uv pip install 'jax[cuda12]==0.10.0'"

# 2) pull latest code on the requested branch
$SSH "cd /workspace/orbit-wars && git fetch origin -q && git checkout -q ${BRANCH} && git pull -q origin ${BRANCH}"

# 3) write tmux launcher + start
$SSH "cat > /workspace/run_train.sh <<EOF
#!/bin/bash
cd /workspace/orbit-wars/bot
export ORBIT_WARS_BEST_S3_PREFIX='${S3_PREFIX}'
export ORBIT_WARS_RUN_ID='${RUN_ID}'
exec .venv/bin/python -m pipeline.reinforce.case8.training.train_jax --config pipeline/reinforce/case8/configs/${CONFIG}
EOF
chmod +x /workspace/run_train.sh; tmux kill-session -t train 2>/dev/null; rm -f /workspace/logs/train.log; mkdir -p /workspace/logs; tmux new-session -d -s train 'bash /workspace/run_train.sh > /workspace/logs/train.log 2>&1'; sleep 3; tmux ls"
echo "launched ${CONFIG} on ${RUN_ID}"
