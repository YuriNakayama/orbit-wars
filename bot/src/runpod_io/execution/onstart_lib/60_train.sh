# preprocess-only mode: <TRAIN_MODULE> が空なら train セクションを全スキップ。
# preprocess の成果物 (data/mart/...) は前段で DVC 永続化済み。
if [ -z "<TRAIN_MODULE>" ]; then
  echo "[onstart] step=train_skip (preprocess-only mode, TRAIN_MODULE empty)"
  mark "70_train_done"
  mark "99_done"
  exit 0
fi

mark "60_before_train"
echo "[onstart] step=train_direct stage=<STAGE> case=<CASE>"
echo "[onstart] train data listing before launch:"
find data/mart/imitation -maxdepth 2 \( -name '*.parquet' -o -name '*.parquet.dvc' \) \
  -printf '%p %s bytes\n' 2>&1 | sort | head -40
# snapshot は引用符を含む JSON なので、RunPod env では base64 で渡されている。
# 学習プロセス起動時に decode して ORBIT_WARS_RUNPOD_OFFER_SNAPSHOT に再注入する。
SNAPSHOT_JSON=""
if [ -n "${ORBIT_WARS_RUNPOD_OFFER_SNAPSHOT_B64:-}" ]; then
  SNAPSHOT_JSON="$(echo "${ORBIT_WARS_RUNPOD_OFFER_SNAPSHOT_B64}" | base64 -d)"
fi
# train.log と gpu.log を run_dir 配下に分離して書き出す。これにより
# `dev/runpod tail <run_id> --source train` / `--source gpu` で SSH 経由
# リアルタイムに眺められる。onstart.log とは tee で分岐済み (整合性確保)。
TRAIN_LOG="${RUN_DIR_ABS}/train.log"
GPU_LOG="${RUN_DIR_ABS}/gpu.log"
mkdir -p "$(dirname "${TRAIN_LOG}")"

# nvidia-smi を background で 10s 周期サンプリング → gpu.log に append
NVSMI_PID=""
if command -v nvidia-smi >/dev/null 2>&1; then
  ( nvidia-smi \
      --query-gpu=timestamp,utilization.gpu,memory.used,memory.total \
      --format=csv -l 10 > "${GPU_LOG}" 2>&1 ) &
  NVSMI_PID=$!
  echo "[onstart] gpu sampler pid=${NVSMI_PID} -> ${GPU_LOG}"
fi

# CPU / RAM を python (psutil) で 10s 周期サンプリング → system.log
# nvidia-smi が不在 (CPU smoke / 古い image) でも常に走る。
# iter11: `uv run` 経由排除して .venv/bin/python 直接呼び。train プロセスと
# 同じ venv を共有しているため uv による sync が起動中に走らないようにする。
SYSMON_LOG="${RUN_DIR_ABS}/system.log"
SYSMON_PID=""
( cd bot && "${PY_BIN}" -m runpod_io.execution.system_monitor \
    --interval 10 --output "${SYSMON_LOG}" >/dev/null 2>&1 ) &
SYSMON_PID=$!
echo "[onstart] system sampler pid=${SYSMON_PID} -> ${SYSMON_LOG}"

# iter11: `uv run` 経由排除。cwd=bot に切り替えて .venv/bin/python 直接呼び。
# train は最長プロセスで、起動時に裏で別の uv コマンドが走ると hang or crash
# (case8 iter4-9 の `60_before_train` 直後 hang はこれが root cause 候補)。
#
# iter18: train 起動直後に 900s 以上 marker が進まず watcher が stalled 判定した。
# 原因特定のため、train pipeline を background 実行し、60s ごとに heartbeat marker
# と train/gpu/system log snapshot を S3 artifacts prefix へアップロードする。
(
  set -o pipefail
  # 2026-05-20: stream best.pt directly to S3 from train.py on every
  # best_updated event so host preempt mid-training still preserves the
  # latest best (observed 011220: epoch 2 progress lost when pod vanished).
  # The prefix is a directory-style S3 URI (no trailing filename); train.py
  # writes both `best_e{N}_vftf{val:.4f}.pt` (history) and `best.pt` (latest).
  BEST_S3_PREFIX="s3://orbit-wars-dvc-286854171013/remote/runpod_artifacts/<RUN_ID>"
  ORBIT_WARS_RUN_DIR="${RUN_DIR_ABS}" \
    ORBIT_WARS_RUN_ID="<RUN_ID>" \
    ORBIT_WARS_GIT_SHA="<COMMIT_SHA>" \
    ORBIT_WARS_GIT_BRANCH="<BRANCH>" \
    ORBIT_WARS_RUNPOD_POD_ID="${INSTANCE_ID}" \
    ORBIT_WARS_RUNPOD_OFFER_SNAPSHOT="${SNAPSHOT_JSON}" \
    ORBIT_WARS_BEST_S3_PREFIX="${BEST_S3_PREFIX}" \
    ORBIT_WARS_COMMAND="cd bot && ${PY_BIN} -m <TRAIN_MODULE> <CONFIG_ARG>" \
    bash -c "cd bot && '${PY_BIN}' -m <TRAIN_MODULE> <CONFIG_ARG>" \
    2>&1 | tee "${TRAIN_LOG}"
) &
TRAIN_PIPE_PID=$!
echo "[onstart] train pipeline pid=${TRAIN_PIPE_PID} -> ${TRAIN_LOG}"
mark "61_train_process_started"
HEARTBEAT=0
while kill -0 "${TRAIN_PIPE_PID}" 2>/dev/null; do
  sleep 60
  if ! kill -0 "${TRAIN_PIPE_PID}" 2>/dev/null; then
    break
  fi
  HEARTBEAT=$((HEARTBEAT + 1))
  echo "[onstart] train heartbeat ${HEARTBEAT}: pid=${TRAIN_PIPE_PID}"
  mark "62_train_heartbeat_${HEARTBEAT}"
  if command -v aws >/dev/null 2>&1; then
    for f in train.log gpu.log system.log; do
      src="${RUN_DIR_ABS}/${f}"
      if [ -f "${src}" ]; then
        aws s3 cp "${src}" "${S3_ARTIFACT_PREFIX}/${f}" 2>/dev/null || true
      fi
    done
    if [ -f /var/log/onstart.log ]; then
      aws s3 cp /var/log/onstart.log "${S3_ARTIFACT_PREFIX}/onstart.log" 2>/dev/null || true
    fi
  fi
done
wait "${TRAIN_PIPE_PID}"
TRAIN_EXIT=$?
echo "[onstart] train pipeline exit=${TRAIN_EXIT}"
mark "63_train_process_exit_${TRAIN_EXIT}"

# 学習終了後に gpu / system sampler を停止
if [ -n "${NVSMI_PID}" ]; then
  kill "${NVSMI_PID}" 2>/dev/null || true
fi
if [ -n "${SYSMON_PID}" ]; then
  # SIGTERM で system_monitor の signal handler が走り fp.close() される
  kill -TERM "${SYSMON_PID}" 2>/dev/null || true
fi

if [ "${TRAIN_EXIT}" -ne 0 ]; then
  echo "[onstart] train exit=${TRAIN_EXIT} FAILED" >&2
  mark "65_train_failed_exit_${TRAIN_EXIT}"
  exit "${TRAIN_EXIT}"
fi

mark "70_train_done"
# onstart log 全体を run_dir に取り込んで DVC 永久化対象にする。dvc add 直前なので
# tee 中の最新行も含めて snapshot される。失敗 path では cleanup_destroy が S3 に
# 直接アップロードするので、ここでは成功 path のみカバーで十分。RUN_DIR_ABS は
# /workspace/orbit-wars/data/output/... の絶対 path (train.py 出力位置と整合)。
if [ -f /var/log/onstart.log ]; then
  cp /var/log/onstart.log "${RUN_DIR_ABS}/onstart.log"
  echo "[onstart] copied onstart.log -> ${RUN_DIR_ABS}/onstart.log"
fi

