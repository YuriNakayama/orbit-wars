# Progress marker — write a tiny file to S3 at each step so we can debug
# without SSH access. AWS env vars are injected by create_pod.
S3_MARKER_PREFIX="s3://orbit-wars-dvc-286854171013/remote/runpod_progress/<RUN_ID>"
S3_ARTIFACT_PREFIX="s3://orbit-wars-dvc-286854171013/remote/runpod_artifacts/<RUN_ID>"
mark() {
  local step="$1"
  local ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  local body="ts=${ts} step=${step} pod=${INSTANCE_ID}"
  echo "[onstart] mark step=${step}"
  if command -v aws >/dev/null 2>&1; then
    # 初回 mark のみ stderr を表示する (silent failure を見逃さない)。
    # 2 回目以降は log を汚さないために suppress。
    if [ -z "${MARK_DEBUG_DONE:-}" ]; then
      echo "${body}" | aws s3 cp - "${S3_MARKER_PREFIX}/${ts}_${step}" \
        || echo "[onstart] mark first-call s3 cp failed (subsequent silent)" >&2
      export MARK_DEBUG_DONE=1
    else
      echo "${body}" | aws s3 cp - "${S3_MARKER_PREFIX}/${ts}_${step}" 2>/dev/null || true
    fi
  else
    echo "[onstart] mark skipped (aws cli not available) step=${step}" >&2
  fi
}
mark "00_container_started"

# RunPod mode: "oneshot" (default, train→DVC push→自動 remove) or "interactive"
# (sleep infinity で pod を維持、auto remove なし、明示的 `dev/runpod destroy`
# で削除する運用)。<RUNPOD_MODE> は render_onstart で展開される。
RUNPOD_MODE="<RUNPOD_MODE>"
echo "[onstart] runpod_mode=${RUNPOD_MODE}"
mark "01_mode_${RUNPOD_MODE}"

# Periodic log flush: 30 秒間隔で /var/log/onstart.log を S3 に push する。
# stall 検出 → 外部から runpodctl remove pod / SDK terminate された場合に
# bash trap (cleanup_destroy) が動かず onstart.log が失われる問題への対策。
# `onstart.live.log` は本セッション中の暫定 path で、cleanup_destroy 成功時
# は最終 `onstart.log` (同 prefix) で上書きされる。aws cli が無ければ noop。
LOG_FLUSHER_PID=""
if command -v aws >/dev/null 2>&1; then
  (
    while true; do
      sleep 30
      if [ -f /var/log/onstart.log ]; then
        aws s3 cp /var/log/onstart.log \
          "${S3_MARKER_PREFIX}/onstart.live.log" \
          --no-progress --only-show-errors >/dev/null 2>&1 || true
      fi
    done
  ) &
  LOG_FLUSHER_PID=$!
  echo "[onstart] log flusher started pid=${LOG_FLUSHER_PID} interval=30s"
fi

# Hard timeout safety net: trap が壊れても 8h で pod を強制 remove する。
# 2026-05-12 観測: case10 base mart preprocess は host CPU 数や IO で
# 大きくばらつき (1.04-3.31 ep/s)、worker=11 cap でも host 性能次第で
# 完走 + dvc commit + push に 4h+ 必要なケースがある。8h に余裕を持たせる。
# remove 前に log を S3 へ最終 snapshot として書き出す (cleanup_destroy が
# 呼ばれない経路の救済 — 例えば `runpodctl remove pod` を外部から叩かれて
# bash に SIGKILL が届いた場合や、kernel panic 等)。
# interactive モードでは timeout guard を張らない (ユーザーが destroy するまで保持)。
TIMEOUT_GUARD_PID=""
if [ "${RUNPOD_MODE}" = "oneshot" ]; then
  (
    sleep 28800
    if [ -f /var/log/onstart.log ] && command -v aws >/dev/null 2>&1; then
      aws s3 cp /var/log/onstart.log "${S3_MARKER_PREFIX}/onstart.log" 2>/dev/null || true
    fi
    runpodctl remove pod "$INSTANCE_ID" 2>/dev/null || true
  ) &
  TIMEOUT_GUARD_PID=$!
else
  echo "[onstart] interactive mode: skipping 8h timeout guard"
fi

cleanup_destroy() {
  local exit_code=$?
  # Stop the timeout guard so it doesn't fire later in the success path.
  if [ -n "${TIMEOUT_GUARD_PID}" ]; then
    kill "$TIMEOUT_GUARD_PID" 2>/dev/null || true
  fi
  # Stop the periodic log flusher: final snapshot is taken below at the
  # canonical `onstart.log` path, so the `onstart.live.log` daemon is no
  # longer needed and would otherwise race with the final upload.
  if [ -n "${LOG_FLUSHER_PID}" ]; then
    kill "${LOG_FLUSHER_PID}" 2>/dev/null || true
  fi
  echo "[onstart] cleanup status=${exit_code}"
  mark "90_cleanup_exit_${exit_code}"
  # 失敗時 (or 成功時の保険) に onstart log 全体を S3 に直接アップロード。
  # 成功 path では後段で run_dir/onstart.log として DVC 永久化されるが、
  # 序盤で失敗すると DVC 経路に到達しないので、必ずここで S3 へ最終 snapshot を残す。
  if [ -f /var/log/onstart.log ] && command -v aws >/dev/null 2>&1; then
    echo "[onstart] uploading log snapshot to s3://${S3_MARKER_PREFIX#s3://}/onstart.log"
    aws s3 cp /var/log/onstart.log \
      "${S3_MARKER_PREFIX}/onstart.log" 2>/dev/null || \
      echo "[onstart] log upload failed (non-fatal)"
  fi
  if command -v aws >/dev/null 2>&1; then
    # Preserve preprocessed mart parquet even if a later train step fails or the
    # pod is externally terminated before DVC push finishes. This is a fallback
    # artifact path, not the canonical DVC path.
    MART_SNAPSHOT_DIR="data/mart/imitation/<CASE>"
    if [ -d "${MART_SNAPSHOT_DIR}" ]; then
      echo "[onstart] uploading mart snapshot to ${S3_ARTIFACT_PREFIX}/mart/"
      aws s3 cp "${MART_SNAPSHOT_DIR}" "${S3_ARTIFACT_PREFIX}/mart/" \
        --recursive 2>/dev/null || \
        echo "[onstart] mart snapshot upload failed (non-fatal)"
    else
      echo "[onstart] mart snapshot skip (not found: ${MART_SNAPSHOT_DIR})"
    fi
  fi
  # RunPod は docker_args の bash プロセスが exit するとコンテナを再起動して
  # docker_args を再実行する (実 e2e で確認済み: stop pod 後も restart loop)。
  # `runpodctl stop pod` は storage を残して停止するだけで、RunPod の自動
  # restart 機構が発動して再実行ループに入る。`runpodctl remove pod` (alias:
  # delete) なら pod 自体が削除されて確実に止まる。
  if command -v runpodctl >/dev/null 2>&1; then
    if [ "${exit_code}" -ne 0 ]; then
      echo "[onstart] failure path: removing pod (exit=${exit_code})"
    else
      echo "[onstart] success path: removing pod=${INSTANCE_ID}"
    fi
    runpodctl remove pod "${INSTANCE_ID}" || echo "[onstart] remove pod failed"
  else
    echo "[onstart] runpodctl missing; cannot self-destroy."
  fi
  # Hang while RunPod tears the pod down. exec で bash 自体を sleep に置換し、
  # trap 連鎖や set -e による即 exit を抑止する。
  exec sleep infinity
}
# interactive モードでは EXIT trap を張らない。pod は sleep infinity で保持され、
# ユーザーが `dev/runpod destroy <run_id>` で明示的に terminate するまで残る。
if [ "${RUNPOD_MODE}" = "oneshot" ]; then
  trap cleanup_destroy EXIT
else
  echo "[onstart] interactive mode: skipping cleanup_destroy EXIT trap"
fi

