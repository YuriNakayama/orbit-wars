PREPROCESS_RAN=0
# Special branch: parquet_to_npy converters do not produce a new parquet;
# they read the existing mart parquet and emit per-column .npy files for
# mmap consumption. Skip the parquet freshness check + post-preprocess
# DVC track block entirely. The targeted mart pull in 40 already fetched
# the parquet, so we just run the converter once and proceed to train.
if [[ "<PREPROCESS_CMD>" == *"parquet_to_npy"* ]]; then
  echo "[onstart] step=parquet_to_npy case=<CASE> (skip parquet-skip + dvc-track)"
  # The .npy files are 100GB+ uncompressed. Container disk on most pods
  # (RTX 4090 SECURE: ~30GB) is too small, causing SIGBUS on mmap write.
  # Force out-root onto /persist (network volume, 300GB). Also keep
  # train_parquet / val_parquet as inputs read from the symlinked mart dir.
  NPY_OUT_ROOT="/persist/data-mart-imitation/<CASE>"
  mkdir -p "${NPY_OUT_ROOT}"
  TRAIN_PQ_ABS="$(pwd)/data/mart/imitation/<CASE>/train.parquet"
  VAL_PQ_ABS="$(pwd)/data/mart/imitation/<CASE>/val.parquet"
  # Cleanup stale .npy directories from prior runs. The network volume
  # has a 300GB quota that is shared across runs (orbit_wars), so the
  # previous run's train_npy (~174GB) lingers and triggers "Disk quota
  # exceeded" on the next conversion (observed retry #16 commit 7324d4f).
  echo "[onstart] /persist before cleanup:"
  df -h /persist 2>&1 | tail -2 || true
  echo "[onstart] /persist usage breakdown (top-level dirs, slow):"
  du -sh /persist/* 2>/dev/null | sort -h | tail -20 || true
  # Drop the per-case .npy directories and any other lingering imitation
  # converters' outputs to maximize free quota for the upcoming write.
  for split in train val; do
    if [ -d "${NPY_OUT_ROOT}/${split}_npy" ]; then
      echo "[onstart] cleaning stale ${NPY_OUT_ROOT}/${split}_npy"
      rm -rf "${NPY_OUT_ROOT}/${split}_npy"
    fi
  done
  # Also remove any sibling case_npy dirs from earlier experiments since
  # they're regenerable from parquet (~191GB recoverable per case).
  find /persist/data-mart-imitation -maxdepth 2 -type d -name "*_npy" 2>/dev/null \
    | while read -r d; do
        echo "[onstart] cleaning stale npy dir ${d}"
        rm -rf "${d}"
      done
  # Volume quota is 300GB. After 75GB dvc-cache + 46GB uv-cache + 6GB uv-venv
  # only ~173GB remains — 1GB short of train_npy's 174GB footprint.
  # Drop dvc-cache (regenerable from S3 in 5-10min via dvc pull) to free
  # 75GB. This is the cheapest way to fit the conversion under quota.
  if [ -d /persist/dvc-cache ]; then
    DVC_CACHE_SIZE=$(du -sh /persist/dvc-cache 2>/dev/null | awk '{print $1}')
    echo "[onstart] cleaning /persist/dvc-cache (was ${DVC_CACHE_SIZE}) to free quota"
    rm -rf /persist/dvc-cache
    mkdir -p /persist/dvc-cache
  fi
  echo "[onstart] /persist after cleanup:"
  df -h /persist 2>&1 | tail -2 || true
  echo "[onstart] parquet_to_npy out_root=${NPY_OUT_ROOT}"
  # parquet_to_npy takes ~14min and emits no marker by itself, which
  # tickles the 900s STALL_THRESHOLD watcher and gets the pod killed
  # mid-conversion (observed retry on commit 9584b73). Run a background
  # heartbeat loop that emits a step=54_parquet_to_npy_heartbeat_N marker
  # every 4min so the watcher sees ongoing progress.
  mark "54_parquet_to_npy_started"
  (
    n=1
    while sleep 240; do
      mark "54_parquet_to_npy_heartbeat_${n}"
      n=$((n + 1))
    done
  ) &
  HB_PID=$!
  if ! ( cd bot && "${PY_BIN}" -m <PREPROCESS_CMD> \
      --train-parquet "${TRAIN_PQ_ABS}" \
      --val-parquet "${VAL_PQ_ABS}" \
      --out-root "${NPY_OUT_ROOT}" ); then
    kill "${HB_PID}" 2>/dev/null || true
    echo "[onstart] step=parquet_to_npy FAILED (exit code != 0)" >&2
    mark "55_parquet_to_npy_failed"
    exit 1
  fi
  kill "${HB_PID}" 2>/dev/null || true
  # Symlink the per-split npy dirs back under data/mart/imitation/<CASE>/
  # so the dataset's `_npy_dir_for` resolution (sibling of train.parquet)
  # picks them up without code changes.
  MART_CASE_DIR="$(pwd)/data/mart/imitation/<CASE>"
  for split in train val; do
    if [ -d "${NPY_OUT_ROOT}/${split}_npy" ]; then
      rm -rf "${MART_CASE_DIR}/${split}_npy" 2>/dev/null || true
      ln -s "${NPY_OUT_ROOT}/${split}_npy" "${MART_CASE_DIR}/${split}_npy"
      echo "[onstart] linked ${MART_CASE_DIR}/${split}_npy -> ${NPY_OUT_ROOT}/${split}_npy"
    fi
  done
  echo "[onstart] step=parquet_to_npy done"
elif [ -n "<PREPROCESS_CMD>" ]; then
  # dvc pull は missing blob を WARNING で済ませて exit 0 を返すため、
  # `.dvc` stub は git にあるが本体 blob が S3 にない (orphan) ケースで
  # `data/mart/imitation/<CASE>/*.parquet` が dangling symlink として
  # 残ることがある。`find -name '*.parquet'` は dangling symlink も拾うので
  # `-xtype f` で「symlink を辿った先が file かつ存在」のみ採用、さらに
  # `-size +0c` で 0 byte を弾く (DVC cache 経由なら本体は MB〜GB のはず)。
  #
  # 加えて (trap #9 永久対策、観測 2026-05-05 case7 iter2): parquet が存在しても
  # **dvc.yaml の deps (preprocess.py / featurizer.py / configs/*.yaml 等) が変わって
  # いれば schema が古い可能性**があるため `dvc status` で stage が up-to-date か
  # を必ず確認する。`dvc status preprocess_imitation_<CASE>` が "Data and pipelines
  # are up to date." を返すときのみ skip。それ以外 (changed deps / changed outs) は
  # 再実行する。
  PARQUET_PRESENT=0
  if [ -d "data/mart/imitation/<CASE>" ] && \
     find "data/mart/imitation/<CASE>" -maxdepth 1 -name '*.parquet' \
       \( -type f -o -xtype f \) -size +0c -print -quit | grep -q . ; then
    PARQUET_PRESENT=1
  fi
  PREPROCESS_STAGE_NAME="preprocess_imitation_<CASE>"
  STAGE_UPTODATE=0
  if [ "${PARQUET_PRESENT}" -eq 1 ]; then
    echo "[onstart] checking ${PREPROCESS_STAGE_NAME} freshness via dvc status"
    DVC_STATUS_OUT=$(${DVC_BIN} status "${PREPROCESS_STAGE_NAME}" 2>&1)
    echo "${DVC_STATUS_OUT}" | head -20
    if echo "${DVC_STATUS_OUT}" | grep -qiE "up to date|nothing to reproduce"; then
      STAGE_UPTODATE=1
    fi
  fi
  if [ "${PARQUET_PRESENT}" -eq 1 ] && [ "${STAGE_UPTODATE}" -eq 1 ]; then
    echo "[onstart] step=preprocess_skip (parquet present and stage up-to-date)"
  else
    if [ "${PARQUET_PRESENT}" -eq 1 ] && [ "${STAGE_UPTODATE}" -eq 0 ]; then
      echo "[onstart] step=preprocess_force_rerun (parquet present but deps changed)"
      # 古い parquet を削除してから走らせる (write_parquet の上書きはするが、
      # symlink target の場合に dvc cache 不整合を防ぐため明示削除)
      find "data/mart/imitation/<CASE>" -maxdepth 1 -name '*.parquet' \
        \( -type f -o -type l \) -print -delete 2>&1 || true
    else
      echo "[onstart] step=preprocess case=<CASE>"
      # 既存の dangling symlink を消してから preprocess を走らせる。残ったまま
      # だと preprocess.py が write_parquet で symlink target に書き込もうとして
      # 元の blob path に書き込もうとするため (dvc cache の symlink type の場合)。
      if [ -d "data/mart/imitation/<CASE>" ]; then
        find "data/mart/imitation/<CASE>" -maxdepth 1 -name '*.parquet' \
          \( -type l -xtype l \) -print -delete 2>&1 || true
      fi
    fi
    # iter11: `uv run` 経由を排除 (race condition 回避)。cwd を bot に揃えて
    # .venv/bin/python を直接起動すれば uv は介在せず deps は固定。
    # preprocess の exit code を check して、 失敗時は train に進まず onstart fail。
    #
    # 2026-05-12 観測: RunPod host の CPU 数によっては自動 worker 数 (cpu-1)
    # が 47 まで膨らみ、worker 間競合で rate が 3.31 ep/s → 1.04 ep/s に低下
    # (~3x slowdown)。case10 base mart preprocess (~16k 1v1 episodes) で
    # 確認。11 workers に固定して前回計測の安定 rate に揃える。
    export ORBIT_WARS_PREPROCESS_WORKERS=11
    echo "[onstart] preprocess workers cap: ORBIT_WARS_PREPROCESS_WORKERS=11"
    if ! ( cd bot && "${PY_BIN}" -m <PREPROCESS_CMD> ); then
      echo "[onstart] step=preprocess FAILED (exit code != 0)" >&2
      mark "55_preprocess_failed"
      exit 1
    fi
    PREPROCESS_RAN=1
  fi
fi

# preprocess を実走した場合、生成された parquet を DVC で track して
# git に .dvc stub を push する。次回以降は git pull → dvc pull だけで
# data/mart/imitation/<CASE>/ が復元され、preprocess は skip される。
#
# DVC は symlinked dir 内のファイルに `dvc add` できないため、persist_setup
# で張った data/mart/imitation -> /persist/data-mart-imitation の symlink を
# 一時的に解除し、中身を実 dir にコピーで戻してから dvc add する。完了後に
# symlink を再構築 (next run の cache 利用は維持)。観測: 2026-05-04 case7 で
# `Cannot add files inside symlinked directories` で onstart 失敗。
if [ "${PREPROCESS_RAN}" -eq 1 ]; then
  echo "[onstart] step=mart_dvc_persist case=<CASE>"
  MART_LINK_TARGET=""
  if [ -L data/mart/imitation ]; then
    MART_LINK_TARGET=$(readlink data/mart/imitation)
    echo "[onstart] mart_dvc_persist: unlinking data/mart/imitation -> ${MART_LINK_TARGET}"
    rm data/mart/imitation
    mkdir -p data/mart/imitation
    # 中身を実 dir にコピーで戻す。`-a` で permissions / mtime を保持
    cp -a "${MART_LINK_TARGET}/." data/mart/imitation/ 2>&1 | tail -3 || true
  fi

  MART_DIR="data/mart/imitation/<CASE>"
  PREPROCESS_STAGE="preprocess_imitation_<CASE>"

  # parquet が存在することを確認 (preprocess が成功したら必ずある)
  shopt -s nullglob
  parquet_files=("${MART_DIR}"/*.parquet)
  shopt -u nullglob
  if [ "${#parquet_files[@]}" -eq 0 ]; then
    echo "[onstart] mart_dvc_persist: no parquet found under ${MART_DIR}; skipping"
  else
    # `dvc.yaml` の preprocess_imitation_<CASE> stage に outs として登録された
    # parquet を track する。`dvc add` は stage outs と重複するので NG。
    # `dvc commit -f <stage>` で `dvc.lock` の該当 stage の output hash を更新し、
    # 続いて `dvc push <stage>` で remote にアップロード、最後に `dvc.lock` を
    # git push する (`*.parquet.dvc` stub は生成されない)。
    # 観測: 2026-05-04 case7 で `dvc add` が "overlaps with an output of stage"
    # で fatal exit (4回目の試行)。
    echo "[onstart] dvc commit -f ${PREPROCESS_STAGE}"
    if ! ${DVC_BIN} commit -f "${PREPROCESS_STAGE}" 2>&1; then
      echo "[onstart] dvc commit ${PREPROCESS_STAGE} FATAL" >&2
      mark "55_mart_dvc_add_failed"
      # symlink を再構築してから exit (次回 run の cache 維持)
      if [ -n "${MART_LINK_TARGET}" ]; then
        rsync -a --delete data/mart/imitation/ "${MART_LINK_TARGET}/" 2>&1 | tail -3 || true
        rm -rf data/mart/imitation
        ln -sfn "${MART_LINK_TARGET}" data/mart/imitation
      fi
      exit 1
    fi
    # dvc push: stage 単位で remote にアップロード。3 回までリトライ。
    # `-j 1` で multipart upload の並列度を 1 に下げる: DVC + s3fs の既知 bug
    # (treeverse/dvc#10374) で、高並列 + 高 RTT remote (ap-northeast-1) では
    # multipart upload の future が timeout なしでハングする現象を回避する。
    # `-v` で進捗を onstart.log に出して運用観測性を確保。
    # 観測: 2026-05-04 case7 試行 #6 で `dvc push <stage>` (default -j) が
    # 20+ 分 hang → terminate 余儀なくされ A100 で $0.88 浪費。
    # iter11: `uv run` 経由を排除して uv race condition を回避。
    push_ok=0
    for attempt in 1 2 3; do
      echo "[onstart] dvc push -j 1 -v ${PREPROCESS_STAGE} (attempt ${attempt}/3)"
      if ${DVC_BIN} push -j 1 -v "${PREPROCESS_STAGE}" 2>&1; then
        push_ok=1
        break
      fi
      echo "[onstart] dvc push attempt=${attempt} FAILED; retrying in 30s" >&2
      sleep 30
    done
    if [ "${push_ok}" -ne 1 ]; then
      echo "[onstart] dvc push ${PREPROCESS_STAGE} FATAL — exiting" >&2
      mark "55_mart_dvc_push_failed"
      exit 1
    fi

    # dvc.lock を git に commit & push (dvc push 成功時のみ)。これが pipeline
    # 管理下 outs の hash を保持する正本。`*.parquet.dvc` stub は使わない。
    git config user.email "runpod-bot@orbit-wars.local"
    git config user.name "runpod-bot"
    if [ -n "${GIT_PAT:-}" ]; then
      PUSH_URL="https://x-access-token:${GIT_PAT}@$(echo "<REPO_URL>" | sed 's|https://||')"
      git remote set-url origin "${PUSH_URL}" 2>&1 | tail -2
    fi
    TARGET_BRANCH="<BRANCH>"
    git fetch origin "${TARGET_BRANCH}" || true
    git checkout -B "${TARGET_BRANCH}" "origin/${TARGET_BRANCH}" || git checkout "${TARGET_BRANCH}" || true
    # dvc.lock は repo root にある。`.dvc` stub の whitelist と違い無条件で track 対象。
    git add -f dvc.lock 2>/dev/null || true
    if git diff --cached --quiet; then
      echo "[onstart] mart_dvc_persist: nothing to commit (dvc.lock unchanged)"
    else
      git commit --no-verify \
        -m ":robot: runpod: update dvc.lock for ${PREPROCESS_STAGE}" \
        || echo "[onstart] mart_dvc_persist: commit failed"
      for attempt in 1 2 3; do
        if git push origin "HEAD:${TARGET_BRANCH}"; then
          echo "[onstart] mart_dvc_persist: pushed to ${TARGET_BRANCH}"
          break
        fi
        echo "[onstart] mart_dvc_persist: attempt=${attempt} rejected; rebasing"
        git pull --rebase origin "${TARGET_BRANCH}" || true
        if [ "${attempt}" -eq 3 ]; then
          echo "[onstart] mart_dvc_persist: push exhausted retries"
        fi
      done
    fi
    mark "55_mart_dvc_persisted"
  fi

  # symlink を再構築して /persist の cache 利用を維持する。先に rsync で
  # data/mart/imitation/ の最新内容を /persist/data-mart-imitation/ に
  # 反映してから symlink に張り直す (--delete で stale entry も掃除)。
  if [ -n "${MART_LINK_TARGET}" ]; then
    echo "[onstart] mart_dvc_persist: re-linking data/mart/imitation -> ${MART_LINK_TARGET}"
    rsync -a --delete data/mart/imitation/ "${MART_LINK_TARGET}/" 2>&1 | tail -3 || true
    rm -rf data/mart/imitation
    ln -sfn "${MART_LINK_TARGET}" data/mart/imitation
  fi
fi

