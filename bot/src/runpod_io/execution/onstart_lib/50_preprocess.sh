PREPROCESS_RAN=0
if [ -n "<PREPROCESS_CMD>" ]; then
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

