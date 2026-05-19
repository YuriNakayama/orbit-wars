mark "20_clone_done"
# uv は冒頭で前倒し install 済みのはず。万一 PATH が落ちていれば再 export。
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "[onstart] step=install_uv (early install missed; fallback)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
fi

mark "30_before_uv_sync"
echo "[onstart] step=uv_sync"
# venv が persist 経由で残っているかを log に残す。lock 不変なら uv sync は
# 「何もしない」ので、この行は分単位の高速化指標として後段の解析に役立つ。
if [ -L bot/.venv ] && [ -f bot/.venv/pyvenv.cfg ]; then
  echo "[onstart] uv venv reuse: bot/.venv -> $(readlink bot/.venv) (pre-existing)"
else
  echo "[onstart] uv venv fresh: building bot/.venv from scratch"
fi
UV_SYNC_START=$(date +%s)

# Fast path: bot/uv.lock の hash が前回 sync 時と同じで、persisted venv
# (/persist/uv-venv-bot) に working python が居るなら uv sync を完全 skip。
# attempt 5 (38c84d9) で uv sync が 291s かかっていた根本原因は
# `Removed virtual environment` が走り全 wheel を install し直したこと。
# lock 不変条件下では venv をそのまま再利用するのが最速 (~数秒)。
LOCK_HASH=""
LOCK_HASH_FILE="/persist/uv-venv-bot/.uv_lock_sha256"
if [ -f bot/uv.lock ] && command -v sha256sum >/dev/null 2>&1; then
  LOCK_HASH=$(sha256sum bot/uv.lock | awk '{print $1}')
fi
SKIP_UV_SYNC=0
if [ -d /persist ] && [ -n "${LOCK_HASH}" ] \
   && [ -f "${LOCK_HASH_FILE}" ] \
   && [ "$(cat "${LOCK_HASH_FILE}" 2>/dev/null)" = "${LOCK_HASH}" ] \
   && [ -x bot/.venv/bin/python ]; then
  echo "[onstart] uv sync SKIP: lock hash unchanged (${LOCK_HASH:0:12}), venv intact"
  SKIP_UV_SYNC=1
fi

if [ "${SKIP_UV_SYNC}" -eq 0 ]; then
  echo "[onstart] uv sync RUN: lock hash=${LOCK_HASH:0:12} (file=${LOCK_HASH_FILE})"
  # iter13 fix (case9 retry 2026-05-06 11:33): broken venv detection.
  # bot/.venv (or its target) が中身は持つが bin/python が無い壊れた状態だと
  # uv sync が "not a valid Python environment" で 3 retry 全失敗する。
  # 復旧: bin/python が無ければ bot/.venv (symlink ごと) と persist 側を
  # 強制的に空に戻して fresh build に倒す。
  if [ ! -x bot/.venv/bin/python ]; then
    echo "[onstart] uv sync: detected broken venv (bin/python missing); resetting"
    rm -rf bot/.venv 2>/dev/null || true
    if [ -d /persist/uv-venv-bot ]; then
      find /persist/uv-venv-bot -mindepth 1 -delete 2>/dev/null \
        || rm -rf /persist/uv-venv-bot/* /persist/uv-venv-bot/.[!.]* 2>/dev/null \
        || true
      ln -sfn /persist/uv-venv-bot bot/.venv
    fi
  fi
  for attempt in 1 2 3; do
    # --frozen は --locked より厳格 (lock の依存解決を一切再計算しない)。
    # 既存 lock を信用するので smoke run のような短命 pod では正しい挙動。
    if uv sync --frozen --no-dev --directory bot; then
      break
    fi
    echo "[onstart] uv_sync attempt=${attempt} failed; retrying in 30s"
    sleep 30
    if [ "${attempt}" -eq 3 ]; then
      echo "[onstart] uv_sync exhausted retries"
      exit 1
    fi
  done
  # lock hash を /persist に書き出して次回 run 用にメモ。
  if [ -d /persist/uv-venv-bot ] && [ -n "${LOCK_HASH}" ]; then
    echo "${LOCK_HASH}" > "${LOCK_HASH_FILE}" 2>/dev/null \
      && echo "[onstart] uv lock hash recorded -> ${LOCK_HASH_FILE}" \
      || echo "[onstart] uv lock hash record FAILED (non-fatal)" >&2
  fi
fi
UV_SYNC_ELAPSED=$(( $(date +%s) - UV_SYNC_START ))
echo "[onstart] uv sync elapsed=${UV_SYNC_ELAPSED}s (skip=${SKIP_UV_SYNC})"

mark "40_uv_sync_done"

# iter11 fix: `uv run --project bot` / `uv run --directory bot` を以後の
# step (dvc / python) で**一切**使わない。`uv run` は呼ぶたびに
# pyproject.toml と uv.lock の整合性を再確認し、必要なら裏で `uv sync` を
# 走らせる仕様。同じ venv を 2 つの `uv run` がほぼ並行で触ると、片方の
# sync 中間状態を他方が踏んで `ModuleNotFoundError` (dvc / pathspec /
# diskcache / rich._emoji_codes / s3fs) が頻発する race condition を
# iter9 (`Installed 180 packages` retry chain) と iter10
# (`Uninstalled 25 packages + Installed 5` during dvc pull) で観測。
# 直接 .venv/bin の bin を呼べば uv は介在せず venv は line 319 の
# `uv sync --locked --no-dev` 状態のまま固定される。
PY_BIN="$(pwd)/bot/.venv/bin/python"
# iter12 fix (case9 連続失敗 2026-05-06 11:28): bot/.venv/bin/dvc shim の
# shebang が persisted volume の古い path を指して "bad interpreter" で死ぬ
# ケースを観測。 PY_BIN -m dvc で経由すれば shim を介さず実 dvc package を
# 起動できるので shebang 経路を回避できる。
DVC_BIN="${PY_BIN} -m dvc"
echo "[onstart] iter12 fix: pinned PY_BIN=${PY_BIN} DVC_BIN=${DVC_BIN}"

echo "[onstart] step=dvc_pull cwd=$(pwd) case=<CASE>"
# case0 は RunPod 基盤の E2E smoke 専用なので、dvc pull 経路の正常性検証だけ
# 行い、他 case の outs を巻き込まない (memory: runpod_5_traps)。
# data/lake/case0_smoke は数十バイトの sentinel ファイル 1 つだけ。
if [ "<CASE>" = "case0" ]; then
  echo "[onstart] dvc pull SCOPED to data/lake/case0_smoke (case0 smoke path)"
  ls -la data/lake/case0_smoke.dvc 2>&1 | head -3
  if ! ${DVC_BIN} pull data/lake/case0_smoke.dvc; then
    echo "[onstart] dvc pull (case0_smoke) FAILED" >&2
    ls -la data/lake/ 2>&1 | head -5
    git status --short 2>&1 | head -5
    mark "45_dvc_pull_case0_smoke_failed"
    exit 1
  fi
  echo "[onstart] case0_smoke contents:"
  ls -la data/lake/case0_smoke/ 2>&1 | head -5
else
  # mart-only path: preprocess を pod 側で走らせない設計の case (mart は
  # 事前 push 済) では kaggle_episodes (60GB+, 62k hive parquet files) の
  # pull は不要。skip して直接 case 別 mart targeted pull に進む。
  # PREPROCESS_CMD は instance.render_onstart が CASE_DEFAULTS の
  # preprocess_cmd を埋め込む。空文字なら "mart 事前 push 済" と判断。
  # 2026-05-18 case11 retry で /persist のキャッシュ不在 + 60GB pull が
  # hang して 40_uv_sync_done で stall 連発した trap への対処。
  if [ -z "<PREPROCESS_CMD>" ] || [[ "<PREPROCESS_CMD>" == *"parquet_to_npy"* ]]; then
    echo "[onstart] dvc pull SKIP kaggle_episodes (PREPROCESS_CMD empty; mart-only path)"
  else
    # 診断: cwd と repo の dvc-tracked 状態を log に残す
    echo "[onstart] dvc pull diagnostic:"
    ls -la data/lake/kaggle_episodes/matches.dvc 2>&1 | head -3
    # set -e は外しているので明示的に exit code を確認
    if ! ${DVC_BIN} pull data/lake/kaggle_episodes/matches.dvc; then
      echo "[onstart] dvc pull (kaggle_episodes) FAILED" >&2
      echo "[onstart] dvc pull diagnostic listing:"
      ls -la data/lake/kaggle_episodes/ 2>&1 | head -5
      ls -la 2>&1 | head -10
      git status --short 2>&1 | head -10
      git log --oneline -3 2>&1 | head -3
      mark "45_dvc_pull_kaggle_failed"
      exit 1
    fi
  fi
  # iter15 fix (case9 retry3 2026-05-06 12:00): `dvc pull --allow-missing` は
  # graph 解析で衝突 (case8 outs vs .dvc 重複) を検出すると case9 parquet を
  # "hash info not found" の WARNING で skip して exit 0 になる。 case9 parquet
  # を target 直接指定で pull すれば衝突を bypass できるので、 case 別に pull
  # を **--allow-missing の前** に 1 段追加する (--allow-missing 後に置くと
  # block 自体に到達しない事故が retry3 で発生)。
  CASE_SUBDIR=$(echo '<CASE>' | sed 's/_three_head$//;s/_candidate_ships$//;s/_candidate$//;s/_dual$//;s/_sweep_.*$//;s/_base_preprocess$//;s/_template_ships$//')
  CASE_MART_DIR="data/mart/imitation/${CASE_SUBDIR}"
  echo "[onstart] iter15 targeted pull: case='<CASE>' subdir='${CASE_SUBDIR}' dir='${CASE_MART_DIR}'"
  ls -la "${CASE_MART_DIR}/" 2>&1 | head -10
  if [ -f "${CASE_MART_DIR}/train.parquet.dvc" ] \
     && [ -f "${CASE_MART_DIR}/val.parquet.dvc" ]; then
    # iter19 (case10 2026-05-14): index.parquet.dvc が存在すれば併せて pull する。
    # case10 では side index (train_index.parquet / val_index.parquet) を train.py が
    # in-memory フィルタで参照するため、parquet 本体と同時に取得が必要。
    EXTRA_DVCS=()
    [ -f "${CASE_MART_DIR}/train_index.parquet.dvc" ] && EXTRA_DVCS+=("${CASE_MART_DIR}/train_index.parquet.dvc")
    [ -f "${CASE_MART_DIR}/val_index.parquet.dvc" ] && EXTRA_DVCS+=("${CASE_MART_DIR}/val_index.parquet.dvc")
    echo "[onstart] dvc pull TARGETED (iter15): ${CASE_MART_DIR}/{train,val}.parquet.dvc${EXTRA_DVCS[*]:+ + index pair}"
    ${DVC_BIN} pull --force \
      "${CASE_MART_DIR}/train.parquet.dvc" \
      "${CASE_MART_DIR}/val.parquet.dvc" \
      "${EXTRA_DVCS[@]}" 2>&1 | tail -30
    echo "[onstart] iter15 targeted pull exit=$?"
  else
    echo "[onstart] iter15 targeted pull SKIP: case .dvc files not found" >&2
  fi
  # --allow-missing: dvc.yaml stage の outs が S3 にまだ無い (= 未学習の case の
  # weights.pt などが orphan stub として残っている) ケースで FAIL せず WARNING に
  # 降格させる。
  # --force: /persist volume に前回 preprocess の残骸 parquet が unsaved file として
  # 残っている場合に dvc pull が "Can't remove unsaved files" で fail するのを回避。
  # 残骸自体は preprocess skip ロジックで再利用される (新規 preprocess も上書き可能)。
  # memory `project_runpod_5_traps_2026_05_04.md` および
  # `project_runpod_onstart_pitfalls.md` 参照。
  #
  # 2026-05-18 case11 retry7: mart-only path (PREPROCESS_CMD="") では
  # targeted pull で必要な mart は取得済 + kaggle_episodes も不要なので、
  # この full pull は graph 解析で 60GB+ の outs を fetch しようとして
  # hang する。SKIP して targeted pull の結果を尊重する。
  if [ -z "<PREPROCESS_CMD>" ] || [[ "<PREPROCESS_CMD>" == *"parquet_to_npy"* ]]; then
    echo "[onstart] dvc pull --allow-missing SKIP (PREPROCESS_CMD empty; targeted pull complete)"
  else
    if ! ${DVC_BIN} pull --allow-missing --force; then
      echo "[onstart] dvc pull (full) FAILED" >&2
      mark "45_dvc_pull_full_failed"
      exit 1
    fi
  fi
  # iter17 fix (case9 retry5 2026-05-06 12:12): `dvc pull --allow-missing --force`
  # が graph 整合のため targeted pull で取得した case 別 parquet を削除して
  # しまうケースを観測 (`D data/mart/imitation/case9/train.parquet`)。
  # --allow-missing 後に case 別 mart parquet を**再 pull**して復活させる。
  #
  # 2026-05-19 case11 retry trap: PREPROCESS_CMD empty path では full pull
  # を SKIP しているので「`--allow-missing` が消した」前提が成立しない。
  # しかし MFS の遅延コミットで file 存在 check が false negative になり、
  # 不要な re-pull が hang する。empty path では iter17 をスキップする。
  if [ -n "<PREPROCESS_CMD>" ] && [[ "<PREPROCESS_CMD>" != *"parquet_to_npy"* ]] \
     && [ -f "${CASE_MART_DIR}/train.parquet.dvc" ] \
     && [ -f "${CASE_MART_DIR}/val.parquet.dvc" ]; then
    NEED_REPULL=0
    [ ! -f "${CASE_MART_DIR}/train.parquet" ] && NEED_REPULL=1
    [ ! -f "${CASE_MART_DIR}/val.parquet" ] && NEED_REPULL=1
    [ -f "${CASE_MART_DIR}/train_index.parquet.dvc" ] && [ ! -f "${CASE_MART_DIR}/train_index.parquet" ] && NEED_REPULL=1
    [ -f "${CASE_MART_DIR}/val_index.parquet.dvc" ] && [ ! -f "${CASE_MART_DIR}/val_index.parquet" ] && NEED_REPULL=1
    if [ "${NEED_REPULL}" = "1" ]; then
      echo "[onstart] iter17 re-pull: parquet (or index) was deleted by --allow-missing"
      EXTRA_DVCS=()
      [ -f "${CASE_MART_DIR}/train_index.parquet.dvc" ] && EXTRA_DVCS+=("${CASE_MART_DIR}/train_index.parquet.dvc")
      [ -f "${CASE_MART_DIR}/val_index.parquet.dvc" ] && EXTRA_DVCS+=("${CASE_MART_DIR}/val_index.parquet.dvc")
      ${DVC_BIN} pull --force \
        "${CASE_MART_DIR}/train.parquet.dvc" \
        "${CASE_MART_DIR}/val.parquet.dvc" \
        "${EXTRA_DVCS[@]}" 2>&1 | tail -10
      echo "[onstart] iter17 re-pull exit=$?"
    fi
  fi
fi

mark "50_dvc_pull_done"
echo "[onstart] step=mkdir_run"
# 絶対 path で固定: train.py は ORBIT_WARS_RUN_DIR を Path.resolve() するので
# 相対 path だと cwd (uv run --directory bot) 基準になり bot/data/... に
# ずれる。後段の dvc add / S3 upload とも整合させるため絶対 path で持つ。
RUN_DIR_ABS="$(pwd)/data/output/models/imitation/<CASE>/runs/<RUN_ID>"
mkdir -p "${RUN_DIR_ABS}"

# iter4 fix: preprocess の前に mart symlink を物理 dir に materialize する。
# 後段 (mart_dvc_persist) で materialize するパターンだと、preprocess が
# /persist 配下に書き込んだ parquet が cp -RL でコピー後に glob で見えない
# 事象を A6000 host で観測 (case8 iter4 1st run の `65_train_failed_exit_1`)。
# 順序を「materialize → preprocess」に変えれば preprocess は materialize 済みの
# 実 dir に直接書き込むため symlink chain を経由せず安全。
MART_PARENT_PRE="data/mart/imitation"
if [ -L "${MART_PARENT_PRE}" ]; then
  echo "[onstart] (pre-preprocess) mart parent ${MART_PARENT_PRE} is symlink — materialize"
  MART_TARGET_PRE="$(readlink -f "${MART_PARENT_PRE}")"
  rm "${MART_PARENT_PRE}"
  mkdir -p "${MART_PARENT_PRE}"
  if [ -d "${MART_TARGET_PRE}" ]; then
    cp -RL "${MART_TARGET_PRE}"/. "${MART_PARENT_PRE}/" 2>&1 | tail -3 || true
  fi
fi

