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
# 2026-05-20: venv on /persist (MFS network volume) repeatedly produced
# ImportError (libgomp .so unloadable; pyarrow missing despite install
# log showing 265 packages installed). Build venv on container-local
# disk instead. Cost: ~30-45s `uv sync` per pod (vs ~5s SKIP), but
# eliminates the dominant failure class.
#
# Remove any stale /persist symlink left by older onstart scripts so
# uv creates a real container-local directory.
if [ -L bot/.venv ]; then
  echo "[onstart] removing stale /persist symlink bot/.venv -> $(readlink bot/.venv)"
  rm -f bot/.venv
fi
UV_SYNC_START=$(date +%s)
for attempt in 1 2 3; do
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
UV_SYNC_ELAPSED=$(( $(date +%s) - UV_SYNC_START ))
echo "[onstart] uv sync elapsed=${UV_SYNC_ELAPSED}s (container-local)"
# Sanity check: site-packages actually usable. Catches partial installs
# or future MFS-style breakage early so 50/60 don't have to retry.
if ! bot/.venv/bin/python -c "import pyarrow, torch, numpy, sklearn" 2>&1; then
  echo "[onstart] FATAL: venv import smoke failed post uv sync" >&2
  exit 1
fi

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

# reinforce family は on-policy で env.step() を回す唯一の case 群なので、
# Rust 製シミュレータ `orbit_wars_rust` (PyO3 + maturin) の Linux .so を
# 学習開始前に build する。imitation case は parquet-based BC で env を
# 一切使わないため build 不要だった (rust .so は worktree の darwin 版が
# git untracked で残っているのみ)。
# Pre-check the venv before any secondary `uv sync --group <X>` call.
# The original detection at line ~47 only runs before the FIRST sync;
# subsequent group syncs can hit a poisoned persisted venv from a
# previous failed bench run and die with
# "Project virtual environment directory bot/.venv cannot be used
#  because it is not a valid Python environment (no Python executable
#  was found)" (observed on RunPod EU-RO-1 pods 20260520-102638 and
# 20260520-103216 back-to-back). The reset logic is identical to the
# first-sync block but lives in a function so we can call it before
# both `--group env` and `--group cuda`.
check_and_reset_broken_venv() {
  if [ ! -x bot/.venv/bin/python ]; then
    echo "[onstart] secondary uv sync: detected broken venv (bin/python missing); resetting"
    rm -rf bot/.venv 2>/dev/null || true
    if [ -d /persist/uv-venv-bot ]; then
      find /persist/uv-venv-bot -mindepth 1 -delete 2>/dev/null \
        || rm -rf /persist/uv-venv-bot/* /persist/uv-venv-bot/.[!.]* 2>/dev/null \
        || true
      ln -sfn /persist/uv-venv-bot bot/.venv
    fi
    # Re-run the base sync so the env is rebuilt before we add a group.
    echo "[onstart] secondary uv sync: rebuilding base env"
    if ! ( cd bot && uv sync --frozen --no-dev ); then
      echo "[onstart] secondary uv sync: base rebuild FAILED" >&2
      return 1
    fi
  fi
  return 0
}

if [ "<CASE_FAMILY>" = "reinforce" ]; then
  echo "[onstart] step=build_rust_sim (reinforce family のみ)"
  # reinforce は on-policy で env.step() を回すため Rust 製シミュレータの
  # Linux .so が必要。bot の dependency-groups.env に maturin を入れて
  # `uv sync --group env` で入れる方が pip 経路を回避できて確実。
  # imitation case は parquet-based BC で env を使わないため env group を
  # 入れずに済む (default sync は --no-dev で env group も skip)。
  if ! check_and_reset_broken_venv; then
    mark "45_build_rust_sim_failed"
    exit 1
  fi
  echo "[onstart] build_rust_sim: installing maturin via uv sync --group env"
  # Capture full output to /tmp/uv_env_sync.log so we have at least a
  # post-mortem hint when the pod gets cleaned up (onstart.log は S3 へ
  # 上がる前に terminate されることがあり診断できないため)。
  UV_ENV_LOG="/tmp/uv_env_sync.log"
  if ! ( cd bot && uv sync --no-dev --group env --frozen ) >"${UV_ENV_LOG}" 2>&1; then
    echo "[onstart] build_rust_sim: maturin install FAILED — uv sync log tail:" >&2
    tail -30 "${UV_ENV_LOG}" >&2 || true
    # S3 へ即 upload。pod が消える前に保全。
    if command -v aws >/dev/null 2>&1; then
      aws s3 cp "${UV_ENV_LOG}" \
        "s3://orbit-wars-dvc-286854171013/remote/runpod_artifacts/<RUN_ID>/uv_env_sync.log" \
        2>&1 | head -3 || true
    fi
    mark "45_build_rust_sim_failed"
    exit 1
  fi
  echo "[onstart] uv sync --group env ok; checking maturin binary"
  if [ ! -x bot/.venv/bin/maturin ]; then
    echo "[onstart] build_rust_sim: bot/.venv/bin/maturin not present after sync" >&2
    ls -la bot/.venv/bin/ 2>&1 | grep -iE "maturin|^total" >&2 || true
    mark "45_build_rust_sim_failed"
    exit 1
  fi
  echo "[onstart] maturin $(bot/.venv/bin/maturin --version 2>&1 | head -1)"
  # Rust toolchain (cargo) の存在確認。RunPod base image に含まれていない場合
  # rustup で導入する。memory `project_runpod_5_traps` の trap には未記録。
  if ! command -v cargo >/dev/null 2>&1; then
    echo "[onstart] cargo not found — installing rust toolchain via rustup"
    if ! curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal 2>&1 | tail -10; then
      echo "[onstart] rustup install FAILED" >&2
      mark "45_build_rust_sim_failed"
      exit 1
    fi
    # rustup の env script を source して PATH を反映。
    if [ -f "$HOME/.cargo/env" ]; then
      # shellcheck disable=SC1091
      source "$HOME/.cargo/env"
    fi
  fi
  echo "[onstart] cargo $(cargo --version 2>&1 | head -1)"
  RUST_BUILD_START=$(date +%s)
  # maturin develop は VIRTUAL_ENV を読んで install 先を決める。
  # bot/.venv は cwd/bot/.venv にあるので絶対 path で明示 export する。
  BOT_VENV_ABS="$(pwd)/bot/.venv"
  if [ ! -d "${BOT_VENV_ABS}" ]; then
    echo "[onstart] build_rust_sim: bot/.venv missing at ${BOT_VENV_ABS}" >&2
    mark "45_build_rust_sim_failed"
    exit 1
  fi
  if ! ( cd simulator/rust && VIRTUAL_ENV="${BOT_VENV_ABS}" "${BOT_VENV_ABS}/bin/maturin" develop --release ) 2>&1 | tail -30; then
    echo "[onstart] build_rust_sim FAILED — orbit_wars_rust の build に失敗" >&2
    mark "45_build_rust_sim_failed"
    exit 1
  fi
  RUST_BUILD_ELAPSED=$(( $(date +%s) - RUST_BUILD_START ))
  echo "[onstart] build_rust_sim ok (elapsed=${RUST_BUILD_ELAPSED}s)"
  if ! "${PY_BIN}" -c "import sys; sys.path.insert(0, 'simulator/rust/python'); import orbit_wars_rust; print('orbit_wars_rust import OK')" 2>&1; then
    echo "[onstart] build_rust_sim verify FAILED" >&2
    mark "45_build_rust_sim_verify_failed"
    exit 1
  fi
fi

echo "[onstart] step=dvc_pull cwd=$(pwd) case=<CASE> family=<CASE_FAMILY>"
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
elif [ "<CASE_FAMILY>" = "reinforce" ]; then
  # reinforce family は on-policy rollout で学習データを生成するため、
  # kaggle_episodes mart の pull は不要。BC warm-start に使う重み (case9
  # per_planet best.pt) だけを targeted pull する。BC 重みのパスは config
  # YAML 内に hardcode されているので、配布されている .dvc を一括 pull する
  # 方式 (recursive) で `data/output/models/imitation/case9_per_planet/` 配下
  # の全 runs を取得する。
  BC_RUNS_PARENT="data/output/models/imitation/case9_per_planet/runs"
  echo "[onstart] dvc pull SCOPED to ${BC_RUNS_PARENT}/ (reinforce BC warm-start)"
  ls -la "${BC_RUNS_PARENT}/" 2>&1 | head -8
  # *.dvc は per-run-dir 単位で push 済み。`dvc pull <dvc>` で個別取得。
  # train.yaml は単一 run の best.pt を参照するので、find で全 .dvc を渡す。
  mapfile -t BC_DVCS < <(find "${BC_RUNS_PARENT}" -maxdepth 1 -name "*.dvc" 2>/dev/null)
  if [ ${#BC_DVCS[@]} -eq 0 ]; then
    echo "[onstart] reinforce BC pull: no .dvc files under ${BC_RUNS_PARENT}" >&2
    mark "45_dvc_pull_reinforce_bc_missing"
    exit 1
  fi
  echo "[onstart] reinforce BC pull: ${#BC_DVCS[@]} .dvc files to fetch"
  if ! ${DVC_BIN} pull -j 4 "${BC_DVCS[@]}" 2>&1 | tail -30; then
    echo "[onstart] dvc pull (reinforce BC) FAILED" >&2
    mark "45_dvc_pull_reinforce_bc_failed"
    exit 1
  fi
  echo "[onstart] reinforce BC pull complete; verifying best.pt..."
  find "${BC_RUNS_PARENT}" -name "best.pt" -maxdepth 3 2>&1 | head -5
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
  CASE_SUBDIR=$(echo '<CASE>' | sed 's/_three_head$//;s/_candidate_ships$//;s/_candidate$//;s/_dual$//;s/_sweep_.*$//;s/_base_preprocess$//;s/_template_ships$//;s/_smoke$//')
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
RUN_DIR_ABS="$(pwd)/data/output/models/<CASE_FAMILY>/<CASE_SUBDIR>/runs/<RUN_ID>"
mkdir -p "${RUN_DIR_ABS}"

# iter4 fix: preprocess の前に mart symlink を物理 dir に materialize する。
# 後段 (mart_dvc_persist) で materialize するパターンだと、preprocess が
# /persist 配下に書き込んだ parquet が cp -RL でコピー後に glob で見えない
# 事象を A6000 host で観測 (case8 iter4 1st run の `65_train_failed_exit_1`)。
# 順序を「materialize → preprocess」に変えれば preprocess は materialize 済みの
# 実 dir に直接書き込むため symlink chain を経由せず安全。
# reinforce family は on-policy で mart を使わないので skip。
if [ "<CASE_FAMILY>" != "reinforce" ]; then
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
fi

# CUDA12 jax plugin for GPU bench cases. Run AFTER dvc pull because
# `uv sync` is declarative — a `--group cuda` sync (even combined
# with `--group env`) prunes the `dvc[s3]` extra's transitive
# `s3fs` from the venv, breaking subsequent dvc operations. By
# deferring cuda install to after dvc pull, base+env state stays
# intact during dvc, and the train process gets a venv with both
# the rust simulator (env group) and cuda12 plugin (cuda group).
# Pod 20260520-110209 hit `ERROR: URL 's3://' is supported but
# requires these missing dependencies: ['s3fs']` when cuda sync
# ran before dvc.
case "<CASE>" in
  bench_*_gpu)
    echo "[onstart] step=install_cuda_jax (case=<CASE>): uv sync --group env --group cuda"
    if ! check_and_reset_broken_venv; then
      mark "47_install_cuda_jax_failed"
      exit 1
    fi
    UV_CUDA_LOG="/tmp/uv_cuda_sync.log"
    if ! ( cd bot && uv sync --no-dev --group env --group cuda --frozen ) >"${UV_CUDA_LOG}" 2>&1; then
      echo "[onstart] install_cuda_jax FAILED — uv sync log tail:" >&2
      tail -30 "${UV_CUDA_LOG}" >&2 || true
      if command -v aws >/dev/null 2>&1; then
        aws s3 cp "${UV_CUDA_LOG}" \
          "s3://orbit-wars-dvc-286854171013/remote/runpod_artifacts/<RUN_ID>/uv_cuda_sync.log" \
          2>&1 | head -3 || true
      fi
      mark "47_install_cuda_jax_failed"
      exit 1
    fi
    echo "[onstart] install_cuda_jax ok"
    mark "47_install_cuda_jax_done"
    ;;
esac

