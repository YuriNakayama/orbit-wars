echo "[onstart] step=env_persist"
env >> /etc/environment

# Persistent network volume (mount path is supplied by RunPod create_pod).
# When a network volume is attached at /persist we symlink:
#   uv cache, DVC cache, data/lake (raw episodes), data/mart (preprocessed).
# data/lake は dvc pull で取得する大型 raw データ (kaggle_episodes 等) で、
# 学習データ追加時に再ダウンロードを避けるため volume に永続化する。
# When the volume is absent we fall back to ephemeral disk. mkdir/ln are idempotent.
if [ -d /persist ]; then
  echo "[onstart] step=persist_setup mount=/persist"
  mkdir -p /persist/uv-cache /persist/dvc-cache /persist/data-mart /persist/data-lake
  mkdir -p "$HOME/.cache"
  rm -rf "$HOME/.cache/uv" 2>/dev/null || true
  ln -sfn /persist/uv-cache "$HOME/.cache/uv"
  export UV_CACHE_DIR="$HOME/.cache/uv"
  echo "[onstart] persist: uv cache -> /persist/uv-cache, data-lake/data-mart 用意済み"
else
  echo "[onstart] persist: /persist not mounted (no volume) — using ephemeral disk"
fi

mark "10_before_clone"
echo "[onstart] step=clone repo=<REPO_URL> sha=<COMMIT_SHA>"
mkdir -p /workspace
cd /workspace
if [ ! -d orbit-wars ]; then
  if [ -n "${GIT_PAT:-}" ]; then
    git clone "https://x-access-token:${GIT_PAT}@$(echo "<REPO_URL>" | sed 's|https://||')" orbit-wars
  else
    git clone "<REPO_URL>" orbit-wars
  fi
fi
cd orbit-wars
git fetch origin "<COMMIT_SHA>" || true
git checkout "<COMMIT_SHA>"

if [ -d /persist ]; then
  echo "[onstart] step=persist_link_repo"
  mkdir -p .dvc
  rm -rf .dvc/cache 2>/dev/null || true
  ln -sfn /persist/dvc-cache .dvc/cache
  # bot/.venv を /persist/uv-venv-bot に symlink。uv は venv の
  # python 絶対 path を pyvenv.cfg に焼き込むので、persist 側で venv を作る
  # ことで再起動後も再利用できる。lock 不変なら uv sync は no-op に近い。
  # DVC では管理しない (依存情報は uv.lock が source of truth)。
  mkdir -p /persist/uv-venv-bot
  # attempt 6 (commit 9b49d74) で `uv sync` が以下のエラーで失敗した:
  #   error: failed to remove directory `/workspace/orbit-wars/bot/.venv/lib`:
  #   Directory not empty (os error 39)
  # 原因: lock hash mismatch を検出した uv が venv を rebuild しようとして
  # rmdir したが、network volume 上で rmdir が空 dir を空でないと報告した。
  # 対策: lock hash mismatch を検出したら、symlink 化の前に persisted venv
  # の中身を完全クリアしてしまう (uv の rmdir に頼らない)。
  PERSIST_LOCK_HASH_FILE="/persist/uv-venv-bot/.uv_lock_sha256"
  CURRENT_LOCK_HASH=""
  if [ -f bot/uv.lock ] && command -v sha256sum >/dev/null 2>&1; then
    CURRENT_LOCK_HASH=$(sha256sum bot/uv.lock | awk '{print $1}')
  fi
  if [ -n "${CURRENT_LOCK_HASH}" ] \
     && [ -d /persist/uv-venv-bot ] \
     && { [ ! -f "${PERSIST_LOCK_HASH_FILE}" ] \
          || [ "$(cat "${PERSIST_LOCK_HASH_FILE}" 2>/dev/null)" != "${CURRENT_LOCK_HASH}" ]; }; then
    # bot/.venv/bin/python が現存していれば中身は意味がある可能性があるが、
    # uv が rebuild すると判断するので予防的に消す。
    if [ -e /persist/uv-venv-bot/bin/python ] \
       || [ -d /persist/uv-venv-bot/lib ]; then
      echo "[onstart] persisted venv lock hash mismatch — clearing /persist/uv-venv-bot/*"
      # find -delete で network volume の rmdir 不整合を回避
      find /persist/uv-venv-bot -mindepth 1 -delete 2>/dev/null \
        || rm -rf /persist/uv-venv-bot/* /persist/uv-venv-bot/.[!.]* 2>/dev/null \
        || true
    fi
  fi
  if [ -L bot/.venv ]; then
    : # already linked
  elif [ -d bot/.venv ]; then
    # 既存 venv (古い ephemeral 由来) は破棄して symlink に置換
    rm -rf bot/.venv
    ln -sfn /persist/uv-venv-bot bot/.venv
  else
    ln -sfn /persist/uv-venv-bot bot/.venv
  fi
  # data/ は git tree から復元される (各種 .dvc stub を含む)。
  # rm -rf data/lake / data/mart は git-tracked .dvc ファイルを破壊するので
  # ディレクトリ全体ではなく、dvc が pull で書き出す末端 dir のみ symlink にする。
  # 2026-05 以降 replays は S3 直接、DVC 管理は matches/index.parquet のみ。
  # matches/ dir はローカル selfplay / preprocess の作業領域として symlink 永続化する。
  # data/lake/{kaggle_episodes,selfplay} と data/mart の親 dir のみ作る
  # (末端 dir は symlink にする想定なので mkdir しない)。
  mkdir -p data/lake/kaggle_episodes data/lake/selfplay data/mart
  mkdir -p /persist/lake-kaggle-matches /persist/lake-selfplay-matches \
           /persist/data-mart-imitation
  # data/lake/<src>/matches だけを symlink で persist 化 (raw episodes の本体)
  if [ ! -L data/lake/kaggle_episodes/matches ] && [ ! -d data/lake/kaggle_episodes/matches ]; then
    ln -sfn /persist/lake-kaggle-matches data/lake/kaggle_episodes/matches
  fi
  if [ ! -L data/lake/selfplay/matches ] && [ ! -d data/lake/selfplay/matches ]; then
    ln -sfn /persist/lake-selfplay-matches data/lake/selfplay/matches
  fi
  # data/mart/imitation 全体を /persist/data-mart-imitation に symlink。
  # この時点で git tree からは復元されていない (新規 case の場合) か、
  # 復元されていたとしても .dvc stub のみなので symlink で問題ない。
  # 既に dir として実体がある (例: 過去 run の遺物) 場合は中身を /persist に
  # 引っ越してから symlink (再起動で持続化される ⇒ 次回以降 dvc pull が冪等)。
  #
  # 重要: 過去 run の preprocess 中途破損 parquet が /persist に残ると新 run の
  # train が `parquet: File must end with PAR1` で死亡する (memory
  # `project_runpod_5_traps_2026_05_04` の trap #7)。冒頭で /persist 側を
  # 完全クリアし、毎 run preprocess を fresh で走らせる方が信頼性が高い。
  rm -rf /persist/data-mart-imitation
  mkdir -p /persist/data-mart-imitation
  if [ -L data/mart/imitation ]; then
    : # already linked
  elif [ -d data/mart/imitation ]; then
    # iter16 fix (case9 retry4 2026-05-06 12:04): git clone で復元される
    # case 別の .dvc ファイル (`data/mart/imitation/<case>/{train,val}.parquet.dvc`)
    # を rm -rf で消す前に /persist 側へ copy 退避する。 過去 cases (case3/4/8)
    # は preprocess を pod 内部で走らせるので .dvc を不要にしていたが、 case9 は
    # preprocess を local で済ませ、 mart parquet は dvc pull で取りにいく形。
    # そのため .dvc ファイルが symlink target に存在することが必須。
    cp -an data/mart/imitation/. /persist/data-mart-imitation/ 2>/dev/null \
      || echo "[onstart] mart .dvc copy failed (non-fatal)" >&2
    rm -rf data/mart/imitation
    ln -sfn /persist/data-mart-imitation data/mart/imitation
  else
    ln -sfn /persist/data-mart-imitation data/mart/imitation
  fi

  # data/output/models/<CASE_FAMILY> も /persist に持続化。
  # train artifact の正本は DVC + S3 (`step=dvc_add_run` で push 済み) だが、
  # 学習中に pod が落ちた場合の checkpoint 救出や、複数 run の train.log /
  # gpu.log を残しておきたいケースに有用。サイズ管理は cost-report で観測。
  mkdir -p data/output/models "/persist/data-output-models-<CASE_FAMILY>"
  if [ -L "data/output/models/<CASE_FAMILY>" ]; then
    : # already linked
  elif [ -d "data/output/models/<CASE_FAMILY>" ]; then
    cp -an "data/output/models/<CASE_FAMILY>/." "/persist/data-output-models-<CASE_FAMILY>/" \
      2>/dev/null || true
    rm -rf "data/output/models/<CASE_FAMILY>"
    ln -sfn "/persist/data-output-models-<CASE_FAMILY>" "data/output/models/<CASE_FAMILY>"
  else
    ln -sfn "/persist/data-output-models-<CASE_FAMILY>" "data/output/models/<CASE_FAMILY>"
  fi
fi

