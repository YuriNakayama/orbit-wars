# Command Execution Rules

Conventions for running scripts and tooling in this repository. Prefer the wrappers under `dev/` over invoking `uv` / `dvc` / package managers directly: the wrappers `cd` into `bot/` and pin the right interpreter, so the same command works from any worktree.

## Top-level Commands

```bash
dev/setup             # Install dependencies (uv sync)
dev/format            # Code formatting (ruff)
dev/lint              # Static analysis (ruff + mypy)
dev/test-bot          # CI (format check → lint → type check → pytest)
dev/create-worktree   # Create git worktree with .env copy
dev/dvc               # DVC operations (setup / pull / repro / push / dag / add)
dev/vast              # Vast.ai GPU pod control (train / pull / promote / cost-report / volume)
dev/runpod            # RunPod GPU pod control (train / pull / promote / cost-report / volume)
dev/kaggle            # Kaggle Notebook free-tier GPU (train / pull / promote / dataset / cost-report)
```

## Long-running Training Checkpoint Policy

長時間学習 (1h 以上想定の RunPod / Kaggle / Vast 学習) は **iter ごとに best.pt 等の中間成果物を S3 (or DVC remote) に即 upload** すること。Kaggle Kernel は完走時のみ `/kaggle/working` を output コミットするため、timeout / ERROR で中間 weights が破棄される (RunPod は preempt で消える)。`ORBIT_WARS_BEST_S3_PREFIX` 等の env var を train script に渡し、新 best 更新ごとに `s3.upload_file()` を呼ぶ実装パターンを必ず採ること。新規学習基盤を追加する際もこの規約を満たしてから本番投入する。

## DVC Commands

```bash
dev/dvc setup                       # Configure local DVC (cache dir + AWS profile)
dev/dvc pull                        # Fetch real data from S3 remote
dev/dvc repro                       # Re-run pipeline on diffs
dev/dvc push                        # Upload artifacts to S3
dev/dvc dag                         # Stage dependency graph
dev/dvc add <path>                  # Track a path with DVC
dev/dvc <subcommand> [args...]      # Pass-through to `dvc <subcommand>`
```

`data/lake/selfplay/matches/` (selfplay runner output) and `data/lake/kaggle_episodes/matches/` (Kaggle scraper output) are tracked at the directory level via `dev/dvc add`. When the selfplay run produces new history, either pass `--dvc-add` to update automatically, or run `dev/dvc add data/lake/selfplay/matches` → `git add *.dvc` → `dev/dvc push` manually.

**Concurrent execution across multiple worktrees is discouraged**: the DVC cache at `/Users/user/project/orbit-wars/.dvc/cache` is shared between worktrees, so simultaneously running `dev/dvc repro` / `dev/dvc pull` / `dev/dvc add` may cause lock contention.

## Vast.ai GPU Training

```bash
# 1) commit & push, then launch on Vast
git push origin <branch>
dev/vast train <commit-sha> [--stage train_imitation_case1]

# 2) once finished, fetch locally
dev/vast pull <run_id>

# 3) if adopted, promote to canonical weights
dev/vast promote <run_id>

# Cost check
dev/vast cost-report --month 2026-04
```

Candidate weights are saved to `data/output/models/imitation/case1/runs/<run_id>/best.pt` and managed via DVC/S3. `policy/weights.pt` (the canonical Kaggle submit weights) is updated only when `dev/vast promote` runs. `VAST_API_KEY` is recorded in `bot/.env`. See [`docs/plans/vast-ai-basis/`](../../docs/plans/vast-ai-basis/) for details.

## RunPod GPU Training

RunPod 基盤は Vast.ai と並走するもう一つの GPU プロバイダ。Secure Cloud (T3/T4 DC + network volume 可) と Community Cloud (P2P, 安価) の 2 系統を `--cloud-type` で選べる。

```bash
# 1) commit & push, then launch on RunPod
git push origin <branch>
dev/runpod train <commit-sha> [--case case1] [--cloud-type SECURE|COMMUNITY|ALL]

# 2) once finished, fetch locally
dev/runpod pull <run_id> [--case case1]

# 3) if adopted, promote to canonical weights
dev/runpod promote <run_id> [--case case1] [--eval-results PATH]

# Cost check (RunPod 専用、vast とは別ファイル)
dev/runpod cost-report --month 2026-05

# Network volume 管理 (Secure Cloud 専用、Pod 作成時のみ attach 可能)
dev/runpod volume list
dev/runpod volume search [--data-center-id US-KS-2]
dev/runpod volume create <name> --data-center-id US-KS-2 [--size 15]

# 進捗確認 / 完了監視
dev/runpod ps                         # 起動中 pod 一覧 (launch.json と突合)
dev/runpod status <run_id>            # 単一 run の pod state + S3 marker + DVC 状況
dev/runpod summary <run_id>           # status / cost / metrics / artifacts を 1 画面集約

# ライブ tail (pod RUNNING 中のみ、SSH 経由、永続化なし)
dev/runpod tail <run_id> --source onstart  # /var/log/onstart.log を tail -F
dev/runpod tail <run_id> --source train    # 学習プロセス stdout のみ
dev/runpod tail <run_id> --source gpu      # nvidia-smi 10s サンプル

# 永続化済ログ (terminate 後でも S3 経由で参照可)
dev/runpod logs <run_id>              # S3 progress marker を timestamp 順に表示
dev/runpod logs <run_id> --source onstart  # /var/log/onstart.log 全文 (run_dir or S3 fallback)
dev/runpod logs <run_id> --tail 5     # 末尾のみ
dev/runpod logs <run_id> --grep done  # 行フィルタ

dev/runpod watch <run_id>             # 既存 pod の終了まで poll → 完了/失敗で desktop 通知

# 成果物取得 (DVC 失敗時の S3 fallback あり)
dev/runpod pull <run_id>              # auto: DVC → 失敗時 S3 artifacts へ自動切替
dev/runpod pull <run_id> --from s3    # 強制 S3 artifacts 経由
dev/runpod pull <run_id> --from dvc   # 強制 DVC 経由 (fallback なし)

# `dev/runpod train --watch` で起動と同時に監視も開始可能 (推奨)。
# 終了通知は macOS osascript / Linux notify-send / fallback stdout。
# 観測性の詳細は docs/plans/runpod-basis/06_observability.md
```

Vast.ai 基盤と同じ `data/output/models/imitation/case<N>/runs/<run_id>/` に成果物を保存し、DVC/S3 remote も共有。run.json には provider 別フィールド (`vast_*` / `runpod_*`) が記録され、両基盤の run を区別可能。`RUNPOD_API_KEY` は `backend/.env` に置き、key は <https://runpod.io/console/user/settings> で発行。デフォルト cost limit は $1.5/run (Vast の $1.0 より高め)。詳細は [`docs/plans/runpod-basis/`](../../docs/plans/runpod-basis/)。

両基盤の使い分け方針は [`docs/plans/runpod-basis/README.md`](../../docs/plans/runpod-basis/README.md) の「Vast.ai 基盤との使い分け」表を参照。

### Interactive Mode (dev / debug pods)

`dev/runpod dev` は **インタラクティブモード** で pod を確保し、`sleep infinity` で
保持する。auto-cleanup / 8h timeout guard / `trap cleanup_destroy EXIT` を全て無効化
するので、SSH 接続でコード変更・再実行・デバッグを繰り返せる。終了は明示的に
`dev/runpod destroy <run_id>` で行う必要がある (放置すると課金が止まらない)。

```bash
# 起動 (commit を origin に push 済みであること)
dev/runpod dev <commit-sha> [--case caseN] [--cloud-type SECURE|COMMUNITY|ALL]

# 状態確認 (50_interactive_ready が出れば SSH 接続可)
dev/runpod status <run_id> --case caseN

# SSH 接続 (proxy=ssh.runpod.io 既定、direct=TCP/22 公開 port も可)
dev/runpod ssh <run_id> [--case caseN] [--via proxy|direct] [--key PATH] [--exec "<cmd>"]

# コード同期 (rsync 経由、bot/ のみ。.venv / data / __pycache__ 等は exclude)
dev/runpod sync <run_id> [--case caseN] --push [--dry-run] [--delete]
dev/runpod sync <run_id> [--case caseN] --pull

# 明示的に terminate (interactive モードでは必須)
dev/runpod destroy <run_id> [--case caseN] [-y]
```

oneshot モード (`dev/runpod train`) と interactive モード (`dev/runpod dev`) の比較や
proxy SSH 用 key の登録方法は [`docs/plans/runpod-basis/07_interactive_mode.md`](../../docs/plans/runpod-basis/07_interactive_mode.md)
を参照。`dev/runpod ps` は interactive pod を黄色で表示し destroy リマインダを出す。

## Kaggle Kernel GPU Training (Free Tier)

Vast.ai / RunPod と並ぶ第三の GPU 学習基盤。Kaggle Notebooks (Save & Run All のバッチ実行) の **無料 GPU 枠 (T4x2 / P100、週 30h)** を利用してコスト 0 で学習を回す。9h GPU 上限 / ~5 同時 kernel 上限あり、長時間 RL には不向きだが imitation の小規模 case 向け。

```bash
# 0) (初回のみ) Kaggle API key を bot/.env に追加
#    https://www.kaggle.com/settings → Create New API Token で kaggle.json を取得し
#    KAGGLE_USERNAME=<your-username> と KAGGLE_KEY=<your-key> を bot/.env に追記

# 1) (初回のみ) bot/ を Kaggle Dataset として upload
dev/kaggle dataset push --commit-sha "$(git rev-parse HEAD)"

# 2) commit & push, then launch on Kaggle
git push origin <branch>
dev/kaggle train "$(git rev-parse HEAD)" --case case1 --accelerator gpu-t4x2 --watch
#   → bot/ snapshot を dataset の新 version として push
#   → notebook 自動生成 → kernel push
#   → --watch で QUEUED → RUNNING → COMPLETE / ERROR まで polling

# 3) 成果物 pull (kaggle kernels output → ローカル dvc add)
dev/kaggle pull <run_id> --case case1

# 4) 採用なら canonical weights に昇格 (vast/runpod と同等)
dev/kaggle promote <run_id> --case case1 [--eval-results PATH]

# 5) 月次 free GPU 時間レポート (金額は 0)
dev/kaggle cost-report --month 2026-05

# 進捗確認 / ログ
dev/kaggle ps                            # active kernel 一覧
dev/kaggle status <run_id>               # 単一 run の launch + kernel status + run.json
dev/kaggle watch <run_id>                # 終了まで polling、完了で desktop 通知
dev/kaggle logs <run_id> [--tail N]      # 完了済 kernel の train.log (要事前 pull)

# Dataset 管理
dev/kaggle dataset push --label "<note>"  # commit SHA を version_notes に記録
dev/kaggle dataset status                 # 現在の dataset の processing 状態
```

Vast.ai / RunPod と同じ `data/output/models/imitation/case<N>/runs/<run_id>/` に成果物を保存し、DVC/S3 remote も共有。`run.json` の `kaggle_kernel_meta` field で kaggle 経由かを区別可能。`KAGGLE_USERNAME` / `KAGGLE_KEY` は `bot/.env` に置き、key は <https://www.kaggle.com/settings> で発行。

Kaggle Kernel は **学習用** であり、Kaggle competition への submit kernel ではない (submit は `dev/submit` の責務)。詳細は [`docs/plans/kaggle-kernel-basis/`](../../docs/plans/kaggle-kernel-basis/)。

### Interactive Mode (Kaggle Notebook を sleep loop で常駐 + S3 command channel)

Kaggle には SSH がないため、`dev/kaggle dev` で **S3 を双方向 channel として使う sleep-loop notebook** を push する。Claude (local) が S3 inbox にコマンドを put すると、kernel が拾って実行し outbox に結果を書き戻す。RunPod `dev/ssh/sync/destroy` と機能的に等価。**kernel が ERROR / OOM で死亡しても S3 に heartbeat + 直前の outbox が残るため、SSH なしで死亡直前の状況が掴める** のが核心。

```bash
# 1) interactive kernel 起動 (RunPod dev 相当)
git push origin <branch>
dev/kaggle dev "$(git rev-parse HEAD)" --case case1 --accelerator gpu-t4x2

# 2) heartbeat 確認 (state=ready / running / idle / shutdown / voluntary_exit)
dev/kaggle info <run_id>

# 3) 任意 bash を kernel 上で実行 (S3 経由、ssh-exec 相当)
dev/kaggle exec <run_id> -- python -c "import torch; print(torch.cuda.is_available())"
dev/kaggle exec <run_id> --cwd /tmp/orbit-wars-repo/bot -- pytest tests/unit/

# 4) ローカル file を kernel に転送 (rsync push 相当)
dev/kaggle sync <run_id> --file bot/pipeline/imitation/case1/training/train.py

# 5) 明示的に終了 (Kaggle 側は次の Quota cycle で自動停止、即時停止は Web UI から)
dev/kaggle destroy <run_id> -y
```

制約: Internet ON 必須 (S3 アクセスのため、submit kernel には流用不可)、AWS creds は kernel 側にも必要 (`bot/.env` を dataset に同梱 or Kaggle Secrets 登録)、Kaggle 9h 上限超過で kernel は強制停止 (`--max-idle-minutes` で voluntary exit を早めることが推奨)。

## Kaggle Submission Policy

Any real remote submission (`uv run python -m submit submit`, `dev/submit`, `kaggle competitions submit`, the `cd-kaggle-submit.yml` workflow_dispatch) is irreversible and consumes the daily 5-submission quota (note: `SubmissionStatus.ERROR` does NOT count against the quota — validation failures can be retried immediately). Always obtain explicit user approval immediately before executing, showing the case / message / mode to be submitted. Dry-run, archive build, and read-only history checks do NOT require approval. Prior approval covers only that single submission and does not extend to later submissions or auto-mode / autonomous loops.

## Direct `uv` / `dvc` Usage

Only fall back to running `uv run --directory bot ...` or `dvc ...` directly when no wrapper covers the case. If the same direct invocation appears more than once, add it as a subcommand under the relevant `dev/` script instead of duplicating it across docs.
