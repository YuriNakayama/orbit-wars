# vast-ai-basis — Requirements Definition

## Background and Purpose

`backend/pipeline/imitation/case1/` の DeepSets 模倣学習は CPU でも数十分で完走するが、今後の改善イテレーション（hyperparameter sweep / 大型モデル / RL 派生）では学習時間が支配的になりつつある。GitHub Actions を介すと CI 環境の制約・実行時間制限・反復速度の遅さがネックになる一方、Vast.ai であれば 数百 USD/月 の従量で RTX3090 GPU を on-demand に確保でき、ローカルからの直接起動で「PR commit → GPU 学習 → 結果評価」のループを 30分単位に短縮できる。

本 feature は **Vast.ai を使い捨て GPU 計算ノードとして扱い、学習成果物の正本は Git + DVC/S3 に置く** ためのローカル起動基盤を整備する。`dev/vast-train <commit-sha>` というシングルコマンドで「Vast 上で `dvc repro <stage>` を走らせ、生成物を DVC/S3 にプッシュ」が完結し、ローカルでは `dev/vast-pull <run_id>` で結果を取得 → 評価 → 採用なら `dev/promote-weights <run_id>` で `policy/weights.pt` を昇格、という流れに統一する。

副次目的として、(a) `imitation/case1` だけでなく将来の RL / 他 case の学習にも使える汎用基盤にする、(b) どの run が誰のどの commit から生まれたかを `run.json` で完全に追跡し、(c) 1 run 当たりのクラウドコストを `docs/experiment/` に月次集計できるよう可視化する。

## User Stories

- As a **developer**, I want to run `dev/vast-train <commit-sha>` from my laptop and have the GPU training start on Vast.ai automatically, so that I don't need to maintain a CI workflow or local GPU.
- As a **developer**, I want the CLI to show me the cheapest 10 GPU offers and let me pick one, so that I can balance cost and availability per run.
- As a **developer**, I want each run's outputs (best.pt + metrics.json + run.json) saved to a unique directory under `artifacts/models/imitation/case1/runs/<run_id>/`, so that experiments don't overwrite each other.
- As a **developer**, I want to run `dev/vast-pull <run_id>` locally to get the artifacts via DVC, so that I can evaluate without re-pulling the whole DVC tree.
- As a **developer**, I want `dev/promote-weights <run_id>` to copy `runs/<run_id>/best.pt` to `pipeline/imitation/case1/policy/weights.pt` and `dvc commit` it, so that adoption is one explicit step that cannot happen by accident.
- As a **researcher**, I want `run.json` to record git SHA / branch / params hash / seed / Vast instance id / GPU type / training command / training metrics / local eval results / status, so that I can reproduce or audit any past run.
- As a **researcher**, I want to extend the same basis to `dvc repro <any_stage>` (preprocess / train / eval / future RL stages) so the basis is reusable beyond imitation/case1.
- As a **cost-conscious developer**, I want a soft warning when the estimated cost (dph_total × estimated runtime) exceeds 1.0 USD per run, so that runaway training doesn't surprise me.
- As an **operator**, I want a monthly cost report aggregator that scans `runs/*/run.json` and produces a markdown summary in `docs/experiment/`, so that I can review usage at a glance.
- As a **developer**, I want the Vast instance to **self-destroy** after `dvc push` succeeds, so that idle instances don't accrue cost.

## Functional Requirements

### F1. CLI: `dev/vast-train <commit-sha> [--stage <name>] [--seed <n>] [--label <text>]`

1. F1.1: `<commit-sha>` を必須引数で受け取り、ローカル git で存在確認 + origin に push 済みかチェック（未 push なら fail fast）。
2. F1.2: `--stage` オプション（デフォルト `train_imitation_case1`）で任意の `dvc.yaml` stage を指定可能。
3. F1.3: vastai SDK 経由で `search_offers` を実行し、フィルタ `gpu_name in [RTX_3090, RTX_4090, RTX_A6000, A100], num_gpus=1, reliability >= 0.99, cuda_max_good >= 12, type=on-demand, dph_total < 1.0` で `dph_total asc` ソートし、上位 10 件を rich table で表示。
4. F1.4: 表示後、stdin で番号入力 → 対応する `offer_id` を確定。
5. F1.5: 推定コスト (`dph_total * estimated_runtime_minutes / 60`) を表示。1 USD を超える場合は再確認プロンプト。
6. F1.6: `run_id` を `<YYYYMMDD-HHMMSS>__<branch_slug>__<sha7>__seed<seed>` で生成。
7. F1.7: `backend/src/vast/onstart.sh` をテンプレートとして読み、placeholder（`<COMMIT_SHA>`, `<RUN_ID>`, `<STAGE>`, `<BRANCH>`, `<REPO_URL>`）を sed 置換した tmp ファイルを生成。
8. F1.8: AWS credentials (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`) と `VAST_API_KEY` を `--env` で注入。
9. F1.9: `create_instance(offer_id, image=<pytorch:cuda12.4>, disk=40, label=<run_id>, ssh=True, direct=True, onstart=<tmp_file>, env=<env_string>)` を呼ぶ。
10. F1.10: 起動後、instance id を表示し、`vastai logs <id>` 監視のためのコマンドをユーザーに案内。
11. F1.11: 部分的な再現性確保のため、commit-sha が dirty / unpushed の場合は警告して終了。

### F2. Onstart script: `backend/src/vast/onstart.sh`

1. F2.1: `set -euo pipefail` で堅牢化。`trap` で異常終了時にも self-destroy が呼ばれることを保証。
2. F2.2: `env >> /etc/environment` を冒頭で実行（SSH デバッグ時に env が見える）。
3. F2.3: `git clone <REPO_URL> /workspace/orbit-wars && cd /workspace/orbit-wars && git checkout <COMMIT_SHA>`。
4. F2.4: `curl -LsSf https://astral.sh/uv/install.sh | sh` で uv をインストール。
5. F2.5: `uv sync --locked --all-extras --dev --directory backend` で依存解決（lockfile based）。
6. F2.6: `uv run --directory backend dvc remote modify --local s3 profile default` で AWS profile を default に切替（env 経由 credentials を使うため）。
7. F2.7: `uv run --directory backend dvc pull` で deps を取得。
8. F2.8: `mkdir -p artifacts/models/imitation/case1/runs/<RUN_ID>` で run dir 作成。
9. F2.9: `ORBIT_WARS_RUN_DIR=artifacts/models/imitation/case1/runs/<RUN_ID> uv run --directory backend dvc repro <STAGE>` で学習を実行。`train.py` 側はこの env を読んで `weights_out` を override（後述 F4）。
10. F2.10: 学習完了後、`uv run --directory backend dvc push artifacts/models/imitation/case1/runs/<RUN_ID>` で S3 にプッシュ。
11. F2.11: 最後に `vastai destroy instance "$VAST_INSTANCE_ID"` で自分自身を破壊（`VAST_API_KEY` は env 経由）。
12. F2.12: 各ステップの開始/終了を `echo "[onstart] step=... status=ok|fail"` で stdout に出力。

### F3. Run metadata: `run.json` schema

1. F3.1: `train.py` 完了時、`<run_dir>/run.json` を以下のフィールドで生成:
   - `run_id` (str), `git_sha` (str), `git_branch` (str), `params_hash` (str, sha256[:12]), `seed` (int)
   - `vast_instance_id` (int|null, env 経由), `gpu_name` (str|null), `vast_offer_snapshot` (object|null)
   - `command` (str), `weights_path` (str, relative to repo root)
   - `train_metrics` (object: epochs_run, best_epoch, best_val_loss, train_loss_history, val_loss_history)
   - `local_eval_results` (object|null) — 後でローカル評価 CLI が追記
   - `status` (enum: `running` / `pushed` / `evaluated` / `adopted` / `failed`)
   - `created_at` (ISO8601), `updated_at` (ISO8601)
2. F3.2: `metrics.json` も別途同 dir に保存（DVC metrics 互換、cache: false 相当）。中身は `train_metrics` の subset で良い。
3. F3.3: 既存 `docs/experiment/imitation_case1_iter*.json` の schema は変更しない（後方互換、移行は段階的）。

### F4. `train.py` の改修

1. F4.1: `_seed_all(seed)` に `torch.cuda.manual_seed_all(seed)` を追加。
2. F4.2: `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` を取得し、`model.to(device)` と batch tensor の `.to(device)` 注入。`pin_memory=True` (loader)、`non_blocking=True` (transfer)。
3. F4.3: `_run_epoch` 内の loss/acc 計算は変更なし（DataParallel など並列は不要、1 GPU 想定）。
4. F4.4: 環境変数 `ORBIT_WARS_RUN_DIR` が設定されていれば、`weights_out` を `<run_dir>/best.pt` に置き換え、追加で `metrics.json` と `run.json` を `<run_dir>/` に保存。`ORBIT_WARS_RUN_DIR` が無ければ既存挙動（`policy/weights.pt` への直接書き出し）を維持。
5. F4.5: `TrainReport` に `train_loss_history`, `val_loss_history`, `device` フィールドを追加。
6. F4.6: GPU 推論への path は今回のスコープ外（policy/agent.py は CPU 推論のまま、Kaggle Sandbox は CPU 環境のため）。

### F5. CLI: `dev/vast-pull <run_id>`

1. F5.1: `artifacts/models/imitation/case1/runs/<run_id>/` を `dvc pull <path>` でローカルに取得。
2. F5.2: pull 後、`run.json` を表示（cat）し、`status` が `pushed` 以外なら警告。

### F6. CLI: `dev/promote-weights <run_id> [--message <text>]`

1. F6.1: `artifacts/models/imitation/case1/runs/<run_id>/best.pt` の存在確認。
2. F6.2: `cp <run_dir>/best.pt backend/pipeline/imitation/case1/policy/weights.pt`。
3. F6.3: `uv run --directory backend dvc commit pipeline/imitation/case1/policy/weights.pt`（既に DVC 管理されているので `add` ではなく `commit`）。
4. F6.4: `git status` を表示し、ユーザーに git commit を促す（自動 commit は行わない、メッセージは人間が確認）。
5. F6.5: `run.json` の `status` を `adopted` に更新し再 push（`dvc add` した run dir 全体）。

### F7. Cost Aggregator: `dev/vast-cost-report [--month <YYYY-MM>]`

1. F7.1: `artifacts/models/imitation/case1/runs/*/run.json` を全走査。
2. F7.2: 月単位（デフォルト当月）で `gpu_name`, `vast_offer_snapshot.dph_total`, `train_metrics.runtime_seconds` を集計。
3. F7.3: 推定コスト合計 / run 数 / 採用 run 数 を表で出力し、`docs/experiment/vast_cost_report_<YYYY-MM>.md` に保存。

### F8. Configuration: 環境変数とローカル設定

1. F8.1: `~/.vast/config.toml` 等を新設せず、既存の `.env`（リポジトリ内）か `~/.aws/credentials` の延長で完結させる。
2. F8.2: `VAST_API_KEY` は `~/.env` か `backend/.env` から読み取り（Python `python-dotenv` 経由、既存依存）。
3. F8.3: AWS は `~/.aws/credentials` の `orbit-wars` profile を **そのまま** 使用（Vast 側に渡すのは access key id/secret のみ、profile 名は Vast 側で `default` に切替）。
4. F8.4: 新規 IAM user は作らず、既存 `orbit-wars-dev-dvc-user` のキーを Vast に渡す（リスク受容、漏洩時は Terraform で revoke）。

## Non-Functional Requirements

### NFR-1. 性能 / レスポンス

- `dev/vast-train` 実行から create_instance 完了までの体感 < 30 秒（search offers 5s + create 10-15s + 表示等）
- onstart 終了までの実時間 < 30 分（git clone 1m + uv sync 3-5m + dvc pull 1-2m + train 10-15m + dvc push 2-3m + destroy 数秒）
- 1 run の典型コスト < 0.30 USD（RTX 3090 @ $0.13/h × 30 min × 余裕係数 1.5 = 約 $0.10）

### NFR-2. セキュリティ

- AWS access key / VAST_API_KEY は `--env` 経由で Vast コンテナにのみ渡し、git にコミットしない（`.gitignore` で `.env` 確認済み）。
- DVC IAM ポリシーは S3 の `s3:DeleteObject` を含まないため、Vast 側の bug でも S3 オブジェクトを破壊できない（既存設計の継承）。
- `vastai destroy instance` の self-destroy は Vast 側 API key を knowing する必要があるため、漏洩時の影響範囲は **個人の Vast アカウントの delete 権限のみ**（クレジット盗用は可能だが、課金は個人）。
- `run.json` には credentials を一切含めない（`vast_offer_snapshot` は public な offer 情報のみ）。

### NFR-3. 可用性 / 障害耐性

- Vast インスタンス起動失敗（offer が他者に取られた等）は再試行不要、エラーメッセージを返して `dev/vast-train` 終了。ユーザーが再実行で別 offer 選択。
- onstart 中の `git clone` / `uv sync` / `dvc pull` 失敗は `set -e` で即時中断 → trap が destroy 実行 → ローカルでは `vastai logs <id>` で原因特定。
- `dvc push` 失敗時は **destroy しない**（成果物が消える）。trap で `status=failed` を `run.json` に書いてから人間判断を待つ。

### NFR-4. 拡張性

- `--stage` 引数で任意 stage を受け取る設計のため、将来 `train_rl_caseN` などを追加する際は dvc.yaml に stage を追加するだけで `dev/vast-train` は無改修。
- `train.py` の `ORBIT_WARS_RUN_DIR` env override パターンは他 case の train script にも横展開可能なシンプルな仕様。
- GPU フィルタは `dph_total < 1.0` を初期上限とし、将来 RL 用 A100 など必要なら CLI フラグで上書き可能（後の改修）。

## Out of Scope

- **Multi-GPU 学習 (DDP)**: DeepSets MLP は単 GPU で十分。将来必要になったら別 feature で。
- **インクリメンタル学習 / checkpoint resume**: interruptible インスタンスを使わないため不要。
- **Hyperparameter sweep の自動化**: 1 commit = 1 run の原則を維持。将来 `dev/vast-sweep` の別 feature で対応。
- **GitHub Actions による自動トリガ**: 本 feature の主旨はローカル起動。CI 化は将来検討。
- **Vast 専用 IAM user の新設**: 既存 `orbit-wars-dev-dvc-user` を流用、リスク受容。
- **GPU 推論 (Kaggle submit 改修)**: Kaggle Sandbox は CPU 環境前提。`policy/agent.py` は CPU 推論を維持。
- **既存 `weights_iter*.pt` (git 管理) の DVC 移行**: 段階移行とし、新 run のみ DVC 経路。

## Glossary

| Term | Description |
|------|-------------|
| `run_id` | `<YYYYMMDD-HHMMSS>__<branch_slug>__<sha7>__seed<N>` 形式の 1 学習実行を一意特定する文字列 |
| `run dir` | `artifacts/models/imitation/case1/runs/<run_id>/`。1 run の全成果物を含むディレクトリ |
| canonical weights | `backend/pipeline/imitation/case1/policy/weights.pt`。Kaggle submit 時の正本 |
| candidate weights | `<run_dir>/best.pt`。採用前の学習成果物 |
| onstart | Vast.ai インスタンス起動時に自動実行されるシェルスクリプト |
| self-destroy | onstart 末尾で `vastai destroy instance` を呼んでインスタンスを破壊する仕組み |
| promote | candidate weights を canonical weights にコピーして採用する操作 |
