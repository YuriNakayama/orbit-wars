# vast-ai-basis — Web Technical Research

## Official Documentation

### Vast.ai Python SDK / CLI（公式）

- **パッケージ**: `pip install vastai` (PyPI)。リポジトリは [vast-ai/vast-cli](https://github.com/vast-ai/vast-cli) に統合済み（旧 `vast-ai/vast-sdk` は deprecated だが `from vastai_sdk import VastAI` の互換 import は維持）。
- **バージョン**: 2026-04 時点で活発に更新（latest 2026-04-14）。Python は 3.9+ が想定（明示記載なし、`vastai` CLI は単体バイナリ的に動く）。
- **API キー**: `https://cloud.vast.ai/manage-keys/` で発行。`VAST_API_KEY` 環境変数または `VastAI(api_key=...)` で渡す。
- **主要コマンド** ([Vast CLI Commands](https://docs.vast.ai/cli/commands)):
  - `vastai search offers '<query>' [--type on-demand|bid] [--order ...] [--raw]` — JSON 出力可。
  - `vastai create instance <offer_id> --image <docker> --disk <GB> --label <name> --ssh --direct --env '...' --onstart-cmd '...' --raw`
  - `vastai show instances [--raw]`、`vastai stop instance <id>`、`vastai destroy instance <id>`、`vastai logs <id>`
- **search offers の主要フィルタ** ([Search API](https://docs.vast.ai/api-reference/search/search-offers)):
  - `gpu_name` (`{"in": ["RTX_3090", "RTX_4090"]}`)、`num_gpus`、`gpu_ram` (MB)、`cuda_max_good`、`disk_space`、`dph_total` (USD/hour)、`reliability` (0..1)、`inet_down` (MB/s)、`geolocation`、`rentable`、`verified`
  - sort は `[["dph_total", "asc"]]` 等
- **onstart の渡し方** (両系統サポート):
  - `--onstart <file>` — ローカルの shell script ファイルパス
  - `--onstart-cmd '<inline>'` — 単一文字列のインライン script（quoting に注意）
- **環境変数の SSH/onstart 共有** ([SSH instance env]):
  - `--env '-e KEY=VAL ...'` で Docker entrypoint 環境変数に注入される。
  - **SSH 起動モードでは onstart 内では見えるが SSH session には見えない** → onstart の最後に `env >> /etc/environment` を入れて永続化するのが定石。
  - 機密値は **アカウントレベル env** (Vast.ai 管理画面) に置く方法もあるが、PR 単位の commit-sha 駆動には不向き → ローカルから `--env` で渡す方が再現性が高い。
- **Docker image vs SSH-only**: `--image pytorch/pytorch:2.6.0-cuda12.4-runtime` のような OCI 互換 image を `--ssh --direct` と組み合わせるのが現代的。`--ssh` 無しだと onstart 終了後にコンテナが exit して instance 状態が `stopped` になる（手動デバッグ不可）。
- **コスト管理**: 残高 0 で **stopped** 扱い（destroy ではない）→ ストレージ課金は継続。学習完了後は **明示的に `vastai destroy instance`** で破壊するのが必須。
- **GPU 価格帯** (2026-04 [vast.ai pricing](https://vast.ai/pricing)):
  - RTX 3090: on-demand $0.13/h〜、interruptible $0.10〜
  - RTX 4090: on-demand $0.29/h〜、interruptible $0.10〜
  - imitation/case1 の現状学習 (CPU で数十分想定の DeepSets MLP) は RTX 3090 1枚で十分以上、interruptible は中断リスクあり ⇒ **on-demand RTX 3090 1 枚推奨**。

### DVC + Cloud GPU Worker パターン

- **公式推奨パターン**:
  - 学習サーバ側で `git checkout <sha>` → `dvc pull` で deps を取得 → `dvc repro <stage>` → `dvc push` でアウトプットを remote へ。
  - DVC remote は git 同様に SSO key と独立した IAM 認証で動くため、ephemeral worker でも `~/.aws/credentials` または env (`AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`) で完結する。
  - `dvc.lock` の更新（新しい outs hash）はワーカー側で発生 → ワーカー側で `git commit` して push するか、ワーカー側では git commit せず `dvc push` だけ実行し、ローカルで `dvc pull` 後に手動 commit する 2 つの流派がある。
- **DVC の lightweight model registry 機能** ([artifacts get](https://dvc.org/doc/command-reference/artifacts/get)):
  - `dvc.yaml` に `artifacts:` ブロックを追加すると `dvc artifacts get` で git tag / branch 名を指定して任意 commit の成果物を pull 可能。
  - 本プロジェクトは正本 weights 1 個 + 候補 N 個という規模なので、DVC artifacts API までは使わず **`dvc.yaml` の outs ディレクトリと git SHA + run.json で運用** で十分。

## Similar OSS Projects

### Project 1: vast-ai/vast-cli — [github.com/vast-ai/vast-cli](https://github.com/vast-ai/vast-cli)

- **Relevance**: 公式 CLI / SDK 本体。`vast.py` 内の関数群を直接読める。
- **Approach**: 単一の `vast.py` (5000+ 行) に全コマンドを実装。Python SDK は同じ関数群を class methods として薄く wrap。
- **Reusable patterns**:
  - **`--raw` フラグで JSON を吐く設計** → bash thin wrapper から `jq` でパースしたり、Python から呼ぶ場合は `subprocess.run(..., capture_output=True)` で取得しやすい。
  - **search offers → create instance の二段呼び出しが基本フロー** → 我々の `dev/vast-train` も同じ二段に。
- **Pitfalls found**: `--onstart-cmd` の quoting に弱く、複雑な script は `--onstart <file>` で渡したほうが確実（CLI の issue tracker で頻出）。複数行 onstart は **必ずファイル経由** が無難。

### Project 2: jjziets/vasttools — [github.com/jjziets/vasttools](https://github.com/jjziets/vasttools)

- **Relevance**: コミュニティ製の Vast.ai 周辺ツール集。idle shutdown / cost monitor の実装例あり。
- **Approach**: bash + python の混在。`vastai show instances --raw | jq` で状態取得 → 条件マッチで destroy という idle watcher を bash で実装。
- **Reusable patterns**:
  - **idle watcher**: cron で 5 分おきに `vastai show instances --raw` → `gpu_util_percent < 5` & `runtime > N min` なら destroy。我々の場合は **学習スクリプト末尾に `vastai destroy instance $VAST_INSTANCE_ID` を仕込む** 方がシンプル（onstart 終了 = 学習完了）。
- **Pitfalls found**: API rate limit (1 req/sec 程度の体感) があり、watcher を 30s 間隔で動かすと 429 を踏みやすい → 60s+ 間隔推奨。

### Project 3: skypilot-org/skypilot との連携 (参考) — [vast.ai blog](https://vast.ai/article/vast-ai-gpus-can-now-be-rentend-through-skypilot)

- **Relevance**: SkyPilot は YAML 1 つで GPU クラウド (AWS/GCP/Vast/RunPod) を抽象化。
- **Approach**: `sky launch task.yaml` で provisioning + sync + run を一括実行。ファイル sync は rsync over SSH。
- **Reusable patterns**: 「**ファイル sync 派**（rsync）」 vs 「**git checkout 派**（onstart 内で clone）」 の比較が明確。本 feature は git/DVC で正本管理するため後者が自然。
- **Why NOT use SkyPilot directly**: 抽象レイヤの追加コスト（YAML 1 個のために skypilot 依存を増やす）、SkyPilot の vast.ai 統合は比較的新しく安定性に懸念、本リポジトリの DVC + git 中心ワークフローと整合性が低い → **直接 vastai SDK/CLI を叩く** ほうが軽量。

### Pattern Comparison

| Aspect | Our Project | vast-cli | vasttools | SkyPilot |
|--------|-------------|----------|-----------|----------|
| Code transfer | `git clone <repo> && git checkout <sha>` (onstart 内) | 同上 (推奨) | rsync ベースの例 | rsync (workdir) |
| Artifact 管理 | DVC + S3 (既存) | 言及なし | 言及なし | cloud-native (S3/GCS) |
| Instance teardown | onstart 末尾で self-destroy | 推奨 | watcher | 自動 (idle / done) |
| Secrets 注入 | `--env '-e AWS_*=...'` + onstart で /etc/environment | 同 | 同 | YAML envs |
| 依存追加 | `vastai` のみ | – | bash + jq | skypilot + 多数 |
| Recommendation | ⭐ vast-cli + git/DVC + onstart self-destroy | – | watcher 部分のみ参考 | スコープ外 |

## Library/Service Selection

### Vast.ai 操作レイヤ

| Candidate | Pros | Cons | Maintenance | Recommendation |
|-----------|------|------|-------------|----------------|
| ⭐ `vastai` Python SDK (`from vastai import VastAI`) | 公式、型ヒント有り、Python から直接 dict 受け取り | 依存追加、Python 3.9+ | 活発 (2026-04 active) | **推奨**。`backend/src/vast/` から `VastAI()` クラスを使用 |
| `vastai` CLI を subprocess で呼ぶ | SDK バージョン乖離リスクなし | quoting が脆弱、JSON parse を自前で | 同 | 補助。ログ閲覧 (`vastai logs`) のみ subprocess で十分 |
| HTTP API + `requests` 直叩き | 依存最小 | バージョン追従コスト、認証ヘッダ手書き | – | 不採用。SDK で済む |

💡 推薦理由: 本プロジェクトはすでに pyproject.toml で Python 中心の依存管理。`vastai` SDK は単一パッケージで CLI も含むため、CLI/SDK 両方を 1 依存で得られる。

### Onstart script の管理

| Candidate | Pros | Cons | Recommendation |
|-----------|------|------|----------------|
| ⭐ shell script ファイル (`backend/src/vast/onstart.sh`) を `--onstart` で渡す | 行数を気にせず書ける、git で diff 追跡可能、quoting 問題回避 | tiny script でもファイルが増える | **推奨**。リポジトリにコミットし可読性を確保 |
| `--onstart-cmd '<inline>'` | wrapper 側でテンプレ化（commit-sha を sed 埋め込み等） | quoting fragile、長い script で破綻 | 不採用 |
| Jinja2 等でテンプレート → 一時ファイル化 | テンプレ変数の差し込みが構造化 | 依存追加、複雑性 | 補助。commit-sha 等のごく少数の変数は `sed` か bash heredoc で十分 |

💡 推薦理由: vast-cli リポジトリの issue でも quoting trap が頻出。ファイル化で確実性を取り、commit-sha など最小限の変数だけ wrapper 側で sed 置換。

### Run metadata schema (run.json)

| Field | Type | Purpose |
|-------|------|---------|
| `run_id` | string | `<YYYYMMDD-HHMMSS>__<branch_slug>__<sha7>__seed<N>` |
| `git_sha` | string (40) | Vast 側で `git rev-parse HEAD` |
| `git_branch` | string | feature/exp branch name |
| `params_hash` | string (sha256, 12 chars) | `params.yaml` の hash |
| `seed` | int | `params.yaml: seed` の解決値 |
| `vast_instance_id` | int | `vastai show instances` 由来 |
| `vast_offer_snapshot` | object | `gpu_name`, `dph_total`, `geolocation` |
| `gpu_name` | string | `RTX_3090` 等 |
| `command` | string | 実行した `dvc repro` コマンド全文 |
| `weights_path` | string | `artifacts/models/imitation/case1/runs/<run_id>/best.pt` |
| `train_metrics` | object | `epochs_run`, `best_val_loss`, `best_epoch` 等 |
| `local_eval_results` | object | `wins/losses/draws/win_rate` (ローカル評価後に追記) |
| `status` | enum | `running` / `pushed` / `evaluated` / `adopted` / `failed` |
| `created_at` / `updated_at` | ISO8601 | – |

## API/Protocol Research

- **Vast.ai REST API base**: `https://console.vast.ai/api/v0/`。SDK が薄く wrap しているのでアプリコードでは直接叩かない。
- **Authentication**: `Bearer` token (API key)。CLI/SDK 共通。
- **Instance lifecycle**:
  1. `search_offers(...)` → 候補 offers の id 一覧
  2. `create_instances(id, image, disk, onstart, env, label, ssh=True)` → instance id 返却
  3. ステータス `creating → loading → running` を `show_instances` でポーリング (15-30s 間隔)
  4. SSH ログイン or `vastai logs <id>` で進捗確認
  5. onstart の最終ステップ自身で `vastai destroy instance` を呼ぶ（API key を Vast 側に渡す必要あり → `--env` で `VAST_API_KEY` を注入）
  6. ローカルで `dvc pull` → 評価
- **Failure modes**:
  - host disconnection (reliability < 1 で稀に発生) → `dvc push` 前に消失、再実行が必要。`reliability >= 0.99` で絞る。
  - onstart 中の docker pull 失敗 → instance は loading で stuck → `vastai destroy` で手動回復。

## Research Summary

### Key findings that impact design

1. **Vast SDK は 1 個で十分**: `vastai` パッケージのみ追加すれば CLI と Python SDK 両方が手に入る。`backend/src/vast/` を新設して `VastAI()` を import。
2. **onstart はファイル化**: `backend/src/vast/onstart.sh` を git 管理。wrapper 側で `<COMMIT_SHA>` などのプレースホルダを sed で置換 → tmp ファイル → `--onstart`。
3. **Self-destroy が標準**: onstart の最終ステップで `vastai destroy instance "$VAST_INSTANCE_ID"` を呼ぶ。失敗パスでも必ず destroy するため `trap` を仕込む。
4. **環境変数は二段渡し**: `--env '-e AWS_ACCESS_KEY_ID=... -e AWS_SECRET_ACCESS_KEY=... -e AWS_DEFAULT_REGION=ap-northeast-1 -e VAST_API_KEY=... -e ORBIT_WARS_RUN_ID=... -e ORBIT_WARS_GIT_SHA=...'` で渡し、onstart 冒頭で `env >> /etc/environment` を実行（SSH デバッグ時に見える）。
5. **GPU 選定**: RTX 3090 1 枚（on-demand、$0.13/h〜、reliability ≥ 0.99、cuda_max_good ≥ 12）が DeepSets MLP には十分。`gpu_name={"in": ["RTX_3090", "RTX_4090"]}` で柔軟性を持たせるのが安全。
6. **Run metadata は run.json（git untracked）+ DVC 管理**: 1 run = 1 ディレクトリ `artifacts/models/imitation/case1/runs/<run_id>/{best.pt, metrics.json, run.json}`。**ディレクトリ全体を DVC stage out として管理** すれば `dvc push` で S3 同期。
7. **採用フロー**: `dev/promote-weights <run_id>` の bash thin wrapper で run dir → `policy/weights.pt` を `cp` し、`dvc add policy/weights.pt`（既に DVC 管理なので `dvc commit`）→ git commit + push → main merge。
8. **dvc.yaml の改修方針 (要 Step 5 で議論)**:
   - 案 A: `train_imitation_case1` の outs を runs ディレクトリ全体に変更（既存 `weights.pt` 経路を破壊）
   - 案 B: 新 stage `train_imitation_case1_run` を追加し、既存 stage は維持。Vast 側は新 stage を repro
   - 案 C: DVC stage 外で `python -m pipeline.imitation.case1.training.train` を直接実行し、出力先を CLI 引数で run dir に切替。dvc.yaml は触らない
   - 推奨は **案 C**（後述、Step 5 で詳細化）。dvc.yaml を壊さず、run.json/metrics.json も train script の責務に統合可能。

### Patterns adopted from external research

- **`vastai search offers --raw | jq` フィルタ後 create instance** ([vast-cli](https://github.com/vast-ai/vast-cli) パターン)
- **onstart 末尾で self-destroy** ([vasttools](https://github.com/jjziets/vasttools) パターンの簡略版)
- **`env >> /etc/environment`** ([Vast docs SSH env best practice](https://docs.vast.ai/cli/commands))
- **git SHA + DVC ディレクトリ outs での model registry**（[DVC artifacts](https://dvc.org/doc/command-reference/artifacts/get) の lightweight 派生）

### Recommended approach

`dev/vast-train <commit-sha>` (bash thin wrapper) → `backend/src/vast/cli.py` (typer + vastai SDK) → search offers → create instance with `--onstart backend/src/vast/onstart.sh` (sed 置換 tmp) and AWS/VAST_API_KEY env → onstart 内で repo clone + uv sync + dvc pull + 学習 (run dir に書き出し) + dvc push + self-destroy → ローカル `dev/vast-pull <run_id>` で `dvc pull artifacts/.../runs/<run_id>/` → 評価スクリプトで `replay_one_match` 等を回し `docs/experiment/<run_id>.md` に記録 → 採用なら `dev/promote-weights <run_id>` で `policy/weights.pt` 上書き + git commit + PR。

## Sources

- [Vast.ai Python SDK Quickstart](https://docs.vast.ai/sdk/python/quickstart)
- [Vast.ai CLI Commands](https://docs.vast.ai/cli/commands)
- [Vast.ai search-offers API](https://docs.vast.ai/api-reference/search/search-offers)
- [Vast.ai create-instance API](https://docs.vast.ai/api/create-instance)
- [Vast.ai Instances FAQ](https://docs.vast.ai/documentation/reference/faq/instances)
- [Vast.ai pricing 2026](https://vast.ai/pricing)
- [vast-ai/vast-cli (GitHub)](https://github.com/vast-ai/vast-cli)
- [jjziets/vasttools (GitHub)](https://github.com/jjziets/vasttools)
- [DVC Pipelines (公式)](https://doc.dvc.org/start/data-pipelines/data-pipelines)
- [DVC artifacts API](https://dvc.org/doc/command-reference/artifacts/get)
- [DVC + Cloud GPU CI/CD](https://doc.dvc.org/use-cases/ci-cd-for-machine-learning)
