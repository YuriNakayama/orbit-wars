# dvc-data-control — Requirements Definition

## 背景と目的

Kaggle Orbit Wars の深層学習エージェント (`pipeline/imitation/case1` 以降) は、
Kaggle replay / self-play replay を入力に preprocess → train → evaluation のパイプラインを
回すが、現状は **ファイル命名と手動運用だけ** で世代管理している。

過去の `memory/project_imitation_case1_phase2_breakthrough.md` / `project_imitation_case1_phase3_non_determinism.md` に
残るように、「勝率 5/100 が再評価で 0/300」等の非決定性が既に発生しており、
**「その時の preprocess output + weights + 評価コード」の完全な再現** が
必要な局面が増えている。

本プランはローカル環境のデータ・モデル管理を DVC + S3 remote 方式へ統一し、
次の 2 つを第一優先とする:

1. **実験再現性の保証** — 全 stage が `dvc.lock` でハッシュ固定され、`dvc repro` で同じ成果物を再生成できる。
2. **重み・データの世代管理** — 実験ブランチごとに候補モデル / 前処理 parquet を無損保存し、任意時点に戻れる。

## スコープ

- ローカル環境 (macOS Darwin) のみ。vast.ai 等の GPU サーバ側セットアップは **明示的に scope 外**（設計メモのみ 03-architecture.md に記載）。
- 既存 YAML/CLI の後方互換性は考慮しない。移行時に不要ファイルは都度削除する（検証プロジェクトのため）。

## User Stories

- As an ML 開発者, I want `dvc repro` を叩くだけで preprocess → train → eval が再実行できる, so that コードやパラメータ変更のたびに再現手順を書き起こさなくて済む。
- As an ML 開発者, I want 過去の `git commit` にチェックアウトしたら `dvc pull` で当時の学習データ・重みが揃う, so that 過去実験を完全再現して比較できる。
- As an ML 開発者, I want 実験ごとの候補重み (`weights.pt`) を S3 に溜める, so that ディスクが枯れず、手動リネーム運用から脱却できる。
- As an 提出オペレーター, I want Kaggle 提出時に `dvc pull` 1 発で最新重みを取得できる, so that tar.gz ビルドフローが壊れずに済む。
- As an インフラ担当, I want Terraform で bucket/IAM/policy を再現可能に定義する, so that 手動コンソール操作や権限の取り違いを避けられる。

## Functional Requirements

1. **DVC 初期化**: リポジトリに `.dvc/config`, `.dvcignore` を配置し、`data/` 配下を DVC 管理下に置く。
2. **S3 Remote**: `s3://orbit-wars-dvc-<account_id>/remote` を default remote として設定。認証は AWS profile (`.dvc/config.local` は gitignore)。
3. **共有 Cache**: メインリポ `/Users/user/project/orbit-wars/.dvc/cache` を `.dvc/config.local` で指定し、複数 worktree で共有。
4. **Pipeline 定義**: `dvc.yaml` に以下の stage を定義。
   - `scrape_kaggle` (always_changed=true 相当 / または frozen 選択可能)
   - `preprocess_imitation_case1`
   - `train_imitation_case1`
   - `eval_imitation_case1`
5. **Params ファイル**: リポジトリルートに `params.yaml` を新設し、既存 `configs/il_baseline.yaml` から移行する（旧ファイルは最終的に削除）。
6. **DVC 管理対象**:
   - `data/lake/kaggle_episodes/`, `data/lake/selfplay/`（writer stage の outs）
   - `data/mart/imitation/case1/*.parquet`（preprocess outs）
   - `data/processed/`（将来の中間成果物置き場として予約、現 stage では未使用）
   - `backend/pipeline/imitation/case1/policy/weights.pt`（train outs、DVC 化）
7. **Kaggle 提出フロー更新**: `dev/submit` / `backend/src/submit/` が `dvc pull <weights.pt>` を事前実行してから tar.gz ビルド。
8. **Terraform モジュール**: `infra/module/application/dvc_remote/` で以下を定義:
   - S3 bucket（versioning on, SSE AES256, public access block）
   - IAM policy (least privilege)
   - IAM user または role（1 AWS profile 対応分）
   - 出力: `bucket_name`, `bucket_arn`, `iam_access_key_id` (sensitive), `iam_secret_access_key` (sensitive)
9. **環境別ルート**: `infra/environment/dev/` から上記 module を呼び、`terraform plan` が通る状態まで仕上げる（apply は別途ユーザー承認）。
10. **ドキュメント更新**: `README.md` / `.claude/CLAUDE.md` / `backend/pipeline/imitation/case1/README.md` に DVC 運用手順を追記。

## Non-Functional Requirements

- **Performance**:
  - 初回 `dvc push`: 850 MB 上り、数分〜10 分程度を許容（個人回線で完結）。
  - 差分 `dvc push/pull`: 秒〜数十秒。ハッシュが同じなら実転送は発生しない。
- **Security**:
  - secret を git に commit しない（`.dvc/config.local`, `.tfvars` は .gitignore）。
  - IAM は least privilege（bucket ごと `s3:GetObject/PutObject/ListBucket`、Delete は管理者ロールのみ）。
  - S3 bucket は public access block 全 on、server-side encryption AES256。
- **Reliability**:
  - S3 bucket versioning 有効（誤 `dvc gc -c` からの救済）。
  - `dvc.lock` は git 追跡必須。`.dvc/tmp/` は ignored。
- **Portability**:
  - cache の絶対パスは `.dvc/config.local` に分離し、`.dvc/config` は worktree 横断で再利用可能な内容のみ。
- **CI**:
  - `dev/test-backend` は DVC に非依存で動く（テストは DVC pull を要求しないように fixtures を保つ）。
  - `dvc.yaml` / `dvc.lock` の構文チェックは scope 外（将来 pre-commit で追加）。

## Out of Scope (今回は実装しない)

- vast.ai / EC2 / SageMaker 等のリモート GPU 環境での DVC 運用（将来 GPU チーム用 IAM ロールは残しつつ今回は実装しない）。
- CI/CD ワークフロー (`.github/workflows/*`) からの DVC 操作自動化。
- `dvc exp run` / `dvc exp show` を使った実験比較 UI。
- DVC Studio / MLflow / W&B 連携。
- DVC remote の複数化（stage remote, backup remote 等）。

## Glossary

| Term | Description |
|------|-------------|
| DVC | Data Version Control。Git 親和性の高いデータ・パイプライン管理ツール |
| DVC remote | DVC の content-addressable storage 先。今回は S3 |
| DVC cache | ローカルの content-addressable キャッシュ (`.dvc/cache`)。remote との差分同期の起点 |
| stage | `dvc.yaml` における 1 実行単位 (cmd + deps + outs + params) |
| params.yaml | DVC stage が granular 追跡するパラメータファイル |
| dvc.lock | 各 stage の deps/outs ハッシュ + params 値のスナップショット (git 追跡) |
| S3 bucket versioning | 同一 key への PutObject を世代保存する S3 機能 |
