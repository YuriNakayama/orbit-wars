# dvc-data-control — Implementation Steps

## 実行方針

- **Order**: backend-first で DVC コア → pipeline 化 → 提出フロー、次に infra を並行。
- **Granularity**: 1 stage = 1 技術関心。Step 単位で `dev/test-backend` が通る状態を維持。
- **Parallelizable**: Step 6 (Terraform module) と Step 3-5 (Python pipeline) は依存なし → 並行可能。
- **Cross-cutting**: ドキュメント更新は最後の Step 9 で一括。

## Dependency Graph

```
Step 1 (dvc init + .gitignore)
  ├─→ Step 2 (params.yaml 作成)
  │     └─→ Step 3 (preprocess.py 改修)
  │           └─→ Step 4 (train.py 改修)
  │                 └─→ Step 5 (dvc.yaml 定義 + dvc add 既存データ)
  │                       ├─→ Step 7 (Kaggle submit フック)
  │                       └─→ Step 8 (DVC smoke test)
  └─→ Step 6 (Terraform dvc_remote module) ← 並行可

Step 9 (ドキュメント更新) ← すべて完了後
```

---

## Step 1: DVC 初期化と .gitignore / .dvcignore 整備

**Target**: cross-cutting
**Dependencies**: None

### Overview
リポジトリに DVC を導入し、git 管理境界を確定する。

### Work Items
- [ ] `backend/pyproject.toml` に `dvc[s3]>=3.55` を追加、`uv sync` で lock 更新
- [ ] リポジトリルートで `uv run --directory backend dvc init` を実行
- [ ] `.dvc/config` を生成（remote 設定は後段、core と cache 初期値のみ）
- [ ] `.dvcignore` を作成
- [ ] `.gitignore` に `/.dvc/tmp`, `/.dvc/config.local`, `/.dvc/cache`, `/data`, `backend/pipeline/imitation/case1/policy/weights.pt`, `infra/**/.terraform/`, `infra/**/*.tfstate*`, `infra/**/terraform.tfvars` を追記
- [ ] `dev/dvc-setup` シェルスクリプト作成（後段の remote / cache.dir 設定を冪等に実行）

### Target Files
- `backend/pyproject.toml`
- `backend/uv.lock`
- `.dvc/config`
- `.dvcignore`
- `.gitignore`
- `dev/dvc-setup` (新規)

### Acceptance Criteria
- `uv run --directory backend dvc version` が動く
- `git status` で data/ や .dvc/cache が ignored になっている
- `dev/dvc-setup` を 2 回実行しても冪等

---

## Step 2: params.yaml を新設し、既存 YAML を調査

**Target**: backend
**Dependencies**: Step 1

### Overview
ルート `params.yaml` を新設し、`configs/il_baseline.yaml` の値をそのまま移植する。
検証プロジェクト方針に従い、最終的には `configs/` を削除するが本 Step では併存。

### Work Items
- [ ] リポジトリルートに `params.yaml` を作成（03-architecture.md の雛形をコピー）
- [ ] `configs/il_baseline.yaml` と値が一致することを diff で確認
- [ ] `params.yaml` の形式を pytest で軽く検証（任意）

### Target Files
- `params.yaml` (新規)

### Acceptance Criteria
- `params.yaml` が yaml.safe_load で parse 可能
- `configs/il_baseline.yaml` と意味論的に同値

---

## Step 3: `preprocess.py` を params.yaml 固定読み込みに改修

**Target**: backend
**Dependencies**: Step 2

### Overview
`--config` 引数を廃止し、`params.yaml` を固定読み込みする。

### Work Items
- [ ] `backend/pipeline/imitation/case1/training/preprocess.py` の typer CLI から `--config` を削除
- [ ] `params.yaml` を `Path("params.yaml")` で固定読み込み（cwd=`backend/` の前提で `../params.yaml`、または `dvc.yaml` の `wdir` で調整）
- [ ] 出力先は params.data.out_train / out_val を参照
- [ ] `tests/pipeline/imitation/case1/test_preprocess*.py` のパッチ（config 引数依存を除去）

### Target Files
- `backend/pipeline/imitation/case1/training/preprocess.py`
- `backend/tests/pipeline/imitation/case1/test_preprocess*.py`（該当あれば）

### Acceptance Criteria
- `uv run --directory backend python -m pipeline.imitation.case1.training.preprocess` が動く
- 既存テストが通る

---

## Step 4: `train.py` を params.yaml 固定読み込みに改修

**Target**: backend
**Dependencies**: Step 3

### Overview
同様に `train.py` からも `--config` を除去。

### Work Items
- [ ] `backend/pipeline/imitation/case1/training/train.py` の CLI 改修
- [ ] 出力重みパスを params.train.weights_out 参照に
- [ ] `evaluation/eval_vs_baseline.py` も params.yaml 読みに揃える（メトリクス JSON を `data/mart/imitation/case1/eval_metrics.json` に書く）
- [ ] 対応テスト修正

### Target Files
- `backend/pipeline/imitation/case1/training/train.py`
- `backend/pipeline/imitation/case1/evaluation/eval_vs_baseline.py`
- `backend/tests/pipeline/imitation/case1/*`

### Acceptance Criteria
- train / eval の CLI が params.yaml のみで動く
- eval_metrics.json が生成される

---

## Step 5: `dvc.yaml` 定義と既存データの DVC 取り込み

**Target**: backend
**Dependencies**: Step 4

### Overview
pipeline を DVC stage に落とし込み、既存の lake / mart / weights を DVC 管理下に置く。

### Work Items
- [ ] `dvc.yaml` を 03-architecture.md の例を元に作成（stages 4 つ）
- [ ] `uv run --directory backend dvc add data/lake/kaggle_episodes/matches` — 既存 replay を DVC 管理に登録（.dvc stub 生成）
- [ ] 同様に selfplay も必要なら dvc add
- [ ] `dvc repro preprocess_imitation_case1` を dry run（deps があれば即 skip）
- [ ] `dvc.lock` が生成されることを確認
- [ ] `configs/il_baseline.yaml` を削除（検証プロジェクト方針、params.yaml に移行完了）

### Target Files
- `dvc.yaml` (新規)
- `dvc.lock` (自動生成)
- `data/lake/kaggle_episodes/matches.dvc` (自動生成)
- `backend/pipeline/imitation/case1/configs/` 削除

### Acceptance Criteria
- `dvc status` が clean（またはローカルで想定通りの差分）
- `dvc dag` で依存グラフが描画される

---

## Step 6: Terraform dvc_remote module 作成 (Step 1-5 と並行可)

**Target**: infra
**Dependencies**: None (Step 1 と並行可、実行だけ Step 7 前に)

### Overview
S3 bucket + IAM user/policy を Terraform 化。apply は scope 外、plan まで通す。

### Work Items
- [ ] `infra/module/application/dvc_remote/versions.tf` (required_providers)
- [ ] `infra/module/application/dvc_remote/variables.tf` (bucket_name, prefix)
- [ ] `infra/module/application/dvc_remote/main.tf` (03-architecture.md 参照)
- [ ] `infra/module/application/dvc_remote/outputs.tf` (bucket_name, iam_access_key_id sensitive, iam_secret_access_key sensitive)
- [ ] `infra/environment/dev/versions.tf`, `main.tf`, `variables.tf`, `outputs.tf`
- [ ] `infra/environment/dev/terraform.tfvars.example` を作成、実 tfvars は gitignore
- [ ] `terraform fmt` / `terraform validate` / `terraform plan`（credentials 未設定なら plan は dry）

### Target Files
- `infra/module/application/dvc_remote/*.tf`
- `infra/environment/dev/*.tf`
- `infra/environment/dev/terraform.tfvars.example`

### Acceptance Criteria
- `terraform fmt -check -recursive` が通る
- `terraform validate` が通る
- `terraform plan`（mock credentials で OK）の出力に破綻がない

---

## Step 7: Kaggle 提出フローに dvc pull を組み込み

**Target**: backend
**Dependencies**: Step 5

### Overview
`weights.pt` が DVC 管理に移ったことで、tar.gz ビルド前に `dvc pull` が必要。

### Work Items
- [ ] `backend/src/submit/packager.py` に `ensure_weights(case_dir)` を追加
- [ ] 内部で `subprocess.run(["dvc", "pull", str(weights)], cwd=repo_root, check=True)` を実行
- [ ] `.dvc` stub の検出ロジック（ファイルサイズ < 数 KB のテキスト等）
- [ ] `dev/submit` は既存のまま（packager が内部で対応）
- [ ] 単体テストで `dvc pull` をモック化してフックが呼ばれることを確認

### Target Files
- `backend/src/submit/packager.py`
- `backend/tests/submit/test_packager.py`

### Acceptance Criteria
- `uv run python -m submit submit imitation/case1 --dry-run -m "dvc integration test"` が tar.gz ビルドまで到達
- weights.pt がない状態から dvc pull が走り、build_tar 前に実体化されている

---

## Step 8: DVC smoke test（実データなし）

**Target**: backend
**Dependencies**: Step 5, Step 7

### Overview
DVC が想定どおり動いていることを CI 非依存で確認。

### Work Items
- [ ] `dvc status` / `dvc dag` の出力を確認
- [ ] ダミー remote（ローカル `/tmp/dvc-smoke-remote/`）を `dvc remote add -d --local smoke /tmp/...` で一時追加し、`dvc push` / `dvc pull` の往復動作を確認
- [ ] smoke 完了後、--local 設定をリセット（実 remote は S3 に戻す）
- [ ] `dev/dvc-setup` の冪等性再確認

### Acceptance Criteria
- dvc push / pull がローカルダミー remote で動作
- 実 S3 remote への push は本 Step では **実施しない**（apply まで完了後の別作業）

---

## Step 9: ドキュメント更新

**Target**: cross-cutting
**Dependencies**: 全 Step 完了後

### Overview
README / CLAUDE.md / imitation case1 README を DVC 運用手順込みに更新。

### Work Items
- [ ] ルート `README.md`: "Data Management" 節を追加。DVC + S3 remote の概略、`dev/dvc-setup` / `dvc pull` / `dvc repro` の使い方
- [ ] `.claude/CLAUDE.md`: Folder Structure に `dvc.yaml`, `params.yaml`, `.dvc/`, `infra/module/application/dvc_remote/` を追記。Commands に `dev/dvc-setup` 追記
- [ ] `backend/pipeline/imitation/case1/README.md`: 手順を `dvc repro preprocess_imitation_case1` ... の流儀に更新、旧 `--config` コマンドを削除
- [ ] `docs/plans/dvc-data-control/` に本 plan 6 ファイル（既存）
- [ ] `.claude/rules/` に `dvc.md` を追加するか検討（パス指定は `dvc.yaml`, `params.yaml`, `.dvc/**`）— 必要なら追加
- [ ] `infra/environment/dev/README.md` を追加し、apply 手順と credentials セットアップを記載

### Target Files
- `README.md`
- `.claude/CLAUDE.md`
- `.claude/rules/dvc.md` (任意新規)
- `backend/pipeline/imitation/case1/README.md`
- `infra/environment/dev/README.md`

### Acceptance Criteria
- 新規開発者が README を読むだけで `dev/dvc-setup` → `dvc pull` → `dvc repro` まで辿れる
- `dev/test-backend` が通る
