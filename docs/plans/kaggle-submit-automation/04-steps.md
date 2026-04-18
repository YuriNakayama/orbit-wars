# 実装ステップ

## Step 1: 依存追加と pyproject 更新

**Target**: backend
**Dependencies**: None

### 作業項目
- [ ] `pyproject.toml` の `dependencies` に `kaggle>=1.6` を追加
- [ ] `uv sync` を実行し、ロック更新を確認

### 対象ファイル
- `pyproject.toml`
- `uv.lock`（自動生成）

### 受け入れ基準
- `uv run kaggle --version` が成功
- `uv run python -c "import kaggle_environments; kaggle_environments.make('orbit_wars')"` が成功

## Step 2: `src/submit/` パッケージを実装

**Target**: backend
**Dependencies**: Step 1

### 作業項目
- [ ] `src/submit/__init__.py`（空）
- [ ] `src/submit/packager.py`（tar.gz 生成）
- [ ] `src/submit/validator.py`（kaggle_environments でドライラン）
- [ ] `src/submit/auth.py`（認証有無チェック）
- [ ] `src/submit/kaggle_api.py`（CLIラッパ）
- [ ] `src/submit/history.py`（submissions.jsonl追記）
- [ ] `src/submit/__main__.py`（typer CLI）

### 対象ファイル
上記 7 ファイル

### 受け入れ基準
- `uv run python -m submit --help` がヘルプを表示
- `uv run python -m submit submit case0 -m "test" --dry-run` が提出なしで完了

## Step 3: `dev/submit` bash ラッパーを追加

**Target**: dev infra
**Dependencies**: Step 2

### 作業項目
- [ ] `dev/submit` スクリプト作成（既存 `dev/deploy` パターン踏襲）
- [ ] 実行権限 `chmod +x dev/submit`

### 受け入れ基準
- `./dev/submit --help` がヘルプ表示
- `./dev/submit case0 -m "test" --dry-run` が成功

## Step 4: pytest テストを追加

**Target**: backend
**Dependencies**: Step 2

### 作業項目
- [ ] `tests/__init__.py`, `tests/submit/__init__.py`
- [ ] `tests/submit/test_packager.py`（tar.gz構造検証、除外パターン）
- [ ] `tests/submit/test_auth.py`（env変数あり/なし、config file存在判定のモック）
- [ ] `tests/submit/test_validator.py`（case0 を ドライラン対象に使う、smoke test）
- [ ] `tests/submit/test_history.py`（jsonl追記）

### 受け入れ基準
- `uv run pytest tests/submit -v` が全件パス
- カバレッジ80%以上（`src/submit/`）

## Step 5: GitHub Actions ワークフロー追加

**Target**: infra
**Dependencies**: Step 2

### 作業項目
- [ ] `.github/workflows/cd-kaggle-submit.yml` を新規作成
- [ ] `workflow_dispatch` で `case`, `message`, `dry_run`, `wait` 入力
- [ ] secrets で `KAGGLE_USERNAME`, `KAGGLE_KEY`

### 受け入れ基準
- `gh workflow list` に `cd-kaggle-submit.yml` が表示される
- （実際の実行は Secrets 設定後に可能）

## Step 6: `pipeline/case0` を検証してローカル提出

**Target**: validation / smoke test
**Dependencies**: Step 1–5

### 作業項目
- [ ] `dev/submit case0 -m "case0 baseline: nearest planet sniper" --dry-run` でローカル検証
- [ ] 認証設定があれば `dev/submit case0 -m "..."` で本番提出
- [ ] `kaggle competitions submissions orbit-wars` で確認

### 受け入れ基準
- dry-run が成功し、`data/submissions/case0/<ts>.tar.gz` が生成
- `submissions.jsonl` に1行追記
- 本番提出時、`status` が `complete` になる（Kaggle 側非同期処理のため時間かかる場合あり）

## Step 7: README／docs 追記

**Target**: docs
**Dependencies**: Step 6

### 作業項目
- [ ] ルート `README.md` に「提出フロー」セクション
- [ ] `pipeline/case0/README.md` に提出コマンドの追記は既に充足

## 並列化可能

- Step 2 と Step 5 は並列可
- Step 3 は Step 2 完了後すぐ
- Step 4 は Step 2 と並行して書き始め可能
