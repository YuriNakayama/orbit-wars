# Kaggle 提出自動化フロー — コードベース調査

## 調査対象リポジトリ

- パス: `/Users/user/project/orbit-wars.worktrees/feature-kaggle-deploy-flow`
- ブランチ: `feature/kaggle-deploy-flow`

## 詳細コードベース分析

### Area 1: `pipeline/case0/`（テスト対象エージェント）

- **Files analyzed**:
  - `pipeline/case0/main.py` (1–60行)
  - `pipeline/case0/agents.md` (1–213行)
  - `pipeline/case0/README.md` (1–173行)
- **Current implementation**:
  - `main.py` は `agent(obs)` を1関数で定義。`kaggle_environments.envs.orbit_wars.orbit_wars.Planet` を `import` し、`math` のみ使用。外部ファイルへの依存なし。
  - `agents.md` にすでに kaggle CLI を使った提出手順が記載されている（`kaggle competitions submit orbit-wars -f main.py -m "..."`, `tar -czf submission.tar.gz main.py ...`）。
- **Key interfaces**:
  - シグネチャ `def agent(obs): -> list[list[int|float]]` — 戻り値は `[[from_planet_id, angle, num_ships], ...]`
  - Kaggle側の要求: ルート直下に `main.py` と `agent(obs)` 関数があること。
- **Patterns used**: 単一ファイルエージェント。依存は kaggle-environments のみ。
- **Coupling & side effects**: なし。stateless（各ターン独立）。
- **Test coverage**: なし（`tests/` ディレクトリ自体未作成）。
- **Gaps identified**:
  - ローカル自己対戦スクリプトがない（`env.run()` を呼ぶ雛形なし）。
  - 提出自動化（CLIラッパ、GitHub Actions、認証管理）は未実装。
  - 複数ファイル対応（tar.gz バンドル）の仕組みもない。

### Area 2: `dev/` スクリプト群

- **Files analyzed**:
  - `dev/setup` (23行): `uv sync` のみ。`backend/` ディレクトリ前提で書かれているが本プロジェクトはモノレポではない。
  - `dev/deploy` (66行): `gh workflow run` を呼ぶ bash。`target` × `environment` で条件分岐。
  - `dev/test-backend`, `dev/format`, `dev/lint`: 既存ユーティリティ。
- **Key interfaces**: `#!/bin/bash` + `set -euo pipefail`。引数検証 → `gh` コマンド実行。
- **Patterns used**: コマンド単位の bash スクリプト。`gh workflow run` で手動CDトリガー。
- **Gaps identified**:
  - Kaggle 提出専用スクリプト `dev/submit` は存在しない。
  - `dev/setup` が `backend/` 前提で本プロジェクトに合わない（ルート `pyproject.toml` を見ない）。

### Area 3: `.github/workflows/`

- **Files analyzed**:
  - `build-push.yml` (138行): main push時ECRへpush + 手動でECSタスク実行。`backend/**` パス前提。
  - `ci-backend.yml` (59行): PR時 ruff format/lint。`working-directory: backend` 前提で本プロジェクトではパスが合わない（将来の修正が必要だが本タスクの範囲外）。
- **Key patterns**:
  - `astral-sh/setup-uv@v6` で uv インストール
  - `uv sync --locked --all-extras --dev`
  - `permissions: id-token: write` で OIDC
- **Gaps identified**: Kaggle 提出用ワークフロー未整備。secrets 管理方針も未定義。

### Area 4: `pyproject.toml`（依存関係）

- **Files analyzed**: `pyproject.toml` (1–182行)
- **Key findings**:
  - Python 3.13 固定、`kaggle-environments>=1.17.0` は依存済み。
  - `typer>=0.15.2`, `rich>=13.9.4` が利用可能 → 提出CLIに最適。
  - `python-dotenv>=1.0.0` が既に依存 → Kaggle認証の `.env` 読み込みが自然。
  - **`kaggle` パッケージ自体は依存に含まれていない** → 追加が必要（または PATH に存在する CLI を呼ぶ方針）。
  - `pipeline` は `[tool.hatch.build.targets.wheel]` で packages に含まれている。

### Area 5: `src/`

- **現状**: 空ディレクトリ。 `agents/`, `env/`, `features/`, `policies/`, `utils/` の想定構造はあるが未実装。
- **Gaps**: 提出用共通ユーティリティを置く自然な場所は `src/submit/` もしくは `pipeline/` 直下のツール。本タスクでは将来の拡張を見据えて `src/submit/` に提出処理を置くのが妥当。

## 技術的制約

1. **Kaggle CLI 認証**: `~/.kaggle/kaggle.json` または `KAGGLE_USERNAME` / `KAGGLE_KEY` 環境変数の両対応が必要。現状ローカルには未設定（`.kaggle/` 未作成、`KAGGLE_USERNAME` 未設定）。
2. **ルール受諾**: `kaggle competitions submit` 前に Web UI で Join Competition が必須。CLI で事前に `kaggle competitions list --group entered` で確認可能。
3. **提出形式**:
   - 単一ファイル: `main.py` を直接投げる
   - 複数ファイル: `tar -czf submission.tar.gz main.py ...` でルートに `main.py` を含める
4. **1日5提出まで**。自動化で誤爆防止が重要（確認プロンプト or dry-run モード）。
5. **Python バージョン**: ローカル 3.13 指定だが、Kaggle 実行環境は Kaggle 側Pythonランタイムで走るため、標準ライブラリ＋ kaggle-environments のみ依存という点に注意。
6. **venv 未作成**: 初回実行前に `uv sync` が必要。

## 主要な発見まとめ

- **活用可能**:
  - `typer` + `rich` で CLI を綺麗に作れる。
  - `pipeline/case0/agents.md` には提出コマンドが網羅されている（これを自動化の仕様にできる）。
  - `dev/deploy` のパターンを流用して `dev/submit` を作れる。
  - `gh workflow run` パターンで GitHub Actions 側トリガーも統一できる。
- **要実装**:
  - `src/submit/` に提出処理（tar.gz生成、kaggle CLI呼出し、ローカル検証、提出履歴取得）
  - `dev/submit` bash ラッパー
  - `.github/workflows/cd-kaggle-submit.yml` CD パイプライン
  - `tests/submit/` ユニットテスト
- **意図的に範囲外**:
  - `ci-backend.yml` の `working-directory: backend` 修正（既存バグ。別タスクで）
  - `dev/setup` の `backend/` 前提修正（同上）
