# KDD Cup 2026: Data Agent

KDD Cup 2026 参戦プロジェクト。DataAgent-Bench に対して自律的に多段階推論を実行し、正確な回答を導出するデータエージェントを開発する。

## Architecture

```
Natural Language Question
  ↓
Planner (質問分解・実行計画生成)
  ↓
Executor (ツール選択・実行)
  ├── Python Script — データ処理・集計
  ├── SQL Query — DB問い合わせ (DuckDB/SQLite)
  ├── Document Parser — PDF/DOCX/JSON読み取り
  └── Image Analyzer — PNG/チャート解析
  ↓
Reasoner (中間結果の推論・統合)
  ↓
Integrator (最終回答の合成)
```

## Technology Stack

- **Language**: Python 3.12+
- **AI**: Claude API / Claude CLI
- **Data Processing**: Pandas, DuckDB/SQLite, Pillow
- **Document Processing**: PyPDF2, python-docx, PyYAML
- **Testing**: Pytest + pytest-cov, Ruff, Mypy
- **Package Management**: UV

## Folder Structure

```
pipeline/
  case1/                     分析ケース1
    eda/                     探索的データ分析
  case2/                     分析ケース2（以降同様）
backend/
  pyproject.toml             Python dependencies (UV managed)
  uv.lock                    Lock file
  src/                       共通ライブラリ・ユーティリティ
  tests/                     Pytest unit tests
infra/
  main.tf                    Root module (provider, backend)
  variables.tf               Input variables
  outputs.tf                 Output values
  modules/                   Terraform modules
dev/                         Development scripts
  setup                      Install dependencies (uv sync)
  format                     Code formatting (ruff)
  lint                       Static analysis (ruff + mypy)
  test-backend               Backend CI (format check → lint → type check → pytest)
  create-worktree            Create git worktree with .env copy
docs/
  plans/                     Feature plans
  research/                  Research prompts and outputs
```

## Commands

```bash
dev/setup            # Install dependencies (uv sync in backend/)
dev/format           # Code formatting (backend: ruff)
dev/lint             # Static analysis (backend: ruff + mypy)
dev/test-backend     # Backend CI (format check → lint → type check → pytest)
dev/create-worktree  # Create git worktree with .env copy
```

## Glossary

| Term | Description |
|------|-------------|
| DataAgent-Bench | KDD Cup 2026公式ベンチマーク。異種データパッケージ + 自然言語質問 |
| Data Package | CSV, JSON, SQLite, PDF, PNG, DOCX等を含むデータセット |
| Reasoning Topology | 推論の構造パターン（逐次チェーン、分岐・統合、反復ループ） |
| Column Matching | 評価方式。予測が全正答列を完全一致で網羅した場合のみスコア1 |
| Planner | 質問を多段階実行計画へ分解するモジュール |
| Executor | Python, SQL, API等のツールを選択・実行するモジュール |
| Reasoner | 中間結果を推論・統合するモジュール |

## Rules

| Rule file | Auto-loaded for | When to read manually |
|-----------|----------------|----------------------|
| `.claude/rules/backend.md` | `backend/**` | Python code changes, pytest, ruff/mypy configuration |
| `.claude/rules/pipeline.md` | `pipeline/**` | 分析パイプライン、ケース別データ処理、推論ロジック |
| `.claude/rules/infra.md` | `infra/**` | Terraform changes, AWS resource design, module structure decisions |
| `.claude/rules/security.md` | Always loaded | Commits, secret handling, CI/CD pipeline changes |

## Response Language

- Internal reasoning should be in English
- All user-facing output must be in Japanese(全てのユーザー向けの出力は日本語で行うこと)
