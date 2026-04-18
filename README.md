# KDD Cup 2026: Data Agents for Complex Data Analysis

KDD Cup 2026 参戦リポジトリ。複雑なデータ分析に対して自律的に推論・回答する **データエージェント** を開発します。

## 大会概要

- **主催**: 清華大学・HKUST（広州）
- **目的**: 分析質問を分解し、複数データソースにおける多段階推論を実行して正確な回答を導出する自律AIエージェントの構築
- **ベンチマーク**: DataAgent-Bench（構造化DB、PDF、JSON、PNG画像、DOCX等の異種データパッケージ + 自然言語質問）

### エージェントに求められる4つの能力

| 能力 | 説明 |
|------|------|
| **分解・計画** | 高度な分析質問を多段階実行計画へ自動分解 |
| **ツール選択** | Python, SQL, API呼び出し等から適切なツールを選択・実行 |
| **異種データ推論** | テーブル、文書、チャート、マルチモーダルデータに対応 |
| **結果統合** | 複数ステップの中間結果を総合して最終回答を合成 |

### 推論トポロジー

- **逐次チェーン**: ステップ間の依存関係が線形
- **分岐・統合**: 複数データソースの並列サブクエリ後に結果を統合
- **反復ループ**: 中間結果の反復的改善

### 難易度レベル

| レベル | データ構成 | 特徴 |
|--------|-----------|------|
| Easy | 構造化ファイル + 知識文書 | データ分析ワークフロー生成 |
| Medium | JSON/CSV + DB + 知識文書 | Text-to-SQL処理 |
| Hard | 非構造化文書を含む複合推論 | 10K-128Kトークン |
| Extreme | 超長文書入力 | 128K+トークン |

## 大会スケジュール

| フェーズ | 期間 | 内容 |
|---------|------|------|
| Phase 1 | 2026/4/24 〜 5/23 | 公開リーダーボードで自動評価 |
| Phase 2 | 2026/5/28 〜 6/30 | Leaderboard トラック or Creative トラック |

### 評価基準

**列マッチング正確性**: 予測が全ての正答列を完全かつ正確に網羅した場合のみスコア1、それ以外は0。列名は比較対象外、余分な列は許容。

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
infra/                       Terraform infrastructure (将来のデプロイ用)
dev/                         Development scripts
  setup                      Install dependencies (uv sync)
  format                     Code formatting (ruff)
  lint                       Static analysis (ruff + mypy)
  test-backend               Backend CI (format check -> lint -> type check -> pytest)
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
dev/test-backend     # Backend CI (format check -> lint -> type check -> pytest)
dev/create-worktree  # Create git worktree with .env copy
```

## Glossary

| Term | Description |
|------|-------------|
| DataAgent-Bench | KDD Cup 2026の公式ベンチマーク。異種データパッケージ + 自然言語質問で構成 |
| Data Package | CSV, JSON, SQLite, PDF, PNG, DOCX等を含むデータセット |
| Reasoning Topology | 推論の構造パターン（逐次チェーン、分岐・統合、反復ループ） |
| Column Matching | 評価方式。正答列の完全一致で判定 |

## Links

- [KDD Cup 2026 公式サイト](https://dataagent.top/)
- [Discord: KDD Cup 2026 | DataAgents](https://discord.gg/kdd-cup-2026)
