---
name: Dev Tools
description: Verification, checkpoints, refactoring, documentation management, strategic compaction, and project setup utilities
activation: Running verification checks, managing checkpoints, cleaning dead code, updating docs, or configuring project tooling
---

# Dev Tools

開発ユーティリティ群。検証、チェックポイント、リファクタリング、ドキュメント管理、コンパクション、プロジェクト設定。

## Commands

- `/verify` - コードベースの包括的検証（build→types→lint→tests→security→git status）
- `/checkpoint` - チェックポイントの作成・比較・管理
- `/remove-unused` - デッドコードの検出・安全な削除
- `/update-codemaps` - コードベース構造の分析・アーキテクチャドキュメント更新
- `/update-docs` - ソースオブトゥルースからのドキュメント同期

## Verification

### Phases (in order)

1. **Build** - コンパイル確認
2. **Type Check** - 型チェック（tsc / mypy）
3. **Lint** - リント（ESLint / ruff）
4. **Test Suite** - テスト実行
5. **Security Scan** - セキュリティチェック
6. **Diff Review** - 変更差分レビュー

### Modes

- `quick` - build + types のみ
- `full` - 全チェック
- `pre-commit` - コミット前検証
- `pre-pr` - PR 前の完全検証

## Checkpoints

```bash
/checkpoint create <name>    # チェックポイント作成
/checkpoint verify <name>    # 現在の状態と比較
/checkpoint list             # 一覧表示
/checkpoint clear            # クリア
```

## Refactoring

デッドコード分析ツール:
- `knip` - 未使用の exports/dependencies
- `depcheck` - 未使用パッケージ
- `ts-prune` - 未使用 TypeScript exports

削除前後にテストスイートを実行して安全性を検証。

## Documentation Management

- `/update-codemaps` - `docs/codemaps/` にアーキテクチャドキュメントを生成
  - 変更差分が 30% 超の場合は承認を要求
  - 鮮度タイムスタンプを付与
- `/update-docs` - `package.json` scripts、`.env.example` からドキュメントを同期
  - 90日以上古いドキュメントを検出

## Strategic Compaction

コンテキストウィンドウの効率的な利用のため、論理的な区切りで手動コンパクションを推奨:
- 計画フェーズ完了後
- デバッグセッション完了後
- 大きなファイル変更の後
- フェーズ切り替え時

## Project Setup

パッケージマネージャの検出優先順:
1. 環境変数
2. プロジェクト設定
3. package.json
4. ロックファイル
5. グローバル設定
6. フォールバック（npm）
