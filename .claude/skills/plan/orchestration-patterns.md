---
name: Orchestration
description: Agent orchestration, implementation planning, and iterative context retrieval for multi-agent workflows
activation: Complex features, multi-step tasks, agent coordination, or implementation planning
---

# Orchestration

エージェントオーケストレーション、実装計画、マルチエージェントワークフロー。

## Commands

- `/plan` - 要件整理・リスク評価・ステップバイステップの実装計画作成
- `/orchestrate` - 複数エージェントの逐次・並列ワークフロー実行

## Workflow Types

- `feature` - 新機能実装
- `bugfix` - バグ修正
- `refactor` - リファクタリング
- `security` - セキュリティ監査
- `custom` - カスタムワークフロー

## Iterative Retrieval Pattern

マルチエージェントワークフローにおける「コンテキスト問題」を解決するパターン。

### 問題

サブエージェントは限られたコンテキストで起動される:
- どのファイルに関連コードがあるか不明
- コードベースのパターンが不明
- プロジェクト用語が不明

### 解決策: 4フェーズループ

```
DISPATCH → EVALUATE → REFINE → LOOP (最大3サイクル)
```

1. **DISPATCH** - 初期クエリでコンテキストを取得
2. **EVALUATE** - 取得した情報の関連性を評価
3. **REFINE** - 不足情報を特定し検索を改善
4. **LOOP** - 十分なコンテキストが得られるまで繰り返し（最大3回）

### ベストプラクティス

- 最初のクエリは広めに設定
- 各サイクルで具体性を上げる
- 3サイクルで収束しない場合は人間に確認
- 取得情報はハンドオフドキュメントに集約
