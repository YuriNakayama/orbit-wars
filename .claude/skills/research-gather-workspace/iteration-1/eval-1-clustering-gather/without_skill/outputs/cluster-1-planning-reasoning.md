# クラスタ1: 計画・推論 - リソース収集結果

## 概要

LLMの推論能力を活用し、複雑なタスクを段階的に分解・計画・実行するアプローチに関する学術論文および特許を収集した。Chain-of-Thought、ReAct、Tree-of-Thoughtsなどの推論戦略が含まれる。

**対象期間**: 2024年 - 2026年
**収集日**: 2026-03-29

---

## 学術論文

### 1. Pre-Act: Multi-Step Planning and Reasoning Improves Acting in LLM Agents
- **URL**: https://arxiv.org/abs/2505.09970
- **発表時期**: 2025年5月
- **概要**: ReActパラダイムを拡張し、各ステップに対する詳細なマルチステップ計画と推論を生成するアプローチ。会話型・非会話型の両AIエージェントに適用可能。

### 2. Why Reasoning Fails to Plan: A Planning-Centric Analysis of Long-Horizon Decision Making in LLM Agents
- **URL**: https://arxiv.org/abs/2601.22311
- **発表時期**: 2026年1月
- **概要**: 計画の観点から、標準的なLLM推論（Chain-of-Thoughtなど）がステップごとの貪欲な方策に過ぎないことを分析。LLMエージェントの長期的意思決定における推論の限界を明らかにした。

### 3. Model-First Reasoning LLM Agents: Reducing Hallucinations through Explicit Problem Modeling
- **URL**: https://arxiv.org/abs/2512.14474
- **発表時期**: 2025年12月
- **概要**: Chain-of-ThoughtやReActなどの既存戦略が暗黙の状態追跡に依存し、明示的な問題表現を欠いている点を指摘。明示的問題モデリングによるハルシネーション低減を提案。

### 4. Large Language Models for Planning: A Comprehensive and Systematic Survey
- **URL**: https://arxiv.org/abs/2505.19683
- **発表時期**: 2025年5月
- **概要**: LLMベースの計画に関する包括的サーベイ。タスク分解、メモリ拡張アプローチ、強化学習との統合など、計画能力向上のための手法を体系的に整理。

### 5. Recursive Decomposition of Logical Thoughts (RDoLT): Framework for Superior Reasoning and Knowledge Propagation in LLMs
- **URL**: https://arxiv.org/abs/2501.02026
- **発表時期**: 2025年1月
- **概要**: 思考の再帰的分解とロバストなスコアリングシステムを活用し、LLMの推論性能を向上させるフレームワーク。

### 6. Towards Autonomous Agents: Adaptive-planning, Reasoning, and Acting in Language Models
- **URL**: https://arxiv.org/abs/2408.06458
- **発表時期**: 2024年8月
- **概要**: 適応的計画、推論、行動を統合した自律エージェントの構築アプローチ。環境フィードバックに基づく計画の動的調整を実現。

### 7. MAPLE: Multi-Agent Adaptive Planning with Long-Term Memory for Table Reasoning
- **URL**: https://arxiv.org/abs/2506.05813
- **発表時期**: 2025年6月
- **概要**: 動的計画、リフレクション機構、自己改善、長期記憶を統合したマルチエージェントアーキテクチャ。テーブル推論タスクに適用。

### 8. Reasoning Beyond Limits: Advances and Open Problems for LLMs
- **URL**: https://arxiv.org/abs/2503.22732
- **発表時期**: 2025年3月
- **概要**: LLMの推論能力に関する最新の進展と未解決問題を包括的にレビュー。自己反省、計画、複雑な推論タスクにおける課題を議論。

---

## 特許

### 1. Dynamic Agents with Real-time Alignment (Microsoft)
- **特許番号**: US20250368219A1
- **URL**: https://patents.google.com/patent/US20250368219A1/en
- **公開年**: 2025年
- **概要**: 目的達成のためにタスクを含む計画を生成・実行するマルチエージェントシステム。複雑なマルチステップタスクにおけるLLMの予測不可能な出力に対処。

### 2. Prompting Language Models with Workflow Plans (ServiceNow)
- **特許番号**: US20240176958A1
- **URL**: https://patents.google.com/patent/US20240176958A1
- **公開年**: 2024年
- **概要**: 訓練時に見られなかったプランでLLMを条件付けすることで、ゼロショット学習設定における新しいタスクへの適応能力を向上。ワークフロー計画を用いたプロンプティング手法。

### 3. Declarative Agent with Hierarchical Components Using LLMs (Sierra Technologies)
- **特許番号**: US20250258856A1
- **URL**: https://patents.google.com/patent/US20250258856A1/en
- **公開年**: 2025年
- **概要**: 階層的なスキルセットを展開する宣言型エージェントサービス。分類に基づいてLLM呼び出しを反復的に特定し、ユーザー入力に応答を生成。

### 4. Enterprise Generative AI Architecture (C3.AI)
- **特許番号**: US12111859B2
- **URL**: https://patents.google.com/patent/US12111859B2/en
- **発行年**: 2024年
- **概要**: オーケストレーターが複数のエージェントを管理し、LLMを用いてプロンプトをタスク系列に分解。推論機能を提供するエージェントアーキテクチャ。

### 5. Composite Symbolic and Non-Symbolic AI System for Advanced Reasoning and Automation
- **特許番号**: US20250258852A1
- **URL**: https://patents.google.com/patent/US20250258852A1/en
- **公開年**: 2025年
- **概要**: シンボリックAIと非シンボリックAIを組み合わせた高度な推論・自動化システム。LLMエージェントの推論能力を構造化された方法で強化。

### 6. Systems and Methods for Orchestrating LLM-Augmented Autonomous Agents (Salesforce)
- **特許番号**: US20250053793A1
- **URL**: https://patents.google.com/patent/US20250053793A1/en
- **公開年**: 2025年
- **概要**: LLM拡張自律エージェント（LAA）のオーケストレーション。コントローラーがタスク指示を受け取り、環境を使用してタスクを実行するエージェントの行動を予測。

---

## まとめ

計画・推論クラスタでは、2024-2026年の間にReActの拡張（Pre-Act）、計画と推論の関係性の再分析、明示的問題モデリング、再帰的思考分解など、多様なアプローチが提案されている。特許面では、Microsoft、Salesforce、Sierra Technologies等の大手企業がエージェントの計画・推論アーキテクチャに関する特許を積極的に出願している。
