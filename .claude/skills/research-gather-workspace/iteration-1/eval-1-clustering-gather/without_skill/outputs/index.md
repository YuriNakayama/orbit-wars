# LLMエージェント技術 - リソース収集結果サマリ

## 収集概要

| 項目 | 内容 |
|------|------|
| **テーマ** | LLMエージェント技術 |
| **対象期間** | 2024年 - 2026年（直近2年） |
| **収集日** | 2026-03-29 |
| **リソースタイプ** | 学術論文（arXiv等）、特許（Google Patents） |
| **入力** | clustering-output-llm-agents.md（3クラスタ） |

---

## クラスタ別収集結果

| # | クラスタ名 | 論文数 | 特許数 | ファイル |
|---|-----------|--------|--------|----------|
| 1 | 計画・推論 | 8 | 6 | [cluster-1-planning-reasoning.md](./cluster-1-planning-reasoning.md) |
| 2 | ツール使用・API連携 | 9 | 6 | [cluster-2-tool-use-api.md](./cluster-2-tool-use-api.md) |
| 3 | マルチエージェント協調 | 9 | 6 | [cluster-3-multi-agent.md](./cluster-3-multi-agent.md) |
| **合計** | | **26** | **18** | |

---

## 主要な知見

### クラスタ1: 計画・推論
- ReActの拡張版であるPre-Actが2025年に登場し、マルチステップ計画と推論を統合
- 2026年初頭に「推論は計画ではない」という分析論文が発表され、CoTの限界を指摘
- MCPの包括的サーベイ（2025年5月）がLLM計画研究を体系化
- Microsoft、Salesforce、Sierra Technologies等が計画・推論エージェントの特許を出願

### クラスタ2: ツール使用・API連携
- Anthropicが2024年11月に導入したMCP（Model Context Protocol）が大きな転換点
- MCPに関するセキュリティ分析、スケーラビリティ、ベンチマーク研究が急増
- Function Calling精度向上のための構造化テンプレート手法が提案
- プロトコル非依存のツール統合アプローチが注目を集める

### クラスタ3: マルチエージェント協調
- マルチエージェント強化学習（MARL）とLLM協調の融合（MAGRPO）
- 最大100エージェントまでスケール可能なベンチマーク（AgentsNet）の登場
- 潜在空間を活用したエージェント間通信という新しいアプローチ
- CrewAI、AutoGen、LangGraph、MetaGPTの比較分析が活発化

---

## 検索ソース

- **学術論文**: arXiv, Semantic Scholar, ACL Anthology
- **特許**: Google Patents
- **検索キーワード**: LLM planning, reasoning agent, chain-of-thought, ReAct, tool use, function calling, MCP, multi-agent LLM, collaborative agents, AutoGen, CrewAI
