# ツール使用・API連携 — リソース一覧

## 領域概要

LLMが外部ツール（検索エンジン、計算機、API等）を呼び出し、テキスト生成だけでは不可能なタスクを遂行する技術。Function callingの標準化やModel Context Protocol（MCP）の登場により、ツール連携のエコシステムが急速に形成されつつある。

**関連キーワード**: `tool use`, `function calling`, `API integration`, `code execution`, `MCP`, `retrieval augmented`, `web browsing`

---

## 学術論文

| # | タイトル | 著者 | 年 | Venue | 概要 |
|---|---------|------|-----|-------|------|
| 1 | [Model Context Protocol (MCP): Landscape, Security Threats, and Future Research Directions](https://arxiv.org/abs/2503.23278) | 著者複数 | 2025 | arXiv | MCPのアーキテクチャとセキュリティの観点からの体系的研究。MCPサーバーのライフサイクルを定義し、包括的な脅威分類法を構築 |
| 2 | [ToolACE: Winning the Points of LLM Function Calling](https://arxiv.org/abs/2409.00920) | 著者複数 | 2024 | ICLR 2025 | ツール自己進化合成、自己誘導型対話生成、二層検証モジュールからなる関数呼び出しのための自動データパイプライン |
| 3 | [Unified Tool Integration for LLMs: A Protocol-Agnostic Approach to Function Calling](https://arxiv.org/abs/2508.02979) | 著者複数 | 2025 | arXiv | プロトコルに依存しないツール統合アプローチを提案し、LLMの関数呼び出し能力を統一的に扱う |
| 4 | [Natural Language Tools: A Natural Language Approach to Tool Calling in Large Language Agents](https://arxiv.org/abs/2510.14453) | 著者複数 | 2025 | arXiv | 自然言語を用いたツール呼び出しアプローチを提案し、より柔軟なツール連携を実現 |
| 5 | [Help or Hurdle? Rethinking Model Context Protocol-Augmented LLMs](https://arxiv.org/abs/2508.12566) | 著者複数 | 2025 | arXiv | MCPがLLMの能力を強化する一方で課題も生じることを分析。MCPがAI駆動プラットフォームの基本構成要素に変化したことを指摘 |
| 6 | [Gorilla: Large Language Model Connected with Massive APIs](https://arxiv.org/abs/2305.15334) | Patil et al. | 2024 | NeurIPS 2024 | Retriever Aware Training（RAT）を用いてLLMを大規模APIに接続。Berkeley Function Calling Leaderboard（BFCL）の基盤 |
| 7 | [MCP at First Glance: Studying the Security and Maintainability of MCP Servers](https://arxiv.org/abs/2506.13538) | 著者複数 | 2025 | arXiv | MCPサーバーのセキュリティと保守性の実態調査。週800万以上のSDKダウンロードを記録するMCPエコシステムの分析 |
| 8 | [From Language to Action: A Review of LLMs as Autonomous Agents and Tool Users](https://arxiv.org/abs/2508.17281) | 著者複数 | 2025 | arXiv | LLMエージェントのアーキテクチャ、ツール使用、推論、計画、メモリを包括的にレビュー |

### 注目論文

- **Model Context Protocol (MCP): Landscape, Security Threats, and Future Research Directions** — 2024年11月にAnthropicが発表し、2025年末にLinux Foundationに寄贈されたMCPの包括的分析。ツール連携の事実上の標準となったMCPのアーキテクチャとセキュリティ課題を体系的に整理しており、この領域の必読論文。
- **ToolACE** — ICLR 2025に採択された関数呼び出しのベンチマーク論文。自動データパイプラインによるツール使用能力の評価と向上手法を提案。
- **Gorilla** — NeurIPS 2024に採択。大規模APIとLLMの接続という根本的な課題に取り組み、Berkeley Function Calling Leaderboardとして業界標準のベンチマークを確立。

---

## 特許

| # | タイトル | 番号 | 出願人 | 年 | 特許庁 | 概要 |
|---|---------|------|--------|-----|--------|------|
| 1 | [Generating and executing action plans involving software tools via LLM](https://patents.google.com/patent/US20250272544A1/en) | US20250272544A1 | Adobe Inc. | 2025 | USPTO | LLMがソフトウェアツールのAPIを呼び出して一連のアクションを実行する計画生成・実行システム |
| 2 | [Declarative agent with hierarchical components using LLMs](https://patents.google.com/patent/US20250258856A1/en) | US20250258856A1 | Sierra Technologies | 2025 | USPTO | 階層的スキルセットを備えた宣言的エージェントサービス。LLMを用いてユーザー入力に対する応答を生成 |
| 3 | [Assisting users in interactions with LLMs](https://patents.google.com/patent/US20250173043A1/en) | US20250173043A1 | — | 2025 | USPTO | LLMとのユーザーインタラクションを支援するシステム |
| 4 | [Systems and methods for interacting with a large language model](https://patents.google.com/patent/US12051205B1/en) | US12051205B1 | — | 2024 | USPTO | LLMとの対話システムにおけるツール統合手法 |
| 5 | [AI assisted integration of new digital model types and tools](https://patents.google.com/patent/WO2025006790A2/en) | WO2025006790A2 | — | 2025 | WIPO | AIモデルを用いたAPI/SDK関数呼び出しスクリプトの自動生成 |

### 注目特許

- **US20250272544A1（Adobe）** — LLMがAPIを通じてソフトウェアツールを操作するための計画生成・実行システム。Adobe製品との統合を想定した実用的なアプローチ。
- **US20250258856A1（Sierra Technologies）** — 階層的スキル構造による宣言的エージェントの特許。MCPに類似した概念を特許化しており、ツール連携の商用化動向を示す。

---

## 次のステップ

- **論文の詳細調査**: research-papers スキルでこのリストの論文を詳しく調査できます
- **追加の領域マッピング**: research-clustering スキルで関連領域をさらに探索できます
