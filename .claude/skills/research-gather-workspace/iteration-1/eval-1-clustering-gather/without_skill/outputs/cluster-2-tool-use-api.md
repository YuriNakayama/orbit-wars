# クラスタ2: ツール使用・API連携 - リソース収集結果

## 概要

LLMが外部ツール（検索エンジン、計算機、API等）を呼び出し、テキスト生成だけでは不可能なタスクを遂行する技術に関する学術論文および特許を収集した。Function callingやMCP（Model Context Protocol）などの標準化の動向も含む。

**対象期間**: 2024年 - 2026年
**収集日**: 2026-03-29

---

## 学術論文

### 1. Unified Tool Integration for LLMs: A Protocol-Agnostic Approach to Function Calling
- **URL**: https://arxiv.org/abs/2508.02979
- **発表時期**: 2025年8月
- **概要**: ツール拡張LLMの断片化されたエコシステムに対し、プロトコル差異を抽象化する統一的なツール統合アプローチを提案。統合シナリオ全体で60-80%のコード削減と最大3.1倍のパフォーマンス改善を実証。

### 2. Training LLMs for Multi-Step Tool Orchestration with Constrained Data Synthesis and Graduated Rewards
- **URL**: https://arxiv.org/abs/2603.24709
- **発表時期**: 2026年3月
- **概要**: LLMが外部ツールをFunction Callingで使用する際のマルチステップツールオーケストレーションに焦点。正しい順序でAPIを呼び出し、前のステップの出力を次のステップの入力として伝播する手法を提案。

### 3. ToolACE: Winning the Points of LLM Function Calling
- **URL**: https://arxiv.org/abs/2409.00920
- **発表時期**: 2024年9月
- **概要**: 正確で複雑かつ多様なツール学習データを生成する自動エージェンティックパイプライン。自己進化合成プロセスにより26,507の多様なAPIからなるAPIプールを構築。

### 4. Natural Language Tools: A Natural Language Approach to Tool Calling in Large Language Agents
- **URL**: https://arxiv.org/abs/2510.14453
- **発表時期**: 2025年10月
- **概要**: LLMのツール呼び出し能力を自然言語ベースで強化するアプローチ。エージェンティックアーキテクチャにおいて外部システムとのインターフェースを実現。

### 5. ToolRegistry: A Protocol-Agnostic Tool Management Library for Function-Calling LLMs
- **URL**: https://arxiv.org/abs/2507.10593
- **発表時期**: 2025年7月
- **概要**: Anthropicが2024年11月に導入したMCPを踏まえ、ツールプロバイダーとLLM/エージェント開発者間のインターフェースを標準化するプロトコル非依存ツール管理ライブラリ。

### 6. Model Context Protocol (MCP): Landscape, Security Threats, and Future Research Directions
- **URL**: https://arxiv.org/abs/2503.23278
- **発表時期**: 2025年3月
- **概要**: MCPの全体像、セキュリティ脅威、将来の研究方向性を包括的に分析。LLMと外部ツール、データソース、プロンプト間の接続を標準化するオープンプロトコルとしてのMCPを評価。

### 7. ScaleMCP: Dynamic and Auto-Synchronizing Model Context Protocol Tools for LLM Agents
- **URL**: https://arxiv.org/abs/2505.06416
- **発表時期**: 2025年5月
- **概要**: ツール適用型RAGの技術進歩により、LLMエージェントが大量のツールへ効率的にスケールすることを可能にするMCPの拡張。

### 8. Help or Hurdle? Rethinking Model Context Protocol-Augmented Large Language Models
- **URL**: https://arxiv.org/abs/2508.12566
- **発表時期**: 2025年8月
- **概要**: MCP拡張LLMの再評価。LLMがツールの必要性をどのように認識し、呼び出しを実行し、取得情報を統合するかという基本的行動側面を分析。

### 9. Improving Large Language Models Function Calling and Interpretability via Guided-Structured Templates
- **URL**: https://arxiv.org/abs/2509.18076
- **発表時期**: 2025年9月
- **概要**: 事前実行型の構造化推論とテンプレートベースのアプローチによるFunction Calling精度の向上手法。

---

## 特許

### 1. Agentic Orchestration (UiPath)
- **特許番号**: US12412138B1
- **URL**: https://patents.google.com/patent/US12412138B1/en
- **発行年**: 2024-2025年
- **概要**: AIエージェント、RPAロボット、エージェンティックオーケストレーションプロセス（AOP）のための自己修復機能付きオーケストレーションソリューション。メモリ、ナレッジベース、生成AIおよびLLM機能を備えたエージェントによる自然言語通信と意思決定。

### 2. Techniques for Automating Tasks Using LLMs and Software Agents (Ally Financial)
- **特許番号**: US20250315626A1
- **URL**: https://patents.google.com/patent/US20250315626A1/en
- **公開年**: 2025年
- **概要**: LLMとソフトウェアエージェントを使用したタスク自動化技術。データセットを処理してLLMを用いたタスクの特定・実行を実現。

### 3. Generative AI Driven Software Fixing (Veracode)
- **特許番号**: US12229040B2
- **URL**: https://patents.google.com/patent/US12229040B2/en
- **発行年**: 2025年
- **概要**: トランスフォーマーベースのLLMを使用して欠陥のあるプログラムコードにパッチを適用する生成AIコード修正パイプライン。LLMの特定の計算タスクへのツール的活用。

### 4. Systems for Interacting with a Large Language Model
- **特許番号**: US12051205B1
- **URL**: https://patents.google.com/patent/US12051205B1/en
- **発行年**: 2024年
- **概要**: LLMとのインタラクションシステム。外部ツールやAPIとの統合を含むLLMベースのシステムアーキテクチャ。

### 5. Optimizing Behavior and Deployment of Large Language Models
- **特許番号**: US12423064B2
- **URL**: https://patents.google.com/patent/US12423064B2
- **発行年**: 2025年
- **概要**: LLMの動作最適化とデプロイメント。ツール使用を含むLLMの行動パターンの最適化手法。

### 6. Prompt Engineering for AI Assisted Industrial Automation System Design (Rockwell Automation)
- **特許番号**: US20250005224A1
- **URL**: https://patents.google.com/patent/US20250005224A1/en
- **公開年**: 2025年
- **概要**: 産業自動化環境にAIを統合するプロンプトエンジニアリングサービス。産業自動化ワークフローで訓練されたLLM機能を活用したツール連携。

---

## まとめ

ツール使用・API連携クラスタでは、2024-2026年にMCP（Model Context Protocol）の登場が大きな転換点となっている。Anthropicが2024年11月に導入したMCPを中心に、ツール統合の標準化、セキュリティ分析、スケーラビリティ向上に関する研究が急速に進展。特許面では、UiPath、Ally Financial、Veracode等がLLMのツール使用に関する自動化・オーケストレーション技術を出願している。
