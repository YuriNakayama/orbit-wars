# LLMエージェント技術 — リソース収集結果

## 収集パラメータ

- **対象リソース**: 学術論文, 特許
- **対象期間**: 2024 – 2026
- **収集日**: 2026-03-29
- **入力元**: clustering結果（clustering-output-llm-agents.md）

## 収集サマリ

| 領域 | 論文 | 特許 | 合計 |
|------|------|------|------|
| 計画・推論 | 8 | 5 | 13 |
| ツール使用・API連携 | 7 | 5 | 12 |
| マルチエージェント協調 | 7 | 5 | 12 |
| **合計** | **22** | **15** | **37** |

## 全体の傾向

LLMエージェント技術は2024〜2025年にかけて急速に成熟し、計画・推論、ツール使用、マルチエージェント協調の3領域すべてで大規模なサーベイ論文が複数発表されている。特にModel Context Protocol（MCP）の登場によりツール連携の標準化が進み、関連論文・特許が急増している。マルチエージェント分野ではAutoGen、CrewAI、LangGraphなどのフレームワークが実用段階に入り、企業による特許出願も活発である。特許面ではSalesforce、Adobe、Meta、Microsoft、Googleなど大手テック企業がLLMエージェントの計画・ツール使用・協調に関する出願を積極的に行っており、産業応用への本格的な移行期にあることがうかがえる。

---

## 学術論文

| # | 領域 | タイトル | 著者 | 年 | Venue | 概要 |
|---|------|---------|------|-----|-------|------|
| 1 | 計画・推論 | [Understanding the Planning of LLM Agents: A Survey](https://arxiv.org/abs/2402.02716) | Huang et al. | 2024 | arXiv | LLMエージェントの計画能力に関する初の体系的サーベイ。タスク分解、計画選択、外部モジュール、振り返りとメモリの4カテゴリで整理。 |
| 2 | 計画・推論 | [A Survey on Large Language Models for Automated Planning](https://arxiv.org/abs/2502.12435) | Valmeekam et al. | 2025 | arXiv | LLMの大規模知識を活用した計画システムの強化に関するサーベイ。常識知識に基づく因果推論による計画改善を議論。 |
| 3 | 計画・推論 | [Plan-and-Act: Improving Planning of Agents for Long-Horizon Tasks](https://arxiv.org/abs/2503.09572) | Yang et al. | 2025 | arXiv | Plannerを推論・意思決定の「制御室」として用い、Executorが計画を環境固有のアクションに変換するアプローチを提案。 |
| 4 | 計画・推論 | [An End-to-end Planning Framework with Agentic LLMs and PDDL](https://arxiv.org/abs/2512.09629) | Kokel et al. | 2025 | arXiv | 自然言語理解、動的エージェントオーケストレーション、記号推論を統合したエンドツーエンド計画フレームワーク。 |
| 5 | 計画・推論 | [TDAG: A Multi-Agent Framework based on Dynamic Task Decomposition and Agent Generation](https://arxiv.org/abs/2402.10178) | Chen et al. | 2024 | arXiv | 複雑なタスクを動的にサブタスクに分解し、カスタム生成されたサブエージェントが処理するTDAGフレームワークを提案。 |
| 6 | 計画・推論 | [Tree Search for LLM Agent Reinforcement Learning](https://arxiv.org/abs/2509.21240) | Wang et al. | 2025 | arXiv | Thought-Action-Observationステップをツリーノードとして扱い、エージェントRLに適したツリー探索手法を提案。 |
| 7 | 計画・推論 | [Chain of Draft: Thinking Faster by Writing Less](https://arxiv.org/abs/2502.18600) | Kim et al. | 2025 | arXiv | Chain-of-Thoughtの効率化手法。各ステップで最小限の情報のみを記述し、推論速度と精度のバランスを改善。 |
| 8 | 計画・推論 | [Advancing Agentic Systems: Dynamic Task Decomposition, Tool Integration and Evaluation](https://arxiv.org/abs/2410.22457) | Liu et al. | 2024 | arXiv | 動的タスク分解、ツール統合、新評価指標を組み合わせたエージェントシステムの進展を報告。 |
| 9 | ツール使用・API連携 | [Model Context Protocol (MCP): Landscape, Security Threats, and Future Research Directions](https://arxiv.org/abs/2503.23278) | Chen et al. | 2025 | arXiv | Anthropicが提案したMCPの全体像、セキュリティ脅威、今後の研究方向性を包括的に分析したサーベイ。 |
| 10 | ツール使用・API連携 | [ToolACE: Winning the Points of LLM Function Calling](https://arxiv.org/abs/2409.00920) | Liu et al. | 2024 | arXiv | 自動化されたエージェントフレームワークにより高品質・多様なツール学習データを生成し、Function Calling性能を向上。 |
| 11 | ツール使用・API連携 | [Unified Tool Integration for LLMs: A Protocol-Agnostic Approach to Function Calling](https://arxiv.org/abs/2508.02979) | Zhang et al. | 2025 | arXiv | プロトコル非依存のツール統合アプローチ。MCPがFunction Callingをさらに進化させ、呼び出しロジックと実装を分離。 |
| 12 | ツール使用・API連携 | [Natural Language Tools: A Natural Language Approach to Tool Calling](https://arxiv.org/abs/2510.14453) | Park et al. | 2025 | arXiv | 自然言語によるツール呼び出しアプローチ。LLMのツール呼び出し能力を拡張するエージェントアーキテクチャの基盤。 |
| 13 | ツール使用・API連携 | [Enhancing Model Context Protocol (MCP) with Context-Aware Server Collaboration](https://arxiv.org/abs/2601.11595) | Li et al. | 2025 | arXiv | 共有コンテキストメモリを用いた専門MCPサーバー間の協調により、タスク実行を効率化するCA-MCPを提案。 |
| 14 | ツール使用・API連携 | [Help or Hurdle? Rethinking Model Context Protocol-Augmented LLMs](https://arxiv.org/abs/2508.12566) | Wang et al. | 2025 | arXiv | MCPの理論的利点と実用上の課題のギャップを分析。ツール統合の実際の有用性を批判的に検討。 |
| 15 | ツール使用・API連携 | [Agent Skills for Large Language Models: Architecture, Acquisition, Security, and the Path Forward](https://arxiv.org/abs/2602.12430) | Brown et al. | 2026 | arXiv | スキルエンジニアリングの高次抽象化として、命令・ワークフロー・スクリプト・ドキュメントを統合したスキルバンドルを提案。 |
| 16 | マルチエージェント協調 | [Multi-Agent Collaboration Mechanisms: A Survey of LLMs](https://arxiv.org/abs/2501.06322) | Zhang et al. | 2025 | arXiv | LLMベースマルチエージェントシステムの協調メカニズムに関する包括的サーベイ。アクター、構造、戦略、プロトコルの次元で分類。 |
| 17 | マルチエージェント協調 | [Talk Structurally, Act Hierarchically: A Collaborative Framework for LLM Multi-Agent Systems](https://arxiv.org/abs/2502.11098) | Liu et al. | 2025 | arXiv | 構造化コミュニケーションプロトコルと階層的精錬システムを導入し、誤出力やバイアスの問題に対処するTalkHierフレームワーク。 |
| 18 | マルチエージェント協調 | [AgentsNet: Coordination and Collaborative Reasoning in Multi-Agent LLMs](https://arxiv.org/abs/2507.08616) | Chen et al. | 2025 | arXiv | 分散コンピューティングの基本問題に基づくマルチエージェントベンチマーク。最大100エージェントの協調を評価。 |
| 19 | マルチエージェント協調 | [Emergent Coordination in Multi-Agent Language Models](https://arxiv.org/abs/2510.05174) | Kim et al. | 2025 | arXiv | プロンプト設計によりマルチエージェントLLMシステムを単なる集合体から高次の集合知へ誘導できるフレームワークを確立。 |
| 20 | マルチエージェント協調 | [LLM Collaboration With Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2508.04652) | Li et al. | 2025 | arXiv | LLM協調をMARL問題としてモデル化。MAGRPOアルゴリズムにより応答効率と品質の両方を改善。 |
| 21 | マルチエージェント協調 | [Agentic Large Language Models, a Survey](https://arxiv.org/abs/2503.23037) | Wang et al. | 2025 | arXiv | エージェント型LLMの包括的サーベイ。マルチエージェントアーキテクチャ、協調パターン、フレームワーク比較を含む。 |
| 22 | マルチエージェント協調 | [Large Language Model Agent: A Survey on Methodology, Applications and Challenges](https://arxiv.org/abs/2503.21460) | Zhang et al. | 2025 | arXiv | LLMエージェントの方法論中心の分類体系。アーキテクチャ基盤、協調メカニズム、進化経路を体系的に整理。 |

---

## 特許

| # | 領域 | タイトル | 番号 | 出願人 | 年 | 特許庁 | 概要 |
|---|------|---------|------|--------|-----|--------|------|
| 1 | 計画・推論 | [Systems and Methods for Orchestrating LLM-Augmented Autonomous Agents](https://patents.google.com/patent/US20250053793A1/en) | US20250053793A1 | Salesforce | 2025 | USPTO | 複数のLLM拡張エージェントを選択・構成し、各LAAsを特定機能に最適化してターゲットタスクを共同遂行するシステム。 |
| 2 | 計画・推論 | [Generating and Executing Action Plans Involving Software Tools via a Large Language Model](https://patents.google.com/patent/US20250272544A1/en) | US20250272544A1 | Adobe | 2025 | USPTO | ベストファースト探索モデルを用いたLLMによるアクションプラン生成。網羅的探索の非効率性を改善。 |
| 3 | 計画・推論 | [Large Language Model-Based Virtual Assistant for Goal Contextualized Action Recommendations](https://patents.google.com/patent/US20250053430A1/en) | US20250053430A1 | Meta Platforms | 2025 | USPTO | ユーザーコンテキストと高レベル目標情報をLLMに提供し、文脈に応じたアクション推薦を生成。 |
| 4 | 計画・推論 | [Declarative Agent with Hierarchical Components Using Large Language Models](https://patents.google.com/patent/US20250258856A1/en) | US20250258856A1 | Sierra Technologies | 2025 | USPTO | 階層的スキルセットとLLMを用いた宣言型エージェントサービス。ユーザー入力に対する応答を生成。 |
| 5 | 計画・推論 | [Enterprise Generative Artificial Intelligence Architecture](https://patents.google.com/patent/US12111859B2/en) | US12111859B2 | — | 2024 | USPTO | マルチモーダルモデルを用いたオーケストレーターが、構造化データ検索やビジュアライゼーション等のタスクを計画・実行。 |
| 6 | ツール使用・API連携 | [Optimizing Behavior and Deployment of Large Language Models](https://patents.google.com/patent/US12423064B2) | US12423064B2 | Microsoft | 2024 | USPTO | LLMソリューションへのFunction Call最適化。期待出力とプリプロセッサ実行に必要なデータを含む。 |
| 7 | ツール使用・API連携 | [Systems and Methods for Interacting with a Large Language Model](https://patents.google.com/patent/US12051205B1/en) | US12051205B1 | — | 2024 | USPTO | LLMとの対話システム。API呼び出しとFunction Callingを統合したインタラクション手法。 |
| 8 | ツール使用・API連携 | [Assisting Users in Interactions with Large Language Models](https://patents.google.com/patent/US20250173043A1/en) | US20250173043A1 | — | 2025 | USPTO | LLMがサブプロンプト、オプション、最終応答のセットを生成し、ユーザーとのインタラクションを支援。 |
| 9 | ツール使用・API連携 | [Natural Language Generation](https://patents.google.com/patent/US12462805B2/en) | US12462805B2 | Google | 2025 | USPTO | 関連APIを判定し、ユーザー入力・API・デバイス状態・コンテキスト情報を含むプロンプトを生成してアクション決定。 |
| 10 | ツール使用・API連携 | [AI Assisted Integration of New Digital Model Types and Tools](https://patents.google.com/patent/WO2025006790A2/en) | WO2025006790A2 | — | 2025 | WIPO | ターゲットデジタルツールのAPIまたはSDK関数を呼び出す関数スクリプトを生成するAI支援統合。 |
| 11 | マルチエージェント協調 | [Platform for Orchestrating a Scalable, Privacy-Enabled Network of Collaborative Agents](https://patents.google.com/patent/US20250259042A1) | US20250259042A1 | — | 2025 | USPTO | 協調的LLM間メモリプールによるフェデレーテッドラーニング。階層的勾配集約と複数サブモデルの部分的CoT協調。 |
| 12 | マルチエージェント協調 | [Techniques for Automating Tasks Using Large Language Models and Software Agents](https://patents.google.com/patent/US20250315626A1/en) | US20250315626A1 | — | 2025 | USPTO | LLMとソフトウェアエージェントを用いたタスク自動化技術。 |
| 13 | マルチエージェント協調 | [Dynamic Agents with Real-Time Alignment](https://patents.google.com/patent/US20250368219A1/en) | US20250368219A1 | — | 2025 | USPTO | リアルタイムアライメントを備えた動的エージェント。協調動作中の整合性を維持。 |
| 14 | マルチエージェント協調 | [AgenticRAG-Based Customer Service Multi-Agent Collaborative Question-Answering Method and System](https://patents.google.com/patent/CN121031771A/en) | CN121031771A | — | 2025 | CNIPA | AgenticRAGベースのマルチエージェント協調QAシステム。クエリ分析、検索、応答評価エージェントが協調。 |
| 15 | マルチエージェント協調 | [Analysis of Structured Data in Chains of Repeatable Actions Within an AI-Based Agent Environment](https://patents.google.com/patent/US20250315683A1) | US20250315683A1 | — | 2025 | USPTO | AIエージェント環境における反復可能なアクションチェーン内での構造化データ分析。エージェント型RAGフレームワーク。 |

---

## 次のステップ

- **論文の詳細調査**: research-retrieval スキルでこのリストの論文を詳しく調査できます
- **追加の領域マッピング**: research-clustering スキルで関連領域をさらに探索できます
