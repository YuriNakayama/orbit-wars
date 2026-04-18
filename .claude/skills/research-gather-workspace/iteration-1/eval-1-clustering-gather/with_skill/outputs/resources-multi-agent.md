# マルチエージェント協調 — リソース一覧

## 領域概要

複数のLLMエージェントが役割分担しながら協力してタスクを遂行するフレームワーク。議論、投票、分業などの協調戦略が研究されている。AutoGen、CrewAI、LangGraphなどのフレームワークが実用段階に達し、1000エージェント規模のスケーリングや潜在空間での協調など新しいパラダイムも探索されている。

**関連キーワード**: `multi-agent`, `agent communication`, `role assignment`, `collaborative agents`, `debate`, `agent framework`

---

## 学術論文

| # | タイトル | 著者 | 年 | Venue | 概要 |
|---|---------|------|-----|-------|------|
| 1 | [Multi-Agent Collaboration Mechanisms: A Survey of LLMs](https://arxiv.org/abs/2501.06322) | 著者複数 | 2025 | arXiv | LLMベースマルチエージェントの協調メカニズムに関する包括的サーベイ。協調の種類（協力・競争・共競争）、構造（P2P・集中型・分散型）、戦略、プロトコルを体系化 |
| 2 | [Large Language Model based Multi-Agents: A Survey of Progress and Challenges](https://arxiv.org/abs/2402.01680) | 著者複数 | 2024 | arXiv | LLMベースマルチエージェントシステムの進歩と課題に関するサーベイ。複雑な問題解決とワールドシミュレーションでの成果を概観 |
| 3 | [Scaling Large Language Model-based Multi-Agent Collaboration](https://arxiv.org/abs/2406.07155) | 著者複数 | 2024 | arXiv | 1000エージェント以上の協調をサポートし、不規則なトポロジーが規則的なものを上回ることを実証。ニューラルスケーリング則との類似性を探究 |
| 4 | [AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation](https://arxiv.org/abs/2308.08155) | Wu et al. | 2024 | ICLR 2024 | Microsoftが開発したマルチエージェント会話フレームワーク。カスタマイズ可能なエージェント間の会話を通じた柔軟なタスク遂行を実現 |
| 5 | [Towards Effective GenAI Multi-Agent Collaboration: Design and Evaluation for Enterprise Applications](https://arxiv.org/abs/2412.05449) | 著者複数 | 2024 | arXiv | エンタープライズアプリケーション向けマルチエージェント協調の設計と評価に関する研究 |
| 6 | [AgentsNet: Coordination and Collaborative Reasoning in Multi-Agent LLMs](https://arxiv.org/abs/2507.08616) | 著者複数 | 2025 | arXiv | 分散コンピューティング問題に基づくマルチエージェントベンチマーク。最大100エージェントの協調を評価（既存ベンチマークの2-5エージェントを大幅に超える） |
| 7 | [LLM Collaboration With Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2508.04652) | 著者複数 | 2025 | arXiv | LLMの協調を協調的マルチエージェント強化学習（MARL）問題としてモデル化。MAGRPOアルゴリズムを提案 |
| 8 | [Latent Collaboration in Multi-Agent Systems](https://arxiv.org/abs/2511.20639) | 著者複数 | 2025 | arXiv | LatentMASフレームワーク：エージェントが潜在空間で推論・通信し、最終回答のみテキストでデコードする新しい協調パラダイム |

### 注目論文

- **Multi-Agent Collaboration Mechanisms: A Survey of LLMs** — マルチエージェント協調の全体像を把握するための必読サーベイ。協調の5つの次元（アクター、種類、構造、戦略、プロトコル）で体系的に整理しており、この領域の研究マップとして最適。
- **Scaling Large Language Model-based Multi-Agent Collaboration** — 1000エージェント規模のスケーリングに挑戦した先駆的研究。ニューラルスケーリング則との類似性を検証し、大規模マルチエージェント協調の可能性を示した。
- **AutoGen** — ICLR 2024に採択されたMicrosoftのマルチエージェントフレームワーク。CrewAI、LangGraphと並ぶ三大フレームワークの一つとして広く採用されている。

---

## 特許

| # | タイトル | 番号 | 出願人 | 年 | 特許庁 | 概要 |
|---|---------|------|--------|-----|--------|------|
| 1 | [Systems and methods for orchestrating LLM-augmented autonomous agents](https://patents.google.com/patent/US20250053793A1/en) | US20250053793A1 | — | 2025 | USPTO | LLM拡張型自律エージェントのアーキテクチャ構築。各エージェントをファインチューニングまたはプロンプト最適化で特化させ、コントローラが最適なエージェントを選択 |
| 2 | [Platform for orchestrating a scalable, privacy-enabled network of collaborative and negotiating agents](https://patents.google.com/patent/US20250259042A1) | US20250259042A1 | — | 2025 | USPTO | プライバシーを保護しながらスケーラブルな協調エージェントネットワークをオーケストレーション。協調的LLM間メモリプールによる連合学習 |
| 3 | [Enterprise generative AI architecture](https://patents.google.com/patent/US12111859B2/en) | US12111859B2 | C3.ai | 2024 | USPTO | エンタープライズ生成AIアーキテクチャ。オーケストレータエージェントが複数のエージェントとツールを監督・制御 |
| 4 | [Orchestration of parallel generative AI pipelines](https://patents.google.com/patent/US12039263B1/en) | US12039263B1 | McKinsey & Company | 2024 | USPTO | 並列LLMクエリを用いた生成AIパイプラインのオーケストレーション |
| 5 | [Adaptive multi-agent cooperative computation and inference](https://patents.google.com/patent/US12169792B2/en) | US12169792B2 | — | 2024 | USPTO | 適応型マルチエージェント協調計算・推論システム |

### 注目特許

- **US20250053793A1** — LLM拡張型自律エージェントのオーケストレーションに関する特許。各エージェントの専門化とコントローラによる動的選択という、AutoGenやCrewAIに類似したアーキテクチャを商用化する試み。
- **US12111859B2（C3.ai）** — エンタープライズ向けマルチエージェントAIアーキテクチャの特許。オーケストレータパターンの商用実装として注目される。

---

## 次のステップ

- **論文の詳細調査**: research-papers スキルでこのリストの論文を詳しく調査できます
- **追加の領域マッピング**: research-clustering スキルで関連領域をさらに探索できます
