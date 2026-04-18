# クラスタ3: マルチエージェント協調 - リソース収集結果

## 概要

複数のLLMエージェントが役割分担しながら協力してタスクを遂行するフレームワークに関する学術論文および特許を収集した。議論、投票、分業などの協調戦略、AutoGen・CrewAI等のフレームワークを含む。

**対象期間**: 2024年 - 2026年
**収集日**: 2026-03-29

---

## 学術論文

### 1. Multi-Agent Collaboration Mechanisms: A Survey of LLMs
- **URL**: https://arxiv.org/abs/2501.06322
- **発表時期**: 2025年1月
- **概要**: LLMベースのマルチエージェントシステム（MAS）における協調メカニズムの包括的サーベイ。協調の次元（アクター、タイプ、構造、戦略、協調プロトコル）に基づくフレームワークを提示。協力、競争、共競（coopetition）などのタイプを体系化。

### 2. LLM Collaboration With Multi-Agent Reinforcement Learning (MAGRPO)
- **URL**: https://arxiv.org/abs/2508.04652
- **発表時期**: 2025年8月
- **概要**: LLM協調を協調型MARL問題としてモデル化し、分散部分観測マルコフ決定過程（Dec-POMDP）として定式化。マルチターン設定でLLMを訓練するMulti-Agent GRPO（MAGRPO）を提案。

### 3. Emergent Coordination in Multi-Agent Language Models
- **URL**: https://arxiv.org/abs/2510.05174
- **発表時期**: 2025年10月（2026年3月更新）
- **概要**: マルチエージェントLLMシステムがプロンプト設計により単なる集合体から高次の集合体へ誘導可能であることを実証。創発的な協調行動の分析。

### 4. AgentsNet: Coordination and Collaborative Reasoning in Multi-Agent LLMs
- **URL**: https://arxiv.org/abs/2507.08616
- **発表時期**: 2025年7月
- **概要**: 分散コンピューティングの基本問題に基づくマルチエージェントベンチマーク。既存のベンチマークが2-5エージェントに限定される中、最大100エージェントまでスケール可能な評価スイートを提案。

### 5. Latent Collaboration in Multi-Agent Systems
- **URL**: https://arxiv.org/abs/2511.20639
- **発表時期**: 2025年11月
- **概要**: LLMの連続潜在空間をモデル間の情報交換のための新しい「モデル言語」として活用する潜在的協調の探索。

### 6. LLM Multi-Agent Systems: Challenges and Open Problems
- **URL**: https://arxiv.org/abs/2402.03578
- **発表時期**: 2024年2月（2026年1月更新）
- **概要**: マルチエージェントシステムにおけるタスク割り当ての最適化、反復的議論を通じたロバストな推論、複雑な文脈情報管理、メモリ管理の強化に関する課題と未解決問題を議論。

### 7. Agentic AI Frameworks: Architectures, Protocols, and Design Challenges
- **URL**: https://arxiv.org/abs/2508.10146
- **発表時期**: 2025年8月
- **概要**: CrewAI、LangGraph、AutoGen、MetaGPTなどの主要エージェンティックAIフレームワークのアーキテクチャ、メモリ、通信、ガードレール、サービスコンピューティングサポートを分析・比較。

### 8. Towards Effective GenAI Multi-Agent Collaboration: Design and Evaluation for Enterprise Applications
- **URL**: https://arxiv.org/abs/2412.05449
- **発表時期**: 2024年12月
- **概要**: エンタープライズアプリケーション向けの効果的な生成AIマルチエージェント協調の設計と評価。実用的な企業環境での適用を重視。

### 9. LLM-Based Human-Agent Collaboration and Interaction Systems: A Survey
- **URL**: https://arxiv.org/abs/2505.00753
- **発表時期**: 2025年5月
- **概要**: LLMベースの人間-エージェント協調とインタラクションシステムに関するサーベイ。エージェント間だけでなく人間との協調も含めた包括的レビュー。

---

## 特許

### 1. Platform for Orchestrating a Scalable, Privacy-Enabled Network of Collaborative and Negotiating Agents (QOMPLX LLC)
- **特許番号**: US20250259042A1
- **URL**: https://patents.google.com/patent/US20250259042A1
- **公開年**: 2025年
- **概要**: 特化型AIエージェントのネットワークを協調させるプラットフォーム。トークンベースの通信とリアルタイム結果ストリーミングを通じたセキュアな協調を実現。

### 2. Systems and Methods for Orchestrating LLM-Augmented Autonomous Agents (Salesforce)
- **特許番号**: US20250053793A1
- **URL**: https://patents.google.com/patent/US20250053793A1/en
- **公開年**: 2025年
- **概要**: LLM拡張自律エージェント（LAA）のオーケストレーション。複数のLAAが共同でターゲットタスクを実行するアーキテクチャを構築し、コントローラーが利用可能なLAAプールから最適なエージェントを選択。

### 3. Human-AI Collaborative Prompt Engineering (IBM)
- **特許番号**: US20250292093A1
- **URL**: https://patents.google.com/patent/US20250292093A1
- **公開年**: 2025年
- **概要**: LLM向けタスク記述プロンプトの最適化技術。候補プロンプトをLLMに適用し、応答を評価する協調的プロンプトエンジニアリング。

### 4. Enterprise Generative AI Architecture (C3.AI)
- **特許番号**: US12111859B2
- **URL**: https://patents.google.com/patent/US12111859B2/en
- **発行年**: 2024年
- **概要**: オーケストレーターがLLMを含むマルチモーダルモデルを使用してプロンプトを複数エージェントへの指示系列に分解・管理するシステム。

### 5. Dynamic Agents with Real-time Alignment (Microsoft)
- **特許番号**: US20250368219A1
- **URL**: https://patents.google.com/patent/US20250368219A1/en
- **公開年**: 2025年
- **概要**: 自動化エージェントを使用して計画を生成・実行するマルチエージェントシステム。リアルタイムアライメントにより複雑なマルチステップタスクにおけるエージェント間の協調を強化。

### 6. Method and System for Multi-Level Artificial Intelligence Supercomputer Design
- **特許番号**: US12299017B2
- **URL**: https://patents.google.com/patent/US12299017B2
- **発行年**: 2025年
- **概要**: マルチレベルAIスーパーコンピュータ設計。複数のAIモデルが階層的に協調動作するシステムアーキテクチャ。

---

## まとめ

マルチエージェント協調クラスタでは、2024-2026年にサーベイ論文の充実、MARL（マルチエージェント強化学習）との融合、大規模ベンチマークの開発、潜在空間を活用した新しい協調手法など、研究が急速に多様化している。フレームワーク比較研究（AutoGen、CrewAI、LangGraph、MetaGPT）も活発に行われている。特許面では、QOMPLX、Salesforce、IBM、Microsoft等がマルチエージェントオーケストレーションに関する特許を出願しており、エンタープライズ向けの実用化が進んでいる。
