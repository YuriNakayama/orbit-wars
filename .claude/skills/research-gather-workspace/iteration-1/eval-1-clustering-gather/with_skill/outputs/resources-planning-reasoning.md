# 計画・推論 — リソース一覧

## 領域概要

LLMの推論能力を活用し、複雑なタスクを段階的に分解・計画・実行するアプローチ。Chain-of-Thought、ReAct、Tree-of-Thoughtsなどの推論戦略が含まれ、OpenAI o1やDeepSeek R1などの推論特化モデルの登場により研究が加速している。

**関連キーワード**: `chain-of-thought`, `ReAct`, `tree-of-thoughts`, `task decomposition`, `planning`, `reasoning`, `self-reflection`, `step-by-step`

---

## 学術論文

| # | タイトル | 著者 | 年 | Venue | 概要 |
|---|---------|------|-----|-------|------|
| 1 | [Understanding the Planning of LLM Agents: A Survey](https://arxiv.org/abs/2402.02716) | Huang et al. | 2024 | arXiv | LLMベースエージェントの計画能力に関する体系的サーベイ。タスク分解、計画選択、外部モジュール、振り返りとメモリの4カテゴリに分類 |
| 2 | [A Survey on Large Language Models for Automated Planning](https://arxiv.org/abs/2502.12435) | 著者複数 | 2025 | arXiv | LLMの自動計画における能力と限界を包括的にカバーするサーベイ |
| 3 | [Plan-and-Act: Improving Planning of Agents for Long-Horizon Tasks](https://arxiv.org/abs/2503.09572) | 著者複数 | 2025 | arXiv | 長期的タスクにおける計画と実行を分離するアプローチを提案 |
| 4 | [Agentic Artificial Intelligence: Architectures, Taxonomies, and Evaluation of LLM Agents](https://arxiv.org/abs/2601.12560) | 著者複数 | 2026 | arXiv | LLMエージェントのアーキテクチャ、分類、評価に関する包括的調査。メモリ、ツール使用、環境フィードバックを統合した認知コントローラとしてのLLMを記述 |
| 5 | [Self-Reflection in LLM Agents: Effects on Problem-Solving Performance](https://arxiv.org/abs/2405.06682) | 著者複数 | 2024 | arXiv | LLMエージェントにおける自己振り返りが問題解決能力に与える影響を実証的に分析 |
| 6 | [Why Reasoning Fails to Plan: A Planning-Centric Analysis](https://arxiv.org/abs/2601.22311) | 著者複数 | 2026 | arXiv | 推論と計画の概念的ギャップを分析。Chain-of-Thoughtのような標準的推論はローカルスコアに基づく貪欲方策に過ぎないと指摘 |
| 7 | [Agentic Large Language Models: A Survey](https://arxiv.org/abs/2503.23037) | 著者複数 | 2025 | arXiv | エージェント型LLMを推論・行動・対話の3カテゴリに整理した包括的サーベイ |
| 8 | [An End-to-end Planning Framework with Agentic LLMs and PDDL](https://arxiv.org/abs/2512.09629) | 著者複数 | 2025 | arXiv | 自然言語理解、動的エージェントオーケストレーション、記号推論、計画解釈を統合した初のエンドツーエンド計画フレームワーク |

### 注目論文

- **Understanding the Planning of LLM Agents: A Survey** — LLMエージェントの計画能力を体系的に整理した重要なサーベイ。タスク分解、計画選択、外部モジュール活用、振り返りの4つの軸で既存研究を分類しており、この領域の全体像を把握する出発点として最適。
- **Why Reasoning Fails to Plan** — 推論と計画の本質的な違いを分析した洞察に富む論文。Chain-of-Thoughtなどの推論手法が計画タスクで失敗する理由を理論的に解明しており、今後の研究方向に大きな示唆を与える。
- **Plan-and-Act** — 計画と実行の分離というシンプルかつ効果的なアプローチを提案し、長期的タスクにおけるエージェントの性能向上を実証。

---

## 特許

| # | タイトル | 番号 | 出願人 | 年 | 特許庁 | 概要 |
|---|---------|------|--------|-----|--------|------|
| 1 | [Large language model-based virtual assistant for high-level goal contextualized action recommendations](https://patents.google.com/patent/US20250053430A1/en) | US20250053430A1 | — | 2025 | USPTO | LLMベースの仮想アシスタントが高レベルの目標に基づいてアクション推奨を生成する |
| 2 | [Chain of thought reasoning for ASR](https://patents.google.com/patent/US20250118293A1/en) | US20250118293A1 | — | 2025 | USPTO | Chain-of-Thought推論を音声認識に適用し、会話の論理的関係を活用する |
| 3 | [Systems and methods for generating code using language models](https://patents.google.com/patent/US20240020096A1/en) | US20240020096A1 | — | 2024 | USPTO | 言語モデルを用いたコード生成システムで、計画的推論プロセスを組み込む |
| 4 | [Computer implemented methods for automated analysis using LLM](https://patents.google.com/patent/US20230259705A1/en) | US20230259705A1 | — | 2024 | USPTO | 直接検索、計算ユニットの使用、推論の3つのアプローチでデータ分析を行うLLM手法 |
| 5 | [Aligning LMMs/LLMs using post training with domain-specific principles](https://patents.google.com/patent/WO2025188958A1/en) | WO2025188958A1 | — | 2025 | WIPO | ドメイン固有の原則を用いたポストトレーニングによるLLMの推論能力整合化 |

### 注目特許

- **US20250053430A1** — 高レベル目標からの行動計画生成という、LLMエージェントの計画能力を直接的に商用化する特許。目標の文脈化とアクション推奨の組み合わせが実用性の高いアプローチを示している。
- **US20250118293A1** — Chain-of-Thought推論をASRに応用した特許で、推論技術の応用範囲の広がりを示す好例。

---

## 次のステップ

- **論文の詳細調査**: research-papers スキルでこのリストの論文を詳しく調査できます
- **追加の領域マッピング**: research-clustering スキルで関連領域をさらに探索できます
