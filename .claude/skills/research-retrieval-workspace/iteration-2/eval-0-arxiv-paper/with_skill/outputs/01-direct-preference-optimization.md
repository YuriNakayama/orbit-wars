# Direct Preference Optimization: Your Language Model is Secretly a Reward Model

- **Link**: https://arxiv.org/abs/2305.18290
- **Authors**: Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn
- **Year**: 2023
- **Venue**: NeurIPS 2023 (37th Conference on Neural Information Processing Systems)
- **Type**: Academic Paper

## Abstract

While large-scale unsupervised language models (LMs) learn broad world knowledge and some reasoning skills, achieving precise control of their behavior is difficult due to the completely unsupervised nature of their training. Existing methods for gaining such steerability collect human labels of the relative quality of model generations, and fine-tune the unsupervised LM to align with these preferences, often with reinforcement learning from human feedback (RLHF). However, RLHF is a complex and often unstable procedure, first fitting a reward model that reflects the human preferences, and then fine-tuning the large unsupervised LM using reinforcement learning to maximize this estimated reward without drifting too far from the original model. In this paper, we introduce a new parameterization of the reward model in RLHF that enables extraction of the corresponding optimal policy in closed form, allowing us to solve the standard RLHF problem with only a simple classification loss. The resulting algorithm, which we call Direct Preference Optimization (DPO), is stable, performant, and computationally lightweight, eliminating the need for fitting a reward model, sampling from the LM during fine-tuning, or performing significant hyperparameter tuning. Our experiments show that DPO can fine-tune LMs to align with human preferences as well as or better than existing methods. Notably, fine-tuning with DPO exceeds PPO-based RLHF in ability to control sentiment of generations, and matches or improves response quality in summarization and single-turn dialogue while being substantially simpler to implement and train.

## Abstract（日本語訳）

大規模な教師なし言語モデル（LM）は幅広い世界知識やある程度の推論能力を学習するが、完全に教師なしで訓練されるため、その振る舞いを正確に制御することは困難である。既存の手法では、モデル生成物の相対的な品質に関する人間のラベルを収集し、人間のフィードバックからの強化学習（RLHF）を用いて教師なしLMをこれらの選好に合わせて微調整する。しかし、RLHFは複雑で不安定な手順であり、まず人間の選好を反映する報酬モデルを学習し、次にこの推定報酬を最大化しつつ元のモデルから大きく逸脱しないように大規模教師なしLMを強化学習で微調整する必要がある。本論文では、RLHFにおける報酬モデルの新しいパラメータ化を導入し、対応する最適方策を閉形式で抽出可能にすることで、標準的なRLHF問題を単純な分類損失のみで解くことを可能にした。得られたアルゴリズムをDirect Preference Optimization（DPO）と呼び、安定的で高性能かつ計算コストが低く、報酬モデルの学習、微調整中のLMからのサンプリング、大規模なハイパーパラメータ調整が不要である。実験により、DPOは既存手法と同等以上に人間の選好に合わせてLMを微調整できることを示した。特に、DPOによる微調整はPPOベースのRLHFを感情制御能力で上回り、要約や単一ターン対話の応答品質において同等以上の性能を達成しつつ、実装と訓練が大幅に簡単である。

## Overview

DPOは、RLHF（人間のフィードバックからの強化学習）のパイプラインを根本的に簡素化するアルゴリズムである。従来のRLHFでは、(1) 人間の選好データから報酬モデルを訓練し、(2) その報酬モデルを用いてPPO等の強化学習でポリシーを最適化する、という2段階のプロセスが必要であった。DPOの核心的な洞察は、Bradley-Terryモデルにおける報酬関数と最適ポリシーの間に閉形式の関係が存在し、これを利用することで報酬モデルの学習と強化学習の両方を省略し、単純な二値クロスエントロピー損失による教師あり学習のみでポリシーを直接最適化できるという点にある。DPOは感情制御（IMDb）、要約（TL;DR）、単一ターン対話（Anthropic HH）の3つのタスクにおいて、PPOベースのRLHFと同等以上の性能を、大幅に低い計算コストと実装の簡素さで達成した。

## Problem

DPOが解決する問題は以下の通り：

- **RLHFの複雑さ**: 従来のRLHFパイプラインは、報酬モデル訓練 → 強化学習（PPO）→ ポリシー最適化という多段階プロセスであり、各段階で大量の計算資源とハイパーパラメータ調整が必要
- **強化学習の不安定性**: PPOベースの最適化は不安定であり、学習率、クリッピング範囲、バッチサイズなどの多数のハイパーパラメータに敏感
- **計算コストの高さ**: 訓練中にLMからサンプリングする必要があり、大規模モデルでは特に計算コストが高い
- **報酬モデルの品質への依存**: ポリシーの最終的な性能が報酬モデルの品質に大きく依存し、報酬ハッキング（reward hacking）のリスクがある

## Proposed Method

**Direct Preference Optimization (DPO)**

DPOの手法は以下のように構成される：

### Core Idea

RLHFの標準的な目的関数において、報酬モデルと最適ポリシーの間の閉形式の関係を利用し、報酬モデルを介さずに人間の選好データからポリシーを直接最適化する。

### 理論的導出

#### Step 1: RLHF目的関数

標準的なRLHF目的関数は、報酬を最大化しつつ参照ポリシーからのKLダイバージェンス制約を課す：

$$\max_{\pi} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi(y|x)}[r(x, y)] - \beta \mathbb{D}_{\text{KL}}[\pi(y|x) \| \pi_{\text{ref}}(y|x)]$$

ここで $r(x,y)$ は報酬関数、$\pi_{\text{ref}}$ は参照ポリシー（SFTモデル）、$\beta$ はKL制約の強さを制御するパラメータ。

#### Step 2: 最適ポリシーの閉形式解

上記の目的関数の最適解は以下の閉形式で表される：

$$\pi_r(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta} r(x, y)\right)$$

ここで $Z(x) = \sum_y \pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta} r(x, y)\right)$ は分配関数。

#### Step 3: 報酬の再パラメータ化

最適ポリシーの式を報酬について解くと：

$$r(x, y) = \beta \log \frac{\pi_r(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$$

#### Step 4: Bradley-Terryモデルへの代入

人間の選好はBradley-Terryモデルで表現される：

$$p^*(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l))$$

ここで $\sigma$ はシグモイド関数、$y_w$ は選好される応答、$y_l$ は選好されない応答。上記の報酬再パラメータ化をBradley-Terryモデルに代入すると、分配関数 $Z(x)$ が相殺される。

#### Step 5: DPO損失関数

最終的なDPO損失関数：

$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$

### 既存手法との差異

| 特徴 | RLHF (PPO) | DPO |
|------|------------|-----|
| 報酬モデル訓練 | 必要 | 不要 |
| 強化学習 | PPO使用 | 不要（教師あり学習のみ） |
| 訓練中のサンプリング | 必要 | 不要 |
| ハイパーパラメータ | 多数（学習率、クリッピング等） | 少数（$\beta$のみが主要） |
| 計算コスト | 高い | 低い |
| 安定性 | 不安定になりやすい | 安定 |
| 理論的保証 | 近似的 | 厳密（Bradley-Terry仮定下） |

**Key Formulas**:

DPO損失の勾配は以下の直感的な形を持つ：

$$\nabla_\theta \mathcal{L}_{\text{DPO}} = -\beta \mathbb{E}_{(x,y_w,y_l)} \left[ \underbrace{\sigma(\hat{r}_\theta(x,y_l) - \hat{r}_\theta(x,y_w))}_{\text{重み付け項}} \left[ \nabla_\theta \log \pi_\theta(y_w|x) - \nabla_\theta \log \pi_\theta(y_l|x) \right] \right]$$

ここで $\hat{r}_\theta(x,y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$ は暗黙的報酬モデル。この勾配は、モデルが間違った選好を割り当てているサンプルに対して大きな更新を行う。

**Features**:

- 報酬モデルの明示的な訓練が不要
- 強化学習ループが不要で、標準的な教師あり学習パイプラインで実装可能
- $\beta$ パラメータにより参照ポリシーからの逸脱度を制御可能
- 暗黙的な報酬モデルを内包しており、「言語モデルは密かに報酬モデルである」

## Algorithm (Pseudocode)

```
Algorithm: Direct Preference Optimization (DPO)
Input:
  - D = {(x_i, y_w_i, y_l_i)}  // 選好データセット（プロンプト、選好応答、非選好応答）
  - π_ref                        // 参照ポリシー（SFTモデル）
  - β                            // KL制約パラメータ
  - α                            // 学習率
Output: π_θ  // 最適化されたポリシー

1. π_θ ← π_ref                  // 参照ポリシーで初期化
2. π_ref をフリーズ（勾配計算なし）  // 参照モデルは固定
3. for each batch B ⊂ D do
4.   for each (x, y_w, y_l) ∈ B do
5.     // 選好応答と非選好応答の対数確率比を計算
6.     r_w ← β * (log π_θ(y_w|x) - log π_ref(y_w|x))  // 選好応答の暗黙的報酬
7.     r_l ← β * (log π_θ(y_l|x) - log π_ref(y_l|x))  // 非選好応答の暗黙的報酬
8.     // DPO損失を計算（二値クロスエントロピー）
9.     loss ← -log σ(r_w - r_l)
10.  end for
11.  L ← mean(loss over batch)    // バッチ内の平均損失
12.  θ ← θ - α * ∇_θ L           // 勾配降下でパラメータ更新
13. end for
14. return π_θ
```

## Architecture / Process Flow

### 従来のRLHFパイプライン vs DPOパイプライン

```
【従来のRLHF】
人間の選好データ D
    ↓
[Phase 1: SFT]
教師あり微調整 → π_SFT (参照ポリシー)
    ↓
[Phase 2: 報酬モデル訓練]
(x, y_w, y_l) → Bradley-Terry損失 → r_φ(x, y)
    ↓
[Phase 3: PPOによるRL最適化]
π_SFT → PPO(r_φ, π_ref) → π_RLHF
         ↑ サンプリング + 報酬計算ループ
         └─── 不安定、計算コスト大

【DPO】
人間の選好データ D
    ↓
[Phase 1: SFT]
教師あり微調整 → π_SFT = π_ref (参照ポリシー)
    ↓
[Phase 2: DPO最適化（Phase 2とPhase 3を統合）]
(x, y_w, y_l) → DPO損失（二値クロスエントロピー） → π_DPO
                 ↑ 単純な教師あり学習
                 └─── 安定、計算コスト小
```

### DPO訓練ステップの詳細フロー

```
入力: (prompt x, chosen y_w, rejected y_l)
    ↓                    ↓
┌───────────┐    ┌───────────────┐
│ π_θ (学習) │    │ π_ref (固定)  │
│  Forward   │    │   Forward     │
└─────┬─────┘    └──────┬────────┘
      ↓                  ↓
log π_θ(y_w|x)    log π_ref(y_w|x)
log π_θ(y_l|x)    log π_ref(y_l|x)
      ↓                  ↓
      └────────┬─────────┘
               ↓
    Δr = β[log(π_θ(y_w)/π_ref(y_w)) - log(π_θ(y_l)/π_ref(y_l))]
               ↓
    Loss = -log σ(Δr)
               ↓
    ∇_θ Loss → パラメータ更新
```

## Figures & Tables

### 1. 感情制御タスク：報酬 vs KLダイバージェンスのフロンティア

DPOは、IMDb感情制御タスクにおいて、全てのKLダイバージェンス値に対して最高の期待報酬を達成した。

| 手法 | 特性 |
|------|------|
| DPO | 全KL値で最高報酬を達成。最も効率的なフロンティア |
| PPO (学習済み報酬) | DPOに次ぐ性能。報酬モデル訓練が必要 |
| PPO (真の報酬) | 真の報酬を使用しても DPO に劣る場合あり |
| Unlikelihood | 低KL領域で低い報酬 |

### 2. TL;DR要約タスク：勝率比較（GPT-4評価）

GPT-4による人間参照要約との比較勝率：

| 手法 | 温度 0.0 での勝率 | 備考 |
|------|-------------------|------|
| **DPO** | **約61%** | **最高性能、温度変化に頑健** |
| PPO | 約57% | 最適温度 0.0 |
| Best of 128 | PPOと同等 | 計算コストが非常に高い |
| Preferred-FT (SFT) | PPO以下 | ベースライン |

DPO vs PPO 直接比較：DPOサンプル（温度0.25）がPPOサンプル（温度0.0）に対して**58%**の勝率

### 3. Anthropic HH対話タスク：勝率比較（GPT-4評価）

| 手法 | テストセットの選好応答に対する勝率 | 備考 |
|------|----------------------------------|------|
| **DPO** | **選好応答を上回る唯一の効率的手法** | 最高性能 |
| Best of 128 | DPOと同等レベル | 計算コスト極大 |
| PPO | 選好応答を下回る | — |
| Unlikelihood | 選好応答を下回る | — |
| Preferred-FT (SFT) | ベースライン | — |

### 4. 手法間の全体比較表

| 比較項目 | DPO | PPO (RLHF) | Best of N | Unlikelihood |
|----------|-----|------------|-----------|-------------|
| 報酬モデル必要 | No | Yes | Yes | No |
| RL必要 | No | Yes | No | No |
| 訓練中サンプリング | No | Yes | 推論時N回 | No |
| 感情制御 | 最高 | 良好 | — | 低い |
| 要約（勝率） | 約61% | 約57% | PPO同等 | PPO以下 |
| 対話品質 | 最高 | 中程度 | 高い | 低い |
| 計算効率 | 高い | 低い | 非常に低い | 高い |

## Experiments & Evaluation

### Setup

#### データセット

| データセット | タスク | 詳細 |
|-------------|--------|------|
| IMDb | 感情制御 | 映画レビューデータセット。GPT-2をSFTし、siebert/sentiment-roberta-large-englishを真の報酬モデルとして使用。25,000プレフィックスから4つの補完をサンプリングし、各プレフィックスに対して6つの選好ペアを作成 |
| Reddit TL;DR | 要約 | Redditフォーラム投稿の要約。GPT-Jベース |
| Anthropic HH | 単一ターン対話 | 有用性と無害性の対話データセット。Pythia-2.8Bベース |

#### ベースライン

- **Preferred-FT (SFT)**: 選好応答のみで教師あり微調整
- **Unlikelihood**: 非選好応答の尤度を低下させる訓練
- **PPO**: Proximal Policy Optimizationによる標準的RLHF
- **PPO (真の報酬)**: 真の報酬関数を使用したPPO（感情制御タスクのみ、上界として）
- **Best of N**: N個のサンプルから報酬モデルスコアが最高のものを選択（N=128）

#### 評価方法

- **感情制御**: 期待報酬 vs KLダイバージェンスのフロンティア比較
- **要約・対話**: GPT-4による自動評価（勝率計算）。GPT-4の判断は人間との一致率が高く、人間同士の一致率と同等以上

### Main Results

#### 感情制御（IMDb）
- DPOは全てのKLダイバージェンス値において最高の期待報酬を達成
- PPO（学習済み報酬モデル使用）よりも優れたフロンティアを形成
- PPO（真の報酬使用）と比較しても競争力のある結果

#### TL;DR要約
- DPO: 温度0.0で**約61%**の勝率（人間参照要約に対するGPT-4評価）
- PPO: 温度0.0で**約57%**の勝率
- DPO（温度0.25）vs PPO（温度0.0）の直接比較: DPOが**58%**で選好される
- DPOはサンプリング温度の変化に対してPPOよりも頑健

#### Anthropic HH対話
- DPOはテストセットの選好応答を上回る**唯一の計算効率の高い手法**
- Best of 128（計算コストが非常に高い）と同等以上の性能
- PPOおよびUnlikelihoodは選好応答を上回ることができなかった

### Ablation Study

#### 温度に対する頑健性
DPOは異なるサンプリング温度間での性能変動がPPOよりも小さく、温度に対してより頑健であることが示された。PPOは温度が上昇すると性能が大幅に低下するが、DPOは比較的安定した性能を維持する。

#### GPT-4評価の妥当性
GPT-4の判断と人間の判断の一致率は、人間同士の評価者間一致率と同等以上であることが確認され、GPT-4を評価者として使用することの妥当性が実証された。

## Notes

### 実用的インパクト
- DPOは発表後、LLMアラインメントの標準的手法の一つとなった
- Llama 3 Instruct、Zephyr等の主要なオープンソースLLMの訓練に採用された
- 実装が非常に簡単であり、標準的な教師あり学習フレームワーク（PyTorch等）で数十行のコードで実装可能

### 公式実装
- GitHub: https://github.com/eric-mitchell/direct-preference-optimization（公式リファレンス実装）
- HuggingFace TRLライブラリにDPOTrainerとして統合

### 制限事項と後続研究
- Bradley-Terry選好モデルの仮定に依存（より一般的な選好モデルへの拡張: f-DPO, IPO等）
- オフポリシーデータの品質に性能が依存する可能性
- 反復的DPO（Iterative DPO）やオンラインDPOなど、データ収集と学習を交互に行う手法が後続研究で提案
- KTO（Kahneman-Tversky Optimization）等、ペアデータを必要としない手法も提案されている

### 関連研究
- RLHF (Ouyang et al., 2022) — InstructGPTで使用されたPPOベースの手法
- SLiC (Zhao et al., 2023) — 類似の教師あり学習ベースのアプローチ
- RRHF (Yuan et al., 2023) — ランキングベースのアラインメント手法
- IPO (Azar et al., 2023) — Bradley-Terry仮定を緩和した手法
