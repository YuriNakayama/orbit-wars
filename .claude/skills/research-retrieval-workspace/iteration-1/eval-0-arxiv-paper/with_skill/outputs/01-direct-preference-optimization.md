# Direct Preference Optimization: Your Language Model is Secretly a Reward Model

- **Link**: https://arxiv.org/abs/2305.18290
- **Authors**: Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D. Manning, Stefano Ermon, Chelsea Finn
- **Year**: 2023
- **Venue**: NeurIPS 2023 (37th Conference on Neural Information Processing Systems)
- **Type**: Academic Paper

## Abstract

While large-scale unsupervised language models (LMs) learn broad world knowledge and some reasoning skills, achieving precise control of their behavior is difficult due to the completely unsupervised nature of their training. Existing methods for gaining such steerability collect human labels of the relative quality of model generations and fine-tune the unsupervised LM to align with these preferences, often with reinforcement learning from human feedback (RLHF). However, RLHF is a complex and often unstable procedure, first fitting a reward model that reflects the human preferences, and then fine-tuning the large unsupervised LM using reinforcement learning to maximize this estimated reward without drifting too far from the original model. In this paper, we leverage a mapping between reward functions and optimal policies to show that this constrained reward maximization problem can be optimized exactly with a single stage of policy training, essentially solving a classification problem on the human preference data. The resulting algorithm, which we call Direct Preference Optimization (DPO), is stable, performant, and computationally lightweight, eliminating the need for fitting a reward model, sampling from the LM during fine-tuning, or performing significant hyperparameter tuning. Our experiments show that DPO can fine-tune LMs to align with human preferences as well as or better than existing methods. Notably, fine-tuning with DPO exceeds RLHF's ability to control sentiment of generations and improves response quality in summarization and single-turn dialogue while being substantially simpler to implement and train.

## Abstract（日本語訳）

大規模な教師なし言語モデル（LM）は幅広い世界知識やある程度の推論能力を学習するが、完全に教師なしの訓練という性質上、その振る舞いを精密に制御することは困難である。既存の手法では、モデル生成物の相対的な品質に関する人間のラベルを収集し、教師なしLMを人間の選好に合わせてファインチューニングするが、多くの場合、人間のフィードバックからの強化学習（RLHF）が用いられる。しかし、RLHFは複雑で不安定な手続きであり、まず人間の選好を反映する報酬モデルを学習し、次にその推定報酬を最大化するように大規模教師なしLMを強化学習でファインチューニングする必要があり、その際に元のモデルから大きく逸脱しないようにしなければならない。本論文では、報酬関数と最適方策の間の写像を活用し、この制約付き報酬最大化問題が、人間の選好データに対する分類問題を解くことで、単一段階の方策訓練により厳密に最適化できることを示す。得られたアルゴリズムを**Direct Preference Optimization（DPO）**と呼び、安定的で高性能かつ計算コストが低く、報酬モデルの学習、ファインチューニング中のLMからのサンプリング、大規模なハイパーパラメータ調整を不要にする。実験では、DPOが既存手法と同等またはそれ以上に人間の選好に沿ったLMのファインチューニングを実現できることを示す。特に、DPOによるファインチューニングは、感情制御においてRLHFを上回り、要約や単一ターン対話における応答品質を向上させつつ、実装と訓練が大幅に簡単である。

## 概要

本論文は、大規模言語モデル（LLM）を人間の選好に合わせてアライメントする新しい手法「**Direct Preference Optimization（DPO）**」を提案する。従来のRLHF（Reinforcement Learning from Human Feedback）パイプラインでは、(1) 人間の選好データから報酬モデルを学習し、(2) その報酬モデルを用いてPPO等の強化学習で方策を最適化する、という2段階の複雑な手続きが必要であった。DPOは、報酬関数と最適方策の間の解析的な関係を利用することで、報酬モデルの明示的な学習を省略し、選好データに対する単純な分類損失のみで直接的に方策を最適化する。これにより、強化学習のループ、報酬モデルの学習、LMからのサンプリングが一切不要となり、実装の簡潔さと訓練の安定性を大幅に向上させた。

## 問題設定

本論文が解決を目指す問題は以下の通りである。

- **RLHFの複雑性**: 従来のRLHFパイプラインは、報酬モデル学習 → 強化学習による方策最適化という2段階の手続きが必要であり、実装が複雑でハイパーパラメータ調整が困難である。
- **訓練の不安定性**: PPO（Proximal Policy Optimization）を用いた強化学習は不安定になりやすく、報酬ハッキング（reward hacking）のリスクがある。
- **計算コスト**: 報酬モデルの学習、方策からのサンプリング、価値関数の学習など、複数のモデルを同時に保持・更新する必要があり、計算資源の消費が大きい。
- **スケーラビリティの課題**: モデルサイズが大きくなるにつれ、RLHF全体のパイプラインの維持がさらに困難になる。

## 提案手法

**Direct Preference Optimization（DPO）**

DPOの核心的なアイデアは、RLHFの制約付き報酬最大化問題において、報酬関数と最適方策の間に閉形式の解析的関係が存在することを利用する点にある。

### 核心的アイデア

1. 標準的なRLHF目的関数（報酬最大化 + KLペナルティ）の最適方策を閉形式で導出
2. その最適方策の式を逆に解いて、報酬関数を方策の関数として表現
3. Bradley-Terryモデルに代入することで、報酬モデルを経由せず直接方策を最適化する損失関数を導出

### 既存手法との違い

| 特性 | RLHF (PPO) | DPO |
|------|------------|-----|
| 報酬モデル学習 | 必要 | 不要 |
| 強化学習 | 必要（PPO） | 不要 |
| LMからのサンプリング | 訓練中に必要 | 不要 |
| 価値関数 | 必要 | 不要 |
| 損失関数 | 方策勾配 | 二値分類（クロスエントロピー） |
| ハイパーパラメータ | 多数 | βのみ |
| 実装の複雑さ | 高 | 低 |
| 訓練の安定性 | 低〜中 | 高 |

### 主要な数式

**RLHF目的関数**（出発点）:

$$\max_{\pi} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi(y|x)} \left[ r(x, y) \right] - \beta \cdot D_{\mathrm{KL}} \left[ \pi(y|x) \| \pi_{\mathrm{ref}}(y|x) \right]$$

ここで $r(x, y)$ は報酬関数、$\pi_{\mathrm{ref}}$ は参照方策（SFTモデル）、$\beta$ はKLペナルティの強さを制御するパラメータである。

**最適方策の閉形式解**:

$$\pi_r(y|x) = \frac{1}{Z(x)} \pi_{\mathrm{ref}}(y|x) \exp\left(\frac{1}{\beta} r(x, y)\right)$$

ここで $Z(x) = \sum_y \pi_{\mathrm{ref}}(y|x) \exp\left(\frac{1}{\beta} r(x, y)\right)$ は分配関数である。

**報酬関数の方策による再パラメータ化**:

$$r(x, y) = \beta \log \frac{\pi_r(y|x)}{\pi_{\mathrm{ref}}(y|x)} + \beta \log Z(x)$$

**DPO損失関数**（最終形）:

$$\mathcal{L}_{\mathrm{DPO}}(\pi_\theta; \pi_{\mathrm{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)} \right) \right]$$

ここで $y_w$ は選好された応答、$y_l$ は非選好の応答、$\sigma$ はシグモイド関数である。Bradley-Terryモデルへの代入時に分配関数 $Z(x)$ がキャンセルされることが鍵となる。

### 特徴

- **教師あり学習のみ**: 強化学習を一切使わず、選好ペアに対する二値分類損失のみで最適化
- **暗黙的報酬モデル**: 方策自体が暗黙的に報酬モデルを定義しており、明示的な報酬モデルが不要
- **理論的保証**: Bradley-Terryモデルの下で、RLHFと同じ最適解に収束することが証明されている
- **動的な重み付け**: 勾配は暗黙的報酬の推定値に基づく動的な重みを持ち、モデルの退化を防止

## アルゴリズム（擬似コード）

```
Algorithm: Direct Preference Optimization (DPO)
Input: 選好データセット D = {(x_i, y_w_i, y_l_i)}_{i=1}^N,
       参照方策 π_ref (SFTモデル),
       パラメータ β (KLペナルティ強度)
Output: 最適化された方策 π_θ

1. π_θ ← π_ref をコピーして初期化    // 参照方策から開始
2. for each ミニバッチ B ⊂ D do
3.   for each (x, y_w, y_l) ∈ B do
4.     // 選好・非選好応答の対数確率比を計算
5.     r_w ← β * (log π_θ(y_w|x) - log π_ref(y_w|x))  // 選好応答の暗黙的報酬
6.     r_l ← β * (log π_θ(y_l|x) - log π_ref(y_l|x))  // 非選好応答の暗黙的報酬
7.     // DPO損失を計算
8.     loss ← -log σ(r_w - r_l)                          // 二値分類損失
9.   end for
10.  L ← (1/|B|) * Σ loss                                // バッチ平均
11.  θ ← θ - α * ∇_θ L                                   // 勾配降下で更新
12. end for
13. return π_θ
```

## アーキテクチャ / プロセスフロー

### 従来のRLHFパイプライン vs DPO

```
=== 従来のRLHF（3段階） ===

[人間の選好データ]
       ↓
[Stage 1: SFT]  教師ありファインチューニング
       ↓
[Stage 2: 報酬モデル学習]  選好データから報酬関数を学習
       ↓                         ↓
[Stage 3: PPO]  ←── [報酬スコア] ←── [報酬モデル]
  方策最適化         サンプル評価
       ↓
[アライメント済みモデル]


=== DPO（2段階） ===

[人間の選好データ]
       ↓
[Stage 1: SFT]  教師ありファインチューニング → π_ref
       ↓
[Stage 2: DPO]  選好ペアに対する分類損失で直接最適化
  (報酬モデル不要、サンプリング不要、RL不要)
       ↓
[アライメント済みモデル]
```

### DPO損失の勾配の直感的理解

```
DPO勾配の構造:

∇L_DPO ∝ -β · σ(r̂_l - r̂_w) · [∇log π_θ(y_w|x) - ∇log π_θ(y_l|x)]
            ↑                       ↑                      ↑
       動的重み係数           選好応答の確率↑        非選好応答の確率↓
   (誤分類されやすい例ほど
    大きな重みで更新)
```

## 図表

### 1. 手法比較表：RLHF vs DPO vs その他のアライメント手法

| 手法 | 報酬モデル | RL | サンプリング | 安定性 | 実装容易性 |
|------|-----------|-----|------------|--------|-----------|
| **DPO** | 不要（暗黙的） | 不要 | 不要 | 高 | 非常に高 |
| RLHF (PPO) | 必要 | 必要 | 必要 | 低〜中 | 低 |
| Best-of-N | 必要 | 不要 | N回必要 | 高 | 中 |
| SLiC-HF | 不要 | 不要 | 不要 | 高 | 高 |

### 2. 感情制御実験の結果

| 手法 | 報酬（↑） | KL距離（↓） |
|------|----------|------------|
| Zero-shot | 低 | 0 |
| SFT | 中 | 小 |
| PPO (best) | 高 | 中 |
| **DPO** | **最高** | **小〜中** |

DPOは、感情制御タスクにおいて、同程度のKLダイバージェンスにおいてPPOよりも高い報酬を達成する（報酬-KLフロンティアの改善）。

### 3. TL;DR要約タスク: GPT-4による勝率評価

| 比較 | DPO勝率 | 引き分け | 相手勝率 |
|------|---------|---------|---------|
| DPO vs SFT | 高い | - | 低い |
| DPO vs PPO (temp=0) | ~58% | - | ~42% |
| DPO vs 人間書き要約 | 競争的 | - | 競争的 |

人間評価でも、DPOサンプル（温度0.25）がPPOサンプル（温度0）に対して約58%の勝率を示す。

### 4. Anthropic-HH 単一ターン対話タスク

| 手法 | テストセット勝率 |
|------|---------------|
| Preferred (人間選択) | ベースライン |
| SFT | 改善なし |
| PPO | 改善なし |
| **DPO** | **唯一の改善** |

DPOはAnthopic-HHテストセットにおいて、選好された応答を上回る唯一の手法である。

## 実験と評価

### 実験設定

#### データセット
- **感情制御**: IMDbデータセット、GPT-2-largeをベースモデルとして使用
- **要約**: Reddit TL;DRデータセット（選好ペア付き）
- **単一ターン対話**: Anthropic Helpful and Harmless（HH）データセット

#### ベースライン
- SFT（教師ありファインチューニングのみ）
- RLHF with PPO（標準的なRLHFパイプライン）
- Best-of-N サンプリング
- SLiC-HF（Sequence Likelihood Calibration）
- Unlikelihood training

#### 評価指標
- 感情制御: 地上真値報酬モデル（siebert/sentiment-roberta-large-english）のスコア + KLダイバージェンス
- 要約・対話: GPT-4による勝率評価（人間評価との高い相関を確認済み）

### 主要な結果

1. **感情制御**: DPOは報酬-KLダイバージェンスのフロンティアにおいて、PPOを含む全ベースラインを上回る。同じKL予算でより高い報酬を達成。

2. **TL;DR要約**: DPOはGPT-4評価において、PPOベースのRLHFと同等またはそれ以上の勝率を達成。CNN/DailyMailデータセット（分布外）への汎化においても、DPOがPPOを大幅に上回る。

3. **Anthropic-HH対話**: DPOはテストセットの選好応答を上回る唯一の手法。PPOやSFTは改善を示さなかった。

4. **人間評価との一致**: GPT-4評価と人間評価の間に高い相関が確認された（150件のDPO vs PPO比較、200件のPPO vs PPO比較で検証）。

### アブレーション研究

- **β（KLペナルティ強度）の影響**: βが大きすぎると参照方策に近すぎる保守的な方策になり、小さすぎると参照方策から過度に離れる。タスクに応じた適切な設定が重要。
- **サンプリング温度の影響**: DPOは温度0.25で最良の結果を示し、PPOは温度0で最良。
- **分布外汎化**: TL;DRで訓練した方策をCNN/DailyMailで評価した場合、DPOはPPOよりも頑健に汎化する。

## 備考

- **後続研究への影響**: DPOはLLMアライメント分野に大きな影響を与え、多数の派生手法が提案されている（IPO, KTO, ORPO, SimPO, ADPOなど）。
- **実用的採用**: Zephyr 7B、TULU 2（70Bパラメータ）など、大規模な実用モデルの訓練にDPOが採用されている。
- **限界**: Bradley-Terryモデルの仮定に依存しており、選好がこのモデルに従わない場合の性能保証はない。また、オンラインのデータ収集を行わないため、方策の改善に伴う新しい選好データの活用が困難。
- **実装**: HuggingFaceのTRLライブラリにDPOTrainerとして実装されており、数行のコードでDPO訓練が可能。
- **引用数**: NeurIPS 2023で発表以降、LLMアライメント分野で最も引用される論文の一つとなっている。
