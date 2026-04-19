# Imitation Learning Baseline (case3) — Web Technical Research

## 調査目的

模倣学習 (Imitation Learning) を Kaggle Orbit Wars に適用するにあたり、(a) 戦略ゲーム文脈での先行事例、(b) 連続/離散ハイブリッドな action 空間の扱い、(c) 軽量な PyTorch 実装パターン、を外部文献で確認する。

## 公式ドキュメント / 学術文献

### 行動クローニング (Behavior Cloning, BC) の基礎

- BC は「状態 → 行動」マッピングを教師あり学習で行う最も単純な IL 手法。報酬関数も環境フィードバックも不要 ([Emergent Mind](https://www.emergentmind.com/topics/behavior-cloning))。
- 弱点は **compounding error**: 学習分布から外れた状態に出会った時に誤差が蓄積する ([Underactuated Robotics Ch.21](https://underactuated.mit.edu/imitation.html))。
- 緩和策: DAgger (オンライン補正) / データ多様化 / アンサンブル。Orbit Wars はターンベース確定遷移なので drift は他より穏やか、まず **vanilla BC をベースライン**として組むのが妥当。

### 連続 vs 離散 action space

- 連続値回帰 (Gaussian policy `π(a|s)=N(μ, Σ)`) は精度が高いが **信頼度推定が弱い** ([Deep RL 2 Imitation - Puyuan Peng](https://jasonppy.github.io/deeprl/deeprl-2-imitation/))。
- 多次元 action を全部 m-bin で離散化すると `m^n` 爆発するが、**自己回帰因子化** `p(a₁|s)·p(a₂|s,a₁)…` で n 個のヘッドに分解できる ([同上](https://jasonppy.github.io/deeprl/deeprl-2-imitation/))。
- Orbit Wars の action は `[from_planet_id (離散 0〜35), angle (連続 [-π, π]), num_ships (整数 1〜N)]`。**from_planet を分類**, **angle を回帰** or **target_planet を分類して angle は predict_aim で計算**, **num_ships を回帰 or 5-bin 分類**, の 3 軸で設計する。
- 角度を **target_planet 分類で代替** すれば連続値回帰の難しさを回避でき、かつ既存 `aim_with_prediction()` を decoder にできる。**この設計は Orbit Wars 特有の geometry を活用** ([Continuous Control with Action Quantization from Demonstrations (ICML 2022)](https://proceedings.mlr.press/v162/dadashi22a/dadashi22a.pdf))。

## 類似 OSS プロジェクト

### Lux AI Season 1 (Kaggle 2021) — 最も近い先行事例

#### [shoheiazuma/lux-ai-with-imitation-learning](https://www.kaggle.com/shoheiazuma/lux-ai-with-imitation-learning)
- **Relevance**: Kaggle のターンベース戦略ゲーム × 模倣学習 × 上位入賞という Orbit Wars と完全に同型のセットアップ。
- **アプローチ**:
  - Kaggle API で上位プレイヤーの match record を一括 download。
  - obs を **盤面マルチチャンネル画像** (各チャンネルが「自陣 unit」「敵 unit」「資源量」など) に変換。
  - U-Net / 全結合 ResNet で「セル単位の action 分類」を学習。
  - 推論は U-Net の出力を argmax で各 unit の action にマッピング。
- **再利用可能なパターン**:
  - **Top-rated submission を expert として教師化** (Orbit Wars でも `agent_*_rating_mu` で同じことが可能)。
  - **action は分類タスクに帰着** させる (角度・距離は離散カテゴリ化)。
- **落とし穴**: 盤面画像化は Lux のグリッド世界には自然だが、**Orbit Wars は連続座標の点群** (惑星 ~36 個) なので CNN より **Set/Graph encoder (DeepSets, Set Transformer, GNN) の方が適切**。

#### [IsaiahPressman/Kaggle_Lux_AI_2021](https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021) — 上位ソリューション
- **Relevance**: Lux AI 2021 の上位入賞ソリューション。BC + RL ハイブリッド。
- **アプローチ**: Squeeze-Excitation 付き ResNet (128ch, 5×5 conv) を fully convolutional に構成。
- **再利用可能なパターン**:
  - **小さな conv ResNet (~5MB) でも十分強い**ことを示している → Orbit Wars でも軽量で OK。
  - **マルチタスクヘッド** (value head + policy head) で BC + Q-learning を共有 backbone から学習。case3 では **まず policy head のみ**, 後続 case で RL 化する余地。
- **落とし穴**: グリッドベース前提の論文・コードを Orbit Wars にそのまま流用すると過剰実装。

#### [Two Sigma — Best Practices from Building a Halite ML Bot](https://www.twosigma.com/articles/best-practices-from-building-a-machine-learning-bot-for-halite/)
- **Relevance**: Kaggle Halite (2018) で IL を本気で適用した社内チームの教訓集。
- **再利用可能なパターン**:
  - "**Imitate the winner, not the average**" — episode の `winner` 側だけを教師にする。
  - "**Filter by ELO/rating cutoff**" — レーティング下位は学習を悪化させる。
  - "**Frame-level loss が偏る**" — 序盤手と終盤手のバランスを weighted sampling で調整。
- **落とし穴**: 完全な BC だけだと **ノベルなボードに弱い** → 自己対戦データ追加 (DAgger 風) の検討が必要。

### Pattern Comparison

| Aspect | 本プロジェクト (Orbit Wars) | Lux AI (shoheiazuma) | Lux AI (IsaiahPressman) | Halite (Two Sigma) | 推奨 |
|--------|--------------------------|----------------------|--------------------------|---------------------|------|
| 観測表現 | 惑星点群 (~36) + fleet (動的) | グリッド画像 | グリッド画像 | グリッド画像 | **DeepSets / Set Transformer** (惑星集合は順不同) |
| Action 表現 | (from, angle, ships) | (unit, action_type) | per-cell action | (ship, dir) | **(from_id 分類 × target_id 分類 × ships 5-bin 分類)** |
| Backbone | (新規) | U-Net | SE-ResNet | CNN | **MLP + Set encoder (~1MB)** |
| Expert filter | rating_mu > 1200 | top kernel | top entries | ELO top 10% | rating_mu top quartile |
| 学習目標 | BC | BC | BC + Q-learning | BC + 自己対戦 | **vanilla BC (MVP)** → 後続で DAgger |
| サブミッション size | <10MB | ~30MB | ~50MB | ~10MB | <5MB target |

## ライブラリ / サービス選定

### Deep Learning フレームワーク

| Candidate | Pros | Cons | Maintenance | 推奨 |
|-----------|------|------|-------------|------|
| **PyTorch (CPU 版)** | Kaggle 標準同梱、研究エコシステム広い、torch.jit/onnx エクスポート可 | サイズ ~150MB (Kaggle にプリインストールなので追加 0) | ⭐ アクティブ | ⭐ 推奨 |
| JAX/Flax | 速い、関数型で純粋 | Kaggle ランタイムでバージョン要確認 | アクティブ | ✗ |
| scikit-learn (MLP) | 軽量・依存最小 | 表現力不足、GPU 学習不可 | アクティブ | △ MVP の対照群として |
| TensorFlow/Keras | Kaggle で動く | 重く、最近のコミュニティ熱量が PyTorch より低い | アクティブ | ✗ |

→ **PyTorch CPU 版 + ローカル学習時のみ GPU (mps/cuda 任意)**。

### 推論時のテンソル化

| Candidate | Pros | Cons | 推奨 |
|-----------|------|------|------|
| 純粋 NumPy + 手書き transform | 依存最小、明示的 | コード量 | ⭐ |
| polars + pyarrow | バッチ前処理が爆速 | 推論時には不要 | 学習データ前処理に |
| pandas | 既存依存 | 大量小ファイルで遅い | ✗ |

→ **学習データ前処理: polars / 推論時: NumPy**。

## API / Protocol Research

### Kaggle Submission ランタイム — PyTorch サポート

- Kaggle competitions の Python kernel ランタイムは PyTorch 同梱。`import torch` で動作実績あり (Lux AI の上位ソリューションも全て torch)。
- ただし **submit tar.gz 内に torch を入れる必要はない** (ランタイムに既存)。`pyproject.toml` の dev/test 依存に追加するだけで済む。
- **モデル重み形式**: `torch.save(model.state_dict(), "weights.pt")` 推奨。`pickle` 経由は **バージョン互換でハマる**。

### kaggle-environments の Determinism

- `kaggle_environments.make("orbit_wars", configuration={"seed": N})` で seed 固定可能 (`src/env/executor.py:99`)。**ただし完全再現ではない** (`pipeline/case1/README.md:99`)。
- 模倣学習エージェントは **observation 固定 → action 完全再現** をテストで担保 (case1 と同じスナップショット方式)。

## Research Summary

### 設計に効く主要 finding

1. **教師データのフィルタは必須**: rating_mu トップ層 (例: > 1200, 上位 ~25 submissions) のリプレイから、**winner 側のみ**を学習データ化する。
2. **Action 表現は分類に寄せる**: `from_planet_id (36-class)`, `target_planet_id (36-class or no-op)`, `num_ships_bucket (5-class: 25%, 50%, 75%, 100%, exact_need)` の 3 ヘッド分類。`angle` は `aim_with_prediction(src, target)` から決定論的に再構成 (Orbit Wars geometry を活用)。
3. **Backbone は軽量に**: 惑星 ~36 + 自陣 fleet ~20 を Set encoder (DeepSets or 1-layer Set Transformer) で埋め込み → MLP head。重み 1〜5MB に収まる。
4. **MVP は vanilla BC**: DAgger / 自己対戦 augmentation は case3 のスコープ外、後続の case4 で扱う。
5. **マルチエージェント (FFA) は当面 1v1 のみ**: ffa4 のデータも豊富だが、player perspective の正規化 (相対座標) を 1v1 で確立してから 4P 拡張すべき。

### 採用する外部パターン

| パターン | 出典 | 適用方法 |
|----------|------|---------|
| Top-rated expert filter | Two Sigma Halite, Lux AI shoheiazuma | `rating_mu > τ` and winner == player のフレームのみ |
| Action quantization from demonstrations | Dadashi+ ICML 2022 | num_ships を 5-bin に分割、bin 境界はデモ分布から学習 |
| Set encoder for variable point clouds | DeepSets (Zaheer 2017) | 惑星集合 (順不同) を順不同 invariant に embed |
| Snapshot-based determinism test | pipeline/case1 既存パターン | obs 固定で action JSON 完全一致を assert |

### 推奨するアプローチ (要約)

- **Backbone**: 各惑星の特徴 (位置・所有者・船数・生産・自陣相対距離) を MLP embed → mean pool で全惑星集約 + 自陣 summary 特徴 concat → 2-layer MLP。
- **Heads**: `from_planet (logits over my_planets)`, `target_planet (logits over all planets ∪ no-op)`, `num_ships_bucket (5-class)`。
- **Action decoder**: 推論時は (a) 各 my_planet について from_planet=自分の確率を取り、(b) 確率閾値を超えた from について target をサンプル/argmax、(c) `aim_with_prediction()` で angle を計算、(d) num_ships は bucket × 自陣ships で決定。
- **Loss**: 3 ヘッドの cross entropy 合計 (重み 1:1:0.5 程度)。
- **データ規模**: 1v1 リプレイ 456 本 × 平均 ~250 turn × 平均 ~3 actions/turn ≈ **数十万 (action, obs) ペア**。MVP には十分。

### Open Questions (Step 3 のヒアリングで確定)

- 1v1 のみで開始するか、ffa4 も含めるか
- Action 表現: 分類 3-head か, target 分類 + angle 回帰のハイブリッドか
- 評価指標: vs case1 の勝率を主か, 別の baseline 群を用意するか
- Kaggle 提出のスコープ: case3 を実提出して LB スコア計測まで含めるか, ローカル比較で止めるか
