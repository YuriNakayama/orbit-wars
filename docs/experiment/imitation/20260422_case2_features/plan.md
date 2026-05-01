# Orbit Wars 模倣学習 — 特徴量設計メモ

> 作成日: 2026-04-22
> 関連: `docs/experiment/imitation/20260420_case1_baseline/plan.md`（IL ベースライン全体設計）
> スコープ: IL モデルへ与える特徴量の設計と過去コンペ事例ベースのバリデーション

過去 Kaggle Simulation IL 解法を web search + writeup で裏取りし、Orbit Wars に転用可能な
特徴量設計の工夫を 5 つの観点で表形式に整理する。

---

## 目次

1. [空間特徴マップ（spatial channels）](#1-空間特徴マップspatial-channels)
2. [スカラー特徴（scalar / game progress）](#2-スカラー特徴scalar--game-progress)
3. [派生・手計算特徴（engineered features）](#3-派生手計算特徴engineered-features)
4. [履歴 / Entity / 正規化](#4-履歴--entity--正規化)
5. [Augment / Auxiliary Head / 適用優先度](#5-augment--auxiliary-head--適用優先度)
6. [Orbit Wars への適用提言](#6-orbit-wars-への適用提言)
7. [参考文献](#7-参考文献)

---

## 1. 空間特徴マップ（spatial channels）

CNN / U-Net 入力としての (C, H, W) テンソルにおけるチャネル設計事例。

| コンペ / 解法 | 入力形状 | チャネル数 | 主要チャネル内容 | 出典 |
|--------------|----------|-----------|------------------|------|
| Halite IV (Kha Vo 8位) | 21×21 | 多数 (EfficientNet encoder) | halite 量, シップ位置, シップヤード位置, 所有者 | voanhkha.github.io |
| Halite IV (公式 env) | 21×21 | 基本 4 ch | halite 量, シップヤード owner, シップ位置, owner | kaggle-environments |
| Hungry Geese (Maxwell 3位) | 7×11 | **46 ch (20 feat)** | 17 base + floodFill + food + opp tail | hoxomaxwell SpeakerDeck |
| Kore 2022 (khanhvu207) | 21×21 | **18 ch** → 12-layer ResNet | ship 位置, cargo, board state | github.com/khanhvu207/kore2022 |
| Lux AI S1 (No face, 93位) | 32×32 | 128 ch 5×5 conv, 16 residual | U-Net + CBAM attention | Kaggle writeup |
| AlphaStar | 変動 | multi-modal | spatial encoder (minimap) + entity list | DeepMind Nature 2019 |

**所感**: Kaggle Simulation の IL 上位は **18〜46 ch** のレンジ。Orbit Wars でも 14–18 ch を起点に
アブレーションで増減するのが妥当。

---

## 2. スカラー特徴（scalar / game progress）

| 特徴 | 採用コンペ | 実装方法 |
|------|-----------|---------|
| 現ターン番号 | Halite IV, Kore 2022, Hungry Geese, AlphaStar | 正規化 (/total_turns) |
| 残りターン比率 | Halite IV, Hungry Geese | broadcast channel として H×W に敷く |
| 自スコア / 敵スコア | Kore 2022, AlphaStar | scalar → MLP → broadcast |
| スコア差 | Halite IV 上位 | リード/ビハインドで戦術切替 |
| 自リソース保有量 | Kore 2022, AlphaStar | scalar |
| 収集率 / 生産率 | AlphaStar | scalar |
| 破壊済みユニット総価値 | AlphaStar | scalar |
| 総収集ミネラル / ベスピン | AlphaStar | scalar |
| アイドル生産時間 | AlphaStar | scalar |

**実装パターン共通**: `scalar → MLP → spatial broadcast (H×W 全面)` または **FiLM 変調** で
CNN backbone に注入する。

---

## 3. 派生・手計算特徴（engineered features）

CNN に手渡してやることで学習を加速できる「明示計算可能な」特徴量。

| 特徴 | 目的 | 採用コンペ | 計算方法 |
|------|------|-----------|---------|
| **FloodFill** | 領域支配・閉塞検知 | Hungry Geese (Maxwell 3位) | 自ヘッドから BFS で到達可能セル数マップ |
| **距離マップ** | 最寄りユニット検索の暗黙化 | Halite IV, Kore 2022 | 自/敵シップからの Chebyshev 距離を H×W に |
| **脅威マップ** | 敵到達可能領域の可視化 | Hungry Geese, AlphaStar 的 | 敵が N ターンで到達可能なセルを減衰値で |
| **リソース再生マップ** | 経時変化するセル価値 | Kore 2022 | セル依存の regen rate を channel 化 |
| **未来位置予測 (fog-of-war)** | POMDP 補完 | Lux S3 | 過去観測から次 N step を予測して channel 化 |
| **軌道予測（Orbit Wars 固有）** | 軌道力学の NN 明示 | — (新規) | 次 N ターンの惑星位置を channel 化 |
| **敵頭隣接食料フラグ** | 食料競合検知 | Hungry Geese Maxwell | opp_head 隣接セルに food があれば 1 |
| **opp tail position** | 衝突回避 | Hungry Geese Maxwell | 他蛇の tail セルを channel 化 |

**Orbit Wars 固有のキラー特徴**: 軌道予測 channel — 軌道力学を NN にブラックボックス学習させるのは
コスパが悪いため、**事前計算した次 N ターンの惑星位置を channel として手渡す** ことで学習を加速できる
（Lux S3 の fog-of-war 予測の発想と同型）。

---

## 4. 履歴 / Entity / 正規化

### 4-A. 履歴特徴（history features, POMDP 補完）

| 方式 | 採用コンペ | 詳細 |
|------|-----------|------|
| 過去 K ターン stack | Halite IV 上位, Hungry Geese | 過去 4 ターンの位置/行動を channel 方向に concat |
| LSTM 内部状態 | AlphaStar | Transformer core の後段に LSTM |
| 過去フラグ channel | Halite IV | 直前ターンに launch したか 0/1 |

### 4-B. Entity 特徴（Transformer 入力）

| 解法 | token あたり特徴 | embed dim | 出典 |
|------|-----------------|-----------|------|
| **AlphaStar entity encoder** | unit_type, unit_attr, alliance, current_health, was_selected, position, cooldown | 128 (2-layer, 2-head) | DeepMind Nature 2019 |
| **Kore 2022 ship plan** | 文字 embedding + position embedding, concat | dim/2 × 2 = Transformer dim | github.com/khanhvu207 |

### 4-C. 正規化・スケーリング

| 特徴 | 推奨正規化 | 理由 |
|------|-----------|------|
| 艦隊量 / halite 量 | log1p or 盤面 max 除算 | 分布が裾重い |
| ターン数 | / total_turns (0–1) | phase 比較可能に |
| 距離 | / 盤面対角長 | スケール不変 |
| 所有者 | one-hot (3–4 クラス) | embed より解釈性高 |
| 惑星 ID | **embedding** | one-hot は惑星 10 超で generalize 悪化 |
| 位置 (x, y) | / 盤面辺長 | 0–1 範囲 |

---

## 5. Augment / Auxiliary Head / 適用優先度

### 5-A. Augmentation（特徴生成としての augment）

| 手法 | 効果 | 出典 |
|------|------|------|
| **盤面ピクセル 60% ランダムマスク** | robust 化、DAgger 代替的効果 | **Kore 2022 khanhvu207（裏取り済み）** |
| 盤面 90°/180°/270° 回転 | 実効データ ×4 | Halite IV, Lux S1, Kore 2022 |
| 左右反転 | 実効データ ×2 | 多数 |
| 時系列 subsampling | 重要局面抽出 | AlphaStar |
| 勝率 weighting | 負け軌跡も活用 | Hungry Geese Maxwell |
| action 圧縮（6→3 via 回転） | 学習クラス削減 | Lux S1 No face |

### 5-B. Auxiliary Heads

| Aux head | 効果 | 採用コンペ |
|----------|------|-----------|
| Value head (勝敗/スコア予測) | MCTS prior 強化、dual head | Hungry Geese, AlphaGo, AlphaZero |
| 次ターン敵行動予測 | opponent modeling | AlphaStar |
| 得点差予測 | dense reward 代替 | Halite IV 上位 |

---

## 6. Orbit Wars への適用提言

### 6-A. 優先度マップ

| 優先 | 特徴 / 工夫 | 参照表 | 根拠コンペ |
|------|------------|--------|-----------|
| ★★★ | 自/敵艦隊量、惑星所有者、生産速度 | §1 | 全コンペ共通 |
| ★★★ | 残りターン比率、自/敵スコアの broadcast | §2 | Halite IV, Kore 2022 |
| ★★★ | **軌道予測チャネル（次 N ターン惑星位置）** | §3 | Orbit Wars 固有、Lux S3 類似 |
| ★★★ | 過去 4 ターンの敵発射履歴 stack | §4-A | Halite IV 上位 |
| ★★★ | 艦隊量 log1p、距離 /対角長 正規化 | §4-C | 全コンペ共通 |
| ★★ | 惑星 ID を embedding | §4-C | AlphaStar |
| ★★ | FloodFill / 到達コストマップ | §3 | Hungry Geese Maxwell |
| ★★ | 距離マップ / 脅威マップ | §3 | Halite IV, Kore 2022 |
| ★★ | **60% ランダムピクセルマスク augment** | §5-A | Kore 2022 khanhvu207 |
| ★★ | Entity Transformer + scalar broadcast | §4-B | AlphaStar, Kore 2022 |
| ★ | Value head (auxiliary) | §5-B | Hungry Geese, AlphaGo |
| ★ | 敵行動予測 head | §5-B | AlphaStar |
| ★ | 盤面回転 TTA (推論時) | §5-A | Halite IV 上位 |
| ★ | LSTM 長期履歴 | §4-A | AlphaStar（短期 stack で代替可） |

### 6-B. 推奨初期チャネル構成（Day 1 ベースライン）

```
ch  0:    自艦隊量（log1p 正規化）
ch  1:    敵艦隊量（log1p 正規化）
ch  2:    惑星中心位置マスク
ch  3:    惑星 ID（embedding 入力用、後段で展開）
ch  4-6:  所有者 one-hot（自/敵/中立）
ch  7:    残りターン比率（broadcast scalar）
ch  8:    スコア差（broadcast scalar）
ch  9:    生産速度マップ
ch 10-13: 過去 4 ターンの敵発射履歴
ch 14-17: 軌道予測: 次 1, 2, 4, 8 ターン後の惑星位置  ← Orbit Wars 固有
合計: 18 ch（Kore 2022 と同等）
```

### 6-C. Phase 2 で追加するもの

```
+ FloodFill / 到達コストマップ                 (1-2 ch)
+ 距離マップ（自/敵 最寄り惑星距離）             (2 ch)
+ 脅威マップ                                   (1 ch)
→ 合計 22-24 ch
+ 60% ランダムピクセルマスク augment（学習時）
+ Entity Transformer ヘッド（並行構成）
+ Value / 得点差 auxiliary head
```

---

## 7. 参考文献

- [Imitation Learning by Semantic Segmentation for Halite IV (Kha Vo)](https://voanhkha.github.io/2020/09/15/halite/)
- [khanhvu207/kore2022 GitHub](https://github.com/khanhvu207/kore2022)
- [Kore 2022 Feature Generator (Kaggle)](https://www.kaggle.com/code/huikang/kore-2022-feature-generator)
- [Lux AI S3 3rd Place Imitation Learning Writeup](https://www.kaggle.com/competitions/lux-ai-season-3/writeups/adg4b-imitation-learning-3rd-place-solution)
- [Hungry Geese 3rd Place Solution (hoxomaxwell SpeakerDeck)](https://speakerdeck.com/hoxomaxwell/kaggle-hungry-geese)
- [An Exploration of Deep Reinforcement Learning Methods with Hungry Geese (arXiv)](https://arxiv.org/pdf/2109.01954)
- [Deciphering AlphaStar on StarCraft II (Yekun's ML Notes)](https://ychai.uk/notes/2019/07/21/RL/DRL/Decipher-AlphaStar-on-StarCraft-II/)
- [Standard Architecture | google-deepmind/alphastar (DeepWiki)](https://deepwiki.com/google-deepmind/alphastar/3.3-standard-architecture)
- [Fkaneko/kaggle_lux_ai GitHub](https://github.com/Fkaneko/kaggle_lux_ai)
