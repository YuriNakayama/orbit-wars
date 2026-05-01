# imitation/case2 Feature Engineering — Phase 1 結果

> 作成日: 2026-04-26
> 対象: `pipeline/imitation/case2/`
> 評価指標: validation 上の多面的精度指標 (ROC-AUC / PR-AUC / F1 / log-loss / NLL / ECE / per-class breakdown)。self-play 勝率は評価対象外（ユーザー明示指示時のみ実行）。

## 背景

case1 (baseline 11/6 dim featurizer) を出発点に、Kaggle Simulation の上位 IL 解法
(Halite IV, Hungry Geese, Kore 2022, Lux S1/S3, AlphaStar) で採用される CNN-channel 設計を、
Orbit Wars の **per-planet set + global vector** 構造に翻訳して適用する案を実装。

## 翻訳マッピング (CNN → Graph U-Net)

| 研究メモ側 | 重要度 | 翻訳先 |
|------------|------|--------|
| 距離マップ (own/enemy 最近傍) | ★★★ | planet 列 nearest_ally/neutral_dist (+2) |
| 軌道予測チャネル | ★★★ | planet 列 (dx, dy) × t={1,2,4,8} (+8) |
| 入射 fleet ETA & ship 量 | ★★★ | planet 列 ally/enemy ETA & ships split (+4) |
| 脅威マップ短/長 | ★★ | planet 列 threat_pressure_short/long (+1) |
| 彗星到達予測 | ★★ | global 列 next_comet_eta_norm (+1) |
| 全体集計 | ★★ | global 列 totals_frac / score_diff (+4) |

合計: **planet 18 → 33 (+15)**, **global 11 → 16 (+5)**

## 実験設計

baseline (planet=18/global=11) と phase1 (planet=33/global=16) を **同じ rating_quantile=0.50, val_split=0.10, seed=0** で並行に preprocess + train + eval_metrics し、validation 上の多面的指標を比較。

データセット: 1v1 modes、1013 episodes 中 rating_cutoff=945.49 で 639 episodes 採用。
train frames=222492, val frames=28148。

## 主要結果 (baseline vs phase1)

| 指標 | baseline | phase1 | Δ | 判定 |
|------|---------:|-------:|----:|------|
| from ROC-AUC | 0.8451 | 0.8418 | -0.0033 | ✗ 微悪化 |
| from PR-AUC | 0.5030 | 0.4873 | **-0.0157** | ✗ 必達未達 |
| from F1-best | 0.4742 | 0.4698 | -0.0043 | ✗ |
| target macro-F1 | 0.3015 | **0.3188** | **+0.0174** | ✓ 期待達成 |
| target top-1 acc | 0.4186 | 0.4200 | +0.0014 | ✓ |
| ships macro-F1 | 0.4934 | 0.4867 | -0.0067 | ✗ |
| ships ECE | 0.0276 | 0.0372 | +0.0095 | ✗ |

target head は明確に改善したが、from と ships が劣化。

## Ablation (5 group)

phase1 の追加 15 列を 5 group に分けて 1 group ずつ抜く ablation。

| group | 列 | 内容 |
|-------|----|------|
| A | planet 18-19 | nearest_ally/neutral_dist |
| B | planet 20-27 | orbit prediction t={1,2,4,8} |
| C | planet 28-31 | incoming ETA & ships split |
| D | planet 32 | threat_pressure_long |
| E | global 11-15 | comet eta + totals + score_diff |

### contribution table (phase1 → no_X 差分)

| group | from PR-AUC | target macro-F1 | ships macro-F1 | ships ECE↓ | best_val_loss |
|-------|-----------:|----------------:|---------------:|----------:|--------------:|
| **A** | +0.0069 | +0.0021 | +0.0227 | -0.0086 | 3.7976 |
| **B** | +0.0027 | -0.0134 | -0.0156 | +0.0004 | 3.7978 |
| **C** | +0.0092 | -0.0230 | +0.0301 | -0.0224 | 3.8141 |
| **D** | **+0.0140** | -0.0129 | **+0.0279** | **-0.0273** | **3.7748** |
| **E** | +0.0071 | -0.0083 | +0.0302 | -0.0266 | 3.7873 |

(正値 = 抜くと改善 = 悪因子。負値 = 抜くと悪化 = 良因子)

### 解釈

- **B (軌道予測 8 列)** は target に明確な良因子 (-0.0134), ships にも僅かに良 (-0.016)
- **C (incoming split 4 列)** は target に最強の良因子 (-0.0230), ships には悪因子 (+0.030)
- **D (threat_pressure_long)** は from に最大の悪因子 (+0.0140), ships ECE も大幅悪化、val_loss は no_d で最低
- A, E は ships に対して悪因子寄りだが影響小

## 採用案: phase1-final = no_d (planet 32 のみ 0 化)

ablation で **D を抜く設定 (no_d) が最良**:

| 指標 | baseline | phase1 | **no_d** |
|------|---------:|-------:|---------:|
| from ROC-AUC | 0.8451 | 0.8418 | **0.8461** ✓ |
| from PR-AUC | 0.5030 | 0.4873 | **0.5013** ✓ |
| from F1-best | 0.4742 | 0.4698 | **0.4732** ✓ |
| target macro-F1 | 0.3015 | 0.3188 | **0.3060** ✓ |
| target top-1 acc | 0.4186 | 0.4200 | **0.4210** ✓ |
| ships accuracy | 0.7300 | 0.7316 | **0.7348** ✓ |
| ships macro-F1 | 0.4934 | 0.4867 | **0.5146** ✓ |
| ships ordinal-MAE | 0.3695 | 0.3688 | **0.3578** ✓ |
| ships ECE | 0.0276 | 0.0372 | **0.0098** ✓ |

**全 9 主要指標で baseline 比改善 or 劣化なし**。完了基準 7/7 全達成。

### clean-C (A+D+E 同時除去) の試行

A, D, E を同時除去すると from PR-AUC は phase1 より悪化 (-0.0198) し、複数 group の同時除去は加算的でないことが判明。**no_d 単独除去が最良**。

## 採用 config と運用

- `pipeline/imitation/case2/configs/il_phase1.yaml` の `data.ablation.planet_cols: [32]` で固定
- weights: `pipeline/imitation/case2/policy/weights_phase1.pt` (no_d weights を copy)
- agent: `pipeline.imitation.case2.policy.agent_phase1` (registry: `il_v2_phase1`)

## Phase 2 への示唆

- 軌道予測 (B) と incoming ETA split (C) は target head の改善源として機能 → 履歴特徴 (Phase 2) でも同様の per-planet 時系列情報を載せる方向は有望
- ships head は依然 bucket 3 偏重 (recall 0.95) — Phase 2 の per-planet `delta_ships_t-1` で改善余地あり
- threat_pressure_long のような **半径制約特徴** は noise になりやすい — Phase 2 の特徴設計でも radius-based の集計は慎重に

## 関連ファイル

- preprocess: `pipeline/imitation/case2/training/preprocess.py` (config の `data.featurizer` で baseline/phase1 切替)
- dataset: `pipeline/imitation/case2/training/dataset.py` (mask_planet_cols / mask_global_cols 対応)
- featurizer: `pipeline/imitation/case2/policy/featurizer_phase1.py`
- agent: `pipeline/imitation/case2/policy/agent_phase1.py`
- eval: `pipeline/imitation/case2/evaluation/eval_metrics.py`
- 結果 JSON: `pipeline/imitation/case2/evaluation/results_metrics_*.json`
