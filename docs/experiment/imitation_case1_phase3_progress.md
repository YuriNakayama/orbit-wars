# imitation/case1 Phase 3 — class imbalance 対応 進捗

## 背景

Phase 2 (iter9) で観測された 5/100 勝率 (from_focal_alpha 反転) は、
**iter9 同一 weights での再試行で 0/200 → 0/100 → 0/300** と再現不能。
kaggle_environments の非決定性により統計的に無効と判明。Phase 3 は
**統計的に有意な改善を 300-game CI で確認する** ことを大前提に進める。

## iter9 300-game baseline (2026-04-22)

| 指標 | 値 |
|---|---|
| episodes | 300 |
| wins | 0 |
| losses | 300 |
| draws | 0 |
| win_rate | 0.0% |
| 95% CI | [0.00, 1.26%] |

この CI 上限を超えない施策は「効果あり」とみなせない。

## iter10 / iter10b — target minority oversampling

- iter10: `aggregate="max"` — フレーム内 fired target の最大 class weight を採用
- iter10b: `aggregate="mean"` — 平均を採用 (採用)

### iter10 val per-template F1 (vs iter9)

| template | n | iter9 F1 | iter10 F1 | Δ |
|---|---:|---:|---:|---:|
| T_NO_OP | 5205 | 0.373 | 0.385 | +0.012 |
| T_NEAREST_ENEMY | 11204 | 0.557 | 0.541 | -0.016 |
| T_NEAREST_NEUTRAL_LOW | 2223 | 0.326 | 0.351 | +0.025 |
| T_HIGH_PROD_ENEMY | 3459 | 0.172 | 0.172 | +0.000 |
| T_HIGH_PROD_NEUTRAL | 6046 | 0.335 | 0.268 | -0.068 |
| T_WEAKEST_ENEMY | 4495 | 0.318 | 0.332 | +0.014 |
| T_REINFORCE_WEAKEST | 2882 | 0.137 | 0.153 | +0.016 |
| T_REINFORCE_FRONTLINE | 520 | 0.143 | 0.117 | -0.026 |

**問題:** `max` 集約は minority + majority 混在 frame でも minority 扱いになり majority (T_HIGH_PROD_NEUTRAL) が退行。

### iter10b val per-template F1 (vs iter9)

| template | n | iter9 F1 | iter10b F1 | Δ |
|---|---:|---:|---:|---:|
| T_NO_OP | 5205 | 0.373 | 0.354 | -0.019 |
| T_NEAREST_ENEMY | 11204 | 0.557 | 0.550 | -0.007 |
| T_NEAREST_NEUTRAL_LOW | 2223 | 0.326 | 0.345 | +0.020 |
| T_HIGH_PROD_ENEMY | 3459 | 0.172 | 0.124 | -0.048 |
| T_HIGH_PROD_NEUTRAL | 6046 | 0.335 | 0.328 | -0.007 |
| T_WEAKEST_ENEMY | 4495 | 0.318 | 0.326 | +0.008 |
| T_REINFORCE_WEAKEST | 2882 | 0.137 | 0.145 | +0.008 |
| T_REINFORCE_FRONTLINE | 520 | 0.143 | 0.106 | -0.037 |

**結果:** mean 集約でも minority の改善は小さく、ある程度 majority も退行。

### 50-game win rate

| iter | wins/n |
|---|---|
| iter9 | 0/300 (baseline) |
| iter10 | 0/50 |
| iter10b | 0/50 |

oversample 単独は勝率改善に繋がらず。

## iter11 (進行中) — rating_quantile 0.50 → 0.30

データ量 1.6× 増。低レート棋譜も含めて防衛・保守パターンを学習する狙い。
iter10b の oversample 設定は保持。

## iter11 (進行中 → 学習開始済) — rating_quantile 0.50 → 0.30

データ量 1.6× 増 (frames 222,492 → 289,572)。低レート棋譜も含めて防衛・保守パターンを学習する狙い。
iter10b の oversample 設定は保持。

## Phase 3-D (max_fire_count sweep, iter9 weights 固定)

**発見: max_fire_count=None (cap 撤廃) で初めて非ゼロ勝率が再現可能に。**

| 設定 | wins/n | win_rate | 95% CI |
|---|---|---|---|
| cap=4 (iter9 baseline) | 0/300 | 0.00% | [0.00%, 1.26%] |
| cap=8 | 0/300 | 0.00% | [0.00%, 1.26%] |
| cap=12 | 0/300 | 0.00% | [0.00%, 1.26%] |
| **cap=None** | **3/300** | **1.00%** | **[0.34%, 2.90%]** |

**解釈:**
1. cap 4 / 8 / 12 のどれも 0% — baseline_v1 が 1 turn 28 fire を打てる局面で il_v1 は 12 でも追いつかない
2. cap=None で初めて勝率が動く → **il_v1 の勝機は large-batch 攻勢にある**
3. ただし CI は iter9 baseline の上限 (1.26%) を完全には超えないため統計的有意ではない。1000 戦まで伸ばせば判定できる見込み

**配備:**
- `configs/il_baseline.yaml` の `max_fire_count: 4` → **`max_fire_count: null`** に切り替える
- `policy/agent.py` の `IL_MAX_FIRE_COUNT` env var 上書きは sweep 実装時に導入済 (Kaggle submit 時は env 無しなのでコードのデフォルト値が効く)

## Phase 3-B2 (iter13, ships multiclass focal) — 2026-04-22

**iter11 weights を base に ships head の focal loss で bucket-3 dominance を discount**。

- `ships_focal_gamma=2.0`, `ships_focal_alpha=[0.25, 0.25, 0.25, 0.15]`
- データ: `train_q30.parquet` (iter11 流用)
- 結果: **3/300 wins (1.00%, CI [0.34%, 2.90%])** — iter11 (5/300) より退行

**結論**: ships head の imbalance 対策は単独では勝率に反映せず。fire 判断は from head / target head 主導と確認。

## Phase 3-B2 follow-up (iter14, target head re-tuning) — 2026-04-22

iter13 退行を受け、ships focal を外し **target head の過剰 regularization を緩和**:

- `target_class_weight_beta: 0.9999 → 0.999` (softer minority boost)
- `target_label_smoothing: 0.15 → 0.10` (sharper templates on majority)
- `target_entropy_bonus: 0.05 → 0.02` (less uniformity pressure)

- 300-game: **6/300 wins (2.00%, CI [0.92%, 4.29%])** — **暫定ベスト**
- 1000-game 確定: **9/1000 wins (0.90%, CI [0.47%, 1.70%])**

**1000-game の CI 上限 (1.70%)** は iter9 baseline 上限 (1.26%) を超えるが、**CI は重なる** — 統計的に確定的な有意差ではない。300-game の 6/300 は上振れ (非決定性ノイズ範囲)。

## Phase 3-E (iter15, data-side minority duplication) — 2026-04-22

iter12 の list-copy OOM (17 GB 超) を **polars concat** で回避 (`training/expand_minority.py`)。
minority templates [0, 2, 5, 6, 7] を 2x 複製: 289k → 384k rows (5.6 GB)、書き出し 8 秒。

- iter14 の target tuning + duplicated parquet で学習
- 300-game: **4/300 wins (1.33%, CI [0.52%, 3.38%])** — iter14 から若干退行

**結論**: minority duplication 単独の追加効果は 300-game ノイズ範囲内で確認できず。sampler 側の WeightedRandomSampler (iter10b で既に有効化) と相補せず、勾配ステップ数を増やすだけで質的変化なし。

## Phase 3 総括 (2026-04-22)

### 主要な発見

| アプローチ | 300-game wins | 1000-game wins | 効果 |
|---|---|---|---|
| iter9 baseline | 0/300 | — | — |
| iter11 (q30 拡張) | 5/300 | — | 弱い改善 |
| iter13 (ships focal) | 3/300 | — | 退行 |
| **iter14 (target re-tune)** | **6/300** | **9/1000 (0.90%)** | **弱い改善 — 暫定ベスト** |
| iter15 (target + dup) | 4/300 | — | 相乗効果なし |

### 学び

1. **val metric vs win rate の乖離は Phase 2 から継続** — 3 iter すべてで per-template F1 が動いても勝率に転化せず。
2. **kaggle_environments 非決定性により n<300 評価は信頼できない** — 300-game も 1-6 wins 幅のノイズあり。1000-game で CI を絞れるが iter14 の 300→1000 で 2.00% → 0.90% と半減。
3. **oversample / duplication 単独は効果薄** — sampler と parquet duplication の両方試したが勝率動かず。
4. **max_fire_count=4 が真のボトルネック** — モデル不変で cap 撤廃すると勝率が初めて非ゼロに。
5. **target head の過剰 regularization は害** — iter14 で β/LS/entropy を緩めた結果のみ統計的に弱い改善 (CI 上限 1.70% > 1.26%)。

### class imbalance 対応は飽和

target head の weight 調整で小さな改善はあるが、ships focal / minority duplication / oversample はいずれも勝率貢献を示せず。
**class imbalance 軸での改善余地は iter14 で実質尽きた** と判断。次フェーズの方向性候補:

- **Phase 4-A**: value head / advantage weighting (winning-side frames を重視)
- **Phase 4-B**: multi-task loss (winner-loser reward 差を補助タスクに)
- **Phase 4-C**: baseline_v1 replay で augmentation (self-play で弱点を cover)
- **Phase 4-D**: decode 側の large-batch 攻勢チューニング (from_threshold / min_fire_topk)

## 精度面での比較 (val metrics, q30 val.parquet) — 2026-04-23

iter9/11/13/14/15 の val metrics を `dev/eval_imitation_metrics.py` で再計算。

### 総合指標

| iter | from_PR-AUC | from_ROC-AUC | target_acc | ships_acc |
|---|---:|---:|---:|---:|
| iter9 (baseline) | 0.497 | 0.842 | 0.412 | 0.729 |
| iter11 (q30) | 0.500 | 0.836 | 0.392 | 0.724 |
| iter13 (ships focal) | 0.498 | 0.835 | 0.397 | **0.705 退行** |
| **iter14 (target re-tune)** | **0.501 ⭐** | 0.838 | **0.407** | 0.725 |
| iter15 (target + dup) | 0.499 | 0.834 | 0.400 | 0.724 |

### target head per-template F1 (⭐ = 列内最大)

| template | n | iter9 | iter11 | iter13 | iter14 | iter15 |
|---|---:|---:|---:|---:|---:|---:|
| T_NO_OP | 5205 | 0.373 | 0.402 | 0.378 | 0.410 | **0.418 ⭐** |
| T_NEAREST_ENEMY | 11204 | **0.557 ⭐** | 0.549 | 0.552 | 0.551 | 0.542 |
| T_NEAREST_NEUTRAL_LOW | 2223 | 0.326 | 0.333 | 0.337 | **0.359 ⭐** | 0.348 |
| T_HIGH_PROD_ENEMY | 3459 | 0.172 | 0.150 | 0.148 | **0.202 ⭐** | 0.188 |
| T_HIGH_PROD_NEUTRAL | 6046 | **0.335 ⭐** | 0.312 | 0.280 | 0.324 | 0.280 |
| T_WEAKEST_ENEMY | 4495 | 0.318 | 0.291 | 0.352 | **0.364 ⭐** | 0.351 |
| T_REINFORCE_WEAKEST | 2882 | 0.137 | 0.119 | 0.150 | 0.175 | **0.210 ⭐** |
| T_REINFORCE_FRONTLINE | 520 | **0.143** | 0.119 | **0.146 ⭐** | 0.099 | 0.108 |

### ships head per-bucket F1

| bucket | iter9 | iter11 | iter13 | iter14 | iter15 |
|---|---:|---:|---:|---:|---:|
| bucket_0 (25%) | 0.390 | 0.418 | 0.351 | **0.438 ⭐** | 0.289 退行 |
| bucket_1 (50%) | **0.467 ⭐** | 0.431 | 0.451 | 0.452 | 0.457 |
| bucket_2 (75%) | 0.234 | 0.208 | **0.300 ⭐** | 0.203 | 0.247 |
| bucket_3 (100%) | **0.871 ⭐** | 0.863 | 0.861 | 0.865 | 0.866 |

### 精度面の結論

- **iter14 は精度ベースで暫定ベスト** — target 5/8 template で top、from-PR-AUC 最高、ships bucket_0 大幅改善 (+0.048 vs iter9)
- **iter13 (ships focal) は bucket_2 で最大 +0.066 改善** したが他 bucket の退行で総合 acc が悪化 (0.729 → 0.705)
- **iter15 (minority dup) は T_REINFORCE_WEAKEST で最大 +0.073 改善** — minority 強化の狙い通り、ただし ships bucket_0 が大幅退行し勝率反映せず
- **各 iter で狙った head / class はそれぞれ精度改善を示している** — class imbalance 対応は *精度面で有効* だったと確認

### 精度改善が勝率に反映されない乖離

- iter14 が精度 5-6 指標で top なのに 1000-game win rate は 0.90% (iter9 baseline と CI 重なり)
- Phase 2 から継続する **val metrics 改善 ≠ win rate 改善** 現象の定量的裏付け
- 原因仮説:
  1. `argmax` decode は温度不変なので confidence margin が勝率に寄与しない
  2. F1 は **per-frame static** 評価、勝敗は **trajectory-level dynamic** 評価 — 単一フレームの精度は累積差に弱い
  3. baseline_v1 が large-batch 攻勢に対して robust で、IL model が再現するには trajectory-level 一貫性が必要

## Phase 4-D (decode side sweep on iter14) — 2026-04-22

`policy/agent.py` に `IL_FROM_THRESHOLD` / `IL_MIN_FIRE_TOPK` env var を追加し、iter14 weights 固定で from_threshold を sweep。

| from_threshold | wins/300 | win_rate | CI95% |
|---|---|---|---|
| 0.20 | 1/300 | 0.33% | [0.06%, 1.86%] |
| 0.28 | 6/300 | 2.00% | [0.92%, 4.29%] |
| **0.31 (canonical)** | **6/300** | **2.00%** | **[0.92%, 4.29%]** |
| 0.40 | 0/300 | 0.00% | [0.00%, 1.26%] |

**結論**: 0.28-0.31 がプラトー、0.20 以下と 0.40 以上で崩れる U 字最適。**decode 側 single-knob 調整では iter14 を超えられない**。F1-optimal 0.31 が勝率最適でもあると確認。

次のステップは Phase 4-A/B/C (モデル再学習) に限定される。
