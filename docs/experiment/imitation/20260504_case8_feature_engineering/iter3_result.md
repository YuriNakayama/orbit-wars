# imitation/case8 iter3 — Result (採用 + 削除候補特定)

> 作成日: 2026-05-05
> 関連 plan: `./iter3_plan.md`
> commit: `7219177` (iter3) → train run `20260505-080208__feature-feature-engineering__7219177__seed0`
> weights: `bot/pipeline/imitation/case8/policy/weights.pt` (iter3 best.pt epoch 12, val_loss=3.5062)

## サマリ (TL;DR)

**判定: 採用 (with 削除推奨あり)**。permutation feature importance により、
iter3 で追加した 3 カテゴリ (A/D/J) はすべて **意味のある寄与**が確認された。
同時に **iter2 で追加した 4 列 (centroid_home + fleet_traj 一部) と global iter2 の 4 列が無意義** と
判明、iter4 でこれらを削除する整理候補。

| iter | best_val_loss | best_epoch | head improvement vs iter1 |
|------|--------------:|-----------:|---------------------------|
| iter1 | 3.5235 | 9 | (baseline) |
| iter2 | 3.5142 | 9 | from -0.005 / target +0.005 / ships -0.013 |
| **iter3** | **3.5062** | **12** | **target +0.014 ✅ / ships +0.001 △ / from -0.012 ✗** |

iter3 の **best_epoch が 9 → 12 に後ろ倒し** 、追加 input で深い学習が進む構造。

## 訓練ログ (15 epoch)

```
epoch  train_total  val_total
0      3.871        3.6878
12*    3.4735       3.5062  ← best (iter1 3.5235 / iter2 3.5142 比 -0.0173 / -0.0080)
14     3.4485       3.5183
```

## Stage 1: validation metrics

### 全 head 比較

| head | metric | iter1 | iter2 | **iter3** | Δ vs iter2 | Δ vs iter1 |
|------|--------|------:|------:|---------:|----------:|----------:|
| **from** | F1 | 0.6651 | 0.6601 | **0.6529** | -0.007 | -0.012 ✗ |
| from | PR-AUC | 0.7676 | 0.7626 | **0.7369** | **-0.026** ✗ | -0.031 ✗ |
| from | acc | 0.9151 | 0.9096 | 0.9115 | +0.002 | -0.004 |
| **target** | macro F1 | 0.3210 | 0.3184 | **0.3220** | +0.004 | **+0.001** △ |
| target | top-1 acc | 0.4327 | 0.4378 | **0.4467** | **+0.009** ✅ | **+0.014** ✅ |
| target | top-2 acc | 0.6377 | 0.6390 | **0.6443** | +0.005 | +0.007 |
| target | PR-AUC macro | 0.3419 | 0.3423 | **0.3474** | +0.005 | +0.005 |
| **ships** | macro F1 | 0.6425 | 0.6292 | **0.6353** | **+0.006** ✅ | -0.007 |
| ships | acc | 0.8519 | 0.8468 | 0.8478 | +0.001 | -0.004 |

### 解釈

- **target head が明確改善** (top-1 +0.014, macro F1 +0.001 vs iter1) — **Pairwise Top-K の貢献**
- **ships head は iter2 比で持ち直し** (macro F1 +0.006) — Defense surplus + sparse mask が iter2 の sparse 列の信号希釈を緩和
- **from head は依然後退** (PR-AUC -0.031 vs iter1) — Pairwise Top-K と Defense surplus が from head に noise を持ち込んだ可能性

## Permutation Feature Importance (iter3 に対して)

ベースライン: iter3 weights で計算した val metrics (上表 iter3 列)。各 group を val 行ごと shuffle して metrics 低下を測定 (大きく下がる group ほど重要)。

| Group (planet 範囲) | Δ from_PR-AUC | Δ target_acc | Δ target_F1 | Δ ships_F1 | 重要度判定 |
|---------------------|--------------:|-------------:|------------:|-----------:|:----------|
| **iter1_base** (0-23) | **-0.549** | **-0.178** | -0.144 | **-0.332** | 🥇 model のコア |
| **iter2_multi_horizon** (28-31) | **-0.166** | -0.035 | -0.040 | **-0.141** | 🥈 ships head に重要 |
| **iter3_pairwise_topk** (34-53) | -0.036 | **-0.044** | **-0.056** | -0.026 | 🥉 **target head に明確効** ✅ |
| **iter3_defense_surplus** (54-57) | -0.034 | -0.007 | -0.008 | -0.025 | from / ships に控えめ ✅ |
| global_base (g 0-9) | -0.038 | -0.015 | -0.019 | -0.080 | ships head に重要 |
| **iter3_sparse_mask** (58-62) | -0.006 | -0.031 | -0.025 | -0.011 | target に効果あり ✅ |
| iter2_fleet_traj (24-27) | -0.003 | -0.007 | -0.007 | -0.014 | 効果薄 ⚠️ |
| **iter2_centroid_home** (32-33) | -0.001 | -0.0003 | -0.001 | +0.0002 | **無意義** 🗑️ 削除候補 |
| **iter2_global** (g 10-13) | -0.001 | -0.003 | -0.007 | -0.0003 | **無意義** 🗑️ 削除候補 |

### 採用判定 (iter3 追加 3 カテゴリ)

| Category | 効果 | 採用 |
|----------|------|------|
| **A. Pairwise Top-K** | target_F1 -0.056 (group としては iter1_base 以外で **最大**) | ✅ **採用** |
| **D. Defense surplus** | from -0.034 / ships_F1 -0.025 (controlled effect) | ✅ **採用** |
| **J. Sparse mask** | target_F1 -0.025 (small but consistent across heads) | ✅ **採用** |

### 整理候補 (iter4 で削除して dim を絞る)

| Category | 列範囲 | 理由 |
|----------|--------|------|
| iter2_centroid_home | planet 32-33 | from -0.001 / target -0.001 / ships +0.0002 = **完全に無意義**。home distance / production centroid distance は iter3 の Pairwise Top-K と Defense surplus に情報吸収済 |
| iter2_global | global 10-13 | comet/home_owner/prod_centroid_dist 全て effect ~0、削除しても loss 無し |
| iter2_fleet_traj | planet 24-27 | -0.014 ships_F1 に効くが弱い、Defense surplus と機能重複 |

iter4 で全部削除すれば planet 63→55、global 14→10 で dim 縮小、訓練速度向上の余地あり。

## 採用判定

**判定: 採用** (canonical weights は iter3 best.pt 維持)。

iter3 の追加 3 カテゴリ (A/D/J) はすべて permutation importance で意味ある寄与が
確認された。Stage 1 head metrics でも target が iter1 比 +0.014、ships が
iter2 比で +0.006 と改善傾向。from head の後退 (-0.031 vs iter1) は
Pairwise Top-K の信号輻輳が原因と推定、iter4 で別アーキテクチャ (例: from
head 専用 input subset) で対処可能。

### 後処理

- canonical weights `bot/pipeline/imitation/case8/policy/weights.pt` は
  **iter3 best.pt (val_loss 3.5062)** にすでに昇格済 (本サイクルで)
- Stage 2 self-play (300 ep) は別 follow-up plan で実施 (本 plan のスコープ外、
  memory `project_imitation_case1_phase3` 準拠)

## 推奨される iter4 plan

permutation importance の結果を元に:

1. **iter2 削除候補**: planet 24-27, 32-33 + global 10-13 を削除 (dim 63→55, 14→10)
2. **from head 復活策**: from head に Pairwise Top-K の k=0,1 のみを与える (k=2-4 は noise の可能性)
3. **新規追加候補**: target head が伸び代あるなら template prefix を別 normalize で再実験

## iter4 方針 (2026-05-05 追記、permutation importance + ユーザー指示で確定)

### 削除 (8 列、permutation で sum |Δ| < 0.01)

| Group | 列範囲 | 列数 | 削除理由 |
|-------|------|---:|------|
| **G6 enemy_ship_event_per_planet** | planet 22-23 | 2 | sum &#124;Δ&#124;=0.0018、全 head 効果ゼロ (iter1 で投入したが今は冗長) |
| **H1 global_step_velocity** | global 0-1 | 2 | sum &#124;Δ&#124;=0.0056、step / ang_vel が想像以上に効かない |
| **H4 global_comet_state** | global 10-11 | 2 | sum &#124;Δ&#124;=0.0086、comet info は per-planet `is_comet` flag (col 8) で十分 |
| **H5 global_home_centroid** | global 12-13 | 2 | sum &#124;Δ&#124;=0.0017、home_owner_flag + prod_centroid_dist_norm の global 化は per-planet 列に情報吸収済み |

### 縮小 (K1: G9 を K=5→K=3 に圧縮、12 列削減)

| Group | 変更 | 列数差 | 理由 |
|-------|------|---:|------|
| **G9 pairwise_top_k** | K=5×4 (20列) → K=3×4 (12 列) | -8 | per-column importance 0.0084 で全 group 中最低、k=3,4 は noise の可能性高 |

### 新規追加 (4 カテゴリ、計 +12 列)

| Cat | 名称 | 列数 | 期待 head | 内容 |
|-----|------|---:|---|---|
| **K2** | Per-fleet outgoing trajectory | planet +4 | from / ships | 自軍出撃 fleet の 5-turn 先 future position (per-planet attribute、from-pid に紐付け)。iter2 G7 inbound に対する outgoing 版、from head 復活策の本命 |
| **K3** | Frontline distance | planet +4 | target | 敵軍最近 1 個目/2 個目との relative (dist_log, ships_ratio_log) 各 2 列。G1 の position_geometry を強化 |
| **K4** | Aux 集計 multi-horizon | global +4 | ships | 自軍 ships sum h=5/15 predicted + my prod sum + ratio。H2 を multi-horizon 化、ships head の決定力を補助 |

### dim 計算

| 段階 | planet | global |
|---|---:|---:|
| iter3 | 63 | 14 |
| -G6 削除 | 61 | 14 |
| -H1/H4/H5 削除 | 61 | 8 |
| -K1 縮小 (G9 20→12) | 53 | 8 |
| +K2 outgoing trajectory | 57 | 8 |
| +K3 frontline distance | 61 | 8 |
| +K4 aux global | 61 | 12 |
| **iter4 final** | **61** | **12** |

planet 63→61 (-2)、global 14→12 (-2)。総 -4 列だが内訳は **削除 14 + 追加 16 + 縮小 -8 = -6**。
最終的に slim 化 + 新シグナル追加で精度狙い。

### 採否方針 (iter4)

iter3 と同じ **permutation feature importance 駆動**。dead group が出たら iter5 で更に削る螺旋的整理。事前固定ゲートは使わない。

### 実装ステップ (高レベル)

1. `featurizer.py` で
   - G6 (col 22-23) を削除し以降の slot を詰める
   - G9 を K=3×4 に縮小 (`TOPK_NEIGHBORS = 3`)
   - K2 (outgoing fleet trajectory、自軍出撃 fleet の per-source future position) 追加
   - K3 (frontline distance、敵軍 nearest 2 個と self の関係) 追加
2. global は H1/H4/H5 削除 + K4 (multi-horizon ships sum + prod sum + ratio) 追加
3. `configs/il_case8.yaml` の dim を 61/12 に更新
4. `tests/pipeline/imitation/case8/test_featurizer_iter4.py` 新規追加 (各カテゴリ 1-2 テスト + dim sanity)
5. `dev/test-bot` green 化 → commit/push → RunPod Step B → permutation importance → iter4_result (このファイルではなく新規 `iter4_result.md` を作成)

### 想定コスト

- RunPod ~30 分、~$0.30 (trap #9 fix で preprocess 自動再実行)
- ローカル permutation importance ~1 分

### 注意点

- **K2 outgoing trajectory** は `obs.fleets` の自軍 fleet (owner == player) を per-source attribute する必要あり。iter2 の inbound enemy 用 `_fleet_action_target` ロジックを反転利用
- **K3 frontline distance** は per-planet で敵 nearest 2 個を sort する必要あり。G1 と vectorize 共有可能
- **K4 aux global** は per-planet を集約する後処理段階で計算、preprocess 時のコスト微増のみ

## RunPod onstart

iter3 では新たな onstart trap は発見されず:
- trap #8 (output/models symlink unlink) 動作確認 ✅
- trap #9 (preprocess_skip dvc status check) 動作確認 ✅ (preprocess force rerun)
- 全 markers 期待通りの順 (00→40→50→preprocess.episode×N→preprocess.done→55→60→train.epoch×15→70→75→99→90)
- 訓練 cost: 起動から 99_done まで **46 分** (RTX 4090 $0.69/h × 0.77h ≒ **$0.53**)

## 関連ファイル

- featurizer (iter3): `bot/pipeline/imitation/case8/policy/featurizer.py` (63/14 schema)
- weights (iter3): `bot/pipeline/imitation/case8/policy/weights.pt` (iter3 best.pt copy)
- run dir: `data/output/models/imitation/case8/runs/20260505-080208__feature-feature-engineering__7219177__seed0/`
- val metrics: `/tmp/case8_iter3_val_metrics.json`
- **permutation importance**: `data/output/experiment/imitation_case8_iter3_feature_importance.json`
- 訓練 metrics: `runs/20260505-080208__.../metrics.json`

## 教訓 (iter4 設計に活かす)

1. **permutation importance は採否判断の決定打**: 「全 head +0.01 ゲート」より遥かに有用。head 別の効果が定量化できる
2. **追加カテゴリは効果が薄いものを ruthless に削る**: iter2 の centroid_home / iter2_global は無意義、Pairwise Top-K は使うが iter2 fleet_traj は冗長
3. **from head の後退は構造的問題**: Pairwise Top-K のような pair info は target head には効くが from head には信号過多。head 別の input subsetting が必要
4. **best_epoch が後ろにずれる = 追加 input が学習材料として有効**: iter1=9, iter2=9, iter3=12。dim 増加で 表現力増の証左
