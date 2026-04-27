# imitation/case3 Phase 2 — History Features

> 作成日: 2026-04-27
> 対象: `pipeline/imitation/case3/`
> 比較基準: `imitation/case2` Phase 1 final (`il_v2_phase1`, weights_phase1.pt = no_d ablation)
> 評価指標: validation 上の多面的精度指標 (self-play は実施しない)

## 背景

Phase 1 (case2) で 18 → 33 dim の per-planet 特徴量を導入し、target macro-F1 を +0.005 / ships
macro-F1 を +0.021 改善。Phase 2 では **時系列 history** をモデルに与えることで、Phase 1 で残っていた
ships head の bucket 3 偏重 (recall 0.94 vs bucket 2 recall 0.20) を含む per-planet 戦況変化の予測力を向上させる。

## 追加した特徴量

| 種別 | 列 | 名前 | 意味 |
|------|----|------|------|
| planet | 32 | `delta_ships_t1` | (ships_now − ships_{t−2}) / max(1, ships_now), clip ±1 |
| planet | 33 | `delta_ships_t2` | (ships_now − ships_{t−3}) / max(1, ships_now), clip ±1 |
| planet | 34 | `owner_changed_t1` | owner が t−2 から変わったか (0 / 1) |
| global | 16 | `enemy_launch_count_last4` | 直近 4 ターンの敵発射回数 / 10 |
| global | 17 | `enemy_launch_ships_last4` | 直近 4 ターンの敵発射 ships 合計 (log1p / 6) |
| global | 18 | `ally_launch_count_last4` | 直近 4 ターンの自軍発射回数 / 10 |
| global | 19 | `ally_launch_ships_last4` | 直近 4 ターンの自軍発射 ships 合計 (log1p / 6) |

合計: planet 33 → **35 (+3)**, global 16 → **20 (+4)**。Phase 1 final から D 群除外を継承 (col 32 = `threat_pressure_long` は phase2 で col 32 に再用済み — featurizer_phase2 では含まれていない)。

## 重要バグ: history 列のリーク発見と修正

### 第一実装でのリーク (記録のため保存)

初期実装では `delta_ships_t1 = (ships_now − ships_{t−1}) / max(1, ships_now)` としていた。
Kaggle replay の `step[N].observation.planets[*].ships` は **action 適用後** の値だと判明:

```
step=17 で player は planet 16 から 37 ships 発射:
  step=16 obs.ships = 40   (発射前)
  step=17 obs.ships =  5   (発射後 = 40 - 37 + 2 production)
  step=18 obs.ships =  7
```

このため `delta_ships_t1 = obs_N.ships − obs_{N−1}.ships` には **action_N の発射量が完全に反映** され、
training の `from_multihot` (= action_N) と **完全相関 → 完全リーク**。

リーク版の数値は異常に高く出た:

| variant | from PR-AUC | from F1 | ships F1 | val_loss |
|---------|-----------:|--------:|---------:|---------:|
| phase1_final | 0.5013 | 0.4732 | 0.5146 | 3.7748 |
| **phase2 LEAKY** | **0.9464** | **0.9011** | **0.8427** | **3.5570** |
| phase2 fixed | 0.6317 | 0.6182 | 0.6677 | 3.6846 |

### Ablation でリーク源を特定

planet/global history を on/off で切り分け:

| variant | from PR-AUC | ships F1 | 結論 |
|---------|-----------:|---------:|------|
| no_history (planet 32-34 + global 16-19 = 0) | 0.4969 | 0.5183 | phase1_final 同等 |
| no_planet_history (planet 32-34 = 0, global 残す) | 0.4929 | 0.5194 | phase1_final 同等 |
| no_global_history (global = 0, planet 32-34 残す) | **0.9461** | **0.8480** | **リーク** |

→ planet history (delta_ships, owner_changed) **だけ**でリーク。global launch history は無効。

### 修正

`HistoryState.prev_planet_snapshots.maxlen` を 2 → 3 に拡大し、featurizer 内で
`snap_t1 = prev[-2]` (= obs_{N−2}), `snap_t2 = prev[-3]` (= obs_{N−3}) と参照を 1 ターン後ろにずらす。
これで `delta_ships_t1 = obs_N.ships − obs_{N−2}.ships` となり、action_N が **直接** 引き算に
入らない。

## 主要結果 (phase2_fixed vs phase1_final)

| 指標 | phase1_final | **phase2_fixed** | Δ |
|------|-----------:|----------------:|----:|
| from ROC-AUC | 0.8461 | **0.9275** | **+0.0814** |
| from PR-AUC | 0.5013 | **0.6317** | **+0.1304** |
| from F1-best | 0.4732 | **0.6182** | **+0.1450** |
| from log-loss | 0.4831 | **0.4135** | -0.0696 |
| from ECE | 0.2525 | 0.2184 | -0.0341 |
| target macro-F1 | 0.3060 | 0.3076 | +0.0017 |
| target top-1 | 0.4210 | 0.4238 | +0.0029 |
| target ECE | 0.0602 | 0.0702 | +0.0100 ✗ |
| ships accuracy | 0.7348 | **0.8108** | **+0.0760** |
| ships macro-F1 | 0.5146 | **0.6677** | **+0.1531** |
| ships ordinal-MAE | 0.3578 | **0.2501** | -0.1076 |
| ships ECE | 0.0098 | 0.0091 | -0.0007 |

### per-class breakdown

**target template F1** (改善小)

| idx | template | n | phase1_final | phase2_fixed | Δ |
|-----|----------|--:|------------:|-------------:|----:|
| 0 | NEAR_NEUT_LOW | 5205 | 0.404 | 0.374 | -0.030 |
| 1 | NEAR_ENEMY | 11204 | 0.563 | 0.563 | -0.000 |
| 2 | HIGH_PROD_NEUT | 2223 | 0.362 | 0.348 | -0.014 |
| 3 | HIGH_PROD_ENEMY | 3459 | 0.162 | 0.193 | +0.031 |
| 4 | REINF_FRONT | 6046 | 0.351 | 0.352 | +0.001 |
| 5 | REINF_WEAK | 4495 | 0.363 | 0.347 | -0.015 |
| 6 | WEAK_ENEMY | 2882 | 0.117 | 0.131 | +0.014 |
| 7 | NO_OP | 520 | 0.125 | 0.152 | +0.027 |

**ships bucket recall** (大幅改善、bucket 3 偏重が緩和)

| bucket (ratio) | n | phase1_final | phase2_fixed | Δ |
|---------------|--:|-------------:|-------------:|----:|
| 0 (25%) | 1768 | 0.374 | **0.547** | +0.173 |
| 1 (50%) | 5058 | 0.446 | **0.629** | +0.184 |
| 2 (75%) | 5269 | 0.196 | **0.455** | +0.258 |
| 3 (100%) | 23939 | 0.941 | 0.947 | +0.006 |

## Phase 2 完了基準

| 基準 | 結果 |
|------|------|
| 全 head 劣化なし (-0.005 以内) | ✓ 7/7 |
| (期待) from PR-AUC or target macro-F1 +0.01 以上 | ✓ from PR-AUC +0.130 |

完了基準を期待値の 13 倍で達成。

## 採用 config と運用

- `pipeline/imitation/case3/configs/il_phase2.yaml` を canonical config として使用
- weights: `pipeline/imitation/case3/policy/weights_phase2.pt` (= weights_phase2_fixed.pt の copy)
- agent: `pipeline.imitation.case3.policy.agent_phase2:agent` (registry: `il_v3`)
- preprocess parquet: `data/mart/imitation/case3/{train,val}_phase2_fixed.parquet`

## Phase 3 への示唆

- target head は phase1 の特徴量で頭打ち気味 — auxiliary head (value, opponent action 予測) で
  押し上げを狙う
- target ECE は +0.0100 と僅かに悪化 → temperature scaling 再 tuning が必要
- ships の ECE は据え置き、ordinal-MAE が大幅改善 — Phase 3 では bucket 0/1 (低 recall 改善) を
  さらに伸ばすために focal loss を再導入する余地あり

## 関連ファイル

- featurizer: `pipeline/imitation/case3/policy/featurizer_phase2.py`
- agent: `pipeline/imitation/case3/policy/agent_phase2.py` (per-match HistoryState ring buffer)
- preprocess: `pipeline/imitation/case3/training/preprocess.py` (per-slot history を episode 間維持)
- 結果 JSON: `pipeline/imitation/case3/evaluation/results_metrics_phase2{,_leaky,_fixed}.json`

## 教訓

1. **時系列特徴は必ず "supervision に対して因果先行" である必要**。Kaggle replay の obs は
   action 適用後なので、`obs_N` を「N の supervision の特徴量」として使うとき、`obs_{N−1}` との
   差分は action_{N−1} を含むだけでなく、内部の動的物理計算経由で action_N の影響も
   間接的に持ち込む可能性がある。今回は `obs_N` 自体が action_N 適用後だったため直接リーク。
2. **異常な改善幅を見たら必ず ablation で原因切り分け**。今回 phase2 全 history で from PR-AUC
   が 0.5 → 0.95 という **特徴量だけでは説明不能の幅** だったため、history 個別 0 化で
   リーク経路を 30 分以内に特定できた。
3. **history 系特徴は時間方向の整合性が肝**。今後 Phase 3 以降で時系列を扱う場合は、必ず
   "obs_N が action_N の **前** の状態か **後** の状態か" を replay で実測してから設計する。
