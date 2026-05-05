# imitation/case7 iter5 — Result (🚨 K2 causal leak 確定)

> 作成日: 2026-05-06
> 関連: `./iter5_plan.md` / `./iter4_result.md`
> 訓練ラン: 無し (診断のみ、ローカル ~10 分)

## サマリ (TL;DR)

**判定: iter4 ships F1 0.9349 は K2 (outgoing fleet trajectory) の causal leak によるもの**。
iter4 を「採用」とした判断は **撤回** 、K2 を削除した iter6 を再学習する必要がある。

3 診断の結論:

| 診断 | 結果 | 解釈 |
|------|------|------|
| A. K4 ablation | K4 全 zero-out で ships_F1 -0.0007 のみ | **K4 は無効** (削除候補) |
| B. permutation @ iter4 | K2 sum &#124;Δ&#124;=1.70 で異常巨大 | **K2 が ships head 0.93 の主因** |
| C. class 分布 + kappa | balanced_acc=0.94, kappa=0.94 | imbalance 由来ではない、真に高精度 |

K2 は **自軍 fleet (action_N の結果)** を per-source-planet に attribute する特徴量。`ships_per_src` ラベルは action_N の発射 ships で、`obs.fleets` の自軍 fleet は action 適用後の値 → **policy が逆算して ships_per_src を学んでしまう leak**。

case3 phase2 result.md の教訓 (`obs_N` は action 適用後) と完全に一致。

## Diagnosis A: K4 ablation 結果

各 K4 列を 1 つずつ zero-out した val metrics。

| Variant | Δ ships_F1 | Δ ships_acc |
|---------|-----------:|------------:|
| zero `aux_my_ships_h5` | +0.0000 | +0.0000 |
| zero `aux_my_ships_h15` | -0.0003 | +0.0000 |
| zero `aux_my_prod_log` | -0.0003 | -0.0003 |
| zero `aux_ships_ratio` | -0.0002 | -0.0001 |
| zero `K4_all (cols 8-11)` | **-0.0007** | +0.0000 |

**結論**: K4 は ships head の真の主因では**なかった**。iter4_result.md の解釈は誤り。

## Diagnosis B: Permutation importance @ iter4

functional group 別の sum |Δ| ranking (重要度順):

| Rank | Group | n_col | sum &#124;Δ&#124; | Δ ships_F1 | Δ target_F1 |
|---:|-------|---:|------:|---------:|---------:|
| 🥇 | **K2_outgoing_fleet_trajectory** | 4 | **1.70** | **-0.648** | -0.088 |
| 🥈 | G2_ships_production_owner | 6 | 1.21 | -0.466 | -0.071 |
| 🥉 | G8_multi_horizon | 4 | 0.79 | -0.425 | -0.032 |
| 4 | G4_ship_prediction_h30 | 6 | 0.74 | -0.415 | -0.016 |
| 5 | G3_incoming_eta | 2 | 0.16 | -0.000 | -0.056 |
| 6 | G1_position_geometry | 7 | 0.14 | +0.000 | -0.050 |
| 7 | G5_history_obs_n2_n3 | 3 | 0.13 | +0.002 | -0.004 |
| 8 | G9_pairwise_top_k_k3 | 12 | 0.11 | -0.006 | -0.038 |
| 9 | G11_sparse_mask | 5 | 0.06 | -0.005 | -0.016 |
| 10 | G10_defense_surplus | 4 | 0.05 | -0.009 | -0.002 |
| 11 | H2_global_aggregate_resources | 4 | 0.03 | +0.001 | -0.007 |
| 12 | K3_frontline_distance | 4 | 0.03 | -0.003 | -0.009 |
| 13 | G7_inbound_fleet_trajectory | 4 | 0.03 | -0.000 | -0.008 |
| 14 | H3_global_launch_history | 4 | 0.02 | -0.000 | -0.003 |
| 🗑️ | **K4_aux_multi_horizon_global** | 4 | **0.001** | +0.000 | -0.000 |

**核心**: K2 outgoing fleet trajectory が ships head に **-0.648** の影響、これは
groupings の中で**断トツ最大**。次の G2 ships_production_owner (-0.466) を抜いている。
K2 を shuffle すると ships F1 が 0.93 → 0.29 まで一気に落ちる = **K2 抜きでは ships
の予測が壊滅的**。

これは「K2 の真の効果」ではなく「K2 経由で ships_per_src を逆算する leak」が
permutation で完璧にキャンセルされた結果と解釈する。

## Diagnosis C: ships class 分布 + balanced metrics

```
fired sources total: 41,446
class counts:  bucket 0=1,390 (3.4%) / 1=7,062 (17.0%) / 2=3,863 (9.3%) / 3=29,131 (70.3%)
```

**class imbalance はある (bucket 3 が 70%)** が、**これだけでは 0.93 は出ない**:

| Metric | Majority predict (= always 3) | iter4 model |
|--------|------------------------------:|------------:|
| acc | 0.7029 | **0.9728** |
| macro F1 | 0.2064 | **0.9349** |
| Cohen's kappa | (n/a, kappa=0 by definition) | **0.9416** |
| Balanced accuracy | 0.25 (avg recall when always predicting 3) | **0.9375** |

per-class recall:
| class | recall |
|------|------:|
| 0 (25%) | 0.910 |
| 1 (50%) | 0.954 |
| 2 (75%) | 0.895 |
| 3 (100%) | 0.991 |

**全 class で recall ≥ 0.89** → これは imbalance 補正済みの真の高精度。imbalance による
偽の高 F1 ではない。**しかしこれが leak から来ているのが本問題**。

confusion matrix:
```
         pred_0   pred_1   pred_2   pred_3
true_0    1,266     101       8       15    (recall 0.91)
true_1       89   6,738     196       39    (recall 0.95)
true_2       15      73   3,455      320    (recall 0.89)
true_3      110      94      67   28,860    (recall 0.99)
```

## 真因解釈: K2 outgoing fleet trajectory の leak メカニズム

`obs.fleets` には **action 適用後** の fleet が含まれる (`step[N].observation.fleets` は
turn N の action 結果)。case3 phase2 で同事象を確認済み。

K2 featurizer は:
1. `raw_fleets` から `fowner == player` の fleet を集める
2. `from_pid` (出撃源 planet id) で per-source-planet にグループ化
3. その fleet の 5-turn 後 future position を per-source attribute

ここで **`from_pid` がそもそも action_N の選択結果**で、自軍 fleet の **ships 数も action_N の発射量**そのもの。
従って K2 の `outgoing_fleet_ships_log` (col 56) は `ships_per_src` ラベルと**完全に同じ情報**。

policy network はこれを 1 layer で学べる:
- `from_multihot[slot] = 1` ⇔ K2 の `inbound_dist[slot] != -1`
- `ships_per_src[slot]` ⇔ K2 の `outgoing_fleet_ships_log[slot]`

これが ships F1 0.93 の正体。target head の +0.005 改善も同じ leak から来ている可能性が高い (from_pid leak は target template 推定にも一部寄与)。

## 採用判定 (撤回)

iter4 採用 → **取消**。canonical weights は iter4 best.pt のままだが、ships head が
leak なので **Stage 2 self-play で実戦勝率には反映しない** 見込み。

### iter6 推奨方針

1. **K2 outgoing fleet trajectory を完全削除** (4 列)
2. **K4 も削除** (effect 0.001、無意義)
3. iter4 dim 61/12 → **iter6 dim 57/8** (planet -4, global -4)
4. K2 削除後の真の ships head 効果を再測定 (おそらく iter1 レベルの 0.62〜0.65 に戻る)
5. K3 frontline と G10 defense surplus は残す (target に効いている)
6. permutation importance を再度取り直して真の改善寄与を可視化

### iter6 で必要な追加診断

- iter1〜3 weights でも同じ permutation を取り、K2 削除後の baseline と比較
- target head improvement (+0.014 vs iter1) が leak 副作用 か K3 frontline 真効果 か切り分け

## RunPod onstart

iter5 では RunPod 起動なし、コスト 0。

## 関連ファイル

- val metrics (baseline iter4): val data baseline は変わらず ships_F1 0.9349
- A K4 ablation: `data/output/experiment/imitation_case7_iter5_k4_ablation.json`
- B permutation importance @ iter4: `data/output/experiment/imitation_case7_iter4_grouped_importance.json`
- C class 分布 + confusion: `data/output/experiment/imitation_case7_iter5_ships_class_distribution.json`
- 元 iter4 metrics.json: `runs/20260505-102543__.../metrics.json` (val_loss 3.3576 は leak 込みなので true value 不明)

## 教訓

1. **「劇的改善 = leak 疑い」は鉄則**: case3 phase2 で同じ問題を経験していたのに iter4 で見落とした。Δ ships_F1 +0.29 は通常の特徴量追加で出る幅を超えている時点で疑うべきだった
2. **`obs.fleets` の自軍 fleet は action_N の結果**: case3 phase2 result.md の教訓が普遍。**outgoing fleet を input に入れる場合は必ず obs_{N-1} を参照する**
3. **permutation importance は leak 検出の決定打**: K2 が target head ではなく ships head に -0.648 と巨大寄与 = action_N の情報が input に直接通っている証拠
4. **K4 のように一見効きそうな global aggregate も実は無意義**な場合あり: ablation 必須
5. **「真の効果」と「leak」は metric だけでは区別できない** が、permutation importance + 機能カテゴリ別の Δ で見ると leak は groups 1 つに突出した形で現れる
