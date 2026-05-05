# imitation/case7 iter6 — K2 leak 除去 + K4 整理 (slim 化)

> 作成日: 2026-05-06
> 関連:
> - `./iter5_result.md` — K2 outgoing fleet trajectory が causal leak と確定 (sum |Δ|=1.70, Δships_F1=-0.648)
> - `./iter4_result.md` — iter4 で ships F1 0.93 という偽改善
> - `./iter3_result.md` — permutation importance の方法論
> - `bot/pipeline/imitation/case7/policy/featurizer.py` — iter4 で 61/12 schema、K2 (planet 53-56) と K4 (global 8-11) を保持
>
> スコープ: K2 (4 列) + K4 (4 列) 削除して dim 61/12 → **57/8** に slim 化、leak 除去後の真の Stage 1 metrics を再測定。

## 仮説 (Hypothesis)

iter4 の ships head 大幅改善 (F1 0.6353 → 0.9349) は K2 outgoing fleet trajectory の
**causal leak** (`obs.fleets` の自軍 fleet = action_N の結果) が原因と iter5 で確定。
K2 を削除すれば ships F1 は **iter1 base レベル (0.62〜0.65)** に戻り、target head
の改善 (top-1 +0.014) も leak 副作用かどうか判別できる。同時に K4 (importance 0.001
で完全 dead) も削除して dim を slim 化する。

メカニズム:

- K2 削除: `obs.fleets` の自軍 fleet を 一切 input に入れない (inbound enemy fleet G7 はそのまま、こちらは leak free)
- K4 削除: aux_my_ships_h5/h15 + my_prod_log + my_ships_ratio は H2_global_aggregate_resources (my_total_ships log) と相関で **意義なし**

採否ゲート: 「leak の有無」を断定するための再測定なので **metric 達成ゲートは設けない**。代わりに **iter1 base からの真の Δ** を 4 head 全てで報告し、permutation importance で残り group の貢献度を再確認 (iter5 で K2 の異常巨大 Δ が消えるかを再確認)。

期待される結果:

| Head | iter1 | iter4 (leak) | **iter6 期待値** |
|------|------:|-------------:|----------------:|
| from F1 | 0.6651 | 0.6838 | 0.65〜0.69 (leak 副作用なら下がる) |
| target macro F1 | 0.3210 | 0.3296 | 0.32〜0.33 (leak 副作用なら下がる) |
| **ships macro F1** | 0.6425 | **0.9349** | **0.62〜0.66 (大幅後退想定)** |
| ships balanced acc | (n/a) | 0.94 | 0.50〜0.65 想定 |

## 既存コードの現状 (from Step 1)

- featurizer.py iter4 layout (PLANET=61, GLOBAL=12):
  - planet 53-56: **K2 outgoing_fleet_trajectory** (削除対象)
  - planet 57-60: K3 frontline_distance (保持、iter5 importance 0.03 で控えめだが target に微効)
  - global 8-11: **K4 aux_my_ships_h5/h15/prod_log/ratio** (削除対象、importance 0.001)
- iter5 の permutation importance ranking (iter4 weights 上):
  - 🥇 K2 (sum |Δ|=1.70, Δships_F1=-0.648) ← 削除
  - 🥈 G2 ships_production_owner (1.21)
  - 🥉 G8 multi_horizon (0.79)
  - 🗑️ K4 (0.001) ← 削除
- 訓練済 weights は dim 61/12 用なので **iter6 で再訓練必須**
- preprocess も dim 変更 (61→57) で再生成 (trap #9 fix で onstart が自動再実行)

## スコープ (Scope)

### 変更ファイル

| Path | 変更内容 |
|------|----------|
| `bot/pipeline/imitation/case7/policy/featurizer.py` | PLANET_FEAT_DIM 61→**57**、GLOBAL_FEAT_DIM 12→**8**。K2 (planet 53-56) と K4 (global 8-11) の出力削除、関連 pre-compute 削除 |
| `bot/pipeline/imitation/case7/configs/il_case7.yaml` | `planet_in_dim: 61→57`、`global_in_dim: 12→8` |
| `bot/tests/pipeline/imitation/case7/test_featurizer_dim.py` | `PLANET_FEAT_DIM == 57`、`GLOBAL_FEAT_DIM == 8` に更新 |
| `bot/tests/pipeline/imitation/case7/test_featurizer_iter4.py` | K2/K4 の test を **削除**、K3 frontline (col 53-56 に shift) の test は保持して col 値だけ更新 |

### 変更なし

- AGENT_REGISTRY (`il_v7` のまま、weights 上書き)
- model.py / decoder.py / agent.py / main.py
- iter1〜3 で追加した特徴量 (G1〜G11、H2/H3) はすべて保持
- K3 frontline_distance (iter4 の planet 57-60 → iter6 の **53-56** に slot shift)

### 列 layout (iter6 final)

| idx | group | 内容 |
|-----|-------|------|
| 0-23 | iter1 base + history + (iter1 ship-event は iter4 で削除済) | |
| 22-25 | G7 inbound_fleet_trajectory (iter4 で 22-25 に移動) | |
| 26-29 | G8 multi-horizon (iter4 で 26-29) | |
| 30-31 | iter2 home/centroid distance (per-planet) | |
| 32-43 | G9 pairwise_top_k K=3 (12 列) | |
| 44-47 | G10 defense_surplus | |
| 48-52 | G11 sparse_mask_flags | |
| ~~53-56~~ | ~~K2 outgoing_fleet_trajectory~~ | **削除** |
| **53-56** | **K3 frontline_distance** (元 57-60 から shift) | |

global:
| idx | group |
|-----|-------|
| 0-3 | H2 global_aggregate_resources |
| 4-7 | H3 global_launch_history |
| ~~8-11~~ | ~~K4 aux_multi_horizon_global~~ | **削除** |

## 実装ステップ (Implementation outline)

1. **featurizer.py 編集**:
   - PLANET_FEAT_DIM 61→57、GLOBAL_FEAT_DIM 12→8 定数更新
   - K2 pre-compute (`outgoing_fleet_dx/dy/dist/ships_log`) 削除
   - K4 計算 (`aux_my_ships_h5`, `aux_my_ships_h15`, `aux_my_prod_log`, `aux_ships_ratio`) 削除
   - planet feats list から K2 4 列を削除、K3 4 列の slot は変えず (元 57-60 → 53-56)
   - global feats list から K4 4 列を削除
2. **`configs/il_case7.yaml` 更新**: `planet_in_dim: 57`, `global_in_dim: 8`
3. **テスト更新**:
   - `test_featurizer_dim.py`: dim assertion 更新、K2/K4 言及テストを削除
   - `test_featurizer_iter4.py`:
     - K2 (test_k2_*) 4 件削除
     - K4 (test_k4_*) 4 件削除
     - K3 (test_k3_*) は col 値を 53-56 に shift、4 件保持
4. `dev/test-bot` で format/lint/mypy/pytest green 化
5. commit & push
6. (execution side) RunPod Step B: trap #9 fix で preprocess 再実行 (deps hash 変更)、~30 分・~$0.35
7. (execution side) Stage 1 evaluation + permutation importance 再計算
8. (execution side) iter6_result.md に「真の Stage 1 metrics + leak 除去前後比較」を記録

## 検証方法 (Validation method)

### ローカル

```bash
dev/test-bot
uv run --directory bot pytest tests/pipeline/imitation/case7 -x

# featurizer dim sanity
uv --project bot run python -c "from pipeline.imitation.case7.policy.featurizer import PLANET_FEAT_DIM, GLOBAL_FEAT_DIM; print(PLANET_FEAT_DIM, GLOBAL_FEAT_DIM)"
# → 57 8 を期待
```

### リモート (execution skill 側)

```bash
git push origin feature/feature-engineering
dev/runpod train <iter6-sha> --case case7 \
  --gpu-name "NVIDIA GeForce RTX 4090" --gpu-name "NVIDIA GeForce RTX 3090" --gpu-name "NVIDIA RTX A6000"
# trap #9 fix で deps invalidate → preprocess 再実行
# 想定: preprocess ~7 分 + train ~7 分 + dvc/git push ~5 分 = ~30 分、~$0.35
```

### 評価

iter1〜4 と同じ `diagnose_weights.py` で full head metrics、加えて permutation importance を iter5 と同じ functional group で再計算。

報告フォーマット:

| head | iter1 | iter4 (leak) | **iter6 (clean)** | iter6 - iter1 |
|------|------:|-------------:|------------------:|----:|
| from F1, PR-AUC, ROC-AUC | ... | ... | ... | ... |
| target top-1, macro F1, PR-AUC | ... | ... | ... | ... |
| ships acc, macro F1, kappa, balanced_acc | ... | ... | ... | ... |

採否しきい値: **leak が除去されたかの確認** が主目的。
- ships F1 が iter1 レベル (0.62〜0.65) に戻れば leak 完全除去確認 → iter6 を新 baseline として採用
- target / from が iter1 比 +0.005〜+0.01 残れば K3/G10 など他 iter3-4 改良が真効果と確認
- もし ships F1 が依然 0.85+ なら **K2 以外の leak 経路** がある (G7 inbound などを再疑う)

## リスク / 想定失敗モード

1. **K2 削除でも ships F1 が下がりきらない**: G7 inbound_fleet_trajectory にも leak 経路がある可能性。iter5 では sum |Δ|=0.03 で control 下だが、K2 を抜いた相対重要度で再評価必須
2. **K3 frontline_distance も leak**: K3 は敵軍 nearest との dist だけなので action 反映なし、leak リスク低だが念のため iter6 importance で確認
3. **target / from head の改善が完全消失**: iter4 の target +0.014 / from F1 +0.019 が leak 副作用だった場合。その場合 iter6 は **iter1 と parity** という結果になる
4. **dim 縮小で訓練が早すぎて underfit**: planet 57 / global 8 でも model capacity は十分のはず。問題出れば iter7 で hidden dim を上げる
5. **trap #9 (preprocess force rerun)** は iter4 で動作確認済、再発リスクは低

## Stop conditions

- [ ] case7 featurizer/configs/tests が iter6 仕様 (PLANET=57, GLOBAL=8) に整理、unit test pass
- [ ] `dev/test-bot` green
- [ ] commit & push
- [ ] (execution side) RunPod Step B が `99_done` で終了、weights pull
- [ ] Stage 1 metrics + permutation importance 再計算
- [ ] iter6_result.md に「leak 除去後の真の Stage 1 metrics」+ iter1/iter4 比較が書かれる

## 参考

- iter5_result.md の核心ポイント (`obs_N` は action 適用後)
- case3 phase2 result.md の history leak 教訓
- iter5 permutation importance JSON: `data/output/experiment/imitation_case7_iter4_grouped_importance.json`
