# Hypotheses — imitation/case9 head_setting_optimization

> 作成日: 2026-05-06
> 最終更新: 2026-05-08
> 状態: in_progress
> 最大 iteration: 上限なし (ユーザー停止指示まで)
> 主要メトリクス: 学習中 val_loss / val_target_acc / val_ships_acc (case7/case8 同形式) + ローカル self-play 10 戦の挙動評価 (vs baseline_v1)
> 既定 episode 数: 30 (挙動確認のみ、n<300 で結論は出さない)

## スコープと固定軸

case9 は **head 設計の最適化** にスコープを絞る。特徴量 / backbone は以下で **固定**:

| 軸 | 固定値 | 出典 |
|----|--------|------|
| Featurizer | PLANET=63 (case8 iter6 leak free 57 + case5 timeline 6, 重複 11 列は base 共有) / GLOBAL=8 (case8 iter6 同一) | `docs/experiment/imitation/20260504_case8_feature_engineering/iter6_result.md` (clean baseline) + `docs/experiment/imitation/20260503_case5_ship_prediction/plan.md` (timeline 6 列) |
| Backbone | Set Transformer (ISAB ×3, m=16, hidden=128, PMA) | `docs/experiment/imitation/20260505_case7_model_structure_roadmap/iter3_result.md` (val_loss 3.6211 ⭐ best) |

可変軸は **head 構造のみ**。

## 初期 3 パターン比較 (priority P1)

ユーザー指定により、まず以下 3 head 構成を並べて比較する:

1. **3-head 単独** (from + template + ships, case3 phase2 由来)
2. **candidate head + ships head** (candidate で source/slot 選択 + ships で発射数 4-bucket categorical)
3. **template head + ships head** (template incl no-op で sourceごとの抽象行動を選択 + ships で発射数 4-bucket categorical)

## 実施しない検証 / 評価 (skip list)

### 評価
- Kaggle publicScore は引用しない (project rule, memory `project_om_finding` / `project_case5_validation`)
- skill rating は使わない (project rule)
- **300 対戦による評価はしない** — 学習中 val 指標 + ローカル 10 戦挙動 + 対戦ログ分析で採否

### 分析
- n<300 結果で結論を出さない (memory `project_imitation_case1_phase3`) — 10 / 30 ep 結果は **inconclusive 固定**、val_loss / val_target_acc / val_ships_acc + 対戦ログの定性評価で確定的判断

### 実行
- (なし) — smoke / dev/test-bot / RunPod / auto-recover はすべて実施

### 例外条件
- val 指標で大幅改善 (val_loss が case8 iter6 比 -0.05 以上) かつローカル 10 戦で挙動異常なしの仮説に限り、ユーザー判断で 300 ep フォローアップを検討可能 (本 hypotheses.md のスコープ外)

## 仮説リスト (priority 順)

- [x] (P1) **H1: 3-head 単独** — case3 phase2 の 3-head (from + template + ships) のみ。SetT 128 + P=63/G=8 上での 3-head ベースライン。比較 3 パターンの 1 本目 — **inconclusive** (val_target_acc=0.928 / 10 ep vs baseline_v1 = 0/10, n<300 で結論不可)
- [x] (P1) **H2: candidate head 単独** — case4/case8 の per-source × CAND_K=8 categorical head のみ (CAND_FEAT_DIM=14)。ships rule = `max(target.ships+1, 20)` (notebook 流)。比較 3 パターンの 2 本目 — **inconclusive** (val_cand_fire_acc=0.211 / 10 ep = 0/10)
- [x] (P1) **H3: candidate head + ships head** — candidate head で source/slot 選択 + 別 ships_head で発射数を 4-bucket categorical 学習 (ships rule を learned head に置換)。H1/H2 のハイブリッド。比較 3 パターンの 3 本目 — **inconclusive** (val_cand_fire_acc=0.211 / 10 ep = 0/10)
- [x] (P1 follow-up) **H4: 3 head family 挙動比較 (3-head / candidate×ships / template×ships)** — no-op/fire の表現方法が異なる 3 パターンを同条件で比較し、学習時 train/val 精度推移とローカル対戦挙動の差を確認する — **inconclusive** (template×ships: best val_total=0.8758 / fire_acc=0.2888, 3-head: val_total=1.0772, candidate×ships: loss異常大, local 5ep は全て0/5)
- [ ] (P2, depends on H4) **H5: template/candidate の no-op bias 調整** — H4 で fire recall が低い head に対して no-op class weight / threshold を調整する
- [ ] (P2) **H6: candidate head の K 拡張 (K=8 → K=12)** — candidate slot 数を増やし、target 探索空間を広げる。CAND_FEAT_DIM=14 は維持
- [ ] (P3) **H7: 3-head の ships を regression 化** — ships_head を 4 buckets categorical → 1-dim regression (log1p ships) に置換。case8 ship_count_regression 系の知見を head に適用
- [ ] (P3) **H8: candidate head の slot 0 = no-op 重み付け強化** — class_weight_on_slots を no-op に bias し、過剰発射を抑制 (case4 iter2 の fire/no-op 振動への対策)
- [ ] (P3) **H9: 3-head の target_head に timeline 列を直接 concat** — timeline 6 列を target_head 直前で再注入し、defensive hold 判断を強化
- [ ] (deferred) **H10: triple head (3-head + candidate + value head)** — value head を追加し state value 推定を補助タスク化。工数大

## Iteration log

(各 iter 完了時に experiment-analysis / experiment が追記)

| iter | 開始 | 仮説# | plan path | run_id | 主要メトリクス | 採否 | result path | analysis path |
|---|---|---|---|---|---|---|---|---|
| 1 | 2026-05-07 | H1 three_head | hypotheses.md | `20260507-023323__...__98ea59d__seed0` | val_target_acc=0.928 / 10ep=0/10 | inconclusive | iter1_result.md | - |
| 1 | 2026-05-07 | H2 candidate | hypotheses.md | `20260506-133924__...__a6a7bee__seed0` | val_cand_fire_acc=0.211 / 10ep=0/10 | inconclusive | iter1_result.md | - |
| 1 | 2026-05-07 | H3 candidate_ships | hypotheses.md | `20260507-022131__...__ea87185__seed0` | val_cand_fire_acc=0.211 / 10ep=0/10 | inconclusive | iter1_result.md | - |
| 2 | 2026-05-08 | H4 head_family_behavior_comparison | iter2_plan.md | `20260508-115939...` / `20260508-120122...` / `20260508-120124...` | template val_total=0.8758 fire=0.2888 / 3head val_total=1.0772 / cand×ships loss異常大; local 5ep all 0/5 | inconclusive | iter2_result.md | iter2_analysis.md |
