# Hypotheses — imitation/case9 head_setting_optimization

> 作成日: 2026-05-06
> 最終更新: 2026-05-13
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
> **方針転換 (2026-05-09)**: 今後はデータ量 / モデルサイズの増加が見込まれるため、**精緻な予測**を優先する。class weight / focal / label smoothing / threshold / temperature 等の **曖昧化方針は採用しない**。代わりに (a) 行動空間の解像度を上げ (b) 量的予測は regression、(c) target は planet_id 直接、(d) per-slot 入力情報量を増やす方向で head を再設計する。imbalance はデータ量で吸収する前提。
>
> **運用方針 (H6 以降)**: 各仮説は **candidate×ships variant** と **template×ships variant** を**同条件・同 iter** で並行学習し、val_total / fire_acc / ローカル挙動を 1 表で比較する (iter2 の 3-variant 並行と同じスタイル)。両 variant 比較せずに片側だけ回した場合は inconclusive 固定。前提として candidate×ships の loss 異常 (iter2 val_total=246719) は **ラベル / 構造の正しさで** 修正する (label index 範囲 / mask / `ignore_index=-100` / CE input scale)。weight や smoothing で吸収するアプローチは採らない。
>
> **削除済み仮説 (曖昧化方針のため)**: 旧 H5 (no-op bias 調整: class weight / threshold) / 旧 H8 (candidate slot 0 class_weight 強化)。
> **統合済み仮説**: 旧 H7 (ships regression) → H12 / 旧 H11 (ships bucket 最適化) → H12 + H15。

- [ ] (P1) **H6: 候補空間の精緻化 (candidate K=8→16 / template +4 種)** — fire 側の行動解像度を上げる。**両 variant を同 iter で並行学習** だが**主軸は candidate** (K↑ で planet_id 直接予測に漸近)。template 側は副軸として比較指標用に残す。
  - **candidate variant (主軸)**: `CAND_K=8 → 16` (no-op + fire 15 slot)、`CAND_FEAT_DIM=14` 維持。`candidates.py` の slot 構築 + `candidate.py` head 出力次元を更新。計算コスト 2×、データ量増を前提に過学習リスクを許容
  - **template variant (副軸)**: `NUM_TEMPLATES=8 → 12` (NO_OP 含む)、追加 4 種:
    - `DEFEND_HOMEWORLD` (自 home へ集結、防御)
    - `ATTACK_PRODUCING_ENEMY` (production>0 enemy のうち ships 最少)
    - `INTERCEPT_INCOMING_FLEET` (受けている艦隊経路上の友軍へ援軍。fleet obs 参照要)
    - `COMET_RIDE` (comet 出現ターン 50/150/250/350/450 で comet 狙い)

    `templates.py` の `resolve_template` / `template_features` / `TEMPLATE_CTX_DIM` を更新
  - 比較指標: candidate は val_cand_fire_acc / val_cand_acc、template は val_template_fire_acc / val_template_noop_acc
  - 期待効果: candidate K↑ で target 解像度 2×、planet_id 直接予測に漸近。曖昧化なしで fire の正解 slot を直接学習

- [ ] (P1) **H12: ships head の regression 化** — 4-bucket categorical を **log1p ships の 1-dim regression (MSE / Huber)** に置換。量を連続値で直接予測する精緻化。candidate×ships_reg と template×ships_reg を**両 variant 並行**。case8 ship_count_regression 系の知見を head に適用。
  - 比較指標: val_ships_mae (log1p 空間) / val_ships_acc (bucket 換算)
  - 期待効果: 4-bucket 量子化を排し、発射量の細かい差を学習可能に

- [ ] (P1) **H13: candidate slot feature の精緻化 (CAND_FEAT_DIM 14 → 22)** — per-slot の入力情報量を拡張。timeline 6 列 (在庫推移 / 到着 ETA) + fleet pressure 2 列 (incoming / outgoing) を slot 単位で注入。`candidates.py` の `candidate_features` を拡張、`candidate.py` head 入力次元を更新。
  - candidate variant: 主対象 (slot feature 直接拡張)
  - template variant: 並行比較として `template_features` にも timeline / fleet pressure を per-template 注入
  - 期待効果: 各 slot / template の判断材料が増え、精緻な選択が可能に

- [~] (P2) **H14: planet_id 直接予測 head (3-head の target_head 復活)** — 3-head の target_head を template (8 class) ではなく **`planet_id` (37 class) 直接予測**に戻す。case3 phase2 由来の最も精緻な target 選択肢。データ量増を前提に収束を狙う。
  - candidate variant: candidate K=37 拡張 (full planet 集合) + ships_reg
  - template variant: 3-head の target_head を planet_id 37 class に戻す
  - 期待効果: target 解像度を最大化 (8 → 37)、抽象化による情報損失を排除
  - **partial adoption (iter3, 2026-05-13)**: per_planet variant 単独で実装・学習 (template variant は未走)。val_target_fire_acc=0.351 / val_target_acc=0.920。両 variant 並行ルール違反のため iter は **inconclusive** 扱い、`H14 template variant` を deepen 候補として残す。

- [ ] (P2) **H15: ships head の高解像度 bucket** — 4-bucket → **16 / 32-bucket** categorical (log spacing)。H12 (regression) と並走する精緻化選択肢。両 variant 並行。
  - 比較指標: val_ships_acc / 各 bucket の precision
  - 期待効果: regression が学習しづらい場合の代替、量の解像度を 4× / 8× に

- [ ] (P2) **H9: target_head に timeline 列を直接 concat** — timeline 6 列を target_head 直前で再注入。補助情報の精緻化。両 variant 並行 (candidate は slot feature 注入、template は template feature 注入で H13 と一部重複、H13 後に差分検証)。

- [ ] (P2) **H10: triple head (3-head + candidate + value head)** — value head を追加し state value (勝率 / 期待リターン) を補助タスク化。多タスク学習で表現を精緻化。両 variant とも value head を共有 backbone から分岐。工数大だが精緻化方針と整合のため P2 に昇格。

## Iteration log

(各 iter 完了時に experiment-analysis / experiment が追記)

| iter | 開始 | 仮説# | plan path | run_id | 主要メトリクス | 採否 | result path | analysis path |
|---|---|---|---|---|---|---|---|---|
| 1 | 2026-05-07 | H1 three_head | hypotheses.md | `20260507-023323__...__98ea59d__seed0` | val_target_acc=0.928 / 10ep=0/10 | inconclusive | iter1_result.md | - |
| 1 | 2026-05-07 | H2 candidate | hypotheses.md | `20260506-133924__...__a6a7bee__seed0` | val_cand_fire_acc=0.211 / 10ep=0/10 | inconclusive | iter1_result.md | - |
| 1 | 2026-05-07 | H3 candidate_ships | hypotheses.md | `20260507-022131__...__ea87185__seed0` | val_cand_fire_acc=0.211 / 10ep=0/10 | inconclusive | iter1_result.md | - |
| 2 | 2026-05-08 | H4 head_family_behavior_comparison | iter2_plan.md | `20260508-115939...` / `20260508-120122...` / `20260508-120124...` | template val_total=0.8758 fire=0.2888 / 3head val_total=1.0772 / cand×ships loss異常大; local 5ep all 0/5 | inconclusive | iter2_result.md | iter2_analysis.md |
| 3 | 2026-05-12 | H14 per_planet (planet_id 直接予測 + ships regression / partial: candidate variant only, template variant 未実施) + データフィルタ (winners + top_team_rank 80, max_episodes 8000) | (plan 未作成、直接実装) | `20260512-080505__feature-imitation-data-volume-sweep__593a7c4__seed0` | val_target_acc=0.9205 / val_target_fire_acc=0.3512 / val_target_noop_acc=0.991 / val_ship_mae=0.371 / local 1ep vs random=WIN, 1ep vs baseline_v1=LOSS | inconclusive (両 variant 未並行、n=2 で結論不可、n<300 ruleにより) | (result.md 未作成) | iter3_analysis.md (本セッション) |
