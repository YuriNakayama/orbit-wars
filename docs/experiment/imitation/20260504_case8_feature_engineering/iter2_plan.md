# imitation/case8 iter2 — 特徴量追加 (template prefix / fleet trajectory / multi-horizon / production-comet)

> 作成日: 2026-05-05
> 関連:
> - `./iter1_plan.md` — case8 iter1 の plan (predicted dist + history + ship event 11 列追加)
> - `./iter1_result.md` — case8 iter1 の結果 (Stage 1 from PR-AUC +0.136 ✅、Stage 2 0/50 ⚠、採否保留)
> - `bot/pipeline/imitation/case8/policy/featurizer.py` — iter1 で 24 列 / 10 global 列に拡張済み
> - `bot/pipeline/imitation/case8/policy/templates.py` — 8 template (NEAREST_NEUTRAL_LOW / NEAREST_ENEMY / HIGH_PROD_NEUTRAL / HIGH_PROD_ENEMY / REINFORCE_FRONTLINE / REINFORCE_WEAKEST / WEAKEST_ENEMY / NO_OP)
> - `bot/pipeline/imitation/case8/policy/timeline.py` — `simulate_planet_timeline`, horizon 30
>
> スコープ: case8 iter1 の featurizer を **in-place 拡張** し、3 カテゴリ ~14 列を追加。focus は **学習精度 (Stage 1 val metrics) 向上**、Stage 2 self-play は今回のスコープ外 (300 ep フォローアップは別途)。
>
> **更新 (2026-05-05 implementation start)**: 元の 4 カテゴリ案のうち「template prefix score」は既存の `template_context_features` (`templates.py`) で `TEMPLATE_CTX_DIM=40` 列が `BatchFeatures.template_ctx` 経由で model に渡っていることが判明。**重複なので削除**、残り 3 カテゴリで実装する。

## 仮説 (Hypothesis)

iter1 では target head が val_acc 0.42 で plateau に達した。これは current featurizer
が「per-planet の状態 + ship-prediction + history + per-fleet 集計」で停滞しており、
**(a) decoder template scoring の中間結果**、**(b) fleet 単位の future trajectory**、
**(c) 多 horizon ship-prediction**、**(d) home / production-weighted spatial 情報** の 4 軸を
追加すれば、target head は val_acc 0.45+ (+0.03)、ships head は macro F1 +0.025 で
iter1 の唯一の後退を逆転、from head も PR-AUC 0.80+ に押し上げできる、という推測。

メカニズム:

- **Template prefix scores (8 列 / source planet)**: decoder が runtime で計算する
  template 評価値 (例: `score(NEAREST_NEUTRAL_LOW | src=p)`) を featurizer 段階で
  pre-compute して input に含める。これは BC が「state→action mapping」を学ぶ際に、
  **expert が暗黙にやっている template scoring を直接観測値として与える**戦略 (BC literature の
  "Knowledge-Informed Models" / `Sample-Efficient BC w/ General Domain Knowledge` の系譜)。
- **Per-fleet trajectory prediction**: 現状 `incoming` / `nearest_eta` は per-planet 集計のみで、
  fleet 単位の **「5/10 turn 後のどこに居るか」** を policy は推測できない。fleet level の
  future position 4 列 + ship_log を per-planet に attribution して **per-planet × fleet-level 4 列** を
  追加することで、harass timing (敵 fleet が 5 turn 後に着弾するか) と reinforce 必要量の判断を
  policy 側が直接する。
- **Multi-horizon ship-prediction**: iter1 は horizon=30 のみ。短期 (5 turn) / 中期 (15 turn) /
  長期 (30 turn、既存) の 3 horizon を持たせて **「すぐ陥落」と「中期厳しい」と「長期問題なし」を区別**。
  iter1 の `loss_3turn` は短期だが、`min_owned` / `keep_needed` などは horizon=30 一択。
- **Production-weighted centroid / home distance / next comet ETA**: iter1 の centroid は
  単純平均。production 重みつき centroid と **home planet (initial_planets) との距離**、
  次 comet 出現 turn (50/150/250/350/450) までの距離を入れて、空間判断の解像度を上げる。

成功指標 (Stage 1 val metrics — Stage 2 は今 plan のスコープ外):

- **case8 iter1 比 +0.01 以上**を全 head で達成
  - target macro F1: 0.3210 → 0.3310+
  - from PR-AUC: 0.7676 → 0.7776+
  - ships macro F1: 0.6425 → 0.6525+
- 採否ゲートは memory `project_imitation_case1_phase3` に従い、**全 head が +0.01 ゲートを満たす**ケースのみ採用、
  1 head でも -0.005 を超える後退があれば破棄
- Kaggle publicScore は使わない (`.claude/rules/bot/pipeline.md` 準拠)

評価メトリクス:
- **case8/evaluation/diagnose_weights.py** で per-head val metrics (case3 phase2 と同体系)
- **ablation 必須**: 採用判定後に 4 カテゴリを **個別に zero-out** して val 評価、
  どのカテゴリが最も寄与するか / 害になるか / 無効かを切り分け

## 既存コードの現状 (from Step 1)

- **featurizer.py (24 列 / planet, 10 / global)**:
  - planet 0-10: case1 base (位置 / 半径 / ships / production / owner flags / comet flag / incoming ratio / nearest_eta)
  - planet 11-16: timeline 6 列 (loss_3turn, ttf_norm, min_owned, surplus, fall_predicted, keep_needed)
  - planet 17-18: future_dist_to_my/enemy_centroid (orbit 5-turn 先位置)
  - planet 19-21: delta_ships_t1, delta_ships_t2, owner_changed_t1 (`obs_{N-2}` / `obs_{N-3}` 参照)
  - planet 22-23: enemy_targeted_count_last4, enemy_targeted_ships_last4
  - global 0-5: step / ang_vel / 各 owner ships / production diff
  - global 6-9: enemy/ally launch_count_last4 + launch_ships_last4
- **timeline.py**: `simulate_planet_timeline(planet, arrivals, player, horizon=30)` のみ。short_window=3 で `loss_3turn` を出すが multi-horizon は未対応。
- **templates.py**: 8 template、`template_context_features(src_row, planet_rows, player, BOARD_SIZE)` は **per-source の context vector** を返すが TEMPLATE_CTX_DIM (= templates.py の定数) とのみ連動。template 評価スコア自体は decoder 経由で computed、featurizer には流れていない。
- **HistoryState**: planet snapshots maxlen=4、fleet snapshots maxlen=5。LAUNCH_HISTORY_WINDOW=4。
- **iter1 のテストは 16 件**: dim sanity (24/10) / history / launch / predicted distance / preprocess speed。iter2 では追加列のテストを各カテゴリで 1-2 本ずつ。

iter1 の所見 (`iter1_result.md` より要点):
- target head は epoch 5-9 で val_acc 0.4165 plateau、その後微減
- ships head macro F1 -0.025 (唯一の後退)、bucket 0/1 recall が下がった可能性
- 50 ep / 30 ep self-play で 0 勝 → BC のタクティカル決定力課題、特徴量追加で緩和したい

## スコープ (Scope)

### 変更ファイル

| Path | 変更内容 |
|------|----------|
| `bot/pipeline/imitation/case8/policy/featurizer.py` | PLANET_FEAT_DIM 24 → **34** (+10)、GLOBAL_FEAT_DIM 10 → **14** (+4)。下記 catalogue の列を追加 |
| `bot/pipeline/imitation/case8/policy/timeline.py` | `summarize_timeline_multi(timeline, horizons=[5, 15, 30])` を追加して 3 horizon の `loss_window` / `min_owned_window` を 1 回の simulate から取り出す |
| `bot/pipeline/imitation/case8/configs/il_case8.yaml` | `model.planet_in_dim: 24 → 34`、`model.global_in_dim: 10 → 14`。他は iter1 と同じ (epochs=15, lr=1e-3, batch=256) |
| `bot/pipeline/imitation/case8/training/preprocess.py` / `dataset.py` / `train.py` | dim 変更のみ (featurizer の constants 経由で自動的に追従、明示変更は不要のはず) |
| `bot/tests/pipeline/imitation/case8/` | 新規テスト追加: `test_featurizer_template_prefix.py` / `test_featurizer_fleet_trajectory.py` / `test_featurizer_multihorizon.py` / `test_featurizer_production_centroid.py` |

### 変更なし

- AGENT_REGISTRY (`il_v7` のまま、weights.pt 上書き)
- dvc.yaml の case8 stage 構造 (deps の hash が変わるので重 rebuild になるのは想定通り)
- model.py (DeepSetsPolicy / GraphUNetPolicy。input dim は `cfg.planet_in_dim` / `cfg.global_in_dim` 経由で受ける)
- decoder.py (template scoring は別経路で動く)
- main.py (Kaggle entry)

### Feature catalogue (追加予定)

#### planet 列 (24 → 34, +10)

| idx | 名前 | 定義 | カテゴリ |
|----|------|------|----------|
| 24 | `inbound_fleet_future_dx` | この planet を target とする敵 fleet (LAUNCH_HISTORY_WINDOW 内、ETA 最早の 1 件) の **5 turn 後 future position** からの relative dx / BOARD_SIZE | fleet trajectory |
| 25 | `inbound_fleet_future_dy` | 同 dy / BOARD_SIZE | fleet trajectory |
| 26 | `inbound_fleet_future_dist` | 同 distance / BOARD_SIZE。fleet なしなら -1 (sentinel) | fleet trajectory |
| 27 | `inbound_fleet_ships_log` | 該当 fleet の ships を log1p。fleet なしなら 0 | fleet trajectory |
| 28 | `loss_5turn_log` | timeline horizon=5 で `loss_in_window` (現在 ships - 5turn 後 ships) を log1p | multi-horizon |
| 29 | `loss_15turn_log` | timeline horizon=15 の loss_window log1p | multi-horizon |
| 30 | `min_owned_5turn_log` | timeline horizon=5 で「自所有期間中の最小 ships」を log1p | multi-horizon |
| 31 | `min_owned_15turn_log` | timeline horizon=15 の min_owned log1p | multi-horizon |
| 32 | `dist_to_home_planet_log` | home planet (initial_planets[player]) からの距離 log1p / BOARD_SIZE | production / centroid |
| 33 | `dist_to_prod_centroid_log` | 自軍 production-weighted centroid からの距離 log1p / BOARD_SIZE。自軍が無いとき sun (50,50) から | production / centroid |

#### global 列 (10 → 14, +4)

| idx | 名前 | 定義 |
|----|------|------|
| 10 | `next_comet_eta_norm` | 次 comet 出現 turn (50, 150, 250, 350, 450 のうち > current step の最小) との差 / 100。最後 (>450) は 1.0 |
| 11 | `comet_active_flag` | 現在 comet がアクティブ (`comet_planet_ids` 非空) なら 1.0 |
| 12 | `home_planet_owner_flag` | 自分の home planet (initial_planets) が現在自軍所有か (1.0)、奪われたか (0.0)、initial_planets 不明なら 0.5 |
| 13 | `prod_centroid_dist_to_enemy_norm` | 自軍 production-weighted centroid と敵軍 production-weighted centroid の距離 / BOARD_SIZE。「戦線距離」の 1 値 summary |

#### 設計原則

- **Template prefix の causal safety**: template scoring は **現在の状態のみ**で計算するので causal leak リスクなし (history 不要)。ただし `score_template()` が `simulate_planet_timeline` を 8 回呼ぶと per-frame で重くなるので、 timeline は 1 回だけ計算して 8 template に共有する形で実装。
- **Fleet trajectory feature の per-planet 帰属**: `_fleet_action_target` で reverse-resolve した target slot に future_position を attribute。複数 fleet が同 planet を狙う場合は ETA 最早の 1 件 (ECA = effective closest arrival) を採用。
- **Multi-horizon の重複回避**: `timeline.py:simulate_planet_timeline` を 1 回だけ呼んで `ships_at` から horizon=5/15/30 全部の `loss_window`, `min_owned_window` を取り出す。3 回 simulate を呼ばない。
- **Production-weighted centroid の発火条件**: production_sum が 0 の側 (全 planet が neutral or home 全滅) では centroid を sun (50, 50) に fallback。

## 実装ステップ (Implementation outline)

1. **timeline.py 拡張**: `summarize_timeline_multi(timeline, horizons=[5, 15, 30])` を追加。`ships_at[h]` から `loss_window`, `min_owned_window`, `keep_needed_window` を h ごとに返す。featurizer は horizon=30 の simulate を 1 回だけ呼ぶ。
2. **templates.py 拡張**: `score_template_prefix(src_row, planet_rows, player, ang_vel, comet_ids, timeline_for_src) -> np.ndarray (8,)` を追加。各 template について resolved target を選び、scoring (target ships / src ships ratio + production gain など) を返す。NO_OP は 0.5 fix。
3. **featurizer.py 拡張**: PLANET_FEAT_DIM 42, GLOBAL_FEAT_DIM 14 に変更。新規 18 列 (planet) + 4 列 (global) を catalogue 通りに埋める。`fleet_trajectory_to_planets` ヘルパで per-planet attribution を vectorize。
4. **configs/il_case8.yaml 更新**: `planet_in_dim: 42`, `global_in_dim: 14`。
5. **テスト追加** (4 ファイル × 2-3 テスト):
   - `test_featurizer_template_prefix.py`: my_planet では非ゼロ、enemy_planet では 0 fill、NO_OP score の値域。
   - `test_featurizer_fleet_trajectory.py`: ang_vel=0 で fleet_future_dist == 現在距離、ships_log の log1p 単調性。
   - `test_featurizer_multihorizon.py`: horizon=5 の loss は horizon=30 の loss 以下、`min_owned_15` ≥ `min_owned_30`。
   - `test_featurizer_production_centroid.py`: 自軍が 1 planet のみのとき home distance = 0、production_centroid が production の重みで偏る。
   - **causal leak 回帰テスト**: template prefix と fleet trajectory のいずれも `obs_{N-1}` を参照しないことを assert (case3 phase2 trap 回帰防止)。
6. **ローカル smoke**: `dev/test-bot` (format/lint/mypy/pytest) green 化。`uv --project bot run python -c "from pipeline.imitation.case8.policy.agent import agent; print(agent)"` で import sanity。
7. **commit & push** (この plan は execution skill のスコープ外、`/experiment-execution` で実施)。
8. (execution side) **RunPod Step B 一発**: `dev/runpod train <sha> --case case8 --gpu-name "NVIDIA GeForce RTX 4090" --gpu-name "NVIDIA GeForce RTX 3090" --gpu-name "NVIDIA RTX A6000"`. preprocess は dim 変更で再実行 + train。`dvc.lock` の case8 stage が deps hash 変更で invalidate されるので preprocess 自動 rerun (~7 分) + train (~7 分) の合計 ~25 分予想。
9. (execution side) **Stage 1 evaluation**: `diagnose_weights.py` で iter1 比 全 head +0.01 を確認。
10. (execution side) **Ablation**: 採否ゲート通過なら、4 カテゴリを個別 zero-out (template prefix=24-31 / fleet trajectory=32-35 / multihorizon=36-37,40-41 / production-centroid=38-39, global=10-13) して val metrics を取得、寄与切り分け。`iter2_result.md` の Ablation 表に列挙。
11. (フォローアップ) **300 ep self-play**: iter1 と合わせた最終 Stage 2 判定 (本 plan のスコープ外)。

## 検証方法 (Validation method)

### ローカル

```bash
# format/lint/mypy/pytest
dev/test-bot

# case8 iter2 unit test のみ (新規追加 test を含む)
uv run --directory bot pytest tests/pipeline/imitation/case8 -x

# featurizer dim sanity
uv --project bot run python -c "from pipeline.imitation.case8.policy.featurizer import PLANET_FEAT_DIM, GLOBAL_FEAT_DIM; print(PLANET_FEAT_DIM, GLOBAL_FEAT_DIM)"
# → 42 10 を期待

# Causal leak 回帰防止 (`test_featurizer_history.py` の history パターンを延長)
uv run --directory bot pytest tests/pipeline/imitation/case8 -k "no_leak" -x
```

### リモート (execution skill 側のスコープ)

```bash
git push origin feature/feature-engineering
# preprocess + train を 1 pod で。iter1 の onstart trap 8 個は修正済み (commit d12ec24)。
dev/runpod train <iter2-commit-sha> --case case8 \
  --gpu-name "NVIDIA GeForce RTX 4090" --gpu-name "NVIDIA GeForce RTX 3090" --gpu-name "NVIDIA RTX A6000"
# 想定所要時間: preprocess ~7 分 + train ~7 分 + dvc push + git push 含めて ~25 分。コスト ~$0.30
dev/runpod pull <run_id> --case case8
```

### 評価

- **Stage 1 (validation)**:
  - `uv --project bot run python -m pipeline.imitation.case8.evaluation.diagnose_weights --config bot/pipeline/imitation/case8/configs/il_case8.yaml --weights bot/pipeline/imitation/case8/policy/weights.pt --report data/output/experiment/imitation_case8_iter2_val_metrics.json`
  - **採否しきい値**: from PR-AUC + target macro F1 + ships macro F1 が **すべて iter1 比 +0.01 以上**
  - 1 head でも -0.005 を超える後退があれば **破棄**
- **Ablation (採用後必須)**:
  - 4 カテゴリを 1 つずつ zero-out した weights で同じ diagnose_weights を回す
  - `data/output/experiment/imitation_case8_iter2_ablation_table.md` (or json) に整理、`iter2_result.md` 内に転載
- **Stage 2 (out of scope for this plan)**: 採用後の 300 ep self-play vs baseline_v1 はフォローアップ別 plan / result で実施

### リーク回帰防止

- `test_featurizer_template_prefix.py` で template scoring が **同一 obs に対して deterministic** であることを assert (history 非依存)
- `test_featurizer_fleet_trajectory.py` で fleet future position が `obs_{N-1}` を参照しない (現在 fleet 状態のみ使う) ことを assert
- iter1 と同様に Stage 1 で異常 PR-AUC (e.g. > 0.95) が出た variant があれば即 ablation で原因切り分け

## リスク / 想定失敗モード

1. **Template prefix の dual signal 効果が薄い**: decoder が runtime で同じ scoring を計算しているので、policy network から見ると input と output が同じ情報源 → 学習しないリスク。緩和策: scoring を **正規化** (z-score per-frame) して decoder 側 raw scoring とは形式を変える。
2. **Fleet trajectory が sparse**: 多くの planet には inbound enemy fleet が無く、32-35 列が 0 fill ばかりになる。緩和策: 0 fill が多い → mask flag (`has_inbound_fleet`) を追加で渡し、policy が bias 学習を回避できるようにする。
3. **Multi-horizon が collinear**: horizon=5/15/30 の loss は強相関で、新規列が冗長になる。緩和策: ablation で確認、寄与小なら 1 horizon に削減して iter3 へ。
4. **Production-centroid が global 5 と redundant**: global 5 は production diff、global 12 は production diff norm。重複なら global 12 削除して別 feature (例: `enemy_home_distance`) に置換。
5. **学習コスト**: PLANET_FEAT_DIM 42 でモデル parameter は微増 ((42-24) × 128 = 2304 個)、訓練時間は ~7 分のまま。問題なし。
6. **RunPod onstart trap #8 (`data/output/models/imitation` symlink) 未検証**: iter1 後の commit `d12ec24` で修正済みだが本格 RunPod run で動作未確認。Step B run で初顕在化する可能性あり、その場合は failure log を見て即対応。

## Stop conditions

以下を満たしたら本 plan のスコープは完了:

- [ ] case8 featurizer/timeline/templates が iter2 仕様 (PLANET_FEAT_DIM=42, GLOBAL_FEAT_DIM=14) に拡張され、unit test が pass
- [ ] `dev/test-bot` が green
- [ ] commit & push まで完了
- [ ] (execution side) RunPod Step B が `99_done` で終了、`weights.pt` がローカルに pull
- [ ] Stage 1 採否判定が完了 (`diagnose_weights.py`)
- [ ] (採用なら) Ablation 4 件実行 + `iter2_result.md` 完成

## 参考 (References)

- [Sample-Efficient Behavior Cloning Using General Domain Knowledge (arXiv 2501.16546)](https://arxiv.org/html/2501.16546v1) — domain knowledge をモデルに直接埋め込む系統。template prefix score を input に与えるのは KIM (Knowledge-Informed Models) の精神に近い
- [Result Constraint Behavior Cloning (RCBC, ScienceDirect 2025)](https://www.sciencedirect.com/science/article/abs/pii/S0893608025012365) — BC の結果状態を制約として組み込む手法。本 plan の直接適用は無いが、 iter3 以降の auxiliary head 設計で参考に
- [Lux AI Challenge S3 (`Lux-AI-Challenge/Lux-Design-S3`)](https://github.com/Lux-AI-Challenge/Lux-Design-S3) — 上位解法の per-unit observation stacking が参考になる (CNN ベースだが per-unit feature engineering の発想は共通)
- 既存リポジトリ: `iter1_result.md`、`docs/experiment/imitation/20260427_case3_feature_engineering_phase2/result.md` (history 列の causal leak 事例)、memory `project_imitation_case1_phase3` (n<300 self-play は信頼不可、Stage 1 で採否判定する妥当性)
