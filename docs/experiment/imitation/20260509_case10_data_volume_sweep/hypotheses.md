# Hypotheses — imitation/case10 data_volume_sweep

> 作成日: 2026-05-09
> 最終更新: 2026-05-09
> 状態: in_progress
> 最大 iteration: 5 (data point 5 段階)
> 主要メトリクス:
>   - 精度 (学習): val_loss / val_cand_fire_acc / val_template_fire_acc / val_ships_acc
>   - 精度 (実戦): vs `rulebase/case1` の勝率 (20 戦 × 先手/後手入替 = 40 戦、seed 0–19)
>   - 実行時間: preprocess wall-clock [s] / train epoch wall-clock [s] / train total wall-clock [s] / 1 epoch あたりサンプル数
> 既定 episode 数: vs rulebase/case1 で **40 戦** (20 戦 × P1/P2 入替)

## 背景

case10 は `top_submission_limit=80` (~1,224 episode, 1v1) で固定されてきた。データ量を変えた時に
- (a) 精度 (val_* + 実戦勝率) がどこで頭打ちするか
- (b) 学習・前処理の wall-clock がどう増えるか
- (c) 過学習の挙動 (train vs val gap) がどう変わるか
を 1 軸 (`top_submission_limit`) スイープで定量化し、後続 case の data 設計の根拠にする。

## スコープと固定軸

case10 candidate_ships variant を主軸に、以下を**固定**:

| 軸 | 固定値 | 出典 |
|----|--------|------|
| Featurizer | P=41 / G=20 (case10 既定) | `bot/pipeline/imitation/case10/configs/il_case10_candidate.yaml` |
| Backbone | Set Transformer hidden=192 / ISAB×4 / heads=8 / m=24 | 同上 |
| Head | candidate_ships (主軸) | 同上 |
| Optimizer / scheduler | 既定 (lr=3e-4, cosine_warmup, 50 epoch) | 同上 |
| match mode | **`["1v1"]` 固定** | 本実験要件 |
| turn_min / turn_max / val_split | 80 / 400 / 0.10 (固定) | 同上 |
| seed (学習) | 0 (固定) | 同上 |

**可変軸は `data.top_submission_limit` のみ**。

## データ拡張: 敗者側 episode の反転 (loser swap)

case10 既定では 1 episode を player 0 / player 1 双方から featurize してサンプル化している (`_iter_episode_frames`, `bot/pipeline/imitation/case10/training/preprocess.py:286`)。本実験ではこれを以下に変更し、**全サンプルが「勝者視点」のラベル**になるようにする:

| 元の player slot | 元の勝敗 | 取り扱い |
|---|---|---|
| slot=0 (winner) | 勝 | そのまま featurize (obs / action とも slot=0) |
| slot=1 (winner) | 勝 | そのまま featurize (obs / action とも slot=1) |
| slot=0 (loser)  | 敗 | **obs を swap して player=1 視点に正規化、action は同 step の slot=1 (winner) のものを使用** |
| slot=1 (loser)  | 敗 | **obs を swap して player=0 視点に正規化、action は同 step の slot=0 (winner) のものを使用** |

実装方針 (新規 preprocess option `loser_swap: true` として preprocess に追加):
- index の `winner` 列で `winner_slot ∈ {0, 1}` を決定 (draw は既存通り除外)
- 各 step ごとに `loser_slot = 1 - winner_slot` のフレームを **obs.players の 0↔1 swap + obs.player = winner_slot** で featurize
- action_list は `step[winner_slot].action` を使う (= 「敗者の盤面でも勝者ならどう打ったか」)
- データ量は現状維持 (1 ep → 2 サンプル)、ただし全サンプルが winner ラベル

→ counterfactual 学習 (敗者盤面 + 勝者 action) を含むため、まず loser_swap=true で全 sweep を回し、参考点として loser_swap=false (既定) も 1 点 (top80) のみ計測して比較する。

## 可変軸 (data volume) と評価対象

`top_submission_limit` を 5 段階でスイープ。すべての sweep 点で **vs `rulebase/case1` 40 戦評価** を実施:

| iter | 仮説 | top_submission_limit | 期待 episode 数 | loser_swap |
|------|------|----------------------|-----------------|------------|
| 1 | H1 | 20                   | ~310            | true       |
| 2 | H2 | 40                   | ~620            | true       |
| 3 | H3 | 80 (case10 既定相当) | 1,224           | true       |
| 4 | H4 | 160                  | ~2,400          | true       |
| 5 | H5 | null (上限なし)      | lake 全件       | true       |
| 参考 | R1 | 80                | 1,224           | **false** (case10 既定再現)  |
| 参考 | R2 | -                  | -               | -          | `baseline_v1` (rulebase/case1 自身) を vs rulebase/case1 にぶつけた self-mirror 勝率 (50% ± seed noise の noise floor 確認) |

## 評価プロトコル (vs rulebase/case1)

各学習済み weights について以下を実施:

| 項目 | 値 |
|------|---|
| 相手 | `rulebase/case1` (= `baseline_v1`) |
| 戦数 | **20 戦 (seed 0–19) × 先手/後手入替 = 40 戦** |
| 評価対象 | 勝率 [%] / draw 率 [%] / mean turn count / mean turn p95 wall-clock |
| 実行 | `bot/pipeline/imitation/case10/evaluation/compare_*.py` 系を sweep 用に拡張 (新 driver: `eval_data_volume_sweep.py`) |
| 結果先 | `data/output/experiment/case10_data_volume_sweep/eval_<tag>.json` |

- 全 sweep 点で同じ seed list を共有 → 点間の勝率差をそのまま比較可能
- 先手/後手入替で先手バイアスを除去
- skip list: 300 戦は実施しない / Kaggle publicScore は引用しない

## 記録するメトリクス

各 iter で以下を記録 (driver が `summary.json` に集約):

- **データ**: `data.kept_episode_count` / train.parquet 行数 / val.parquet 行数
- **時間**: preprocess wall-clock / train epoch 平均 / train total / 1 epoch あたりサンプル数
- **学習精度**: best val_loss / best val_cand_fire_acc / best val_ships_acc / best epoch / final train_loss vs val_loss gap
- **実戦精度**: vs rulebase/case1 勝率 / draw 率 / mean turns

## 実施しない検証 / 評価 (skip list)

### 評価
- Kaggle publicScore は引用しない (project rule, memory `project_om_finding` / `project_case5_validation`)
- skill rating は使わない (project rule)
- **300 対戦は実施しない** — 本実験は 40 戦評価で sweep 形状を見るのが目的
- 1v1 以外のモード (FFA) は学習・評価とも実施しない

### 分析
- replay 分析は本 hypotheses では実施しない (40 戦勝率 + val 指標で採否、必要なら follow-up で個別実施)
- n<300 結果で「絶対勝率」を結論にしない (memory `project_imitation_case1_phase3`) — sweep 内での **相対比較** にのみ使用

### 実行
- smoke test (1-episode self-play) は実施
- dev/test-bot は実施
- RunPod GPU を使う (case10 既定。CPU では 50 epoch × 5 iter は非現実的)
- auto-recover loop は使う

### 例外条件
- vs case1 勝率が iter3 (top80) 比 +10pp 以上の iter があれば、その data 量を **新 case の base** として採用検討 (300 戦 follow-up は本 hypotheses スコープ外でユーザー判断)

## 仮説リスト (priority 順)

- [ ] (P1) **H1: top_submission_limit=20 (smallest)** — 最小データ。過学習 / 高 train-val gap / wall-clock 最短の下限点 / 実戦勝率も最小と予想
- [ ] (P1) **H2: top_submission_limit=40** — 倍増。サンプル不足域の中間点
- [ ] (P1) **H3: top_submission_limit=80** — case10 既定。リファレンス点
- [ ] (P1) **H4: top_submission_limit=160** — 既定の 2 倍。精度頭打ち判定の主要点
- [ ] (P1) **H5: top_submission_limit=null (全件)** — lake 全件。data scaling 曲線の右端
- [ ] (P2) **R1: loser_swap=false (top80)** — case10 既定再現。loser_swap 効果の単独計測 (差分は H3 vs R1 の対比で読む)

## 成果物

iter5 (+ R1) 完了後、`summary.md` に以下を集約:

| metric | iter1 (top20) | iter2 (top40) | iter3 (top80) | iter4 (top160) | iter5 (all) | R1 (top80, swap off) |
|--------|---------------|---------------|---------------|----------------|-------------|----------------------|
| episodes | | | | | | |
| train samples | | | | | | |
| val samples | | | | | | |
| preprocess [s] | | | | | | |
| epoch wall [s] | | | | | | |
| total train [s] | | | | | | |
| best val_loss | | | | | | |
| best val_cand_fire_acc | | | | | | |
| best val_ships_acc | | | | | | |
| best epoch | | | | | | |
| train-val loss gap | | | | | | |
| **vs case1 勝率 (40 戦)** | | | | | | |
| draw 率 | | | | | | |

加えて精度 / 実行時間 vs データ量の log-log プロット (matplotlib) を `data/output/experiment/case10_data_volume_sweep_*.png` に出力。

## Iteration log

(各 iter 完了時に experiment-analysis / experiment が追記)

| iter | 開始 | 仮説# | plan path | run_id | 主要メトリクス | 採否 | result path | analysis path |
|------|------|-------|-----------|--------|----------------|------|-------------|---------------|
