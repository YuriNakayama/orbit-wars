# Hypotheses — imitation/case10 data_volume_sweep

> 作成日: 2026-05-09
> 最終更新: 2026-05-10
> 状態: in_progress
> 最大 iteration: **2** (1 仮説 = 1 iter = 1 チャート、各 iter 内で必要な全 data point を取得してから次 iter へ)
> 主要メトリクス:
>   - 学習: `val_loss` / `val_template_fire_acc` / `val_ships_acc`
>   - 実戦: vs `rulebase/case1` 勝率 (20 戦 × 先手/後手入替 = 40 戦、seed 0–19)
>   - コスト: preprocess wall-clock [s] / train total wall-clock [s] / 1 epoch 平均 [s]
> 既定 episode 数: vs rulebase/case1 で **40 戦** (20 戦 × P1/P2 入替)
> 使用 head: **template_ships** (case10 既定の `il_case10_template.yaml`)

## 目的

**実験用 dataset (= 後続 case 検証で常用するデータ量) を決定する**こと。
本実験単体での「最終的な施策採否」は目的としない (それは後続 case で大データを用いて行う)。

二段構えの設計:

1. **本実験 (case10)**: 小データで施策効果の **順位付け** ができる最小データ量を確定する
2. **後続 case (本実験スコープ外)**: 採否確定は full data で再走して判定する

## 仮説の構造

**1 仮説 = 1 iter = 1 チャート = 1 意思決定**。各 iter 内で必要な全 data point を取得してから次 iter に進む (個別 data point を別 iter にしない)。

| iter | 仮説 | 取得する data point | 出力チャート | 意思決定 |
|------|------|---------------------|--------------|---------|
| 1 | **H1: top_K sweep で精度・速度の頭打ち点が存在** | top_K ∈ {20, 40, 80, 160, all} の **5 点** すべて | chart 1: x=top_K (log) / 左 y=val_loss・val_template_fire_acc・40戦勝率 / 右 y=total wall-clock / 判定基準帯 (H5±閾値) を破線で重ねる | **採用 top_K の決定** |
| 2 | **H2: loser_swap=true は val 指標 / 勝率を改善** | top80 固定で swap=on / swap=off の **2 点** | chart 2: 棒グラフ (val_loss / val_template_fire_acc / 40戦勝率) を swap on vs off で並置 | **loser_swap の採否** |

## 共通フィルタ (両 iter 共通)

- **試合長 (turn count) 50〜400 ターンの episode のみ採用** (`turn_min=50`, `turn_max=400`)
  - 下限 50: 序盤 1〜2 行動で決着した極短試合 (ノイズ / 接続切れ等) を除外
  - 上限 400: 終盤グダ展開や draw 寸前の長期戦を除外、学習信号の質を優先
- match mode `1v1` 固定 (FFA は除外)
- draw episode は除外
- これらは preprocess (`_filter_index`) で適用、全 data point に共通

## H1: top_K sweep (iter1)

### スコープと固定軸

case10 template_ships variant を主軸に、以下を**固定**:

| 軸 | 固定値 | 出典 |
|----|--------|------|
| Featurizer | P=41 / G=20 (case10 既定) | `bot/pipeline/imitation/case10/configs/il_case10_template.yaml` |
| Backbone | Set Transformer hidden=192 / ISAB×4 / heads=8 / m=24 | 同上 |
| Head | **template_ships** | 同上 |
| Optimizer / scheduler | 既定 (lr=3e-4, cosine_warmup, 50 epoch) | 同上 |
| match mode | **`["1v1"]` 固定** | 本実験要件 |
| turn_min / turn_max / val_split | **50 / 400** / 0.10 (試合長 50〜400 ターンに絞る) | 同上 |
| seed (学習) | 0 (固定) | 同上 |
| **loser_swap** | **true 固定** (H2 が swap 効果を別途検証) | 本実験要件 |

**可変軸は `data.top_submission_limit` のみ**。

### 取得する data point (5 点)

| point | top_submission_limit | 期待 episode 数 |
|-------|----------------------|-----------------|
| top20 | 20                   | ~310            |
| top40 | 40                   | ~620            |
| top80 | 80 (case10 既定相当) | 1,224           |
| top160 | 160                 | ~2,400          |
| all (基準点) | null (上限なし) | lake 全件 (rating_quantile=0.50 で絞った後) |

→ **iter1 完了 = この 5 点すべての train + 40戦評価が揃った時点**。途中で打ち切らない。

### 採否判定基準 (実験用 dataset 選択)

H5 (top_K=all) を **基準点** とし、以下 3 条件を**すべて**満たす最小 top_K を「実験用 dataset」として採用する:

| 指標 | 閾値 | 根拠 |
|---|---|---|
| `best val_template_fire_acc` | all 比 **−1.0pp 以内** | 学習信号は seed 再現性高く、後続 case の施策順位付けに直結 |
| `best val_loss` | all 比 **+2.0% 以内** | 同上 |
| 40 戦勝率 | all 比 **−10pp 以内** | n=40 では noise floor が広い (memory `project_imitation_case1_phase3`)、参考扱い。±10pp は経験則 |

**最終決定の優先順位**: val_loss / val_template_fire_acc → 40戦勝率 → wall-clock。
val 指標を主軸とする理由は、(1) 40戦勝率は seed variance が ±10pp 級で個別 point の勝率差から data 量効果を分離できないこと、(2) val 指標は seed=0 固定で再現性が高いこと、の 2 点。

**採否ルール**:
- 上記 3 条件を満たす最小 top_K → 実験用 dataset に採用
- 全 top_K が条件を満たさない場合 → 全件 (all) を実験用にも採用 (= データ削減失敗)
- 同等の top_K が複数ある場合 → wall-clock が最小のものを採用

## H2: loser_swap 効果 (iter2)

### スコープと固定軸

iter1 と同じ固定軸 (template_ships / 1v1 / seed=0 / 50 epoch / 既定 hyperparams)。**top_K=80 に固定**、`loser_swap` のみ可変。

### 取得する data point (2 点)

| point | loser_swap | 備考 |
|-------|-----------|------|
| swap=true | true | iter1 H1 の top80 と同条件、再利用可 (再学習不要なら結果を借りる) |
| swap=false | false | case10 既定再現 |

### 採否判定

- val_template_fire_acc / val_loss / 40戦勝率の **すべてで swap=true ≧ swap=false** → loser_swap=true を後続 case の既定として採用
- 1 つでも swap=false が上回る指標があれば inconclusive 扱い (loser_swap は default false に倒す)

## データ拡張: 敗者側 episode の反転 (loser swap) — 仕様

case10 既定では 1 episode を player 0 / player 1 双方から featurize してサンプル化している (`_iter_episode_frames`, `bot/pipeline/imitation/case10/training/preprocess.py:286`)。`loser_swap=true` ではこれを以下に変更し、**全サンプルが「勝者視点」のラベル**になるようにする:

| 元の player slot | 元の勝敗 | 取り扱い |
|---|---|---|
| slot=0 (winner) | 勝 | そのまま featurize (obs / action とも slot=0) |
| slot=1 (winner) | 勝 | そのまま featurize (obs / action とも slot=1) |
| slot=0 (loser)  | 敗 | **obs を swap して player=1 視点に正規化、action は同 step の slot=1 (winner) のものを使用** |
| slot=1 (loser)  | 敗 | **obs を swap して player=0 視点に正規化、action は同 step の slot=0 (winner) のものを使用** |

実装方針 (新規 preprocess option `loser_swap: true`):
- index の `winner` 列で `winner_slot ∈ {0, 1}` を決定 (draw は既存通り除外)
- 各 step ごとに `loser_slot = 1 - winner_slot` のフレームを **obs.players の 0↔1 swap + obs.player = winner_slot** で featurize
- action_list は `step[winner_slot].action` を使う (= 「敗者の盤面でも勝者ならどう打ったか」)
- データ量は現状維持 (1 ep → 2 サンプル)、ただし全サンプルが winner ラベル

## 評価プロトコル (vs rulebase/case1)

各学習済み weights について以下を実施:

| 項目 | 値 |
|------|---|
| 相手 | `rulebase/case1` (= `baseline_v1`) |
| 戦数 | **20 戦 (seed 0–19) × 先手/後手入替 = 40 戦** |
| 評価対象 | 勝率 [%] / draw 率 [%] / mean turn count / mean turn p95 wall-clock |
| 実行 | `bot/pipeline/imitation/case10/evaluation/eval_data_volume_sweep.py` (新規 driver) |
| 結果先 | `data/output/experiment/case10_data_volume_sweep/eval_<tag>.json` |

- 全 data point で同じ seed list を共有 → 点間の勝率差をそのまま比較可能
- 先手/後手入替で先手バイアスを除去
- skip list: 300 戦は実施しない / Kaggle publicScore は引用しない

## 記録するメトリクス

各 data point ごとに以下を記録 (driver が `summary.json` に集約):

- **データ**: `data.kept_episode_count` / train.parquet 行数 / val.parquet 行数
- **時間**: preprocess wall-clock / train epoch 平均 / train total / 1 epoch あたりサンプル数
- **学習精度**: best val_loss / best val_template_fire_acc / best val_ships_acc / best epoch / final train_loss vs val_loss gap
- **実戦精度**: vs rulebase/case1 勝率 / draw 率 / mean turns

## 実施しない検証 / 評価 (skip list)

### 評価
- Kaggle publicScore は引用しない (project rule, memory `project_om_finding` / `project_case5_validation`)
- skill rating は使わない (project rule)
- **300 対戦は実施しない** — 本実験は 40 戦評価で sweep 形状を見るのが目的
- n<300 結果で「絶対勝率」を結論にしない (memory `project_imitation_case1_phase3`) — sweep 内での **相対比較** にのみ使用
- 1v1 以外のモード (FFA) は学習・評価とも実施しない
- 後続 case の施策採否は本実験では行わない (実験用 dataset の決定のみ)

### 分析
- replay 分析は本 hypotheses では実施しない (40 戦勝率 + val 指標で採否、必要なら follow-up で個別実施)
- step 方向 (turn_max) の削減 sweep は本実験では実施しない (top_K 1 軸固定)
- candidate_ships head の sweep は実施しない (template_ships のみ)
- baseline_v1 self-mirror noise floor 実測は実施しない (40戦勝率は経験則 ±10pp で扱い、参考指標)
- iter 内の data point を個別の iter として切り出すことは行わない (1 仮説 = 1 iter)

### 実行
- smoke test (1-episode self-play) は実施
- dev/test-bot は実施
- RunPod GPU を使う (case10 既定。CPU では 50 epoch × 5+2 point は非現実的)
- auto-recover loop は使う

### 例外条件
- iter1 で **vs case1 勝率が all 比 +10pp 以上** の data point があれば、その top_K を **新 case の base** として採用検討 (300 戦 follow-up は本 hypotheses スコープ外でユーザー判断)
- iter1 で全 top_K が判定基準を満たさない場合は全件採用 + 後続で削減方針再検討 (step 方向 / loser_swap 効果の単独計測など)

## 仮説リスト (priority 順)

- [ ] (P1) **H1: top_K sweep (5 point: 20/40/80/160/all)** — 1 iter で 5 data point すべて取得し chart 1 で意思決定。**採用 top_K を決定する主要仮説**
- [ ] (P2, depends on H1) **H2: loser_swap on/off 比較 (2 point: top80×{true,false})** — 1 iter で 2 data point 取得し chart 2 で意思決定。loser_swap の採否

## 成果物

iter2 完了後、`summary.md` に以下を集約:

### chart 1 用テーブル (H1)

| metric | top20 | top40 | top80 | top160 | all (基準) |
|--------|-------|-------|-------|--------|-----------|
| episodes | | | | | |
| train samples | | | | | |
| val samples | | | | | |
| preprocess [s] | | | | | |
| epoch wall [s] | | | | | |
| total train [s] | | | | | |
| best val_loss | | | | | |
| best val_template_fire_acc | | | | | |
| best val_ships_acc | | | | | |
| best epoch | | | | | |
| train-val loss gap | | | | | |
| **vs case1 勝率 (40 戦)** | | | | | |
| draw 率 | | | | | |
| **判定 (val 3 条件)** | | | | | |

### chart 2 用テーブル (H2)

| metric | swap=true (top80) | swap=false (top80) |
|--------|-------------------|---------------------|
| episodes | | |
| train samples | | |
| best val_loss | | |
| best val_template_fire_acc | | |
| best val_ships_acc | | |
| **vs case1 勝率 (40 戦)** | | |
| **swap=true 採否** | | |

### chart 出力先

`data/output/experiment/case10_data_volume_sweep/chart1_top_k_sweep.png`
`data/output/experiment/case10_data_volume_sweep/chart2_loser_swap.png`

最後に summary.md 末尾で **採用する実験用 dataset (top_K) と loser_swap の既定** を明記し、後続 case の base として参照可能にする。

## Iteration log

(各 iter 完了時に experiment-analysis / experiment が追記)

| iter | 開始 | 仮説# | data points | plan path | run_ids | 主要メトリクス | 採否 | result path | analysis path |
|------|------|-------|-------------|-----------|---------|----------------|------|-------------|---------------|
