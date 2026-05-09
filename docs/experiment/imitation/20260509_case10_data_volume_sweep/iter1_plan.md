# iter1 plan — H1: top_submission_limit=20 (smallest, loser_swap=true)

> 仮説: data 量を case10 既定 (top80, ~1224 ep) から 1/4 (top20, ~310 ep) に削った場合、
> (a) val_cand_fire_acc は明確に低下、(b) train_loss と val_loss の gap が拡大 (過学習傾向)、
> (c) preprocess / train wall-clock は episode 数に概ね比例して短縮、
> (d) vs `rulebase/case1` 40 戦勝率 (20 戦 × P1/P2 入替) は 40% 前後 (sweep 下限) になる、
> を確認する。

## 設定 (case10 candidate_ships からの差分のみ)

| 項目 | 値 |
|------|---|
| config | `bot/pipeline/imitation/case10/configs/data_volume_sweep/il_case10_candidate_top20.yaml` |
| `data.modes` | **`["1v1"]` 固定** |
| `data.top_submission_limit` | **20** (case10 既定 80 から削減) |
| `data.loser_swap` | **true** (敗者側 episode を勝者視点に反転して使用) |
| `data.out_train` | `data/mart/imitation/case10/data_volume_sweep/top20/train.parquet` |
| `data.out_val`   | `data/mart/imitation/case10/data_volume_sweep/top20/val.parquet` |
| `train.weights_out` | `pipeline/imitation/case10/policy/data_volume_sweep/weights_candidate_ships_top20.pt` |
| その他 (model / loss_weights / scheduler / batch / epoch / seed) | case10 candidate_ships と同一 |

## 前提となる preprocess 改修 (本 plan の前にマージ済みであること)

`bot/pipeline/imitation/case10/training/preprocess.py` に以下を追加:

1. `_filter_index` の戻り値に `winner_by_match_id: dict[str, int]` を含める (index の `winner` 列から)
2. `data.loser_swap: bool` config を読む (デフォルト false)
3. `_iter_episode_frames` 引数に `winner_slot: int | None` と `loser_swap: bool` を追加
4. `loser_swap=true` かつ `slot != winner_slot` の場合:
   - obs を `_swap_players(obs)` (player 0↔1 swap、obs.player を winner_slot に書換)
   - action_list を `step[winner_slot]["action"]` から取得
   - history も winner 側を使う (各 step ごとに history を追跡)
5. draw episode は引き続き除外

## 実行手順

```bash
# 1. preprocess + train を sweep driver 経由で実行 (top20 のみ)
uv run --directory bot python -m pipeline.imitation.case10.training.run_data_volume_sweep \
  --configs top20

# 2. 学習成果物に対して vs rulebase/case1 評価 (20 戦 × P1/P2 入替 = 40 戦)
uv run --directory bot python -m pipeline.imitation.case10.evaluation.eval_data_volume_sweep \
  --tag top20 --opponent baseline_v1 --episodes 20 --swap-sides

# 3. RunPod 投入する場合 (推奨、5 iter まとめて並列でも可)
git push origin feature/imitation-data-volume-sweep
dev/runpod train <commit-sha> --case case10 \
  --config-path pipeline/imitation/case10/configs/data_volume_sweep/il_case10_candidate_top20.yaml
```

## 記録するメトリクス

driver が `data/output/experiment/case10_data_volume_sweep/summary.json` に書き込む値:

- `data.kept_episode_count`
- train.parquet / val.parquet 行数
- preprocess wall-clock [s]
- train total wall-clock [s] / 1 epoch 平均 [s]
- best val_loss / best val_cand_fire_acc / best val_ships_acc / best epoch
- final epoch の train_loss / val_loss (gap = val_loss - train_loss)
- **vs rulebase/case1 勝率 / draw 率 / mean turns (40 戦)**

## 検証方法

skip list に従い:
- 1v1 のみ、FFA は使わない
- ローカル self-play 300 戦は実施しない (40 戦のみ)
- replay 分析は本 iter では実施しない
- 学習中 val 指標 + 40 戦勝率 + wall-clock のみで採否

→ iter1 単体では「H1 完了 (data point 1 取得)」を確認するだけ。最終的な「データ量 vs 精度・速度」曲線は iter5 + R1 完了後に summary.md で評価する。
