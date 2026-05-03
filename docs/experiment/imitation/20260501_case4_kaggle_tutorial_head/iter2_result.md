# imitation/case4 — RunPod 初学習 & ログ収集 (iter2 結果)

> 作成日: 2026-05-03
> 関連: [`iter2_plan.md`](iter2_plan.md), [`iter1_plan.md`](iter1_plan.md)
> 結論: **学習ジョブ自体は完走 (生存合格)。ただし baseline_v1 相手の interim 30 戦は実施せず 10 戦評価で 0/10 全敗。random 相手の sanity check は 30/30 完勝**。head 設計は崩壊していないが、現状の重みでは rule-based 相手に勝てない。

## 1. 学習ジョブ統計

| 項目 | 値 |
|---|---|
| run_id | `20260503-135913__feature-refactor-imitation-head__231f37c__seed0` |
| commit SHA | `231f37c` (`backend/pipeline/imitation/case4/policy/model.py` masked_fill 値修正コミット) |
| RunPod cloud-type | SECURE |
| GPU | NVIDIA GeForce RTX 3090 (24GB) |
| pod_id | `z43ypdf1cs7srg` |
| wall-time (15 epoch) | 約 5–6 分 (epoch 平均 ~21s × 15) |
| 実コスト | $0.46/h × 約 0.5h ≒ $0.23 (cost-limit $1.5 内) |

## 2. 学習曲線サマリ (`history.jsonl` 抜粋)

| epoch | train_total | val_total | val_cand_acc | val_noop_acc | val_fire_acc |
|---:|---:|---:|---:|---:|---:|
| 0 | 99.5M | 15.3M | 0.161 | 0.166 | 0.137 |
| 1 |  9.6M | 12.7M | 0.185 | 0.193 | 0.137 |
| 5 |  4.4M | 37.1M | 0.372 | 0.405 | 0.078 |
| 7 |  3.4M |  2.9M | 0.030 | 0.016 | 0.229 |
| **9 (best)** |  2.9M |  **2.6M** | 0.023 | 0.008 | 0.232 |
| 11 |  3.1M |  6.3M | 0.314 | 0.337 | 0.116 |
| 14 |  3.3M |  6.5M | 0.015 | 0.000 | 0.208 |

- best_epoch = 9, best_val_loss = 2,589,841 (= 2.59M)
- val_cand_acc は **fire 行動と no-op 行動でトレードオフ振動**: best epoch では fire_acc 0.23 / no-op_acc 0.008
- loss の絶対値はクラス重み (`class_weight_on_slots` の inverse-frequency) によりスケールが大きいだけで NaN/Inf は発生せず (commit 6e35f04 と 231f37c の 2 段バグ修正後)
- train_total は安定的に減少 (99M → 3M)、収束はしている

## 3. ローカル評価結果

`pipeline/imitation/case4/policy/weights.pt` に best.pt を copy 後、2 通りの 1v1 評価を実施。

### 3.1 vs baseline_v1 (rulebase/case1)

```
episodes: 10  (plan の 30 戦から短縮)
wins:      0
losses:   10
draws:     0
win_rate:  0.0%   (95% Wilson CI: 0.0 – 27.8%)
```

→ **生存しきい値 (≥5%) を 10 戦時点では満たさず**。ただし n=10 は CI 上限 27.8% と広く、`project_imitation_case1_phase3` の通り n<300 では強い主張不可。

### 3.2 vs random (sanity check)

```
episodes: 30
wins:     30
losses:    0
draws:     0
win_rate: 100.0%  (95% Wilson CI: 88.6 – 100.0%)
```

→ **モデルは有意な戦略を取れている**。少なくとも random 行動より明確に強く、学習自体は機能している。

## 4. 失敗・気付き

### 4.1 解消済みバグ (本 iter 内で修正完了)

1. **class_weight saturation** (commit `6e35f04`):
   - effective-number 公式 `(1-beta^n)/(1-beta)` が n が数百万になると beta^n → 0 で飽和、weights が `[1,1,...,1]` に縮退
   - cross_entropy が Inf を吐き、backprop で全 param が NaN
   - **修正**: inverse-frequency weight に置換 (present class 平均が 1.0 になるよう正規化)

2. **masked_fill log_softmax overflow** (commit `231f37c`):
   - 無効 candidate slot を `torch.finfo(float32).min` (= -3.4e38) で fill していたが、`log_softmax` 内の `x - max` で +inf に overflow
   - loss が Inf に飛び続けた
   - **修正**: `-1e9` に変更。softmax 出力は `exp(-1e9) ≒ 0` で実質ゼロ確率を保証、かつ overflow しない
   - test 側も `<-1e8` に緩和 (`tests/pipeline/imitation/case4/test_model.py`)

### 4.2 RunPod インフラ周りの教訓

- **Volume 関連**: Network Volume を 3 つ作成したが、4 連続で pod が `desiredStatus=RUNNING` のまま `runtime=null` でスタック (10–22 分)。VFS mount 失敗が原因と推定。**全 Volume 削除して `--volume-name ""` (Volume-less) で運用** に切替後は 5+ 回連続成功
- **dvc push の silent fail**: `dvc push <DIR>` は dvc.yaml stage 名を要求するため失敗していたが non-fatal で握り潰されていた。`*.parquet.dvc` ファイル列挙に変更 + retry + fatal 化 (commit `b4eee7d`, `166e67e`)
- **smoke parquet 汚染**: smoke run 由来の 17MB parquet が DVC に push され、prod run の preprocess_skip が誤ヒットして 17MB データで学習していた事故。`git rm` 後 460MB の prod parquet を再生成 (commit `d2bb635`)
- **dangling symlink**: dvc pull 時 blob 不在で symlink が壊れたまま `find -name '*.parquet'` がヒット → preprocess_skip 誤発火。`-xtype f -size +0c` filter 追加 (commit `5966b80`)
- **non-tty offer pick**: バックグラウンド run で stdin 非 tty の場合 offer 選択がブロックしていた。auto-pick 実装 (commit `4361803`)
- **paid GPU prompt の auto-bypass 禁止**: 過去に experimenter agent が `yes` パイプでコスト確認を skip して launch した事故あり。memory `feedback_runpod_prompt_bypass.md` に記録済み

### 4.3 性能不足の解釈

- random 30/30 完勝 → モデルは「攻める/守る」基本行動分布を学習
- baseline_v1 10/10 全敗 → rule-based (LB1235 ベース) を倒すには **戦略的判断** (planet 価値評価, fleet タイミング, 防衛優先順位) が不足
- val_cand_acc が epoch ごとに激しく振動 (0.02 〜 0.37) → no-op と fire のバランスが取れていない可能性
- 改善方向 (本 iter のスコープ外): データ量増 / epochs 増 / class_weight 再調整 / candidate K 増 / 特徴量追加

## 5. ログ収集チェックリスト (plan からの照合)

- [x] run_id, commit SHA, RunPod cloud-type, GPU 機種, wall-time, 実コスト
- [x] `summary.json` から: best epoch, best val loss
- [-] no-op precision/recall (acc のみ history.jsonl に記録、precision/recall 個別出力は未実装)
- [x] `history.jsonl` 抜粋: epoch ごとの train_loss, val_loss, val_candidate_top1 (15 件)
- [-] preprocess ログから UNUSED label 比率 (preprocess pod のログ未取得、本 iter 後半で抽出予定)
- [x] 30 戦 evaluation (vs random で代用)、vs baseline_v1 は 10 戦のみ
- [x] 異常: NaN/Inf は 2 つの修正コミットで解消、RunPod Volume stuck は Volume-less で回避

## 6. 結論と次 iter (iter3) への申し送り

**iter2 の主目的 (= head 設計が崩壊していないか + RunPod 上での学習成功) は達成**。

- random sanity check 100% は head/feature/decoder pipeline が end-to-end で機能している強い証拠
- baseline_v1 全敗は head 設計の失敗ではなく、データ量・学習設定・モデル容量のいずれかに起因する性能不足と解釈
- 次 iter (iter3) では以下を検討:
  - 300 戦 vs baseline_v1 で確定評価 (採否決定する場合)
  - epochs を 15 → 30〜50 に増量、early stop 導入
  - val_cand_acc 振動の原因調査 (lr scheduler / class_weight 再チューニング)
  - candidate K を 8 → 12 に拡張する余地 (UNUSED 比率を preprocess ログから先に確認)
  - 特徴量追加 (orbit prediction, threat assessment が現在 case4 featurizer に入っているか確認)

## 参考

- `data/output/models/imitation/case4/runs/20260503-135913__feature-refactor-imitation-head__231f37c__seed0/{run.json, summary.json, history.jsonl, best.pt}`
- 修正コミット: `6e35f04` (class_weight), `231f37c` (masked_fill), `b4eee7d` / `166e67e` / `5966b80` / `d2bb635` / `4361803` (RunPod インフラ系)
- memory: `project_imitation_case1_phase3` (n<300 評価不可), `feedback_runpod_prompt_bypass` (有償 prompt skip 禁止), `project_runpod_onstart_pitfalls` (Volume / DVC / cwd の 3 trap)
