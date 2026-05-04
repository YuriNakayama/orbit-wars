# imitation/case8 — Ship-Count Regression Head (case4 拡張)

> 作成日: 2026-05-04
> 関連:
> - `bot/pipeline/imitation/case4/policy/model.py` (case4 candidate head — そのまま流用)
> - `bot/pipeline/imitation/case4/policy/decoder.py:38 _fixed_ship_count` (現行 rule-based ship 数)
> - `docs/experiment/imitation/20260501_case4_kaggle_tutorial_head/iter2_result.md` (case4 baseline: vs random 30/30, vs baseline_v1 0/10)
> - `docs/experiment/imitation/20260503_case5_ship_prediction/plan.md` (**別物**: case5 は敵 fleet 到達後の planet 残存予測 = timeline featurizer。本 plan は自軍発射 ship 数の predicton head 追加)
>
> スコープ: case4 の per-source candidate head に **発射 ship 数の連続値 regression 副 head** を追加した imitation/case8 を新設。決定変数を 1 個増やすだけで backbone / candidate slot 構造は不変。

## 仮説 (Hypothesis)

case4 は per-source × CAND_K=8 categorical で「どこを撃つか」だけ学習し、発射 ship 数は
`max(tgt.ships+1, 20)` という rule-based 固定値で決定している。
**発射 ship 数を replay の actual `act[2]` から regression で学習する副 head** を追加すれば、
ship 数のニュアンス (large attack / 計算された minimum / over-fire 抑制) を policy が学べ、
case4 の現行 weights (0/10 vs baseline_v1) より勝率が改善するはず。

メカニズム:

- 現行 case4 は target 選択を学べても **過小・過剰発射の判断は rule に丸投げ**
- replay top players の発射 ship 数は `max(tgt+1, 20)` よりも文脈依存 (序盤 small probe / 中盤 keep-needed 計算 / 終盤 burst)
- ship 数 regression head を joint loss で同時学習 → fire 判断 (cand head) と量判断 (ship head) が独立変数として decode 段で組み合わせ可能

## 既存コードの現状 (from Step 1)

- **case4 model**: `policy/model.py` の `CandidatePolicy` は Graph U-Net backbone + per-source candidate categorical head 1 本のみ。`PolicyOutput` は `candidate_logits (B, P, K)` のみ
- **case4 decoder**: `policy/decoder.py:113 ships = _fixed_ship_count(target.ships)` で rule 固定。`act[2]` (replay の発射 ship 数) は preprocess 段で reverse-resolve に使うが教師として保存していない
- **case4 preprocess**: `training/preprocess.py:184 ships = int(act[2])` で取得済み。parquet 列追加 (`ship_label_per_src`) は容易
- **case4 losses**: `training/losses.py:compute_loss` は cand_loss 1 本。joint loss (cand + ship) への拡張は副 head 学習の標準パターン
- **case3 先例**: case3 にあった 4-bucket `ships_head` (`max(target.ships+1, 20)` を超える行動を学習) を case4 で削除した経緯あり。今回は continuous regression として復活させる
- **iter2 結果**: case4 は random 30/30 / baseline_v1 0/10。head 設計は機能、ship 数 rule 固定は性能上限を抑える候補要因の 1 つ
- **混同注意**: 同名イベント `imitation/case5` の plan は **「敵 fleet 到達後の planet 残存予測 (timeline featurizer)」** で別仮説。本 plan は自軍発射 ship 数の **action 側** prediction head 追加であり、両者を混同しない

## スコープ (Scope)

### 新規 case 切り出し (`bot/pipeline/imitation/case8/`)

case6/case7 は他で使用中とのユーザー情報により **case8** を新設。case4 を `cp -r` し、relative import を case8 に書き換えた上で head のみ拡張する。

| Path | 変更内容 |
|---|---|
| `bot/pipeline/imitation/case8/` (新規, `cp -r case4`) | base となる copy |
| `bot/pipeline/imitation/case8/policy/types.py` | `PolicyOutput` に `ship_pred: torch.Tensor (B, P)` を追加 (continuous scalar) |
| `bot/pipeline/imitation/case8/policy/model.py` | `CandidatePolicy` に `ship_head = MLP(self_h ⊕ global_h ⊕ chosen_cand_h)` を追加。学習時は **all candidate slots ではなく per-source 1 scalar** を返す (= 「このソースから発射するなら何隻か」) |
| `bot/pipeline/imitation/case8/policy/decoder.py` | argmax cand_slot で no-op 以外なら `ships = clamp(round(ship_pred[slot]), [1, src.ships])`。`max(tgt.ships+1, 20)` rule は **fallback** として残し、ship_pred が `< tgt.ships+1` なら rule 値を採用 (under-fire 防止) |
| `bot/pipeline/imitation/case8/training/preprocess.py` | parquet schema に `ship_label_per_src: int32 (MAX_PLANETS,)` 追加。fired src のみ `act[2]`、その他は `-1` (loss から除外) |
| `bot/pipeline/imitation/case8/training/losses.py` | `compute_loss` を 2-head 化: `total = cand_loss + λ * ship_loss`。`ship_loss = SmoothL1Loss(ship_pred[fired], ship_label[fired])`。`λ = 1.0` start (sweep 余地あり) |
| `bot/pipeline/imitation/case8/training/dataset.py` | parquet 読み込みに `ship_label_per_src` を追加 |
| `bot/pipeline/imitation/case8/training/train.py` | log に `ship_loss` / `ship_mae` を追加 |
| `bot/src/dataset/selfplay/agents.py` | `"il_v8": "pipeline.imitation.case8.policy.agent:agent"` を AGENT_REGISTRY に追加 |
| `bot/pipeline/imitation/case8/configs/il_case8.yaml` | case4 の config を copy、`λ_ship: 1.0` を追加 |
| `dvc.yaml` | `preprocess_imitation_case8 / train_imitation_case8` stage を追加 |
| `bot/src/runpod_io/cli.py` の `CASE_DEFAULTS` | `case8` entry 追加 |
| `bot/pipeline/imitation/case8/evaluation/eval_vs_baseline.py` | case4 と同形 |

### 変更なし

- case1-5 (imitation), rulebase 全 case — **触らない**
- Submit-shape: case8 は新 case で canonical (`il_v4` weights.pt) に影響なし

### Hyperparameters

| Knob | Before (case4) | After (case8) |
|---|---|---|
| Head 数 | 1 (cand_logits) | 2 (cand_logits + ship_pred) |
| Ship label | なし | `act[2]` continuous, fired src のみ (no-op = -1 で loss から除外) |
| Ship head 出力 | なし | per-source 1 scalar (B, P) |
| Ship loss | なし | `SmoothL1Loss` (Huber δ=1, fired src のみ平均) |
| `λ_ship` | — | `1.0` (initial), 不調なら `0.5 / 2.0` sweep |
| Decoder ship rule | `max(tgt.ships+1, 20)` 固定 | `clamp(round(ship_pred), [max(tgt.ships+1, 20), src.ships])` (under-fire 防止に rule 値が下限) |

`SmoothL1` 採用理由: replay の発射 ship 数は右裾長 (20-30 が大半、序盤 burst で 200+) → MSE は外れ値で勾配爆発、SmoothL1 は外れ値ロバスト + 小誤差で MSE 的挙動 ([Huber loss - Wikipedia](https://en.wikipedia.org/wiki/Huber_loss))。

## 実装ステップ (Implementation outline)

1. `cp -r bot/pipeline/imitation/case4 bot/pipeline/imitation/case8` → 全 import path / `__init__.py` を case8 に書換
2. `policy/types.py`: `PolicyOutput.ship_pred: torch.Tensor (B, P)` を追加
3. `policy/model.py`: `CandidatePolicy.__init__` に `self.ship_head = nn.Sequential(Linear(3*h, h), ReLU(), Linear(h, 1))` を追加。`forward` 末尾で `ship_pred = self.ship_head(joint).squeeze(-1).max over k slots` ではなく **per-source 1 scalar** (例: `joint` を `cand_logits` の argmax slot で gather してから ship_head に通す、または cand と独立に self_h/global_h のみで produce)
4. `policy/decoder.py`: `decode` で `ship_pred[slot]` を取得、上記 clamp rule を適用
5. `training/preprocess.py`: `ship_label_per_src` 列を出力 (fired src の `int(act[2])`、その他 `-1`)
6. `training/losses.py`: `compute_loss` を 2-head に拡張、`SmoothL1` を fired src 上で平均、`total = cand_loss + λ * ship_loss`
7. `training/dataset.py / train.py`: schema 追加 + log 拡張
8. `tests/pipeline/imitation/case8/` に最小テスト (model forward output shape / loss compute / decode clamp rule)
9. `src/dataset/selfplay/agents.py` / `dvc.yaml` / `runpod_io/cli.py CASE_DEFAULTS` を更新
10. `dev/test-bot` で format/lint/type/pytest を pass
11. RunPod Step A (preprocess smoke) → Step B (train, ~$1.0)
12. ローカル `eval_vs_baseline.py` で baseline_v1 vs il_v8 を 50 戦実行
13. `result.md` を本ディレクトリに記述

## 検証方法 (Validation method)

- **ローカル**:
  - `dev/test-bot` (format / lint / type / pytest)
  - `uv run --directory bot pytest tests/pipeline/imitation/case8 -x`
- **リモート**:
  - `dev/runpod train <commit-sha> --case case8` (preprocess + train)
  - 想定所要時間: preprocess ~10 分 / train 15 epoch で ~5-6 分 (case4 iter2 ベース)
  - 想定コスト: ~$1.0 (Step A + Step B)
- **評価**:
  - 対戦相手: `baseline_v1` (rulebase/case1)
  - エピソード数: **50 戦** (ユーザー指定。memory `project_imitation_case1_phase3` の n<300 不可信頼ルールを承知の上での縮小 sanity 評価)
  - 主要メトリクス:
    1. **学習ログの正常性目視確認**: NaN/Inf なし、`train_total` が単調減少傾向、`val_cand_acc` と `val_ship_mae` の epoch 推移
    2. **vs baseline_v1 50 戦勝率 > 0%** (iter2 case4 の 0/10 を上回ること)
  - 採否しきい値: 50 戦で **勝率 > 0%** (チラージ判定)。**より厳格な採否を後で行う場合は別 iter で 300 戦評価を推奨**

### 採否しきい値の補足

ユーザー判断により本 iter の評価は 50 戦のみで完結する。50 戦は Wilson 95% CI が広く (例: 5/50 = 10%, CI 3.3-21.8%)、case4 比 +5pp 程度の差を有意に検出することはできない。本 plan は **「ship-count head が pipeline として動作し、case4 の `0/10` を超えるか」** の生存判定として 50 戦を運用する。本格採否 (production weights 昇格 / il_v8 を `il_v4` に置換) を行う場合は別途 300 戦を回す前提とする。

## 既知のリスク / 注意点

1. **Ship head の出力位置**: `joint = (self ⊕ global ⊕ cand_h)` は (B, P, K, 3H)。ship head が「per-candidate ship 数」になると K 倍の出力が要るが、replay ラベルは fired src あたり 1 個なので **per-source 1 scalar** (= 「このソースから発射するなら何隻か」) に decoder 段で gather すべき。実装時は `model.forward` 末尾で `argmax_slot = cand_logits.argmax(-1)` → `gathered = joint.gather(2, argmax_slot[..., None, None].expand(-1,-1,1,3*h)).squeeze(2)` → `ship_head(gathered)` の流れ。学習時は teacher forcing で `cand_slot_per_src` の正解 slot を gather すれば label と整合する
2. **iter2 で発生した bug の再発防止**: `class_weight saturation` (commit 6e35f04) と `masked_fill log_softmax overflow` (commit 231f37c) は case4 の修正済み実装を case8 にコピー後も維持。`-1e9` masking と inverse-frequency weight のままにする
3. **DVC stage 追加時の cache 競合**: 他 worktree との DVC cache 共有で lock contention の懸念あり (`.claude/rules/command.md`)。並行 `dev/dvc repro` を避ける
4. **RunPod 対話 prompt**: `feedback_runpod_prompt_bypass` の通り cost 確認 prompt を yes/auto bypass しない
5. **Cross-case import 禁止**: case8 から case4 を import しない (`.claude/rules/bot/pipeline.md`)。共通コードは copy で対応する

## 参考 (References — Step 3 web 調査)

- [Huber Loss — Wikipedia](https://en.wikipedia.org/wiki/Huber_loss): SmoothL1 = Huber(δ=1)。MSE の outlier 過剰 penalty を回避しつつ小誤差は二次的に振る舞う。fired ship 数の右裾長分布に適合
- [Huber Loss for Regression with Outliers (mlexplained)](https://mlexplained.blog/2023/07/31/huber-loss-loss-function-to-use-in-regression-when-dealing-with-outliers/): replay 起源の noisy label に対し Huber が MSE / MAE より頑健と整理
- [Lux AI with Imitation Learning (Kaggle, shoheiazuma)](https://www.kaggle.com/shoheiazuma/lux-ai-with-imitation-learning) / [IsaiahPressman/Kaggle_Lux_AI_2021](https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021): Lux S1 imitation 系は CNN + multi-head actor。明示的な count regression は標準パターンとして文献化されておらず、本実験は notebook 流 head 拡張として独自実装の余地あり

## 次 iter 候補 (本 plan の範囲外)

- 50 戦で >0% を確認できたら 300 戦で +5pp 採否判定
- `λ_ship` sweep (0.5 / 1.0 / 2.0)
- ship head を categorical (Round 1 で議論した 6-bucket [20, 30, 50, 80, 120, 200]) に切替えての A/B
- ship_pred を decoder で `1.0×` ではなく learned overfire-suppression として利用
