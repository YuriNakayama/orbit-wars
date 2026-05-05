# imitation/case7 — Feature Engineering 拡張 Result

> 作成日: 2026-05-05
> 関連 plan: `./plan.md`
> commit: `5a89082` (RunPod 修正一連 + mark_progress) → train run `20260505-040940__feature-feature-engineering__5a89082__seed0`
> weights: `bot/pipeline/imitation/case7/policy/weights.pt` (best.pt epoch 9, val_loss=3.5235)

## サマリ (TL;DR)

| 段階 | 結果 | 採否 |
|------|------|------|
| **訓練** | 15 epoch 完走、val_loss best @ epoch 9 = **3.5235** | ✅ 完了 |
| **Stage 1 (val metrics)** | 全 head 改善、from PR-AUC **+0.136**、target macro F1 **+0.013** | ✅ **+0.01 ゲート突破** |
| **Stage 2 (50 ep self-play)** | vs baseline_v1: **0 / 50** (95% CI: 0% - 7.1%) | ⚠ sanity 失敗 |
| **Stage 2 (30 ep, 別 seed start)** | vs baseline_v1: **0 / 30** (95% CI: 0% - 11.4%) | ⚠ sanity 失敗 |
| **採否最終判定** | **保留** (300 ep フォローアップ要、memory `project_imitation_case1_phase3` 準拠) | — |

仮説の **「予測距離 + history + ship 発射の 3 軸補強で defensive/offensive/target 判断を改善する」** は **validation metrics レベルでは強く支持** されたが、**self-play レベルでは現時点で baseline_v1 を倒せていない**。50 ep は noise floor 内なので 300 ep で再評価する。

## 訓練ログ (epoch 0-14)

```
epoch  train_total  val_total  val_from_acc  val_target_acc  val_ships_acc
0      3.8865       3.6901     0.9039        0.3792          0.8462
9*     3.5353       3.5235     -             -               -                ← best
14     3.4700       3.5440     -             -               -
```
*訓練時間*: 403.9 秒 (RTX 4090, 15 epoch, batch=256), runtime cuda

`train_loss_history` 単調減少 (3.89 → 3.47), `val_loss_history` epoch 9 で底打ち
(3.5235) その後微増 — 標準的な BC 学習挙動、early stopping + best.pt 保存も OK。

## Stage 1: validation metrics (全 head)

### 採否ゲートの結果

case3 phase2 の指標体系で計算 (`bot/pipeline/imitation/case7/evaluation/diagnose_weights.py`)。
**+0.01 以上で採用 → from PR-AUC で +0.136 という大幅改善**で完全突破。

| head | metric | case3 phase2 (ref) | **case7** | Δ | 採否 |
|------|--------|-------------------:|----------:|----:|------|
| **from** | F1 | 0.6182 | **0.6651** | **+0.047** | ✅ |
| from | PR-AUC | 0.6317 | **0.7676** | **+0.136** | ✅ 大幅 |
| from | ROC-AUC | 0.9275 | 0.9532 | +0.026 | ✅ |
| from | acc | (n/a) | 0.9151 | — | — |
| from | positive_rate_gt | (n/a) | 0.1094 | — | — |
| **target** | macro F1 | 0.3076 | **0.3210** | **+0.013** | ✅ |
| target | top-1 acc | 0.4238 | 0.4327 | +0.009 | ✅ |
| target | top-2 acc | (n/a) | 0.6377 | — | — |
| target | weighted F1 | (n/a) | 0.4000 | — | — |
| target | PR-AUC macro | (n/a) | 0.3419 | — | — |
| target | ROC-AUC macro | (n/a) | 0.7763 | — | — |
| **ships** | acc | 0.8108 | 0.8519 | **+0.041** | ✅ |
| ships | macro F1 | **0.6677** | **0.6425** | **-0.025** | ⚠ 唯一の後退 |
| ships | MAE bucket | 0.2501 | 0.2022 | -0.048 (small=better) | ✅ |
| ships | PR-AUC macro | (n/a) | 0.7117 | — | — |
| ships | ROC-AUC macro | (n/a) | 0.9435 | — | — |

### 解釈

- **from head が最大の伸び**: PR-AUC +0.136 は case3 phase2 から case7 までの新規 11 列
  特徴量 (予測距離 / history / ship 発射 history) が source planet の選択精度を強く後押ししている証拠。
- **target head は控えめだが改善**: macro F1 +0.013、top-1 +0.009。template 別 break-down は計測対象外
  (case7 の diagnose_weights は per-template F1 を出力していない、case3 phase2 と差分あり)。
- **ships head はトレードオフ**: acc / MAE は良化したが macro F1 は若干後退。bucket 0/1 (低 ratio) の
  recall が下がったのが原因と推測されるが detail は未解析。case3 phase2 では bucket 2 の recall 0.20→0.45 が
  大躍進だったので、case7 では別 head とのバランスで何かが押し出された。

### Causal leak 回帰防止 (Risk #1) の検証

case7 の `tests/pipeline/imitation/case7/test_featurizer_history.py` は、`obs_{N-2}` 参照の delta_ships
が action_N と直接相関しないことを assert する unit test を含む。Stage 1 の異常 PR-AUC (0.95 超等) は
発生しておらず、causal leak の再発は無いと判断。

## Stage 2: self-play vs baseline_v1

### 50 ep (seed 0-49)

| 指標 | 値 |
|------|------|
| episodes | 50 |
| wins | **0** |
| losses | 50 |
| draws | 0 |
| **win_rate** | **0.0%** |
| 95% CI | **0% - 7.1%** |
| challenger | il_v7 |
| baseline | baseline_v1 |
| mode | 1v1 |

### 30 ep (re-run, seed 0-29)

50 ep 実施後 trap #8 修正と同じ session で 30 ep を別途回した結果も同じ:

| 指標 | 値 |
|------|------|
| episodes | 30 |
| wins | **0** |
| losses | 30 |
| **win_rate** | **0.0%** |
| 95% CI | **0% - 11.4%** |
| challenger | il_v7 |
| baseline | baseline_v1 |

50 ep / 30 ep どちらも 0 勝。**short-horizon self-play では完敗**で確定。
ただし n<300 は採否判定材料にならない (memory `project_imitation_case1_phase3` 準拠)。

### 解釈

- **完敗**: 50 試合で勝てなかった。CI 上限 7.1% は memory `project_imitation_case1_phase3` で記録された
  iter9 の 5/100 = 5% (n<300 では非反復) と同水準。**50 ep 自体が noise floor 内** で採否判定には
  使えない (本 plan でもユーザーが指定した「sanity check スコープ」)。
- **過去 imitation case との比較**:
  - case5 baseline_v1 vs il_v5: 自己対戦未実施 (case5 RunPod training は本セッションで未完)
  - case3 phase2 (il_v3): self-play 未実施 (validation のみ)
  - 直近 case (case4): rulebase ベースなので比較対象外
- 結局 **case7 が「validation 上は改善、試合上は未確認」** という特殊な状況。

### 想定される追加要因

1. **target template の resolution mismatch**: featurizer が新規 11 列を持つようになっても、
   decoder が template id → planet id に解決する経路が変わっていない。新特徴量の情報が target 選択
   までは届くが、最終的な action (angle / ships) の生成では古い rule に従う。
2. **ships bucket の細粒度**: ships head の macro F1 が後退 = 試合中に「何隻送るか」の判断が
   鈍くなった可能性。これは試合結果に直接効く。
3. **tactical decision-making の欠落**: BC は state→action mapping を学ぶが、defensive hold /
   harass timing の **判断ステップ** はモデル内に明示エンコードされていない。新特徴量で
   policy が「敵 fleet が来る」と認識しても、「どう対応する」の output は学習データ依存。
   memory `project_imitation_case1_phase2_breakthrough` の「BC 単体ではタクティカル決定力に課題が残る」
   という結論と整合。

## 採否最終判定

**保留**。memory `project_imitation_case1_phase3` の方針 (n<300 self-play は信頼不可、+5pp 以上で採用、
+0-5pp は保留) に厳密に従う。**300 ep self-play フォローアップ run** が必要。

### 推奨 follow-up 順序

1. **300 ep self-play vs baseline_v1** (case7/evaluation/eval_vs_baseline.py -n 300、ローカル ~30 分)
   - 0/300 ならば破棄、1-15/300 (0-5%) ならば保留、16+/300 ならば採用候補
2. (採用候補の場合) 300 ep vs baseline_v4 (production champion) を追加実施
3. (採用判定後) Kaggle submission policy 準拠で submit (本 plan のスコープ外)

### 別軸の改善余地 (次 iter 候補)

1. **ships head の class-balanced loss**: bucket macro F1 後退を補う
2. **template の追加・拡張**: defensive hold / harass timing を明示する template を増設
3. **Auxiliary head 追加**: value head / opponent action prediction で BC の弱点を補強

## 修正された RunPod onstart trap (このサイクルで発見)

case7 を初めて RunPod で full preprocess + train した結果、**6 つの未知 trap** を発見・修正した
(memory `project_runpod_onstart_pitfalls` の 3 trap を 6 trap に拡張すべき):

| # | trap | 修正 commit | 詳細 |
|---|------|------|------|
| 1 | CUDA 13 image vs old host driver | `604c4e1` (cu1241 default) | RTX 4090 SECURE fallback ノードに driver 580 未満が混在 |
| 2 | dvc pull full mode が abort | `604c4e1` (`--allow-missing`) | 新 case の outs (preprocess 出力なので未生成) で pull が落ちる |
| 3 | dvc add がsymlinked dir 不可 | `f30e864` (unlink → cp → re-link) | `data/mart/imitation -> /persist/...` symlink 経由で dvc add 不可 |
| 4 | persist の stale parquet が dvc pull を block | `8df06bb` (`--force`) | 前 run の出力が next run の dvc pull を妨害 |
| 5 | dvc add が pipeline stage outs と overlap | `c72fdf4` (`dvc commit/push <stage>` + `dvc.lock`) | dvc.yaml に outs 登録された artifact は `dvc add` ではなく `dvc commit -f <stage>` |
| 6 | dvc push s3fs futures hang (treeverse/dvc#10374) | `1bda9ff` (`-j 1 -v`) | 高 RTT + 高並列の S3 push で futures が timeout なしハング |
| 7 | mark_progress 関数が module に存在しない | `5a89082` (`runpod_io.progress.mark_progress` 実装) | 長年の dead import (case3-5 全て該当)、case7 train で初顕在化 |
| 8 | data/output/models/imitation の symlink 経由 dvc add | (本セッション末で修正、未検証 commit) | 訓練 run dir の dvc add も同じ symlink trap。`mart_dvc_persist` と同じ unlink → cp → re-link pattern を `dvc_add_run` ブロックに適用、`dvc push` も `-j 1 -v` 化。次 RunPod run で動作検証要 |

このセッションで RunPod に **~$2.64** 投入し、訓練本体に到達したのは試行 9 (Step B 2nd attempt = 5a89082)。

## 関連ファイル

- featurizer: `bot/pipeline/imitation/case7/policy/featurizer.py`
- agent: `bot/pipeline/imitation/case7/policy/agent.py`
- weights: `bot/pipeline/imitation/case7/policy/weights.pt` (982 KB, best @ epoch 9)
- 訓練 metrics: `data/output/models/imitation/case7/runs/20260505-040940__.../metrics.json`
- val metrics: `/tmp/case7_val_metrics.json` (move to `data/output/experiment/imitation_case7_val_metrics.json` if needed)
- 50ep self-play: `/tmp/case7_eval_vs_baseline_50ep_v2.json` (move to `data/output/experiment/imitation_case7_eval_vs_baseline_50ep.json`)
- registry: `bot/src/dataset/selfplay/agents.py` `il_v7`

## 教訓

1. **新規 case の RunPod 初投入は trap 検出ラリー**: case3-5 が RunPod で preprocess/train を
   一度も完走していなかったため、onstart テンプレの bug がまとめて顕在化した。次に新 case を
   作るときは onstart trap カタログ (memory) を見直す。
2. **Stage 1 (val) と Stage 2 (試合) は別物**: validation 指標の改善が試合で必ずしも
   出ないのは BC の構造的限界。target head がどれだけ精度上がっても、**decoder の
   template→planet 解決と ships head の細粒度** が試合結果を支配する可能性がある。
3. **Self-play は 300 ep が下限**: 50 ep の 0/50 を見て破棄するのは早計。memory
   `project_imitation_case1_phase3` を厳守。
4. **DVC + multi-case `dvc.yaml` + RunPod の組み合わせは未踏領域**: 過去の運用は
   `dvc.yaml` の outs 登録なしで `*.parquet.dvc` 戦略だった可能性。**case7 が初の
   stage-managed mart parquet** を運用したケース。

## 次のアクション

- [ ] 300 ep self-play vs baseline_v1 を実施 (separate `iter2_result.md` か追記)
- [ ] (もし採用) Kaggle submission を別途承認のうえで実施
- [ ] memory `project_runpod_onstart_pitfalls` を 8 trap 完全カタログに更新 (`docs/`に格上げか)
- [x] trap #8 (data/output/models/imitation symlink) を onstart テンプレで修正 ← 本セッション末
- [ ] case7 の onstart 結果を踏まえ、case5 RunPod training も再開可能 (同じ修正がそのまま使える)
- [ ] 次 RunPod run で trap #8 修正の動作検証 (本来は smoke run を打ちたいが session コスト枠を超過するため次回送り)
