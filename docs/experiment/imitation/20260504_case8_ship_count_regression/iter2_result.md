# imitation/case8 — iter2 結果: cosine LR + warmup + early stop

> 作成日: 2026-05-05
> plan.md: [`./iter2_plan.md`](./iter2_plan.md)
> 関連: [`iter1_plan.md`](./iter1_plan.md), [`iter1_result.md`](./iter1_result.md)
> 結論: **scheduler / early stop 自体は機能、best.pt 選定基準を val_cand_fire_acc max に変更も、vs baseline_v1 50 戦は 0/50 = 0%。iter1 (1/50 = 2.0%) より悪化。仮説 (cand head 振動抑制) は win rate に翻訳されず**

## 1. 学習ジョブ統計

| 項目 | 値 |
|---|---|
| run_id | `20260505-050417__feature-candidate_k-with-ship-prediction__b77aa8e__seed0` |
| commit SHA | `b77aa8e` (epochs=30, cosine warmup 2ep, early stop val_cand_fire_acc patience=5) |
| RunPod cloud-type | SECURE |
| GPU | NVIDIA GeForce RTX 4090 (24GB) |
| pod_id | `6mawxeczb72ng4` |
| early stop | epoch 15 で発動 (best=epoch 10、その後 5 epoch 改善なし) |
| wall-time (16 epoch run) | 約 9 分 (epoch 平均 ~18s × 16) |
| 実コスト | $0.69/h × 約 0.8h ≒ $0.55 (container 起動 16 min が支配的) |

## 2. 学習曲線サマリ (`train.log` 抜粋、early stop で epoch 0-15 の 16 周のみ)

| epoch | lr | train_total | val_cand_acc | val_noop_acc | val_fire_acc | val_ship_mae |
|---:|---:|---:|---:|---:|---:|---:|
| 0 (warmup) | 1.0e-4 | 319M | 0.346 | 0.364 | 0.158 | 19,729 |
| 1 (warmup) | 5.5e-4 | 16.0M | 0.223 | 0.235 | 0.098 | 90,029 |
| 2 (cosine peak) | 1.0e-3 | 8.1M | 0.017 | 0.005 | 0.203 | 123,716 |
| 5 | 9.6e-4 | 3.8M | 0.014 | 0.001 | 0.213 | 8,184 |
| 9 | 8.4e-4 | 2.4M | 0.017 | 0.004 | 0.215 | 30 |
| **10 (best)** | 8.1e-4 | 3.0M | 0.027 | 0.015 | **0.227** | 30 |
| 13 | 7.0e-4 | 2.3M | 0.314 | 0.334 | 0.059 | 30 |
| 15 (early stop) | 6.3e-4 | 1.7M | 0.013 | 0.000 | 0.210 | 40 |

- **early stop 発動**: epoch 10 がbest 後、5 epoch 改善なしで break (patience 通り) ✅
- **ship_loss 安定収束**: 31,193 → 19.4 (epoch 14)、val_ship_mae 30 で iter1 と同等 ✅
- **best epoch 10**: val_cand_fire_acc = 0.227 (iter1 best epoch 8 = 0.223、+0.4%) — scheduler 効果は微小
- **val_cand_acc 振動は iter1 と同パターン**: 0.346 (ep0) → 0.014 (ep5) → 0.314 (ep13) → 0.013 (ep14) で逆相関 oscillation 維持
- cosine annealing は lr を 1e-3 → 6.3e-4 (ep15) まで滑らかに減衰させたが、cand head 振動は scheduler では抑制不可

## 3. ローカル評価結果

`pipeline/imitation/case8/policy/weights.pt` に best.pt (epoch 10) を copy 後、1v1 評価 50 戦実施。

### 3.1 vs baseline_v1 (rulebase/case1)

```
episodes: 50
wins:      0
losses:   50
draws:     0
win_rate:  0.0%   (95% Wilson CI: 0.0 – 7.1%)
challenger: il_v8
baseline:   baseline_v1
mode:       1v1
seed_start: 0, seed_end_exclusive: 50
label:      case8_iter2_cosine_warmup_earlystop
```

→ **iter1 (1/50 = 2.0%) より悪化** (95% CI のみで判定すれば iter1=0.35-10.5% / iter2=0.0-7.1% で実質有意差なし、両者 noise 範囲内)

## 4. 失敗・気付き

### 4.1 scheduler / early stop 自体は機能、性能には繋がらず

iter2 plan の仮説は「cand head の振動を scheduler で抑え、best.pt を val_cand_fire_acc max で選べば fire 寄り predictor が選ばれて勝率上昇」。実際:

- ✅ scheduler は意図通り動作 (lr 1e-4 warmup → 1e-3 → 6.3e-4 cosine decay)
- ✅ early stop 動作 (epoch 15 で break)
- ✅ best metric 切り替え動作 (val_cand_fire_acc max で epoch 10 選定)
- ✅ best 値は iter1 より僅かに高い (0.227 vs 0.223)
- ❌ **win_rate は 2.0% → 0.0%** に下がった

**val_cand_fire_acc と win_rate の相関が崩れている** = val metric を最大化しても実 game で勝てない。

### 4.2 cand head 振動は scheduler 単独では消えない

epoch 0 (warmup): val_cand_acc=0.346
epoch 5: 0.014
epoch 13: 0.314
epoch 14: 0.013

cosine decay でも val_cand_acc が予想外に **激しく振動** しており、no-op 寄り epoch (0.3+) と fire 寄り epoch (0.01) が交互に出る。これは **lr の問題ではなく cand head の構造的問題** (label distribution の偏り = noop が圧倒的多数) と推察。

### 4.3 ship_head は引き続き機能

- val_ship_mae 30 で安定 (iter1 23 と同等)
- ship_loss は cosine 後半 (epoch 8+) で 19 で完全収束
- ship_head は cand head 振動の影響を受けず独立して学習

### 4.4 onstart 改善は引き続き機能

- ✅ `dvc pull --allow-missing --force` (orphan blob を skip)
- ✅ mart symlink materialize (`55_mart_dvc_persisted` が初めて発火)
- ✅ output/models symlink materialize (本 iter で `75_dvc_add_run_failed` が再発しなかったことで検証)

container 起動が 16 分と異常に遅かった (通常 80s) のは RunPod 側の image cache 問題と推察、本 PR とは無関係。

## 5. 結論と次 iter への申し送り

**iter2 結論: scheduler の機構は成功、しかし狙った win_rate uplift は得られず。cand head の構造的問題が支配的で、iter1 → iter2 で 1pp の振動範囲内 (0%→2%、両者ノイズ)。**

| 指標 | iter1 | iter2 | 評価 |
|---|---|---|---|
| best epoch | 8 (val_total min) | 10 (val_cand_fire_acc max) | iter2 設計通り |
| val_cand_fire_acc (best) | 0.223 | 0.227 (+0.4%) | わずか改善 |
| val_cand_acc 振動 | 0.01-0.37 | 0.01-0.35 | 抑制できず |
| ship_mae (best) | 32 | 30 | 同等 (両 iter とも機能) |
| **vs baseline_v1 50 戦** | **2.0% (1/50)** | **0.0% (0/50)** | **悪化 (CI 重複、ノイズ範囲内)** |

### 推奨される次の iter (案)

1. **iter3: cand head の構造的修正** — scheduler では抑えきれなかった振動を解消
   - **focal loss** 導入: cand_loss を `FocalLoss(alpha=0.25, gamma=2.0)` に変更し、label distribution の偏りに頑健化
   - もしくは **class_weight の inverse-frequency をさらに強める** (iter1/iter2 共通で「present class 平均が 1.0」だが、効きが薄い → α=0.5 で fire 重視を強制)
   - data 側の改善: training data の sub-sampling で **fire vs noop frame ratio を 1:1 に強制** (iter1/iter2 は noop 圧倒的多数)

2. **iter4: ship_head の loss weight sweep** (ship_w = 0.5 / 1.0 / 2.0)
   - 本 iter では joint loss `cand + ship` の比率を 1:1 にしているが、ship_w を上げると cand 学習を犠牲に ship 精度が上がる、下げると逆。スイートスポット探索

3. **iter5: 300 戦評価** で scheduler 試行の有意差判定
   - iter1 と iter2 の 50 戦差は CI 重複でノイズ範囲、300 戦回せば真の差が見える可能性

### 短期的方針 (iter1 / iter2 にこれ以上 RunPod を投じない)

- **本ブランチ (`feature/candidate_k-with-ship-prediction`) は iter2 で頭打ち**。iter3 (cand head 構造) 以降は別ブランチ / 別 PR で進めるのが clean
- 本 PR としては iter1 + iter2 + onstart 6 fix を merge 候補にできる (case8 が **完走可能** なことは検証済み、機能要件は満たす)

## 6. 参考

- `data/output/models/imitation/case8/runs/20260505-050417__feature-candidate_k-with-ship-prediction__b77aa8e__seed0/{run.json, train.log, gpu.log, onstart.log, best.pt}`
- `bot/pipeline/imitation/case8/policy/weights.pt` (= best.pt copy、評価で使用)
- `bot/pipeline/imitation/case8/evaluation/results.json` (50 戦 win_rate=0.0)
- 修正 commit chain (iter1+iter2): `6397c82` (case8 init) → `6108571` (parallel preprocess) → `1b93ecb` (--allow-missing) → `fd4b639` (mart materialize) → `384ac3b` (cu1241) → `1a94cd0` (--force) → `e874df9` (dvc.yaml stage 削除) → `741c77e` (output/models materialize) → `f16b561` (iter2 scheduler) → `04164ed` (debug stamp) → `16e33c0` (sanity epochs=2) → `b77aa8e` (iter2 full)
- memory: `project_imitation_case1_phase3` (n<300 評価不可), `project_runpod_onstart_pitfalls`
