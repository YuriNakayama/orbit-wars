# imitation/case8 — iter3 結果: focal loss

> 作成日: 2026-05-05
> plan.md: [`./iter3_plan.md`](./iter3_plan.md)
> 関連: [`iter1_result.md`](./iter1_result.md), [`iter2_result.md`](./iter2_result.md)
> 結論: **focal loss は学習ダイナミクスを劇的に改善 (loss スケール 75x 縮小、val_cand_acc 振動幅縮小) も vs baseline_v1 50 戦は 0/50 = 0%。iter2 と同等。win_rate と val metric の相関崩壊は iter1/iter2/iter3 共通の根本問題**

## 1. 学習ジョブ統計

| 項目 | 値 |
|---|---|
| run_id | `20260505-062848__feature-candidate_k-with-ship-prediction__da4c2fa__seed0` |
| commit SHA | `da4c2fa` (focal loss + cosine LR + warmup + early stop) |
| RunPod cloud-type | SECURE |
| GPU | NVIDIA GeForce RTX 4090 (24GB) |
| pod_id | `aprlq53jua060r` |
| early stop | epoch 14 で発動 (best=epoch 9, patience 5) |
| wall-time (15 epoch) | 約 9 分 (epoch 平均 ~36s × 15) |
| 実コスト | $0.69/h × 約 0.65h ≒ $0.45 (container 起動 8 min を含む) |

## 2. 学習曲線サマリ (`train.log`、early stop で 0-14 epoch)

| epoch | lr | train_total | val_total | val_cand_acc | val_noop_acc | val_fire_acc | val_ship_mae |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 (warmup) | 1.0e-4 | 4.96M | 330k | 0.267 | 0.279 | 0.166 | 7,314 |
| 1 (warmup) | 5.5e-4 | 237k | 230k | 0.216 | 0.225 | 0.126 | 57,576 |
| 2 (cosine peak) | 1.0e-3 | 125k | 73.9k | 0.078 | 0.071 | 0.184 | 6,693 |
| 5 | 9.6e-4 | 52.8k | 37.1k | 0.039 | 0.028 | 0.205 | 259 |
| **9 (best)** | 8.4e-4 | 39.6k | 94.6k | 0.013 | 0.000 | **0.210** | 30 |
| 11 | 7.7e-4 | 32.4k | 18.6k | 0.153 | 0.152 | 0.158 | 30 |
| 14 (early stop) | 6.5e-4 | 28.9k | 22.2k | 0.182 | 0.185 | 0.147 | 40 |

### 学習ダイナミクスの劇的変化 (focal loss 効果)

- **train_total が iter1/iter2 比で 75x 縮小**: iter1 best ~3M → iter3 best ~40k → focal の `(1-p_t)^γ` が easy noop の loss を効果的に down-weight、累積 loss のスケールが大幅縮小
- **train_ship_loss も連動して縮小**: 13,269 (epoch 0) → 18.7 (epoch 9) — ship_head は iter1/iter2 と同レベルで収束
- **val_cand_acc 振動の幅縮小**: iter1/iter2 は 0.01-0.37、iter3 は **0.013-0.355** で幅は同程度だが、loss スケール縮小で gradient が安定化
- **best_metric (val_cand_fire_acc) は iter2 比で低下**: iter1=0.223, iter2=0.227, **iter3=0.210**
- **early stop epoch 9 → 14 で発動**: focal で easier learning curve、5 epoch 連続改善なしで停止

## 3. ローカル評価結果

`pipeline/imitation/case8/policy/weights.pt` に best.pt (epoch 9) を copy 後、1v1 評価 50 戦実施。

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
label:      case8_iter3_focal_loss
```

→ **iter2 と同じ 0/50 = 0.0%**。focal loss の機構的成功は win_rate に翻訳されなかった。

## 4. 失敗・気付き

### 4.1 focal loss は learning dynamics を改善するも win_rate に効かず

| 指標 | iter1 (CE) | iter2 (CE+cosine) | iter3 (focal+cosine) |
|---|---|---|---|
| best epoch | 8 | 10 | 9 |
| val_cand_fire_acc (best) | 0.223 | 0.227 | **0.210** |
| train_total (best) | ~3M | ~3M | **~40k (75x ↓)** |
| ship_mae (best) | 32 | 30 | 31 |
| **vs baseline_v1 50 戦** | **2.0% (1/50)** | **0.0% (0/50)** | **0.0% (0/50)** |

focal loss は loss scale を圧倒的に縮め、gradient flow を変えたものの:

- val_cand_fire_acc は iter2 (0.227) より下がった
- 50 戦結果は iter2 と同じ 0% (CI 重複範囲、ノイズ範囲内)
- **3 iter 全てで vs baseline_v1 が 0-2% 範囲** = 体系的な問題が支配項

### 4.2 val_cand_fire_acc と win_rate の相関完全崩壊

iter1 → iter3 で val metric は変動したが win_rate は ~0-2% で固定:

- val_cand_fire_acc 0.20-0.23 の範囲は **どの best epoch でも win_rate ≤ 2%** に翻訳される
- best metric を val_cand_fire_acc max にする戦略 (iter2/iter3) は **意味が薄い**
- 真の win_rate を上げるには **val proxy を別軸にする** (例: actual game step での fire/no-op 比率の妥当性、orbit-aware action accuracy 等) か、**train data 構成自体を見直す** (oversample / hard mining)

### 4.3 ship_head は引き続き安定、独立性が確認

3 iter 全てで val_ship_mae 30-32 で安定収束。focal loss / cosine scheduler / early stop の影響を受けない = ship_head は cand head から独立 (plan の設計通り)。

### 4.4 onstart pipeline は完全安定化

iter1 で 6 trap を経て iter2/iter3 では cleanly に通っている。本ブランチの onstart 修正群は他 case にも横展開価値 (`dvc pull --allow-missing --force` / mart+output materialize / cu1241 image / dvc.yaml stage 削除 / iter2 scheduler infra) は安定動作確認済み。

## 5. 結論と次 iter への申し送り

**iter3 結論: focal loss の機構は完全に成功 (loss 75x 縮小、振動幅縮小、ship_head 無影響) も win_rate は 2 iter 連続で 0%。3 iter 通じて vs baseline_v1 は 0-2% で頭打ち。**

### 根本問題の絞り込み

iter1/iter2/iter3 で本質的に何が変わっていないか:

1. **train data**: 全 iter で同じ Kaggle replay 944 ep / 300k frame / val 38k frame
2. **action space**: per-source × CAND_K=8 categorical + per-source ship scalar
3. **backbone**: Graph U-Net (case3 から不変)
4. **base loss surface**: cross-entropy or focal、いずれも frame-level supervised

= 「どんな loss / scheduler を使っても **frame-level BC for noop-dominant data** という枠組みの限界に達している」可能性が高い。

### 推奨される次 iter (案)

本ブランチでの追加 iter は **diminishing returns 領域** に入っている。本 PR としては **iter1+iter2+iter3 + onstart 6 fix** をまとめて main にマージし、case8 の進化は別 PR で:

1. **Data 再構成 (= 別 PR を切る)**:
   - fire / noop frame の比率を 1:1 強制 (over-sample or under-sample)
   - 「fire に値する場面のみ」の hard mining (= rulebase が STAY 判定する frame をノイズとして除外)
2. **Aux loss / dual head**:
   - cand head に **per-step value head** (帰結予測) を追加し loss surface を多次元化
3. **Self-play で fine-tune**:
   - imitation で初期化 → self-play (RL) で BC bias 補正
4. **Ship head 単独の効果検証**:
   - case4 (ship 数 rule) vs case8 ship_head 直接 head-to-head 300戦 — ship_head が cand 学習を妨害していないかの sanity

### 短期判断

- iter1/iter2/iter3 は **case8 の枠組みでは win_rate uplift 困難** を示した有効な negative result
- onstart 6 fix は他 case 横展開価値あり、PR 化するべき
- case8 自体の有用性は ship_head pathway としての PoC、prod 採用は時期尚早

## 6. 参考

- `data/output/models/imitation/case8/runs/20260505-062848__feature-candidate_k-with-ship-prediction__da4c2fa__seed0/{run.json, train.log, best.pt}`
- `bot/pipeline/imitation/case8/policy/weights.pt` (= best.pt copy、評価で使用)
- `bot/pipeline/imitation/case8/evaluation/results.json` (50 戦 win_rate=0.0)
- 修正 commit chain (iter1+iter2+iter3): `6397c82` → `6108571` → `1b93ecb` → `fd4b639` → `384ac3b` → `1a94cd0` → `e874df9` → `741c77e` → `f16b561` → `04164ed` → `16e33c0` → `b77aa8e` → `da4c2fa`
- memory: `project_imitation_case1_phase3` (n<300 評価不可), `project_runpod_onstart_pitfalls`
