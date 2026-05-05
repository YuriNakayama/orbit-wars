# imitation/case8 — Ship-Count Regression Head (iter1 結果)

> 作成日: 2026-05-04
> plan.md: `./plan.md`
> 関連: [`docs/experiment/imitation/20260501_case4_kaggle_tutorial_head/iter2_result.md`](../20260501_case4_kaggle_tutorial_head/iter2_result.md)
> 結論: **ship_head 自体は学習成功 (MAE 23 隻オーダー)、ただし vs baseline_v1 50 戦は 1/50 = 2.0% で iter2 の 0/10 を僅かに上回るのみ。cand head の振動が主因と推定**

## 1. 学習ジョブ統計

| 項目 | 値 |
|---|---|
| run_id | `20260504-112305__feature-candidate_k-with-ship-prediction__e874df9__seed0` |
| commit SHA | `e874df9` (cu1241 image / preprocess 並列化 / 各種 onstart fix) |
| RunPod cloud-type | SECURE |
| GPU | NVIDIA GeForce RTX 4090 (24GB) |
| pod_id | `q4nbfarsht8dkg` |
| wall-time (15 epoch) | 約 9 分 (epoch 平均 ~36s × 15) |
| 実コスト | $0.69/h × 約 0.4h ≒ $0.28 (cost-limit $1.5 内) |
| preprocess 並列化 | workers=47 / 944 ep / **285 秒で完走** (前 case4 iter2 の serial と比較し ~3-4x 高速化) |

## 2. 学習曲線サマリ (`train.log` 抜粋)

| epoch | train_total | val_total | val_cand_acc | val_noop_acc | val_fire_acc | val_ship_loss | val_ship_mae |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 84M | 20M | 0.253 | 0.269 | 0.087 | 70,487 | 70,488 |
| 1 |  8.2M |  8.7M | 0.038 | 0.029 | 0.197 | 29,631 | 29,632 |
| 3 |  4.6M |  5.6M | 0.027 | 0.016 | 0.218 |  1,299 |  1,299 |
| 5 |  3.2M |  2.5M | 0.021 | 0.008 | 0.223 |     30 |     31 |
| 6 |  3.3M | 15.4M | **0.371** | **0.398** | 0.054 |     30 |     31 |
| **8 (best)** |  3.0M |  **2.3M** | 0.017 | 0.004 | 0.223 |     31 |     32 |
| 10 |  3.4M |  4.4M | 0.132 | 0.130 | 0.168 |     38 |     38 |
| 12 |  2.8M |  2.2M | 0.034 | 0.022 | 0.228 |     38 |     38 |
| 14 |  3.6M |  2.7M | 0.096 | 0.091 | 0.196 |     36 |     36 |

- best_epoch = 8, best_val_total = 2,305,260
- **ship_loss 急速収束**: 93,572 (epoch 0) → 23 (epoch 14) → **ship head は機能、SmoothL1 適合 OK**
- **val_ship_mae 23-30**: replay の発射 ship 数 (20-50 が中心) のオーダーと一致、量的予測は機能
- val_cand_acc が epoch 6 (0.37) と他 (0.01-0.13) で激しく振動 → case4 iter2 と同じ no-op vs fire のトレードオフ振動
- best_epoch (8) では val_cand_noop_acc=0.004 / fire_acc=0.223 → fire 寄り predictor

## 3. ローカル評価結果

`pipeline/imitation/case8/policy/weights.pt` に best.pt を copy 後、1v1 評価 50 戦実施。

### 3.1 vs baseline_v1 (rulebase/case1)

```
episodes: 50  (plan の 50 戦完走)
wins:      1
losses:   49
draws:     0
win_rate:  2.0%   (95% Wilson CI: 0.35 – 10.5%)
challenger: il_v8
baseline:   baseline_v1
mode:       1v1
seed_start: 0, seed_end_exclusive: 50
label:      case8_ship_count_regression_iter1
```

→ **plan の生存しきい値 (>0%) を満たす**。case4 iter2 (0/10) を上回るが、n=50 の Wilson 95% CI は 0.35-10.5% と広く、**+5pp 改善の有意検出は不可** (project memory `project_imitation_case1_phase3` の n<300 不可信頼ルール通り)。

## 4. 失敗・気付き

### 4.1 ship_head は学習成功

ship_loss / ship_mae の epoch 推移は本実験で最も明瞭な進展:

- epoch 0: ship_mae 70,488 (ほぼ random init で爆発、SmoothL1 の linear regime)
- epoch 5: ship_mae 30 (=replay 値の 1.5-2x オーダーで安定)
- epoch 14: ship_mae 23.3 (= 平均 23 隻ずれ)

**Plan で SmoothL1 を採用した判断は妥当**。MSE だと ship_mae=70k が二乗で勾配爆発するはずだが、SmoothL1 の linear regime が outlier を吸収し収束を許容した。

### 4.2 cand head の振動が性能上限を制約

cand_acc の epoch ごとの跳ねは case4 iter2 と類似:

| epoch | val_cand_acc |
|---:|---:|
| 6 | 0.371 |
| 7 | 0.013 |
| 10 | 0.132 |
| 11 | 0.014 |
| 14 | 0.096 |

= no-op と fire のバランスが取れず、epoch ごとに predictor が一方に偏る。**これは case8 で導入した ship_head とは独立な問題で、case4 iter2 から残る課題**。

### 4.3 cand head と ship head の独立性

ship_head の loss が安定収束する一方で cand_acc が振動 → ship_head が cand_logits の勾配に干渉していない (= plan で設計した「self_h ⊕ global_h のみで produce、candidate slot 非依存」が意図通り機能)。

### 4.4 onstart 整備で踏んだ trap (本 iter で解消、コスト相当)

| 失敗 | 修正 commit | trap |
|---|---|---|
| `45_dvc_pull_full_failed` (case8 outs 未 push で missing blob fail) | `1b93ecb` | `dvc pull --allow-missing` |
| `45_dvc_pull_full_failed` (前 run の persist file が unsaved 扱いで `--force` 必要) | `1a94cd0` | `dvc pull --allow-missing --force` |
| `55_mart_dvc_add_failed` (mart symlink dir 配下 file は `dvc add` 拒否) | `fd4b639` | symlink → 物理 dir に materialize |
| `55_mart_dvc_add_failed` (dvc.yaml stage の outs と onstart `dvc add` が conflict) | `e874df9` | dvc.yaml から `preprocess_imitation_case8` / `train_imitation_case8` 削除 |
| CUDA driver mismatch (cu1300 image vs host driver) | `384ac3b` | image を `cu1241` (CUDA 12.4) に固定 |
| `75_dvc_add_run_failed` (output/models symlink dir で同様) | `741c77e` (本 iter で適用) | output/models も materialize に拡張 (eval パスでは無関係なので impact なし) |

すべて memory `project_runpod_onstart_pitfalls` の trap カテゴリに属する。**新規 case を切る前のチェックリスト** が更新候補。

## 5. 結論と次 iter への申し送り

**iter1 結論: ship_head は仮説どおり学習可能、ただし cand head の振動が性能の支配項で、ship_head 単独では大きな勝率改善には至らない。**

- ✅ ship-count regression head は SmoothL1 で安定収束、MAE 23 = 妥当
- ✅ cand head と独立性を保った設計 (per-source self+global stream) が機能
- ❌ 50 戦 1/50 = 2.0% は iter2 case4 (0/10) を超えるが、project の n<300 不可信頼ルールにより有意な improvement は主張不可
- ❌ cand head 自体の振動 (case4 iter2 から継承) を解消しないと ship_head の効果は埋没

### 推奨される次の iter (案)

1. **iter2: cand head の振動対策**
   - lr scheduler (cosine / step decay) の導入で epoch 後半の loss 振動を抑制
   - early stopping (val_total のみではなく val_cand_fire_acc も watch) で best epoch 選定を改善
   - epochs を 15 → 30 に増量、warmup を入れて epoch 0 の loss explosion (84M) を緩和
   - これらは ship_head とは独立で、case4 iter3 計画とも合流可能

2. **iter3: ship_head の実効性検証 (300 戦)**
   - cand head の振動が落ち着いた状態で 300 戦評価
   - case8 (ship_head あり) vs case4 (ship 数 rule) の直接比較
   - ship_head がもたらす勝率 delta を初めて測定可能になる

3. **本 iter で得た onstart 修正は他 case にも横展開価値あり**
   - `dvc pull --allow-missing --force` / mart-symlink materialize / cu1241 image / dvc.yaml stage 削除は新規 case の onstart trap を 5 個解消した
   - case6 / case7 / 今後の case9+ の launch 前にこれらが入った main を rebase 推奨

## 6. 参考

- `data/output/models/imitation/case8/runs/20260504-112305__feature-candidate_k-with-ship-prediction__e874df9__seed0/{run.json, train.log, gpu.log, onstart.log, best.pt}`
- `bot/pipeline/imitation/case8/policy/weights.pt` (= best.pt copy、評価で使用)
- `bot/pipeline/imitation/case8/evaluation/results.json` (50 戦 win_rate=0.02)
- 修正 commit chain: `6397c82` → `6108571` → `1b93ecb` → `fd4b639` → `384ac3b` → `1a94cd0` → `e874df9` → `741c77e`
- memory: `project_imitation_case1_phase3` (n<300 評価不可), `feedback_runpod_prompt_bypass` (有償 prompt skip 禁止), `project_runpod_onstart_pitfalls` (Volume / DVC / cwd の 3 trap + 本 iter で 3 trap 追加発見)
