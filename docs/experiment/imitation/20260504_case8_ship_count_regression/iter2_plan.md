# imitation/case8 — iter2: cand head 振動対策 (lr scheduler / 30 epoch / early stop)

> 作成日: 2026-05-04
> 関連:
> - [`iter1_plan.md`](./iter1_plan.md) (ship_head 設計、論拠)
> - [`iter1_result.md`](./iter1_result.md) (ship_head ✅ 機能、cand head ❌ 振動 → vs baseline_v1 1/50 = 2.0%)
> - `bot/pipeline/imitation/case8/training/train.py:264-268` (現行 AdamW、lr 1e-3 固定、scheduler なし)
>
> スコープ: case8 内に lr scheduler + warmup + early stopping を追加。ship_head と features は **不変**。

## 仮説 (Hypothesis)

iter1 の cand head は val_cand_acc が epoch 4 (0.27) → 5 (0.02) → 6 (0.37) → 7 (0.01)
と**激しく振動** (no-op acc と fire acc が逆相関で上下)。これは:

1. lr=1e-3 固定で epoch 後半の loss landscape が安定しない
2. val_total が epoch 6 (15.4M) と epoch 8 (2.3M) で 6 倍以上揺れる
3. best epoch (8) が val_total 最小だが val_cand_acc=0.017 で実は **fire 偏重 predictor** = 不要発射が連発し勝率を引き下げ

**cosine annealing + warmup + epochs 15→30 + early stopping (val_cand_fire_acc を watch)** で
振動を抑え、より balanced な (noop / fire 両方学べた) checkpoint を選択できれば、
ship_head の効果が決定値に乗りやすくなる。**iter1 の 2.0% → 5%+ への uplift が期待される**。

メカニズム:
- **warmup (epoch 0-1, lr 0 → 1e-3)** が epoch 0 の loss explosion (84M → 8M) を緩和
- **cosine decay (lr 1e-3 → 1e-5)** が epoch 後半の振動を抑える (gradient step が小さくなる)
- **epochs 30** で best epoch が後半に移る余地が生まれる (iter1 best=8/15 だが scheduler 入れると best=20-25/30 想定)
- **early stop (val_cand_fire_acc が 5 epoch 連続改善なし)** で過学習回避 + 計算節約

## 既存コードの現状 (iter1 から継承)

- `policy/model.py`: ship_head + cand_head の 2-head 構成 — **不変**
- `training/losses.py`: SmoothL1 ship_loss + CE cand_loss、joint loss — **不変**
- `training/preprocess.py`: ProcessPoolExecutor 並列化 (workers=cpu-1) — **不変** (iter1 で 285s 完走確認済)
- `training/train.py`: optim.AdamW + scheduler 無し + epochs=15 + best=val_total → **本 iter で改修**
- `data/mart/imitation/case8/{train,val}.parquet`: iter1 run の preprocess 出力 (300,084 / 38,278 frames) を **再利用**可能 (DVC remote にあり) → preprocess 再実行不要、コスト ~50% 削減

## スコープ (Scope)

### 変更ファイル

| Path | 変更内容 |
|---|---|
| `bot/pipeline/imitation/case8/training/train.py` | (1) AdamW 後に `optim.lr_scheduler.CosineAnnealingLR` 追加。(2) warmup (epoch 0-1) を `LinearLR` で実装、`SequentialLR` で結合。(3) `epochs` を config 読みのままにして config で 15→30。(4) early stop: `val_cand_fire_acc` が `patience=5` epoch 連続改善なしで break |
| `bot/pipeline/imitation/case8/configs/il_case8.yaml` | `epochs: 15 → 30`、`scheduler:` セクション追加 (cosine_t_max=30, warmup_epochs=2, eta_min=1e-5)、`early_stop:` セクション追加 (`metric: val_cand_fire_acc, patience: 5, mode: max`) |
| `bot/tests/pipeline/imitation/case8/test_train_scheduler.py` | (新規) scheduler が config から正しく構築されるか / warmup の lr 推移 / early stop の発火条件 |

### 変更なし

- `policy/{model,types,featurizer,candidates,decoder,geometry,agent}.py` — head 構造・特徴量変更なし
- `training/{preprocess,losses,dataset}.py` — 並列化・loss 設計を維持
- DVC stage / AGENT_REGISTRY / CASE_DEFAULTS — case8 は引き続き登録なし (onstart 直呼び pathway)

### Hyperparameters

| Knob | iter1 | iter2 |
|---|---|---|
| epochs | 15 | **30** |
| optimizer | AdamW(lr=1e-3, wd=1e-4) | 同じ (初期値) |
| lr scheduler | なし | **CosineAnnealingLR (T_max=30, eta_min=1e-5) + LinearLR warmup (start_factor=0.1, total_iters=2)** |
| early stop | なし | **val_cand_fire_acc patience=5 (mode=max)** |
| best metric | val_total | **val_cand_fire_acc が最大の epoch** (val_total tie-break) ※ best.pt の選択基準を変更 |
| ship head / loss | 不変 (SmoothL1, λ=1.0) | 不変 |

## 実装ステップ (Implementation outline)

1. `bot/pipeline/imitation/case8/training/train.py`:
   - `train()` の冒頭で `train_cfg.get("scheduler", {})` を読む。`type: cosine_warmup` で warmup + cosine を `SequentialLR` で組む
   - epoch loop の末尾で `scheduler.step()` を呼ぶ
   - `train_cfg.get("early_stop", {})` を読み、`metric: val_cand_fire_acc / patience: 5 / mode: max` で best 追跡を val_total から `val_cand_fire_acc` (mode=max) に変更
   - `best.pt` 保存条件: 新 metric で更新時のみ save (val_total と decouple)
   - early stop は `patience` 連続改善なしで break (epochs を切り上げ)
   - history.jsonl の log 行に `lr` と `early_stop_counter` を追加
2. `bot/pipeline/imitation/case8/configs/il_case8.yaml`:
   - `epochs: 30`
   - `scheduler: {type: cosine_warmup, t_max: 30, eta_min: 1.0e-5, warmup_epochs: 2, warmup_start_factor: 0.1}`
   - `early_stop: {metric: val_cand_fire_acc, patience: 5, mode: max}`
3. `bot/tests/pipeline/imitation/case8/test_train_scheduler.py`:
   - smoke test: 5 epoch 縮小 config で `train()` を呼ぶ → scheduler の lr が warmup → cosine の通り推移すること
   - early stop test: `val_cand_fire_acc` が patience=2 で改善しなくなる stub data → early break すること
4. `dev/test-bot` で format / lint / type / pytest 全通過確認
5. RunPod launch (case8、commit + push) — preprocess 再実行不要
6. `dev/runpod pull --from s3` で best.pt 取得 → 50 戦評価 (vs baseline_v1)
7. **survival** (>iter1 の 2.0%) なら 300 戦に拡張、`iter2_result.md` 作成

## 検証方法 (Validation method)

- **ローカル**:
  - `dev/test-bot` (format / lint / type / pytest)
  - `uv run --directory bot pytest tests/pipeline/imitation/case8 -x`
- **リモート**:
  - `dev/runpod train <new_sha> --case case8`
  - 想定所要時間: uv sync 30s (persist hit) + dvc pull 4-5min (mart parquet を S3 から取得) + train 30 epoch ≒ 18 min (iter1 の 2x) + dvc add/push + git push ≒ **~25 min total**
  - 想定コスト: ~$0.30 (RTX 4090 SECURE $0.69/h × 0.4-0.5h)
- **評価**:
  - 対戦相手: `baseline_v1` (rulebase/case1)
  - エピソード数: **50 戦 (sanity)** → 勝率 >iter1 (2.0%) を満たせば **300 戦に拡張**
  - 主要メトリクス: vs baseline_v1 win_rate
  - 採否しきい値:
    - 50 戦で **>5%** (3/50 以上) → 300 戦昇格
    - 300 戦で **>7% (Wilson 95% CI lower bound > 5%)** → adopted、`dev/runpod promote` を user 確認後実行
    - 50 戦で <=2.0% → cand head 振動以外の問題、別アプローチ検討

## リスク / 注意点

1. **best.pt 選択基準変更の影響**: val_total → val_cand_fire_acc は per-iter で違う epoch を選ぶ可能性あり。iter1 の best (epoch 8、val_total 最小) は val_cand_fire_acc=0.223。iter2 で同じ checkpoint が選ばれれば結果は変わらない。**scheduler が cand head を安定化させて、後半 epoch でより高い fire_acc が出る** という前提が崩れると無効。
2. **scheduler の hyperparameter 感度**: warmup_epochs=2 は経験則。epoch 0 (lr=1e-4) → epoch 2 (lr=1e-3) → epoch 30 (lr=1e-5)。loss が epoch 0-1 で発散したら warmup_start_factor を 0.01 に下げて再起動 (= 別 iter)
3. **epochs=30 でも振動が消えない場合**: cand head 自体の構造的問題 (e.g. softmax temperature、class_weight saturation) が残り、scheduler は対症療法に留まる。次 iter (iter3) では label smoothing 強化 / focal loss / cand_score MLP 構造変更を検討
4. **DVC pull で前 iter の parquet が再利用される確認**: `dvc pull --allow-missing --force` 後、`/persist/data-mart-imitation/case8/{train,val}.parquet` が iter1 と同じ blob hash であること (preprocess 再走を回避)
5. **memory `project_imitation_case1_phase3` の n<300 ルール**: 50 戦で >5% は **proceed signal**、300 戦で +5pp は **adoption decision**。50 戦で 1/50 を 1 件追加で超えたとしても本採用判定にはしない (確率 11% で偶然起きる)

## 次 iter 候補 (本 plan の範囲外)

- iter3: cand head 構造変更 (focal loss / structural change)
- iter4: ship head λ_ship sweep (0.5 / 1.0 / 2.0)
- iter5: ship head を categorical (6-bucket) と continuous で A/B
- 横展開: 本 iter で得た scheduler + early stop pattern を case4 / case5 / case7 へ移植 (各 case が train.py を独立コピーしているため manual port)
