# imitation/case8 — iter5: dataset lazy 化 で iter4 hyperparams を再実験

> 作成日: 2026-05-05
> 関連:
> - [`iter4_plan.md`](./iter4_plan.md) (grad_clip + EMA + head dropout 設計)
> - iter4-v1/v2/v3/v4 (4 連続 RTX 4090 host RAM OOM at `loading train_ds`)
>
> スコープ: backbone も head 構造も hyperparams も**完全不変**、`training/dataset.py`
> のみを **chunk-by-chunk lazy read** に refactor し、iter4 の RAM bloat を
> 解消して iter4 hyperparams を本来の意図通り検証する。

## 仮説 (Hypothesis)

iter4 の grad_clip + EMA + head_dropout 3 点セットは振動抑制設計として完成
しているが、4 回連続で `60_before_train` 直前の `CaseFourDataset.__init__`
において **host RAM OOM で hang** し検証不可能。

OOM の root cause は `pl.read_parquet(...).to_list()` 中間 Python list +
polars internal Arrow buffer pool + numpy 複製の **三重 alloc** が
`candidate_feats` (6.9GB / 328k rows) 処理時にピークし、RTX 4090 host の
~32GB RAM を超過するため。column-by-column drop_in_place fix (iter4-v4)
でも Arrow buffer pool 自体が **次の column 処理開始まで release されない**
allocator semantics により効果不十分だった。

**真の解決策**: `pyarrow.parquet.ParquetFile` の **row group 単位 iteration**
で chunk-by-chunk に読み込み、全 row を同時に in-memory に持たない設計に
refactor する。これでピーク RAM が `1 row group 分` に圧縮 (preprocess の
`FLUSH_EVERY_FRAMES=5000` がそのまま row group 境界なので ~800MB/group)、
**iter4 hyperparams を本来意図した通り** に試せる。

期待: iter4 plan 通りの train/val curve 平滑化 (val_total max/min < 2.0) と
副次 win_rate を計測。

## 既存コードの現状 (from Step 1)

- `training/dataset.py:CaseFourDataset.__init__`: `pl.read_parquet → to_list →
  np.array` で全 row を一括 in-memory load。`__getitem__` は `self._planet_feats[idx]`
  で O(1) アクセス。
- iter4-v4 で column-by-column `drop_in_place + gc.collect` を試したが、
  `polars` 内部の Arrow buffer pool は Python GC では release されないため
  ピーク RAM 削減は限定的だった。
- preprocess (`training/preprocess.py`) は既に `StreamingParquetWriter` で
  flush_every=5000 行単位の row group を生成しており、chunk read 側で活用可能。

## スコープ (Scope)

### 変更ファイル

| Path | 変更内容 |
|---|---|
| `bot/pipeline/imitation/case8/training/dataset.py` | `CaseFourDataset` を **`pyarrow.parquet.ParquetFile`** ベースに書き換え。`__init__` では metadata のみ読み (`num_rows`, `schema`)、行 data は読まない。`__getitem__(idx)` で対応する row group を on-demand load (LRU cache size=1-2)。`class_weight_on_slots` は事前計算用に **`cand_slot_per_src` のみ全 row scan** して in-memory に保持 (~5MB) |
| `bot/pipeline/imitation/case8/training/dataset_lazy.py` (新規 helper) | row group cache class、必要なら別 module に切り出し |
| `bot/tests/pipeline/imitation/case8/test_dataset_lazy.py` (新規) | (1) 小 synthetic parquet で `len(ds) / ds[0]` が正常、(2) row group 跨ぎの idx 取得、(3) class_weight 集計の正しさ、(4) RAM が in-memory 版より小さい (psutil で粗い測定 or `_active_row_group` 内部状態確認) |

### 変更なし

- `policy/{model,types,decoder,featurizer,candidates,geometry,agent}.py` — 完全不変
- `training/{preprocess,losses,train}.py` — 完全不変
- `configs/il_case8.yaml` — iter4 から不変 (grad_clip 1.0 + EMA 0.999 + head_dropout 0.2 + cosine LR + warmup + early stop)

### Hyperparameters

iter4 から **完全不変**:

| Knob | iter4 = iter5 |
|---|---|
| grad_clip max_norm | 1.0 |
| EMA | enabled, decay=0.999 |
| head_dropout | 0.2 |
| epochs | 30 |
| best_metric | val_cand_fire_acc max |
| early_stop | val_cand_fire_acc patience=5 |
| cand_loss | focal (α=0.25, γ=2.0) + class_weight |
| ship loss | SmoothL1, λ=1.0 |

## 実装ステップ (Implementation outline)

1. `dataset.py`: `CaseFourDataset.__init__` を `pq.ParquetFile(path)` ベースに変更
   - metadata から `num_rows`, `num_row_groups` を取得
   - `cand_slot_per_src` のみ全 row scan で読み込み (class_weight_on_slots 用、RAM ~5MB)
   - 他の列は読み込まない
2. `__getitem__(idx)`:
   - `idx` から該当 `row_group_idx` と group 内 offset を計算 (preprocess の
     flush_every=5000 を仮定して `idx // 5000`)
   - cache miss なら `pf.read_row_group(rg_idx, columns=[...])` で 1 group を読む
   - cached numpy view を返却 (LRU=2 で sequential / random access 両対応)
3. `class_weight_on_slots` は init で読んだ `cand_slot_per_src` を使う (互換)
4. `tests/pipeline/imitation/case8/test_dataset_lazy.py`: 上記 4 観点
5. `dev/test-bot` pass を確認
6. RunPod launch (config 変更なし、dataset 変更のみ)
7. `dev/runpod pull --from s3` → 50 戦評価 → `iter5_result.md`

## 検証方法 (Validation method)

- **ローカル**:
  - `dev/test-bot` (format / lint / type / pytest)
  - `uv run --directory bot pytest tests/pipeline/imitation/case8 -x`
- **リモート**:
  - `dev/runpod train <new_sha> --case case8`
  - 想定所要時間: container 起動 + uv sync + dvc pull + train 30 epoch
    (lazy dataset → I/O bound で iter4 比 +20-30% wall-time、~25 min train)
    + post ≈ **~55 min total**
  - 想定コスト: ~$0.65 (RTX 4090 SECURE $0.69/h × 0.95h)
- **評価 (主要メトリクス: train/val curve の平滑さ)**:
  - **(主)** train.log の val_total max/min 比 **< 2.0** (iter2=6x, iter3=8x からの改善目標)
  - **(主)** val_cand_fire_acc の moving max が単調増加
  - **(副)** vs baseline_v1 50戦 win_rate (>5% で 300戦昇格)
  - 採否しきい値:
    - **(主)** val_total max/min < 2.0 → 振動 fix 成功と判定
    - **(主)** val_total max/min ≥ 4.0 → 振動 fix 失敗、別アプローチ
    - **(副)** 50 戦で >5% (3/50) → 300 戦昇格

## リスク / 注意点

1. **chunk-by-chunk read で I/O bound 化、wall-time が長くなる**: 30 epoch 完走に iter1-3 (9 min)、iter4 計画値 (~18 min) より大幅に長くなる可能性。RTX 4090 fast SSD 前提、想定 25-30 min。dev/runpod の watch threshold (15 min) を超える可能性あり、`60_before_train` 後の hang 誤検知に注意 (実際は train 中)。
2. **DataLoader の shuffle と row group cache の相性**: iter1-3 の DataLoader は `shuffle=True` で random index 取得。row group cache size=1 だと毎 batch でグループ swap が発生し I/O が hot。**LRU cache size=4 程度** で吸収する。
3. **class_weight_on_slots の事前計算**: `cand_slot_per_src` のみ全 row scan する init 時間 (~5-10 sec) は加算。
4. **PyArrow と polars の挙動差**: 元コードは polars で `.to_list()` していたが、PyArrow は struct/list dtype の serialization が違う。preprocess の出力 schema (List(float32) など) が PyArrow ParquetFile で読めるか事前確認、必要なら preprocess 側も合わせて変更。

## 次 iter 候補 (本 plan の範囲外)

- iter6: iter5 で fix 効くなら 300 戦昇格 + epochs=60 に拡張
- iter7: ship_w sweep / focal_alpha sweep
- 本 PR の最終形: iter1+iter2+iter3+iter4 (実装) + iter5 (dataset refactor + 検証) を一括 main マージ

## 参考

- iter4 の OOM 試行 chain: `del self._df` (v2) → `column drop_in_place` (v3,v4) いずれも Arrow buffer pool semantics で不十分
- pyarrow `ParquetFile.iter_batches` / `read_row_group`: 標準 API、batch_size を row group size に揃えれば I/O 最適
