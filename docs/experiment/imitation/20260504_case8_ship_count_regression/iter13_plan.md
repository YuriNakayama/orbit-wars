# imitation/case8 — iter13: pyarrow zero-copy in-memory dataset

> 作成日: 2026-05-06
> 関連: `iter5_plan.md` (lazy refactor の起源), `iter1_plan.md` (in-memory baseline)
> スコープ: dataset 層のみ。学習 hyperparams は iter11/12-B (EMA OFF) を維持

## 仮説 (Hypothesis)

iter5-12 の lazy parquet dataset (`pq.ParquetFile.read_row_group` + `to_pylist` + LRU cache size=4) が **epoch=0 の最初の batch を返さず +58 min hang する root cause** は、shuffle=True 下で row group cache thrash + arrow buffer pool overhead が host RAM swap を引き起こしているため。pyarrow `flatten().to_numpy(zero_copy_only=False)` で list-typed column を **python list 中間なし** に numpy 化 + column drop_in_place で peak RAM を抑えれば、in-memory load でも RTX 4090 host (24-32 GB) で完走できる。

## 既存コードの現状

- iter1-3 in-memory: `pl.read_parquet → to_list → np.array`。polars DF + python list + numpy が三重常駐 → ~30 GB peak。iter4-v1 で OOM Killed。
- iter5-12 lazy: `pq.ParquetFile.read_row_group(rg)` → cache size 4。epoch=0 hang。EMA OFF (iter12-B) でも再現 → EMA は無罪、dataset 側が真因。
- 中間サイズ見積:
  - 300k rows × 32 KB/row → in-memory 9.6 GB
  - polars + pylist + numpy = 9.6 × 3 = 28.8 GB (iter1 OOM 寸前)
  - lazy cache 4 × 800 MB row group + arrow buffer pool overhead ≈ 6-10 GB transient + dataloader prefetch ⇒ host swap 飽和

## スコープ (Scope)

- 変更ファイル: `bot/pipeline/imitation/case8/training/dataset.py` のみ (in-memory load に書き換え)
- 周辺: `bot/tests/pipeline/imitation/case8/test_dataset_lazy.py` の lazy 内部状態テストを in-memory 動作テストに差し替え
- config: `bot/pipeline/imitation/case8/configs/il_case8.yaml` のコメントだけ更新 (param 変化なし)
- 学習側 (train.py / model.py / preprocess.py / losses.py): **完全不変**

## 実装ステップ (Implementation outline)

1. `dataset.py` の `CaseFourDataset.__init__` を書き換え:
   - `pq.read_table(path)` で 1-shot read
   - 各 list-typed column について `combine_chunks() → flatten() (再帰) → to_numpy(zero_copy_only=False) → np.array(copy=True)` の流れで numpy buffer 化
   - 取得直後に `table.drop_columns([name])` で arrow buffer を即解放 → peak RAM = (largest column copy ~6 GB) + (table - that column)
   - `is_noop` (primitive) は `to_numpy(zero_copy_only=False)`
2. lazy 専用 helper (`_RowGroupArrays`, `_load_row_group`, `_locate`, `_CACHE_SIZE`, `_cache`, `OrderedDict`, `gc`) を全削除
3. `__getitem__` を `torch.from_numpy(self._planet_feats[idx])` 等の pure numpy slice に簡略化
4. test を in-memory 観点で書き直し (`_num_row_groups` / `_cache` 依存を削除)、新規に `mask_planet_cols` 効果テストを追加

## 検証方法 (Validation method)

- ローカル:
  - `uv run pytest tests/pipeline/imitation/case8/test_dataset_lazy.py -x` → 5/5 pass 必須
  - `dev/test-bot` の format / lint pass (mypy duplicate module は pre-existing)
- リモート:
  - `dev/runpod train <sha> --case case8` (RTX 4090 SECURE)
  - 期待: `60_before_train` 通過後 epoch=0 が ~5 min 内に完了し epoch_end log を吐く (iter1 と同水準)
  - 失敗判定: epoch=0 開始から +15 min で epoch_end が出なければ host RAM 不足の可能性 → A6000 (48 GB host) に切替検討
- 評価:
  - 完走できれば iter1-3 と同 metric シグネチャ (val_total / val_cand_fire_acc / val_ship_mae)
  - **win rate 評価は完走確認 → 別 iter で 300戦実施**。iter13 の主目的は **「学習が完走するか」だけ**

## 参考

- 過去 iter4 OOM 履歴: `docs/experiment/imitation/20260504_case8_ship_count_regression/iter4_plan.md`
- iter5 lazy 設計の background: `iter5_plan.md`
- pyarrow `to_numpy(zero_copy_only=False)` の挙動 (内部 buffer 共有のため writable=False) — 本実装は `np.array(..., copy=True)` で書き込み可能 buffer に変換し、torch.from_numpy 互換 + mask 適用可能化

## リスク

- **R1**: pyarrow `flatten().to_numpy()` が依然 python overhead を抱える可能性 → 直接ベンチマーク取れないが、iter1 の polars 経由より少ないことは確実 (中間 list が無い)
- **R2**: candidate_feats の reshape (300k × 48 × 8 × 14 × 4 = 6.4 GB) が peak RAM 内で完結するか → 本実装は drop_columns で table 側を逐次解放、peak は 1 column 分のみ
- **R3**: A6000 が必要になる場合 → cloud_type=SECURE で `gpu_type_id="NVIDIA RTX A6000"` 指定で対応可
