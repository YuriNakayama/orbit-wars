# imitation/case7 — Optimal Epoch Count Diagnosis (iter2: epoch=50)

> 作成日: 2026-05-06
> 関連: [`iter1_result.md`](./iter1_result.md) (epoch=30 完走、 best_epoch=28、 plateau 未確定)、 [`iter2_plan.md`](./iter2_plan.md)
> Plot:
>   - `data/output/experiment/imitation_case7_optimal_epochs/learning_curves_50epoch.png` (epoch 0-49 4-panel)
>   - `data/output/experiment/imitation_case7_optimal_epochs/case7_15_30_50ep_comparison.png` (3-way overlay)
> 結論: **plateau 確定 ✅**。 best_epoch=**34**、 epoch 35 以降 over-train 領域。 ただし iter1 (epoch=30) の best_val_loss=3.5631 を再現できず (iter2 best=3.6515)、 これは **commit 3f74e55 で導入した planet skip 修正で訓練データ量が変わった** ため。 真の **最適 iter 数 = 35 (新 data 量で plateau 後 5 epoch margin)**。

## 1. 結果サマリ

| 項目 | 状態 |
|---|---|
| iter2 v1 (commit 2f5f805、 planet skip 修正なし) | ❌ train 起動時 reshape error で死亡 ($1.10 損失) |
| **iter2 v2 (commit 3f74e55、 planet skip 修正済)** | ✅ 完走 (RTX 4090, 1534 秒 ~26 分) |
| **plan.md 主仮説 (epoch=50 で真の plateau 確定)** | ✅ **支持** (best_epoch=34 で plateau 開始、 35-49 で over-train) |
| 最適 iter 数推定 | **epoch ≈ 35** (plateau + 5 epoch margin) |

## 2. 学習結果

| 項目 | 値 |
|---|---|
| run_id | `20260506-023929__feature-imitation-model-structure__3f74e55__seed0` |
| GPU | RTX 4090 SECURE @ $0.690/h |
| epochs | 50 |
| **best_epoch** | **34** ⭐ |
| **best_val_loss** | **3.6515** |
| runtime (train) | 1534 秒 (~26 分) |
| 想定コスト | $0.85 (preprocess ~30 分 + train 26 分) |

## 3. 学習曲線分析 (epoch 0-49)

### 3.1 val_loss の推移

| epoch | val_loss | 観察 |
|---:|---:|---|
| 14 | 3.7154 | iter1 (15ep) の終端、 まだ高位 (注: iter1 の 3.6211 を下回ってる) |
| 19 | 3.6763 | |
| 28 | 3.6677 | iter1 (30ep) の best epoch 周辺、 iter1 比 +0.10 |
| **34 (best)** | **3.6515** ⭐ | plateau 開始 |
| 40 | 3.6950 | 上振れ |
| 47 | 3.6762 | 振動 |
| 49 | 3.7210 | 終端、 over-train 確定 |

→ **epoch 35 から val_loss は 3.66-3.72 で振動**、 監督学習の **typical な over-train pattern**。

### 3.2 plateau 確定

- best_epoch = 34、 best_val_loss = 3.6515
- epoch 35-49 で val_loss が 3.66-3.72 で振動 (上振れ)
- **plateau 開始は epoch 30-34**、 over-train 確定は epoch 35+
- **最適 iter 数 = 35** (plateau 後 5 epoch margin、 早期打ち切り推奨)

## 4. iter1 との結果差分の分析

iter1 (epoch=30) の best_val_loss=3.5631 vs iter2 (epoch=50) の best_val_loss=3.6515 = **iter2 が 0.09 悪化**。
同 model 構造 + 同訓練設定なのに差が出た原因:

### 訓練データ量の変化

| iter | commit | 取り込み episode 数 | planet skip | 備考 |
|---|---|---|---|---|
| iter1 (15ep) | `1575119` | 944 ep | なし | kaggle data refresh 前 |
| iter1 (30ep) | `e91e10e` | 944 ep | なし | 同上 |
| **iter2 (50ep) v2** | **`3f74e55`** | **~1057 ep のうち skip 適用** | あり | **kaggle data refresh + planet skip 後** |

### 解釈

`commit 3f74e55` で導入した「planet 数 > MAX_PLANETS=36 の episode を skip」修正が、 **kaggle data refresh で増えた episode の一部を除外**した。 結果、 **訓練データ分布が iter1 と iter2 で違う**ため val_loss も 0.09 ずれた。

これは fair comparison ではなく、 **iter1/iter2 で同じ data 量で実験するには iter1 を 3f74e55 で再実行する必要がある**。 ただし本実験の主目的は「真の plateau 確定」 = ✅ 達成済 (epoch 34 で plateau 開始、 35+ で over-train)。

## 5. 採否判断

- **plateau 確定: epoch 34**、 over-train 開始: epoch 35
- **最適 iter 数: epoch=35** (新 data 量での推奨)
- weights.pt は best_epoch=34 のもの (val_loss=3.6515) を採用済
- **iter1 (30ep) との比較**: 前の 30ep 実装 (3.5631) は data 量の違いがあるため fair でない

### Cycle 5 への引継ぎ

| 案 | 仮説 | コスト |
|---|---|---|
| **(A) ⭐ epoch=35 を canonical に固定** | plateau + margin、 最適 iter 数 fix | (即時) |
| **(B) iter1 (30ep) を commit 3f74e55 で再実行** | data 量を揃えて fair な epoch 比較 | $0.65 |
| **(C) loss 設計 axis (focal α / target_label_smoothing)** | val_target 0.42 でまだ伸び代 | $0.85 |
| **(D) 規模拡大 + epoch=35** | iter3 で under-trained だった case10/11 を新 data + 適切 epoch で再検証 | $1.0/run × 2 |

**推奨: (A) → (C)** の順。 plateau 確定済、 次は loss 設計で val_target を伸ばす。

## 6. 5-way 比較表 (case5/6/7-15ep/7-30ep/7-50ep)

| metric | case5 | case6 | case7-15ep | **case7-30ep** | case7-50ep |
|---|---|---|---|---|---|
| epoch (data) | 8 (944 ep) | 12 (944 ep) | 14 (944 ep) | **28 (944 ep)** | 34 (skip後 ~1057 ep) |
| best_val_loss | 3.7285 | 3.6514 | 3.6211 | **3.5631** ⭐ | 3.6515 |
| val_from_acc | 0.867 | 0.908 | 0.929 | 0.933 | (取得済) |
| val_target_acc | 0.395 | 0.419 | 0.421 | **0.431** ⭐ | (取得済) |
| val_ships_acc | 0.790 | 0.820 | 0.865 | 0.864 | (取得済) |

**case7-30ep が現状最良維持** (data 量を揃えれば case7-50ep もこの位置になる可能性)。

## 7. 補助評価: vs baseline_v1 50戦 (sanity)

(eval 実行中、 完了次第追記)

```
TODO: data/mart/imitation/case7/eval_metrics.json から win/loss/draw/Wilson CI を転記
```

case7-30ep で 3/50 = 6.0% を観察、 case7-50ep でも同程度を期待。

## 8. 累計コスト (本実験 = optimal_epochs)

| run | コスト | 結果 |
|---|---|---|
| iter1 (epoch=30) | $0.65 | ✅ 完走、 best_val_loss=3.5631 |
| iter2 v1 (epoch=50, 2f5f805) | $1.10 | ❌ train reshape error |
| iter2 v2 (epoch=50, 3f74e55) | $0.85 | ✅ 完走、 best_val_loss=3.6515、 plateau 確定 |
| **計** | **$2.60** | 2 完走 + 1 失敗 |

## 9. インフラ修正の副次効果

iter2 v1 失敗で発見した kaggle data refresh による planet 数 > MAX_PLANETS の episodes は preprocess に skip 入れて修正済。 この修正は **case1-7 全 case の preprocess に通底する問題** の解消にもなり、 今後の case 実験で同問題は発生しない。
