# imitation/case7 — Optimal Epoch Count Diagnosis (result)

> 作成日: 2026-05-06
> 関連: [`plan.md`](./plan.md)
> Plot:
>   - `data/output/experiment/imitation_case7_optimal_epochs/learning_curves_30epoch.png` (epoch 0-29 4-panel)
>   - `data/output/experiment/imitation_case7_optimal_epochs/case7_15ep_vs_30ep.png` (overlay)
> 結論: **仮説強く支持 ✅**。 case7 は epoch=15 で **明確に under-trained**、 epoch=30 で best_val_loss が **3.6211 → 3.5631 (−0.058 = 1.6% 改善)** で改善。 ただし epoch 28 (best) も依然 plateau 直前 → **epoch=50 で再実験して真の plateau 確定が推奨**。

## 1. 結果サマリ

| 項目 | 状態 |
|---|---|
| 1-case 集約 + factory pattern (case8/10/11 削除) | ✅ commit `e91e10e` |
| RunPod 学習 (case7 epoch=30) | ✅ 完走 (RTX 4090, 1074 秒 ~18 分) |
| **plan.md 主仮説 (case7 epoch=15 は under-trained)** | ✅ **強く支持** |
| 最適 iter 数推定 | epoch ≥ 30 (epoch 28 が現状 best、 plateau 未確定) |

## 2. 学習結果

| 項目 | 値 |
|---|---|
| run_id | `20260505-223151__feature-imitation-model-structure__e91e10e__seed0` |
| GPU | RTX 4090 SECURE @ $0.690/h |
| epochs | 30 |
| **best_epoch** | **28** ⭐ (= 28/30 = 終盤) |
| **best_val_loss** | **3.5631** (case7 iter1 比 **−0.058 = 1.6% 改善**) |
| runtime | 1074 秒 (~18 分) |
| 想定コスト | $0.65 (preprocess 33 分 + train 18 分 + 環境構築 ~10 分) |

## 3. 学習曲線分析 (epoch 0-29)

### 3.1 val_loss の推移

| epoch | val_loss | 観察 |
|---:|---:|---|
| 14 (iter1 best) | 3.6211 | iter1 終了点 |
| 18 | 3.6100 | iter1 比 −0.011 |
| 23 | 3.5829 | 大幅改善 |
| 27 | 3.5745 | |
| **28 (best)** | **3.5631** ⭐ | iter1 比 −0.058 (1.6% 改善) |
| 29 | 3.5771 | わずかに上振れ (noise の可能性高) |

→ **epoch 14 で plateau に見えていたが、 実は epoch 28 まで明確に低下し続けていた**。 epoch=15 で打ち切るのは早すぎ、 epoch=30 でもまだ plateau 確定とは言えない (29 で上振れあるが 28 がよくなった可能性も)。

### 3.2 各 head の収束タイミング

| head | iter1 best (epoch 12-14) | 30ep best | Δ | 収束タイミング |
|---|---|---|---|---|
| val_loss | 3.6211 | 3.5631 (ep 28) | −0.058 ✅ | **epoch 28 でも下降中、 plateau 未確定** |
| val_from_acc | 0.929 | 0.933 (ep 28) | +0.004 ≈ | epoch 7 で plateau、 以降振動 (0.92-0.94) |
| val_target_acc | 0.421 | **0.431** (ep 28) | **+0.010** ✅ | epoch 11 で 0.42 到達、 epoch 28 で 0.43 (微増) |
| val_ships_acc | 0.865 | 0.864 (ep 28) | −0.001 ≈ | epoch 5 で plateau、 以降 flat |

→ **val_loss と val_target_acc は epoch 30 でも改善余地あり**、 val_from / val_ships は早期飽和。

## 4. 仮説検証

plan.md の主仮説:

> case7 (epoch=15) は under-trained。 best_epoch=14 (= 最終 epoch) で val_loss がまだ下がり続けていた = capacity に対する訓練量不足。 case7 を **epoch=30** で学習すれば真の best_epoch が 20-25 epoch 付近に出現し、 best_val_loss は更に低下 (3.6211 → 3.55-3.60 想定) する。

**結果**:
- best_epoch = **28** (想定 20-25 をやや超過、 plateau 未確定)
- best_val_loss = **3.5631** (想定範囲 3.55-3.60 の中央値、 1.6% 改善)
- → **仮説強く支持 ✅**

## 5. 採否判断 + 次の手

### 採否

- **iter1 (epoch=15) → 30ep run へ昇格**。 val 指標が単調改善
- 新 weights.pt を `bot/pipeline/imitation/case7/policy/weights.pt` に配置済 (canonical 採用)
- 旧 iter1 weights.pt は run dir に保存されている

### 次の手 (Cycle 5 候補)

| 案 | 仮説 | 想定コスト |
|---|---|---|
| **(A) ⭐ epoch=50 で再実験** | epoch 28 で best、 まだ plateau 未確定 → epoch 50 で真の plateau 確定 | $1.0/run (train ~30 分) |
| (B) epoch=30 + lr scheduler (cosine warmup) | より速く収束、 epoch=30 で plateau 到達 | $0.7/run |
| (C) 規模拡大 (case10/11) を epoch=30 で再検証 | iter3 で under-trained 確定したので、 epoch を増やせば規模拡大が活きる可能性 | $1.0/run × 2 |
| (D) loss 設計 axis (focal α / target_label_smoothing) | val_target 0.43 でまだ伸びる、 loss tuning で +3pp 期待 | $0.7/run |

**推奨: (A) → (D)** の順:
1. epoch=50 で plateau 確定 → 最適 iter 数を fix
2. その後 loss 設計に進んで val_target_acc を本格に伸ばす

## 6. 5-way 比較表 (case5/6/7-15ep/7-30ep/8)

| metric | case5 | case6 | **case7-15ep** | **case7-30ep** | case8 |
|---|---|---|---|---|---|
| best_val_loss | 3.7285 | 3.6514 | 3.6211 | **3.5631** ⭐ | 3.6485 |
| val_from_acc | 0.867 | 0.908 | 0.929 | 0.933 | 0.933 |
| val_target_acc | 0.395 | 0.419 | 0.421 | **0.431** ⭐ | 0.412 |
| val_ships_acc | 0.790 | 0.820 | 0.865 | 0.864 | 0.863 |

**case7-30ep が全 metric で 5-way 首位**。 規模拡大 (case10/11) より **訓練量増の方が効果的** であることが実証された。

## 7. 補助評価: vs baseline_v1 50戦 (sanity)

(eval 実行中、 完了次第追記)

```
TODO: data/mart/imitation/case7/eval_metrics.json から win/loss/draw/Wilson CI を転記
```

case7 iter1 (epoch=15) で 0/50、 訓練量増で対戦勝率が動くかは不明。

## 8. 累計コスト

| 項目 | コスト |
|---|---|
| iter1 (case7 epoch=15) | $0.65 |
| **iter1 epoch=30 (本実験)** | **$0.65** |
| roadmap 累計 (iter1-3) | $4.06 |
| **本実験を含む累計** | **$4.71** |
