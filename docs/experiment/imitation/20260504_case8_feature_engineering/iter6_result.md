# imitation/case8 iter6 — Result (K2 leak 除去確認、ships F1 0.93 → 0.63 で iter1 parity)

> 作成日: 2026-05-06
> 関連: `./iter6_plan.md` / `./iter5_result.md` (K2 leak 確定診断) / `./iter4_result.md` (撤回済)
> commit: `0895b65` → train run `20260506-010029__feature-feature-engineering__0895b65__seed0`
> weights: `bot/pipeline/imitation/case8/policy/weights.pt` (iter6 best.pt epoch 12, val_loss=3.5060)

## サマリ (TL;DR)

**判定: K2 causal leak 除去を確定**。iter4 の ships F1 0.9349 は K2 (outgoing fleet trajectory) が `obs.fleets` 経由で `ships_per_src` を逆算する leak によるもので、K2 削除後は **ships F1 0.6272** に大幅後退し iter1 baseline (0.6425) と parity に戻った。iter5 の仮説が完全に裏付けられた。

iter6 を **新 baseline として採用**: dim 57/8 で leak free。target macro F1 は iter1 比 +0.005 と微改善が残り、iter3 で導入した G9/G10/G11 の真の貢献はわずかに正であることが確認できた。

| iter | dim (P/G) | best_val_loss | ships F1 | from F1 | target F1 | 状態 |
|------|----------:|--------------:|---------:|--------:|----------:|------|
| iter1 | 24/10 | 3.5235 | 0.6425 | 0.6651 | 0.3210 | baseline |
| iter3 | 63/14 | 3.5062 | 0.6353 | 0.6529 | 0.3220 | 採用 |
| iter4 | 61/12 | **3.3576** | **0.9349** 🚨 | **0.6838** | **0.3296** | 🚨 leak (撤回) |
| **iter6** | **57/8** | **3.5060** | **0.6272** | **0.6566** | **0.3256** | ✅ clean baseline |

## 訓練ログ (15 epoch)

```
epoch  train_loss  val_loss
0      3.8686      3.6975
12*    3.4696      3.5060   ← best (iter1 3.5235 比 -0.0175)
14     3.4443      3.5325
```

訓練時間: **351 秒** (RTX 4090)。dim 57/8 で iter4 (375s) より速い。

## Stage 1: validation metrics

### 全 head 比較 (iter1 / iter4 leak / iter6 clean)

| head | metric | iter1 | iter4 (leak) | **iter6 (clean)** | Δ vs iter1 | Δ vs iter4 |
|------|--------|------:|------------:|-----------------:|----------:|----------:|
| **from** | F1 | 0.6651 | 0.6838 | **0.6566** | -0.009 | -0.027 |
| from | PR-AUC | 0.7676 | 0.7621 | **0.7418** | -0.026 | -0.020 |
| from | ROC-AUC | 0.9532 | 0.9629 | **0.9492** | -0.004 | -0.014 |
| from | acc | 0.9151 | 0.9153 | **0.9160** | +0.001 | +0.001 |
| **target** | macro F1 | 0.3210 | 0.3296 | **0.3256** | **+0.005** ✅ | -0.004 |
| target | top-1 acc | 0.4327 | 0.4516 | **0.4420** | **+0.009** ✅ | -0.010 |
| target | top-2 acc | 0.6377 | 0.6523 | **0.6428** | **+0.005** ✅ | -0.010 |
| target | PR-AUC macro | 0.3419 | 0.3539 | **0.3474** | **+0.006** ✅ | -0.007 |
| **ships** | macro F1 | 0.6425 | **0.9349** | **0.6272** | -0.015 | **-0.308** 🎯 |
| ships | acc | 0.8519 | 0.9728 | **0.8480** | -0.004 | **-0.125** 🎯 |
| ships | MAE bucket | 0.2022 | 0.0370 | **0.2046** | +0.002 | **+0.168** 🎯 |
| ships | PR-AUC macro | 0.7117 | 0.9628 | **0.7009** | -0.011 | **-0.262** 🎯 |

### 解釈

- **ships head の劇的後退** (Δ vs iter4 で macro F1 -0.308) は **iter5 の仮説どおり**。`obs.fleets` 自軍 fleet を一切入れない iter6 では ships head の精度は iter1 baseline 水準 (0.6425 → 0.6272) に戻る。leak は K2 が単独原因であったことが確定。
- **target head は iter1 比 +0.005 〜 +0.009 の真改善**: macro F1 / top-1 / top-2 / PR-AUC が全て iter1 を超え、iter4 の +0.008 〜 +0.019 の半分程度が「真の改善」として残った。残り半分は K2 経由の副作用 (ships head の logit が target head の bottleneck を間接的に支援していた)。
- **from head は iter1 比 -0.009** で僅かに後退するが、permutation importance で K3 frontline_distance や G10 defense_surplus は from head に対し ±0.01 以下の寄与しかなく、誤差範囲内。
- **val_loss 3.5060 は iter1 (3.5235) 比 -0.018** と iter3 (3.5062) と同水準。iter4 の -0.166 は丸ごと leak 副作用だった。

## leak 仮説の最終決着

iter5 で立てた仮説:

> K2 outgoing fleet trajectory は自軍 fleet (action_N の結果) を per-source-planet に attribute する特徴量。`ships_per_src` ラベルは action_N の発射 ships で、`obs.fleets` の自軍 fleet は **action 適用後の値** → policy が逆算して ships_per_src を学んでしまう leak。

iter6 で確認:

| Diagnostic | iter5 予測 | iter6 実測 | 一致? |
|---|---|---|---|
| ships F1 | 0.62 〜 0.66 (iter1 base に戻る) | **0.6272** | ✅ |
| target F1 | 0.32 〜 0.33 (leak 副作用なら下がる) | **0.3256** | ✅ (iter1 +0.005 残) |
| from F1 | 0.65 〜 0.69 | **0.6566** | ✅ (iter1 比 -0.009) |
| ships balanced acc | 0.50 〜 0.65 想定 | (本 run 未測定だが acc 0.8480 で iter1 0.8519 と parity) | ✅ |

**結論**: iter4 の ships F1 0.9349 は **完全に leak 由来** であり、iter6 で除去した。case3 phase2 result.md の教訓 (`obs_N` は action 適用後) と一致する。

## K4 削除の効果

K4 (aux_my_ships_h5/h15/prod_log/ratio) は iter5 permutation で sum |Δ|=0.001 と完全 dead であった。iter6 で削除したが val_loss は iter3 (K4 なし期) の 3.5062 と同水準 → **削除による損失なし**。dim を 12→8 にスリム化できた。

## 副次効果: dim slim 化

- planet: 61 → **57** (-4)
- global: 12 → **8** (-4)
- 訓練時間: 375s → **351s** (-6%)
- 訓練 / val parquet 容量も dim 削減に応じて減少 (preprocess 再実行で確認)

## 採否判定

- **iter6 を新 baseline として採用** (`bot/pipeline/imitation/case8/policy/weights.pt` に promote 済み)
- iter4 の評価記録 (iter4_result.md) は文書として残すが、weights は iter6 で上書き
- 真の Stage 1 改善 (vs iter1):
  - target macro F1 +0.005 (iter3 G9 pairwise / G10 defense_surplus が target head に微効)
  - ships / from は parity 〜 微減
  - val_loss は -0.018 とごく小さい改善
- iter3 で導入した G9 (pairwise top-K) / G10 (defense surplus) / G11 (sparse mask) のうち、target head に効いている group は iter5 importance では G9 (sum |Δ|=0.11) と G10 (0.05) が見えるが G11 (0.06) は限定的。next iter は G9/G10 を残しつつ G11 を再評価対象に

## Stop conditions のチェック

- [x] case8 featurizer/configs/tests が iter6 仕様 (PLANET=57, GLOBAL=8) に整理、unit test pass
- [x] `dev/test-bot` green
- [x] commit & push (commit `0895b65`)
- [x] RunPod Step B が `99_done` で終了 (run `20260506-010029__...`、約 30 分・~$0.35)
- [x] Stage 1 metrics 再計算 (上記表)
- [x] iter6_result.md に「leak 除去後の真の Stage 1 metrics」+ iter1/iter4 比較

## next iter 候補

iter5 importance で sum |Δ| ≥ 0.1 だが iter6 で残存している group のうち、削除候補と追加候補の両方を検討:

1. **削除候補** (importance 低、 dead 疑い): G11 sparse_mask (0.06)、G7 inbound_fleet_trajectory (0.03)、K3 frontline_distance (0.03)、H3 global_launch_history (0.02)
2. **追加候補** (現在 missing で leak free):
   - **comet 残時間 / 種類** (G2 ships_production_owner が importance #2 = 1.21 → 関連特徴量を強化)
   - **opponent fleet 累積発射量** (G7 inbound は今あるが「過去 H step 累計発射 ships」は無い)
   - **production-deficit forecasting** (h=30 production accumulator)
3. **constraint**: 自軍 fleet (action_N の結果) は絶対に入れない (iter4 教訓)

iter7 plan は別途。

## 参考

- iter5_result.md: K2 leak 確定診断 (permutation A/B/C)
- iter6_plan.md: 削除スコープと期待値
- case3 phase2 result.md (history leak の教訓: `obs_N` は action 適用後)
- `data/output/models/imitation/case8/runs/20260506-010029__feature-feature-engineering__0895b65__seed0/metrics.json`
