# iter9 敗北ケース解析 — seed=0

**試合情報:**
- match_id: `20260422T052741Z_1v1_seed0`
- agents: il_v1 (idx 0) vs baseline_v1 (idx 1)
- 結果: **il_v1 敗北** (turn 126 で終局)
- 最終 reward: il_v1=-1, baseline_v1=+1

## アクション分布

| actions/turn | il_v1 (回) | baseline_v1 (回) |
|---|---|---|
| 0 | 46 (36.5%) | 52 |
| 1 | 21 | 16 |
| 2 | 35 | 14 |
| 3 | 19 | 12 |
| 4 | 5 | 8 |
| 5+ | 0 | 24 |

- il_v1 は 36.5% の turn で何もしない (gating は iter6 より改善したがまだ passive)
- baseline_v1 は 1 turn で最大 12 fire (強攻撃の turn あり)

## ターン別トレース (艦/惑星/艦隊)

| turn | il_ships | base_ships | il_planets | base_planets | il_fleets | base_fleets |
|---|---|---|---|---|---|---|
| 10 | 5 | 20 | 2 | 2 | 6 | 0 |
| 20 | 42 | 21 | 3 | 2 | 7 | 7 |
| 30 | 38 | 27 | 3 | 3 | 18 | 13 |
| 40 | 57 | 55 | 4 | 5 | 18 | 13 |
| 50 | 123 | 176 | 5 | 9 | 28 | 17 |
| 60 | 171 | 232 | 7 | 14 | 24 | 22 |
| 70 | 200 | 374 | 5 | 17 | 32 | 22 |
| 80 | 123 | 412 | 5 | 19 | 44 | 40 |
| 90 | 128 | 364 | 4 | 17 | 36 | 52 |
| 100 | 0 | 393 | 0 | 22 | 26 | 63 |
| 110 | 0 | 1149 | 0 | 24 | 8 | 43 |
| 120 | 0 | 2421 | 0 | 24 | 2 | 7 |
| 126 | 0 | 2854 | 0 | 24 | 0 | 2 |

## パターン

**Phase 1 (turn 0-30): 序盤互角**
- il_v1 はむしろ艦数リード (turn 20: 42 vs 21)
- Home planet からの初期派兵が効いている

**Phase 2 (turn 30-50): 拡張戦で遅れる**
- Turn 50: baseline_v1 が planets 5→9 に急拡大、艦数も 176 vs 123 と逆転
- il_v1 は中立惑星確保のペースが追いつかない (iter9 val metrics でも HIGHEST_PROD_NEUTRAL F1=0.13 と過小予測)

**Phase 3 (turn 70-100): 戦線崩壊**
- Turn 80: il_v1 艦数が 200→123 に減少 (反撃で 77 艦失う)
- Turn 100: **il_v1 全惑星喪失** (planet 5→0)
- この間 il_v1 の fleets は 32→44 と出ていたが、守りが破綻

**Phase 4 (turn 100-126): 一方的**
- il_v1 は全惑星なしで残艦隊のみ、baseline_v1 は増産続く (393→2854)
- Turn 126 で終局

## 考察

1. **中盤の拡張ペースが致命的に遅い**: NEAREST_NEUTRAL F1 0.33, HIGHEST_PROD_NEUTRAL F1 0.13 という val の欠点が、生産潜在力の高い中立惑星を取り逃す → 中盤以降の艦数スケールで負ける
2. **防衛ロジックがない**: iter9 も含めて、模倣対象には「自惑星が攻撃されたら守る」概念がない。WEAKEST_OWN template は存在するが F1=0.21 と機能してない
3. **36.5% の 0アクション turn** は「撃たないのが最適」ではなく「gating が依然 conservative すぎる」可能性

## 改善仮説

- NEAREST_NEUTRAL / HIGHEST_PROD_NEUTRAL を up-weight (target class weight β を 0.9999→0.999 に強化)
- from head の pred_pos_rate を gt=0.127 にもっと近づける (現状 0.158 で over-fire 気味なので、threshold を 0.31→0.40 に上げる方が勝率に効くかも)
- データ追加 (rating_quantile 0.50→0.30) で弱い棋譜も含めて防衛パターンを学ぶ
