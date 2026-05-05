# iter3 — Loop Resume State (STOPPED EARLY)

> 作成日: 2026-05-05
> Status: **iter3 ユーザー判断で 180/200 (90%) で停止**

## 中断時の数値

180戦 v9=86 / v4=94 / draws=0 → **47.8%**。

iter2 (49.5%, 200戦) 比 -1.7pp、iter1 (46.0%, 100戦) より +1.8pp。
bypass=8 → 10 への緩和は **逆効果**。残り 20戦で挽回見込みなしと判断。

## 判定: iter3 棄却

- `LOW_PLANET_BYPASS_THRESHOLD: 10 → 8` に戻す (iter4 の起点)
- 序盤から一貫して iter2 を下回る → 緩和し過ぎは ping-pong 抑止効果を毀損

## chunk 別比較 (iter2 vs iter3)

| chunk | iter2 (seed 3000-) | iter3 (seed 4000-) | 差 |
|---|---|---|---|
| 0–60 | 56.7% | 50.0% | -6.7pp |
| 60–120 (累積) | 55.8% | 45.8% | -10pp |
| 120–180 (累積) | 51.1% | 47.8% | -3.3pp |

iter3 は最初から iter2 を下回って推移。bypass=10 単独効果は **-2pp 程度のマイナス**と推察 (seed 差 ±5pp の範囲を超える)。

## 真のボトルネック (再認識)

1. **評価コスト**: 200戦 ~100 分 → 6 周回で 1 設計サイクル。rust simulator なしでは仮説回しが現実的でない
2. **seed 分散**: 200戦でも CI ±7pp、設計差 +2-3pp の検出が困難
3. **余剰 ship 流用未実装**: iter2 analysis の本命課題、cooldown 系の小修正で +5pp は届きそうにない

## iter4 で次にやること (優先順、必須)

1. **`LOW_PLANET_BYPASS_THRESHOLD: 10 → 8` に戻す** (iter2 の知見を維持)
2. **case7 から ACCUMULATE port** (本命): stay.py / strategy.py / strategy_helpers.py / config.py の関連 ~600 行をコピペ + 配線
3. **300戦評価**: rust 導入できれば 5-10 分、できなければ 200戦のまま
4. (オプション) rustc 導入はユーザー権限要、確認したい
