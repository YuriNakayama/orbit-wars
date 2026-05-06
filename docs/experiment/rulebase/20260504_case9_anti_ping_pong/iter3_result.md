# rulebase/case9 — anti_ping_pong (iter3 result, EARLY STOP)

> 作成日: 2026-05-05
> 関連: `iter3_plan.md`, `iter3_state.md`, `iter2_*.md`
> Status: **棄却** (180/200 戦で中断、47.8%)

## サマリ

`LOW_PLANET_BYPASS_THRESHOLD` を 8 → 10 に緩和した結果、200戦中 180戦 (90%) 時点で **47.8%** (v9=86 / v4=94 / draws=0)。
iter2 (49.5%, 200戦) 比 **-1.7pp の悪化**。残り 20 戦での挽回見込みなしと判断し、ユーザー指示で停止。
**bypass 緩和単独はマイナス効果**。次は iter4 で bypass=8 に戻し ACCUMULATE port を主役にする。

## 数値

### Phase B: vs baseline_v4 180戦 (seed 4000-4179)

- 完了: 180/200 (90%)
- v9 wins: 86, v4 wins: 94, draws: 0
- **勝率: 47.8%**, 信頼区間 (Wilson 95%): 約 [40.6%, 55.0%]

### iter1–iter3 推移

| iter | 主要変更 | n | 勝率 | iter2 比 |
|---|---|---|---|---|
| 1 | 抑止導入 (cooldown 3,5; MIN_DEFICIT=3) | 100 | 46.0% | (base) |
| 2 | bypass=8 + 緩和 (1,2; MIN_DEFICIT=1) | 200 | **49.5%** | (best) |
| 3 (中断) | bypass=8 → 10 | 180 | 47.8% | **-1.7pp** |

### chunk 別累積勝率

| chunk | iter2 (3000-3199) | iter3 (4000-4179) |
|---|---|---|
| 0–60 | 56.7% | 50.0% |
| 0–120 | 55.8% | 45.8% |
| 0–180 | 51.1% | 47.8% |

iter3 は最初から iter2 を下回る軌跡。

## 診断

**仮説 (bypass=10 で「16-19 惑星帯の僅差負け」を救う) は成立せず**。理由 (推察):
- bypass=10 で「劣勢シグナル」のしきい値が低すぎ、**通常のミッドゲーム (10-15 惑星帯) でも cooldown が無効化** され ping-pong 抑止効果が消失
- iter2 で稼いでいた「ping-pong 抑止による launches -17.7%」効果がほぼなくなり、iter1 寄りの挙動に回帰

## 判定

**棄却**。iter4 では:
1. `LOW_PLANET_BYPASS_THRESHOLD` を **8 に戻す** (iter2 の知見を維持)
2. **case7 ACCUMULATE port** を主役に (iter2 analysis NEXT ACTION の最重要項)
3. 評価は 200戦継続 (rust 導入は別議題)

## 成果物

- 中断ログ: `/tmp/compare_v4_iter3.log` (180/200 までの中間値)
- (run parquet なし、compare_v4 はリプレイ非保存)

## NEXT (iter4)

`iter3_state.md` 参照。bypass を 8 に戻し ACCUMULATE port を実装。
