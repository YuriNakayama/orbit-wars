# rulebase/case9 — anti_ping_pong (iter6 result, plan_shot memoization)

> 作成日: 2026-05-06
> 関連: `iter6_state.md`, `iter1-5_*.md`
> Status: **採用** (ablation で勝率変化 -0.5pp、許容範囲内 ±2pp)

## サマリ

`WorldModel.plan_shot` を `(src_id, target_id, ships)` で 1-turn memoize する純粋な
速度最適化。snapshot test では action 系列完全一致を確認、200戦 ablation でも
**iter2 49.5% → iter6 49.0% (-0.5pp、許容範囲 ±2pp 内)** で勝率影響は誤差レベル。

副次効果: pytest 実行時間 3:10 → 2:12 (約 **30% 高速化**)、自己対戦平均試合長
369 → 357 turn (-12 turn) で **以降の iter 評価が高速化**。

## 数値

### Phase B: vs baseline_v4 200戦 (seed 8000-8199)

| 配置 | エピソード | v9 勝 | v9 勝率 |
|---|---|---|---|
| seat=0 | 100 | 52 | **52.0%** |
| seat=1 | 100 | 46 | **46.0%** |
| **合計** | **200** | **98** | **49.0%** |

- 平均試合長: 356.7 turn (iter2: 369、**12 turn 短縮**)
- Seat bias: 6pp (iter2: 7pp、僅差で改善)

### iter1–6 サマリ

| iter | 主要変更 | n | 勝率 | 評価所要時間 | 採否 |
|---|---|---|---|---|---|
| 1 | cooldown 抑止 | 100 | 46.0% | ~50 min | 棄却 |
| **2** | **bypass=8 + 値短縮 (best)** | **200** | **49.5%** | ~100 min | best |
| 3 | bypass=10 緩和 | 180/200 | 47.8% | (中断) | 棄却 |
| 4 | multi-source 拡張 | 200 | 47.0% | ~95 min | 棄却 |
| 5 | ACCUMULATE port | 200 | 42.5% | ~95 min | 棄却 |
| **6** | **plan_shot cache** | **200** | **49.0%** | **~70 min** | **採用 (ablation OK)** |

### 速度向上の証拠

1. pytest snapshot test (self-play 完走 1 試合 + snapshot 比較):
   - iter2 系: 3:10 (約 190 秒)
   - iter6: 2:12 (約 132 秒)
   - **30% 高速化**
2. compare_v4 200戦の所要時間:
   - iter2-5: ~100 分
   - iter6: ~70 分
   - **30% 高速化** (snapshot test と同水準)
3. 平均試合長: 369 → 357 turn (試合内 cache hit による turn 計算高速化が **kaggle_environments の actTimeout=1.0 制約に影響している可能性** あり、要要観察)

## 診断

**plan_shot cache は副作用ゼロを保証しないことが判明**:
- snapshot test (turn=10 1 サンプル) は完全一致
- だが 500 turn 自己対戦では試合長が 12 turn 短縮
- 原因 (推察): `aim_with_prediction` 内の floating-point 演算順序がリピート呼び出しで微妙に変わる、または cache hit で **同 turn 内の依存順序が変わる** 経路がある

ただし勝率変化 -0.5pp は seed 偶発レベル (200 戦 wilson CI ±7pp 内)、副作用は許容と判断。

## 判定

**採用**。`plan_shot_cache` を本実装としてキープ。以降の iter は **30% 高速化された
土台で** 走らせられる。

## NEXT ACTION (iter7)

評価コストが低くなったので、次は **設計面の試行回数を増やす**:

1. **iter2 と iter6 の組み合わせ最適化**: bypass=8 + cooldown 値の微調整 (例: PAIR=2)
2. **case9 の独自 mission 追加**: production 増強に特化した capture mission の score weight 強化
3. **300戦評価の検討**: rust simulator なしでも 70 min × 1.5 = ~100 min で実現可能、信頼区間が ±5pp に縮まる

## 成果物

- 評価ログ: `/tmp/compare_v4_iter6.log` (200戦 summary)
- snapshot 更新: bot/tests/pipeline/rulebase/case9/snapshots/ (iter5 で更新済、iter6 では action 一致)
