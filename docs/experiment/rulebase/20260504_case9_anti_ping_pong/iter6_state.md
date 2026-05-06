# iter6 — Loop Resume State (RUNNING)

> 作成日: 2026-05-05
> Status: **iter6 評価実行中** (PID 39060, seed 8000-8199, 200戦)

## 進行中

- 速度最適化: `WorldModel.plan_shot` を `(src, dst, ships)` で memoize
- 実装: 内部関数 `_plan_shot_uncached` に旧ロジックを移し、`plan_shot` は cache lookup
  → 不一致時 `_plan_shot_uncached` 呼び出し → cache 書込み + return
- snapshot test 3/3 pass で **動作完全一致** (= cache は純粋な副作用ゼロ最適化)
- pytest 実行時間: 3分10秒 → 2分12秒 (**約 30% 高速化**)
- 評価ログ: `/tmp/compare_v4_iter6.log`、ETA ~70 分 (高速化後の試算)

## 採否判定

**ablation スタイル**: 性能を下げない条件で採用
- 200戦で iter2 (49.5%) ± 2pp 以内 (47.5% – 51.5%) → 採用
- 採用後に **iter7 で別軸 (cooldown tuning など) を試す高速化された土台に**
- 範囲外なら最適化に bug があるとして revert

## 副次効果 (期待値)

- 200戦評価時間が 100分 → 70分程度に短縮見込み (cache 効果は実 self-play でより強く出るため snapshot test の 30% より大きい可能性)
- iter7+ の試行回数が増やせる

## 次のループ周回でやること

1. PID 39060 の重複ガード確認
2. 完了後:
   - 勝率 47.5–51.5% 範囲内 → 採用、iter6_result.md 作成 + commit
   - 範囲外 → 棄却、plan_shot cache を revert
3. 評価所要時間を記録 (iter1-5 平均と比較)

## 過去 iter サマリ

- iter1 (cooldown 抑止): 46.0%
- **iter2 (bypass=8 + 値短縮)**: **49.5% (best)** ← 比較基準
- iter3 (bypass=10): 47.8%/180中断
- iter4 (multi-source): 47.0%
- iter5 (ACCUMULATE port): 42.5% (最低)
- **iter6 (plan_shot cache, 速度最適化)**: 評価中
