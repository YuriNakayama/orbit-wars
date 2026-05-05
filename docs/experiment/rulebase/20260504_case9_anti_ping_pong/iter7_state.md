# iter7 — Loop Resume State (RUNNING)

> 作成日: 2026-05-06
> Status: **iter7 評価実行中** (PID 8747, seed 9000-9299, 300戦)

## 進行中

- 300戦 vs baseline_v4 (seed 9000-9299)
- 設定: case9 = iter2 best 設計 (`bypass=8`, `cooldown 1/2`, `MIN_DEFICIT=1`) + iter6 plan_shot cache
- ETA: ~100 分 (iter6 70 min × 1.5)
- ログ: `/tmp/compare_v4_iter7.log`
- PID: 8747

## 採否分岐 (300戦完了後)

- **300戦 ≥ 55%**: +5pp 達成 → **iter 採択 + loop 完了** → memory 記録 + cron 停止
- **51% ≤ x < 55%**: iter2 ベース設計で +5pp は届かないと結論、iter8 で別軸
- **x < 51%**: iter6 cache の副作用疑い、要検証

## 次のループ周回でやること

1. PID 8747 の重複ガード確認
2. 完了済みなら結果に応じて分岐:
   - **55% 以上**: memory に成功事例書込み → CronDelete 11c931e9 → 完了報告
   - **51-55%**: iter7_result.md (棄却) → iter8_state.md で別軸計画
   - **51% 未満**: iter7_result.md + iter8 で iter6 cache 検証

## 過去 iter サマリ

- iter1 (cooldown 抑止): 46.0%
- **iter2 (bypass=8 + 値短縮)**: **49.5% (200戦、best 設計)**
- iter3 (bypass=10): 47.8%/180中断
- iter4 (multi-source): 47.0%
- iter5 (ACCUMULATE port): 42.5%
- iter6 (plan_shot cache): 49.0% → ablation 採用、30% 高速化
- **iter7 (300戦 confirm)**: 評価中
