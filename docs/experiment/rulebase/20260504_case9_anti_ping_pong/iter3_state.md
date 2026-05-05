# iter3 — Loop Resume State

> 作成日: 2026-05-05
> Status: **iter3 評価実行中** (PID 15098, seed 4000-4199, 200戦)

## 進行中

- 変更: `LOW_PLANET_BYPASS_THRESHOLD: 8 → 10` (1 行のみ)
- 評価コマンド: `compare_v4.py -n 100 -p 4 --seed 4000`
- ETA: ~100分 (iter2 同等)
- ログ: `/tmp/compare_v4_iter3.log`

## 完了したこと

- iter3_plan.md 作成 (bypass 緩和単独効果測定にスコープ絞った)
- config.py 1 行変更
- ruff/mypy green

## 次のループ周回でやること

1. **重複ガード確認**: PID 15098 が生きていれば skip (compare_v4 進行中)
2. プロセス完了後:
   - `/tmp/compare_v4_iter3.log` から最終 summary を読む
   - iter3_result.md 作成 (採否判定: iter2 比 +2pp で採択)
   - 採択 → iter4 で ACCUMULATE port (case7 から)
   - 棄却 → bypass を 8 に戻し iter4 で ACCUMULATE port を主役に

## iter4 候補 (iter3 結果次第)

- case7 ACCUMULATE port: stay.py (488 行) + strategy.py + strategy_helpers.py + config.py (~30 行) を case9 にコピペ + 配線。重い作業 (~1-2 周回かかる)
- 余剰 ship 用途として最有望 (production 増強につながる遠距離 capture mission)
