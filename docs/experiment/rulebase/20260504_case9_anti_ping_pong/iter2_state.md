# iter2 — Loop Resume State (CLOSED)

> 作成日: 2026-05-05
> Status: **iter2 評価完了、棄却**

iter2 結果: 200戦 49.5% (iter1 比 +3.5pp、しきい値 55% 未達)。

## 完了したこと

- iter2_plan.md / iter2_result.md 作成済み
- case9 実装変更 (LOW_PLANET_BYPASS_THRESHOLD=8、cooldown 短縮、REINFORCE_MIN_DEFICIT=1)
- 200戦 vs baseline_v4 評価 (seed 3000)
- pytest 79/79 pass

## iter3 で次にやること

1. **`LOW_PLANET_BYPASS_THRESHOLD=8 → 10` に緩和** (config.py のみ 1 行変更)
2. **case7 から ACCUMULATE port** (余剰 ship 流用): `bot/pipeline/rulebase/case7/baseline/missions/stay.py` の ACCUMULATE 関連を case9 に移植 + `ACCUMULATE_ENABLED` フラグ
3. **rust simulator 切替検討**: rustup 未導入のため要確認 → 導入できれば `orbit_wars_rust` で 200戦を ~20 分に短縮

詳細は `iter2_result.md` の NEXT ACTION 節を参照。
