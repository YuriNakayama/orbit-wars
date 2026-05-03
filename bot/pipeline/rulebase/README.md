# Rulebase Cases

Pure-Python rule-based agents. Each case is an independent submission package
(see `.claude/rules/pipeline.md`).

## Status table

| Case | Status | Strategy summary | publicScore | LB 順位時 | 備考 |
|------|--------|------------------|-------------|----------|------|
| case0 | archive | 単純スナイパー参考実装 | n/a | n/a | 学習用、refactor 対象外 |
| case1 | active (legacy) | baseline_v1 (sniper + reinforcement) | LB 897 | 2026-03 | `strategy.py` を `planner/` に分割 (2026-04) |
| case2 | active | baseline_v2 (OM, lookahead, harass) | n/a | n/a | OM ablation 結果あり |
| case3 | active | baseline_v3 (rollout) | n/a | n/a | case2 + rollout |
| case4 | **production** | baseline_v4 (fleet consolidation) | 745 | 2026-04 | 現役チャンピオン (mission resolver 抽出済) |
| case5 | active (verification) | baseline_v5 (LB1224 port) | 600 | 2026-04 | `agent_full.py` は notebook verbatim port、refactor しない |
| case6 | active (experiment) | baseline_v6 (case4 + STAY judge) | n/a | n/a | defense hold + burst hold、`docs/experiment/rulebase/20260502_case6_stay_mission/` |

## Conventions

- `case<N>/baseline/` が agent body
- `evaluation/snapshot_update.py` は `src/evaluation/snapshot_update.py` 経由
- 大型 `strategy.py` は `case<N>/baseline/planner/` に分割可 (case1 が参考実装)
- cross-case import は禁止 (case 内で完結。共通化したい場合は重複保持)

詳細: `docs/plans/refactor-directory/`
