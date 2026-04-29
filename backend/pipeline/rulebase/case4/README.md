# case4 — baseline_v4 (production)

現役チャンピオン case。`fleet_consolidation` を追加した case3 の進化形。

## 採用戦略

- case3 の構成 + `baseline/missions/fleet_consolidation.py`
- 余剰艦の集約による効率向上

## 成績

- vs baseline_v3: **70.3%** 勝率 (300 戦)、`project_case4_phase_results.md` 参照
- Phase C (10 iter) で V=0.95 / DENSE=0.90 採用済
- publicScore 745 (2026-04 観測)

## 構造

case3 と同型。違いは:
- `baseline/missions/fleet_consolidation.py` 追加 (test_fleet_consolidation.py で覆われている)
- `baseline/core/physics.py` に `predict_target_position_fractional` + `SAFE_INTERCEPT_HALF_STEP` 追加 (case1/2/3 にはない)

## Refactor 2026-04-29

`strategy.py` 内の mission ループを `_process_single_source_mission` /
`_process_multi_source_mission` / `_enforce_inventory_cap` に抽出。
出力 (action 系列) は不変、snapshot test pass 済み。
