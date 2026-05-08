# Imitation/case9 — 3 head families behavior comparison (iter2) RESULT

> 関連: iter2_plan.md / hypotheses.md
> run_id: 実行中
> commit: 未確定
> case: case9_template_ships + 既存 case9_three_head / case9_candidate_ships 比較
> 開始: 2026-05-08
> 終了: 実行中
> コスト: 集計中

## Summary

ユーザー指示により、iter2 は dual head ではなく **3-head / candidate×ships / template×ships** の 3 パターン比較へ書き換えた。現時点では `template_ships` head 実装・config・registry・test 対応を進めている。

## Numbers

未集計。

| variant | train/val curve | local behavior | note |
|---|---|---|---|
| 3-head | 既存runあり、再集計予定 | 既存10epは0/10 | `val_target_acc` 軸 |
| candidate×ships | 既存runあり、再集計予定 | 既存10epは0/10 | `val_cand_fire_acc` 軸 |
| template×ships | 未学習 | 未評価 | 新規 `head_mode=template_ships` |

## Diagnosis

未実行。

## Decision

- 採否: 実行中
- 次の一手: targeted tests → smoke → template_ships 学習 → 3者の曲線/ローカル挙動比較。

## Artifacts

- plan: `docs/experiment/imitation/20260506_case9_head_setting_optimization/iter2_plan.md`
- result: `docs/experiment/imitation/20260506_case9_head_setting_optimization/iter2_result.md`
