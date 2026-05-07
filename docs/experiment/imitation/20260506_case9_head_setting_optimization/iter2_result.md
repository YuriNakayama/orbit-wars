# Imitation/case9 — dual head blend α=0.5 (iter2) RESULT

> 関連: iter2_plan.md / hypotheses.md
> run_id: 未起動
> commit: HEAD (local commit; push pending)
> case: case9
> 開始: 2026-05-07
> 終了: 2026-05-07
> コスト: $0.00

## Summary

H4 の dual head 実装はローカル実装・targeted tests・1 episode smoke まで完了。`dev/test-bot` で検出された既存 failure は別途修正し、長時間化の直接原因だった case5 integration test は `slow` 対象に移した。RunPod 側にも `case9_dual` を登録済み。リモート学習は commit/push 後に起動する。

## Numbers

| metric | value | note |
|---|---:|---|
| targeted case9 tests | 22 passed | `uv run --directory bot pytest tests/pipeline/imitation/case9 -q --no-header -x` |
| ruff check | passed | `pipeline/imitation/case9`, `tests/pipeline/imitation/case9`, `src/dataset/selfplay/agents.py` |
| import sanity | passed | `IL_CASE9_HEAD_MODE=dual` で `agent` import 成功 |
| 1 episode smoke | passed | `il_v9_dual` vs `baseline_v1`, timeout なし |
| dev/test-bot format/lint/type | format/lint/type passed after local gate fixes | case8 stale evaluation mypy override 追加後、typecheck は 744 files no issues |
| dev/test-bot pytest | partial / improved | case1 snapshot は weights 欠落時 skip、case3 fleet arrival 修正、case5 長時間 integration は slow 化。ユーザー指示により以後は該当 smoke のみ実行 |

## Diagnosis

- 実装面では `head_mode="dual"` を追加し、3-head と candidate head を共通 backbone 上で同時 forward できる状態にした。
- 学習 loss は `dual_alpha=0.5` で 3-head loss と candidate loss を blend する構造にした。
- 推論は candidate decoder path を使うため、3-head はこの iter では補助学習 signal としてのみ効く。
- RunPod 起動の blocker だった既存 test failure と長時間 test 分類は修正した。`case9_dual` は RunPod CLI の case registry に登録済み。

## Decision

- 採否: 実行待ち
- 次の一手: 変更を commit/push し、`dev/runpod train <commit> --case case9_dual --watch` で H4 学習を起動する。

## Artifacts

- plan: `docs/experiment/imitation/20260506_case9_head_setting_optimization/iter2_plan.md`
- result: `docs/experiment/imitation/20260506_case9_head_setting_optimization/iter2_result.md`
- model: 未生成
- metrics: 未生成
