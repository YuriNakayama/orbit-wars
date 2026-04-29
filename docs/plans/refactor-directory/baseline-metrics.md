# ディレクトリリファクタリング — Baseline Metrics

**目的**: refactor 前の状態をスナップショットし、refactor 後との回帰検証に使う。

**測定日**: 2026-04-29 (実装着手時点)
**対象 commit**: `feature/refactor-directory` ブランチ HEAD

---

## 1. Snapshot Test 結果

実測 (refactor 前):

```
355 passed, 4 failed, 1 skipped (実行時間 198 秒)
```

**Failure / Skip (既存問題、refactor 範囲外)**:
- `tests/pipeline/imitation/case1/test_agent_integration.py::*` (2件) — `pipeline/imitation/case1/policy/weights.pt` が DVC 管理で local 不在
- `tests/pipeline/imitation/case1/test_agent_snapshot.py::*` (2件) — 同上
- `tests/pipeline/rulebase/case4/test_baseline_agent.py` skip — `case4/evaluation/snapshot_update.py` 未実行

**refactor 後の合格条件**: 上記 4 failed / 1 skipped が **悪化しない**こと (= 同一 failure のみ、新規 failure なし)。

Snapshot は以下 case にあり:
- `tests/pipeline/rulebase/case1/snapshots/`
- `tests/pipeline/rulebase/case2/snapshots/`
- `tests/pipeline/rulebase/case3/snapshots/`
- `tests/pipeline/rulebase/case5/snapshots/`
- `tests/pipeline/imitation/case1/snapshots/`

---

## 2. Selfplay 勝率 (記録のみ — Phase F-2 でフル測定)

実機 selfplay 50 戦は重い (約 30 分) ため、Phase F-2 (Step 15) で実施する。
現状のメモリ記録 (project_*) を参照値として残す:

| 条件 | 期待勝率 | ソース |
|------|---------|--------|
| baseline_v4 vs baseline_v3 (300戦) | ~70.3% | `project_case4_phase_results.md` |
| baseline_v5 vs baseline_v4 (自己対戦) | ~56% | `project_case5_validation.md` |
| baseline_v5 vs baseline_v1 | ~70% | `project_case5_validation.md` |

**refactor 後の合格条件**: 同条件で勝率 ± 5pp 以内 (Wilson 95% CI overlap)。

---

## 3. 1 ターン実行時間 (target ≦ 100ms)

実測は Phase F-2 で `uv run --directory backend python -m pipeline.imitation.case1.evaluation.replay_match` 経由で実施。

---

## 4. 既存テスト件数 (Phase A 追加前)

```
backend/tests/pipeline/
├── rulebase/case1: 2 tests + snapshots
├── rulebase/case2: 3 tests + snapshots
├── rulebase/case3: 3 tests + snapshots
├── rulebase/case4: 4 tests
├── rulebase/case5: 4 tests + snapshots
├── imitation/case1: 7 tests + snapshots
├── imitation/case2: 5 tests
└── imitation/case3: 3 tests
```

## 5. Phase A で追加したテスト

| ファイル | テスト数 |
|---------|---------|
| tests/pipeline/imitation/case1/test_geometry.py | 24 |
| tests/pipeline/imitation/case2/test_geometry.py | 24 |
| tests/pipeline/imitation/case3/test_geometry.py | 24 |
| tests/pipeline/imitation/case2/test_decoder.py | 4 |
| tests/pipeline/imitation/case3/test_decoder.py | 4 |
| tests/pipeline/rulebase/case1/test_core_geometry.py | 19 |
| tests/pipeline/rulebase/case1/test_core_physics.py | 22 |
| tests/pipeline/rulebase/case4/test_core_geometry.py | 19 |
| tests/pipeline/rulebase/case4/test_core_physics.py | 24 (case4 固有 fractional 含む) |
| **合計** | **164** |
