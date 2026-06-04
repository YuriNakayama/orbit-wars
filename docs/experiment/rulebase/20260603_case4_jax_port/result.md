# rulebase/case4 JAX port — result

> 記録: 2026-06-03 ~13:30 / 状態: done (Step1+非劣化) / 親: ../20260603_case8_jax_port/

## case4 = case8 (predict-cache 差のみ) → JAX port は free

- case4 vs case8 baseline 差分は **agent.py の reset_predict_cache() 呼び出し +
  core/physics.py 実装のみ** = predict-cache (perf 最適化、挙動透過)。
- config delta **0 行** (case4 config == case8 config)。
- **検証: case4 Python == case8 Python を実測** (200 turn, mismatch 0)。
  → JAX core は predict-cache を持たない (Python 側 memoization) ので case4/case8 で
  同一 JAX agent。case8 baseline_jax を **verbatim copy** で case4 に再利用可。

## 実装 + 検証

- case8 baseline_jax/core_jax を case4 へ copy (cross-case independence: import 不可)。
- 結合テスト `tests/e2e/pipeline/rulebase/case4/test_agent_jax_identity.py` 作成。
- **foreground 4-game gate: JAX 2/4 = 50%** (case8 と同一、非劣化)。import clean。

→ **case4 完了** (Step1 + 非劣化、case8 の harass 込み core 共有)。1 tick で 2 case目。

## lineage 残 (case2/3/6/7/9) の scope

| case | config delta vs case8 | files differ | 備考 |
|------|----------------------|--------------|------|
| case2 | 18 | 10 | |
| case3 | 31 | 11 | rollout.py 追加 |
| case6 | 33 | 7 | stay.py 追加 |
| case7 | 68 | 8 | stay.py 追加 (t14 trap, memory) |
| case9 | 18 | 6 | anti_ping_pong (棄却済, memory) |

これらは case4 のような free copy でなく、config delta + 新 module の port が要る。
tick 毎に config 適用 → foreground gate → 必要なら mission 追加、で進める。
