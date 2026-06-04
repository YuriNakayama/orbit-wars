# rulebase/case9 JAX port — result

> 記録: 2026-06-03 ~13:45 / 状態: done (Step1+非劣化) / 親: ../20260603_case8_jax_port/

## case9 = case8 + anti-ping-pong (stateful, 但し実運用で dormant)

- case9 vs case8 の差 = **anti-ping-pong cooldown** (reinforce の (src,dst) pair を
  1turn、harass target を 2turn lock、low-planet bypass)。**ターン跨ぎの launch 履歴**を
  module-level state で保持 = stateless な JAX `compute_actions(state, seat)` では
  原理的に保持不可。
- **但し実運用では dormant**: case9 Python == case8 Python を実測 (200 turn, **mismatch 0**)。
  cooldown が 1-2turn と短く同 (src,dst) 即再射が稀 + low-planet bypass のため発火せず。
  memory [[project_case9_anti_ping_pong_2026_05_06]] でも case9 は +5pp 不可・棄却済。

## 実装 + 検証

- case8 baseline_jax を case9 へ verbatim copy (cooldown は JAX に持てないが dormant)。
- 結合テスト作成、**foreground 4-game gate: JAX 2/4 = 50%** (非劣化)。

## 限界 (honest)

case9 の特徴 (anti-ping-pong) は **faithful には未 port** (stateless JAX の制約)。本 port は
cooldown OFF 相当 = case8 と同一挙動。実運用 dormant なので結合テストでは劣化として現れない
が、cooldown が効く稀な盤面 (collapse 期の連続 reinforce 等) では Python と乖離し得る。
case9 は元々棄却済施策につき、この近似で許容。

→ **case9 完了** (近似 port + 非劣化)。case4/case8/case9 が core 共有。
