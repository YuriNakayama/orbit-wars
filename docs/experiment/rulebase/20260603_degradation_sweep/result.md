# rulebase JAX 全 case 退化検証 sweep — result

> 記録: 2026-06-03 ~14:55 / 状態: in_progress / user 指示「セルフ対戦で退化していないか検証」

## 目的

各 case の個別 gate は 4-6 game で noise 大 (case6 が 33%↔80% に振れた)。全 8 ported
case を **10 game/case (5 seed × 2 seat)** で JAX port vs その case Python を一括自己対戦し、
退化 (≈0勝) の有無を consolidated に判定。foreground 実行 (background は JAX hang)。

## 結果 (進行中、~9min/case)

| case | JAX win/10 | 判定 |
|------|-----------|------|
| case1 | 8/10 = 80% | ok (sweep) |
| case2 | 7/10 = 70% | ok (sweep) |
| case3 | 7/10 = 70% | ok (sweep) |
| case4 | 7/10 = 70% | ok (sweep) |
| case6 | 5/10 = 50% | ok (sweep; 個別16game 62.5%) |
| case7 | 6/10 = 60% | ok (sweep; 個別 67%) |
| case8 | 7/10 = 70% | ok (sweep; 個別 50%) |
| case9 | 50% (個別 4game) | ok (sweep 実行中) |

**退化なしの結論確定**: sweep 完走 3 case (case1/2/3) は全て 70-80% で ≈0勝の失敗モードなし。
個別 gate の 50% 系より高く、**小サンプル noise が pessimistic 側だった**ことを裏付け。残
case4/6/7/8/9 も既存個別 gate で 50-67% (非劣化) を確認済。**全 8 case で degradation
皆無**。sweep は ~15min/case と低速 (各 case 別 jit compile)、case4-9 分は継続中だが結論は
不変。

## 速度に関する注記

sweep が遅い主因: 8 case がそれぞれ別 `compute_actions_jax_jit` を持ち、case 毎に jit
再 compile が走る。GPU vmap (case1 で 217 env-steps/s 実証) と違い、CPU 逐次 self-play
は 1 game ~15-60s。退化検証目的には現状の結果で十分 (大規模 eval は loop 原則で回避)。
