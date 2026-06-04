# rulebase JAX 全 case 退化検証 sweep — result

> 記録: 2026-06-03 ~14:55 / 状態: in_progress / user 指示「セルフ対戦で退化していないか検証」

## 目的

各 case の個別 gate は 4-6 game で noise 大 (case6 が 33%↔80% に振れた)。全 8 ported
case を **10 game/case (5 seed × 2 seat)** で JAX port vs その case Python を一括自己対戦し、
退化 (≈0勝) の有無を consolidated に判定。foreground 実行 (background は JAX hang)。

## 結果 (完走、SWEEP DONE)

各 case 10 game (5 seed × 2 seat)、JAX port vs その case の Python:

| case | JAX win/10 | 判定 |
|------|-----------|------|
| case1 | 8/10 = **80%** | ok |
| case2 | 7/10 = **70%** | ok |
| case3 | 7/10 = **70%** | ok |
| case4 | 7/10 = **70%** | ok |
| case6 | 5/10 = **50%** | ok |
| case7 | 6/10 = **60%** | ok |
| case8 | 7/10 = **70%** | ok |
| case9 | 6/10 = **60%** | ok |

**結論: 全 8 case で退化なし (degradation 皆無)**。最低でも 50% (case6)、平均 ~65%、
最高 80% (case1)。ユーザー懸念の「≈0勝」失敗モードは **どの case にも発生せず**。
[ok] = win-rate ≥ 20% (≈0勝でない) を全 case が clear。

### 個別 gate との対照 (小サンプル noise の確認)

| case | 個別 gate (4-16 game) | sweep (10 game) |
|------|----------------------|-----------------|
| case6 | 33%→62.5% (振れ大) | 50% |
| case7 | 67% | 60% |
| case8 | 50% | 70% |
| case4 | 50% | 70% |

個別 gate (特に n=4-6) は ±20pp 振れる。sweep (n=10) でも completely stable ではないが、
**全 case が非劣化レンジ (50-80%) に収まる**点は一貫。退化判定には十分。

## 速度に関する注記

sweep が遅い主因: 8 case がそれぞれ別 `compute_actions_jax_jit` を持ち、case 毎に jit
再 compile が走る。GPU vmap (case1 で 217 env-steps/s 実証) と違い、CPU 逐次 self-play
は 1 game ~15-60s。退化検証目的には現状の結果で十分 (大規模 eval は loop 原則で回避)。
