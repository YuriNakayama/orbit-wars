# rulebase JAX 全 case 退化検証 sweep — result

> 記録: 2026-06-03 ~14:55 / 状態: in_progress / user 指示「セルフ対戦で退化していないか検証」

## 目的

各 case の個別 gate は 4-6 game で noise 大 (case6 が 33%↔80% に振れた)。全 8 ported
case を **10 game/case (5 seed × 2 seat)** で JAX port vs その case Python を一括自己対戦し、
退化 (≈0勝) の有無を consolidated に判定。foreground 実行 (background は JAX hang)。

## 結果 (進行中、~9min/case)

| case | JAX win/10 | 判定 |
|------|-----------|------|
| case1 | 8/10 = 80% | ok |
| case2 | 7/10 = 70% | ok |
| case3 | (実行中) | |
| case4/6/7/8/9 | (待機) | |

case1/2 とも非劣化 (≈0勝の失敗モードなし)。個別 gate の 50-89% より高め = 小サンプル
noise だったことを裏付け。残 6 case の完了後に全表更新。
