# case8 ladder23 — 案B (T0=0段に dense信号 cliff_dense_boost=0.05) RESULT

> run_id: 20260616-145325__feature-poc-v-mpo__21513cb__seed0 / commit: 21513cb2
> 早期停止: iter6 (boost過大確定) / pod destroy済 (0確認)

## Summary
案B (T0=0段に dense差分報酬上乗せ + skip OFF) は機構として動作 (T0=0段が初めて
[updated]) したが、**cliff_dense_boost=0.05 が過大**: T0=0段で reward -26〜-27、
entropy 12-15 に崩壊 (3段とも)。dense が 500手累積で ±1終端を桁違いに支配 →
ladder13 の material支配落とし穴の再現。win は床 (0.02-0.04)。他段 (T0=225 0.80,
self 0.62-0.68) は entropy 47/37 健全で地力維持 — per-rung boost の隔離は正しく機能。

## Numbers (T0=0 dense段)
| iter | win | reward | entropy |
|---|---|---|---|
| 0 | 0.036 | -26.61 | 14.71 |
| 2 | 0.021 | -27.29 | 12.73 |
| 4 | 0.021 | -27.64 | 12.94 |

## Decision
- 採否: rejected (boost過大)。設計は健全 (隔離OK・skip OFFで学習走る) だが magnitude誤り。
- 次: ladder24 = cliff_dense_boost 0.05→0.01 (1/5)。dense が終端勝利を埋もれさせない
  大きさ (reward ~-5程度) に。他は同一。resume ladder22 best.pt。
