# case8 ladder24 — 案B boost縮小 (cliff_dense_boost=0.01) RESULT

> run_id: 20260616-153511__feature-poc-v-mpo__2fa4784__seed0 / commit: 2fa47848
> 早期停止: iter18 (案B 不成立確定 + full低下懸念) / pod destroy済 (0確認)

## Summary
cliff_dense_boost 0.05→0.01 で T0=0段 reward を -27→-7.9 に縮小 (dense支配緩和)。
しかし T0=0段 win は床のまま (0.005-0.042)、entropy ~13 崩壊継続 — **dense territory
報酬は「territory比改善」を学ぶが「勝利」に変換されない**。held-out: iter0 full 0.875
(campaign新高値) → iter10 full 0.75 に低下 + strict 1/64。cliff-dense が policy を
territory偏重に引っ張り full を僅かに害する兆候。案B 棄却。

## Numbers
| held-out | strict | full |
|---|---|---|
| iter0  | 0/64 | 0.875 |
| iter10 | 1/64 | 0.75 |

T0=0段 win: 0.031/0.005/0.016/0.042 (entropy ~13-15)。

## Decision
- 採否: rejected (案B dense-shaping、boost 0.05/0.01 とも strict不可)。
- strict 序盤は dense territory報酬では学習不能 (territory劣勢を減らしても strict の
  位置的優位が複利で効き勝てない)。
