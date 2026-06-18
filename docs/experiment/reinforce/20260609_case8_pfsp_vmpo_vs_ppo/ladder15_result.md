# case8 ladder15 — 逆カリキュラム 強制schedule後退 RESULT

> 関連: ladder14_result.md / hypotheses.md
> run_id: 20260615-042347__feature-poc-v-mpo__fb82945__seed0 / commit: fb82945c
> case: reinforce_case8_vmpo_ladder15 / GPU: A100 80GB SECURE ($1.39/h)
> 開始: 2026-06-15T04:28Z / 早期停止: iter~28/70 (再compile非効率を確認し中断) / pod destroy済 (0確認)

## Summary

ladder14 で勝率gate (promote 0.55) が plateau から到達不能で warmup が後退しなかった
問題を、`retreat_every_strict=1` (strict段ごとに無条件後退) で修正。**強制後退は
機能した** — warmup が 300→275→250→225 と確実に降下し、strict段勝率も 0.307→0.339
→0.365 と warmup 降下に伴い上昇 (ladder14 の固定 0.33 plateau より良好)。しかし
warmup が毎 strict 段で新値になるため warmup advance graph が毎回 ~870s 再compile
され、**1時間で iter9 (warmup=225) 止まり**。warmup→0 到達に数時間 ($5-6) かかり
実験予算超過。機構は正しいが実装が非効率。

## Numbers

### strict 段 (warmup 降下に伴う学習勝率)

| iter | rc_warmup | win | reward | rollout 時間 |
|---|---|---|---|---|
| 1 | 275 | 0.307 | -0.680 | 911s |
| 4 | 250 | 0.339 | -0.671 | 870s |
| 9 | 225 | 0.365 | -0.560 | 832s |

- self_snapshot iter は ~19s (cached)、strict 段のみ毎回 ~870s = **warmup 新値の再compile**
- 用量反応は良好: warmup を下げる (序盤を多く戦わせる) ほど勝率↑・reward↑

### held-out

| iter | strict_v1 | full |
|---|---|---|
| 0 | 0.0 | 0.75 |

(iter10+ の held-out は再compile遅延で flush 前に停止)

## Diagnosis

`_advance_strict_self_jit(warmup_turns, seat)` が `warmup_turns` を **static (lru_cache
キー)** にしていたため、強制後退で warmup が変わるたび XLA が新 graph を ~870s かけて
再compile。ladder14 は warmup=300 固定で1回だけ compile していたので速かった。強制
後退の設計と static-warmup 実装が衝突した。

## Decision

- 採否: **rejected (実装非効率)** — ただし機構 (強制後退) は機能、用量反応も良好
- 次の一手: **ladder16 = warmup advance 単一compile化**。`warmup_turns` を traced
  scalar 化し、scan を static-max (RC_MAX_WARMUP=320) 固定 + 各 step を
  `t >= warmup_turns` で freeze する mask 方式に変更 (`opp_start_turn` と同じ
  traced-scalar パターン)。warmup 値が変わっても再compile ゼロ。config は ladder15
  と同一、rollout コードのみ修正の re-run。

## Artifacts

- model: `data/output/models/reinforce/case8_vmpo_ladder15/runs/20260615-042347__feature-poc-v-mpo__fb82945__seed0/best_i5_win0.7500.pt`
- metrics: 同 dir / metrics.json (iter 0-10)
