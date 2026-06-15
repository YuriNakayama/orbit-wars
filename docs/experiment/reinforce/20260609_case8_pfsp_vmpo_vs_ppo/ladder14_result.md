# case8 ladder14 — 逆カリキュラム (Florensa+2017, strict自己対戦warmup) RESULT

> 関連: strict_heldout_result.md / hypotheses.md
> run_id: 20260615-030043__feature-poc-v-mpo__1ff3a9c__seed0 / commit: 1ff3a9cd
> case: reinforce_case8_vmpo_ladder14 / GPU: RTX 4090 SECURE ($0.69/h)
> 開始: 2026-06-15T03:00Z / 早期停止: iter~19/70 (機構未発火を確認し中断) / pod destroy済 (0確認)

## Summary

T0時間窓ラダーが strict を「終盤だけ」対面させ序盤を学習しない根本問題に対し、
逆カリキュラム (strict段の rollout を「両seat strict_v1で warmup手 自己対戦させた
中立な序盤盤面」からseed、勝率上昇で warmup→0 へ後退) を実装。GPUで動作したが、
**strict段の学習勝率が 0.31-0.38 で頭打ちし promote_win=0.55 に届かず、warmup が
300 のまま一度も後退しなかった** → 逆カリキュラムの核心 (warmup→0 で序盤を学習し
転移) が発火せず、bare strict_v1 held-out は 0.0156 (campaign共通の noise) に留まる。
仮説は機構未発火により棄却。ただし診断が次の一手 (強制schedule後退=ladder15) を直接
規定する。

## Numbers

### strict段 (warmup=300 盤面からの学習勝率)

| iter | win | rc_warmup |
|---|---|---|
| 1 | 0.307 | 300 |
| 4 | 0.349 | 300 |
| 9 | 0.375 | 300 |
| 10 | 0.307 | 300 |
| 17-18 | 0.339 / 0.375 | 300 |

→ n=5 strict段すべて [0.307, 0.375]、max 0.375 < promote 0.55。**後退ゼロ回。**

### held-out (bare, turn 0)

| iter | strict_v1 | baseline_jax_full | elo |
|---|---|---|---|
| 0 (ladder11 resume基準) | 0.0 | 0.703 | 1484 |
| 10 | 0.0156 (1/64) | 0.75 | 1469 |

- bare strict_v1: 0.0 → 0.0156 = campaign共通の noise レベル、**転移せず**
- full: 0.703 → 0.75 (+4.7pp、基準維持。RC は movement-detector baseline を損なわず)

## Diagnosis

逆カリキュラムは「warmup→0 へ後退して agent に序盤を段階学習させる」ことで初めて
turn-0 strict に転移する設計。だが win-gate (promote 0.55) が warmup=300 の plateau
(~0.35) から到達不能で、**後退が一度も起きず warmup=300 固定**。agent は「終盤を
strict有利盤面から」だけ訓練し続け、序盤を主体的に学習しないため held-out turn-0
strict は noise のまま。機構そのものは正しく動作 (start-state seed・rollout消費・
metrics記録すべてライブ確認済) だが、後退trigger の設計が plateau を越えられない。

## Decision

- 採否: **rejected (機構未発火)** — ただし診断が次手を規定
- 次の一手: **ladder15 = 強制schedule後退**。win-gate を bypass し strict段ごとに
  無条件で warmup を25手ずつ後退 (~12 strict段で 0 到達、~iter43 で turn-0 到達、
  残~27 iter を素strictで訓練)。「序盤を学習させれば turn-0 strict に転移するか」を
  本来検証する。他は ladder14 と同一 (reward=ladder11、resume=ladder11 best.pt) —
  後退trigger のみの A/B。実装: train_jax `retreat_every_strict` knob。

## Artifacts

- model: `data/output/models/reinforce/case8_vmpo_ladder14/runs/20260615-030043__feature-poc-v-mpo__1ff3a9c__seed0/best_i5_win0.7812.pt`
- metrics: 同 dir / metrics.json (iter 0-19)
