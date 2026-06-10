# Phase 3 RESULT — V-MPO HP チューニング (H3)

> 確定日: 2026-06-10
> sweep: V-MPO eps_alpha ∈ {0.01, 0.05, 0.10}
> runs: eps_a=0.01 (Phase2 vmpo_frozen 流用) / 0.05 (20260610-033732__…a0d95c4) / 0.10 (20260610-033748__…a0d95c4)
> 関連: phase2_result.md, hypotheses.md

## Summary

V-MPO の trust-region KL bound ε_α を {0.01, 0.05, 0.10} で sweep した (凍結条件、
ε_α のみ変更の A/B)。結論: **ε_α=0.01 (論文デフォルト) が最良、緩めると劣化**。
Phase 2 の仮説「α が学習を過拘束 (KL 0.0003 ≪ ε_α)」は **外れ** — ε_α を緩めても
trust-region KL は ~0.0004 で一定 (bound に追従せず)。KL が小さいのは α の clamp で
なく policy 更新が intrinsically 小さい (lr/学習量律速) ため。クリーンな負の結果。

## Numbers (50 iter, 凍結条件, eps_alpha のみ違い)

| ε_α | held-out last5 | max | slope | pool std (安定性) | entropy min | trust-region KL | α 推移 |
|---|---|---|---|---|---|---|---|
| **0.01** (default) | **0.269** | 0.375 | **+0.0022** | **0.188** | 11.7 | 0.0003 | 4.97→3.94 |
| 0.05 | 0.231 | 0.375 | +0.0015 | 0.193 | 10.3 | 0.0004 | 4.97→3.96 |
| 0.10 | 0.194 | 0.406 | +0.0003 | 0.208 | 10.1 | 0.0004 | 4.97→3.93 |

## Diagnosis

- **ε_α=0.01 が最良**: held-out last5・slope・pool std すべてで default が勝ち。ε_α を
  緩めるほど held-out 収束が悪化 (0.269→0.231→0.194) し安定性も悪化 (std 0.188→0.208)。
- **仮説外れ (重要)**: trust-region KL は ε_α を上げても ~0.0004 で一定で bound (点線) に
  全く近づかない。= KL を抑えているのは α の clamp ではなく、policy 更新自体が小さいこと
  (lr 3e-5 / 50 iter の有効ステップが律速)。Phase 2 で「α 過拘束」と読んだのは誤り。
- **示唆**: V-MPO の改善余地は ε_α (trust-region) でなく lr / 学習量 / env 難度側。ただし
  これらは凍結条件 (PPO と共通) を変えるため Phase 3 (V-MPO 内部 HP のみ) の枠外。

## Decision

- 採否: **ε_α=0.01 (default) を採用**。HP チューニングで改善せず = V-MPO は無調整で
  ロバスト (チューニング不要)。
- 次の一手 (任意): ε_η / top-k の sweep で完全性を期す、または最良 V-MPO を本物 case8 と
  offline paired 300戦で最終確認。ただし当初目的 (V-MPO vs PPO の安定性・収束性比較) は
  Phase 2+3 で達成済み。

## Artifacts
- phase3_eps_alpha_{005,01}.yaml (vmpo_frozen と eps_alpha のみ差分)
- sweep グラフ: /tmp/case8_phase3_eps_alpha.png
- 実験総括グラフ: /tmp/case8_experiment_summary.png
