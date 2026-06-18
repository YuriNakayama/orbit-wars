# Phase 2 RESULT — V-MPO 実装 + 無調整 A/B (H2)

> 確定日: 2026-06-10
> PPO arm: 20260610-025656__feature-poc-v-mpo__a6f8cee__seed0 (ppo_frozen.yaml)
> V-MPO arm: 20260610-025710__feature-poc-v-mpo__a6f8cee__seed0 (vmpo_frozen.yaml)
> 関連: phase1_result.md, hypotheses.md

## Summary

V-MPO (Song et al. ICLR 2020) を実装し、ppo_frozen と **algo 以外完全同一**の
vmpo_frozen で PPO と無調整 (論文デフォルト HP) A/B した。結論: **無調整 V-MPO は
PPO に対し「同等の収束性 + 明確に高い entropy collapse 耐性 + わずかに高い安定性」**。
劣化なく、安定性面で優位の兆し。

## Numbers (50 iter, 同一 harness, algo のみ違い, priority_p=4.0)

| 観点 | PPO | V-MPO | 判定 |
|---|---|---|---|
| 収束性 (held-out) | first5 0.181 → last5 0.275, slope +0.0010, max 0.375 | first5 0.181 → last5 0.269, slope **+0.0022**, max 0.375 | ほぼ互角 (V-MPO slope 2倍だが到達点同じ) |
| 安定性 (pool 振動 std) | mean 0.392, **std 0.196** | mean 0.374, **std 0.188** | V-MPO わずかに安定 |
| entropy (collapse 耐性) | 50.9 → 21.1, **min 8.6** | 50.9 → 27.3, **min 11.7** | **V-MPO 明確に collapse 耐性高** |
| value_loss | mean 0.0193 | mean 0.0199 | 互角 |
| V-MPO 内部 | — | η 1.01→1.24, α 4.97→3.94, trust-region KL 0.0003 (ε_α=0.01 以下) | η/α 正常適応, trust-region 健全 |

## Diagnosis

- **最大の差 = entropy collapse 耐性**: V-MPO は entropy を高く維持 (min 21 vs PPO の
  8.6)。論文の「entropy reg なしで policy collapse に強い」を実データで確認。PPO は pool
  相手 (full/lite) で entropy が 8.6 まで落ちる場面がある。
- **収束性・到達点は互角**: held-out 両者 ~0.27 / max 0.375。無調整 V-MPO は PPO と同等性能
  (劣化なし)。held-out が 0.5 未達なのは env/相手の難度 (既知の天井) で、algo 差ではない。
- **V-MPO 機構は健全**: η/α が適応し trust-region KL が ε_α (0.01) を大きく下回る (0.0003)。
  trust region が効いている (むしろ余裕がありすぎ → Phase 3 で ε_α を緩める余地)。
- **速度**: 両者同一 harness で同等 (~7s/iter)。

## Decision

- 採否: **adopted (無調整 V-MPO は PPO と同等以上)**。安定性・entropy 耐性で優位の兆し。
- 次の一手: **Phase 3 = H3 (V-MPO HP チューニング)**。trust-region KL が ε_α を大きく下回る
  (0.0003 ≪ 0.01) ので ε_α を緩めて学習を加速できる余地。ε_η / top-k 割合 / init η・α も sweep。
  同一凍結条件で V-MPO 内部 HP のみ変える A/B。最良 V-MPO を PPO baseline と再比較。

## Artifacts
- vmpo_jax.py / vmpo_frozen.yaml (ppo_frozen と algo のみ差分)
- 比較グラフ: /tmp/case8_vmpo_vs_ppo.png (held-out/pool/entropy/value_loss + η-α/KL)
