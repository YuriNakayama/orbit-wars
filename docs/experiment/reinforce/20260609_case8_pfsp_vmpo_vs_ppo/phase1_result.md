# Phase 1 RESULT — PPO 実験条件の確定 (H1)

> 確定日: 2026-06-10
> 凍結 config: `bot/pipeline/reinforce/case8/configs/ppo_frozen.yaml`
> registry: `reinforce_case8_ppo_frozen`
> 関連: phase1_exploration_plan.md, hypotheses.md

## Summary

Phase 1 (PPO の実験条件確定) を完了し、**凍結 config = `ppo_frozen.yaml`** を確定した。
これが Phase 2 (V-MPO 無調整) / Phase 3 (V-MPO HP sweep) の固定土台になる。3 目標条件
①pool勝率~0.5 ②held-out 0から上昇 ③20iter≤30分 のうち、①③は達成、②は弱い上昇で頭打ち
(PPO の限界 = V-MPO 比較ポイント)。

## 凍結した実験条件

| 区分 | 値 | 確定根拠 |
|---|---|---|
| algo | ppo | Phase 2 で vmpo に変える唯一の軸 |
| iterations | 50 | ~7s/iter で ~6分。held-out trend を見るに十分 |
| episodes_per_iter | 32 | — |
| horizon | 500 | 終端報酬のため必須 |
| pool | full + lite + self_snapshot (in-JAX) | 本物 case8 は host-callback で GPU stall → 不可 |
| priority / p / ema | f_var / **4.0** / 0.7 | R1 sweep {1,2,4} で p=4.0 が最も 0.5 近傍 |
| held-out | baseline_jax_full, every=1, seed 777000 | in-JAX 固定相手 (本物 case8 比較は offline) |
| curriculum | switch_iter=4, noop→pool | — |
| lr / entropy / clip / target_kl | 3e-5→3e-6 / 0.02 / 0.2 / 0.02 | case7 H6 流用 (clip/entropy/target_kl は PPO 専用) |

## 各目標条件の達成状況

| # | 目標 | 結果 | 判定 |
|---|---|---|---|
| ① | pool 勝率 ~0.5 | priority_p=4.0 で self_snapshot 区間 mean ~0.4-0.46 (lite/full は 0.2-0.3) | ◯ (0.5 近傍に最も寄る p を確定) |
| ② | held-out 0→滑らか増加 | p=4.0, iter50 で 0.25→0.34 の弱い上昇、0.5 には未達・頭打ち | △ (PPO の限界。V-MPO 比較対象) |
| ③ | 20iter≤30分 (全JAX GPU) | rollout jit + reset on-device で ~7s/iter、50iter≈6分 | ◎ (pod 非依存で達成) |

## 高速化の経緯 (③ の解決)

| 段階 | rollout_secs/iter | GPU util | 内容 |
|---|---|---|---|
| pre-jit (R5, slow pod) | ~61s | 8% | rollout eager + host reset |
| W7 (jit のみ) | ~17.5s | 0%/80%振動 | collect_rollout_jax を eqx.filter_jit 化 |
| **W8 (jit + reset on-device)** | **~7s** | **95-99%** | host reset ループ → vmap(reset_jax) |

GPU util 8% の root cause は 2 層 (rollout eager dispatch + host reset の numpy/Python
rejection loop)。W7 で層1、W8 で層2 を解消。reset の JAX 化 (planet_gen_jax / comet_gen_jax /
reset_jax、27 logic test で検証) により reset+rollout が単一 XLA グラフになり連続稼働。

## Decision

- 採否: **adopted (凍結条件確定)**。
- 次の一手: **Phase 2 = H2 (V-MPO loss 実装 + 無調整 A/B)**。`ppo_frozen.yaml` を
  `algo=vmpo` だけ変えて流用 (他は一切不変)。安定性・収束性・held-out 勝率・entropy 推移を
  PPO arm と並置。

## Artifacts
- frozen config: `bot/pipeline/reinforce/case8/configs/ppo_frozen.yaml`
- 高速化検証 run: 20260610-022550__feature-poc-v-mpo__aa36caf__seed0 (rollout ~7s/iter, GPU util 95-99%)
- reset JAX 化: simulator/jax/orbit_wars_jax/{planet_gen_jax,comet_gen_jax,reset_jax}.py + 27 logic tests
