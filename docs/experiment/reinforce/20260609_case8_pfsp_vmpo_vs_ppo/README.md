# reinforce/case8 — V-MPO vs PPO 比較実験 (総括)

> 期間: 2026-06-09 〜 06-10 / 状態: completed
> 目的: PFSP (自己 snapshot + in-JAX rulebase pool, f_var で勝率0.5付近を選択) +
> 固定相手 held-out 評価の枠組みで、**V-MPO と PPO を安定性・収束性の観点から比較**する。

## 結論 (3行)

1. **無調整 V-MPO は PPO と同等以上** — 収束性は互角、**entropy collapse 耐性と安定性で V-MPO がわずかに優位**。
2. **V-MPO は HP チューニング不要でロバスト** — ε_α を緩めても改善せず、論文デフォルト (0.01) が最良。
3. held-out 到達点 (~0.4) の頭打ちは **algo でなく env (固定相手の難度) 起因** — どちらの algo でも 0.5 未達。

## 実験設計 (A/B 規律)

- **凍結条件** (`ppo_frozen.yaml`): f_var priority_p=4.0, pool=full+lite+self_snapshot (in-JAX),
  held-out=baseline_jax_full (in-JAX 固定相手), 50 iter, lr 3e-5→3e-6, entropy 0.02。
- PPO arm と V-MPO arm は **algo 以外完全同一** (`vmpo_frozen.yaml` は algo のみ差分)。Phase 3 の
  sweep も V-MPO 内部 HP のみ変更 — **実験条件を変えたら比較不能** という規律を厳守。
- 本物 rulebase case8 (python_v8) は host-callback で GPU stall するため学習ループに入れず、
  in-JAX proxy (baseline_jax_full) を held-out yardstick に使用 (本物比較は offline paired が前提)。

## Phase 別の結果

| Phase | 内容 | 結論 | doc |
|---|---|---|---|
| 1 | PPO 実験条件確定 + 高速化 | 凍結 config 確定。rollout jit (W7) + reset on-device (W8) で **61s→7s/iter, GPU util 8%→95-99%** | phase1_result.md |
| 2 | V-MPO 実装 + 無調整 A/B | **V-MPO ≈ PPO**。entropy 耐性 (min 11.7 vs 8.6) と安定性 (pool std 0.188 vs 0.196) で V-MPO 優位、収束性互角 | phase2_result.md |
| 3 | V-MPO HP (ε_α) sweep | **ε_α=0.01 (default) が最良**。緩めると劣化。trust-region KL は bound 非追従 (lr/学習量律速) | phase3_result.md |

## 主要メトリクス (50 iter, 同一 harness)

| | held-out last5 | held-out max | pool std (安定性) | entropy min (collapse 耐性) |
|---|---|---|---|---|
| PPO | 0.275 | 0.375 | 0.196 | 8.6 |
| V-MPO (ε_α=0.01) | 0.269 | 0.375 | **0.188** | **11.7** |
| V-MPO ε_α=0.05 | 0.231 | 0.375 | 0.193 | 10.3 |
| V-MPO ε_α=0.10 | 0.194 | 0.406 | 0.208 | 10.1 |

## 副産物 (本実験で構築した基盤)

- **env reset の JAX 化** (`simulator/jax/orbit_wars_jax/{planet_gen_jax,comet_gen_jax,reset_jax}.py`):
  惑星/コメットの rejection sampling を `lax.while_loop` + 固定バッファで JAX-native 化。jit/vmap で
  reset を rollout グラフ内に取り込み (host reset ボトルネック解消)。27 logic test (構造/分布/同一楕円
  での host vs JAX 数値一致)。vendor RNG byte-parity は意図的放棄、旧 reset.py は保持。
- **rollout jit 化** (`rollout_jax.py`): vmap(scan) を eqx.filter_jit。
- **V-MPO 実装** (`vmpo_jax.py`): L_η + L_π + L_α + value MSE、η/α 学習可能スカラー。ppo_jax と
  loss 以外共有。
- **`dev/runpod metrics`**: 学習中にローカルから per-iter metrics を見る CLI (観測性 REQ1)。

## 残課題 (任意)

- ε_η / top-k 割合の sweep (改善余地は薄い見込み — KL が lr 律速のため)。
- 最良 V-MPO (ε_α=0.01) を **本物 rulebase case8 と offline paired 300戦**で最終確認 (env 天井の
  絶対値を本物相手で測る)。
- lr / 学習量を上げる方向 (ただし PPO と共通条件のため A/B 規律の外、別実験として)。

## run_id 一覧

| arm | run_id |
|---|---|
| Phase1 凍結検証 (W8) | 20260610-022550__feature-poc-v-mpo__aa36caf__seed0 |
| Phase2 PPO | 20260610-025656__feature-poc-v-mpo__a6f8cee__seed0 |
| Phase2 V-MPO (ε_α=0.01) | 20260610-025710__feature-poc-v-mpo__a6f8cee__seed0 |
| Phase3 ε_α=0.05 | 20260610-033732__feature-poc-v-mpo__a0d95c4__seed0 |
| Phase3 ε_α=0.10 | 20260610-033748__feature-poc-v-mpo__a0d95c4__seed0 |
