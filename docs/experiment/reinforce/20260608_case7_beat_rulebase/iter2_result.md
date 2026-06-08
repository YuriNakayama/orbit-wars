# iter2 result: H2 dense differential reward — REJECTED (marginal)

run_id: 20260608-060947...a1c5e8b / GPU: RTX 4090 / 12/20 iter (early-stopped) / config: h2_dense_diff.yaml

## 設定
dense差分報酬: `c·(mine-enemy)/(mine+enemy)` (ship+planet, c=0.003, 累積±1.5)。
従来の dense=mine_count(溜め込み助長) を「相手を上回る」差分に変更。
schedule: noop(0-2) → baseline_jax_full(3-19)。h500/batch32, ratio PBRS併用。

## 結果
| stage | iters | opp | win 推移 | 平均 |
|---|---|---|---|---|
| noop | 0-2 | noop | 0.72/0.72/0.78 | 0.74 |
| target | 3-12 | baseline_jax_full | 0.19/0.31/0.375/0.16/0.28/0.28/0.16/0.375/0.22/0.09 | **0.260** |

- reward は dense項で正方向に押されるが(noop期 1.2-1.7)、full戦は -1.2〜-2.4。
- vs full: max 0.375, mean 0.260, **学習 trend なし**(振動)。entropy 16→10 collapse 継続。
- 前 baseline(~0.22) / H1(~0.32) と **有意差なし**。早期に iter12 で打ち切り。

## 結論: REJECTED (marginal)
- dense差分報酬は reward を整形するが **plateau を破れない**。max 0.375 は H1 と同程度。
- **H1(opponent)とH2(reward)が同じ失敗形** = 振動 ~0.3 + entropy collapse。
  → ボトルネックは opponent でも reward shaping でもなく、**from-scratch PPO の探索/初期化**。
  20iter では policy が早期に mediocre 戦略へ collapse し、rulebase-beating 行動に到達しない。
- → 次は **H3 BC warm-start**: imitation 学習済 policy から開始し RL refine。entropy collapse from-scratch を回避。
  (`cpu_stable_v1.yaml` が参照する case9_per_planet imitation best.pt を warm-start に使用、DVC pull 可)

## コスト: ~$0.12 (4090 ~11分)
