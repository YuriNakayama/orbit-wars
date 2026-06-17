# iter3 result: H3 BC warm-start — REJECTED (harmful)

run_id: 20260608-062728...acbab5c / GPU: RTX 4090 / 5/20 iter (early-stopped) / config: h3_bc_warmstart.yaml

## 設定
case9_per_planet imitation best.pt から warm-start (`loaded=133 missing=0` 確認)、
kl_beta=0.5 で BC reference に anchor。dense差分報酬併用。noop(0-2)→full(3-19)。

## pod 運用の摩擦 (記録)
- dev pod `--case case7` が **family=imitation に誤分類** → onstart が reinforce 用
  BC weights pull をせず、代わりに kaggle_episodes (60GB) を pull 開始。
- 対処: onstart の dvc pull プロセスを kill + lock 解除 → ローカル DVC cache から
  `best.pt` を **scp 直送** (DVC pull は lock 競合で失敗)。launch_poc.sh に BC scp を
  組み込むべき (次の改善点)。

## 結果
| iter | opp | win | reward | ent | bc_kl |
|---|---|---|---|---|---|
| 0 | noop | 0.219 | -0.64 | 34.5 | 0.109 |
| 1 | noop | 0.281 | -0.62 | 24.7 | 0.104 |
| 2 | noop | 0.375 | -0.33 | 19.4 | 0.110 |
| 3 | baseline_jax_full | **0.062** | -2.60 | — | — |
| 4 | baseline_jax_full | **0.031** | -2.98 | — | — |

## 結論: REJECTED (harmful)
- **BC warm-start は case7 pipeline で逆効果**。BC init は **vs noop ですら 0.22**
  (from-scratch は同時点 0.72)、vs full は **0.03-0.06**（H1/H2 の 0.19 スタートより悪い）。
- 原因: case9_per_planet imitation policy が case7 の featurizer/decoder に**転移しない**
  (重みは load されるが、出力挙動が case7 推論経路で弱い)。kl_beta=0.5 anchor が
  その弱い policy に固定し、RL の脱出を妨げる。
- → BC checkpoint は case7 用でない。使うなら **同 featurizer の imitation 重み**が必要、
  かつ kl_beta=0 で自由に脱出させるべき。現状の case9 重みは不適。

## コスト: ~$0.08 (4090 ~7分、early stop)
