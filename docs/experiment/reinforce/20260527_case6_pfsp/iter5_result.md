# Reinforce/case6 — H6d horizon=50 (iter5) RESULT

> 関連: iter4_result.md (H5 棄却 0/30) / hypotheses.md
> run_id: 20260602-064748__feature-agent-pool-learning__08bd34a__seed0 / commit: 08bd34a0
> case: reinforce_case6_kaggle_jax_train_pool_v1_h50
> 開始: 2026-06-02 06:52 / 終了: 07:47 (55min 完走) / コスト: $0 (Kaggle T4x2 free)

## Summary

**H6d (horizon 500→50) は微改善のみ**:
- live baseline_v1 30戦 **1/30 = 3.3%** (Wilson CI 0.6-16.7%)
- pool_v1 (H5) の 0/30 → 1/30 で +3.3pp、純粋運の可能性高い
- 学習時の python_v1 iter は依然全敗 (iter 5-29 で 0/8 連続)

## 学習 metrics

| iter | opponent | rollout (s) | win | approx_kl |
|------|----------|------------:|----:|----------:|
| 0-4  | noop     | 55-60 | 0.250-0.500 | -0.0015 to 0.0012 |
| 5-29 | python_v1 | 97-135 | **0.000** | -0.0014 to 0.0015 |

- **noop 戦 win 低下** (pool_v1 は 1.000 だったが H6d は 0.250-0.500)
- horizon=50 が短すぎて noop 戦も決着しないため (本来勝率 1.0 のはず)
- python_v1 戦は依然 0/8 連続、approx_kl ~ 0 で PPO 停止

## Why 改善が小さかったか

H6d 仮説「horizon を短縮し reward 密度 10x で勾配信号回復」は半分しか正しくなかった:
- ✅ rollout 時間は 10x 短縮 (pool_v1 ~2900s/iter → H6d ~120s/iter)
- ❌ shaping reward も短縮中に累積しない (Δplanets が小さい時間内で起きる)
- ❌ そもそも 50 step では v1 戦の決着が見えないので reward = 0 が多い

つまり horizon 短縮は **PPO 計算量を減らせるが、reward signal 自体は薄いまま**。
本物 v1 が強敵な事実は変わらず、勝った episode を **構造的に observe できない**。

## Next

次の有望候補 (実装難度の昇順):

1. **H6c (shaping_coef 5.0)** — config 準備済 (`kaggle_jax_train_pool_v1_shaping5.yaml`、7090b788)
   horizon=500 のまま dense reward を 10x で signal 強化。即起動可能
2. **H6a (BC warm-start)** — case9_per_planet imitation weights から開始、
   初手で「勝てる」状態を確保。本命だが実装コスト高い
3. **H6b (3 段 curriculum)** — noop → baseline_jax_lite → python_v1。
   中間に勝てる相手 (jax_lite) を挟むことで勝った episode を observe させる
