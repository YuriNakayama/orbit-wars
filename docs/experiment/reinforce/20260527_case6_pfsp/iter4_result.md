# Reinforce/case6 — Real Python baseline_v1 opponent via host callback (iter4) RESULT

> 関連: iter4_plan.md / iter3_result.md / hypotheses.md
> run_id: 20260601-191229__feature-agent-pool-learning__06b6faf__seed0 / commit: 06b6faf3
> case: reinforce_case6_kaggle_jax_train_pool_v1
> 開始: 2026-06-01 19:17 / 終了: 2026-06-02 06:39 (約 11h22 完走) / コスト: $0 (Kaggle T4x2 free)
> 環境: Kaggle Kernel gpu-t4x2

## Summary (核心結論)

**仮説 H5 (本物 baseline_v1 opponent で train/eval gap 解消) → 棄却**。
学習後 rl_v6 vs baseline_v1 (live 30戦) **win_rate = 0/30 = 0.0%** (CI95 0-11.4%)。
iter3 H4 と同じ 0/30、改善なし。

## 学習 metrics

| iter | opponent | rollout (s) | win | policy_loss | approx_kl | entropy |
|------|----------|------------:|----:|------------:|----------:|--------:|
| 0    | noop     |  492 | **1.000** | 0.0005 | 0.0022 | 45.4 |
| 1    | noop     |  494 | 1.000 | -0.0010 | 0.0024 | 42.0 |
| 2    | noop     |  553 | 1.000 | -0.0026 | 0.0027 | 37.5 |
| 3    | noop     |  552 | 0.875 | -0.0013 | 0.0030 | - |
| 4    | noop     |  565 | 1.000 | -0.0012 | 0.0040 | - |
| **5** | python_v1 | 2981 | **0.000** | -0.0002 | 0.0002 | - |
| 6-19 | python_v1 | 1460-3395 | **0.000** | -0.0026 to 0.0011 | 0.0-0.0012 | - |

合計 runtime: 11h22 (40915s)、iter 5 以降平均 ~2300s/iter (host callback 重い)。
**iter 5 で v1 に切り替わった瞬間に勝率 0、以降 15 iter 連続で 0/8**。

## なぜ改善しなかったか (Analysis)

1. **Reward sparseness**: v1 戦は terminal ±1 のみ、horizon 500 で 99%+ ステップが reward 0
2. **Approx KL ~ 0**: PPO update が clip 範囲内で **実質停止** (勾配信号ゼロ)
3. **Entropy 低下**: noop 戦で no_op 寄りに collapse 済み、v1 戦で探索する余地なし
4. **opponent strength gap**: noop (random walk) と v1 (本物 LB897 ルール) の難度差が極端、
   curriculum 中間が欠落

仮説 H5 の前提「opponent を本物にすれば gap 解消」は **正しい必要条件だが不十分**:
本物 opponent でも reward 信号が薄ければ PPO は何も学べない。

## 学んだ事 (memory に保存すべき)

- **host callback opponent 自体は機能**: pure_callback (vmap_method='sequential') で
  本物 baseline_v1 の moves が正しく seat 1 に届く (iter 5 以降 win=0/8 は v1 が想定通り強い証拠)
- **Kaggle 9h 公称上限は実際 12h 程度**: 40915s = 11h22 が COMPLETE で commit された
- **iter ごとの S3 upload なしの場合、完走しないと全損**: 今回偶然 COMPLETE したが
  危機一髪 (rules `.claude/rules/command.md` に追記済)

## 次の方針 (iter5 plan へ)

H5 棄却後の有望候補 (採用順):

1. **BC warm-start (H6a)**: kaggle_episodes から baseline_v1 self-play data 収集 →
   policy を v1 の moves で初期化 → PPO で局所最適化。学習信号薄問題を root cause で解消
2. **3 段 curriculum (H6b)**: noop (5 iter) → **baseline_jax_lite** (10 iter) →
   python_v1 (10 iter)。中間に弱目 rule を挟む
3. **Dense shaping 強化 (H6c)**: shaping_coef 0.50 → 5.0、planets diff を強く reward 化
4. **Horizon 短縮 (H6d)**: 500 → 50、reward 密度を 10x

実装難度: H6a > H6c > H6b > H6d。短期検証は H6d (config 1 行変更) → H6c → H6b → H6a。
