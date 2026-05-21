# reinforce/case1 — ppo_stabilization (iter1) ANALYSIS

> 関連: iter1_plan.md / iter1_result.md / hypotheses.md
> 分析対象: iter1 (H1 target_kl=0.05)、run_id `20260518-013025__feature-reinforce-learning-case0__f406f85__seed0`
> 分析モード: **skip mode** (`replay 分析を行わない` per hypotheses.md skip list)
> 入力: `iter1_result.md` + `metrics.json` (history 100 iter) のみ

## why_lost (仮説部分支持の原因)

primary metric (a) `max approx_kl < 0.1` を 27/100 iter で破った原因の候補:

1. **target_kl の判定粒度がepoch単位** (最有力): `ppo_update` の早期停止は当該 epoch の minibatch 平均 approx_kl を集計してから次 epoch を skip する設計。epoch 内 1 minibatch が approx_kl=1.5+ を出しても当該 update はそのまま走り、次 epoch の抑止にしか効かない。実装で `target_kl` 違反を **minibatch 単位** で break すれば spike 抑制効果は上がるはず (CleanRL は epoch 単位、SB3 は minibatch 単位の実装もあり)。
2. **lr=1e-4 が PPO+BC warm-start には大きすぎる**: per-update の Δw が大きく、特に entropy_coef=0.001 で探索 noise が混入した時に ratio=exp(Δlogp) が外れ値化。
3. **BC warm-start point が saturated**: BC 学習済 policy は p(no-op)≈1 の山に張り付いている (no_op_bias=8.0 baked-in)。advantage signal の小さなずれが log_p 上では大きく増幅されやすい。kl_beta=0.5 で anchor したが、初期 anchor 自体が saturated だと効果薄。

## what_worked

1. **early stop 機構そのもの**: epochs_run mean=1.43/2、57/100 iter で 2 epoch 目を skip。PPO update の累積量はちゃんと減っている。実装は正しい。
2. **bc_kl 安定化**: std=0.14 (cpu_stable_v1 0.31 比 0.44× 改善)。policy ドリフトの「変動」は緩和方向。target_kl=0.05 設定は副作用なしで継続価値あり。
3. **完走自体**: 100 iter × 16 ep を 25 304s で完走、approx_kl が爆発して NaN 化することなく学習継続。trust region の効果は出ている。

## where_to_focus_next

n<300 のため結論は出さない (per skip list)。次 iter で確認すべき点:

- **H2 (lr 1e-4→3e-5) を H1 と stacking** して、per-update Δ 縮小が minibatch スパイクを直接潰すか検証。期待: max approx_kl が 0.1-0.3 程度に圧縮。
- **代替案**: target_kl 判定を minibatch 単位 break に切替 (`ppo.py` で `if approx_kl_val > cfg.target_kl: break` を inner loop に追加)。実装コスト小、stack 評価可。H1 改修 deepen 仮説の候補。
- **bc_kl curve の単調性**: iter0 100 iter run の DVC pull 完了後に iter1 と直接比較。iter0 比 0.5× 改善が真に達成できているか確定させる。

## 信頼度の前提 (per skip list)

- 学習中の win_rate (n=16/iter) は **inconclusive 固定**。0.125 や 0.0625 は変動範囲内、参考値のみ。
- replay 由来の手筋分析は実施せず、すべて学習指標ベース。
- Kaggle publicScore は引用していない。
