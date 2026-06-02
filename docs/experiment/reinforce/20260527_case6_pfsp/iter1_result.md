# Reinforce/case6 — PFSP self-snapshot opponent (iter1) RESULT

> 関連: iter1_plan.md / hypotheses.md
> run_id: 20260527-145442__feature-agent-pool-learning__fb36504__seed0 / commit: fb36504 / case: reinforce_case6_kaggle_jax_train_h1
> 開始: 2026-05-27 14:54 / 終了: ~20:01 (200 iter 完走) / コスト: ~$7.1 ⚠️ (A100 80GB $1.39/h × 5.1h、cap $1.5 を大幅超過)

## Summary

H1 (PFSP の土台 = self_snapshot opponent 経路) は **配線として完全に成立**した。
200 iter が NaN/発散なく完走し、`self_snapshot` opponent (frozen iter0 snapshot を
決定論で対戦) が rollout に正しく組み込まれ、best.pt/metrics も S3 に退避された。
**ただし win_rate は学習を通じてほぼ 1.0 で飽和**し、採否の主軸である reward trend
としては **uninformative**。原因は plan のリスク欄通り「凍結 iter0 snapshot が弱すぎて
相手にならない」こと。PFSP の本質 (相手も強くなる) は H2 (pool + 周期更新) を待つ。

## Numbers (学習ログのみ — 300 対戦 skip, n<300 結論不可)

| metric | first 10 iter | last 10 iter | note |
|---|---|---|---|
| win_rate (vs opponent) | 0.905 | 0.988 | iter0-4=noop, iter5-199=self_snapshot |
| win_rate (self_snapshot phase) | 0.847 | 0.988 | 195 iter、ほぼ飽和 |
| win_rate trend slope | — | +0.00034/iter | ほぼ平坦 (飽和) |
| value_loss | 0.104 | 0.052 | 半減 = 価値推定は学習進行 |
| entropy | 46.1 | 95.3 | **増加** = policy が拡散 (収束せず) |
| approx_kl | 0.0023 | 0.0028 | trust region 内で安定 |
| best_win_rate | — | 1.0 | best_i98 |

- ② vs 初期 snapshot 勝率 = 上記 self_snapshot phase そのもの (相手 = 初期 snapshot)。
- ③ vs baseline_v1 20 戦 = **未実施** (win_rate 飽和で H1 配線判定には不要、かつ
  n<300 で結論不可。H2 以降で pool 化した後にまとめて評価する)。

## Diagnosis

- **配線は成功**: OPPONENT_SELF_SNAPSHOT=3 + lax.switch 4 分岐 + opp_model thread が
  vmap/scan 下で 200 iter 完走。unit test 5 件 + smoke + 本番完走で多重に検証済。
- **信号が無意味**: 相手 (iter0 frozen) が固定で弱いため、agent は数 iter で勝ち切り
  win_rate≈1.0 に張り付く。entropy 増加は「勝てるので policy を絞る圧力が無い」状態。
  → self_snapshot を **単独 late 相手にするのは PFSP として不適**。pool 化と周期更新
    (H2) で「適度に強い相手」を供給して初めて学習圧が生まれる。
- **コスト超過が重大**: self_snapshot は相手も毎 step model forward する分 rollout が
  ~2 倍重く、200 iter が 5.1h に膨張。`dev/runpod status` の est_total=$0.000 バグで
  cost ガードが発火せず ~$7.1 を消費 (cap $1.5 の ~4.7 倍)。要メモリ化。

## Decision

- **採否: inconclusive** — 配線は成立 (土台として H2 へ進める) だが、win_rate 飽和で
  「より強い agent になったか」は判定不能。n<300 + 飽和信号のため結論は出さない。
- **次の一手 (H2)**: snapshot を iter0 固定でなく **K iter ごとに pool 追加 + late を
  pool からサンプリング**。相手が学習に追従して強くなり、勝率が中間域に収まることで
  reward trend が意味を持つ。baseline_jax_full も late に混ぜる。
- **コスト対策 (必須)**: ① self_snapshot 系は rollout 2 倍重を見込み iterations を
  100 に半減 or episodes_per_iter を 64 に、② A100 fallback 時は事前に runtime 見積を
  上方修正、③ est_total バグの cost ガードを当てにせず uptime ベースで手動監視。

## Artifacts
- model: `data/output/models/reinforce/case6_kaggle_jax_train_h1/runs/20260527-145442__feature-agent-pool-learning__fb36504__seed0/best.pt`
- best (iter98, win=1.0): `.../best_i98_win1.0000.pt`
- metrics: `.../metrics.json` (200 iter history)
- logs: `.../train.log` / `onstart.log` / `gpu.log`
