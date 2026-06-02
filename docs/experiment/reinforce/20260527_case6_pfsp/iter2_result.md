# Reinforce/case6 — PFSP snapshot pool + periodic refresh (iter2) RESULT

> 関連: iter2_plan.md / iter1_result.md / hypotheses.md
> run_id: 20260528-005806__feature-agent-pool-learning__36982a3__seed0 / commit: 36982a3 / case: reinforce_case6_kaggle_jax_train_h2
> 開始: 2026-05-28 00:58 / 終了: 01:59 (100 iter 完走) / コスト: ~$0.70 (RTX 4090 $0.69/h × 1.0h) ✅ H1 の $7.1 から大幅改善

## Summary

H2 (snapshot pool + 周期更新 + baseline_jax_full 混合 late) は **H1 の win_rate 飽和を
解消**した。H1 では frozen iter0 相手で last10=0.988 に張り付いていたのが、H2 では
last10=0.661 / overall=0.614 と中間域に収まった。決定打は **late に baseline_jax_full を
50% 混ぜたこと** — full は手強く (vs full 平均 win=0.274)、持続的な学習圧を供給した。
さらに **vs full の win_rate が iter を追って 0.138→0.359 と上昇** (slope +0.0027/iter)、
agent が強いルール相手に勝てるよう学習が進んだことを示す。PFSP の前提 (相手が強くなる
ことで学習が進む) が初めて機能。ただし n<300 のため結論は inconclusive。

## Numbers (学習ログのみ — 300 対戦 skip, n<300 結論不可)

| metric | H1 (iter1) | H2 (iter2) | 解釈 |
|---|---|---|---|
| win last10 | 0.988 (飽和) | **0.661** | 飽和解消、中間域 |
| win overall | 0.976 | **0.614** | 同上 |
| late iters in mid-band [0.3,0.7] | ~0% | **23%** | 適度な難度の相手が増えた |
| entropy first→last10 | 46→97 (暴走) | **38→47** (有界) | policy 崩壊せず |
| value_loss | 0.10→0.05 | 0.20→0.20 | 相手が多様で価値推定が難化 |
| vs self_snapshot (pool) | — | win=0.828 (n=55) | 過去自分は概ね上回る |
| **vs baseline_jax_full** | — | **win 0.138→0.359 (slope +0.0027/it)** | **強い相手に勝てるよう学習進行** |
| コスト | $7.1 ⚠️ | **$0.70** ✅ | 軽量化 + 4090 で 1/10 |

- ② vs 初期 snapshot 勝率 = vs self_snapshot phase (pool 経由)。
- ③ vs baseline_v1 20 戦 = 未実施 (win_rate signal が明確、n<300 で結論不可のため H4/H5 後にまとめて 300 戦予定)。

## Diagnosis

- **pool 化で相手が学習に追従** → win が 1.0 に張り付かず中間域維持。H1 で予測した通り。
- **baseline_jax_full の混合が学習圧の主因**: vs self_snapshot は 0.828 と高め (過去自分は
  surpass 済) だが、vs full は 0.274 と低く、ここが伸びしろ。full vs の上昇トレンド
  (+0.0027/iter) が「より強い agent になっている」直接証拠。
- **entropy が有界 (38→47)**: H1 の暴走 (46→97) と対照的。勝てない相手がいるので policy が
  絞られ続け、健全な学習ダイナミクス。
- **コスト対策が奏功**: iterations 100 / episodes 64 + RTX 4090 で $0.70。H1 反省の実証。

## Decision

- **採否: inconclusive (positive-leaning)** — メカニズムは明確に機能 (飽和解消 + vs full
  上昇トレンド)。ただし n<300 + win_rate ベースのため断定不可 (skip list)。土台として
  H4/H5 (PFSP 優先度 sampling) へ確実に進める価値あり。
- **次の一手 (H4)**: vs full が 0.359 止まりで伸びしろ大 → PFSP `f_hard=(1−x)^p` で
  「勝てない相手 (= full や強い pool snapshot)」を優先 sampling し、難敵への学習を加速。
  uniform mix (H2) → 優先度 sampling (H4) で vs full の到達点が上がるかを検証。
- **300 戦評価のタイミング**: H4/H5 完了後に rl_v6 vs baseline_v1 / baseline_jax_full /
  rl_v3 を 300 戦 (例外条件) で測り、PFSP 系が実際に強い agent を生んだか確定。

## Artifacts
- model: `data/output/models/reinforce/case6_kaggle_jax_train_h2/runs/20260528-005806__feature-agent-pool-learning__36982a3__seed0/best.pt`
- metrics: `.../metrics.json` (100 iter history)
- logs: `.../train.log` / `onstart.log` / `gpu.log`
