# Reinforce/case5 — support_reward (iter5) RESULT

> 関連: iter5_plan.md / hypotheses.md / iter3_result.md (H4 ratio coef=1.0, 現行最良)
> run_id: 20260529-231332__feature-support-reward__7da3e2c__seed0 / commit: 7da3e2c / case: reinforce_case5_kaggle_jax_train_h7_ratio_clip
> 開始: 2026-05-29T23:13Z / 終了: 2026-05-30T03:06Z / wall-clock: 222 min / GPU: NVIDIA A100 80GB PCIe (COMMUNITY, $1.19/h offer→$1.39/h 実費) / コスト: ~$5.4
> 備考: RunPod 全 GPU 型に長時間障害発生 (~16h、consumer/DC全 phantom)。3090/4090 確保不能のためユーザー承認のもと H7 のみ A100 で実行 (one-off)、H3/H6 は consumer 復活待ち。

## Summary
H7 (ratio shaping の per-turn 報酬を `[-0.1, +0.1]` に band clip) は lite phase
last-10 = **0.8234** で、現行最良 H4 (ratio coef=1.0, 0.820) と **+0.34pp の差**。
この差は seed/サンプル variance 域 (n<300 結論禁止スキップに該当)、trend (+0.585 < H4 +0.668)
と max win_rate (0.867 < 0.906) は H4 比でやや弱め。事前見込みどおり、ratio 正規化で
ΔΦ が既に [-1,1] に収まり H4 で value_loss 0.0066 / approx_kl 0.005 と十分安定なため、
追加 clip による改善余地が小さい。**clip は不要と確認、H4 維持を推奨**。

## Numbers (学習ログのみ — 300 対戦 skip)

| metric | H7 ratio_clip=0.1 | H4 ratio (best) | note |
|---|---|---|---|
| last-10 win_rate (lite) | **0.8234** | 0.820 | +0.34pp (seed variance 域) |
| trend (slope×n) | +0.5847 | +0.668 | やや低下 |
| max win_rate (lite) | 0.8672 | 0.906 | −3.9pp |
| 20-iter chunk means (lite) | 0.255/0.38/0.45/0.56/0.59/0.67/0.72/0.73/0.76/0.80/0.85 | (H4 0.83 まで上昇) | 終盤 0.85 まで上昇 |
| last-10 value_loss | 0.00800 | 0.00660 | +0.0014 (微増) |
| last-10 approx_kl | 0.00267 | — | 健全 |
| max approx_kl (lite) | 0.00474 | 0.00500 | 健全 |
| iterations | 200 完走 | 200 完走 | = |
| runtime | 222 min (A100) | 126 min (4090) | A100 oversized・4090 より遅い |

## Diagnosis
- H4 の ratio shaping は Φ∈[0,1] で正規化済なので ΔΦ∈[-1,1]、coef=1.0 倍でも報酬は [-1,1] 域。
  序盤 spike 抑制を狙った clip=0.1 は通常域 signal を素通しさせるが、H4 で value_loss/approx_kl
  が既に十分安定 (0.0066 / 0.005) のため、clip による安定化マージンが小さい。
- last-10 +0.34pp の改善らしき差は n<300 では確定不可 (project rule)。trend と max は H4 比劣後で、
  「clip が小さく効いた」よりも「clip 不要なまま H4 並み」と読むのが素直。
- value_loss が H4 0.0066 → H7 0.0080 と微増したのは、clip による信号の不連続性が小幅にノイズを
  加えた可能性 (clip 境界での非可微分性)。trend 低下とも整合。
- A100 vs 4090 で runtime が 222min vs 126min と遅いのは A100 の GPU が JAX W4 spike の rollout
  サイズに対して oversized (vmap batch=128 はメモリ十分な環境ではどちらも一致するはず) で、
  メモリ帯域やランタイム差により逆転した可能性。コスト効率は 4090 が圧勝で、これは想定どおり。

## Decision
- 採否: **inconclusive (clip 不要を確認、H4 維持)** — n<300 ルールに従い win-rate 単独では結論不可。
  trend/max が H4 比劣後 + value_loss 微増の組み合わせから、改善は実質ないと判断。
- 次の一手: H7 系の deepen はしない。残る H3 (絶対保持数 dense 加算, 対照群) を消化、その後 H6
  (time bonus, H3 pair)。promote 候補は引き続き H4 (ユーザー承認後)。
  H3/H6 は consumer 3090/4090 限定 (A100 は H7 の one-off)、RunPod 復旧待ち。

## Artifacts
- model: `data/output/models/reinforce/case5_kaggle_jax_train_h7_ratio_clip/runs/20260529-231332__feature-support-reward__7da3e2c__seed0/best.pt`
- metrics: 同ディレクトリ `metrics.json` (history 200 iter 全 + iter197 で 0.8672 達成)
- best snapshots: best_i0/i2/i4/i147/i181/i186/i197.pt (RunPod onstart の S3 upload-on-improvement で永続化済)
