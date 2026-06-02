# Reinforce/case5 — support_reward (iter3) RESULT

> 関連: iter3_plan.md / hypotheses.md / iter2_result.md
> run_id: 20260528-004854__feature-support-reward__8e2e4a3__seed0 / commit: 8e2e4a3 / case: reinforce_case5_kaggle_jax_train_h4_ratio_coef1
> 開始: 2026-05-28T00:48Z / 終了: 2026-05-28T03:00Z / wall-clock: 126.5 min / GPU: RTX 4090 (SECURE) / コスト: ~$1.5

## Summary
H4 (ratio 係数 sweep, shaping_coef 0.50→1.0) は **支持された**。lite phase last-10 = **0.820**、
H2 (coef=0.50, 0.763) を **+5.6pp** 上回り採用しきい値 (+3pp) を満たす。trend +0.668・max 0.906 で
chunk は 0.824→0.830 と依然上昇傾向、plateau なし。value_loss 0.0066 / max approx_kl 0.005 と
over-shaping の兆候なし。ratio の [0,1] 正規化が大きな headroom を与え、shaping 強化が素直に効いた。
**coef=1.0 を coef=0.50 より採用**。

## Numbers (学習ログのみ — 300 対戦 skip)

| metric | H4 ratio coef=1.0 | H2 ratio coef=0.50 | note |
|---|---|---|---|
| last-10 win_rate (lite) | **0.820** | 0.763 | **+5.6pp** |
| trend (slope×n) | +0.668 | +0.651 | 同等以上 |
| max win_rate (lite) | **0.906** | 0.813 | +9pp |
| 20-chunk means (lite) | 0.207/0.355/0.458/0.575/0.604/0.699/0.746/0.764/0.824/0.830 | (0.77 で頭打ち気味) | 依然上昇 |
| max approx_kl | 0.005 | 0.0047 | trust region 健全 |
| value_loss (last-10) | 0.0066 | 0.005 | 微増だが極小維持 |
| iterations | 200 完走 | 200 完走 | exit_0 |
| runtime | 126.5 min (4090) | 215.8 min (A100) | 4090 で高速・低コスト |

## Diagnosis
- ratio potential は [0,1] 正規化で ΔΦ が小さい (≤1/turn) ため、coef=0.50 では shaping が弱め
  だった。coef=1.0 に倍増しても value_loss はほぼ不変 (0.005→0.0066) で over-shaping にならず、
  より強い勾配が収束を押し上げた。diff mode で coef=1.0 が over-shaping (value_loss 0.47) だった
  のとは対照的 — 正規化された ratio は係数に対して頑健。
- chunk が最終 0.830 でまだ上昇中 → coef をさらに上げる (1.5/2.0) 余地、または iter 数増の余地あり。

## Decision
- 採否: **adopted (n<300 で確定保留だが H2 比 +5.6pp で noise floor 超過)** — ratio coef=1.0 が
  現状ベスト構成 (lite last-10 0.820)。case5 support reward の現行最良。
- 次の一手: coef をさらに上げる sweep (deepen 候補) / H5 (production potential) を ratio に重畳 /
  H7 (ratio clip) は coef=1.0 でも value 安定なので優先度低下。promote 候補筆頭 (H2 から更新、user 承認必須)。

## Artifacts
- model: `data/output/models/reinforce/case5_kaggle_jax_train_h4_ratio_coef1/runs/20260528-004854__feature-support-reward__8e2e4a3__seed0/best.pt`
- metrics: 同ディレクトリ `metrics.json`
- logs: train.log / onstart.log / system.log
