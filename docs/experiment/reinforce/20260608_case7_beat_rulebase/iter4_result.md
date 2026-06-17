# iter4 result: H4 scale-up (150 iter) — train/eval gap (vs JAX proxy ✓, vs real rulebase ✗)

## ⚠️ 重要な訂正 (外部 paired 評価後)
学習中 win は **JAX近似相手 (baseline_jax_full)** に対する値。最終 ckpt を **本物 rulebase**
で評価すると **全敗**:
- ckpt_i131 × **baseline_v8** (本物, 30戦) = **0/30**
- ckpt_i131 × **baseline_v1** (本物, 12戦) = **0/12**

→ 「scale で rulebase を破った」は **誤り**。scale が破ったのは **JAX 近似opponent** のみ。
本物 rulebase には全く転移しない = **train(JAX)/eval(本物) parity gap** が真のボトルネック
(memory `project_reinforce_case6_live_eval` を精密に再現)。scale も機構も、この gap の前では無力。

---

## 当初の (proxy 相手の) 結果 — scale 自体は効く

run_id: 20260608-064345...500bc5d / GPU: RTX 4090 / 137/150 iter (hung at iter137, result intact) / config: h4_scaleup.yaml

## 設定
H1-H3 が示した「20-iter ~0.3 天井」が under-training か本質的天井かを判定。
最も健全な設定: PFSP self_snapshot pool (f_hard) + baseline_jax_full mix (40%)
+ dense差分報酬。BC無し、kl_beta=0。iterations 20 → 150。h500/batch32。

## 結果: vs baseline_jax_full が scale で単調上昇 ★
| 区間 | mean | max | n |
|---|---|---|---|
| iter 0-50 | 0.231 | 0.344 | 10 |
| iter 50-100 | 0.338 | 0.531 | 11 |
| **iter 100+** | **0.475** | **0.625** | 5 |

直近 full-iter: 92:0.50 / 96:0.53 / 112:0.56 / 117:0.47 / **121:0.62** / 123:0.34。
- **vs rulebase が 0.23 → 0.34 → 0.47 へ単調上昇、勝ち越し(0.50-0.62)到達**。
- 20-iter 勢 (H1-H3) は決して 0.4 を超えなかった → **天井は under-training だった**。
- self_snapshot の自己改善が rulebase に**転移している**（train/eval ギャップを scale が埋めた）。
- entropy collapse せず、健全に学習継続。

## 結論: 重要な前進 (loop の転換点)
- **「小規模RLは rulebase に勝てない」(memory case6_live_eval) は規模の問題**と判明。
- H4 の climb は明確 → **H5: 300 iter で勝ち越しを定着**させるのが次の一手。
- ⚠️ n=32/点なので 0.50-0.62 は学習中の指標。最終採否は **最終 ckpt の paired 30戦**で確定要。

## 運用メモ
- iter137 で hang (GPU 0%, log 停止)。memory `feedback_jax_selfplay_foreground_only` 系の
  JAX self-play hang の可能性。crash-safe S3 で全 138 ckpt 保全済、結果は無傷。
- H4 final metrics + ckpt_i137 を S3 の `h4_final_metrics.json` / `h4_ckpt_i137.pt` に退避
  (同 run_id で H5 が上書きするため)。

## コスト: ~$1.0 (4090 ~90分)

## 真の結論 (訂正後)
- scale は **JAX 近似相手** には効く (0.23→0.47) が、**本物 rulebase には 0勝** (train/eval gap)。
- **ループの本質的ボトルネックは parity gap**: 学習は JAX sim + JAX featurizer、評価は本物
  Python env。agent は JAX 固有の癖に過適合し本物に通用しない。
- → 次の正しいレバーは **scale でも機構でもなく parity**:
  (a) 学習相手に本物 rulebase を host_callback で混ぜる (rollout 重いが本物経験)、
  (b) JAX featurizer ↔ 本物 featurizer の parity を取る、
  (c) 本物 env での学習 (PufferLib 等で高速化) 。
- H5 (300iter scale) は **この gap を埋めないので無意味** → 中止が正しい。
