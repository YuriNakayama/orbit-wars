# iter4 result: H4 scale-up (150 iter) — BREAKTHROUGH (scale works)

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
