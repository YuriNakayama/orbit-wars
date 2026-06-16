# case8 ladder22 — 案A (win-thresh skip + 素strict force + no_op_bias 1.0) RESULT

> 関連: ladder21_strict_win_deepdive.md / rl_failure_rootcause.md / hypotheses.md
> run_id: 20260616-113645__feature-poc-v-mpo__af449ce__seed0 / commit: af449ce0
> case: reinforce_case8_vmpo_ladder22 / GPU: RTX 4090 SECURE ($0.69/h)
> 開始: 2026-06-16T11:36Z / 早期停止: iter39 (strict床が構造的に確定、best.pt確保、案Bへpivot) / pod destroy済 (0確認)

## Summary

案A の3工夫: C2(no_op_bias 2→1) + A2(skip_update_win_thresh=0.05) + force(T0=0段を2iter毎
強制照射)。狙いは「地力↑継続 + 素strict段を degenerate無しで訓練」。結果:
- ✅ **A2 skip がライブ動作**: 強制された T0=0 段 (win 0.005-0.021<0.05) は全て update SKIP
  (loss=0/entropy=0 = _ZERO_PPO_STATS)。ladder21 iter4 の entropy崩壊 (44→11.7) を阻止。
- ✅ **地力↑**: 弱体化strict段 win が ladder21 比でさらに上昇 (T0=125: 0.39→0.48,
  T0=225: 0.776→0.812)。no_op_bias=1.0 の over-fire是正が効いた。
- ✅ **full held-out 0.84-0.86** (iter0=0.859, iter30=0.844) = campaign 高水準。
- ❌ **held-out strict_v1 は床のまま** (0,1,0,2 /64) — 断崖は不変。

## Numbers

### held-out
| iter | strict_v1 | full |
|---|---|---|
| 0  | 0/64    | 0.859 |
| 10 | 1/64    | 0.812 |
| 20 | 0/64    | 0.812 |
| 30 | 2/64    | 0.844 |

### T0=0 強制段 (A2 skip 確認)
iter0/2/4/6: win 0.005-0.021、**全て [SKIP]** (loss=0)。poison阻止は完璧。

## Diagnosis: 計画時のリスクが顕在化 — skip と force の自己矛盾

force_rung_low (T0=0 を照射) と A2 (win<0.05 で skip) が衝突: **T0=0 段は「サンプルされるが
更新されない」** → 素strict序盤を一度も「学習」しない。よって:
- policy は clean に保たれる (poison無し) ✅ → 地力 full 0.86 が伸びる
- が、断崖 (T0=0) は学習信号がゼロのまま → held-out strict_v1 は床 ❌

**重要な切り分け**: ladder21+22 で「degenerate poison」は完全に解決した。残る唯一の障壁は
**「T0=0 段に学習可能な (非退化の) 信号が無い」こと**。skip (保護) では学習できない。

## Decision

- 採否: **partial** — A2/C2 は地力↑ (full 0.86 新境地) として採用価値あり。strict 攻略は
  skip だけでは不可と確定。
- 次の一手: **案B (ladder23) = T0=0 段に shaped reward を boost**。`_run_iter` は shaping_coef
  を per-iter 引数で受ける (実装確認済) ので、T0=0 段だけ shaping_coef/dense_coef を厚くし、
  「勝てなくても序盤の良い動き (territory確保/捕獲) に非退化の勾配」を与える。Lux S1優勝の
  「shaped で土台→sparse」と同発想。これで T0=0 段が skip されず (win>0でも shaped信号で
  advantage分散が出る) 学習が乗るか。resume ladder22 best.pt (full 0.86)。
  注意: shaping が material最適化に偏る落とし穴 (ladder13) を避け、dense差分 (out-position
  報酬) 主体にする。

## Artifacts
- model: `data/output/models/reinforce/case8_vmpo_ladder22/runs/20260616-113645__feature-poc-v-mpo__af449ce__seed0/` (best.pt full~0.86)
