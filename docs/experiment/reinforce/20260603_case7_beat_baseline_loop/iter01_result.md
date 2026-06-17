# case7 「ルールベースに勝つ」ループ — iter01 RESULT

時刻: 2026-06-03 02:00 (cron tick 2)

## やったこと
1. 評価経路を確立: `best.pt(npz) → jax_to_torch → policy/weights.pt → eval_vs_baseline`。
   `CHALLENGER` を `rl_v6 → rl_v7` に修正。
2. **出発点 measure**: 16-iter best.pt vs baseline_v1 10戦 = **0/10 (win 0.0)**。
3. **H-a+H-b 試行**: curriculum `noop → baseline_jax_lite`(=v1相当) で 16-iter best.pt
   から resume +16 iter 追加学習を起動。

## Numbers (vs baseline_jax_lite, 学習中 trace)
| iter | opp | win | reward | rollout |
|---|---|---|---|---|
| 0-1 | noop | 0.75, 1.00 | +0.15, +0.21 | ~65s |
| 2 | lite | 0.250 | -0.20 | 107s |
| 3 | lite | 0.000 | -0.61 | 214s |
| 4 | lite | 0.125 | -0.63 | 173s |
| 5 | lite | 0.000 | -0.62 | 202s |
| 6 | lite | 0.125 | -0.39 | 171s |
| 7 | lite | 0.125 | -0.47 | 152s |

## Diagnosis
- **noop には圧勝 (0.75-1.0)、baseline_jax_lite(=v1相当) には 0-0.25 で完敗**。
  → 0/10 vs 本物 v1 は妥当。モデルは v1 流の打ち合いに勝てる方策を持たない。
- vs lite で **iter 2→7 で改善が見えない** (win 0-0.25 横ばい、reward -0.5 前後)。
  16 iter 程度の追加 PPO では足りない兆候。
- **rollout が 150-214s/iter に激増** (lite は host callback で重い)。CPU ローカルで
  16 iter ≈ 45分 → 30分 tick に収まらない。効率が悪い。

## 最大の見落とし (次 tick の主仮説)
- **case7 の全 run は BC warm-start OFF** (`bc_warmstart.enabled: false`)。
  case5/6 の設計と memory `project_reinforce_case6_pool_v1_rejected` は
  **「BC warm-start + 3段 curriculum」を v1 攻略の本筋**としている。
  random-ish init から強い rule 相手に PPO すると reward sparse で勾配消失 → 伸びない。
- BC 元 weights (`imitation/case9_per_planet/.../best.pt`) は**ローカル未取得**
  (DVC 管理)。→ 次 tick: `dev/dvc pull` で取得 → bc_warmstart 有効化。

## NEXT ACTION (iter02)
1. iter01 の追加学習を完走させ、その best.pt を 10戦 vs v1 で測定 (0/10 から動くか)。
2. **BC warm-start を有効化**: BC weights を DVC pull → `bc_warmstart.enabled: true` +
   `kl_beta` 小 (0.1) で resume 学習。これが本筋の改善。
3. 効率対策: lite 相手は重いので episodes 8→4 or horizon 200→150 で iter 数を稼ぐ。
