# RESULT — 合流構成 (combo): 蒸留クローン + strict 強制 pool + strict held-out

> 関連: strict_heldout_result.md / distilled_result.md / vmpo_combo.yaml
> run_id: 20260611-034156__feature-poc-v-mpo__a4a47b2__seed0 / commit: a4a47b2d
> 実行: 2026-06-11 / RTX 4090 ~52分 ≈ $0.60

## Summary

**インフラ: adopted / 学習: 50 iter 規模では strict 信号は flat 0 (想定通り)。**
実証済み3部品 — ①蒸留クローン (bc_warmstart 教師 + 常設 NN 相手 133/133)、
②`force_strict_every: 10` による strict_v1 強制対戦 (iter 4/14/24/34/44 で
正確に5回発火、各 ~170秒)、③strict_v1 held-out (6点、warm ~3分) — が
1 つの run で全て設計通りに動作した。学習面では強制 strict 対戦 win と
held-out が全点 0.000 で、50 iter 小規模 RL の限界 (case7 15iter / PoC /
distilled と一貫) を本物のものさしで再確認した。

## Numbers

| 信号 | 値 |
|---|---|
| held-out vs strict_v1 (iter 0/10/20/30/40/49) | 0.0 ×6点 |
| 強制 strict 対戦 win (iter 4/14/24/34/44) | 0.000 ×5回 (reward ≈ -3.4 一定) |
| Elo (held-out 連動) | 1484 → 1414 (全敗による単調減) |
| pool 勝率 (self/clone/lite) | ~0.5-0.7 帯で健全 (PFSP 正常) |
| 強制 strict rollout | ~170秒/回 (5回 = +14分、見積り通り) |
| 総所要 | ~52分 (学習 ~26分 + held-out ~19分 + setup) |

## 運用上の確定事項

- `force_strict_every` は f_var トラップ (1敗で重み恒久ゼロ) を完全に回避し、
  時間コストは決定論的 (+170秒×回数)。
- 50 iter 構成の標準コスト: **~52分 / ~$0.60** — 長時間学習の単価基準になる。
- 全敗 iter の reward は -3.4 で一定 = 学習信号としての分散はほぼゼロ。
  勝ち始めるまで強制対戦は「検知器」であり「教師」ではない (設計通り)。

## Decision

- 採否: **adopted (インフラ)** / 学習効果は 50 iter では検出されず (想定内)
- 次の一手:
  1. **この config のまま iter 500+ の長時間学習** (~6-8時間 / ~$5-6)。
     held-out 6点→50点の曲線で liftoff を監視。文献 (Generals.io 36 GPU-h) 比
     ではまだ 1 桁下だが、最初のスケール検証として妥当。
     iter 500 で全点 0 なら「規模を 1 桁上げても 0」が確定し、mid-agent
     (per-target 集約 rulebase, <10ms) や報酬設計に軸足を移す。
  2. 並行で DAgger によるクローン強化 (fire_acc 52.5→75%) — 教師と相手の
     質を上げてから長時間学習に入る方が効率的な可能性。

## Artifacts

- run dir: `data/output/models/reinforce/case8_vmpo_combo/runs/20260611-034156__feature-poc-v-mpo__a4a47b2__seed0/` (metrics.json)
