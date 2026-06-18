# PoC RESULT — BC warm-start × V-MPO は rulebase への勝率を動かすか

> 関連: hypotheses.md / phase2_result.md / vmpo_poc_bc.yaml
> run_id: 20260610-141501__feature-poc-v-mpo__d512d2c__seed0 / commit: d512d2c2 / case: reinforce_case8_vmpo_poc_bc
> 実行: 2026-06-10 / GPU: RTX 4090 (~25分, ≈$0.29) / 判定コスト: ローカル CPU eval ×5 (無料)
> 成功基準 (ユーザー緩和後): 学習中もしくは前後で実 rulebase への勝率が向上していること

## Summary

**不支持 (NO-GO in current form)。** vmpo_frozen から `bc_warmstart` のみ有効化した
single-variable A/B (case9_per_planet BC 重み = Kaggle 上位棋譜由来) で V-MPO 50 iter を
学習し、ckpt 軌跡を実 Python rulebase `baseline_v1` と各 24 戦で評価した。
**実 rulebase への勝率軌跡は 0% で横ばい**(i030 の 1 勝のみ、ノイズ範囲)であり、
事前に ~50% と見積もったリスク「Kaggle 棋譜由来 BC は rulebase に traction を持たない」が
顕在化した。学習自体は健全 (BC 133/133 ロード、pool 勝率 0.47-0.72 で PFSP 正常、
entropy 崩壊なし) で、インフラ/配線の問題ではなく**戦略上の不足**が原因。

## Numbers

### 実 baseline_v1 (Python rulebase) との offline 対戦 — 各 24 戦

| 評価点 | 勝率 | 勝-敗-分 | 備考 |
|---|---|---|---|
| S1: vmpo_frozen (BC なし, 50it) | 0.0% | 0-24-0 | no-BC 基準 |
| ckpt_i000 (≈BC 初期) | 0.0% | 0-24-0 | **BC 自体が 0%** ← 主因 |
| ckpt_i015 | 0.0% | 0-24-0 | |
| ckpt_i030 | 4.2% | 1-23-0 | CI95 0.2-21%、ノイズ範囲 |
| ckpt_i049 (最終) | 0.0% | 0-24-0 | |

### 学習中の固定相手曲線 (held-out: in-JAX baseline_jax_full, 毎 iter 32 ep)

- 前半 10 iter 平均 **3.4%** → 後半 10 iter 平均 **5.0%** (+1.6pp、実質横ばい)
- pool 勝率 (PFSP): 0.47-0.72 で健全。entropy 42.9→27.3 (min 5.0、崩壊なし)

n=24/点 は project rule 上 inconclusive 扱いだが、PoC の判定対象「上昇軌跡の有無」に
対しては全 5 点 ×24 戦 + 毎 iter held-out が一貫して横ばいであり、判定に十分。

## Diagnosis

1. **BC の質が律速 (事前リスク ~50% が顕在化)**: case9_per_planet は Kaggle 上位棋譜の
   クローンであり、rulebase (baseline_v1) に対し 0/24。出発点に traction がなければ
   V-MPO はそれを増幅できない。
2. **pool 相手が「対 rulebase 能力」を教えない**: 学習相手は in-JAX baseline_jax
   full/lite + self snapshot。rulebase 的な挙動と対戦しないため、anti-rulebase の
   勾配信号が存在しない (過去知見「self-play 改善は実相手に転移しない」と整合)。
3. V-MPO/学習配線は正常 (pool 勝率・entropy・BC ロードすべて健全)。アルゴリズム
   ではなく**教師と相手の選択**の問題。

## Decision

- 採否: **rejected** (現形態の BC warm-start 単独では rulebase への勝率は動かない)
- 次の一手 (優先順):
  1. **rulebase 蒸留** — 実 baseline_v1/v2/v3 の selfplay 棋譜 (CPU 高速・無料) から
     BC 蒸留し、(a) 蒸留クローンを opp_model として pool/held-out に投入、
     (b) 同じ蒸留重みを bc_warmstart 教師に使う。教師と相手の両方が rulebase 化する
     ため、本 PoC で欠けていた 2 要素を同時に埋める (planner 計画 Phase 1-2)。
  2. 蒸留クローン相手に直接 V-MPO (PFSP f_var で勝率 0.5 帯に維持)
  3. 上記が動いてから長時間学習 (iter 数 50→200+)

## Artifacts

- run dir: `data/output/models/reinforce/case8_vmpo_poc_bc/runs/20260610-141501__feature-poc-v-mpo__d512d2c__seed0/` (best.pt / ckpt_i{000,015,030,049}.pt / metrics.json)
- 評価 JSON: `data/output/experiment/case8_poc_bc/{s1_vmpo_frozen_vs_v1,traj_i000,traj_i015,traj_i030,traj_i049}.json`
- S3: `s3://orbit-wars-dvc-286854171013/remote/runpod_artifacts/20260610-141501__feature-poc-v-mpo__d512d2c__seed0/`
