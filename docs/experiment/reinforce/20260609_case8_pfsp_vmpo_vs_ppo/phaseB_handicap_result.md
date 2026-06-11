# Phase B RESULT — ハンディキャップ・カリキュラム（蒸留不要計画）

> 関連: phase1_v8_gate.md / vmpo_handicap.yaml
> v1 run: 20260611-072644 (ships, 10 iter で中止) / v2 run: 20260611-075834 (weaken, 49/50 iter)
> commit: a2338628 (ships) → 3d75a17b (weaken) / 実行: 2026-06-11 / GPU 計 ~85分 ≈ $1.0
> 注: v2 は私の pod destroy が早く iter49 の最終 held-out 1 点を喪失 (it0-48 は完全)

## Summary

**機構 adopted / ε 降下はスケール待ち。** 2 つのカリキュラム軸を検証した:

- **v1 (ships, 棄却)**: 学習者の初期 ships を h=3.0 倍しても強制 strict 戦は 0/64
  (reward_max -2.59 = 最良エピソードすら大敗)。受動的な方策は物量を活かせない —
  「材料ハンディは準有能な学習者にしか効かない」ことを実証。
- **v2 (ε-weaken, 機構成立)**: strict の行動をターン毎確率 ε で noop 化。
  ε=1.0 で勝率 0.81→0.92 (単調成長、正の reward = **strict 戦で初めて勾配が生きた**)。
  コントローラは設計通り昇降し、副次効果として f_var の自然選択も復活
  (iter23/31 は強制外で strict を選択 — 恒久ゼロ化トラップ解消を実証)。
  ただし **ε=0.85 (strict 実働 15%) が 50 iter での壁** (0.0 ×5回)。

## Numbers

| 計測 | v1 (ships h=3.0) | v2 (ε-weaken) |
|---|---|---|
| 強制 strict 戦 | 0/64, 0/64 (iter4/9) | ε=1.0: **0.81/0.84/0.86/0.81/0.88/0.92** ↗ / ε=0.85: 0.0 ×5 |
| ladder 軌跡 | idx 0 固定 (壁) | 1.0 ⇄ 0.85 振動 ×5周期 |
| held-out strict_v1 (素) | — | 0.0 ×5点 |
| held-out baseline_jax_full (素) | — | 0.156-0.219 横ばい |
| pool 勝率 (f_var) | ~0.4-0.5 | ~0.43 (設計帯) |

## Diagnosis (学習が進まない原因の検討)

1. **絶対経験量が 2-3 桁不足 (主因)**: 総経験 3,200 エピソード。noop 化相手に
   8-19% 落とし、full への held-out が横ばい — strict 以前にゲーム自体が未習得。
   文献 (Generals.io) は数百万ゲーム相当で bot 撃破。
2. **報酬信号が薄い**: ±1 終端 + dense 0.003 では 500 手のクレジット割当て不能。
   序盤拡張 (勝敗の決定打) に勾配が届かない。
3. **ε-uniform は「効く手」を消せない**: ターン一様 Bernoulli では strict の決定的な
   序盤拡張の 15% が依然通る → 1.0→0.85 の急峻な崖。**時間窓型** (最初の T0 手のみ
   noop 化、T0: 500→0 anneal) の方が学習可能な中間タスクを作る。
4. 副因: no_op_bias 8.0 の受動的初期化。除外済み: V-MPO 自体 (PPO 同等)、
   entropy 崩壊、インフラ。

## Decision

- 採否: **機構 adopted** (勾配・コントローラ・f_var 鮮度維持の全部品が動作) /
  ε 降下 (本物到達) は未達 — descent-pending-scale
- 次の一手 (診断①②③に対応):
  1. **shaping 強化**: Generals.io 式 3 項 potential (planets/ships/production の
     log 比) — 最安で②を検証
  2. **時間窓カリキュラム**: ε-uniform → 序盤 T0 手 noop 化 + T0 anneal (③)
  3. **スケール**: 上記 2 つを入れた iter 300-500 run (~$4-6) で①を検証
  4. ラダー上部の細分化 [1.0, 0.95, 0.9, 0.85, ...] は時間窓化するなら不要

## Artifacts

- v2 run dir: `data/output/models/reinforce/case8_vmpo_handicap/runs/20260611-075834__feature-poc-v-mpo__3d75a17__seed0/` (metrics.json, 49 iter)
- plots: `data/output/experiment/case8_handicap/plots/{phaseB_v2_training_log,dual_heldout_progress,curriculum_experiments_log}.png`
