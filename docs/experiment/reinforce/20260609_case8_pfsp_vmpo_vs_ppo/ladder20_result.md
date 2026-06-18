# case8 ladder20 — handicap で勝ちを人工生成 RESULT

> 関連: ladder19_result.md / strict_loss_websearch_analysis.md / hypotheses.md
> run_id: 20260615-151944__feature-poc-v-mpo__d22b630__seed0 / commit: d22b630b
> case: reinforce_case8_vmpo_ladder20 / GPU: A100 80GB SECURE ($1.39/h)
> 開始: 2026-06-16T00:28Z / 早期停止: iter6 (h=3.0でwin~4%固定が明確) / pod destroy済 (0確認)

## Summary

zero-variance (勝ちが無く勾配が立たない) を破るため handicap (mode=ships) で learner の
初期 ship を 3倍 boost し勝てる局面を人工生成、勝利勾配を bootstrap → anneal する狙い。
resume ladder18 best.pt (aim修正込)。結果、**h=3.0 (3倍ship) で訓練しても win は
0.036→0.005→0.042 と ~4% 帯で固定、hcap_idx=0 (boost最大) から動けず** (win が demote
0.2 すら下回り promote 0.6 には程遠い)。仮説は棄却。

## Numbers

### handicap (strict) 段 — h=3.0 固定

| iter | win | handicap | hcap_idx |
|---|---|---|---|
| 0 | 0.036 | 3.0 | 0 |
| 2 | 0.005 | 3.0 | 0 |
| 4 | 0.042 | 3.0 | 0 |

held-out iter0: strict_v1=0.0, full=0.828 (ladder18 base 維持)。

## Diagnosis (決定的所見)

**3倍の戦力を与えても strict に ~4% しか勝てない** = strict の優位は **material (ship数)
でなく位置/戦略 (positioning / decision quality)**。材料を3倍にしても out-maneuver される。
これは phaseB の handicap 棄却 (h=3.0→0/64) を aim修正後も再確認 (0→4% の僅差改善のみ)。
→ **material-based curriculum では戦略ギャップを埋められない。**

web search 分析 (strict_loss_websearch_analysis.md) とも整合: handicap は curriculum 策の
一種だが、文献でも「相手が強すぎ skill gap が大きいと curriculum/BC でも性能が頭打ち/低下」
と警告されており、3倍 boost でも勝てない = skill gap が material 補償の範囲を超えている。

## campaign 総括 (ladder1-20): strict 攻略で否定されたレバー

| レバー | run | 結果 |
|---|---|---|
| curriculum 形状 (T0/ε/逆カリ) | ladder1-16 | 素strict 0% |
| 対戦量 | ladder17 | 0% |
| aim修正 (intercept) | ladder18 | **full 0.828 新記録**、strict 0% |
| reward整形 + 難段強制 | ladder19 | 0% (zero-variance) |
| handicap (material boost) | ladder20 | ~4% (戦略的劣位、material無効) |

→ 残る文献最有力策は **BC bootstrap (imitation)**: strict_v1 の意思決定を直接模倣学習し
RL で洗練。ただし (a) strict→action の BC データ/clone 構築が必要 (大きめ)、(b) 文献の
skill-gap 警告 (強すぎる相手の BC は頭打ち) に注意。

## Decision

- 採否: **rejected (handicap/material)**
- 次の一手 (要ユーザー判断 — 7連続 strict 棄却を踏まえた分岐):
  - **(A) BC bootstrap 継続**: 最大の未投入レバー。strict_v1 のリプレイから state→action BC →
    warm-start + KL anchor (bc_warmstart 配線済) → RL洗練。effort 大。
  - **(B) full 0.828 を成果として確定**: strict は構造的に困難と結論し、movement-detector
    に強い汎用エージェント (full 0.828) を成果として ladder18 best.pt を確定。
  両者の判断をユーザーに諮る。

## Artifacts

- model: `data/output/models/reinforce/case8_vmpo_ladder20/runs/20260615-151944__feature-poc-v-mpo__d22b630__seed0/`
