# Phase 1 GATE — baseline_v8 蒸留クローンは実 v1 に traction を持つか

> 関連: distilled_result.md / il_case9_rulebase_v8.yaml
> BC run: 20260611-063257__feature-poc-v-mpo__a091f5c__seed0 (case9_rulebase_v8) / commit a091f5c3
> 実行: 2026-06-11 / データ生成 CPU ~1h + BC GPU ~22分 ≈ $0.3 / gate 評価 CPU 50戦

## Summary

**NO-GO (0/50, CI95 上限 7.1%)。** v1 を 68.7% で倒す baseline_v8 を教師に、
対 v1 状態分布そのもの (v8 vs v1 棋譜 757 episodes / 224k frames) から蒸留した
クローンでも、実 baseline_v1 への勝率は **0%**。v1-clone (0/24) と合わせ、
**「~50% の行動一致率の BC クローンは、教師の強さに関係なく決定論的 rulebase に
0%」**という強い負の結果が確定した。律速は教師の質ではなく**蒸留忠実度**である。

## Numbers

| エージェント | 教師 | fire_acc (exact) | vs 実 baseline_v1 |
|---|---|---|---|
| 実 baseline_v8 | — | — | **68.7%** (515/750) |
| clone-v8 (今回) | v8 | 47.9% | **0/50 (0%)** |
| clone-v1 | v1 | 52.5% | 0/24 (0%) |
| Kaggle-BC | 上位棋譜 | — | 0/24 (0%) |

- teacher_agents フィルタは設計通り動作 (757/1765 採用、v1-mirror 1008 件除外)
- BC 学習: early-stop ep40 付近、best ep35。v8 は v1 より複雑で一致率が低い
  (47.9% vs 52.5%) — 教師が強いほど蒸留は難しいという追加知見

## Diagnosis

ゼロサム × 決定論的エキスパート相手では、1 つの悪手 (テンポ損失) が回復不能で、
~50% 一致のクローンは毎ターン ~50% の確率で教師と違う(多くは劣る)手を打つ。
勝率がゼロになるのは「半分正しい」では足りないため。BC 忠実度をどこまで上げれば
勝率が出るかは未知 (DAgger で 75%+ にしても保証なし)。

## Decision

- ゲート: **不通過** → クローン init による「traction のある出発点」路線は棚上げ
- ただし **Phase 2 (ハンディキャップ・カリキュラム) の正当性は不変**:
  Phase 2 は「本物の strict_v1 相手に初期条件ハンディで勝てる試合を製造し、
  ゼロ勾配を解く」アプローチであり、クローンの強さに依存しない。
  clone-v8 は pool の多様性要員 (v8 スタイルのスパーリング) として残す。
- 次の選択肢:
  1. **Phase 2 着手** (推奨): reset の seat0 初期 ships スケール + 勝率連動 anneal
  2. DAgger で忠実度 75%+ を狙う (効果不確実、~1日)
  3. 探索ハイブリッド (決定論搾取、RL 外)

## Artifacts

- clone-v8: `data/output/models/imitation/case9_rulebase/runs/20260611-063257__feature-poc-v-mpo__a091f5c__seed0/best.pt` (DVC 済)
- mart: `data/mart/imitation/case9_rulebase_v8/` (224k/29k frames)
- gate 評価: `data/output/experiment/case8_distilled/clone_v8_vs_v1.json`
