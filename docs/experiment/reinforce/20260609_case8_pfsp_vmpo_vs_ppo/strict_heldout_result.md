# RESULT — 高速化 strict_v1 の低頻度 held-out は実用になったか（軽い実験）

> 関連: poc_bc_result.md / distilled_result.md / vmpo_strict_heldout.yaml
> run_id: 20260611-024907__feature-poc-v-mpo__e4a6549__seed0 / commit: e4a6549e
> 前提 merge: feature/reinforcement-learning-pooling-simple (allocator top-K=64 截断 15-18x + aim vectorize、identity GPU 32/32 検証済)
> 実行: 2026-06-11 / RTX 4090 ~22分 ≈ $0.25

## Summary

**採用 (adopted)。** 旧 strict (per-turn 18.5-29s) では 16戦の held-out eval が
1回も完了しなかったが、高速化 merge 後は **cold（compile込）~3.8分 / warm ~2.9分/eval**
で安定完走。V-MPO 30 iter + strict_v1 eval 3回（iter 0/15/29）が計 ~19分で完了し、
**「本物 ~90% parity の真の強さ yardstick を学習中に持つ」運用が初めて実用化**した。
通常 iter は 7-8 秒で in-JAX 相手と同等（strict branch の compile 同居による劣化なし）。

## Numbers

| 計測 | 旧 strict | 今回（高速化後） |
|---|---|---|
| held-out 16戦×500手 (cold) | >13分でも未完 ×2回 | **~3.8分**（compile 込） |
| 同 (warm) | — | **~2.9分**（2回計測: 2:54 / 2:49） |
| 実効 per-turn | 18.5-29s | **~0.35s**（batch16 償却） |
| 通常 iter (in-JAX pool) | 7-8s | 7-8s（劣化なし） |
| 30 iter + eval3回 総所要 | 完走不能 | **~19分** |

- held-out 曲線 (vs strict_v1): iter0=0.0 / iter15=0.0 / iter29=0.0
  （本物級相手に対する現方策の真の位置 = 0% を学習中に直接観測。蒸留クローン
  yardstick の ~16% が「甘い ものさし」だったことも同時に裏付け）
- 学習自体は健全: pool 勝率 ~0.5 帯 (PFSP)、entropy 崩壊なし、V-MPO KL 制御下

## 運用ガイド（今後の実験設計用）

- every:15 / 16 ep → 50 iter run に **+6分**程度。every:10 でも +9分で許容。
- episodes を 32 に戻しても per-turn 律速のため eval 時間はほぼ不変（バッチ償却）
  → **32戦に上げて分解能を稼ぐのが得**。
- strict を学習 pool に入れるのは依然非推奨（rollout 7s/iter に対し2桁重い)。

## Decision

- 採否: **adopted**（インフラ実用性の検証として成功）
- 次の一手:
  1. **vmpo_distilled + strict held-out の合流**: 蒸留クローン教師/相手 + strict_v1
     yardstick (every:10, 32ep)。学習信号と真のものさしを両立した本命構成。
  2. 長時間学習 (iter 500+) を 1 の構成で実施し、strict held-out が 0 から動くかを
     本物のものさしで監視（rulebase 撃破ロードマップの①計算規模に着手）。
  3. 並行して mid-agent（per-target 集約、<10ms 級）を pool 用に構築すれば
     「学習相手も rulebase 級」に近づく（調査済み方針 P0-P3）。

## Artifacts

- run dir: `data/output/models/reinforce/case8_vmpo_strict_heldout/runs/20260611-024907__feature-poc-v-mpo__e4a6549__seed0/` (metrics.json)
- plots: `data/output/experiment/case8_strict_heldout/plots/{strict_heldout_run_detail,session_three_arm_comparison}.png`
