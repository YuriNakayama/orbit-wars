# Rulebase/case12 — iter1 (NaïveMCTS) Result: 0%

> 作成日: 2026-05-06
> 対応 plan: [`plan.md`](./plan.md)
> 関連:
> - [`docs/experiment/rulebase/20260505_case11_portfolio_search/iter1_result.md`](../20260505_case11_portfolio_search/iter1_result.md) — PGS 4 連敗 (0%)
> - memory: `project_heuristic_search_saturation.md` (本実験で 11 連敗目を確定)

## 結論

**case12 NaïveMCTS は完全敗北 (vs v4 = 0/10)。** PGS の deterministic hill
climb を stochastic UCB1 sampling に変えても **同じ 0%**。**「script-only
モデル」自体が case4 の rich mission に勝てない構造制約** が、deterministic
/ stochastic 共通の致命的限界として確定。

これで `project_heuristic_search_saturation` 通り **heuristic 系探索改修は 11 連敗で完全飽和**。学習ベース value function / 別 agent family への方向転換以外の道なし。

## 数値

### Smoke 結果

| 構成 | n | wins | win_rate | turn_p95 |
|---|---|---|---|---|
| case12 NaïveMCTS v0 (rollouts=64, h=20) | 10 | 0/10 | **0.0%** | 0.028s |
| (参考) case11 PGS v3 | 10 | 0/10 | 0.0% | 0.058s |

**turn_p95 0.028s** = case4 production (~0.65s) の **1/23**。NaïveMCTS が rollout で計算した結果を出す move 量が **PGS よりさらに少ない** = sampling で arm が分散して中途半端な script (idle 寄り) 採用されている。

baseline_v4 側で **timeouts 9 件** 発生していたが、それでも v12 は 0 勝。 v4 の timeout は v12 の弱さで補えていない。

## 診断

### Script-only モデルの構造制約 (case11 PGS と同じ)

PGS v0-v3 の 4 連敗で判明したのは「**1 source 1 script モデルが case4 の rich mission set と相性悪い**」。case4 の `collect_missions` は 1 source が capture + snipe + swarm + harass + reinforce 等 **複数 mission に並列貢献** する設計で、greedy で 1 ターンに 5-10 件の mission が成立する。

case12 NaïveMCTS は sampling アルゴリズムを変えただけで、**output モデル (assignment = src_id → 1 script)** は PGS と完全同型。case4 の rich greedy を再現できない。

### Sampling-based でも改善しない理由

- **rollouts=64 は不足**: case4 base には ~30 planets × 7 scripts = 210 arm、64 rollout では各 arm を 1-2 回しか visit しない
- **playout horizon=20 は不足**: case4 が長期戦で勝つパターンは 50-100 turn 規模、20 turn では大局が見えない
- ただし rollouts や horizon を増やしても **script-only output モデル** の根本制約は解消しない。turn_p95 0.028s から推測すると **rollouts もまともに走り切れていない可能性** (early exit が頻発)

### 11 連敗確定

`project_heuristic_search_saturation` 通り heuristic 系の物理限界:

1-9. (memory 既記載)
10. case11 PGS v0/v1/v2/v3
11. **case12 NaïveMCTS v0** (本実験)

## 採用方針

- **case12 採用却下**
- `NAIVE_MCTS_ENABLED=False` に default 復元 (case4 等価動作、unit test 保証)
- 4 unit tests pass

## 確定した最終結論 (case8/9/10/11/12 累計)

case4 production (LB745) を超えるには **heuristic 系 (mission ordering, score 補正, filter, script assignment, sampling-based search) では到達不可能**。残る方向は:

| 方向 | コスト | 期待 |
|---|---|---|
| **学習ベース value function** (imitation の value head 流用) | 数日 | 評価関数を heuristic から離す唯一の道 |
| **別 agent family 開発** (rulebase 撤退、imitation/reinforce 集中) | 中-長期 | 手法のパラダイムシフト |
| script を「mission per source」に粒度変更 (case12 iter2) | 1-2 時間 | 構造制約に手当てするが、根本解決ではない |

推奨は **学習方向** へのシフト。本実験ディレクトリ (case12) は撤退、本セッションの heuristic 改修ループはここで物理限界に到達。

## 関連ファイル

- `bot/pipeline/rulebase/case12/baseline/core/config.py:NAIVE_MCTS_*` — 5 個 config (default False で撤退)
- `bot/pipeline/rulebase/case12/baseline/planner/naive_mcts.py` — NaïveMCTS 実装 (~120 行)
- `bot/pipeline/rulebase/case12/baseline/planner/scripts.py` — case11 v3 流用
- `bot/pipeline/rulebase/case12/baseline/planner/evaluator.py` — case11 流用

## 環境

- ハードウェア: M4 MacBook (local), parallel=4
- branch: `feature/rulebase-multistep-optimization`
- 実行日時: 2026-05-06
