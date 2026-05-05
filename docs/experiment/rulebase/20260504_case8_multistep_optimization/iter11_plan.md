# Rulebase/case13 — Action-level MCTS (Mission per arm)

> 作成日: 2026-05-06
> 関連:
> - [`docs/experiment/rulebase/20260505_case11_portfolio_search/iter1_result.md`](../20260505_case11_portfolio_search/iter1_result.md) — PGS 4 連敗 (1 source 1 script 制約で 0%)
> - [`docs/experiment/rulebase/20260505_case12_naive_mcts/iter1_result.md`](../20260505_case12_naive_mcts/iter1_result.md) — NaïveMCTS でも script-only モデルでは 0%
> - [`docs/experiment/rulebase/20260504_case8_multistep_beam/iter2_plan.md`](../20260504_case8_multistep_beam/iter2_plan.md) — `score_mission_score` evaluator (本実験で再利用)
> - memory: `project_heuristic_search_saturation` (heuristic 系 11 連敗で飽和、Action-level は未試行)
> スコープ: 新規 `case13` を切り、case4 base 全複製 + `collect_missions` 出力を直接 arm 化した CMAB-based MCTS

## 仮説 (Hypothesis)

case11 PGS と case12 NaïveMCTS が 0% で完敗した主因は **「1 source 1 script」モデル** が case4 の rich mission set (1 source が capture / snipe / swarm / harass / reinforce 等に並列貢献する設計) を再現できなかったこと。

**Action-level MCTS** は `collect_missions` の出力 (8-15 mission/turn) を **個別の arm として独立に sampling**、各 mission の include/exclude を CMAB で UCB1 探索する。case4 の rich mission space を保ちながら multi-step 最適化を実現する。期待: vs `baseline_v4` で **≥55% 達成** (case11/12 の 0% から +50pp 級飛躍可能)。

**Mechanism**:
- 各 mission を独立 arm 化 → 1 source が複数 mission に貢献する rich pattern を維持
- `score_mission_score` (case8 iter2 流用、採用 mission の `mission.score` 合計) を reward
- UCB1 sampling で stochastic 探索、PGS の deterministic local optimum を bypass
- 採用 mission 集合は case4 既存の `_process_*_mission` で commit、moves emit は既存資産を完全再利用

## 既存コードの現状

- **新規 case 番号**: `case13` (free)、`baseline_v13` 未登録
- **Base**: `bot/pipeline/rulebase/case4/baseline/` (LB745 production)
- **`collect_missions` 出力**: `missions/__init__.py` で reinforcement → capture (with snipe) → swarm → crash_exploit → harass の順に builder を呼び合算、典型 8-15 mission/turn
- **Mission 構造**: `core/types.py:Mission` = (kind, score, target_id, turns, options[ShotOption])。`mission.score` がある
- **commit ロジック**: case4 `strategy.py:_process_single_source_mission` / `_process_multi_source_mission` (既存資産)
- **過去 iter の所見**:
  - case11 PGS 4 連敗 (script-only 制約)、case12 NaïveMCTS 0% (同制約)
  - case8 iter2 で `score_mission_score` evaluator を実装済 (本実験で再利用)
  - case8 candidate.py の `commit_missions_in_order` が「mission リストを順に commit」する関数 (case13 で類似コピー)

## スコープ (Scope)

### 新規 case 構成

```
bot/pipeline/rulebase/case13/                           # case4 全複製
├── main.py / __init__.py / README.md
├── baseline/
│   ├── ...                                              # case4 と同一資産
│   ├── core/config.py                                   # ★ ACTION_MCTS_* 追加
│   └── planner/                                         # ★ 新設
│       ├── __init__.py                                  # `run_action_mcts` を export
│       ├── candidate.py                                 # case8 から複製した commit_missions_in_subset (改名版)
│       ├── evaluator.py                                 # score_mission_score 評価
│       └── action_mcts.py                               # ★ 新規: Action-level MCTS 実装

bot/src/dataset/selfplay/agents.py                      # `"baseline_v13": ...` 追加
bot/tests/pipeline/rulebase/case13/                     # 新規 (3 種)
bot/pyproject.toml                                       # case13 ignore 追加
```

### config 追加

```python
# core/config.py
ACTION_MCTS_ENABLED: bool = True
ACTION_MCTS_ROLLOUTS: int = 128
ACTION_MCTS_EXPLORATION: float = 1.41  # sqrt(2)
ACTION_MCTS_TIME_BUDGET_S: float = 0.6
```

### Action-level MCTS アルゴリズム

各 mission は独立な **「include / exclude」2-arm の bandit** として扱われる:

```python
def run_action_mcts(world, missions, modes, ...) -> list[move]:
    """
    Algorithm:
    1. Get all missions from collect_missions (already done by caller).
    2. For each mission `m_i`, treat (include, exclude) as 2-arm bandit.
       Per-arm stats: (visits, total_reward).
    3. For `rollouts` iterations:
       a. For each m_i, pick include/exclude via UCB1 over its 2 arms.
          - First visit each arm at least once.
       b. The picked subset of missions = subset[m_i for include[m_i]]
       c. Run commit_missions_in_subset(world, subset, modes, ...)
          - Returns moves emitted + accepted_missions list
       d. Compute reward = sum(m.score for m in accepted_missions)
       e. Backprop reward to each m_i's arm (include or exclude as picked)
    4. Final subset: per m_i, pick most-visited arm.
       (Tie-break: pick include if both arms ≥1 visit and exclude won.)
    5. Materialize moves via commit_missions_in_subset and emit followup/evac/rear_guard.
    """
```

**重要**: `commit_missions_in_subset` は **score 順ではなく PGS で決まった subset 内の順序** を使うが、subset 内で score 順 sort して順次 commit する (case4 greedy と同じ)。これで「採用 mission のみ greedy commit」が成立。

### case8 candidate.py からの複製ロジック

case8 の `_process_single_source_mission` / `_process_multi_source_mission` は cross-case import 禁止のため、本実験では:
1. case4 の `strategy.py` の同名関数を本実験 `planner/candidate.py` に再配置 (case4 の動作を保つ thin wrapper)
2. `commit_missions_in_subset(world, missions, modes, source_inventory_left, source_attack_left, append_move)` で missions を score-desc に sort、順に `_process_*_mission` を呼ぶ

## 実装ステップ

1. `cp -r bot/pipeline/rulebase/case4 bot/pipeline/rulebase/case13`
2. `case13/{main.py, __init__.py, README.md}` の case4 → case13 参照置換
3. `core/config.py` に `ACTION_MCTS_*` 4 個追加
4. `planner/__init__.py` 新規 + 3 ファイル:
   - `candidate.py` — case4 strategy.py から `_process_single/multi_source_mission` を抽出、`commit_missions_in_subset` を export
   - `evaluator.py` — `score_mission_score(world, accepted_missions) -> sum(m.score)` (薄い、case8 iter2 流用)
   - `action_mcts.py` — `run_action_mcts(...)` 新規 ~150 行
5. `strategy.py:plan_moves` の冒頭で `ACTION_MCTS_ENABLED` なら:
   - `collect_missions` で missions を取得 (既存)
   - `run_action_mcts(world, missions, modes, ...)` で **採用 subset の moves** を取得
   - 既存 case4 の `emit_followup_moves` / `emit_evacuation_moves` / `emit_rear_guard_moves` を続けて呼ぶ (movement layer は変更しない)
   - `_enforce_inventory_cap` で最終 cap
6. `bot/src/dataset/selfplay/agents.py` に `baseline_v13` 追加、`bot/pyproject.toml` ignore 追加
7. `bot/tests/pipeline/rulebase/case13/`:
   - `test_baseline_agent.py` (smoke + slow integration)
   - `test_action_mcts_basic.py` (UCB1 unvisited explore、`run_action_mcts` shape)
   - `test_mcts_off_equals_case4.py` (`ACTION_MCTS_ENABLED=False` で case4 等価)
8. lint / format / mypy / pytest 緑、`dev/test-bot` 通過

## 検証方法

### ローカル

```bash
uv run --directory bot pytest tests/pipeline/rulebase/case13 -m "not slow" -x
```

### 性能評価 (3 段階)

#### Stage A: 10戦 smoke (early signal)

```bash
uv run --directory bot python -m dataset run --agents baseline_v4,baseline_v13 \
    --mode 1v1 -n 10 --seed 110000 --parallel 4 --no-save-replay
```

判定:
- ≥30% → Stage B 30戦に進む
- 10-30% → debug (rollouts 不足 / evaluator 不適合)
- <10% → 構造問題、撤退検討

#### Stage B: 30戦 + replay

```bash
uv run --directory bot python -m dataset run --agents baseline_v4,baseline_v13 \
    --mode 1v1 -n 30 --seed 110100 --parallel 4
```

判定:
- ≥55% → Stage C 200戦
- 50-55% → 改善方向、hyperparameter sweep (rollouts 128→256)
- <50% → 撤退、別方向

#### Stage C: 200戦

```bash
uv run --directory bot python -m dataset run --agents baseline_v4,baseline_v13 \
    --mode 1v1 -n 100 --seed 110500 --parallel 4 --no-save-replay
uv run --directory bot python -m dataset run --agents baseline_v13,baseline_v4 \
    --mode 1v1 -n 100 --seed 111000 --parallel 4 --no-save-replay
```

しきい値 ≥55% (合算) で採用、production 候補に上げる。

- **対戦相手**: `baseline_v4` (production, LB745) を主軸
- **エピソード数**: 合算 200 戦 (seat0=100, seat1=100)、user 指定
- **主要メトリクス**: 合算勝率 (vs v4)。**Kaggle publicScore は使用しない**
- **採否しきい値**: **+5pp 以上 (合算 ≥55%)** で採用
- **time budget**: turn_p95 ≤ 0.7s
- **wall-clock 想定**: 200戦合算 ~15-20 分 (case12 NaïveMCTS と同等の rollout 量)

## リスクと早期撤退条件

- **計算予算超過**: rollouts=128 で turn_p95 > 0.7s が安定 → rollouts を 64 に削減
- **commit_missions_in_subset の budget 共有 bug**: 同 source に複数 mission が割り当てられる場合の ship 不足。case4 既存の `spent_total` / `source_inventory_left` の closure を使って解決
- **case4 等価 regression 破壊**: `ACTION_MCTS_ENABLED=False` で case4 等価動作する事を unit test で保証
- **3 連敗のパターン (PGS / NaïveMCTS / Action-level すべて 0%)**: 探索系全体の構造問題確定 → 学習方向への切替判断材料

## 期待される結果のシナリオ

| case13 vs v4 (n=30 Stage B) | 解釈 | 次 |
|---|---|---|
| ≥55% | 採用候補、Stage C で確定 | production 化検討 |
| 40-55% | 改善方向、hyperparameter sweep で +α | rollouts 256 で iter2 |
| 10-40% | 構造的に case4 base に届かないが case11/12 の 0% は脱出 | 別 evaluator (timeline-based) 試行 |
| <10% | mission-level でも script と同じ構造問題 | 学習方向に切替決定 |

## 進行管理

iter1 = Action-level MCTS v0、Stage A → B → C を順次実施。
v0 で <40% なら撤退判断、≥40% なら hyperparameter / evaluator tuning iter2 に進む。
