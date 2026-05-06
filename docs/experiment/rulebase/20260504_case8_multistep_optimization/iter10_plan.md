# Rulebase/case12 — Naïve MCTS (Combinatorial Multi-armed Bandit)

> 作成日: 2026-05-05
> 関連:
> - [`docs/experiment/rulebase/20260505_case11_portfolio_search/iter1_result.md`](../20260505_case11_portfolio_search/iter1_result.md) — PGS 4 連敗 (0%)、構造的に「1 source 1 script」が case4 の rich mission と相性悪い
> - memory: `project_heuristic_search_saturation.md` (10 連敗で heuristic 飽和、NaïveMCTS は推奨方向 #2)
> - memory: `project_thrash_filter_harm.md`, `project_case7_t14_trap.md`
> - 文献: Ontañón 2017, "Combinatorial Multi-armed Bandits for Real-Time Strategy Games" (arxiv 1710.04805)

## 仮説 (Hypothesis)

case11 PGS の構造制約 **「各 source は 1 script のみ」** が 0% 完敗の主因。case4 の **「1 source が複数 mission に並列貢献」** を再現するには、**action space を per-source の独立 arm として扱う Combinatorial Multi-armed Bandit (CMAB)** で sampling-based に探索する必要がある。

NaïveMCTS は Ontañón 2017 で large branching factor RTS で最良性能を示した手法。各 source の **action choice を独立 arm として UCB1 で sampling**、組合せた assignment を rollout 評価し、最頻採用の組合せを最終 move として出力。期待: vs `baseline_v4` で **≥50%** に到達 (case4 noise floor 50%、+5pp は heuristic 飽和を踏まえれば理想的でなくても上等)。

**Mechanism**:
- PGS の hill climbing は **deterministic で local optimum に固定**。case11 で `script_idle` 主体の局所最適に陥った
- NaïveMCTS は **stochastic sampling** で広範囲を探索、PGS が見ない組合せを試行
- 各 source の choice 空間を「mission family + send ratio」のように **より granular** に拡張可能 (PGS の 7 script より細かい branching)

## 既存コードの現状

- **新規 case 番号**: `case12` (free)、`baseline_v12` 未登録
- **Base**: `bot/pipeline/rulebase/case4/baseline/` (LB745 production)
- **再利用資産** (case11 から):
  - `planner/scripts.py` (7 script) — そのまま流用
  - `planner/evaluator.py` の `evaluate_assignment` — そのまま流用
- **case11 PGS の learning**:
  - HORIZON=20 でないと capture が評価されない (memory 化済)
  - capture_aggressive は reserve=max(5, prod*3) を残す
  - 「smaller is better」evaluator bias は sent ships を引かないことで自然解消

## スコープ (Scope)

### 新規 case 構成

```
bot/pipeline/rulebase/case12/                           # case4 全複製
├── main.py / __init__.py / README.md
├── baseline/
│   ├── core/config.py                                  # ★ NAIVE_MCTS_* config 追加
│   └── planner/                                         # ★ 新設
│       ├── __init__.py                                  # `run_naive_mcts` を export
│       ├── scripts.py                                   # case11 から複製 (改修済み版)
│       ├── evaluator.py                                 # case11 から複製
│       └── naive_mcts.py                                # ★ 新規: NaïveMCTS 実装
├── configs/                                             # case4 と同一
└── evaluation/                                          # case4 と同一

bot/src/dataset/selfplay/agents.py                      # `"baseline_v12": ...` 追加
bot/tests/pipeline/rulebase/case12/                     # 新規
bot/pyproject.toml                                       # case12 ignore 追加
```

### config 追加

```python
# core/config.py
NAIVE_MCTS_ENABLED: bool = True
NAIVE_MCTS_ROLLOUTS: int = 64       # sampling 回数 (per-turn)
NAIVE_MCTS_EXPLORATION: float = 1.41  # UCB1 c parameter (sqrt(2))
NAIVE_MCTS_PLAYOUT_HORIZON: int = 20  # case11 v1 と同じ
NAIVE_MCTS_TIME_BUDGET_S: float = 0.6  # 1 ターン上限
```

### NaïveMCTS アルゴリズム (Ontañón 2017)

各 source planet ごとに **独立な arm 選択** を行い、その組合せ全体を 1 サンプルとして評価する:

```python
def run_naive_mcts(world, scripts, rollouts, horizon, time_budget_s):
    # Per-source arm statistics: {(src_id, script_name) -> (visits, total_reward)}
    arm_stats: dict[tuple[int, str], tuple[int, float]] = {}
    deadline = time.perf_counter() + time_budget_s

    for _ in range(rollouts):
        if time.perf_counter() >= deadline: break

        # Sample assignment: each source picks a script via UCB1
        assignment: dict[int, str] = {}
        for src in world.my_planets:
            # UCB1 over scripts for this source
            best_script = ucb1_select(src.id, scripts, arm_stats, exploration_c)
            assignment[src.id] = best_script

        # Evaluate the assignment via playout
        reward = evaluate_assignment(world, assignment, horizon)

        # Backprop reward to each (src, script) arm used
        for src_id, script_name in assignment.items():
            v, r = arm_stats.get((src_id, script_name), (0, 0.0))
            arm_stats[(src_id, script_name)] = (v + 1, r + reward)

    # Final assignment: per source, pick the most-visited script
    final: dict[int, str] = {}
    for src in world.my_planets:
        best_script, best_visits = "idle", 0
        for s in scripts:
            v, _ = arm_stats.get((src.id, s.name), (0, 0.0))
            if v > best_visits:
                best_script, best_visits = s.name, v
        final[src.id] = best_script

    return materialize_moves(world, final)
```

### UCB1 selection per source

```python
def ucb1_select(src_id, scripts, arm_stats, c):
    total_visits = sum(v for (s, _), (v, _) in arm_stats.items() if s == src_id)
    if total_visits == 0:
        return random.choice([s.name for s in scripts])
    best, best_score = None, -inf
    for s in scripts:
        v, r = arm_stats.get((src_id, s.name), (0, 0.0))
        if v == 0:
            return s.name  # explore unvisited arms first
        avg = r / v
        ucb = avg + c * math.sqrt(math.log(total_visits) / v)
        if ucb > best_score:
            best, best_score = s.name, ucb
    return best
```

## 実装ステップ

1. `cp -r bot/pipeline/rulebase/case4 bot/pipeline/rulebase/case12`
2. `case12/` の case4 → case12 参照置換 (main.py / README.md)
3. `core/config.py` に `NAIVE_MCTS_*` 5 個追加
4. `planner/__init__.py` 新規 + 3 ファイル:
   - `scripts.py` — case11 から複製 (v3 改修版、reserve=max(5, prod*3))
   - `evaluator.py` — case11 から複製
   - `naive_mcts.py` — 新規 ~120 行
5. `strategy.py:plan_moves` 冒頭で `NAIVE_MCTS_ENABLED` なら `run_naive_mcts(...)` に delegate
6. `bot/src/dataset/selfplay/agents.py` に `baseline_v12` 追加、`pyproject.toml` ignore 追加
7. `tests/pipeline/rulebase/case12/`:
   - `test_baseline_agent.py` (smoke + slow integration)
   - `test_naive_mcts_basic.py` (UCB1 が unvisited arm を最初に explore する unit test)
   - `test_mcts_off_equals_case4.py` (`NAIVE_MCTS_ENABLED=False` で case4 等価)

## 検証方法

### ローカル

```bash
uv run --directory bot pytest tests/pipeline/rulebase/case12 -m "not slow" -x
```

### 性能評価 (3 段階)

#### Stage A: 10戦 smoke (early signal)

```bash
uv run --directory bot python -m dataset run --agents baseline_v4,baseline_v12 \
    --mode 1v1 -n 10 --seed 100000 --parallel 4 --no-save-replay
```

判定:
- ≥30% → Stage B 30戦に進む
- 10-30% → debug (rollouts 不足 / horizon 短すぎ)
- <10% → 構造問題、撤退検討

#### Stage B: 30戦 + replay

```bash
uv run --directory bot python -m dataset run --agents baseline_v4,baseline_v12 \
    --mode 1v1 -n 30 --seed 100100 --parallel 4
```

判定:
- ≥50% → Stage C 200戦
- 40-50% → 改善方向、hyperparameter sweep (rollouts 64→128)
- <40% → 撤退、別方向

#### Stage C: 200戦

```bash
uv run --directory bot python -m dataset run --agents baseline_v4,baseline_v12 \
    --mode 1v1 -n 100 --seed 110000 --parallel 4 --no-save-replay
uv run --directory bot python -m dataset run --agents baseline_v12,baseline_v4 \
    --mode 1v1 -n 100 --seed 110500 --parallel 4 --no-save-replay
```

しきい値 ≥55% で採用、production 候補に上げる。

## リスクと早期撤退条件

- **計算予算超過**: rollouts=64 で turn_p95 > 0.7s が安定 → rollouts を 32 に削減 + horizon を 15 に削減
- **PGS と同じ low-throughput 0%**: NaïveMCTS でも script-only 制約が問題なら、scripts を「mission per source」に粒度変更 (iter2 で対応)
- **memory `project_heuristic_search_saturation` の予言通り 10 連敗 → 11 連敗**: heuristic 系の物理限界が確定、本ディレクトリも撤退して学習方向 (推奨 #1) へ

## 期待される結果のシナリオ

| case12 NaïveMCTS vs v4 (n=30) | 解釈 | 次 |
|---|---|---|
| ≥55% | 採用、Stage C で確定 | production 化検討 |
| 40-55% | sampling は機能、hyperparameter で +α 期待 | rollouts 倍増で iter2 |
| <40% | script-only モデルの構造問題が NaïveMCTS でも解消せず | 学習方向 (推奨 #1) に切替 |

## 参考 (References)

- [Combinatorial Multi-armed Bandits for Real-Time Strategy Games (Ontañón 2017, arxiv 1710.04805)](https://arxiv.org/abs/1710.04805) — 本実験の理論基盤、NaïveMCTS は large branching factor で他 sampling 手法を上回る
- [The Combinatorial Multi-armed Bandit Problem and Its Application to RTS Games (Ontañón 2013, AIIDE)](https://ojs.aaai.org/index.php/AIIDE/article/view/12681) — CMAB の元論文
- case11 PGS の経験的 learnings: `docs/experiment/rulebase/20260505_case11_portfolio_search/iter1_result.md`

## 進行管理

iter1 = NaïveMCTS v0 base 実装、Stage A → B → C を順次実施。
v0 で <40% なら撤退判断、≥40% なら hyperparameter tuning iter2 に進む。
本ディレクトリでの学習方向への切替判断は v0 結果次第。
