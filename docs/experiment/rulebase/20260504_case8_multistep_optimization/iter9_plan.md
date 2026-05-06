# Rulebase/case11 — Multi-step Optimization (Portfolio Search Family)

> 作成日: 2026-05-05
> 関連:
> - [`docs/experiment/rulebase/20260504_case8_multistep_beam/iter3_result.md`](../20260504_case8_multistep_beam/iter3_result.md) — beam search 飽和 (vs v4 ~30%)
> - [`docs/experiment/rulebase/20260505_case10_accumulate_step_guard/iter4_result.md`](../20260505_case10_accumulate_step_guard/iter4_result.md) — heuristic 系 8 連敗、完全飽和
> - memory: `project_thrash_filter_harm.md` (減点系 filter 飽和)
> スコープ: case4 base 全複製で新規 `case11` を切り、**3 段階の portfolio search 系手法を iter 単位で順次検証**

## 全体仮説 (Hypothesis)

case8 (beam search) と case10 (heuristic 改修) で確認した **「heuristic score の補正/並び替え」方針の飽和** を、**「mission ordering の探索」ではなく「source 単位の script 割当の探索」** という直交軸で打破する。

具体的には RTS AI 文献 (Churchill & Buro 2013、Moraes 2018、Ontañón 2013/2017) で実績のある **portfolio search 系 3 手法** を順次評価:

1. **iter1: Portfolio Greedy Search (PGS)** — 各 source に script を hill climbing で割り当て (本命、最低コスト)
2. **iter2: Nested-Greedy Search (NGS)** — PGS の上位互換、enemy reaction を inner loop で folding (中コスト)
3. **iter3: Naïve MCTS (CMAB sampling)** — combinatorial action space を CMAB で sampling-based MCTS (高コスト、最終手段)

各 iter で個別に Stage A (30 戦) → Stage B (200 戦) を実施し、しきい値 ≥55% で採用判定。前段 iter が ≥55% に届いたら次 iter は **任意** (cost-benefit で判断)。届かなかった場合は次 iter に進む。

## 既存コードの現状

- **新規 case 番号**: `case11` (free)、`baseline_v11` 未登録
- **Base**: `bot/pipeline/rulebase/case4/baseline/` (LB745 production、case7 base での試行が全敗していたため case4 base に変更)
- **case4 missions** (script 候補ソース): `capture, snipe, swarm, harass, reinforcement, crash_exploit, fleet_consolidation` (案配は再構成)
- **score 集約点**: `strategy_helpers.score_attack` → `apply_score_modifiers`
- **過去 iter の所見**:
  - case8 iter1-3 で beam 系飽和 (vs v4 ~30%)
  - case9 で thrash filter 害 (-10pp)、case10 iter1-4 で step guard / 動的化など 8 連敗
  - case4 vs case4 ablation = 50% (noise floor)
- **ゴール**: vs `baseline_v4` で ≥55% を達成、production 候補に上げる

## スコープ (Scope)

### 新規 case 構成

```
bot/pipeline/rulebase/case11/                           # case4 全複製
├── __init__.py
├── main.py                                              # 参照を case11 に置換
├── README.md
├── baseline/
│   ├── ...                                              # case4 全コピー
│   ├── core/config.py                                   # ★ PORTFOLIO_*, NGS_*, NAIVE_MCTS_* 追加 (iter ごとに増)
│   └── planner/                                         # ★ 新設サブパッケージ
│       ├── __init__.py
│       ├── scripts.py                                   # ★ iter1 で各 source に割り当てる script を定義
│       ├── portfolio_greedy.py                          # ★ iter1: PGS 実装
│       ├── nested_greedy.py                             # ★ iter2: NGS 実装 (PGS extends)
│       ├── naive_mcts.py                                # ★ iter3: NaïveMCTS 実装
│       └── evaluator.py                                 # ★ playout / value function (3 iter で共通)
└── (configs / evaluation は case4 と同一)

bot/src/dataset/selfplay/agents.py                      # `"baseline_v11": ...` 追加
bot/tests/pipeline/rulebase/case11/                     # 新規
bot/pyproject.toml                                       # case11 ignore 追加
```

### config 追加 (各 iter で増)

```python
# core/config.py — iter1 開始時
PORTFOLIO_ENABLED: bool = True
PORTFOLIO_HILL_CLIMB_ITERS: int = 3       # 各 source 1 回ずつ × 3 ラウンド
PORTFOLIO_PLAYOUT_HORIZON: int = 8         # forward simulation 深さ
PORTFOLIO_TIME_BUDGET_S: float = 0.6      # 1 ターン上限

# iter2 で追加
NGS_ENABLED: bool = False                  # iter2 で True
NGS_INNER_HILL_CLIMB_ITERS: int = 2

# iter3 で追加
NAIVE_MCTS_ENABLED: bool = False           # iter3 で True
NAIVE_MCTS_ROLLOUTS: int = 64
NAIVE_MCTS_EXPLORATION: float = 1.41
```

## 共通実装基盤 (3 iter で再利用)

### `planner/scripts.py` — script ライブラリ

各 source planet に割り当てる **script** を 5-7 個定義:

```python
@dataclass(frozen=True)
class Script:
    name: str
    fn: Callable[[Planet, WorldModel, dict[str, Any]], list[list[int|float]]]

SCRIPTS = [
    Script("idle",           script_idle),           # 何もしない (hold all)
    Script("capture_safe",   script_capture_safe),   # 最も近い safe neutral を確保
    Script("capture_max",    script_capture_max),    # 最高 value target に full-commit
    Script("snipe",          script_snipe),          # 敵 fleet を撃ち落とす
    Script("harass",         script_harass),         # 敵 production 削り
    Script("reinforce",      script_reinforce),      # 自軍弱点に補給
    Script("consolidate",    script_consolidate),    # case4 fleet_consolidation
]
```

各 script は **1 source のみ** を見て move を返す (純粋関数化)。既存の `missions/*.py` ロジックを script に re-wrap するだけで済む (実装は ~150 行)。

### `planner/evaluator.py` — playout-based value function

```python
def evaluate_assignment(
    world: WorldModel,
    assignment: dict[int, str],   # source_id -> script_name
    horizon: int,
) -> float:
    """各 source に assignment を適用 → moves を集約 → simulate_planet_timeline で
    horizon ターン展開 → 自軍 net ships + production を返す。"""
```

**`simulate_planet_timeline`** (case8 で既存) を再利用、敵は 1-step greedy と仮定 (iter1)。iter2 では NGS 内で敵 reaction を folding。

## iter 別の詳細

### iter1: Portfolio Greedy Search (PGS)

**仮説**: 各 source に script を hill climbing で割り当てる方が、greedy mission ordering より勝率が高い。case4 base 上で +5pp 改善 (50% → 55%)。

**アルゴリズム** (Churchill & Buro 2013):
1. 初期 assignment: 全 source `script_idle`
2. 評価: `evaluate_assignment(world, current, horizon=PLAYOUT_HORIZON)`
3. Hill climb (PGS の核):
   ```
   for iter in range(HILL_CLIMB_ITERS=3):
     improved = False
     for source in sources:
       best_script = current[source]
       best_score = current_score
       for script in SCRIPTS:
         trial = current.copy(); trial[source] = script
         score = evaluate_assignment(world, trial, horizon)
         if score > best_score:
           best_script, best_score = script, score
       if best_script != current[source]:
         current[source] = best_script
         current_score = best_score
         improved = True
     if not improved: break
   ```
4. 最終 assignment を `script` ごとに展開して moves emit

**コスト分析**: O(sources × scripts × iters × playout) = 8 × 7 × 3 × O(planets × horizon=8) ≈ 大規模板で 1万 ops、turn_p95 0.6s 内見込み。

**Stage A (30 戦) → Stage B (200 戦) → result.md** で +5pp 達成判定。

### iter2: Nested-Greedy Search (NGS)

**仮説**: iter1 の PGS が ~52% で止まった場合、NGS で敵 reaction を inner loop で取り込めば +2-5pp 上乗せできる。

**アルゴリズム** (Moraes 2018):
- PGS の改良: 各 candidate script の評価時に **enemy も同じ PGS を 1 iter 走らせて反応を取得**
- 自分の playout に敵 reaction を folding して再評価
- 計算コストは PGS の 2-3x (= 各評価で enemy mini-PGS が走る)

**実装**: `planner/nested_greedy.py` を `portfolio_greedy.py` を継承して書く (recursion guard 付き、case3 true2p_light と同パターン)。

**iter1 が ≥55% 達成済の場合**: iter2 は skip (cost > benefit)。
**iter1 が 50-55% の場合**: iter2 を実施し +2-5pp の上乗せ確認。

### iter3: Naïve MCTS (CMAB sampling)

**仮説**: iter1/iter2 が <55% で止まった場合、MCTS で広範囲を sampling すれば +5pp 以上の改善が見込める。

**アルゴリズム** (Ontañón 2013/2017):
- 各 source の script choice を **arm** として CMAB を構成
- Naïve sampling: 各 source 単位で UCB1 で arm を選び、組合せた assignment を rollout
- `NAIVE_MCTS_ROLLOUTS=64` 回 sampling、最頻採用 assignment を出力

**実装**: `planner/naive_mcts.py`。rollout は PGS の `evaluate_assignment` を再利用。

**iter1/iter2 が ≥55% 達成済の場合**: iter3 は skip。
**iter1/iter2 ともに <55% の場合**: iter3 を最終手段として実施。

## 実装ステップ (iter1 のみ詳細、iter2/iter3 は別 plan で詰める)

### iter1 実装ステップ

1. `cp -r bot/pipeline/rulebase/case4 bot/pipeline/rulebase/case11` で全複製
2. `case11/__init__.py` / `main.py` / `README.md` の case4 → case11 参照置換
3. `baseline/core/config.py` に `PORTFOLIO_*` config 4 個追加
4. `baseline/planner/__init__.py` 新規 + 4 ファイル (scripts.py, portfolio_greedy.py, evaluator.py、nested_greedy.py / naive_mcts.py は空 stub で iter2/iter3 用予約)
5. `baseline/planner/scripts.py` で 7 script を実装 (各 ~20 行、既存 mission ロジックの薄いラッパ)
6. `baseline/planner/evaluator.py` で `evaluate_assignment(world, assignment, horizon)`
7. `baseline/planner/portfolio_greedy.py` で `run_pgs(world, scripts, ...)` を実装
8. `baseline/strategy.py` の `plan_moves` 冒頭で `PORTFOLIO_ENABLED` なら `run_pgs(...)` に delegate、戻り値の moves を採用
9. `bot/src/dataset/selfplay/agents.py` に `baseline_v11` 追加
10. `bot/pyproject.toml` に case11 ignore 追加
11. `bot/tests/pipeline/rulebase/case11/`:
    - `test_baseline_agent.py` (smoke + slow integration)
    - `test_pgs_basic.py` (PGS が `script_idle` と `script_capture_safe` から正しく上位を選ぶ)
    - `test_pgs_off_equals_case4.py` (`PORTFOLIO_ENABLED=False` で case4 等価)
12. lint / format / mypy / pytest 緑、`dev/test-bot` 通過

## 検証方法 (各 iter 共通)

### ローカル

```bash
uv run --directory bot pytest tests/pipeline/rulebase/case11 -m "not slow" -x
```

### 性能評価

#### Stage A: 30戦 + replay

```bash
uv run --directory bot python -m dataset run --agents baseline_v4,baseline_v11 \
    --mode 1v1 -n 30 --seed 90000 --parallel 4
```

判定:
- ≥55% → Stage B 200戦 で確認、しきい値達成
- 50-55% → 微妙、Stage B で seed variance 確認
- <50% → 実装 / hyperparameter 確認、必要なら撤退

#### Stage B: 200戦

```bash
uv run --directory bot python -m dataset run --agents baseline_v4,baseline_v11 \
    --mode 1v1 -n 100 --seed 90000 --parallel 4 --no-save-replay
uv run --directory bot python -m dataset run --agents baseline_v11,baseline_v4 \
    --mode 1v1 -n 100 --seed 90500 --parallel 4 --no-save-replay
```

しきい値 ≥55% で採用 (iter1/iter2/iter3 共通)。

### 副次評価

各 iter の result.md で **平均 turn_p95** と **PGS hill climb 平均反復数** を記録。

## リスクと早期撤退条件

### 全 iter 共通

- **計算予算超過**: turn_p95 > 0.7s が安定発生 → playout horizon を 8→4 に削減 / hill climb iters を 3→2 に削減
- **case4 base 互換性破壊**: `PORTFOLIO_ENABLED=False` で case4 等価動作を unit test で保証
- **PGS / NGS / NaïveMCTS 全敗**: heuristic 系の 8 連敗に portfolio 系も加わると、構造的に「 case4 + 何か」が production を超えるルートが無いことが確定 → 学習ベース value function に方向転換

### iter 別

| iter | 主リスク | 撤退条件 |
|---|---|---|
| iter1 PGS | script 数不足で hill climb が局所解 | Stage A <50% で iter1 撤退、iter2 (NGS) はさらにコスト上、撤退判断慎重に |
| iter2 NGS | enemy mini-PGS が turn_p95 破綻 | turn_p95 > 0.8s で recursion guard 強化、無理なら撤退 |
| iter3 NaïveMCTS | 64 rollouts で turn_p95 破綻 | rollouts 削減、無理なら学習方向転換 |

## 期待される結果のシナリオ

| iter1 PGS vs v4 | 解釈 | 次の iter |
|---|---|---|
| ≥55% | 採用候補、production 化検討 | iter2 任意 (+α 狙い) |
| 50-55% | 改善方向だが微妙、hyperparameter sweep 検討 | iter2 必須 (NGS で reaction folding) |
| <50% | PGS では足りず | iter2 で改善期待薄、iter3 (MCTS) を最終手段 |

## 参考 (References) — Step 3 Web research

- [Portfolio Greedy Search and Simulation for Large-Scale Combat in StarCraft (Churchill & Buro 2013)](https://skatgame.net/mburo/ps/combat13.pdf) — PGS の元論文。hill climbing で各 unit に script を割り当てる手法、本実験 iter1 の直接基盤
- [Nested-Greedy Search for Adversarial Real-Time Games (Moraes 2018, AAAI)](https://cdn.aaai.org/ojs/13017/13017-52-16534-1-2-20201228.pdf) — PGS 上位互換、中小マップで state-of-the-art。本実験 iter2 の根拠
- [Combinatorial Multi-armed Bandits for Real-Time Strategy Games (Ontañón 2017, arXiv 1710.04805)](https://arxiv.org/abs/1710.04805) — Naïve Sampling MCTS、large branching factor で他手法を上回る。本実験 iter3 の根拠
- [Portfolio Search and Optimization for General Strategy Game-Playing (arxiv 2104.10429)](https://arxiv.org/pdf/2104.10429) — General Strategy Game Playing 視点で PGS 系を整理、評価関数設計の参考

## 進行管理

iter 単位で plan.md (本ファイル) を `iter1_result.md` / `iter2_result.md` / `iter3_result.md` に追記する形で結果を記録。各 iter が ≥55% に達した時点で次 iter は 任意化、達しない場合のみ次に進む。最終 result は `final_result.md` で 3 iter を統合してまとめる。
