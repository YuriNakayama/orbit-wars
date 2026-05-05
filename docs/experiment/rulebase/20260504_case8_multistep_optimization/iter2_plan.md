# Rulebase/case8 — Multi-step Beam (iter2)

> 作成日: 2026-05-05
> 関連:
> - [`iter1_plan.md`](./iter1_plan.md) / [`iter1_result.md`](./iter1_result.md) — vs v4 32.3% で却下
> - [`docs/experiment/rulebase/20260420_case3_rollout_ablation/result.md`](../20260420_case3_rollout_ablation/result.md) — true2p の seat 対称性 ±0.35pp の知見
> - replay analysis: `data/output/experiment/rulebase/case8/replay_analysis/20260505_1230/result_{1,2}.md`
> スコープ: case8 内 (新 case を切らない)。`baseline/planner/` に手を入れ、評価関数を差し替える + 敵反応 folding を有効化する

## 仮説 (Hypothesis)

iter1 の case8 が greedy より大幅劣化 (vs v4 = 32.3%) した主因は、**評価関数 `score_commitments` が選ぶ ordering と、greedy の `mission.score` が選ぶ ordering が逆方向に効いている** こと。これに加え `BEAM_OPPONENT_MODE="static"` が **守備手薄化** を招いている (replay 分析の仮説 A/B)。

iter2 では以下 2 軸を同時に改修して 200 戦で再評価する:

1. **評価関数を `mission.score` 合計ベース** に切替 — greedy と完全に整合する評価軸の上で beam が ordering 探索だけ行う。これにより最悪でも greedy と同等 (= seed=greedy ordering を採用) になる
2. **`BEAM_OPPONENT_MODE="true2p_light"`** を default に — case3 G の知見 (敵反応 folding で seat 対称性 ±0.35pp、勝率 +1.0pp) を取り込む

評価関数の不整合と敵モデル不在の **2 重失敗** を一括で潰すと、合算勝率は **iter1 の 32.3% → 50% 近辺まで戻る** はず。greedy と同等までは保証できる (シード ordering なので)。**+5pp 以上 (≥55%)** 改善は理論上未保証だが、敵反応 folding の case3 の +1.0pp + ordering 効果で達成を狙う。

## 既存コード (case8 iter1 から)

- `baseline/planner/beam.py` — 既存。orderings を beam 探索、greedy ordering を seed
- `baseline/planner/evaluator.py` — `score_commitments(world, planned_commitments, horizon, weights)`: horizon-end の self net_ships + production - enemy_threat を線形和。**iter1 の主犯**
- `baseline/planner/candidate.py` — `commit_missions_in_order` 既存維持
- `baseline/core/config.py:262-279` — `BEAM_*` 7 個。`BEAM_OPPONENT_MODE` は宣言だけで実装に未配線

## スコープ (Scope)

### 変更ファイル

```
bot/pipeline/rulebase/case8/baseline/
├── planner/
│   ├── evaluator.py            ★ score_commitments を mission.score 合計ベースに置換
│   │                             (旧版は score_commitments_legacy として残し ablation 可能に)
│   ├── beam.py                 ★ true2p_light モード時に enemy reaction を folding
│   └── opponent_reaction.py    ★ NEW: case3/baseline/rollout.py から複製
│                                  (_strongest_enemy / _enemy_reaction_arrivals / _infer_action_target /
│                                   plan_moves(light=True) 呼び出し)
└── strategy.py                 ★ plan_moves(world, light=False) を追加 (敵反応用の軽量 path)
```

### 設計詳細

#### 1. `evaluator.py`: 評価関数の置換

```python
def score_commitments(
    world: WorldModel,
    planned_commitments: dict[int, list[tuple[int, int, int]]],
    selected_missions: list[Mission],   # 新規引数: ordering で採用された missions
    horizon: int,
    weights: tuple[float, float, float],   # 後方互換のため維持、このバージョンは未使用
) -> float:
    return sum(m.score for m in selected_missions)
```

- beam 側 (`commit_missions_in_order`) に「採用された mission のリスト」を返させる必要あり (現状は `MovesPlan.moves` のみ)。`MovesPlan` に `accepted_missions: list[Mission]` を追加
- 旧版は `score_commitments_legacy` にリネームして残す。`BEAM_EVALUATOR_MODE: str = "mission_score"` config で切替可能 (default = mission_score)

#### 2. `opponent_reaction.py`: 敵反応 folding (case3 流用)

case3/baseline/rollout.py から下記関数を複製 (cross-case import 禁止のため):

```python
def _strongest_enemy(world: WorldModel) -> int | None: ...
def _infer_action_target(src, angle, planets, ships) -> Planet | None: ...
def _enemy_reaction_arrivals(world, enemy_id, our_send_src, our_send_ships) -> dict[int, list]: ...
```

`_enemy_reaction_arrivals` は内部で `from ..strategy import plan_moves` を呼び `plan_moves(enemy_world, light=True)` で敵 1-ply を取得。再帰防止のため module-local `_REACTION_DEPTH` で制限。

#### 3. `beam.py`: true2p folding 統合

```python
def run_beam(..., opponent_mode: str = "static"):
    ...
    for new_order in expansions:
        plan = commit_missions_in_order(...)
        extra_arrivals = {}
        if opponent_mode == "true2p_light":
            extra_arrivals = _enemy_reaction_arrivals(world, ...)
        cand_score = score_commitments(
            world, merge(plan.planned_commitments, extra_arrivals),
            plan.accepted_missions, horizon, weights,
        )
        ...
```

新評価関数は `mission.score` 合計だけを見るので、敵反応 folding は **score 合計には影響しない** (mission 採否 = 自軍ロジック)。ただし将来 `score_commitments_legacy` に戻す場合のために統合経路を作っておく。**iter2 default では敵反応 folding は score に効かないが、`commit_missions_in_order` で敵反応を `predicted_arrivals` に取り込むことで mission 採否 (`ships_needed_to_capture` の計算) に間接的に効く** 設計とする。

#### 4. `strategy.py`: light=True エントリポイント

```python
def plan_moves(world, consecutive_holds=None, accumulate_holds_consec=None, light=False):
    ...
    if light:
        # 敵反応用の軽量 path: greedy + followup/evacuation/rear_guard skip
        ...
        return moves_only
    # 通常 path
    ...
```

`light=True` では `BEAM_ENABLED` を無視して greedy ordering を使う (再帰防止 + 軽量化、case3 と同じ方針)。

### config 変更 (`baseline/core/config.py`)

```python
BEAM_OPPONENT_MODE: str = "true2p_light"   # iter1: "static" → iter2: "true2p_light"
BEAM_EVALUATOR_MODE: str = "mission_score" # iter2 新規; "legacy_net_ships" で旧版
# 他は据え置き
```

## 実装ステップ

1. `MovesPlan` に `accepted_missions: list[Mission]` を追加 (`candidate.py`)
2. `commit_missions_in_order` で commit 成功した mission を `accepted_missions` に append (move が 1 件以上 emit された mission のみ)
3. `evaluator.py` を rewrite: `score_commitments` (mission_score モード) + `score_commitments_legacy` (旧 net_ships モード) を両方持つ
4. `opponent_reaction.py` を case3/rollout.py からコピー → 不要関数 (`_baseline_net_ships`, `_mission_value*`, `rollout_*`) を削除して必要 3 関数だけ残す
5. `strategy.py` に `light=False` 引数追加 + 既存 path を温存 + light path 実装
6. `beam.py` で `opponent_mode == "true2p_light"` の場合 `extra_arrivals` を `commit_missions_in_order` に渡す (新引数 `extra_arrivals: dict | None = None`)
7. `commit_missions_in_order` で `world.predicted_arrivals` の代わりに merge した arrivals を見るように `world` を浅コピーした dummy で渡す (or 引数追加)
8. `core/config.py` に `BEAM_EVALUATOR_MODE` 追加、`BEAM_OPPONENT_MODE` を `"true2p_light"` default
9. `tests/pipeline/rulebase/case8/test_beam_off_equals_greedy.py` 引き続き green を確認
10. `tests/pipeline/rulebase/case8/test_evaluator_mission_score.py` を新規追加 (mission_score モードの sanity test)
11. `dev/test-bot` 緑、`uv run pytest tests/pipeline/rulebase/case8 -x` 緑

## 検証方法

### ローカル

```bash
# fast loop
uv run --directory bot pytest tests/pipeline/rulebase/case8 -x --no-header -q

# beam off 等価性 regression
uv run --directory bot pytest tests/pipeline/rulebase/case8/test_beam_off_equals_greedy.py -x

# smoke 30 戦
uv run --directory bot python -m dataset run --agents baseline_v4,baseline_v8 --mode 1v1 -n 30 --seed 50000 --parallel 4 --no-save-replay
```

### 性能評価

```bash
# 200 戦 (seat 入替: seed_a=50000+150, seed_b=50500+150)
uv run --directory bot python -m dataset run --agents baseline_v4,baseline_v8 --mode 1v1 -n 100 --seed 50000 --parallel 4 --no-save-replay
uv run --directory bot python -m dataset run --agents baseline_v8,baseline_v4 --mode 1v1 -n 100 --seed 50500 --parallel 4 --no-save-replay
```

- **対戦相手**: `baseline_v4` (production, LB745) を主軸
- **エピソード数**: 合算 **200 戦** (seat0=100, seat1=100)、user 指定。300 戦より粗いが iter2 は方向性確認のため許容
- **主要メトリクス**: 合算勝率 (vs v4)、Kaggle publicScore は使用しない
- **採否しきい値**: **+5pp 以上 (合算 ≥55%)** で iter3 へ進む。**iter1 の 32.3% から 50% 復帰** が最小成功線、それ未満は本方向の撤退検討
- **Time budget**: turn_p95 ≤ 0.7s。true2p_light で `plan_moves(light=True)` を全 ordering 候補で呼ぶため iter1 より遅くなる懸念。0.7s 超なら `BEAM_BRANCH_LIMIT` を 8→4 に下げる

## リスク

- **計算予算超過**: true2p_light は 1 ordering につき敵 plan_moves 1 回呼び出し。BEAM_BRANCH_LIMIT=8, BEAM_WIDTH=4, HORIZON=2 だと最大 ~32 回 / ターン × 軽量 plan_moves。case3 G で 0.47s だったので 0.5-0.6s 帯と推定、安全帯ぎりぎり
- **mission.score 評価でも greedy 同等止まり**: 仮説 B (敵モデル) の効果がメイン。0.6-0.8 倍 = 32% × 1.5 = 48% 程度の復帰でも、+5pp 達成は不確か
- **無限再帰**: `plan_moves(light=True) → enemy_reaction → plan_moves(light=True)` の再帰。`_REACTION_DEPTH` で 1 ply に制限済 (case3 と同パターン)
