# case5 (LB 1224 rulebase) — アーキテクチャ設計

## 1. 全体図

```
              ┌────────────────────┐
              │   Kaggle Runtime   │
              │   env.step(obs)    │
              └─────────┬──────────┘
                        │
                        ▼
        ┌─────────────────────────────────┐
        │  pipeline/rulebase/case5/main.py │  ← 20行 wrapper (Path.cwd() 注入)
        └─────────────────┬───────────────┘
                          │ from baseline.agent import agent
                          ▼
        ┌─────────────────────────────────┐
        │       baseline/agent.py          │
        │   - _read / _detect_game_id     │
        │   - build_world(obs)            │
        │   - agent(obs, config) ◀─────── deadline 計算
        └─────┬─────────────┬─────────────┘
              │             │
              ▼             ▼
       core/world_model    strategy.plan_moves(world, *, deadline)
              │                       │
   ┌──────────┼──────────┐            ├──── collect_missions ── missions/*.py
   │          │          │            │       (capture/snipe/swarm/rescue/
   │     core/types  core/physics     │        recapture/reinforcement/crash_exploit)
   │     core/geometry core/config    │
   │     core/timing                  ├──── resolve_missions ── strategy_helpers
   │                                  │       (settle_plan / apply_score_modifiers)
   │                                  │
   │                                  └──── apply_movements ── movements/*.py
   │                                          (evacuation/rear_guard/followup/
   │                                           proactive_defense)
   │
   └── 各層は core/timing.Deadline を引数で受け取り should_skip() でスキップ判定
```

## 2. ディレクトリ構造

```
pipeline/rulebase/case5/
├── main.py                          # 20行 wrapper (case4 と同テンプレ)
├── baseline/
│   ├── __init__.py                  # 出典/ライセンスコメント + from .agent import agent
│   ├── LICENSE                      # Apache 2.0 全文
│   ├── agent.py                     # ~120行: build_world + agent + deadline 計算
│   ├── strategy.py                  # ~150行: plan_moves (collect → resolve → apply)
│   ├── strategy_helpers.py          # ~400行: build_modes/target_value/settle_plan/apply_score_modifiers
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py                # ~250行: 120+ 定数 (notebook L7-177 を移植)
│   │   ├── geometry.py              # ~80行: dist, basic geometry
│   │   ├── physics.py               # ~280行: orbit_radius/predict_*/aim_with_prediction
│   │   ├── timing.py                # ★新規 ~60行: Deadline dataclass + should_skip
│   │   ├── types.py                 # ~70行: Planet/Fleet namedtuple, ShotOption/Mission frozen dataclass
│   │   └── world_model.py           # ~750行: WorldModel + 集計ヘルパー (build_arrival_ledger 等)
│   ├── missions/
│   │   ├── __init__.py              # collect_missions(world, deadline) を集約
│   │   ├── capture.py               # ~160行
│   │   ├── snipe.py                 # ~100行
│   │   ├── swarm.py                 # ~200行 (2源/3源切替を含む)
│   │   ├── reinforcement.py         # ~120行
│   │   ├── rescue.py                # ★新規 ~120行
│   │   ├── recapture.py             # ★新規 ~140行
│   │   └── crash_exploit.py         # ~110行 (notebook の洗練版)
│   └── movements/
│       ├── __init__.py
│       ├── evacuation.py            # ~120行
│       ├── rear_guard.py            # ~120行
│       ├── followup.py              # ~120行
│       └── proactive_defense.py     # ★新規 ~120行
├── configs/
│   └── baseline.yaml                # ablation 設定 (case4 と同形式)
└── evaluation/
    ├── __init__.py
    ├── ablation.py                  # configs/baseline.yaml で各 mission on/off
    ├── compare_v2.py                # case4 と勝率比較
    └── snapshot_update.py           # snapshot 用 obs/action 更新
```

`tests/pipeline/rulebase/case5/` (テスト):

```
tests/pipeline/rulebase/case5/
├── __init__.py
├── snapshots/
│   ├── obs_seed0_turn10.json
│   └── action_seed0_turn10.json
├── test_baseline_agent.py           # smoke (env.run) + snapshot + action shape
├── test_world_model.py              # WorldModel timeline / blood_in_water / exposed_planets
├── test_timing.py                   # Deadline.should_skip / fake clock
├── test_missions_capture.py
├── test_missions_rescue.py          # 落ちる惑星の救援判定
├── test_missions_recapture.py       # 失った惑星の再奪還判定
├── test_missions_crash_exploit.py
├── test_strategy_helpers.py         # build_modes / settle_plan / apply_score_modifiers
└── test_movements_proactive.py      # 複数敵 stack window 検出
```

## 3. フロントエンド設計

該当なし (バックエンドエージェントのみ)。

## 4. バックエンド設計

### 4.1 主要モジュールの責務

#### `core/types.py`

```python
from dataclasses import dataclass, field, replace
from typing import NamedTuple

class Planet(NamedTuple):
    id: int
    owner: int
    x: float
    y: float
    radius: float
    ships: int
    production: int

class Fleet(NamedTuple):
    id: int
    owner: int
    x: float
    y: float
    angle: float
    from_planet_id: int
    ships: int

@dataclass(frozen=True, slots=True)
class ShotOption:
    angle: float
    eta: int
    arrival_x: float
    arrival_y: float
    safe: bool

@dataclass(frozen=True, slots=True)
class Mission:
    kind: str            # "capture" | "snipe" | "rescue" | "recapture" | "reinforce" | "crash_exploit" | "swarm"
    source_id: int
    target_id: int
    angle: float
    ships: int
    score: float
    eta: int
    extras: tuple = ()   # mission-specific options

    def with_score(self, score: float) -> "Mission":
        return replace(self, score=score)
```

#### `core/timing.py` (★新規)

```python
import time
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class Deadline:
    started_at: float
    deadline_at: float

    @classmethod
    def from_config(cls, config: dict | None, *, soft_max: float, fraction: float) -> "Deadline":
        start = time.perf_counter()
        act_timeout = (config or {}).get("actTimeout", 1.0)
        return cls(start, start + min(soft_max, act_timeout * fraction))

    def remaining(self, now: float | None = None) -> float:
        return self.deadline_at - (now if now is not None else time.perf_counter())

    def should_skip(self, min_time: float) -> bool:
        return self.remaining() < min_time
```

#### `core/world_model.py`

case4 の `WorldModel` をベースに、以下のフィールドを **追加**:

- `exposed_planet_ids: frozenset[int]`
- `blood_in_water_owners: frozenset[int]`
- `enemy_fights_at_neutrals: dict[int, list[int]]`
- `indirect_feature_map: dict[int, IndirectFeatures]`
- `keep_needed_map: dict[int, int]`  (notebook 仕様に拡張)
- `holds_full_map: dict[int, int]`
- `fall_turn_map: dict[int, int]`
- `stacked_enemy_keep: dict[int, int]` (proactive defense 用)

case4 の `predicted_arrivals`/`opponent_threat_score` フィールドは **削除**。

#### `strategy.py`

```python
def plan_moves(world: WorldModel, *, deadline: Deadline) -> list[list]:
    missions = collect_missions(world, deadline=deadline)
    planned = resolve_missions(world, missions, deadline=deadline)
    actions = apply_movements(world, planned, deadline=deadline)
    return [a.to_action_list() for a in actions]
```

各フェーズはサブモジュールに委譲し、ファイル本体は ~150 行に抑える。

#### `missions/__init__.py`

```python
def collect_missions(world: WorldModel, *, deadline: Deadline) -> list[Mission]:
    missions: list[Mission] = []
    missions.extend(build_capture_missions(world, deadline=deadline))
    if not deadline.should_skip(HEAVY_PHASE_MIN_TIME):
        missions.extend(build_snipe_missions(world, deadline=deadline))
        missions.extend(build_swarm_missions(world, deadline=deadline))
        missions.extend(build_rescue_missions(world, deadline=deadline))
        missions.extend(build_recapture_missions(world, deadline=deadline))
    missions.extend(build_reinforce_missions(world, deadline=deadline))
    if not deadline.should_skip(OPTIONAL_PHASE_MIN_TIME):
        missions.extend(build_crash_exploit_missions(world, deadline=deadline))
    return apply_score_modifiers(world, missions)
```

#### `agent.py`

```python
def agent(obs: dict, config: dict | None = None) -> list[list]:
    deadline = Deadline.from_config(
        config, soft_max=SOFT_ACT_DEADLINE, fraction=SOFT_ACT_FRACTION
    )
    world = build_world(obs)
    return plan_moves(world, deadline=deadline)
```

`build_world(obs)` は notebook と同様、obs を NumPy ベクトル化して `WorldModel` を返す。

### 4.2 ライセンス・出典の配置

`baseline/__init__.py` 冒頭:

```python
"""baseline_v5 — port of Kaggle notebook `orbit-star-wars-lb-max-1224`.

Adapted from https://www.kaggle.com/code/romantamrazov/orbit-star-wars-lb-max-1224
by Roman Tamrazov (Apache License 2.0).

Refactored for readability:
  - 480-line plan_moves split into collect/resolve/apply phases
  - Mission as frozen dataclass instead of mutable
  - Deadline control extracted to core/timing.py
"""
from .agent import agent

__all__ = ["agent"]
```

`baseline/LICENSE`: Apache 2.0 本文 (notebook 由来であることを NOTICE 部に明記)。

## 5. データモデル

新規スキーマ無し。すべての obs/action は既存の Orbit Wars 形式に従う。

## 6. インフラ変更

新規インフラ無し。既存の selfplay (`src/dataset/selfplay`) と submit (`src/submit`) パイプラインを利用。`.submitignore` も既存設定 (`evaluation/`, `configs/`) で対応可能。

## 7. 外部統合

- `src/dataset/selfplay/agents.py` の `AGENT_REGISTRY` に 1 行追加:
  ```python
  "baseline_v5": "pipeline.rulebase.case5.baseline.agent:agent",
  ```
- `kaggle-environments` の `make("orbit_wars")` を利用 (既存依存)
- 新規外部 API なし

## 8. 設計原則の遵守

| backend.md ルール | 遵守方法 |
|------------------|---------|
| 関数 < 50 行 | `plan_moves` を 3 段階に分解、各 mission builder も 50 行以内に保つ |
| ファイル 200-400 行 | 18 ファイルに分割、最大は `core/world_model.py` ~750 行 (case4 の 707 行と同レンジ) |
| `Any` 禁止 | `dict | None` / 具体的な型ヒントで対応、`obs` は `Mapping[str, object]` で受ける |
| `print` 禁止 | `logging.getLogger(__name__)` を使用、agent ホットパスでは原則ログ出力なし |
| frozen dataclass | `Mission` / `ShotOption` / `Deadline` / `ModeFlags` をすべて frozen 化 |
| 型ヒント必須 | 全関数に厳格に付与、mypy strict 通過 |
| ベクトル化 | `build_world` で planets/fleets を NumPy 化、ホットパスでループを避ける |
