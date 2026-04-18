# baseline-reinforce — アーキテクチャ設計

## 全体図

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Kaggle Submission (本番実行環境)                     │
│                                                                             │
│   pipeline/case1/baseline/main.py  ─── from .agent import agent             │
│                                          │                                  │
│                                          ▼                                  │
│                                   agent(obs, config) ─── list[[id,ang,n]]  │
└─────────────────────────────────────────────────────────────────────────────┘
                                          ▲
                                          │
┌─────────────────────────────────────────┴──────────────────────────────────┐
│                      pipeline/case1/baseline/  (core/missions/agent 3層)    │
│                                                                             │
│  ┌─────────────────── core/ ─────────────────┐                              │
│  │  config.py   CONFIG 定数 (80+)            │                              │
│  │  types.py    Planet / Fleet / ShotOption  │                              │
│  │              / Mission                     │                              │
│  │  geometry.py dist, segment_hits_sun, ...  │                              │
│  │  physics.py  fleet_speed, predict_*,      │                              │
│  │              search_safe_intercept        │                              │
│  │  world_model.py  WorldModel class         │                              │
│  │              (plan_shot, projected_state, │                              │
│  │               base_need_cache, reaction…) │                              │
│  │              build_arrival_ledger         │                              │
│  │              simulate_planet_timeline     │                              │
│  └────────────────────────────────────────────┘                             │
│                    ▲                                                         │
│   ┌────────────────┴──── missions/ ──────────────────┐                      │
│   │  snipe.py           build_snipe_mission (単数)     │                    │
│   │  reinforcement.py   build_reinforcement_missions  │                    │
│   │  crash_exploit.py   build_crash_exploit_missions  │                    │
│   │                     (4P 専用)                      │                    │
│   └─────────────────────────────────────────────────────┘                   │
│                    ▲                                                         │
│   ┌────────────────┴──── strategy.py ────────────────────┐                  │
│   │  plan_moves(world)  メインオーケストレーション        │                  │
│   │  (expansion/attack/swarm/followup/doomed/rear は     │                  │
│   │   plan_moves 内にインライン実装 — ノートブック準拠) │                  │
│   │  build_modes, target_value, preferred_send,          │                  │
│   │  apply_score_modifiers, opening_filter,              │                  │
│   │  is_safe_neutral, is_contested_neutral,              │                  │
│   │  planet_distance                                     │                  │
│   └───────────────────────────────────────────────────────┘                 │
│                    ▲                                                         │
│   ┌────────────────┴────────────────────────────────────┐                   │
│   │  agent.py   build_world(obs) / agent(obs)           │                   │
│   │             - obs から WorldModel 構築               │                   │
│   │             - strategy.plan_moves(world) を呼ぶ      │                   │
│   └──────────────────────────────────────────────────────┘                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                          ▲
                                          │ import
┌─────────────────────────────────────────┴──────────────────────────────────┐
│                      pipeline/case1/evaluation/  (自己対局 CLI)             │
│                                                                             │
│  selfplay.py  ── typer app                                                  │
│      ├── uv run python -m pipeline.case1.evaluation.selfplay [opts]        │
│      ├── 1v1 or ffa4                                                        │
│      ├── env.run([agent, agent])                                            │
│      ├── 計測: time.perf_counter / turn                                     │
│      ├── data/replays/case1/<ts>/episode_<i>.json                           │
│      └── rich table サマリ (勝率 / タイムアウト率 / 平均ターン)             │
└─────────────────────────────────────────────────────────────────────────────┘

┌───────────────── pipeline/case1/configs/ ─────────────────┐
│  baseline.yaml   CONFIG 全パラメータの YAML 化（将来用）  │
└────────────────────────────────────────────────────────────┘

┌───────────────── pipeline/case1/notebook/ ─────────────────┐
│  orbit-wars-2026-reinforce.ipynb  (kaggle kernels pull)    │
│  kernel-metadata.json                                       │
└─────────────────────────────────────────────────────────────┘

┌───────────────── tests/pipeline/case1/ ─────────────────┐
│  test_baseline_agent.py   動作確認 + snapshot           │
│  test_world_state.py      単体 (arrival ledger 等)      │
│  snapshots/                                             │
│    episode_seed0.json     全ターンの action 列           │
│    observation_seed0.json 初期観測                       │
└──────────────────────────────────────────────────────────┘
```

## ディレクトリ構造（新規作成物）

```
pipeline/case1/
├── __init__.py
├── README.md                         # ケース概要・実行方法
├── baseline/
│   ├── __init__.py                   # agent を再エクスポート
│   ├── LICENSE                       # Apache 2.0 本文
│   ├── main.py                       # Kaggle submission entrypoint
│   ├── agent.py                      # build_world + agent(obs)
│   ├── strategy.py                   # plan_moves + 戦略ヘルパー
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── types.py
│   │   ├── geometry.py
│   │   ├── physics.py
│   │   └── world_model.py
│   └── missions/
│       ├── __init__.py
│       ├── snipe.py
│       ├── reinforcement.py
│       └── crash_exploit.py
├── evaluation/
│   ├── __init__.py
│   └── selfplay.py                   # typer CLI
├── configs/
│   └── baseline.yaml                 # CONFIG を YAML 化
└── notebook/
    ├── orbit-wars-2026-reinforce.ipynb
    └── kernel-metadata.json

tests/
├── __init__.py
└── pipeline/
    ├── __init__.py
    └── case1/
        ├── __init__.py
        ├── test_baseline_agent.py
        ├── test_world_model.py
        └── snapshots/
            └── episode_seed0.json
```

### Python ファイルの役割と期待される行数

| ファイル | 責務 | 想定行数 |
|----------|------|----------|
| `baseline/core/config.py` | CONFIG 定数 | ~150 |
| `baseline/core/types.py` | Planet/Fleet/ShotOption/Mission | ~40 |
| `baseline/core/geometry.py` | 2D 幾何 | ~80 |
| `baseline/core/physics.py` | 速度・軌道予測・intercept | ~210 |
| `baseline/core/world_model.py` | WorldModel + forward sim + 補助関数 | ~550 |
| `baseline/missions/snipe.py` | build_snipe_mission (単数) | ~80 |
| `baseline/missions/reinforcement.py` | build_reinforcement_missions | ~130 |
| `baseline/missions/crash_exploit.py` | build_crash_exploit_missions (4P) | ~150 |
| `baseline/strategy.py` | plan_moves + ヘルパー | ~600 |
| `baseline/agent.py` | build_world + agent(obs) | ~60 |
| `baseline/main.py` | 再エクスポート | ~5 |
| `evaluation/selfplay.py` | typer CLI | ~200 |

## モジュール依存グラフ

```
main.py ─► agent.py ─► strategy.py
                          │
                          ├─► core/world_model ─► core/physics, geometry, types, config
                          └─► missions/*       ─► core/world_model, physics, geometry, config, types
```

- 循環依存は無し。`core/` は leaf、`missions/` が `core` に依存、`strategy` が `core + missions` を束ね、`agent` が `strategy` を呼ぶ。
- 本 feature では `src/` は一切 import しない（scope 外）。

## 主要インターフェース

### `agent.agent`
```python
def agent(observation: Any) -> list[list[int | float]]:
    """Kaggle entrypoint. Returns list of [from_planet_id, angle, num_ships]."""
```

### `core.world_model.WorldModel`
```python
class WorldModel:
    player: int
    planets: list[Planet]
    fleets: list[Fleet]
    remaining_steps: int
    ang_vel: float
    comets: list[dict[str, Any]]
    comet_ids: set[int]
    initial_by_id: dict[int, Planet]

    my_planets: list[Planet]
    enemy_planets: list[Planet]
    neutral_planets: list[Planet]
    planet_by_id: dict[int, Planet]
    arrivals_by_planet: dict[int, list[tuple[int, int, int]]]  # (eta, owner, ships)
    reserve: dict[int, int]
    doomed_planets: set[int]
    threatened_candidates: dict[int, dict[str, Any]]
    base_need_cache: dict[tuple[int, int], int]

    def plan_shot(self, src_id: int, tgt_id: int, cap: int) -> ShotOption | None: ...
    def projected_state(self, tgt_id: int, cutoff: int, commits: list, extras: list) -> dict: ...
    def ships_needed_to_capture(self, tgt_id: int, turns: int) -> int: ...
    def reinforcement_needed_for(self, planet_id: int) -> int: ...
    def reaction_times(self, tgt_id: int) -> dict[int, int]: ...
```

### `missions/*` インターフェース
```python
# snipe.py
def build_snipe_mission(world: WorldModel, ...) -> Mission | None: ...

# reinforcement.py
def build_reinforcement_missions(world: WorldModel, ...) -> list[Mission]: ...

# crash_exploit.py
def build_crash_exploit_missions(world: WorldModel, ...) -> list[Mission]: ...
```

expansion / attack / swarm / followup / doomed / rear のミッション組立ロジックは `strategy.plan_moves(world)` 内にインライン実装（ノートブックの `plan_moves` を忠実再現）。

`Mission` は `dataclass` 形式（`kind: str, score: float, target_id: int, turns: int, options: list[ShotOption]`）とする。

### `evaluation.selfplay.app` (typer)
```python
@app.command()
def run(
    episodes: int = 20,
    mode: Literal["1v1", "ffa4"] = "1v1",
    seed: int = 0,
    output_dir: Path | None = None,
    save_replay: bool = True,
) -> None: ...
```

実行例:
```bash
uv run python -m pipeline.case1.evaluation.selfplay run --episodes 100 --mode 1v1 --seed 0
uv run python -m pipeline.case1.evaluation.selfplay run --episodes 50 --mode ffa4
```

## データフロー

```
Kaggle env.step(observation) ─► agent(obs)
                                   │
  1. build_world(obs) → WorldModel
     (arrival ledger, doomed, threatened, base_need_cache を構築)
                                   │
  2. strategy.plan_moves(world)
     │
     ├─ phase 判定 (early / opening / normal / late / very_late)
     ├─ expansion / attack / swarm (2+3 source) — インライン
     ├─ build_snipe_mission(world, …)
     ├─ build_reinforcement_missions(world, …)
     ├─ if 4P: build_crash_exploit_missions(world, …)
     ├─ followup / doomed 撤退 / rear expansion — インライン
     └─ score ソート → リソース割当 → moves 生成
                                   │
  return moves ─────────────────► Kaggle env
```

## インフラ変更

### `.gitignore` 追加行
```
# replays (large JSON)
data/replays/

# kaggle kernel metadata secrets
.kaggle/
kaggle.json
```

### `pyproject.toml` 変更
1. `[dependency-groups.dev]` に `"kaggle>=1.7.4"` を追加。
2. `[tool.ruff.lint.per-file-ignores]` に以下を追加:
   ```toml
   [tool.ruff.lint.per-file-ignores]
   "pipeline/case1/baseline/**/*.py" = ["C901", "E501", "PLR0912", "PLR0913", "PLR0915"]
   "pipeline/case1/notebook/**" = ["ALL"]
   "tests/pipeline/case1/snapshots/**" = ["ALL"]
   ```
3. `[tool.pytest.ini_options]` は現状維持（`--cov=src` のため、pipeline/case1 はカバレッジ計算に含めない）。

### 外部統合
- **Kaggle API**: `kaggle kernels pull` でノートブック取得、必要に応じて `kaggle competitions submit` で提出。認証は `~/.kaggle/kaggle.json` または環境変数 `KAGGLE_USERNAME` / `KAGGLE_KEY`。
- **kaggle-environments**: `env = make("orbit_wars", configuration={"episodeSteps": 500})`, `env.run([agent, agent])`。

## 設計上の判断とトレードオフ

| 判断 | 理由 | 代替案を却下した理由 |
|------|------|----------------------|
| core / missions / strategy / agent の 4層構造 | import グラフが一方向、各層の責務が明確 | 完全フラット化はファイル数が増え、検索性が下がる |
| `WorldModel` を単一クラスに集約 | ノートブックと同構造 (クラス名も揃える)、キャッシュ管理が容易 | モジュール関数群に分解すると共有状態引き回しが煩雑化 |
| `missions/` はノートブックで関数として切り出されている 3 種のみ分離 (snipe/reinforcement/crash_exploit) | ノートブック準拠で diff を最小化 | expansion/attack/swarm も無理に切り出すと `plan_moves` のインライン制御フロー (commits, score modifiers) が壊れる |
| expansion/attack/swarm は `strategy.plan_moves` にインライン保持 | ノートブックの挙動を 1:1 再現、snapshot 一致性を担保 | 切り出すと関数間で world の mutable state を引き回す必要があり複雑化 |
| Snapshot は action 列の全一致 | ノートブック挙動の deviation を即検知 | 10 ターンのみだと後半バグを見逃す |
| `data/replays/` を gitignore | 100 エピソードで 100MB+ になる可能性 | LFS 導入は scope 過大 |
| kaggle CLI は dev 依存のみ | 本番 submission には不要 | `main` 依存にすると bundle が重くなる |
| `src/` は touch しない | scope を守る、Step B で段階的に | 同時実行するとノートブック diff 検証が曖昧化 |
| per-file-ignores で complexity 緩和 | ノートブック互換性最優先 | 関数分割すると diff が膨大化 |
