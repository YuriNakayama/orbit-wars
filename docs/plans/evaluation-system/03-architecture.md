# Evaluation System — Architecture Design

プロジェクト既存アーキテクチャ（CLAUDE.md 記載のパイプライン図、`src/` と `pipeline/` の責務分担）を踏まえて、評価システムを `src/env/` 配下の汎用ライブラリとして構築する。`pipeline/case1/evaluation/` は新モジュールを薄く呼ぶラッパーに置換（最終的には削除し、`src/env/` に完全移行）。

## 1. 全体構成

```
                    ┌──────────────────────────────────────────┐
                    │              CLI (src/env/cli.py)         │
                    │  python -m env.cli run|analyze|replay ... │
                    └───────────┬──────────────────────────────┘
                                │
             ┌──────────────────┼──────────────────┐
             ▼                  ▼                  ▼
      ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
      │   runner    │    │   analyze   │    │   replay    │
      │  (dispatch) │    │ (reporting) │    │  (loader)   │
      └──────┬──────┘    └──────┬──────┘    └──────┬──────┘
             │                  │                  │
             ▼                  ▼                  ▼
      ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
      │  executor   │    │   loader    │    │   recorder  │
      │ (worker fn) │◀───│ (parquet IO)│    │ (disk save) │
      └──────┬──────┘    └─────────────┘    └──────▲──────┘
             │                                     │
             ▼                                     │
      ┌─────────────┐                       ┌─────────────┐
      │   agents    │                       │   agents    │
      │ (registry)  │                       │  (registry) │
      └──────┬──────┘                       └─────────────┘
             ▼
      kaggle_environments.make("orbit_wars", ...).run([...])
             │
             ▼
      env.toJSON() ──gzip──▶ data/matches/replays/{match_id}.json.gz
             │
             └─ summarize ──▶ data/matches/index.parquet (hive: mode=)
```

### 責務分割

| モジュール | 責務 | 想定行数 |
|---|---|---|
| `src/env/__init__.py` | 公開 API（`run_match`, `load_match`, `AGENT_REGISTRY`） | 〜30 |
| `src/env/agents.py` | エージェントレジストリ、version 解決 | 〜80 |
| `src/env/runner.py` | 対戦ディスパッチ。multiprocessing.Pool 管理、progress bar | 〜150 |
| `src/env/executor.py` | ワーカー関数。`env.make + env.run + timing 計測 + match dict 生成`。pickle 可 | 〜200 |
| `src/env/recorder.py` | Parquet への append、gzip JSON 書き出し | 〜150 |
| `src/env/loader.py` | Parquet scan、リプレイ読み込み（`load_replay(match_id)`） | 〜120 |
| `src/env/analyze.py` | 集計ユーティリティ（winrate, timing, mission） | 〜200 |
| `src/env/report.py` | rich.Table フォーマットでの CLI 出力 | 〜100 |
| `src/env/cli.py` | typer CLI。`run` / `analyze` / `replay-inspect` サブコマンド | 〜200 |
| `src/env/types.py` | `MatchRecord` pydantic、`AgentSpec` dataclass 等 | 〜80 |

合計 〜1100行。backend.md の「200-400行/file、800行max」方針に合致。

## 2. データフロー

### 2.1 実行時フロー
1. `cli.py::run` が引数を typer でパース。
2. `runner.py::run_episodes(spec: RunSpec)` が `multiprocessing.Pool(P)` を起動。
3. Pool に `executor.run_one_match(match_spec: MatchSpec) -> MatchRecord` を map。`match_spec` は pickle 可な dict/dataclass。
4. 各ワーカーで:
   - `kaggle_environments.make("orbit_wars", configuration={agents, seed})`
   - agent 関数は **ワーカー内で lazy import**（pickle 困難なクロージャを避ける）。
   - `agents = [timing_wrapper(resolve(name)) for name in spec.agent_names]`
   - `env.run(agents)` 実行。経過時間・タイムアウト数を計測。
   - `env.toJSON()` を recorder に渡す。
5. recorder は:
   - リプレイ: `replays/{match_id}.json.gz` に `gzip.compress(json.dumps(toJSON).encode())` で保存。
   - 集計: `MatchRecord` を dict 化し、**run 終了時にまとめて** `pl.DataFrame(records).write_parquet(index_path, partition_by=["mode"])` する。エピソード都度 append は polars の制約上非推奨のため、run 単位 append とする（hive 側に既存 parquet が居る場合は新パーティションファイルが追加される）。
6. runner は結果を集めて `report.py` に渡し、rich.Table で stdout 表示。

### 2.2 再生フロー
1. `loader.load_replay(match_id)` → `replays/{match_id}.json.gz` を読み、gzip.decompress + json.loads。
2. `make("orbit_wars", configuration=loaded["configuration"], steps=loaded["steps"])` で env を再構成。
3. ノートブックから `env.render(mode="ipython")` で描画。
4. サンプル: `pipeline/case1/eda/replay_viewer.py` (percent-format、jupytext の .py 表現)。

### 2.3 分析フロー
1. `loader.scan_index(filters=None)` → `pl.scan_parquet(DATA_ROOT / "matches" / "index.parquet", hive_partitioning=True)`。
2. `analyze.agent_winrate(lf)` / `analyze.timing_distribution(lf)` / `analyze.mission_distribution(lf)` などで Polars DataFrame を返す。
3. `cli.py analyze --since ... --agents ...` で CLI からも呼べる。

## 3. Frontend 設計

本件に frontend UI は存在しない。**「可視化 = ノートブック上で env.render("ipython")」** が最終形。

- `pipeline/case1/eda/replay_viewer.py` — percent format の Python ファイル。VS Code / Jupyter で「セルとして実行」できる。内容:
  ```python
  # %%
  from env import load_replay, list_matches
  # %%
  df = list_matches(filters={"mode": "1v1"})
  df.head(10)
  # %%
  match_id = df["match_id"][0]
  env = load_replay(match_id)
  env.render(mode="ipython", width=800, height=600)
  ```

## 4. Backend 設計

### 4.1 エンドポイント（CLI コマンド）

| コマンド | 役割 | 例 |
|---|---|---|
| `env run` | 対戦実行 | `python -m env.cli run --agents baseline_v1,case0 --mode 1v1 -n 100 --parallel 8` |
| `env analyze` | index 集計レポート | `python -m env.cli analyze --mode 1v1 --since 2026-04-01` |
| `env list` | マッチ一覧 | `python -m env.cli list --mode 1v1 --limit 20` |
| `env replay-inspect` | リプレイ検査 | `python -m env.cli replay-inspect {match_id}` — 最終 obs と勝者を表示 |

### 4.2 ドメインモデル

```python
# src/env/types.py
from dataclasses import dataclass
from typing import Callable

@dataclass(frozen=True)
class AgentSpec:
    name: str
    version: str          # git short sha
    callable_path: str    # "pipeline.case1.baseline.agent:agent"

@dataclass(frozen=True)
class MatchSpec:
    match_id: str
    run_id: str
    mode: str             # "1v1" | "ffa4"
    seed: int
    agents: tuple[AgentSpec, ...]
    save_replay: bool
    data_root: str        # workers can't inherit Path easily

@dataclass(frozen=True)
class AgentTiming:
    timeouts: int
    p50: float
    p95: float
    max: float

@dataclass(frozen=True)
class MatchRecord:
    match_id: str
    run_id: str
    mode: str
    seed: int
    started_at: str       # ISO8601
    elapsed_sec: float
    turns: int
    winner: int           # -1 for draw
    draw: bool
    agent_names: tuple[str, ...]
    agent_versions: tuple[str, ...]
    agent_scores: tuple[int, ...]
    agent_timings: tuple[AgentTiming, ...]
    replay_path: str      # "" if not saved
    git_sha: str
```

### 4.3 ユースケース

- **UC1: 新 baseline vs 旧 baseline** — `env run --agents baseline_v1,baseline_v0 --mode 1v1 -n 200 --parallel 8`
- **UC2: 自己対戦（同一 agent × 2）** — `env run --agents baseline_v1,baseline_v1 --mode 1v1 -n 100`（agent_name は同じでも record に 2 エントリ）
- **UC3: FFA4 混成** — `env run --agents baseline_v1,case0,random,baseline_v0 --mode ffa4 -n 50`
- **UC4: 敗戦リプレイ検査** — `env list --mode 1v1 --agents baseline_v1 --losses` → match_id 取得 → notebook で `load_replay(mid).render("ipython")`
- **UC5: タイムアウトリスク調査** — `env analyze --metric timing --agents baseline_v1` → p95 が 800ms 超えたら警告

### 4.4 エージェント解決ロジック

```python
# src/env/agents.py
AGENT_REGISTRY: dict[str, str] = {
    "baseline_v1": "pipeline.case1.baseline.agent:agent",
    "case0": "pipeline.case0.main:agent",
    "random": "random",   # kaggle_environments 標準
}

def resolve(name: str) -> Callable:
    path = AGENT_REGISTRY[name]
    if ":" not in path:       # kaggle 標準エージェント
        return path            # str のまま env.run に渡す
    module_name, attr = path.split(":")
    return getattr(importlib.import_module(module_name), attr)

def agent_version() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
```

### 4.5 multiprocessing の具体設計

- **起動方式**: `ctx = multiprocessing.get_context("spawn")` で forkedless → macOS/Linux 両対応、pickle 明示。
- **ワーカー関数**: `executor.run_one_match(match_spec: MatchSpec) -> dict` — top-level関数。agent は spec から lazy resolve。kaggle_environments もワーカー内で import。
- **Pool 引数**: `Pool(processes=P)` + `imap_unordered(run_one_match, specs)` で 進捗バー。
- **pickle 回避**: `AgentSpec.callable_path` は文字列。関数オブジェクト自体は渡さない。
- **並列書き込み回避**: recorder はメインプロセスのみ。ワーカーは `MatchRecord` を返すだけ、リプレイの gzip JSON も「バイト列を返してメインで書く」方式（writeは並列化の恩恵小）。

## 5. Data Model

### 5.1 ディレクトリ構造

```
data/
  matches/
    index.parquet/            # hive partitioning
      mode=1v1/
        run_20260418_101500_ab12cd3.parquet
        run_20260418_113000_ab12cd3.parquet
      mode=ffa4/
        run_20260418_140000_ab12cd3.parquet
    replays/
      20260418T101500Z_ab12cd3_0.json.gz
      20260418T101500Z_ab12cd3_1.json.gz
      ...
```

- `index.parquet` は hive ディレクトリ方式（Polars `write_parquet(..., partition_by=["mode"])`）。
- 1 run = 1 parquet ファイル / パーティション。同一 run は 1 ファイル append（実際は run 終了時に 1 回書くだけ）。
- replay のファイル名: `{match_id}.json.gz`。`match_id = "{ISO_utc_compact}_{git_sha7}_{seed}"`。

### 5.2 Parquet スキーマ

```
match_id              : Utf8
run_id                : Utf8
mode                  : Utf8   ← partition key
seed                  : Int64
started_at            : Datetime(time_unit=us, time_zone=UTC)
elapsed_sec           : Float64
turns                 : Int32
winner                : Int8  (-1 for draw)
draw                  : Boolean
agent_0_name, agent_1_name, agent_2_name, agent_3_name : Utf8 (空文字で不在)
agent_0_version, ..., agent_3_version                  : Utf8
agent_0_score, ..., agent_3_score                      : Int32
agent_0_timeouts, ..., agent_3_timeouts                : Int32
agent_0_turn_p50, ..., agent_3_turn_p50                : Float64
agent_0_turn_p95, ..., agent_3_turn_p95                : Float64
agent_0_turn_max, ..., agent_3_turn_max                : Float64
replay_path           : Utf8
git_sha               : Utf8
```

- 4人固定列にする理由: 可変長 list より Polars 操作が単純・type-stable。
- hive partition は当面 `mode` のみ。日付パーティションは Phase2（データ量が増えたら）。

### 5.3 .gitignore 追記

```
# evaluation-system
data/matches/**
data/replays/**
data/lake/**
!data/**/.gitkeep
```

## 6. Infrastructure 変更

- AWS 等のクラウド不使用。
- pyproject.toml の `[tool.hatch.build.targets.wheel]` に `src/env` を追加（`packages = ["src/submit", "src/env", "pipeline"]`）。
- `dev/test-backend` の `cd backend` バグは本件スコープ外だが、`env` モジュールが import できないと CI が壊れるので、**スクリプトの `cd` を除去してリポジトリルートで `uv run pytest tests` を走らせる修正を FR の一部に含める**。
- `ruff.per-file-ignores` に `"src/env/cli.py" = ["B008"]`（typer 慣習）を追加。

## 7. External Integrations

- **kaggle_environments** — 既存依存。`make("orbit_wars", ...)` と `env.toJSON()` に全面依存。バージョン制約は `kaggle-environments>=1.17.0`（pyproject 既存）。`orbit_wars` 未含有のバージョンがあれば 1.28.0 以降にバンプ必要（docs/competition/20260418_evaluation.md 参照）。
- **polars / pyarrow** — 既存依存。hive partitioning は Polars 1.39 で安定。
- **rich / typer** — 既存依存。
- **外部ネットワーク通信なし、シークレット不要。**

## 8. 既存コードへの変更

| ファイル | 変更内容 |
|---|---|
| `pipeline/case1/evaluation/selfplay.py` | **削除**（src/env/ に移行） |
| `pipeline/case1/evaluation/snapshot_update.py` | そのまま残す（別責務） |
| `pipeline/case1/evaluation/__init__.py` | selfplay 削除に伴い内容更新 |
| `pyproject.toml` | `packages` に `src/env` 追加、ruff ignore 追加 |
| `.gitignore` | `data/matches/**`, `data/replays/**` 追加 |
| `.claude/rules/backend.md` | （オプション）`src/env/` の位置づけを追記 |
| `README.md` | （オプション）`env` セクション追加 |
| `dev/test-backend` | `cd backend` バグ修正（関連スコープ） |
