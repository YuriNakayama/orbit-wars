# Evaluation System — Implementation Steps

実装方針:
- **1 ステップ = 1 モジュール（実装 + unit テスト）**。
- **実装順**: types → agents → executor → recorder → runner → cli → analyze → replay → viewer → cleanup。
- 各ステップで unit テスト必須。最終ステップで横断 integration を集約。
- **並列可能**: Step 8 (analyze) と Step 9 (replay) は互いに独立。Step 10 (viewer) と Step 11 (tests 仕上げ) も独立。

## Step 0: プロジェクト足場準備

**Target**: cross-cutting  
**Dependencies**: None

### Overview
`src/env/` を作成するための `pyproject.toml` / gitignore / ruff 設定を先に整える。

### Work Items
- [ ] `pyproject.toml` の `[tool.hatch.build.targets.wheel]` の `packages` に `"src/env"` を追加
- [ ] `[tool.ruff.lint.per-file-ignores]` に `"src/env/cli.py" = ["B008"]` を追記
- [ ] `.gitignore` に以下を追記:
  ```
  data/matches/**
  data/replays/**
  data/lake/**
  !data/**/.gitkeep
  ```
- [ ] `data/matches/.gitkeep`, `data/replays/.gitkeep` を作成
- [ ] `src/env/__init__.py` を空で作成（次ステップ以降で公開 API を書く）
- [ ] `dev/test-backend` の `cd backend` を削除し、リポジトリルートから `uv run pytest tests "$@"` が走るように修正

### Target Files
- `pyproject.toml`
- `.gitignore`
- `data/matches/.gitkeep`, `data/replays/.gitkeep`
- `src/env/__init__.py`
- `dev/test-backend`

### Acceptance Criteria
- `uv run python -c "import env"` が成功する
- `dev/test-backend` が既存テストを全てパスする（`tests/` がルートから読める）
- `git status` で `data/matches/**` が untracked にならない

---

## Step 1: types — ドメインモデル定義

**Target**: backend  
**Dependencies**: Step 0

### Overview
`AgentSpec`, `MatchSpec`, `AgentTiming`, `MatchRecord` を frozen dataclass で定義。Parquet への直列化関数を併設。

### Work Items
- [ ] `src/env/types.py` に 4 つの dataclass を実装（全て `frozen=True`）
- [ ] `MatchRecord.to_row() -> dict` — Parquet 行用の flat dict へ変換。4人固定の列幅、不在 player は `""` / `0`
- [ ] `MatchRecord.from_row(row: dict) -> MatchRecord` — 逆変換（Phase2 用だが unit テスト対象）
- [ ] 型ヒントは `tuple` / `str | None` ではなく `str` + `""` を使い、Parquet 互換を保つ
- [ ] `tests/env/test_types.py` で dataclass の frozen 性・to_row/from_row ラウンドトリップを検証

### Target Files
- `src/env/types.py`
- `tests/env/__init__.py`
- `tests/env/test_types.py`

### Acceptance Criteria
- `ruff check` / `mypy --strict` パス
- `MatchRecord.to_row()` と `from_row()` が同一インスタンスを返す（ラウンドトリップ）
- テストカバレッジ 95%+

---

## Step 2: agents — レジストリと version 解決

**Target**: backend  
**Dependencies**: Step 1

### Overview
`AGENT_REGISTRY` と `resolve(name)`、`agent_version()` を実装。

### Work Items
- [ ] `src/env/agents.py` に `AGENT_REGISTRY: dict[str, str]` を定義（`baseline_v1`, `case0`, `random` の 3 つを初期登録）
- [ ] `resolve(name: str) -> Callable | str` — `"module:attr"` 形式は importlib で解決、`"random"` 等の文字列はそのまま
- [ ] `agent_version(name: str) -> str` — `subprocess.run(["git", "rev-parse", "--short", "HEAD"])` を 1 回呼んでキャッシュ
- [ ] `list_agents() -> list[str]`
- [ ] 未登録名に対して `KeyError` を投げる
- [ ] `tests/env/test_agents.py`:
  - `resolve("case0")` が callable を返す
  - `resolve("random")` が文字列 `"random"` を返す
  - `resolve("unknown")` が KeyError
  - `agent_version()` が 7 文字前後のヘックスを返す（もしくは `""` を許容、git が無い環境対策）

### Target Files
- `src/env/agents.py`
- `tests/env/test_agents.py`

### Acceptance Criteria
- 3 種類 (`baseline_v1`, `case0`, `random`) の resolve が通る
- `mypy --strict` パス

---

## Step 3: executor — ワーカー関数（対戦1回）

**Target**: backend  
**Dependencies**: Step 2

### Overview
`run_one_match(spec: MatchSpec) -> tuple[MatchRecord, bytes | None]` を実装。`kaggle_environments` を lazy import し、timing 計測を含む。戻り値の 2 要素目は gzip リプレイの bytes（`save_replay=False` なら None）。

### Work Items
- [ ] `src/env/executor.py` に top-level 関数 `run_one_match(spec: MatchSpec) -> tuple[dict, bytes | None]` を実装（メインに返すため dict で返す）
- [ ] タイミングラッパー: `time.perf_counter()` で `agent(obs)` を計測、ターン毎の経過を記録
- [ ] 勝者判定: `_episode_winner(final_state)`（既存 selfplay.py のロジック踏襲）
- [ ] スコア計算: `_player_scores(env)`（既存踏襲）
- [ ] `env.toJSON()` を `json.dumps(..., default=str)` → `gzip.compress(...)` で bytes 化
- [ ] draw 判定: 勝者 -1 または reward 同点で `draw=True`
- [ ] `tests/env/test_executor.py`:
  - `@pytest.mark.integration` で `kaggle_environments` 実環境を使い、1v1 1エピソードを実行
  - `MatchRecord` 風 dict が返ること、`turns > 0`, `winner in {-1, 0, 1}`
  - `save_replay=False` で bytes が None になること
  - `save_replay=True` で bytes の gzip を decompress すると valid JSON であること
- [ ] unit: `_episode_winner([...])`, `_player_scores(mock_env)` を mock で検証

### Target Files
- `src/env/executor.py`
- `tests/env/test_executor.py`

### Acceptance Criteria
- integration テストが `kaggle_environments` 環境で通る
- unit テストは `pytest.importorskip` 不要（モックのみ）で通る
- 1v1 エピソードの `run_one_match` 実行時間が 5 秒以内（開発機目安）

---

## Step 4: recorder — Parquet / リプレイ書き込み

**Target**: backend  
**Dependencies**: Step 3

### Overview
`MatchRecord` リストを Parquet に hive append、リプレイ bytes を `.json.gz` として書き出す。

### Work Items
- [ ] `src/env/recorder.py` に `write_run(records: list[MatchRecord], replay_bytes: dict[str, bytes], data_root: Path) -> None`
- [ ] Parquet 書き込み: 各モードごとに `pl.DataFrame([r.to_row() for r in records])` を作り、`data_root/matches/index.parquet/mode={mode}/run_{run_id}.parquet` に書く
- [ ] 既存ファイル検出時の衝突回避（同 run_id は末尾 `_1`, `_2` でサフィックス）
- [ ] リプレイ: `data_root/matches/replays/{match_id}.json.gz` に bytes を書く
- [ ] 親ディレクトリは `mkdir(parents=True, exist_ok=True)`
- [ ] `tests/env/test_recorder.py`:
  - tmp_path に 2 records × 2 mode を書く → hive ディレクトリが正しく作られる
  - 同一 run_id の 2 回目書き込みで suffix が付く
  - リプレイ bytes が round-trip する（gzip decompress → json.loads が dict）
- [ ] unit のみ（`kaggle_environments` 不要）

### Target Files
- `src/env/recorder.py`
- `tests/env/test_recorder.py`

### Acceptance Criteria
- Polars `scan_parquet(..., hive_partitioning=True)` で書いたデータが mode 列付きで読める
- リプレイファイルが gzip として valid

---

## Step 5: runner — dispatch と progress bar

**Target**: backend  
**Dependencies**: Step 4

### Overview
`run_episodes(spec: RunSpec)` を実装。`multiprocessing.Pool` または逐次実行、`rich.Progress` で進捗、結果を recorder に渡す。

### Work Items
- [ ] `src/env/runner.py` に `RunSpec` dataclass（`agents: tuple[str, ...]`, `mode: str`, `episodes: int`, `seed: int`, `parallel: int`, `save_replay: bool`, `data_root: Path`）を定義
- [ ] `run_episodes(run_spec: RunSpec) -> list[MatchRecord]` 関数:
  - mode 妥当性チェック（1v1=2 agents, ffa4=4 agents）
  - `match_specs` を生成（`seed=base+i`, `match_id` 生成）
  - `parallel == 1`: 直列ループ
  - `parallel > 1`: `multiprocessing.get_context("spawn").Pool(parallel)` + `imap_unordered(run_one_match, specs)`
  - `rich.Progress` で進捗バー（完了数）
  - 結果を `recorder.write_run(...)` に渡す
- [ ] runner からは **CLI 表示用のデータ**を返すだけ、print は禁止
- [ ] `tests/env/test_runner.py`:
  - `@pytest.mark.integration` で `parallel=1` と `parallel=2` それぞれ 2 エピソード実行
  - 両方同じ件数のレコードを返す
  - unit: `RunSpec` のバリデーション、mode 不一致で `ValueError`

### Target Files
- `src/env/runner.py`
- `tests/env/test_runner.py`

### Acceptance Criteria
- `parallel=2, episodes=4` のテストが 20 秒以内に完了
- progress bar は `verbose=False` でも最低限（1 行更新）表示
- strict mypy パス

---

## Step 6: cli — typer エントリポイント

**Target**: backend  
**Dependencies**: Step 5

### Overview
`python -m env.cli run ...` を実装。

### Work Items
- [ ] `src/env/cli.py` に `app = typer.Typer(...)`
- [ ] `run` サブコマンド:
  ```
  env run --agents NAME[,NAME...] --mode {1v1,ffa4} -n N --seed S --parallel P
          [--save-replay/--no-save-replay] [--data-root PATH] [-v]
  ```
  - `--agents` はカンマ区切りで 2 or 4 個
  - runner.run_episodes → report.summarize で rich.Table 出力
- [ ] `list` サブコマンド（最小）: Parquet を scan して最新 N 件を stdout に出す
- [ ] `replay-inspect` サブコマンド: match_id を受け取り、最終 obs・勝者・turns を出力
- [ ] `src/env/__main__.py` を追加（`if __name__ == "__main__": cli.app()`）
- [ ] `tests/env/test_cli.py`:
  - `typer.testing.CliRunner` で `run --agents case0,case0 --mode 1v1 -n 1 --parallel 1 --no-save-replay` を実行
  - exit_code == 0
  - stdout に "Summary" を含む
  - integration マーカー

### Target Files
- `src/env/cli.py`
- `src/env/__main__.py`
- `src/env/report.py` 雛形
- `tests/env/test_cli.py`

### Acceptance Criteria
- `uv run python -m env run --agents case0,case0 --mode 1v1 -n 2 --parallel 1 --no-save-replay` が成功
- `data/matches/index.parquet/mode=1v1/...` が生成される

---

## Step 7: report — rich.Table サマリ

**Target**: backend  
**Dependencies**: Step 6（cli が呼ぶ）

### Overview
`summarize(records: list[MatchRecord]) -> str / rich.Table` を実装。agent 別勝率、平均ターン、タイムアウト率、1ターン時間分位を表示。

### Work Items
- [ ] `src/env/report.py` に `summarize(records, console=None) -> None`（print するのではなく `console.print(table)`）
- [ ] 列: `agent`, `wins`, `games`, `win_rate`, `avg_turns`, `timeout_rate`, `turn_p95`
- [ ] 各 agent_name 単位で集計（同名 agent が複数 player にいる場合は合算）
- [ ] `tests/env/test_report.py`: fixture の `MatchRecord` リストから `summarize` が期待通りの Table を返すこと（Table のセル比較）

### Target Files
- `src/env/report.py`（Step 6 の雛形を本実装に）
- `tests/env/test_report.py`

### Acceptance Criteria
- モック records で集計が数学的に正しい
- `rich.Table` の列幅やフォーマットが崩れない（smoke test）

---

## Step 8: analyze — Polars 集計ユーティリティ（並列可: Step 9 と独立）

**Target**: backend  
**Dependencies**: Step 4（recorder） — Step 6 と独立

### Overview
`data/matches/index.parquet` を読み、agent × 相手 × モード別の集計を返す。

### Work Items
- [ ] `src/env/analyze.py`:
  - `scan_index(data_root, filters=None) -> pl.LazyFrame`
  - `agent_winrate(lf, by=["agent_name", "mode"]) -> pl.DataFrame` — agent×mode×opponent 勝率
  - `timing_distribution(lf) -> pl.DataFrame` — agent 別 p50/p95/max
  - `mode_summary(lf) -> pl.DataFrame` — mode 別平均ターン・引き分け率
- [ ] CLI 側 `analyze` サブコマンド追加（Step 6 に後付け）
- [ ] `tests/env/test_analyze.py`:
  - fixture で 10 records の parquet を書き、agent_winrate が期待値を返す

### Target Files
- `src/env/analyze.py`
- `src/env/cli.py`（analyze サブコマンド追加）
- `tests/env/test_analyze.py`

### Acceptance Criteria
- 合成データで agent_winrate が正しい
- scan_parquet が hive partition を利用（filter push-down を確認）

---

## Step 9: replay/loader — リプレイ復元（並列可: Step 8 と独立）

**Target**: backend  
**Dependencies**: Step 4

### Overview
`load_replay(match_id)`, `list_matches(filters)` を実装。

### Work Items
- [ ] `src/env/loader.py`:
  - `list_matches(filters=None, data_root=DEFAULT) -> pl.DataFrame` — Parquet を scan して filter 適用
  - `load_replay(match_id: str, data_root=DEFAULT) -> Environment` — gzip JSON を読み、`kaggle_environments.make(name=loaded["name"], configuration=loaded["configuration"], steps=loaded["steps"])` で再構成
- [ ] `src/env/__init__.py` で `run_match, load_replay, list_matches, AGENT_REGISTRY` を export
- [ ] `tests/env/test_loader.py`:
  - Step 3 で生成した実リプレイ（integration fixture）を load_replay で読めること
  - `env.render(mode="json")` が dict を返すこと（ipython は pytest でスキップ）

### Target Files
- `src/env/loader.py`
- `src/env/__init__.py`（export 追加）
- `tests/env/test_loader.py`

### Acceptance Criteria
- `env.run` → `toJSON` → `save` → `load_replay` → `toJSON` が元と同等（steps 数が一致）
- integration テストでラウンドトリップが成功

---

## Step 10: viewer notebook（並列可: Step 11 と独立）

**Target**: pipeline / docs  
**Dependencies**: Step 9

### Overview
`pipeline/case1/eda/replay_viewer.py` を percent format で作成し、Jupyter から `env.render("ipython")` が動くことを確認。

### Work Items
- [ ] `pipeline/case1/eda/replay_viewer.py` を作成:
  - `# %%` セル 3 つ: import → list_matches → load_replay + render
  - docstring で「VS Code / Jupyter で Run Cell ボタンから実行」と明記
- [ ] `pipeline/case1/eda/README.md`（任意）— 使い方を記述
- [ ] 手動検証: ノートブック上でリプレイが表示されることを確認（CI は対象外）

### Target Files
- `pipeline/case1/eda/replay_viewer.py`
- `pipeline/case1/eda/README.md`（optional）

### Acceptance Criteria
- `.py` ファイルが ruff 合格
- 手動検証で Jupyter 上で惑星が描画される

---

## Step 11: 横断 integration テストと最終クリーンアップ

**Target**: cross-cutting  
**Dependencies**: Step 1-9 全部

### Overview
複数モジュールを通しで動かす integration テストを追加し、`pipeline/case1/evaluation/selfplay.py` を削除（または削除に代わる薄いラッパー化）。

### Work Items
- [ ] `tests/env/test_end_to_end.py`:
  - `@pytest.mark.integration @pytest.mark.slow`
  - CliRunner で `run --agents case0,case0 -n 2 --parallel 2 --save-replay` 実行
  - 生成された `index.parquet` を load_matches で読み、2 件あること
  - replay_path の json.gz が読め、`load_replay` で env が再構成できる
  - `render("json")` の dict が `steps` キーを持つ
- [ ] `pipeline/case1/evaluation/selfplay.py` を削除
- [ ] `pipeline/case1/evaluation/__init__.py` から selfplay エクスポートを削除
- [ ] `pipeline/case1/evaluation/snapshot_update.py` はそのまま残す
- [ ] README.md に `env` セクションを追加（`python -m env run ...` の使い方を 10 行程度）

### Target Files
- `tests/env/test_end_to_end.py`
- `pipeline/case1/evaluation/selfplay.py`（削除）
- `pipeline/case1/evaluation/__init__.py`
- `README.md`

### Acceptance Criteria
- `dev/test-backend` が全テストパス
- `dev/lint` / ruff / mypy が全てパス
- end-to-end テストが 60 秒以内に完了（2 エピソード × 2 並列）
- `pipeline/case1/evaluation/selfplay.py` が存在しない
- README に新 CLI の使い方が記載

---

## 依存関係まとめ

```
Step 0 ─▶ Step 1 ─▶ Step 2 ─▶ Step 3 ─▶ Step 4 ─▶ Step 5 ─▶ Step 6 ─▶ Step 7 ─┐
                                                                     │         ├─▶ Step 11
                                                           ┌─────────┴─────┐   │
                                                           ▼               ▼   │
                                                        Step 8          Step 9 ┤
                                                                           │   │
                                                                           ▼   │
                                                                        Step 10┘
```

- Step 8 / 9 は Step 4 以降に並列可。
- Step 10 / 11 の一部は Step 9 完了後に並列可。
- 全体で 12 ステップ、1人実装で 2-4 日の規模感。
