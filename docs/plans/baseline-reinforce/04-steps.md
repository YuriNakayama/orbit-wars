# baseline-reinforce — 実装ステップ分解

順序: **下から積み上げ（notebook取得 → core → missions → strategy → agent → selfplay → tests）**  
粒度: **1関心事（モジュール1群）= 1ステップ**  
並列化: **core/ 4ファイル同時作成可、missions/ 3ファイル同時作成可、notebook pull × pyproject 更新は並列可能**

---

## Step 1: ノートブック取得とメタデータ整理 ✅（本 feature 実施時点で完了済み）

**対象**: notebook / cross-cutting  
**依存**: なし

### 概要
Kaggle API 経由で原典ノートブックを取得し、`.ipynb` と `.py` を並置する。

### 作業項目
- [x] `~/.kaggle/kaggle.json` を配置し `chmod 600` 設定。
- [x] `uv tool install kaggle` で CLI 導入（`uv sync` は pygame の SDL.h 依存で失敗するため tool install を採用）。
- [x] `kaggle kernels pull sigmaborov/lb-897-orbit-wars-2026-reinforce -p pipeline/case1/notebook/ -m` 実行。
- [x] `lb-897-orbit-wars-2026-reinforce.ipynb` の単一セルを `.py` にエクスポート（1896 行）。

### 対象ファイル
- `pipeline/case1/notebook/lb-897-orbit-wars-2026-reinforce.ipynb`
- `pipeline/case1/notebook/lb-897-orbit-wars-2026-reinforce.py`
- `pipeline/case1/notebook/kernel-metadata.json`

### 完了基準
- [x] `.ipynb` と `.py` と `kernel-metadata.json` が `pipeline/case1/notebook/` に揃っている。
- [x] `.py` の行数がノートブックと一致している（1896行）。

### 備考
- slug は公開時点で `lb-897-orbit-wars-2026-reinforce` に改名されている（元は `orbit-wars-2026-reinforce`）。公開名の変更に追随するため、README には `kaggle kernels list -s "orbit-wars-2026-reinforce"` で最新 slug を確認する手順も併記する。

---

## Step 2: 環境整備 — pyproject.toml / .gitignore 更新

**対象**: cross-cutting  
**依存**: なし（Step 1 と並列実施可）

### 概要
Kaggle CLI を dev 依存に追加し、Ruff の per-file-ignores を設定、`.gitignore` にリプレイ出力と kaggle credential を追加する。

### 作業項目
- [ ] `pyproject.toml` の `[dependency-groups.dev]` に `"kaggle>=1.7.4"` を追加（`uv sync` 実行は pygame ビルド失敗のため別途解決 — 暫定回避策として `uv tool install kaggle` を `pipeline/case1/README.md` に記載）。
- [ ] `pyproject.toml` の `[tool.ruff.lint.per-file-ignores]` に以下を追加:
  - `"pipeline/case1/baseline/**/*.py" = ["C901", "E501", "PLR0912", "PLR0913", "PLR0915"]`
  - `"pipeline/case1/notebook/**" = ["ALL"]`
  - `"tests/pipeline/case1/snapshots/**" = ["ALL"]`
- [ ] プロジェクトルートの `.gitignore` に以下を追記:
  - `data/replays/`
  - `.kaggle/`
  - `kaggle.json`

### 対象ファイル
- `pyproject.toml`
- `.gitignore`

### 完了基準
- [ ] `git status` で対象ファイルのみ変更が検出される。
- [ ] `uv run ruff check pipeline/case1/notebook/` がエラー無しで抜ける（ignores が効いている）。

---

## Step 3: パッケージスケルトン作成

**対象**: cross-cutting  
**依存**: Step 2

### 概要
`pipeline/case1/` の `__init__.py` 群と空のサブディレクトリを一式作成し、import が通る状態を作る。

### 作業項目
- [ ] `pipeline/__init__.py`（既存なければ）、`pipeline/case1/__init__.py`。
- [ ] `pipeline/case1/baseline/__init__.py`、`baseline/core/__init__.py`、`baseline/missions/__init__.py`。
- [ ] `pipeline/case1/evaluation/__init__.py`、`pipeline/case1/configs/` を作成。
- [ ] `pipeline/case1/baseline/LICENSE` に Apache 2.0 本文を配置。
- [ ] `pipeline/case1/README.md` に戦略概要・実行コマンド・notebook 取得手順を記載。
- [ ] `tests/__init__.py`、`tests/pipeline/__init__.py`、`tests/pipeline/case1/__init__.py`。

### 対象ファイル
- 上記 `__init__.py`, `LICENSE`, `README.md`

### 完了基準
- [ ] `uv run python -c "import pipeline.case1"` がエラー無し。
- [ ] README に 3 コマンド（notebook pull / selfplay / test）が揃っている。

---

## Step 4: core/ モジュール群（並列作業可）

**対象**: backend  
**依存**: Step 3

### 概要
ノートブックの CONFIG / TYPES / GEOMETRY / PHYSICS セクションを 4 ファイルに分割して移植する。相互依存は types → geometry, physics（minor）のみ。

### 作業項目（並列実施可能なサブタスク）
- [ ] **4a. `baseline/core/config.py`**: CONFIG セクションの全定数を定数として書き写す。値はノートブック v2 と完全一致。
- [ ] **4b. `baseline/core/types.py`**: `Planet`, `Fleet` namedtuple 定義。`typing.NamedTuple` にして型付けを厳格化。
- [ ] **4c. `baseline/core/geometry.py`**: `dist`, `segment_hits_sun`, `point_to_segment_distance` を移植。関数シグネチャにフル型ヒントを追加。
- [ ] **4d. `baseline/core/physics.py`**: `fleet_speed`, `orbital_radius`, `is_static_planet`, `travel_time`, `predict_planet_position`, `predict_comet_position`, `predict_target_position` を移植。

### 対象ファイル
- `pipeline/case1/baseline/core/config.py`
- `pipeline/case1/baseline/core/types.py`
- `pipeline/case1/baseline/core/geometry.py`
- `pipeline/case1/baseline/core/physics.py`

### 完了基準
- [ ] 各ファイル先頭に Apache 2.0 ヘッダー。
- [ ] `uv run mypy pipeline/case1/baseline/core/` が strict でパス（`Any`, `type: ignore` 無し）。
- [ ] `uv run ruff check pipeline/case1/baseline/core/` がパス（E501 は ignores 対象）。

---

## Step 5: world_model 移植

**対象**: backend  
**依存**: Step 4

### 概要
ノートブック中核の `WorldModel` クラスを `baseline/core/world_model.py` に移植。`build_arrival_ledger`, `simulate_planet_timeline`, `state_at_timeline`, `projected_state` (`base_need_cache` 付き), `plan_shot`, `ships_needed_to_capture`, `reinforcement_needed_for`, `reaction_times` を含める。トップレベル関数群（`fleet_target_planet`, `resolve_arrival_event`, `normalize_arrivals`, `count_players`, `nearest_distance_to_set`, `indirect_wealth`, `detect_enemy_crashes`）も同ファイルに配置。

### 作業項目
- [ ] `WorldModel` クラスをノートブックと同フィールド・同名で定義。
- [ ] `build_arrival_ledger`, `resolve_arrival_event`, `normalize_arrivals` を実装。
- [ ] `simulate_planet_timeline`, `state_at_timeline` を実装。
- [ ] `projected_state`（`base_need_cache` 付き）を実装。
- [ ] `plan_shot`, `ships_needed_to_capture`, `reinforcement_needed_for`, `reaction_times` を実装。
- [ ] `_multi_enemy_proactive_keep`, `_compute_defense_buffers`, `is_static`, `comet_life`, `source_inventory_left`, `source_attack_left` を実装。
- [ ] 補助関数 `fleet_target_planet`, `count_players`, `nearest_distance_to_set`, `indirect_wealth`, `detect_enemy_crashes` を実装。

### 対象ファイル
- `pipeline/case1/baseline/core/world_model.py`

### 完了基準
- [ ] ダミー observation で `WorldModel(...)` 構築がエラー無し。
- [ ] `WorldModel.arrivals_by_planet` が期待通り `{planet_id: [(eta, owner, ships)]}` を返す（後続 Step 10 のテストで検証）。
- [ ] Mypy strict パス。

---

## Step 6: missions/ 群（並列作業可、3サブタスク）

**対象**: backend  
**依存**: Step 5

### 概要
ノートブック内で独立関数として切り出されている 3 種のミッション生成関数のみを各ファイルに実装。`Mission` 型は `core/types.py` に配置。expansion / attack / swarm は `plan_moves` インライン実装に留めるため、ここでは触れない。

### 作業項目（並列実施可）
- [ ] **6a. `missions/snipe.py`**: `build_snipe_mission` (単数、ノートブックの関数名と揃える)。
- [ ] **6b. `missions/reinforcement.py`**: `build_reinforcement_missions`。
- [ ] **6c. `missions/crash_exploit.py`**: `build_crash_exploit_missions` (4P 専用)。

### 対象ファイル
- `pipeline/case1/baseline/missions/__init__.py`
- `pipeline/case1/baseline/missions/snipe.py`
- `pipeline/case1/baseline/missions/reinforcement.py`
- `pipeline/case1/baseline/missions/crash_exploit.py`

### 完了基準
- [ ] 各ファイル先頭に Apache 2.0 ヘッダー。
- [ ] 各ミッションが `WorldModel` を受け取り、ノートブックと同じ戻り値型 (`Mission | None` または `list[Mission]`) を返す。
- [ ] Mypy strict, Ruff (per-file-ignores 後) パス。

---

## Step 6.5: strategy.py 移植（plan_moves オーケストレーション）

**対象**: backend  
**依存**: Step 6

### 概要
ノートブック STRATEGY セクション（928〜1849行）を `baseline/strategy.py` に移植する。中核は `plan_moves(world)` でここに expansion / attack / swarm (2+3 source) / followup / doomed / rear のインライン制御フローを全量置く。ノートブック準拠で snapshot 一致性を担保するため、関数内ブロック分割やサブ関数化はしない。

### 作業項目
- [ ] ヘルパ関数群: `planet_distance`, `build_modes`, `is_safe_neutral`, `is_contested_neutral`, `opening_filter`, `target_value`, `preferred_send`, `apply_score_modifiers` を移植。
- [ ] `plan_moves(world) -> list[list[int | float]]` を移植。内部で `build_snipe_mission`, `build_reinforcement_missions`, `build_crash_exploit_missions` を呼び出す。
- [ ] `print(` を全量削除し、`logging.getLogger(__name__).debug(...)` に置換（または単純削除）。

### 対象ファイル
- `pipeline/case1/baseline/strategy.py`

### 完了基準
- [ ] `print` がファイル内に残存していない (`grep` で 0 件)。
- [ ] Mypy strict パス（複雑度は per-file-ignores で緩和済）。
- [ ] Ruff パス。

---

## Step 7: agent.py 実装

**対象**: backend  
**依存**: Step 6

### 概要
ノートブックの `agent()` 関数を `baseline/agent.py` に移植する。`WorldState` 構築、フェーズ判定、全 `build_*_missions` 呼び出し、ソート、リソース制約下での順次実行、doomed evacuation、rear expansion を含む。

### 作業項目
- [ ] `classify_phase(world) -> str` ヘルパー実装。
- [ ] `agent(observation, configuration=None) -> list[list[int | float]]` 実装。
- [ ] Doomed planet 退避処理。
- [ ] Rear expansion 処理。
- [ ] `num_ships` は必ず `int` キャストして返す。
- [ ] `print` は全て削除（`logging.getLogger(__name__)` に置換、デフォルト NullHandler）。

### 対象ファイル
- `pipeline/case1/baseline/agent.py`

### 完了基準
- [ ] ダミー observation で `agent(obs)` が 1 秒以内にアクション列を返す。
- [ ] 戻り値が `list[list[int | float]]` 型で、各要素長が 3。

---

## Step 8: main.py (Kaggle Submission entrypoint)

**対象**: backend  
**依存**: Step 7

### 概要
Kaggle 提出用のエントリポイント。相対 import でエージェントを再エクスポート。

### 作業項目
- [ ] `pipeline/case1/baseline/main.py` に `from .agent import agent` を記述。
- [ ] Apache 2.0 ヘッダー。
- [ ] `__all__ = ["agent"]`。

### 対象ファイル
- `pipeline/case1/baseline/main.py`

### 完了基準
- [ ] `uv run python -c "from pipeline.case1.baseline.main import agent; print(agent)"` が関数を表示。
- [ ] Mypy strict パス。

---

## Step 9: selfplay CLI 実装

**対象**: tooling  
**依存**: Step 8

### 概要
`pipeline/case1/evaluation/selfplay.py` を `typer` CLI として実装。1v1 / 4P FFA、seed 制御、リプレイ JSON 保存、rich テーブルでサマリ出力。

### 作業項目
- [ ] `typer.Typer()` アプリ作成。
- [ ] `run(episodes, mode, seed, output_dir, save_replay)` コマンド実装。
- [ ] `kaggle_environments.make("orbit_wars", ...)` で環境構築、`env.run([agent, agent, ...])` 実行。
- [ ] 各ターンの `time.perf_counter` でエージェント実行時間を計測し、`data/replays/case1/<ts>/episode_<i>_timings.json` に保存。
- [ ] `episode_<i>.json` に `env.toJSON()` または `env.render(mode="json")` のリプレイを保存。
- [ ] `summary.csv` に列: `episode, winner, turns, p0_score, p1_score, p2_score, p3_score, timeouts, elapsed_sec`。
- [ ] 標準出力に `rich.table.Table` で勝率・タイムアウト率・平均ターン。
- [ ] `logging` 経由の出力（`print` 禁止、`rich.console.Console` は可）。

### 対象ファイル
- `pipeline/case1/evaluation/selfplay.py`

### 完了基準
- [ ] `uv run python -m pipeline.case1.evaluation.selfplay run --episodes 2 --mode 1v1 --seed 0` が成功し、`data/replays/case1/<ts>/` が生成される。
- [ ] `--mode ffa4` も動作。
- [ ] Mypy strict / Ruff パス。

---

## Step 10: 単体テスト + スナップショットテスト

**対象**: testing  
**依存**: Step 9

### 概要
`tests/pipeline/case1/` に単体テストとスナップショットテストを配置。snapshot は seed=0 の 1v1 全ターンの action 列を diff。

### 作業項目
- [ ] **10a. `test_world_state.py`**:
  - 小規模 observation fixture を作成し、`WorldState.from_observation` が planet_by_id / arrivals_by_planet / doomed_planets を期待通り計算することを AAA で検証。
- [ ] **10b. `test_baseline_agent.py`**:
  - `kaggle_environments.make("orbit_wars")` を使い、`env.run([agent, agent])` が DONE に到達することを確認。
  - `agent(obs)` 戻り値の型と要素数を assert。
- [ ] **10c. スナップショット**:
  - seed=0 の 1v1 エピソードを回し、各ターンの action 列を `tests/pipeline/case1/snapshots/episode_seed0.json` にコミット。
  - テスト実行時に現行 agent の出力と snapshot を diff し、一致しなければ失敗。
  - スナップショット更新用のヘルパ CLI `uv run python -m pipeline.case1.evaluation.snapshot_update` を提供（scope小・任意）。

### 対象ファイル
- `tests/pipeline/case1/test_world_state.py`
- `tests/pipeline/case1/test_baseline_agent.py`
- `tests/pipeline/case1/snapshots/episode_seed0.json`
- （任意）`pipeline/case1/evaluation/snapshot_update.py`

### 完了基準
- [ ] `uv run pytest tests/pipeline/case1/` が全パス。
- [ ] スナップショットは git コミットされ、リポジトリに固定される。

---

## Step 11: configs/baseline.yaml + load_config helper

**対象**: configuration  
**依存**: Step 4（config.py）

### 概要
CONFIG を YAML 化してチューニング基盤の事前配線を行う。本 feature では Python 定数が優先される。

### 作業項目
- [ ] `pipeline/case1/configs/baseline.yaml` に CONFIG を転記。
- [ ] `pipeline/case1/baseline/core/config.py` に `load_config(path: Path) -> dict[str, Any]` ヘルパを追加（YAML パースのみ、差し替えは scope外）。

### 対象ファイル
- `pipeline/case1/configs/baseline.yaml`
- `pipeline/case1/baseline/core/config.py`（追記）

### 完了基準
- [ ] `yaml.safe_load(open(...))` で読み込める。
- [ ] Python 定数値と YAML 値が一致する（テスト不要、目視で足りる）。

---

## Step 12: README 完成と手動動作確認

**対象**: documentation  
**依存**: Step 10

### 概要
`pipeline/case1/README.md` を完成させ、手動で一連の flow を確認する。

### 作業項目
- [ ] README に以下のセクション:
  - 「Case1 Baseline — Orbit Wars 2026 Reinforce (sigmaborov)」見出し
  - 目的・ライセンス・出典
  - notebook 取得方法（kaggle 認証の両方式を記載）
  - selfplay 実行方法
  - テスト実行方法
  - ディレクトリ構造図
  - 既知の制約（ノートブックとの挙動 diff ポリシー、`uv sync` の pygame ビルドハマりどころ等）
- [ ] 手動確認:
  - `uv run python -m pipeline.case1.evaluation.selfplay run --episodes 5 --mode 1v1` が完走。
  - `uv run pytest tests/pipeline/case1/` が緑。
  - `uv run ruff check pipeline/case1 tests/pipeline/case1` / `uv run mypy pipeline/case1` が緑。

### 対象ファイル
- `pipeline/case1/README.md`

### 完了基準
- [ ] すべての手動確認コマンドがエラー無し。
- [ ] README から本 feature のスコープと操作方法が完結している。

---

## 依存関係まとめ

```
Step 1 (notebook)  ──┐
Step 2 (pyproject) ──┼─► Step 3 (skeleton) ─► Step 4 (core) ─► Step 5 (world_state) ─► Step 6 (missions) ─► Step 7 (agent) ─► Step 8 (main.py) ─► Step 9 (selfplay) ─► Step 10 (tests) ─► Step 12 (README + verify)
                     │
                     └─► Step 11 (YAML)  ← Step 4 の後なら並列可
```

## 並列化可能なワークストリーム

| ストリーム | ステップ | 並列作業 |
|------------|----------|----------|
| A | Step 4a/b/c/d | config / types / geometry / physics を同時作成 |
| B | Step 6a/b/c/d/e/f/g | missions 7 ファイルを同時作成 |
| C | Step 1 + Step 2 | notebook 取得と pyproject 更新は独立 |
| D | Step 11 | Step 4 後ならば並行可能 |

## Cross-cutting: Apache 2.0 ライセンス表示
全 `pipeline/case1/baseline/**/*.py` 先頭に以下のヘッダー:
```python
# Adapted from "Orbit Wars 2026 - Reinforce" by sigmaborov
# https://www.kaggle.com/code/sigmaborov/lb-897-orbit-wars-2026-reinforce
# Licensed under Apache License 2.0
```
