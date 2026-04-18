# baseline-reinforce — コードベース調査

`pipeline/case1/` 配下に Kaggle ノートブック「Orbit Wars 2026 - Reinforce」(sigmaborov, Public Score 928.5) を再現したベースラインを構築するための事前調査結果。

## 深いコードベース分析

### 1. 既存コードベース

- **`src/`**: 現状 **空ディレクトリすら未作成**。`pyproject.toml` の `[tool.hatch.build.targets.wheel].packages` には `"src"` と `"pipeline"` が列挙されているが、実体はこれから作成する。
- **`pipeline/case1/`**: 空ディレクトリのみ存在（`.gitkeep` なし）。`eda/` `training/` `evaluation/` `configs/` `output/` は未作成。
- **`tests/`**: 未作成。`pyproject.toml` の `testpaths=['tests']` と `--rootdir=tests` が既定値としてあるため、このパスで作成する必要がある。
- **`.env`, `.venv`**: いずれも未作成。`dev/setup` 実行時に `uv sync` で生成される想定。
- **`data/`**: 既にディレクトリのみ存在。リプレイ保存先として `data/replays/case1/` を想定。

### 2. 設定ファイル (`pyproject.toml`)

- **Python**: `>=3.14,<3.15` 固定。`target-version = "py314"`。
- **既存依存**: `kaggle-environments>=1.17.0`, `numpy>=2.2.6`, `pandas>=2.3.3`, `polars>=1.39.0`, `pyarrow>=23.0.1`, `pydantic>=2.10.6`, `pyyaml>=6.0.2`, `rich>=13.9.4`, `typer>=0.15.2`, `python-dotenv>=1.0.0`。
- **Dev**: `mypy>=1.15.0`, `pylint>=3.3.2`, `pytest>=8.3.4`, `pytest-coverage`, `ruff>=0.8.4`。
- **Ruff**: `line-length=88`, `select=["E","F","W","B","I"]`, `mccabe.max-complexity=5`（**ノートブックのオリジナル関数は複雑度が高いため per-file-ignores で緩和が必要**）。
- **Mypy**: `strict=true`, `disallow_any_decorated=false`, `disallow_any_unimported=false`, `ignore_missing_imports=true`（`kaggle_environments` に型情報なしでもOK）。
- **Pytest**: `--cov=src`（**`pipeline` はカバレッジ対象外**）。`asyncio_mode="auto"`。マーカーに `slow` (自己対戦/学習) が既定。
- **Hatch**: `packages = ["src","pipeline"]` — `pipeline.case1.xxx` としてimport可能。
- **Coverage**: `omit=["**/__init__.py"]`, HTML出力 `tests/coverage/report`。

### 3. 既存開発スクリプト (`dev/*`)

現状のスクリプト群は **旧プロジェクト（frontend/backend構造）の残骸** で、本リポジトリ構造と整合していない：

- `dev/setup`: `frontend` / `backend` ディレクトリを探しに行くが、それらは存在しない。`uv sync` 単発呼び出しに作り直すか、現状維持で `uv sync` を手動で打つか要判断。
- `dev/test-backend`: `cd backend` を含むため、このままでは動作しない。
- `dev/lint`, `dev/format`: 同上。

**判断**: 本機能で dev スクリプト修正まで踏み込むと scope が膨らむため、**CI 実行コマンドはドキュメントに `uv run ruff check . && uv run mypy src pipeline && uv run pytest` を明記**し、スクリプト修正は別タスク扱いにする。

### 4. ルール (`.claude/rules/*.md`)

- **`backend.md`** (`src/**`, `tests/**`): 200-400行/ファイル目安・最大800行、`NEVER mutate objects`、`print()` 禁止、`Any`/`cast`/`type: ignore` 禁止、`numpy` ベクトル化、軌道予測は初期化時にキャッシュ、1ターン1秒制約を明記。**テストは AAA パターン、`kaggle_environments.make("orbit_wars")` でシナリオ構築**。
- **`pipeline.md`** (`pipeline/**`): 戦略別に `pipeline/caseN/` を切る。各ケースは `eda/training/evaluation/configs/output/` を持ち、`README.md` 必須。評価指標は **勝率・平均最終スコア・レーティング更新予測・タイムアウト率**。自己対戦スクリプトは `typer` で CLI 化。出力先はタイムスタンプ付きディレクトリ。
- **`security.md`**: `.env`, `*.pem`, `*.key` 参照禁止。`print()` 禁止、環境変数経由。

### 5. 再現対象ノートブック仕様 (`docs/competition/20260418_baseline.md`)

- **エージェント**: ミッション計画型ヒューリスティック。`agent(obs)` がエントリポイント。
- **主要コンポーネント**:
  - `CONFIG`（80+ パラメータ）
  - `TYPES`（Planet / Fleet namedtuple — kaggle_environments と同一シグネチャ）
  - `GEOMETRY`（`dist`, `segment_hits_sun`, `point_to_segment_distance`）
  - `PHYSICS`（`fleet_speed`, `orbital_radius`, `predict_position`）
  - `WorldState` dataclass（中核の状態管理）
  - MISSIONS（`build_expansion/attack/snipe/swarm/reinforcement/crash_exploit_missions`）
  - `agent()` エントリポイント
- **主要アルゴリズム**: `build_arrival_ledger`, `simulate_planet_future`, `projected_state`, `search_safe_intercept`。
- **実行時間**: 22s / エピソード。**1ターン1秒以内**のタイムアウト制約を満たす。

### 6. Kaggle Orbit Wars 環境仕様 (`docs/competition/abstract.md`)

- **observation** は dict: `planets`, `fleets`, `player`, `angular_velocity`, `initial_planets`, `comets`, `comet_planet_ids`, `remainingOverageTime`。
- **action**: `list[[from_planet_id:int, angle:float, num_ships:int]]`。
- **configuration**: `episodeSteps=500`, `actTimeout=1`, `shipSpeed=6.0`, `cometSpeed=4.0`, `sunRadius=10.0`, `boardSize=100.0`。
- **コンビニエンスAPI**: `from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet, CENTER, ROTATION_RADIUS_LIMIT`。

## 技術的制約

1. **1ターン 1秒 (actTimeout=1)**: ホットパスで NumPy ベクトル化、軌道位置の事前キャッシュ必須。
2. **Kaggle Submission 実行環境**: 提出物は単一 `.py` または軽量モジュール群。重い依存（PyTorch等）は避ける。このベースラインは純粋な heuristic で、**NumPy のみで完結**可能。
3. **Mypy strict + Ruff mccabe max-complexity=5**: ノートブックの巨大関数（`agent`, `build_*_missions`）はそのまま移植すると両方に引っかかる。
   - **対応**: `pipeline/case1/baseline/` 配下には `per-file-ignores` を設定するか、関数を早めに分割する。
4. **`ships: int` 必須**: kaggle_environments は `num_ships` が float の場合に挙動不定。必ず `int(...)` キャスト。
5. **numpy>=2.2.6**: `np.log1p` 等のAPIはそのまま使えるが、旧 `np.int` / `np.float` は削除済み。`int`, `float` を使う。

## テスト方針の前提

- `tests/` が未作成なため、**tests/pipeline/case1/** のような階層を新規作成する。
- `pytest-cov` で `--cov=src` が既定。`pipeline/case1` はカバレッジ計算対象外のため、**共通処理は `src/` に置き、`pipeline/case1/` は CLI・設定・EDAに留める**方針と相性が良い。
- ただし、ベースライン再現の初期段階では「ノートブックと同一動作を保証する」ことが最優先のため、最初は単一ファイル `pipeline/case1/baseline/agent.py` に丸ごと置き、動作検証後に `src/agents/`, `src/features/`, `src/policies/`, `src/utils/` へ段階的に分解する二段階アプローチを採用する。

## キーファインディングス要約

| 項目 | 状態 | インプリケーション |
|------|------|----------------------|
| `src/`, `tests/`, `pipeline/case1/*` | **全て未作成** | 本機能ですべて新規作成 |
| `pyproject.toml` | 整備済み | そのまま利用可、ノートブック互換のため依存追加不要（NumPyのみ） |
| `kaggle-environments>=1.17.0` | インストール済み設定 | `make("orbit_wars")` が使える想定 |
| `dev/*` スクリプト | 旧プロジェクト残骸 | scope外で修正せず、コマンドを直接文書化 |
| Ruff `max-complexity=5` | 厳しい | per-file-ignoresで `pipeline/case1/baseline/**` を緩和 |
| Mypy strict | 有効 | Planet/Fleet は `typing.NamedTuple` で型付け、`cast` は使わずに条件分岐で narrow |
| 1ターン1秒制約 | 厳守 | NumPy ベクトル化 + 軌道位置キャッシュ |
| ノートブック取得手段 | Kaggle API 経由 | `~/.kaggle/kaggle.json` 設定後 `kaggle kernels pull sigmaborov/orbit-wars-2026-reinforce -p pipeline/case1/notebook/` |

## 出典

- `.claude/CLAUDE.md`, `.claude/rules/backend.md`, `.claude/rules/pipeline.md`, `.claude/rules/security.md`
- `pyproject.toml`
- `docs/competition/abstract.md`, `docs/competition/20260418_baseline.md`
- `dev/setup`, `dev/test-backend`, `dev/lint`, `dev/format`
