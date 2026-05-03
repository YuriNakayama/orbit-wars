# Rust Simulator — Codebase Research

ユーザーのリクエスト: kaggle_environments の `orbit_wars` 環境を Rust で書き直して10倍以上高速化する。あわせて `backend/` を `bot/` にリネームし、ルートに `simulator/` を作って Python 版（公式コピー）と Rust 版を並走させる。

## Deep Codebase Analysis

### Area 1: 既存 simulator 利用箇所 (kaggle_environments への依存)

`backend/` 配下で `kaggle_environments` を import している箇所を特定。リネーム + Rust 移行で必ず影響する API surface。

- **Files analyzed**:
  - `backend/src/submit/validator.py:58` — `from kaggle_environments import make`（Kaggle 提出 dry-run 用）
  - `backend/src/dataset/storage/loader.py:44-45` — `from kaggle_environments import make`（保存済み replay の再構築）
  - `backend/src/dataset/selfplay/executor.py:95-110` — `from kaggle_environments import make` + `env.run(wrapped)` が self-play の中核
  - `backend/src/evaluation/snapshot_update.py:19-27` — `kaggle_environments.make("orbit_wars", configuration={"agents": agents, "seed": seed})` + `env.step(no_op_actions)`（snapshot capture）
  - `backend/pipeline/rulebase/case0/main.py:21` — `from kaggle_environments.envs.orbit_wars.orbit_wars import Planet`（namedtuple のみを使用、interpreter は呼ばない）
  - `backend/pipeline/rulebase/case5/evaluation/{compare_v1,compare_v4,debug_splits}.py` — `kaggle_environments.make("orbit_wars", ...)` で評価マッチ実行
  - `backend/pipeline/rulebase/case4/{evaluation/compare_v2.py,baseline/core/physics.py}` — 前者は `make`、後者は **コメントで** `kaggle_environments.envs.orbit_wars.point_to_segment_distance` を mirror していると明記
  - `backend/tests/pipeline/rulebase/case[1-5]/test_baseline_agent.py` 系 — 単体テスト経路で `kaggle_environments` が import される
  - `backend/tests/pipeline/imitation/case1/test_agent_integration.py`

- **Current implementation**:
  - 一貫して **`env = make("orbit_wars", configuration={"agents": N, "seed": S})` → `env.run(agents)` または `env.step(actions)`** のパターン。
  - 観測 / アクションの schema は `orbit_wars.json`（97行）と Python 側 `Planet`/`Fleet` namedtuple（`backend/.venv/.../orbit_wars.py:9-14`）に固定。
  - `env.steps[-1]` から `obs.planets / fleets / step / player / angular_velocity / initial_planets / next_fleet_id / comets / comet_planet_ids` を読む形式。
  - `env.toJSON()` で replay 保存（gzip 圧縮、`backend/src/dataset/selfplay/executor.py:123`）。

- **Key interfaces** (Rust 移行で互換維持が必要なもの):
  - `make(env_name, configuration) -> Environment`
  - `Environment.step(actions: list[list[Action]]) -> None` （内部で `env.steps` を append）
  - `Environment.run(agents: list[Callable | str]) -> list[list[State]]`
  - `Environment.steps[-1][i]["observation"] -> dict`
  - `Environment.toJSON() -> dict`
  - `interpreter(state, env)` シグネチャ（snapshot/test レイヤで直接呼ぶ可能性が将来的にあり得る）

- **Patterns used**:
  - lazy import: `kaggle_environments` は worker process 内で import（multiprocessing 起因）
  - 観測は **dict + list の素のデータ**（`Planet`/`Fleet` namedtuple は agent 側で `[*p]` 解構するときだけ使う）
  - Self-play はすべて `multiprocessing.Pool` で並列化（`backend/src/dataset/selfplay/executor.py`）

- **Coupling & side effects**:
  - `kaggle_environments` バージョンは pyproject に `kaggle-environments>=1.17.0` として固定。新バージョンで spec/observation が変わると agent 全部に影響する。
  - 全 agent が **dict-based observation** を前提にしているので、Rust 版でも **同等の dict / list を Python に返す** API が必要（PyO3 bindings の場合）。
  - `env.toJSON()` の出力構造は Kaggle replay viewer と互換性を持つので変更できない。
  - `from_planet_id` / `next_fleet_id` の発番は決定論的でなければならない。観測の deep copy（`obs0.initial_planets`）も同様。

- **Test coverage**:
  - 公式パッケージ内 `test_orbit_wars.py:585行 / 27 test cases`（symmetry, combat, fleet/planet motion, comet, reward）
  - 既存リポでは `backend/tests/pipeline/rulebase/case4/test_engine_alignment.py` が **エンジン挙動の port** を検証している実績あり（`physics.py:173,200` の Python mirror に対応）

- **Gaps identified**:
  - `kaggle_environments` の `random.*` 呼び出しは **グローバル random state** に依存（seed なしで `generate_planets` が呼ばれる箇所がある）。Rust port ではここを seed 経由で再現可能にしておかないと parity test が崩れる。
  - 現状リポには Rust toolchain / `Cargo.toml` / maturin 設定が **存在しない**（`grep` 結果ゼロ）。新規追加が必要。
  - 公式 `orbit_wars.py` のライセンス確認が未実施（要確認: Kaggle / kaggle-environments の Apache-2.0 と思われる）。

---

### Area 2: `backend/` 参照箇所 (リネームで影響する surface)

`backend/` を `bot/` に rename するために更新が必要なすべての参照を列挙。

- **dev/ scripts** (リネームで全部書き換え):
  - `dev/setup:11-13`、`dev/test-backend:6-23`、`dev/lint:4`、`dev/format:4-6`、`dev/submit:10`、`dev/dvc:36-46`、`dev/vast:12`、`dev/create-worktree:46`（`ENV_DIRS` 配列）
  - `dev/test-backend` のスクリプト名そのものが残るかは要確認（CI workflow が参照）

- **GitHub Actions workflows** (`.github/workflows/`):
  - `build-push.yml:8` (`paths: backend/**`), `:51` (`working-directory: backend`)
  - `cron-kaggle-scrape.yml:56,59-86` (`working-directory: backend` + `uv run --project backend ...`)
  - `ci-backend.yml`: 名前のみ `backend` だが内容は repo root から `uv sync`（リネーム影響軽微）

- **infra/**:
  - `infra/runtime/Dockerfile:29-30` — `WORKDIR /opt/orbit-wars/backend` + `COPY backend/pyproject.toml backend/uv.lock ./`
  - `infra/module/application/ecr_runtime/main.tf:2` — コメント文字列のみ
  - `infra/environment/dev/README.md` — terraform の `backend "s3"` ブロック説明（不変、文脈が違うので無視）

- **DVC config** (`dvc.yaml`):
  - `cmd: uv run --directory backend python -m pipeline.imitation.case1.training.{preprocess,train,...}` — 全 stage が `--directory backend` 指定
  - `deps: backend/pipeline/...`、`outs: backend/pipeline/imitation/case1/policy/weights.pt`

- **`.claude/CLAUDE.md` & rules**:
  - `CLAUDE.md:25,38,45,64,65` — folder structure 説明、auto-load rule path
  - `.claude/rules/python.md`, `.claude/rules/backend/{pipeline,tests}.md`、`.claude/rules/command.md`、`.claude/rules/data.md` — ほぼ全 rule に `backend/` 文字列が散在

- **docs/plans/** (既存プラン):
  - 既存 9 計画ディレクトリは text 上の参照のみ（履歴的価値あり）。リネーム済 path に書き換えるか、明示的に "後方互換のため当時の表記を残す" と注記するか方針判断が必要。

---

### Area 3: `data/` symlink 構造

ルート直下の `data/` は **メイン repo の `data/` への symlink**（`dev/create-worktree:38-43`）。`simulator/` を新設する際、`data/` の扱いは変えない。

- **gitignore**: `/data` 全体が ignore 対象、`.dvc` ファイルと `*.md` だけ git 追跡。
- **DVC cache**: `/Users/user/project/orbit-wars/.dvc/cache` を全 worktree で共有（`command.md` の concurrent execution warning）。

---

### Area 4: pyproject.toml / hatch ビルドスコープ

- `backend/pyproject.toml:25-26` — `[tool.hatch.build.targets.wheel] packages = ["src/submit", "src/dataset", "src/vast", "src/evaluation", "src/utils", "pipeline"]`
- リネーム後は `bot/pyproject.toml` がそのまま生きる（hatch package list はディレクトリ相対なので変更不要）。
- ただし mypy/ruff の per-file-ignores パスは `pipeline/...` で **bot/ 内相対** なのでこれも変えなくてよい。
- 一方で `dev/test-backend` 等が `cd backend` していた箇所を `cd bot` に変えるだけで lint/test は通る想定。

---

## Technical Constraints

- **Python 3.13** 固定 (`backend/.python-version` / pyproject `requires-python = ">=3.13,<3.14"`).
- **Kaggle Submission constraint** — Kaggle ランタイムには Rust binary は持ち込めない（pure Python only）。よって **Rust simulator は self-play / 評価 / 学習 high-throughput 経路でのみ使用**、Kaggle 提出 agent 自体は Python 側（既存どおり）。
- **NumPy / PyTorch wheels** が依存にあるので、Rust を加える場合は CI のビルド時間が増えないよう **maturin の wheel を pre-build** するか **CI で skip** する戦略が必要。
- **Determinism** — 学習データ収集と回帰テストの両方で **同一 seed → 同一 trajectory** が必須。Python の `random.Random(seed)` 呼び出し順序を Rust 側でも完全に再現する必要があり、これは **大きな実装難度**。
- **公式 simulator コードのライセンス** — Kaggle/kaggle-environments は Apache-2.0 と推定。`simulator/python/` にコピーする際は LICENSE を同梱する。
- **既存の `backend/.dvc/` cache パス** は変えると DVC pull が再走するのでコスト大。`.dvc/cache` は repo root 直下なのでリネーム影響なし（要確認）。

---

## Key Findings Summary

1. **rename の影響面は dev/ scripts + .github/workflows + infra/Dockerfile + dvc.yaml + .claude/rules/**。pyproject.toml 内部相対パスは変えなくてよい。コミット 1 回で全体置換可能。
2. **`kaggle_environments` の API surface は dict + list ベース**で素直。Rust 側で同型 PyDict / PyList を返せば既存 agent コードに変更不要。
3. **`orbit_wars.py` 本体は 806 行 + 585 行の test**。物理は `interpreter()` 関数 1 つに集約され、副作用は `state[*].observation.{planets,fleets,...}` への `list` mutation のみ。Rust への port は十分現実的。
4. **`generate_planets` / `generate_comet_paths` の rejection sampling が Python `random` 依存**。Rust 側で `rand` crate の seed compat を取るか、planet 生成だけ Python に残す（hybrid）かの戦略選択が必要。
5. **公式 `test_orbit_wars.py` の 27 test cases がそのまま受け入れテスト**になる。これを Python 側で再生成 → Rust 側でも parity を確認、が TDD のゴールドスタンダード。
6. **Kaggle ランタイム制約**で Rust simulator を提出 agent には組み込めない。利用シーンは "self-play / RL training / 評価" に限定。`Vast.ai` 学習（GPU）と組み合わせる場合に最大の旨味がある。
