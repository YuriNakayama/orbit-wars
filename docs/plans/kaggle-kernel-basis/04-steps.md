# kaggle-kernel-basis — Implementation Steps

> **Implementation status (2026-05-20)**: Step 1-9 完了 (Step 2 は case1/3/4/5/6/7/8/9/10 全 9 case に適用済)。Step 10 (実 Kaggle 上の e2e smoke) は API key 必須のため手動実行。

実装は **vast.run_meta の拡張 → train.py 改修 → kaggle_kernel パッケージ → notebook template → CLI サブコマンド → e2e + ドキュメント** の順で進める。Step 内の **並列可能** マークは独立着手可能なペアを示す。各 Step で unit test を同時にコミットし、`dev/test-bot` をグリーンに維持する。

---

## Step 1: vast.run_meta.RunMetadata に kaggle_kernel_meta field 追加 (後方互換)

**Target**: bot
**Dependencies**: None

### Overview
`bot/src/vast/run_meta.RunMetadata` に optional フィールド (`kaggle_kernel_meta`) を追加。`schema_version=1` を維持し、既存 vast/runpod run.json の読み書きに影響を与えない。

### Work Items
- [ ] `bot/src/vast/run_meta.py` の `RunMetadata` dataclass に `kaggle_kernel_meta: dict[str, Any] | None = None` を追加 (default で後方互換)。
- [ ] field 順序: `runpod_*` の直後に `kaggle_kernel_meta` を置き、JSON 出力の可読性を確保。
- [ ] `bot/tests/src/vast/test_run_meta.py` に新 field の round-trip テスト追加 (既存テストはそのまま pass)。
- [ ] 既存 vast / runpod run.json fixture (`kaggle_kernel_meta` 未含) が `read_run_json` で default `None` で埋まることを確認するテスト。

### Target Files
- `bot/src/vast/run_meta.py`
- `bot/tests/src/vast/test_run_meta.py`

### Acceptance Criteria
- 既存 vast/runpod run.json を `read_run_json` で読めば default `None` で埋まる。
- `write_run_json` → `read_run_json` の round-trip で `kaggle_kernel_meta` が保持される。
- `dev/test-bot` グリーン。

---

## Step 2: train.py の Kaggle Kernel env 検出を case1/case3/case4/case8/case9 で追加 (並列可能 with Step 3)

**Target**: bot
**Dependencies**: Step 1

### Overview
複数 train.py で `ORBIT_WARS_KAGGLE_KERNEL_SLUG` / `ORBIT_WARS_KAGGLE_KERNEL_META` env を検出し、`RunMetadata.kaggle_kernel_meta` を埋める。三 provider (vast / runpod / kaggle_kernel) の env が同時セットされたら明示エラー (Risk 防御)。

### Work Items
- [ ] `bot/pipeline/imitation/case1/training/train.py` に env 検出ロジック追加:
  ```python
  vast_id = os.environ.get("ORBIT_WARS_VAST_INSTANCE_ID")
  rp_id = os.environ.get("ORBIT_WARS_RUNPOD_POD_ID")
  kk_slug = os.environ.get("ORBIT_WARS_KAGGLE_KERNEL_SLUG")
  active = [bool(vast_id), bool(rp_id), bool(kk_slug)]
  if sum(active) > 1:
      raise RuntimeError(
          "Multiple provider env vars are set simultaneously. "
          "Set only one of ORBIT_WARS_VAST_INSTANCE_ID / "
          "ORBIT_WARS_RUNPOD_POD_ID / ORBIT_WARS_KAGGLE_KERNEL_SLUG."
      )
  ```
- [ ] `kaggle_kernel_meta` を `ORBIT_WARS_KAGGLE_KERNEL_META` env (JSON 文字列) から `json.loads`。空/未設定なら `None`。malformed JSON は `RuntimeError`。
- [ ] case3 / case4 / case8 / case9 の train.py にも同じパッチを適用 (該当箇所は env 解決ブロック)。
- [ ] tests:
  - `bot/tests/pipeline/imitation/case1/training/test_train_run_dir.py` に新ケース:
    - `ORBIT_WARS_KAGGLE_KERNEL_SLUG="username/foo"` のみ → run.json に `kaggle_kernel_meta` が dict、他 provider field は None。
    - 三者から 2 つ以上 set → `RuntimeError`。
    - `ORBIT_WARS_KAGGLE_KERNEL_META='{"accelerator":"gpu-t4x2"}'` で snapshot が dict に展開される。
  - case3/case4/case8/case9 にも同等 mini test を追加。

### Target Files
- `bot/pipeline/imitation/case{1,3,4,8,9}/training/train.py`
- `bot/tests/pipeline/imitation/case{1,3,4,8,9}/training/test_train_run_dir.py`

### Acceptance Criteria
- vast / runpod / kaggle_kernel の env 単独でそれぞれ run.json の対応 field のみ埋まる。
- 複数 set で RuntimeError、メッセージに 3 経路名前を含む。
- `dev/test-bot` グリーン。

---

## Step 3: bot/src/kaggle_kernel/ package skeleton + auth (並列可能 with Step 2)

**Target**: bot
**Dependencies**: None

### Overview
パッケージの骨組みと auth helper を実装。

### Work Items
- [ ] `bot/src/kaggle_kernel/__init__.py` (docstring + `import kaggle as kaggle_sdk` 規約明記)
- [ ] `bot/src/kaggle_kernel/__main__.py` で `from .cli.app import app; app()`
- [ ] `bot/src/kaggle_kernel/auth.py`:
  - `KaggleCreds` frozen dataclass (`username`, `key`)
  - `load_kaggle_creds()` の 3 段 fallback:
    1. env (`KAGGLE_USERNAME` + `KAGGLE_KEY`)
    2. `bot/.env` を `python-dotenv` で読む
    3. `~/.kaggle/kaggle.json` を JSON 解析
  - 失敗時 `CredentialsError` (3 経路の actionable hint 含む)
- [ ] `bot/pyproject.toml` の `[project.dependencies]` に `kaggle>=1.6` を追加 (runtime dep)。
- [ ] `bot/tests/src/kaggle_kernel/test_auth.py` で 3 fallback パスを mock fixtures で網羅。

### Target Files
- `bot/src/kaggle_kernel/__init__.py`
- `bot/src/kaggle_kernel/__main__.py`
- `bot/src/kaggle_kernel/auth.py`
- `bot/tests/src/kaggle_kernel/test_auth.py`
- `bot/pyproject.toml`

### Acceptance Criteria
- `python -c "from kaggle_kernel.auth import load_kaggle_creds"` 成功。
- `dev/test-bot` グリーン (新規 unit test が pass)。

---

## Step 4: dataset builder + metadata + api

**Target**: bot
**Dependencies**: Step 3

### Overview
`bot/` を Kaggle Dataset として upload するための snapshot 作成と CRUD wrapper。

### Work Items
- [ ] `bot/src/kaggle_kernel/dataset/builder.py`:
  - `build_snapshot(repo_root: Path, dest_dir: Path, commit_sha: str, include_wheels: list[Path] | None = None) -> Path`
  - 除外規則: `data/`, `.venv/`, `__pycache__/`, `*.pyc`, `.dvc/`, `.git/`, `docs/`, `infra/`, `bot/tests/`, `node_modules/`
  - 包含: `bot/src/`, `bot/pipeline/`, `bot/pyproject.toml`, `bot/uv.lock`, `simulator/python/`, `simulator/rust/src/` (ビルド済 wheel は別 dir)
  - wheel 同梱: `include_wheels` で渡された .whl を `<dest>/wheels/` にコピー
- [ ] `bot/src/kaggle_kernel/dataset/metadata.py`:
  - `write_dataset_metadata(dest_dir: Path, slug: str, title: str, commit_sha: str)` で `dataset-metadata.json` を生成
  - `is_private=true` 固定、license は Apache-2.0
- [ ] `bot/src/kaggle_kernel/dataset/api.py`:
  - `push_dataset_version(dest_dir: Path, version_notes: str, dry_run: bool = False)` — 既存 dataset の version up
  - `create_new_dataset(dest_dir: Path)` — 初回作成
  - `dataset_status(slug: str) -> dict` — processing 状態取得
  - `latest_version_commit(slug: str) -> str | None` — 最新 version の `version_notes` から commit SHA を抽出 (`commit=abc1234` 形式)
- [ ] tests:
  - `test_dataset_builder.py`: 除外規則の検証 (`data/` が混入しないこと、wheel が同梱されること)
  - `test_dataset_api.py`: KaggleApi を mock、create/version パスを round-trip

### Target Files
- `bot/src/kaggle_kernel/dataset/{__init__,builder,metadata,api}.py`
- `bot/tests/src/kaggle_kernel/test_dataset_{builder,api}.py`

### Acceptance Criteria
- snapshot dir に `data/` / `.git/` が含まれない。
- mock 経由で `dataset_create_new` / `dataset_create_version` が呼ばれることをテストで検証。
- `dev/test-bot` グリーン。

---

## Step 5: notebook template render

**Target**: bot
**Dependencies**: Step 3

### Overview
`.ipynb` を生成する純関数を実装。

### Work Items
- [ ] `bot/src/kaggle_kernel/kernel/template.py`:
  - `render_notebook(ctx: RenderContext) -> dict` — Jupyter notebook の dict (nbformat 4) を返す
  - `RenderContext` dataclass: `run_id`, `commit_sha`, `branch`, `case`, `train_module`, `config_arg`, `dataset_slug`, `dataset_version`, `accelerator`, `kaggle_kernel_meta_initial: dict`
  - placeholder バリデーション: `_VALID_VALUE = re.compile(r"^[A-Za-z0-9._\-/:=]+$")` で env 値を検証
  - cell A-E を生成 (`03-architecture.md` の Notebook Template Structure 通り)
  - 結果 dict は `json.dumps` で serializable、`nbformat.read` で valid と検証可能
- [ ] tests:
  - `test_kernel_template.py`:
    - 生成された notebook が `nbformat.validate` を pass
    - placeholder が全て展開済 (`<RUN_ID>` 等が残らない)
    - shell injection 試行 (`;rm -rf /`) は `ValueError` で reject

### Target Files
- `bot/src/kaggle_kernel/kernel/{__init__,template,state}.py`
- `bot/tests/src/kaggle_kernel/test_kernel_template.py`

### Acceptance Criteria
- 生成 notebook が `nbformat.validate` を pass。
- 全 placeholder が展開済。
- shell injection を reject。
- `dev/test-bot` グリーン。

---

## Step 6: kernel runner (push + status polling)

**Target**: bot
**Dependencies**: Step 3, Step 5

### Overview
KaggleApi を使って kernel push と status polling を行う runner。

### Work Items
- [ ] `bot/src/kaggle_kernel/kernel/state.py`:
  - `KernelStatus` enum: `QUEUED`, `RUNNING`, `COMPLETE`, `ERROR`, `CANCEL_ACKNOWLEDGED`, `UNKNOWN`
  - `parse_status(raw: dict) -> KernelStatus`
- [ ] `bot/src/kaggle_kernel/kernel/runner.py`:
  - `push_kernel(kernel_dir: Path) -> KernelPushResult` — `kernels_push_cli` の wrapper、戻り値に `slug` / `version` を含む
  - `poll_status(slug: str, interval: float = 60.0, timeout: float = 36000.0) -> KernelStatus` — 完了 / エラー / cancel まで polling
  - `KernelPushResult` dataclass
- [ ] tests:
  - `test_kernel_runner.py`: KaggleApi を mock、push → polling → COMPLETE 経路と ERROR 経路を round-trip

### Target Files
- `bot/src/kaggle_kernel/kernel/{state,runner}.py`
- `bot/tests/src/kaggle_kernel/test_kernel_runner.py`

### Acceptance Criteria
- mock 経由で QUEUED → RUNNING → COMPLETE の遷移をテストで再現。
- ERROR / CANCEL_ACKNOWLEDGED で例外 or 戻り値で expressive に表現。
- `dev/test-bot` グリーン。

---

## Step 7: artifacts (launch, output, cost)

**Target**: bot
**Dependencies**: Step 1, Step 6

### Overview
launch.json の round-trip、kernel output pull → run_dir 配置 → dvc add、月次 free-hour 集計。

### Work Items
- [ ] `bot/src/kaggle_kernel/artifacts/launch.py`:
  - `LaunchMeta` dataclass (`03-architecture.md` 通り)
  - `write_launch_json(run_dir: Path, meta: LaunchMeta)` / `read_launch_json(run_dir: Path) -> LaunchMeta`
- [ ] `bot/src/kaggle_kernel/artifacts/output.py`:
  - `pull_kernel_output(slug: str, tmp_dir: Path)` — `KaggleApi().kernels_output_cli(slug, path=tmp_dir)`
  - `place_into_run_dir(tmp_dir: Path, run_dir: Path)` — `/kaggle/working/runs/<run_id>/` 内容を `run_dir` にコピー
  - `dvc_add(run_dir: Path)` — `subprocess.run(["dvc", "add", str(run_dir)], cwd=repo_root, check=True)`
- [ ] `bot/src/kaggle_kernel/artifacts/cost.py`:
  - `aggregate_runs(runs_root: Path, month: str) -> CostReport` — `kaggle_kernel_meta != None` の run のみ集計
  - `runtime_seconds` の合計を `total_gpu_hours_used` として算出
  - `render_markdown(report: CostReport) -> str`
  - 出力先: `docs/experiment/kaggle_kernel_cost_report_<YYYY-MM>.md`
- [ ] tests:
  - `test_artifacts_output.py`: KaggleApi mock + subprocess mock で round-trip
  - `test_artifacts_cost.py`: 模擬 run.json 群から正しい free-hour 集計と markdown 出力

### Target Files
- `bot/src/kaggle_kernel/artifacts/{__init__,launch,output,cost}.py`
- `bot/tests/src/kaggle_kernel/test_artifacts_{output,cost}.py`

### Acceptance Criteria
- mock 経由で kernel output pull → run_dir 配置 → dvc add の subprocess 呼び出しが正しい引数で行われる。
- 月次集計の markdown 出力が想定 schema 通り。
- `dev/test-bot` グリーン。

---

## Step 8: CLI integration (Typer subcommands)

**Target**: bot
**Dependencies**: Step 4, Step 5, Step 6, Step 7

### Overview
`dev/kaggle-kernel <subcmd>` の全エントリポイントを Typer で構築。

### Work Items
- [ ] `bot/src/kaggle_kernel/cli/app.py`:
  - `train(commit_sha, case="case1", accelerator="gpu-t4x2", seed=0, label=None, no_internet=False, watch=False, dataset_bump_only=False, max_hours=8.5)`
  - `dataset_push(commit_sha=None, force_new=False)` / `dataset_status(slug=None)`
  - `pull(run_id, case="case1")`
  - `promote(run_id, case="case1", eval_results=None)` — `runpod_io.artifacts.run_meta.promote_to_canonical` を共有 import
  - `status(run_id, case="case1")`
  - `ps(case=None, all_=False)`
  - `logs(run_id, case="case1", tail=None, grep=None)`
  - `watch(run_id, case="case1", poll_interval=60, max_wait=36000)`
  - `cost_report(month=None)`
- [ ] `bot/src/kaggle_kernel/__main__.py` 経由で `python -m kaggle_kernel` が動く。
- [ ] tests:
  - `test_cli_train.py`: train サブの統合テスト (`kaggle.api` 全 mock、`subprocess` mock)
  - `test_cli_pull.py`: pull サブの統合テスト

### Target Files
- `bot/src/kaggle_kernel/cli/{__init__,app}.py`
- `bot/tests/src/kaggle_kernel/test_cli_{train,pull}.py`

### Acceptance Criteria
- `python -m kaggle_kernel --help` で全サブコマンドが表示される。
- `python -m kaggle_kernel train <sha>` を mock 環境下で round-trip 実行できる。
- `dev/test-bot` グリーン。

---

## Step 9: dev/kaggle-kernel wrapper + docs 更新

**Target**: dev/, .claude/
**Dependencies**: Step 8

### Work Items
- [ ] `dev/kaggle-kernel` を作成 (3 行 bash):
  ```bash
  #!/bin/bash
  set -euo pipefail
  cd "$(dirname "$0")/.."
  exec uv run --directory bot python -m kaggle_kernel "$@"
  ```
- [ ] `chmod +x dev/kaggle-kernel`
- [ ] `.claude/rules/command.md` に "Kaggle Kernel Training" 章を追加 (RunPod 章の直下)。サブコマンド一覧 + 1 サイクル例。
- [ ] CLAUDE.md (or 関連 doc) に三基盤の使い分け表を追加 (既に README.md にあるためリンクで足りる場合は省略可)。

### Target Files
- `dev/kaggle-kernel`
- `.claude/rules/command.md`

### Acceptance Criteria
- `dev/kaggle-kernel --help` で help 出力。
- `dev/kaggle-kernel train --help` で train サブの引数が出る。
- doc 更新で worktree クリーン。

---

## Step 10: e2e smoke + 06-testing.md 仕上げ

**Target**: bot, docs
**Dependencies**: Step 9

### Work Items
- [ ] **手動 smoke** (1 epoch case1 で本番 Kaggle に push):
  - `dev/kaggle-kernel dataset push --commit-sha HEAD` (初回のみ)
  - `dev/kaggle-kernel train HEAD --case case1 --accelerator gpu-t4x2 --watch`
  - `dev/kaggle-kernel pull <run_id> --case case1`
  - `data/output/models/imitation/case1/runs/<run_id>/best.pt` の存在確認
  - `git status` で `.dvc` ファイル更新確認
- [ ] `06-testing.md` に実行コマンド + 期待出力を記録。
- [ ] README.md の troubleshooting 章を smoke で発見した issue で補強。
- [ ] PR 用 summary (Why / What / Test plan) を準備。

### Target Files
- `docs/plans/kaggle-kernel-basis/06-testing.md`
- `docs/plans/kaggle-kernel-basis/README.md`

### Acceptance Criteria
- 1 epoch case1 が Kaggle Kernel 上で COMPLETE。
- artifact が `policy/weights.pt` に promote 可能。
- DVC commit + git status クリーンで PR 準備完了。

---

## Dependency Graph

```
Step 1 (run_meta) ──► Step 2 (train.py パッチ)
                  ├─► Step 7 (artifacts)
                  └─► Step 8 (CLI)

Step 3 (skeleton + auth) ──► Step 4 (dataset)
                           ├─► Step 5 (template)
                           └─► Step 6 (runner)

Step 4, 5, 6, 7 ──► Step 8 (CLI)
Step 8 ──► Step 9 (wrapper + docs)
Step 9 ──► Step 10 (e2e smoke)
```

並列着手可能ペア:
- (Step 2, Step 3)
- (Step 4, Step 5, Step 6) — Step 3 完了後
