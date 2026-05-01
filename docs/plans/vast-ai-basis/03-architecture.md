# vast-ai-basis — Architecture Design

## Overall Diagram

```
┌──────────────── Local Machine ─────────────────────┐
│                                                    │
│  developer                                         │
│    │                                               │
│    ▼                                               │
│  dev/vast-train <sha> [--stage NAME]  (bash)       │
│    │                                               │
│    ▼                                               │
│  uv run -m vast train ...  (typer CLI)             │
│    │                                               │
│    ├── offers.search_offers(filter)──┐             │
│    ├── pick_offer (rich table + stdin)             │
│    ├── instance.create(offer_id, ...)              │
│    │                                               │
│    │   AWS creds (--env)  ┌───────────────────────┐│
│    └─────────────────────►│       Vast.ai         ││
│                           │   instance (GPU)      ││
│                           │                       ││
│                           │   onstart.sh:         ││
│                           │   1. env >> /etc/env  ││
│                           │   2. git clone+co     ││
│                           │   3. uv sync          ││
│                           │   4. dvc pull         ││
│                           │   5. dvc repro <stage>││
│                           │      (writes run dir) ││
│                           │   6. dvc push         ││
│                           │   7. self-destroy     ││
│                           └───────────┬───────────┘│
│                                       │            │
│                          dvc push     │            │
│                                       ▼            │
│                          ┌────────────────────────┐│
│                          │   S3: orbit-wars-dvc   ││
│                          │   (DVC remote)         ││
│                          └────────────┬───────────┘│
│                                       │            │
│  developer (later)                    │            │
│    │                                  │            │
│    ▼                                  │            │
│  dev/vast-pull <run_id>               │            │
│    │ uv run -m vast pull ...          │            │
│    │ -> dvc pull data/output/.../runs/<run_id>/      │
│    │ <-- run.json + best.pt + metrics.json         │
│    ▼                                               │
│  Local evaluation:                                 │
│   uv run python -m pipeline.imitation.case1.       │
│     evaluation.eval_vs_baseline                    │
│     (with weights override path)                   │
│    │                                               │
│    ├── update run.json: local_eval_results         │
│    │                                               │
│    ▼                                               │
│  dev/promote-weights <run_id> [if adopted]         │
│    cp run_dir/best.pt -> policy/weights.pt         │
│    dvc commit, then human git commit               │
│                                                    │
└────────────────────────────────────────────────────┘
```

## Directory Layout (new files only)

```
backend/
  src/
    vast/                            # NEW package
      __init__.py
      __main__.py                    # `python -m vast` entrypoint
      cli.py                         # typer Typer() with subcommands: train, pull, promote, cost-report
      offers.py                      # search_offers, format_offers_table, pick_offer (interactive)
      instance.py                    # create_instance, build_env_string, render_onstart
      run_meta.py                    # RunMetadata dataclass + run_id generation + JSON I/O
      cost.py                        # aggregate run.json -> markdown report
      onstart.sh.tmpl                # bash template with placeholders <COMMIT_SHA> <RUN_ID> <STAGE> <BRANCH> <REPO_URL>
      auth.py                        # read AWS creds from ~/.aws/credentials (orbit-wars profile), VAST_API_KEY from .env
  tests/
    vast/
      test_offers.py
      test_instance.py
      test_run_meta.py
      test_cost.py
      test_onstart_template.py       # sed replacement coverage

dev/
  vast-train                         # NEW bash thin wrapper -> python -m vast train "$@"
  vast-pull                          # NEW
  vast-promote                       # NEW (renamed from promote-weights for consistency)
  vast-cost-report                   # NEW

data/output/                           # NEW directory (gitignored, DVC managed via stage outs or `dvc add`)
  models/
    imitation/
      case1/
        runs/
          <run_id>/
            best.pt                  # weights produced by train
            metrics.json             # epoch history + best metrics (small, optionally cache:false)
            run.json                 # full lineage record
```

## Backend Design

### `backend/src/vast/cli.py` (typer)

```python
app = typer.Typer(no_args_is_help=True)

@app.command()
def train(
    commit_sha: str = typer.Argument(..., help="must be pushed to origin"),
    stage: str = typer.Option("train_imitation_case1", "--stage", "-s"),
    seed: int = typer.Option(0, "--seed"),
    label: str | None = typer.Option(None, "--label"),
    cost_limit_usd: float = typer.Option(1.0, "--cost-limit"),
) -> None:
    ...

@app.command()
def pull(run_id: str) -> None: ...

@app.command()
def promote(run_id: str, message: str | None = None) -> None: ...

@app.command("cost-report")
def cost_report(month: str | None = None) -> None: ...
```

- `train` の流れ:
  1. `commit_sha` の git 検証 (`git cat-file -e <sha>` + `git branch -r --contains <sha>` で push 確認)
  2. `auth.load_aws_creds(profile="orbit-wars")` と `auth.load_vast_api_key()` を実行（失敗時は actionable エラー）
  3. `offers.search_offers(filter)` → 上位 10 件 → `offers.pick_offer()` (rich table + Prompt.ask)
  4. 推定コスト計算（`dph_total * 0.5h`）→ `cost_limit_usd` 超過時は `typer.confirm()`
  5. `run_meta.generate_run_id(branch, commit_sha, seed)`
  6. `instance.render_onstart(template_path, vars)` → tmp file
  7. `instance.build_env_string(aws_creds, vast_api_key, run_id, ...)`
  8. `instance.create_instance(...)` を呼び、結果を表示
  9. ユーザーへ `vastai logs <id>` でモニタリング案内
- `pull` の流れ: `subprocess.run(["uv", "run", "--directory", "backend", "dvc", "pull", f"data/output/.../runs/{run_id}"])` → `run.json` 表示
- `promote` の流れ: F6 の通り `cp` + `dvc commit` + git status 表示（自動 commit はしない）
- `cost-report`: `glob data/output/.../runs/*/run.json` → 月単位集計 → markdown 出力

### `backend/src/vast/offers.py`

```python
@dataclass(frozen=True)
class Offer:
    id: int
    gpu_name: str
    num_gpus: int
    dph_total: float
    reliability: float
    geolocation: str | None
    cuda_max_good: float | None
    inet_down: float | None
    verified: bool

def search_offers(
    *,
    gpu_names: tuple[str, ...] = ("RTX_3090", "RTX_4090", "RTX_A6000", "A100"),
    max_dph: float = 1.0,
    min_reliability: float = 0.99,
    min_cuda: float = 12.0,
    limit: int = 10,
) -> list[Offer]:
    """vastai SDK の search_offers をラップ。返り値は dph_total asc。"""

def format_table(offers: list[Offer]) -> Table:
    """rich.Table を返す。# / GPU / num / dph / reliability / region / cuda の列。"""

def pick_offer(offers: list[Offer]) -> Offer:
    """rich Prompt で番号入力を取り、対応する Offer を返す。範囲外は再入力。"""
```

### `backend/src/vast/instance.py`

```python
def render_onstart(
    template_path: Path,
    *,
    commit_sha: str,
    run_id: str,
    stage: str,
    branch: str,
    repo_url: str,
) -> Path:
    """sed 相当の str.replace を 5 箇所適用し tmp ファイルを返す。"""

def build_env_string(
    *,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    aws_default_region: str,
    vast_api_key: str,
    run_id: str,
    git_sha: str,
) -> str:
    """vastai --env に渡す '-e KEY=VAL ...' 形式の文字列を組み立てる。"""

def create_instance(
    offer_id: int,
    *,
    onstart_path: Path,
    env_string: str,
    image: str = "pytorch/pytorch:2.6.0-cuda12.4-runtime",
    disk_gb: int = 40,
    label: str,
) -> int:
    """VastAI().create_instance(...) を呼び、instance id を返す。"""
```

### `backend/src/vast/onstart.sh.tmpl` (placeholders で sed 置換)

```bash
#!/bin/bash
set -euo pipefail
exec > >(tee -a /var/log/onstart.log) 2>&1

VAST_API_KEY="${VAST_API_KEY:-}"
INSTANCE_ID="${VAST_CONTAINERLABEL:-unknown}"  # vast 提供 env

cleanup_destroy() {
  local exit_code=$?
  echo "[onstart] cleanup status=${exit_code}"
  if [ "${exit_code}" -ne 0 ]; then
    # 失敗パス: instance を残して人間が ssh で原因確認できるようにする (NFR-3)
    echo "[onstart] failure: leaving instance up for inspection"
    exit "${exit_code}"
  fi
  # 成功パス: self-destroy
  if command -v vastai >/dev/null 2>&1; then
    vastai destroy instance "${INSTANCE_ID}" || true
  fi
}
trap cleanup_destroy EXIT

echo "[onstart] step=env_persist"
env >> /etc/environment

echo "[onstart] step=clone"
mkdir -p /workspace && cd /workspace
git clone <REPO_URL> orbit-wars
cd orbit-wars
git checkout <COMMIT_SHA>

echo "[onstart] step=install_uv"
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

echo "[onstart] step=uv_sync"
uv sync --locked --all-extras --dev --directory backend

echo "[onstart] step=install_vastai"
# self-destroy 用 CLI
uv pip install --directory backend vastai || pip install vastai

echo "[onstart] step=dvc_remote_profile"
# AWS profile を default に切替 (env 経由 credentials を読む)
uv run --directory backend dvc remote modify --local s3 profile default

echo "[onstart] step=dvc_pull"
uv run --directory backend dvc pull

echo "[onstart] step=mkdir_run"
mkdir -p data/output/models/imitation/case1/runs/<RUN_ID>

echo "[onstart] step=dvc_repro"
ORBIT_WARS_RUN_DIR=data/output/models/imitation/case1/runs/<RUN_ID> \
  ORBIT_WARS_RUN_ID=<RUN_ID> \
  ORBIT_WARS_GIT_SHA=<COMMIT_SHA> \
  ORBIT_WARS_GIT_BRANCH=<BRANCH> \
  ORBIT_WARS_VAST_INSTANCE_ID="${INSTANCE_ID}" \
  uv run --directory backend dvc repro <STAGE>

echo "[onstart] step=dvc_push_runs"
uv run --directory backend dvc push data/output/models/imitation/case1/runs/<RUN_ID>

echo "[onstart] step=done"
```

- placeholder は **山括弧フォーマット** で sed 置換 (`s|<COMMIT_SHA>|abc1234|g`)。
- self-destroy は `trap EXIT` で実装。**失敗時は残す（人間がデバッグ可能）、成功時のみ自動破壊** (NFR-3 の方針)。
- `<RUN_ID>` 等は `instance.render_onstart` で Python 側が安全な文字種（英数 + `_-`）に sanitize 済みなので shell injection リスクはなし。

### `backend/src/vast/run_meta.py`

```python
@dataclass
class RunMetadata:
    run_id: str
    git_sha: str
    git_branch: str
    params_hash: str
    seed: int
    vast_instance_id: int | None
    gpu_name: str | None
    vast_offer_snapshot: dict[str, Any] | None
    command: str
    weights_path: str
    train_metrics: dict[str, Any]
    local_eval_results: dict[str, Any] | None
    status: str  # running / pushed / evaluated / adopted / failed
    created_at: str
    updated_at: str

def generate_run_id(branch: str, sha: str, seed: int, *, now: datetime | None = None) -> str:
    """<YYYYMMDD-HHMMSS>__<branch_slug>__<sha7>__seed<N>"""

def hash_params(params_yaml_path: Path) -> str:
    """sha256 of canonicalized YAML, return first 12 chars."""

def write_run_json(run_dir: Path, meta: RunMetadata) -> None:
    """json.dumps with 2-space indent, ensure_ascii=False, sort_keys=False."""

def update_run_json(run_dir: Path, **patch: Any) -> RunMetadata:
    """Read, patch, write atomically; updates updated_at."""
```

### `backend/src/vast/cost.py`

```python
def aggregate_runs(runs_root: Path, month: str | None = None) -> CostReport:
    """run.json を読み込み、月単位で集計。"""

def render_markdown(report: CostReport) -> str:
    """| run_id | gpu | dph | runtime | cost_usd | status | のテーブル + summary。"""
```

### `train.py` 改修箇所

```python
# 既存 _seed_all に CUDA seed 追加
def _seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)

# train() 内で device 解決
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepSetsPolicy(model_config).to(device)

# DataLoader: pin_memory + non_blocking transfer
train_loader = DataLoader(..., pin_memory=(device.type == "cuda"))
# _to_batch_features 後に .to(device, non_blocking=True) を chain

# weights_out の override
run_dir_env = os.environ.get("ORBIT_WARS_RUN_DIR")
if run_dir_env:
    run_dir = Path(run_dir_env).resolve()
    weights_out = run_dir / "best.pt"
    metrics_out = run_dir / "metrics.json"
    write_run_metadata = True  # run.json も書く
else:
    weights_out = _abspath(train_cfg["weights_out"])
    metrics_out = None
    write_run_metadata = False

# 学習後
torch.save(best_state.cpu_state_dict, weights_out)  # GPU -> CPU 変換
if metrics_out:
    metrics_out.write_text(json.dumps({...}))
if write_run_metadata:
    from vast.run_meta import RunMetadata, write_run_json
    write_run_json(run_dir, RunMetadata(...))
```

## Data Model

### `run.json` schema (v1)

```json
{
  "schema_version": 1,
  "run_id": "20260425-143022__feature-vast-ai-basis__abc1234__seed0",
  "git_sha": "abc1234567890...",
  "git_branch": "feature/vast-ai-basis",
  "params_hash": "a1b2c3d4e5f6",
  "seed": 0,
  "vast_instance_id": 12345678,
  "gpu_name": "RTX_3090",
  "vast_offer_snapshot": {
    "dph_total": 0.13,
    "geolocation": "US",
    "reliability": 0.995,
    "cuda_max_good": 12.4
  },
  "command": "uv run --directory backend dvc repro train_imitation_case1",
  "weights_path": "data/output/models/imitation/case1/runs/<run_id>/best.pt",
  "train_metrics": {
    "epochs_run": 15,
    "best_epoch": 9,
    "best_val_loss": 3.84,
    "runtime_seconds": 612.3,
    "device": "cuda",
    "train_loss_history": [4.2, 4.0, ...],
    "val_loss_history": [4.1, 3.95, ...]
  },
  "local_eval_results": null,
  "status": "pushed",
  "created_at": "2026-04-25T14:30:22Z",
  "updated_at": "2026-04-25T14:42:14Z"
}
```

### DVC Tracking

- `data/output/` ディレクトリは **`.gitignore` に追加**。
- run dir 全体は **`dvc add data/output/models/imitation/case1/runs/<run_id>/`** で 1 個の `.dvc` ファイル化（onstart 末尾で実行）。または、shared `runs/` 全体を 1 つの `.dvc` で管理する案もあるが、複数 run を独立に push/pull するため **run dir 単位で `.dvc`** を推奨。
- DVC remote (S3) には自動同期。`dvc.lock` 内で `train_imitation_case1` stage の outs 自体は変えない（依然 `policy/weights.pt`）ため、stage repro の hash 影響なし。

### `params.yaml`

- 変更なし。`weights_out` パスは canonical path のまま。
- 将来 hyperparameter sweep を作る際は `params.yaml` の差し替え運用を維持し、`run.json` の `params_hash` で識別。

## Infrastructure Changes

### AWS / Terraform

- **変更なし**。既存 `infra/module/application/dvc_remote/` の IAM user (`orbit-wars-dev-dvc-user`) のキーをそのまま Vast にも渡す（要件 F8.4）。
- IAM ポリシーには `s3:DeleteObject` が含まれないため、Vast 側からの誤破壊リスクなし（versioning + bucket policy で多重防御済み）。

### Local Configuration

- `~/.aws/credentials` の `orbit-wars` profile をそのまま使用（dev/dvc-setup と同じ）。
- `VAST_API_KEY` は `backend/.env` に追加（`.env` は git ignore 済み）。`backend/.env.example` に `VAST_API_KEY=` 行を追加して documentation。
- `pyproject.toml` の dependencies に `vastai>=0.3.0` を追加（最新安定）。

## External Integrations

- **Vast.ai REST API**: `https://console.vast.ai/api/v0/`、`vastai` SDK 経由で呼ぶ。`Bearer` token (`VAST_API_KEY`) で認証。
- **AWS S3 (DVC remote)**: 既存。Vast 側でも env 経由 credentials で同じ bucket にアクセス。
- **GitHub (origin remote)**: `git clone https://github.com/<user>/orbit-wars.git` を Vast 側で実行。**public repo であれば認証不要**。private 化されている場合は HTTPS PAT or SSH key を Vast に追加する必要があり、要前提確認 (本リポジトリは現在 private と推定 → onstart で `GIT_PAT` env を使う設計か、deploy key 用意か検討)。

## File-Level Changes Summary

| File | Action | Notes |
|------|--------|-------|
| `backend/src/vast/__init__.py` | create | empty |
| `backend/src/vast/__main__.py` | create | `from .cli import app; app()` |
| `backend/src/vast/cli.py` | create | typer Typer |
| `backend/src/vast/offers.py` | create | search/format/pick |
| `backend/src/vast/instance.py` | create | render_onstart/create_instance |
| `backend/src/vast/run_meta.py` | create | RunMetadata + I/O |
| `backend/src/vast/cost.py` | create | aggregator |
| `backend/src/vast/auth.py` | create | aws/vast key load |
| `backend/src/vast/onstart.sh.tmpl` | create | bash template |
| `backend/tests/vast/*` | create | unit tests |
| `backend/pipeline/imitation/case1/training/train.py` | edit | device + run dir override |
| `backend/pyproject.toml` | edit | `vastai>=0.3.0` 追加、`packages` に `src/vast` 追加 |
| `backend/.env.example` | edit | `VAST_API_KEY=` 行追加 |
| `dev/vast-train` | create | bash thin wrapper |
| `dev/vast-pull` | create | bash thin wrapper |
| `dev/vast-promote` | create | bash thin wrapper |
| `dev/vast-cost-report` | create | bash thin wrapper |
| `.gitignore` | edit | `data/output/` 追加 |
| `params.yaml` | unchanged | – |
| `dvc.yaml` | unchanged | – |
| `infra/` | unchanged | – |

## Cross-Cutting Concerns

- **Logging**: 既存 `logger = logging.getLogger(__name__)` パターンを踏襲。`vast` パッケージ内も同様。Vast 側 onstart は `tee /var/log/onstart.log` でファイル化。
- **エラーハンドリング**: typer の `typer.Exit(code=N)` で外向き status code を返す。ローカル CLI は失敗ケースを actionable なメッセージで返す（"VAST_API_KEY not found in backend/.env" 等）。
- **テスト方針**: vast パッケージは vastai SDK 呼び出しを `unittest.mock` で stub。`onstart.sh.tmpl` の sed 置換は文字列比較で deterministic にテスト。`train.py` の GPU path は `pytest.mark.cuda` で skip 可能化（ローカル CI は CPU のみ実行）。
- **後方互換**: `params.yaml`, `dvc.yaml`, `policy/weights.pt`, Kaggle submit パスはすべて不変。既存 `weights_iter*.pt` は git に残置。
