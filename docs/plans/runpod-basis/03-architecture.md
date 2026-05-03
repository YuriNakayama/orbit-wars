# runpod-basis — Architecture Design

## Overall Diagram

```
┌──────────────── Local Machine ─────────────────────┐
│                                                    │
│  developer                                         │
│    │                                               │
│    ▼                                               │
│  dev/runpod train <sha> [--case ...] [--cloud-type]│
│    │                                               │
│    ▼                                               │
│  uv run -m runpod train ...     (typer CLI)        │
│    │   ※ 内部では import runpod as runpod_sdk      │
│    │      (パッケージ自身も runpod。同名衝突回避)   │
│    │                                               │
│    ├── auth.load_aws_creds("orbit-wars")           │
│    ├── auth.load_runpod_api_key()                  │
│    ├── offers.search_offers(sdk, cloud_type, ...)  │
│    │     │                                         │
│    │     ├── sdk.get_gpus()  ─ id 一覧             │
│    │     └── for each id: sdk.get_gpu(id) ─ price  │
│    ├── volumes.find_or_create(sdk, name, dc) ─┐    │
│    ├── offers.pick_offer (rich + IntPrompt)   │    │
│    ├── instance.render_onstart(...)           │    │
│    ├── instance.create_pod(sdk, gpu_type_id,  │    │
│    │       network_volume_id=...,             │    │
│    │       env={...}, docker_args=onstart)    │    │
│    │                                          │    │
│    │   AWS creds + RUNPOD_API_KEY (env)       │    │
│    └─────────────────────────────────► RunPod │    │
│                                  ┌────────────┴────┐│
│                                  │  Pod (GPU)      ││
│                                  │  /persist mount ││
│                                  │                 ││
│                                  │  onstart.sh:    ││
│                                  │  1. trap EXIT   ││
│                                  │  2. timeout BG  ││
│                                  │     (sleep 7200 ││
│                                  │      && stop)   ││
│                                  │  3. env persist ││
│                                  │  4. clone+co    ││
│                                  │  5. uv sync     ││
│                                  │  6. dvc pull    ││
│                                  │  7. preprocess  ││
│                                  │  8. python -m   ││
│                                  │     <TRAIN_MOD> ││
│                                  │  9. dvc add+push││
│                                  │ 10. git push    ││
│                                  │     <RUN>.dvc   ││
│                                  │ 11. runpodctl   ││
│                                  │     stop pod    ││
│                                  └────────┬────────┘│
│                                           │         │
│                                dvc push   │         │
│                                           ▼         │
│                              ┌─────────────────────┐│
│                              │  S3: orbit-wars-dvc ││
│                              │  (DVC remote)       ││
│                              └──────────┬──────────┘│
│                                         │           │
│  developer (later)                      │           │
│    │                                    │           │
│    ▼                                    │           │
│  dev/runpod pull <run_id> [--case ...]  │           │
│    └── dvc pull data/output/.../runs/<id>           │
│        ← run.json + best.pt + metrics.json         │
│                                                    │
│  Local evaluation (provider 不問、既存 evaluator)   │
│    └── 評価結果 JSON を作成                          │
│                                                    │
│  dev/runpod promote <run_id> [--eval-results PATH] │
│    └── cp run_dir/best.pt → policy/weights.pt      │
│        + dvc commit + git status 表示               │
│                                                    │
│  dev/runpod cost-report [--month YYYY-MM]          │
│    └── runs/*/run.json から runpod のみ集計         │
│        → docs/experiment/runpod_cost_report_*.md   │
│                                                    │
└────────────────────────────────────────────────────┘
```

## Directory Layout (new files only)

```
bot/
  src/
    runpod_io/                       # NEW package (SDK との衝突回避のため _io suffix)
      __init__.py                    # docstring に SDK alias 規約を明記
      __main__.py                    # `python -m runpod_io` entrypoint
      cli.py                         # typer Typer() with subcommands: train, pull, promote, cost-report, volume {list,search,create}
      offers.py                      # search_offers (get_gpus + get_gpu loop), Offer dataclass, pick_offer
      instance.py                    # render_onstart, create_pod, build_env_dict
      run_meta.py                    # vast.run_meta から RunMetadata を import + runpod_pod_id/runpod_offer_snapshot 用 helper
      cost.py                        # aggregate_runs (runpod のみ filter) + render_markdown
      volumes.py                     # network volume CRUD (REST/GraphQL 直叩き含む)
      auth.py                        # load_aws_creds (vast から共有 import) + load_runpod_api_key
      onstart.sh.tmpl                # bash template (Vast から差分のみ修正)
  tests/
    src/
      runpod_io/                     # NEW
        __init__.py
        test_auth.py
        test_cli.py
        test_cost.py
        test_instance.py
        test_offers.py
        test_volumes.py
        test_onstart_template.py     # vast 同等の bash -n smoke test

dev/
  runpod                             # NEW thin wrapper (CLI 名は短く `runpod`)

bot/
  .env.example                       # +RUNPOD_API_KEY=
  pyproject.toml                     # +runpod>=1.7.0, +packages: src/runpod_io

bot/pipeline/imitation/case{1,3,4}/training/
  train.py                           # MINOR EDIT: ORBIT_WARS_RUNPOD_POD_ID 検出ロジック追加
                                     # (Vast 既存パスは無変更、両 env 同時セットなら assert)
```

## SDK ⇔ パッケージ命名 (確定)

公式 SDK `runpod` と衝突するため、内部パッケージ名は **`runpod_io`** を採用 (ユーザ確定)。CLI 名はユーザ要件通り `dev/runpod` を維持し、起動コマンドは `python -m runpod_io` となる。

- **パッケージ命名の整合性**:
  - 内部 import: `from runpod_io.auth import load_runpod_api_key` (絶対) / `from .cli import app` (相対) の両方 OK。
  - SDK import: `import runpod as runpod_sdk` を各モジュールで採用。`runpod_io` パッケージは SDK のトップレベルと別名なので、Python の import システムが両方を独立に解決可能。
  - typer の wiring (`uv run python -m runpod_io train ...`) は `__main__.py` から `from runpod_io.cli import app; app()`。
- **CLI 体験**:
  - `dev/runpod train <sha>`: thin wrapper の中身は `exec uv run --directory bot python -m runpod_io "$@"`。
  - vast 基盤との対称性: `dev/vast` (= `python -m vast`) ⇔ `dev/runpod` (= `python -m runpod_io`)。
- **テスト命名**: `bot/tests/src/runpod_io/` 配下に置き、test ファイル名は vast の構造をミラー (`test_auth.py`, `test_cli.py`, `test_cost.py`, `test_instance.py`, `test_offers.py`, `test_run_meta.py` (run_meta は薄い wrapper)、`test_volumes.py`, `test_onstart_template.py`)。
- **mypy / ruff 設定**: `bot/pyproject.toml` の `[tool.mypy]` / `[tool.ruff]` の対象に `src/runpod_io` を追加。

## Backend Design

### `bot/src/runpod_io/cli.py` (typer)

```python
from __future__ import annotations

import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import runpod as runpod_sdk
import typer
from rich.console import Console

from .auth import (
    DEFAULT_AWS_PROFILE,
    AwsCreds,
    CredentialsError,
    load_aws_creds,
    load_runpod_api_key,
)
from .cost import aggregate_runs, parse_month, render_markdown
from .instance import (
    DEFAULT_DISK_GB,
    DEFAULT_IMAGE,
    build_env_dict,
    create_pod,
    render_onstart,
)
from .offers import Offer, pick_offer, search_offers
from .run_meta import generate_run_id, update_run_json
from .volumes import (
    create_volume as create_volume_fn,
    find_volume_by_name,
    list_volumes,
    pick_volume_offer,
    render_volume_offers,
    search_volume_offers,
)

app = typer.Typer(
    add_completion=False,
    help="RunPod-driven GPU training basis for Orbit Wars.",
    no_args_is_help=True,
)
console = Console()

DEFAULT_CASE = "case1"
DEFAULT_CLOUD_TYPE = "SECURE"
DEFAULT_COST_LIMIT_USD = 1.5
ESTIMATED_RUNTIME_HOURS = 0.5
DEFAULT_GPU_NAMES: tuple[str, ...] = (
    "NVIDIA GeForce RTX 3090",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA RTX A6000",
    "NVIDIA A100 80GB PCIe",
)
DEFAULT_VOLUME_NAME = "orbit_wars"
DEFAULT_MOUNT_PATH = "/persist"
DEFAULT_PORTS = "22/tcp,8888/http"

CASE_DEFAULTS: dict[str, dict[str, str]] = {
    # vast.cli の構造を完コピ (stage / train_module / config_arg / preprocess_cmd / canonical_weights)
    "case1": {...},
    "case3": {...},
    "case4": {...},
}


@app.command()
def train(
    commit_sha: str = typer.Argument(..., help="must be pushed to origin"),
    case: str = typer.Option(DEFAULT_CASE, "--case"),
    cloud_type: str = typer.Option(
        DEFAULT_CLOUD_TYPE, "--cloud-type", help="SECURE / COMMUNITY / ALL"
    ),
    gpu_names: list[str] = typer.Option(list(DEFAULT_GPU_NAMES), "--gpu-name"),
    max_dph: float = typer.Option(2.0, "--max-dph"),
    seed: int = typer.Option(0, "--seed"),
    label: str | None = typer.Option(None, "--label"),
    cost_limit_usd: float = typer.Option(DEFAULT_COST_LIMIT_USD, "--cost-limit"),
    aws_profile: str = typer.Option(DEFAULT_AWS_PROFILE, "--aws-profile"),
    image: str = typer.Option(DEFAULT_IMAGE, "--image"),
    disk_gb: int = typer.Option(DEFAULT_DISK_GB, "--disk-gb"),
    volume_id: str | None = typer.Option(None, "--volume-id"),
    volume_name: str = typer.Option(DEFAULT_VOLUME_NAME, "--volume-name"),
    mount_path: str = typer.Option(DEFAULT_MOUNT_PATH, "--mount-path"),
    auto_create_volume: bool = typer.Option(False, "--auto-create-volume"),
    volume_size_gb: int = typer.Option(15, "--volume-size"),
    data_center_id: str | None = typer.Option(None, "--data-center-id"),
) -> None:
    """Search RunPod GPUs, pick one, and launch GPU training for a given commit."""
    # 1. case + git 検証 (vast.cli.train と同じ)
    # 2. credentials load
    # 3. SDK 初期化: runpod_sdk.api_key = api_key
    # 4. volume 解決 (--volume-id 明示 / --volume-name 一致再利用 / --auto-create-volume)
    # 5. offers.search_offers(sdk=runpod_sdk, cloud_type=..., gpu_names=...,
    #                         max_dph=..., limit=10)
    # 6. pick_offer + estimated cost confirm
    # 7. run_id 生成
    # 8. render_onstart (9 placeholder)
    # 9. build_env_dict
    # 10. instance.create_pod(...)
    # 11. 起動メッセージ + monitor 案内


@app.command()
def pull(run_id: str, case: str = typer.Option(DEFAULT_CASE)) -> None:
    """`dvc pull` で run dir をローカルに取得し run.json を表示。vast.pull と同設計"""


@app.command()
def promote(
    run_id: str,
    case: str = typer.Option(DEFAULT_CASE),
    eval_results: Path | None = typer.Option(None, "--eval-results"),
) -> None:
    """candidate を canonical に昇格、dvc commit、status=adopted。vast.promote と同設計"""


@app.command("cost-report")
def cost_report_cmd(
    month: str | None = typer.Option(None, "--month"),
    case: str = typer.Option(DEFAULT_CASE),
) -> None:
    """runpod_offer_snapshot != null の run のみ集計 → markdown。vast.cost-report と分離"""


volume_app = typer.Typer(...)
app.add_typer(volume_app, name="volume")


@volume_app.command("list")
def volume_list_cmd() -> None: ...


@volume_app.command("search")
def volume_search_cmd(...) -> None: ...


@volume_app.command("create")
def volume_create_cmd(...) -> None: ...
```

### `bot/src/runpod_io/auth.py`

vast.auth と完全に同じ `load_aws_creds()` + RunPod 用の `load_runpod_api_key()` を追加。

```python
from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values

from .auth_aws import (  # = vast.auth から import or 同等関数を独立実装
    DEFAULT_AWS_PROFILE,
    DEFAULT_AWS_REGION,
    AwsCreds,
    CredentialsError,
    load_aws_creds,
)


def load_runpod_api_key(*, env_path: Path | None = None) -> str:
    """bot/.env または環境変数から RUNPOD_API_KEY を読む。"""
    if env_path is None:
        env_path = _default_env_path()
    if env_path.is_file():
        values = dotenv_values(env_path)
        api_key = values.get("RUNPOD_API_KEY")
        if api_key:
            return api_key.strip()
    fallback = os.environ.get("RUNPOD_API_KEY", "").strip()
    if fallback:
        return fallback
    raise CredentialsError(
        "RUNPOD_API_KEY not found. Add `RUNPOD_API_KEY=<your-key>` to "
        f"{env_path} or export it as an environment variable. "
        "Get a key from https://runpod.io/console/user/settings."
    )


def _default_env_path() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent / ".env"
    return Path(".env").resolve()
```

`load_aws_creds()` の重複を避けるため、**実装方針 (採用)**: `vast.auth` から再 import:

```python
# bot/src/runpod_io/auth.py
from vast.auth import (  # type: ignore[import-untyped]
    DEFAULT_AWS_PROFILE,
    DEFAULT_AWS_REGION,
    AwsCreds,
    CredentialsError,
    load_aws_creds,
)
```

これにより: (a) 重複ロジックなし、(b) vast 側を改修したら runpod 側も追従、(c) `auth_aws` の独立 module 化は将来 phase 2 で `bot/src/cloud/auth.py` に切り出す候補となる。

### `bot/src/runpod_io/offers.py`

Vast の `search_offers` (DSL query) と異なり、RunPod は `get_gpus()` + `get_gpu(id)` の 2 段階呼び出し。

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import runpod as runpod_sdk
from rich.console import Console
from rich.prompt import IntPrompt
from rich.table import Table


@dataclass(frozen=True)
class Offer:
    """RunPod GPU offer."""
    gpu_type_id: str
    display_name: str
    memory_gb: int
    secure_cloud: bool
    community_cloud: bool
    secure_price: float | None       # on-demand $/h on Secure
    community_price: float | None    # on-demand $/h on Community
    secure_spot_price: float | None
    community_spot_price: float | None
    cloud_type: Literal["SECURE", "COMMUNITY"]  # 選択された側
    dph_total: float                  # cloud_type に応じた選択価格
    data_center_id: str | None = None  # GPU 単位では null、pod 起動時に決定

    def to_snapshot(self) -> dict[str, Any]:
        return {...}


def search_offers(
    sdk: Any,
    *,
    gpu_names: list[str],
    cloud_type: Literal["SECURE", "COMMUNITY", "ALL"] = "SECURE",
    max_dph: float = 2.0,
    min_memory_gb: int = 16,
    limit: int = 10,
) -> list[Offer]:
    """get_gpus → 各 id を get_gpu で詳細取得 → cloud_type filter → dph asc。"""
    all_gpus = sdk.get_gpus()  # [{"id": "NVIDIA GeForce RTX 3090", ...}, ...]
    target_ids = {g["id"] for g in all_gpus if g["id"] in set(gpu_names)}
    candidates: list[Offer] = []
    for gpu_id in target_ids:
        detail = sdk.get_gpu(gpu_id)
        memory_gb = int(detail.get("memoryInGb", 0))
        if memory_gb < min_memory_gb:
            continue
        secure_price = detail.get("securePrice")
        community_price = detail.get("communityPrice")
        if cloud_type in ("SECURE", "ALL") and secure_price and detail.get("secureCloud"):
            if secure_price <= max_dph:
                candidates.append(_build_offer(detail, "SECURE", secure_price))
        if cloud_type in ("COMMUNITY", "ALL") and community_price and detail.get("communityCloud"):
            if community_price <= max_dph:
                candidates.append(_build_offer(detail, "COMMUNITY", community_price))
    candidates.sort(key=lambda o: o.dph_total)
    return candidates[:limit]


def format_table(offers: list[Offer]) -> Table: ...
def pick_offer(offers: list[Offer], *, console: Console | None = None) -> Offer: ...
```

### `bot/src/runpod_io/instance.py`

Vast の `instance.py` と placeholder + sanitize ロジックは同じ。差分は `create_pod` 呼び出しのみ。

```python
from __future__ import annotations

import os
import re
import shlex
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import runpod as runpod_sdk

# Placeholder/正規表現は vast.instance と同一
_VALID_VALUE = re.compile(r"^[A-Za-z0-9._\-/:]+$")
_VALID_CONFIG_ARG = re.compile(r"^(|--config [A-Za-z0-9._\-/]+)$")
_VALID_PREPROCESS_CMD = re.compile(
    r"^(|[A-Za-z0-9._\-/]+( --config [A-Za-z0-9._\-/]+)?)$"
)
_TEMPLATE_PLACEHOLDERS = (
    "<COMMIT_SHA>", "<RUN_ID>", "<STAGE>", "<BRANCH>", "<REPO_URL>",
    "<CASE>", "<TRAIN_MODULE>", "<CONFIG_ARG>", "<PREPROCESS_CMD>",
)

DEFAULT_IMAGE = os.environ.get(
    "ORBIT_WARS_RUNPOD_IMAGE",
    "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
)
DEFAULT_DISK_GB = 40


class TemplateError(ValueError): ...


def render_onstart(
    template_path: Path,
    *,
    commit_sha: str, run_id: str, stage: str, branch: str, repo_url: str,
    case: str = "case1",
    train_module: str = "pipeline.imitation.case1.training.train",
    config_arg: str = "",
    preprocess_cmd: str = "",
) -> str:
    """vast.instance.render_onstart と同じ実装。"""
    ...


def build_env_dict(env: Mapping[str, str]) -> dict[str, str]:
    """vast.instance.build_env_dict と同じ。env 名を ^[A-Z_][A-Z0-9_]*$ で validate。"""
    ...


def create_pod(
    sdk: Any,
    *,
    name: str,
    gpu_type_id: str,
    cloud_type: str,
    onstart_script: str,
    env: Mapping[str, str],
    image: str = DEFAULT_IMAGE,
    container_disk_gb: int = DEFAULT_DISK_GB,
    network_volume_id: str | None = None,
    volume_mount_path: str = "/persist",
    ports: str = "22/tcp,8888/http",
) -> str:
    """runpod.create_pod ラッパ. pod_id を返す。"""
    docker_args = f"bash -c {shlex.quote(onstart_script)}"
    response = sdk.create_pod(
        name=name,
        image_name=image,
        gpu_type_id=gpu_type_id,
        cloud_type=cloud_type,
        gpu_count=1,
        container_disk_in_gb=container_disk_gb,
        volume_in_gb=0,
        network_volume_id=network_volume_id,
        volume_mount_path=volume_mount_path,
        docker_args=docker_args,
        env=dict(env),
        support_public_ip=True,
        start_ssh=True,
        ports=ports,
    )
    if not isinstance(response, Mapping):
        raise RuntimeError(f"unexpected create_pod response: {type(response).__name__}")
    pod_id = response.get("id") or response.get("podId")
    if pod_id is None:
        raise RuntimeError(f"create_pod response missing pod id: keys={list(response)}")
    return str(pod_id)
```

### `bot/src/runpod_io/onstart.sh.tmpl`

vast.onstart.sh.tmpl から差分のみ:

1. **冒頭の自殺タイムアウト保険** (新規):
   ```bash
   ( sleep 7200 && runpodctl stop pod "$INSTANCE_ID" 2>/dev/null || true ) &
   TIMEOUT_GUARD_PID=$!
   ```

2. **`INSTANCE_ID="${RUNPOD_POD_ID:-unknown}"`** に変更 (vast の `VAST_CONTAINERLABEL` から)。

3. **`cleanup_destroy()`** 関数:
   ```bash
   cleanup_destroy() {
     local exit_code=$?
     kill "$TIMEOUT_GUARD_PID" 2>/dev/null || true
     echo "[onstart] cleanup status=${exit_code}"
     if [ "${exit_code}" -ne 0 ]; then
       echo "[onstart] failure: leaving pod up for manual inspection"
       exit "${exit_code}"
     fi
     # 成功時: runpodctl で自殺
     if command -v runpodctl >/dev/null 2>&1; then
       echo "[onstart] self-destroy pod=${INSTANCE_ID}"
       runpodctl stop pod "${INSTANCE_ID}" || echo "[onstart] self-destroy failed"
     else
       echo "[onstart] runpodctl missing; cannot self-destroy. Stop manually."
     fi
   }
   trap cleanup_destroy EXIT
   ```

4. **train env**: `ORBIT_WARS_VAST_INSTANCE_ID` → `ORBIT_WARS_RUNPOD_POD_ID`:
   ```bash
   ORBIT_WARS_RUN_DIR="data/output/models/imitation/<CASE>/runs/<RUN_ID>" \
     ORBIT_WARS_RUN_ID="<RUN_ID>" \
     ORBIT_WARS_GIT_SHA="<COMMIT_SHA>" \
     ORBIT_WARS_GIT_BRANCH="<BRANCH>" \
     ORBIT_WARS_RUNPOD_POD_ID="${INSTANCE_ID}" \
     ORBIT_WARS_COMMAND="uv run --directory bot python -m <TRAIN_MODULE> <CONFIG_ARG>" \
     uv run --directory bot python -m <TRAIN_MODULE> <CONFIG_ARG>
   ```

5. **`uv pip install vastai` の行は削除** (runpodctl は image に pre-install されている)。

6. **bot コミット (`<RUN_ID>.dvc` を origin に push)** は vast 同設計のまま、user.email/name を `runpod-bot@orbit-wars.local` に変更。

### `bot/src/runpod_io/run_meta.py`

vast.run_meta から `RunMetadata` を re-export し、RunPod 用の helper を追加。

```python
from __future__ import annotations

from typing import Any

from vast.run_meta import (  # type: ignore[import-untyped]
    SCHEMA_VERSION,
    RUN_ID_PATTERN,
    RunMetadata,
    RunStatus,
    generate_run_id,
    hash_params,
    read_run_json,
    update_run_json,
    write_run_json,
)


def build_runpod_offer_snapshot(
    *, gpu_type_id: str, display_name: str, memory_gb: int,
    cloud_type: str, secure_cloud: bool, community_cloud: bool,
    dph_total: float, data_center_id: str | None,
) -> dict[str, Any]:
    """run.json.runpod_offer_snapshot に格納する dict を組み立てる."""
    return {
        "gpu_type_id": gpu_type_id,
        "display_name": display_name,
        "memory_gb": memory_gb,
        "cloud_type": cloud_type,
        "secure_cloud": secure_cloud,
        "community_cloud": community_cloud,
        "dph_total": dph_total,
        "data_center_id": data_center_id,
    }
```

`vast.run_meta.RunMetadata` の dataclass フィールドに `runpod_pod_id: int | None = None` と `runpod_offer_snapshot: dict[str, Any] | None = None` を **追加** する PR を Step 4 (要件の F3) で行う:

```python
# bot/src/vast/run_meta.py (修正後の dataclass)
@dataclass
class RunMetadata:
    schema_version: int = SCHEMA_VERSION  # = 1 のまま
    run_id: str = ""
    git_sha: str = ""
    git_branch: str = ""
    params_hash: str = ""
    seed: int = 0
    vast_instance_id: int | None = None
    runpod_pod_id: str | None = None              # NEW (Optional)
    gpu_name: str | None = None
    vast_offer_snapshot: dict[str, Any] | None = None
    runpod_offer_snapshot: dict[str, Any] | None = None  # NEW (Optional)
    command: str = ""
    weights_path: str = ""
    train_metrics: dict[str, Any] = field(default_factory=dict)
    local_eval_results: dict[str, Any] | None = None
    status: RunStatus = "running"
    created_at: str = ""
    updated_at: str = ""
```

`schema_version=1` を維持し、後方互換のために `RunMetadata(**data)` の `data` から欠如している field がある場合 (古い vast run.json) は default で埋まる (dataclass default 機構)。

### `bot/src/runpod_io/cost.py`

vast.cost と同じ structure。差分は: (a) `runpod_offer_snapshot` を読む、(b) `runpod_offer_snapshot is None` なら skip (vast の run を除外)、(c) markdown ヘッダが `# RunPod Cost Report — <month>`、(d) 出力 path が `docs/experiment/runpod_cost_report_<YYYY-MM>.md`。

```python
from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunCost:
    run_id: str
    git_sha: str
    gpu_type_id: str | None
    cloud_type: str | None
    dph_total: float | None
    runtime_seconds: float
    cost_usd: float
    status: str
    created_at: str


@dataclass(frozen=True)
class CostReport:
    month: str
    runs: tuple[RunCost, ...]

    @property
    def total_cost_usd(self) -> float: ...
    @property
    def adopted_count(self) -> int: ...
    @property
    def average_cost_usd(self) -> float: ...


def _load_run(run_json: Path) -> RunCost | None:
    """runpod_offer_snapshot が null なら None を返して skip。"""
    try:
        data: dict[str, Any] = json.loads(run_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    snapshot = data.get("runpod_offer_snapshot")
    if snapshot is None:
        return None  # vast run はここでスキップ
    metrics = data.get("train_metrics") or {}
    runtime = float(metrics.get("runtime_seconds", 0.0))
    dph = snapshot.get("dph_total")
    cost = (float(dph) * runtime / 3600.0) if dph else 0.0
    return RunCost(...)


def aggregate_runs(runs_root: Path, month: str | None = None) -> CostReport: ...
def render_markdown(report: CostReport) -> str: ...
def parse_month(value: str | None) -> str | None: ...
def iter_run_dirs(runs_root: Path) -> Iterable[Path]: ...
```

### `bot/src/runpod_io/volumes.py`

RunPod の network volume 操作は SDK が薄い。GraphQL 直叩き or REST API:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import runpod as runpod_sdk
from runpod.api.graphql import run_graphql_query


@dataclass(frozen=True)
class VolumeOffer:
    data_center_id: str
    storage_per_hour_per_gb: float  # $/GB/hour or per month converted
    available_size_gb: int


@dataclass(frozen=True)
class Volume:
    id: str
    name: str
    size_gb: int
    data_center_id: str
    storage_per_hour: float


def list_volumes(sdk: Any) -> list[Volume]:
    """所有 network volumes を一覧。GraphQL queryUserNetworkVolumes 経由。"""
    query = """
    query MyVolumes {
      myself {
        networkVolumes { id name size dataCenterId }
      }
    }
    """
    response = run_graphql_query(query)
    raw = response.get("data", {}).get("myself", {}).get("networkVolumes", [])
    return [Volume(...) for r in raw]


def search_volume_offers(
    sdk: Any, *, min_size_gb: int = 15, data_center_id: str | None = None,
) -> list[VolumeOffer]:
    """available data center を返す。価格は固定 ($0.07/GB/月)、availability 確認用。"""
    # 簡略化: data center の固定リストを返す (公式 docs 由来)
    centers = ["US-KS-2", "US-CA-1", "EU-RO-1", ...]
    if data_center_id:
        centers = [c for c in centers if c == data_center_id]
    return [VolumeOffer(data_center_id=c, storage_per_hour_per_gb=0.07/720, available_size_gb=4096) for c in centers]


def create_volume(sdk: Any, *, name: str, size_gb: int, data_center_id: str) -> str:
    """新規 volume 作成. id を返す。"""
    mutation = """
    mutation CreateVolume($input: NetworkVolumeInput!) {
      createNetworkVolume(input: $input) { id name size }
    }
    """
    variables = {"input": {"name": name, "size": size_gb, "dataCenterId": data_center_id}}
    response = run_graphql_query(mutation, variables=variables)
    return str(response["data"]["createNetworkVolume"]["id"])


def find_volume_by_name(volumes: list[Volume], name: str) -> Volume | None:
    matches = [v for v in volumes if v.name == name]
    if not matches:
        return None
    return matches[0]


def render_volume_offers(offers: list[VolumeOffer], console: Any | None = None) -> None: ...
def pick_volume_offer(offers: list[VolumeOffer], ...) -> VolumeOffer: ...
def validate_volume_name(name: str) -> None: ...
```

### `train.py` 改修箇所 (`bot/pipeline/imitation/case{1,3,4}/training/train.py`)

vast 基盤の改修済み箇所を踏まえ、最小変更:

```python
# (改修前: vast 専用)
vast_instance_id = os.environ.get("ORBIT_WARS_VAST_INSTANCE_ID")
if vast_instance_id:
    vast_instance_id = int(vast_instance_id)
else:
    vast_instance_id = None

# (改修後: 両 provider 対応)
vast_instance_id_env = os.environ.get("ORBIT_WARS_VAST_INSTANCE_ID")
runpod_pod_id_env = os.environ.get("ORBIT_WARS_RUNPOD_POD_ID")

# 両方 set はおかしい
if vast_instance_id_env and runpod_pod_id_env:
    raise RuntimeError(
        "Both ORBIT_WARS_VAST_INSTANCE_ID and ORBIT_WARS_RUNPOD_POD_ID are set. "
        "Only one provider should be active per run."
    )

vast_instance_id = int(vast_instance_id_env) if vast_instance_id_env else None
runpod_pod_id = runpod_pod_id_env if runpod_pod_id_env else None

meta = RunMetadata(
    ...
    vast_instance_id=vast_instance_id,
    runpod_pod_id=runpod_pod_id,
    runpod_offer_snapshot=None,  # onstart 経由では埋めない (cli.py の create_pod 後に append が必要なら追加実装)
    ...
)
```

`runpod_offer_snapshot` の埋め方:
- **オプション A** (採用): onstart スクリプト内で env `ORBIT_WARS_RUNPOD_OFFER_SNAPSHOT` (JSON 文字列) を渡し、train.py で `json.loads` して `runpod_offer_snapshot` に格納。
- **オプション B**: `dev/runpod pull` 後にローカルで `update_run_json` してマージ。複雑なので採用しない。

オプション A: `cli.py` で `runpod_offer_snapshot` を JSON 化して env に注入:
```python
env["ORBIT_WARS_RUNPOD_OFFER_SNAPSHOT"] = json.dumps(chosen.to_snapshot())
```
train.py 側:
```python
snapshot_env = os.environ.get("ORBIT_WARS_RUNPOD_OFFER_SNAPSHOT")
runpod_offer_snapshot = json.loads(snapshot_env) if snapshot_env else None
```

## Data Model

### `run.json` schema (v1, RunPod 用 fields 追加版)

```json
{
  "schema_version": 1,
  "run_id": "20260502-143022__feature-runpod-basis__abc1234__seed0",
  "git_sha": "abc1234567890...",
  "git_branch": "feature/runpod-basis",
  "params_hash": "a1b2c3d4e5f6",
  "seed": 0,
  "vast_instance_id": null,
  "runpod_pod_id": "abcd1234ef",
  "gpu_name": "NVIDIA GeForce RTX 3090",
  "vast_offer_snapshot": null,
  "runpod_offer_snapshot": {
    "gpu_type_id": "NVIDIA GeForce RTX 3090",
    "display_name": "RTX 3090",
    "memory_gb": 24,
    "cloud_type": "SECURE",
    "secure_cloud": true,
    "community_cloud": true,
    "dph_total": 0.43,
    "data_center_id": "US-KS-2"
  },
  "command": "uv run --directory bot python -m pipeline.imitation.case1.training.train",
  "weights_path": "data/output/models/imitation/case1/runs/<run_id>/best.pt",
  "train_metrics": {
    "epochs_run": 15,
    "best_epoch": 9,
    "best_val_loss": 3.84,
    "runtime_seconds": 612.3,
    "device": "cuda",
    "train_loss_history": [...],
    "val_loss_history": [...]
  },
  "local_eval_results": null,
  "status": "pushed",
  "created_at": "2026-05-02T14:30:22Z",
  "updated_at": "2026-05-02T14:42:14Z"
}
```

### DVC Tracking

vast 基盤と完全に同じ。`runs/<run_id>/` 単位で `dvc add` → `.dvc` ファイル → `git push` で origin 反映 → `dvc push` で S3 反映。

### `params.yaml` / `dvc.yaml`

変更なし。両基盤共有。

## Infrastructure Changes

### AWS / Terraform

変更なし。`infra/module/application/dvc_remote/` の `orbit-wars-dev-dvc-user` を継続再利用。

### Local Configuration

- `bot/.env` に `RUNPOD_API_KEY=...` 追加 (vast の `VAST_API_KEY` と並列)。
- `~/.aws/credentials` の `orbit-wars` profile はそのまま使用。
- `pyproject.toml`:
  ```toml
  dependencies = [
    "vastai>=0.3.0",
    "runpod>=1.7.0",  # NEW
    ...
  ]
  ```
- `bot/.env.example`:
  ```
  VAST_API_KEY=your-vast-api-key
  RUNPOD_API_KEY=your-runpod-api-key  # NEW
  ```

## External Integrations

- **RunPod GraphQL API**: `https://api.runpod.io/graphql`、`runpod` SDK 経由で呼ぶ。`Bearer` token (`RUNPOD_API_KEY`)。
- **AWS S3 (DVC remote)**: 既存。両基盤で共通の bucket。
- **GitHub (origin remote)**: clone 元 (両基盤で同じ私的リポジトリ)。private 化されている場合は `GIT_PAT` env で認証。

## File-Level Changes Summary

| File | Action | Notes |
|------|--------|-------|
| `bot/src/runpod_io/__init__.py` | create | docstring に SDK alias 規約を明記 |
| `bot/src/runpod_io/__main__.py` | create | `from .cli import app; app()` |
| `bot/src/runpod_io/cli.py` | create | typer Typer (train/pull/promote/cost-report/volume {list,search,create}) |
| `bot/src/runpod_io/auth.py` | create | `from vast.auth import load_aws_creds, ...` + `load_runpod_api_key()` |
| `bot/src/runpod_io/offers.py` | create | `Offer` + `search_offers` (get_gpus + get_gpu loop) |
| `bot/src/runpod_io/instance.py` | create | `render_onstart` + `create_pod` ラッパ |
| `bot/src/runpod_io/run_meta.py` | create | `from vast.run_meta import ...` + `build_runpod_offer_snapshot` |
| `bot/src/runpod_io/cost.py` | create | `aggregate_runs` (runpod_offer_snapshot のみ) |
| `bot/src/runpod_io/volumes.py` | create | GraphQL 直叩きで CRUD |
| `bot/src/runpod_io/onstart.sh.tmpl` | create | bash template (vast から差分のみ修正) |
| `bot/tests/src/runpod_io/*.py` | create | unit tests (vast tests を mirror) |
| `bot/src/vast/run_meta.py` | edit | `RunMetadata` に `runpod_pod_id` + `runpod_offer_snapshot` 追加 |
| `bot/pipeline/imitation/case1/training/train.py` | edit | `ORBIT_WARS_RUNPOD_POD_ID` + `ORBIT_WARS_RUNPOD_OFFER_SNAPSHOT` 検出 |
| `bot/pipeline/imitation/case3/training/train.py` | edit | 同上 |
| `bot/pipeline/imitation/case4/training/train.py` | edit | 同上 |
| `bot/pyproject.toml` | edit | `runpod>=1.7.0`, packages に `src/runpod_io` 追加 |
| `bot/.env.example` | edit | `RUNPOD_API_KEY=` 行追加 |
| `dev/runpod` | create | bash thin wrapper: `exec uv run --directory bot python -m runpod_io "$@"` |
| `.gitignore` | unchanged | `data/output/` は既に ignored |
| `dvc.yaml` | unchanged | – |
| `infra/` | unchanged | – |

## Cross-Cutting Concerns

- **Logging**: 既存パターン (`logger = logging.getLogger(__name__)`)。pod 側は `tee /var/log/onstart.log`。
- **エラーハンドリング**: `typer.Exit(code=N)` + actionable な `CredentialsError`。
- **テスト**: `runpod` SDK は `unittest.mock` で stub。vast tests と同パターン。
- **後方互換**:
  - `bot/src/vast/run_meta.RunMetadata` への field 追加は default=None なので vast 既存 run.json は変わらず読める。
  - `train.py` の env 検出は vast / runpod / どちらでもない (ローカル) の 3 ケースを動作。
- **mypy**: `runpod>=1.7.0` の型 stub が無い場合 `# type: ignore[import-untyped]` で凌ぐ (vast でも同等)。

## Package Naming Decision (確定)

ユーザ確定: **パッケージ名は `runpod_io`、CLI 名は `dev/runpod`**。

- 内部パッケージ: `bot/src/runpod_io/`
- entry point: `python -m runpod_io <subcommand>`
- thin wrapper: `dev/runpod` (中身は `exec uv run --directory bot python -m runpod_io "$@"`)
- SDK 参照: 各モジュール先頭で `import runpod as runpod_sdk`。`runpod_io` パッケージは SDK のトップレベル `runpod` と別名なので衝突しない。
- 検証: `python -c "import runpod_io; import runpod"` の両方が成功することを Step 6 (Implementation) の最初に CI / smoke test で確認。
