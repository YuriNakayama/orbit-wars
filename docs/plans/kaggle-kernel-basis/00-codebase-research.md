# kaggle-kernel-basis — Codebase Research

本ドキュメントは Vast.ai 基盤 (`bot/src/vast/`) / RunPod 基盤 (`bot/src/runpod_io/`) を踏襲して Kaggle Kernel 基盤を作るための、既存実装の詳細分析。Kaggle Kernel 基盤は両基盤と coexist する mirror 実装になる前提で、再利用ポイントと差分を明示する。

## Deep Codebase Analysis

### Vast.ai / RunPod CLI パッケージ

詳細な分析は `docs/plans/runpod-basis/00-codebase-research.md` に既出。本基盤の観点で重要な再利用ポイントは:

- **`bot/src/vast/run_meta.py`** — `RunMetadata` dataclass (`schema_version=1`), `generate_run_id()`, `hash_params()`, `write_run_json` / `read_run_json` / `update_run_json` を **そのまま共有**。`kaggle_kernel_meta` field を後方互換で追加する (Vast 既存 RunMetadata.runpod_* 拡張パターン踏襲)。
- **`bot/src/runpod_io/cli/app.py`** — Typer サブコマンド構成 (`train` / `pull` / `promote` / `cost-report` / `ps` / `status` / `logs` / `watch`) を mirror する。RunPod の `volume` / `tail` / `stock` は Kaggle に該当機能がないため skip。
- **`bot/src/runpod_io/artifacts/run_meta.py`** — 提案: 共通の `promote_to_canonical(run_id, case)` 実装を Kaggle Kernel からも呼ぶ。RunPod 側に既に存在するため、本基盤からは `from runpod_io.artifacts.run_meta import promote_to_canonical` で import する。
- **`bot/pipeline/imitation/case{1,3,4,8,9}/training/train.py`** — `ORBIT_WARS_VAST_INSTANCE_ID` / `ORBIT_WARS_RUNPOD_POD_ID` env 検出ブロックに `ORBIT_WARS_KAGGLE_KERNEL_SLUG` 検出を追加。**三 provider 同時セットは RuntimeError**。

### Vast / RunPod との差分まとめ

| Layer | Vast | RunPod | Kaggle Kernel |
|-------|------|--------|---------------|
| Auth helper | `vast.auth.load_vast_api_key()` | `runpod_io.auth.load_runpod_api_key()` | `kaggle_kernel.auth.load_kaggle_creds()` — `KAGGLE_USERNAME` + `KAGGLE_KEY` (env → bot/.env → ~/.kaggle/kaggle.json の 3 段 fallback) |
| Offer 検索 | `vast.offers.search_offers(sdk)` | `runpod_io.runpod.gpus.search_gpus(sdk)` | **なし** (Kaggle に GPU marketplace なし、accelerator enum 二択) |
| ノード起動 | `sdk.create_instance(...)` | `sdk.create_pod(...)` | `kaggle.api.kernels_push_cli(...)` (notebook の slug + metadata 経由) |
| Onstart | `onstart.sh.tmpl` の bash | `onstart.sh.tmpl` の bash | **notebook の cell** (Python セル、kaggle.api 経由で render) |
| データ配送 | `git clone` + `dvc pull` | `git clone` + `dvc pull` | **bot/ snapshot を Kaggle Dataset として upload** + notebook の `Add Data` 参照 |
| Self-destroy | `vastai destroy instance` | `runpodctl stop pod` | **不要** (kernel は run 完了で自動停止) |
| Artifact 出力 | onstart で `dvc push` → ローカル `dvc pull` | 同左 | **`/kaggle/working/runs/<run_id>/` に保存** → ローカルで `kaggle kernels output` → `dvc add` |
| Cost 計算 | `dph_total × runtime` | 同左 | **無料、`free_gpu_hours_used` のみ集計** |
| Volume | network volume CRUD | secure 専用 volume | **なし** (Dataset 全体が暗黙的に volume の役割) |

### `bot/pipeline/imitation/case<N>/training/train.py` の env 解決ブロック

既存 (case1 を例に):

```python
vast_id = os.environ.get("ORBIT_WARS_VAST_INSTANCE_ID")
rp_id = os.environ.get("ORBIT_WARS_RUNPOD_POD_ID")
if vast_id and rp_id:
    raise RuntimeError("Both ORBIT_WARS_VAST_INSTANCE_ID and ORBIT_WARS_RUNPOD_POD_ID are set. ...")
```

このブロックに第三 env を追加:

```python
kk_slug = os.environ.get("ORBIT_WARS_KAGGLE_KERNEL_SLUG")
active = [bool(vast_id), bool(rp_id), bool(kk_slug)]
if sum(active) > 1:
    raise RuntimeError("Multiple provider env vars are set simultaneously. ...")
```

そして `RunMetadata.kaggle_kernel_meta` を `ORBIT_WARS_KAGGLE_KERNEL_META` env (JSON 文字列) から `json.loads()` で展開する。

### `RunMetadata` schema 拡張

`bot/src/vast/run_meta.py` の `RunMetadata` dataclass に optional field を追加:

```python
@dataclass(frozen=True)
class RunMetadata:
    schema_version: int = 1
    # ... 既存 vast_* / runpod_* fields ...
    kaggle_kernel_meta: dict[str, Any] | None = None
```

`kaggle_kernel_meta` の中身 (例):
```json
{
  "kernel_slug": "<user>/orbit-wars-case1-20260520",
  "kernel_version": 3,
  "dataset_slug": "<user>/orbit-wars-bot",
  "dataset_version": "v17",
  "accelerator": "gpu-t4x2",
  "runtime_seconds": 1820,
  "internet_enabled": true,
  "free_gpu_hours_remaining_at_start": 24.5
}
```

`schema_version` は 1 を維持 (新フィールドは optional、既存読み込みに影響なし)。

### `dev/` wrappers のパターン

```bash
# dev/runpod
exec uv run --directory bot python -m runpod_io "$@"
# dev/vast
exec uv run --directory bot python -m vast "$@"
# dev/kaggle-kernel (新規)
exec uv run --directory bot python -m kaggle_kernel "$@"
```

3 行で済むため新規実装は trivial。

## Implementation Plan Mapping

| 既存資産 | Kaggle Kernel 基盤での扱い |
|---------|---------------------------|
| `bot/src/vast/run_meta.RunMetadata` | **拡張** (kaggle_kernel_meta field 追加、後方互換) |
| `bot/src/runpod_io/artifacts/run_meta.promote_to_canonical` | **共有 import** (Kaggle Kernel CLI から再利用) |
| `bot/src/runpod_io/artifacts/cost.aggregate_runs` | **参考にして別実装** (free-hour 集計に置換) |
| `bot/src/vast/onstart.sh.tmpl` | **不要** (Kaggle は notebook cell ベース) |
| `bot/pipeline/imitation/case*/training/train.py` env 解決 | **パッチ** (`ORBIT_WARS_KAGGLE_*` 検出追加) |
| `dev/runpod` / `dev/vast` shell wrapper | **mirror** (3 行で `dev/kaggle-kernel` 新設) |
| `.claude/rules/command.md` の RunPod 章 | **mirror** (Kaggle Kernel 章を末尾に追加) |

## 公式 Kaggle Python API の最小利用面

`kaggle` Python package (v1.6+) で本基盤が使う API は:

- `KaggleApi().authenticate()` — `KAGGLE_USERNAME` + `KAGGLE_KEY` env or `~/.kaggle/kaggle.json` から認証。
- `KaggleApi().dataset_create_new(folder, ...)` — 初回 dataset 作成。
- `KaggleApi().dataset_create_version(folder, version_notes, ...)` — 既存 dataset の version up。
- `KaggleApi().dataset_status(slug)` — dataset の processing 完了確認。
- `KaggleApi().kernels_push_cli(folder)` — notebook + metadata json を push して run。
- `KaggleApi().kernels_status(slug)` — `{"status": "queued|running|complete|error", "failureMessage": "..."}` を返す。
- `KaggleApi().kernels_output(slug, path)` — `/kaggle/working/` の中身をローカル path に download。
- `KaggleApi().kernels_list(user=..., page_size=...)` — 過去 kernel 一覧 (cost-report 用)。

これらは全て **HTTPS over kaggle.com**、API key はアカウント単位のスコープ。
