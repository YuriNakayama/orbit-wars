"""1 つの run の "現在の状態 + 成果物" を 1 dataclass にまとめる集約器。

`dev/runpod summary <run_id>` の本体。複数の source (launch.json / run.json /
S3 marker / S3 artifacts / DVC meta) を merge して以下を返す:

- 現在の status (running / succeeded / failed / stalled / unknown)
- last_step + 何秒前から進捗なし
- 推定コスト (実行時間 × dph_total)
- 成果物の存在状況 (local / S3 artifacts / S3 DVC meta の 3 列)
- final_train_loss / final_val_loss / epochs_completed (run.json から)

外部副作用 (S3, SDK) は呼び出し側で stub できるよう module 引数で受ける。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from runpod_io import progress as progress_mod
from runpod_io.launch import read_launch_json


@dataclass(frozen=True)
class ArtifactStatus:
    """1 ファイルの存在状況を 3 経路で確認した結果。"""

    name: str
    in_run_dir: bool  # ローカル run_dir に存在
    in_s3_artifacts: bool  # s3://.../runpod_artifacts/<run>/<name>


@dataclass(frozen=True)
class RunSummary:
    """1 run の集約状態。"""

    run_id: str
    pod_id: str | None
    status: str  # "succeeded" / "failed" / "stalled" / "running" / "unknown"
    last_step: str | None
    last_step_age_seconds: int | None
    elapsed_seconds: int | None
    estimated_cost_usd: float | None
    artifacts: tuple[ArtifactStatus, ...]
    final_train_loss: float | None
    final_val_loss: float | None
    epochs_completed: int | None
    dvc_meta_committed: bool


def _safe_load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(data, dict):
        return data
    return None


def _classify_status_from_markers(steps: list[str]) -> str:
    """marker 列から `succeeded`/`failed`/`running` のいずれかを返す。"""
    if "99_done" in steps:
        return "succeeded"
    cleanup_failures = [
        s for s in steps if s.startswith("90_cleanup_exit_") and not s.endswith("_0")
    ]
    if cleanup_failures:
        return "failed"
    return "running"


def build_summary(
    run_id: str,
    *,
    repo_root: Path,
    case: str,
    aws_profile: str | None = None,
    progress_module: Any = progress_mod,
) -> RunSummary:
    """主要な source を merge して RunSummary を返す。"""
    relative = Path(f"data/output/models/imitation/{case}/runs/{run_id}")
    run_dir = repo_root / relative

    # 1) launch.json
    pod_id: str | None = None
    dph_total: float | None = None
    try:
        launch = read_launch_json(run_dir)
        pod_id = str(launch.get("pod_id") or "") or None
        dph_total = launch.get("dph_total")
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    # 2) S3 markers
    try:
        markers = progress_module.list_markers(run_id, profile=aws_profile)
    except Exception:  # noqa: BLE001
        markers = []
    steps = [m.step for m in markers]
    last_step: str | None = steps[-1] if steps else None
    status = _classify_status_from_markers(steps)
    last_step_age_seconds: int | None = None
    elapsed_seconds: int | None = None
    if markers:
        from datetime import UTC, datetime

        try:
            first_ts = datetime.fromisoformat(
                markers[0].timestamp.replace("Z", "+00:00")
            )
            last_ts = datetime.fromisoformat(
                markers[-1].timestamp.replace("Z", "+00:00")
            )
            now = datetime.now(UTC)
            elapsed_seconds = int((last_ts - first_ts).total_seconds())
            last_step_age_seconds = int((now - last_ts).total_seconds())
        except ValueError:
            pass

    # 3) run.json (local or s3 artifacts)
    run_json = _safe_load_json(run_dir / "run.json")
    final_train_loss: float | None = None
    final_val_loss: float | None = None
    epochs_completed: int | None = None
    if run_json is not None:
        metrics = run_json.get("train_metrics") or {}
        if isinstance(metrics, dict):
            tl = metrics.get("train_loss_history") or []
            vl = metrics.get("val_loss_history") or []
            if isinstance(tl, list) and tl:
                try:
                    final_train_loss = float(tl[-1])
                except (TypeError, ValueError):
                    pass
            if isinstance(vl, list) and vl:
                try:
                    final_val_loss = float(vl[-1])
                except (TypeError, ValueError):
                    pass
            ec = metrics.get("epochs_run")
            if isinstance(ec, int):
                epochs_completed = ec

    # 4) artifact 存在確認
    try:
        s3_files = set(progress_module.list_artifacts(run_id, profile=aws_profile))
    except Exception:  # noqa: BLE001
        s3_files = set()

    artifact_names = ("best.pt", "metrics.json", "run.json", "onstart.log")
    artifacts = tuple(
        ArtifactStatus(
            name=name,
            in_run_dir=(run_dir / name).is_file(),
            in_s3_artifacts=(name in s3_files),
        )
        for name in artifact_names
    )

    # 5) DVC meta が origin に上がっているか
    dvc_meta = repo_root / f"{relative}.dvc"
    dvc_meta_committed = dvc_meta.is_file()

    # 6) 推定コスト
    estimated_cost_usd: float | None = None
    if dph_total is not None and elapsed_seconds is not None:
        try:
            estimated_cost_usd = round(float(dph_total) * (elapsed_seconds / 3600.0), 4)
        except (TypeError, ValueError):
            pass

    return RunSummary(
        run_id=run_id,
        pod_id=pod_id,
        status=status,
        last_step=last_step,
        last_step_age_seconds=last_step_age_seconds,
        elapsed_seconds=elapsed_seconds,
        estimated_cost_usd=estimated_cost_usd,
        artifacts=artifacts,
        final_train_loss=final_train_loss,
        final_val_loss=final_val_loss,
        epochs_completed=epochs_completed,
        dvc_meta_committed=dvc_meta_committed,
    )


__all__ = ["ArtifactStatus", "RunSummary", "build_summary"]
