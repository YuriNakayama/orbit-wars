"""RunPod 操作の Typer CLI: train / pull / promote / cost-report / volume。

`dev/runpod` thin wrapper から `python -m runpod_io <cmd>` で呼ばれる前提。
SDK との衝突回避のため、SDK は `import runpod as runpod_sdk` の alias 経由
で読む規約 (パッケージ名は `runpod_io`)。
"""

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

from runpod_io import cost as cost_mod
from runpod_io.auth import (
    DEFAULT_AWS_PROFILE,
    AwsCreds,
    CredentialsError,
    load_aws_creds,
    load_runpod_api_key,
)
from runpod_io.instance import (
    DEFAULT_DISK_GB,
    DEFAULT_IMAGE,
    build_env_dict,
    create_pod,
    render_onstart,
)
from runpod_io.offers import (
    DEFAULT_GPU_NAMES,
    Offer,
    pick_offer,
    search_offers,
)
from runpod_io.run_meta import generate_run_id, update_run_json
from runpod_io.volumes import (
    KNOWN_DATA_CENTERS,
    Volume,
    find_volume_by_name,
    list_volumes,
    render_volume_offers,
    search_volume_offers,
)
from runpod_io.volumes import (
    create_volume as create_volume_fn,
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
DEFAULT_VOLUME_NAME = "orbit_wars_cache_runpod"
DEFAULT_MOUNT_PATH = "/persist"
DEFAULT_COST_REPORT_DIR = Path("docs/experiment")

# vast.cli.CASE_DEFAULTS と同期。両基盤で同じ stage / train_module / config_arg
# / preprocess_cmd / canonical_weights を共有する。
CASE_DEFAULTS: dict[str, dict[str, str]] = {
    "case1": {
        "stage": "train_imitation_case1",
        "train_module": "pipeline.imitation.case1.training.train",
        "config_arg": "",
        "preprocess_cmd": "",
        "canonical_weights": "backend/pipeline/imitation/case1/policy/weights.pt",
    },
    "case3": {
        "stage": "train_imitation_case3",
        "train_module": "pipeline.imitation.case3.training.train",
        "config_arg": "--config pipeline/imitation/case3/configs/il_phase2.yaml",
        "preprocess_cmd": (
            "pipeline.imitation.case3.training.preprocess "
            "--config pipeline/imitation/case3/configs/il_phase2.yaml"
        ),
        "canonical_weights": (
            "backend/pipeline/imitation/case3/policy/weights_phase2.pt"
        ),
    },
    "case4": {
        "stage": "train_imitation_case4",
        "train_module": "pipeline.imitation.case4.training.train",
        "config_arg": "--config pipeline/imitation/case4/configs/il_case4.yaml",
        "preprocess_cmd": (
            "pipeline.imitation.case4.training.preprocess "
            "--config pipeline/imitation/case4/configs/il_case4.yaml"
        ),
        "canonical_weights": "backend/pipeline/imitation/case4/policy/weights.pt",
    },
}

DEFAULT_TEMPLATE_PATH = Path(__file__).resolve().parent / "onstart.sh.tmpl"


def _runs_root_for(case: str) -> Path:
    return Path(f"data/output/models/imitation/{case}/runs")


def _case_defaults(case: str) -> dict[str, str]:
    if case not in CASE_DEFAULTS:
        raise typer.BadParameter(
            f"unknown case={case!r}; supported: {sorted(CASE_DEFAULTS)}"
        )
    return CASE_DEFAULTS[case]


def _repo_root() -> Path:
    """`backend/` の親ディレクトリ (リポジトリ root) を返す。"""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "backend").is_dir() and (parent / ".git").exists():
            return parent
    raise RuntimeError(f"repo root not found from {here}")


def _git(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd or _repo_root()),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_remote_url() -> str:
    return _git("remote", "get-url", "origin")


def _git_current_branch() -> str:
    return _git("rev-parse", "--abbrev-ref", "HEAD")


def _verify_commit_pushed(commit_sha: str) -> None:
    """commit が origin に push 済みか確認。"""
    try:
        _git("cat-file", "-e", commit_sha)
    except subprocess.CalledProcessError as exc:
        raise typer.BadParameter(
            f"commit {commit_sha!r} does not exist locally"
        ) from exc
    try:
        remote_branches = _git("branch", "-r", "--contains", commit_sha)
    except subprocess.CalledProcessError as exc:
        raise typer.BadParameter(
            f"failed to check remote branches for {commit_sha!r}"
        ) from exc
    if not remote_branches.strip():
        raise typer.BadParameter(
            f"commit {commit_sha!r} is not pushed to origin. Run `git push` first."
        )


def _build_sdk(api_key: str) -> Any:
    """RunPod SDK を初期化して返す (module-level state なので module 自体を返す)。"""
    runpod_sdk.api_key = api_key
    return runpod_sdk


@app.command()
def train(
    commit_sha: str = typer.Argument(..., help="must be pushed to origin"),
    case: str = typer.Option(
        DEFAULT_CASE,
        "--case",
        help=f"imitation case to train (one of: {sorted(CASE_DEFAULTS)})",
    ),
    cloud_type: str = typer.Option(
        DEFAULT_CLOUD_TYPE,
        "--cloud-type",
        help="SECURE / COMMUNITY / ALL",
    ),
    gpu_names: list[str] = typer.Option(
        list(DEFAULT_GPU_NAMES),
        "--gpu-name",
        help="GPU type ids to consider (repeatable)",
    ),
    max_dph: float = typer.Option(2.0, "--max-dph"),
    seed: int = typer.Option(0, "--seed"),
    label: str | None = typer.Option(None, "--label"),
    cost_limit_usd: float = typer.Option(
        DEFAULT_COST_LIMIT_USD,
        "--cost-limit",
        help="confirm before launching if estimated cost exceeds this",
    ),
    aws_profile: str = typer.Option(DEFAULT_AWS_PROFILE, "--aws-profile"),
    image: str = typer.Option(DEFAULT_IMAGE, "--image"),
    disk_gb: int = typer.Option(DEFAULT_DISK_GB, "--disk-gb"),
    volume_id: str | None = typer.Option(
        None,
        "--volume-id",
        help="既存 network volume id (Secure Cloud のみ attach 可)",
    ),
    volume_name: str = typer.Option(
        DEFAULT_VOLUME_NAME,
        "--volume-name",
        help="自動再利用用の volume name",
    ),
    mount_path: str = typer.Option(DEFAULT_MOUNT_PATH, "--mount-path"),
    auto_create_volume: bool = typer.Option(
        False,
        "--auto-create-volume",
        help="一致 volume が無ければ --data-center-id で新規作成",
    ),
    volume_size_gb: int = typer.Option(15, "--volume-size"),
    data_center_id: str | None = typer.Option(
        None,
        "--data-center-id",
        help="pod / 新規 volume の DC (例: US-KS-2)",
    ),
) -> None:
    """Search RunPod GPU offers, pick one, and launch training for a given commit."""
    defaults = _case_defaults(case)

    try:
        repo_url = _git_remote_url()
    except subprocess.CalledProcessError as exc:
        raise typer.BadParameter(
            "could not resolve `git remote get-url origin`"
        ) from exc
    _verify_commit_pushed(commit_sha)
    branch = _git_current_branch()

    try:
        aws_creds = load_aws_creds(profile=aws_profile)
        runpod_api_key = load_runpod_api_key()
    except CredentialsError as exc:
        console.print(f"[red]credentials error:[/] {exc}")
        raise typer.Exit(code=2) from exc

    sdk = _build_sdk(runpod_api_key)

    # Volume 解決 (Secure Cloud + Pod 作成時 attach のみ).
    # 1) --volume-id 明示 → そのまま attach
    # 2) --volume-name 一致 → 既存 network volume を再利用
    # 3) --auto-create-volume + --data-center-id → 新規作成
    # Community Cloud のときは attach 不可。
    volume_id_resolved: str | None = None
    if cloud_type == "COMMUNITY" and (volume_id or auto_create_volume):
        console.print(
            "[yellow]volume:[/] Community Cloud は network volume 不可。"
            " volume 設定は無視されます。"
        )
    elif volume_id is not None:
        volume_id_resolved = volume_id
        console.print(
            f"[cyan]volume:[/] linking id={volume_id_resolved} at {mount_path}"
        )
    else:
        existing = find_volume_by_name(list_volumes(_volume_sdk()), volume_name)
        if existing is not None:
            volume_id_resolved = existing.id
            console.print(
                f"[cyan]volume:[/] reusing {existing.name!r} id={existing.id} "
                f"size={existing.size_gb}GB at {mount_path}"
            )
            if data_center_id is None:
                data_center_id = existing.data_center_id
        elif auto_create_volume:
            if data_center_id is None:
                console.print(
                    "[red]--auto-create-volume requires --data-center-id "
                    f"(known: {KNOWN_DATA_CENTERS})[/]"
                )
                raise typer.Exit(code=1)
            new_id = create_volume_fn(
                _volume_sdk(),
                name=volume_name,
                size_gb=volume_size_gb,
                data_center_id=data_center_id,
            )
            volume_id_resolved = new_id
            console.print(
                f"[green]created network volume:[/] id={new_id} "
                f"name={volume_name!r} dc={data_center_id}"
            )
        else:
            console.print(
                "[yellow]volume: not configured (uv cache 等は永続化されない)。[/]\n"
                "  --auto-create-volume + --data-center-id で初回作成、"
                "以降は --volume-name で自動再利用。"
            )

    offers = search_offers(
        sdk,
        gpu_names=gpu_names,
        cloud_type=cloud_type,
        max_dph=max_dph,
    )
    if not offers:
        console.print(
            "[red]No offers matched.[/] Loosen --max-dph, broaden --gpu-name, or "
            "switch --cloud-type=ALL."
        )
        raise typer.Exit(code=1)
    chosen = pick_offer(offers, console=console)
    estimated = chosen.dph_total * ESTIMATED_RUNTIME_HOURS
    console.print(
        f"\nSelected {chosen.gpu_type_id} ({chosen.cloud_type}) "
        f"@ ${chosen.dph_total:.3f}/h "
        f"(est. ${estimated:.3f} for {ESTIMATED_RUNTIME_HOURS}h)"
    )
    if estimated > cost_limit_usd and not typer.confirm(
        f"Estimated cost ${estimated:.3f} exceeds limit ${cost_limit_usd:.2f}. "
        "Proceed anyway?",
        default=False,
    ):
        raise typer.Exit(code=1)

    run_id = generate_run_id(branch, commit_sha, seed)
    onstart_cmd = render_onstart(
        DEFAULT_TEMPLATE_PATH,
        commit_sha=commit_sha,
        run_id=run_id,
        stage=defaults["stage"],
        branch=branch,
        repo_url=repo_url,
        case=case,
        train_module=defaults["train_module"],
        config_arg=defaults["config_arg"],
        preprocess_cmd=defaults["preprocess_cmd"],
    )
    snapshot_json = json.dumps(chosen.to_snapshot())
    env = build_env_dict(
        {
            "AWS_ACCESS_KEY_ID": aws_creds.access_key_id,
            "AWS_SECRET_ACCESS_KEY": aws_creds.secret_access_key,
            "AWS_DEFAULT_REGION": aws_creds.region,
            "RUNPOD_API_KEY": runpod_api_key,
            "ORBIT_WARS_RUN_ID": run_id,
            "ORBIT_WARS_GIT_SHA": commit_sha,
            "ORBIT_WARS_GIT_BRANCH": branch,
            "ORBIT_WARS_CASE": case,
            "ORBIT_WARS_RUNPOD_OFFER_SNAPSHOT": snapshot_json,
        }
    )
    pod_id = create_pod(
        sdk,
        name=label or run_id,
        gpu_type_id=chosen.gpu_type_id,
        cloud_type=chosen.cloud_type,
        onstart_script=onstart_cmd,
        env=env,
        image=image,
        container_disk_gb=disk_gb,
        network_volume_id=volume_id_resolved,
        volume_mount_path=mount_path,
        data_center_id=data_center_id,
    )
    console.print(
        f"\n[green]Pod launched![/] id=[bold]{pod_id}[/] run_id={run_id} case={case}"
    )
    console.print(f"  Monitor logs: [cyan]runpodctl pod logs {pod_id}[/]")
    console.print(f"  Stop manually: [cyan]runpodctl pod stop {pod_id}[/]")
    console.print(
        f"  After completion: [cyan]dev/runpod pull {run_id} --case {case}[/] then "
        f"[cyan]dev/runpod promote {run_id} --case {case}[/] to adopt"
    )


@app.command()
def pull(
    run_id: str = typer.Argument(..., help="run_id from runpod train"),
    case: str = typer.Option(DEFAULT_CASE, "--case"),
    runs_root: Path | None = typer.Option(None, "--runs-root"),
) -> None:
    """`dvc pull` で run dir をローカルに取得し、run.json を表示する。"""
    if runs_root is None:
        runs_root = _runs_root_for(case)
    relative = runs_root / run_id
    repo_root = _repo_root()
    dvc_meta = repo_root / f"{relative}.dvc"
    if not dvc_meta.is_file():
        console.print(
            f"[yellow]missing:[/] {dvc_meta.relative_to(repo_root)} — "
            "RunPod onstart should have committed it back to origin. Try "
            "`git fetch origin && git pull --rebase` on the training branch, "
            "then retry `dev/runpod pull`."
        )
        raise typer.Exit(code=1)
    cmd = ["uv", "run", "--project", "backend", "dvc", "pull", str(relative)]
    console.print(f"[dim]$ {' '.join(cmd)}[/]")
    result = subprocess.run(cmd, cwd=str(repo_root))
    if result.returncode != 0:
        raise typer.Exit(code=result.returncode)
    run_json_path = _repo_root() / relative / "run.json"
    if not run_json_path.is_file():
        console.print(
            f"[yellow]warning:[/] {run_json_path} not found after pull "
            "(check run_id and DVC remote)"
        )
        raise typer.Exit(code=1)
    payload = json.loads(run_json_path.read_text(encoding="utf-8"))
    console.print_json(data=payload)
    if payload.get("status") != "pushed":
        console.print(
            f"[yellow]warning:[/] status={payload.get('status')!r}; "
            "expected 'pushed'. Verify the run completed."
        )


@app.command()
def promote(
    run_id: str = typer.Argument(..., help="run_id to promote"),
    case: str = typer.Option(DEFAULT_CASE, "--case"),
    runs_root: Path | None = typer.Option(None, "--runs-root"),
    canonical: Path | None = typer.Option(None, "--canonical"),
    eval_results_path: Path | None = typer.Option(
        None, "--eval-results", help="optional JSON file with local eval"
    ),
) -> None:
    """Run dir の best.pt を canonical weights.pt にコピーし status=adopted。"""
    defaults = _case_defaults(case)
    if runs_root is None:
        runs_root = _runs_root_for(case)
    if canonical is None:
        canonical = Path(defaults["canonical_weights"])
    repo_root = _repo_root()
    run_dir = repo_root / runs_root / run_id
    src = run_dir / "best.pt"
    if not src.is_file():
        console.print(f"[red]not found:[/] {src}")
        raise typer.Exit(code=1)
    dst = repo_root / canonical
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    console.print(f"[green]copied[/] {src} -> {dst}")

    dvc_commit_cmd = [
        "uv",
        "run",
        "--project",
        "backend",
        "dvc",
        "commit",
        str(canonical),
    ]
    console.print(f"[dim]$ {' '.join(dvc_commit_cmd)}[/]")
    subprocess.run(dvc_commit_cmd, cwd=str(repo_root), check=False)

    patch: dict[str, Any] = {"status": "adopted"}
    if eval_results_path is not None:
        patch["local_eval_results"] = json.loads(
            eval_results_path.read_text(encoding="utf-8")
        )
    update_run_json(run_dir, **patch)
    console.print(f"[green]run.json status -> adopted[/] in {run_dir}")

    dvc_add_cmd = [
        "uv",
        "run",
        "--project",
        "backend",
        "dvc",
        "add",
        str(runs_root / run_id),
    ]
    console.print(f"[dim]$ {' '.join(dvc_add_cmd)}[/]")
    subprocess.run(dvc_add_cmd, cwd=str(repo_root), check=False)

    status_result = subprocess.run(
        ["git", "status", "--short"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    console.print("\n[bold]git status:[/]")
    console.print(status_result.stdout or "(clean)")
    console.print("\nNext: review the diff, then `git commit` and open a PR to main.")


@app.command("cost-report")
def cost_report_cmd(
    month: str | None = typer.Option(
        None, "--month", help="filter by YYYY-MM (default: current month)"
    ),
    case: str = typer.Option(DEFAULT_CASE, "--case"),
    runs_root: Path | None = typer.Option(None, "--runs-root"),
    output_dir: Path = typer.Option(DEFAULT_COST_REPORT_DIR, "--output-dir"),
) -> None:
    """run.json を集計して RunPod 月次コストレポートを生成する。"""
    if runs_root is None:
        runs_root = _runs_root_for(case)
    if month is None:
        month = datetime.now(UTC).strftime("%Y-%m")
    cost_mod.parse_month(month)
    repo_root = _repo_root()
    report = cost_mod.aggregate_runs(repo_root / runs_root, month=month)
    md = cost_mod.render_markdown(report)
    out_dir = repo_root / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"runpod_cost_report_{month}.md"
    out_path.write_text(md, encoding="utf-8")
    console.print(md)
    console.print(f"\n[green]written:[/] {out_path}")


def _volume_sdk() -> Any:
    """volume 操作は graphql 経由なので runpod.api.graphql を返す。"""
    from runpod.api import graphql as runpod_graphql

    return runpod_graphql


volume_app = typer.Typer(
    add_completion=False,
    help="Manage RunPod network volumes (uv / DVC cache の永続化用).",
    no_args_is_help=True,
)
app.add_typer(volume_app, name="volume")


@volume_app.command("list")
def volume_list_cmd() -> None:
    """所有 network volumes を一覧。"""
    try:
        load_runpod_api_key()
    except CredentialsError as exc:
        console.print(f"[red]credentials error:[/] {exc}")
        raise typer.Exit(code=2) from exc
    runpod_sdk.api_key = load_runpod_api_key()
    volumes = list_volumes(_volume_sdk())
    if not volumes:
        console.print("[yellow]No volumes found.[/]")
        return
    from rich.table import Table

    table = Table(title="RunPod network volumes")
    table.add_column("id")
    table.add_column("name")
    table.add_column("size_gb", justify="right")
    table.add_column("data_center")
    for v in volumes:
        table.add_row(v.id, v.name, str(v.size_gb), v.data_center_id)
    console.print(table)


@volume_app.command("search")
def volume_search_cmd(
    min_size_gb: int = typer.Option(15, "--min-size"),
    data_center_id: str | None = typer.Option(None, "--data-center-id"),
) -> None:
    """利用可能な data center 一覧を表示。"""
    offers = search_volume_offers(
        min_size_gb=min_size_gb, data_center_id=data_center_id
    )
    if not offers:
        console.print(f"[red]No matching DCs.[/] Known: {KNOWN_DATA_CENTERS}")
        raise typer.Exit(code=1)
    render_volume_offers(offers, console=console)


@volume_app.command("create")
def volume_create_cmd(
    name: str = typer.Argument(..., help="volume name"),
    size_gb: int = typer.Option(15, "--size"),
    data_center_id: str = typer.Option(..., "--data-center-id"),
) -> None:
    """新規 volume を作成し id を表示。"""
    try:
        runpod_sdk.api_key = load_runpod_api_key()
    except CredentialsError as exc:
        console.print(f"[red]credentials error:[/] {exc}")
        raise typer.Exit(code=2) from exc
    new_id = create_volume_fn(
        _volume_sdk(),
        name=name,
        size_gb=size_gb,
        data_center_id=data_center_id,
    )
    console.print(
        f"[green]created volume:[/] id={new_id} name={name!r} "
        f"size={size_gb}GB dc={data_center_id}"
    )


__all__ = [
    "AwsCreds",
    "CASE_DEFAULTS",
    "DEFAULT_CASE",
    "DEFAULT_CLOUD_TYPE",
    "DEFAULT_TEMPLATE_PATH",
    "Offer",
    "Volume",
    "app",
]
