"""Vast.ai 操作の Typer CLI: train / pull / promote / cost-report。

`dev/vast-*` thin wrappers から `python -m vast <cmd>` で呼ばれる前提。
"""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from vast import cost as cost_mod
from vast.auth import (
    DEFAULT_AWS_PROFILE,
    AwsCreds,
    CredentialsError,
    load_aws_creds,
    load_vast_api_key,
)
from vast.instance import (
    DEFAULT_DISK_GB,
    DEFAULT_IMAGE,
    build_env_dict,
    create_instance,
    render_onstart,
)
from vast.offers import (
    Offer,
    pick_offer,
    search_offers,
)
from vast.run_meta import generate_run_id

app = typer.Typer(
    add_completion=False,
    help="Vast.ai-driven GPU training basis for Orbit Wars.",
    no_args_is_help=True,
)
console = Console()

DEFAULT_STAGE = "train_imitation_case1"
DEFAULT_RUNS_ROOT = Path("artifacts/models/imitation/case1/runs")
DEFAULT_TEMPLATE_PATH = Path(__file__).resolve().parent / "onstart.sh.tmpl"
DEFAULT_CANONICAL_WEIGHTS = Path("backend/pipeline/imitation/case1/policy/weights.pt")
DEFAULT_COST_REPORT_DIR = Path("docs/experiment")
DEFAULT_COST_LIMIT_USD = 1.0
ESTIMATED_RUNTIME_HOURS = 0.5


def _repo_root() -> Path:
    """`backend/` の親ディレクトリ (リポジトリ root) を返す。"""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "backend").is_dir() and (parent / ".git").exists():
            return parent
    raise RuntimeError(f"repo root not found from {here}")


def _git(*args: str, cwd: Path | None = None) -> str:
    """git サブコマンドを実行して stdout を返す (失敗時は CalledProcessError)。"""
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
    """commit が origin に push 済みか確認。なければ BadParameter。"""
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


@app.command()
def train(
    commit_sha: str = typer.Argument(..., help="must be pushed to origin"),
    stage: str = typer.Option(
        DEFAULT_STAGE, "--stage", "-s", help="dvc.yaml stage name"
    ),
    seed: int = typer.Option(0, "--seed", help="seed override (recorded in run.json)"),
    label: str | None = typer.Option(None, "--label", help="custom Vast label"),
    cost_limit_usd: float = typer.Option(
        DEFAULT_COST_LIMIT_USD,
        "--cost-limit",
        help="confirm before launching if estimated cost exceeds this",
    ),
    aws_profile: str = typer.Option(
        DEFAULT_AWS_PROFILE, "--aws-profile", help="AWS profile name"
    ),
    image: str = typer.Option(DEFAULT_IMAGE, "--image"),
    disk_gb: int = typer.Option(DEFAULT_DISK_GB, "--disk-gb"),
) -> None:
    """Search Vast offers, pick one, and launch GPU training for a given commit."""
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
        vast_api_key = load_vast_api_key()
    except CredentialsError as exc:
        console.print(f"[red]credentials error:[/] {exc}")
        raise typer.Exit(code=2) from exc

    sdk = _build_sdk(vast_api_key)
    offers = search_offers(sdk)
    if not offers:
        console.print("[red]No offers matched the filter. Try again later.[/]")
        raise typer.Exit(code=1)
    chosen = pick_offer(offers, console=console)
    estimated = chosen.dph_total * ESTIMATED_RUNTIME_HOURS
    console.print(
        f"\nSelected offer #{chosen.id} {chosen.gpu_name} @ ${chosen.dph_total:.3f}/h "
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
        stage=stage,
        branch=branch,
        repo_url=repo_url,
    )
    env = build_env_dict(
        {
            "AWS_ACCESS_KEY_ID": aws_creds.access_key_id,
            "AWS_SECRET_ACCESS_KEY": aws_creds.secret_access_key,
            "AWS_DEFAULT_REGION": aws_creds.region,
            "VAST_API_KEY": vast_api_key,
            "ORBIT_WARS_RUN_ID": run_id,
            "ORBIT_WARS_GIT_SHA": commit_sha,
            "ORBIT_WARS_GIT_BRANCH": branch,
        }
    )
    instance_id = create_instance(
        sdk,
        chosen.id,
        onstart_cmd=onstart_cmd,
        env=env,
        image=image,
        disk_gb=disk_gb,
        label=label or run_id,
    )
    console.print(
        f"\n[green]Instance launched![/] id=[bold]{instance_id}[/] run_id={run_id}"
    )
    console.print(f"  Monitor logs: [cyan]vastai logs {instance_id}[/]")
    console.print(f"  Stop manually: [cyan]vastai destroy instance {instance_id}[/]")
    console.print(
        f"  After completion: [cyan]dev/vast-pull {run_id}[/] then "
        f"[cyan]dev/vast-promote {run_id}[/] to adopt"
    )


@app.command()
def pull(
    run_id: str = typer.Argument(..., help="run_id from vast-train"),
    runs_root: Path = typer.Option(
        DEFAULT_RUNS_ROOT, "--runs-root", help="root of run dirs"
    ),
) -> None:
    """`dvc pull` で run dir をローカルに取得し、run.json を表示する。"""
    relative = runs_root / run_id
    cmd = [
        "uv",
        "run",
        "--project",
        "backend",
        "dvc",
        "pull",
        str(relative),
    ]
    console.print(f"[dim]$ {' '.join(cmd)}[/]")
    result = subprocess.run(cmd, cwd=str(_repo_root()))
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
    runs_root: Path = typer.Option(
        DEFAULT_RUNS_ROOT, "--runs-root", help="root of run dirs"
    ),
    canonical: Path = typer.Option(
        DEFAULT_CANONICAL_WEIGHTS,
        "--canonical",
        help="canonical weights.pt path",
    ),
    eval_results_path: Path | None = typer.Option(
        None,
        "--eval-results",
        help="optional JSON file with local eval to merge into run.json",
    ),
) -> None:
    """Run dir の best.pt を canonical weights.pt にコピーし、run.json を adopted に更新する。"""
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

    # canonical は DVC 管理下なので commit (add ではなく commit が正しい)
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

    # run.json の status 更新
    from vast.run_meta import update_run_json

    patch: dict[str, Any] = {"status": "adopted"}
    if eval_results_path is not None:
        patch["local_eval_results"] = json.loads(
            eval_results_path.read_text(encoding="utf-8")
        )
    update_run_json(run_dir, **patch)
    console.print(f"[green]run.json status -> adopted[/] in {run_dir}")

    # 採用後の run dir 自体も DVC で push 対象に
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
    runs_root: Path = typer.Option(
        DEFAULT_RUNS_ROOT, "--runs-root", help="root of run dirs"
    ),
    output_dir: Path = typer.Option(
        DEFAULT_COST_REPORT_DIR, "--output-dir", help="markdown output dir"
    ),
) -> None:
    """run.json を集計して markdown のコストレポートを生成する。"""
    if month is None:
        month = datetime.now(UTC).strftime("%Y-%m")
    cost_mod.parse_month(month)
    repo_root = _repo_root()
    report = cost_mod.aggregate_runs(repo_root / runs_root, month=month)
    md = cost_mod.render_markdown(report)
    out_dir = repo_root / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"vast_cost_report_{month}.md"
    out_path.write_text(md, encoding="utf-8")
    console.print(md)
    console.print(f"\n[green]written:[/] {out_path}")


def _build_sdk(api_key: str) -> Any:
    """vastai SDK の VastAI を遅延 import する。"""
    from vastai import VastAI

    return VastAI(api_key=api_key)


# Re-exports kept for downstream wrapper usage.
__all__ = [
    "Offer",
    "AwsCreds",
    "DEFAULT_RUNS_ROOT",
    "DEFAULT_STAGE",
    "DEFAULT_TEMPLATE_PATH",
    "app",
]
