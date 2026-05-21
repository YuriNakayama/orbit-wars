"""Kaggle Notebook (.ipynb) を render する純関数。

学習 run 1 回ごとに生成する notebook。cell 構成:
    A: env setup (os.environ.update + run_dir 作成)
    B: pip install /kaggle/input/<dataset>/wheels/*.whl  (Rust wheel)
    C: pip install -e /kaggle/input/<dataset>/bot/
    D: train 実行 (subprocess → train.log)
    E: artifact 収集 (run_dir に集約)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

_VALID_VALUE = re.compile(r"^[A-Za-z0-9._\-/:=@]+$")
_VALID_RUN_ID = re.compile(r"^\d{8}-\d{6}__[A-Za-z0-9_-]+__[0-9a-f]{7}__seed\d+$")


@dataclass(frozen=True)
class RenderContext:
    """Notebook render 用パラメータ。"""

    run_id: str
    commit_sha: str
    branch: str
    case: str
    train_module: str
    config_arg: str
    dataset_slug: str
    accelerator: str
    kaggle_kernel_meta_initial: dict[str, Any]
    seed: int = 0


def render_notebook(ctx: RenderContext) -> dict[str, Any]:
    """nbformat 4 の notebook dict を返す。"""
    _validate(ctx)
    dataset_dirname = ctx.dataset_slug.split("/")[-1]
    dataset_mount = f"/kaggle/input/{dataset_dirname}"
    run_dir = f"/kaggle/working/runs/{ctx.run_id}"
    meta_json = json.dumps(ctx.kaggle_kernel_meta_initial, ensure_ascii=False)

    cells: list[dict[str, Any]] = [
        _code_cell(_cell_env_setup(ctx, run_dir, meta_json)),
        _code_cell(_cell_install_wheels(dataset_mount)),
        _code_cell(_cell_install_bot(dataset_mount)),
    ]
    if _needs_parquet_to_npy(ctx.case):
        cells.append(_code_cell(_cell_parquet_to_npy(ctx)))
    cells.extend(
        [
            _code_cell(_cell_train(ctx, run_dir, dataset_mount)),
            _code_cell(_cell_collect_artifacts(ctx, run_dir)),
        ]
    )

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "name": "python3",
                "language": "python",
                "display_name": "Python 3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _validate(ctx: RenderContext) -> None:
    if not _VALID_RUN_ID.match(ctx.run_id):
        raise ValueError(f"invalid run_id format: {ctx.run_id!r}")
    for name, value in [
        ("commit_sha", ctx.commit_sha),
        ("branch", ctx.branch),
        ("case", ctx.case),
        ("train_module", ctx.train_module),
        ("dataset_slug", ctx.dataset_slug),
        ("accelerator", ctx.accelerator),
    ]:
        if not _VALID_VALUE.match(value):
            raise ValueError(
                f"invalid {name} value (only A-Za-z0-9._-/:=@ allowed): {value!r}"
            )
    if "/" not in ctx.dataset_slug:
        raise ValueError(f"dataset_slug must be <owner>/<name>: {ctx.dataset_slug!r}")
    if ctx.config_arg and not re.match(r"^[A-Za-z0-9._\-/=,\s]+$", ctx.config_arg):
        raise ValueError(f"invalid config_arg: {ctx.config_arg!r}")


def _code_cell(source: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def _cell_env_setup(ctx: RenderContext, run_dir: str, meta_json: str) -> str:
    payload = {
        "ORBIT_WARS_RUN_ID": ctx.run_id,
        "ORBIT_WARS_GIT_SHA": ctx.commit_sha,
        "ORBIT_WARS_GIT_BRANCH": ctx.branch,
        "ORBIT_WARS_CASE": ctx.case,
        "ORBIT_WARS_KAGGLE_KERNEL_SLUG": ctx.kaggle_kernel_meta_initial.get(
            "kernel_slug", ""
        ),
        "ORBIT_WARS_KAGGLE_KERNEL_VERSION": str(
            ctx.kaggle_kernel_meta_initial.get("kernel_version", "")
        ),
        "ORBIT_WARS_KAGGLE_ACCELERATOR": ctx.accelerator,
        "ORBIT_WARS_KAGGLE_KERNEL_META": meta_json,
        "ORBIT_WARS_RUN_DIR": run_dir,
    }
    return (
        "# cell A: env setup\n"
        "import os\n"
        f"_env = {json.dumps(payload, ensure_ascii=False)}\n"
        "os.environ.update({k: str(v) for k, v in _env.items()})\n"
        f"os.makedirs({run_dir!r}, exist_ok=True)\n"
    )


def _cell_install_wheels(dataset_mount: str) -> str:
    """Discover the actual dataset mount path, copy / symlink it into a writable
    REPO_ROOT (Kaggle datasets normally arrive at /kaggle/input/<slug>), then
    install Rust wheels (if any).

    Kaggle ``dir_mode='zip'`` uploads compress folders during upload, but the
    files are **unpacked on the dataset side**: at runtime the dataset mount
    contains a normal directory tree (``bot/``, ``simulator/``, ``data/``,
    ``params.yaml``). We mirror that into ``/tmp/orbit-wars-repo/`` so
    ``find_repo_root()``'s ``(bot/, .git/)`` sentinel pair is satisfied and
    editable installs work.
    """
    return (
        "# cell B: discover dataset mount, mirror to writable REPO_ROOT\n"
        "import os, glob, shutil, subprocess\n"
        "REPO_ROOT = '/tmp/orbit-wars-repo'\n"
        f"EXPECTED = {dataset_mount!r}\n"
        "def _find_mount(target_marker='bot'):\n"
        "    # Kaggle mounts datasets in a few different layouts depending on\n"
        "    # public/private/version. Walk /kaggle/input up to 3 levels and pick\n"
        "    # the first dir that contains both ``bot`` and ``params.yaml``.\n"
        "    for root, dirs, files in os.walk('/kaggle/input'):\n"
        "        depth = root[len('/kaggle/input'):].count('/')\n"
        "        if depth > 3:\n"
        "            dirs[:] = []\n"
        "            continue\n"
        "        if target_marker in dirs and 'params.yaml' in files:\n"
        "            return root\n"
        "    return None\n"
        "print('## /kaggle/input listing:')\n"
        "for root, dirs, files in os.walk('/kaggle/input'):\n"
        "    depth = root[len('/kaggle/input'):].count('/')\n"
        "    if depth > 3:\n"
        "        dirs[:] = []\n"
        "        continue\n"
        "    print(' ', root, 'dirs=', dirs[:5], 'files=', files[:5])\n"
        "DATASET_MOUNT = EXPECTED if os.path.isdir(EXPECTED) else _find_mount()\n"
        "if DATASET_MOUNT is None:\n"
        "    raise RuntimeError(f'dataset mount not found; expected {EXPECTED}')\n"
        "print('using DATASET_MOUNT =', DATASET_MOUNT)\n"
        "os.makedirs(REPO_ROOT, exist_ok=True)\n"
        "for entry in os.listdir(DATASET_MOUNT):\n"
        "    src = os.path.join(DATASET_MOUNT, entry)\n"
        "    dst = os.path.join(REPO_ROOT, entry)\n"
        "    if os.path.isdir(src):\n"
        "        if os.path.exists(dst):\n"
        "            shutil.rmtree(dst)\n"
        "        shutil.copytree(src, dst)\n"
        "    else:\n"
        "        shutil.copy2(src, dst)\n"
        "# Sentinel so find_repo_root() recognizes REPO_ROOT.\n"
        "os.makedirs(os.path.join(REPO_ROOT, '.git'), exist_ok=True)\n"
        "with open(os.path.join(REPO_ROOT, '.git', 'HEAD'), 'w') as f:\n"
        "    f.write('ref: refs/heads/kaggle-snapshot\\n')\n"
        "# Builder rewrites data/ -> mart_payload/ to avoid Kaggle reserved\n"
        "# top-level name. Restore it inside REPO_ROOT so train.py can find\n"
        "# `data/mart/imitation/<case>/{train,val}.parquet`.\n"
        "_mart_src = os.path.join(REPO_ROOT, 'mart_payload')\n"
        "_mart_dst = os.path.join(REPO_ROOT, 'data')\n"
        "if os.path.isdir(_mart_src):\n"
        "    os.makedirs(_mart_dst, exist_ok=True)\n"
        "    for sub in os.listdir(_mart_src):\n"
        "        shutil.move(os.path.join(_mart_src, sub),\n"
        "                    os.path.join(_mart_dst, sub))\n"
        "    os.rmdir(_mart_src)\n"
        "    print('restored mart_payload -> data')\n"
        "# Install any pre-built wheels\n"
        "wheels = glob.glob(os.path.join(REPO_ROOT, 'wheels', '*.whl'))\n"
        "if wheels:\n"
        "    subprocess.check_call(['pip', 'install', '-q'] + wheels)\n"
        "    print('installed wheels:', wheels)\n"
        "else:\n"
        "    print('no wheels dir, skipping')\n"
    )


def _cell_install_bot(dataset_mount: str) -> str:
    """Make the bot package importable by adding ``bot/src`` and ``bot/`` to
    ``sys.path`` and install missing third-party deps.

    Editable `pip install -e .` is fragile on Kaggle because pyproject.toml's
    ``[tool.uv.sources]`` references local paths that pip doesn't understand.
    Direct sys.path injection avoids the metadata-generation step entirely.

    We also probe the GPU's torch+CUDA pairing; if the smoke fails (CC mismatch
    on T4), reinstall torch with the official PyTorch CUDA 12.1 wheel.
    """
    return (
        "# cell C: make bot importable + ensure torch covers the assigned GPU\n"
        "# We probe GPU compatibility in a subprocess (so the live kernel\n"
        "# process never imports the stale Kaggle torch wheel — once a\n"
        "# C extension is loaded it cannot be replaced safely).\n"
        "import sys, subprocess, json, os\n"
        "REPO_ROOT = '/tmp/orbit-wars-repo'\n"
        "for p in [REPO_ROOT + '/bot/src', REPO_ROOT + '/bot', REPO_ROOT]:\n"
        "    if p not in sys.path:\n"
        "        sys.path.insert(0, p)\n"
        "# Install third-party deps that the Kaggle image may lack.\n"
        "deps = ['polars', 'kaggle-environments']\n"
        "subprocess.run(['pip', 'install', '--no-deps'] + deps, check=False)\n"
        "\n"
        "_PROBE = (\n"
        "    'import json, sys\\n'\n"
        "    'try:\\n'\n"
        "    '    import torch\\n'\n"
        "    '    info = {\"version\": torch.__version__,\\n'\n"
        "    '            \"cuda\": torch.version.cuda,\\n'\n"
        "    '            \"available\": torch.cuda.is_available(),\\n'\n"
        '    \'            "smoke_ok": False, "err": None}\\n\'\n'
        "    '    if torch.cuda.is_available():\\n'\n"
        "    '        try:\\n'\n"
        "    '            (torch.zeros(2, 2).cuda() + 1).sum().item()\\n'\n"
        "    '            info[\"smoke_ok\"] = True\\n'\n"
        "    '        except Exception as e:\\n'\n"
        "    '            info[\"err\"] = repr(e)\\n'\n"
        "    '    else:\\n'\n"
        "    '        info[\"smoke_ok\"] = True\\n'\n"
        "    'except Exception as e:\\n'\n"
        '    \'    info = {"err": repr(e), "smoke_ok": False}\\n\'\n'
        "    'sys.stdout.write(json.dumps(info))\\n'\n"
        ")\n"
        "\n"
        "def _probe():\n"
        "    r = subprocess.run(\n"
        "        [sys.executable, '-c', _PROBE],\n"
        "        capture_output=True, text=True, check=False,\n"
        "    )\n"
        "    try:\n"
        "        info = json.loads(r.stdout.strip().splitlines()[-1])\n"
        "    except Exception:\n"
        "        info = {'smoke_ok': False, 'err': r.stderr[:500]}\n"
        "    print('torch probe:', info)\n"
        "    return info\n"
        "\n"
        "info = _probe()\n"
        "if not info.get('smoke_ok'):\n"
        "    # Kaggle's bundled torch wheel doesn't cover this GPU's CC.\n"
        "    # cu118 wheel carries sm_50/sm_60/sm_70/sm_75/sm_80/sm_86/sm_90\n"
        "    # so it covers every Kaggle-allocated GPU (P100=sm_60, T4=sm_75,\n"
        "    # V100=sm_70, A100=sm_80).\n"
        "    print('reinstalling torch with cu118 wheel (sm_50-sm_90)...')\n"
        "    subprocess.check_call([\n"
        "        'pip', 'install', '--quiet', '--force-reinstall',\n"
        "        '--index-url', 'https://download.pytorch.org/whl/cu118',\n"
        "        'torch',\n"
        "    ])\n"
        "    info = _probe()\n"
        "    if not info.get('smoke_ok'):\n"
        "        # Fail loudly rather than silently fall back to CPU — the\n"
        "        # whole point of Kaggle Kernel is GPU acceleration.\n"
        "        raise RuntimeError(\n"
        "            f'GPU unusable after cu118 reinstall; aborting. info={info}'\n"
        "        )\n"
        "\n"
        "# Verify bot package is importable in a subprocess too (matches\n"
        "# the env that cell D's training subprocess will see).\n"
        "_IMPORT_PROBE = (\n"
        "    'import sys\\n'\n"
        '    f\'sys.path[:0] = {[REPO_ROOT + "/bot/src", REPO_ROOT + "/bot", REPO_ROOT]!r}\\n\'\n'  # noqa: E501
        "    'import pipeline, utils.repo_root\\n'\n"
        "    'print(\"bot import OK\")\\n'\n"
        ")\n"
        "subprocess.run([sys.executable, '-c', _IMPORT_PROBE], check=False)\n"
    )


def _needs_parquet_to_npy(case: str) -> bool:
    """case11 / case11_smoke は per-column npy (mmap) を要求するため、
    notebook 上で parquet -> npy 変換を 1 度だけ実行する。
    """
    return case == "case11" or case.startswith("case11_")


def _cell_parquet_to_npy(ctx: RenderContext) -> str:
    """case11 mart parquet を per-column .npy に変換 (cell D の直前で実行)。

    case と同名の subdir を見る (smoke 用は data/mart/imitation/case11_smoke/、
    本番は data/mart/imitation/case11/)。出力 npy は dataset.py の
    _npy_dir_for() が見るパス (<stem>_npy/) に揃える。
    """
    case = ctx.case
    mart_dir = f"/tmp/orbit-wars-repo/data/mart/imitation/{case}"
    return (
        "# cell C2: parquet -> per-column .npy (case11 only)\n"
        "import os, sys, subprocess\n"
        f"mart_dir = {mart_dir!r}\n"
        "train_pq = os.path.join(mart_dir, 'train.parquet')\n"
        "val_pq = os.path.join(mart_dir, 'val.parquet')\n"
        "for p in [train_pq, val_pq]:\n"
        "    if not os.path.isfile(p):\n"
        "        raise RuntimeError(f'mart parquet missing: {p}')\n"
        "    sz_gb = os.path.getsize(p) / (1024**3)\n"
        "    print(f'  {p}: {sz_gb:.2f} GB')\n"
        "cmd = [\n"
        "    sys.executable, '-m',\n"
        "    'pipeline.imitation.case11.training.parquet_to_npy',\n"
        "    '--train-parquet', train_pq,\n"
        "    '--val-parquet', val_pq,\n"
        "    '--out-root', mart_dir,\n"
        "]\n"
        "env = os.environ.copy()\n"
        "env['PYTHONPATH'] = (\n"
        "    '/tmp/orbit-wars-repo/bot/src:/tmp/orbit-wars-repo/bot:'\n"
        "    '/tmp/orbit-wars-repo'\n"
        ")\n"
        "print('running:', ' '.join(cmd))\n"
        "proc = subprocess.run(\n"
        "    cmd, cwd='/tmp/orbit-wars-repo/bot', env=env, check=False,\n"
        ")\n"
        "if proc.returncode != 0:\n"
        "    raise SystemExit(\n"
        "        f'parquet_to_npy failed with exit_code={proc.returncode}'\n"
        "    )\n"
        "# Report npy directory sizes for sanity.\n"
        "for split in ('train_npy', 'val_npy'):\n"
        "    d = os.path.join(mart_dir, split)\n"
        "    if os.path.isdir(d):\n"
        "        total = sum(\n"
        "            os.path.getsize(os.path.join(d, f))\n"
        "            for f in os.listdir(d)\n"
        "        )\n"
        "        print(f'  {d}: {total / (1024**3):.2f} GB')\n"
    )


def _cell_train(ctx: RenderContext, run_dir: str, dataset_mount: str) -> str:
    args = ctx.config_arg.strip().split() if ctx.config_arg.strip() else []
    return (
        "# cell D: train\n"
        "import subprocess, sys, os\n"
        f"run_dir = {run_dir!r}\n"
        "log_path = os.path.join(run_dir, 'train.log')\n"
        f"cmd = [sys.executable, '-m', {ctx.train_module!r}] + {args!r}\n"
        "cwd = '/tmp/orbit-wars-repo/bot'\n"
        "env = os.environ.copy()\n"
        "env['PYTHONPATH'] = (\n"
        "    '/tmp/orbit-wars-repo/bot/src:/tmp/orbit-wars-repo/bot:'\n"
        "    '/tmp/orbit-wars-repo'\n"
        ")\n"
        "# (CUDA enabled — relying on Kaggle's pre-installed torch wheel which\n"
        "#  ships sm_75 kernels for T4. Set ORBIT_WARS_KAGGLE_FORCE_CPU=1 in\n"
        "#  the kernel env to fall back to CPU.)\n"
        "if os.environ.get('ORBIT_WARS_KAGGLE_FORCE_CPU') == '1':\n"
        "    env['CUDA_VISIBLE_DEVICES'] = ''\n"
        "    print('CUDA disabled via ORBIT_WARS_KAGGLE_FORCE_CPU')\n"
        "print('running:', ' '.join(cmd), 'cwd=', cwd)\n"
        "with open(log_path, 'w') as logf:\n"
        "    proc = subprocess.run(\n"
        "        cmd, cwd=cwd, env=env,\n"
        "        stdout=logf, stderr=subprocess.STDOUT, check=False,\n"
        "    )\n"
        "print('exit_code=', proc.returncode)\n"
        "with open(log_path) as f:\n"
        "    tail = f.readlines()[-80:]\n"
        "print(''.join(tail))\n"
        "if proc.returncode != 0:\n"
        "    raise SystemExit(proc.returncode)\n"
    )


def _cell_collect_artifacts(ctx: RenderContext, run_dir: str) -> str:
    src = (
        f"/tmp/orbit-wars-repo/data/output/models/imitation/"
        f"{ctx.case}/runs/{ctx.run_id}/"
    )
    return (
        "# cell E: collect artifacts into run_dir\n"
        "import os, shutil, glob\n"
        f"src = {src!r}\n"
        f"dst = {run_dir!r}\n"
        "os.makedirs(dst, exist_ok=True)\n"
        "if os.path.isdir(src):\n"
        "    for f in glob.glob(os.path.join(src, '*')):\n"
        "        if os.path.isfile(f):\n"
        "            shutil.copy2(f, dst)\n"
        "for f in sorted(os.listdir(dst)):\n"
        "    print(f, os.path.getsize(os.path.join(dst, f)))\n"
    )
