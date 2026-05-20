"""runpod_io.runpod.sync のユニットテスト。"""

from __future__ import annotations

from pathlib import Path

import pytest

from runpod_io.runpod.ssh import SshEndpoint, build_proxy_endpoint
from runpod_io.runpod.sync import (
    DEFAULT_EXCLUDES,
    DEFAULT_REMOTE_BOT_DIR,
    build_sync_plan,
)


def _direct_endpoint() -> SshEndpoint:
    return SshEndpoint(
        host="1.2.3.4",
        port=40124,
        key_path=Path("/tmp/key"),
        kind="direct",
    )


def test_build_sync_plan_push_direct() -> None:
    ep = _direct_endpoint()
    plan = build_sync_plan(
        endpoint=ep,
        local_bot_dir=Path("/repo/bot"),
        direction="push",
    )
    assert plan.direction == "push"
    assert plan.src == "/repo/bot/"
    # direct は user@host 形式で組む必要あり (rsync がローカルユーザーで接続するのを防ぐ)
    assert plan.dst == f"root@1.2.3.4:{DEFAULT_REMOTE_BOT_DIR}/"
    assert plan.cmd[0] == "rsync"
    assert "-avz" in plan.cmd
    # -e に ssh コマンド文字列が含まれ、末尾の user@host は除去されていること
    e_idx = plan.cmd.index("-e")
    rsh = plan.cmd[e_idx + 1]
    assert "ssh" in rsh
    assert "/tmp/key" in rsh
    assert "40124" in rsh
    assert "1.2.3.4" not in rsh  # 末尾 host は除去済


def test_build_sync_plan_pull_proxy() -> None:
    ep = build_proxy_endpoint("podx-hash", key_path=Path("/tmp/k"))
    plan = build_sync_plan(
        endpoint=ep,
        local_bot_dir=Path("/repo/bot"),
        direction="pull",
    )
    assert plan.direction == "pull"
    assert plan.src == f"podx-hash@ssh.runpod.io:{DEFAULT_REMOTE_BOT_DIR}/"
    assert plan.dst == "/repo/bot/"
    # proxy は -p を含まない
    e_idx = plan.cmd.index("-e")
    rsh = plan.cmd[e_idx + 1]
    assert "-p" not in rsh.split()


def test_build_sync_plan_includes_default_excludes() -> None:
    plan = build_sync_plan(
        endpoint=_direct_endpoint(),
        local_bot_dir=Path("/repo/bot"),
        direction="push",
    )
    for pattern in DEFAULT_EXCLUDES:
        assert pattern in plan.cmd
    # 重要: .venv が必ず exclude されていること (Linux ↔ Mac の混在防止)
    assert ".venv/" in plan.cmd


def test_build_sync_plan_dry_run_and_delete() -> None:
    plan = build_sync_plan(
        endpoint=_direct_endpoint(),
        local_bot_dir=Path("/repo/bot"),
        direction="push",
        dry_run=True,
        delete=True,
    )
    assert "--dry-run" in plan.cmd
    assert "--delete" in plan.cmd


def test_build_sync_plan_rejects_relative_local() -> None:
    with pytest.raises(ValueError):
        build_sync_plan(
            endpoint=_direct_endpoint(),
            local_bot_dir=Path("bot"),  # relative
            direction="push",
        )


def test_build_sync_plan_rejects_bad_direction() -> None:
    with pytest.raises(ValueError):
        build_sync_plan(
            endpoint=_direct_endpoint(),
            local_bot_dir=Path("/repo/bot"),
            direction="sideways",
        )
