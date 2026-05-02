"""runpod_io.ssh のユニットテスト。"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from runpod_io.ssh import (
    SshEndpoint,
    SshUnavailable,
    get_pod_ssh_endpoint,
    parse_ssh_endpoint,
)


def _pod_with_ports(ports: list[dict[str, object]]) -> dict[str, object]:
    return {"id": "pod-x", "runtime": {"ports": ports}}


def test_parse_ssh_endpoint_picks_public_tcp_22() -> None:
    pod = _pod_with_ports(
        [
            {"privatePort": 8888, "publicPort": 60001, "isIpPublic": False, "ip": "x"},
            {
                "privatePort": 22,
                "publicPort": 40124,
                "isIpPublic": True,
                "ip": "1.2.3.4",
            },
            {
                "privatePort": 22,
                "publicPort": 99999,
                "isIpPublic": False,
                "ip": "internal",
            },
        ]
    )
    ep = parse_ssh_endpoint(pod)
    assert ep.host == "1.2.3.4"
    assert ep.port == 40124
    assert ep.user == "root"


def test_parse_ssh_endpoint_no_public_port_raises() -> None:
    pod = _pod_with_ports(
        [{"privatePort": 22, "publicPort": 99, "isIpPublic": False, "ip": "x"}]
    )
    with pytest.raises(SshUnavailable, match="no public TCP/22"):
        parse_ssh_endpoint(pod)


def test_parse_ssh_endpoint_empty_runtime() -> None:
    with pytest.raises(SshUnavailable):
        parse_ssh_endpoint({"id": "pod-x", "runtime": None})


def test_to_command_includes_key_and_options() -> None:
    ep = SshEndpoint(host="1.2.3.4", port=40124, key_path=Path("/tmp/key"))
    cmd = ep.to_command("tail -f /var/log/onstart.log")
    assert cmd[0] == "ssh"
    assert "/tmp/key" in cmd
    assert "40124" in cmd
    assert "root@1.2.3.4" in cmd
    assert "tail -f /var/log/onstart.log" in cmd
    assert "StrictHostKeyChecking=accept-new" in cmd


def test_to_command_no_remote_cmd_omits_trailing_arg() -> None:
    ep = SshEndpoint(host="h", port=1)
    cmd = ep.to_command()
    assert cmd[-1] == "root@h"


def test_get_pod_ssh_endpoint_via_sdk() -> None:
    sdk = MagicMock()
    sdk.get_pod.return_value = _pod_with_ports(
        [{"privatePort": 22, "publicPort": 40000, "isIpPublic": True, "ip": "5.6.7.8"}]
    )
    ep = get_pod_ssh_endpoint(sdk, "pod-x")
    assert ep.host == "5.6.7.8"
    assert ep.port == 40000
    sdk.get_pod.assert_called_once_with("pod-x")


def test_get_pod_ssh_endpoint_pod_missing() -> None:
    sdk = MagicMock()
    sdk.get_pod.return_value = None
    with pytest.raises(SshUnavailable, match="not found"):
        get_pod_ssh_endpoint(sdk, "pod-x")
