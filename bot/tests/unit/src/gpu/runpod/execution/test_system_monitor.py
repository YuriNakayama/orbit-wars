"""Tests for runpod_io.execution.system_monitor sample formatting."""

from __future__ import annotations

import io
import json

from gpu.runpod.execution import system_monitor


def test_take_sample_fields_present() -> None:
    sample = system_monitor._take_sample()
    assert sample.cpu_percent >= 0.0
    assert sample.ram_total_mb > 0.0
    assert 0.0 <= sample.ram_percent <= 100.0
    # ISO-8601 Z timestamp
    assert sample.timestamp.endswith("Z")
    assert "T" in sample.timestamp


def test_emit_writes_valid_json_line() -> None:
    sample = system_monitor.Sample(
        timestamp="2026-05-05T10:00:00Z",
        cpu_percent=12.5,
        ram_used_mb=1024.0,
        ram_total_mb=4096.0,
        ram_percent=25.0,
        load_avg_1m=0.42,
    )
    buf = io.StringIO()
    system_monitor._emit(sample, buf)
    line = buf.getvalue().strip()
    payload = json.loads(line)
    assert payload["cpu_percent"] == 12.5
    assert payload["ram_total_mb"] == 4096.0
    assert payload["load_avg_1m"] == 0.42


def test_emit_handles_no_load_avg() -> None:
    sample = system_monitor.Sample(
        timestamp="2026-05-05T10:00:00Z",
        cpu_percent=0.0,
        ram_used_mb=0.0,
        ram_total_mb=1.0,
        ram_percent=0.0,
        load_avg_1m=None,
    )
    buf = io.StringIO()
    system_monitor._emit(sample, buf)
    payload = json.loads(buf.getvalue().strip())
    assert payload["load_avg_1m"] is None
