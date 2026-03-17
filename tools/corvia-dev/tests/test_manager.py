"""Tests for process manager."""

import json
from pathlib import Path

from corvia_dev.manager import ProcessManager, ManagedProcess
from corvia_dev.models import ServiceDefinition, ServiceState


def test_managed_process_initial_state() -> None:
    svc = ServiceDefinition(name="test", tier=0, port=9999, start_cmd=["echo", "hi"])
    mp = ManagedProcess(service=svc)
    assert mp.state == ServiceState.STOPPED
    assert mp.pid is None
    assert mp.restart_count == 0
    assert mp.backoff_s == 1.0


def test_managed_process_backoff_escalation() -> None:
    svc = ServiceDefinition(name="test", tier=0, port=9999, start_cmd=["echo", "hi"])
    mp = ManagedProcess(service=svc)
    mp.escalate_backoff()
    assert mp.backoff_s == 2.0
    mp.escalate_backoff()
    assert mp.backoff_s == 4.0
    for _ in range(10):
        mp.escalate_backoff()
    assert mp.backoff_s == 60.0


def test_managed_process_backoff_reset() -> None:
    svc = ServiceDefinition(name="test", tier=0, port=9999, start_cmd=["echo", "hi"])
    mp = ManagedProcess(service=svc)
    mp.backoff_s = 32.0
    mp.reset_backoff()
    assert mp.backoff_s == 1.0


def test_process_manager_write_state(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    svc = ServiceDefinition(name="test-svc", tier=0, port=8020, start_cmd=["echo"])
    mgr = ProcessManager(
        services=[svc],
        workspace_root=tmp_path,
        state_path=state_path,
    )
    mgr.write_state()
    data = json.loads(state_path.read_text())
    assert data["manager"]["pid"] > 0
    assert len(data["services"]) == 1
    assert data["services"][0]["name"] == "test-svc"
    assert data["services"][0]["state"] == "stopped"
