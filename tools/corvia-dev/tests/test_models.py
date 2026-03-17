"""Tests for corvia-dev Pydantic models."""

from corvia_dev.models import (
    ServiceDefinition,
    ServiceState,
    ServiceStatus,
    ManagerStatus,
    ConfigSummary,
    StatusResponse,
)


def test_service_definition_corvia_inference() -> None:
    svc = ServiceDefinition(
        name="corvia-inference",
        tier=0,
        port=8030,
        health_path="/health",
        start_cmd=["corvia-inference", "serve", "--port", "8030"],
        depends_on=[],
    )
    assert svc.name == "corvia-inference"
    assert svc.tier == 0
    assert svc.exclusive_group is None


def test_service_definition_with_exclusive_group() -> None:
    svc = ServiceDefinition(
        name="ollama",
        tier=1,
        port=11434,
        health_path="/api/tags",
        start_cmd=["ollama", "serve"],
        depends_on=[],
        exclusive_group="embedding",
    )
    assert svc.exclusive_group == "embedding"


def test_service_status_stopped() -> None:
    s = ServiceStatus(name="ollama", state=ServiceState.STOPPED)
    assert s.pid is None
    assert s.port is None
    assert s.reason is None


def test_service_status_healthy() -> None:
    s = ServiceStatus(
        name="corvia-server",
        state=ServiceState.HEALTHY,
        port=8020,
        pid=1234,
        uptime_s=3600.0,
    )
    assert s.state == ServiceState.HEALTHY
    assert s.uptime_s == 3600.0


def test_status_response_json_roundtrip() -> None:
    resp = StatusResponse(
        manager=ManagerStatus(pid=100, uptime_s=60.0, state="running"),
        services=[
            ServiceStatus(name="corvia-inference", state=ServiceState.HEALTHY, port=8030, pid=101, uptime_s=55.0),
            ServiceStatus(name="corvia-server", state=ServiceState.HEALTHY, port=8020, pid=102, uptime_s=50.0),
        ],
        config=ConfigSummary(
            embedding_provider="corvia",
            merge_provider="corvia",
            storage="lite",
            workspace="corvia-workspace",
        ),
        enabled_services=[],
        logs=[],
    )
    data = resp.model_dump()
    assert data["manager"]["pid"] == 100
    assert len(data["services"]) == 2
    assert data["config"]["embedding_provider"] == "corvia"
    json_str = resp.model_dump_json()
    assert '"corvia-inference"' in json_str
