# corvia-dev CLI Implementation Plan

> **Status:** Superseded — Python CLI abandoned in favor of Rust CLI (`corvia workspace`)

**Goal:** Replace bash scripts and VS Code extension with a Python CLI that manages dev services, config mutation, and process supervision with structured JSON output.

**Architecture:** Python CLI (`corvia-dev`) using click for commands, pydantic for models, tomllib/tomli_w for config. Single ProcessManager supervises all services. VS Code extension becomes a thin consumer of `corvia-dev status --json`.

**Tech Stack:** Python 3.13, click, pydantic, tomli_w (write), tomllib (read, stdlib), asyncio (subprocess management)

**Design doc:** `docs/plans/archive/2026-03-09-corvia-dev-cli-design.md`

---

### Task 1: Project Scaffolding & Package Setup

**Files:**
- Create: `tools/corvia-dev/pyproject.toml`
- Create: `tools/corvia-dev/corvia_dev/__init__.py`
- Create: `tools/corvia-dev/corvia_dev/cli.py`

**Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "corvia-dev"
version = "0.1.0"
description = "Dev environment orchestration for corvia-workspace"
requires-python = ">=3.11"
dependencies = [
    "click>=8.1",
    "pydantic>=2.0",
    "tomli_w>=1.0",
]

[project.scripts]
corvia-dev = "corvia_dev.cli:main"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

**Step 2: Create `__init__.py`**

```python
"""corvia-dev: Dev environment orchestration for corvia-workspace."""
```

**Step 3: Create minimal CLI entry point**

```python
"""CLI entry point for corvia-dev."""

import click


@click.group()
def main() -> None:
    """Dev environment orchestration for corvia-workspace."""


@main.command()
def status() -> None:
    """Show service health and config summary."""
    click.echo("corvia-dev status: not yet implemented")


if __name__ == "__main__":
    main()
```

**Step 4: Install and verify**

Run:
```bash
cd /workspaces/corvia-workspace && python3 -m pip install -e tools/corvia-dev
corvia-dev --help
corvia-dev status
```

Expected: Help text shows `status` command. `status` prints placeholder message.

**Step 5: Commit**

```bash
git add tools/corvia-dev/
git commit -m "feat(corvia-dev): scaffold Python CLI package with click"
```

---

### Task 2: Pydantic Models

**Files:**
- Create: `tools/corvia-dev/corvia_dev/models.py`
- Create: `tools/corvia-dev/tests/__init__.py`
- Create: `tools/corvia-dev/tests/test_models.py`

**Step 1: Write the failing test**

```python
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
    # Verify JSON serialization works
    json_str = resp.model_dump_json()
    assert '"corvia-inference"' in json_str
```

**Step 2: Run test to verify it fails**

Run: `cd /workspaces/corvia-workspace && python3 -m pytest tools/corvia-dev/tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'corvia_dev.models'`

**Step 3: Write the implementation**

```python
"""Pydantic models for corvia-dev."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class ServiceState(str, Enum):
    """Possible states for a managed service."""

    STARTING = "starting"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    STOPPED = "stopped"
    CRASHED = "crashed"


class ServiceDefinition(BaseModel):
    """Static definition of a service in the registry."""

    name: str
    tier: int  # 0=core, 1=provider, 2=additive
    port: int | None = None
    health_path: str = "/health"
    start_cmd: list[str] = []
    stop_signal: str = "SIGTERM"
    depends_on: list[str] = []
    exclusive_group: str | None = None


class ServiceStatus(BaseModel):
    """Runtime status of a single service."""

    name: str
    state: ServiceState
    port: int | None = None
    pid: int | None = None
    uptime_s: float | None = None
    reason: str | None = None


class ManagerStatus(BaseModel):
    """Status of the process manager itself."""

    pid: int
    uptime_s: float
    state: str


class ConfigSummary(BaseModel):
    """Summary of current corvia.toml config."""

    embedding_provider: str
    merge_provider: str
    storage: str
    workspace: str


class StatusResponse(BaseModel):
    """Full status response — the JSON contract for the VS Code extension."""

    manager: ManagerStatus | None = None
    services: list[ServiceStatus] = []
    config: ConfigSummary
    enabled_services: list[str] = []
    logs: list[str] = []
```

**Step 4: Run test to verify it passes**

Run: `cd /workspaces/corvia-workspace && python3 -m pytest tools/corvia-dev/tests/test_models.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add tools/corvia-dev/corvia_dev/models.py tools/corvia-dev/tests/
git commit -m "feat(corvia-dev): add Pydantic models for services and status"
```

---

### Task 3: Service Registry

**Files:**
- Create: `tools/corvia-dev/corvia_dev/services.py`
- Create: `tools/corvia-dev/tests/test_services.py`

**Step 1: Write the failing test**

```python
"""Tests for service registry and dependency resolution."""

from corvia_dev.services import SERVICES, get_service, resolve_startup_order


def test_registry_has_core_services() -> None:
    names = [s.name for s in SERVICES]
    assert "corvia-inference" in names
    assert "corvia-server" in names


def test_registry_tiers() -> None:
    for svc in SERVICES:
        if svc.name in ("corvia-inference", "corvia-server"):
            assert svc.tier == 0, f"{svc.name} should be tier 0"
        elif svc.name in ("ollama", "vllm", "surrealdb", "postgres"):
            assert svc.tier == 1, f"{svc.name} should be tier 1"
        elif svc.name == "coding-llm":
            assert svc.tier == 2, f"{svc.name} should be tier 2"


def test_get_service_found() -> None:
    svc = get_service("corvia-server")
    assert svc is not None
    assert svc.port == 8020


def test_get_service_not_found() -> None:
    svc = get_service("nonexistent")
    assert svc is None


def test_resolve_startup_order_default() -> None:
    """Default config: corvia-inference → corvia-server, no extras."""
    order = resolve_startup_order(
        embedding_provider="corvia",
        storage="lite",
        enabled_services=[],
    )
    names = [s.name for s in order]
    assert names == ["corvia-inference", "corvia-server"]


def test_resolve_startup_order_ollama() -> None:
    """Ollama as provider: ollama → corvia-server."""
    order = resolve_startup_order(
        embedding_provider="ollama",
        storage="lite",
        enabled_services=[],
    )
    names = [s.name for s in order]
    assert names == ["ollama", "corvia-server"]
    # corvia-inference should NOT be in the list
    assert "corvia-inference" not in names


def test_resolve_startup_order_surrealdb() -> None:
    """SurrealDB enabled: corvia-inference → surrealdb → corvia-server."""
    order = resolve_startup_order(
        embedding_provider="corvia",
        storage="surrealdb",
        enabled_services=["surrealdb"],
    )
    names = [s.name for s in order]
    assert "surrealdb" in names
    assert names.index("surrealdb") < names.index("corvia-server")


def test_resolve_startup_order_coding_llm() -> None:
    """coding-llm adds ollama after corvia-server (even if corvia uses corvia-inference)."""
    order = resolve_startup_order(
        embedding_provider="corvia",
        storage="lite",
        enabled_services=["coding-llm"],
    )
    names = [s.name for s in order]
    assert "corvia-inference" in names
    assert "ollama" in names  # needed for coding-llm
    assert names.index("corvia-server") < names.index("ollama")


def test_resolve_startup_order_ollama_plus_coding_llm() -> None:
    """Ollama for corvia + coding-llm: ollama only appears once."""
    order = resolve_startup_order(
        embedding_provider="ollama",
        storage="lite",
        enabled_services=["coding-llm"],
    )
    names = [s.name for s in order]
    assert names.count("ollama") == 1
    assert names.index("ollama") < names.index("corvia-server")
```

**Step 2: Run test to verify it fails**

Run: `cd /workspaces/corvia-workspace && python3 -m pytest tools/corvia-dev/tests/test_services.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write the implementation**

```python
"""Service registry and dependency resolution."""

from __future__ import annotations

from corvia_dev.models import ServiceDefinition

# --- Service Definitions ---

SERVICES: list[ServiceDefinition] = [
    ServiceDefinition(
        name="corvia-inference",
        tier=0,
        port=8030,
        health_path="/health",
        start_cmd=["corvia-inference", "serve", "--port", "8030"],
        depends_on=[],
        exclusive_group="embedding",
    ),
    ServiceDefinition(
        name="corvia-server",
        tier=0,
        port=8020,
        health_path="/health",
        start_cmd=["corvia", "serve"],
        depends_on=["_active_embedding"],  # resolved dynamically
    ),
    ServiceDefinition(
        name="ollama",
        tier=1,
        port=11434,
        health_path="/api/tags",
        start_cmd=["ollama", "serve"],
        depends_on=[],
        exclusive_group="embedding",
    ),
    ServiceDefinition(
        name="vllm",
        tier=1,
        port=None,  # TBD
        health_path="/health",
        start_cmd=[],  # Docker-managed
        depends_on=[],
        exclusive_group="embedding",
    ),
    ServiceDefinition(
        name="surrealdb",
        tier=1,
        port=8000,
        health_path="/health",
        start_cmd=["docker", "compose", "-f", "repos/corvia/docker/docker-compose.yml", "up", "-d"],
        depends_on=[],
        exclusive_group="storage",
    ),
    ServiceDefinition(
        name="postgres",
        tier=1,
        port=5432,
        health_path="",  # uses pg_isready instead of HTTP
        start_cmd=["docker", "compose", "-f", "repos/corvia/docker/docker-compose-pg.yml", "up", "-d"],
        depends_on=[],
        exclusive_group="storage",
    ),
    ServiceDefinition(
        name="coding-llm",
        tier=2,
        port=None,
        health_path="",
        start_cmd=[],  # virtual — depends on ollama, configures Continue
        depends_on=["ollama"],
    ),
]

_SERVICES_BY_NAME: dict[str, ServiceDefinition] = {s.name: s for s in SERVICES}

# Maps provider config value to service name
EMBEDDING_PROVIDERS: dict[str, str] = {
    "corvia": "corvia-inference",
    "ollama": "ollama",
    "vllm": "vllm",
}

STORAGE_BACKENDS: dict[str, str | None] = {
    "lite": None,  # in-process, no service to start
    "surrealdb": "surrealdb",
    "postgres": "postgres",
}


def get_service(name: str) -> ServiceDefinition | None:
    """Look up a service definition by name."""
    return _SERVICES_BY_NAME.get(name)


def resolve_startup_order(
    embedding_provider: str,
    storage: str,
    enabled_services: list[str],
) -> list[ServiceDefinition]:
    """Resolve which services to start and in what order.

    Returns services in dependency order: embedding → storage → server → additive.
    """
    to_start: list[str] = []
    seen: set[str] = set()

    def _add(name: str) -> None:
        if name in seen:
            return
        svc = get_service(name)
        if svc is None:
            return
        # Add dependencies first
        for dep in svc.depends_on:
            if dep == "_active_embedding":
                _add(EMBEDDING_PROVIDERS.get(embedding_provider, "corvia-inference"))
            else:
                _add(dep)
        if name not in seen:
            seen.add(name)
            to_start.append(name)

    # 1. Active embedding provider
    emb_service = EMBEDDING_PROVIDERS.get(embedding_provider, "corvia-inference")
    _add(emb_service)

    # 2. Active storage backend (if not lite)
    storage_service = STORAGE_BACKENDS.get(storage)
    if storage_service:
        _add(storage_service)

    # 3. corvia-server
    _add("corvia-server")

    # 4. Additive enabled services
    for svc_name in enabled_services:
        svc = get_service(svc_name)
        if svc is None:
            continue
        # Resolve dependencies (e.g., coding-llm → ollama)
        _add(svc_name)

    return [_SERVICES_BY_NAME[name] for name in to_start if name in _SERVICES_BY_NAME]
```

**Step 4: Run test to verify it passes**

Run: `cd /workspaces/corvia-workspace && python3 -m pytest tools/corvia-dev/tests/test_services.py -v`
Expected: All 7 tests PASS

**Step 5: Commit**

```bash
git add tools/corvia-dev/corvia_dev/services.py tools/corvia-dev/tests/test_services.py
git commit -m "feat(corvia-dev): add service registry with dependency resolution"
```

---

### Task 4: Config Reader & Mutator

**Files:**
- Create: `tools/corvia-dev/corvia_dev/config.py`
- Create: `tools/corvia-dev/tests/test_config.py`

**Step 1: Write the failing test**

```python
"""Tests for corvia.toml config read/mutate/write."""

import tomllib
from pathlib import Path
from textwrap import dedent

from corvia_dev.config import load_config, use_provider, read_enabled_services, set_enabled_service


SAMPLE_TOML = dedent("""\
    [project]
    name = "test"
    scope_id = "test"

    [workspace]
    repos_dir = "repos"

    [[workspace.repos]]
    name = "corvia"
    url = "https://github.com/chunzhe10/corvia"
    namespace = "kernel"

    [storage]
    store_type = "lite"
    data_dir = ".corvia"

    [embedding]
    provider = "corvia"
    model = "nomic-embed-text-v1.5"
    url = "http://127.0.0.1:8030"
    dimensions = 768

    [server]
    host = "127.0.0.1"
    port = 8020
""")


def test_load_config(tmp_path: Path) -> None:
    toml_path = tmp_path / "corvia.toml"
    toml_path.write_text(SAMPLE_TOML)
    cfg = load_config(toml_path)
    assert cfg.embedding_provider == "corvia"
    assert cfg.merge_provider == "corvia"  # defaults when [merge] absent
    assert cfg.storage == "lite"
    assert cfg.workspace_name == "test"


def test_use_ollama(tmp_path: Path) -> None:
    toml_path = tmp_path / "corvia.toml"
    toml_path.write_text(SAMPLE_TOML)
    use_provider("ollama", toml_path)

    # Re-read and verify
    raw = tomllib.loads(toml_path.read_text())
    assert raw["embedding"]["provider"] == "ollama"
    assert raw["embedding"]["url"] == "http://127.0.0.1:11434"
    # Model and dimensions preserved
    assert raw["embedding"]["model"] == "nomic-embed-text-v1.5"
    assert raw["embedding"]["dimensions"] == 768
    # Merge section created
    assert raw["merge"]["provider"] == "ollama"
    assert raw["merge"]["url"] == "http://127.0.0.1:11434"
    # Unrelated sections preserved
    assert raw["project"]["name"] == "test"
    assert raw["server"]["port"] == 8020
    assert len(raw["workspace"]["repos"]) == 1


def test_use_corvia_inference(tmp_path: Path) -> None:
    toml_path = tmp_path / "corvia.toml"
    toml_path.write_text(SAMPLE_TOML)
    # First switch to ollama, then back to corvia
    use_provider("ollama", toml_path)
    use_provider("corvia-inference", toml_path)

    raw = tomllib.loads(toml_path.read_text())
    assert raw["embedding"]["provider"] == "corvia"
    assert raw["embedding"]["url"] == "http://127.0.0.1:8030"
    assert raw["merge"]["provider"] == "corvia"
    assert raw["merge"]["url"] == "http://127.0.0.1:8030"


def test_use_invalid_provider(tmp_path: Path) -> None:
    toml_path = tmp_path / "corvia.toml"
    toml_path.write_text(SAMPLE_TOML)
    try:
        use_provider("invalid", toml_path)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "invalid" in str(e).lower()


def test_enabled_services_empty(tmp_path: Path) -> None:
    flags_path = tmp_path / ".corvia-workspace-flags"
    services = read_enabled_services(flags_path)
    assert services == []


def test_enabled_services_roundtrip(tmp_path: Path) -> None:
    flags_path = tmp_path / ".corvia-workspace-flags"
    set_enabled_service("coding-llm", True, flags_path)
    set_enabled_service("surrealdb", True, flags_path)
    services = read_enabled_services(flags_path)
    assert sorted(services) == ["coding-llm", "surrealdb"]

    set_enabled_service("surrealdb", False, flags_path)
    services = read_enabled_services(flags_path)
    assert services == ["coding-llm"]
```

**Step 2: Run test to verify it fails**

Run: `cd /workspaces/corvia-workspace && python3 -m pytest tools/corvia-dev/tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write the implementation**

```python
"""Config reader and mutator for corvia.toml."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

import tomli_w


PROVIDER_MAP: dict[str, dict[str, str]] = {
    "ollama": {
        "provider": "ollama",
        "url": "http://127.0.0.1:11434",
    },
    "corvia-inference": {
        "provider": "corvia",
        "url": "http://127.0.0.1:8030",
    },
    "vllm": {
        "provider": "vllm",
        "url": "http://127.0.0.1:8000",
    },
}

VALID_PROVIDERS = set(PROVIDER_MAP.keys())


@dataclass(frozen=True)
class ConfigSnapshot:
    """Read-only snapshot of the fields we care about."""

    embedding_provider: str
    embedding_url: str
    merge_provider: str
    merge_url: str
    storage: str
    workspace_name: str


def load_config(path: Path) -> ConfigSnapshot:
    """Load corvia.toml and extract the fields we need."""
    raw = tomllib.loads(path.read_text())
    embedding = raw.get("embedding", {})
    merge = raw.get("merge", {})
    storage = raw.get("storage", {})
    project = raw.get("project", {})
    return ConfigSnapshot(
        embedding_provider=embedding.get("provider", "corvia"),
        embedding_url=embedding.get("url", "http://127.0.0.1:8030"),
        merge_provider=merge.get("provider", embedding.get("provider", "corvia")),
        merge_url=merge.get("url", embedding.get("url", "http://127.0.0.1:8030")),
        storage=storage.get("store_type", "lite"),
        workspace_name=project.get("name", "unknown"),
    )


def use_provider(provider: str, path: Path) -> None:
    """Switch embedding and merge providers in corvia.toml.

    Preserves all other fields. Creates [merge] section if absent.
    """
    if provider not in VALID_PROVIDERS:
        raise ValueError(
            f"Unknown provider '{provider}'. Valid: {sorted(VALID_PROVIDERS)}"
        )

    mapping = PROVIDER_MAP[provider]
    raw = tomllib.loads(path.read_text())

    # Mutate embedding
    if "embedding" not in raw:
        raw["embedding"] = {}
    raw["embedding"]["provider"] = mapping["provider"]
    raw["embedding"]["url"] = mapping["url"]

    # Mutate merge (create if absent)
    if "merge" not in raw:
        raw["merge"] = {}
    raw["merge"]["provider"] = mapping["provider"]
    raw["merge"]["url"] = mapping["url"]

    path.write_text(tomli_w.dumps(raw))


def read_enabled_services(flags_path: Path) -> list[str]:
    """Read the list of enabled optional services from the flags file."""
    if not flags_path.exists():
        return []
    enabled: list[str] = []
    for line in flags_path.read_text().splitlines():
        line = line.strip()
        if "=" in line:
            name, value = line.split("=", 1)
            if value.strip() == "enabled":
                enabled.append(name.strip())
    return enabled


def set_enabled_service(name: str, enabled: bool, flags_path: Path) -> None:
    """Set a service as enabled or disabled in the flags file."""
    # Read existing flags
    flags: dict[str, str] = {}
    if flags_path.exists():
        for line in flags_path.read_text().splitlines():
            line = line.strip()
            if "=" in line:
                k, v = line.split("=", 1)
                flags[k.strip()] = v.strip()

    if enabled:
        flags[name] = "enabled"
    else:
        flags[name] = "disabled"

    # Write back
    lines = [f"{k}={v}" for k, v in sorted(flags.items())]
    flags_path.write_text("\n".join(lines) + "\n")
```

**Step 4: Run test to verify it passes**

Run: `cd /workspaces/corvia-workspace && python3 -m pytest tools/corvia-dev/tests/test_config.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add tools/corvia-dev/corvia_dev/config.py tools/corvia-dev/tests/test_config.py
git commit -m "feat(corvia-dev): add config reader/mutator with TOML round-tripping"
```

---

### Task 5: Health Checker

**Files:**
- Create: `tools/corvia-dev/corvia_dev/health.py`
- Create: `tools/corvia-dev/tests/test_health.py`

**Step 1: Write the failing test**

```python
"""Tests for health checking."""

import http.server
import threading
from unittest.mock import patch

from corvia_dev.health import check_http, check_service
from corvia_dev.models import ServiceDefinition


def test_check_http_unreachable() -> None:
    """Health check against a port with nothing listening."""
    result = check_http("127.0.0.1", 19999, "/health", timeout=1.0)
    assert result.healthy is False
    assert result.latency_ms < 0


def test_check_http_healthy() -> None:
    """Health check against a real HTTP server."""
    handler = http.server.BaseHTTPRequestHandler

    class OkHandler(handler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *args: object) -> None:
            pass  # suppress output

    server = http.server.HTTPServer(("127.0.0.1", 0), OkHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()

    result = check_http("127.0.0.1", port, "/health", timeout=2.0)
    assert result.healthy is True
    assert result.latency_ms >= 0
    server.server_close()


def test_check_service_no_port() -> None:
    """Virtual services (like coding-llm) with no port are always 'unknown'."""
    svc = ServiceDefinition(name="coding-llm", tier=2, port=None, depends_on=["ollama"])
    result = check_service(svc, timeout=1.0)
    assert result.healthy is None  # indeterminate
```

**Step 2: Run test to verify it fails**

Run: `cd /workspaces/corvia-workspace && python3 -m pytest tools/corvia-dev/tests/test_health.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write the implementation**

```python
"""Health checking for services."""

from __future__ import annotations

import time
import urllib.request
import urllib.error
from dataclasses import dataclass

from corvia_dev.models import ServiceDefinition


@dataclass
class HealthResult:
    """Result of a health check."""

    healthy: bool | None  # None = indeterminate (no port)
    latency_ms: float  # -1 if unhealthy/indeterminate


def check_http(host: str, port: int, path: str, timeout: float = 3.0) -> HealthResult:
    """Check health via HTTP GET. Returns HealthResult, never raises."""
    url = f"http://{host}:{port}{path}"
    start = time.monotonic()
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout):
            elapsed = (time.monotonic() - start) * 1000
            return HealthResult(healthy=True, latency_ms=round(elapsed, 1))
    except (urllib.error.URLError, OSError, TimeoutError):
        return HealthResult(healthy=False, latency_ms=-1)


def check_service(svc: ServiceDefinition, timeout: float = 3.0) -> HealthResult:
    """Check health of a service. Dispatches to appropriate check method."""
    if svc.port is None:
        return HealthResult(healthy=None, latency_ms=-1)
    return check_http("127.0.0.1", svc.port, svc.health_path, timeout=timeout)
```

**Step 4: Run test to verify it passes**

Run: `cd /workspaces/corvia-workspace && python3 -m pytest tools/corvia-dev/tests/test_health.py -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add tools/corvia-dev/corvia_dev/health.py tools/corvia-dev/tests/test_health.py
git commit -m "feat(corvia-dev): add HTTP health checker for services"
```

---

### Task 6: Process Manager (Core Supervision)

**Files:**
- Create: `tools/corvia-dev/corvia_dev/manager.py`
- Create: `tools/corvia-dev/tests/test_manager.py`

**Step 1: Write the failing test**

```python
"""Tests for process manager."""

import asyncio
import json
import signal
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

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
    assert mp.backoff_s == 60.0  # capped


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
```

**Step 2: Run test to verify it fails**

Run: `cd /workspaces/corvia-workspace && python3 -m pytest tools/corvia-dev/tests/test_manager.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write the implementation**

```python
"""Process manager for service supervision."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import time
from pathlib import Path

from corvia_dev.health import check_service
from corvia_dev.models import (
    ConfigSummary,
    ManagerStatus,
    ServiceDefinition,
    ServiceState,
    ServiceStatus,
    StatusResponse,
)

logger = logging.getLogger(__name__)

MAX_BACKOFF = 60.0
BACKOFF_RESET_AFTER = 300.0  # reset after 5 min stable
HEALTH_CHECK_INTERVAL = 5.0
DEFAULT_STATE_PATH = Path("/tmp/corvia-dev-state.json")


class ManagedProcess:
    """Tracks a single managed child process."""

    def __init__(self, service: ServiceDefinition) -> None:
        self.service = service
        self.state = ServiceState.STOPPED
        self.pid: int | None = None
        self.process: asyncio.subprocess.Process | None = None
        self.restart_count: int = 0
        self.backoff_s: float = 1.0
        self.started_at: float | None = None

    def escalate_backoff(self) -> None:
        self.backoff_s = min(self.backoff_s * 2, MAX_BACKOFF)

    def reset_backoff(self) -> None:
        self.backoff_s = 1.0
        self.restart_count = 0

    @property
    def uptime_s(self) -> float | None:
        if self.started_at is None:
            return None
        return time.time() - self.started_at

    def to_status(self) -> ServiceStatus:
        return ServiceStatus(
            name=self.service.name,
            state=self.state,
            port=self.service.port,
            pid=self.pid,
            uptime_s=round(self.uptime_s, 1) if self.uptime_s is not None else None,
        )


class ProcessManager:
    """Manages multiple child processes with health checks and restart."""

    def __init__(
        self,
        services: list[ServiceDefinition],
        workspace_root: Path,
        state_path: Path = DEFAULT_STATE_PATH,
        config_summary: ConfigSummary | None = None,
        enabled_services: list[str] | None = None,
    ) -> None:
        self.processes: dict[str, ManagedProcess] = {
            svc.name: ManagedProcess(svc) for svc in services
        }
        self.workspace_root = workspace_root
        self.state_path = state_path
        self.config_summary = config_summary or ConfigSummary(
            embedding_provider="unknown",
            merge_provider="unknown",
            storage="unknown",
            workspace="unknown",
        )
        self.enabled_services = enabled_services or []
        self._started_at = time.time()
        self._running = False
        self._log_lines: list[str] = []

    def _log(self, msg: str) -> None:
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        line = f"{ts} {msg}"
        self._log_lines.append(line)
        # Keep last 100 lines
        if len(self._log_lines) > 100:
            self._log_lines = self._log_lines[-100:]
        logger.info(msg)

    async def start_service(self, name: str) -> None:
        """Start a single service process."""
        mp = self.processes.get(name)
        if mp is None:
            return

        if not mp.service.start_cmd:
            # Virtual service (e.g., coding-llm) — mark as healthy
            mp.state = ServiceState.HEALTHY
            mp.started_at = time.time()
            self._log(f"{name}: virtual service marked healthy")
            return

        mp.state = ServiceState.STARTING
        self._log(f"{name}: starting ({' '.join(mp.service.start_cmd)})")

        try:
            proc = await asyncio.create_subprocess_exec(
                *mp.service.start_cmd,
                cwd=str(self.workspace_root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            mp.process = proc
            mp.pid = proc.pid
            mp.started_at = time.time()
            mp.state = ServiceState.STARTING
            self._log(f"{name}: started (pid {proc.pid})")
        except FileNotFoundError:
            mp.state = ServiceState.CRASHED
            self._log(f"{name}: binary not found")
        except OSError as e:
            mp.state = ServiceState.CRASHED
            self._log(f"{name}: failed to start: {e}")

    async def stop_service(self, name: str) -> None:
        """Stop a single service process."""
        mp = self.processes.get(name)
        if mp is None or mp.process is None:
            return

        self._log(f"{name}: stopping (pid {mp.pid})")
        try:
            mp.process.terminate()
            try:
                await asyncio.wait_for(mp.process.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                mp.process.kill()
                await mp.process.wait()
        except ProcessLookupError:
            pass  # already dead

        mp.state = ServiceState.STOPPED
        mp.pid = None
        mp.process = None
        mp.started_at = None
        self._log(f"{name}: stopped")

    async def health_check_all(self) -> None:
        """Run health checks on all managed processes."""
        for name, mp in self.processes.items():
            if mp.state in (ServiceState.STOPPED,):
                continue

            if mp.process is not None and mp.process.returncode is not None:
                # Process exited
                mp.state = ServiceState.CRASHED
                mp.pid = None
                self._log(f"{name}: crashed (exit code {mp.process.returncode})")
                mp.process = None
                continue

            result = check_service(mp.service)
            if result.healthy is True:
                if mp.state != ServiceState.HEALTHY:
                    self._log(f"{name}: now healthy")
                mp.state = ServiceState.HEALTHY
                # Reset backoff after stable running
                if mp.uptime_s is not None and mp.uptime_s > BACKOFF_RESET_AFTER:
                    mp.reset_backoff()
            elif result.healthy is False and mp.state == ServiceState.HEALTHY:
                mp.state = ServiceState.UNHEALTHY
                self._log(f"{name}: health check failed")
            elif result.healthy is None:
                pass  # indeterminate (no port), keep current state

    async def restart_crashed(self) -> None:
        """Restart any crashed services with backoff."""
        for name, mp in self.processes.items():
            if mp.state != ServiceState.CRASHED:
                continue
            if mp.service.tier == 2:
                self._log(f"{name}: tier 2 service crashed, skipping restart")
                mp.state = ServiceState.STOPPED
                continue

            mp.restart_count += 1
            self._log(f"{name}: restarting (attempt {mp.restart_count}, backoff {mp.backoff_s}s)")
            await asyncio.sleep(mp.backoff_s)
            mp.escalate_backoff()
            await self.start_service(name)

    def write_state(self) -> None:
        """Write current state to the state file as JSON."""
        resp = StatusResponse(
            manager=ManagerStatus(
                pid=os.getpid(),
                uptime_s=round(time.time() - self._started_at, 1),
                state="running" if self._running else "stopped",
            ),
            services=[mp.to_status() for mp in self.processes.values()],
            config=self.config_summary,
            enabled_services=self.enabled_services,
            logs=self._log_lines[-20:],
        )
        self.state_path.write_text(resp.model_dump_json(indent=2))

    async def run(self) -> None:
        """Main supervision loop. Runs until cancelled or SIGTERM."""
        self._running = True
        loop = asyncio.get_event_loop()

        # Handle SIGTERM/SIGINT
        stop_event = asyncio.Event()

        def _signal_handler() -> None:
            self._log("Received shutdown signal")
            stop_event.set()

        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, _signal_handler)

        # Start services in order
        for name in self.processes:
            await self.start_service(name)
            # Wait for health before starting dependents
            for _ in range(30):  # 30s max wait
                await asyncio.sleep(1)
                result = check_service(self.processes[name].service)
                if result.healthy is True or result.healthy is None:
                    self.processes[name].state = ServiceState.HEALTHY
                    break
            self.write_state()

        # Supervision loop
        while not stop_event.is_set():
            await self.health_check_all()
            await self.restart_crashed()
            self.write_state()
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=HEALTH_CHECK_INTERVAL)
            except asyncio.TimeoutError:
                pass

        # Shutdown: stop all services in reverse order
        self._log("Shutting down all services")
        for name in reversed(list(self.processes.keys())):
            await self.stop_service(name)
        self._running = False
        self.write_state()
```

**Step 4: Run test to verify it passes**

Run: `cd /workspaces/corvia-workspace && python3 -m pytest tools/corvia-dev/tests/test_manager.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add tools/corvia-dev/corvia_dev/manager.py tools/corvia-dev/tests/test_manager.py
git commit -m "feat(corvia-dev): add process manager with health checks and restart"
```

---

### Task 7: Wire Up CLI Commands

**Files:**
- Modify: `tools/corvia-dev/corvia_dev/cli.py`

**Step 1: Write the full CLI**

```python
"""CLI entry point for corvia-dev."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
from pathlib import Path

import click

from corvia_dev.config import (
    load_config,
    read_enabled_services,
    set_enabled_service,
    use_provider,
)
from corvia_dev.health import check_service
from corvia_dev.manager import DEFAULT_STATE_PATH, ProcessManager
from corvia_dev.models import ConfigSummary, ServiceState, StatusResponse
from corvia_dev.services import get_service, resolve_startup_order


def _workspace_root() -> Path:
    return Path(os.environ.get("CORVIA_WORKSPACE", os.getcwd()))


def _config_path() -> Path:
    return _workspace_root() / "corvia.toml"


def _flags_path() -> Path:
    return _workspace_root() / ".devcontainer" / ".corvia-workspace-flags"


@click.group()
def main() -> None:
    """Dev environment orchestration for corvia-workspace."""


@main.command()
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def status(as_json: bool) -> None:
    """Show service health and config summary."""
    # Try reading state file first (from running manager)
    if DEFAULT_STATE_PATH.exists():
        try:
            data = json.loads(DEFAULT_STATE_PATH.read_text())
            resp = StatusResponse.model_validate(data)

            # Check if manager is still alive
            if resp.manager and resp.manager.pid:
                try:
                    os.kill(resp.manager.pid, 0)
                except OSError:
                    resp.manager.state = "dead"

            if as_json:
                click.echo(resp.model_dump_json(indent=2))
                return

            _print_status_human(resp)
            return
        except (json.JSONDecodeError, ValueError):
            pass

    # No state file — build status from live checks
    cfg = load_config(_config_path())
    enabled = read_enabled_services(_flags_path())
    services = resolve_startup_order(cfg.embedding_provider, cfg.storage, enabled)

    service_statuses = []
    for svc in services:
        result = check_service(svc)
        state = (
            ServiceState.HEALTHY
            if result.healthy is True
            else ServiceState.STOPPED
            if result.healthy is None
            else ServiceState.UNHEALTHY
        )
        service_statuses.append({
            "name": svc.name,
            "state": state.value,
            "port": svc.port,
            "pid": None,
            "uptime_s": None,
        })

    resp = StatusResponse(
        manager=None,
        services=[StatusResponse.model_validate(s) if False else
                  type("_", (), s)  # handled below
                  for s in []],
        config=ConfigSummary(
            embedding_provider=cfg.embedding_provider,
            merge_provider=cfg.merge_provider,
            storage=cfg.storage,
            workspace=cfg.workspace_name,
        ),
        enabled_services=enabled,
        logs=[],
    )
    # Build properly
    from corvia_dev.models import ServiceStatus
    resp.services = [ServiceStatus(**s) for s in service_statuses]

    if as_json:
        click.echo(resp.model_dump_json(indent=2))
    else:
        _print_status_human(resp)


def _print_status_human(resp: StatusResponse) -> None:
    """Print status in human-readable format."""
    click.echo("=== corvia-dev status ===\n")

    if resp.manager:
        click.echo(f"Manager: {resp.manager.state} (pid {resp.manager.pid})")
    else:
        click.echo("Manager: not running")

    click.echo(f"\nConfig:")
    click.echo(f"  Embedding: {resp.config.embedding_provider}")
    click.echo(f"  Merge:     {resp.config.merge_provider}")
    click.echo(f"  Storage:   {resp.config.storage}")
    click.echo(f"  Workspace: {resp.config.workspace}")

    click.echo(f"\nServices:")
    for svc in resp.services:
        icon = {"healthy": "+", "unhealthy": "!", "stopped": "-", "starting": "~", "crashed": "x"}.get(svc.state, "?")
        pid_str = f" (pid {svc.pid})" if svc.pid else ""
        port_str = f" :{svc.port}" if svc.port else ""
        click.echo(f"  [{icon}] {svc.name}{port_str}{pid_str} — {svc.state}")

    if resp.enabled_services:
        click.echo(f"\nEnabled: {', '.join(resp.enabled_services)}")

    if resp.logs:
        click.echo(f"\nRecent logs:")
        for line in resp.logs[-10:]:
            click.echo(f"  {line}")


@main.command()
@click.option("--no-foreground", is_flag=True, help="Run in background")
def up(no_foreground: bool) -> None:
    """Start and supervise all enabled services."""
    cfg = load_config(_config_path())
    enabled = read_enabled_services(_flags_path())
    services = resolve_startup_order(cfg.embedding_provider, cfg.storage, enabled)

    config_summary = ConfigSummary(
        embedding_provider=cfg.embedding_provider,
        merge_provider=cfg.merge_provider,
        storage=cfg.storage,
        workspace=cfg.workspace_name,
    )

    mgr = ProcessManager(
        services=services,
        workspace_root=_workspace_root(),
        config_summary=config_summary,
        enabled_services=enabled,
    )

    if no_foreground:
        # Fork to background
        pid = os.fork()
        if pid > 0:
            click.echo(f"corvia-dev manager started (pid {pid})")
            return
        # Child: detach
        os.setsid()
        sys.stdin.close()

    click.echo(f"Starting {len(services)} services...")
    asyncio.run(mgr.run())


@main.command()
def down() -> None:
    """Stop all managed services."""
    if not DEFAULT_STATE_PATH.exists():
        click.echo("No running manager found.")
        return

    try:
        data = json.loads(DEFAULT_STATE_PATH.read_text())
        resp = StatusResponse.model_validate(data)
        if resp.manager and resp.manager.pid:
            os.kill(resp.manager.pid, signal.SIGTERM)
            click.echo(f"Sent SIGTERM to manager (pid {resp.manager.pid})")
        else:
            click.echo("Manager PID not found in state file.")
    except (json.JSONDecodeError, ValueError, ProcessLookupError) as e:
        click.echo(f"Failed to stop manager: {e}")


@main.command()
@click.argument("provider", type=click.Choice(["ollama", "corvia-inference", "vllm"]))
def use(provider: str) -> None:
    """Switch corvia's embedding and merge provider."""
    path = _config_path()
    if not path.exists():
        click.echo(f"Error: {path} not found", err=True)
        raise SystemExit(1)

    use_provider(provider, path)
    click.echo(f"Switched to {provider}")
    click.echo(f"  embedding.provider = {provider}")
    click.echo(f"  merge.provider = {provider}")
    click.echo("\nRestart corvia-server for changes to take effect:")
    click.echo("  corvia-dev restart corvia-server")


@main.command()
@click.argument("service", type=click.Choice(["coding-llm", "surrealdb", "postgres"]))
def enable(service: str) -> None:
    """Enable and start an optional service."""
    set_enabled_service(service, True, _flags_path())
    click.echo(f"Enabled {service}")

    svc = get_service(service)
    if svc and svc.depends_on:
        for dep in svc.depends_on:
            dep_svc = get_service(dep)
            if dep_svc:
                result = check_service(dep_svc)
                if not result.healthy:
                    click.echo(f"  Note: dependency '{dep}' is not running")

    click.echo("Run 'corvia-dev up' to start all enabled services.")


@main.command()
@click.argument("service", type=click.Choice(["coding-llm", "surrealdb", "postgres"]))
def disable(service: str) -> None:
    """Disable an optional service."""
    set_enabled_service(service, False, _flags_path())
    click.echo(f"Disabled {service}")


@main.command()
@click.argument("service", required=False)
def restart(service: str | None) -> None:
    """Restart a service or all services."""
    if not DEFAULT_STATE_PATH.exists():
        click.echo("No running manager. Use 'corvia-dev up' to start.")
        return

    # Send SIGHUP to manager (could be extended to restart individual services)
    try:
        data = json.loads(DEFAULT_STATE_PATH.read_text())
        resp = StatusResponse.model_validate(data)
        if resp.manager and resp.manager.pid:
            # For now, full restart via stop + start
            os.kill(resp.manager.pid, signal.SIGTERM)
            click.echo("Restarting all services...")
            import time
            time.sleep(2)
            # Re-invoke up
            ctx = click.get_current_context()
            ctx.invoke(up, no_foreground=True)
    except (json.JSONDecodeError, ValueError, ProcessLookupError) as e:
        click.echo(f"Failed: {e}")


@main.command()
def config() -> None:
    """Show current corvia.toml config summary."""
    cfg = load_config(_config_path())
    click.echo(f"Embedding: {cfg.embedding_provider} ({cfg.embedding_url})")
    click.echo(f"Merge:     {cfg.merge_provider} ({cfg.merge_url})")
    click.echo(f"Storage:   {cfg.storage}")
    click.echo(f"Workspace: {cfg.workspace_name}")


@main.command()
@click.argument("service", required=False)
@click.option("--tail", "-n", default=20, help="Number of log lines")
def logs(service: str | None, tail: int) -> None:
    """Show recent logs."""
    if not DEFAULT_STATE_PATH.exists():
        click.echo("No state file found. Manager may not be running.")
        return

    try:
        data = json.loads(DEFAULT_STATE_PATH.read_text())
        resp = StatusResponse.model_validate(data)
        for line in resp.logs[-tail:]:
            click.echo(line)
    except (json.JSONDecodeError, ValueError) as e:
        click.echo(f"Failed to read logs: {e}")


if __name__ == "__main__":
    main()
```

**Step 2: Run the CLI to verify all commands register**

Run:
```bash
corvia-dev --help
corvia-dev status --help
corvia-dev up --help
corvia-dev use --help
corvia-dev enable --help
```

Expected: All commands show help text with correct options.

**Step 3: Test `status` and `config` against live workspace**

Run:
```bash
cd /workspaces/corvia-workspace
corvia-dev config
corvia-dev status
corvia-dev status --json
```

Expected: `config` shows embedding/merge/storage info from `corvia.toml`. `status` shows service health. `--json` outputs valid JSON.

**Step 4: Commit**

```bash
git add tools/corvia-dev/corvia_dev/cli.py
git commit -m "feat(corvia-dev): wire up all CLI commands (status, up, down, use, enable, disable)"
```

---

### Task 8: Update Devcontainer Integration

**Files:**
- Modify: `.devcontainer/scripts/post-create.sh`
- Modify: `.devcontainer/scripts/post-start.sh`

**Step 1: Update post-create.sh to install corvia-dev**

At the end of `post-create.sh`, replace the `corvia-workspace` symlink with pip install:

Replace:
```bash
# Install corvia-workspace toggle command
chmod +x "$WORKSPACE_ROOT/.devcontainer/scripts/corvia-workspace.sh"
ln -sf "$WORKSPACE_ROOT/.devcontainer/scripts/corvia-workspace.sh" "$INSTALL_DIR/corvia-workspace"
```

With:
```bash
# Install corvia-dev CLI
python3 -m pip install -e "$WORKSPACE_ROOT/tools/corvia-dev" --quiet
```

Update help text at the end:
```bash
echo "=== Post-Create Complete ==="
echo "Run 'corvia-dev status' to see available services."
echo "Run 'corvia-dev use ollama' to switch to Ollama embeddings."
echo "Run 'corvia-dev enable coding-llm' to enable local coding LLM."
echo "Run 'corvia-dev enable surrealdb' to enable SurrealDB FullStore."
```

**Step 2: Update post-start.sh to use corvia-dev**

Replace the supervisor startup block and service restart logic with:

```bash
# Start corvia-dev manager (replaces corvia-supervisor.sh)
if corvia-dev status --json 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); exit(0 if d.get('manager',{}).get('state')=='running' else 1)" 2>/dev/null; then
    echo "corvia-dev manager already running"
else
    corvia-dev up --no-foreground
    sleep 2
    echo "corvia-dev manager started"
fi
```

Keep the MCP registration and VS Code extension install sections unchanged.

**Step 3: Verify post-start.sh still works**

Run: `bash .devcontainer/scripts/post-start.sh`
Expected: Server starts via corvia-dev manager, MCP registration succeeds.

**Step 4: Commit**

```bash
git add .devcontainer/scripts/post-create.sh .devcontainer/scripts/post-start.sh
git commit -m "feat(corvia-dev): integrate with devcontainer lifecycle scripts"
```

---

### Task 9: Rewrite VS Code Extension (Thin Skin)

**Files:**
- Modify: `.devcontainer/extensions/corvia-services/extension.js`
- Modify: `.devcontainer/extensions/corvia-services/package.json`

**Step 1: Rewrite extension.js**

Replace the entire 818-line extension with a thin ~200-line version that consumes `corvia-dev status --json`:

```javascript
const vscode = require("vscode");
const { exec } = require("child_process");

let statusBarItem;
let panel;
let pollTimer;

const POLL_INTERVAL = 10000;

function activate(context) {
    statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 50);
    statusBarItem.command = "corvia.openDashboard";
    statusBarItem.text = "$(loading~spin) Corvia";
    statusBarItem.show();
    context.subscriptions.push(statusBarItem);

    context.subscriptions.push(
        vscode.commands.registerCommand("corvia.openDashboard", () => openDashboard(context))
    );

    refresh();
    pollTimer = setInterval(refresh, POLL_INTERVAL);
    context.subscriptions.push({ dispose: () => clearInterval(pollTimer) });
}

function run(cmd) {
    return new Promise((resolve) => {
        exec(cmd, { timeout: 8000 }, (err, stdout) => {
            resolve(err ? null : stdout);
        });
    });
}

async function refresh() {
    const raw = await run("corvia-dev status --json");
    if (!raw) {
        statusBarItem.text = "$(error) Corvia";
        statusBarItem.backgroundColor = new vscode.ThemeColor("statusBarItem.errorBackground");
        statusBarItem.tooltip = "corvia-dev not responding";
        if (panel) panel.webview.postMessage({ type: "status", data: null });
        return;
    }

    let data;
    try {
        data = JSON.parse(raw);
    } catch {
        statusBarItem.text = "$(error) Corvia";
        return;
    }

    const tier0 = (data.services || []).filter(
        (s) => ["corvia-inference", "corvia-server"].includes(s.name)
    );
    const allHealthy = tier0.every((s) => s.state === "healthy");
    const anyDown = tier0.some((s) => s.state !== "healthy");

    if (allHealthy) {
        statusBarItem.text = "$(check) Corvia";
        statusBarItem.backgroundColor = undefined;
    } else if (anyDown) {
        statusBarItem.text = "$(error) Corvia";
        statusBarItem.backgroundColor = new vscode.ThemeColor("statusBarItem.errorBackground");
    } else {
        statusBarItem.text = "$(warning) Corvia";
        statusBarItem.backgroundColor = new vscode.ThemeColor("statusBarItem.warningBackground");
    }

    const svcSummary = (data.services || [])
        .map((s) => `${s.name}: ${s.state}`)
        .join(" | ");
    statusBarItem.tooltip = svcSummary;

    if (panel) {
        panel.webview.postMessage({ type: "status", data });
    }
}

function openDashboard(context) {
    if (panel) {
        panel.reveal();
        return;
    }

    panel = vscode.window.createWebviewPanel(
        "corviaDashboard",
        "Corvia Dashboard",
        vscode.ViewColumn.One,
        { enableScripts: true }
    );

    panel.webview.html = getDashboardHtml();

    panel.webview.onDidReceiveMessage((msg) => {
        if (msg.type === "command") {
            const terminal = vscode.window.createTerminal("corvia-dev");
            terminal.show();
            terminal.sendText(msg.command);
        } else if (msg.type === "refresh") {
            refresh();
        }
    });

    panel.onDidDispose(() => { panel = undefined; });
    refresh();
}

function getDashboardHtml() {
    return `<!DOCTYPE html>
<html>
<head>
<style>
    body { font-family: var(--vscode-font-family); color: var(--vscode-foreground); padding: 16px; }
    .card { border: 1px solid var(--vscode-panel-border); border-radius: 6px; padding: 12px; margin: 8px 0; }
    .healthy { border-left: 4px solid #4caf50; }
    .unhealthy, .crashed { border-left: 4px solid #f44336; }
    .stopped { border-left: 4px solid #666; }
    .starting { border-left: 4px solid #ff9800; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 8px; }
    .name { font-weight: bold; }
    .state { opacity: 0.8; font-size: 0.9em; }
    .actions { margin-top: 16px; }
    button {
        background: var(--vscode-button-background); color: var(--vscode-button-foreground);
        border: none; padding: 6px 14px; border-radius: 4px; cursor: pointer; margin: 4px;
    }
    button:hover { background: var(--vscode-button-hoverBackground); }
    .config { opacity: 0.8; margin-top: 12px; }
    .logs { margin-top: 12px; font-family: monospace; font-size: 0.85em; white-space: pre-wrap; max-height: 200px; overflow-y: auto; }
    h2 { margin: 16px 0 8px; }
    .none { opacity: 0.5; }
</style>
</head>
<body>
<h1>Corvia Dashboard</h1>
<div id="content"><p class="none">Loading...</p></div>
<script>
    const vscode = acquireVsCodeApi();

    window.addEventListener("message", (e) => {
        if (e.data.type === "status") render(e.data.data);
    });

    function render(data) {
        const el = document.getElementById("content");
        if (!data) { el.innerHTML = '<p class="none">corvia-dev not responding</p>'; return; }

        let html = "<h2>Services</h2><div class='grid'>";
        for (const svc of data.services || []) {
            const port = svc.port ? ":" + svc.port : "";
            const pid = svc.pid ? " pid " + svc.pid : "";
            html += '<div class="card ' + svc.state + '">';
            html += '<div class="name">' + svc.name + port + '</div>';
            html += '<div class="state">' + svc.state + pid + '</div>';
            html += "</div>";
        }
        html += "</div>";

        html += '<h2>Config</h2><div class="config">';
        html += "Embedding: " + data.config.embedding_provider + "<br>";
        html += "Merge: " + data.config.merge_provider + "<br>";
        html += "Storage: " + data.config.storage + "</div>";

        html += '<h2>Actions</h2><div class="actions">';
        html += btn("Status", "corvia-dev status");
        html += btn("Use Ollama", "corvia-dev use ollama");
        html += btn("Use Corvia-Inference", "corvia-dev use corvia-inference");
        html += btn("Enable coding-llm", "corvia-dev enable coding-llm");
        html += btn("Restart", "corvia-dev restart");
        html += btn("Refresh", null, "refresh");
        html += "</div>";

        if (data.logs && data.logs.length) {
            html += '<h2>Logs</h2><div class="logs">' + data.logs.join("\\n") + "</div>";
        }

        el.innerHTML = html;
        el.querySelectorAll("[data-cmd]").forEach((b) => {
            b.onclick = () => vscode.postMessage({ type: "command", command: b.dataset.cmd });
        });
        el.querySelectorAll("[data-action]").forEach((b) => {
            b.onclick = () => vscode.postMessage({ type: b.dataset.action });
        });
    }

    function btn(label, cmd, action) {
        if (action) return '<button data-action="' + action + '">' + label + "</button>";
        return '<button data-cmd="' + cmd + '">' + label + "</button>";
    }
</script>
</body>
</html>`;
}

function deactivate() {
    if (pollTimer) clearInterval(pollTimer);
}

module.exports = { activate, deactivate };
```

**Step 2: Rebuild the .vsix**

Run:
```bash
cd .devcontainer/extensions/corvia-services
# If vsce is available:
# vsce package --no-dependencies -o corvia-services.vsix
# Otherwise just update the raw files — the extension is loaded from source
```

**Step 3: Verify extension loads without errors**

Open VS Code, check that the status bar item appears and the dashboard opens.

**Step 4: Commit**

```bash
git add .devcontainer/extensions/corvia-services/
git commit -m "refactor(extension): rewrite as thin skin consuming corvia-dev status --json"
```

---

### Task 10: Delete Old Scripts & Final Cleanup

**Files:**
- Delete: `.devcontainer/scripts/corvia-workspace.sh`
- Delete: `.devcontainer/scripts/corvia-supervisor.sh`

**Step 1: Remove old scripts**

```bash
git rm .devcontainer/scripts/corvia-workspace.sh
git rm .devcontainer/scripts/corvia-supervisor.sh
```

**Step 2: Remove the symlink if it exists**

```bash
rm -f /usr/local/bin/corvia-workspace
```

**Step 3: Verify corvia-dev works end-to-end**

Run:
```bash
corvia-dev config
corvia-dev status
corvia-dev status --json
corvia-dev use ollama
corvia-dev config  # verify it switched
corvia-dev use corvia-inference  # switch back
corvia-dev enable coding-llm
corvia-dev status
```

Expected: All commands work. Config mutation round-trips correctly. Status shows service health.

**Step 4: Run all tests**

Run: `cd /workspaces/corvia-workspace && python3 -m pytest tools/corvia-dev/tests/ -v`
Expected: All tests pass.

**Step 5: Commit**

```bash
git rm .devcontainer/scripts/corvia-workspace.sh .devcontainer/scripts/corvia-supervisor.sh
git add -A
git commit -m "chore: remove old bash scripts replaced by corvia-dev CLI"
```

---

### Summary

| Task | What | Files | Tests |
|------|------|-------|-------|
| 1 | Scaffolding | pyproject.toml, __init__.py, cli.py | manual |
| 2 | Pydantic models | models.py | 5 tests |
| 3 | Service registry | services.py | 7 tests |
| 4 | Config mutator | config.py | 6 tests |
| 5 | Health checker | health.py | 3 tests |
| 6 | Process manager | manager.py | 4 tests |
| 7 | CLI wiring | cli.py (rewrite) | manual |
| 8 | Devcontainer integration | post-create.sh, post-start.sh | manual |
| 9 | VS Code extension rewrite | extension.js | manual |
| 10 | Delete old scripts | remove 2 files | full suite |

Total: 10 tasks, ~25 tests, 10 commits.
