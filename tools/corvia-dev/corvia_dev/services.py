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
        health_proto="grpc",
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
        name="postgres",
        tier=1,
        port=5432,
        health_path="",  # uses pg_isready instead of HTTP
        start_cmd=["docker", "compose", "-f", "repos/corvia/docker/docker-compose-pg.yml", "up", "-d"],
        depends_on=[],
        exclusive_group="storage",
    ),
    ServiceDefinition(
        name="corvia-dashboard",
        tier=0,
        port=8021,
        health_path="/",
        start_cmd=[
            "bash", "-c",
            "cd tools/corvia-dashboard && npx vite --port 8021 --host 0.0.0.0",
        ],
        depends_on=["corvia-server"],
    ),
    ServiceDefinition(
        name="playwright-mcp",
        tier=0,
        port=8050,
        health_proto="tcp",
        start_cmd=[
            "npx", "@playwright/mcp@latest",
            "--port", "8050",
            "--host", "0.0.0.0",
            "--allowed-hosts", "*",
            "--headless",
            "--no-sandbox",
            "--isolated",
        ],
        depends_on=[],
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

    # 4. corvia-dashboard (always starts after server)
    _add("corvia-dashboard")

    # 5. Shared MCP servers
    _add("playwright-mcp")

    # 6. Additive enabled services
    for svc_name in enabled_services:
        svc = get_service(svc_name)
        if svc is None:
            continue
        _add(svc_name)

    return [_SERVICES_BY_NAME[name] for name in to_start if name in _SERVICES_BY_NAME]
