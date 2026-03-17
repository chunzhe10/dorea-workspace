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
        elif svc.name in ("ollama", "vllm", "postgres"):
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
    order = resolve_startup_order(
        embedding_provider="corvia",
        storage="lite",
        enabled_services=[],
    )
    names = [s.name for s in order]
    assert names == ["corvia-inference", "corvia-server"]


def test_resolve_startup_order_ollama() -> None:
    order = resolve_startup_order(
        embedding_provider="ollama",
        storage="lite",
        enabled_services=[],
    )
    names = [s.name for s in order]
    assert names == ["ollama", "corvia-server"]
    assert "corvia-inference" not in names


def test_resolve_startup_order_coding_llm() -> None:
    order = resolve_startup_order(
        embedding_provider="corvia",
        storage="lite",
        enabled_services=["coding-llm"],
    )
    names = [s.name for s in order]
    assert "corvia-inference" in names
    assert "ollama" in names
    assert names.index("corvia-server") < names.index("ollama")


def test_resolve_startup_order_ollama_plus_coding_llm() -> None:
    order = resolve_startup_order(
        embedding_provider="ollama",
        storage="lite",
        enabled_services=["coding-llm"],
    )
    names = [s.name for s in order]
    assert names.count("ollama") == 1
    assert names.index("ollama") < names.index("corvia-server")
