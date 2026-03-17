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
    assert cfg.merge_provider == "corvia"
    assert cfg.storage == "lite"
    assert cfg.workspace_name == "test"


def test_use_ollama(tmp_path: Path) -> None:
    toml_path = tmp_path / "corvia.toml"
    toml_path.write_text(SAMPLE_TOML)
    use_provider("ollama", toml_path)

    raw = tomllib.loads(toml_path.read_text())
    assert raw["embedding"]["provider"] == "ollama"
    assert raw["embedding"]["url"] == "http://127.0.0.1:11434"
    assert raw["embedding"]["model"] == "nomic-embed-text-v1.5"
    assert raw["embedding"]["dimensions"] == 768
    assert raw["merge"]["provider"] == "ollama"
    assert raw["merge"]["url"] == "http://127.0.0.1:11434"
    assert raw["project"]["name"] == "test"
    assert raw["server"]["port"] == 8020
    assert len(raw["workspace"]["repos"]) == 1


def test_use_corvia_inference(tmp_path: Path) -> None:
    toml_path = tmp_path / "corvia.toml"
    toml_path.write_text(SAMPLE_TOML)
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
    set_enabled_service("postgres", True, flags_path)
    services = read_enabled_services(flags_path)
    assert sorted(services) == ["coding-llm", "postgres"]

    set_enabled_service("postgres", False, flags_path)
    services = read_enabled_services(flags_path)
    assert services == ["coding-llm"]
