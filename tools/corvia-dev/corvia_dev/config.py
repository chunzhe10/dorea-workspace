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

    lines = [f"{k}={v}" for k, v in sorted(flags.items())]
    flags_path.write_text("\n".join(lines) + "\n")
