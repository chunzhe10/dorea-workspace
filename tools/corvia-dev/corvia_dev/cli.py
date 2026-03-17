"""CLI entry point for corvia-dev."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import time
from pathlib import Path

import click

from corvia_dev.config import (
    load_config,
    read_enabled_services,
    set_enabled_service,
    use_provider,
)
from corvia_dev.health import check_service
from corvia_dev.manager import DEFAULT_STATE_PATH, LOG_DIR, ProcessManager, tail_any_log, tail_service_log
from corvia_dev.models import ConfigSummary, ServiceState, ServiceStatus, StatusResponse
from corvia_dev.rebuild import cargo_build, check_staleness, install_binaries, DEFAULT_INSTALL_DIR
from corvia_dev.services import get_service, resolve_startup_order
from corvia_dev.traces import collect_traces


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

            resp.stale_binaries = check_staleness(workspace_root=_workspace_root())

            if as_json:
                click.echo(resp.model_dump_json(indent=2))
                return

            _print_status_human(resp)
            return
        except (json.JSONDecodeError, ValueError):
            pass

    # No state file — build status from live checks
    config_path = _config_path()
    if not config_path.exists():
        click.echo("Error: corvia.toml not found", err=True)
        raise SystemExit(1)

    cfg = load_config(config_path)
    enabled = read_enabled_services(_flags_path())
    services = resolve_startup_order(cfg.embedding_provider, cfg.storage, enabled)

    service_statuses: list[ServiceStatus] = []
    for svc in services:
        result = check_service(svc)
        if result.healthy is True:
            state = ServiceState.HEALTHY
        elif result.healthy is None:
            state = ServiceState.STOPPED
        else:
            state = ServiceState.UNHEALTHY
        service_statuses.append(ServiceStatus(
            name=svc.name,
            state=state,
            port=svc.port,
        ))

    # Collect per-service logs from log files (available even without manager)
    svc_logs: dict[str, list[str]] = {}
    for svc in services:
        lines = tail_service_log(svc.name, 30)
        if lines:
            svc_logs[svc.name] = lines

    # Also check legacy supervisor log
    supervisor_log = Path("/tmp/corvia-supervisor.log")
    sup_lines = tail_any_log(supervisor_log, 30)
    if sup_lines:
        svc_logs["supervisor"] = sup_lines

    resp = StatusResponse(
        manager=None,
        services=service_statuses,
        config=ConfigSummary(
            embedding_provider=cfg.embedding_provider,
            merge_provider=cfg.merge_provider,
            storage=cfg.storage,
            workspace=cfg.workspace_name,
        ),
        enabled_services=enabled,
        logs=[],
        service_logs=svc_logs,
        traces=collect_traces(LOG_DIR),
    )

    resp.stale_binaries = check_staleness(workspace_root=_workspace_root())

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
        state_val = svc.state.value if hasattr(svc.state, "value") else str(svc.state)
        icon = {
            "healthy": "+",
            "unhealthy": "!",
            "stopped": "-",
            "starting": "~",
            "crashed": "x",
        }.get(state_val, "?")
        pid_str = f" (pid {svc.pid})" if svc.pid else ""
        port_str = f" :{svc.port}" if svc.port else ""
        click.echo(f"  [{icon}] {svc.name}{port_str}{pid_str} — {state_val}")

    if resp.enabled_services:
        click.echo(f"\nEnabled: {', '.join(resp.enabled_services)}")

    if resp.stale_binaries:
        names = ", ".join(resp.stale_binaries)
        click.echo(f"\n  Warning: stale binaries: {names}")
        click.echo(f"  Run 'corvia-dev rebuild' to update.")

    if resp.logs:
        click.echo(f"\nRecent logs:")
        for line in resp.logs[-10:]:
            click.echo(f"  {line}")


@main.command()
@click.option("--no-foreground", is_flag=True, help="Run in background")
def up(no_foreground: bool) -> None:
    """Start and supervise all enabled services."""
    config_path = _config_path()
    if not config_path.exists():
        click.echo("Error: corvia.toml not found", err=True)
        raise SystemExit(1)

    cfg = load_config(config_path)
    enabled = read_enabled_services(_flags_path())
    services = resolve_startup_order(cfg.embedding_provider, cfg.storage, enabled)

    # Check for stale binaries
    stale = check_staleness(workspace_root=_workspace_root())
    if stale:
        names = ", ".join(stale)
        if sys.stdin.isatty() and not no_foreground:
            if click.confirm(
                f"Newer build detected for {names}. Install and restart?",
                default=True,
            ):
                installed = install_binaries(
                    target_dir=_workspace_root() / "repos" / "corvia" / "target" / "debug",
                )
                for name in installed:
                    click.echo(f"  Installed {name}")
        else:
            click.echo(
                f"Warning: installed binaries are older than local build: {names}",
                err=True,
            )
            click.echo(
                "  Run 'corvia-dev rebuild' to update.",
                err=True,
            )

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
        pid = os.fork()
        if pid > 0:
            click.echo(f"corvia-dev manager started (pid {pid})")
            return
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
@click.argument("service", type=click.Choice(["coding-llm", "postgres"]))
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
@click.argument("service", type=click.Choice(["coding-llm", "postgres"]))
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

    try:
        data = json.loads(DEFAULT_STATE_PATH.read_text())
        resp = StatusResponse.model_validate(data)
        if resp.manager and resp.manager.pid:
            os.kill(resp.manager.pid, signal.SIGTERM)
            click.echo("Stopping manager...")
            time.sleep(2)
            # Re-invoke up in background
            click.echo("Restarting...")
            ctx = click.get_current_context()
            ctx.invoke(up, no_foreground=True)
    except (json.JSONDecodeError, ValueError, ProcessLookupError) as e:
        click.echo(f"Failed: {e}")


@main.command("config")
def show_config() -> None:
    """Show current corvia.toml config summary."""
    path = _config_path()
    if not path.exists():
        click.echo(f"Error: {path} not found", err=True)
        raise SystemExit(1)

    cfg = load_config(path)
    click.echo(f"Embedding: {cfg.embedding_provider} ({cfg.embedding_url})")
    click.echo(f"Merge:     {cfg.merge_provider} ({cfg.merge_url})")
    click.echo(f"Storage:   {cfg.storage}")
    click.echo(f"Workspace: {cfg.workspace_name}")


@main.command()
@click.argument("service", required=False)
@click.option("--tail", "-n", default=50, help="Number of log lines")
def logs(service: str | None, tail: int) -> None:
    """Show recent service logs."""
    if service:
        # Show logs for a specific service
        lines = tail_service_log(service, tail)
        if not lines:
            click.echo(f"No logs for {service}")
            click.echo(f"  (checked {LOG_DIR / f'{service}.log'})")
            return
        click.echo(f"=== {service} (last {len(lines)} lines) ===")
        for line in lines:
            click.echo(line)
    else:
        # Show all service logs
        found = False
        if LOG_DIR.exists():
            for lf in sorted(LOG_DIR.glob("*.log")):
                svc_name = lf.stem
                lines = tail_service_log(svc_name, tail)
                if lines:
                    found = True
                    click.echo(f"\n=== {svc_name} (last {len(lines)} lines) ===")
                    for line in lines:
                        click.echo(line)
        # Also check legacy supervisor log
        sup = tail_any_log(Path("/tmp/corvia-supervisor.log"), tail)
        if sup:
            found = True
            click.echo(f"\n=== supervisor (last {len(sup)} lines) ===")
            for line in sup:
                click.echo(line)
        if not found:
            click.echo("No log files found.")


@main.command()
@click.option("--no-build", is_flag=True, help="Skip cargo build, only install and restart")
@click.option("--release", is_flag=True, help="Build with --release")
def rebuild(no_build: bool, release: bool) -> None:
    """Build from source, install binaries, and restart services."""
    workspace = _workspace_root()
    profile = "release" if release else "debug"
    target_dir = workspace / "repos" / "corvia" / "target" / profile

    if not no_build:
        click.echo(f"Building corvia binaries ({profile})...")
        if not cargo_build(workspace_root=workspace, release=release):
            click.echo("Build failed.", err=True)
            raise SystemExit(1)
        click.echo("Build succeeded.")

    if not target_dir.exists():
        click.echo(f"Target directory not found: {target_dir}", err=True)
        raise SystemExit(1)

    # Stop services BEFORE installing (can't overwrite running binaries on Linux)
    manager_was_running = False
    if DEFAULT_STATE_PATH.exists():
        try:
            data = json.loads(DEFAULT_STATE_PATH.read_text())
            resp = StatusResponse.model_validate(data)
            if resp.manager and resp.manager.pid:
                click.echo("Stopping services...")
                os.kill(resp.manager.pid, signal.SIGTERM)
                time.sleep(2)
                manager_was_running = True
        except (json.JSONDecodeError, ValueError, ProcessLookupError):
            pass

    click.echo("Installing binaries...")
    installed = install_binaries(target_dir=target_dir)
    if not installed:
        click.echo("No binaries found to install.", err=True)
        raise SystemExit(1)
    for name in installed:
        click.echo(f"  {name} -> {DEFAULT_INSTALL_DIR / name}")

    # Restart services if they were running before
    if manager_was_running:
        click.echo("Restarting services...")
        ctx = click.get_current_context()
        ctx.invoke(up, no_foreground=True)
    else:
        click.echo("No running manager. Run 'corvia-dev up' to start services.")


if __name__ == "__main__":
    main()
