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
from corvia_dev.traces import collect_traces
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
LOG_DIR = Path("/tmp/corvia-dev-logs")


def _tail_log(service_name: str, lines: int = 30) -> list[str]:
    """Read the last N lines of a service's log file."""
    log_file = LOG_DIR / f"{service_name}.log"
    if not log_file.exists():
        return []
    try:
        text = log_file.read_text(errors="replace")
        return text.strip().splitlines()[-lines:]
    except OSError:
        return []


def tail_service_log(service_name: str, lines: int = 50) -> list[str]:
    """Public API: read recent log lines for a service."""
    return _tail_log(service_name, lines)


def tail_any_log(path: Path, lines: int = 50) -> list[str]:
    """Read the last N lines of any log file."""
    if not path.exists():
        return []
    try:
        text = path.read_text(errors="replace")
        return text.strip().splitlines()[-lines:]
    except OSError:
        return []


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
        if len(self._log_lines) > 100:
            self._log_lines = self._log_lines[-100:]
        logger.info(msg)

    async def start_service(self, name: str) -> None:
        """Start a single service process."""
        mp = self.processes.get(name)
        if mp is None:
            return

        if not mp.service.start_cmd:
            mp.state = ServiceState.HEALTHY
            mp.started_at = time.time()
            self._log(f"{name}: virtual service marked healthy")
            return

        mp.state = ServiceState.STARTING
        self._log(f"{name}: starting ({' '.join(mp.service.start_cmd)})")

        # Write stdout/stderr to per-service log files
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_file = LOG_DIR / f"{name}.log"
        try:
            fh = open(log_file, "a")  # noqa: SIM115
            proc = await asyncio.create_subprocess_exec(
                *mp.service.start_cmd,
                cwd=str(self.workspace_root),
                stdout=fh,
                stderr=fh,
            )
            mp.process = proc
            mp.pid = proc.pid
            mp.started_at = time.time()
            mp.state = ServiceState.STARTING
            mp._log_fh = fh  # keep reference so GC doesn't close it
            self._log(f"{name}: started (pid {proc.pid}), logs -> {log_file}")
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
            pass

        mp.state = ServiceState.STOPPED
        mp.pid = None
        mp.process = None
        mp.started_at = None
        if hasattr(mp, '_log_fh') and mp._log_fh:
            mp._log_fh.close()
            mp._log_fh = None
        self._log(f"{name}: stopped")

    async def health_check_all(self) -> None:
        """Run health checks on all managed processes."""
        for name, mp in self.processes.items():
            if mp.state in (ServiceState.STOPPED,):
                continue

            if mp.process is not None and mp.process.returncode is not None:
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
                if mp.uptime_s is not None and mp.uptime_s > BACKOFF_RESET_AFTER:
                    mp.reset_backoff()
            elif result.healthy is False and mp.state == ServiceState.HEALTHY:
                mp.state = ServiceState.UNHEALTHY
                self._log(f"{name}: health check failed")
            elif result.healthy is None:
                pass

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
            service_logs={name: _tail_log(name) for name in self.processes},
            traces=collect_traces(LOG_DIR),
        )
        self.state_path.write_text(resp.model_dump_json(indent=2))

    def _cleanup_stale_processes(self) -> None:
        """Kill leftover processes from a previous manager that didn't shut down cleanly."""
        if not self.state_path.exists():
            return
        try:
            old_state = json.loads(self.state_path.read_text())
        except (json.JSONDecodeError, OSError):
            return

        old_mgr_pid = old_state.get("manager", {}).get("pid")
        if old_mgr_pid and old_mgr_pid != os.getpid():
            # Check if old manager is still alive — if so, don't touch its children
            try:
                os.kill(old_mgr_pid, 0)
                self._log(f"previous manager (pid {old_mgr_pid}) still alive, skipping cleanup")
                return
            except OSError:
                pass  # old manager is dead, clean up its orphans

        for svc in old_state.get("services", []):
            pid = svc.get("pid")
            name = svc.get("name", "unknown")
            if pid is None:
                continue
            try:
                os.kill(pid, 0)  # check if alive
            except OSError:
                continue
            self._log(f"killing stale {name} process (pid {pid}) from previous manager")
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass

    async def run(self) -> None:
        """Main supervision loop. Runs until cancelled or SIGTERM."""
        self._running = True
        loop = asyncio.get_event_loop()

        stop_event = asyncio.Event()

        def _signal_handler() -> None:
            self._log("Received shutdown signal")
            stop_event.set()

        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, _signal_handler)

        # Clean up orphaned processes from previous manager crash
        self._cleanup_stale_processes()

        # Start services in order
        for name in self.processes:
            await self.start_service(name)
            for _ in range(30):
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

        # Shutdown
        self._log("Shutting down all services")
        for name in reversed(list(self.processes.keys())):
            await self.stop_service(name)
        self._running = False
        self.write_state()
