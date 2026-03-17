"""Health checking for services."""

from __future__ import annotations

import socket
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


def check_grpc(host: str, port: int, timeout: float = 3.0) -> HealthResult:
    """Check gRPC health via TCP connect + HTTP/2 preface handshake."""
    # HTTP/2 connection preface — if the server responds with a SETTINGS
    # frame, it's alive and speaking gRPC/HTTP2.
    h2_preface = b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n"
    start = time.monotonic()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect((host, port))
        s.sendall(h2_preface)
        data = s.recv(64)
        s.close()
        elapsed = (time.monotonic() - start) * 1000
        # A valid HTTP/2 SETTINGS frame starts with \x00\x00 length + type 0x04
        if len(data) >= 9 and data[3] == 0x04:
            return HealthResult(healthy=True, latency_ms=round(elapsed, 1))
        return HealthResult(healthy=False, latency_ms=-1)
    except (OSError, TimeoutError):
        return HealthResult(healthy=False, latency_ms=-1)


def check_tcp(host: str, port: int, timeout: float = 3.0) -> HealthResult:
    """Check health via TCP connect only."""
    start = time.monotonic()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect((host, port))
        s.close()
        elapsed = (time.monotonic() - start) * 1000
        return HealthResult(healthy=True, latency_ms=round(elapsed, 1))
    except (OSError, TimeoutError):
        return HealthResult(healthy=False, latency_ms=-1)


def check_service(svc: ServiceDefinition, timeout: float = 3.0) -> HealthResult:
    """Check health of a service. Dispatches to appropriate check method."""
    if svc.port is None or svc.health_proto == "none":
        return HealthResult(healthy=None, latency_ms=-1)
    host = "127.0.0.1"
    if svc.health_proto == "grpc":
        return check_grpc(host, svc.port, timeout=timeout)
    if svc.health_proto == "tcp":
        return check_tcp(host, svc.port, timeout=timeout)
    return check_http(host, svc.port, svc.health_path, timeout=timeout)
