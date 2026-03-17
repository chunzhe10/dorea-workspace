"""Tests for health checking."""

import http.server
import threading

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
            pass

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
    assert result.healthy is None
