"""Tests for trace log parsing and aggregation."""

from corvia_dev.traces import parse_trace_line, collect_traces_from_lines, SPAN_TO_MODULE


def test_parse_json_span_line():
    """Parse a JSON tracing line with span timing."""
    line = '{"timestamp":"2026-03-10T14:31:52","level":"INFO","span":{"name":"corvia.entry.write"},"fields":{"session_id":"s1"},"elapsed_ms":12}'
    result = parse_trace_line(line)
    assert result is not None
    assert result["span"] == "corvia.entry.write"
    assert result["elapsed_ms"] == 12
    assert result["level"] == "INFO"


def test_parse_json_event_line():
    """Parse a JSON tracing event (no span timing)."""
    line = '{"timestamp":"2026-03-10T14:31:52","level":"WARN","fields":{"message":"Slow embed: 210ms"},"target":"corvia_kernel::agent_coordinator"}'
    result = parse_trace_line(line)
    assert result is not None
    assert result["level"] == "WARN"
    assert "Slow embed" in result["msg"]


def test_parse_non_json_line_returns_none():
    """Non-JSON lines return None."""
    result = parse_trace_line("INFO some plain text log")
    assert result is None


def test_parse_empty_line():
    result = parse_trace_line("")
    assert result is None


def test_span_to_module_mapping():
    """Verify specific-first matching: entry.embed -> inference."""
    assert SPAN_TO_MODULE["corvia.entry.embed"] == "inference"
    assert "corvia.entry.write" not in SPAN_TO_MODULE


def test_collect_traces_from_lines():
    """Full aggregation from a set of log lines."""
    lines = [
        '{"timestamp":"2026-03-10T14:00:00","level":"INFO","span":{"name":"corvia.entry.write"},"elapsed_ms":10}',
        '{"timestamp":"2026-03-10T14:00:01","level":"INFO","span":{"name":"corvia.entry.write"},"elapsed_ms":14}',
        '{"timestamp":"2026-03-10T14:00:02","level":"INFO","span":{"name":"corvia.entry.embed"},"elapsed_ms":80}',
        '{"timestamp":"2026-03-10T14:00:03","level":"WARN","fields":{"message":"Slow embed: 210ms"},"target":"corvia_kernel::agent_coordinator"}',
        'not a json line',
    ]
    traces = collect_traces_from_lines(lines)
    assert "corvia.entry.write" in traces.spans
    assert traces.spans["corvia.entry.write"].count == 2
    assert traces.spans["corvia.entry.write"].avg_ms == 12.0
    assert traces.spans["corvia.entry.write"].last_ms == 14.0
    assert traces.spans["corvia.entry.embed"].count == 1
    assert len(traces.recent_events) == 1
    assert traces.recent_events[0].level == "warn"


def test_collect_traces_empty():
    traces = collect_traces_from_lines([])
    assert traces.spans == {}
    assert traces.recent_events == []
