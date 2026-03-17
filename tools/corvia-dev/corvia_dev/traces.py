"""Parse JSON tracing output and aggregate span timings."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

from corvia_dev.models import SpanStats, TraceEvent, TracesData


# Specific span -> module overrides (checked before prefix matching)
SPAN_TO_MODULE: dict[str, str] = {
    "corvia.entry.embed": "inference",
}

# Prefix -> module mapping (evaluated in order)
_PREFIX_MAP: list[tuple[str, str]] = [
    ("corvia.agent.", "agent"),
    ("corvia.session.", "agent"),
    ("corvia.entry.", "entry"),
    ("corvia.merge.", "merge"),
    ("corvia.store.", "storage"),
    ("corvia.rag.", "rag"),
    ("corvia.gc.", "gc"),
]

# Target (Rust module path) -> module mapping for events without span names
_TARGET_MAP: list[tuple[str, str]] = [
    ("agent_coordinator", "agent"),
    ("merge_worker", "merge"),
    ("lite_store", "storage"),
    ("knowledge_store", "storage"),
    ("postgres_store", "storage"),
    ("rag_pipeline", "rag"),
    ("graph_store", "storage"),
    ("chunking", "entry"),
    ("embedding_service", "inference"),
    ("chat_service", "inference"),
    ("model_manager", "inference"),
]


def span_to_module(span_name: str) -> str:
    """Map a span name to its module."""
    if span_name in SPAN_TO_MODULE:
        return SPAN_TO_MODULE[span_name]
    for prefix, module in _PREFIX_MAP:
        if span_name.startswith(prefix):
            return module
    return "unknown"


def target_to_module(target: str) -> str:
    """Map a Rust target path to a module name."""
    for pattern, module in _TARGET_MAP:
        if pattern in target:
            return module
    return "unknown"


def parse_trace_line(line: str) -> dict | None:
    """Parse a single JSON tracing line. Returns dict or None."""
    line = line.strip()
    if not line or not line.startswith("{"):
        return None
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None

    result: dict = {}
    result["level"] = obj.get("level", "INFO")
    result["timestamp"] = obj.get("timestamp", "")

    # Span with timing
    span = obj.get("span", {})
    span_name = span.get("name") if isinstance(span, dict) else None
    elapsed = obj.get("elapsed_ms")

    if span_name and elapsed is not None:
        result["span"] = span_name
        result["elapsed_ms"] = float(elapsed)
        result["fields"] = obj.get("fields", {})
        return result

    # Structured event (no span timing)
    fields = obj.get("fields", {})
    msg = fields.get("message", "")
    target = obj.get("target", "")
    if msg or target:
        result["msg"] = msg
        result["target"] = target
        result["module"] = target_to_module(target)
        return result

    return None


def _parse_ts_epoch(ts_str: str) -> float | None:
    """Parse an ISO timestamp string to epoch seconds, or None."""
    if not ts_str:
        return None
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return dt.timestamp()
    except (ValueError, OSError):
        return None


def collect_traces_from_lines(lines: list[str]) -> TracesData:
    """Aggregate span stats and collect recent events from log lines."""
    span_all: dict[str, list[float]] = {}
    span_1h: dict[str, list[float]] = {}
    span_errors: dict[str, int] = {}
    events: list[TraceEvent] = []
    now = time.time()
    one_hour_ago = now - 3600

    for line in lines:
        parsed = parse_trace_line(line)
        if parsed is None:
            continue

        ts_str = parsed.get("timestamp", "")
        ts_short = ""
        ts_epoch = _parse_ts_epoch(ts_str)
        if ts_str:
            ts_short = ts_str.split("T")[1][:8] if "T" in ts_str else ts_str[:8]

        level = parsed.get("level", "INFO").upper()

        if "span" in parsed:
            span_name = parsed["span"]
            elapsed = parsed["elapsed_ms"]
            span_all.setdefault(span_name, []).append(elapsed)
            if ts_epoch is None or ts_epoch >= one_hour_ago:
                span_1h.setdefault(span_name, []).append(elapsed)
            if level in ("ERROR", "ERR"):
                span_errors[span_name] = span_errors.get(span_name, 0) + 1
        elif "msg" in parsed and parsed["msg"]:
            norm_level = level.lower()
            if norm_level in ("warn", "warning"):
                norm_level = "warn"
            elif norm_level in ("error", "err"):
                norm_level = "error"
            elif norm_level == "debug":
                norm_level = "debug"
            else:
                norm_level = "info"
            module = parsed.get("module", "unknown")
            events.append(TraceEvent(
                ts=ts_short,
                level=norm_level,
                module=module,
                msg=parsed["msg"],
            ))

    # Build SpanStats
    spans: dict[str, SpanStats] = {}
    for name, timings in span_all.items():
        count = len(timings)
        timings_1h = span_1h.get(name, [])
        avg = sum(timings) / count if count else 0
        spans[name] = SpanStats(
            count=count,
            count_1h=len(timings_1h),
            avg_ms=round(avg, 1),
            last_ms=round(timings[-1], 1) if timings else 0,
            errors=span_errors.get(name, 0),
        )

    # Keep last 50 events
    recent = events[-50:]

    return TracesData(spans=spans, recent_events=recent)


def collect_traces(log_dir: Path) -> TracesData:
    """Collect traces from all service log files in log_dir."""
    all_lines: list[str] = []
    if log_dir.exists():
        for log_file in log_dir.glob("*.log"):
            try:
                text = log_file.read_text(errors="replace")
                lines = text.strip().split("\n")
                # Take last 500 lines per file to bound memory
                all_lines.extend(lines[-500:])
            except OSError:
                continue
    return collect_traces_from_lines(all_lines)
