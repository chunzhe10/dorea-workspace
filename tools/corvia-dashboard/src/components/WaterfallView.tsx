import { useState, useCallback } from "preact/hooks";
import { usePoll } from "../hooks/use-poll";
import { fetchRecentTraces } from "../api";
import type { TraceTree, SpanNode } from "../types";

const MODULE_COLORS: Record<string, string> = {
  agent: "var(--peach)",
  entry: "var(--gold)",
  merge: "var(--mint)",
  storage: "var(--lavender)",
  rag: "var(--sky)",
  inference: "var(--coral)",
  gc: "var(--amber)",
  unknown: "var(--text-dim)",
};

function WaterfallBar({
  span,
  totalMs,
  depth,
}: {
  span: SpanNode;
  totalMs: number;
  depth: number;
}) {
  const leftPct = totalMs > 0 ? (span.start_offset_ms / totalMs) * 100 : 0;
  const widthPct = totalMs > 0 ? Math.max(0.5, (span.elapsed_ms / totalMs) * 100) : 1;
  const color = MODULE_COLORS[span.module] ?? MODULE_COLORS.unknown;
  const shortName = span.span_name.replace("corvia.", "");

  return (
    <div class="waterfall-row" style={{ paddingLeft: `${depth * 16}px` }}>
      <div class="waterfall-label" title={span.span_name}>
        {shortName}
      </div>
      <div class="waterfall-track">
        <div
          class="waterfall-bar"
          style={{
            left: `${leftPct}%`,
            width: `${widthPct}%`,
            background: color,
          }}
          title={`${span.span_name}: ${span.elapsed_ms.toFixed(1)}ms`}
        >
          <span class="waterfall-bar-label">
            {span.elapsed_ms.toFixed(1)}ms
          </span>
        </div>
      </div>
    </div>
  );
}

function TraceDetail({ trace }: { trace: TraceTree }) {
  const renderSpans = (spans: SpanNode[], depth: number) => {
    const elements: any[] = [];
    for (const span of spans) {
      elements.push(
        <WaterfallBar key={span.span_id} span={span} totalMs={trace.total_ms} depth={depth} />
      );
      if (span.children.length > 0) {
        elements.push(...renderSpans(span.children, depth + 1));
      }
    }
    return elements;
  };

  return (
    <div class="waterfall-detail">
      <div class="waterfall-header">
        <span class="waterfall-root">{trace.root_span.replace("corvia.", "")}</span>
        <span class="waterfall-total">{trace.total_ms.toFixed(1)}ms</span>
        <span class="waterfall-count">{trace.span_count} spans</span>
      </div>
      <div class="waterfall-chart">
        {renderSpans(trace.spans, 0)}
      </div>
    </div>
  );
}

export function WaterfallView() {
  const [selectedTrace, setSelectedTrace] = useState<string | null>(null);
  const fetcher = useCallback(() => fetchRecentTraces(20), []);
  const { data } = usePoll(fetcher, 10000);

  if (!data || data.traces.length === 0) {
    return (
      <div class="trace-card">
        <div style={{ fontSize: "12px", color: "var(--text-dim)", textAlign: "center", padding: "16px 0" }}>
          Enable OTEL exporter to see trace trees.
        </div>
      </div>
    );
  }

  const active = data.traces.find((t) => t.trace_id === selectedTrace) ?? data.traces[0];

  return (
    <div class="waterfall-view">
      <div class="trace-card">
        <h2 style={{ fontSize: "12px", fontWeight: 600, color: "var(--text-dim)", textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: "10px" }}>
          Recent Traces ({data.traces.length})
        </h2>
        <div class="trace-list">
          {data.traces.map((t) => (
            <button
              key={t.trace_id}
              class={`trace-list-item${t.trace_id === active.trace_id ? " active" : ""}`}
              onClick={() => setSelectedTrace(t.trace_id)}
            >
              <span class="trace-list-name">{t.root_span.replace("corvia.", "")}</span>
              <span class="trace-list-ms">{t.total_ms.toFixed(0)}ms</span>
              <span class="trace-list-count">{t.span_count} spans</span>
            </button>
          ))}
        </div>
      </div>

      <TraceDetail trace={active} />
    </div>
  );
}
