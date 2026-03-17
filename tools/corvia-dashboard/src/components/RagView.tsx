import { useState, useCallback } from "preact/hooks";
import { ragAsk } from "../api";
import type { RagResponse, PipelineTrace } from "../types";

function WaterfallBar({ trace }: { trace: PipelineTrace }) {
  const total = trace.total_latency_ms || 1;
  const rPct = (trace.retrieval.latency_ms / total) * 100;
  const aPct = (trace.augmentation.latency_ms / total) * 100;
  const gPct = trace.generation ? (trace.generation.latency_ms / total) * 100 : 0;

  return (
    <div class="rag-waterfall">
      <div class="waterfall-header">
        <span class="waterfall-total">{total}ms total</span>
      </div>
      <div class="waterfall-track">
        <div
          class="waterfall-seg retrieve"
          style={{ width: `${Math.max(rPct, 2)}%` }}
          title={`Retrieve: ${trace.retrieval.latency_ms}ms`}
        >
          <span class="waterfall-label">Retrieve</span>
        </div>
        <div
          class="waterfall-seg augment"
          style={{ width: `${Math.max(aPct, 2)}%` }}
          title={`Augment: ${trace.augmentation.latency_ms}ms`}
        >
          <span class="waterfall-label">Augment</span>
        </div>
        {trace.generation && (
          <div
            class="waterfall-seg generate"
            style={{ width: `${Math.max(gPct, 2)}%` }}
            title={`Generate: ${trace.generation.latency_ms}ms`}
          >
            <span class="waterfall-label">Generate</span>
          </div>
        )}
      </div>
      <div class="waterfall-legend">
        <span class="waterfall-ms retrieve">{trace.retrieval.latency_ms}ms</span>
        <span class="waterfall-ms augment">{trace.augmentation.latency_ms}ms</span>
        {trace.generation && <span class="waterfall-ms generate">{trace.generation.latency_ms}ms</span>}
      </div>
    </div>
  );
}

function TokenGauge({ trace }: { trace: PipelineTrace }) {
  const used = trace.augmentation.token_estimate;
  const budget = trace.augmentation.token_budget;
  const pct = budget > 0 ? Math.min((used / budget) * 100, 100) : 0;
  const color = pct > 90 ? "var(--coral)" : pct > 70 ? "var(--amber)" : "var(--mint)";

  return (
    <div class="token-gauge">
      <div class="token-gauge-header">
        <span class="token-gauge-label">Token Budget</span>
        <span class="token-gauge-val" style={{ color }}>
          {used.toLocaleString()} / {budget.toLocaleString()}
        </span>
      </div>
      <div class="token-gauge-track">
        <div class="token-gauge-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
    </div>
  );
}

function TraceStats({ trace }: { trace: PipelineTrace }) {
  return (
    <div class="rag-stats">
      <div class="rag-stat">
        <div class="rag-stat-val">{trace.retrieval.vector_results}</div>
        <div class="rag-stat-label">Vector hits</div>
      </div>
      <div class="rag-stat">
        <div class="rag-stat-val">{trace.retrieval.graph_expanded}</div>
        <div class="rag-stat-label">Graph expanded</div>
      </div>
      <div class="rag-stat">
        <div class="rag-stat-val">{trace.augmentation.sources_included}</div>
        <div class="rag-stat-label">Sources used</div>
      </div>
      <div class="rag-stat">
        <div class="rag-stat-val">{trace.augmentation.sources_truncated}</div>
        <div class="rag-stat-label">Truncated</div>
      </div>
      {trace.generation && (
        <>
          <div class="rag-stat">
            <div class="rag-stat-val">{trace.generation.input_tokens}</div>
            <div class="rag-stat-label">Input tokens</div>
          </div>
          <div class="rag-stat">
            <div class="rag-stat-val">{trace.generation.output_tokens}</div>
            <div class="rag-stat-label">Output tokens</div>
          </div>
        </>
      )}
    </div>
  );
}

function SourcePanel({ response }: { response: RagResponse }) {
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null);

  return (
    <div class="rag-sources">
      <h3 class="rag-section-title">Sources ({response.sources.length})</h3>
      {response.sources.map((src, i) => (
        <div
          class={`rag-source${expandedIdx === i ? " expanded" : ""}`}
          key={i}
          onClick={() => setExpandedIdx(expandedIdx === i ? null : i)}
        >
          <div class="rag-source-header">
            <span class="rag-source-score">{(src.score * 100).toFixed(1)}%</span>
            <span class="rag-source-file">{src.source_file ?? "inline"}</span>
            {src.language && <span class="rag-source-lang">{src.language}</span>}
          </div>
          {expandedIdx === i && (
            <pre class="rag-source-content">{src.content}</pre>
          )}
        </div>
      ))}
    </div>
  );
}

export function RagView({ navigateToHistory }: { navigateToHistory?: (entryId: string) => void }) {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [response, setResponse] = useState<RagResponse | null>(null);

  const handleSubmit = useCallback(async (e: Event) => {
    e.preventDefault();
    if (!query.trim() || loading) return;
    setLoading(true);
    setError(null);
    try {
      const resp = await ragAsk(query.trim(), "corvia");
      setResponse(resp);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
    setLoading(false);
  }, [query, loading]);

  return (
    <div class="rag-view">
      {/* Query explorer */}
      <div class="card">
        <h2>RAG Pipeline Explorer</h2>
        <form class="rag-form" onSubmit={handleSubmit}>
          <input
            type="text"
            class="rag-input"
            placeholder="Ask the knowledge base..."
            value={query}
            onInput={(e) => setQuery((e.target as HTMLInputElement).value)}
            disabled={loading}
          />
          <button type="submit" class="rag-submit" disabled={loading || !query.trim()}>
            {loading ? "Thinking..." : "Ask"}
          </button>
        </form>
      </div>

      {error && <div class="error-banner">{error}</div>}

      {response && (
        <>
          {/* Answer */}
          {response.answer && (
            <div class="card rag-answer-card">
              <h2>Answer</h2>
              <div class="rag-answer">{response.answer}</div>
            </div>
          )}

          {/* Pipeline trace */}
          <div class="card">
            <h2>Pipeline Trace</h2>
            <WaterfallBar trace={response.trace} />
            <TokenGauge trace={response.trace} />
            <TraceStats trace={response.trace} />
          </div>

          {/* Sources */}
          <div class="card">
            <SourcePanel response={response} />
          </div>
        </>
      )}

      {!response && !error && (
        <div class="card">
          <div style={{ color: "var(--text-dim)", textAlign: "center", padding: "40px 20px" }}>
            Type a question above to see the full RAG pipeline in action.
            <br />
            <span style={{ fontSize: "12px" }}>
              Each query shows: retrieval → augmentation → generation stages with timing, token budget, and source attribution.
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
