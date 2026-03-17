import { useState, useCallback, useEffect, useRef } from "preact/hooks";
import { usePoll } from "../hooks/use-poll";
import { fetchTraces, fetchMergeQueue, retryMergeEntries } from "../api";
import type { SpanStats, TracesResponse, TraceEvent, MergeQueueStatus, ModuleStats } from "../types";
import { GcPanel } from "./GcPanel";
import { WaterfallView } from "./WaterfallView";

// --- Architecture topology ---
interface ModuleDef {
  label: string;
  color: string;
  desc: string;
  icon: string;
  pos: [number, number]; // [left%, top%]
  navTarget?: string; // tab to navigate to on "View all"
}

const MODULES: Record<string, ModuleDef> = {
  agent:     { label: "Agent",     color: "peach",    desc: "Agent registration & session lifecycle",    icon: "\u{1F916}", pos: [3, 5],  navTarget: "agents" },
  entry:     { label: "Entry",     color: "gold",     desc: "Write, embed, insert pipeline",             icon: "\u{1F4DD}", pos: [35, 2] },
  merge:     { label: "Merge",     color: "mint",     desc: "Conflict detection & resolution",           icon: "\u{1F500}", pos: [35, 42] },
  storage:   { label: "Storage",   color: "lavender", desc: "LiteStore / Postgres persistence",          icon: "\u{1F4BE}", pos: [67, 2] },
  rag:       { label: "RAG",       color: "sky",      desc: "Retrieval-augmented generation",            icon: "\u{1F50E}", pos: [67, 42], navTarget: "rag" },
  inference: { label: "Inference", color: "coral",    desc: "ONNX embedding via gRPC",                  icon: "\u26A1",    pos: [3, 42] },
  gc:        { label: "GC",        color: "amber",    desc: "Garbage collection sweeps",                 icon: "\u{1F9F9}", pos: [3, 75] },
};

const EDGES: [string, string][] = [
  ["agent", "entry"], ["agent", "gc"],
  ["entry", "storage"], ["entry", "merge"], ["entry", "inference"],
  ["merge", "storage"],
  ["storage", "rag"],
  ["gc", "storage"],
];

const SPAN_MODULE_SPECIFIC: Record<string, string> = { "corvia.entry.embed": "inference" };
const SPAN_MODULE_PREFIX: [string, string][] = [
  ["corvia.agent.", "agent"], ["corvia.session.", "agent"],
  ["corvia.entry.", "entry"], ["corvia.merge.", "merge"],
  ["corvia.store.", "storage"], ["corvia.rag.", "rag"],
  ["corvia.gc.", "gc"],
];

const SPAN_FIELDS: Record<string, string> = {
  "corvia.agent.register": "display_name",
  "corvia.session.create": "agent_id, with_staging",
  "corvia.session.commit": "session_id",
  "corvia.entry.write": "session_id",
  "corvia.entry.embed": "gRPC / Ollama",
  "corvia.entry.insert": "entry_id, scope_id",
  "corvia.merge.process": "",
  "corvia.merge.process_entry": "entry_id",
  "corvia.merge.conflict": "entry_id, scope_id",
  "corvia.merge.llm_resolve": "new_id, existing_id",
  "corvia.store.insert": "entry_id, scope_id",
  "corvia.store.search": "scope_id",
  "corvia.store.get": "",
  "corvia.rag.context": "scope_id",
  "corvia.rag.ask": "scope_id",
  "corvia.gc.run": "",
};

function spanToModule(name: string): string {
  if (SPAN_MODULE_SPECIFIC[name]) return SPAN_MODULE_SPECIFIC[name];
  for (const [prefix, mod] of SPAN_MODULE_PREFIX) {
    if (name.startsWith(prefix)) return mod;
  }
  return "unknown";
}

type TraceMode = "map" | "dataflow" | "heat";

function getMaxCount(modules: Record<string, ModuleStats>): number {
  let max = 1;
  for (const ms of Object.values(modules)) {
    if (ms.count > max) max = ms.count;
  }
  return max;
}

// --- SVG edge drawing ---
function EdgeLayer({ mode, canvasRef }: { mode: TraceMode; canvasRef: { current: HTMLDivElement | null } }) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    const svg = svgRef.current;
    const canvas = canvasRef.current;
    if (!svg || !canvas) return;

    const cw = canvas.offsetWidth;
    const ch = canvas.offsetHeight;
    svg.setAttribute("viewBox", `0 0 ${cw} ${ch}`);

    let html = "";
    let anims = "";
    for (const [from, to] of EDGES) {
      const fromMod = MODULES[from];
      const toMod = MODULES[to];
      if (!fromMod || !toMod) continue;

      const x1 = (fromMod.pos[0] / 100) * cw + 60;
      const y1 = (fromMod.pos[1] / 100) * ch + 40;
      const x2 = (toMod.pos[0] / 100) * cw + 60;
      const y2 = (toMod.pos[1] / 100) * ch + 40;
      const mx = (x1 + x2) / 2;

      const pathId = `edge-${from}-${to}`;
      const d = `M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`;
      html += `<path id="${pathId}" class="edge-path" d="${d}"/>`;

      if (mode === "dataflow") {
        const color = `var(--${fromMod.color})`;
        anims += `<circle r="3" fill="${color}" style="filter:drop-shadow(0 0 3px ${color})">` +
          `<animateMotion dur="3s" repeatCount="indefinite"><mpath href="#${pathId}"/></animateMotion>` +
          `</circle>`;
      }
    }

    svg.innerHTML = html + anims;
  }, [mode, canvasRef]);

  return <svg ref={svgRef} class="edge-layer" />;
}

// --- Module node ---
function ModuleNode({
  id, mod, modStats, maxCount, mode, selected, onSelect,
}: {
  id: string;
  mod: ModuleDef;
  modStats: ModuleStats;
  maxCount: number;
  mode: TraceMode;
  selected: boolean;
  onSelect: (id: string) => void;
}) {
  const barW = maxCount > 0 ? Math.max(5, Math.round((modStats.count / maxCount) * 100)) : 5;

  let heatCls = "";
  if (mode === "heat") {
    const score = (modStats.count / maxCount) * 0.6 + (modStats.errors > 0 ? 0.4 : 0);
    heatCls = score > 0.7 ? " heat-hot" : score > 0.3 ? " heat-warm" : " heat-cool";
  }

  const selStyle = selected ? { borderColor: `var(--${mod.color})` } : {};

  return (
    <div
      class={`tnode${selected ? " selected" : ""}${heatCls}`}
      style={{ left: `${mod.pos[0]}%`, top: `${mod.pos[1]}%`, ...selStyle }}
      onClick={() => onSelect(id)}
    >
      <div class="tnode-icon" style={{ background: `var(--${mod.color}-soft)`, color: `var(--${mod.color})` }}>
        {mod.icon}
      </div>
      <div class="tnode-label" style={{ color: `var(--${mod.color})` }}>{mod.label}</div>
      <div class="tnode-stat">{modStats.count.toLocaleString()} ops &middot; {modStats.span_count} spans</div>
      <div class="tnode-bar">
        <div class="tnode-bar-fill" style={{ width: `${barW}%`, background: `var(--${mod.color})` }} />
      </div>
    </div>
  );
}

// --- Merge queue panel (inline in detail panel for "merge" module) ---
function MergeQueuePanel() {
  const fetcher = useCallback(() => fetchMergeQueue(20), []);
  const { data, error } = usePoll(fetcher, 3000);

  const handleRetry = useCallback(async (entryId: string) => {
    try {
      await retryMergeEntries([entryId]);
    } catch { /* ignore */ }
  }, []);

  if (error) return <div class="error-banner" style={{ fontSize: "12px" }}>{error}</div>;
  if (!data) return null;

  return (
    <div class="trace-card">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "10px" }}>
        <h2 style={{ fontSize: "12px", fontWeight: 600, color: "var(--text-dim)", textTransform: "uppercase", letterSpacing: "0.5px", margin: 0 }}>
          Merge Queue
        </h2>
        <span style={{ fontSize: "12px", color: "var(--mint)", fontFamily: "var(--font-mono)" }}>
          {data.depth} pending
        </span>
      </div>
      {data.entries.length === 0 ? (
        <div style={{ fontSize: "12px", color: "var(--text-dim)", padding: "8px 0" }}>Queue is empty</div>
      ) : (
        data.entries.map((entry) => (
          <div class="merge-entry" key={entry.entry_id}>
            <div class="merge-entry-info">
              <code class="merge-entry-id">{entry.entry_id.slice(0, 8)}</code>
              <span class="merge-entry-agent">{entry.agent_id}</span>
              {entry.last_error && (
                <span class="merge-entry-error">{entry.last_error}</span>
              )}
            </div>
            <div class="merge-entry-actions">
              {entry.retry_count > 0 && (
                <span class="merge-retry-count">{entry.retry_count}x</span>
              )}
              {entry.last_error && (
                <button
                  class="merge-retry-btn"
                  onClick={() => handleRetry(entry.entry_id)}
                  title="Retry"
                >
                  \u21BB
                </button>
              )}
            </div>
          </div>
        ))
      )}
    </div>
  );
}

// --- Detail panel ---
function DetailPanel({
  moduleId, modDef, modStats, spans, events, onNavigate,
}: {
  moduleId: string;
  modDef: ModuleDef;
  modStats: ModuleStats;
  spans: Record<string, SpanStats>;
  events: TraceEvent[];
  onNavigate?: (tab: string) => void;
}) {
  const moduleSpans = Object.entries(spans)
    .filter(([name]) => spanToModule(name) === moduleId)
    .map(([name, stats]) => ({ name, stats }));

  const modEvents = events.filter((ev) => ev.module === moduleId).slice(0, 10);

  const avgColor = modStats.avg_ms < 50 ? "var(--mint)" : modStats.avg_ms < 150 ? "var(--peach)" : "var(--coral)";
  const errColor = modStats.errors === 0 ? "var(--mint)" : "var(--coral)";

  return (
    <>
      {/* Module summary */}
      <div class="trace-card">
        <div class="module-hdr">
          <div class="module-dot" style={{ background: `var(--${modDef.color})`, boxShadow: `0 0 6px var(--${modDef.color}-soft)` }} />
          <div>
            <div class="module-name">{modDef.label}</div>
            <div class="module-desc">{modDef.desc}</div>
          </div>
        </div>
        <div class="mini-stats">
          <div class="mini-stat">
            <div class="mini-stat-val">{modStats.count.toLocaleString()}</div>
            <div class="mini-stat-lbl">Total</div>
          </div>
          <div class="mini-stat">
            <div class="mini-stat-val">{modStats.count_1h.toLocaleString()}</div>
            <div class="mini-stat-lbl">Last hour</div>
          </div>
          <div class="mini-stat">
            <div class="mini-stat-val" style={{ color: avgColor }}>
              {modStats.avg_ms}<span style={{ fontSize: "11px", fontWeight: 500 }}>ms</span>
            </div>
            <div class="mini-stat-lbl">Avg latency</div>
          </div>
          <div class="mini-stat">
            <div class="mini-stat-val" style={{ color: errColor }}>{modStats.errors}</div>
            <div class="mini-stat-lbl">Errors</div>
          </div>
        </div>

        {/* Navigation link for modules with dedicated tabs */}
        {modDef.navTarget && onNavigate && (
          <button
            class="nav-link"
            onClick={() => onNavigate(modDef.navTarget!)}
          >
            View all {modDef.label.toLowerCase()} &rarr;
          </button>
        )}
      </div>

      {/* Merge queue (only for merge module) */}
      {moduleId === "merge" && <MergeQueuePanel />}

      {/* Spans */}
      <div class="trace-card">
        <h2 style={{ fontSize: "12px", fontWeight: 600, color: "var(--text-dim)", textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: "10px" }}>
          Instrumented Spans
        </h2>
        {moduleSpans.length === 0 ? (
          <div style={{ fontSize: "12px", color: "var(--text-dim)", padding: "8px 0" }}>No span data available</div>
        ) : (
          moduleSpans.map(({ name, stats }) => {
            const shortName = name.replace("corvia.", "");
            const fields = SPAN_FIELDS[name] || "";
            const ms = stats.avg_ms;
            const pillCls = ms < 50 ? "span-fast" : ms < 150 ? "span-medium" : "span-slow";
            return (
              <div class="span-row" key={name}>
                <div>
                  <div class="span-name">{shortName}</div>
                  {fields && <div class="span-fields">{fields}</div>}
                  {(stats.p50_ms ?? 0) > 0 && (
                    <div class="span-percentiles">
                      p50: {stats.p50_ms?.toFixed(0)}ms &middot; p95: {stats.p95_ms?.toFixed(0)}ms &middot; p99: {stats.p99_ms?.toFixed(0)}ms
                    </div>
                  )}
                </div>
                <span class={`span-pill ${pillCls}`}>{Math.round(ms)}ms</span>
              </div>
            );
          })
        )}
      </div>

      {/* Events */}
      <div class="trace-card">
        <h2 style={{ fontSize: "12px", fontWeight: 600, color: "var(--text-dim)", textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: "10px" }}>
          Recent Events
        </h2>
        {modEvents.length === 0 ? (
          <div style={{ fontSize: "12px", color: "var(--text-dim)", padding: "8px 0" }}>No recent events</div>
        ) : (
          modEvents.map((ev, i) => (
            <div class="evt-row" key={i}>
              <div class={`evt-dot ${ev.level}`} />
              <span class="evt-msg">{ev.msg}</span>
              <span class="evt-time">{shortTs(ev.ts)}</span>
            </div>
          ))
        )}
      </div>
    </>
  );
}

function shortTs(ts: string): string {
  try { return new Date(ts).toLocaleTimeString([], { hour12: false }); }
  catch { return ts; }
}

// --- Main component ---
export function TracesView({ onNavigate }: { onNavigate?: (tab: string) => void }) {
  const [mode, setMode] = useState<TraceMode>("map");
  const [selectedModule, setSelectedModule] = useState<string | null>(null);
  const canvasRef = useRef<HTMLDivElement>(null);

  const fetcher = useCallback(() => fetchTraces(), []);
  const { data, error, loading } = usePoll(fetcher, 5000);

  if (loading) return <div class="loading">Loading traces...</div>;
  if (error) return <div class="error-banner">{error}</div>;
  if (!data) return null;

  // Use pre-aggregated module stats from server (no client-side computation)
  const defaultMs: ModuleStats = { count: 0, count_1h: 0, avg_ms: 0, errors: 0, span_count: 0 };
  const modStats = data.modules;
  const maxCount = getMaxCount(modStats);
  const ms = (id: string) => modStats[id] ?? defaultMs;

  return (
    <div class="traces-workspace">
      {/* Graph panel */}
      <div class="graph-panel">
        <div class="graph-toolbar">
          <div class="mode-switcher">
            {(["map", "dataflow", "heat"] as TraceMode[]).map((m) => (
              <button
                key={m}
                class={`mode-btn${mode === m ? " active" : ""}`}
                onClick={() => setMode(m)}
              >
                {m === "map" ? "Map" : m === "dataflow" ? "Data Flow" : "Heat"}
              </button>
            ))}
          </div>
          <span class="graph-hint">Click a module to inspect</span>
        </div>

        <div class="graph-canvas" ref={canvasRef}>
          <EdgeLayer mode={mode} canvasRef={canvasRef} />
          {Object.entries(MODULES).map(([id, mod]) => (
            <ModuleNode
              key={id}
              id={id}
              mod={mod}
              modStats={ms(id)}
              maxCount={maxCount}
              mode={mode}
              selected={selectedModule === id}
              onSelect={setSelectedModule}
            />
          ))}
        </div>
      </div>

      {/* Detail panel */}
      <div class="trace-detail">
        {!selectedModule ? (
          <div class="trace-card">
            <div class="trace-empty">Select a module to inspect its telemetry</div>
          </div>
        ) : selectedModule === "gc" ? (
          <GcPanel />
        ) : (
          <DetailPanel
            moduleId={selectedModule}
            modDef={MODULES[selectedModule]}
            modStats={ms(selectedModule)}
            spans={data.spans}
            events={data.recent_events}
            onNavigate={onNavigate}
          />
        )}
      </div>

      {/* Waterfall drill-down */}
      <div class="trace-waterfall-section">
        <WaterfallView />
      </div>
    </div>
  );
}
