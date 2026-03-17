import { useState, useCallback, useEffect } from "preact/hooks";
import { createContext } from "preact";
import { useContext } from "preact/hooks";
import { usePoll } from "../hooks/use-poll";
import { fetchStatus, fetchHealth } from "../api";
import { StatusBar } from "./StatusBar";
import { LogsView } from "./LogsView";
import { TracesView } from "./TracesView";
import { GraphView } from "./GraphView";
import { AgentsView } from "./AgentsView";
import { RagView } from "./RagView";
import { HistoryView, SidebarEntryDetail } from "./HistoryView";
import { ConfigPanel } from "./ConfigPanel";
import { HealthPanel } from "./HealthPanel";
import type { HealthResponse } from "../types";

type Tab = "traces" | "agents" | "rag" | "logs" | "graph" | "history";

const TABS: { id: Tab; label: string }[] = [
  { id: "traces", label: "Traces" },
  { id: "agents", label: "Agents" },
  { id: "rag", label: "RAG" },
  { id: "logs", label: "Logs" },
  { id: "graph", label: "Graph" },
  { id: "history", label: "History" },
];

const FRIENDLY_NAMES: Record<string, string> = {
  "corvia-server": "API Server",
  "corvia-inference": "Inference",
};

// --- Sidebar state types ---

export type SidebarState = "collapsed" | "narrow" | "wide";

export type SidebarContent =
  | { kind: "config" }
  | { kind: "health" }
  | { kind: "cluster"; data: any }
  | { kind: "entry"; data: any }
  | { kind: "agent"; data: any }
  | { kind: "finding"; data: any }
  | { kind: "history"; entryId: string };

const SIDEBAR_WIDTHS: Record<SidebarState, number> = {
  collapsed: 0,
  narrow: 320,
  wide: 480,
};

// --- Sidebar context for child components ---

export interface SidebarAPI {
  openSidebar: (content: SidebarContent, width?: SidebarState) => void;
  closeSidebar: () => void;
  sidebarState: SidebarState;
  sidebarContent: SidebarContent | null;
  navigateToHistory: (entryId: string) => void;
}

export const SidebarContext = createContext<SidebarAPI>({
  openSidebar: () => {},
  closeSidebar: () => {},
  sidebarState: "collapsed",
  sidebarContent: null,
  navigateToHistory: () => {},
});

export function useSidebar(): SidebarAPI {
  return useContext(SidebarContext);
}

// --- Layout ---

export function Layout() {
  const [tab, setTab] = useState<Tab>("traces");
  const [healthData, setHealthData] = useState<HealthResponse | null>(null);
  const [healthLoading, setHealthLoading] = useState(false);

  // Sidebar state
  const [sidebarState, setSidebarState] = useState<SidebarState>("collapsed");
  const [sidebarContent, setSidebarContent] = useState<SidebarContent | null>(null);

  // Cross-tab deeplink state
  const [deeplinkEntryId, setDeeplinkEntryId] = useState<string | null>(null);

  const fetcher = useCallback(() => fetchStatus(), []);
  const { data, error, loading } = usePoll(fetcher, 5000);

  const navigateToTab = useCallback((t: string) => setTab(t as Tab), []);

  /** Navigate to History tab with a specific entry pre-loaded. */
  const navigateToHistory = useCallback((entryId: string) => {
    setDeeplinkEntryId(entryId);
    setTab("history");
  }, []);

  // --- Sidebar API ---

  const openSidebar = useCallback((content: SidebarContent, width: SidebarState = "narrow") => {
    setSidebarContent(content);
    setSidebarState(width);
  }, []);

  const closeSidebar = useCallback(() => {
    setSidebarState("collapsed");
    // Delay clearing content so the collapse animation finishes before content disappears
    setTimeout(() => setSidebarContent(null), 220);
  }, []);

  // Auto-collapse on tab switch and clear deeplink when leaving history
  useEffect(() => {
    setSidebarState("collapsed");
    setSidebarContent(null);
    if (tab !== "history") {
      setDeeplinkEntryId(null);
    }
  }, [tab]);

  // --- Health ---

  const loadHealth = useCallback(async () => {
    openSidebar({ kind: "health" });
    if (healthData) return; // use cached
    setHealthLoading(true);
    try {
      const h = await fetchHealth();
      setHealthData(h);
    } catch { /* ignore */ }
    setHealthLoading(false);
  }, [healthData, openSidebar]);

  const refreshHealth = useCallback(async () => {
    setHealthLoading(true);
    try {
      const h = await fetchHealth();
      setHealthData(h);
    } catch { /* ignore */ }
    setHealthLoading(false);
  }, []);

  // Summarize health for header dots
  const healthSummary = healthData
    ? {
        total: healthData.count,
        hasErrors: healthData.findings.some((f) => f.confidence > 0.7),
        hasWarnings: healthData.findings.some((f) => f.confidence > 0.3 && f.confidence <= 0.7),
      }
    : null;

  const sidebarAPI: SidebarAPI = {
    openSidebar,
    closeSidebar,
    sidebarState,
    sidebarContent,
    navigateToHistory,
  };

  const sidebarWidth = SIDEBAR_WIDTHS[sidebarState];

  return (
    <SidebarContext.Provider value={sidebarAPI}>
      <div class="layout">
        <header class="header">
          <div class="brand">
            <div class="brand-icon">C</div>
            <span class="brand-name">Corvia</span>
          </div>

          <div class="status-pills">
            {data?.services.map((s) => (
              <div class="pill" key={s.name}>
                <div class={`pill-dot ${s.state === "healthy" ? "ok" : s.state === "starting" ? "warn" : "down"}`} />
                <span class="pill-label">{FRIENDLY_NAMES[s.name] ?? s.name}</span>
                {s.latency_ms != null && (
                  <span class="pill-latency">{s.latency_ms.toFixed(1)}ms</span>
                )}
              </div>
            ))}
          </div>

          {/* Health pulse dots */}
          <button
            class={`health-pulse${sidebarContent?.kind === "health" ? " active" : ""}`}
            onClick={loadHealth}
            title="Knowledge health"
          >
            <span class={`health-dot ${healthSummary?.hasErrors ? "red" : healthSummary?.hasWarnings ? "amber" : healthSummary ? "green" : ""}`} />
            <span class={`health-dot ${healthSummary && !healthSummary.hasErrors ? "green" : healthSummary?.hasErrors ? "red" : ""}`} />
            <span class={`health-dot ${healthSummary ? "green" : ""}`} />
            {healthSummary && healthSummary.total > 0 && (
              <span class="health-count">{healthSummary.total}</span>
            )}
          </button>

          <nav class="tabs">
            {TABS.map((t) => (
              <button
                key={t.id}
                class={`tab ${tab === t.id ? "active" : ""}`}
                onClick={() => setTab(t.id)}
              >
                {t.label}
              </button>
            ))}
          </nav>

          <div class="header-right">
            {/* Gear icon for config */}
            <button
              class={`header-config-btn${sidebarContent?.kind === "config" ? " active" : ""}`}
              onClick={() => {
                if (sidebarContent?.kind === "config" && sidebarState !== "collapsed") {
                  closeSidebar();
                } else {
                  openSidebar({ kind: "config" });
                }
              }}
              title="Configuration"
            >
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="8" cy="8" r="2.5" />
                <path d="M6.8 1.5h2.4l.3 1.7a5 5 0 011.2.7l1.6-.6.9 1.5-1.3 1.1a5 5 0 010 1.4l1.3 1.1-.9 1.5-1.6-.6a5 5 0 01-1.2.7l-.3 1.7H6.8l-.3-1.7a5 5 0 01-1.2-.7l-1.6.6-.9-1.5 1.3-1.1a5 5 0 010-1.4L2.8 4.8l.9-1.5 1.6.6a5 5 0 011.2-.7z" />
              </svg>
            </button>
            {data && (
              <span class="scope-badge">{data.config.workspace}</span>
            )}
            <span class="header-time">{new Date().toLocaleTimeString([], { hour12: false })}</span>
          </div>
        </header>

        <div class="main main--sidebar-collapsible">
          <div class="content">
            {loading && !data && <div class="loading">Connecting to corvia-server...</div>}
            {error && !data && <div class="error-banner">Unable to reach corvia-server: {error}</div>}

            {data && <StatusBar data={data} />}

            {tab === "traces" && <TracesView onNavigate={navigateToTab} />}
            {tab === "agents" && <AgentsView navigateToHistory={navigateToHistory} />}
            {tab === "rag" && <RagView navigateToHistory={navigateToHistory} />}
            {tab === "logs" && <LogsView navigateToHistory={navigateToHistory} />}
            {tab === "graph" && <GraphView navigateToHistory={navigateToHistory} />}
            {tab === "history" && <HistoryView deeplinkEntryId={deeplinkEntryId} />}
          </div>

          {/* Chevron toggle — always visible at sidebar edge */}
          <button
            class={`sidebar-toggle${sidebarState !== "collapsed" ? " sidebar-toggle--open" : ""}`}
            style={{ right: `${sidebarWidth}px` }}
            onClick={() => {
              if (sidebarState === "collapsed") {
                openSidebar({ kind: "config" });
              } else {
                closeSidebar();
              }
            }}
            title={sidebarState === "collapsed" ? "Open sidebar" : "Close sidebar"}
          >
            {sidebarState === "collapsed" ? "\u25C0" : "\u25B6"}
          </button>

          {/* Collapsible sidebar */}
          <aside
            class="sidebar sidebar--collapsible"
            style={{
              width: `${sidebarWidth}px`,
              minWidth: sidebarState !== "collapsed" ? `${sidebarWidth}px` : "0px",
            }}
          >
            {sidebarState !== "collapsed" && sidebarContent && (
              <div class="sidebar-inner">
                <div class="sidebar-header">
                  <span class="sidebar-title">
                    {sidebarContent.kind === "config" && "Configuration"}
                    {sidebarContent.kind === "health" && "Knowledge Health"}
                    {sidebarContent.kind === "cluster" && "Cluster Detail"}
                    {sidebarContent.kind === "entry" && "Entry Detail"}
                    {sidebarContent.kind === "agent" && "Agent Detail"}
                    {sidebarContent.kind === "finding" && "Finding Detail"}
                    {sidebarContent.kind === "history" && "History Detail"}
                  </span>
                  <button
                    class="sidebar-close"
                    onClick={closeSidebar}
                    title="Close sidebar"
                  >
                    &#x2715;
                  </button>
                </div>

                {sidebarContent.kind === "config" && (
                  data ? (
                    <ConfigPanel config={data.config} />
                  ) : (
                    <div style={{ color: "var(--text-dim)" }}>Waiting for server...</div>
                  )
                )}
                {sidebarContent.kind === "health" && (
                  <HealthPanel
                    data={healthData}
                    loading={healthLoading}
                    onRefresh={refreshHealth}
                    navigateToHistory={navigateToHistory}
                  />
                )}
                {sidebarContent.kind === "history" && (
                  <SidebarEntryDetail entryId={sidebarContent.entryId} />
                )}
              </div>
            )}
          </aside>
        </div>
      </div>
    </SidebarContext.Provider>
  );
}
