import { useState, useCallback } from "preact/hooks";
import { usePoll } from "../hooks/use-poll";
import { fetchGcStatus, triggerGcRun } from "../api";
import type { GcReportDto } from "../types";

function Sparkline({ history }: { history: GcReportDto[] }) {
  if (history.length === 0) return null;

  const maxDuration = Math.max(...history.map((r) => r.duration_ms), 1);
  const barW = Math.max(4, Math.floor(280 / history.length));
  const h = 60;

  return (
    <svg width={barW * history.length + 4} height={h} class="gc-sparkline">
      {history.map((r, i) => {
        const barH = Math.max(2, (r.duration_ms / maxDuration) * (h - 4));
        const color =
          r.orphans_rolled_back === 0 ? "var(--mint)" :
          r.orphans_rolled_back > 10 ? "var(--coral)" : "var(--amber)";
        return (
          <rect
            key={i}
            x={i * barW + 2}
            y={h - barH - 2}
            width={barW - 2}
            height={barH}
            fill={color}
            rx={1}
          >
            <title>
              {r.started_at}: {r.duration_ms}ms, {r.orphans_rolled_back} orphans
            </title>
          </rect>
        );
      })}
    </svg>
  );
}

function LastRunCard({ report }: { report: GcReportDto }) {
  return (
    <div class="gc-last-run">
      <div class="mini-stats">
        <div class="mini-stat">
          <div class="mini-stat-val">{report.duration_ms}<span style={{ fontSize: "11px" }}>ms</span></div>
          <div class="mini-stat-lbl">Duration</div>
        </div>
        <div class="mini-stat">
          <div class="mini-stat-val">{report.orphans_rolled_back}</div>
          <div class="mini-stat-lbl">Orphans</div>
        </div>
        <div class="mini-stat">
          <div class="mini-stat-val">{report.stale_transitioned}</div>
          <div class="mini-stat-lbl">Stale</div>
        </div>
        <div class="mini-stat">
          <div class="mini-stat-val">{report.closed_sessions_cleaned}</div>
          <div class="mini-stat-lbl">Cleaned</div>
        </div>
        <div class="mini-stat">
          <div class="mini-stat-val">{report.agents_suspended}</div>
          <div class="mini-stat-lbl">Suspended</div>
        </div>
      </div>
      <div style={{ fontSize: "11px", color: "var(--text-dim)", marginTop: "6px" }}>
        Last run: {new Date(report.started_at).toLocaleString()}
      </div>
    </div>
  );
}

export function GcPanel() {
  const [running, setRunning] = useState(false);
  const [triggerResult, setTriggerResult] = useState<GcReportDto | null>(null);
  const [triggerError, setTriggerError] = useState<string | null>(null);
  const fetcher = useCallback(() => fetchGcStatus(), []);
  const { data } = usePoll(fetcher, 10000);

  const handleRun = useCallback(async () => {
    setRunning(true);
    setTriggerError(null);
    try {
      const result = await triggerGcRun();
      setTriggerResult(result);
    } catch (e) {
      setTriggerError(e instanceof Error ? e.message : "GC trigger failed");
    }
    setRunning(false);
  }, []);

  if (!data) return null;

  // Show immediate trigger result until next poll refreshes data
  const displayReport = triggerResult ?? data.last_run;

  return (
    <>
      <div class="trace-card">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "10px" }}>
          <h2 style={{ fontSize: "12px", fontWeight: 600, color: "var(--text-dim)", textTransform: "uppercase", letterSpacing: "0.5px", margin: 0 }}>
            Garbage Collection
          </h2>
          <span style={{ fontSize: "11px", color: "var(--text-dim)" }}>Manual trigger only</span>
        </div>

        {displayReport ? (
          <LastRunCard report={displayReport} />
        ) : (
          <div style={{ fontSize: "12px", color: "var(--text-dim)", padding: "16px 0", textAlign: "center" }}>
            No GC runs yet
          </div>
        )}

        {triggerError && (
          <div class="error-banner" style={{ fontSize: "11px", color: "var(--coral)", marginTop: "6px" }}>
            {triggerError}
          </div>
        )}

        <button
          class="gc-trigger-btn"
          onClick={handleRun}
          disabled={running}
          style={{ marginTop: "12px" }}
        >
          {running ? "Running..." : "Run GC Now"}
        </button>
      </div>

      {data.history.length > 0 && (
        <div class="trace-card">
          <h2 style={{ fontSize: "12px", fontWeight: 600, color: "var(--text-dim)", textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: "10px" }}>
            History ({data.history.length} runs)
          </h2>
          <Sparkline history={data.history} />
        </div>
      )}
    </>
  );
}
