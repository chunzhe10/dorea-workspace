import { useState } from "preact/hooks";
import type { HealthResponse, HealthFinding } from "../types";

const CHECK_LABELS: Record<string, { label: string; icon: string }> = {
  stale_entry: { label: "Stale Entries", icon: "\u23F0" },
  broken_chain: { label: "Broken Chains", icon: "\u26D3" },
  orphaned_node: { label: "Orphaned Entries", icon: "\u{1F47B}" },
  dangling_import: { label: "Dangling Edges", icon: "\u{1F517}" },
  dependency_cycle: { label: "Cycles", icon: "\u{1F504}" },
  misplaced_doc: { label: "Misplaced Doc", icon: "\u{1F4C1}" },
  temporal_contradiction: { label: "Temporal Contradiction", icon: "\u26A1" },
  coverage_gap: { label: "Coverage Gap", icon: "\u{1F4DD}" },
};

function severityColor(confidence: number): string {
  if (confidence > 0.7) return "var(--coral)";
  if (confidence > 0.3) return "var(--amber)";
  return "var(--mint)";
}

function FindingRow({
  finding,
  navigateToHistory,
}: {
  finding: HealthFinding;
  navigateToHistory?: (entryId: string) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const meta = CHECK_LABELS[finding.check_type] ?? { label: finding.check_type, icon: "\u2753" };
  const color = severityColor(finding.confidence);

  return (
    <div class="health-finding" onClick={() => setExpanded(!expanded)}>
      <div class="health-finding-header">
        <span class="health-finding-icon">{meta.icon}</span>
        <span class="health-finding-label">{meta.label}</span>
        <span class="health-finding-badge" style={{ background: color }}>
          {Math.round(finding.confidence * 100)}%
        </span>
      </div>
      <div class="health-finding-rationale">{finding.rationale}</div>
      {expanded && finding.target_ids.length > 0 && (
        <div class="health-finding-ids">
          {finding.target_ids.slice(0, 5).map((id) => (
            <code
              key={id}
              class={`health-entry-id${navigateToHistory ? " entry-link" : ""}`}
              onClick={(e) => {
                if (navigateToHistory) {
                  e.stopPropagation();
                  navigateToHistory(id);
                }
              }}
              title={navigateToHistory ? `View history for ${id}` : id}
            >
              {id.slice(0, 8)}
            </code>
          ))}
          {finding.target_ids.length > 5 && (
            <span style={{ fontSize: "10px", color: "var(--text-dim)" }}>
              +{finding.target_ids.length - 5} more
            </span>
          )}
        </div>
      )}
    </div>
  );
}

interface Props {
  data: HealthResponse | null;
  loading: boolean;
  onRefresh: () => void;
  navigateToHistory?: (entryId: string) => void;
}

export function HealthPanel({ data, loading, onRefresh, navigateToHistory }: Props) {
  if (loading && !data) {
    return <div class="health-panel"><div class="loading">Running health checks...</div></div>;
  }

  if (!data) {
    return (
      <div class="health-panel">
        <div style={{ color: "var(--text-dim)", textAlign: "center", padding: "20px 0" }}>
          Click to run health checks
        </div>
      </div>
    );
  }

  // Group findings by check_type
  const grouped = new Map<string, HealthFinding[]>();
  for (const f of data.findings) {
    const arr = grouped.get(f.check_type) ?? [];
    arr.push(f);
    grouped.set(f.check_type, arr);
  }

  // All check types (show green for missing ones)
  const allChecks = ["stale_entry", "broken_chain", "orphaned_node", "dangling_import", "dependency_cycle", "misplaced_doc", "temporal_contradiction", "coverage_gap"];

  return (
    <div class="health-panel">
      <div class="health-panel-header">
        <h3>Knowledge Health</h3>
        <button class="health-refresh" onClick={onRefresh} disabled={loading}>
          {loading ? "\u21BB" : "\u21BB Run"}
        </button>
      </div>

      {/* Summary dots */}
      <div class="health-summary">
        {allChecks.map((type) => {
          const meta = CHECK_LABELS[type] ?? { label: type, icon: "?" };
          const findings = grouped.get(type);
          const color = findings
            ? findings.some((f) => f.confidence > 0.7)
              ? "var(--coral)"
              : "var(--amber)"
            : "var(--mint)";
          return (
            <div class="health-check-dot" key={type} title={meta.label}>
              <span class="health-check-indicator" style={{ background: color }} />
              <span class="health-check-name">{meta.label}</span>
              <span class="health-check-count" style={{ color }}>
                {findings ? findings.length : 0}
              </span>
            </div>
          );
        })}
      </div>

      {/* Findings detail */}
      {data.findings.length === 0 ? (
        <div style={{ color: "var(--mint)", textAlign: "center", padding: "16px 0", fontSize: "13px" }}>
          All checks passed
        </div>
      ) : (
        <div class="health-findings">
          {data.findings.map((f, i) => (
            <FindingRow key={i} finding={f} navigateToHistory={navigateToHistory} />
          ))}
        </div>
      )}
    </div>
  );
}
