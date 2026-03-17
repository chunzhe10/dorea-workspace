import { useState, useCallback } from "preact/hooks";
import { usePoll } from "../hooks/use-poll";
import { fetchAgents, fetchAgentSessions } from "../api";
import type { AgentRecord, SessionRecord, SessionState, ActivitySummary } from "../types";
import { LiveSessionsBar } from "./LiveSessionsBar";

const STATE_COLORS: Record<SessionState, string> = {
  Created: "var(--text-dim)",
  Active: "var(--mint)",
  Committing: "var(--gold)",
  Merging: "var(--peach)",
  Closed: "var(--text-muted)",
  Stale: "var(--amber)",
  Orphaned: "var(--coral)",
};

function relativeTime(iso: string): string {
  try {
    const diff = Date.now() - new Date(iso).getTime();
    if (diff < 60_000) return `${Math.floor(diff / 1000)}s ago`;
    if (diff < 3_600_000) return `${Math.floor(diff / 60_000)}m ago`;
    if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)}h ago`;
    return `${Math.floor(diff / 86_400_000)}d ago`;
  } catch {
    return iso;
  }
}

function HeartbeatDot({ status, lastSeen }: { status: string; lastSeen: string }) {
  const diff = Date.now() - new Date(lastSeen).getTime();
  const isRecent = diff < 60_000; // within 1 minute
  const cls =
    status === "Suspended" ? "hb-suspended" :
    isRecent ? "hb-active" :
    diff < 300_000 ? "hb-stale" : "hb-dead";
  return <span class={`heartbeat-dot ${cls}`} />;
}

function SessionTimeline({ sessions }: { sessions: SessionRecord[] }) {
  if (sessions.length === 0) {
    return <div class="agent-empty">No sessions</div>;
  }

  return (
    <div class="session-list">
      {sessions.map((s) => (
        <div class="session-row" key={s.session_id}>
          <div class="session-state-badge" style={{ color: STATE_COLORS[s.state], borderColor: STATE_COLORS[s.state] }}>
            {s.state}
          </div>
          <div class="session-info">
            <span class="session-id">{s.session_id.split("/").pop()}</span>
            <span class="session-meta">
              {s.entries_written} written &middot; {s.entries_merged} merged
            </span>
          </div>
          <span class="session-time">{relativeTime(s.created_at)}</span>
        </div>
      ))}
    </div>
  );
}

function AgentCard({
  agent,
  expanded,
  onToggle,
}: {
  agent: AgentRecord;
  expanded: boolean;
  onToggle: () => void;
}) {
  const [sessions, setSessions] = useState<SessionRecord[] | null>(null);
  const [sessLoading, setSessLoading] = useState(false);

  const handleToggle = useCallback(async () => {
    onToggle();
    if (!expanded && sessions === null) {
      setSessLoading(true);
      try {
        const s = await fetchAgentSessions(agent.agent_id);
        setSessions(s);
      } catch { /* ignore */ }
      setSessLoading(false);
    }
  }, [expanded, sessions, agent.agent_id, onToggle]);

  const scopes =
    agent.permissions === "ReadOnly" ? "read-only" :
    agent.permissions === "Admin" ? "admin" :
    typeof agent.permissions === "object" && "ReadWrite" in agent.permissions
      ? agent.permissions.ReadWrite.scopes.join(", ")
      : "";

  const openSessions = sessions?.filter((s) =>
    s.state === "Active" || s.state === "Committing" || s.state === "Merging"
  ).length ?? 0;

  const totalSessions = sessions?.length ?? 0;

  return (
    <div class={`agent-card${expanded ? " expanded" : ""}`}>
      <div class="agent-card-header" onClick={handleToggle}>
        <HeartbeatDot status={agent.status} lastSeen={agent.last_seen} />
        <div class="agent-card-info">
          <div class="agent-name">{agent.display_name}</div>
          <div class="agent-id">{agent.agent_id}</div>
          {agent.description && (
            <div class="agent-description">{agent.description}</div>
          )}
          {agent.activity_summary && agent.activity_summary.topic_tags.length > 0 && (
            <div class="agent-topics">
              {agent.activity_summary.topic_tags.map((tag) => (
                <span class="topic-pill" key={tag}>{tag}</span>
              ))}
              {agent.activity_summary.drifted && (
                <span
                  class="drift-indicator"
                  title={`Last session: ${agent.activity_summary.last_topics.join(", ")}`}
                >
                  drifted
                </span>
              )}
            </div>
          )}
        </div>
        <div class="agent-card-stats">
          <span class="agent-stat">
            <span class="agent-stat-label">type</span>
            <span class="agent-stat-val">{agent.identity_type}</span>
          </span>
          {agent.activity_summary && (
            <span class="agent-stat">
              <span class="agent-stat-label">entries</span>
              <span class="agent-stat-val">{agent.activity_summary.entry_count}</span>
            </span>
          )}
          {sessions !== null && (
            <span class="agent-stat">
              <span class="agent-stat-label">sessions</span>
              <span class="agent-stat-val">{openSessions}/{totalSessions}</span>
            </span>
          )}
          <span class="agent-stat">
            <span class="agent-stat-label">seen</span>
            <span class="agent-stat-val">{relativeTime(agent.last_seen)}</span>
          </span>
        </div>
        <span class="agent-expand-icon">{expanded ? "\u25B2" : "\u25BC"}</span>
      </div>

      {expanded && (
        <div class="agent-card-body">
          <div class="agent-detail-row">
            <span class="agent-detail-label">Registered</span>
            <span class="agent-detail-val">{new Date(agent.registered_at).toLocaleString()}</span>
          </div>
          <div class="agent-detail-row">
            <span class="agent-detail-label">Scopes</span>
            <span class="agent-detail-val">{scopes}</span>
          </div>
          <div class="agent-detail-row">
            <span class="agent-detail-label">Status</span>
            <span class="agent-detail-val" style={{ color: agent.status === "Active" ? "var(--mint)" : "var(--coral)" }}>
              {agent.status}
            </span>
          </div>
          {agent.description && (
            <div class="agent-detail-row">
              <span class="agent-detail-label">Purpose</span>
              <span class="agent-detail-val">{agent.description}</span>
            </div>
          )}

          {agent.activity_summary && (
            <>
              <h3 class="agent-section-title">Activity Summary</h3>
              <div class="agent-detail-row">
                <span class="agent-detail-label">Entries</span>
                <span class="agent-detail-val">{agent.activity_summary.entry_count}</span>
              </div>
              <div class="agent-detail-row">
                <span class="agent-detail-label">Sessions</span>
                <span class="agent-detail-val">{agent.activity_summary.session_count}</span>
              </div>
              {agent.activity_summary.topic_tags.length > 0 && (
                <div class="agent-detail-row">
                  <span class="agent-detail-label">Topics</span>
                  <span class="agent-detail-val">
                    {agent.activity_summary.topic_tags.map((tag) => (
                      <span class="topic-pill" key={tag}>{tag}</span>
                    ))}
                  </span>
                </div>
              )}
              {agent.activity_summary.drifted && agent.activity_summary.last_topics.length > 0 && (
                <div class="agent-detail-row">
                  <span class="agent-detail-label">Drift</span>
                  <span class="agent-detail-val drift-indicator">
                    Last session topics: {agent.activity_summary.last_topics.join(", ")}
                  </span>
                </div>
              )}
              <div class="agent-detail-row">
                <span class="agent-detail-label">Last active</span>
                <span class="agent-detail-val">{relativeTime(agent.activity_summary.last_active)}</span>
              </div>
            </>
          )}

          <h3 class="agent-section-title">Sessions</h3>
          {sessLoading && <div class="agent-empty">Loading sessions...</div>}
          {sessions && <SessionTimeline sessions={sessions} />}
        </div>
      )}
    </div>
  );
}

export function AgentsView({ navigateToHistory }: { navigateToHistory?: (entryId: string) => void }) {
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const fetcher = useCallback(() => fetchAgents(), []);
  const { data, error, loading } = usePoll(fetcher, 5000);

  if (loading && !data) return <div class="loading">Loading agents...</div>;
  if (error) return <div class="error-banner">{error}</div>;
  if (!data) return null;

  return (
    <div class="agents-view">
      <LiveSessionsBar onSessionClick={(agentId) => setExpandedId(agentId)} />
      <div class="agents-header">
        <h2>Registered Agents</h2>
        <span class="agents-count">{data.length} agents</span>
      </div>

      {data.length === 0 ? (
        <div class="card">
          <div style={{ color: "var(--text-dim)", textAlign: "center", padding: "40px 20px" }}>
            No agents registered. Agents appear here when they connect via the REST or MCP API.
          </div>
        </div>
      ) : (
        <div class="agents-grid">
          {data.map((agent) => (
            <AgentCard
              key={agent.agent_id}
              agent={agent}
              expanded={expandedId === agent.agent_id}
              onToggle={() => setExpandedId(expandedId === agent.agent_id ? null : agent.agent_id)}
            />
          ))}
        </div>
      )}
    </div>
  );
}
