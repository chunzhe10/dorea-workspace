import { useCallback } from "preact/hooks";
import { usePoll } from "../hooks/use-poll";
import { fetchLiveSessions } from "../api";
import type { LiveSession } from "../types";

const STATE_DOT_COLORS: Record<string, string> = {
  Active: "var(--mint)",
  Stale: "var(--amber)",
  Created: "var(--text-dim)",
  Committing: "var(--gold)",
  Merging: "var(--peach)",
  Orphaned: "var(--coral)",
};

function formatDuration(secs: number): string {
  if (secs < 60) return `${secs}s`;
  if (secs < 3600) return `${Math.floor(secs / 60)}m`;
  return `${Math.floor(secs / 3600)}h ${Math.floor((secs % 3600) / 60)}m`;
}

function SessionPill({ session, onClick }: { session: LiveSession; onClick?: () => void }) {
  const dotColor = STATE_DOT_COLORS[session.state] ?? "var(--text-dim)";
  return (
    <button class="session-pill" onClick={onClick} title={`${session.agent_name} — ${session.state}`}>
      <span class="session-pill-dot" style={{ background: dotColor }} />
      <span class="session-pill-name">{session.agent_name}</span>
      <span class="session-pill-stat">{session.pending_entries} pending</span>
      <span class="session-pill-time">{formatDuration(session.duration_secs)}</span>
    </button>
  );
}

export function LiveSessionsBar({ onSessionClick }: { onSessionClick?: (agentId: string) => void }) {
  const fetcher = useCallback(() => fetchLiveSessions(), []);
  const { data } = usePoll(fetcher, 5000);

  if (!data || data.sessions.length === 0) return null;

  return (
    <div class="live-sessions-bar">
      <div class="live-sessions-header">
        <span class="live-sessions-label">Live Sessions</span>
        <span class="live-sessions-count">
          {data.summary.total_active} active
          {data.summary.total_stale > 0 && ` \u00b7 ${data.summary.total_stale} stale`}
        </span>
      </div>
      <div class="live-sessions-pills">
        {data.sessions.map((s) => (
          <SessionPill
            key={s.session_id}
            session={s}
            onClick={() => onSessionClick?.(s.agent_id)}
          />
        ))}
      </div>
    </div>
  );
}
