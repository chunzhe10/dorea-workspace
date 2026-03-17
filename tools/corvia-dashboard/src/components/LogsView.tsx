import { useState, useCallback } from "preact/hooks";
import { usePoll } from "../hooks/use-poll";
import { fetchLogs } from "../api";

/** Regex for UUID v7 (or any UUID) in log messages. */
const UUID_RE = /\b([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\b/gi;

/** Renders a log message with UUIDs as clickable links. */
function LogMessage({
  message,
  navigateToHistory,
}: {
  message: string;
  navigateToHistory?: (entryId: string) => void;
}) {
  if (!navigateToHistory) {
    return <>{message}</>;
  }
  const parts: (string | { uuid: string })[] = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;
  UUID_RE.lastIndex = 0;
  while ((match = UUID_RE.exec(message)) !== null) {
    if (match.index > lastIndex) {
      parts.push(message.slice(lastIndex, match.index));
    }
    parts.push({ uuid: match[1] });
    lastIndex = match.index + match[0].length;
  }
  if (lastIndex < message.length) {
    parts.push(message.slice(lastIndex));
  }
  if (parts.length <= 1 && typeof parts[0] === "string") {
    return <>{message}</>;
  }
  return (
    <>
      {parts.map((p, i) =>
        typeof p === "string" ? (
          p
        ) : (
          <span
            key={i}
            class="entry-link"
            onClick={(e) => {
              e.stopPropagation();
              navigateToHistory(p.uuid);
            }}
            title={`View history for ${p.uuid}`}
          >
            {p.uuid}
          </span>
        ),
      )}
    </>
  );
}

export function LogsView({ navigateToHistory }: { navigateToHistory?: (entryId: string) => void }) {
  const [module, setModule] = useState("");
  const [level, setLevel] = useState("");

  const fetcher = useCallback(
    () => fetchLogs({ module: module || undefined, level: level || undefined, limit: 200 }),
    [module, level],
  );
  const { data, error, loading } = usePoll(fetcher, 5000);

  return (
    <div class="card">
      <div class="filters">
        <select value={level} onChange={(e) => setLevel((e.target as HTMLSelectElement).value)}>
          <option value="">All levels</option>
          <option value="ERROR">ERROR</option>
          <option value="WARN">WARN</option>
          <option value="INFO">INFO</option>
          <option value="DEBUG">DEBUG</option>
          <option value="TRACE">TRACE</option>
        </select>
        <input
          type="text"
          placeholder="Filter module..."
          value={module}
          onInput={(e) => setModule((e.target as HTMLInputElement).value)}
        />
        {data && (
          <span style={{ color: "var(--text-dim)", fontSize: "12px", alignSelf: "center", fontFamily: "var(--font-mono)" }}>
            {data.total} entries
          </span>
        )}
      </div>

      {loading && <div class="loading">Loading logs...</div>}
      {error && <div class="error-banner">{error}</div>}

      {data && data.entries.length > 0 && (
        <table>
          <thead>
            <tr>
              <th style={{ width: "120px" }}>Time</th>
              <th style={{ width: "60px" }}>Level</th>
              <th style={{ width: "140px" }}>Module</th>
              <th>Message</th>
            </tr>
          </thead>
          <tbody>
            {data.entries.map((entry, i) => (
              <tr key={i}>
                <td style={{ color: "var(--text-dim)" }}>{entry.timestamp}</td>
                <td class={`level-${entry.level}`}>{entry.level}</td>
                <td style={{ color: "var(--text-muted)" }}>{entry.module}</td>
                <td style={{ color: "var(--text-primary)", wordBreak: "break-all" }}>
                  <LogMessage message={entry.message} navigateToHistory={navigateToHistory} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {data && data.entries.length === 0 && (
        <div style={{ color: "var(--text-dim)", textAlign: "center", padding: "20px" }}>No log entries</div>
      )}
    </div>
  );
}

