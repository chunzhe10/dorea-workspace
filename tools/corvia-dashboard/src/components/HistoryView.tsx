import { useState, useEffect, useCallback } from "preact/hooks";
import { fetchActivityFeed, fetchEntryHistory, fetchEntryDetail } from "../api";
import { useSidebar } from "./Layout";
import type {
  ActivityItem,
  ActivityFeedResponse,
  HistoryEntry,
  HistoryResponse,
  EntryDetail,
} from "../types";

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

function agentColor(agentId?: string): string {
  if (!agentId) return "#666";
  let hash = 0;
  for (const ch of agentId) hash = ((hash << 5) - hash) + ch.charCodeAt(0);
  const hue = Math.abs(hash) % 360;
  return `hsl(${hue}, 60%, 50%)`;
}

function relativeTime(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

/** When groups are collapsed, show only the first item per group. */
function groupFeedItems(
  items: ActivityItem[],
  expandedGroups: Set<string>,
): ActivityItem[] {
  const seen = new Set<string>();
  return items.filter((item) => {
    if (!item.group_id) return true;
    if (expandedGroups.has(item.group_id)) return true; // group expanded
    if (seen.has(item.group_id)) return false;
    seen.add(item.group_id);
    return true;
  });
}

function formatDelta(bytes?: number): string {
  if (bytes == null) return "";
  if (bytes >= 0) return `+${bytes}`;
  return String(bytes);
}

// ---------------------------------------------------------------------------
// Diff utilities (kept from old HistoryView for sidebar detail)
// ---------------------------------------------------------------------------

function computeDiff(
  a: string,
  b: string,
): { type: "same" | "add" | "del"; text: string }[] {
  const aLines = a.split("\n");
  const bLines = b.split("\n");

  const m = aLines.length;
  const n = bLines.length;
  const dp: number[][] = Array.from({ length: m + 1 }, () =>
    Array(n + 1).fill(0),
  );
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      dp[i][j] =
        aLines[i - 1] === bLines[j - 1]
          ? dp[i - 1][j - 1] + 1
          : Math.max(dp[i - 1][j], dp[i][j - 1]);
    }
  }

  let i = m;
  let j = n;
  const ops: { type: "same" | "add" | "del"; text: string }[] = [];
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && aLines[i - 1] === bLines[j - 1]) {
      ops.push({ type: "same", text: aLines[i - 1] });
      i--;
      j--;
    } else if (j > 0 && (i === 0 || dp[i][j - 1] >= dp[i - 1][j])) {
      ops.push({ type: "add", text: bLines[j - 1] });
      j--;
    } else {
      ops.push({ type: "del", text: aLines[i - 1] });
      i--;
    }
  }
  ops.reverse();
  return ops;
}

function formatTimestamp(iso: string): string {
  try {
    const d = new Date(iso);
    return d.toLocaleString(undefined, {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    });
  } catch {
    return iso;
  }
}

function truncate(s: string, maxLen: number): string {
  return s.length > maxLen ? s.slice(0, maxLen) + "..." : s;
}

// ---------------------------------------------------------------------------
// Feed item component
// ---------------------------------------------------------------------------

function FeedItem({
  item,
  expandedGroups,
  onExpandGroup,
  onClickItem,
}: {
  item: ActivityItem;
  expandedGroups: Set<string>;
  onExpandGroup: (groupId: string) => void;
  onClickItem: (entryId: string) => void;
}) {
  const isGroupHeader =
    item.group_id &&
    item.group_count &&
    item.group_count > 1 &&
    !expandedGroups.has(item.group_id);

  return (
    <div
      class="feed-item"
      onClick={() => onClickItem(item.entry_id)}
    >
      <span
        class="agent-dot"
        style={{ background: agentColor(item.agent_id) }}
      />
      <span class="feed-agent">
        {item.agent_name || item.agent_id || "unknown"}
      </span>
      <span class="feed-action">{item.action}</span>
      <span class="feed-title" title={item.title}>
        {truncate(item.title, 60)}
      </span>
      {item.topic_tags.length > 0 && (
        <span class="feed-topics">
          {item.topic_tags.map((t) => (
            <span class="topic-pill small" key={t}>
              {truncate(t, 20)}
            </span>
          ))}
        </span>
      )}
      <span
        class={`feed-delta ${
          (item.delta_bytes || 0) >= 0 ? "positive" : "negative"
        }`}
      >
        {formatDelta(item.delta_bytes)}
      </span>
      <span class="feed-time">{relativeTime(item.recorded_at)}</span>
      {isGroupHeader && (
        <button
          class="expand-group"
          onClick={(e) => {
            e.stopPropagation();
            onExpandGroup(item.group_id!);
          }}
        >
          +{item.group_count! - 1} more
        </button>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sidebar entry detail (timeline + diff)
// ---------------------------------------------------------------------------

export function SidebarEntryDetail({ entryId }: { entryId: string }) {
  const [history, setHistory] = useState<HistoryResponse | null>(null);
  const [detail, setDetail] = useState<EntryDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedIdx, setSelectedIdx] = useState(0);
  const [diffPair, setDiffPair] = useState<[number, number] | null>(null);

  useEffect(() => {
    setLoading(true);
    setSelectedIdx(0);
    setDiffPair(null);
    Promise.all([fetchEntryHistory(entryId), fetchEntryDetail(entryId)])
      .then(([h, d]) => {
        setHistory(h);
        setDetail(d);
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [entryId]);

  if (loading) return <div class="loading">Loading entry...</div>;

  const chain = history?.chain ?? [];

  return (
    <div class="sidebar-entry-detail">
      {/* Entry metadata */}
      {detail && (
        <div class="entry-meta-card">
          <div class="meta-row">
            <span class="meta-key">ID</span>
            <span class="meta-val mono">{detail.id}</span>
          </div>
          {detail.metadata.source_file && (
            <div class="meta-row">
              <span class="meta-key">Source</span>
              <span class="meta-val">{detail.metadata.source_file}</span>
            </div>
          )}
          <div class="meta-row">
            <span class="meta-key">Recorded</span>
            <span class="meta-val">{formatTimestamp(detail.recorded_at)}</span>
          </div>
          {detail.superseded_by && (
            <div class="meta-row">
              <span class="meta-key">Superseded by</span>
              <span class="meta-val mono">{detail.superseded_by}</span>
            </div>
          )}
        </div>
      )}

      {/* Timeline */}
      {chain.length > 0 && (
        <div class="sidebar-timeline">
          <div class="timeline-count">
            {chain.length} version{chain.length !== 1 ? "s" : ""}
          </div>
          {chain.map((entry, i) => (
            <button
              key={entry.id}
              class={`timeline-node${selectedIdx === i ? " selected" : ""}`}
              onClick={() => {
                setSelectedIdx(i);
                setDiffPair(null);
              }}
            >
              <div class="timeline-connector">
                <div
                  class={`timeline-dot${entry.is_current ? " current" : ""}`}
                />
                {i < chain.length - 1 && <div class="timeline-line" />}
              </div>
              <div class="timeline-content">
                <div class="timeline-header">
                  <span class="timeline-version">
                    v{chain.length - i}
                  </span>
                  {entry.is_current && (
                    <span class="timeline-badge current">current</span>
                  )}
                </div>
                <div class="timeline-time">
                  {formatTimestamp(entry.valid_from)}
                </div>
                <div class="timeline-preview">
                  {truncate(entry.content, 80)}
                </div>
              </div>
            </button>
          ))}

          {/* Diff controls */}
          {chain.length >= 2 && (
            <div class="diff-controls">
              <div class="diff-controls-label">Compare versions</div>
              <div class="diff-selectors">
                <select
                  class="diff-select"
                  value={diffPair ? String(diffPair[0]) : ""}
                  onChange={(e) => {
                    const v = parseInt(
                      (e.target as HTMLSelectElement).value,
                    );
                    if (!isNaN(v))
                      setDiffPair([v, diffPair?.[1] ?? Math.max(v - 1, 0)]);
                  }}
                >
                  <option value="" disabled>
                    Older...
                  </option>
                  {chain.map((_, i) => (
                    <option key={i} value={String(i)}>
                      v{chain.length - i}
                    </option>
                  ))}
                </select>
                <span class="diff-vs">vs</span>
                <select
                  class="diff-select"
                  value={diffPair ? String(diffPair[1]) : ""}
                  onChange={(e) => {
                    const v = parseInt(
                      (e.target as HTMLSelectElement).value,
                    );
                    if (!isNaN(v))
                      setDiffPair([
                        diffPair?.[0] ?? Math.min(v + 1, chain.length - 1),
                        v,
                      ]);
                  }}
                >
                  <option value="" disabled>
                    Newer...
                  </option>
                  {chain.map((_, i) => (
                    <option key={i} value={String(i)}>
                      v{chain.length - i}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Diff view */}
      {diffPair &&
        chain[diffPair[0]] &&
        chain[diffPair[1]] &&
        diffPair[0] !== diffPair[1] && (
          <div class="history-diff">
            <div class="diff-header">
              <span class="diff-label del-label">
                v{chain.length - diffPair[0]}
              </span>
              <span class="diff-arrow">&rarr;</span>
              <span class="diff-label add-label">
                v{chain.length - diffPair[1]}
              </span>
            </div>
            <pre class="diff-body">
              {computeDiff(
                chain[diffPair[0]].content,
                chain[diffPair[1]].content,
              ).map((line, i) => (
                <div key={i} class={`diff-line ${line.type}`}>
                  <span class="diff-marker">
                    {line.type === "add"
                      ? "+"
                      : line.type === "del"
                        ? "-"
                        : " "}
                  </span>
                  <span class="diff-text">{line.text}</span>
                </div>
              ))}
            </pre>
          </div>
        )}

      {/* Selected entry content */}
      {chain[selectedIdx] && (
        <div class="entry-detail-card">
          <div class="content-label">Content</div>
          <pre class="content-body">{chain[selectedIdx].content}</pre>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main HistoryView component
// ---------------------------------------------------------------------------

export interface HistoryViewProps {
  deeplinkEntryId?: string | null;
}

export function HistoryView({ deeplinkEntryId }: HistoryViewProps) {
  const [feed, setFeed] = useState<ActivityFeedResponse | null>(null);
  const [selectedTopic, setSelectedTopic] = useState<string | null>(null);
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null);
  const [expandedGroups, setExpandedGroups] = useState<Set<string>>(
    new Set(),
  );
  const [loading, setLoading] = useState(true);

  const { openSidebar } = useSidebar();

  // Load activity feed
  useEffect(() => {
    setLoading(true);
    const load = () =>
      fetchActivityFeed({
        limit: 100,
        topic: selectedTopic || undefined,
        agent: selectedAgent || undefined,
      })
        .then((f) => {
          setFeed(f);
          setLoading(false);
        })
        .catch((err) => {
          console.error(err);
          setLoading(false);
        });
    load();
    const interval = setInterval(load, 10000);
    return () => clearInterval(interval);
  }, [selectedTopic, selectedAgent]);

  // Handle deeplink from other tabs
  useEffect(() => {
    if (deeplinkEntryId) {
      openSidebar({ kind: "history", entryId: deeplinkEntryId }, "wide");
    }
  }, [deeplinkEntryId]);

  const handleExpandGroup = useCallback((groupId: string) => {
    setExpandedGroups((prev) => new Set([...prev, groupId]));
  }, []);

  const handleClickItem = useCallback(
    (entryId: string) => {
      openSidebar({ kind: "history", entryId }, "wide");
    },
    [openSidebar],
  );

  // Collect unique agents from feed for the dropdown
  const agents: string[] = feed
    ? [
        ...new Set(
          feed.items
            .map((i) => i.agent_id)
            .filter((a): a is string => !!a),
        ),
      ]
    : [];

  if (loading && !feed)
    return <div class="loading">Loading activity feed...</div>;

  if (!feed)
    return (
      <div class="card">
        <div
          style={{
            color: "var(--text-dim)",
            textAlign: "center",
            padding: "40px 20px",
          }}
        >
          Unable to load activity feed.
        </div>
      </div>
    );

  const visibleItems = groupFeedItems(feed.items, expandedGroups);

  return (
    <div class="history-view">
      {/* Filter bar */}
      <div class="activity-filters">
        {/* Topic pills */}
        <div class="topic-filters">
          {feed.topics.map((topic) => (
            <button
              key={topic}
              class={`topic-pill${selectedTopic === topic ? " active" : ""}`}
              onClick={() =>
                setSelectedTopic(selectedTopic === topic ? null : topic)
              }
              title={topic}
            >
              {truncate(topic, 24)}
            </button>
          ))}
        </div>

        {/* Agent dropdown */}
        {agents.length > 0 && (
          <select
            class="agent-filter"
            value={selectedAgent || ""}
            onChange={(e) => {
              const val = (e.target as HTMLSelectElement).value;
              setSelectedAgent(val || null);
            }}
          >
            <option value="">All agents</option>
            {agents.map((a) => (
              <option key={a} value={a}>
                {a}
              </option>
            ))}
          </select>
        )}

        <span class="feed-count">
          {feed.total} {feed.total === 1 ? "entry" : "entries"}
        </span>
      </div>

      {/* Activity feed */}
      {visibleItems.length === 0 ? (
        <div class="card">
          <div
            style={{
              color: "var(--text-dim)",
              textAlign: "center",
              padding: "40px 20px",
            }}
          >
            No activity found
            {selectedTopic ? ` for topic "${selectedTopic}"` : ""}
            {selectedAgent ? ` by agent "${selectedAgent}"` : ""}.
          </div>
        </div>
      ) : (
        <div class="activity-feed">
          {visibleItems.map((item) => (
            <FeedItem
              key={item.entry_id}
              item={item}
              expandedGroups={expandedGroups}
              onExpandGroup={handleExpandGroup}
              onClickItem={handleClickItem}
            />
          ))}
        </div>
      )}
    </div>
  );
}
