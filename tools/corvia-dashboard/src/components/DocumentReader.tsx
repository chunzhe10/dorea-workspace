import { useEffect, useState } from "preact/hooks";
import { fetchEntryDetail, fetchEntryNeighbors } from "../api";
import { NeighborCards } from "./NeighborCards";
import type { EntryDetail, NeighborEntry } from "../types";

interface DocumentReaderProps {
  entryId: string | null;
  onNavigate: (entryId: string) => void;
  onClose: () => void;
  navigateToHistory?: (entryId: string) => void;
}

export function DocumentReader({ entryId, onNavigate, onClose, navigateToHistory }: DocumentReaderProps) {
  const [entry, setEntry] = useState<EntryDetail | null>(null);
  const [neighbors, setNeighbors] = useState<NeighborEntry[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!entryId) {
      setEntry(null);
      setNeighbors([]);
      return;
    }
    setLoading(true);
    Promise.all([
      fetchEntryDetail(entryId),
      fetchEntryNeighbors(entryId),
    ]).then(([detail, nbrs]) => {
      setEntry(detail);
      setNeighbors(nbrs.neighbors);
    }).finally(() => setLoading(false));
  }, [entryId]);

  if (!entryId) {
    return (
      <div class="reader-placeholder">
        <p>Click a node to view its content</p>
      </div>
    );
  }

  if (loading) {
    return <div class="reader-loading">Loading...</div>;
  }

  if (!entry) {
    return <div class="reader-error">Entry not found</div>;
  }

  return (
    <div class="document-reader">
      <div class="reader-header">
        <h3>{entry.metadata.source_file ?? entry.id}</h3>
        <button class="reader-close" onClick={onClose}>&times;</button>
      </div>

      <div class="reader-meta">
        {entry.metadata.content_role && (
          <span class="badge badge-role">{entry.metadata.content_role}</span>
        )}
        {entry.metadata.source_origin && (
          <span class="badge badge-origin">{entry.metadata.source_origin}</span>
        )}
        <span class="reader-date">
          {new Date(entry.recorded_at).toLocaleDateString()}
        </span>
        {navigateToHistory && (
          <button
            class="history-link"
            onClick={() => navigateToHistory(entry.id)}
            style={{
              background: "none",
              border: "none",
              color: "var(--lavender)",
              cursor: "pointer",
              fontSize: "12px",
              fontWeight: 600,
              padding: "2px 0",
              marginLeft: "auto",
            }}
          >
            View history &rarr;
          </button>
        )}
      </div>

      <div class="reader-content">
        <pre>{entry.content}</pre>
      </div>

      <NeighborCards neighbors={neighbors} onSelect={onNavigate} />
    </div>
  );
}
