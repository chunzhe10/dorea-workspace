# Task: Temporal History Explorer

> **You are Agent B** in a parallel development setup. Another agent (Agent A) is simultaneously
> building the Interactive Knowledge Graph in a separate worktree. Coordinate via corvia MCP.

## Goal

Build a new Temporal History Explorer tab that shows entry supersession chains,
diff views between versions, and a "time travel" slider.

## What exists today

- No history view exists in the dashboard
- Backend API: `GET /v1/entries/{id}/history` returns supersession chain (REST API in `rest.rs`)
- Backend API: `GET /v1/entries/{id}` returns a single entry
- Dashboard has 5 tabs: Traces, Agents, RAG, Logs, Graph — defined in `Layout.tsx`
- Tab type is `type Tab = "traces" | "agents" | "rag" | "logs" | "graph"`

## Implementation Plan

### 1. Backend: Add history endpoint to dashboard API

**File:** `repos/corvia/crates/corvia-server/src/dashboard/mod.rs`

Add two new endpoints:

```rust
.route("/api/dashboard/entries/{id}", get(entry_detail_handler))
.route("/api/dashboard/entries/{id}/history", get(entry_history_handler))
```

`entry_detail_handler`: thin wrapper around `state.store.get(&uuid)`, returns the full entry JSON.

`entry_history_handler`: calls `state.store.history(&uuid)`, returns the supersession chain as:
```json
{
  "entry_id": "...",
  "chain": [
    { "id": "...", "content": "...", "created_at": "...", "superseded_by": "..." },
    ...
  ],
  "count": 3
}
```

### 2. Frontend: Add API functions

**File:** `tools/corvia-dashboard/src/api.ts`

```typescript
// --- History ---
export function fetchEntryDetail(entryId: string): Promise<EntryDetail> {
  return get(`/entries/${encodeURIComponent(entryId)}`);
}

export function fetchEntryHistory(entryId: string): Promise<HistoryResponse> {
  return get(`/entries/${encodeURIComponent(entryId)}/history`);
}
```

### 3. Frontend: Add types

**File:** `tools/corvia-dashboard/src/types.ts`

```typescript
export interface EntryDetail {
  id: string;
  content: string;
  scope_id: string;
  created_at: string;
  superseded_by: string | null;
  metadata: {
    source_file: string | null;
    language: string | null;
    [key: string]: unknown;
  };
}

export interface HistoryEntry {
  id: string;
  content: string;
  created_at: string;
  superseded_by: string | null;
}

export interface HistoryResponse {
  entry_id: string;
  chain: HistoryEntry[];
  count: number;
}
```

### 4. Frontend: Create HistoryView.tsx

**File:** `tools/corvia-dashboard/src/components/HistoryView.tsx` (NEW FILE)

Build a temporal history explorer with:

- **Entry lookup**: Text input for entry ID (with paste support), submit button
- **Timeline**: Vertical timeline showing each version in the supersession chain
  - Each node shows: version number, created_at timestamp, content preview (first 100 chars)
  - Visual connector lines between versions
  - Current (latest) version highlighted
- **Diff view**: Click any two versions to see a simple text diff
  - Side-by-side or unified diff display
  - Highlight additions (green) and deletions (red)
  - Use a simple line-by-line diff algorithm (no external dep needed)
- **Time travel slider**: Range input at the bottom
  - Sliding shows the content at that point in the chain
  - Full content display for the selected version
- **Entry detail panel**: Shows full metadata for selected version

### 5. Frontend: Add History tab to Layout

**File:** `tools/corvia-dashboard/src/components/Layout.tsx`

- Add `"history"` to the `Tab` type union
- Add `{ id: "history", label: "History" }` to `TABS` array
- Import and render `HistoryView` when `activeTab === "history"`

### 6. Test the dashboard

Run: `cd tools/corvia-dashboard && npm run dev -- --port 8024`
Verify: Navigate to History tab, enter an entry ID, see timeline and diff view

## Files you will touch

| File | Change type |
|------|-------------|
| `repos/corvia/crates/corvia-server/src/dashboard/mod.rs` | Add routes + handlers |
| `tools/corvia-dashboard/src/api.ts` | Add `fetchEntryDetail()`, `fetchEntryHistory()` |
| `tools/corvia-dashboard/src/types.ts` | Add entry/history types |
| `tools/corvia-dashboard/src/components/HistoryView.tsx` | New file |
| `tools/corvia-dashboard/src/components/Layout.tsx` | Add History tab |

## Files you must NOT touch

These files are being modified by Agent A — do not touch them to avoid merge conflicts:

- `tools/corvia-dashboard/src/components/GraphView.tsx` (Agent A rewrites this)
- Any kernel files in `crates/corvia-kernel/src/traits.rs`
- Any files outside the ones listed above

## Coordination

- Use `corvia_write` to record any design decisions you make
- Use `corvia_search` to check if Agent A has recorded anything relevant
- The shared corvia server is at `http://localhost:8020`
- Your dev server should run on port **8024** (not 8021 which is the main dashboard)

## Commit convention

Prefix commits with `feat(dashboard):` and note `[history]` in the message.
