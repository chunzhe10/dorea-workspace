import type { NeighborEntry } from "../types";

// Role -> color mapping (matches graph node colors from spec)
const ROLE_COLORS: Record<string, string> = {
  design: "#7dd3fc",    // blue
  code: "#5eead4",      // green
  plan: "#ffb07c",      // orange
  memory: "#c4b5fd",    // purple
  finding: "#ff8a80",   // red
  decision: "#f0c94c",  // gold
  instruction: "#fcd34d", // amber
  learning: "#6ee7b7",  // emerald
};

interface NeighborCardsProps {
  neighbors: NeighborEntry[];
  onSelect: (entryId: string) => void;
}

export function NeighborCards({ neighbors, onSelect }: NeighborCardsProps) {
  if (neighbors.length === 0) {
    return <div class="neighbor-empty">No connected entries</div>;
  }

  return (
    <div class="neighbor-cards">
      <h4>Connected Knowledge</h4>
      {neighbors.map((n) => (
        <button
          key={n.id}
          class="neighbor-card"
          onClick={() => onSelect(n.id)}
          title={`${n.relation} (${n.direction})`}
        >
          <span
            class="role-dot"
            style={{ backgroundColor: ROLE_COLORS[n.content_role ?? ""] ?? "#8a8279" }}
          />
          <span class="neighbor-label">
            {n.source_file ?? n.content.slice(0, 60)}
          </span>
          <span class="neighbor-role">{n.content_role ?? "unknown"}</span>
        </button>
      ))}
    </div>
  );
}
