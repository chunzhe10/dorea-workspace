import { useState } from "preact/hooks";

interface GraphControlsProps {
  onFilterChange: (filters: {
    contentRole: string | null;
    sourceOrigin: string | null;
    depth: number;
  }) => void;
  availableRoles: string[];
  availableOrigins: string[];
}

const ROLE_LABELS: Record<string, string> = {
  design: "Design",
  decision: "Decision",
  plan: "Plan",
  code: "Code",
  memory: "Memory",
  finding: "Finding",
  instruction: "Instruction",
  learning: "Learning",
};

export function GraphControls({
  onFilterChange,
  availableRoles,
  availableOrigins,
}: GraphControlsProps) {
  const [role, setRole] = useState<string | null>(null);
  const [origin, setOrigin] = useState<string | null>(null);
  const [depth, setDepth] = useState(2);

  const handleRoleChange = (e: Event) => {
    const val = (e.target as HTMLSelectElement).value || null;
    setRole(val);
    onFilterChange({ contentRole: val, sourceOrigin: origin, depth });
  };

  const handleOriginChange = (e: Event) => {
    const val = (e.target as HTMLSelectElement).value || null;
    setOrigin(val);
    onFilterChange({ contentRole: role, sourceOrigin: val, depth });
  };

  const handleDepthChange = (e: Event) => {
    const val = parseInt((e.target as HTMLInputElement).value, 10);
    setDepth(val);
    onFilterChange({ contentRole: role, sourceOrigin: origin, depth: val });
  };

  return (
    <div class="graph-controls">
      <select value={role ?? ""} onChange={handleRoleChange}>
        <option value="">All roles</option>
        {availableRoles.map((r) => (
          <option key={r} value={r}>{ROLE_LABELS[r] ?? r}</option>
        ))}
      </select>

      <select value={origin ?? ""} onChange={handleOriginChange}>
        <option value="">All sources</option>
        {availableOrigins.map((o) => (
          <option key={o} value={o}>{o}</option>
        ))}
      </select>

      <label class="depth-control">
        <span>Depth: {depth}</span>
        <input
          type="range"
          min={1}
          max={3}
          value={depth}
          onInput={handleDepthChange}
        />
      </label>
    </div>
  );
}
