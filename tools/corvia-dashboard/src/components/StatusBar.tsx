import type { DashboardStatusResponse } from "../types";

interface MetricProps {
  color: string;
  icon: string;
  label: string;
  value: number;
}

function MetricCard({ color, icon, label, value }: MetricProps) {
  return (
    <div class={`metric-card ${color}`}>
      <div class={`metric-icon ${color}`}>{icon}</div>
      <div class="metric-label">{label}</div>
      <div class="metric-row">
        <span class="metric-value">{value.toLocaleString()}</span>
      </div>
    </div>
  );
}

export function StatusBar({ data }: { data: DashboardStatusResponse }) {
  return (
    <div class="metrics">
      <MetricCard color="gold" icon="📝" label="Entries" value={data.entry_count} />
      <MetricCard color="peach" icon="🤖" label="Agents" value={data.agent_count} />
      <MetricCard color="mint" icon="🔀" label="Merge Queue" value={data.merge_queue_depth} />
      <MetricCard color="lavender" icon="💬" label="Sessions" value={data.session_count} />
    </div>
  );
}
