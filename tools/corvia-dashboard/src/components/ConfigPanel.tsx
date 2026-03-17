import type { DashboardConfig } from "../types";

export function ConfigPanel({ config }: { config: DashboardConfig }) {
  const items = [
    { key: "Storage", val: config.storage },
    { key: "Embedding", val: config.embedding_provider },
    { key: "Merge", val: config.merge_provider },
    { key: "Workspace", val: config.workspace },
  ];

  return (
    <div>
      <h2 style={{
        fontSize: "12px",
        fontWeight: 600,
        color: "var(--text-dim)",
        textTransform: "uppercase",
        letterSpacing: "0.5px",
        marginBottom: "14px",
      }}>
        Configuration
      </h2>
      <ul class="config-list">
        {items.map((item) => (
          <li key={item.key}>
            <span class="key">{item.key}</span>
            <span class="val">{item.val}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
