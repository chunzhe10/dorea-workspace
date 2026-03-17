/** Mirrors corvia-common::dashboard Rust types */

export type ServiceState = "healthy" | "unhealthy" | "starting" | "stopped";

export interface ServiceStatus {
  name: string;
  state: ServiceState;
  port?: number;
  latency_ms?: number;
}

export interface SpanStats {
  count: number;
  count_1h: number;
  avg_ms: number;
  last_ms: number;
  errors: number;
  p50_ms?: number;
  p95_ms?: number;
  p99_ms?: number;
}

export interface TraceEvent {
  ts: string;
  level: string;
  module: string;
  msg: string;
}

export interface TracesData {
  spans: Record<string, SpanStats>;
  recent_events: TraceEvent[];
}

export interface DashboardConfig {
  embedding_provider: string;
  merge_provider: string;
  storage: string;
  workspace: string;
}

export interface DashboardStatusResponse {
  services: ServiceStatus[];
  entry_count: number;
  agent_count: number;
  merge_queue_depth: number;
  session_count: number;
  config: DashboardConfig;
  traces?: TracesData;
}

export interface LogEntry {
  timestamp: string;
  level: string;
  module: string;
  message: string;
}

export interface LogsResponse {
  entries: LogEntry[];
  total: number;
}

export interface ModuleStats {
  count: number;
  count_1h: number;
  avg_ms: number;
  errors: number;
  span_count: number;
}

export interface TracesResponse {
  spans: Record<string, SpanStats>;
  recent_events: TraceEvent[];
  modules: Record<string, ModuleStats>;
}

export interface GraphEdge {
  from_id: string;
  relation: string;
  to_id: string;
  weight: number;
}

// --- Graph scope types ---

export interface GraphNode {
  id: string;
  label: string;
  preview?: string;
  source_file?: string;
  language?: string;
  group?: string;
  content_role?: string;
  source_origin?: string;
}

export interface GraphScopeEdge {
  from: string;
  relation: string;
  to: string;
  weight?: number;
}

export interface GraphScopeResponse {
  nodes: GraphNode[];
  edges: GraphScopeEdge[];
  degraded?: boolean;
}

export interface ClusterGraphNode {
  id: string;
  label: string;
  level: number;
  entry_count?: number;
  preview?: string;
  source_file?: string;
  group?: string;
  content_role?: string;
  source_origin?: string;
}

// --- Agent types ---

export type AgentStatus = "Active" | "Suspended";
export type IdentityType = "Registered" | "McpClient";

export interface ActivitySummary {
  entry_count: number;
  topic_tags: string[];
  last_topics: string[];
  last_active: string;
  session_count: number;
  drifted: boolean;
}

export interface AgentRecord {
  agent_id: string;
  display_name: string;
  identity_type: IdentityType;
  registered_at: string;
  permissions: AgentPermission;
  last_seen: string;
  status: AgentStatus;
  description?: string;
  activity_summary?: ActivitySummary;
}

export type AgentPermission =
  | "ReadOnly"
  | { ReadWrite: { scopes: string[] } }
  | "Admin";

export type SessionState =
  | "Created"
  | "Active"
  | "Committing"
  | "Merging"
  | "Closed"
  | "Stale"
  | "Orphaned";

export interface SessionRecord {
  session_id: string;
  agent_id: string;
  created_at: string;
  last_heartbeat: string;
  state: SessionState;
  git_branch: string | null;
  staging_dir: string | null;
  entries_written: number;
  entries_merged: number;
}

// --- Merge queue types ---

export interface MergeQueueEntry {
  entry_id: string;
  agent_id: string;
  session_id: string;
  scope_id: string;
  enqueued_at: string;
  retry_count: number;
  last_error: string | null;
}

export interface MergeQueueStatus {
  depth: number;
  entries: MergeQueueEntry[];
}

// --- RAG types ---

export interface RetrievalMetrics {
  latency_ms: number;
  vector_results: number;
  graph_expanded: number;
  graph_reinforced: number;
  post_filter_count: number;
  retriever_name: string;
}

export interface AugmentationMetrics {
  latency_ms: number;
  token_estimate: number;
  token_budget: number;
  sources_included: number;
  sources_truncated: number;
  augmenter_name: string;
  skills_used: string[];
}

export interface GenerationMetrics {
  latency_ms: number;
  model: string;
  input_tokens: number;
  output_tokens: number;
}

export interface PipelineTrace {
  trace_id: string;
  retrieval: RetrievalMetrics;
  augmentation: AugmentationMetrics;
  generation: GenerationMetrics | null;
  total_latency_ms: number;
}

export interface RagSource {
  content: string;
  score: number;
  source_file: string | null;
  language: string | null;
  content_role: string | null;
  source_origin: string | null;
}

export interface RagResponse {
  answer: string | null;
  sources: RagSource[];
  trace: PipelineTrace;
}

// --- Entry / History types ---

export interface EntryDetail {
  id: string;
  content: string;
  scope_id: string;
  recorded_at: string;
  valid_from: string;
  valid_to: string | null;
  superseded_by: string | null;
  metadata: {
    source_file: string | null;
    language: string | null;
    content_role: string | null;
    source_origin: string | null;
  };
}

export interface HistoryEntry {
  id: string;
  content: string;
  recorded_at: string;
  valid_from: string;
  valid_to: string | null;
  superseded_by: string | null;
  is_current: boolean;
  metadata: {
    source_file: string | null;
    language: string | null;
    content_role: string | null;
    source_origin: string | null;
  };
}

export interface HistoryResponse {
  entry_id: string;
  chain: HistoryEntry[];
  count: number;
}

// --- Neighbor types ---

export interface NeighborEntry {
  id: string;
  content: string;
  relation: string;
  direction: "outgoing" | "incoming";
  score: number | null;
  source_file: string | null;
  content_role: string | null;
  source_origin: string | null;
}

export interface NeighborsResponse {
  neighbors: NeighborEntry[];
  count: number;
}

// --- Health types ---

export interface HealthFinding {
  check_type: string;
  confidence: number;
  rationale: string;
  target_ids: string[];
}

export interface HealthResponse {
  scope_id: string;
  findings: HealthFinding[];
  count: number;
}

// --- Activity feed types ---

export interface ActivityItem {
  entry_id: string;
  action: string;
  title: string;
  agent_id?: string;
  agent_name?: string;
  topic_tags: string[];
  delta_bytes?: number;
  recorded_at: string;
  superseded_id?: string;
  group_id?: string;
  group_count?: number;
}

export interface ActivityFeedResponse {
  items: ActivityItem[];
  total: number;
  topics: string[];
}

// --- GC types ---

export interface GcReportDto {
  orphans_rolled_back: number;
  duration_ms: number;
  stale_transitioned: number;
  closed_sessions_cleaned: number;
  agents_suspended: number;
  entries_deduplicated: number;
  started_at: string;
}

export interface GcStatusResponse {
  last_run: GcReportDto | null;
  history: GcReportDto[];
  scheduled: boolean;
}

// --- Live session types ---

export interface LiveSession {
  session_id: string;
  agent_id: string;
  agent_name: string;
  state: string;
  started_at: string;
  duration_secs: number;
  entries_written: number;
  entries_merged: number;
  pending_entries: number;
  git_branch: string | null;
  has_staging_dir: boolean;
}

export interface LiveSessionsSummary {
  total_active: number;
  total_stale: number;
  total_entries_pending: number;
}

export interface LiveSessionsResponse {
  sessions: LiveSession[];
  summary: LiveSessionsSummary;
}

// --- Trace tree types ---

export interface SpanNode {
  span_id: string;
  parent_span_id: string;
  trace_id: string;
  span_name: string;
  elapsed_ms: number;
  start_offset_ms: number;
  depth: number;
  module: string;
  fields: Record<string, unknown>;
  children: SpanNode[];
}

export interface TraceTree {
  trace_id: string;
  root_span: string;
  total_ms: number;
  span_count: number;
  started_at: string;
  spans: SpanNode[];
}

export interface RecentTracesResponse {
  traces: TraceTree[];
}
