import { useCallback, useEffect, useRef, useState } from "preact/hooks";
import { usePoll } from "../hooks/use-poll";
import { fetchGraphScope, fetchClusteredGraph } from "../api";
import { DocumentReader } from "./DocumentReader";
import { GraphControls } from "./GraphControls";
import type { GraphNode, GraphScopeEdge, GraphScopeResponse } from "../types";
import "../styles/graph-reader.css";

// --- Breadcrumb type ---

interface Breadcrumb {
  level: number;
  id?: string;
  label: string;
}

// --- Cluster types ---

interface ClusterNode {
  id: string; // group name
  label: string;
  count: number;
  files: GraphNode[];
  x: number;
  y: number;
  vx: number;
  vy: number;
  pinned: boolean;
}

interface ClusterEdge {
  from: string;
  to: string;
  count: number;
  relations: Set<string>;
}

// --- Entry-level node type ---

interface EntryNode {
  id: string;
  label: string;
  content_role?: string;
  source_origin?: string;
  source_file?: string;
  x: number;
  y: number;
  vx: number;
  vy: number;
  pinned: boolean;
}

// --- LOD level names ---

const LOD_LABELS = ["Super-clusters", "Sub-clusters", "File groups", "Entries"];

// --- Zoom-to-LOD level with hysteresis ---

function zoomToLevel(zoom: number, lastZoom: number): number {
  const HYSTERESIS = 0.05;
  if (zoom < 0.8 - (lastZoom >= 0.8 ? HYSTERESIS : 0)) return 0;
  if (zoom < 1.5 - (lastZoom >= 1.5 ? HYSTERESIS : 0)) return 1;
  if (zoom < 3.0 - (lastZoom >= 3.0 ? HYSTERESIS : 0)) return 2;
  return 3;
}

// --- Viewport culling for L3 ---

function cullToViewport(
  nodes: EntryNode[],
  pan: { x: number; y: number },
  zoom: number,
  canvasW: number,
  canvasH: number,
): EntryNode[] {
  const margin = 0.2;
  const left = -pan.x / zoom - (canvasW * margin) / zoom;
  const right = (canvasW - pan.x) / zoom + (canvasW * margin) / zoom;
  const top = -pan.y / zoom - (canvasH * margin) / zoom;
  const bottom = (canvasH - pan.y) / zoom + (canvasH * margin) / zoom;
  return nodes.filter(
    (n) => n.x >= left && n.x <= right && n.y >= top && n.y <= bottom,
  );
}

// --- Relation color palette (matches theme) ---

const RELATION_COLORS: Record<string, string> = {
  imports: "#7dd3fc",       // sky
  depends_on: "#c4b5fd",   // lavender
  implements: "#5eead4",   // mint
  supersedes: "#f0c94c",   // gold
  relates_to: "#ffb07c",   // peach
  contradicts: "#ff8a80",  // coral
  tests: "#fcd34d",        // amber
  documents: "#7dd3fc",    // sky
  uses: "#c4b5fd",         // lavender
};

function relationColor(rel: string): string {
  return RELATION_COLORS[rel] ?? "#8a8279";
}

// --- Cluster color palette ---

const CLUSTER_COLORS = [
  "#5eead4", // mint
  "#f0c94c", // gold
  "#7dd3fc", // sky
  "#c4b5fd", // lavender
  "#ffb07c", // peach
  "#ff8a80", // coral
  "#fcd34d", // amber
  "#a78bfa", // violet
  "#6ee7b7", // emerald
  "#f9a8d4", // pink
  "#93c5fd", // blue
  "#fdba74", // orange
  "#86efac", // green
  "#d8b4fe", // purple
  "#fca5a5", // red-light
  "#67e8f9", // cyan
  "#bef264", // lime
];

function clusterColor(index: number): string {
  return CLUSTER_COLORS[index % CLUSTER_COLORS.length];
}

// --- Entry-level node colors by content_role ---

const ROLE_NODE_COLORS: Record<string, string> = {
  design: "#7dd3fc",
  code: "#5eead4",
  plan: "#ffb07c",
  memory: "#c4b5fd",
  finding: "#ff8a80",
  decision: "#f0c94c",
  instruction: "#fcd34d",
  learning: "#6ee7b7",
};

// --- Entry-level node shape by source_origin ---

function drawNodeShape(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  r: number,
  origin: string | undefined,
) {
  ctx.beginPath();
  if (origin?.startsWith("repo:")) {
    // Circle for repo entries
    ctx.arc(x, y, r, 0, Math.PI * 2);
  } else if (origin === "workspace") {
    // Diamond for workspace entries
    ctx.moveTo(x, y - r);
    ctx.lineTo(x + r, y);
    ctx.lineTo(x, y + r);
    ctx.lineTo(x - r, y);
    ctx.closePath();
  } else if (origin === "memory") {
    // Triangle for memory entries
    ctx.moveTo(x, y - r);
    ctx.lineTo(x + r, y + r);
    ctx.lineTo(x - r, y + r);
    ctx.closePath();
  } else {
    ctx.arc(x, y, r, 0, Math.PI * 2); // default circle
  }
}

// --- Build clusters from nodes/edges ---

function buildClusters(
  nodes: GraphNode[],
  edges: GraphScopeEdge[],
): { clusters: ClusterNode[]; clusterEdges: ClusterEdge[] } {
  // Group nodes by their group field
  const groups = new Map<string, GraphNode[]>();
  for (const n of nodes) {
    const g = n.group ?? "other";
    if (!groups.has(g)) groups.set(g, []);
    groups.get(g)!.push(n);
  }

  // Build node-to-group lookup
  const nodeGroup = new Map<string, string>();
  for (const n of nodes) {
    nodeGroup.set(n.id, n.group ?? "other");
  }

  // Create cluster nodes
  const clusterList = [...groups.entries()].sort((a, b) => b[1].length - a[1].length);
  const clusters: ClusterNode[] = clusterList.map(([name, files], _i) => ({
    id: name,
    label: name,
    count: files.length,
    files,
    x: 0,
    y: 0,
    vx: 0,
    vy: 0,
    pinned: false,
  }));

  // Aggregate edges between clusters
  const edgeKey = (a: string, b: string) => (a < b ? `${a}::${b}` : `${b}::${a}`);
  const edgeMap = new Map<string, ClusterEdge>();

  for (const e of edges) {
    const fromG = nodeGroup.get(e.from) ?? "other";
    const toG = nodeGroup.get(e.to) ?? "other";
    if (fromG === toG) continue; // skip intra-cluster edges
    const key = edgeKey(fromG, toG);
    if (!edgeMap.has(key)) {
      edgeMap.set(key, { from: fromG, to: toG, count: 0, relations: new Set() });
    }
    const ce = edgeMap.get(key)!;
    ce.count++;
    ce.relations.add(e.relation);
  }

  return { clusters, clusterEdges: [...edgeMap.values()] };
}

// --- Force simulation for clusters ---

const SPRING_LENGTH = 200;
const SPRING_K = 0.003;
const REPULSION = 50000;
const DAMPING = 0.82;
const CENTER_PULL = 0.001;
const MIN_VELOCITY = 0.05;

function initClusters(clusters: ClusterNode[], w: number, h: number) {
  const cx = w / 2;
  const cy = h / 2;
  const radius = Math.min(w, h) * 0.35;
  for (let i = 0; i < clusters.length; i++) {
    const angle = (2 * Math.PI * i) / clusters.length;
    clusters[i].x = cx + radius * Math.cos(angle) + (Math.random() - 0.5) * 30;
    clusters[i].y = cy + radius * Math.sin(angle) + (Math.random() - 0.5) * 30;
    clusters[i].vx = 0;
    clusters[i].vy = 0;
  }
}

function stepClusters(clusters: ClusterNode[], edges: ClusterEdge[], cx: number, cy: number): boolean {
  const cMap = new Map<string, ClusterNode>();
  for (const c of clusters) cMap.set(c.id, c);

  for (const n of clusters) {
    if (n.pinned) continue;
    let fx = 0;
    let fy = 0;

    // Repulsion
    for (const m of clusters) {
      if (m.id === n.id) continue;
      const dx = n.x - m.x;
      const dy = n.y - m.y;
      const distSq = dx * dx + dy * dy + 1;
      // Scale repulsion by node size
      const sizeFactor = Math.sqrt(n.count) * Math.sqrt(m.count);
      const f = (REPULSION * sizeFactor) / distSq;
      const dist = Math.sqrt(distSq);
      fx += (f * dx) / dist;
      fy += (f * dy) / dist;
    }

    // Spring attraction along edges
    for (const e of edges) {
      let other: ClusterNode | undefined;
      if (e.from === n.id) other = cMap.get(e.to);
      else if (e.to === n.id) other = cMap.get(e.from);
      if (!other) continue;
      const dx = other.x - n.x;
      const dy = other.y - n.y;
      const dist = Math.sqrt(dx * dx + dy * dy) + 0.1;
      const displacement = dist - SPRING_LENGTH;
      // Stronger spring for more edges
      const k = SPRING_K * Math.log2(e.count + 1);
      fx += k * displacement * (dx / dist);
      fy += k * displacement * (dy / dist);
    }

    // Gentle center pull
    fx += (cx - n.x) * CENTER_PULL;
    fy += (cy - n.y) * CENTER_PULL;

    n.vx = (n.vx + fx) * DAMPING;
    n.vy = (n.vy + fy) * DAMPING;
    n.x += n.vx;
    n.y += n.vy;
  }

  let maxV = 0;
  for (const n of clusters) {
    const v = Math.abs(n.vx) + Math.abs(n.vy);
    if (v > maxV) maxV = v;
  }
  return maxV > MIN_VELOCITY;
}

// --- Entry-level force simulation ---

const ENTRY_SPRING_LENGTH = 80;
const ENTRY_SPRING_K = 0.005;
const ENTRY_REPULSION = 3000;

function initEntryNodes(nodes: EntryNode[], w: number, h: number) {
  const cx = w / 2;
  const cy = h / 2;
  const radius = Math.min(w, h) * 0.3;
  for (let i = 0; i < nodes.length; i++) {
    const angle = (2 * Math.PI * i) / nodes.length;
    nodes[i].x = cx + radius * Math.cos(angle) + (Math.random() - 0.5) * 20;
    nodes[i].y = cy + radius * Math.sin(angle) + (Math.random() - 0.5) * 20;
    nodes[i].vx = 0;
    nodes[i].vy = 0;
  }
}

function stepEntryNodes(nodes: EntryNode[], edges: GraphScopeEdge[], cx: number, cy: number): boolean {
  const nMap = new Map<string, EntryNode>();
  for (const n of nodes) nMap.set(n.id, n);

  for (const n of nodes) {
    if (n.pinned) continue;
    let fx = 0;
    let fy = 0;

    // Repulsion
    for (const m of nodes) {
      if (m.id === n.id) continue;
      const dx = n.x - m.x;
      const dy = n.y - m.y;
      const distSq = dx * dx + dy * dy + 1;
      const f = ENTRY_REPULSION / distSq;
      const dist = Math.sqrt(distSq);
      fx += (f * dx) / dist;
      fy += (f * dy) / dist;
    }

    // Spring attraction along edges
    for (const e of edges) {
      let other: EntryNode | undefined;
      if (e.from === n.id) other = nMap.get(e.to);
      else if (e.to === n.id) other = nMap.get(e.from);
      if (!other) continue;
      const dx = other.x - n.x;
      const dy = other.y - n.y;
      const dist = Math.sqrt(dx * dx + dy * dy) + 0.1;
      const displacement = dist - ENTRY_SPRING_LENGTH;
      fx += ENTRY_SPRING_K * displacement * (dx / dist);
      fy += ENTRY_SPRING_K * displacement * (dy / dist);
    }

    // Center pull
    fx += (cx - n.x) * CENTER_PULL;
    fy += (cy - n.y) * CENTER_PULL;

    n.vx = (n.vx + fx) * DAMPING;
    n.vy = (n.vy + fy) * DAMPING;
    n.x += n.vx;
    n.y += n.vy;
  }

  let maxV = 0;
  for (const n of nodes) {
    const v = Math.abs(n.vx) + Math.abs(n.vy);
    if (v > maxV) maxV = v;
  }
  return maxV > MIN_VELOCITY;
}

// --- Canvas rendering ---

function nodeRadius(count: number): number {
  return 16 + Math.sqrt(count) * 2.5;
}

function drawClusters(
  ctx: CanvasRenderingContext2D,
  clusters: ClusterNode[],
  edges: ClusterEdge[],
  selectedId: string | null,
  hoveredId: string | null,
  zoom: number,
  panX: number,
  panY: number,
  w: number,
  h: number,
  colorMap: Map<string, string>,
) {
  ctx.save();
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#1e2230";
  ctx.fillRect(0, 0, w, h);

  ctx.translate(panX, panY);
  ctx.scale(zoom, zoom);

  const cMap = new Map<string, ClusterNode>();
  for (const c of clusters) cMap.set(c.id, c);

  // Build connected set for hover highlight
  const connectedSet = new Set<string>();
  if (hoveredId) {
    connectedSet.add(hoveredId);
    for (const e of edges) {
      if (e.from === hoveredId) connectedSet.add(e.to);
      if (e.to === hoveredId) connectedSet.add(e.from);
    }
  }

  // Draw edges
  for (const e of edges) {
    const from = cMap.get(e.from);
    const to = cMap.get(e.to);
    if (!from || !to) continue;

    const thickness = Math.min(1 + Math.log2(e.count + 1) * 1.5, 8);
    let alpha = Math.min(0.15 + e.count * 0.02, 0.6);

    // Hover fade: dim edges not connected to hovered node
    if (hoveredId && !connectedSet.has(e.from) && !connectedSet.has(e.to)) {
      alpha *= 0.15;
    }

    ctx.beginPath();
    ctx.moveTo(from.x, from.y);
    ctx.lineTo(to.x, to.y);
    ctx.strokeStyle = "#504b44";
    ctx.globalAlpha = alpha;
    ctx.lineWidth = thickness / zoom;
    ctx.stroke();
    ctx.globalAlpha = 1;

    // Edge count label
    if (e.count > 1) {
      const mx = (from.x + to.x) / 2;
      const my = (from.y + to.y) / 2;
      ctx.font = `bold ${10 / zoom}px 'Cascadia Code', monospace`;
      ctx.fillStyle = "#8a8279";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.globalAlpha = hoveredId && !connectedSet.has(e.from) && !connectedSet.has(e.to) ? 0.15 : 1;
      ctx.fillText(`${e.count}`, mx, my);
      ctx.globalAlpha = 1;
    }
  }

  // Draw nodes
  for (const c of clusters) {
    const r = nodeRadius(c.count) / zoom;
    const isSelected = c.id === selectedId;
    const color = colorMap.get(c.id) ?? "#8a8279";

    // Hover fade: dim unconnected nodes
    const isConnected = hoveredId ? connectedSet.has(c.id) : true;
    ctx.globalAlpha = isConnected ? 1.0 : 0.15;

    // Glow for selected
    if (isSelected) {
      ctx.beginPath();
      ctx.arc(c.x, c.y, r + 4 / zoom, 0, 2 * Math.PI);
      ctx.fillStyle = color;
      const prevAlpha = ctx.globalAlpha;
      ctx.globalAlpha = prevAlpha * 0.15;
      ctx.fill();
      ctx.globalAlpha = prevAlpha;
    }

    // Circle
    ctx.beginPath();
    ctx.arc(c.x, c.y, r, 0, 2 * Math.PI);
    ctx.fillStyle = isSelected ? "#252a3a" : "#1e2230";
    ctx.fill();
    ctx.strokeStyle = color;
    ctx.lineWidth = (isSelected ? 3 : 2) / zoom;
    ctx.stroke();

    // Count inside circle
    ctx.font = `bold ${14 / zoom}px 'Cascadia Code', monospace`;
    ctx.fillStyle = color;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(`${c.count}`, c.x, c.y);

    // Label below
    const displayLabel = c.label.replace("corvia-", "");
    ctx.font = `bold ${12 / zoom}px Inter, sans-serif`;
    ctx.fillStyle = isSelected ? "#ffffff" : "#e0ddd8";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText(displayLabel, c.x, c.y + r + 6 / zoom);

    ctx.globalAlpha = 1;
  }

  ctx.restore();
}

// --- Entry-level rendering ---

const ENTRY_NODE_RADIUS = 10;

function drawEntryNodes(
  ctx: CanvasRenderingContext2D,
  nodes: EntryNode[],
  edges: GraphScopeEdge[],
  selectedId: string | null,
  hoveredId: string | null,
  zoom: number,
  panX: number,
  panY: number,
  w: number,
  h: number,
) {
  ctx.save();
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#1e2230";
  ctx.fillRect(0, 0, w, h);

  ctx.translate(panX, panY);
  ctx.scale(zoom, zoom);

  const nMap = new Map<string, EntryNode>();
  for (const n of nodes) nMap.set(n.id, n);

  // Build connected set for hover highlight
  const connectedSet = new Set<string>();
  if (hoveredId) {
    connectedSet.add(hoveredId);
    for (const e of edges) {
      if (e.from === hoveredId) connectedSet.add(e.to);
      if (e.to === hoveredId) connectedSet.add(e.from);
    }
  }

  // Draw edges
  for (const e of edges) {
    const from = nMap.get(e.from);
    const to = nMap.get(e.to);
    if (!from || !to) continue;

    const isConnected = hoveredId
      ? connectedSet.has(e.from) && connectedSet.has(e.to)
      : true;

    ctx.beginPath();
    ctx.moveTo(from.x, from.y);
    ctx.lineTo(to.x, to.y);
    ctx.strokeStyle = relationColor(e.relation);
    ctx.globalAlpha = isConnected ? 0.4 : 0.06;
    ctx.lineWidth = 1.5 / zoom;
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  // Draw nodes
  for (const n of nodes) {
    const r = ENTRY_NODE_RADIUS / zoom;
    const isSelected = n.id === selectedId;
    const color = ROLE_NODE_COLORS[n.content_role ?? ""] ?? "#8a8279";

    const isConnected = hoveredId ? connectedSet.has(n.id) : true;
    ctx.globalAlpha = isConnected ? 1.0 : 0.15;

    // Glow for selected
    if (isSelected) {
      ctx.beginPath();
      ctx.arc(n.x, n.y, r + 3 / zoom, 0, 2 * Math.PI);
      ctx.fillStyle = color;
      const prevAlpha = ctx.globalAlpha;
      ctx.globalAlpha = prevAlpha * 0.2;
      ctx.fill();
      ctx.globalAlpha = prevAlpha;
    }

    // Shape by origin
    drawNodeShape(ctx, n.x, n.y, r, n.source_origin);
    ctx.fillStyle = isSelected ? "#252a3a" : "#1e2230";
    ctx.fill();
    ctx.strokeStyle = color;
    ctx.lineWidth = (isSelected ? 2.5 : 1.5) / zoom;
    ctx.stroke();

    // Label below
    const displayLabel = n.source_file
      ? n.source_file.split("/").pop() ?? n.label
      : n.label.length > 25 ? n.label.slice(0, 23) + "\u2026" : n.label;
    ctx.font = `${10 / zoom}px Inter, sans-serif`;
    ctx.fillStyle = isSelected ? "#ffffff" : "#b0a99f";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText(displayLabel, n.x, n.y + r + 3 / zoom);

    ctx.globalAlpha = 1;
  }

  ctx.restore();
}

// --- Hit test ---

function hitTestCluster(
  clusters: ClusterNode[],
  canvasX: number,
  canvasY: number,
  zoom: number,
  panX: number,
  panY: number,
): ClusterNode | null {
  const worldX = (canvasX - panX) / zoom;
  const worldY = (canvasY - panY) / zoom;
  for (let i = clusters.length - 1; i >= 0; i--) {
    const c = clusters[i];
    const r = nodeRadius(c.count) / zoom;
    const dx = worldX - c.x;
    const dy = worldY - c.y;
    if (dx * dx + dy * dy <= r * r * 1.3) return c;
  }
  return null;
}

function hitTestEntry(
  nodes: EntryNode[],
  canvasX: number,
  canvasY: number,
  zoom: number,
  panX: number,
  panY: number,
): EntryNode | null {
  const worldX = (canvasX - panX) / zoom;
  const worldY = (canvasY - panY) / zoom;
  const r = ENTRY_NODE_RADIUS / zoom;
  for (let i = nodes.length - 1; i >= 0; i--) {
    const n = nodes[i];
    const dx = worldX - n.x;
    const dy = worldY - n.y;
    if (dx * dx + dy * dy <= r * r * 1.5) return n;
  }
  return null;
}

// --- Component ---

type ViewMode = "cluster" | "entry";

export function GraphView({ navigateToHistory }: { navigateToHistory?: (entryId: string) => void }) {
  const [filters, setFilters] = useState<{
    contentRole: string | null;
    sourceOrigin: string | null;
    depth: number;
  }>({ contentRole: null, sourceOrigin: null, depth: 2 });

  // --- LOD state ---
  const [lodLevel, setLodLevel] = useState(0);
  const [breadcrumbs, setBreadcrumbs] = useState<Breadcrumb[]>([
    { level: 0, label: "All" },
  ]);
  const lastZoomRef = useRef(1.0);
  const [isDegraded, setIsDegraded] = useState(false);

  // Derive parent ID from breadcrumbs (the last breadcrumb with an id)
  const parentId = breadcrumbs.length > 1
    ? breadcrumbs[breadcrumbs.length - 1].id
    : undefined;

  // Level-aware fetcher: try clustered endpoint first, fall back on degraded
  const fetcher = useCallback(() => {
    if (isDegraded) {
      // Degraded mode: use the original path-based fetch
      return fetchGraphScope(
        filters.contentRole || filters.sourceOrigin
          ? {
              content_role: filters.contentRole ?? undefined,
              source_origin: filters.sourceOrigin ?? undefined,
            }
          : undefined,
      );
    }
    return fetchClusteredGraph(lodLevel, parentId).then((resp) => {
      if (resp.degraded) {
        // Mark degraded and fall back to path-based fetch
        setIsDegraded(true);
        return fetchGraphScope(
          filters.contentRole || filters.sourceOrigin
            ? {
                content_role: filters.contentRole ?? undefined,
                source_origin: filters.sourceOrigin ?? undefined,
              }
            : undefined,
        );
      }
      return resp;
    });
  }, [lodLevel, parentId, isDegraded, filters.contentRole, filters.sourceOrigin]);

  const { data, error, loading } = usePoll(fetcher, 10000);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const clustersRef = useRef<ClusterNode[]>([]);
  const clusterEdgesRef = useRef<ClusterEdge[]>([]);
  const entryNodesRef = useRef<EntryNode[]>([]);
  const entryEdgesRef = useRef<GraphScopeEdge[]>([]);
  const allEntryNodesRef = useRef<EntryNode[]>([]);
  const animRef = useRef<number>(0);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [selectedEntryId, setSelectedEntryId] = useState<string | null>(null);
  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>("cluster");
  const [focusCluster, setFocusCluster] = useState<string | null>(null);
  const zoomRef = useRef(1);
  const panRef = useRef({ x: 0, y: 0 });
  const dragRef = useRef<{ node: ClusterNode | EntryNode; offsetX: number; offsetY: number } | null>(null);
  const panDragRef = useRef<{ startX: number; startY: number; panX: number; panY: number } | null>(null);
  const sizeRef = useRef({ w: 800, h: 600 });
  const dataIdRef = useRef("");
  const colorMapRef = useRef(new Map<string, string>());
  const [, forceRender] = useState(0);

  // Build clusters when data changes
  useEffect(() => {
    if (!data) return;
    const resp = data as GraphScopeResponse;
    if (!resp.nodes || !resp.edges) return;

    const dataId = resp.nodes.length + ":" + resp.edges.length + ":" + lodLevel;
    if (dataId === dataIdRef.current) return;
    dataIdRef.current = dataId;

    const { clusters, clusterEdges } = buildClusters(resp.nodes, resp.edges);

    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.parentElement?.getBoundingClientRect();
    const w = rect?.width ?? 800;
    const h = rect?.height ?? 600;
    sizeRef.current = { w, h };
    canvas.width = w * devicePixelRatio;
    canvas.height = h * devicePixelRatio;
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;

    // For L3 (individual entries), switch to entry mode
    if (lodLevel === 3 && !isDegraded) {
      const entryNodes: EntryNode[] = resp.nodes.map((n) => ({
        id: n.id,
        label: n.label,
        content_role: n.content_role,
        source_origin: n.source_origin,
        source_file: n.source_file,
        x: 0,
        y: 0,
        vx: 0,
        vy: 0,
        pinned: false,
      }));
      initEntryNodes(entryNodes, w, h);
      allEntryNodesRef.current = entryNodes;
      entryNodesRef.current = entryNodes;
      entryEdgesRef.current = resp.edges;
      setViewMode("entry");
    } else {
      initClusters(clusters, w, h);
      clustersRef.current = clusters;
      clusterEdgesRef.current = clusterEdges;

      // Assign colors
      const cm = new Map<string, string>();
      clusters.forEach((c, i) => cm.set(c.id, clusterColor(i)));
      colorMapRef.current = cm;

      setViewMode("cluster");
    }

    zoomRef.current = 1;
    panRef.current = { x: 0, y: 0 };
    setSelectedId(null);
  }, [data, lodLevel, isDegraded]);

  // Navigate to a specific breadcrumb level
  const navigateToLevel = useCallback((level: number, id?: string) => {
    // Trim breadcrumbs to the target level
    setBreadcrumbs((prev) => {
      const idx = prev.findIndex((bc) => bc.level === level && bc.id === id);
      if (idx >= 0) return prev.slice(0, idx + 1);
      return [{ level: 0, label: "All" }];
    });
    setLodLevel(level);
    setSelectedId(null);
    setSelectedEntryId(null);
    setHoveredNodeId(null);
    setFocusCluster(null);
    zoomRef.current = 1;
    panRef.current = { x: 0, y: 0 };
  }, []);

  // Drill into next level (double-click on cluster)
  const drillIntoCluster = useCallback(
    (clusterId: string, clusterLabel: string) => {
      if (isDegraded) {
        // In degraded mode, use the existing entry-mode expansion
        switchToEntryMode(clusterId);
        return;
      }
      const nextLevel = Math.min(lodLevel + 1, 3);
      setBreadcrumbs((prev) => [
        ...prev,
        { level: nextLevel, id: clusterId, label: clusterLabel.replace("corvia-", "") },
      ]);
      setLodLevel(nextLevel);
      setSelectedId(null);
      setSelectedEntryId(null);
      setHoveredNodeId(null);
      setFocusCluster(null);
      zoomRef.current = 1;
      panRef.current = { x: 0, y: 0 };
    },
    [lodLevel, isDegraded],
  );

  // Switch to entry-level view when a cluster is double-clicked (degraded mode)
  const switchToEntryMode = useCallback((clusterId: string) => {
    if (!data) return;
    const resp = data as GraphScopeResponse;
    const clusterNodes = resp.nodes.filter((n) => (n.group ?? "other") === clusterId);
    const clusterNodeIds = new Set(clusterNodes.map((n) => n.id));

    // Filter edges to only those within this cluster
    const clusterEntryEdges = resp.edges.filter(
      (e) => clusterNodeIds.has(e.from) && clusterNodeIds.has(e.to),
    );

    // Create entry nodes
    const entryNodes: EntryNode[] = clusterNodes.map((n) => ({
      id: n.id,
      label: n.label,
      content_role: n.content_role,
      source_origin: n.source_origin,
      source_file: n.source_file,
      x: 0,
      y: 0,
      vx: 0,
      vy: 0,
      pinned: false,
    }));

    const { w, h } = sizeRef.current;
    initEntryNodes(entryNodes, w, h);
    allEntryNodesRef.current = entryNodes;
    entryNodesRef.current = entryNodes;
    entryEdgesRef.current = clusterEntryEdges;

    zoomRef.current = 1;
    panRef.current = { x: 0, y: 0 };
    setViewMode("entry");
    setFocusCluster(clusterId);
    setSelectedId(null);
    setHoveredNodeId(null);
  }, [data]);

  const switchToClusterMode = useCallback(() => {
    setViewMode("cluster");
    setFocusCluster(null);
    setSelectedEntryId(null);
    setHoveredNodeId(null);
    zoomRef.current = 1;
    panRef.current = { x: 0, y: 0 };
    // In LOD mode, navigate back to parent level
    if (!isDegraded && breadcrumbs.length > 1) {
      const parent = breadcrumbs[breadcrumbs.length - 2];
      navigateToLevel(parent.level, parent.id);
    }
  }, [isDegraded, breadcrumbs, navigateToLevel]);

  // Viewport culling for L3 entry-level nodes
  useEffect(() => {
    if (viewMode !== "entry" || lodLevel !== 3 || isDegraded) return;
    const all = allEntryNodesRef.current;
    if (all.length === 0) return;
    const { w, h } = sizeRef.current;
    const culled = cullToViewport(all, panRef.current, zoomRef.current, w, h);
    entryNodesRef.current = culled;
  }, [viewMode, lodLevel, isDegraded]);

  // Animation loop
  useEffect(() => {
    let running = true;
    let settled = 0;

    function tick() {
      if (!running) return;
      const canvas = canvasRef.current;
      if (!canvas) { animRef.current = requestAnimationFrame(tick); return; }
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      const { w, h } = sizeRef.current;

      if (viewMode === "cluster") {
        const clusters = clustersRef.current;
        const edges = clusterEdgesRef.current;

        if (clusters.length > 0) {
          const moving = stepClusters(clusters, edges, w / 2, h / 2);
          if (moving) settled = 0; else settled++;
        }

        const dpr = devicePixelRatio;
        ctx.save();
        ctx.scale(dpr, dpr);
        drawClusters(
          ctx, clusters, edges, selectedId, hoveredNodeId,
          zoomRef.current, panRef.current.x, panRef.current.y,
          w, h, colorMapRef.current,
        );
        ctx.restore();
      } else {
        // At L3 with LOD, apply viewport culling each frame
        let renderNodes = entryNodesRef.current;
        if (lodLevel === 3 && !isDegraded) {
          renderNodes = cullToViewport(
            allEntryNodesRef.current,
            panRef.current,
            zoomRef.current,
            w,
            h,
          );
          entryNodesRef.current = renderNodes;
        }
        const edges = entryEdgesRef.current;

        if (renderNodes.length > 0) {
          const moving = stepEntryNodes(renderNodes, edges, w / 2, h / 2);
          if (moving) settled = 0; else settled++;
        }

        const dpr = devicePixelRatio;
        ctx.save();
        ctx.scale(dpr, dpr);
        drawEntryNodes(
          ctx, renderNodes, edges, selectedEntryId, hoveredNodeId,
          zoomRef.current, panRef.current.x, panRef.current.y,
          w, h,
        );
        ctx.restore();
      }

      if (settled < 200 || dragRef.current) {
        animRef.current = requestAnimationFrame(tick);
      } else {
        animRef.current = window.setTimeout(() => {
          if (running) animRef.current = requestAnimationFrame(tick);
        }, 200) as unknown as number;
      }
    }

    animRef.current = requestAnimationFrame(tick);
    return () => { running = false; cancelAnimationFrame(animRef.current); };
  }, [selectedId, selectedEntryId, hoveredNodeId, viewMode, lodLevel, isDegraded]);

  // Mouse handlers
  const onMouseDown = useCallback((e: MouseEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;

    if (viewMode === "cluster") {
      const hit = hitTestCluster(clustersRef.current, cx, cy, zoomRef.current, panRef.current.x, panRef.current.y);
      if (hit) {
        hit.pinned = true;
        const wx = (cx - panRef.current.x) / zoomRef.current;
        const wy = (cy - panRef.current.y) / zoomRef.current;
        dragRef.current = { node: hit, offsetX: hit.x - wx, offsetY: hit.y - wy };
      } else {
        panDragRef.current = { startX: e.clientX, startY: e.clientY, panX: panRef.current.x, panY: panRef.current.y };
      }
    } else {
      const hit = hitTestEntry(entryNodesRef.current, cx, cy, zoomRef.current, panRef.current.x, panRef.current.y);
      if (hit) {
        hit.pinned = true;
        const wx = (cx - panRef.current.x) / zoomRef.current;
        const wy = (cy - panRef.current.y) / zoomRef.current;
        dragRef.current = { node: hit, offsetX: hit.x - wx, offsetY: hit.y - wy };
      } else {
        panDragRef.current = { startX: e.clientX, startY: e.clientY, panX: panRef.current.x, panY: panRef.current.y };
      }
    }
  }, [viewMode]);

  const onMouseMove = useCallback((e: MouseEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;

    if (dragRef.current) {
      const { node, offsetX, offsetY } = dragRef.current;
      const wx = (cx - panRef.current.x) / zoomRef.current;
      const wy = (cy - panRef.current.y) / zoomRef.current;
      node.x = wx + offsetX;
      node.y = wy + offsetY;
      node.vx = 0;
      node.vy = 0;
    } else if (panDragRef.current) {
      panRef.current = {
        x: panDragRef.current.panX + (e.clientX - panDragRef.current.startX),
        y: panDragRef.current.panY + (e.clientY - panDragRef.current.startY),
      };
    } else {
      // Hover detection
      if (viewMode === "cluster") {
        const hit = hitTestCluster(clustersRef.current, cx, cy, zoomRef.current, panRef.current.x, panRef.current.y);
        setHoveredNodeId(hit?.id ?? null);
      } else {
        const hit = hitTestEntry(entryNodesRef.current, cx, cy, zoomRef.current, panRef.current.x, panRef.current.y);
        setHoveredNodeId(hit?.id ?? null);
      }
    }
  }, [viewMode]);

  const onMouseUp = useCallback((e: MouseEvent) => {
    if (dragRef.current) {
      const node = dragRef.current.node;
      node.pinned = false;
      const canvas = canvasRef.current;
      if (canvas) {
        const rect = canvas.getBoundingClientRect();
        const cx = e.clientX - rect.left;
        const cy = e.clientY - rect.top;

        if (viewMode === "cluster") {
          const hit = hitTestCluster(clustersRef.current, cx, cy, zoomRef.current, panRef.current.x, panRef.current.y);
          if (hit && hit.id === node.id) {
            setSelectedId((prev) => (prev === node.id ? null : node.id));
          }
        } else {
          const hit = hitTestEntry(entryNodesRef.current, cx, cy, zoomRef.current, panRef.current.x, panRef.current.y);
          if (hit && hit.id === node.id) {
            setSelectedEntryId((prev) => (prev === node.id ? null : node.id));
          }
        }
      }
      dragRef.current = null;
    }
    panDragRef.current = null;
  }, [viewMode]);

  const onDoubleClick = useCallback((e: MouseEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;

    if (viewMode === "cluster") {
      const hit = hitTestCluster(clustersRef.current, cx, cy, zoomRef.current, panRef.current.x, panRef.current.y);
      if (hit) {
        drillIntoCluster(hit.id, hit.label);
      }
    }
    // No drill-down from entry mode — entries are the leaf level
  }, [viewMode, drillIntoCluster]);

  const onWheel = useCallback((e: WheelEvent) => {
    e.preventDefault();
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    const oldZoom = zoomRef.current;
    const factor = e.deltaY < 0 ? 1.1 : 0.9;
    const newZoom = Math.max(0.2, Math.min(5, oldZoom * factor));
    panRef.current = {
      x: cx - (cx - panRef.current.x) * (newZoom / oldZoom),
      y: cy - (cy - panRef.current.y) * (newZoom / oldZoom),
    };
    zoomRef.current = newZoom;

    // LOD switching based on zoom level (only in non-degraded mode)
    if (!isDegraded) {
      const newLevel = zoomToLevel(newZoom, lastZoomRef.current);
      lastZoomRef.current = newZoom;
      if (newLevel !== lodLevel) {
        // Reset breadcrumbs when zoom-driven LOD changes (keep only root)
        setBreadcrumbs([{ level: 0, label: "All" }]);
        setLodLevel(newLevel);
        setSelectedId(null);
        setSelectedEntryId(null);
        setHoveredNodeId(null);
      }
    }

    forceRender((n) => n + 1);
  }, [lodLevel, isDegraded]);

  // Resize observer
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !canvas.parentElement) return;
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const w = entry.contentRect.width;
        const h = entry.contentRect.height;
        sizeRef.current = { w, h };
        canvas.width = w * devicePixelRatio;
        canvas.height = h * devicePixelRatio;
        canvas.style.width = `${w}px`;
        canvas.style.height = `${h}px`;
      }
    });
    observer.observe(canvas.parentElement);
    return () => observer.disconnect();
  }, []);

  if (loading) return <div class="loading">Loading graph...</div>;
  if (error) return <div class="error-banner">{error}</div>;
  if (!data) return null;

  const resp = data as GraphScopeResponse;
  const edges = resp.edges ?? [];
  const nodes = resp.nodes ?? [];

  if (edges.length === 0 && nodes.length === 0) {
    return <EdgeTable edges={edges} nodes={nodes} />;
  }

  // Compute available roles/origins for filter dropdowns
  const availableRoles = [...new Set(nodes.map((n) => n.content_role).filter(Boolean) as string[])];
  const availableOrigins = [...new Set(nodes.map((n) => n.source_origin).filter(Boolean) as string[])];

  // Build cluster info for UI
  const clusters = clustersRef.current;
  const clusterEdges = clusterEdgesRef.current;
  const selectedCluster = clusters.find((c) => c.id === selectedId);
  const selectedEdges = selectedId
    ? clusterEdges.filter((e) => e.from === selectedId || e.to === selectedId)
    : [];

  // Intra-cluster edge count for selected
  const intraCount = selectedId
    ? edges.filter((e) => {
        const fg = nodes.find((n) => n.id === e.from)?.group;
        const tg = nodes.find((n) => n.id === e.to)?.group;
        return fg === selectedId && tg === selectedId;
      }).length
    : 0;

  return (
    <div class="graph-split-panel">
      <div class={`graph-panel ${selectedEntryId ? "graph-panel--narrow" : ""}`}>
        <div style={{ display: "flex", flexDirection: "column", height: "100%", minHeight: "500px" }}>
          {/* Toolbar */}
          <div class="graph-toolbar">
            <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
              {viewMode === "entry" && isDegraded && (
                <button
                  class="graph-back-btn"
                  onClick={switchToClusterMode}
                  title="Back to clusters"
                >
                  &larr; Clusters
                </button>
              )}
              <span style={{ fontSize: "12px", fontWeight: 600, color: "var(--text-dim)", textTransform: "uppercase", letterSpacing: "0.5px" }}>
                {isDegraded
                  ? (viewMode === "cluster" ? "Knowledge Graph" : `Cluster: ${focusCluster?.replace("corvia-", "")}`)
                  : `Knowledge Graph \u00b7 ${LOD_LABELS[lodLevel] ?? "L" + lodLevel}`
                }
              </span>
              <span style={{ fontSize: "11px", color: "var(--text-dim)", fontFamily: "var(--font-mono)" }}>
                {viewMode === "cluster"
                  ? `${nodes.length} entries \u00b7 ${edges.length} edges \u00b7 ${clusters.length} clusters`
                  : `${entryNodesRef.current.length} entries \u00b7 ${entryEdgesRef.current.length} edges`
                }
              </span>
              {isDegraded && (
                <span style={{
                  fontSize: "10px", fontWeight: 600, color: "var(--gold)",
                  background: "var(--gold-soft)", padding: "2px 8px",
                  borderRadius: "4px",
                }}>
                  Path-based fallback
                </span>
              )}
            </div>
            <span class="graph-hint">
              {viewMode === "cluster"
                ? "Drag clusters \u00b7 Scroll to zoom \u00b7 Click to inspect \u00b7 Double-click to expand"
                : "Drag nodes \u00b7 Scroll to zoom \u00b7 Click to read entry"
              }
            </span>
          </div>

          {/* Breadcrumb bar (LOD mode only, when navigated past root) */}
          {!isDegraded && breadcrumbs.length > 1 && (
            <div class="graph-breadcrumbs">
              {breadcrumbs.map((bc, i) => (
                <span key={`${bc.level}-${bc.id ?? "root"}`}>
                  {i > 0 && <span class="graph-breadcrumb-sep">{" \u2192 "}</span>}
                  <button
                    class={`graph-breadcrumb-btn${i === breadcrumbs.length - 1 ? " graph-breadcrumb-btn--active" : ""}`}
                    onClick={() => navigateToLevel(bc.level, bc.id)}
                  >
                    {bc.label}
                  </button>
                </span>
              ))}
            </div>
          )}

          {/* Controls */}
          <GraphControls
            onFilterChange={setFilters}
            availableRoles={availableRoles}
            availableOrigins={availableOrigins}
          />

          {/* Canvas */}
          <div style={{
            flex: 1, position: "relative", background: "var(--bg-card)",
            borderRadius: "0 0 var(--radius-md) var(--radius-md)",
            overflow: "hidden", border: "1px solid var(--border)", borderTop: "none",
          }}>
            <canvas
              ref={canvasRef}
              onMouseDown={onMouseDown}
              onMouseMove={onMouseMove}
              onMouseUp={onMouseUp}
              onMouseLeave={(e: MouseEvent) => { onMouseUp(e); setHoveredNodeId(null); }}
              onDblClick={onDoubleClick}
              onWheel={onWheel}
              style={{ cursor: dragRef.current ? "grabbing" : "grab", display: "block" }}
            />

            {/* Zoom / LOD indicator overlay */}
            {!isDegraded && (
              <div style={{
                position: "absolute", bottom: "8px", left: "8px",
                fontSize: "10px", fontFamily: "var(--font-mono)",
                color: "var(--text-dim)", background: "rgba(30,34,48,0.85)",
                padding: "4px 8px", borderRadius: "4px",
                pointerEvents: "none",
              }}>
                {zoomRef.current.toFixed(1)}x \u00b7 L{lodLevel}
              </div>
            )}
          </div>
        </div>

        {/* Cluster detail panel (only in cluster mode) */}
        {viewMode === "cluster" && selectedCluster && (
          <div style={{ width: "280px", flexShrink: 0, display: "flex", flexDirection: "column", gap: "12px", overflowY: "auto", position: "absolute", right: 0, top: 0, bottom: 0, padding: "12px", background: "var(--bg-primary)" }}>
            {/* Cluster header */}
            <div class="trace-card">
              <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "12px" }}>
                <span style={{
                  width: "12px", height: "12px", borderRadius: "50%",
                  background: colorMapRef.current.get(selectedId!) ?? "#8a8279",
                  flexShrink: 0,
                }} />
                <span style={{ fontSize: "15px", fontWeight: 700, color: "var(--text-bright)" }}>
                  {selectedCluster.label.replace("corvia-", "")}
                </span>
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "8px" }}>
                <div style={{ background: "var(--bg-elevated)", borderRadius: "var(--radius-xs)", padding: "8px", textAlign: "center" }}>
                  <div style={{ fontSize: "18px", fontWeight: 700, fontFamily: "var(--font-mono)", color: "var(--text-bright)" }}>
                    {selectedCluster.count}
                  </div>
                  <div style={{ fontSize: "9px", color: "var(--text-dim)", textTransform: "uppercase", letterSpacing: "0.5px" }}>
                    Entries
                  </div>
                </div>
                <div style={{ background: "var(--bg-elevated)", borderRadius: "var(--radius-xs)", padding: "8px", textAlign: "center" }}>
                  <div style={{ fontSize: "18px", fontWeight: 700, fontFamily: "var(--font-mono)", color: "var(--text-bright)" }}>
                    {selectedEdges.reduce((sum, e) => sum + e.count, 0)}
                  </div>
                  <div style={{ fontSize: "9px", color: "var(--text-dim)", textTransform: "uppercase", letterSpacing: "0.5px" }}>
                    Cross-links
                  </div>
                </div>
                <div style={{ background: "var(--bg-elevated)", borderRadius: "var(--radius-xs)", padding: "8px", textAlign: "center" }}>
                  <div style={{ fontSize: "18px", fontWeight: 700, fontFamily: "var(--font-mono)", color: "var(--text-bright)" }}>
                    {intraCount}
                  </div>
                  <div style={{ fontSize: "9px", color: "var(--text-dim)", textTransform: "uppercase", letterSpacing: "0.5px" }}>
                    Internal
                  </div>
                </div>
              </div>
            </div>

            {/* Expand cluster button */}
            <button
              class="graph-expand-btn"
              onClick={() => drillIntoCluster(selectedId!, selectedCluster.label)}
            >
              Expand entries &rarr;
            </button>

            {/* Connected clusters */}
            <div class="trace-card">
              <div style={{ fontSize: "11px", fontWeight: 600, color: "var(--text-dim)", textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: "10px" }}>
                Connected Clusters ({selectedEdges.length})
              </div>
              {selectedEdges.length === 0 ? (
                <div class="trace-empty">No cross-cluster edges</div>
              ) : (
                <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                  {selectedEdges
                    .sort((a, b) => b.count - a.count)
                    .map((e, i) => {
                      const otherId = e.from === selectedId ? e.to : e.from;
                      const otherColor = colorMapRef.current.get(otherId) ?? "#8a8279";
                      return (
                        <div
                          key={i}
                          style={{
                            display: "flex", alignItems: "center", gap: "8px",
                            padding: "6px 8px", background: "var(--bg-elevated)",
                            borderRadius: "var(--radius-xs)", cursor: "pointer",
                          }}
                          onClick={() => setSelectedId(otherId)}
                        >
                          <span style={{ width: "8px", height: "8px", borderRadius: "50%", background: otherColor, flexShrink: 0 }} />
                          <span style={{ fontSize: "12px", fontWeight: 600, color: "var(--text-primary)", flex: 1 }}>
                            {otherId.replace("corvia-", "")}
                          </span>
                          <span style={{
                            fontSize: "11px", fontFamily: "var(--font-mono)", fontWeight: 700,
                            color: "var(--gold)", padding: "1px 6px",
                            background: "var(--gold-soft)", borderRadius: "8px",
                          }}>
                            {e.count}
                          </span>
                        </div>
                      );
                    })}
                </div>
              )}
            </div>

            {/* Top files in cluster */}
            <div class="trace-card">
              <div style={{ fontSize: "11px", fontWeight: 600, color: "var(--text-dim)", textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: "10px" }}>
                Files ({selectedCluster.files.length})
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: "4px", maxHeight: "300px", overflowY: "auto" }}>
                {selectedCluster.files.slice(0, 30).map((f, i) => (
                  <div key={i} style={{
                    fontSize: "11px", fontFamily: "var(--font-mono)", color: "var(--text-muted)",
                    padding: "4px 6px", background: "var(--bg-elevated)", borderRadius: "var(--radius-xs)",
                    overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                  }} title={f.source_file ?? f.label}>
                    {f.source_file ?? f.label}
                  </div>
                ))}
                {selectedCluster.files.length > 30 && (
                  <div style={{ fontSize: "10px", color: "var(--text-dim)", padding: "4px 6px" }}>
                    +{selectedCluster.files.length - 30} more
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Document reader panel (only in entry mode when entry selected) */}
      {selectedEntryId && (
        <div class="reader-panel">
          <DocumentReader
            entryId={selectedEntryId}
            onNavigate={(id) => setSelectedEntryId(id)}
            onClose={() => setSelectedEntryId(null)}
            navigateToHistory={navigateToHistory}
          />
        </div>
      )}
    </div>
  );
}

// --- Fallback table ---

function EdgeTable({ edges, nodes }: { edges: GraphScopeEdge[]; nodes: GraphNode[] }) {
  const nodeMap = new Map<string, GraphNode>();
  for (const n of nodes) nodeMap.set(n.id, n);

  function nodeLabel(id: string): string {
    const n = nodeMap.get(id);
    if (n?.source_file) return n.source_file;
    if (n?.label) return n.label.length > 40 ? n.label.slice(0, 38) + "\u2026" : n.label;
    return shortId(id);
  }

  return (
    <div class="card graph-edges">
      <h2>Knowledge Graph ({edges.length} edges)</h2>
      {edges.length === 0 ? (
        <div style={{ color: "var(--text-dim)", textAlign: "center", padding: "20px" }}>
          No graph edges found
        </div>
      ) : (
        <table>
          <thead>
            <tr>
              <th>From</th>
              <th>Relation</th>
              <th>To</th>
              <th style={{ width: "80px" }}>Weight</th>
            </tr>
          </thead>
          <tbody>
            {edges.map((e, i) => (
              <tr key={i}>
                <td title={e.from}>{nodeLabel(e.from)}</td>
                <td class="relation">{e.relation}</td>
                <td title={e.to}>{nodeLabel(e.to)}</td>
                <td>{e.weight?.toFixed(2) ?? "\u2014"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

function shortId(id: string): string {
  if (id.length > 12) return id.slice(0, 8) + "\u2026" + id.slice(-4);
  return id;
}
