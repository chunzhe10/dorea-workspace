# Interactive Graph + Document Browsing UX Research

> **Status:** Reference

Research date: 2026-03-13

Comparative analysis of products that combine document reading with graph navigation,
focused on concrete UX patterns, interaction models, and lessons for corvia's dashboard.

---

## 1. Obsidian

**Category**: Personal knowledge management (local-first, Markdown)

### Graph-to-Document Navigation UX

- **Global Graph View**: Force-directed layout showing all notes as nodes, links as edges.
  Opens as a full tab. Clicking a node opens the corresponding note in the main editor pane.
- **Local Graph View**: Shows only connections for the currently active note. Can be dragged
  into the right sidebar where it dynamically updates as users select different notes. This
  is the recommended "graph + document" workflow.
- **Click behavior**: Clicking a node in the graph opens the note in the main pane. When the
  graph is pinned in a sidebar, clicking a node has historically been buggy -- sometimes it
  replaces the graph panel itself instead of opening in the main pane.
- **Ctrl+Click**: Opens the note in a new split pane while keeping the graph visible.
- **Hover**: Highlights connected nodes and fades unrelated ones (one of the best-received features).

### Panel Pattern

**Split-panel with sidebar graph**. The primary pattern is: editor in the main pane, local graph
in the right sidebar. The graph acts as a navigation aid, not the primary workspace.

### Large Graph Handling

- Force-directed layout with physics simulation
- Filtering by file groups (folders, tags), orphan nodes, attachments
- Color groups for visual categorization
- Depth slider to control how many hops of connections are shown (local graph)
- **Performance issues**: Vaults with 100K+ notes can take 10+ minutes to index. Graph becomes
  unusable above ~10K nodes without aggressive filtering. Nodes shake/lag when toggling settings.

### What Works Well

- Hover-to-highlight-connected-nodes is universally praised
- Local graph in sidebar is the actual useful pattern (not global graph)
- Depth slider for controlling connection radius
- Color coding by folder/tag groups

### Common Complaints

- Graph view feels "secondary" and decorative, not integral to the workflow
- Clicking nodes is imprecise -- you often click and nothing happens or the wrong thing opens
- No way to preview note content without opening it (no hover preview on graph nodes)
- Overwhelming at scale -- becomes a "hairball" that provides no insight
- No shape/size differentiation for node types beyond connection count
- Users request: node folding, progressive disclosure, graph sculpting (hide/show individual nodes)
- Feature request for "graph navigation mode" where clicking selects nodes without opening notes

### Innovative Patterns to Steal

- **Hover-highlight with fade**: Hovering a node highlights its direct connections and fades everything else
- **Local graph depth slider**: Controls how many hops of relationships to show
- **Dynamic sidebar graph**: Graph updates reactively as user navigates documents

---

## 2. Notion

**Category**: Team workspace (documents + databases)

### Graph-to-Document Navigation UX

Notion has **no built-in graph view**. It is fundamentally document-and-database-first.
Connections between pages exist via @mentions, database relations, and inline page links,
but there is no visual graph representation.

### Third-Party Solutions

- **IVGraph**: Upcoming tool (beta Q1 2026) that provides 3D interactive graph visualization
  of Notion workspaces. Claims 86% performance improvement in latest architecture. Connects
  via Notion API, reads @mentions, relations, and links to build the graph. Not yet publicly
  available.
- **Graphify**: Transforms Notion workspace into an interactive knowledge graph using existing
  mentions and relations. Available as a Notion integration.
- **notion-graph-view**: Open-source tool that generates Roam-like network graphs from Notion pages.

### Panel Pattern

N/A -- Notion's native UX is linear document navigation with breadcrumbs and sidebar tree.

### What Works Well

- Page linking via @mentions is frictionless
- Database relations create structured connections
- Synced blocks allow content reuse across pages

### Common Complaints

- No visual way to see the network of connections
- Can't discover unexpected connections between documents
- Navigation is always hierarchical (tree) or explicit (link), never emergent

### Lessons

Notion proves that **graph visualization is not necessary for linked knowledge** -- but its
absence is the single most common feature request from power users migrating from Obsidian/Roam.
The gap matters most for discovery of implicit relationships.

---

## 3. Roam Research

**Category**: Networked thought (outliner with bidirectional links)

### Graph-to-Document Navigation UX

- **Graph Overview**: Full knowledge graph accessible from the sidebar. Clicking any node
  navigates graph-to-graph, showing connections for each clicked node.
- **Page Graph**: Small graph icon in the top-right of every page shows a local graph of
  directly connected pages. Clicking nodes transitions between page graphs.
- **Shift+Click**: Opens any link in the right sidebar without losing current context.
  This is Roam's signature interaction -- you can stack multiple pages in the right sidebar
  for cross-referencing.
- **Linked/Unlinked References**: Below every page, automatic sections show all pages that
  reference the current page (linked) and pages that mention the page title without explicit
  links (unlinked). These can be filtered.

### Panel Pattern

**Main page + right sidebar stack**. The main editing area shows one page. Shift+click opens
additional pages stacked vertically in a right sidebar. Users can have many pages open
simultaneously for reference. The graph is a discovery tool, not the primary workspace.

### Large Graph Handling

- Graph overview becomes unwieldy with large databases
- Primary navigation is through daily notes + bidirectional links, not the graph
- Filtering linked references by keyword or page

### What Works Well

- **Shift+click to sidebar** is the single most innovative interaction pattern in the space.
  It solves the "I want to read this but not lose my place" problem.
- Unlinked references surface hidden connections automatically
- Block-level references allow granular linking (not just page-to-page)
- Daily notes as entry point creates a natural knowledge accumulation pattern

### Common Complaints

- Graph view is rarely used in practice -- it's more of a demo feature
- $15/month pricing for what competitors offer free
- Performance degrades with large graphs
- No offline access (cloud-only)

### Innovative Patterns to Steal

- **Shift+click to right sidebar** for non-destructive navigation
- **Unlinked references** (automatic mention detection without explicit linking)
- **Block-level references** (linking to specific paragraphs, not just pages)
- **Stacked sidebar pages** for multi-document cross-referencing

---

## 4. Athens Research

**Category**: Open-source Roam alternative (discontinued)

### Status

Athens is **no longer maintained**. It was a YC W21-backed open-source clone of Roam Research
built in Clojure/ClojureScript. The project has been archived on GitHub.

### What It Was

- Outliner with bidirectional links and block references (identical model to Roam)
- Built on a graph database backend
- Open-source and local-first (addressing Roam's cloud-only limitation)
- Collaborative editing features

### Why It Failed

- Could not sustain development pace against Roam, Logseq, and Obsidian
- Community fragmented across too many similar tools
- Business model challenges for open-source knowledge management

### Lesson

Open-source Roam clones struggle because the bidirectional linking UX is table-stakes now.
The differentiator is in the ecosystem (plugins, integrations, community) not the core graph model.

---

## 5. Logseq

**Category**: Open-source knowledge management (outliner + graph, local-first)

### Graph-to-Document Navigation UX

- **Global Graph**: Full graph view of all pages. Clicking a node opens the page in the
  editor, replacing the graph view (no split-panel behavior).
- **Page Graph**: Shows in the sidebar as a graph of pages connected to the current page.
  Similar to Obsidian's local graph.
- **Shift+Click limitation**: Shift+click opens normal links in the sidebar, but does NOT
  work from the graph view. This is a frequently requested feature.
- **Block-based**: Like Roam, Logseq uses block-level linking, and pages are trees of blocks.

### Panel Pattern

**Full-screen replacement**. Clicking a graph node replaces the current view with the document.
No built-in split between graph and document reading. Users have requested the ability to open
pages from graph view in the sidebar.

### Large Graph Handling

- No special handling beyond the standard force-directed layout
- Community complaints about the graph being "useless" for large vaults
- Feature requests for additional layout algorithms (hierarchical, radial, etc.)
- Users report that the graph view has never been practically useful even after years of use

### What Works Well

- Open-source and local-first (Markdown/Org-mode files)
- Block-level bidirectional linking
- PDF annotation integration
- Plugin ecosystem

### Common Complaints

- Graph view is considered decorative, not functional
- Clicking a node loses your graph position (no back navigation)
- Cannot open pages in sidebar from graph view
- No meaningful way to filter or focus the graph
- Users describe a "conceptual gap" between how Logseq is supposed to be used and what
  the graph actually provides

### Lessons

Logseq demonstrates the failure mode of "graph view as afterthought." When graph navigation
destroys the graph context (by replacing the view), users abandon it entirely. The graph
MUST coexist with the document reader.

---

## 6. Neo4j Bloom

**Category**: Graph database visualization (enterprise)

### Graph-to-Document Navigation UX

- **Natural language search**: Users type near-natural-language queries into a search bar.
  Bloom suggests matching graph patterns and renders results as node-link diagrams.
- **Click to select**: Left-click selects a node, highlighting it. No document opens --
  this is a graph-native tool, not a document reader.
- **Double-click for Inspector**: Double-clicking a node opens the **Inspector panel**, which
  shows all properties, relationships, and neighbor nodes as interactive cards. You can click
  neighbor cards to navigate to them.
- **Right-click context menu**: Offers Expand (show immediate neighbors), Dismiss (remove
  from scene), Select Related, Reveal Relationships, Group Nodes, and Shortest Path.

### Panel Pattern

**Graph canvas + card list sidebar + inspector overlay**.

Three-tier information architecture:
1. **Scene** (main canvas): The graph visualization itself
2. **Card List** (sidebar drawer): Lists all nodes/relationships in the scene as cards.
   Selecting a card highlights the node in the scene and vice versa (bidirectional selection).
   Can filter by nodes/relationships and search within the list.
3. **Inspector** (overlay on double-click): Shows all properties with type annotations (string,
   integer, float), relationships as navigable cards, neighbor nodes as navigable cards.
   Supports inline property editing.

Additional panels:
- **Legend**: Shows available entity types with styling controls
- **Perspective drawer**: Defines which categories/relationships are visible

### Large Graph Handling

- **Advanced Expansion**: When expanding a node, users can choose specific relationship types,
  target categories, and result limits to avoid overwhelming the scene
- **Grouping**: Multiple nodes can be collapsed into a single visual group
- **Dismiss**: Remove individual nodes or all unconnected nodes
- **Map Overview**: Mini-map for navigating large scenes

### What Works Well

- **Inspector showing neighbors as navigable cards** is excellent for graph exploration
- Natural language search lowers the barrier to query construction
- Bidirectional selection (card list <-> graph scene) keeps context synchronized
- Property type annotations on hover help users understand the data model
- Advanced expansion with relationship type filtering prevents information overload

### Common Complaints

- Enterprise-focused pricing
- Requires Neo4j database (not a general-purpose tool)
- Scene can still become cluttered without discipline

### Innovative Patterns to Steal

- **Double-click Inspector with navigable neighbor cards**: Clicking a neighbor card in the
  inspector navigates to that node, creating seamless graph traversal from a detail panel
- **Bidirectional card list <-> graph selection**: Selecting in either view highlights in both
- **Advanced Expansion dialog**: Choose relationship types and limits before expanding
- **Right-click context menu** with graph-specific actions (expand, dismiss, group, shortest path)

---

## 7. Kumu

**Category**: Relationship mapping / systems visualization

### Graph-to-Document Navigation UX

- **Click to select**: Clicking an element on the map opens its **Profile** in the right sidebar.
- **Profile panel**: Shows Label, Type, Description, Tags, Image, and custom fields.
  Description supports rich text for narrative content.
- **Tab to toggle**: Press Tab on keyboard to show/hide the sidebar profile panel.
- **Multi-select**: Selecting multiple elements shows a bulk editing tool in the sidebar.

### Panel Pattern

**Map canvas + right sidebar profile**. Clean two-panel layout where the map is always visible
and the sidebar shows details for the selected element. Sidebar is toggle-able.

### Large Graph Handling

- **Organize button**: Activates a built-in graph algorithm that centers the most-linked element
- **Map templates**: System (loop diagrams), Stakeholder (people relationships), SNA (network
  analysis) -- each with different layout defaults
- **Views**: Save different filter/styling configurations as named views
- **Decorations**: Conditional formatting based on data values (size, color, shape by field values)

### What Works Well

- Simple and clean: click element, see profile in sidebar. No complexity.
- Custom fields allow domain-specific metadata
- Description field supports narrative/explanatory text alongside the graph
- Conditional styling (decorations) makes data patterns visible
- Easy to embed and share maps

### Common Complaints

- Limited to relationship mapping (not a general knowledge tool)
- No content preview on hover
- Profile sidebar is basic -- no structured content or rich documents

### Innovative Patterns to Steal

- **Tab key to toggle sidebar**: Simple, memorable keyboard shortcut
- **Profile with custom fields**: Extensible metadata alongside the graph
- **Decorations**: Data-driven visual styling (node size/color/shape from field values)
- **Multi-select bulk editing** in the sidebar

---

## 8. Scrintal

**Category**: Visual-first knowledge management (infinite canvas)

### Graph-to-Document Navigation UX

- **Canvas IS the graph**: Unlike other tools where graph view is separate, Scrintal's
  infinite canvas IS the primary workspace. Cards (notes) are placed spatially on the canvas
  and connected with visual links.
- **Click card to read**: Clicking a card on the canvas expands it for reading. Cards can
  be resized from folded (post-it size) to full-screen.
- **In-text linking**: [[wikilinks]] inside cards create bidirectional connections that also
  appear as visual edges on the canvas.
- **Visual linking**: Drag-to-connect creates visual links between cards independent of
  text content.

### Panel Pattern

**Canvas-as-workspace with expandable cards + sidebar navigation**.

The canvas replaces both the graph view and the document editor. Cards are the documents
and their spatial arrangement IS the graph. A sidebar provides access to all cards, boards,
tags, and starred items for navigation beyond the current canvas view.

### Large Graph Handling

- **Organize button**: Graph algorithm centers the most-linked card and arranges others
- **Boards**: Separate canvases for different topics (equivalent to multiple graphs)
- **Card folding**: Cards can be collapsed to post-it size to reduce visual clutter
- **Color coding**: Instant visual segmentation by card color
- **Sidebar navigation**: Search and browse all cards/boards outside the canvas

### What Works Well

- Eliminates the graph/document duality entirely -- they are the same surface
- Spatial arrangement creates intuitive visual memory
- Card folding for information density control
- Two types of connections (visual links and text links) serve different purposes
- Drag-to-connect is intuitive

### Common Complaints

- Performance degrades on very large canvases
- Limited compared to text-first tools for long-form writing
- No hierarchical organization (everything is flat on canvases)

### Innovative Patterns to Steal

- **Canvas as unified graph+document surface**: No mode switching between graph and reading
- **Card folding/sizing**: Post-it (collapsed) to full-screen (expanded) on the same surface
- **Dual connection types**: Visual (spatial) and textual (semantic) links as distinct concepts
- **Organize algorithm**: One-click graph layout that centers most-connected nodes

---

## 9. Heptabase

**Category**: Visual knowledge management (whiteboards + cards)

### Graph-to-Document Navigation UX

- **Whiteboard as workspace**: Cards are placed on infinite whiteboards. Connections are
  drawn between cards as visual edges.
- **Click card to open in side panel**: Clicking a card on the whiteboard opens it in the
  right side panel for reading. The whiteboard stays visible. This is the key interaction.
- **Alt+Click**: Alternative way to open a card in the side panel.
- **Block link navigation**: Clicking a block link inside a card opens the target card in
  the side panel.
- **Tab system**: Cards can also be opened as full tabs (like browser tabs) in the left sidebar.
- **Focus Mode**: Minimizes UI distractions when editing a card on the whiteboard.

### Panel Pattern

**Whiteboard + right side panel (reader) + left sidebar (tabs/navigation)**.

Three-zone layout:
1. **Left sidebar**: Tab system with pinned tabs, tab folders/groups, card library
2. **Center**: Whiteboard canvas with cards and connections
3. **Right sidebar**: Context panel showing card content, PDF highlights, table of contents,
   metadata, AI summaries, chat

The right sidebar is the "reader" -- you click a card on the whiteboard and read it on
the right while keeping the visual context. Multiple sidebar panels available: Chat, Card
Library, Highlight, Info, Table of Contents, Insight (AI summary).

### Large Graph Handling

- **Whiteboard tidy-up**: Batch tools for organizing multiple objects simultaneously
- **Multiple whiteboards**: Separate visual spaces for different topics
- **Card Library**: Searchable index of all cards across all whiteboards
- **Global Search (Cmd+O)**: Searches cards, whiteboards, chats, and tags with advanced
  filtering by card type, content source, and sorting
- **Tag tables**: Clicking a tag gathers all cards with that tag into a single table view

### What Works Well

- **Click-to-side-panel** keeps visual context while reading -- best implementation of
  graph+reader split
- Cards can exist on multiple whiteboards (same content, different spatial contexts)
- AI summaries chunk content into ~300 character units for quick scanning
- Deeplink support for cross-whiteboard navigation
- PDF annotation with highlights visible in the sidebar
- Bi-directional links with backlink display

### Common Complaints

- Performance slowdowns on large boards (mentioned in multiple reviews)
- Mobile experience is limited
- Collaboration features still maturing
- UI has "rough edges" according to reviewers
- Pricing can be a barrier

### Innovative Patterns to Steal

- **Click card on canvas, read in side panel**: The cleanest graph-to-content pattern found
- **Card on multiple whiteboards**: Same content in different visual contexts
- **AI Insight panel**: Auto-summarizes long content into digestible chunks
- **Right sidebar as multi-purpose reader**: Switches between content, metadata, highlights, TOC
- **Tag tables**: Automatic aggregation view that gathers tagged cards across all whiteboards
- **Focus mode**: Strips UI when editing a card on the whiteboard

---

## 10. GitHub Dependency Graph

**Category**: Code dependency visualization (supply chain security)

### Graph-to-Document Navigation UX

- **Not a visual graph**: Despite the name, GitHub's dependency "graph" is a **tabular list**,
  not a node-link diagram. It shows dependencies as rows with version, license, manifest
  file, and vulnerability status.
- **Click to navigate**: Clicking a dependency name navigates to that package's repository.
  Hovering shows repository metadata.
- **Dependents tab**: Separate tab shows which other repos depend on the current one.
- **Show Paths**: For transitive dependencies, a dropdown reveals the dependency chain.

### Panel Pattern

**Tabular list under Insights tab**. No split panel or graph visualization.

### Large Graph Handling

- Auto-sorting with vulnerable dependencies at top
- Keyword filters: `ecosystem:npm`, `relationship:direct`
- Pagination for large dependency lists
- Separate tabs for Dependencies vs Dependents

### What Works Well

- Vulnerability highlighting is immediately useful (security context)
- License information at a glance
- Transitive dependency chain visualization via "Show Paths"

### Common Complaints

- Not actually a visual graph -- many users expect a node-link diagram
- Limited to manifest-declared dependencies (misses dynamic imports)
- No way to visualize the full dependency tree as a graph

### Lessons

GitHub proves that **tabular views with smart sorting can be more useful than visual graphs**
for certain use cases (security auditing, license compliance). The "Show Paths" feature for
transitive dependencies is a clever progressive disclosure pattern.

---

## 11. Graphistry

**Category**: GPU-accelerated graph visualization (analytics)

### Graph-to-Document Navigation UX

- **Click node for details**: Click a node to see its properties. Click again to drag.
- **Data Brush tool**: Select a region of the graph to populate the inspector with details
  of all selected nodes.
- **Data Inspector (bottom panel)**: Table view of underlying data. Searchable, sortable,
  downloadable (CSV/Parquet/JSON).
- **Histogram panel (right panel)**: Shows attribute distributions. Click-drag to filter
  by value range. Clicking an attribute value adds a filter.

### Panel Pattern

**Graph canvas + right histogram panel + bottom data inspector**.

Three-panel layout optimized for data analysis:
1. **Center**: WebGL-rendered graph with GPU-accelerated layout
2. **Right**: Histogram panel for attribute analysis, filtering, and coloring
3. **Bottom**: Data inspector table for detail examination

### Large Graph Handling

- **GPU acceleration**: Server-side NVIDIA GPUs run layout and clustering algorithms.
  Client WebGL renders up to 8 million nodes+edges.
- **Streaming architecture**: Heavy computation on server GPU, light render data streamed
  to browser. Works on any device.
- **Timebar animation**: For temporal data, animates the graph over time
- **Pruning controls**: Toggle labels/icons, remove isolated nodes
- **Visual templates**: Pre-configured investigation workflows

### What Works Well

- **Scale**: Handles orders of magnitude more data than any other tool in this comparison
  (100K-8MM elements vs ~10K for consumer tools)
- **Filter-by-histogram** is intuitive for data exploration
- **Data Brush** for inspecting graph regions
- **Streaming GPU rendering** keeps the client responsive regardless of data size
- Timebar animation for temporal exploration

### Common Complaints

- Requires infrastructure (GPU server)
- Oriented toward data analysts, not knowledge workers
- No concept of "documents" -- purely node/edge property data
- Setup complexity

### Innovative Patterns to Steal

- **Histogram-based filtering**: Click-drag on a histogram to filter the graph by value range
- **Data Brush**: Select a region of the graph to inspect aggregated properties
- **GPU streaming architecture**: Server-side layout, client-side rendering for scale
- **Timebar animation**: Temporal graph exploration
- **Attribute-driven coloring/sizing**: Color and size nodes by any property value

---

## 12. Developer Documentation Tools (Dendron, Foam, TheBrain)

### Dendron (VS Code extension, discontinued active development)

- Graph view built with Cytoscape.js
- Keyboard navigation: up/down between hierarchies, cycle between children, open notes
- Hierarchical note organization (dot-separated paths like `project.module.component`)
- Graph as supplementary view within VS Code's panel system
- **Status**: No longer actively maintained but still usable

### Foam (VS Code extension, active)

- **Graph visualization**: `Foam: Show Graph` command opens a graph panel in VS Code
- Nodes represent files and tags, edges represent links
- Node size scales with connection count (visual importance indicator)
- Integrates with VS Code's native split-pane system
- [[wikilinks]] for linking, automatic backlinks
- **Limitation**: Graph is view-only, no interaction details documented for clicking to navigate
- **Key value**: Runs inside VS Code, so developers don't leave their editor

### TheBrain (commercial, cross-platform)

**Most innovative graph-to-content UX in the entire comparison.**

- **The Plex**: Dynamic graph visualization with animated transitions. Clicking a node
  re-centers the graph around it with smooth animation, showing parents above, children below,
  and siblings alongside.
- **Unified scroll surface (mobile)**: On mobile, the plex (graph) and content are a single
  scrollable surface. Swiping transitions from full-screen graph to full-screen content.
  No panel switching -- just scroll.
- **Radiant layout**: Force-simulation graph that users can expand/collapse arbitrarily
  and drag nodes to place.
- **Mapped Links (footer navigation)**: Below note content, a linear outline of all connections
  (parents, children, siblings, jumps) as clickable links. Allows graph traversal from
  within the document without seeing the graph at all.
- **Unlinked Mentions**: Automatically detects when note text mentions existing nodes and
  underlines them. Right-click to activate as a link.
- **Aggregated Content**: Backlinks section shows where the current node is referenced
  elsewhere, with paragraph-level context excerpts.
- **Plex hide/show**: The graph can be completely hidden while still navigating via mapped
  links in the content footer.

**Key Innovation**: TheBrain treats graph and content as **a single continuous surface** rather
than two separate panels. The graph is not a view -- it is the navigation system embedded
in and around the content.

---

## Comparative Analysis: UX Patterns

### Pattern 1: Graph-to-Document Navigation Models

| Model | Tools | Description | Best For |
|-------|-------|-------------|----------|
| **Replace** | Logseq, Obsidian (global) | Click node, document replaces graph | Simple but loses graph context |
| **Split panel** | Obsidian (local), Kumu, Neo4j Bloom | Graph in one pane, document in another | Maintaining visual context while reading |
| **Side panel reader** | Heptabase, Roam (sidebar) | Click node on canvas, content opens in side panel | Best balance of context and readability |
| **Canvas-as-document** | Scrintal | Cards ON the canvas ARE the documents | Spatial thinkers, visual organization |
| **Unified surface** | TheBrain | Graph and content are one scrollable surface | Seamless transitions, mobile |
| **Tabular** | GitHub, Graphistry (inspector) | List/table view with click-to-navigate | Data-heavy, analytical use cases |

### Pattern 2: Detail Panel Designs

| Design | Tools | Behavior |
|--------|-------|----------|
| **Inspector overlay** | Neo4j Bloom | Double-click opens property inspector with navigable neighbor cards |
| **Right sidebar** | Heptabase, Kumu, Roam | Click opens detail/content in persistent right panel |
| **Expandable card** | Scrintal | Click card on canvas to resize/expand inline |
| **Bottom data table** | Graphistry | Bottom panel shows tabular data for selected nodes |
| **Footer links** | TheBrain | Navigation links embedded below content in the document itself |

### Pattern 3: Large Graph Handling Strategies

| Strategy | Tools | Description |
|----------|-------|-------------|
| **Local/ego graph** | Obsidian, Logseq, Roam | Show only N hops from current node |
| **Filtering** | Obsidian, Neo4j Bloom, Graphistry | Filter by type, relationship, property value |
| **Grouping/folding** | Neo4j Bloom, TheBrain | Collapse multiple nodes into one visual group |
| **Separate boards** | Heptabase, Scrintal | Multiple canvases for different topics |
| **GPU rendering** | Graphistry | Server-side GPU layout, client WebGL rendering |
| **Progressive expansion** | Neo4j Bloom | Choose relationship types before expanding |
| **Smart sorting** | GitHub | Tabular view with priority sorting (vulnerabilities first) |

### Pattern 4: Filtering Mechanisms

| Mechanism | Tools | Description |
|-----------|-------|-------------|
| **Depth slider** | Obsidian | Control number of hops shown |
| **Relationship type filter** | Neo4j Bloom | Expand only specific edge types |
| **Histogram range filter** | Graphistry | Click-drag on histogram to filter by value range |
| **Text search** | All | Filter nodes by name/content |
| **Tag/type filter** | Obsidian, Kumu, Heptabase | Show only nodes of specific types |
| **Data-driven decorations** | Kumu, Graphistry | Size/color/shape from property values |

---

## Top Patterns to Adopt for corvia Dashboard

### Must-Have (Tier 1)

1. **Click node -> right side panel reader** (Heptabase pattern)
   - Click a node in the graph to open its content in a right side panel
   - Graph stays visible and interactive
   - Side panel shows: content preview, metadata, related entries, graph edges
   - This is the single most effective pattern across all tools studied

2. **Hover-highlight with fade** (Obsidian pattern)
   - Hovering a node highlights its direct connections
   - All unrelated nodes and edges fade to low opacity
   - Provides instant visual context without clicking

3. **Bidirectional selection sync** (Neo4j Bloom pattern)
   - Selecting a node in the graph highlights it in any list/table view
   - Selecting an entry in a list highlights it in the graph
   - Keeps both views synchronized

4. **Local/ego graph mode** (Obsidian/Roam pattern)
   - Option to show only nodes within N hops of the selected node
   - Depth slider to control the radius
   - Essential for large knowledge bases

### Should-Have (Tier 2)

5. **Navigable neighbor cards in detail panel** (Neo4j Bloom pattern)
   - The detail panel shows related entries as clickable cards
   - Clicking a neighbor card navigates to that node
   - Enables graph traversal from the detail panel without returning to the graph

6. **Relationship type filtering on expand** (Neo4j Bloom pattern)
   - When expanding a node, choose which edge types to follow
   - Prevents information overload from showing all connections

7. **Data-driven node styling** (Kumu/Graphistry pattern)
   - Node size from connection count or edge weight
   - Node color from entry type or source
   - Edge color/thickness from relationship type or weight

8. **Keyboard shortcut for panel toggle** (Kumu pattern)
   - Tab or similar key to show/hide the detail panel
   - Quick keyboard-driven workflow

### Nice-to-Have (Tier 3)

9. **Shift+click to open in sidebar without losing position** (Roam pattern)
   - Different click modifiers for different navigation behaviors
   - Regular click: select. Shift+click: open in side panel. Ctrl+click: open in new tab.

10. **Unlinked mentions / automatic relationship discovery** (Roam/TheBrain pattern)
    - Surface entries that mention the selected entry's content but aren't explicitly linked
    - Helps discover hidden connections

11. **Histogram-based filtering** (Graphistry pattern)
    - For large graphs, show distribution of edge weights or node properties
    - Filter by clicking/dragging on the histogram

12. **Graph layout algorithms** (Scrintal/Neo4j pattern)
    - One-click "organize" that centers the most-connected node
    - Multiple layout options (force-directed, hierarchical, radial)

---

## Anti-Patterns to Avoid

1. **Graph replaces document view** (Logseq failure mode)
   - Clicking a node should NEVER destroy the graph context
   - Always use split-panel, side-panel, or overlay

2. **Graph as decoration** (Obsidian/Logseq common complaint)
   - If the graph doesn't provide actionable navigation, users abandon it
   - The graph must be integral to the workflow, not a pretty visualization

3. **No content preview before navigation** (common across tools)
   - Users want to peek at content without committing to full navigation
   - Hover previews or expandable cards solve this

4. **Undifferentiated nodes** (Obsidian complaint)
   - All nodes looking the same (just circles of varying size) limits usefulness
   - Use color, shape, size, and icons to differentiate node types

5. **No way back** (Logseq complaint)
   - After navigating away from a graph view, users need to return to their position
   - Maintain graph state/position across navigation

---

## Key Insight: The Maturity Spectrum

Tools fall on a spectrum from "graph as afterthought" to "graph as core surface":

```
Graph as Afterthought          Graph as Core Surface
        |                              |
   Logseq  Notion  Obsidian  Roam  Neo4j  Heptabase  Scrintal  TheBrain
        |                              |
   Graph is decorative          Graph IS the workspace
```

The most successful tools (Heptabase, TheBrain) treat the graph not as a separate view
but as an integral part of the content interaction model. corvia's dashboard should aim
for the Heptabase zone: graph as a first-class navigation surface with seamless content
access via the side panel.

---

## Sources

- [Obsidian Graph View Design Discussion](https://forum.obsidian.md/t/design-talk-about-the-graph-view/22594)
- [Obsidian Graph View Documentation](https://help.obsidian.md/plugins/graph)
- [Obsidian Graph Navigation Mode Request](https://forum.obsidian.md/t/graph-navigation-mode/11471)
- [Obsidian Local Graph Sidebar](https://forum.obsidian.md/t/how-to-open-a-local-graph-view-pane-on-the-right-sidebar/7190)
- [Obsidian Large Vault Performance](https://forum.obsidian.md/t/obsidian-graph-view-doesnt-work-for-a-large-vault/106287)
- [Roam Research Beginner Guide](https://thesweetsetup.com/a-thorough-beginners-guide-to-roam-research/)
- [Roam Research Review 2025](https://aiproductivity.ai/tools/roam-research/)
- [Roam Research UX for Qualitative Data](https://uxdesign.cc/roam-research-a-new-way-of-working-with-qualitative-research-data-96534b9cd951)
- [Athens Research GitHub](https://github.com/athensresearch/athens)
- [Logseq Graph View Discussion](https://discuss.logseq.com/t/confusion-about-the-graph-view-whats-the-point-of-it-if-you-rely-on-blocks-and-journals/28136)
- [Logseq Sidebar from Graph Feature Request](https://discuss.logseq.com/t/is-it-possible-to-open-a-page-in-the-sidebar-from-the-graph-view/15283)
- [Neo4j Bloom Overview](https://neo4j.com/docs/bloom-user-guide/current/bloom-visual-tour/bloom-overview/)
- [Neo4j Bloom Card List](https://neo4j.com/docs/bloom-user-guide/current/bloom-visual-tour/card-list/)
- [Neo4j Bloom Scene Interactions](https://neo4j.com/docs/bloom-user-guide/current/bloom-visual-tour/bloom-scene-interactions/)
- [Kumu Profiles Documentation](https://docs.kumu.io/guides/profiles)
- [Kumu Map Editor](https://docs.kumu.io/overview/user-interfaces/map-editor)
- [Scrintal Knowledge Graph](https://scrintal.com/features/knowledge-graph)
- [Scrintal vs Obsidian Comparison](https://scrintal.com/comparisons/obsidian-vs-scrintal)
- [Heptabase User Interface Logic](https://wiki.heptabase.com/user-interface-logic)
- [Heptabase 2025 Changelog](https://wiki.heptabase.com/changelog/2025)
- [Heptabase Fundamental Elements](https://wiki.heptabase.com/fundamental-elements)
- [GitHub Dependency Graph Docs](https://docs.github.com/en/code-security/supply-chain-security/understanding-your-software-supply-chain/exploring-the-dependencies-of-a-repository)
- [Graphistry UI Guide](https://hub.graphistry.com/docs/ui/index/)
- [Graphistry GPU Visualization](https://www.graphistry.com/gpu)
- [Graphistry MCP Integration](https://github.com/graphistry/graphistry-mcp)
- [Dendron Graph View Design](https://docs.dendron.so/notes/6e87249b-358f-4f4b-8049-dff6e6a8463b/)
- [Foam Graph Visualization](https://github.com/foambubble/foam)
- [TheBrain Ubiquitous Visual Knowledge](https://www.thebrain.com/blog/enabling-ubiquitous-non-linear-visual-knowledge)
- [TheBrain Navigating Beyond the Plex](https://thebrain.com/blog/navigating-beyond-the-plex)
- [IVGraph Notion Knowledge Graph 2026](https://ivgraph.com/journal/ultimate-notion-knowledge-graph-guide-2026/)
- [Knowledge Graph Visualization Guide](https://datavid.com/blog/knowledge-graph-visualization)
- [UX Design for Enterprise Knowledge Graphs](https://predictiveux.com/insights/ux-design-for-enterprise-search-using-knowledge-graphs)
- [Top Graph-Based KM Tools 2025](https://blog.knowing.app/posts/top-graph-based-knowledge-management-tools-2025/)
- [Interactive Visual Knowledge Graphs Research](https://www.emergentmind.com/topics/interactive-visual-knowledge-graphs)
