# Layout Techniques Master Reference

**Date:** 2026-03-16
**Purpose:** Unified reference for all future dagua layout engine development. Synthesized from
four research reports covering academic literature, OGDF, Graphviz, ELK, D3, Cytoscape.js,
dagre, NetworkX, igraph, yEd/yFiles, and TikZ/pgf.

**Sources:**
- Academic/OGDF report (15 topics, theory-heavy)
- Practical tools report (NetworkX, igraph, yEd/yFiles, OGDF, TikZ)
- Graphviz/ELK report (inline notes)
- D3/Cytoscape/dagre report (js-layout-engines.md)

---

## 1. Skip Connections (Long-Span Edges in Layered Layout)

### Best-in-Class: ELK Layered / yEd Hierarchic

ELK and yEd both implement the full Sugiyama pipeline with mature long-edge handling:
dummy node insertion, dedicated long-edge ordering strategies, and multiple routing modes
(orthogonal, polyline, octilinear, curved). yEd adds backloop routing for reversed edges
and automatic edge grouping for skip-connection bundles. ELK exposes a dedicated
`longEdgeStrategy` option for controlling how dummy chains interact with crossing reduction.

### Comparison

| Tool | Approach | Skip-Connection Awareness | Routing Modes |
|------|----------|--------------------------|---------------|
| **ELK Layered** | Sugiyama + long-edge ordering strategy | Explicit, configurable | Ortho, polyline, spline |
| **yEd/yFiles** | Sugiyama + backloop routing + edge grouping | Explicit, best UX | Ortho, polyline, octilinear, curved |
| **Graphviz dot** | Sugiyama + virtual nodes + visibility-graph splines | Implicit via dummies | Spline, polyline, ortho |
| **dagre** | Sugiyama + Brandes-Kopf coordinate assignment | Explicit dummy chains | Polyline (staircase) |
| **OGDF Sugiyama** | FastHierarchyLayout: at most 2 bends per edge, vertical middle segment | Explicit, <=2 bends | Polyline |
| **TikZ/pgf** | Modular Sugiyama with configurable phases | Structural, via layer assignment | Polyline |
| **igraph Sugiyama** | Dummy vertices, `return_extended_graph` exposes bend points | Structural | Polyline |
| **D3 d3-force** | No special handling; skip edges are just springs | None | N/A (render-only) |
| **Cytoscape Cola** | Stress majorization respects graph distance implicitly | Indirect (shortest-path weights) | N/A |
| **NetworkX** | Force/spectral treat all edges uniformly | None | N/A |

### Key Algorithms

- **Sugiyama dummy expansion** [Sugiyama, Tagawa, Toda 1981]: Replace long edges with dummy
  vertex chains. Standard, O(|V||E|) worst case.
- **Eiglsperger-Siebenhaller-Kaufmann 2005**: Linearized dummy structure, reduces to
  O((|V|+|E|) log |E|) time and O(|V|+|E|) space while preserving crossing count.
- **Brandes-Kopf 2001**: Linear-time coordinate assignment with inner-segment conflict
  resolution. Adapted by dagre for variable node sizes.

### Recommendations for Dagua

**Implement:**
- Eiglsperger-style linearized dummy representation. Dagua's current init_placement uses
  topological sort + barycenter but doesn't have explicit dummy-chain management for
  crossing reduction. This is a key quality gap vs Graphviz.
- Long-edge ordering strategy during crossing reduction (ELK's approach).
- Backloop routing for reversed cycle-breaking edges (yEd's approach) to replace the
  current "wide arc" back-edge routing that overlaps nodes.

**Skip:**
- Multiple routing modes (ortho, octilinear, curved) are render-layer concerns. Dagua's
  Bezier routing is sufficient; invest in control-point quality, not mode variety.

---

## 2. Clustering / Compound Graph Layout

### Best-in-Class: yEd/yFiles Hierarchic

yEd has the most complete compound-graph stack: a true hierarchy tree where group nodes
contain both leaf nodes and child groups, bottom-up bounding box propagation, recursive
edge routing through nested boundaries, proxy edges for collapsed groups, and explicit
controls (Layout Groups, Fix Contents, Fix Bounds, Ignore Groups). No other tool matches
this integrated depth.

### Comparison

| Tool | Compound Model | Deep Nesting | Bbox Propagation | Inter-Cluster Routing |
|------|---------------|-------------|-----------------|----------------------|
| **yEd/yFiles** | Hierarchy tree, group nodes | Full | Bottom-up with insets/min-size | Recursive through boundaries |
| **ELK Layered** | INCLUDE_CHILDREN for global cluster optimization | Full | Layout-integrated | Through boundary ports |
| **Graphviz** | `subgraph cluster_*` + osage recursive packing | Partial | Recursive pack | Straight-line crossing boundaries |
| **OGDF** | ClusterGraph, ClusterPlanarizationLayout | Full (less documented) | Via ClusterGraphAttributes | Dual-graph shortest path |
| **dagre** | Sander's border-node method | Full | Border nodes constrain span | Through LCA path |
| **Cytoscape fCoSE** | Native parent-child compound nodes | Full | Children determine parent bounds | Not routed through boundaries |
| **Cytoscape Cola** | Dummy-node rectangles per group | Full | Dummy top-left/bottom-right | Not routed through boundaries |
| **D3** | No native support | None | N/A | N/A |
| **NetworkX** | No support | None | N/A | N/A |
| **igraph** | No support | None | N/A | N/A |
| **TikZ/pgf** | Syntactic subgraphs, not true compounds | Weak | N/A | N/A |

### Key Algorithms

- **Sander 1996**: Extends Sugiyama to cluster trees. Used by dagre with border-node pairs.
- **OGDF ClusterPlanarizationLayout** [Di Battista, Didimo, Marcandalli 2001]: C-planar
  orthogonal clustered drawing. Routes inter-cluster edges through a dual graph.
- **CoSE/CiSE** [Dogrusoz et al. 2009]: Force-directed compound layout. Separates
  intra-cluster and inter-cluster forces with nesting-factor scaling.
- **ELK INCLUDE_CHILDREN**: Global optimization across cluster boundaries rather than
  recursive independent layout. Better global edge quality at the cost of less modularity.
- **Graphviz osage**: Recursive cluster packing. Each cluster is laid out independently,
  then treated as a rectangle when packed with siblings.

### Recommendations for Dagua

**Implement (high priority -- TODO item "Cluster hierarchy nesting"):**
- Bottom-up bounding box propagation: lay out leaf clusters first, compute bbox + padding,
  then treat each cluster as an atomic rectangle at the parent level. This is the universal
  approach (OGDF, yEd, dagre, fCoSE all do it).
- Cluster containment loss term: attraction toward cluster centroid + penalty for escaping
  bbox. Dagua already has a `Cluster` constraint in constraints.py; extend it for nested
  hierarchy.
- Inter-cluster edge routing: route through LCA path in cluster tree (dagre's approach is
  simplest to implement differentiably).

**Implement (medium priority):**
- ELK-style INCLUDE_CHILDREN mode: optimize node positions globally while enforcing cluster
  containment as a constraint, rather than recursive independent layout. This produces
  better inter-cluster edge straightness. Dagua's differentiable framework makes this
  natural -- just add cluster containment as a loss term in the global optimization.

**Skip:**
- Collapsed-group proxy edges (yEd feature). This is an interactive editor feature, not
  a layout algorithm concern. Dagua is batch-oriented.
- C-planar testing and related planarity theory. Dagua is optimization-based, not
  embedding-based; planarity constraints don't fit the differentiable framework.

---

## 3. Edge Routing

### Best-in-Class: yEd/yFiles (breadth) / Cola GridRouter (obstacle avoidance)

yEd has the most routing modes (orthogonal, polyline, octilinear, curved, bus-style,
organic, generic bundling) and treats routing as a first-class subsystem with obstacle
awareness for labels and grouped graphs. Cola's GridRouter is the strongest open
obstacle-avoiding router: grid construction -> Dijkstra with bend penalty -> VPSC nudging.

### Comparison

| Tool | Edge Routing | Obstacle Avoidance | Spline/Curve Support |
|------|-------------|-------------------|---------------------|
| **yEd/yFiles** | 8+ router types, first-class subsystem | Full (nodes, labels, groups) | Orthogonal, polyline, octilinear, curved |
| **ELK Layered** | Integrated in Sugiyama; can delegate to Libavoid | Partial | Polyline, spline |
| **Graphviz** | Visibility-graph splines (dot), spring-model (neato) | Partial | B-spline, polyline, ortho |
| **Cola GridRouter** | Grid + Dijkstra + VPSC nudge | Full (rectangles) | Orthogonal only |
| **OGDF** | Planarization + orthogonalization; NodeRespecterLayout bends around nodes | Partial | Polyline, Bezier post-process |
| **dagre** | Implicit via dummy nodes in Sugiyama | By construction (no node crossing) | Polyline (staircase) |
| **Adaptagrams libavoid** | Visibility-graph connector routing | Full | Orthogonal, polyline |
| **D3** | None (render-only curves) | None | curveBundle, curveBasis, etc. |
| **Cytoscape** | Render-only styles (bezier, taxi, haystack) | None | Bezier, segments |
| **NetworkX/igraph** | None (node coordinates only) | None | N/A |
| **TikZ/pgf** | Layered polyline routing, necklace routing | Minimal | Polyline |

### Key Algorithms

- **Visibility-graph routing**: Build graph from obstacle corners, run Dijkstra for
  shortest-path polyline routes. Graphviz and libavoid both use this. Quadratic in
  obstacle complexity before path search.
- **Tamassia 1987**: Minimum-bend orthogonal drawing via min-cost flow. Exact for
  fixed-embedding 4-planar graphs. O(n^2 log n).
- **Topology-shape-metrics pipeline**: Fix embedding -> orthogonalize (min bends) ->
  compact (min area/length). The strongest theoretical framework for orthogonal routing.
- **Cola GridRouter**: Grid construction through layout space, Dijkstra with bend penalty
  (1000 per 90-degree turn), VPSC nudging for parallel segment separation.
- **D3 curveBundle** [Holten 2006]: B-spline rendering through hierarchy paths. Beta
  parameter controls bundling tightness. Render-only, not a layout algorithm.

### Recommendations for Dagua

**Implement (high priority -- TODO item "Learnable bezier control points"):**
- Differentiable Bezier control point optimization. Current routing.py is heuristic.
  Make control points learnable parameters with loss terms: edge-node crossing penalty,
  smoothness (G2 curvature), edge-edge crossing penalty, path length regularization.
  This is dagua's unique advantage -- no competitor optimizes routes differentiably.

**Implement (medium priority -- TODO item "Edge-node crossing avoidance"):**
- Edge-through-node penalty as a loss term during the main layout optimization.
  OGDF's NodeRespecterLayout does this with dummy nodes; dagua can do it with a
  differentiable distance-to-segment function.

**Skip:**
- Full visibility-graph routing. This is a discrete algorithm that doesn't fit dagua's
  differentiable framework. The learnable Bezier approach is strictly more powerful
  for dagua's architecture.
- Multiple routing modes (ortho, octilinear). Not needed for v0.1. Bezier curves with
  optimized control points cover the aesthetic space well enough.

---

## 4. Node Size and Shape

### Best-in-Class: yEd/yFiles Hierarchic + OGDF NodeRespecterLayout

yEd treats node geometry as real geometry throughout: alignment by top/center/bottom
border, border-to-border edge length measurement, and even node resizing in orthogonal
mode. OGDF's NodeRespecterLayout explicitly adapts forces to node sizes and prevents
edges from crossing non-incident nodes.

### Comparison

| Tool | Size-Aware Placement | Shape Support | Overlap Handling |
|------|---------------------|--------------|-----------------|
| **yEd/yFiles** | Full (alignment, border-to-border, resize) | Rectangles + shapes | Integrated minimum distances |
| **OGDF NodeRespecterLayout** | Full (forces adapt to sizes) | Bounding circles/boxes | Explicit edge-through-node avoidance |
| **OGDF FMMM** | BoundingCircle mode | Circles around boxes | Approximate |
| **ELK Layered** | Full (spacing options for thickness) | Rectangles | Spacing-based |
| **Graphviz** | neato/fdp have overlap-removal modes | Various shapes | Prism/Voronoi post-pass |
| **dagre** | Full (sep() uses half-widths + nodesep) | Rectangles only | Built-in via separation |
| **Cytoscape fCoSE** | Full (rectangular repulsion, nodeDimensionsIncludeLabels) | Rectangles | Force-based |
| **Cytoscape Cola VPSC** | Full (sweep-line + separation constraints) | Rectangles | Guaranteed no-overlap |
| **D3 forceCollide** | Circle approximation only | Circles only | Quadtree O(n log n) |
| **NetworkX** | None (point positions) | None | None |
| **igraph** | None (point positions) | None | None |
| **TikZ/pgf** | Partial (layer distances adjust for non-uniform sizes) | TeX boxes | Spacing-based |

### Key Algorithms

- **Gansner-Hu 2010 (PRISM)**: Proximity-preserving overlap removal. Builds proximity
  graph, applies stress-like scaling. Near-linear per iteration.
- **Nachmanson et al. 2017**: Delaunay-MST growth method, faster than PRISM empirically.
- **Cola VPSC**: Sweep-line detects overlapping rectangle pairs, generates separation
  constraints, active-set solver resolves all simultaneously. Guaranteed no-overlap for
  rectangles. Decoupled X/Y solving can be suboptimal.
- **Brandes-Kopf with variable sizes** (dagre): sep(u,v) = (width(u) + width(v))/2 +
  nodesep. Different separation for dummy vs real nodes.

### Recommendations for Dagua

**Already implemented (verify quality):**
- Dagua already has grid-based overlap detection and projection (O(N) expected).
  Verify that it handles highly non-uniform node sizes well (e.g., a 200px-wide label
  node next to a 20px default node).

**Implement (low priority):**
- Border-to-border edge length measurement in the attract loss. Currently uses
  center-to-center distance. Subtracting half-widths along the edge direction vector
  would give border-to-border distances, improving visual uniformity.
- Consider PRISM-style proximity preservation as an alternative post-pass for cases
  where the gradient-based overlap projection gets stuck.

**Skip:**
- Non-rectangular shapes in the layout algorithm. Rectangles (with label-inclusive sizing)
  cover 95%+ of graph layout use cases. Shape-aware rendering is a render-layer concern.

---

## 5. Text Placement and Sizing

### Best-in-Class: yEd/yFiles Hierarchic

yEd has integrated label handling: `Consider Node Labels` inflates placement geometry,
`Hierarchic` edge labeling reserves space during layout itself (not as a post-pass),
and routers can treat labels of fixed edges as obstacles.

### Comparison

| Tool | Node Label in Layout | Edge Label in Layout | Label Collision Avoidance |
|------|---------------------|---------------------|--------------------------|
| **yEd/yFiles** | Yes (Consider Node Labels) | Yes (Hierarchic mode reserves space) | Router treats labels as obstacles |
| **ELK** | Yes (spacing options) | Yes (edge-label placement strategies) | Configurable |
| **Graphviz** | Implicit (node size from label) | xlabel collision avoidance | Heuristic |
| **dagre** | Caller inflates node dims | Edge label proxy nodes in layout | Edge labels: yes; Node labels: no |
| **Cytoscape** | nodeDimensionsIncludeLabels | Post-layout only | Node labels only |
| **OGDF** | Weak (application responsibility) | Weak | Limited |
| **D3** | None | None | None |
| **NetworkX/igraph** | None | None | None |
| **TikZ/pgf** | TeX box sizing (automatic) | Path-label placement (syntactic) | Manual |

### Key Algorithms

- **Knuth-Plass 1981**: Optimal line-breaking for text paragraphs. O(n^2) in feasible
  breakpoints. Relevant for multi-line node labels.
- **Label placement as MIS** [Agarwal et al. 1998]: Model as maximum independent set in
  a conflict graph. NP-hard in general; O(log n) approximation for arbitrary rectangles,
  2-approximation for unit-height rectangles.
- **Simulated annealing label placement** [Christensen, Marks, Shieber 1995]: Strong
  practical baseline for point-feature labeling.
- **dagre edge label proxy nodes**: Inject a dummy node with label dimensions at the
  edge's labelRank. Participates in ranking and ordering, so edge labels get
  first-class positioning.

### Recommendations for Dagua

**Implement (high priority):**
- Node label inclusion in layout geometry. Dagua's `compute_node_sizes` in utils.py
  already measures text. Ensure the measured sizes (including multi-line labels) flow
  into the overlap and repulsion constraints correctly.
- Edge label positioning as a differentiable problem: treat each edge label as a small
  rectangle tethered to its edge midpoint, with repulsion from nodes and other labels.
  This is more flexible than dagre's proxy-node approach and fits dagua's framework.

**Implement (medium priority):**
- Graphviz-style xlabel collision avoidance: for external node labels, test candidate
  positions (8 compass directions) and pick the one with least overlap. Fast heuristic,
  big visual improvement.

**Skip:**
- Knuth-Plass line breaking. Dagua's text measurement already handles multi-line labels
  with simple wrapping. Optimal paragraph layout is overkill for graph node labels.
- Full MIS-based label optimization. The differentiable repulsion approach is more
  natural for dagua and produces good-enough results without combinatorial solvers.

---

## 6. Edge and Border Thickness

### Best-in-Class: yEd/yFiles Hierarchic

yEd is the only tool where `Consider Edge Thickness` is a documented, first-class layout
option that participates in minimum-distance computations.

### Comparison

| Tool | Thickness-Aware Layout | How |
|------|----------------------|-----|
| **yEd/yFiles** | Yes | Edge thickness in minimum-distance computation |
| **ELK** | Partial | Edge thickness and spacing options |
| **Graphviz** | Render-only | Stroke width doesn't affect layout |
| **OGDF** | Render-only | Not consistently documented |
| **dagre** | Manual | Caller inflates node dims to account for borders |
| **Cytoscape** | Partial | Compound padding styles |
| **D3** | Manual | Add border width to forceCollide radius |
| **NetworkX/igraph** | No | Not addressed |
| **TikZ/pgf** | No | Rendering-layer concern |

### Key Algorithms

- **Barequet, Goodrich, Riley 2004**: Linear-time algorithm for planar drawings with large
  vertices and thick edges. Theoretical, not widely implemented.
- **Fat edge routing** [Duncan, Efrat, Kobourov, Wenk]: Homotopic rerouting to maximize
  clearance. Exact for special subproblems.
- **Practical rule**: Inflate every routed polyline into a corridor. Node border thickness
  handled by inflating obstacle rectangles. Turns spacing into clearance constraints.

### Recommendations for Dagua

**Implement (low priority, easy win):**
- Include edge thickness in the edge-node crossing penalty. When computing distance from
  an edge path to a node, subtract half the edge stroke width from the clearance threshold.
  This is a one-line change to the loss function.
- Include node border thickness in `compute_node_sizes`. If border_width is 3px, add 3px
  to each dimension. Already trivial with the current architecture.

**Skip:**
- Dedicated thick-edge routing algorithms. The differentiable approach handles this
  naturally through clearance-aware loss terms.

---

## 7. Cycle Handling

### Best-in-Class: ELK Layered (9 strategies) / dagre (clean DFS + greedy)

ELK offers the most cycle-breaking strategies (9 documented options). dagre has the
cleanest implementation with DFS-based and greedy (Eades-Lin-Smyth) options plus
clean reversal/restoration bookkeeping.

### Comparison

| Tool | Cycle Breaking | Strategies | Back-Edge Handling |
|------|---------------|------------|-------------------|
| **ELK Layered** | 9 cycle-breaking strategies | DFS, greedy, interactive, model order, ... | Configurable |
| **Graphviz dot** | DFS-based greedy | Single heuristic | Reversed edges drawn as back-arcs |
| **dagre** | DFS (default) or greedy (Eades-Lin-Smyth) | 2 options | Reversed + restored after layout |
| **OGDF** | DfsAcyclicSubgraph, GreedyCycleRemoval | 2 options | Heuristic |
| **TikZ/pgf** | DFS, prioritized greedy, greedy, naive/random | 4+ options | Configurable heuristic choice |
| **yEd/yFiles** | Automatic detection + backloop routing | Integrated | Back edges as backloops |
| **D3/Cytoscape fCoSE** | N/A (force-directed, direction-agnostic) | N/A | N/A |
| **Cytoscape Cola** | Best-effort DAG flow | Soft constraint | Violated constraints |
| **igraph Sugiyama** | FAS reversal (weight-aware) | 1 heuristic | Lower-weight edges reversed |
| **NetworkX** | N/A (force/spectral) | N/A | N/A |

### Key Algorithms

- **DFS back-edge reversal**: O(|V|+|E|). Simple, fast. May reverse more edges than
  necessary.
- **Eades-Lin-Smyth 1993 (greedy)**: Linear time. Guarantees at least |A|/2 - |V|/6
  non-feedback arcs for connected digraphs without two-cycles.
- **Minimum feedback arc set**: NP-hard. All practical tools use heuristics.

### Recommendations for Dagua

**Already implemented (dagua/layout/cycle.py):**
- Dagua has cycle detection and temporary edge reversal. Verify it uses Eades-Lin-Smyth
  rather than naive DFS for better reversal minimization.

**Implement (medium priority):**
- Backloop routing for reversed edges (yEd's approach). Currently dagua routes back-edges
  as wide arcs that overlap nodes (known issue in TODO.md). Route them as loops that go
  backward along the flow direction, outside the main layout region.
- Weight-aware cycle breaking: prefer reversing lower-weight edges (igraph's approach).
  Dagua edges already have optional weights; use them in the FAS heuristic.

**Skip:**
- 9 cycle-breaking strategies (ELK). Two good ones (DFS + greedy) cover practical needs.
- Cycle rank optimization. No mainstream tool uses it; feedback-edge heuristics are the
  universal approach.

---

## 8. Pinning and General Constraints

### Best-in-Class: Cytoscape Cola (most flexible) / yEd Hierarchic (best UX)

Cola offers the most constraint types: alignment, gap, flow, non-overlap, and fixed
positions, all solved via VPSC. yEd has the best practical constraint UX: incremental
layout, from-sketch mode, swimlane hints, and port constraints as routing constraints.

### Comparison

| Tool | Hard Pin | Soft Anchor | Alignment | Ordering/Gap | Progressive Introduction |
|------|----------|-------------|-----------|-------------|------------------------|
| **Cola (VPSC)** | Fixed nodes | Penalty in Hessian | Axis + offsets | Gap constraints | unconstrIter -> userConstIter -> allConstIter |
| **yEd/yFiles** | Incremental mode | From-sketch | Swimlanes, layers | Port constraints | Sketch -> incremental |
| **ELK** | Interactive mode (current positions) | Partial | Layer constraints | 5-level port constraints | Interactive mode |
| **Graphviz** | pos + pin attribute | Partial | rank=same | Compass ports | N/A |
| **D3 d3-force** | fx/fy (hard) | forceX/Y (soft) | None built-in | None built-in | N/A |
| **Cytoscape fCoSE** | fixedNodeConstraint | N/A | alignmentConstraint | relativePlacementConstraint | Disables tiling when constraints present |
| **dagre** | None | None | None | edge.minlen only | N/A |
| **OGDF** | Module-specific | Limited | Limited | Limited | Not a focus |
| **igraph** | FR/KK: minx/maxx/miny/maxy boxes | Region constraints | Layer hints (Sugiyama) | N/A | N/A |
| **NetworkX** | spring_layout: pos + fixed | None | None | None | N/A |
| **TikZ/pgf** | desired at, anchor here | same layer | same layer | minimum layers | N/A |

### Key Algorithms

- **IPSEP-COLA** [Dwyer, Koren, Marriott 2006, 2009]: Constrained stress majorization
  with gradient projection. Separation constraints solved incrementally. The theoretical
  foundation for all constraint-based layout.
- **VPSC (Variable Placement with Separation Constraints)**: Active-set QP solver for
  axis-aligned separation constraints. Used by Cola, libcola, Adaptagrams.
- **Cola progressive introduction**: Phases of unconstrained -> user constraints ->
  all constraints (including non-overlap). Proven to improve convergence.

### Recommendations for Dagua

**Already implemented (dagua/flex.py, dagua/layout/constraints.py):**
- Pin constraint (hard via projection, soft via loss term).
- Align constraint (axis alignment as loss term).
- Flex system (soft/firm/locked values as differentiable preferences).

**Implement (high priority):**
- Progressive constraint introduction (Cola's curriculum). Currently all constraints
  are active from step 0. Instead: warmup with unconstrained attraction/repulsion ->
  introduce soft user constraints -> introduce hard constraints + overlap avoidance.
  This is the single highest-impact convergence improvement available.
- Gap/ordering constraints: "node A must be left of node B with gap >= 20px."
  Express as a differentiable hinge loss: `max(0, pos[A].x + gap - pos[B].x)`.

**Skip:**
- Full VPSC solver. Dagua's differentiable projection approach already handles separation
  constraints. Reimplementing an active-set QP solver would be architecture-alien.
- 5-level port constraints (ELK). Port support is a separate topic (section 10).

---

## 9. Overlap Avoidance

### Best-in-Class: Cola VPSC (guaranteed rectangles) / Dagua grid-based (scalable)

Cola's VPSC provides guaranteed no-overlap for rectangles via sweep-line + separation
constraints. Dagua's grid-based overlap detection and projection is already competitive
for scalability (O(N) expected) but should be verified for quality vs VPSC on small graphs.

### Comparison

| Tool | Method | Shape | Guarantee | Complexity |
|------|--------|-------|-----------|------------|
| **Cola VPSC** | Sweep-line + separation constraints | Rectangles | Hard guarantee | O(n log n) per pass |
| **Graphviz Prism** | Proximity graph + stress-like scaling | Rectangles | Proximity preservation | Near-linear per iteration |
| **Graphviz Voronoi** | Territorial separation | Points/circles | Heuristic | O(n log n) |
| **OGDF NodeRespecterLayout** | Force adaptation to sizes + edge-through-node avoidance | Bounding boxes | Minimizes violations | Iterative |
| **yEd/yFiles** | Minimum distance constraints | Rectangles | Reliable | Integrated |
| **ELK** | Dedicated overlap-removal algorithms | Rectangles | Configurable | Varies |
| **D3 forceCollide** | Quadtree circle-circle overlap | Circles only | Soft (force-based) | O(n log n) |
| **Cytoscape fCoSE** | Rectangular repulsion forces | Rectangles | Soft (force-based) | Iterative |
| **dagre** | Brandes-Kopf separation guarantees | Rectangles | Hard (by construction) | O(n) |
| **Dagua (current)** | Grid-based spatial hashing + projection | Rectangles | Hard (projection) | O(n) expected |
| **NetworkX** | Point repulsion (0.01 minimum) | Points | Minimal | N/A |
| **igraph** | Force repulsion | Points | Statistical | N/A |

### Key Algorithms

- **Gansner-Hu 2010 (PRISM)**: Proximity-preserving overlap removal. Build proximity
  graph from Delaunay triangulation, apply stress-like scaling. Preserves relative
  structure well. Near-linear per iteration.
- **Nachmanson et al. 2017**: Delaunay-MST growth, faster than PRISM empirically.
- **VPSC** [Dwyer et al.]: Sweep-line detects overlapping pairs, generates separation
  constraints, active-set solver resolves simultaneously. Decoupled X/Y.
- **Grid-based spatial hashing** (dagua): O(N) expected overlap detection via spatial
  hash grid. Projection pushes overlapping nodes apart along minimum-separation axis.

### Recommendations for Dagua

**Already implemented and competitive:**
- Grid-based overlap detection + projection in layout/projection.py. O(N) expected.
  This is already better than most competitors for large graphs.

**Implement (low priority, quality polish):**
- Verify behavior on pathological cases: highly non-uniform sizes, near-degenerate
  grid cells, one huge node among many small ones.
- Consider PRISM-style proximity preservation as a fallback metric: measure how well
  relative distances are preserved after overlap removal, report in eval.

**Skip:**
- VPSC solver. Dagua's projection-based approach is architecturally simpler and scales
  better. VPSC's guarantee advantage is mostly theoretical for practical graphs.
- Voronoi-based overlap removal. Older technique, superseded by PRISM and grid methods.

---

## 10. Port / Anchor Placement

### Best-in-Class: ELK Layered (algorithmic) / yEd/yFiles (practical)

ELK has 5 levels of port constraints (FREE, FIXED_SIDE, FIXED_ORDER, FIXED_RATIO,
FIXED_POS) that participate in crossing minimization, node ordering, and routing.
yEd has real ports in its graph model with port candidates, grid port styles, and
uniform group port assignment.

### Comparison

| Tool | Port Model | Constraint Levels | Affects Layout |
|------|-----------|-------------------|---------------|
| **ELK Layered** | 5-level port constraints | Free -> Fixed side -> Fixed order -> Fixed ratio -> Fixed pos | Yes (crossing min, ordering, routing) |
| **yEd/yFiles** | Real ports in graph model | Port candidates, grid styles, group ports | Yes (routing, ordering) |
| **Graphviz** | Compass ports (n/ne/e/se/s/sw/w/nw/c) | 9 fixed positions | Partial (edge endpoint only) |
| **OGDF** | Orthogonal node types | Limited | Partial |
| **TikZ/pgf** | Node anchors (rendering concept) | Manual | No (rendering only) |
| **Cytoscape** | Render-only endpoint styles | None in layout | No |
| **D3/dagre/NetworkX/igraph** | None | None | No |

### Key Algorithms

- **Schulze, Spoenemann, von Hanxleden 2014**: Extends Sugiyama to port constraints.
  Cycle breaking, crossing minimization, node ordering, and routing all become port-aware.
  Underlies KLay/ELK Layered. This is the definitive reference.
- **Hyperedge port assignment** [Fridman et al. 2022]: MIP-based exact model for ported
  data-flow diagrams. Feasible only for modest instances.
- **Eschbach, Guenther, Becker 2006**: Track assignment and crossing minimization for
  orthogonal hypergraphs. NP-hard.

### Recommendations for Dagua

**Implement (medium priority -- novel differentiable approach):**
- Port positions as learnable parameters per node. Each node has K ports at positions
  along its boundary; each edge is assigned to a specific port. Port assignment and
  position are jointly optimized with the layout.
- Differentiable port-position loss: penalize edge crossings at port sites, penalize
  ports on the wrong side relative to the edge direction, prefer uniform port spacing.
- This would be unique among all competitors. No JS library has layout-aware ports.
  ELK and yEd have them, but via discrete algorithms, not differentiable optimization.

**Implement (low priority):**
- Fixed-side port constraints (FIXED_SIDE level): "this edge must connect on the left
  side of this node." Express as a simple positional constraint on the edge endpoint.

**Skip:**
- Full 5-level ELK port constraint system. Overkill for dagua's current use cases.
- Hyperedge port assignment. MIP-based, doesn't fit the differentiable framework.

---

## 11. Multi-Edge Handling

### Best-in-Class: yEd/yFiles (automatic grouping + routing separation)

yEd handles multi-edges through automatic edge grouping, explicit port grouping, and
routers that separate parallel edges into distinct visual lanes. dagre has the cleanest
Sugiyama-based approach: simplify() merges multi-edges for layout, then restores them.

### Comparison

| Tool | Multi-Edge Model | Visual Separation | During Layout |
|------|-----------------|-------------------|---------------|
| **yEd/yFiles** | Automatic + explicit edge grouping, bus routing | Full (distinct lanes) | Yes |
| **ELK Layered** | Explicit multi-edge support | Yes | Yes |
| **dagre** | simplify() merges, weight=sum, minlen=max | Shared control points | Merged for layout |
| **Graphviz** | Supported | Spline separation | Partial |
| **OGDF** | Module-dependent (some forbid multi-edges) | Module-dependent | Varies |
| **D3** | Cumulative spring force | None (render-only arc offset) | Cumulative force |
| **Cytoscape** | Auto-curved rendering for parallel edges | Yes (bezier curves) | No (render-only) |
| **NetworkX/igraph** | Collapse to stronger connectivity | None | No |
| **TikZ/pgf** | Manual bend left/right | Manual | No |

### Key Algorithms

- **Edge concentration** [Newbery 1989, Sugiyama et al. 2016]: Replace parallel edges with
  shared segments or synthetic hub. Reduces clutter but changes geometry.
- **Geometric offsetting**: After routes are fixed, offset parallel edges by a constant
  spacing. Linear in bundle size.
- **SPQR P-node consistency**: In planar settings, parallel edges form P-nodes in the
  SPQR tree; preserving consistent order prevents unnecessary crossings.

### Recommendations for Dagua

**Implement (medium priority -- TODO item "Multi-edge spacing"):**
- Detect parallel edges (same source-target pair) and offset their Bezier control points
  perpendicular to the edge direction. Offset = edge_index_in_bundle * edge_sep.
  Simple geometric post-processing, big visual improvement.
- Edge weight aggregation for layout: when multiple edges connect the same pair, use
  the sum of weights for attraction force but maintain separate routing.

**Skip:**
- Edge concentration. Changes the graph semantics, not appropriate for a general layout
  engine.
- SPQR-based ordering. Dagua is not embedding-based.

---

## 12. Edge Bundling

### Best-in-Class: D3 curveBundle (hierarchical) / yEd Generic Edge Bundling (force-directed)

D3's curveBundle is the gold standard for hierarchical edge bundling visualization
(Holten 2006). yEd's Generic Edge Bundling implements force-directed bundling with
compatibility measure, strength, and quality controls.

### Comparison

| Tool | Bundling Algorithm | Type | Awareness of Layout |
|------|-------------------|------|-------------------|
| **D3 curveBundle** | Hierarchical bundling (Holten 2006) | Render-only (B-spline) | Requires pre-computed hierarchy |
| **yEd/yFiles** | Generic force-directed bundling + bus-style semantic grouping | Post-process | Ignores existing bends; no node-edge overlap avoidance |
| **Graphviz** | Not a core feature | N/A | N/A |
| **OGDF** | Not a prominent feature | N/A | N/A |
| **ELK** | Not documented as core feature | N/A | N/A |
| **All others** | Not supported | N/A | N/A |

### Key Algorithms

- **Holten 2006 (hierarchical edge bundling)**: Route edges through LCA paths in a
  hierarchy, render as B-splines. Linear per edge once hierarchy is known.
- **Holten-van Wijk 2009 (FDEB)**: Force-directed bundling without hierarchy. Attract
  nearby parallel segments. Iterative, naive cost is quadratic in edge count.
- **Pupyrev-Nachmanson-Kaufmann 2010**: Ink-minimizing bundling for layered layouts.
  Metro-line crossing minimization inside bundles.
- **Ordered bundles** [Pupyrev et al. 2011/2016]: Optimize ink + lengths + widths +
  separations, then solve metro-line crossing variant inside each bundle.

### Recommendations for Dagua

**Implement (medium priority -- TODO item "Edge bundling (FDEB)"):**
- Differentiable edge bundling as a loss term. For each pair of edges with compatible
  direction (angle < threshold) and proximity (midpoints within radius), add an
  attraction term between their Bezier control points. This is FDEB reimagined as a
  differentiable objective -- unique to dagua.
- Bundling strength as a user-controllable parameter (0 = no bundling, 1 = full).
- Important: also add a node-avoidance term so bundled edges don't cross through nodes
  (yEd's documented weakness).

**Implement (low priority):**
- Hierarchical bundling using dagua's cluster tree. Route edges through LCA paths and
  use cluster centroid as control point attractor. Natural extension of the cluster system.

**Skip:**
- Metro-line crossing minimization inside bundles. This is a discrete optimization
  problem that doesn't fit the differentiable framework.
- Ink minimization as a primary objective. Edge bundling should be aesthetic, not
  ink-optimal.

---

## 13. Label Collision Avoidance

### Best-in-Class: yEd/yFiles Hierarchic (integrated) / dagre (edge labels)

yEd's Hierarchic edge labeling reserves space during layout, and its routers treat labels
as obstacles. dagre handles edge labels as first-class layout participants via proxy nodes.

### Comparison

| Tool | Node Label Avoidance | Edge Label Avoidance | Labels as Obstacles |
|------|---------------------|---------------------|-------------------|
| **yEd/yFiles** | Consider Node Labels | Hierarchic (space reservation), Generic (post-process) | Router treats labels as obstacles |
| **ELK** | Spacing options | Edge-label placement strategies | Configurable |
| **Graphviz** | Implicit via node sizing | xlabel collision avoidance | Partial |
| **dagre** | Caller responsibility | Proxy nodes in layout (first-class) | Edge labels participate in ordering |
| **Cytoscape** | nodeDimensionsIncludeLabels | Post-layout only | Node labels only (inflated bbox) |
| **OGDF** | Limited | Limited | Limited |
| **D3** | None | None | None |
| **NetworkX/igraph** | None | None | None |
| **TikZ/pgf** | TeX box sizing (automatic) | Path-label syntax (manual) | Manual |

### Key Algorithms

- **Label placement as MIS** [Agarwal et al. 1998]: NP-hard. O(log n) approximation for
  arbitrary rectangles, 2-approximation for unit-height.
- **Simulated annealing** [Christensen, Marks, Shieber 1995]: Strong practical baseline.
- **Kakoulis-Tollis**: Graph-specific framework for node and edge labels with candidate
  positions and preferred positions.
- **Graphviz xlabel**: Collision avoidance for external labels via candidate-position
  selection and conflict resolution.

### Recommendations for Dagua

**Implement (high priority):**
- Edge label repulsion loss: treat each edge label as a small rectangle tethered to its
  edge midpoint. Add pairwise repulsion between labels and between labels and nodes.
  This is already in dagua's edges.py as `place_edge_labels` but verify it uses repulsion.
- Node label collision avoidance: inflate node bounding boxes to include labels before
  computing overlap constraints. Already partially implemented via compute_node_sizes;
  verify it handles all label positions (inside, above, below).

**Implement (medium priority):**
- Candidate-position heuristic for external labels (xlabel-style): try 8 positions around
  the node, score by overlap with existing labels and nodes, pick the best. Simple greedy
  pass after main layout.

**Skip:**
- Full MIS optimization. The differentiable repulsion approach is sufficient.
- Simulated annealing for labels. Adds a second optimization framework; dagua should use
  its own gradient-based approach.

---

## 14. Aspect Ratio at All Levels

### Best-in-Class: yEd/yFiles (multi-level) / ELK Layered (wrapping + min-width)

yEd controls aspect ratio at all levels: node dimensions, group bounds with insets, and
canvas aspect via orientation and compaction. ELK has dedicated aspect-ratio targets plus
graph wrapping for controlling the aspect ratio of very wide/tall layered layouts.

### Comparison

| Tool | Canvas Aspect | Node Aspect | Cluster Aspect | Control Method |
|------|-------------|------------|---------------|---------------|
| **yEd/yFiles** | Orientation + compaction + stacked placement | Alignment options | Group bounds + insets | Multi-level integrated |
| **ELK Layered** | aspectRatio target, wrapping, min-width layering | Spacing options | Content-determined | Direct parameter |
| **Graphviz** | ratio=compress/expand/auto | Implicit from label | Bottom-up from content | Post-scaling |
| **OGDF FMMM** | pageFormat/pageRatio, component rotation | Bounding geometry | Content-determined | Direct parameter |
| **Cytoscape** | fit + boundingBox + padding | Node dimensions | Compound padding | Scale-to-fit |
| **D3** | forceX/Y strength ratios | Circle radius | N/A | Force tuning |
| **dagre** | marginx/marginy | Node width/height | Border nodes | Margin only |
| **NetworkX** | scale/center | None | N/A | Post-normalization |
| **igraph** | Bounds (minx/maxx/miny/maxy) | None | N/A | Region constraint |
| **TikZ/pgf** | grow direction, level/sibling distance | TeX box | N/A | Declarative |

### Key Algorithms

- **ELK graph wrapping**: For very wide layered layouts, cut the layer sequence and stack
  segments to achieve a target aspect ratio. Dedicated wrapping strategies.
- **ELK min-width layering**: Controls layer width during assignment, trading height for
  width to hit aspect targets.
- **Graphviz ratio=compress**: Compute hierarchy, then scale/compress to fit page
  constraints. Post-hoc, but practical.
- **OGDF FMMM pageRatio**: Aspect-ratio target exposed directly inside the layout engine.
  Also rotates components to find a small bounding rectangle.
- **Balanced Aspect Ratio Trees** [JGAA]: Exact theorems for trees. Not applicable to
  general graphs.

### Recommendations for Dagua

**Implement (medium priority):**
- Canvas aspect ratio as a soft loss term: penalize bounding-box aspect ratios far from
  a target (default 16:9 or user-specified). Compute as
  `(actual_ratio - target_ratio)^2 * weight`. Light weight so it doesn't override
  structural quality.
- Direction switching heuristic: if the graph is much wider than tall, suggest LR layout;
  if taller than wide, suggest TB. Can be automatic with `direction='auto'`.

**Implement (low priority):**
- Cluster aspect ratio: after bottom-up bbox computation, add a mild loss term penalizing
  extremely elongated cluster boxes. This helps visual balance without constraining
  content.

**Skip:**
- Graph wrapping (ELK). Complex to implement, narrow use case (only helps extremely wide
  layered layouts). Direction switching is a simpler solution.
- Min-width layering. Dagua's continuous optimization can achieve similar effects through
  the aspect ratio loss term.

---

## 15. Incremental / Progressive Layout

### Best-in-Class: D3 d3-force (inherently progressive) / yEd Hierarchic (best mental-map)

D3's velocity Verlet simulation is inherently progressive -- every tick is renderable,
and users see the layout settle. yEd's incremental hierarchic mode preserves mental map
best: integrate new elements with minimal disruption to existing positions.

### Comparison

| Tool | Progressive Rendering | Incremental Updates | Mental-Map Preservation |
|------|---------------------|-------------------|----------------------|
| **D3 d3-force** | Native (every tick is a frame) | Add/remove nodes + reheat | Inherent (continuous) |
| **yEd/yFiles** | Not animated (batch) | Explicit incremental mode | Best (from-sketch, selected elements) |
| **ELK** | Not animated | Interactive mode (use current positions) | Anchoring to previous |
| **Cytoscape Cola** | animate: true, tick() for per-step | resume()/stop(), add/remove constraints | Good (stress majorization) |
| **Cytoscape fCoSE** | Quality modes (draft/default/proof) | randomize: false for incremental | initialEnergyOnIncremental |
| **Graphviz** | Batch only | N/A | N/A |
| **OGDF** | Batch only | Not a focus | N/A |
| **dagre** | Batch only | Full recompute | None |
| **NetworkX** | Batch only | Reuse pos as init | Minimal (pos seeding) |
| **igraph** | Batch only | Seed coordinates | Minimal (seed + modes) |
| **TikZ/pgf** | Batch only | Declarative hints | Minimal |

### Key Algorithms

- **IPSEP-COLA** [Dwyer et al. 2006]: Incremental constrained stress layout. Reuses
  previous layout, solves only updated constraint system. The theoretical foundation.
- **Crnovrsanin, Chu, Ma 2017**: Incremental FM3-style dynamic graph layout with local
  refinement. Refines high-energy regions instead of global restart.
- **D3 velocity Verlet**: alpha cooling (alphaDecay ~0.0228), ~300 ticks to convergence.
  Reheat on changes. Inherently progressive.
- **Mental-map preservation** [Archambault-Purchase 2013]: Empirical studies show mixed
  overall results but clear benefits for orientation tasks.

### Recommendations for Dagua

**Implement (high priority -- unique differentiable approach):**
- Progressive optimization rendering: emit intermediate positions every N steps during
  the Adam optimization loop. Dagua's architecture makes this trivial -- the position
  tensor is available at every step. Add a callback interface:
  `layout(g, config, on_step=callback)` that fires every K steps with current positions.
- Anchored incremental layout: when a graph is modified (nodes added/removed), initialize
  new nodes near their neighbors, pin existing nodes with a soft anchor loss, and run
  a short optimization. The anchor loss decays over steps, letting the layout settle
  into a new equilibrium while preserving mental map.

**Implement (medium priority):**
- Cola-style progressive constraint introduction (also mentioned in section 8). This
  is both a convergence strategy and a progressive-rendering feature: users see the
  unconstrained layout first, then watch constraints gradually shape it.

**Skip:**
- Full velocity Verlet simulation. Dagua's Adam optimizer with learning rate scheduling
  already provides smooth convergence. Reimplementing a force simulation would be
  architecturally redundant.

---

## Cross-Topic Interaction Matrix

Interactions between topics that affect implementation priority and design:

| Topic A | Topic B | Interaction | Impact |
|---------|---------|-------------|--------|
| **Skip connections (1)** | **Crossing reduction** | Dummy chains inflate graph for crossing minimization | High -- quality bottleneck |
| **Clustering (2)** | **Edge routing (3)** | Inter-cluster edges need boundary-aware routing | High -- visual quality |
| **Clustering (2)** | **Aspect ratio (14)** | Cluster boxes are bottom-up; aspect follows content | Medium |
| **Clustering (2)** | **Overlap (9)** | Cluster bbox propagation must include overlap padding | Medium |
| **Edge routing (3)** | **Node size (4)** | Routes must avoid non-incident node rectangles | High -- edge-node crossing is a top issue |
| **Edge routing (3)** | **Thickness (6)** | Thick edges need wider clearance corridors | Low |
| **Edge routing (3)** | **Bundling (12)** | Bundled edges share control points; routing quality affects bundle aesthetics | Medium |
| **Node size (4)** | **Text (5)** | Node size includes label text; affects all spacing | High -- foundational |
| **Node size (4)** | **Overlap (9)** | Non-uniform sizes stress overlap detection | Medium |
| **Cycles (7)** | **Skip connections (1)** | Reversed edges become skip connections in the other direction | Medium |
| **Cycles (7)** | **Edge routing (3)** | Back-edges need special routing (backloops) | Medium -- known issue |
| **Pinning (8)** | **Overlap (9)** | Pinned nodes can cause forced overlaps | Medium |
| **Pinning (8)** | **Incremental (15)** | Anchoring is the mechanism for incremental layout | High -- same system |
| **Ports (10)** | **Edge routing (3)** | Port positions constrain edge endpoints | High if ports are implemented |
| **Ports (10)** | **Multi-edge (11)** | Multiple edges to same port need fan-out | Medium |
| **Multi-edge (11)** | **Bundling (12)** | Parallel edges are a natural bundling target | Low |
| **Labels (13)** | **Overlap (9)** | Label boxes participate in overlap detection | High -- visual quality |
| **Labels (13)** | **Node size (4)** | Labels determine effective node size | High -- foundational |
| **Aspect ratio (14)** | **Clustering (2)** | Canvas aspect conflicts with natural cluster layout | Low |
| **Incremental (15)** | **Pinning (8)** | Incremental layout = soft pinning of existing nodes | High -- shared mechanism |

### Key Interaction Chains

1. **Text -> Node Size -> Overlap -> Label Collision**: Label text determines node size,
   which drives overlap detection, which must also consider label boxes. This chain must
   be consistent end-to-end.

2. **Cycles -> Skip Connections -> Crossing Reduction -> Edge Routing**: Cycle breaking
   creates reversed edges that become skip connections, which create dummy chains for
   crossing reduction, which determines edge routes. Quality cascades through this chain.

3. **Pinning -> Incremental -> Progressive**: Soft pinning is the mechanism for incremental
   layout, and progressive rendering is the UX manifestation. Same underlying system.

4. **Clustering -> Overlap -> Aspect Ratio**: Cluster bounding boxes must include overlap
   padding, and cluster aspect ratio emerges from content. Bottom-up propagation ties these.

---

## Prioritized Roadmap

### Phase 1: Convergence and Quality Foundation (high impact, moderate effort)

1. **Progressive constraint introduction** (Section 8)
   - Unconstrained warmup -> soft constraints -> hard constraints + overlap
   - Impact: convergence speed, layout quality, progressive rendering
   - Effort: Modify engine.py optimization loop scheduling
   - Dependencies: None
   - Interactions: Incremental layout (15), Pinning (8)

2. **Edge-node crossing penalty** (Sections 3, 4)
   - Differentiable distance-to-segment function as a loss term
   - Impact: Directly addresses top visual quality issue
   - Effort: New loss function in constraints.py
   - Dependencies: None (can use existing node positions and edge index)
   - Interactions: Edge routing (3), Node size (4), Thickness (6)

3. **Node label inclusion in layout geometry** (Sections 5, 13)
   - Verify compute_node_sizes includes all label positions in bounding box
   - Impact: Prevents label-node overlaps, foundational for label collision
   - Effort: Small verification + fix
   - Dependencies: None
   - Interactions: Node size (4), Overlap (9)

4. **Backloop routing for reversed edges** (Section 7)
   - Replace wide-arc back-edge routing with directed backloops
   - Impact: Directly addresses known TODO issue
   - Effort: Modify routing.py
   - Dependencies: Cycle handling (already implemented)
   - Interactions: Cycles (7), Skip connections (1)

### Phase 2: Differentiable Edge Optimization (high impact, high effort)

5. **Learnable Bezier control points** (Section 3)
   - Make edge control points learnable parameters with loss terms
   - Impact: Unique dagua advantage; replaces heuristic routing
   - Effort: Significant -- new optimization variables, new loss terms, convergence tuning
   - Dependencies: Edge-node crossing penalty (item 2) provides the key loss term
   - Interactions: Edge routing (3), Bundling (12), Multi-edge (11)

6. **Edge label positioning as differentiable problem** (Sections 5, 13)
   - Labels as rectangles tethered to edge midpoints with repulsion
   - Impact: First-class edge labels without proxy-node complexity
   - Effort: Moderate -- new loss term + post-layout refinement pass
   - Dependencies: Learnable Bezier (item 5) for edge midpoint accuracy
   - Interactions: Labels (13), Overlap (9)

7. **Differentiable edge bundling** (Section 12)
   - Attract compatible edge control points as a loss term
   - Impact: Novel approach; reduces clutter on dense graphs
   - Effort: Moderate -- compatibility detection + attraction loss
   - Dependencies: Learnable Bezier (item 5)
   - Interactions: Bundling (12), Multi-edge (11)

### Phase 3: Compound Graph and Constraint System (medium impact, moderate effort)

8. **Bottom-up cluster bbox propagation** (Section 2)
   - Recursive layout: leaf clusters first, then parent levels
   - Impact: Enables deep nesting (TODO item "Cluster hierarchy nesting")
   - Effort: Moderate -- recursive layout orchestration in engine.py
   - Dependencies: Existing cluster constraint in constraints.py
   - Interactions: Clustering (2), Overlap (9), Aspect ratio (14)

9. **Gap/ordering constraints** (Section 8)
   - Hinge loss for "A must be left of B with gap >= K"
   - Impact: Enables swimlane-style layouts and manual ordering
   - Effort: Small -- single new loss function
   - Dependencies: None
   - Interactions: Pinning (8), Clustering (2)

10. **Anchored incremental layout** (Section 15)
    - Soft-anchor existing nodes, short optimization for modified graphs
    - Impact: Mental-map preservation for dynamic graphs
    - Effort: Moderate -- initialization strategy + anchor loss decay
    - Dependencies: Pinning system (already implemented)
    - Interactions: Incremental (15), Pinning (8)

### Phase 4: Polish and Differentiation (lower impact, lower effort)

11. **Multi-edge spacing** (Section 11)
    - Perpendicular offset for parallel-edge Bezier control points
    - Impact: Visual improvement for multigraphs
    - Effort: Small -- geometric post-processing
    - Dependencies: Learnable Bezier (item 5) for best results

12. **Canvas aspect ratio loss** (Section 14)
    - Soft penalty for bounding-box aspect far from target
    - Impact: Better default canvas utilization
    - Effort: Small -- single loss term
    - Dependencies: None

13. **Border-to-border edge length** (Section 4)
    - Subtract half-widths along edge direction in attract loss
    - Impact: Visual uniformity improvement
    - Effort: Small -- modify attract constraint
    - Dependencies: None

14. **Candidate-position xlabel avoidance** (Section 13)
    - Try 8 positions for external labels, pick least-overlapping
    - Impact: Better external label placement
    - Effort: Small -- greedy post-pass
    - Dependencies: None

15. **Differentiable port positions** (Section 10)
    - Port positions as learnable parameters per node
    - Impact: Unique feature; no competitor does this differentiably
    - Effort: High -- new optimization variables, assignment problem
    - Dependencies: Learnable Bezier (item 5)

### What Dagua Should NOT Implement

- **Full visibility-graph routing**: Discrete algorithm, doesn't fit differentiable framework
- **VPSC solver**: Architecturally alien; projection-based approach is simpler and scales better
- **C-planarity testing**: Embedding-based; irrelevant to optimization-based layout
- **Metro-line crossing minimization**: Discrete optimization inside bundles
- **Edge concentration**: Changes graph semantics
- **9 cycle-breaking strategies**: Two good ones (DFS + greedy) are sufficient
- **Full Sugiyama dummy expansion**: Dagua's continuous optimization can achieve similar
  results through loss terms without the O(|V||E|) space overhead of explicit dummy chains
- **Velocity Verlet simulation**: Adam optimizer with learning rate scheduling already
  provides smooth convergence; reimplementing force simulation would be redundant
- **Multiple rendering-layer routing modes**: One good Bezier approach is sufficient;
  variety belongs in the renderer, not the layout engine

---

## References (Most Cited Across Reports)

- Sugiyama, Tagawa, Toda 1981: Layered drawing framework
- Gansner, Koutsofios, North, Vo 1993: Graphviz dot formulation
- Brandes, Kopf 2001: Fast coordinate assignment for hierarchical drawings
- Eiglsperger, Siebenhaller, Kaufmann 2005: Linear dummy representation
- Eades, Lin, Smyth 1993: Greedy cycle removal heuristic
- Sander 1996: Sugiyama for compound graphs
- Dwyer, Koren, Marriott 2006/2009: IPSEP-COLA, constrained stress majorization
- Gansner, Hu 2010: PRISM overlap removal
- Nachmanson et al. 2017: Delaunay-MST overlap removal
- Tamassia 1987: Minimum-bend orthogonal drawing
- Holten 2006: Hierarchical edge bundling
- Holten, van Wijk 2009: Force-directed edge bundling
- Schulze, Spoenemann, von Hanxleden 2014: Port constraints in layered layout
- Dogrusoz et al. 2009: CoSE compound graph layout
- Christensen, Marks, Shieber 1995: Simulated annealing label placement
- Archambault, Purchase 2013: Mental-map preservation studies
