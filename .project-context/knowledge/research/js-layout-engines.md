# JavaScript Graph Layout Engines: Algorithmic Deep Dive

**Libraries covered:** D3.js (d3-force), Cytoscape.js (CoSE/fCoSE/CoSE-Bilkent/Cola), dagre
**Date:** 2026-03-16
**Purpose:** Inform Dagua's differentiable layout engine design with concrete knowledge of
how the dominant JS layout engines handle 15 specific layout/aesthetic challenges.

---

## 1. Skip Connections (Long-Span Edges)

### D3 (d3-force)
**Algorithm:** No special handling. The spring model (forceLink) treats all edges identically
regardless of how many "layers" they span. The default strength formula
`1 / Math.min(count(source), count(target))` weakens links at high-degree nodes, which
indirectly affects skip connections at hub nodes but not skip connections in general.

**Parameters:** `link.distance()` (default 30), `link.strength()` (default degree-based).

**Tradeoffs:** Skip connections distort the layout equally to short connections. In a layered
DAG rendered with d3-force, a skip connection from layer 1 to layer 5 pulls those nodes
toward each other with the same spring stiffness as a layer-1-to-layer-2 edge (unless you
manually set per-edge distance/strength). This makes layered structure hard to preserve.

**Strengths:** Flexible -- you CAN set per-edge distance functions to make skip connections
longer. The force composition model means you can add a custom forceY that respects
topological order, partially recovering layered structure.

**Weaknesses:** No built-in concept of "rank" or "layer." Skip connections are fundamentally
invisible to the layout algorithm.

### Cytoscape.js

**fCoSE:** No special skip-connection handling. The spring embedder treats all edges via
`idealEdgeLength` (default 50) and `edgeElasticity` (default 0.45). Skip connections can be
given custom ideal lengths via a per-edge function, but this is manual. The spectral
initialization (power iteration on a sampled distance matrix via SVD) does encode graph
distance, so nodes separated by many hops will start farther apart. This implicitly helps.

**CoSE-Bilkent:** Same as fCoSE -- all edges get the same idealEdgeLength unless overridden
by a function. No rank-awareness.

**Cola:** The stress majorization objective explicitly minimizes
`sum w[i,j] * (d(i,j) - D[i,j])^2` where D is the shortest-path distance matrix. This
means skip connections (short graph distance) are naturally distinguished from multi-hop
paths (long graph distance). Cola computes all-pairs shortest paths via repeated Dijkstra.
This is the strongest implicit handling of skip connections among the three libraries.

**Dagre extension:** Wraps dagre (see below). Skip connections handled at the dagre level.

### dagre
**Algorithm:** The Sugiyama framework explicitly handles skip connections. During the
**normalize** phase, long edges (spanning multiple ranks) are replaced by chains of dummy
nodes, one per intermediate rank. This ensures the crossing minimization and coordinate
assignment phases see every edge as a single-rank edge.

**Edge labels** on long edges are placed on a specific dummy node whose rank equals the
edge's `labelRank`.

**Coordinate assignment** (Brandes-Kopf) treats dummy nodes as part of the ordering, so long
edges get smooth, non-overlapping routes through intermediate layers.

**Parameters:** `edge.minlen` (default 1) controls the minimum rank span. Setting minlen=3
forces a skip connection to span at least 3 layers.

**Tradeoffs:** Dummy nodes inflate the graph size. An edge spanning k ranks creates k-1 dummy
nodes. This is O(E * average_span) additional nodes, which affects performance on dense
graphs with many long-span edges.

**Strengths:** Best handling of skip connections among the three. The layered structure is
preserved by construction.

**Weaknesses:** Dummy nodes are rectangular and have zero width by default -- the edge route
is a sequence of straight segments through dummy centers, producing the staircase look that
Graphviz users expect but that can be aesthetically stiff.

---

## 2. Clustering / Compound Graphs

### D3 (d3-force)
**Algorithm:** No native compound graph support. The simulation operates on a flat node list.

**Common workarounds:**
- **forceX/forceY per cluster:** Assign each cluster a centroid and use per-node accessor
  functions: `forceX(d => clusterCenters[d.cluster].x).strength(0.3)`. This creates soft
  clustering but cluster boundaries are implicit, not drawn.
- **Hull rendering:** Compute convex hulls of cluster members post-layout for visual
  grouping. The layout doesn't know about hulls.
- **forceCluster (community addon):** Third-party force that attracts nodes to their cluster
  centroid. Not part of d3-force core.

**Tradeoffs:** Cluster forces compete with link forces and many-body repulsion. Tuning
strengths is a manual balancing act. No hard cluster containment -- nodes can drift out of
their cluster if link forces are strong enough.

**Strengths:** Complete flexibility. You can define clusters dynamically during simulation.

**Weaknesses:** No parent-child node hierarchy. No automatic sizing of cluster containers.
No nesting support.

### Cytoscape.js

**fCoSE:** Native compound graph support. Parent (compound) nodes automatically enclose their
children. The algorithm uses:
- `gravityCompound` (default 1.0): attraction force pulling compound node contents inward
- `gravityRangeCompound` (default 1.5): influence radius for compound gravity
- `nestingFactor` (default 0.1): multiplier for ideal edge length of inter-graph edges

The spectral phase computes bounding boxes from children's positions to place parents.
Constraint system supports fixed positions for compound nodes.

**CoSE-Bilkent:** Same compound support lineage (both from iVis Bilkent). CoSE was designed
for compound graphs from the start (Dogrusoz et al., Information Sciences 2009). The spring
embedder computes separate forces for intra-cluster and inter-cluster edges, with nesting
factor scaling edge lengths for cross-cluster connections.

**Cola:** Groups are represented as hierarchical rectangles with two dummy nodes (top-left,
bottom-right) per group. Dummy nodes connect to member nodes with compactness-weighted
edges. During stress majorization, group boundaries participate in the optimization.
`groupCompactness` parameter controls how tightly groups are packed. Supports nested groups.
Non-overlap constraints can be applied between groups.

**Built-in CoSE:** Cytoscape.js core includes a basic CoSE that handles compound graphs,
but fCoSE and CoSE-Bilkent are preferred for quality.

### dagre
**Algorithm:** Compound graph support via Sander's method. The implementation uses:

1. **Nesting graph augmentation:** DFS traversal creates border nodes (top/bottom pairs) for
   each compound node. Border nodes constrain the subgraph's rank span.

2. **Border segments:** Left (_bl) and right (_br) border nodes at each rank within a
   compound node, connected by weight-1 edges. These form the visual boundary.

3. **Parent dummy chains:** When edges cross compound boundaries, dummy nodes are inserted
   along the path through the lowest common ancestor (LCA), maintaining proper nesting.

4. **Constrained crossing reduction:** Forster's method ensures border nodes stay at the
   edges of their compound, and children remain between borders during ordering.

**Parameters:** `nodesep`, `ranksep` apply to compound internals. No explicit
compound-specific parameters.

**Tradeoffs:** Border nodes add significant overhead -- each compound node spanning k ranks
adds 2k border nodes plus connecting edges. Complex nesting (3+ levels) multiplies this.

**Strengths:** Rigorous containment guarantee. Children never escape their parent's bounds.
The Sugiyama framework naturally handles compound hierarchies because ranking and ordering
are constraint-based.

**Weaknesses:** No compound-specific spacing controls (like Cytoscape's nestingFactor).
Deeply nested graphs produce very sparse layouts because border node separations accumulate.

---

## 3. Edge Routing

### D3 (d3-force)
**Algorithm:** D3 does no edge routing. Edges are purely a rendering concern -- the layout
produces node positions only. The standard approach is to draw straight lines between node
centers using SVG `<line>` elements.

**Curve options (d3-shape):** When rendering, you can interpolate through control points:
- `curveBundle` (beta=0.85): Straightened B-spline, purpose-built for **hierarchical edge
  bundling** (Holten 2006). Beta=0 is straight, beta=1 is full B-spline.
- `curveBasis`: Cubic B-spline through control points
- `curveCardinal`: Cubic Hermite with adjustable tension
- `curveCatmullRom` (alpha=0.5): Centripetal Catmull-Rom, avoids self-intersection
- `curveLinear`: Polyline (straight segments)

**Tradeoffs:** Since routing is pure rendering, edges can cross nodes. There is no
obstacle-avoidance. For hierarchical bundling, you must compute a hierarchy separately
(d3-hierarchy) and route edges through LCA paths.

**Strengths:** Full control over curve aesthetics. The bundle curve is the gold standard for
edge bundling visualization.

**Weaknesses:** Zero awareness of node positions during routing. Edges through node
interiors are your problem.

### Cytoscape.js
**Built-in edge types:** Cytoscape.js supports multiple edge rendering styles independent
of layout:
- Straight, bezier (quadratic/cubic), unbundled-bezier
- Haystack (fast approximate for many edges)
- Taxi/segments (orthogonal routing)

**Cola (via WebCola GridRouter):** The most sophisticated edge routing available:
1. Constructs a grid of horizontal/vertical lines through the layout space
2. Creates routing graph vertices at grid intersections
3. Runs **Dijkstra's algorithm** with a bend penalty (cost 1000 per 90-degree turn) to find
   shortest orthogonal paths
4. **Nudging:** Separates parallel edge segments using VPSC constraint solving, maintaining
   consistent bundling via longest common subsequence ordering

This is the only JS library in this comparison with proper obstacle-avoiding edge routing.

**fCoSE/CoSE-Bilkent:** No edge routing. Edges are rendered as straight lines or bezier
curves between node boundaries. The layout doesn't consider edge paths.

### dagre
**Algorithm:** Edge routing is a natural byproduct of the Sugiyama framework:

1. **Dummy node insertion:** Long edges become chains of dummy nodes at each intermediate
   rank. Each dummy has (x,y) coordinates after layout.

2. **Control points:** During denormalization, dummy node positions become the edge's
   `points` array -- a sequence of (x,y) coordinates defining the edge path.

3. **Node intersection:** `intersectRect()` computes where edges meet node boundaries by
   projecting from the control point toward the node center and finding the rectangle edge
   intersection (compares `|dy|*w` vs `|dx|*h` to select top/bottom vs left/right face).

4. **Label placement:** One dummy node per labeled edge carries the label dimensions and
   position (labelpos: l/c/r).

**Parameters:** `edgesep` (default 10) controls spacing between parallel edge segments.

**Tradeoffs:** Routes are rectilinear sequences through rank layers. No spline fitting is
built in -- rendering libraries (D3, JointJS) add curves on top. The staircase pattern is
inherent to the dummy-node approach.

**Strengths:** Edges never cross node interiors (by construction -- dummy nodes participate
in ordering). Edge labels get first-class positioning.

**Weaknesses:** All routes are Manhattan-style. Aesthetic curves require post-processing.

---

## 4. Node Size/Shape Affecting Placement

### D3 (d3-force)
**Algorithm:** The simulation treats nodes as points by default. Node size enters only
through `forceCollide`:
- Each node gets a collision radius (default 1)
- The quadtree-based collision detection pushes overlapping circles apart
- Force: proportional to overlap distance, distributed by `r_j^2 / (r_i^2 + r_j^2)` (mass
  proportional to radius squared)

**Key limitation:** `forceLink` uses center-to-center distance, not edge-to-edge. A link
between a radius-50 node and a radius-5 node will have the same spring force as between two
radius-5 nodes at the same center distance, even though the visual gap is very different.

**forceCollide parameters:** radius (function or constant), strength (default 1), iterations
(default 1).

**Non-circular shapes:** Not supported by forceCollide. You would need a custom force that
computes rectangle-rectangle or arbitrary-shape overlap.

**Strengths:** Simple, fast. Collision detection is O(n log n) via quadtree.

**Weaknesses:** Only circles. No concept of width vs height. No label-inclusive sizing.

### Cytoscape.js

**fCoSE/CoSE-Bilkent:** Full non-uniform node dimensions. Node widths and heights are read
from the data model and affect repulsion forces. The `nodeRepulsion` parameter (default
4500) scales the repulsion force, which is computed based on actual node dimensions, not
just center distances.

`nodeDimensionsIncludeLabels: true` expands the effective node bounding box to include
label text, preventing label overlap during layout.

**Cola:** Nodes are rectangles with explicit width/height. The VPSC overlap avoidance
(`avoidOverlap: true`) generates separation constraints between all overlapping rectangle
pairs using a sweep-line algorithm. Gap = `(w_i + w_j)/2 + minSep`. This handles arbitrary
rectangular shapes properly.

### dagre
**Algorithm:** Node dimensions are first-class inputs:
- Each node has explicit `width` and `height` properties
- `nodesep` (default 50) is the minimum gap between node *edges* (not centers)
- The Brandes-Kopf coordinate assignment uses a `sep()` function that sums half-widths of
  adjacent nodes plus the nodesep constant
- Different separation rules for dummy vs real nodes (dummy uses edgesep=10 instead of
  nodesep=50)

**For horizontal layouts (LR/RL):** Width and height are swapped before layout, then
coordinates are transposed back.

**Strengths:** Node sizes directly participate in placement math. No post-hoc overlap
removal needed.

**Weaknesses:** Only rectangles. No support for ellipses, diamonds, or custom shapes in the
placement algorithm (though rendering can use any shape).

---

## 5. Text Placement and Sizing

### D3 (d3-force)
**Algorithm:** D3 has no text placement in the layout. Text is a rendering concern.
Typical approaches:
- SVG `<text>` elements centered on node coordinates
- `getBBox()` or `getComputedTextLength()` for text measurement after rendering
- Manual offset/anchor adjustments

There is no feedback loop from text size to layout. If a label is wider than the node's
collision radius, it will overlap neighbors.

**Strengths:** Full SVG text rendering capabilities (font selection, wrapping via
`<tspan>`, text-anchor).

**Weaknesses:** Layout-blind. Text measurement requires DOM access (not available in
headless/server environments). No text-aware collision avoidance.

### Cytoscape.js
**Algorithm:** `nodeDimensionsIncludeLabels: true` (supported by fCoSE, CoSE-Bilkent, Cola,
dagre extension) causes the layout to request node dimensions that include label text
bounding boxes. Cytoscape.js computes text dimensions internally using canvas
`measureText()`.

Label positions are a style property (inside node, outside on any side) and are computed
post-layout. The layout sees an inflated bounding box but doesn't know where the label
sits.

**Edge labels:** Positioned at midpoint, source-third, or target-third of the rendered edge
path. Not considered during layout computation.

**Strengths:** Text measurement is built into the platform. Label-inclusive sizing is a
single boolean.

**Weaknesses:** Label position doesn't feed back to layout -- only the aggregate bounding
box does. Two nodes with labels on opposite sides might still overlap labels even though
their inflated boxes don't overlap.

### dagre
**Algorithm:** Edge labels are first-class:
1. Edges with non-zero label dimensions get a **label proxy node** injected at layout time
2. The proxy participates in ranking (its rank becomes the label's vertical position)
3. During normalization, the dummy node carrying the label gets the label's width/height
4. `labelpos` (l/c/r) controls horizontal offset from the edge centerline

Node labels are not directly handled by dagre -- the caller must include label dimensions
in the node's width/height.

**Strengths:** Edge label placement is solved during layout, not as an afterthought.

**Weaknesses:** Node label placement is the caller's responsibility. No text measurement.

---

## 6. Edge/Border Thickness Affecting Layout

### D3 (d3-force)
Not addressed. Edge and border thickness are rendering-only properties. To account for
thick borders in collision avoidance, you would add border width to the forceCollide radius.
This is manual.

### Cytoscape.js
`nodeDimensionsIncludeLabels` inflates the bounding box for labels but not explicitly for
borders. However, Cytoscape's internal dimension calculation can include border width if
the element style's `width`/`height` properties account for it. The compound graph padding
styles (`padding`, `padding-top`, etc.) add space around children within compound nodes,
which effectively accounts for border thickness.

### dagre
Not addressed. Node width/height passed to dagre are expected to include any border
thickness. The caller is responsible for inflating dimensions. dagre's `nodesep` and
`edgesep` provide constant minimum spacing that can absorb border thickness if set
appropriately.

**Summary for all three:** No library has native border-thickness-aware layout. All require
the caller to inflate node dimensions to account for visual chrome.

---

## 7. Graph Cycles

### D3 (d3-force)
**Algorithm:** Cycles are irrelevant. The force simulation operates on undirected physics --
forceLink creates symmetric spring forces regardless of edge direction. A cycle A->B->C->A
is treated identically to three independent links. The simulation will converge to an
equilibrium regardless of cycle structure.

**Strengths:** No preprocessing needed. Works on any graph topology.

**Weaknesses:** Cannot produce layered/hierarchical layouts for cyclic graphs because there
is no concept of direction.

### Cytoscape.js
**fCoSE/CoSE-Bilkent:** Force-directed, direction-agnostic. Cycles handled naturally.

**Cola:** The `flow` option enables DAG-mode with separation constraints enforcing a
specified axis ordering. For cyclic graphs, this will produce violated constraints --
Cola's VPSC solver will satisfy as many as possible. The behavior is "best effort" rather
than failing.

**Dagre extension:** Passes through to dagre's cycle handling (see below).

### dagre
**Algorithm:** Explicit cycle removal as the **first** pipeline phase:

1. **DFS-based (default):** Depth-first search identifies back edges (edges pointing to a
   node on the current stack). These form the **feedback arc set** (FAS).

2. **Greedy (acyclicer='greedy'):** Uses a greedy heuristic for FAS selection, weighting
   edges to minimize total reversed weight. From the paper by Eades/Lin/Smyth.

3. **Reversal:** Each back edge is removed, its direction is reversed, and it's reinserted
   with `label.reversed = true` and `forwardName` stored for later restoration.

4. **Restoration:** After layout, `acyclic.undo()` reverses the edges back, restoring
   original direction. Edge control points are reversed (`reversePoints`) so arrows point
   correctly.

**Parameters:** `acyclicer: 'greedy'` enables the weighted heuristic. Default is DFS.

**Tradeoffs:** DFS-based FAS is fast (O(V+E)) but may reverse more edges than necessary.
Greedy FAS produces fewer reversals but is slower. Reversed edges appear as upward-pointing
arrows in TB layout, which can confuse users.

**Strengths:** Principled handling. The Sugiyama framework requires a DAG, and dagre's
acyclifier cleanly handles this requirement.

**Weaknesses:** The choice of which edges to reverse is heuristic. There is no guarantee
of minimum FAS (NP-hard in general).

---

## 8. Pinning / Constraints

### D3 (d3-force)
**Algorithm:** Two pinning properties per node:
- `node.fx`: If non-null, `node.x = node.fx` and `node.vx = 0` after every tick
- `node.fy`: If non-null, `node.y = node.fy` and `node.vy = 0` after every tick

This is a **hard constraint** -- forces still compute velocities, but the velocity is zeroed
and position is overwritten. The pinned node still contributes forces to other nodes.

**Use pattern:** During drag: set `fx/fy` to mouse position + `alphaTarget(0.3)` to reheat.
On release: set `fx = fy = null` to unpin.

**Custom constraints via forces:**
- `forceX(d => target(d)).strength(s)`: Soft x-position constraint, strength s in [0,1]
- `forceY(d => target(d)).strength(s)`: Soft y-position constraint
- `forceRadial(r, cx, cy).strength(s)`: Soft radial constraint
- Custom force functions can implement arbitrary constraints

**No alignment or gap constraints.** D3 has no built-in way to say "these nodes must be
horizontally aligned" or "node A must be left of node B."

### Cytoscape.js

**fCoSE (most capable):**
1. **fixedNodeConstraint:** Hard pin to exact (x,y). Applied by clamping position during
   both spectral and force-directed phases.
2. **alignmentConstraint:** Align node groups on horizontal or vertical axis. Must be
   specified in "most compact form" (no redundant entries).
3. **relativePlacementConstraint:** Specify that node A must be above/below/left of/right of
   node B with a minimum gap. Supports directional relationships.

When any constraint is present, tiling and component packing are disabled to avoid violating
constraints.

**Cola (most flexible):**
1. **Alignment constraints:** `{axis: 'x'|'y', offsets: [{node: id, offset: px}]}` --
   align nodes on an axis with optional offsets
2. **Gap constraints:** `{axis: 'x'|'y', left: nodeIdx, right: nodeIdx, gap: px}` --
   minimum separation between node pairs, formulated as inequality
   `left.position + gap <= right.position`
3. **Flow constraints:** DAG-style ordering with `{axis: 'y', minSeparation: px}` --
   edges flow in a specified direction
4. **Non-overlap:** `avoidOverlap: true` generates VPSC separation constraints dynamically
5. **Fixed nodes:** `node.fixed = true` locks position via penalty term in Hessian

Iteration phases: `unconstrIter` (no constraints), `userConstIter` (user constraints only),
`allConstIter` (user + non-overlap). Progressive constraint introduction improves
convergence.

**CoSE-Bilkent:** No explicit constraint system beyond compound graph containment.

### dagre
**No general constraint system.** The only "constraints" are:
- `edge.minlen`: Minimum rank separation between connected nodes
- Compound graph containment (border nodes enforce children stay inside parents)
- Implicit rank constraints from edge direction

You cannot pin a node to a specific position, align nodes, or specify gap constraints.

---

## 9. Overlap Avoidance

### D3 (d3-force)
**Algorithm:** `forceCollide` -- quadtree-based circle overlap detection.

1. Build quadtree from node positions + velocities (anticipated positions)
2. For each node, traverse quadtree to find overlapping circles
3. Compute separation force: `(r - d) / d * strength` where r = sum of radii, d = distance
4. Distribute impulse by squared-radius ratio (heavier nodes move less)
5. Repeat for `iterations` passes (default 1)

**Parameters:** radius (per-node function, default 1), strength (default 1), iterations
(default 1).

**Known issues:**
- Ignores fx/fy fixed positions (GitHub issue #213)
- Identical initial positions cause jitter (random jiggle applied)
- Circle-only -- no rectangle overlap avoidance

**Strengths:** O(n log n) per iteration via quadtree. Fast for thousands of nodes.

**Weaknesses:** Circle approximation wastes space for rectangular nodes. Multiple iterations
needed for dense graphs (strength=1, iterations=1 doesn't fully resolve overlaps in one
tick).

### Cytoscape.js

**fCoSE/CoSE-Bilkent:** Node repulsion force (`nodeRepulsion` default 4500) based on actual
rectangular dimensions. Not a separate overlap phase -- embedded in the force iteration.
`nodeSeparation` (fCoSE, default 75) adds space during spectral phase.

**Cola (VPSC):** The most rigorous approach:
1. **Sweep-line algorithm** detects overlapping rectangle pairs
2. Generates **separation constraints**: `center_i + (w_i + w_j)/2 + minSep <= center_j`
3. **VPSC solver** (active set method) resolves all constraints simultaneously
4. Two passes: X-axis then Y-axis (decoupled)
5. Convergence: cost change < 0.0001

VPSC handles rectangles natively, supports non-uniform sizes, and guarantees no overlap
(hard constraint, not just a force).

**Strengths of VPSC:** Guaranteed resolution. Handles arbitrary rectangles. Stable -- once
resolved, overlaps don't return.

**Weaknesses of VPSC:** Decoupled X/Y solving can produce suboptimal layouts (resolving X
may introduce Y overlaps and vice versa). Each iteration is O(n log n) for the sweep line
plus O(c) for the constraint solver where c = number of active constraints.

### dagre
**Built-in by construction.** The layered layout assigns discrete rank positions (y) and the
Brandes-Kopf algorithm assigns x-positions with minimum separation guarantees:
`sep(u,v) = (width(u) + width(v))/2 + nodesep`. Nodes cannot overlap because separation
is a hard constraint in coordinate assignment.

**No post-hoc overlap removal needed.**

---

## 10. Port / Anchor Placement

### D3 (d3-force)
**Not supported.** Edges connect node centers. Any port/anchor logic must be implemented in
the rendering layer by computing intersection points with node boundaries. No API for
declaring that edge E should connect to port P on node N.

### Cytoscape.js
**Edge endpoint styles:** `source-endpoint` and `target-endpoint` can be set to:
- `outside-to-node` (default): Closest point on node boundary
- `outside-to-line`: Closest point on node boundary to the edge line
- A specific point: `'50% 0%'` for top-center of node

These are rendering properties, not layout properties. The layout does not consider port
positions when computing node positions.

**No multi-port nodes.** You cannot declare that a node has named ports at specific positions
and route specific edges to specific ports.

### dagre
**No native port support.** Edge endpoints are computed post-layout by `intersectRect()`,
which finds the intersection of the edge's first/last control point direction with the
node's rectangular boundary. This always produces the closest boundary point -- there is
no way to specify "this edge connects to the left side" or "this edge connects to port 3."

**Workaround:** The Cytoscape.js dagre extension applies Cytoscape's endpoint styles after
dagre computes the layout, so you get Cytoscape's endpoint options but not true ports.

**Summary for all three:** None has true port/anchor placement where port positions
influence layout. All treat edge endpoints as a rendering/post-processing concern.

---

## 11. Multi-Edge Handling

### D3 (d3-force)
**Algorithm:** Multiple edges between the same pair of nodes are treated as independent
links by forceLink. Each contributes its own spring force. The effect is stronger attraction
between multi-connected nodes (cumulative spring force).

**Rendering:** All edges draw as straight lines between the same two centers, overlapping
visually. The standard workaround is to add curvature offset per edge (arc each parallel
edge by a different amount). This is purely a rendering technique.

**Strengths:** The force model naturally handles multi-edges (stronger connection = closer
placement).

**Weaknesses:** No visual separation without custom rendering code.

### Cytoscape.js
**Built-in multi-edge rendering:** Cytoscape.js detects parallel edges and automatically
curves them using bezier control points. The `curve-style: bezier` default handles this
for straight edges, and `curve-style: unbundled-bezier` allows per-edge control point
specification.

**Haystack edges:** For performance with many edges, `curve-style: haystack` draws
straight lines between random points on node boundaries. Parallel edges get different
random offsets, providing visual separation without bezier overhead.

**Layout impact:** Multi-edges are not distinguished during layout. fCoSE, CoSE-Bilkent,
and Cola all see them as independent edges with cumulative force.

### dagre
**Algorithm:** The utility function `simplify()` consolidates multi-edges before layout:
- Edges between the same (source, target) pair are merged
- The merged edge gets `weight = sum(weights)` and `minlen = max(minlens)`

After layout, the original multi-edges are restored. All parallel edges share the same
control points, so rendering must offset them.

**edgesep (default 10):** Controls minimum spacing between edge segments at the same rank.
This helps separate parallel edges that take different paths through dummy nodes, but
edges between the same direct neighbors share the same path.

**Strengths:** Weight aggregation means multi-edges correctly influence crossing
minimization (higher weight edges get priority in ordering).

**Weaknesses:** No built-in visual separation for direct parallel edges.

---

## 12. Edge Bundling

### D3 (d3-force)
**Algorithm:** D3 provides the rendering primitive (`curveBundle`) but not the bundling
algorithm. The full pipeline for hierarchical edge bundling (Holten 2006):

1. Compute a hierarchy (e.g., via `d3.hierarchy` or `d3.cluster`)
2. For each edge, find the path through the hierarchy (source -> LCA -> target)
3. Use the path nodes as control points
4. Render with `d3.line().curve(d3.curveBundle.beta(0.85))` where beta controls bundling
   tightness (0 = straight lines, 1 = full B-spline through hierarchy)

**Force-directed edge bundling (FDEB):** Not built in. Must be implemented separately.
The idea: treat edge segments as particles connected by springs, attract nearby parallel
segments. Some third-party implementations exist.

**Strengths:** `curveBundle` is elegant and produces beautiful results for hierarchical data.
Beta parameter gives fine control.

**Weaknesses:** Requires a hierarchy. Not applicable to general graphs without computing
a cluster hierarchy first. The layout itself has no bundling awareness.

### Cytoscape.js
**No built-in edge bundling.** There is no extension that performs force-directed or
hierarchical edge bundling. The closest features:
- `curve-style: haystack` for fast approximate edge drawing
- Manual control point specification via `unbundled-bezier`

### dagre
**No edge bundling.** Edges are routed individually through dummy nodes. Parallel edges
sharing similar paths will naturally be close together (within edgesep), providing
incidental visual grouping, but there is no deliberate bundling algorithm.

**Summary:** Only D3 offers edge bundling, and only as a rendering technique via
`curveBundle`, not as a layout feature.

---

## 13. Label Collision Avoidance

### D3 (d3-force)
**Not built in.** Approaches:
1. Add labels as additional nodes with forceCollide radius = label bounding box
2. Use a separate label placement library (e.g., d3fc-label-layout)
3. Post-layout greedy label placement (try positions, pick first non-overlapping)

None of these are part of d3-force core. Label collision avoidance is an unsolved problem
in the D3 ecosystem.

### Cytoscape.js
**Partial:** `nodeDimensionsIncludeLabels: true` inflates node bounding boxes to include
labels during layout. This prevents node+label overlap with other nodes+labels when:
- Labels are inside or directly adjacent to nodes
- The layout respects bounding boxes (fCoSE, CoSE-Bilkent, Cola do)

**Not handled:** Edge labels are placed post-layout and can collide with nodes or other edge
labels. No edge label collision avoidance exists.

**Strengths:** The `nodeDimensionsIncludeLabels` flag is simple and effective for the common
case.

**Weaknesses:** Only handles node labels. Edge labels, external labels, and labels on
adjacent sides of neighboring nodes can still collide.

### dagre
**Edge labels:** Handled during layout via label proxy nodes (see section 5). Edge labels
are placed at specific ranks with specific (x,y) coordinates. They participate in the
ordering phase, so they don't overlap other elements at the same rank.

**Node labels:** Not handled. The caller must include label size in node dimensions.

**Inter-label collisions:** Edge labels at different ranks can't collide (they're at
different y-coordinates). Edge labels at the same rank are separated by the ordering
algorithm.

**Strengths:** Best edge label collision avoidance of the three.

**Weaknesses:** No node label handling. No general label-label collision detection.

---

## 14. Aspect Ratio / Canvas Management

### D3 (d3-force)
**No built-in aspect ratio control.** The simulation expands into whatever space the forces
dictate. Common techniques:
- `forceCenter(width/2, height/2)` keeps the centroid at canvas center
- `forceX(width/2).strength(0.05)` + `forceY(height/2).strength(0.05)` provides weak
  centering with aspect ratio bias (set different strengths for X vs Y)
- Post-layout bounding box computation + SVG viewBox fitting
- `forceRadial` to constrain to a circular region

**forceCenter** translates all nodes to keep centroid at target, maintaining relative
positions. It differs from forceX/forceY which pull each node independently.

**Strengths:** Flexible viewport management via SVG viewBox.

**Weaknesses:** No way to say "fit this layout into a 16:9 rectangle." Must be done via
strength ratios or post-processing.

### Cytoscape.js
**Layout options:**
- `fit: true` (default for most layouts): Zoom/pan viewport to fit all elements after
  layout completes
- `padding: 30`: Padding around the fitted layout
- `boundingBox: {x1, y1, w, h}`: Constrain layout to a specific region
- `spacingFactor`: Multiplicative expansion/compression of the layout

**Compound graph padding:** `padding`, `padding-top`, etc. in element styles control
internal spacing of compound nodes.

**Strengths:** Automatic fit is seamless. BoundingBox provides hard aspect ratio control.

**Weaknesses:** `boundingBox` constrains but doesn't optimize for the given aspect ratio --
the layout is computed normally then scaled to fit. This can produce very different node
separations on different axes.

### dagre
**Output control:**
- `marginx` / `marginy` (default 0): Graph-level margins
- Post-layout, the graph object gets `width` and `height` properties representing the
  bounding box

**No aspect ratio control.** The layout expands to whatever the Sugiyama algorithm produces.
The Cytoscape.js dagre extension adds `fit: true` and `padding: 30` on top.

**No bounding box constraint.** Layout size is determined by node count, node sizes, and
separation parameters.

---

## 15. Progressive / Incremental Layout

### D3 (d3-force)
**Algorithm:** Inherently progressive. The simulation runs as an animation:

1. Initialize alpha=1
2. Each tick: `alpha += (alphaTarget - alpha) * alphaDecay`
3. Apply all forces, update velocities and positions
4. Fire "tick" event (render frame)
5. Stop when `alpha < alphaMin` (default 0.001)
6. Default ~300 ticks to convergence (alphaDecay ~0.0228)

**Incremental updates:**
- Add/remove nodes: call `simulation.nodes(newArray)`, forces re-initialize
- Reheat: `simulation.alpha(1).restart()` or `simulation.alphaTarget(0.3).restart()`
- The phyllotaxis spiral initialization means new nodes don't start at (0,0)

**Interaction during layout:**
- Drag: set `node.fx/fy` to mouse position + `alphaTarget(0.3)`
- The simulation continues running, other nodes adjust to the dragged node
- On release: clear `fx/fy`, set `alphaTarget(0)`

**Static mode:** `simulation.stop()` then `simulation.tick(300)` computes layout without
animation. Used for server-side rendering.

**Strengths:** The most natural progressive layout. Every tick is a renderable frame.
Users see the layout "settle" which provides intuitive feedback.

**Weaknesses:** Cannot guarantee how long convergence takes. The layout may oscillate if
forces are poorly balanced.

### Cytoscape.js

**fCoSE:** Three quality modes control cooling:
- `quality: 'draft'`: Spectral only, no force-directed refinement. Instant.
- `quality: 'default'`: Spectral + fast-cooling force-directed. Moderate.
- `quality: 'proof'`: Spectral + slow-cooling force-directed. Best quality.

`animate: true` renders intermediate states during force-directed phase.

**Incremental mode:** Set `randomize: false` to use current positions as starting point
for the force-directed phase (skip spectral). `initialEnergyOnIncremental` controls
starting temperature. Useful for small adjustments after adding/removing a few nodes.

**CoSE-Bilkent:** Same quality modes and incremental support via `randomize: false`.

**Cola:** Fully supports incremental layout:
- `resume()` / `stop()` for runtime control
- `alpha()` for temperature management
- `tick()` for per-step updates
- `maxSimulationTime: 4000` caps animation duration
- `animate: true` shows progressive refinement
- Adding/removing constraints triggers re-layout

Cola's `Descent.rungeKutta()` method enables smooth incremental updates via Runge-Kutta
integration.

### dagre
**Not progressive.** dagre computes the entire layout synchronously in one call:
`dagre.layout(g)`. There are no intermediate states, no animation, no incremental updates.

To "animate" a dagre layout, you must:
1. Compute the full layout
2. Animate node transitions from old positions to new positions (rendering concern)

**No incremental mode.** Adding a single node requires recomputing the entire layout.
The Sugiyama algorithm's phases (ranking, ordering, positioning) are interdependent and
cannot meaningfully produce partial results.

**Strengths:** Deterministic, fast, single-shot. No convergence uncertainty.

**Weaknesses:** No progressive refinement. No user interaction during layout. Cannot
efficiently handle dynamic graphs.

---

## Summary Matrix

| Topic | D3 (d3-force) | Cytoscape.js (best engine) | dagre |
|-------|---------------|---------------------------|-------|
| Skip connections | No special handling | Cola (stress majorization) | Dummy nodes (Sugiyama) |
| Compound graphs | Not supported | fCoSE, Cola (native) | Sander's method (border nodes) |
| Edge routing | Not supported (render only) | Cola (GridRouter + Dijkstra + VPSC nudge) | Implicit via dummy nodes |
| Node size in layout | Circle only (forceCollide) | Rectangle (VPSC/CoSE repulsion) | Rectangle (Brandes-Kopf sep) |
| Text placement | Not supported | nodeDimensionsIncludeLabels | Edge labels via proxy nodes |
| Border thickness | Manual | Compound padding styles | Manual (inflate dimensions) |
| Cycles | Irrelevant (undirected physics) | Cola: best-effort DAG flow | DFS or greedy FAS reversal |
| Pinning/constraints | fx/fy (hard), forceX/Y (soft) | fCoSE: 3 types; Cola: 4 types + VPSC | edge.minlen only |
| Overlap avoidance | forceCollide (circles, quadtree) | VPSC (rectangles, guaranteed) | Built-in (rank + sep) |
| Port placement | Not supported | Render-only (endpoint styles) | Not supported |
| Multi-edge | Cumulative spring force | Auto-curved rendering | simplify() + weight merge |
| Edge bundling | curveBundle (render only) | Not supported | Not supported |
| Label collision | Not supported | Node labels via inflated bbox | Edge labels via layout |
| Aspect ratio | forceX/Y strength ratios | fit + boundingBox + padding | Not supported |
| Progressive layout | Native (velocity Verlet anim) | fCoSE quality modes; Cola tick | Not supported (batch only) |

---

## Key Takeaways for Dagua

1. **Cola's VPSC** is the gold standard for constraint-based overlap avoidance with
   rectangles. Dagua's differentiable approach should match or exceed this guarantee.

2. **dagre's edge label handling** (proxy nodes during layout) is the best approach for
   layered graphs. Worth stealing for Dagua's Sugiyama-inspired mode.

3. **D3's force composition model** (named forces, alpha cooling, velocity Verlet) is the
   most flexible framework for general graphs. Dagua's loss function composition is the
   differentiable analog.

4. **fCoSE's spectral initialization** (SVD on sampled distance matrix + power iteration)
   is a smart way to get a good starting point fast. Dagua's hybrid init
   (topological sort + barycenter) serves the same purpose for DAGs.

5. **No JS library has true port support.** This is a gap Dagua could fill with
   differentiable port-position loss terms.

6. **Edge bundling is render-only everywhere.** Making bundling a layout objective
   (differentiable bundling loss) would be novel.

7. **Cola's progressive constraint introduction** (unconstrIter -> userConstIter ->
   allConstIter) is a proven strategy for convergence. Dagua could adopt a similar
   curriculum: unconstrained warmup -> soft constraints -> hard constraints.

---

## Feature Interactions: Where Independent Systems Collide

Most layout engines treat features as independent modules. The interesting (and broken)
behaviors emerge at the intersections. This section documents what actually happens when
features interact, whether there's dedicated logic, and what breaks.

### Clustering + Skip Connections in dagre Compound Graphs

**Interaction type:** Dedicated logic, but with significant cost.

When a skip connection crosses compound graph boundaries, dagre's pipeline creates
**parent dummy chains** -- dummy nodes at each intermediate rank that are properly parented
to the LCA's compound hierarchy. The algorithm:

1. Acyclifier runs first (on the original flat graph, before nesting augmentation)
2. Nesting graph augmentation creates border nodes for compounds
3. Long edges are normalized into dummy node chains
4. `parentDummyChains()` walks each dummy chain and finds the LCA of source/target,
   re-parenting each dummy to the appropriate compound level

This means a skip connection from a node inside compound A (rank 1) to a node inside
compound B (rank 5) will create 3 dummy nodes, each carefully assigned to the right
nesting level. The border segments of both compounds will adjust to enclose these dummies.

**What breaks:** The overhead compounds multiplicatively. A skip connection spanning k
ranks through n nesting levels creates k-1 dummy nodes, each needing LCA computation.
Deeply nested graphs with many long-span edges produce enormous intermediate graph sizes.
The ordering phase (crossing minimization) is the bottleneck, as it must respect compound
constraints from Forster's method while also handling the inflated node count.

**Dagua implication:** A differentiable approach avoids dummy nodes entirely. The loss
function for "edge should route through intermediate ranks" can be expressed as a penalty
term without materializing dummy nodes. This is a major structural advantage.

### Pinning (fx/fy) + Overlap Avoidance in D3 Force Simulation

**Interaction type:** Independent. And it breaks.

`forceCollide` computes collision forces and modifies `vx/vy` on ALL nodes, including
fixed ones. The tick function then zeroes `vx/vy` and overwrites `x/y` for fixed nodes.
The result:

1. Fixed node A overlaps with mobile node B
2. forceCollide pushes B away from A (modifies B.vx/vy)
3. forceCollide also pushes A away from B (modifies A.vx/vy) -- wasted computation
4. Tick: A.vx = 0, A.x = A.fx -- A doesn't move
5. B moves away, but only got half the displacement it needed (the other half was "given"
   to the immovable A)

**What breaks:** The force distribution uses the squared-radius ratio:
`r = rj^2 / (ri^2 + rj^2)`. If the fixed node has a large radius, it absorbs most of the
"push" but then doesn't move. The mobile node gets only a fraction of the needed
displacement. This means **fixed nodes with large collision radii create persistent overlaps
with nearby mobile nodes.** GitHub issue #213 documents this.

**Workaround:** Increase `forceCollide.iterations()` to 3-4 so mobile nodes get multiple
chances to escape. Or set fixed node radius very small (but this affects collision with
other mobile nodes).

**Dagua implication:** In a differentiable model, pinned nodes should have zero gradient
(position is a constant, not a parameter). Overlap loss between a pinned and a free node
should produce gradient only for the free node. D3's problem is that it distributes the
displacement bilaterally and then discards one side. A loss-function approach naturally
avoids this: `d(loss)/d(pos_free)` automatically captures the full gradient.

### Pinning + Clustering in Cytoscape fCoSE

**Interaction type:** Dedicated logic, with side effects.

When any `fixedNodeConstraint` is present, fCoSE **disables tiling and component packing**:
```
options.tile = false;
options.packComponents = false;
```

This means:
- Disconnected components are NOT packed together (they stay wherever they end up)
- Zero-degree nodes are NOT tiled into neat rows
- The spectral phase still runs, but component connection via dummy nodes may produce
  suboptimal initial positions for fixed nodes

Fixed constraints are enforced by **clamping positions** during the force-directed phase.
The node is held at its target (x,y) and participates in force computation but doesn't
move. Other forces (repulsion, edge attraction) still act on neighbors as if the fixed
node were free, but the fixed node's contribution is one-sided.

**Alignment + fixed interaction:** If a fixed node is also part of an alignment constraint,
fCoSE must satisfy both. The fixed position wins (it's a hard clamp), and the alignment
constraint is satisfied for the remaining non-fixed aligned nodes. If two fixed nodes are
in the same alignment group with incompatible positions, the constraint is silently
violated.

**What breaks:** The tiling/packing disable is a blunt instrument. If you fix one node in
a 100-node graph, ALL tiling and packing stops. Disconnected components float randomly.

**Dagua implication:** Constraint prioritization matters. Dagua's flex system (firm/soft
weights) handles this more gracefully than fCoSE's all-or-nothing approach. But the
interaction between `pin` constraints and `align` constraints needs explicit priority rules.

### Node Size + Edge Routing in dagre

**Interaction type:** Tightly integrated via the sep() function.

dagre's Brandes-Kopf coordinate assignment uses `sep(u, v)` to compute minimum horizontal
distance between adjacent nodes. The function:

```
sep = u.width/2 + (u.dummy ? edgeSep : nodeSep)/2
    + (v.dummy ? edgeSep : nodeSep)/2 + v.width/2
```

Dummy nodes (from edge normalization) have width 0 by default, except for label-carrying
dummies which have width = label width. So:

- **Real node next to real node:** gap = `w1/2 + nodeSep + w2/2`
- **Real node next to edge dummy:** gap = `w1/2 + nodeSep/2 + edgeSep/2 + 0`
- **Edge dummy next to edge dummy:** gap = `0 + edgeSep + 0`
- **Label dummy next to real node:** gap = `labelW/2 + edgeSep/2 + nodeSep/2 + w2/2`

**What breaks:** When node sizes vary dramatically (e.g., a tiny node next to a wide node),
the Brandes-Kopf four-alignment median can produce unexpected results. The algorithm
computes four coordinate assignments (UL, UR, DL, DR), picks the narrowest, then takes the
median of all four for each node. With extreme size variance, the four assignments can
disagree substantially, and the median may not be a good compromise. The dagre source
explicitly notes: "this implementation differs from BK due to a number of problems" with
the original algorithm when node sizes vary.

**Label position (l/c/r) adds additional deltas** to the sep function, shifting the effective
spacing asymmetrically. A right-positioned label adds label width to the right side's gap.

**Dagua implication:** The sep() interaction is clean but rigid. A differentiable approach
can express "minimum separation" as a loss term that handles continuous node sizes without
the discrete dummy-node machinery.

### Multi-Edges in D3 Link Force

**Interaction type:** Independent (no deduplication).

Each link in the `links` array is processed independently. For N parallel edges between
nodes A and B:

1. `count[A] += N` and `count[B] += N` -- parallel edges inflate degree
2. Default strength = `1 / Math.min(count[A], count[B])` -- weaker per-link
3. Bias = `count[A] / (count[A] + count[B])` -- same for all N parallel edges
4. Each link applies spring force independently: total force = N * single_link_force
5. But each link has strength scaled down by 1/N (due to inflated count)

Net effect: N parallel edges between (A,B) produce approximately the **same total force**
as a single edge, because the per-link strength scales as 1/N and there are N links.
The total spring force converges to `1 * (distance - target) / distance * alpha`.

**What breaks:** This is actually well-behaved mathematically. But the rendering problem
is severe: all N edges draw as identical overlapping straight lines. Without custom
rendering code to offset parallel edges, multi-edges are invisible.

**The strength formula interaction:** If A has 5 links to B and 1 link to C, then
count[A] = 6, count[B] = 5, count[C] = 1. Strength to B per link = 1/min(6,5) = 0.2,
total = 1.0. Strength to C = 1/min(6,1) = 1.0, total = 1.0. So the A-B and A-C
connections have equal total attractive force, which is correct. The degree-scaling
naturally normalizes multi-edges.

### Cycles + Compound Graphs in dagre

**Interaction type:** Sequential, with subtle ordering dependency.

The pipeline runs: `removeSelfEdges` -> `acyclic.run` -> `nestingGraph.run`. This means:

1. **Self-loops removed first:** Stored on node objects, completely invisible to subsequent
   phases.
2. **Cycle removal second:** The acyclifier (DFS or greedy) operates on the original flat
   graph. It does NOT see compound structure -- nesting hasn't been augmented yet.
3. **Nesting augmentation third:** Border nodes, nesting edges, and compound constraints
   are added to the already-acyclic graph.

**What breaks:** The acyclifier may reverse an edge that crosses a compound boundary. When
nesting augmentation later adds border nodes and parent dummy chains for this reversed edge,
the dummy chain routes from the "wrong" direction. The final `acyclic.undo()` reverses the
edge back and flips the control points, but the intermediate layout was computed with the
reversed direction.

For most graphs this is fine -- the reversed edge gets valid coordinates either way. But
for graphs where the cycle-causing edge is the only connection between two compound nodes,
reversing it changes which compound appears "above" the other in the layout. The user may
see a counterintuitive direction for the cycle edge, with the compound node ordering looking
wrong.

**No dedicated logic exists** for cycle-compound interaction. The phases are independent.

### Label Placement on Curved Edges in Cytoscape

**Interaction type:** Independent, with visual artifacts.

Cytoscape.js places edge labels at a specified position along the rendered edge path
(source, center, or target). For `curve-style: bezier` (the default for parallel edges),
the label position is computed on the bezier curve, not on the straight line between nodes.

**What breaks:**
- Labels on highly curved edges (e.g., one of several parallel edges) are placed far from
  the visual midpoint of the edge. The label "follows" the curve but may end up in
  unexpected positions.
- Edge label dimensions are NOT considered during layout. The layout computes node positions
  only; edge curves are determined post-layout; label positions are determined post-curve.
  This means labels can overlap with nodes that the layout didn't anticipate.
- For `curve-style: taxi` (orthogonal edges), labels are placed at the midpoint of a
  segment, which can be on a horizontal or vertical section depending on the route.

**Cytoscape provides no edge-label collision avoidance.** If two curved edges cross and
both have labels near the crossing point, labels will overlap.

**Dagua implication:** Edge label placement should be part of the optimization objective,
not a post-hoc calculation. A loss term for "edge label should not overlap any node or
other label" would solve this properly.

### Text Sizing + Node Size in All Three

**D3:** Completely independent. Text is rendered in SVG; node collision radii are set
independently. If you want text-inclusive sizing, you must:
1. Render text to an invisible SVG element
2. Call `getBBox()` or `getComputedTextLength()` to measure
3. Set `forceCollide.radius()` to a function that returns the measured size + padding
4. Re-initialize the simulation

This requires DOM access, making it impossible in headless/worker environments. There is
no `measureText()` equivalent in d3-force.

**Cytoscape.js:** Semi-integrated. `nodeDimensionsIncludeLabels: true` causes the layout
to call `node.layoutDimensions(options)` which returns width/height inclusive of label text.
Cytoscape measures text internally using `canvas.measureText()`, which works headlessly
(canvas can be offscreen).

But the integration is one-directional: text size affects node size, which affects layout.
There is no mechanism for layout to affect text size (e.g., shrinking text to fit a
constrained space). And the label position (top, bottom, center, etc.) is not considered --
only the bounding box is inflated.

**dagre:** Not integrated. Node dimensions are set by the caller before layout. dagre has
no text measurement capability. The Cytoscape-dagre extension bridges this by calling
`node.layoutDimensions()` (which includes labels if configured) and passing the result as
the node's width/height to dagre.

**What breaks across all three:** The feedback loop between text size and node size is
always open-loop. None of the libraries can:
- Automatically size nodes to fit their text
- Adjust font size to fit a target node size
- Balance text truncation against node spacing constraints

This is always delegated to the application layer.

**Dagua implication:** Text measurement in the loss function is possible via a precomputed
lookup (measure all labels once before layout, store as constants). The node-size loss can
then include a "node must be at least as large as its label" constraint. This closes the
loop that all JS engines leave open.

---

## Algorithmic References

Papers referenced by these implementations:

1. Gansner, Koutsofios, North, Vo. "A Technique for Drawing Directed Graphs." (1993)
   -- dagre's network simplex ranking
2. Brandes, Kopf. "Fast and Simple Horizontal Coordinate Assignment." (2001)
   -- dagre's coordinate assignment (with modifications)
3. Barth, Junger, Mutzel. "Simple and Efficient Bilayer Cross Counting." (2002)
   -- dagre's crossing count algorithm, O(|E| log |V|)
4. Sander. "Layout of Compound Directed Graphs." (1996)
   -- dagre's compound graph support
5. Forster. "A Fast and Simple Heuristic for Constrained Two-Level Crossing Reduction."
   (2004) -- dagre's constrained ordering for compound graphs
6. Dogrusoz, Giral, Cetintas, Civril, Demir. "A Layout Algorithm for Undirected Compound
   Graphs." Information Sciences (2009) -- CoSE algorithm (Cytoscape)
7. Dwyer, Koren, Marriott. "Constrained Graph Layout by Stress Majorization and Gradient
   Projection." (2005) -- WebCola/Cola
8. Holten. "Hierarchical Edge Bundles." (2006) -- D3's curveBundle
9. Eades, Lin, Smyth. "A Fast and Effective Heuristic for the Feedback Arc Set Problem."
   (1993) -- dagre's greedy acyclifier
10. Barnes, Hut. "A Hierarchical O(N log N) Force-Calculation Algorithm." (1986)
    -- D3's forceManyBody
