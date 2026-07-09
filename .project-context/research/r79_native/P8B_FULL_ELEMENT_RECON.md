# r80-S5: Full-element layout scope recon (edge routing, labels, node shape/size, eval coverage)

READ-ONLY recon. Repo: /home/jtaylor/.claude/worktrees/dagua-native (branch r79/native).
Zero files modified except this report.

## A. Edge routing today

**A1. Where routing happens: overwhelmingly RENDER-TIME, not layout-time.**

- `dagua/edges.py` module docstring (lines 1-7) says it plainly: "Heuristic bezier edge
  routing -- compute control points **after layout**."
- `route_edges()` (`dagua/edges.py:658`) takes already-final `positions` as input and is
  called from: the public draw path `dagua/__init__.py:167`, `dagua/render/mpl.py:1986`,
  `dagua/animation.py:1004/1329/1437/1602`, `dagua/graphviz_utils.py:348`,
  `dagua/reference_glossary.py:278`, and eval/report code
  (`dagua/eval/report.py:86,216`, `dagua/eval/visual_audit.py:240,609,667,1054`,
  `dagua/eval/benchmark.py:790`). In every call site, `positions` is a finished layout --
  routing never feeds back into node placement.
- The ONE genuinely layout-time, node-avoiding routing mechanism is Sugiyama's dummy-node
  chain: `_ExpandDummyNodes` (`dagua/layout/ops/sugiyama.py:6783`) inserts dummy nodes for
  multi-layer edges, and those dummy nodes participate in the same barycenter-ordering and
  coordinate-assignment machinery as real nodes, so routed paths inherently thread between
  node columns. `_BuildEdgeRoutes` (`dagua/layout/ops/sugiyama.py:7251`) then reconstructs
  polylines from the dummy-node chain positions -- this *is* layout-time routing, but it
  only exists for the Sugiyama/layered pipeline.
- A second, fully differentiable layout-time routing path exists (crossing-aware Bezier
  control-point optimizer with 6 loss terms: edge-edge crossing, edge-node crossing, port
  angular resolution, curvature consistency/penalty, edge-cluster crossing --
  `dagua/layout/edge_optimization.py:82-1617`, wrapped as ops
  `BezierControlPointOpt`/`ReconstructEdgeRoutes` in `dagua/layout/ops/edge_route.py:236-372`)
  but **it is orphaned**: registered for auto-discovery
  (`dagua/layout/ops/__init__.py:244`) and unit-tested
  (`tests/test_ops_edge_route.py`, `tests/test_edge_routing_config.py`), yet grep across
  every pipeline module in `dagua/layout/ops/pipelines/*.py` finds zero references to
  `BezierControlPointOpt`/`bezier_control_point_opt`/`ReconstructEdgeRoutes` -- no pipeline
  composes it into its op sequence. It never runs in production.

**A2. Routing styles and which pipelines use which.**

- Four routing modes exist on `BezierCurve.routing` (`dagua/edges.py:52`):
  `"straight"`, `"bezier"` (default), `"ortho"`, `"taxi"`. Constructors:
  `_compute_straight` (`edges.py:1063`), `_compute_ortho` (`edges.py:1073`),
  `_compute_taxi` (`edges.py:1097`), `_compute_bezier` (`edges.py:1172`).
- Mode selection is a **per-edge style attribute** (`EdgeStyle.routing`), resolved via
  `_get_route_edge_style` (`edges.py:863`) inside `route_edges()` -- i.e. it is a rendering
  choice attached to graph/edge styling, not something any layout pipeline picks based on
  its own topology (layered vs. force). Any pipeline's output can be rendered with any of
  the 4 modes; the pipeline identity (dummy-node-based Sugiyama vs. straight-line force
  layouts) only affects the *positions* the curves are drawn between, not which routing
  mode is chosen.
- Parallel edges: `route_edges()` detects repeated `(src, tgt)` pairs
  (`pair_counts`/`pair_ranks`, `edges.py:691-696`) and flips curvature sign on alternating
  duplicates (`edges.py:822-826`) to fan them across both sides of the chord, mirroring
  Graphviz's alternating spline lanes.

**A3. Node-edge overlap avoidance, self-loops, multi-edges.**

- Self-loops: handled explicitly. `route_edges()` special-cases `s == t`
  (`edges.py:769-771`) and dispatches to `_compute_self_loop_curve` (`edges.py:422`), which
  keeps the loop on the outward-facing side for the layout direction.
- Multi-edges: fanned via the curvature-sign flip above (`edges.py:822-826`); no separate
  "bundling" concept.
- Node-edge overlap avoidance: **only two mechanisms exist, and both are partial.**
  1. Cluster-aware deflection (render time): if a curve crosses a *foreign cluster* bbox,
     `_deflect_around_clusters` (`edges.py:1295`, called at `edges.py:838-845`) pushes
     control points around it. This is cluster-box avoidance, not general node-box
     avoidance.
  2. General edge-vs-any-node-bbox avoidance exists only inside the orphaned differentiable
     optimizer: `_edge_node_crossing_loss` (`dagua/layout/edge_optimization.py:1348`) --
     but as established in A1, no pipeline invokes it, so this avoidance never actually
     runs for a user-facing layout.
  - Port assignment is sorted by connected-node x-position (`out_edges`/`in_edges` sort in
    `route_edges()`, `edges.py:713-724`) to reduce crossings at the node itself, but this
    is a heuristic ordering, not geometric node-avoidance along the route body.

**A4. Does any eval metric measure edge-routing quality?**

Yes, metrics exist, but they are **computed and stored, then thrown away by the scoring
function that actually ranks/tunes layouts.** See section "measured vs unmeasured" table
and gap E-1 below. Concretely:
- `edge_node_crossing_count` (`dagua/metrics.py:1147`) -- edge-vs-node-bbox crossings on
  the *routed* curve (Tier 2).
- `edge_curvature_consistency` (`dagua/metrics.py:1247`) -- curvature CV across edges
  (closest thing to a "spline smoothness" proxy; there is no bend-count metric anywhere).
- `port_angular_resolution` (`dagua/metrics.py:1293`) -- min angle between incident edge
  tangents at actual curve ports (distinct from the node-position-only
  `angular_resolution` in `metrics.py:955`).
- `label_overlap_count` (`dagua/metrics.py:1195`) -- see B5.
- **No metric measures edge-edge crossings of the ROUTED path.** `sampled_crossing_rate`
  (`dagua/metrics.py:749`) -- the only edge-crossing metric wired into the composite score
  -- intersects straight segments between raw node centers
  (`p1 = pos[e1s[valid]]`, `edges.py`-independent, `metrics.py:833-836`), never the actual
  bezier/ortho/taxi geometry. So the benchmark's crossing number and the pixels a user
  actually sees can diverge arbitrarily once routing curves anything.
- Wiring: `full()` only computes the curve-aware Tier-2 metrics `if curves is not None`
  (`metrics.py:1821-1827`), and only `dagua/eval/benchmark.py:790` (the `compute_level ==
  "full"` branch, i.e. small graphs) passes `curves=route_edges(...)`. The `quick()` branch
  for large graphs (`benchmark.py:816-833`, used whenever N>2000 per `composite_large`'s
  docstring at `metrics.py:1566`) never computes `curves` at all, so **large-graph
  benchmark runs have zero edge-routing/label measurement, full stop.**
- Critically, even where computed, none of `edge_node_crossing_rate`,
  `edge_curvature_cv`, `port_angular_res_mean_deg`, `label_overlaps`,
  `label_node_overlaps` appear inside `composite()` (`metrics.py:1394-1459`),
  `composite_undirected()` (`metrics.py:1460-1512`), `composite_large()`
  (`metrics.py:1566-1602`), or `composite_strict()` (`metrics.py:1604-1616`) -- confirmed
  by reading every line of all four functions. The only place these values surface at all
  is the informational `anti_patterns` flag list
  (`_style_anti_patterns`, `dagua/eval/benchmark.py:841-855`, e.g. `label_collisions` at
  line 853), which is not a score and does not feed tuning/ranking.

## B. Labels and text

**B1. Node label text -> node extents -> confirmed used by layout.**

- `DaguaGraph.compute_node_sizes()` (`dagua/graph.py:933-1030`) calls `compute_node_size()`
  (`dagua/utils.py:1025`, which calls `measure_text`/`measure_text_fallback`,
  `utils.py:183,488`) per node, folding in padding, shape, font, wrap, min-size, and
  overflow policy, and writes the result into `self.node_sizes` (`graph.py:1106`).
- This is invoked **before** layout dispatch in both code paths of the engine:
  `graph.compute_node_sizes()` at `dagua/layout/engine.py:1086` (pipeline path) and
  `:1168` (legacy path). The resulting `graph.node_sizes` tensor is then passed as
  `node_sizes=` into every pipeline's `LayoutProblem` (e.g. `engine.py:1113`,
  `dagua/layout/ops/pipelines/sugiyama.py` builds `problem_node_sizes` at line ~303) and is
  read throughout `dagua/layout/ops/project.py` (overlap projection) and
  `dagua/layout/ops/loss_engine.py` (overlap/spacing losses) as `[N, 2]` width/height.
  Confirmed used, not decorative.

**B2. Edge labels: supported, placed at render time, post-placement collision search only.**

- `place_edge_labels()` (`dagua/edges.py:1368-1484`) computes label positions via a greedy
  search over `t_offset` / `label_side` / `perp_scale` candidates
  (`edges.py:1427-1478`), scoring each candidate by overlap area against node bboxes and
  previously-placed labels (`edges.py:1455-1466`), keeping the first zero-overlap hit or
  the least-bad candidate.
- This runs strictly **after** `route_edges()` on fixed positions/curves
  (call sites: `dagua/eval/benchmark.py:791`, mirrored in render/`mpl.py` and
  `visual_audit.py`). It never influences node placement -- it is pure post-placement
  layout of a 2D label given everything else frozen. "Overlap avoidance" here means
  "search a handful of discrete offsets," not an optimizer.

**B3. Cluster labels/titles: DO get layout-time space reservation (unlike edge labels).**

- `compute_cluster_placement_bbox()` (`dagua/layout/ops/cluster_geometry.py:206`) takes a
  `ClusterLabelMetrics` (measured label width/height, `cluster_geometry.py:166-180`) and a
  `label_band_pt` and reserves a top band (`label_band_y_extent` field,
  `cluster_geometry.py:196`) inside the cluster's placement-time bounding box -- this
  literally changes cluster footprint before/during layout, not just at render.
  `dagua/layout/ops/cluster_driver.py:315` constructs `ClusterLabelMetrics` for this path.
- `dagua/layout/engine.py:231` wires `label_band_pt=config.cluster_label_band_pt` into the
  cluster-aware driver, confirming this is a real, configurable layout-time knob.
- The actual drawn label position/anchor is still resolved at render time
  (`_cluster_label_anchor`, `_cluster_label_bounds`, `_resolve_cluster_label_collisions` in
  `dagua/render/mpl.py:4409,5194,5241`), but the *space* for it is reserved during layout
  -- a materially different (better) situation than edge labels.

**B4. Free-standing text/annotations: not supported.**

- No `add_annotation`/`annotate`/standalone-text API exists on `DaguaGraph`
  (checked `dagua/graph.py` -- no matches for `annotation`, `freestanding`, `free_text`).
  `dagua/render/text/` (`layout.py`, `paths.py`, `collection.py`, `decorations.py`)
  contains only the text-shaping/measurement engine used *by* node/edge/cluster labels --
  there is no user-facing way to drop an independent text block onto the canvas outside the
  node/edge/cluster label system.

**B5. Metric coverage for label overlap: computed, not scored (see A4 for the wiring gap).**

- `label_overlap_count()` (`dagua/metrics.py:1195-1244`) counts label-vs-node-bbox and
  label-vs-label bbox overlaps using `measure_text_fallback` bboxes.
  It requires `label_positions` and `edge_labels`, both only supplied on the "full" (small
  graph) benchmark branch (`dagua/eval/benchmark.py:791,799-800`); large graphs never
  compute it. Even when computed, it does not enter `composite()`/`composite_large()` --
  it only feeds the `label_collisions` anti-pattern flag
  (`dagua/eval/benchmark.py:853-854`), which is informational only.

## C. Node shape/size

**C1. 26 named shapes exist; layout sees bounding boxes ONLY, never true shape geometry.**

- `NODE_SHAPE_NAMES` (`dagua/styles.py:113-138`): rect, roundrect, arrow, ellipse, diamond,
  circle, triangle, hexagon, parallelogram, pentagon, octagon, star, cylinder, trapezoid,
  double_circle, cloud, stadium, semicircle(+4 directional variants), tab, note, document,
  box3d -- 26 total.
- Grepping `dagua/layout/ops/project.py` and `dagua/layout/ops/loss_engine.py` (the overlap
  projection and overlap/spacing loss modules) for the string `"shape"` turns up zero shape
  conditionals -- every hit is the unrelated docstring phrase "with shape `[N, 2]`" for
  tensor shapes. Every node, regardless of its `style.shape` value, is treated as an
  axis-aligned rectangle of `node_sizes[i] = (w, h)` for all overlap/spacing purposes.
  True per-shape geometry (ellipse projection, diamond boundary, polygon ray-intersection,
  concave star handling) exists **only** at render time in `_adjust_port_for_shape`
  (`dagua/edges.py:888-1019`, covering `rect`/`roundrect`, `ellipse`/`circle`, 5 semicircle
  variants via `ray_semicircle_intersection`, `diamond`, and 7 polygon shapes via
  `ray_polygon_intersection`) -- used purely to compute where an edge visually touches the
  node boundary for drawing, never fed back into placement.

**C2. No layout-time resizing/aspect adaptation; no port/anchor concept during layout.**

- `graph.compute_node_sizes()` runs exactly once, before pipeline dispatch
  (`dagua/layout/engine.py:1086`/`:1168`), producing a single fixed `node_sizes` tensor
  passed into the pipeline as a constant (`engine.py:1113`). No pipeline mutates node size
  in response to layout pressure (e.g. re-wrapping a long label if its node ends up in a
  tight neighborhood) -- sizing and placement are fully decoupled, one-directional
  (size -> placement, never placement -> size).
- "Ports" (where an edge attaches on a node boundary) have **zero presence** in the layout
  data model: grep of `dagua/layout/ops/state.py` and `dagua/layout/engine.py` for `port`
  turns up nothing but an unrelated code comment. Ports are computed entirely at render
  time inside `route_edges()`: `_compute_directional_ports` (`edges.py:500`) picks a side
  and distributes multiple edges along it, then `_adjust_port_for_shape`
  (`edges.py:888`) projects onto the true shape boundary. Layout-time node placement has no
  knowledge of, or influence from, where edges will eventually attach.

## D. External comparison surface

**D1. Competitor adapters capture positions only; routing/label output is discarded.**

- `CompetitorResult` (`dagua/eval/competitors/base.py:17-23`) is a 4-field dataclass:
  `name`, `pos: Optional[torch.Tensor] # [N, 2]`, `runtime_seconds`, `error`. Every one of
  the 20 adapters in `dagua/eval/competitors/*.py` (graphviz, dagre, elk, igraph, ogdf,
  networkx, cytoscape-fcose, gephi, fa2, linlog, sgd2(-multi), tsne, umap, neulay, classic,
  dagua-self) returns exactly this shape.
- Concretely wasted data, verified by reading the parsers:
  - **Graphviz** (`dagua/eval/competitors/graphviz_competitor.py`):
    `_parse_graphviz_json_positions()` (line 345) reads `dot -Tjson` output, but only
    extracts `obj["pos"]` for objects whose `name` starts with `"n"` (node objects,
    line 369-370) -- it never iterates the JSON's `edges` array at all, so Graphviz's
    native spline control points (`pos` on edge objects) and any label draw positions
    (`_ldraw_`) are silently dropped even though `dot -Tjson` already emits them.
  - **ELK** (`dagua/eval/competitors/elk_competitor.py`): builds an `edges` list for the
    ELK request payload (line 217-233) but the response parser (around line 178) only
    walks node `x`/`y`; ELK's native edge `sections` (bendpoint polylines, ELK's whole
    point of being a routing-aware engine) are never read back.
  - igraph/networkx/OGDF/etc. adapters (`igraph_competitor.py:94-99`,
    `ogdf_competitor.py:158-185`) likewise parse only a flat `[N,2]` position array from
    whatever JSON/text the underlying tool emits.
- Net effect: dagua's benchmark can never say "graphviz's routed edges cross node boxes X
  times but dagua's cross Y times" or compare label placement quality against a reference
  engine, because the reference engines' routing/label output is thrown away at parse time,
  not because the tools don't produce it.

**D2. Effort to capture full-drawing data per engine (estimate).**

- **Graphviz -- low.** `dot -Tjson` already includes edge `pos` (spline control points, one
  Bezier per edge with the `e,` prefix marking the arrow endpoint) and `_ldraw_`/`_hdraw_`
  label draw ops in the same JSON payload currently being parsed. This is a parser
  extension in `_parse_graphviz_json_positions` (or a sibling function), not a new
  integration. Rough effort: a few hours.
- **ELK -- low.** The JSON response already contains `sections[].bendPoints` per edge and
  label geometry when `elk.edgeLabels.inline`/label placement is requested; the adapter
  already round-trips JSON. Rough effort: a few hours.
- **igraph -- N/A for routing** (igraph's R/Python layout functions return positions only;
  it has no built-in edge router), but a "straight vs. dagua's own router applied to
  igraph's positions" comparison is already possible today via `route_edges()`.
- **OGDF -- medium.** OGDF's C++ layout API can emit `DPolyline` per edge if the harness
  requests it explicitly, but the current subprocess/JSON bridge
  (`ogdf_competitor.py`) would need a new field added to both the OGDF-side emitter script
  and the JSON schema. Rough effort: half a day, contingent on the existing OGDF bridge
  script's structure (not inspected in this pass).
- **Cytoscape/Gephi/force-directed Python libs (networkx, fa2, linlog, sgd2, tsne, umap,
  neulay) -- N/A.** These are node-placement-only algorithms with no native routing to
  capture; "full drawing quality" comparisons for these can only ever use dagua's own
  `route_edges()` applied post-hoc to their positions (which the benchmark can already do,
  and does for dagua itself).
- Whatever is captured, `CompetitorResult` and the frozen `positions/*.pt` store
  (per `CLAUDE.md`'s benchmark integrity notes) would need a new optional field/store (e.g.
  `routes: Optional[List[BezierCurve]]`) plus a matching entry in the fidelity-integrity
  validator so partial/missing routing data doesn't silently corrupt the store the way
  missing positions currently do.

## Measured vs unmeasured quality dimensions

| Dimension | Metric exists? | Computed in benchmark? | In composite score? |
|---|---|---|---|
| Node overlap (bbox) | Yes (`count_overlaps_detailed`, `metrics.py:492`) | Always | Yes (`composite`/`composite_large`) |
| Node-node crossing (straight-line) | Yes (`sampled_crossing_rate`, `metrics.py:749`) | Always | Yes |
| Node-node crossing (ROUTED path) | **No** | No | No |
| Edge-node bbox crossing (routed) | Yes (`edge_node_crossing_count`, `metrics.py:1147`) | Only `compute_level="full"` (small N) | **No** |
| Edge curvature / smoothness proxy | Yes (`edge_curvature_consistency`, `metrics.py:1247`) | Only "full" | **No** |
| Bend count | **No metric anywhere** | -- | -- |
| Port angular resolution (real curve tangents) | Yes (`port_angular_resolution`, `metrics.py:1293`) | Only "full" | **No** |
| Node-angle angular resolution (straight edges) | Yes (`angular_resolution`, `metrics.py:955`) | Always | Yes |
| Edge label overlap (label-node, label-label) | Yes (`label_overlap_count`, `metrics.py:1195`) | Only "full" | **No** (anti-pattern flag only) |
| Cluster label space reservation | N/A (layout-time geometry, not a scored metric) | Always (when clusters present) | Indirectly via `cluster_mean_sep_ratio` only |
| Self-loop / multi-edge correctness | **No metric** | -- | -- |
| Reference-engine routing/label fidelity | **No adapter captures it** | No | No |

## E. Gap ranking (impact x feasibility)

1. **Wire the already-computed curve-aware metrics into the composite score
   (POST-PLACEMENT-adjacent measurement gap, not a layout gap).**
   Highest feasibility: `edge_node_crossing_rate`, `edge_curvature_cv`,
   `port_angular_res_mean_deg`, `label_overlaps`/`label_node_overlaps` are already computed
   by `full()` (`metrics.py:1821-1827`) whenever `compute_level="full"` -- they just never
   reach `composite()`. Sketch: add 4 small weighted terms to `composite()` (mirroring the
   existing `composite_undirected` pattern) and extend `composite_large` to compute a
   reduced curves-based subset for large N too (currently `quick()` skips `route_edges()`
   entirely, `benchmark.py:816-823`). This alone would make routing/label quality visible
   to every tuning loop that reads `composite_score` without touching the placement
   algorithm at all.

2. **Wire `BezierControlPointOpt`/`ReconstructEdgeRoutes` into at least one force-directed
   pipeline (POST-PLACEMENT, orthogonal to node-placement benchmark).**
   High feasibility (code exists, tested, just uncomposed -- `dagua/layout/ops/edge_route.py`,
   `tests/test_ops_edge_route.py`): compose these two ops at the tail of e.g. the
   `dagua_native`/`fr`/`kk` pipelines behind an opt-in `LayoutConfig` flag (e.g.
   `edge_opt_steps > 0` already exists as a config field per
   `edge_optimization.py:166-172`, it's just never reached). This gives real edge-edge and
   edge-node crossing avoidance for the majority of pipelines that currently get zero
   avoidance beyond cluster deflection. Does not touch node placement -- purely a
   post-placement pass, so it is safe to add without perturbing the node-placement
   benchmark.

3. **Routed-path-aware crossing metric (replace/augment `sampled_crossing_rate`'s
   straight-line assumption) (measurement gap, POST-PLACEMENT).**
   Medium feasibility: extend `sampled_crossing_rate` (or add a sibling) to sample points
   along `curves` (already available wherever `route_edges()` has run) instead of node
   centers, using the same `segments_intersect` primitive (`metrics.py:146`) on route
   polyline segments. Needed before gap 1's crossing term can be trusted for anything but
   straight-line pipelines.

4. **Node-edge overlap avoidance in the render-time `route_edges()` heuristic itself
   (POST-PLACEMENT, medium impact/medium feasibility).**
   Currently only cluster bboxes get deflection (`_deflect_around_clusters`,
   `edges.py:1295`); ordinary node bboxes get none outside the orphaned optimizer. Sketch:
   extend `_deflect_around_clusters`'s bbox-avoidance logic (or reuse it generically) to
   also test against nearby non-endpoint node bboxes, falling back to the differentiable
   optimizer (gap 2) for stubborn cases. Medium feasibility because the existing deflection
   code is a reasonable template but wasn't designed for O(edges x nodes) traversal on
   larger graphs.

5. **Edge-label placement as a true optimizer rather than a discrete-candidate greedy
   search (POST-PLACEMENT, medium impact/medium feasibility).**
   `place_edge_labels()` (`edges.py:1368`) already frames the problem correctly (overlap
   scoring against node bboxes + previously-placed labels) but only searches 5 t-offsets x
   a handful of side/perp-scale combinations. Sketch: promote to a small gradient-free or
   differentiable local search per label (or extend the edge-optimization framework in gap
   2 to include label anchor as a joint variable), so labels can dodge dense
   neighborhoods rather than just the discrete candidate grid.

6. **General shape-aware overlap/spacing in the LAYOUT-TIME loss functions (LAYOUT
   problem, low feasibility, uncertain impact).**
   `project.py`/`loss_engine.py` treat every shape as its bbox (C1). True geometry-aware
   overlap (e.g. two touching diamonds whose bboxes overlap but silhouettes don't) would
   require rewriting the core overlap-projection/loss primitives to be shape-parametric,
   touching the most performance-critical, most-tested part of the codebase. This is the
   only item in this list that is a genuine LAYOUT (node-position) change rather than
   post-placement, so it is explicitly the one that WOULD interact with the existing
   native-placement benchmark and fidelity gates -- flagged as lowest priority for that
   reason alone, independent of its geometric value.

7. **Cluster-title anchor collision resolution -- already exists
   (`_resolve_cluster_label_collisions`, `mpl.py:5241`) but is render-only; layout-time
   `label_band_pt` reservation (B3) already covers the main risk (labels overlapping their
   own cluster's members). Low priority -- smallest remaining gap in the label story.**

### Layout vs. post-placement classification (for the benchmark-safety question)

- **Post-placement (safe to improve without touching the node-placement benchmark):**
  gaps 1-5, plus D1/D2 (competitor capture), B2 (edge label search), all of section A's
  routing gaps except the Sugiyama dummy-node mechanism itself.
- **Layout (would interact with node-placement benchmark/fidelity gates):** gap 6 only
  (shape-aware overlap/spacing losses). Node-size determination (B1/C2) is technically
  layout-time-consumed but is a one-shot pre-layout input, not something this sprint would
  need to change to close any of the ranked gaps.
