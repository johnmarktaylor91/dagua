# Sprint Design: Clusters as First-Class Placement Primitives

**Date:** 2026-04-27
**Status:** Architect briefing for multi-round implementation sprint
**Author:** Investigation pass (Opus, no code changes)
**Standing directive:** clusters participate in placement as nodes at the same hierarchical layer
(`/home/jtaylor/.claude/projects/-home-jtaylor-projects-dagua/memory/project_clusters_as_nodes.md`).

This document is the design brief for an implementation sprint. It maps the current cluster pipeline
end-to-end (layout, geometry, render), compares to graphviz `dot`, proposes the new architecture
required by the standing directive, and breaks the work into discrete codex-dispatchable phases.

The four observed defects (H4, H5, R9/R10, intra-cluster spacing) are *symptoms* of a single root
cause: clusters are currently treated as soft loss terms over leaf positions, never as primitives
placed against their siblings.

---

## 1. Layout side — current cluster handling

### 1.1 Cluster definitions enter through `DaguaGraph`

- `dagua/graph.py:856` `DaguaGraph.add_cluster(name, members, parent=...)`.
- Storage on the graph (`dagua/graph.py:96-100`):
  - `clusters: Dict[str, List[int]]` — name → flat list of leaf node indices (already flattened
    when a dict-of-dicts is passed in).
  - `cluster_parents: Dict[str, Optional[str]]` — name → parent cluster name (None for roots).
  - `cluster_styles`, `cluster_labels` — render-only metadata.
- A nested cluster's flat member list contains *all* descendants. When `add_cluster` is called with
  a dict-of-dicts, sibling child clusters are auto-created and linked via `cluster_parents`.
- `leaf_cluster_members(name)` in `graph.py:1475` returns the deduplicated leaf indices (flat).

### 1.2 Cluster info is forwarded to the headless layout engine

- `LayoutProblem` (in `dagua/layout/ops/state.py:111`) carries `clusters: Optional[Dict]` and
  `cluster_parents: Optional[Dict]`. These are immutable inputs to all ops.
- The engine entry (`dagua/layout/engine.py:980-1078`) forwards `graph.clusters` and
  `graph.cluster_parents` into the chosen pipeline as kwargs, then into `LayoutProblem`.

### 1.3 Cluster awareness today = three soft loss terms + one degenerate-grid post-process

| Loss | File | Math | Weight |
|---|---|---|---|
| `cluster_compactness_loss` | `constraints.py:1394` | sum over clusters of mean squared distance from each member to cluster centroid | `w_cluster=1.0` |
| `cluster_separation_loss` | `constraints.py:1422` | for sibling cluster pairs (same parent), penalty = (overlap area of inflated bboxes); **uses bbox built from `pos[members].min/.max +/- max-node-size/2 + padding=10`** | `w_cluster_sep=0.5*w_cluster` |
| `cluster_containment_loss` | `constraints.py:1458` | for each (child, parent) pair, ReLU-squared bbox-edge violation, padding=18 | `w_cluster_contain=2.0` |

These are wired in two places:
- Native engine: `engine.py:1909-1948` adds them to the `loss_fns` list when clusters present.
- Ops pipeline: `loss_engine.py:1080-1230` registers `ClusterCompactnessLoss`, `ClusterSeparationLoss`,
  `ClusterContainmentLoss`. Cached via `_get_cluster_cache` (a per-pipeline `_ClusterCache` keyed by
  `id(problem.clusters), id(problem.cluster_parents), node_sizes_shape`). The cache holds
  flattened `(cluster_id, node_idx)` index tensors, sibling pairs, containment pairs, max member
  size per cluster.

There is also a **post-process ops-only opt-in**: `ClusterGridArrange`
(`dagua/layout/ops/cluster_arrange.py`) — only fires on degenerate aspect ratios when clusters
collapsed into vertical stacks. Default OFF. Used only by the legacy native pipeline.

### 1.4 Per-algorithm differences

`grep ClusterCompactness /pipelines/*.py` returns only `dagua_native_legacy.py`. **None of the
classical pipelines (`fr`, `kk`, `fa2`, `sfdp`, `sugiyama`, `stress_sgd`, etc.) wire any cluster
loss.** The current cluster awareness lives only in:
1. The native multilevel/direct engine (`engine.py:_layout_inner`)
2. The legacy native pipeline (`dagua_native_legacy.py`)

This means **picking `algorithm="fr"` or any classical algorithm produces a layout that is fully
cluster-blind** — clusters then become a render-time wrapping of whatever positions came back.
This is the proximate cause of the observed sibling-cluster overlap on `cluster_showcase`: pure FR
(or whichever is dispatched) doesn't know clusters exist; the only thing keeping member nodes
nearby is the (unwired-in-classical-pipelines) compactness loss.

### 1.5 Nested clusters

Recursion is shallow:
- The `cluster_compactness_loss` treats every cluster as a flat set of leaves (no hierarchy
  awareness — `inner_a` is a member of both `cluster_inner` AND `cluster_outer`, so both forces
  pull it).
- `cluster_separation_loss` only repels siblings (same parent), as desired.
- `cluster_containment_loss` is the only force that knows about hierarchy structure: `child_bbox ⊂
  parent_bbox + padding`.

There is **no recursive placement op** — the leaf positions are optimized as a single flat tensor,
with hierarchy expressed only through these three penalty terms. The "cluster bbox" is implicit:
recomputed under no_grad at every loss step from `pos[members].min/.max`.

### 1.6 Edge-cluster interactions

`w_edge_cluster_crossing=4.0` exists in `LayoutConfig` but is **never wired into the placement
loss**. It is consulted only inside `dagua/layout/edge_optimization.py:_edge_cluster_crossing_loss`,
called during the post-layout Bezier control-point optimization (after node positions are frozen).
That helps edges *bend around* foreign clusters but cannot push the node positions to make room.
This is the cause of R9/R10 (edges punching through cluster strokes).

### 1.7 Summary of layout-side current state

- Clusters are **not** placement primitives. They are penalty terms over leaf positions.
- Sibling-cluster separation depends on a soft loss with weight `0.5` against typical
  attract/repel weights of `2.0+`/`5.0+` — easily dominated.
- Most algorithm pipelines (the ones dispatched by `LayoutConfig.algorithm`) don't run cluster
  losses at all.
- No notion of "the cluster's own size" exists during placement — only the live leaf bbox.
- No notion of label-room-at-top exists during placement.
- No notion of external-edge clearance exists during placement.

---

## 2. Cluster bounding box computation — current state

### 2.1 At loss-time (placement)

- `_bbox_min_max_per_cluster` (`constraints.py:1332`) — under `no_grad`, scatter-reduces
  `pos[members]` into per-cluster (min, max). Inflated by `max_node_size_in_cluster / 2 + padding`
  (10pt for separation, 18pt for containment). **No label-band reservation, no margin for incoming
  edges**.

### 2.2 At render-time

Two passes, both in `dagua/render/mpl.py`, both recomputing on the placed leaf positions:

**Pass A — bottom-up: nested header reservation (`_compute_cluster_y_maxes`/`_y_mins` at L3607/L3693):**
- Walks clusters in reverse render order (children first).
- For each cluster, takes leaf bbox `[min, max] +/- padding`.
- For each *child* cluster, propagates child's stored `y_max` upward — so a parent extends to cover
  any child's reserved label band.
- Adds `label_height + label_gap` to the parent's `y_max` if the cluster's `label_position` calls
  for top placement.
- Returns `dict[name → y_max]` and `dict[name → y_min]`.

**Pass B — `_draw_clusters` (`mpl.py:7513`):**
1. Walk in render order (parent-first DFS).
2. For each cluster: `padding = max(style.padding + depth*depth_padding_step, 5.0)` (default
   `padding=38pt`, depth_step=`-3pt`). For root cluster, padding ~ 38pt.
3. `x_min, x_max` from leaf bbox +/- padding.
4. `y_min, y_max` from precomputed maps (Pass A).
5. **Min-width clamp:** `min_cluster_width = cluster_height * 0.65`; expand x by `(min_w - cw)/2`
   on each side if narrower.
6. **Label-fit width clamp:** if label is *inside*, expand x to fit `label_width + 2*label_offset_x`.
7. Build `roundrect` path; depth-modulated fill alpha, stroke alpha, corner radius, stroke width.
8. Compute label anchor position (`top-left` default, supports center/right/outside variants).
9. Optionally add an opaque label background (only in graphviz_strict theme) sized `label_width +
    8pt padding`.
10. Stack patches into `fill_paths_by_depth`, `border_paths_by_depth`. Borders use an "annular path"
    (outer minus inner ring) for solid strokes, dash-ribbon for dashed.
11. Z-order: fill at `0.0 + depth*0.01`, border at `0.05 + depth*0.01`, label at
    `0.12 + depth*0.01`.

### 2.3 Observed issues with current bbox math

- **Padding is huge** (38pt outer cluster). That is *not* what drives sibling overlap — the issue
  is that the sibling cluster bboxes are computed *after* layout and the layout has put the leaves
  too close.
- **Min-width clamp** can push cluster boundaries asymmetrically — the cluster bbox during render
  no longer matches what the layout's `cluster_separation_loss` assumed during placement. So even
  when the loss says "no overlap", the rendered boxes can end up overlapping.
- **Label band is reserved at render time only** — placement doesn't see it, so the `y_max` of a
  cluster used during placement is *strictly less than* the rendered `y_max`. Any external node
  placed just above the cluster's leaves can fall inside the rendered label band → "outer cluster
  top edge cuts through node A" (H4).
- **Sibling repulsion uses `max-node-size/2` for half-extent** (constraints.py:1445), but at render
  the bbox is built from `min` of `pos - size/2` and `max` of `pos + size/2`. When sibling
  clusters have different shapes/sizes, the repulsion can declare "no overlap" while the rendered
  bboxes touch. (Caveat: I did not verify this drives the specific observed defect — there is also
  the much larger problem that classical pipelines don't run the loss at all.)
- **Min-width clamp has no analog during placement.** If a cluster has 1 node and a long label,
  the renderer expands the bbox laterally — but placement doesn't know. Overlap is then guaranteed
  if siblings are closer than the expanded width permits.

---

## 3. Render side — current cluster drawing

### 3.1 Layer order (mpl.py:1446-1469)

```
0.0   Cluster fills (per depth)
0.05  Cluster borders (per depth)
0.12  Cluster labels (per depth)  <-- labels are ON TOP of the border ring
0.5   Edges (then arrowheads on top)
1.0   Nodes (filled, with stroke)
2.0   Node labels
3.0   External labels
4.0   Edge labels
```

The cluster **stroke ring is drawn BEFORE the label**, so the label visually overpaints the stroke
where they cross. In graphviz_strict theme, an opaque fill rectangle is drawn under the label as a
mask (`label_background = bg_color, padding = 4pt`) — a hack that physically erases the stroke
behind the label. In other themes, the label simply overpaints the line.

This is "good enough" cosmetic mask but it's exactly what produced the audit complaint "cluster
border stroke crosses through cluster label text" when the label is dim or transparent — the stroke
shows through. The right answer (per dot, see §4) is to *break the path* around the label: the
top-edge stroke has a gap of `label_width + 2*pad` centered on the label.

### 3.2 Cluster path geometry

- `build_shape_path(ShapeSpec(shape="roundrect", center, w, h, corner_radius, stroke_width))` —
  rounded rectangle (4 arcs + 4 segments).
- For solid strokes, `annular_path = outer minus inset` so the fill stops at the inside of the
  stroke ring (avoids stroke aliasing artifacts).
- For dashed, `dash_ribbon_paths(centerline, dash_pattern, width)` rasterizes the dashes.

### 3.3 Where dot does it differently

- Dot draws cluster polygons as **simple polylines, not annular rings**, and **inserts a gap for
  the label** in the top edge.
- Dot's polygon for `cluster_outer` from the test `nested_clusters` is `8,-56 8,-242 203,-242
  203,-56` — a sharp-cornered rectangle. No corner radius. (Dagua's themes default to `8pt`.)
- Dot's label is positioned by default at the **top-center of the polygon's top edge** (not
  top-left as in dagua's default `ClusterStyle.label_position="top-left"`).

This is a **render-side parity gap** that's separate from the placement issue. We can fix it
inside §4–§5 work, but it's clearly delineated from the placement architecture changes.

### 3.4 External-edge-to-cluster crossings (R9/R10)

- Render side has no knowledge of cluster bboxes. Edges are drawn as Bezier polylines from node
  port to node port. If the route happens to pass through a cluster bbox, nothing prevents the
  draw.
- The post-layout `BezierControlPointOpt` op tries to push control points *away* from foreign
  cluster bboxes via `_edge_cluster_crossing_loss`, but only by adjusting the curve, not by
  adjusting the endpoints. If the placement put the source/target nodes such that any reasonable
  curve passes through a foreign cluster, the optimizer has nowhere to go.
- **Root cause confirmation:** if placement guarantees sibling clusters separated AND external
  source nodes outside the cluster bbox by some clearance, then a smooth edge from source to
  in-cluster node *naturally* enters the cluster only at one point (the perimeter intersection
  closest to the target). R9/R10 collapse to "pre-clip the edge against the cluster perimeter
  starting point."

---

## 4. Graphviz `dot` comparison

I ran `dot -Tsvg` (graphviz 8.0.3) on equivalent definitions for `nested_clusters` and
`cluster_showcase`. Verified observations:

### 4.1 Sibling cluster separation (cluster_showcase)
```
cluster_small   x: 628..722  (94pt wide)
cluster_medium  x: 506..620  (114pt wide)  gap to small: 8pt
cluster_large   x: 300..498  (198pt wide)  gap to medium: 8pt
cluster_outer   x:   8..292  (284pt wide)  gap to large: 8pt
cluster_inner   x:  16..108  (92pt wide)   inside outer (10pt inset)
```
**Fixed 8pt sibling gap.** That's `nodesep=8pt` (graphviz default `0.25in = 18pt` reduced via
clusters' tighter separation rules; in modern dot it's `nodesep` at the rank that contains the
cluster sub-DAG).

### 4.2 Asymmetric label band (nested_clusters)
- `cluster_outer` y: `-242..-56` (height 186pt)
- `cluster_left`/`cluster_right` y: `-211..-64` (height 147pt)
- Outer top = -242, child tops = -211 → outer extends **31pt above** children
- Outer bottom = -56, child bottoms = -64 → outer extends **8pt below** children

The 31pt top band = label height (~14pt font) + reasonable padding for label readability + small
margin. The 8pt bottom = standard padding. **dot reserves room asymmetrically: more above (label),
less below.** dagua's render today also does this (Pass A in §2.2) — so this part matches.

### 4.3 Label position
Dot's labels are **top-center**, not top-left:
```
"Outer Group"  text x=105.5 (cluster center=105.5)  y=-226.8 (cluster top -242 + 15.2pt)
"Left Branch"  text x=154   (cluster center=154)    y=-195.8
```
Dagua default `label_position="top-left"`. **Render-parity gap.**

### 4.4 External nodes vs cluster
- Node `a` at cy=-268, ry=18 → bottom of a at y=-250.
- `cluster_outer` top at y=-242.
- Gap between bottom of a and top of cluster outer: 8pt.

dot reserves **`nodesep=8pt` clearance between an external node and any cluster bbox at the same
rank**. dagua does NOT have this — the clusters share nodes only via the `cluster_compactness`
loss, and external nodes have no special handling.

### 4.5 Label-stroke collision
- dot generates the cluster polygon as a continuous polyline — no gap. The label is drawn on top
  with the **graph background color as the text's fill mask**: SVG renders text after the polygon,
  and matplotlib/cairo painters' algorithm puts text on top.
- No actual gap in the path — but because dot's label uses *transparent* on white background, the
  intersection with the stroke is visible. Default dot output keeps the label at top-center where
  the stroke just barely passes through it.
- For dot's PostScript/PDF/PNG outputs, the rasterized stroke has thickness equal to `penwidth`
  and the label simply masks it visually.

So dagua's "label sits in front of stroke, optionally with a background mask" approach is **the
same as dot**. The "border passes through label" complaint resolves by either (a) ensuring the
label background is opaque (graphviz_strict theme already does this with bg-color mask), or (b)
breaking the top-edge path explicitly. Dot does (a). **dagua should default to (a)** for all
themes when there's a cluster border — current dagua only does this in graphviz_strict.

### 4.6 dot's algorithm in one sentence

Dot uses Sugiyama with a *cluster constraint*: every cluster is treated as a "sub-rank" — the
subgraph layout is computed first and the cluster's bbox becomes a **single dummy node** in the
parent layer's rank/order. This is exactly the cluster-as-node directive.

References (verified by reading dot SVG output, not by reading dot source):
- `cluster_padding` = 8pt default in modern dot (`nodesep`/`ranksep` minus border width).
- Cluster bbox = `inner_layout_bbox + 8pt` (sides) `+ label_height + 16pt` (top), plus 8pt below.
- External nodes vs cluster = `nodesep` (8pt) gap.

---

## 5. Architecture proposal

### 5.1 Conceptual model: clusters are placement primitives

At every level of the cluster hierarchy, the placement op operates on a **placement set**:

```
PlacementSet(parent_cluster) = {
    leaf nodes whose deepest-containing-cluster == parent_cluster
} ∪ {
    child cluster bboxes whose parent == parent_cluster
}
```

Each member of a placement set has:
- `width, height` — its placement footprint
- `position` — center (the variable being optimized)
- `kind` — `leaf` (pure node) or `cluster` (recursive sub-problem)

Nodes have width/height from `node_sizes` tensor. Cluster width/height are **computed
bottom-up** after the inner placement converges.

### 5.2 Recursive placement

Pseudocode:

```
def place(cluster: Optional[str], all_nodes_of_cluster):
    children_clusters = [c for c in cluster_parents if cluster_parents[c] == cluster]
    for ch in children_clusters:
        # Recursively place the child cluster's interior. After return,
        # ch has fixed inner geometry and bbox (w_ch, h_ch).
        place(ch, members_of_cluster(ch, leaves_only=True))
        ch.bbox = compute_cluster_bbox(ch)

    leaves = [n for n in all_nodes_of_cluster if no_deeper_child_cluster(n)]
    placement_set = leaves + [c.as_placement_node() for c in children_clusters]

    # Run normal placement on this set (treat cluster-bboxes as opaque rectangles
    # with their computed width/height). Repulsion, attraction, edge-driven
    # forces all act on placement_set members.
    inner_pos = run_placement_op(placement_set, edges_within_cluster, edges_external)

    # Translate child cluster's inner geometry by the delta between its
    # placeholder's inner_pos and the position it had during its own placement.
    for ch in children_clusters:
        delta = inner_pos[ch] - ch.placeholder_position
        ch.translate_descendants(delta)
```

Key implication: after this routine, **the placement engine never sees a sibling cluster overlap**
because cluster placeholders are full-rectangle rigid obstacles for normal node-vs-node repulsion
and overlap-avoidance.

### 5.3 Cluster bbox formula (during placement, used as the placement-node's footprint)

```
inner_bbox = bbox_of(placement_set_after_inner_placement)
W = (inner_bbox.x_max - inner_bbox.x_min) + 2*side_padding + max(0, label_width - inner_w)
H = (inner_bbox.y_max - inner_bbox.y_min) + 2*side_padding + label_band

side_padding = config.cluster_padding   # default 8.0pt (dot parity)
label_band = label_height + 2 * label_gap  # default 14pt label + 8pt above + 4pt below = 26pt
```

The cluster placement-node's "anchor" can be its inner-bbox center; layout writes back:
`cluster.placement_anchor = leaves_centroid_after_inner`.

When the parent layer places this cluster at a target `(cx, cy)`, the layout translates all
descendants by `(cx, cy) - (anchor_x, anchor_y)`.

### 5.4 Label-band and label-room

Reserve `label_band` exclusively at the top of the cluster bbox (asymmetric: more above than
below, matching dot). Once the cluster placement-node enters the parent layer, normal node-sep
rules push siblings (and external nodes) `nodesep` away from the *full* placement-node footprint
including the label band. This is exactly what fixes H4 (external node A clipping cluster top).

### 5.5 External-edge clearance (fixes R9/R10)

Two-step approach:

**Step 1 — placement clearance:** when computing the placement-node footprint, optionally pad it
by an extra `external_edge_clearance` amount. This is similar to label_band but on all sides.
Since cluster bboxes are now hard rectangles in the parent layer, normal repulsion between
external nodes and the cluster pushes them outside this padded perimeter.

**Step 2 — render-time edge clipping (cosmetic, but cheap):** when an edge has source not in cluster
and target in cluster, re-clip the edge polyline against the cluster perimeter so the visible
endpoint sits on the cluster's outer stroke (not inside). Since we already guarantee the source
is outside the cluster, the polyline crosses the perimeter exactly once — clean visual.

**Step 3 — optional placement-time forbidden-region:** add a small loss term that penalizes
any edge polyline that *enters* a foreign cluster's bbox. This already exists in
`_edge_cluster_crossing_loss` for edge-routing post-process — promote it to placement when needed.
Probably not necessary if Steps 1+2 work.

### 5.6 Render-side cluster path generation (fix label/stroke crossing)

Two options, both consistent with the placement architecture above:

**Option A — Default to opaque label background (matches dot's visual behavior).**
Change `ClusterStyle.label_background` to default to "graph background color" with
`label_background_padding=4pt`. Render the rectangle as the label's `bbox patch`. This is what
graphviz_strict theme already does — make it the default. Trivial change.

**Option B — Break the top-edge path around the label.**
Generate the cluster polygon as four sub-paths: top-left segment, top-right segment, bottom edge,
left edge, right edge — with a gap centered on the label of width `label_width + 2*pad`. Slightly
more work; cleaner for SVG export (no opaque overlay needed).

**Recommendation:** Option A for the implementation sprint (1-line change), Option B as a future
follow-up for SVG-export quality.

### 5.7 What dies and what survives

**Dies (deprecated):**
- `cluster_separation_loss` and its `_ClusterCache.sibling_left/right` machinery. Sibling
  separation now falls out of cluster-as-node sibling repulsion at the parent layer.
- `cluster_containment_loss`. Containment is structural in the recursive placement: leaves of a
  child cluster are translated *inside* the parent's bbox by construction.
- `ClusterGridArrange` post-process (already disabled by default; can stay as ops registry but
  marked deprecated).

**Survives:**
- `cluster_compactness_loss` — useful as a secondary force inside a cluster's interior placement
  to keep tightly-coupled members together. Intra-cluster cohesion is still wanted.
- Render-time bbox computation in `_compute_cluster_y_maxes`/`y_mins` — but we should add an
  assertion that the rendered bbox **matches** what placement assumed, with a warning if they
  diverge by more than 2pt. (Render bbox should equal placement bbox up to font-rendering
  rounding.)
- `_edge_cluster_crossing_loss` for post-layout edge optimization — still useful.

**New machinery:**
- `LayoutProblem.cluster_tree` (or recompute on the fly): a tree representation rather than
  flat dict.
- A `place_with_clusters` op that wraps the chosen leaf-placement op (FR, KK, FA2, etc.) into a
  cluster-aware recursive driver. This is the **one new integration op** that makes all 23
  algorithm pipelines cluster-aware, instead of patching each one.
- A function `compute_cluster_placement_bbox(inner_pos, members, label_text, style, config)` that
  returns `(width, height, anchor_offset)` — single source of truth used by both placement and
  render.

### 5.8 Edge-handling during recursive placement

When a child cluster is being placed internally, edges can be:
- **Internal-internal:** both endpoints leaf nodes inside this cluster → normal layout forces apply
- **Internal-external (cluster boundary):** one endpoint inside, one outside this cluster → during
  *internal* placement, the external endpoint is unavailable; treat the edge as a "phantom anchor"
  pulling toward the perimeter at the side closest to the (eventual) external position. Since the
  internal placement runs *first* (recursion bottom-up), we don't know external positions yet —
  use the *direction* implied by the cluster's parent rank or the edge's port hint. (For Sugiyama
  this falls out naturally from layer/rank).
- **External-external (skipping the cluster):** ignored at this level.

For the first sprint, **simplest workable rule**: during internal placement, the cluster ignores
external endpoints entirely (treats the cluster as an isolated subgraph). External edges affect
only the parent-layer placement, where the cluster placeholder's port is at the cluster perimeter.

This is suboptimal for tightly cross-cluster-coupled graphs (`transformer_block` for example) — the
internal layout doesn't know that `attn_out` should be near the bottom of the `mha` cluster
because `add1` (external) is below. We can layer an additional refinement step in a later phase
(see §6, Phase 5).

### 5.9 Algorithm-pipeline integration

The new "wrap a leaf-placement op into cluster-aware recursion" is one op. Expose as:
```
LayoutConfig(algorithm="fr", cluster_aware=True)  # or:
LayoutConfig(algorithm="cluster:fr")  # composed name
```

Implementation strategy: build a `ClusterAwareDriver` that replaces the chosen pipeline's
top-level `Pipeline.run(...)` with a recursive driver that:
1. Builds the cluster tree from `problem.cluster_parents`.
2. Bottom-up: for each leaf cluster, run the same op pipeline on the sub-graph induced by its
   leaves, freeze positions.
3. For each non-leaf cluster, treat its already-placed children-clusters as rigid placeholder
   nodes; run the pipeline again on the placement set (leaves at this depth + cluster
   placeholders).
4. Top: run on the placement set of root.
5. After all recursion, translate descendant nodes to the final placement-node positions.

The "rigid placeholder node" is implemented as a node with `node_sizes[i] = (W, H)` from §5.3.
All existing ops (Repel, Overlap, Pin, Align) treat it like any other node — no per-op changes
needed. Only the driver wraps the recursion.

For Sugiyama / hierarchical algorithms, special handling may be needed (rank assignment must
respect cluster boundaries — a known dot feature). Defer to Phase 5.

---

## 6. Sprint plan — phased implementation

Each phase = one codex dispatch with self-contained spec. Phases must land sequentially.

### Phase 1 — Foundation: cluster tree representation + bbox primitive

**Goal:** extract a single source-of-truth helper for cluster bbox computation that
will be used by both placement and render.

**Files to touch:**
- `dagua/layout/ops/state.py` — add `cluster_tree` field to `LayoutProblem` (or compute lazily).
- `dagua/layout/ops/cluster_geometry.py` (new) — module with:
  - `class ClusterTree(roots, parents, leaves_per_cluster)`.
  - `compute_cluster_placement_bbox(inner_pos, members, label_metrics, style_padding, label_band)
    -> (width, height, anchor_offset)`. **Pure function, no graph object dependency.**
  - `cluster_descendants(tree, name) -> Iterable[str]`.
  - `cluster_leaves(tree, name) -> Iterable[int]`.
- `dagua/render/mpl.py` — refactor `_compute_cluster_y_maxes`/`_y_mins` to delegate to the
  new function (so render produces identical bboxes).

**Verify:**
- New tests in `tests/test_layout/test_cluster_geometry.py` (file exists; extend it).
- Visual regression: render `nested_clusters` with old vs new bbox computation; bytewise SVG
  diff = 0 (or ≤ 2pt difference per coord).
- Existing `test_sibling_clusters_do_not_overlap_badly` and `test_parent_cluster_contains_child_cluster`
  must still pass.

**Risk:** low — pure refactor.

---

### Phase 2 — Cluster-aware placement driver (the core architectural change)

**Goal:** introduce `ClusterAwareDriver` that recursively runs a leaf-placement op pipeline.

**Files to touch:**
- `dagua/layout/ops/cluster_driver.py` (new) — `ClusterAwareDriver` op:
  - Inputs: `LayoutProblem`, `inner_pipeline` (list of ops to apply at each level).
  - For each cluster (bottom-up): build a sub-`LayoutProblem` with the cluster's
    edge-induced subgraph, run `inner_pipeline`, store inner_pos and bbox in
    `state.extras["cluster_inner_layout"][cluster_name]`.
  - For each cluster's parent layer: build a placement set with leaves + cluster placeholders
    (placeholders have inferred sizes from §5.3), run `inner_pipeline`, store positions.
  - At the end, translate all descendants per the final placement-node position.
- `dagua/config.py` — add `cluster_aware: bool = True` to `LayoutConfig`. When True and
  `clusters` is non-empty, dispatch wraps the chosen pipeline.
- `dagua/layout/engine.py` — engine entry detects `cluster_aware=True`, wraps pipeline.

**Verify:**
- New test fixtures: `nested_clusters`, `cluster_showcase`, `transformer_block`. After layout:
  1. No sibling-cluster bbox overlap (existing assertion, but now zero-tolerance instead of `<
     small fraction`).
  2. External-node-to-cluster clearance ≥ `nodesep` (new assertion).
  3. Parent-cluster bbox strictly contains child bboxes (existing assertion).
- Run `pytest tests/test_layout/ -x` — all green.
- Run on the comparison fixtures in `scripts/graphviz_theme_comparison.py --quick` (quick mode
  only renders programmatic showcase). Visually confirm:
  - cluster_showcase: no overlap.
  - nested_clusters: outer top above all child tops by label-band.

**Risk:** medium — touches the dispatch path. Mitigate by gating behind `cluster_aware=True` with
default rollout in Phase 5.

---

### Phase 3 — Render parity polish

**Goal:** match dot's visual cluster behavior beyond placement.

**Files to touch:**
- `dagua/styles.py` — `ClusterStyle.label_position` default → `"top-center"` (was `"top-left"`).
  Note: this is breaking for users who expect top-left; gate behind theme defaults rather than
  changing the global default if needed.
- `dagua/styles.py` — `ClusterStyle.label_background` default → `"@background"` sentinel resolved
  to graph background color in the renderer. Default `label_background_alpha=1.0`,
  `label_background_padding=(4.0, 2.0)` (data, x, y).
- `dagua/render/mpl.py:_draw_clusters` — apply background mask universally (not just in
  graphviz_strict). Remove the `_is_graphviz_strict_render(graph)` gate around the mask.
- Optional: Implement Option B (path-break around label) as a separate function
  `build_cluster_path_with_label_gap` for users who want crisp SVG export.

**Verify:**
- Visual regression on every cluster fixture in `scripts/graphviz_theme_comparison.py`.
- `tests/test_render/test_cluster_label.py` (new) — assert bbox of any drawn stroke does not
  intersect bbox of label patch.
- Pixel-diff against dot output on `nested_clusters`, `cluster_showcase`. Expect significant
  improvement.

**Risk:** low (cosmetic only).

---

### Phase 4 — Edge clipping at cluster perimeter (fix R9/R10)

**Goal:** when an edge crosses a cluster boundary, clip the visible polyline at the perimeter so
it terminates at the cluster's stroke.

**Files to touch:**
- `dagua/routing.py` or `dagua/edges.py` — post-routing pass:
  - For each edge `(src, tgt)`: if `src ∉ cluster_X.leaves` and `tgt ∈ cluster_X.leaves`, find the
    intersection of the edge polyline with `cluster_X.bbox` (the rectangle from §5.3, computed at
    render time the same way placement did). Clip src-side to the intersection point.
  - Symmetric for the reverse case (src in cluster, tgt out).
  - For both-in or neither-in the cluster: no clipping needed.
- Apply to all clusters the edge crosses (e.g. nested case where outer and inner both bound the
  edge: clip at the outermost-non-shared cluster).

**Verify:**
- Visually confirm `transformer_block` no longer has edges punching through cluster strokes.
- Add regression test: rasterize a cluster + an external→internal edge, assert no edge pixel
  exists strictly inside the cluster *interior* (between perimeter and label band) more than 2px
  past the perimeter.
- Edge length / curvature regression: clipped edges shouldn't get visibly shorter (clipping is
  ≤ stroke-width displacement).

**Risk:** medium — interacts with arrowhead positioning. Need to ensure the clipped endpoint
still has correct port direction for the arrowhead. Careful with bezier sub-curve extraction.

---

### Phase 5 — Hierarchical algorithm support (Sugiyama + cluster ranks)

**Goal:** make Sugiyama-family algorithms (`sugiyama`, `reingold_tilford`, `native_layered_dag`)
respect cluster rank constraints — all members of a cluster occupy a contiguous rank range, with
the cluster's bbox spanning those ranks.

**Files to touch:**
- `dagua/layout/ops/layering.py` — `assign_layers_with_cluster_constraints` op variant.
- `dagua/layout/ops/ordering.py` — within-rank ordering must keep cluster members contiguous in
  the sibling order at the cluster's parent layer.
- `dagua/layout/ops/pipelines/sugiyama.py` — pipeline wires the cluster-aware variants when
  `cluster_aware=True`.

**Verify:**
- `transformer_block` rendered with `algorithm="sugiyama"` produces:
  - All MHA cluster nodes within a contiguous y-range (no FFN node interleaved at the same y).
  - All FFN cluster nodes likewise.
  - Cluster bboxes don't overlap.
- Edge crossings in MHA-FFN cross-cluster region within reasonable bound vs flat sugiyama.

**Risk:** high — Sugiyama with cluster constraints is a known-hard problem (the dot algorithm).
Out of scope for the first sprint pass if Phases 1–4 hit the visual targets on cluster fixtures.

---

### Phase 6 — Cleanup & docs

**Goal:** retire the deprecated cluster-separation/containment losses and document the new
architecture.

**Files to touch:**
- `dagua/layout/constraints.py` — mark `cluster_separation_loss`, `cluster_containment_loss` as
  deprecated; emit a one-line warning when `cluster_aware=True` and these are still used.
- `dagua/layout/ops/cluster_arrange.py` — mark `ClusterGridArrange` as deprecated.
- `dagua/layout/engine.py:1909-1948` — gate the legacy loss wiring behind
  `not cluster_aware or legacy_cluster_losses=True`.
- `docs/LLM_TUTORIAL.md` — update the cluster section to document the new model.
- `dagua/CLAUDE.md` — add a "Clusters as placement primitives" section.

**Verify:**
- All cluster-related tests pass with `cluster_aware=True` default.
- A test sets `cluster_aware=False` to verify the legacy path still works (one-release deprecation).

**Risk:** low.

---

## 7. Open questions / decision points for the architect

1. **Default-on or default-off for `cluster_aware=True`?** Recommend default-on after Phase 4
   verifies the visual fixtures, with a release note. Default-off is conservative but means
   classical algorithm dispatchers still produce cluster-blind layouts.

2. **Cluster-as-rigid-rectangle vs cluster-as-soft-region?** This design assumes rigid rectangles
   (full overlap loss applies). Soft regions (cluster bbox is "fuzzy", with gradient falloff)
   would let the optimizer slightly deform clusters under tension. Stick with rigid for the first
   sprint; soft is a future research direction.

3. **Cluster placeholder anchor: leaves-centroid or leaves-bbox-center?** Two reasonable choices.
   Centroid is more stable for non-uniform leaf distributions; bbox-center is geometrically
   intuitive. Recommend centroid (matches `cluster_compactness_loss` semantics).

4. **Cross-cluster edges during recursion (per §5.8):** for the first sprint, treat external
   endpoints as ignored during internal placement. Full handling (port-aware perimeter pinning)
   is a Phase 5+ refinement. Acceptable?

5. **Default `label_position`:** change to `"top-center"` to match dot, or keep `"top-left"`?
   Top-left is dagua's authored default and may be a deliberate stylistic choice. Recommend
   theme-conditional: top-center in `graphviz_match`/`graphviz_strict`, top-left in dagua's
   custom themes. (Single change in `Theme.cluster_style.label_position` per theme.)

6. **External-edge clearance value:** dot uses `nodesep=8pt`. dagua's current
   `_GRAPHVIZ_STRICT_CLUSTER_EXTERNAL_NODE_GAP_POINTS=36.0`. Either is defensible. Recommend
   bringing dagua to `nodesep` value (probably 18pt = dot's default `nodesep=0.25in`).

7. **Sugiyama support timing:** Phase 5 is hard. Should it block the sprint declaration of
   "clusters bulletproof" or be a separate sprint? Recommend: separate sprint. Phases 1–4
   land first, mark Sugiyama+clusters as "experimental: known suboptimal."

8. **Backward compatibility for `LayoutConfig(w_cluster=, w_cluster_contain=, w_cluster_sep=)`:**
   keep the params but warn when set with `cluster_aware=True` — they only affect the legacy path
   from Phase 6 onward.

9. **Pipeline dispatch surface:** `LayoutConfig(algorithm="fr", cluster_aware=True)` vs
   `LayoutConfig(algorithm="cluster:fr")`? Recommend the boolean — it composes cleanly with
   `algorithm_params`.

10. **What's the success metric for "cluster bulletproof"?** Recommend a small benchmark (the 5
    canonical cluster fixtures: nested_clusters, cluster_showcase, transformer_block,
    deep_nesting_4, deep_nesting_6, flat_many_clusters) where:
    - 0 sibling-cluster bbox overlaps.
    - 0 external-node-inside-cluster-bbox.
    - 0 edges visibly punching through a cluster stroke (renderable via Phase 4 clipping).
    - SSIM ≥ 0.85 vs `dot` reference output for graphviz_strict theme on these fixtures.

---

## 8. Quick reference: file map

| Concern | File | Lines |
|---|---|---|
| Cluster declaration | `dagua/graph.py` | 96-100, 519, 856-931, 1475-1485 |
| Cluster passed to layout | `dagua/layout/engine.py` | 980-1078 |
| Layout cluster losses (native) | `dagua/layout/engine.py` | 1909-1948 |
| Layout cluster losses (ops) | `dagua/layout/ops/loss_engine.py` | 1080-1230 |
| Cluster loss math | `dagua/layout/constraints.py` | 1134-1493 |
| Cluster cache / index tensors | `dagua/layout/constraints.py` | 1148-1304 |
| Cluster bbox helper | `dagua/layout/constraints.py` | 1332-1391 |
| LayoutProblem cluster fields | `dagua/layout/ops/state.py` | 110-152 |
| Degenerate-grid post-process | `dagua/layout/ops/cluster_arrange.py` | 1-222 |
| Cluster post-layout edge optimizer | `dagua/layout/edge_optimization.py` | 200, 266, 394, 453, 1557-1616 |
| Cluster bbox compute (render) | `dagua/render/mpl.py` | 3607-3775 (_y_maxes/_y_mins), 3935-3956 (label bounds) |
| Cluster axes expansion (render) | `dagua/render/mpl.py` | 3778-3933 |
| Cluster path generation | `dagua/render/mpl.py` | 7513-7770 |
| Cluster label anchor | `dagua/render/mpl.py` | 3450-3515 |
| External-edge top cap (graphviz_strict) | `dagua/render/mpl.py` | 3546-3604 |
| Render layer order | `dagua/render/mpl.py` | 1442-1469 |
| ClusterStyle definition | `dagua/styles.py` | 472-526 |
| LayoutConfig cluster fields | `dagua/config.py` | 127-128, 328 |
| Cluster regression tests | `tests/test_layout/test_cluster_geometry.py` | full |
| Comparison fixtures | `scripts/graphviz_theme_comparison.py` | 488-546 (showcase), `_iter_cases` 1600 |

---

## 9. Bottom line

The defects observed in the cosmetic-parity sprint are not cosmetic. They are inevitable
consequences of treating clusters as soft penalties on a flat node-position tensor. Fix the
representation: clusters become placement primitives recursively, with bbox computed bottom-up
from inner placement plus label band plus padding. All four observed defects (H4, H5, R9, R10)
collapse to one architectural change. The render side needs only minor parity polish (Phase 3)
and edge clipping (Phase 4). Sugiyama + clusters is a separate hard problem (Phase 5).
