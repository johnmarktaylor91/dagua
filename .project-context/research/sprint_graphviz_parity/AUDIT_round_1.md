# Graphviz Theme Parity Audit -- Round 1

## Summary
- Total panels audited: 14 (all 8 mandatory + 6 bonus)
- Total distinct departure categories: 7
- Severity: **high** -- two categories (edge routing and arrowhead direction) are
  immediately perceptible as "wrong" to any viewer; the others are cosmetic polish

---

## Departure Categories (priority-ranked, most visually impactful first)

---

### 1. Edge routing -- sweeping bezier curves instead of straight (or gently-bent) lines

- **Native dot:** Edges are near-straight polylines. On simple DAGs with clear
  rank separation (pipeline, diamond, balanced_binary_tree) they are perfectly
  straight. On denser graphs (complete_k5, multi_cycle) they are gently curved
  single-bend B-splines that hug the shortest path between node boundaries.
  At no point does any edge form a loop or double back on itself.
- **Dagua strict:** Edges are rendered as large-radius bezier curves that
  dramatically overshoot the straight path. On diamond and balanced_binary_tree
  nearly every edge forms a visible U-shaped or S-shaped sweep. On complete_k5
  edges fan out into wide arcs on one side of the node column. The effect looks
  like force-directed circular edge routing rather than hierarchical straight-line
  routing.
- **Root cause (hypothesis):** The bezier control-point placement used by dagua's
  edge renderer is apparently inheriting curvature from the layout engine
  (curvature parameter) rather than collapsing to straight lines when nodes are
  in a clean rank-separated layout. The `graphviz_strict` edge style has
  `curvature` unset (defaults to a non-zero value), so curved rendering fires
  even when positions are perfectly layered.
- **Panels exhibiting:** diamond, balanced_binary_tree, complete_k5, multi_cycle,
  data_pipeline (partial), nested_clusters (partial), state_machine (subtle)
- **Likely fix location:**
  - `dagua/styles.py` -- GRAPHVIZ_STRICT_THEME EdgeStyle: add explicit
    `curvature=0.0` (or near-0) to the default edge style so the renderer
    produces straight lines
  - `dagua/render/edges/collection.py` -- verify that curvature=0 actually
    produces a degenerate (straight) bezier; if not, add a straight-line
    fast-path when curvature is below a threshold
  - `dagua/render/edges/geometry.py` -- CubicBezier construction from control
    points; ensure straight-line mode uses coincident inner control points

---

### 2. Arrowhead direction -- pointing INTO source node instead of INTO target node

- **Native dot:** Filled triangular arrowheads point in the direction of the
  edge (toward the target/sink node). The tip of the triangle touches the
  target node boundary, and the body of the edge arrives from the source.
- **Dagua strict:** Arrowheads are inverted -- the filled triangle points BACK
  toward the source. On pipeline.png this is unambiguous: each node has an
  arrowhead whose tip points upward (toward the source) instead of downward
  (toward the target). The arrowhead appears to sit above the node it should
  be entering, with the point aimed at the node it just left.
- **Panels exhibiting:** pipeline (clearest), diamond, balanced_binary_tree,
  multi_cycle, complete_k5, state_machine, colors_showcase, arrow_types (all
  arrow variants affected uniformly)
- **Likely fix location:**
  - `dagua/render/edges/collection.py` -- the `tangent` vector passed to
    `build_arrowhead` must be the incoming direction at the target (from edge
    body toward node surface). If it is currently passed as the outgoing
    direction from source it will be flipped. Check the sign convention in the
    call to `build_arrowhead(spec, tip, tangent, ...)`.
  - `dagua/render/edges/arrowheads.py` -- `build_arrowhead`: `body_direction`
    is defined as "unit vector pointing from the tip back into the edge body"
    (line 1237); verify callers supply the correct direction.

---

### 3. Arrowhead size -- dagua normal arrowhead is ~30-40% smaller than dot's

- **Native dot:** The default filled triangle arrowhead is approximately 10pt
  long x 7pt wide at the base. The head is visually prominent and clearly
  readable even on short edges.
- **Dagua strict:** The arrowhead is noticeably smaller -- approximately 7pt
  long x 4.5pt wide based on the configured `arrow_length=7.0`,
  `arrow_width=4.5` in GRAPHVIZ_STRICT_THEME. On pipeline.png the arrowheads
  look like tiny filled dots rather than triangles at normal viewing distance.
  On arrow_types.png the size difference relative to native is visible across
  all arrow types.
- **Panels exhibiting:** pipeline (most visible), diamond, state_machine,
  colors_showcase, arrow_types
- **Likely fix location:**
  - `dagua/styles.py` GRAPHVIZ_STRICT_THEME EdgeStyle: increase
    `arrow_length` from 7.0 to ~10.0 and `arrow_width` from 4.5 to ~7.0 to
    match dot's proportions

---

### 4. Cluster label positioning and font weight -- dagua uses bold/large, dot uses small regular

- **Native dot:** Cluster labels appear top-left in small (approx 11pt) regular-
  weight serif (Times-Roman). The label is inside the cluster box, tight to the
  top-left corner. The cluster bounding box is a thin light-gray or no-fill
  rectangle with a hairline border (approx 0.5pt stroke, no corner rounding).
- **Dagua strict:** Cluster labels render in a noticeably larger and heavier
  font. On cluster_showcase.png and deep_nesting_4.png the labels "Large Cluster
  With Longer Label," "Outer Cluster," "Medium Cluster," "Level 1," "Level 2,"
  "Level 3," "Level 4 (Core)" are rendered at what appears to be 16-20pt bold,
  dominating the visual weight of the graph. The cluster boxes also carry a
  visible gray fill (opacity 0.6 x #F0F0F0) that is denser than dot's near-
  transparent default. Nested cluster depth shading adds extra fill steps not
  present in dot.
- **Panels exhibiting:** cluster_showcase, nested_clusters, deep_nesting_4,
  transformer_block, microservices, data_pipeline
- **Likely fix location:**
  - `dagua/styles.py` GRAPHVIZ_STRICT_THEME ClusterStyle: reduce `font_size`
    from 12.0 toward 10.0-11.0; confirm `font_weight` is "regular" (already
    set); reduce `opacity` from 0.6 toward 0.15-0.2 (dot clusters are nearly
    transparent); reduce `stroke_width` from 0.8 toward 0.5; confirm
    `label_position="top-left"` matches dot (it already does)
  - Nested-cluster depth-fill stepping (depth_fill_step, depth_stroke_step) is
    an enhancement not present in dot; set both to 0.0 for graphviz_strict

---

### 5. "circle" arrowhead -- dagua renders a large open circle, dot renders a small filled dot

- **Native dot:** The `circle` arrowhead (shown in arrow_types.png) renders as a
  small filled black circle approximately 5-6pt diameter, touching the target
  node boundary cleanly.
- **Dagua strict:** The `circle` arrowhead appears to render as a noticeably
  larger open circle (hollow, stroked outline only) approximately 9-10pt
  diameter. In arrow_types.png the difference is visible: dot has a compact
  filled marker, dagua has a larger hollow ring.
- **Panels exhibiting:** arrow_types
- **Likely fix location:**
  - `dagua/render/edges/arrowheads.py` ARROWHEAD_ALIASES: "circle" maps to
    "odot" (line 1058), which triggers `open_fill=True` on the `dot` primitive
    -- making it hollow. Graphviz's `circle` is a synonym for `dot` (filled),
    not `odot` (hollow). Fix the alias: `"circle": "dot"` (not `"odot"`).
  - Also verify `_dot` radius is sized to match dot's ~5pt circle at the default
    scale; current formula `radius = min(length, width) * 0.5` with
    `arrow_length=7.0` gives radius ~3.5pt which seems right once fill is fixed.

---

### 6. Node stroke weight -- dagua nodes have a slightly heavier border than dot

- **Native dot:** Node ellipse borders are 1px rendered (approx 0.75pt at 96dpi),
  appearing as a single-pixel hairline outline.
- **Dagua strict:** Node borders render at `stroke_width=1.3`, which is
  perceptibly heavier than dot's hairline at normal zoom. On pipeline.png and
  diamond.png the ellipse outlines are clearly thicker than the reference.
  (The comment in styles.py says "slightly above 1.0 to compensate for AA
  thinning" but the visible result overcorrects.)
- **Panels exhibiting:** pipeline, diamond, balanced_binary_tree,
  colors_showcase, arrow_types
- **Likely fix location:**
  - `dagua/styles.py` _GRAPHVIZ_STRICT_DEFAULT_NODE_STYLE: reduce
    `stroke_width` from 1.3 to 1.0 and test whether AA thinning is actually
    a problem at target export DPI. If it is, 1.1 is less overcorrected than 1.3.

---

### 7. Edge label placement -- dagua places labels slightly off-center; dot centers on edge midpoint

- **Native dot:** Edge labels (e.g., "retry," "resume," "restart," "reset" on
  state_machine.png) appear centered horizontally on the edge at its visual
  midpoint, with a small transparent background box.
- **Dagua strict:** Labels on state_machine.png appear in approximately correct
  positions overall, but the label spacing from the edge curve is tighter on
  some edges (labels nearly touching the edge line) compared to dot (labels
  have a small clearance gap above/below the edge). This is a minor issue
  compared to the routing and direction problems.
- **Panels exhibiting:** state_machine (most visible), data_pipeline
- **Likely fix location:**
  - `dagua/render/edges/labels.py` -- label clearance/offset parameters
  - `dagua/styles.py` GRAPHVIZ_STRICT_THEME GraphStyle: `edge_label_font_size`
    is 14.0 which matches node label size; dot uses slightly smaller edge
    labels (~11pt), so reducing to 11.0-12.0 would help visual separation

---

## Per-Panel Notes

### pipeline.png
- Arrowhead direction inversion is the primary defect here; routing is straight
  (correct). Arrowheads point UP into the source instead of DOWN into the target.
- Arrowhead size is small (~half of dot's).
- No edge routing issues (positions are clean rank-separated chain).

### diamond.png
- **Most egregious routing failure in the set.** All four edges form large
  sweeping curves/loops instead of the two gentle diagonal lines that dot
  produces. The routing makes the graph look like a circular layout, not a DAG.
- Arrowhead direction also inverted (visible on the U-curve endpoints).

### balanced_binary_tree.png
- Same routing failure as diamond, compounded over 14 nodes. Every parent-child
  edge is a large bezier arc. In dot the tree uses straight diagonal lines.
  The dagua render is barely recognizable as a binary tree.

### state_machine.png
- Routing is mostly acceptable for the cyclic graph (small deviations from
  dot's long-arc back-edges).
- Arrowhead direction inverted on all edges.
- Edge labels "retry"/"resume" labels are slightly crowded, otherwise correct
  placement.

### nested_clusters.png
- Cluster boxes render correctly positioned and sized.
- The inner-cluster label font is oversized vs dot's small serif label.
- Edge routing within and between clusters has some curvature that dot avoids.
- Cluster borders appear slightly heavier/denser than dot.

### arrow_types.png
- All arrow type shapes present and generally recognizable.
- "circle" type is hollow (odot) in dagua, filled (dot) in native -- wrong alias.
- Arrow sizes are uniformly smaller than dot across all types.
- Direction inverted across all types.

### cluster_showcase.png
- Cluster label font is dramatically oversized vs dot. "Large Cluster With
  Longer Label" in dagua renders at a size that dominates the figure; in dot
  it is small and subordinate.
- Cluster fill opacity is heavier than dot.

### colors_showcase.png
- Node fill colors match dot very closely (red, blue, green, yellow, purple,
  orange all look correct).
- Arrowhead direction inverted (visible on each edge's terminal).
- Arrowhead size is small.
- This panel confirms color handling is not a problem.

### data_pipeline.png
- Cluster rendered around "Transform" subgraph -- fill weight too heavy vs dot.
- Edge routing has curvature issues on inter-cluster edges.
- Arrowhead direction inverted.

### multi_cycle.png
- Back-edges (A->B back-arc, cycle edges) have excess curvature in dagua.
  Dot routes the back-arc as a clean single-bend spline staying near the node
  column; dagua's arcs spread wider.
- Arrowhead direction inverted.

### microservices.png
- Complex multi-cluster graph; cluster label sizes are the most visible defect
  (too large, too heavy).
- Edge routing within clusters shows curvature that dot avoids for short edges
  between adjacent nodes.

### transformer_block.png
- Cluster labels ("Multi-Head Attention," "Feed-Forward Network") render at
  correct sizes in dagua strict -- these appear close to dot. Better cluster
  rendering than cluster_showcase suggests cluster label size depends on graph
  or config context; worth investigating whether the label font size is being
  overridden per-cluster.
- Edge routing within the attention cluster shows some excess curvature.

### deep_nesting_4.png
- 4-level nested clusters; each level label is large and bold in dagua vs small
  regular in dot.
- The cluster depth-fill shading makes inner clusters progressively darker,
  which dot does not do; dot uses a uniform very-light gray for all depths.

### complete_k5.png
- Fully-connected 5-node graph; every edge pair has a routing arc.
- Dot distributes arcs evenly on both sides of node pairs for parallelism;
  dagua fans all arcs to one side, producing a lopsided layout of edge curves.
- Arrowhead direction inverted.

---

## Categories of difference NOT to fix

1. **Font hinting and sub-pixel rendering.** Dot rasterizes text via Freetype
   with native hinting; matplotlib uses its own rasterizer. Sub-pixel glyph
   shape and spacing will never be identical and are not worth chasing.

2. **Antialiasing at node boundaries.** Dot produces 1px aliased ellipse
   outlines at low DPI; matplotlib applies MSAA. The resulting soft vs sharp
   look is a rendering-stack difference.

3. **Output DPI and canvas dimensions.** The comparison gallery is generated at
   a fixed DPI/size that may not match dot's default 72dpi PS output. Small
   absolute-coordinate differences (1-2pt) are acceptable noise.

4. **Exact bezier spline interpolation for back-edges.** Dot uses an internal
   B-spline router (libspline) that computes control points based on a channel
   model. Perfectly replicating the exact curve shape of long back-edges
   (e.g., the Idle->Running arc in state_machine) would require re-implementing
   dot's spline router. Close-but-not-identical curves on back-edges are
   acceptable; what is NOT acceptable is the large-arc sweeping seen on
   straight-line DAG edges (categories 1 and 2 above).

---

## Confidence

**High confidence** on categories 1-4. The arrowhead direction inversion and the
bezier sweeping on clean DAG edges are categorical, visually unambiguous
differences visible in every panel. The cluster label size inflation is equally
unambiguous in cluster_showcase and deep_nesting_4.

**Medium confidence** on categories 5-7. Arrow_types.png is small in the
three-way layout so exact size comparisons have some uncertainty. The "circle"
alias finding (category 5) is based on reading the alias table in code rather
than pixel measurement. Node stroke weight (category 6) is a small difference
that may partially be explained by DPI normalization.

**Would benefit from a follow-up look:**
- A zoom into arrow_types.png at 2x magnification for the "circle," "vee," and
  "open" variants to confirm size and fill mode.
- A single-panel render of a 3-node chain in graphviz_strict at 300dpi to
  measure arrowhead pixel dimensions precisely against a dot reference.
- Confirmation that curvature=0.0 in EdgeStyle actually collapses the bezier to
  a straight line in collection.py (the fix for category 1).
