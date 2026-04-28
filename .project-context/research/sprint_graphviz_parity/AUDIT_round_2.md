# Graphviz Theme Parity Audit -- Round 2

Panels audited: pipeline, diamond, balanced_binary_tree, state_machine,
nested_clusters, arrow_types, cluster_showcase, colors_showcase,
data_pipeline, multi_cycle, complete_k5, deep_nesting_4 (12 total).

---

## Round 1 Fix Verification

- **[1] Edge curvature -> straight DAG edges: PASS**
  diamond.png and balanced_binary_tree.png both now show straight diagonal
  lines that match dot exactly. The dramatic U-shaped/S-shaped sweeping from
  round 1 is gone. Verified on: diamond, balanced_binary_tree, data_pipeline.

- **[2] Arrowhead direction -> toward target: PASS**
  pipeline.png shows arrowheads pointing downward into each receiver node,
  matching dot's direction. The inversion is fully resolved. Verified on:
  pipeline, diamond, colors_showcase.

- **[3] Arrowhead size 10x7pt: PASS**
  The arrowheads on pipeline and colors_showcase are visibly chunkier and now
  proportionally match dot's filled triangles at normal viewing distance.
  No longer look like pinpricks. Verified on: pipeline, arrow_types,
  colors_showcase.

- **[4] Cluster styling subdued: PARTIAL**
  nested_clusters and data_pipeline clusters are now visibly lighter (opacity
  0.15 is a clear improvement; depth darkening is gone). However cluster_showcase
  and deep_nesting_4 still exhibit oversized cluster labels rendered at a size
  that dominates the figure. The codex report acknowledges this: "the renderer
  still scales cluster label size from cluster height." The 10pt theme value is
  set, but the renderer overrides it for large clusters. Fill opacity is fixed;
  label size fix is incomplete. Verified on: nested_clusters (pass),
  cluster_showcase (fail on label size), deep_nesting_4 (fail on label size).

- **[5] "circle" arrowhead filled: PASS**
  arrow_types.png strict panel shows the circle type as a small filled dot,
  matching dot's behavior. The alias fix and forced-fill override both landed.
  Verified on: arrow_types.

---

## Remaining Departures (priority-ranked)

---

### 1. Cluster label size -- renderer ignores 10pt theme value for large clusters

- **Native dot:** Cluster labels render in small (~10-11pt) regular-weight
  Times-Roman, tightly in the top-left corner. On cluster_showcase.png the
  labels "Large Cluster With Longer Label," "Outer Cluster," "Medium Cluster"
  are subordinate text; the label for "Large Cluster With Longer Label" is
  approximately the same visual weight as a node label.
- **Dagua strict:** On cluster_showcase.png and deep_nesting_4.png the cluster
  labels "Large Cluster With Longer Label," "Level 1," "Level 2" render at
  what appears to be 20-28pt -- far larger than any node label in the figure.
  The cluster_showcase strict panel has "Large Cluster With Longer Label"
  spanning nearly the full width of its cluster box in very large text. The
  10.0pt theme setting is being overridden by a renderer-side policy that
  scales cluster label font size from cluster bounding-box height.
- **Panels exhibiting:** cluster_showcase (most egregious), deep_nesting_4,
  data_pipeline (moderate -- "Transform" cluster label slightly large)
- **Likely fix location:**
  - `dagua/render/clusters.py` (or equivalent) -- locate the renderer path
    that computes cluster label font size and bypass/cap the height-based
    scaling when the theme is graphviz_strict, or globally when a fixed
    font_size is explicitly set in ClusterStyle. The fix: if
    `cluster_style.font_size` is set explicitly (non-None / non-zero), use
    that value directly; do not apply any height-based scaling factor.

---

### 2. Node stroke weight -- dagua nodes have a heavier border than dot (CARRY-OVER from round 1, cat. 6)

- **Native dot:** Node ellipse borders are ~0.75pt rendered hairlines (1px at
  96dpi). The outline is thin and barely visible against white fill.
- **Dagua strict:** `stroke_width=1.3` in the theme (unchanged from round 1).
  On pipeline.png, diamond.png, and colors_showcase.png the node outlines are
  visibly heavier than dot. On pipeline the difference is immediately apparent:
  dot's "Preprocess" ellipse has a thin hairline; strict's "Preprocess" has a
  moderately thick border. This is the most visible remaining departure on
  clean DAG panels after the routing fix.
- **Panels exhibiting:** pipeline, diamond, balanced_binary_tree,
  colors_showcase, complete_k5
- **Likely fix location:**
  - `dagua/styles.py` `_GRAPHVIZ_STRICT_DEFAULT_NODE_STYLE`: reduce
    `stroke_width` from 1.3 to 1.0. The "AA thinning compensation" comment
    is overcorrecting; 1.0 is the correct Graphviz default.

---

### 3. Edge label font size -- 14pt is too large relative to dot's ~11pt edge labels

- **Native dot:** Edge labels on state_machine.png ("retry," "resume,"
  "restart," "reset") render at approximately 11pt in Times-Roman, visibly
  smaller than the 14pt node labels. The labels are subordinate text that
  annotates the edge without competing with node text.
- **Dagua strict:** The GRAPHVIZ_STRICT_THEME `GraphStyle.edge_label_font_size`
  is 14.0 and `EdgeStyle.label_font_size` is also 14.0. On state_machine.png
  the "retry"/"resume" labels are the same size as the node labels "Running,"
  "Paused," giving them equal visual weight. In dot, edge labels are clearly
  subordinate.
- **Panels exhibiting:** state_machine (clearest), data_pipeline
- **Likely fix location:**
  - `dagua/styles.py` GRAPHVIZ_STRICT_THEME:
    - `GraphStyle.edge_label_font_size`: reduce from 14.0 to 11.0
    - `EdgeStyle.label_font_size`: reduce from 14.0 to 11.0
    These are two separate fields; both need updating.

---

### 4. Back-edge curvature on cyclic graphs -- arcs spread too wide vs dot

- **Native dot:** On multi_cycle.png and state_machine.png, back-edges (e.g.,
  G->A in multi_cycle, Error->Idle in state_machine) are single-bend splines
  that stay close to the node column. The arc for G->A rises cleanly along the
  left side of the figure with a moderate offset (~30-40pt from the node
  column).
- **Dagua strict:** The "back" EdgeStyle has `curvature=0.6` (unchanged from
  default). On multi_cycle.png the G->A back-arc swings noticeably wider than
  dot's reference -- the arc starts to approach the left margin of the panel
  while dot's stays contained. On state_machine.png the long back-edges
  (Running->Idle, Error->Idle) have wider arcs than dot, though the difference
  is moderate rather than dramatic. The forward edges (curvature=0.0) are now
  straight and correct; only back-edges carry excess width.
- **Panels exhibiting:** multi_cycle, state_machine, complete_k5
- **Likely fix location:**
  - `dagua/styles.py` GRAPHVIZ_STRICT_THEME `edge_styles["back"]`:
    reduce `curvature` from 0.6 to approximately 0.35-0.4 to produce
    tighter arcs matching dot's channel-routed back-edge splines.
    (Cannot go to 0.0 -- back-edges must curve to avoid overlapping forward
    edges.)

---

### 5. Node font size -- 14pt in dagua strict vs ~11-12pt in dot

- **Native dot:** On pipeline.png and diamond.png the node labels ("Input,"
  "Preprocess," "Start," "Left," "Right," "End") render at approximately
  11-12pt Times-Roman. The text fits comfortably inside the ellipse with clear
  internal margins.
- **Dagua strict:** `font_size=14.0` in `_GRAPHVIZ_STRICT_DEFAULT_NODE_STYLE`.
  On pipeline.png the labels are visibly larger than dot's -- "Postprocess" in
  strict occupies nearly the full horizontal extent of the ellipse while dot's
  "Postprocess" has more visible white space around it. On balanced_binary_tree
  the leaf-level labels ("LLL," "LLR," etc.) in strict are noticeably larger
  relative to the node size than in dot.
- **Panels exhibiting:** pipeline, diamond, balanced_binary_tree,
  colors_showcase
- **Likely fix location:**
  - `dagua/styles.py` `_GRAPHVIZ_STRICT_DEFAULT_NODE_STYLE`:
    reduce `font_size` from 14.0 to 11.0 (dot's default label size is
    ~14pt in abstract units but renders to ~11pt at typical graph DPI;
    the discrepancy suggests a DPI or unit conversion mismatch in how
    dagua maps the font_size field to rendered points).

---

### 6. complete_k5 parallel-edge arc distribution -- all arcs fan to one side

- **Native dot:** On complete_k5.png dot distributes parallel arcs
  symmetrically around node pairs: arcs between V1-V3 fan both left and right
  so the dense inter-node crossing is visually balanced.
- **Dagua strict:** All parallel arcs between the same node pair fan to the
  same side (consistent offset direction), producing an asymmetric lopsided
  appearance where all curvature goes rightward. The strict panel is clearly
  identifiable as non-dot at a glance for this graph type.
- **Panels exhibiting:** complete_k5
- **Likely fix location:**
  - `dagua/render/edges/collection.py` or the multi-edge offset logic:
    when multiple edges connect the same node pair, alternate the curvature
    sign (+/-) for successive edges so arcs distribute on both sides.
    This is a rendering-layer fix independent of the curvature=0.0 default;
    the alternation should apply only when `curvature != 0.0`.

---

### 7. Edge label placement -- labels touch edge line; dot has clearance gap (CARRY-OVER from round 1, cat. 7)

- **Native dot:** On state_machine.png edge labels have a small but visible
  clearance gap between the label text and the edge line (~2-3pt of space).
  The label appears to float slightly above/beside the edge.
- **Dagua strict:** On state_machine.png "retry" and "resume" labels sit very
  close to the edge line -- essentially touching it. The label rendering lacks
  the small float offset that dot applies. This is minor but visible on graphs
  with many edge labels.
- **Panels exhibiting:** state_machine
- **Likely fix location:**
  - `dagua/render/edges/labels.py` -- increase label perpendicular offset
    from edge centerline by ~2-3pt.
  - `dagua/styles.py` GRAPHVIZ_STRICT_THEME EdgeStyle: `label_offset` if
    that field exists.

---

## New Issues Introduced by Round 1 Changes

### A. Cluster border now too faint -- opacity 0.15 undershoots dot

- The round-1 cluster opacity reduction (0.6 -> 0.15) overcorrected for nested
  clusters. On nested_clusters.png and deep_nesting_4.png the cluster box
  borders are now nearly invisible -- the outer cluster rectangle in
  nested_clusters strict is a ghost outline whereas dot renders a clearly
  visible (though thin) light-gray hairline box. The fill is now correct
  (near-transparent) but the border stroke should remain visually legible.
  Dot's cluster box is: thin solid stroke (~0.5pt, #AAAAAA or similar), fill
  very light gray (#F0F0F0 at ~15-20% opacity). Dagua's 0.15 opacity applies to
  BOTH fill and stroke, making the stroke too faint.
- **Fix:** Separate fill opacity from stroke opacity in ClusterStyle, or raise
  opacity to ~0.25-0.30 so the border is visible while the fill remains light.
  Alternatively, keep opacity=0.15 but set an explicit higher-opacity stroke
  color (e.g., `stroke="#888888"` at full opacity independent of the fill
  opacity field).

### B. complete_k5 strict panel now shows residual light-gray background rectangle

- After the round-1 changes, complete_k5.png strict panel shows a faint
  gray-filled rectangle in the background (visible as a subtle rectangular wash
  behind the nodes). This is not present in the native dot panel and was not
  present in round 1. The rectangle appears to be a stray cluster or graph
  background element. May be an interaction between curvature=0.0 and how the
  renderer computes its bounding region.
- **Panels exhibiting:** complete_k5 (strict panel only)
- **Likely fix location:** unclear -- investigate whether a graph-level
  background rect is being drawn that should be clipped or not rendered for
  graphviz_strict.

---

## Convergence Assessment

- **Estimated visual gap remaining:** medium
- **Rounds to reach "indistinguishable at a glance":** 2 more rounds
  - Round 3: fix cluster label renderer override (cat. 1), node stroke_width
    1.3->1.0 (cat. 2), node font_size 14->11 (cat. 5), edge label font 14->11
    (cat. 3). These are all mechanical value changes.
  - Round 4: back-edge curvature tuning (cat. 4), parallel-arc alternation
    (cat. 6), edge label clearance gap (cat. 7), cluster border opacity fix
    (new issue A). These require renderer logic changes, not just style values.
- **Diminishing returns:** Not yet. Categories 1, 2, and 5 (cluster label
  inflation, node stroke weight, node font size) are still immediately
  perceptible on first glance. The strict panel still looks noticeably
  different from dot on pipeline and cluster_showcase. After round 3 fixes,
  the remaining delta should shrink to "careful inspection" territory, and
  diminishing returns will apply from round 4 onward.

---

## Confidence

**High confidence** on all PASS/FAIL verdicts for round-1 fixes (categories 1-3
and 5 clearly visible; category 4 partial verdict is unambiguous -- cluster fill
is correct but label size is not). High confidence on remaining departures 1-3
(cluster label inflation, stroke weight, edge label font size -- all measured
against explicit theme values in styles.py). Medium confidence on departure 4
(back-edge curvature -- arc width difference is real but the exact target value
for `curvature=` requires iterative tuning). Medium confidence on departure 6
(parallel-arc distribution -- the asymmetry is visible but the fix location is
inferred). New issue B (gray rectangle on complete_k5) is high confidence on
the observation, medium on root cause.
