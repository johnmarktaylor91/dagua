# Graphviz Theme Parity Audit -- Round 3

Panels audited: pipeline, diamond, balanced_binary_tree, state_machine,
nested_clusters, arrow_types, cluster_showcase, colors_showcase,
data_pipeline, multi_cycle, complete_k5, deep_nesting_4 (12 total).

Gallery: eval_output/graphviz_theme_round_2/three_way/

---

## Round 2 Fix Verification

### 1. Cluster label font fixed at 10pt (cluster_showcase.png, deep_nesting_4.png)

**FAIL.**

cluster_showcase.png: "Large Cluster With Longer Label" in the strict middle
panel is still rendering at a very large size -- visually ~22-28pt, spanning
nearly the full width of its cluster box and dominating the figure. The label
dwarfs every node label in the figure. The native dot panel has this as
small subordinate text. The `font_size_scaling="fixed"` path was added to the
renderer but it is not taking effect for this cluster.

deep_nesting_4.png: "Level 1" and "Level 2" cluster labels in the strict panel
render at what appears to be ~20pt -- clearly larger than the node labels
("Outer 1," "Outer 2," etc.). Native dot has these labels in 10-11pt matching
the small annotation style. The fix has not landed for deep nesting cases.

data_pipeline.png: "Transform" cluster label is moderately oversized relative
to dot but less egregious. Same root cause.

**Justification:** font_size_scaling="fixed" was merged into the renderer path
but the rendered output shows no visible reduction from round 2 behavior.
Either the new code path is not being reached for these cluster types, or the
gallery was not regenerated after the commit.

---

### 2. Cluster border visible while fill subdued (nested_clusters.png, cluster_showcase.png)

**PARTIAL.**

nested_clusters.png: The strict panel now shows cluster rectangles with clearly
visible thin black borders and a near-transparent fill -- very close to dot.
In round 1 the fill was too dark; it is now correctly subdued. The border_opacity
split has landed here. This is a genuine PASS for nested_clusters.

cluster_showcase.png: The cluster boxes have visible borders and subdued fills.
However the overwhelming cluster label size makes it impossible to evaluate the
border/fill balance on its own. The fix is structurally correct; obscured by
the label regression.

deep_nesting_4.png: Same as cluster_showcase -- borders visible, fills subdued,
but the label size problem overshadows the improvement.

**Overall: PASS for border/fill separation; FAIL for cluster_showcase and
deep_nesting_4 because the label fix did not land, which prevents calling
those panels fully fixed.**

---

### 3. Stray gray rectangle on complete_k5 removed

**PASS.**

complete_k5.png strict panel: no residual light-gray background rectangle is
visible. The background is clean white identical to the dot reference panel.
The stray bounding-region rectangle from round 2 is gone.

---

### 4. Node stroke hairline (pipeline.png, diamond.png)

**PARTIAL.**

pipeline.png: The strict panel node borders are noticeably thinner than in
round 2 -- the round 1 "thick border" artifact is reduced. At this zoom level
the strokes look close to dot's hairlines on "Input," "Preprocess,"
"Transform." There is still a slight visible weight difference: dot's borders
are a near-invisible hairline while strict's borders are a thin but clearly
drawn line. The reduction from 1.3 to 1.0 is perceptible but dot is rendering
at something closer to 0.5-0.75pt equivalent.

diamond.png: Same observation -- improvement visible, not yet matching dot's
ghost-thin hairlines exactly.

balanced_binary_tree.png: Leaf-level nodes ("LLL," "LLR," etc.) in strict have
slightly heavier borders than dot, consistent with the above.

**Justification:** stroke_width=1.0 is better than 1.3 but dot's actual
rendered weight is sub-1.0. The PARTIAL is borderline -- at typical viewing
distance this is now a "careful inspection" difference rather than first-glance.

---

### 5. Back-edge curvature reduced (state_machine.png, multi_cycle.png)

**PASS.**

state_machine.png: The back-edges (Running->Idle, Error->Idle) in the strict
panel are now clearly tighter arcs than in round 2. They hug the node column
more closely and the long back-arcs no longer swing to the panel margin. The
curvature is still perceptibly wider than dot's channel-routed splines (dot
uses libspline B-spline routing that can produce very tight offset curves), but
the difference is now "moderate" rather than "dramatic." Qualitatively the
round 2 fix landed.

multi_cycle.png: The G->A back-arc in the strict panel is tighter than round 2.
The arc stays within the figure boundary rather than approaching the left margin.
Still slightly wider than dot's equivalent arc but no longer a prominent
first-glance discrepancy.

---

## Remaining Departures (priority-ranked)

---

### 1. Cluster label size -- renderer fix did NOT land (CARRY-OVER, CRITICAL)

- Described fully in fix verification item 1 above.
- **Panels:** cluster_showcase (critical), deep_nesting_4 (critical),
  data_pipeline (moderate)
- **Impact:** cluster_showcase is still immediately recognizable as non-dot
  at any zoom level. This is the single largest remaining gap.
- **Fix:** The `font_size_scaling="fixed"` code path in `dagua/render/mpl.py`
  `_cluster_font_size_data()` is not being reached. Likely cause: the renderer
  has a separate cluster-label sizing path for large/outer clusters, or the
  gallery was generated before the fix commit. Re-verify with a fresh render;
  if still broken, add a print probe at the top of `_cluster_font_size_data()`
  to confirm the scaling mode is being read.

---

### 2. Node font size -- still 14pt vs dot's ~11pt visual size (CARRY-OVER)

- **Native dot:** Node labels on pipeline, diamond, balanced_binary_tree render
  at a comfortable small size inside the ellipse with clear internal margins.
  At the image zoom level, "Preprocess" in dot has clear white space left and
  right inside the ellipse.
- **Dagua strict:** "Preprocess" in the strict panel fills the ellipse
  horizontally more tightly. "Postprocess" nearly touches the ellipse border
  left-right. The leaf labels on balanced_binary_tree ("LLL," "LLR," etc.) in
  strict are noticeably larger relative to the ellipse than in dot.
- The round 2 report concluded 14pt is the "correct" value because dot's SVG
  declares 14pt -- but the visual comparison shows a persistent size mismatch.
  The discrepancy is likely a DPI normalization difference: dot declares 14pt
  in SVG units at 72dpi (so 14px = ~10.5pt at 96dpi), while dagua renders
  14pt at 96dpi (so 14pt = ~18.7px effective). A 1.33x DPI ratio would explain
  the gap exactly.
- **Panels:** pipeline (clear), diamond (clear), balanced_binary_tree (clear),
  colors_showcase (moderate -- colored fills hide it somewhat)
- **Fix:** Reduce `_GRAPHVIZ_STRICT_DEFAULT_NODE_STYLE.font_size` from 14.0 to
  10.5 (= 14 * 72/96). Do the same for `EdgeStyle.label_font_size` and
  `GraphStyle.edge_label_font_size`. This is a unit-normalization fix, not a
  design choice.

---

### 3. Cluster box border weight and style -- dagua borders are thicker/darker than dot

- **Native dot:** Cluster rectangles on nested_clusters and deep_nesting_4 have
  very thin (~0.5pt), light-gray (#AAAAAA) hairline borders. The box is
  visually subordinate -- you register it as structure, not as a prominent frame.
- **Dagua strict:** On nested_clusters.png the outer "Outer Group" box and inner
  "Right Branch"/"Left Branch" boxes have clearly black, fully-opaque thin
  borders. The boxes are more visually prominent than in dot -- they read as
  black rectangles rather than light-gray guides. border_opacity=1.0 was the
  right fix to restore border visibility after round 1 overcorrected to
  invisible, but 1.0 opacity on a black stroke is too heavy. Dot uses a
  ~0.5-0.6 opacity gray stroke, not a full-black line.
- **Panels:** nested_clusters, deep_nesting_4, data_pipeline
- **Fix:** In the graphviz_strict ClusterStyle, set the border color to a
  medium gray (e.g., `border_color="#999999"` or similar) rather than relying
  on the fill color with full opacity. Alternatively, reduce border_opacity
  to ~0.5-0.6 while keeping the stroke color black. Target: visually similar
  to dot's light dashed-gray cluster boxes.

---

### 4. Node stroke weight -- still slightly heavier than dot (CARRY-OVER, now minor)

- See fix verification item 4. The 1.3->1.0 reduction is visible and helpful,
  but dot renders closer to ~0.7pt equivalent.
- **Panels:** pipeline, diamond, balanced_binary_tree
- **Impact:** Now a "careful inspection" difference rather than first-glance.
  Worth fixing but not blocking.
- **Fix:** Try `stroke_width=0.75` in `_GRAPHVIZ_STRICT_DEFAULT_NODE_STYLE`.

---

### 5. Back-edge curvature -- still slightly wider than dot (CARRY-OVER, now minor)

- See fix verification item 5. The improvement from round 2 is real. The
  remaining gap is: dot's B-spline routing produces arcs that hug the node
  column within ~20-25pt, while dagua's curvature=0.3 arcs offset ~35-40pt.
- **Panels:** state_machine (moderate), multi_cycle (minor)
- **Impact:** Minor at typical viewing distance. No longer a first-glance
  difference.
- **Fix:** Try `curvature=0.20` for the "back" EdgeStyle. The minimum
  useful value is ~0.15 (below that the arc collapses toward the straight edge
  path and back-edges become indistinguishable from forward edges).

---

### 6. complete_k5 parallel-edge arc distribution -- all arcs fan one side (CARRY-OVER)

- **Native dot:** On complete_k5 the parallel arcs between node pairs are
  distributed symmetrically -- arcs fan both left and right of the chord
  between nodes. The dense K5 crossing pattern looks balanced and somewhat
  symmetric.
- **Dagua strict:** All arcs between the same node pair curve to the same side.
  The K5 panel looks asymmetric and lopsided, with all curvature going
  rightward. This is immediately visible on first glance.
- **Panels:** complete_k5
- **Fix:** Alternate curvature sign (+/-) for successive parallel edges between
  the same node pair. This is a renderer logic change, not a style value.

---

### 7. arrow_types "tee" arrowhead shape mismatch

- **Native dot:** The "tee" arrowhead on arrow_types.png renders as a flat
  horizontal bar perpendicular to the edge -- a clean T-stop shape with no
  fill.
- **Dagua strict:** The "tee" arrowhead renders as a small filled triangle
  (same as "normal"/"vee"). The horizontal-bar tee shape is missing entirely.
  At this zoom level both "tee" source and "tee" edge labels show what is
  clearly a filled arrowhead, not a tee.
- **Panels:** arrow_types
- **Impact:** Moderate. Affects only graphs that use named arrowhead styles.
- **Fix:** Implement tee arrowhead as a perpendicular flat bar in the arrowhead
  renderer, or remap "tee" to a minus/bar primitive if one exists.

---

### 8. Edge label font size -- 14pt still equal weight to node labels (CARRY-OVER)

- **Native dot:** On state_machine edge labels "retry," "resume," "restart,"
  "reset" are clearly smaller and lighter than node labels "Running," "Paused."
- **Dagua strict:** Edge labels are the same visual size as node labels.
- **Panels:** state_machine, data_pipeline
- **Fix:** Same DPI normalization fix as departure 2 -- reduce
  `EdgeStyle.label_font_size` from 14.0 to 10.5.
  (This shares the fix with node font size above.)

---

### 9. multi_cycle background rectangle -- new faint gray box visible

- **Native dot:** Clean white background.
- **Dagua strict:** multi_cycle.png strict panel shows a faint light-gray
  rectangular wash behind the graph, similar to the round-2 "stray rectangle"
  issue that was fixed for complete_k5. It is subtle but visible as a slightly
  off-white background box.
- **Panels:** multi_cycle (strict panel only)
- **Impact:** Minor -- only visible on careful inspection.
- **Fix:** Same investigation path as the complete_k5 stray rectangle fix --
  check whether a graph-level background rect is being rendered and needs to
  be suppressed for graphviz_strict.

---

## New Issues Introduced by Round 2 Changes

### A. data_pipeline "Load" cluster label still oversized

- The round 2 cluster label fix landed for some clusters in data_pipeline
  (the outer "Transform" cluster is somewhat better) but a "Load" cluster
  label remains visibly larger than in dot. Consistent with the general
  cluster label fix not fully landing.

### B. No new regressions found

- The back-edge curvature reduction (0.6->0.3) did not introduce new problems.
- The stroke_width reduction (1.3->1.0) did not introduce new problems.
- The stray rectangle fix for complete_k5 did not re-appear elsewhere
  (multi_cycle's faint box may be a pre-existing issue not introduced by round 2).

---

## Convergence Assessment

### Visual gap remaining

- **Without cluster label fix:** medium-large. cluster_showcase and
  deep_nesting_4 remain first-glance failures due to the oversized labels.
- **After cluster label fix (assuming it lands correctly):** small. The
  remaining differences would be: node font size (DPI normalization, departure
  2), cluster border darkness (departure 3), parallel-arc distribution
  (departure 6), tee arrowhead shape (departure 7). Of these, font size and
  cluster border would be the most noticeable.
- **After font+border fixes:** very small. Only parallel-arc alternation,
  tee arrowhead, and sub-pixel stroke weight differences would remain. Those
  are "careful examination" differences, not first-glance failures.

### Round estimate

- Round 3 was supposed to land the cluster label fix, node font size, edge
  label font size, and node stroke weight. The cluster label fix and font
  size changes did not land (cluster label because the renderer path was not
  reached; font sizes because Codex concluded they were correct based on
  SVG declarations, which appears to be a unit-conversion error). Node stroke
  partially landed.
- One more round that correctly delivers: cluster label fix (re-verify
  renderer path), node+edge font size normalization (10.5pt), cluster border
  color/opacity, and parallel-arc alternation for complete_k5 would close the
  majority of the remaining gap.

### Are we hitting diminishing returns?

Not yet on the items that matter. The cluster label inflation (departure 1)
and node font size mismatch (departure 2) are still first-glance differences
on several panels. Diminishing returns territory begins after those two are
fixed -- at that point the remaining differences (parallel arcs, tee arrowhead,
sub-pixel stroke) require effort disproportionate to the visual delta.

---

## CONTINUE / STOP Recommendation

**CONTINUE -- one targeted round.**

The rationale for continuing:

1. The cluster label fix from round 2 did not land (FAIL verdict above). This
   is the single most visually prominent remaining issue and it has a clear
   fix path -- it just needs to be correctly applied to the renderer.

2. The node + edge font size discrepancy (departure 2 + 8) is a DPI
   normalization error (14pt at 96dpi vs 14pt at 72dpi) that produces a
   measurable, consistent mismatch across every node-bearing panel. One value
   change (14.0 -> 10.5) would fix it.

3. After those two fixes, the remaining gaps shift to diminishing-returns
   territory: parallel-arc alternation (complete_k5 only), tee arrowhead
   (arrow_types only), cluster border gray (moderate), back-edge curvature
   trimming (minor). A STOP would be appropriate after round 4 because the
   residual differences at that point would be:
   - Parallel-arc symmetric distribution: requires renderer logic; visually
     prominent only on K5/dense multigraphs, rare in practice.
   - Tee arrowhead: requires new arrowhead primitive; affects only explicit
     `arrowhead=tee` usage, which is uncommon.
   - Sub-pixel stroke weight difference: rendering-stack floor, not fixable
     without matching dot's exact DPI/scaling pipeline.
   - Back-edge spline routing exact match: would require implementing dot's
     libspline B-spline channel routing, out of scope.

The call: **round 4 should deliver cluster labels + font normalization + cluster
border gray. After round 4, STOP and document the residual stack-level
differences as known-and-acceptable.**

---

## Confidence

**High confidence** on all FAIL/PARTIAL verdicts for round-2 fix verification --
the visual evidence is unambiguous (cluster labels are still very large, font
size mismatch is consistent across panels). **High confidence** on departures
1, 2, 3, 6, 7 (clearly visible). **Medium confidence** on the exact target
values for font normalization (10.5pt = 14 * 72/96 is the mechanically correct
number but may need one iteration to confirm). **Medium confidence** on
departure 9 (multi_cycle background box -- observation is clear, root cause
is inferred by analogy with complete_k5).
