# Cluster Sprint Post-Phase-4 Visual Audit

- Auditor: Opus 4.7 (1M context), maximally picky.
- Inputs:
  - `eval_output/parity_metrics.json`, `eval_output/parity_metrics_summary.md`
  - `eval_output/parity_pixel_diff/summary.md`
  - Hi-res pairs: `eval_output/parity_pixel_diff/hires/<slug>/{dot,dagua}.png`
- Cluster panels inspected (8): `nested_clusters`, `cluster_showcase`, `transformer_block`,
  `cross_cluster_edges`, `deep_nesting_4`, `flat_many_clusters`, `microservices`,
  `data_pipeline`.
- Image read budget: 16 of 16 used.
- Critical context: `graphviz_theme_comparison.py` harness uses GRAPHVIZ POSITIONS for the
  Dagua side. Phase 2 placement gains do NOT show. Phase 3 (render parity) and Phase 4
  (edge clipping at cluster perimeter) are observable. Sibling overlap and containment
  failures observed below are downstream of dot's positions filtered through Dagua's
  bbox/padding rules; the actual Phase 2 driver is unverified by this gallery and should
  be checked separately with Dagua's own placement.

---

## Verdict

- Prior items: `N/A` (no prior numbered list to recheck).
- New audit: `FAIL`.
- Stop criteria status: `CONTINUE` — multiple `real_cosmetic_gap` + `fixable_theme_or_render`
  findings remain.

The cluster sprint has **NOT** achieved bulletproof status, and Dagua's clustered output is
**not yet at parity with graphviz**. The most critical render-time defects: missing inner
cluster bboxes (`nested_clusters` shows children as bare vertical lines, no rectangle drawn);
`cluster_showcase` outright omits the "Large Cluster With Longer Label" border; cluster
labels are bisected by their top stroke because the @background mask is narrower than the
label text; Phase 4 edge clipping does not visibly engage on any of 8 panels (every
external->internal edge still pierces the cluster perimeter and arrowheads land *inside*
the bbox); sibling clusters overlap and members spill outside their parent containers in
4 of 8 panels.

Honest answer: this is a long way from "look at least as nice as graphviz." Phase 1 (bbox
primitive) and Phase 4 (edge clipping) appear effectively non-functional in the rendered
gallery. Phase 3 partially landed (asymmetric label band reservation on simple panels works,
mask is universal but undersized).

---

## Prior Item Recheck

| Item | Verdict | Evidence | Notes |
| --- | --- | --- | --- |
| n/a | n/a | n/a | No prior list provided. |

---

## New Findings

| # | Severity | Panel | Element/Region | Finding | Finding Class | Actionability | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F1 | HIGH | `nested_clusters` | child cluster rectangles (`left`, `right`) | Inner cluster bboxes are NOT drawn as rectangles. Only thin near-vertical fragments of the right/left side strokes are visible; top + bottom edges are missing. Their labels "Right Branch" / "Left Branch" float un-anchored. Dot draws full rectangles. | `real_cosmetic_gap` | `fixable_theme_or_render` | hires/nested_clusters/dagua.png vs dot.png. mpl.py path-build for child clusters at depth>0 appears to drop top/bottom segments — likely the annular path or dash-ribbon collapses when narrow. |
| F2 | HIGH | `nested_clusters` | external node A vs outer cluster top stroke | Node A penetrates the outer cluster top edge — A's bottom half is below the cluster top stroke. Dot leaves an 8pt gap between bottom of A and outer cluster top. The outer-cluster label "Outer Group" is also rendered THROUGH node A (label centered horizontally, but at a y where it overlaps the A ellipse interior). | `real_cosmetic_gap` | `needs_layout_scope` for placement; `fixable_theme_or_render` for the rendered overlap if Dagua's own placement produces clearance. | hires/nested_clusters/dagua.png. Driven by harness using dot positions and Dagua's render expanding the cluster bbox upward into A's space. Phase 2 (cluster-as-node) is the real fix; cannot validate from this gallery. |
| F3 | HIGH | `cluster_showcase` | `cluster_large` (large 0..5) | The bounding rectangle for "Large Cluster With Longer Label" is COMPLETELY MISSING. Only the floating label is drawn. Children render naked. | `real_cosmetic_gap` | `fixable_theme_or_render` | hires/cluster_showcase/dagua.png. `_draw_clusters` apparently skips drawing this cluster's path (perhaps min-width clamp triggers a degenerate annular path that emits no segments). Compare dot.png — clean rectangle. |
| F4 | HIGH | all 8 panels (visible: deep_nesting_4, flat_many_clusters, microservices, data_pipeline) | cluster top-edge label collision | The graphviz_strict @background mask under cluster labels is too narrow: the cluster top stroke remains visible *through* the label text on long labels. Examples: "Level 1..4 (Core)" (deep_nesting_4), "Alpha"/"Beta"/"Gamma"/"Delta" (flat_many_clusters), "API Layer"/"Service Layer"/"Worker Layer" (microservices), "Extract"/"Transform" (data_pipeline). | `real_cosmetic_gap` | `fixable_theme_or_render` | The DESIGN.md Phase 3 spec called for `label_background_padding=(4.0, 2.0)` — the rendered mask suggests the padding is sized to the un-typeset text bounding box, not the rasterized text bbox. Likely fix: increase padding x to ~6-8pt and verify mask uses real font metrics. mpl.py:_draw_clusters label-mask block. |
| F5 | HIGH | `deep_nesting_4`, `data_pipeline`, `transformer_block`, `nested_clusters` | external->internal edge clipping at cluster perimeter | Phase 4 clipping does not visibly engage. Specific cases: Source->Outer 1 enters Level 1 with arrowhead inside the bbox (not clipped to perimeter); CSV/API/DB Source->{Parse CSV,Fetch API,Query DB} arrowheads land INSIDE the Extract bbox just below the top edge; Input Embedding -> Add bypass and side ports cross MHA right edge un-clipped; A->C, A->B in nested_clusters cross outer top stroke un-clipped. | `real_cosmetic_gap` | `fixable_theme_or_render` | Phase 4's perimeter clip looks like it was either gated by a flag that's off in the harness, or the perimeter geometry it consults is the placement bbox (which here is dot-derived) and so the intersection math returns "no crossing." Verify edge polyline vs rendered cluster path. |
| F6 | HIGH | `cross_cluster_edges`, `flat_many_clusters`, `microservices`, `nested_clusters` | sibling cluster overlap + containment failure | Sibling clusters visibly overlap their bboxes (Cluster X / Cluster Y / Cluster Z all intersect; Alpha/Beta/Gamma/Delta all share x-ranges; Service Layer / Data Layer / Worker Layer overlap). Members fall outside their parent bbox: Y3/Y4 in cross_cluster_edges sit *inside* Cluster Z bbox; B4 sits below Beta's bottom edge; D4/D5 sit below Delta's bottom edge in flat_many_clusters; "Data Warehouse" partially below Load bottom edge in data_pipeline. | `metric_or_measurement_artifact` partially (harness uses dot positions, not Dagua's), but the bbox-expansion logic IS Dagua's | `needs_layout_scope` (Phase 2 verification); `fixable_theme_or_render` for the bbox padding constants. | hires/cross_cluster_edges/dagua.png; hires/flat_many_clusters/dagua.png. Cause #1: harness forces dot positions onto Dagua. Cause #2: Dagua's render-time padding (38pt outer + 10/18pt loss padding mismatches) swells bboxes beyond the placement assumption. Need to verify with Dagua's own placement. |
| F7 | HIGH | `cluster_showcase`, `cross_cluster_edges` | empty bbox padding above cluster contents | Cluster X bbox extends ~150% of its inner-content height above its top member (X1) — the label sits in deep empty space rather than tight to the top. Dot is much tighter (label ~15pt above top member). Same pattern visible on the missing/tall "Large Cluster" region in cluster_showcase. | `real_cosmetic_gap` | `fixable_theme_or_render` | DESIGN.md §5.3 specifies `label_band = label_height + 2*label_gap`. The rendered label band looks like 3-5x that. Probably a unit/scale mismatch in the bottom-up `_compute_cluster_y_maxes` propagation. |
| F8 | MED | `transformer_block` | edges crossing MHA / FFN cluster right border | Bypass edge from "Input Embedding" travels vertically along the right side of MHA; clearly crosses MHA's right border twice (entering at top, exiting at bottom). Phase 4 should clip a cross-cluster edge segment to the perimeter intersection or push the routing outside. | `real_cosmetic_gap` | `fixable_theme_or_render` | hires/transformer_block/dagua.png. |
| F9 | MED | `data_pipeline` | "Data Warehouse" extends below Load cluster bottom | The Load cluster bottom edge cuts through the Data Warehouse oval near its midline. Containment failure at render time. | `real_cosmetic_gap` | `fixable_theme_or_render` | hires/data_pipeline/dagua.png. The render-side bbox for Load doesn't include enough bottom padding to wrap below member nodes' bottoms. |
| F10 | MED | `cross_cluster_edges` | Cluster Z residual stroke fragments | Top-right area of Cluster Z bbox shows a stray short stroke segment (a few pixels long) outside the main rectangle — looks like a leftover from path-building (possibly an annular or dash-ribbon vertex not closed properly). | `real_cosmetic_gap` | `fixable_theme_or_render` | hires/cross_cluster_edges/dagua.png at top-right of Z bbox. |
| F11 | MED | `cluster_showcase` | "Outer Cluster" placement collides with central cluster | Outer Cluster bbox visually overlaps the (missing) Large Cluster region — outer-a/outer-b sit inside Outer Cluster but the cluster appears physically positioned where Large Cluster should be. | `real_cosmetic_gap` (but layered on harness artifact) | `needs_layout_scope` (Phase 2 verify); `fixable_theme_or_render` for any bbox-clamp behavior contributing to the overlap. | hires/cluster_showcase/dagua.png. Same harness caveat. |
| F12 | LOW | `flat_many_clusters` | label position: top-left vs top-center on `graphviz_strict` | Sibling cluster labels in dot are top-left ("Alpha", "Beta", "Delta", "Gamma" all anchored at top-left of each cluster). Dagua renders them top-center. DESIGN.md §3.3 noted dot labels are sometimes top-center; closer inspection of multiple panels (cluster_showcase too) confirms dot is consistent top-left for siblings under a parent, top-center for outer. | `real_cosmetic_gap` | `fixable_theme_or_render` | hires/flat_many_clusters: dot puts each cluster label flush-left; Dagua centers all. styles.py / Theme override. |
| F13 | LOW | `nested_clusters` | "Outer Group" label vertical position vs node A | Even after fixing the placement of A: Dagua's "Outer Group" label sits below the cluster top stroke, while dot's sits above the children clusters at y= top + ~15pt. After Phase 2 lands proper placement, the label-band reservation must visually lift the label up to match dot, not just clear A. | `uncertain_needs_targeted_probe` | `fixable_theme_or_render` | Verify after Phase 2 with Dagua's own placement. |
| F14 | LOW | universal | node ellipse stroke weight in cluster panels | All cluster panels show Dagua nodes with **slightly thicker stroke** and **rounder aspect** (closer to circles) than dot's ellipses. Per metrics this is the `ellipse_rx_pt` tail (max delta 13.4pt on long_labels / transformer_block). Not a cluster bug per se. | `metric_or_measurement_artifact` for the cluster audit | `not_actionable` in cluster scope | summary.md "ellipse_rx_pt: max 13.40pt"; out of cluster scope per the audit prompt. |

Severity tallies: 7 HIGH, 4 MED, 3 LOW.

---

## Metric Artifact Review

- All declarative cluster features (`cluster_fill`, `cluster_stroke`,
  `cluster_stroke_width_pt`, `cluster_label_font_size_pt`) report 41/41 in tolerance, yet
  the rendered gallery has multiple HIGH-severity cluster defects. The declarative metric
  EXTRACTOR clearly does not see missing rectangles, label-stroke collisions, or edges
  piercing perimeters. This is the audit-instrument blind spot the user is right to call
  out: parity_metrics.json says 100% on cluster features but the eyes say FAIL.
- `nested_clusters` SSIM 0.693 and L1 23.4 (4th-worst panel by L1) corroborates this —
  the pixel-diff knows something is off.
- The `ellipse_rx_pt` tail (13.4pt on long_labels) is non-cluster and out of scope.
- Pixel-diff `Background L1` dominates total L1 across all panels (text/node/edge regions
  read 0.0 because of mask geometry). That is itself a measurement artifact: anything
  inside a node mask doesn't count, including label-stroke collisions which clearly
  deserve to count. Worth noting for future audit instrumentation.

---

## Rendering-Stack Residuals

- Dot's font hinting and rasterizer create slightly different antialiasing on stroke
  endpoints. Not driving any of the findings above.
- One-pixel sub-pixel rounding around cluster corners — not driving the missing-rectangle
  bug (which is many pixels, not one).
- Bezier routing geometry inherited from Graphviz positions is not Dagua's responsibility
  in this harness mode. F2/F6/F11 are partly explained by this and noted as such.

---

## Recommended Next Fixes

Ranked by user-visible impact, only `real_cosmetic_gap` + `fixable_theme_or_render`:

1. **F1 — investigate why inner clusters render as bare lines on `nested_clusters`.**
   Likely root: `_draw_clusters` annular-path build skips top/bottom edges when child cluster
   width is small relative to stroke_width or when the inner ring inset >= outer inset.
   Code area: `dagua/render/mpl.py:7513-7770` (cluster path generation).
   Test: render `nested_clusters` standalone; compare each cluster's PathPatch vertex count
   to expected (8 for closed roundrect: 4 corners + 4 segments).

2. **F3 — investigate missing "Large Cluster With Longer Label" rectangle on
   `cluster_showcase`.**
   Likely root: when label width > inner bbox width, the min-width clamp + label-fit width
   clamp interact badly and produce a degenerate path (zero-width band). Could be a
   sign-error in the `(min_w - cw)/2` expansion.
   Code area: `dagua/render/mpl.py:_draw_clusters`, label-fit width clamp branch.

3. **F4 — increase cluster label background mask padding and use rasterized-text bbox.**
   The mask should use the actual rendered-text bbox (mpl `Text.get_window_extent`), not
   nominal font-size * char-count. Padding (x, y) should default to (6, 3) pt or larger.
   Code area: `dagua/styles.py` `ClusterStyle.label_background_padding` default; renderer
   bbox computation in `_draw_clusters` label block.

4. **F5 — verify Phase 4 edge clipping is wired for the harness path.**
   The harness uses dot positions, but Phase 4's perimeter clip should still operate on
   the rendered cluster bbox at draw time. Trace one sample edge (e.g.
   `data_pipeline` Source->Parse CSV) end to end through `_draw_edges` to confirm:
   (a) cluster bbox is being computed at edge-draw time;
   (b) the polyline is intersected against it;
   (c) the visible polyline gets trimmed to the intersection.
   Likely failure: clipping operates on the placement-time cluster bbox stored in
   state, which in this harness comes from dot — and the rendered bbox (which adds
   render-padding) is larger, so the placement bbox is fully inside the cluster and
   "the polyline never crosses the perimeter" returns true.
   Code area: `dagua/routing.py` or `dagua/edges.py` Phase-4 clip pass.

5. **F7 — fix cluster top label-band reservation so the empty space at top of each
   cluster matches dot.**
   `_compute_cluster_y_maxes` likely double-counts label_height + label_gap when a child
   cluster also has a label band. Bound the reservation to a single `label_band` per
   cluster, not propagated additively up the tree.
   Code area: `dagua/render/mpl.py:3607-3775`.

6. **F12 — sibling cluster label position should be top-left in graphviz themes.**
   Per the audit re-inspection of dot's behavior: outer cluster gets top-center, sibling
   clusters get top-left. Encode as a depth-conditional default.
   Code area: `dagua/styles.py` Theme default; or `_draw_clusters` label-anchor selection.

7. **F8 — extend Phase 4 clipping to bypass edges that ENTER and EXIT a foreign cluster.**
   Today the spec covers only one-sided crossings (src in, tgt out / src out, tgt in).
   Bypass edges (transformer_block) cross a cluster twice on the same straight line — both
   entries should be clipped (or the routing pushed around).
   Code area: same as F5; expand the case enumeration.

8. **F9 — pad Load cluster bbox to fully contain Data Warehouse on `data_pipeline`.**
   Either render-time padding is too small, or member bbox max-y omits the half-height of
   bottom-row nodes. Check the render bbox for the Load cluster against (`max_y + node_h/2 +
   bottom_padding`).
   Code area: `dagua/render/mpl.py:_compute_cluster_y_mins`.

9. **F10 — close the path on `cross_cluster_edges` Cluster Z (stray fragment).**
   Audit the path-build for the case where a cluster's right side is partially outside
   the figure's clip — likely a vertex emitted but not connected.
   Code area: `dagua/render/mpl.py` cluster path generator.

10. **F11/F2 — re-run audit on Dagua's OWN placement (not harness with dot positions).**
    Most cluster-vs-cluster overlap and member-spillover defects collapse to Phase 2
    correctness which this harness cannot exercise. Add a parallel gallery
    `parity_pixel_diff_dagua_placement/` that runs Dagua's full layout pipeline (no dot
    positions) on the same fixtures.
    Code area: `scripts/graphviz_theme_comparison.py` — add `--use-dagua-placement` flag
    that bypasses the dot-position injection.

---

## Inspection Log

For each panel: nodes, labels, edges, arrowheads, cluster borders + labels + masks, and
worst-metric regions inspected.

- **`nested_clusters`** (SSIM 0.693, L1 23.4): outer cluster border ✓ drawn; outer
  label ✓ visible but BAD position (overlaps node A); inner left/right cluster borders
  MISSING (F1); inner labels float; A->C edge crosses outer top un-clipped (F5); F->E,F->D
  edges similarly cross outer bottom un-clipped; A penetrates outer top (F2); ellipses
  thicker than dot (F14).
- **`cluster_showcase`** (SSIM 0.818, L1 13.7): Tiny Cluster + Medium Cluster + Outer
  Cluster + Nested Inner ✓ drawn; "Large Cluster With Longer Label" rectangle MISSING
  (F3); label position varies; Outer Cluster overlaps Large Cluster region (F11); cross-
  cluster edges (medium 1->medium 2->large 0, large 0->large 5, etc) cross perimeters
  un-clipped.
- **`transformer_block`** (SSIM 0.851, L1 11.4): both MHA and FFN clusters ✓ drawn with
  borders; "Multi-Head Attention" label sits on top stroke with mild collision (F4); FFN
  same; bypass Input Embedding -> Add edges cross MHA right edge twice (F8); Add->FFN
  LayerNorm and the FFN bypass into second Add cross FFN top/bottom edges (F8); inner
  layout OK (driven by dot positions).
- **`cross_cluster_edges`** (SSIM 0.717, L1 20.5): three cluster borders drawn but
  intersect each other (F6); Cluster X has massive empty band above X1 (F7); Y3/Y4 inside
  Cluster Z bbox (containment fail, F6); stray stroke fragment near Z top-right (F10);
  cluster labels visible.
- **`deep_nesting_4`** (SSIM 0.733, L1 18.2): all 4 levels drawn as concentric rectangles
  (good!); each level label collides with top stroke (F4); Source->Outer 1 edge enters
  Level 1 un-clipped (F5); Mid 2->Inner 1 enters Level 3 un-clipped (F5); Core->Exit exits
  Level 4 un-clipped (F5); top label band ≈ 1.5x the dot equivalent (F7).
- **`flat_many_clusters`** (SSIM 0.744, L1 18.8): four sibling clusters Alpha/Beta/Gamma/
  Delta drawn but visually overlap each other (F6 / harness artifact); B4 outside Beta
  bottom (F6); D4/D5 outside Delta bottom (F6); labels top-center (should be top-left,
  F12); cluster top-stroke bisects all four labels (F4); Source->{A1,B1,C1,D1} edges
  cross Alpha top un-clipped (F5).
- **`microservices`** (SSIM 0.850, L1 11.9): all four sibling clusters drawn; Service /
  Data / Worker layers visibly overlap (F6 / harness); cluster labels collide with stroke
  on long names (F4); cross-layer edges (Search Service->Order DB etc) cross multiple
  cluster borders un-clipped.
- **`data_pipeline`** (SSIM 0.755, L1 18.5): Extract / Transform / Load all drawn (good!);
  CSV/API/DB Source ->{Parse CSV, Fetch API, Query DB} arrowheads land INSIDE Extract
  bbox un-clipped (F5); Extract members -> Transform Validate edges cross both bottom and
  top borders un-clipped (F5); Data Warehouse partially below Load bottom (F9); cluster
  labels readable but Extract/Transform show stroke through long labels (F4).

Pixel-diff reality check: cluster panels rank from best (transformer_block, microservices
SSIM 0.85) to worst (nested_clusters SSIM 0.69). Not a single cluster panel is in the
top-quartile of SSIM (0.84+) and the four panels with HIGH-severity render defects all sit
below 0.75 SSIM. The pixel-diff agrees: cluster output is the visual quality bottleneck.

---

## Honest answer to user's bar

> "look at least as nice as graphviz and its a CORE FEATURE."

No. Not yet. The renders are **clearly worse than graphviz** on every cluster panel
inspected. Specifically:
- 2 panels have **missing cluster rectangles** that dot draws cleanly.
- All 8 panels show some flavor of label-stroke collision dot does not.
- 0 panels show evidence of edge perimeter-clipping working — in every case where dot
  produces a clean perimeter intersection, Dagua produces an arrowhead inside the cluster
  or a stroke through the cluster boundary.
- 4 panels show sibling cluster overlap or member spillover; even granting the harness
  artifact for placement, the bbox-padding-side is Dagua's and is contributing.

The sprint advanced the architecture (Phase 2's cluster-as-node driver is a real win for
Dagua-native placement) but the **render-side parity gates (Phase 3, Phase 4) have visible
gaps** that need a follow-up round before the user's bar of "bulletproof, at least as nice
as graphviz" is met.

CONTINUE. Specific next-round work in "Recommended Next Fixes" section above (10 items;
top 5 are the bulletproof-blockers).
