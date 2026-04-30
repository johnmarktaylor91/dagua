# Visual Audit Round A2 — Graphviz Strict Parity

Auditor: Opus 4.7 (1M ctx)
Date: 2026-04-27
Scope: cosmetic only (`graphviz_strict` theme + render). `dagua/layout/` OUT OF SCOPE.
Inputs: declarative metrics, pixel-diff summary, 8 hi-res panel pairs (16 PNGs).
Round: A2 (post-B1 commit `9c14892`).

---

## Verdict

- Prior items (A1 F1-F5): `PARTIAL` — F1 partially solved (margin OK, figure
  aspect still leaves whitespace bands on tall content), F2 fully solved
  (long-label wrap fixed), F3 partially solved (single-line OK, long-label
  rx still narrows, max -13.4pt), F4 partially solved (`arrowsize` plumbed
  for `EdgeStyle` but not propagated to per-edge arrowsize attribute parsed
  from DOT; arrow shape regression introduced; edge-label font over-corrected
  smaller than dot), F5 untouched (layout-scope, by design).
- New audit: `FAIL` — multiple new regressions discovered.
- Stop criteria status: `CONTINUE` — many findings classified as
  `real_cosmetic_gap` + `fixable_theme_or_render`. SSIM regression has a
  concrete diagnosis (see "SSIM Regression Diagnosis" below).

---

## SSIM Regression Diagnosis (critical context for round B2)

Mean SSIM dropped from 0.7716 to 0.7615 (-0.010) and worst SSIM dropped
from 0.585 to 0.529 despite an L1 mean improvement. The pattern in the
hi-res images explains this cleanly.

**Two compounding causes:**

1. **Figure-aspect mismatch on tall narrow content** (`pipeline`,
   `colors_showcase`, `tiny_graph`, `ladder`, `single_edge`,
   `nested_clusters`). dot generates a tightly-cropped PNG whose width
   matches the natural content width. Dagua's strict canvas, after the B1
   margin/aspect="auto" changes, now keeps `bbox_inches=False` but appears
   to size the figure to a different aspect ratio than dot's, leaving large
   left/right white bands on tall-narrow panels (clearly visible in
   `pipeline.dagua.png`: ~25% empty band on each side; in
   `single_edge.dagua.png`: similar). dot panels have minimal horizontal
   padding. For SSIM, two images of different actual aspect ratio with
   matched content but different background coverage produce low structural
   similarity scores even when L1 is OK, because SSIM's local-window
   covariance term flips sign in the empty bands.

2. **Aspect cap removal applied broadly** (B1's F2/F3 fix). Short single-line
   labels like `In/Mid/Out` (`tiny_graph`), `Source/Sink` (`single_edge`),
   `Red/Blue/Green` (`colors_showcase`) now render as near-CIRCULAR ellipses
   in dagua, while dot keeps them as wide ovals (~2:1 aspect). This
   structural shape mismatch dominates SSIM on small panels. The B1 1.28x
   single-line factor is not enough to recover the dot aspect when the
   underlying min-height was raised by removing the aspect cap. Compare:
   - `tiny_graph` dot Mid: ~2:1 aspect; dagua Mid: ~1.2:1 aspect.
   - `single_edge` dot Source: ~2.1:1; dagua Source: ~2.1:1 BUT visibly
     taller due to ry not being capped; the bottom curve is misshapen.

The fact that L1 improved while SSIM worsened is consistent: the canvas-fill
fix moved the *content* in the right direction (median pixel intensity
delta closer), but the figure-shape and ellipse-aspect mismatches are
structural and SSIM-dominant.

**Implication for B2:** prioritize fixing the figure-aspect (so dot's
content extents and dagua's content extents fill the same fraction of the
canvas in BOTH x and y) and re-introduce a wider single-line ellipse aspect
without re-introducing the long-label wrap defect.

---

## Prior Item Recheck

| Item | Verdict | Evidence | Notes |
| --- | --- | --- | --- |
| F1 Canvas fill (margin + figure size) | PARTIAL | `parity_metrics_summary.md` row `margin_pt` is 45/45 in tol with 0pt delta. Hi-res `pipeline.dagua.png`, `tiny_graph.dagua.png` show large left/right white bands. | Margin solved; figure aspect/letterboxing not solved on tall-narrow content. |
| F2 Long-label wrap defect | PASS | `parity_metrics_summary.md` `ellipse_ry_pt` 486/487 in tol. `long_labels.n3` no longer multi-line wrapped in metrics; max ry delta 2.76pt. | Wrap defect cleared. |
| F3 Ellipse-rx narrowing | PARTIAL | `ellipse_rx_pt` improved 349->464/487 in tol, but 23 long-label nodes still OOT all-negative; max delta -13.40pt. | 1.28x factor handles short single-line labels but undershoots on labels >=10 chars in `long_labels`, `transformer_block`, `microservices`, `data_pipeline`, `neural_net`, `label_variety`. |
| F4 Arrow defects (size, shape, edge label) | PARTIAL/REGRESSED | `parity_metrics.json` arrow_types e1/e5/e7 still arrow_width -3.46pt; e4 now +3.08pt (over-correct). Hi-res `arrow_types.dagua.png` shows shrunken/diamond arrowheads, edge-labels SMALLER than dot (over-correction from 14pt-too-big -> 11pt-too-small). | `EdgeStyle.arrowsize` was added but per-edge `arrowsize` from DOT source is still ignored; arrow shape primitives now look like flat diamonds rather than triangles; edge-label font over-corrected from too-large to too-small. |
| F5 Nested-cluster geometry | UNCHANGED | Hi-res `nested_clusters.dagua.png` still shows label intrusion, sibling overlap, A protruding out of cluster, F outside cluster. | Layout-scope; out of cosmetic-sprint scope by design. |

---

## New Findings

| Severity | Panel | Element/Region | Finding | Finding Class | Actionability | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| HIGH | all 8 audited panels (esp. `pipeline`, `colors_showcase`, `tiny_graph`, `single_edge`, `nested_clusters`, `ladder`) | figure aspect / outer canvas | Dagua's strict figure now has the right margin (0pt delta) but the figure aspect ratio does NOT match dot. Tall-narrow content (vertical chains) renders with substantial left/right white bands (~20-30% of width) in dagua, while dot crops the figure to the content's natural aspect. This is the dominant SSIM regressor for round B1->B2. Likely cause: the B1 switch to `aspect="auto"` plus disabling `bbox_inches="tight"` keeps the matplotlib default figure size in inches even though the dot reference figure is sized to `(content_width + 2*margin) x (content_height + 2*margin)` with native PT->px. | `real_cosmetic_gap` | `fixable_theme_or_render` | Hi-res `pipeline.dot.png` vs `pipeline.dagua.png` (dagua canvas is roughly 1.6x wider than content, dot is roughly 1.05x). Same pattern in `single_edge`, `tiny_graph`, `nested_clusters`, `ladder`, `colors_showcase`. Pixel-diff `summary.md` mean SSIM 0.7615 with worst SSIM 0.529 on `bipartite_5x5`. |
| HIGH | `arrow_types`, `single_edge`, `tiny_graph`, `colors_showcase`, `bipartite_5x5`, `ladder`, `nested_clusters`, `pipeline` (every panel with arrows) | every arrowhead | Arrowhead shape primitive REGRESSED. Dagua now draws what looks like a small/squat filled DIAMOND (4-vertex rhombus) where dot draws a clean filled isoceles TRIANGLE. Visibly half the height and noticeably shorter than dot's. The change is consistent across panels — it's not per-shape; it's the default `normal` arrowhead path. Effect: every arrow looks "stubby" and "wedge-like" in dagua. SSIM penalty on every panel with arrows. | `real_cosmetic_gap` | `fixable_theme_or_render` | Hi-res `arrow_types.dagua.png` (compare `normal` arrow at far left), `single_edge.dagua.png`, `colors_showcase.dagua.png`, `pipeline.dagua.png`, `tiny_graph.dagua.png`, `ladder.dagua.png`, `nested_clusters.dagua.png`. Apparent height of dagua's `normal` arrow is ~50-60% of dot's; apparent shape is rhombus not triangle. The `EdgeStyle.arrowsize` change in B1 likely reduced the geometry without fixing the polygon vertices. |
| HIGH | `arrow_types` | edges e1, e5, e7 (`vee`, `crow`, `circle`) and e4 (`tee`) | Per-edge `arrowsize` attribute is still NOT respected. e1, e5, e7 have target arrow_width 10.46pt (DOT source `arrowsize=1.5`); dagua emits 7.0pt (delta -3.46). e4 has target 3.92pt (DOT source `arrowsize<1.0`); dagua emits 7.0pt (delta +3.08, OVER). Dagua appears to use a single fixed arrow width (7.0pt) for all per-edge variants. The B1 `EdgeStyle.arrowsize` attribute exists but is not being populated from the DOT-source per-edge `arrowsize` attribute. | `real_cosmetic_gap` | `fixable_theme_or_render` | `parity_metrics.json` arrow_types e1 -3.46, e5 -3.46, e7 -3.46, e4 +3.08, e3 arrow_length -2.0, e7 arrow_filled False vs True. |
| HIGH | `arrow_types` | all 9 edge labels | Edge labels are now visibly SMALLER than dot's edge labels — overcorrected from A1's "too large/bold" to "too small". B1 set graphviz_strict edge-label font to 11pt; dot's edge labels appear to be 14pt (same as node labels) at this hi-res scale. Visual effect: edge labels in dagua are ~20-25% shorter than dot's. | `real_cosmetic_gap` | `fixable_theme_or_render` | Hi-res `arrow_types.dot.png` vs `arrow_types.dagua.png`. Each of `normal`, `vee`, `dot`, `diamond`, `tee`, `crow`, `circle`, `open`, `none` labels is visibly smaller in dagua. (Confirms A1 finding #6 measurement gap: edge-label font size is still not in the parity metric, so the overcorrection landed without metric pushback.) |
| HIGH | `tiny_graph`, `single_edge`, `colors_showcase` (Red/Blue/Green), `bipartite_5x5` (all 10 nodes), `arrow_types` (all 18 nodes) | every short single-line ellipse | Ellipse aspect is too round for short single-line labels. Dot draws short labels (`In`, `Mid`, `Out`, `Source`, `Sink`, `Red`, `Blue`, `Green`, `L1-L5`, `R1-R5`, `normal`, `vee`, `dot`, etc.) as wide ovals at ~2:1 aspect (rx ~ 2*ry). Dagua draws them at ~1.2-1.5:1, visibly more circular. The B1 1.28x rx factor was applied to single-line labels but the underlying ry minimum (after aspect-cap removal) appears to have grown, producing rounder ellipses. The metric doesn't catch this because most short-label ellipses have low absolute rx and are now passing tolerance. | `real_cosmetic_gap` | `fixable_theme_or_render` | Hi-res `tiny_graph.dot.png` vs `tiny_graph.dagua.png` (very obvious — dagua's Mid is almost a circle). Same in `single_edge.dagua.png`, `colors_showcase.dagua.png` (Red/Blue/Green). |
| HIGH | `long_labels`, `transformer_block`, `microservices`, `data_pipeline`, `neural_net`, `label_variety` | every long-label ellipse (23 nodes) | Long-label ellipses (>=10 chars or so) are still narrower than dot. All 23 OOT entries have negative `ellipse_rx_pt` delta; max -13.40pt on `long_labels.n4` (`MultiHeadAttention`). The current 1.28x single-line factor undershoots for long labels — the gap scales with label length, suggesting a per-character padding constant that doesn't keep up. Suggests two padding regimes are needed: short labels and long labels, OR the factor should scale with label-glyph-count rather than be fixed. | `real_cosmetic_gap` | `fixable_theme_or_render` | `parity_metrics_summary.md` Top-10 worst-delta features rows 1-10 (all `ellipse_rx_pt`, all in `long_labels` or `transformer_block`). Out-of-tol list shows linear correlation between label length and deficit. |
| HIGH | `nested_clusters`, `cross_cluster_edges`, `flat_many_clusters`, `cluster_showcase` | cluster bounding boxes and labels | Cluster geometry remains broken (A1 finding #7 still 100% present). Hi-res `nested_clusters.dagua.png`: A node protrudes out the top of `Outer Group`; `Right Branch` and `Left Branch` rectangles share a centerline (no gutter); cluster labels overlap node ellipses; F sits outside the outer cluster bottom edge. Per the sprint scope, this is layout-scope and should not be fixed in cosmetic round B2 — but if any sub-issue is render-time padding (label-clear region, label inset from box top) it COULD be fixed cosmetically. | `uncertain_needs_targeted_probe` | `needs_layout_scope` (most), `fixable_theme_or_render` (label inset/clear-area only) | Hi-res `nested_clusters.dot.png` vs `nested_clusters.dagua.png`. Pattern repeats in `cross_cluster_edges`, `flat_many_clusters`, `cluster_showcase` per pixel-diff worst-list. |
| MED | `single_edge`, `tiny_graph`, `colors_showcase` (Red, Blue) | top arc of ellipse near label | Visible "double-stroke" / segmented look on the top arc of small ellipses in dagua. Likely a Bezier/spline approximation step issue at small scale (matplotlib `Ellipse` patch vs dot's Cairo elliptic arc). Looks like a 1-2px discontinuity on the upper-right of the arc. Subtle but visible at hi-res. | `real_cosmetic_gap` | `fixable_theme_or_render` (or possibly `rendering_stack_residual`) | Hi-res `single_edge.dagua.png` (Source ellipse top-right arc), `tiny_graph.dagua.png` (In/Mid/Out top arcs), `colors_showcase.dagua.png` (Red top arc). Reduces with size; not visible on `transformer_block` larger ellipses. |
| MED | `bipartite_5x5` | leftmost edge L4->R1, rightmost edge L3->R5 | Dagua draws these "outer" edges as CURVED arcs (slight S-bend going outside the column), while dot draws them as straight diagonals. This is a different diagnosis than A1 #8 (ladder) — in `bipartite_5x5` the curving is far more pronounced and clearly outside the column band. Likely Graphviz B-spline routing not being matched by dagua's edge spline interpretation. May or may not be cosmetic depending on whether dagua receives Graphviz-emitted control points. | `uncertain_needs_targeted_probe` | possibly `fixable_theme_or_render` (if dagua is interpreting splines), possibly `rendering_stack_residual` (if it's pure Graphviz routing) | Hi-res `bipartite_5x5.dot.png` (all edges straight) vs `bipartite_5x5.dagua.png` (L4->R1 and L3->R5 visibly arc-curved). |
| MED | `arrow_types` | all 9 source-to-target edge strokes | Edge stroke between source and target ellipses appears LIGHTER GRAY in dagua, while in dot they are SOLID BLACK. This is the same issue A1 finding #5c flagged — declared `edge_stroke_color` 100% in tolerance but visibly lighter in render. May be alpha-blending or stroke-width interaction with anti-aliasing at this scale. | `uncertain_needs_targeted_probe` (declarative metric reports identity; visible difference) | `fixable_theme_or_render` if stroke alpha is implicit; possibly `rendering_stack_residual` if pure AA. | Hi-res `arrow_types.dagua.png` edges look gray; dot's edges look black. |
| LOW | `nested_clusters`, `cluster_showcase` | cluster border rectangle stroke style | Cluster borders look SLIGHTLY thinner in dagua than in dot. Declarative `cluster_stroke_width_pt` is reported 100% in tolerance with delta 0.0, so this may be AA residual at the `1.0pt` line weight. | `metric_or_measurement_artifact` (likely AA) | `rendering_stack_residual` | Hi-res `nested_clusters.dot.png` vs `.dagua.png`. |

Severity scale used:
- HIGH: obvious at normal zoom or affects core Graphviz parity, or impacts many panels.
- MED: visible at full resolution and likely fixable.
- LOW: subtle but real; may be AA residual.

---

## Metric Artifact Review

The same per-region pixel-diff mask artifact persists from A1: every panel
reports Text/Node/Edge L1 = 0.0000 with all the L1 in Background. This
means the per-region mask is computed from one image's mask rather than the
union of both. Implication: the per-region table is currently a no-op for
diagnosing where dagua differs from dot. Recommended fix (out of scope for
B2): use mask = union(dot_mask, dagua_mask) so per-region L1 reflects
content placement differences. Without this fix, the per-region table will
continue to mislead future audits.

`font_size_pt` still reports 487/487 in tolerance, equal to node count;
edge-label font size is still not measured. This let B1's edge-label
font-size over-correction (14pt -> 11pt) land without metric feedback.
Adding edge-label font-size to the parity metric should be a B2
prerequisite for any future edge-label change.

`margin_pt` 100% tolerance with delta 0.0 — B1's tightening fixed the
metric AND the tolerance. Good.

---

## Rendering-Stack Residuals

These are real differences but should NOT drive a theme/render fix in this
sprint:

- **Sub-pixel anti-aliasing differences** between matplotlib's
  `patches.Ellipse` rasterizer and Cairo's elliptic arc on small ellipses
  (drives finding F8 in this round). Visible only at hi-res zoom, not at
  normal viewing.
- **Ladder diagonal cross-edge slight bow** (A1 finding #8): unchanged.
  Likely B-spline routing residual from Graphviz; needs a targeted probe of
  whether dagua receives spline control points or computes its own.
- **Times,serif font hinting differences** between Cairo and matplotlib's
  text path: still not visible at the level of changing apparent letter
  weight or position.

---

## Recommended Next Fixes

Ranked by impact on overall pixel parity (SSIM/L1) and Graphviz fidelity:

### Rank 1 — Figure aspect ratio match (canvas-fill phase 2)

This is the dominant SSIM regressor for B2. The B1 fix tightened margin
but the figure-INCHES sizing still doesn't match dot's content-aspect.
Required behavior: figure size = (graph_bbox_pt + 2*margin) converted to
inches at 72 DPI; matplotlib axes set to match content extent in pt with
`aspect="equal"` (NOT `"auto"`); savefig with `pad_inches=0`; ensure no
implicit padding from rcParams. Numeric target: dagua's PNG width/height
should match dot's PNG width/height to within +/- 2px after the fix.
Likely code area: `dagua/render/mpl.py` figure-size computation and
`fig.savefig` kwargs.

### Rank 2 — Arrowhead polygon shape (rhombus -> triangle)

Every arrow on every panel is currently a stubby diamond. Investigate the
B1 change to arrow geometry (likely the `EdgeStyle.arrowsize` plumbing
into render marker sizing). The marker primitive itself needs to be a
4-vertex isoceles triangle (tip, two base corners, return-to-tip), not a
4-vertex diamond. Likely code area: `dagua/render/edges/` arrow-head
draw routine. Numeric target: `arrow_length_pt` and `arrow_width_pt` in
`parity_metrics.json` for the `normal` shape (which is currently in
tolerance) PLUS a visual confirmation pass. The metric is necessary but
not sufficient — the shape's vertex positions need to match dot's.

### Rank 3 — Per-edge `arrowsize` attribute parsing (A1 F4a residual)

`arrow_types` still has 4 OOT entries on `arrow_width_pt` because per-edge
`arrowsize` from the DOT source is not being threaded into
`EdgeStyle.arrowsize`. Code area: DOT-attr ingestion in graphviz_strict
theme path. The `EdgeStyle.arrowsize` field added by B1 needs to be
populated from per-edge attrs, not just the global theme default.

### Rank 4 — Single-line ellipse aspect (re-introduce wider oval)

Short single-line labels are now too circular. Two fix options:
(a) raise the rx factor from 1.28 to ~1.45 for single-line labels with
length <= 6 chars (would re-widen `In`/`Out`/`Red` cases),
(b) re-introduce a SOFT aspect cap that ensures rx/ry >= 1.8 for
single-line labels but does NOT clamp ry on long labels (so `MultiHead-
Attention` is unaffected).
Option (b) is safer; option (a) is faster. Avoid re-breaking F2.

### Rank 5 — Long-label ellipse rx scaling

23 long-label nodes still narrow by up to 13.4pt. Suggest a label-length-
dependent factor: rx_factor = 1.28 + 0.005 * max(0, label_chars - 8)
or similar. Calibrate against the 23 OOT deltas — they should converge
with one well-chosen formula. Code area: same single-line factor
location in graphviz_strict theme.

### Rank 6 — Edge-label font size (correct over-correction)

B1 set edge-label font to 11pt; dot uses 14pt (matches node-label font).
Set edge-label font back to 14pt for graphviz_strict, OR add per-attr
support if Graphviz docs distinguish edge-label vs node-label defaults
(it does NOT — both are graph-default fontsize). Pre-requisite: extend
parity metric to include edge-label `font_size_pt` so this can be
measured.

### Rank 7 — Edge-stroke color/alpha investigation

`arrow_types` edge strokes look gray vs dot's black. Verify dagua's edge
draw path uses solid black (no alpha). May be linewidth-vs-aa interaction.
Quick probe: render `arrow_types` at 4x DPI and confirm whether the
"gray" stroke is genuine RGB or pure AA.

### Rank 8 — Per-region pixel-diff mask (measurement infra)

Out of cosmetic scope, but blocking future audit fidelity. Change the
per-region mask in `scripts/parity_pixel_diff.py` to be the union of dot
and dagua element masks so the per-region L1 numbers become trustworthy.

### Rank 9 — Edge-label font-size in parity metric (measurement infra)

Add edge-label `font_size_pt` measurement to `scripts/parity_metrics.py`
so future audits don't lose visibility into this dimension.

---

## STOP Criterion Status

`CONTINUE`. Multiple HIGH-severity findings classified as
`real_cosmetic_gap` + `fixable_theme_or_render`:
- F1 figure aspect (the SSIM regressor)
- F2 arrowhead shape (every panel affected)
- F3 per-edge arrowsize parsing (4 OOT in arrow_types)
- F4 edge-label font over-correction
- F5 single-line ellipse aspect (every short-label node)
- F6 long-label ellipse rx scaling
- F8 small-ellipse double-stroke (subtle, MED)

STOP would require zero `real_cosmetic_gap` + `fixable_theme_or_render`
findings. We have at least 6 HIGH-severity, all actionable. STOP =
NO.

---

## Inspection Log

For each of the 8 hi-res panel pairs (16 PNGs read end-to-end), what I
inspected and what I concluded:

### `bipartite_5x5` (SSIM 0.529, worst overall)

Inspected: 5 top L-nodes (L1-L5), 5 bottom R-nodes (R1-R5), 10 vertical
down-edges, 10 diagonal cross-edges, all 10 arrowheads, 10 node label
glyphs, canvas extents, edge curvature.
Result: Outer edges (L4->R1, L3->R5) are curved S-arcs in dagua but
straight in dot — drives finding F9 (uncertain). Arrowheads are
flat-diamond shape vs dot's triangles — drives finding F2. Nodes look
reasonable in aspect but slightly tighter / less stretched than dot (R1-R5
in dagua are noticeably narrower vs dot). Drives F5.

### `arrow_types` (SSIM 0.627)

Inspected: 9 source ellipses (`normal`, `vee`, `dot`, `diamond`, `tee`,
`crow`, `circle`, `open`, `none`), 9 target ellipses (all `target`), 9
edges, 9 arrowheads, 9 edge labels, 9 source-target connection strokes.
Result: every arrowhead is shrunken/squashed (F2). Three arrows still have
wrong absolute width from per-edge `arrowsize` (e1, e5, e7) — drives F3.
Edge labels are smaller than dot's — drives F4. Edge strokes appear gray
not black — drives F10. Source ellipses are also somewhat circular (F5).

### `ladder` (SSIM 0.642)

Inspected: A1-A6, B1-B6 (12 ellipses), all 17 edges, arrowheads, label
glyphs, vertical spacing, diagonal cross-edge curvature, canvas extents.
Result: dagua canvas is wider than content (F1). Diagonals are slightly
bowed (A1 #8 unchanged — rendering-stack residual). Arrowheads are
flat/diamond-shape (F2). Ellipses look noticeably more circular than dot's
(F5). Overall scale of dagua's content within the canvas is smaller than
dot's content within its canvas.

### `single_edge` (SSIM 0.652)

Inspected: 2 ellipses (Source, Sink), 1 edge, 1 arrowhead, 2 label glyphs,
canvas extents.
Result: large left/right white bands (F1). Source ellipse top arc has
visible double-stroke artifact (F8). Source/Sink ellipses look more
circular than dot's wide ovals (F5). Arrowhead is flat-diamond (F2).

### `tiny_graph` (SSIM 0.679)

Inspected: 3 ellipses (In, Mid, Out), 2 edges, 2 arrowheads, label glyphs.
Result: extreme example of F5 — all three ellipses are nearly circular in
dagua, while dot's are wide ovals. Two short-arrows are flat-diamonds
(F2). Canvas has substantial left/right white band (F1). Top arcs of all
three ellipses show subtle double-stroke (F8). This is a worst-case for
short-label aspect mismatch.

### `nested_clusters` (SSIM 0.688)

Inspected: outer cluster border, inner Right Branch border, inner Left
Branch border, all 3 cluster labels, A/B/C/D/E/F nodes, all edges,
arrowheads.
Result: cluster geometry still broken (F7 / A1 #7 unchanged). Cluster
borders look thinner (LOW finding F11). Arrowheads are flat-diamonds
(F2). Cluster labels overlap content (cluster label inset is wrong).
A protrudes top of outer cluster, F sits below outer cluster — layout
problems, out of cosmetic scope.

### `colors_showcase` (SSIM 0.714)

Inspected: 6 colored ellipses, 5 edges, 5 arrowheads, 6 label glyphs,
vertical chain spacing, canvas extents.
Result: large left/right white bands (F1). Red, Blue, Green are noticeably
more circular in dagua (F5). Arrowheads are flat-diamonds (F2). Yellow,
Purple, Orange match aspect better (likely longer label width helps).
Color fills, strokes, and label glyphs match — colors are NOT a finding.

### `pipeline` (SSIM 0.718)

Inspected: 5 ellipses (Input, Preprocess, Transform, Postprocess, Output),
4 edges, 4 arrowheads, 5 label glyphs, canvas extents.
Result: large left/right white bands — most extreme F1 example.
Arrowheads are flat-diamonds (F2). Short-label ellipses (Input, Output)
are visibly more circular than dot's (F5). `Preprocess`, `Transform`,
`Postprocess` (longer labels) match aspect better. Vertical chain spacing
matches.

### Declarative-metric scan (no images)

Inspected the full out-of-tolerance list (54 OOT entries):
- 24 `ellipse_aspect_pct` OOT, 22 negative + 2 positive (mixed_styles n0
  and n3 are positive — these are the only "dagua too wide" cases).
- 23 `ellipse_rx_pt` OOT, 22 negative + 1 positive (mixed_styles n0 again).
- 4 `arrow_width_pt` OOT, 3 negative + 1 positive (e4 is the positive
  outlier — dagua wider than dot at 7.0 vs target 3.92pt; suggests dagua
  flatlines at 7.0 regardless of per-edge arrowsize attribute).
- 1 `arrow_length_pt` OOT (e3 -2.0pt).
- 1 `arrow_filled` OOT (e7).
- 1 `ellipse_ry_pt` OOT (long_labels n5 -2.76).

The mixed_styles n0/n3 positive deltas are a separate small bug worth
noting: dagua over-widens labels with non-default `width=` or `height=`
attributes (mixed_styles uses these). Not in the worst-10 list, but worth
checking the per-attr ingestion path next round.

---

## Notes for Round B2

- Do F1 (figure aspect) FIRST. It's the SSIM regressor and is independent
  of the other fixes. Predict: SSIM jumps 0.05-0.10 on the worst panels.
- Then F2 (arrowhead polygon shape) — broad SSIM impact.
- Then F5 (re-widen single-line ellipses) — paired with F6 (long-label rx
  scaling) — tune both factors so no panel regresses.
- F3 (per-edge arrowsize) is a measurement-driven local fix.
- F4 (edge-label font) needs an edge-label font_size measurement first; if
  added, the fix is trivial.
- Cluster geometry findings (F7) remain layout-scope per sprint scope; do
  NOT divert cosmetic effort to cluster bbox sizing in B2.
- Add per-region pixel-diff mask UNION fix as round-B3 measurement infra
  (not blocking cosmetic work).
