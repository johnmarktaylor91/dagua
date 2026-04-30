# Visual Audit Round A1 — Graphviz Strict Parity

Auditor: Opus 4.7 (1M ctx)
Date: 2026-04-27
Scope: cosmetic only (`graphviz_strict` theme + render). `dagua/layout/` OUT OF SCOPE.
Inputs: see prompt; 7 hi-res panel pairs read in full; metric/diff Markdowns scanned.

---

## Verdict

- Prior items: `N/A` (this is round A1, no prior findings).
- New audit: `FAIL`
- Stop criteria status: `CONTINUE` — multiple findings classified as
  `real_cosmetic_gap` + `fixable_theme_or_render` (canvas fill, multi-line label
  wrapping, ellipse rx padding, custom arrow size, edge label font weight/size,
  edge stroke darkness in `arrow_types`, cluster label/box geometry).

---

## Prior Item Recheck

| Item | Verdict | Evidence | Notes |
| --- | --- | --- | --- |
| (none) | N/A | — | Round A1, no prior findings to re-check. |

---

## New Findings

| Severity | Panel | Element/Region | Finding | Finding Class | Actionability | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| HIGH | all worst panels (`bipartite_5x5`, `ladder`, `tiny_graph`, `colors_showcase`, `pipeline`, `arrow_types`, `nested_clusters`, `single_edge`, etc.) | full canvas (background) | Dagua's rendered graph drawing does NOT scale to fill the canvas the way `dot` does. `dot` produces a tightly cropped image where the graph occupies the whole figure (with native Graphviz margin); Dagua emits the same logical content in the same relative positions but on a larger canvas, leaving substantial asymmetric whitespace (e.g. ~25-30% empty band at the bottom of `colors_showcase` and `pipeline`; large left/right padding in `bipartite_5x5`; sparse-graph panels are worst-hit). This is exactly why the per-region pixel-diff shows `Text/Node/Edge L1 = 0.0000` for every panel while `Background L1 = total L1` — the shapes that exist are pixel-aligned, but a large fraction of the dagua canvas where dot draws content is white background. | `real_cosmetic_gap` | `fixable_theme_or_render` | `eval_output/parity_pixel_diff/summary.md` per-panel region table (Text/Node/Edge L1 all 0.0000, Background L1 = total L1, all 45 panels). Hi-res pairs: `parity_pixel_diff/hires/{tiny_graph,colors_showcase,pipeline,bipartite_5x5,ladder,arrow_types}/{dot,dagua}.png`. |
| HIGH | `long_labels` | node `n3` | Dagua wraps the label of `n3` to multiple lines (`ellipse_ry_pt` 18.00 -> 58.47, +40.47pt; `ellipse_rx_pt` 215.26 -> 204.65, -10.6pt). `dot` keeps the long label on a single line and produces an extreme-aspect oblong; Dagua reflows it into a roughly 3-line stack. This is the single largest declarative parity miss in the whole sweep. | `real_cosmetic_gap` | `fixable_theme_or_render` | `parity_metrics.json` panel `long_labels` -> `nodes` -> `n3.ellipse_ry_pt.delta = +40.4701`, `ellipse_aspect_pct.delta = -8.4589`. Same panel: n4 rx -13.40, n5 rx -12.88, n6 rx -7.70 (all narrower in dagua even when not wrapped). |
| HIGH | `label_variety` | node `n7` | Same wrapping defect: `ellipse_ry_pt` 18.00 -> 39.11 (+21.11pt) plus `ellipse_aspect_pct` -4.38. Dagua wraps a label that `dot` keeps single-line. Indicates a width-budget / wrap-threshold mismatch in dagua's label measurement when the label contains explicit `\n`, HTML `<BR/>`, or simply when the label exceeds an internal soft cap. | `real_cosmetic_gap` | `fixable_theme_or_render` | `parity_metrics.json` panel `label_variety` -> `n7` ry +21.11, aspect -4.38. Co-occurs with rx -4.98 in same node. |
| HIGH | every panel with single-line labels (138 nodes total) | all ellipse nodes | Systematic narrowing: `ellipse_rx_pt` is out of tolerance for 138/487 nodes (28.3%) and EVERY out-of-tolerance delta is negative. Mean delta -3.15pt, max -13.4pt. Co-located with `ellipse_aspect_pct` 167/487 OOT, ALL negative. The pattern is "Dagua nodes are narrower than dot for the same label text". This is a label-padding constant or text-width-measurement difference, not a one-off bug — it scales with label length (longer labels show larger absolute miss in `transformer_block`, `long_labels`, `label_variety`). | `real_cosmetic_gap` | `fixable_theme_or_render` | `parity_metrics.json` aggregate by feature: `ellipse_rx_pt` count=138 mean=-3.15 min=-13.40 max=-2.04. Visible as smaller chains in `colors_showcase` (each colored ellipse is visibly narrower than dot's). |
| HIGH | `arrow_types` | edges `e1` (vee), `e5` (crow), and the `dot`/`circle`/`open` arrow shapes | Two distinct cosmetic defects, both visible at full resolution: (a) Edges with explicit `arrowsize=1.5` (or similar): `arrow_width_pt` target 10.46 vs dagua 7.0 (delta -3.46) — Dagua ignores the per-edge `arrowsize` attribute and uses its default. Same delta for two edges -> consistent. (b) Arrow head shapes for `circle`, `dot`, `open` are visibly mismatched in shape, fill, and stroke weight. `dot` panel: open V triangle for `open`; thin ring for `circle`; small filled disc for `dot`. Dagua: filled diamond shape for `open`; ring of clearly heavier stroke for `circle`; same disc for `dot`. (c) Edge stroke between source and target ellipse is rendered in a noticeably lighter gray in dagua's `arrow_types` panel vs solid black in dot — visible as a "ghosted" line. | `real_cosmetic_gap` | `fixable_theme_or_render` | `parity_metrics.json` panel `arrow_types` arrow_width_pt -3.46 on e1 and e5; visible in `parity_pixel_diff/hires/arrow_types/{dot,dagua}.png`; declarative metrics also show arrow_length_pt -2.0 on one edge. |
| HIGH | `arrow_types` | all 9 edge labels (`normal`, `vee`, `dot`, `diamond`, `tee`, `crow`, `circle`, `open`, `none`) | Edge labels in dagua are visibly LARGER and BOLDER than in dot: comparing the two hi-res images, dagua's edge-label glyphs are roughly 1.4-1.6x the height of dot's edge labels for the same panel scale. The metric `font_size_pt` is reported as 100% in tolerance (487/487) — but those metrics cover NODE labels only (the node-label features list); EDGE labels appear to lack a parity check, so the metric does not catch this. | `real_cosmetic_gap` | `fixable_theme_or_render` | Hi-res `parity_pixel_diff/hires/arrow_types/{dot,dagua}.png`. Note `font_size_pt` summary covers 487 element entries which equals the node count, suggesting edge labels are not being measured. |
| HIGH | `nested_clusters` | clusters `Outer Group`, `Right Branch`, `Left Branch`; node A; node F | Cluster geometry breaks: (a) node `A` protrudes out the top edge of the `Outer Group` rectangle in dagua, while in dot `Outer Group` cleanly encloses `A`; (b) the inner `Right Branch` and `Left Branch` rectangles in dagua share / touch their inner border with no gutter, while in dot they have a clear gap; (c) the cluster labels `Outer Group`, `Right Branch`, `Left Branch` in dagua are positioned where they overlap the contained nodes (label crashes into `A`'s ellipse, and `Right Branch` / `Left Branch` overlap each other in the gutter); (d) node `F` overlaps the bottom edge of the outer cluster border instead of sitting clearly below it. | `uncertain_needs_targeted_probe` | `fixable_theme_or_render` (label/box padding constants and cluster bounding-box render rules); some sub-issues might be `needs_layout_scope` if cluster bounding boxes come from layout coordinates rather than render-time text-extent inflation | `parity_pixel_diff/hires/nested_clusters/{dot,dagua}.png` (very obvious at full resolution). `parity_metrics.json` reports cluster_fill / cluster_stroke / cluster_stroke_width_pt 100% in tolerance — declarative metrics don't measure cluster bounding-box position or label anchoring, so heatmap is the primary evidence. |
| MED | `ladder` | edges between A2->B3, A3->A4/B4, A4->B5, A5->B6, etc. | Diagonal cross-edges in dagua's ladder appear to use a curved/bowed routing (slight S-bend), while dot draws straight diagonals. Same source/target points, but the polyline path differs. This is most likely Graphviz B-spline routing in `dot` vs whatever spline mode dagua's render uses, or a layout-coordinate vs control-point difference. | `real_cosmetic_gap` | `rendering_stack_residual` (likely) — this is most likely the documented "B-spline geometry comes from Graphviz routing" residual class. Worth a targeted probe but should NOT drive a theme/render fix without first confirming whether dagua is rendering Graphviz-emitted control points or its own routing. | Hi-res `parity_pixel_diff/hires/ladder/{dot,dagua}.png`. |
| MED | every panel | margin / page padding | The declarative metric `margin_pt` shows `delta = 14.0000` on EVERY panel and is reported "in tolerance" (45/45). delta=14 is suspicious — it's exactly 1 standard Graphviz margin (8 or 14 pt). Two possibilities: (a) tolerance is too loose and is hiding a real margin gap (likely), or (b) the measurement subtracts/adds 14 somewhere. Either way, given finding #1 (canvas fill), the margin convention almost certainly is mismatched between dagua and dot. | `metric_or_measurement_artifact` (re: the in-tolerance flag) plus probable `real_cosmetic_gap` (the actual margin) | `fixable_theme_or_render` (margin constant) AND fix the tolerance for `margin_pt` | `parity_metrics_summary.md` rows 3-10 of "Top 10 Worst-Delta Features": all `graph:graph` `margin_pt` delta=14.0 marked True (in tolerance). |
| LOW | `single_edge`, `tiny_graph` | overall layout aspect | Disambiguation: the single_edge / tiny_graph SSIM is low primarily because of the canvas-fill issue (finding #1). After that is fixed, these panels should jump significantly because the actual content is pixel-aligned (per-region L1=0). No additional cosmetic gap unique to these two panels beyond canvas fill. | `real_cosmetic_gap` (already covered by #1) | `fixable_theme_or_render` (already covered) | Hi-res tiny_graph dot vs dagua. |

Severity scale used:
- HIGH: obvious at normal zoom, affects core Graphviz parity, or impacts many panels.
- MED: visible at full resolution, likely fixable, but smaller scope.
- LOW: subtle but real.

---

## Metric Artifact Review

The biggest measurement artifact in this round was the **per-region pixel-diff
mask**: every single panel reports `Text L1 = Node L1 = Edge/Arrow L1 = 0.0000`
and `Background L1 = total L1`. Naive interpretation: "the cosmetic theme is
perfect, only background differs." Actual interpretation: the per-region masks
are derived from one of the two images (probably the dot reference) and only
score pixels INSIDE that image's masked element regions. Since dagua positions
its smaller graph drawing within an oversize canvas, large swaths of dagua's
content (and where dot's content was) get scored as background.

Implications:
1. The pixel-diff per-region table should not be treated as evidence of
   element-level parity in this round. Use declarative metrics + hi-res
   eyeballing.
2. If the mask is computed from a union of both images' alpha/edge masks,
   then we should expect non-zero L1 in node/text regions where positions
   actually differ. The all-zero pattern is too clean to trust.
3. Recommended measurement fix (out of scope for the theme sprint, but worth
   logging): score per-region L1 over the UNION of both image masks, not just
   the reference's. Until then, flag this in any future audit prompts.

A second smaller artifact: `font_size_pt` reports 100% (487/487) in tolerance,
but the count exactly equals the node count (NOT the edge count). Edge labels
appear to be excluded from font_size measurement. Finding #6 (arrow_types edge
labels visibly larger/bolder) is therefore a real gap that the declarative
metrics do not catch.

A third artifact: `margin_pt` delta=14.0 marked "in tolerance" on every panel —
the tolerance is 14pt or higher, which makes the metric a no-op. See finding #9.

---

## Rendering-Stack Residuals

These are real differences but should NOT drive a theme/render fix in this
sprint (they are baked-in to either Graphviz routing or rasterizer AA):

- **Edge curvature in `ladder` (finding #8)**: probable B-spline-routing
  residual. Confirm by checking whether dagua's edges in this panel come from
  Graphviz-emitted spline control points or from dagua's own routing. If
  dagua-routed, this becomes a real theme/render gap; if Graphviz-routed, it
  is a known residual.
- **Sub-pixel rounding around ellipse edges**: visible in heatmap noise but
  does not change apparent size or alignment of any individual node. Standard
  AA residual.
- **Times,serif font hinting differences** between Cairo (dot's PNG output)
  and matplotlib's text path: not visible in any of the 7 panels at the level
  of changing apparent letter weight or position; flagged here only for
  completeness.

---

## Recommended Next Fixes

Ranked by impact on overall pixel parity (SSIM/L1) and Graphviz fidelity:

### Rank 1 — Canvas-fill / figure-size scaling (finding #1)

This single fix will pull every one of the 45 panels' background L1 toward 0
because today the entire pixel-diff total is "dagua leaves whitespace where
dot drew content." Likely code area: `dagua/render/mpl.py` around
`fig.set_size_inches`, `ax.set_xlim/set_ylim`, `bbox_inches`, and the DPI/PT
conversion that decides the final canvas dimensions in pixels. Check whether
dagua applies the Graphviz native margin (8pt default for `dot`; ~14pt with
the strict theme's `margin` attribute), then sizes the figure to
`graph_bbox + 2*margin` and disables matplotlib's default whitespace padding
(`tight_layout=False`, `pad_inches=0` to `savefig`, or equivalent). Numeric
target available from the metric: target margin is whatever puts dot's
visible content flush against the figure edge (i.e. the offset between the
graph bbox extents in `dot` PNG and the image edge — small in the dot images,
large in the dagua images).

### Rank 2 — Multi-line label wrap defect (findings #2 and #3)

Two specific reproducer nodes:
- `long_labels.n3`: target rx=215.26, ry=18.00. Dagua: rx=204.65, ry=58.47.
- `label_variety.n7`: target rx=141.88, ry=18.00. Dagua: rx=136.90, ry=39.11.

Dagua is wrapping a label that dot keeps on one line. Likely code area:
`dagua/render/text/` — the text-measurement / wrap pass. Check whether dagua
respects the original label's literal newlines (i.e. only wraps on `\n` or
`\\n` in the DOT source) versus auto-wrapping at a soft width. Graphviz
treats `\n` and `\l`/`\r` as line breaks but never auto-wraps. Dagua should
match: only break on those tokens, never auto-wrap.

### Rank 3 — Systematic ellipse_rx narrowing (finding #4)

138 nodes affected, every delta negative, mean -3.15pt, max -13.4pt. Likely
code area: text-extent + node-padding constants in `dagua/render/text/` and
`dagua/render/borders/`. Possible causes:
- Dagua uses a smaller default text-extent multiplier (e.g. measures
  matplotlib bbox without including the natural side bearings that
  Graphviz/Cairo include).
- Dagua's "node padding" constant is shorter than Graphviz's `width=0.75in,
  height=0.5in` minimum or its `margin="0.11,0.055"` (default node margin).
- Dagua applies `width = max(min_width, label_width + 2*pad)` with a smaller
  `pad` than Graphviz.

Numeric calibration target: compute (target_rx - dagua_rx) / label_pixel_width
across the 138 nodes; it should converge to a single padding constant
difference. That constant goes into the strict theme.

### Rank 4 — Per-edge `arrowsize` attribute support (finding #5a)

Two specific reproducers in `arrow_types`: e1 and e5 each have target
`arrow_width_pt = 10.46` vs dagua 7.0. The DOT source for `arrow_types`
almost certainly sets `arrowsize=1.5` (or similar) on these two edges.
Dagua appears to use the theme default for all edges and ignore the
per-edge override. Likely code area: edge-attr ingestion in the
graphviz_strict theme path; ensure `arrowsize` is plumbed from the
parsed DOT attrs into the render-time arrow-head dimensions.

### Rank 5 — Arrow-head shape parity (finding #5b)

`circle`, `dot`, `open`, possibly `vee` and `crow`. The drawn glyph for
several arrowhead types looks materially different from `dot`'s. Likely
code area: `dagua/render/edges/` — the arrow-head drawing primitives.
Confirmation pass needed (check each arrowhead type one-by-one; some may
already be correct and only `circle/open/dot` are off).

### Rank 6 — Edge label font weight/size (finding #6)

Edge labels in `arrow_types` are ~1.4-1.6x heavier/larger than dot's. Likely
code area: edge-label rendering in `dagua/render/text/` or
`dagua/render/edges/labels.py` (if it exists). Check whether dagua applies
the same `fontsize` default (Times 14pt) to edge labels as to node labels;
the visible gap suggests it doesn't, OR that the bold weight is different.
Also surface this in the parity metric (extend `font_size_pt` measurement
to edge labels — current 487/487 is suspiciously equal to node count).

### Rank 7 — Margin tolerance is too loose (finding #9 measurement leg)

`margin_pt` reports delta=14.0 marked "in tolerance" universally. Tighten the
tolerance to <14 (e.g. 2pt) so this metric actually surfaces gaps. Pair this
with the canvas-fill fix.

### Rank 8 — Cluster layout/label intrusion in `nested_clusters` (finding #7)

Conditional: do this AFTER confirming whether cluster bounding-box
coordinates are rendered from layout output (out of scope) vs computed at
render-time from contained-node bbox + label-extent (in scope). If the
latter, the fix lives in the cluster border/label rendering pass; if the
former, this is `needs_layout_scope` and we drop it from the cosmetic queue.

---

## Inspection Log

For each of the 7 hi-res panel pairs (14 PNGs read end-to-end), what I
inspected and what I concluded:

### `bipartite_5x5` (worst SSIM 0.585)

Inspected: 5 top-row L-nodes, 5 bottom-row R-nodes, 10 down-edges, 10
diagonal cross-edges, all 10 arrowheads, label glyphs of L1-L5 / R1-R5,
canvas extents.
Result: shapes/positions match pixel-for-pixel; dagua canvas is much taller
than the drawn content, leaving large empty bands top and bottom and visible
left/right padding. Drives finding #1.

### `ladder` (worst SSIM 0.663)

Inspected: A1-A6, B1-B6 (12 ellipses), all 17 edges, arrowheads on each,
label glyphs, vertical spacing, diagonal-edge curvature, canvas extents.
Result: same canvas-fill issue as bipartite. Additionally: diagonal cross-
edges in dagua look slightly bowed where dot draws straight (drives finding
#8). Otherwise shapes are pixel-aligned.

### `arrow_types` (worst SSIM 0.668)

Inspected: 9 source ellipses, 9 target ellipses, 9 edges, 9 arrowheads
(normal/vee/dot/diamond/tee/crow/circle/open/none), 9 edge labels,
9 source labels, 9 target labels. Compared each arrowhead glyph carefully.
Result: Several real cosmetic gaps in arrow-head shape (open, circle,
maybe vee), edge stroke lighter in dagua, edge-label font significantly
larger/bolder in dagua. Drives findings #5 and #6. Also the worst arrowsize
attribute mismatch (e1, e5).

### `nested_clusters` (worst SSIM 0.685)

Inspected: outer cluster border, inner Right Branch border, inner Left
Branch border, all three cluster labels, nodes A/B/C/D/E/F, all edges and
arrowheads, label intrusion regions.
Result: significant cluster geometry/label-anchoring issues. Drives finding
#7. Cluster label fill/stroke metrics report 100% in tolerance which is
misleading for this category.

### `colors_showcase` (worst L1 43.17)

Inspected: 6 colored ellipses (Red/Blue/Green/Yellow/Purple/Orange), 5
edges, 5 arrowheads, all 6 node labels, vertical chain spacing, canvas
extents.
Result: same canvas-fill issue (huge whitespace at the bottom of dagua's
output) plus visibly narrower nodes (rx -2 to -3pt each). Drives findings
#1 and #4. Color fills, strokes, and label glyphs match.

### `pipeline` (SSIM 0.719)

Inspected: 5 ellipses (Input -> Preprocess -> Transform -> Postprocess
-> Output), 4 edges, 4 arrowheads, vertical chain spacing, canvas extents.
Result: canvas-fill issue (whitespace at bottom of dagua canvas). Same
narrow-rx pattern visible to the naked eye. Otherwise pixel-aligned. No
unique finding beyond #1 and #4.

### `tiny_graph` (SSIM 0.696)

Inspected: 3 ellipses (In/Mid/Out), 2 edges, 2 arrowheads, label glyphs,
canvas extents.
Result: pure canvas-fill issue with 3-node sparse graph; explains why the
SSIM is so poor on such a simple panel. No unique finding beyond #1.

### Declarative-metric scan (no images)

Inspected the full out-of-tolerance list (314 total OOT entries):
- 167 `ellipse_aspect_pct` OOT, all negative -> finding #4
- 138 `ellipse_rx_pt` OOT, all negative -> finding #4
- 3 `ellipse_ry_pt` OOT, top 2 are findings #2 and #3
- 4 `arrow_width_pt` OOT, top 2 are arrowsize=1.5 reproducers -> finding #5a
- 1 `arrow_length_pt` OOT (-2.0pt) -> grouped under finding #5
- 1 `arrow_filled` OOT -> grouped under finding #5b

No additional findings beyond the 9 above were uncovered in the metric scan.

---

## Notes for the next round

- The canvas-fill fix is THE highest-impact lever; do it first. After it
  lands, re-run pixel-diff and re-baseline the per-region masks (and consider
  fixing the mask to use the union of both images so the per-region table
  becomes trustworthy).
- Add edge-label `font_size_pt` to the declarative metric so finding #6 can
  be caught automatically on the next round.
- Tighten `margin_pt` tolerance below 14pt.
- The cluster-geometry findings (#7) need a quick scope check (render vs
  layout) before being added to the fix queue.
