# Visual Audit Round A3 — Graphviz Strict Parity

Auditor: Opus 4.7 (1M ctx)
Date: 2026-04-27
Scope: cosmetic only (`graphviz_strict` theme + render). `dagua/layout/` OUT OF SCOPE.
Inputs: declarative metrics, pixel-diff summary, 8 hi-res panel pairs (16 PNGs).
Round: A3 (post-B2 commit `27646de`).

---

## Verdict

- Prior items (A2 F1-F11): `PARTIAL`
  - F1 figure aspect: PARTIAL — tiny_graph improved (+0.0287 SSIM) but
    `single_edge` and `pipeline` regressed slightly. B2 admits `figsize=`
    override only fires when caller does NOT pass a figsize, which means
    the parity harness path keeps the same behavior.
  - F2 arrowhead polygon shape: PASS — `normal` arrowhead is now a clean
    triangle in all panels. Visible improvement on `arrow_types`.
  - F3 per-edge `arrowsize`: PARTIAL — code path implemented but the four
    `arrow_types` `arrow_width_pt` misses persist (B2 attributes them to
    fixture extraction, not render).
  - F4 edge-label font: PASS at theme level — restored to 14pt; visible in
    `arrow_types.dagua.png`. Still appears slightly smaller than dot's
    edge labels, but the theme-side over-correction is gone.
  - F5 single-line ellipse aspect (re-widen): IMPLEMENTED but OVER-SHOT.
    The 1.85 oval floor over-widens short-label ellipses past dot's
    actual aspect (typically 1.5 for 2-3 char labels). On `tiny_graph`,
    `bipartite_5x5`, and `single_edge` ellipses are now CLIPPED at the
    canvas edge or visibly oversized.
  - F6 long-label ellipse rx: REVERTED per B2 report — long-label rx
    remains 23/487 OOT.
- New audit: `FAIL` — multiple new regressions discovered, primarily the
  oval-floor over-correction.
- Stop criteria status: `CONTINUE` — at least 5 HIGH-severity findings
  classified as `real_cosmetic_gap` + `fixable_theme_or_render`.

---

## SSIM Regression Diagnosis (the central question)

Mean SSIM dropped 0.7615 -> 0.7592 (-0.0023) and worst SSIM dropped
0.5290 -> 0.5226. Despite F1 fixing tiny_graph (+0.0287), the overall
SSIM regressed because B2's F4 (compact-ellipse oval floor) made many
panels worse. The hi-res images make the cause concrete.

**Root cause: the new `_GRAPHVIZ_STRICT_MIN_OVAL_ASPECT = 1.85` floor
in `dagua/render/mpl.py:116` is set ABOVE dot's natural aspect.**

Code path (verified in `dagua/render/mpl.py:2895-2901`):
```
if str(style.shape) == "ellipse" and adjusted_width <= 70.0:
    adjusted_width = max(
        adjusted_width,
        adjusted_height * _GRAPHVIZ_STRICT_MIN_OVAL_ASPECT,  # 1.85
    )
```

For tiny_graph `In` (label 2 chars):
- dot target: rx=27pt, ry=18pt, aspect = 1.50 (verified in
  `parity_metrics.json` tiny_graph n0).
- dagua spec passes the metric (rx=27, ry=18, 1.50 aspect — metric reads
  the spec, NOT the rendered patch).
- dagua rendered: width = `max(54pt, 36pt * 1.85)` = `max(54pt, 66.6pt)`
  = **66.6pt** at the matplotlib level. So the rendered ellipse is ~23%
  wider than the spec, ~23% wider than dot's actual rendering. This
  pushes the ellipse beyond the figure xlim and CLIPS at the canvas
  edge (visible in `tiny_graph.dagua.png` — In/Mid/Out have only the
  top arc and bottom arc visible, the side strokes are off-canvas).

Same pattern on `single_edge` (Source ellipse extends almost to the canvas
edge), `ladder` (A1/B1/A2/B2 all overlap each other), `bipartite_5x5`
(L1-L5 visibly overlap, R1-R5 likewise), `nested_clusters` (A is now far
wider than dot, C/B/E/D ellipses overlap), `colors_showcase` (Red is
clipped at the bottom).

**B2's F4 is implemented as a ONE-WAY floor, not a calibration.** It
forces aspect >= 1.85 even when dot's actual aspect is 1.50 (any 2-3
character label). dot's actual short-label aspect is what the metric
already reports: 1.50 for 2-3 char labels, ~2.0 for "normal" / "diamond"
(6-7 chars). The floor should target dot's actual minimum which, per the
metric, is 1.50, not 1.85.

**Why this also explains the F1 partial result**: The harness's pixel
path passes an explicit `figsize` derived from dot's pixel dimensions,
so `_strict_content_figsize` does not fire (gated by
`figsize is None`, line 1412). With the canvas size held fixed by the
caller, the over-widened ellipses now overflow because both axes are
pinned to dot's bounds while content widened. This is why `single_edge`
SSIM dropped even after F1 (figure aspect can't be the only fix when
content was widened in the same round).

**Implication for B3**: revert `_GRAPHVIZ_STRICT_MIN_OVAL_ASPECT` from
1.85 to either (a) 1.50 — match the median 2-3 char target — or (b)
remove the unconditional floor entirely and instead enforce
`adjusted_width / adjusted_height` to equal the metric `target_aspect`
when the spec falls below it. Option (b) is safer because it matches
dot's actual behavior; option (a) over-corrects on shorter labels and
under-corrects on slightly longer ones.

---

## Prior Item Recheck

| Item | Verdict | Evidence | Notes |
| --- | --- | --- | --- |
| A2 F1 figure aspect | PARTIAL | `parity_pixel_diff/summary.md` — tiny_graph 0.6790 -> 0.7077 (+0.0287). single_edge 0.6523 -> 0.6433 (-0.0090). pipeline 0.7177 -> 0.7162 (-0.0015). | B2's `_strict_content_figsize` only fires when caller omits `figsize`. The parity harness passes explicit `figsize`. F1 only helped on the targeted-pure path. |
| A2 F2 arrowhead polygon | PASS | Hi-res `arrow_types.dagua.png` `normal` arrowhead is a clean filled triangle. Same in `single_edge.dagua.png`, `pipeline.dagua.png`. | The flat-diamond regression from B1 is gone. |
| A2 F3 per-edge arrowsize | PARTIAL | `parity_metrics.json` arrow_types: e1 `arrow_width_pt` -3.46, e5 -3.46, e7 -3.46, e4 +3.08 — same misses as A2. | B2 implemented the parser path; the four OOT entries persist. Could be that the fixture stays in tolerance now and the metric extractor reports stable values. Re-verify after B2 by reading the arrow_types fixture's per-edge `arrowsize=` values. |
| A2 F4 edge-label font | PASS | Hi-res `arrow_types.dagua.png` `normal`/`vee`/etc. labels are visibly larger and now match dot's font size at this scale. | Restored to 14pt. |
| A2 F5 single-line ellipse aspect | OVER-CORRECTED | Hi-res `tiny_graph.dagua.png` ellipses CLIPPED at canvas edge. `bipartite_5x5.dagua.png` L-row and R-row nodes overlap each other. `ladder.dagua.png` A-column and B-column nodes overlap. | The 1.85 floor pushes width past dot's 1.5-2.0 actual aspect. |
| A2 F6 long-label ellipse rx | UNCHANGED | `parity_metrics_summary.md` — `ellipse_rx_pt` 464/487 in tol, max delta -13.40pt. Top-10 worst-delta features rows 1-10 all `ellipse_rx_pt` on `long_labels` / `transformer_block`. | B2 report says "reverted to avoid broad regression." Confirmed unchanged. |

---

## New Findings

| Severity | Panel | Element/Region | Finding | Finding Class | Actionability | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| HIGH | `tiny_graph`, `single_edge`, `bipartite_5x5`, `ladder`, `nested_clusters`, `colors_showcase`, `star`, `arrow_types` (every panel with short labels) | every short single-line ellipse | Compact-ellipse oval floor (`_GRAPHVIZ_STRICT_MIN_OVAL_ASPECT=1.85`) over-widens short-label ellipses past dot's actual rendering. dot's natural short-label aspect is ~1.50 (per `parity_metrics.json`). At 1.85, dagua renders ~23% wider than dot. On panels where the canvas is tightly cropped to dot's bounds, the over-widened ellipses CLIP at the canvas edge (`tiny_graph.dagua.png`: In/Mid/Out lose left+right strokes; only top/bottom arc visible) or OVERLAP each other (`bipartite_5x5`, `ladder`, `nested_clusters`). This is the dominant SSIM regressor for B1->B2 — much larger absolute effect than the +0.0287 tiny_graph win because it impacts every panel. | `real_cosmetic_gap` | `fixable_theme_or_render` | Code: `dagua/render/mpl.py:116` (`_GRAPHVIZ_STRICT_MIN_OVAL_ASPECT = 1.85`) and `dagua/render/mpl.py:2897-2901` (the floor application). Visual: `tiny_graph.dagua.png` — In/Mid/Out are CLIPPED on each side, only top/bottom arcs visible. Metric: `parity_metrics.json` tiny_graph n0 reports rx=27pt, ry=18pt, aspect=1.50 — but the floor forces visual width to 33.3pt (1.85x). Discrepancy invisible to declarative metric. |
| HIGH | `bipartite_5x5`, `ladder`, `nested_clusters`, `arrow_types` source ellipses | adjacent ellipses | Adjacent ellipses now visibly OVERLAP each other (the over-wide patch from F5 collides with sibling positions that were laid out for narrower nodes). On `bipartite_5x5` L1/L2/L3/L4/L5 overlap each other in a continuous sausage chain (visible in `bipartite_5x5.dagua.png`); same for R1-R5. On `ladder` A1+B1, A2+B2, ... all 6 pairs overlap. On `nested_clusters` C+B and E+D overlap. This is a direct downstream effect of the over-wide F5 floor: layout positioned for narrower spec-width nodes, render emits wider patches. | `real_cosmetic_gap` | `fixable_theme_or_render` (downstream of F5 fix) | `bipartite_5x5.dagua.png`, `ladder.dagua.png`, `nested_clusters.dagua.png`. Compare to `bipartite_5x5.dot.png`: dot's L-nodes have visible gutter between siblings. |
| HIGH | `bipartite_5x5` | edges L4->R1 (leftmost) and L3->R5 (rightmost) | Outer edges still render as exaggerated S-curves bowing outside the column band (worse than A2 due to the wider top/bottom nodes pulling spline anchors further apart). dot draws straight diagonals from L4 to R1 and L3 to R5. dagua's L4->R1 visibly arcs OUTSIDE the L4 column then comes back to R1; L3->R5 mirror. This was MED in A2; with the wider nodes from the 1.85 floor, the visible bow is more pronounced. | `uncertain_needs_targeted_probe` | possibly `fixable_theme_or_render` (if dagua interprets splines), possibly `needs_layout_scope` (if it's pure routing) | `bipartite_5x5.dot.png` (straight diagonals) vs `bipartite_5x5.dagua.png` (visible outer arc on far-left and far-right edges). |
| HIGH | `arrow_types`, `single_edge`, `colors_showcase`, `star`, `bipartite_5x5`, `ladder`, `nested_clusters`, `pipeline`, `tiny_graph` (every panel) | every edge stroke | Edge strokes appear LIGHTER (gray/charcoal) in dagua vs dot's clean solid black. Pattern persists across all panels — this is not per-edge, it's the default stroke rendering. Declarative metric reports `edge_stroke_color` 100% in tolerance with `#000000` on both sides (`parity_metrics.json` arrow_types e0). The visible difference is at the renderer level — likely matplotlib's default Line2D anti-aliasing behavior (sub-pixel-coverage based AA blends with white background, producing visually lighter stroke at 1pt width). | `real_cosmetic_gap` | `fixable_theme_or_render` (likely linewidth or AA setting) | Hi-res `arrow_types.dagua.png` edge strokes vs `.dot.png`. Same on every panel. The visible effect is uniform — strongly suggests a single matplotlib-level fix (e.g., emitting at 1.05x linewidth, or using `solid_capstyle='butt'` for crisper rendering). |
| HIGH | `star` | Spoke 1 (leftmost), Spoke 8 (rightmost) | Outermost spoke ellipses are CLIPPED at the canvas edge — left edge of Spoke 1 and right edge of Spoke 8 are visibly cut off. Same root cause as F1 (compact-ellipse oval floor over-widens; positions stay where layout placed them, render emits wider patch that overflows the figure xlim). dot's `star.png` shows full closed ellipses for Spoke 1 and Spoke 8. Also: the arrowheads landing on Spoke 1 and Spoke 8 appear smaller / less filled than dot's, possibly due to terminal-clipping algorithm interaction with the over-wide patch. | `real_cosmetic_gap` | `fixable_theme_or_render` | `star.dagua.png` outer spokes have left/right strokes cut at canvas border. `star.dot.png` shows full closed ovals for all 8 spokes. |
| MED | `single_edge`, `tiny_graph`, `colors_showcase` (Red) | top arc of small ellipses | Visible "double-stroke" / faint-second-curve artifact on the top arc of small ellipses in dagua. Looks like an inner ghost curve a few pixels below the main top arc on `single_edge.dagua.png` Source. Less prominent than A2 because the over-wide patch hides part of it but still detectable on `single_edge`. Likely matplotlib `patches.Ellipse` anti-aliasing residual at this scale. May be aggravated by the over-wide patch interacting with the figure-aspect mismatch. | `real_cosmetic_gap` | possibly `fixable_theme_or_render` (rendering parameters), possibly `rendering_stack_residual` | `single_edge.dagua.png` Source top-right arc. |
| MED | `arrow_types` | edges e1, e5, e7 (`vee`, `crow`, `circle`), e4 (`tee`) | Per-edge `arrowsize` still not respected at the metric level. e1/e5/e7 deltas remain -3.46pt; e4 delta remains +3.08pt. B2 added the parser code path, but the metric values are unchanged — either the fixture's source DOT does not declare `arrowsize=`, or the parser path is not reaching the metric extraction. Verify by `grep arrowsize` on the arrow_types fixture and confirm whether the per-edge attribute exists. If absent, the misses are NOT a parser issue — they are an extractor / shape-specific width issue that B2 mis-attributed. | `uncertain_needs_targeted_probe` | `fixable_theme_or_render` (after probe) | `parity_metrics.json` arrow_types e1, e4, e5, e7 unchanged from A2. |
| MED | `arrow_types` source ellipses (`vee`, `dot`, `tee`, `crow`, `circle`, `open`, `none`) | source ellipses with 3-4 char labels | Source ellipses for these labels are now visibly WIDER than dot's. dot's `vee` ellipse is ~50pt wide; dagua's renders ~55-60pt wide. Metric reports rx target=27, dagua=27 (in tol) — but the metric reads spec, not rendered. Same root cause as F1 (1.85 floor) but on a different size class. | `real_cosmetic_gap` | `fixable_theme_or_render` (same fix as F1) | `arrow_types.dot.png` vs `arrow_types.dagua.png` — measure left-edge to right-edge in pixels for the `vee` source ellipse. |
| LOW | `nested_clusters` | "Outer Group" cluster label | Cluster label "Outer Group" now overlaps the A node ellipse top in dagua (the top edge of the outer cluster rectangle passes through the top half of A's ellipse). dot keeps A above the outer cluster. Mostly layout-scope (where A is positioned), but the cluster label inset / clear-area for the label could be a render-time fix. | `uncertain_needs_targeted_probe` | most `needs_layout_scope`, possibly `fixable_theme_or_render` for label inset only | `nested_clusters.dagua.png` vs `.dot.png`. |
| LOW | `nested_clusters` | inner clusters Right Branch / Left Branch separator | Right Branch and Left Branch rectangles share a centerline with no gutter (dot has visible vertical gap between them). This is layout-scope (cluster horizontal separation). | `real_cosmetic_gap` | `needs_layout_scope` | `nested_clusters.dagua.png`. Same as A2 unchanged. |

---

## Metric Artifact Review

The same per-region pixel-diff mask artifact persists from A1 and A2:
every panel reports Text/Node/Edge L1 = 0.0000 with all the L1 in
Background. The mask is computed from one image's mask only (likely the
dot reference). Implication: per-region table cannot be used to localize
ellipse over-widening to the "Node" region — the regression hides
entirely in the "Background" L1. Until the mask is fixed to take a union,
audits must continue to rely on hi-res visual inspection.

`font_size_pt` 487/487 in tolerance: edge-label font size is still NOT
in the parity metric. B2's F4 fix to 14pt was therefore unmeasured
(though confirmed visually).

`ellipse_aspect_pct` reads from spec, not rendered patch. The 1.85
oval floor regression is invisible to this metric because the spec
keeps reporting 1.50 for tiny_graph In while the rendered patch is
1.85. Recommended: add a parallel "rendered_aspect_pct" metric that
extracts width/height from the saved PNG, not the spec. This would
have caught B2's F5 over-correction immediately.

---

## Rendering-Stack Residuals

These are real differences but should NOT drive a theme/render fix in
this sprint:

- **Sub-pixel anti-aliasing differences** between matplotlib's
  `patches.Ellipse` rasterizer and Cairo's elliptic arc on small
  ellipses (drives the residual top-arc ghost on `single_edge`).
- **Times,serif font hinting differences** between Cairo and
  matplotlib's text path: still not visible at the level of changing
  apparent letter weight or position.
- **Cluster geometry** (sibling overlap, label inset, outer protrusion):
  layout-scope per sprint scope.

---

## Recommended Next Fixes

Ranked by impact on overall pixel parity (SSIM/L1) and Graphviz fidelity:

### Rank 1 — REVERT the 1.85 oval floor (or calibrate down)

**This is the SSIM regression cause.** Two options:

a) **Drop `_GRAPHVIZ_STRICT_MIN_OVAL_ASPECT` from 1.85 to 1.50.** Median
   target aspect for short labels is 1.50 per `parity_metrics.json`. This
   matches dot exactly for 2-3 char labels and nearly so for 4-5 char.
   Predict: SSIM jumps 0.02-0.04 on `tiny_graph`, `single_edge`,
   `bipartite_5x5`, `ladder`, `nested_clusters`, `colors_showcase`,
   `star`, `arrow_types`. That's 8 of the 10 worst panels.

b) **Remove the unconditional floor; replace with a calibrated lookup**
   keyed on ry: for ry=18pt (the strict default), aspect target = 1.50;
   for larger ry use the metric-extracted aspect from the corresponding
   target. Safer but requires knowing dot's behavior across the ry range.

Option (a) is fast, defensible (matches median target), and the most
likely path to recovering SSIM. Code: `dagua/render/mpl.py:116`.

### Rank 2 — Fix or revert F1's `_strict_content_figsize` gate

Current gate: only fires when `figsize is None`. Parity harness always
passes explicit `figsize`. Either:
a) make the gate fire for graphviz_strict regardless of caller's
   `figsize` (overriding the harness's explicit canvas), OR
b) update the harness to omit `figsize` for graphviz_strict comparison
   panels, OR
c) accept the harness path shows a slight regression on `single_edge`
   and `pipeline` and rely on the targeted-pure path improvement only.

Recommend (a) — make the strict theme always size to its content. The
harness passes `figsize` defensively; if strict overrides, both renders
become content-sized and the comparison is fair.

### Rank 3 — Investigate edge stroke "lighter" appearance

Every edge stroke looks gray vs dot's black. Probe:
1. Save a 1px-cropped strip from `single_edge.dagua.png` containing the
   edge stroke; sample RGB values.
2. Compare to dot's same strip.
3. If dagua's RGB is e.g. (60,60,60) vs dot's (0,0,0), the stroke is
   genuinely rendering as gray — likely an alpha or color setting in
   `_draw_edges`. Fix: ensure `linecolor=(0,0,0,1)` and
   `linewidth=1.0` (not 0.75 or 0.8). If both are correct, the issue
   is matplotlib AA — apply `linewidth=1.05` or `solid_capstyle='butt'`
   to crispen the stroke.

### Rank 4 — Per-edge `arrowsize` probe

Verify whether the `arrow_types` source DOT actually declares per-edge
`arrowsize=` attributes or whether B2's F3 fix was unobservable because
the fixture has no such attributes. If the fixture is missing them,
B2's F3 is correct but unobservable — replace the fixture with one that
exercises per-edge `arrowsize` AND fix the underlying shape-specific
width issue (which IS the residual cause of e1/e5/e7 misses).

### Rank 5 — Add rendered-aspect metric (measurement infra)

Out of cosmetic scope, but blocking: add a "rendered_aspect_pct"
field to parity metrics that reads from the saved PNG bounding box,
not the ShapeSpec. This would have caught B2 F5 immediately and will
catch any future render-only over-correction.

### Rank 6 — Per-region pixel-diff mask UNION fix (measurement infra)

Same as A2 Rank 8. Still blocking.

### Rank 7 — Edge-label font_size_pt in parity metric (measurement
infra)

Same as A2 Rank 9. Still blocking.

---

## STOP Criterion Status

`CONTINUE`. Multiple HIGH-severity findings classified as
`real_cosmetic_gap` + `fixable_theme_or_render`:
- F1 oval-floor over-correction (the new SSIM regressor)
- F2 adjacent ellipse overlap (downstream of F1)
- F4 edge-stroke "lighter" appearance (every panel)
- F5 outer spoke clipping in `star`
- F8 source ellipses too wide in `arrow_types`

STOP would require zero `real_cosmetic_gap` + `fixable_theme_or_render`
findings. We have at least 5 HIGH-severity, all actionable. STOP = NO.

---

## Inspection Log

For each of the 8 hi-res panel pairs (16 PNGs read end-to-end), what I
inspected and what I concluded:

### `bipartite_5x5` (SSIM 0.5226, worst overall, regressed from 0.5290)

Inspected: 5 top L-nodes (L1-L5), 5 bottom R-nodes (R1-R5), 10 vertical
down-edges, 10 diagonal cross-edges, all 10 arrowheads, 10 node label
glyphs, canvas extents, edge curvature.
Result: L-nodes overlap each other in a continuous chain (F2 — F1 oval
floor over-correction makes 2-char ellipses ~33pt wide vs dot's 27pt;
layout positions 5 nodes at dot's spacing, render makes them collide).
Same on R-row. Outer edges L4->R1 and L3->R5 still arc visibly outside
column (F3, exacerbated by wider nodes). Arrowheads are now triangular
(B2 F2 PASS) but visibly smaller than dot's. Edge strokes look
gray/charcoal vs dot's solid black (F4).

### `arrow_types` (SSIM 0.6183, regressed from 0.627)

Inspected: 9 source ellipses, 9 target ellipses, 9 edges, 9 arrowheads,
9 edge labels, 9 source-target connection strokes.
Result: source ellipses for `vee`, `dot`, `tee`, `crow`, `circle`,
`open`, `none` (3-4 char labels) are visibly wider than dot's (F8 — F1
floor over-correction). Arrowheads are triangular (B2 F2 PASS) but
visibly smaller / thinner than dot's. Edge labels match dot's font size
(B2 F4 PASS) but still look slightly small. Edge strokes look
gray/charcoal vs dot's bold black (F4). Per-edge `arrowsize` still
unobservable in metric (F7).

### `ladder` (SSIM 0.6412, regressed from 0.642)

Inspected: A1-A6, B1-B6 (12 ellipses), all 17 edges, arrowheads, label
glyphs, vertical spacing, diagonal cross-edge curvature, canvas extents.
Result: A-column and B-column nodes overlap each other on every row
(F2 — A1+B1 share a chord; A2+B2 likewise; through to A6+B6). dot has
~10pt gutter between sibling pairs. Diagonals slightly bowed (A1 #8,
unchanged residual). Arrowheads triangular (PASS) but smaller. Edge
strokes look gray (F4).

### `single_edge` (SSIM 0.6433, regressed from 0.6523)

Inspected: 2 ellipses (Source, Sink), 1 edge, 1 arrowhead, 2 label
glyphs, canvas extents.
Result: Source ellipse extends almost to the canvas edge — over-wide
(F1, F8). Top arc shows visible double-stroke ghost (F6).
Source/Sink visibly wider than dot's. Arrowhead triangular (PASS),
edge stroke gray (F4).

### `tiny_graph` (SSIM 0.7077, IMPROVED from 0.6790)

Inspected: 3 ellipses (In, Mid, Out), 2 edges, 2 arrowheads, label
glyphs.
Result: SSIM improvement is real (F1 fix path fired here because the
targeted-pure dispatch may have used a content-sized canvas) BUT the
render itself is WORSE than dot — In/Mid/Out ellipses are CLIPPED on
left and right edges; only top and bottom arcs are visible (F1
over-correction). The SSIM number is misleading as a quality signal —
this panel looks visibly broken at hi-res. Arrowheads triangular,
edge strokes gray (F4).

### `nested_clusters` (SSIM 0.6844, regressed from 0.688)

Inspected: outer cluster border, inner Right Branch border, inner Left
Branch border, all 3 cluster labels, A/B/C/D/E/F nodes, all edges,
arrowheads.
Result: A node now extends so wide that the outer cluster top-edge
passes through the upper half of A (F11). C/B and E/D ellipses overlap
each other (F2). Right Branch / Left Branch share centerline (F12,
layout-scope). Cluster label inset wrong on "Outer Group" (intrudes
on A). Most cluster geometry is layout-scope.

### `colors_showcase` (SSIM 0.7106, regressed from 0.714)

Inspected: 6 colored ellipses, 5 edges, 5 arrowheads, 6 label glyphs,
vertical chain spacing, canvas extents.
Result: Red (top) ellipse appears clipped at the bottom — its lower arc
is partially missing (F1 over-correction interacting with vertical
chain spacing). Blue/Green/Yellow/Purple/Orange all visibly wider than
dot's (F1). Arrowheads triangular but smaller; edge strokes gray.
Color fills, strokes, and label glyphs match — colors are NOT a finding.

### `star` (SSIM 0.6970)

Inspected: 1 hub, 8 spokes, 8 edges, 8 arrowheads, 9 label glyphs,
canvas extents.
Result: outer spokes (Spoke 1 leftmost, Spoke 8 rightmost) CLIPPED at
canvas edge — left arc of Spoke 1 and right arc of Spoke 8 visibly
cut off (F5). Hub aspect roughly matches. Arrowheads on Spoke 1 and
Spoke 8 look smaller / less prominent than dot's, possibly because
terminal clipping interacts with the over-wide spoke patch. Edge
strokes gray.

### Declarative-metric scan (no images)

Inspected the full out-of-tolerance list (54 OOT entries):
- 24 `ellipse_aspect_pct` OOT — same as A2.
- 23 `ellipse_rx_pt` OOT — same as A2.
- 4 `arrow_width_pt` OOT — same as A2.
- 1 `arrow_length_pt` OOT.
- 1 `arrow_filled` OOT.
- 1 `ellipse_ry_pt` OOT.

Crucially, **the metric is unchanged from A2** (99.27% in tolerance
both rounds). All the visible regressions diagnosed in A3 are invisible
to the declarative metric — they live entirely in the rendering layer.
Confirms the urgency of Rank 5 (rendered-aspect metric).

---

## Notes for Round B3

- **Do F1 (revert oval floor) FIRST.** It's a one-line change
  (`_GRAPHVIZ_STRICT_MIN_OVAL_ASPECT = 1.50`). Predicted SSIM impact:
  +0.02-0.04 on 8 of the 10 worst panels. This is the highest-leverage
  fix in the sprint by a wide margin.
- **Verify before declaring**: re-run `parity_pixel_diff.py` after the
  one-line change. If mean SSIM is back to >=0.770, B3 is largely
  done modulo edge-stroke and per-edge-arrowsize cleanup.
- **Do not re-add a min-aspect floor without first measuring** dot's
  actual rendered aspect for each label-length class.
- **Add the rendered-aspect parity metric** (Rank 5) before B4 to
  prevent the same trap recurring.
- **Cluster geometry findings remain layout-scope** per sprint scope; do
  NOT divert cosmetic effort to cluster bbox sizing in B3.
- **Per-region pixel-diff mask UNION fix** still blocking measurement
  fidelity; queue for B4.
