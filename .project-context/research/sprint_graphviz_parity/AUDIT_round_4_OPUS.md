# Graphviz Theme Parity Audit -- Round 4 (Opus, maximum pickiness)

## Methodology

- **Panels reviewed (12):** pipeline, diamond, balanced_binary_tree, state_machine,
  arrow_types, cluster_showcase, nested_clusters, deep_nesting_4, transformer_block,
  data_pipeline, microservices, multi_cycle, colors_showcase, long_labels,
  medium_mixed (15 distinct PNGs surveyed; 12 priority panels examined minutely).
- **Inspection level:** at-zoom glyph comparison; ellipse aspect ratio measurement
  by eye in pixel-space; arrow tip vs node-boundary inspection; cluster border
  weight/intersection check; font availability verified at OS level
  (`fc-match`).
- **Column header verification:** YES, confirmed on every PNG. LEFT = "Graphviz dot",
  MIDDLE = "Dagua (strict)", RIGHT = "Dagua (improved)". Round 3's Sonnet
  cluster-label misread is documented but the inverse error is also possible --
  several issues below were independently flagged by the user's expert eye AND
  visible in MIDDLE upon careful re-inspection.
- **Total distinct departures found:** 19 (8 high, 7 medium, 4 low) plus a separate
  acceptable-residual list.
- **Theme code consulted:** `dagua/styles.py:910-1000` (the `GRAPHVIZ_STRICT_THEME`
  block). Current parameter values cross-referenced against rendered output.
- **Font availability cross-check:**
  ```
  fc-match "Times New Roman" -> Times_New_Roman.ttf (msttcorefonts)
  fc-match "Times-Roman"     -> qtmr.pfb (TeXGyreTermes)
  ```
  Graphviz's default "Times-Roman" PostScript name is satisfied by **TeX Gyre
  Termes** on this Linux machine; dagua's `font_family="Times New Roman"`
  resolves to **Times New Roman** (Microsoft Core Fonts). These are physically
  different fonts with different glyph shapes and metrics. **This is the
  root cause of the "different font" complaint.**
- **Graphviz version:** dot 8.0.3 (20230418.1244).

---

## Critical Departures (HIGH severity -- fix in next round)

### H1. Font family -- dagua renders Times New Roman; dot renders TeX Gyre Termes (Times-Roman PS resolution)

- **Native dot:** Glyphs come from TeX Gyre Termes (the system's resolution
  of the "Times-Roman" PostScript name). Distinctive features visible on
  pipeline.png and diamond.png LEFT panel: capital "T" terminals are slightly
  flared with prominent serifs; lowercase "g" has a rounded loop with a
  short ear; capital "S" has nuanced spine curvature; "P" bowl is tighter.
- **Dagua strict:** Glyphs come from Times New Roman (Microsoft Core Fonts).
  Visible in MIDDLE panel: "T" terminals are squarer/more uniform; "g" has
  a slightly different loop shape; "S" spine looks straighter; the entire
  paragraph has a SLIGHTLY different rhythm. On colors_showcase.png the
  word "Yellow" reveals it: the "Y", the "e" eye, the "l" terminal all
  differ visibly between LEFT and MIDDLE at this rendering scale.
- **Panels exhibiting:** every panel with text -- pipeline, diamond,
  colors_showcase, balanced_binary_tree, state_machine, long_labels are
  the clearest comparisons.
- **Likely fix location:** `dagua/styles.py:915` -- change
  `font_family="Times New Roman"` to `font_family="TeX Gyre Termes"` (or
  add a fallback chain: `"TeX Gyre Termes, Nimbus Roman No9 L, Times-Roman,
  serif"`). For maximum parity, query Graphviz's resolved font path on the
  current system at theme load time and use that.
- **Severity rationale:** the user explicitly flagged "different font". This
  is the single largest perceptual gap because EVERY label is wrong. It is
  cheap to fix (one parameter) and immediately visible.

### H2. Node label font is too small in strict (DPI normalization OVERSHOT)

- **Native dot:** Labels render at a comfortable readable size. On
  pipeline.png LEFT, "Preprocess" letters span ~28-30px height inside
  an ellipse ~70px tall, so label-height/ellipse-height ~= 0.40-0.42.
  On colors_showcase.png LEFT, "Yellow" letters are ~22-24px tall.
- **Dagua strict:** Labels are visibly smaller. On pipeline.png MIDDLE,
  "Preprocess" letters span ~18-20px height in an ellipse ~60px tall.
  Ratio ~0.32-0.33. On colors_showcase.png MIDDLE, "Yellow" reads as
  noticeably smaller than dot's "Yellow". The labels float in the
  ellipses with more whitespace than dot.
- **Theme value:** `font_size=10.5` in `dagua/styles.py:916`. The round-3
  reasoning was 14pt * 72/96 = 10.5pt, but matplotlib's default DPI on
  this machine is 100 (not 96), and dot's effective rendered size at
  `dpi=96` SVG with default `fontsize=14` is closer to 14pt physical.
  The DPI normalization formula was correct in spirit but the destination
  scale was wrong; 10.5 is too small.
- **Panels exhibiting:** EVERY text-bearing panel. Most clear on
  colors_showcase, pipeline, diamond, balanced_binary_tree.
- **Likely fix location:** `dagua/styles.py:916` -- raise `font_size`
  from `10.5` to ~`12.0-12.5`. Empirically tune by overlaying LEFT and
  MIDDLE panels at the same zoom and matching cap-height. The `10.5`
  value was an over-correction.
- **Severity rationale:** the user explicitly flagged "text isn't centered"
  -- part of the "isn't centered" perception is that text is too small
  and floating, not actually mis-positioned, but the visual fault is
  still real and immediate.

### H3. Ellipse aspect ratio mismatch -- strict ellipses are flatter/wider than dot's

- **Native dot:** On pipeline.png LEFT, "Preprocess" ellipse looks ~140px
  wide by ~70px tall (ratio ~2.0). "Postprocess" similar. On diamond.png
  LEFT, "Start" ellipse is ~135x80 (ratio ~1.7) -- noticeably more rounded.
- **Dagua strict:** On pipeline.png MIDDLE, "Preprocess" ellipse is
  ~190x60 (ratio ~3.2). On diamond.png MIDDLE, "Start" ellipse is
  ~155x65 (ratio ~2.4) -- visibly squashed compared to dot.
- **Symptom:** dagua's ellipses feel "stretched horizontally" relative
  to dot's "rounder" shape at the same label content.
- **Panels exhibiting:** pipeline (very clear), diamond (very clear),
  colors_showcase (clear), balanced_binary_tree (subtle on inner nodes,
  obvious on Root), long_labels (extreme: BatchNormalization2d ellipse
  is grossly horizontally stretched in MIDDLE vs the more compact
  rounded LEFT).
- **Likely fix location:** the ellipse-fitting routine in the renderer.
  Graphviz computes ellipse semi-axes as `width = max(label_width + margin,
  node_width) / 2 * sqrt(2)` and `height = max(label_height + margin,
  node_height) / 2 * sqrt(2)` (the sqrt-2 makes ellipses circumscribe
  the label box rather than match it -- producing more rounded shapes).
  Dagua's current ellipse fitter likely uses a tighter circumscription
  formula. Check `dagua/render/mpl.py` ellipse path generation; the
  semi-axis multiplier should be `sqrt(2)` not 1 or some smaller value.
- **Severity rationale:** at-a-glance the most "shape doesn't match"
  difference. Even if fonts and arrows were perfect, ellipse shape alone
  would scream "not dot."

### H4. nested_clusters: Outer Group cluster box CUTS THROUGH node A

- **Native dot:** node A is positioned ABOVE the cluster boundary.
  The "Outer Group" rectangle starts BELOW A with clear vertical
  separation (~20-30pt). A connects into the cluster via an edge that
  enters from above the rectangle's top edge.
- **Dagua strict:** node A is positioned ON TOP OF the cluster boundary
  -- the "Outer Group" rectangle's top edge passes RIGHT THROUGH the
  middle of node A's ellipse. Visually catastrophic.
- **Panels exhibiting:** nested_clusters.png MIDDLE (severe).
- **Likely fix location:** the cluster bounding-box computation needs
  to include a top-margin push-out for nodes that are root-of-cluster
  predecessors (i.e. nodes outside the cluster but with edges into the
  cluster's first row should not overlap the cluster's top stroke).
  Either the cluster padding is being computed without considering
  external-incoming edges, or the cluster top-edge is being placed at
  the bounding-y of the contained nodes without leaving room for the
  cluster label, which then pushes the cluster up over external-but-
  adjacent nodes.
- **Severity rationale:** user explicitly flagged "Cluster bounding boxes
  look like shit." This is the literal manifestation of that complaint.

### H5. nested_clusters: Right Branch and Left Branch sub-cluster boxes overlap each other; their labels collide

- **Native dot:** "Right Branch" cluster on the LEFT (containing C, E)
  and "Left Branch" cluster on the RIGHT (containing B, D) are clearly
  separated -- there's about 15-20pt of horizontal whitespace between
  the two sub-cluster rectangles. Each label sits inside its own
  sub-cluster top-left corner, distinct.
- **Dagua strict:** "Right Branch" and "Left Branch" sub-cluster boxes
  in MIDDLE OVERLAP each other in the horizontal middle of the figure
  -- the rectangles share an inner edge or overlap. The labels
  "Right Branch" and "Left Branch" appear so close that they nearly
  touch ("Right BranchLeft Branch" reads almost as one word at this
  zoom). Visible structural defect.
- **Panels exhibiting:** nested_clusters.png MIDDLE.
- **Likely fix location:** sub-cluster horizontal separation is too
  small. In dot, sibling clusters at the same nesting level get a
  default `nodesep` worth of horizontal padding; dagua may be using
  zero or negative padding here. Check the sibling-cluster spacing
  in the cluster layout pipeline. Could be in
  `dagua/layout/ops/cluster_*` or the renderer's bounding-box pass.
- **Severity rationale:** another direct manifestation of the
  "cluster boxes look like shit" complaint. Two named sub-clusters
  bleeding into each other is a clear quality failure.

### H6. Back-edge routing: strict draws straight verticals through the body of the graph; dot routes around the side

- **Native dot:** On state_machine.png LEFT, the "restart" edge from
  Done -> Idle and the "reset" edge from Error -> Idle route as
  CURVED B-spline arcs along the RIGHT MARGIN of the graph -- clean
  arcs that hug the outside.
- **Dagua strict:** On state_machine.png MIDDLE, the same back-edges
  are STRAIGHT VERTICAL LINES that cross THROUGH the middle of the
  graph body -- specifically through the Running -> Paused -> Error
  vertical column. The back-arc-to-curvature parameter (`curvature=0.2`
  in styles.py:970) does not seem to take effect for these long
  back-edges; they render as near-straight lines.
- **Same issue on multi_cycle.png MIDDLE:** the G->A back-arc cuts
  vertically through the entire stack of nodes (D(hub), C, B, A).
- **Panels exhibiting:** state_machine (severe), multi_cycle (severe),
  cluster_showcase (some), microservices (subtle).
- **Likely fix location:** the back-edge routing pipeline. Either
  (a) the `curvature=0.2` value is geometrically too small once the
  Bezier control offset is applied to a long edge (the offset scales
  with chord length and at long chords 0.2 of chord becomes the same
  pixel offset that 0.2 of short chord was -- which means tall narrow
  graphs see the back-edge running nearly straight), or (b) the
  back-edge routing op doesn't push the control point to a distinct
  side-channel for long-chord cases. The fix is likely to compute
  curvature as `min(absolute_offset_pt, curvature_fraction * chord_len)`
  with a non-trivial `absolute_offset_pt` floor (e.g. 30pt) so
  long back-edges still bow visibly outward.
- **Severity rationale:** "arrows are wonky" partially captures this.
  Edges crossing through the body of the graph is a serious visual
  failure -- a domain expert would never confuse this with dot.

### H7. Open arrow forms (vee, open, circle) render as filled in strict

- **Native dot:** On arrow_types.png LEFT:
  - "vee" arrowhead is an OPEN V shape (two strokes, no fill)
  - "open" arrowhead is an UNFILLED triangle outline
  - "circle" arrowhead is an OPEN circle (white fill, black border)
- **Dagua strict:** On arrow_types.png MIDDLE:
  - "vee" renders as a FILLED triangle (looks identical to "normal")
  - "open" renders as a FILLED black triangle
  - "circle" renders as a SOLID black dot (not hollow)
- **Panels exhibiting:** arrow_types.png MIDDLE.
- **Likely fix location:** the arrow primitive registry. Each named
  arrowhead style in graphviz has a fill flag; dagua appears to be
  applying `arrow_fill="filled"` (theme line 940) globally regardless
  of the named arrowhead. The renderer needs per-arrowhead-name fill
  rules, or arrow_types panel needs to override per-arrow with the
  correct fill setting. Alternatively the "open" prefix in graphviz
  semantics -- "ovee", "ocrow", "ocircle", "odot", "obox" -- is what
  yields the open form; dagua may be receiving the bare names without
  the open-prefix translation.
- **Severity rationale:** named-arrowhead graphs render with WRONG
  shapes, not just slightly different. This is a correctness issue
  beyond cosmetics.

### H8. Cluster label text occluded by cluster box top-edge stroke

- **Native dot:** Cluster labels (e.g. "Outer Group", "Right Branch")
  in dot are positioned in the top-left of the cluster but with a
  pixel-perfect transparent-background area around the text, AND
  the cluster's top stroke is BROKEN around the label so the label
  reads cleanly without the stroke crossing through it.
- **Dagua strict:** On nested_clusters.png MIDDLE, the cluster's top
  stroke passes THROUGH the middle of the "Outer Group" label text --
  the horizontal stroke crosses through the lowercase letters at
  about x-height. Same on "Right Branch" and "Left Branch" labels --
  the box's top edge cuts through the labels at descender-height.
- **Panels exhibiting:** nested_clusters (severe), deep_nesting_4
  (severe -- "Level 1", "Level 2" have stroke through them),
  cluster_showcase (moderate), data_pipeline (subtle).
- **Likely fix location:** cluster border drawing routine needs to
  break the top-edge stroke around the cluster label. Standard
  approaches: (a) don't draw the segment of top-edge that lies under
  the label bounding box (with small padding around the label),
  (b) raise the label clear of the stroke by translating its y up
  half its height, (c) draw a white-fill rectangle behind the label
  to mask the stroke. Option (a) is closest to dot. Option (c) is
  cheapest if matplotlib z-order can place the label over the stroke
  with a backdrop -- but currently dagua does the wrong thing
  somehow.
- **Severity rationale:** another direct "cluster boxes look like
  shit" example. Text crossed by box stroke looks unprofessional
  immediately.

---

## Medium Departures (worth fixing if cheap)

### M1. Arrow tip-to-node-boundary spacing inconsistent

- **Native dot:** Arrows touch the ellipse boundary cleanly -- the
  triangle tip lands ON the boundary curve, no gap and no overlap.
- **Dagua strict:** On pipeline.png MIDDLE, the arrow tips into
  Preprocess and Transform ellipses appear to have a SMALL VISIBLE
  GAP (~1-2px) between the triangle tip and the ellipse outline.
  On diamond.png MIDDLE, the arrow tips into Left/Right and into
  End slightly OVERLAP into the ellipse (the back of the triangle
  is inside the node outline). Inconsistent: gap on some panels,
  overlap on others.
- **Panels exhibiting:** pipeline (gap), diamond (overlap), state_machine
  (mixed -- depends on edge angle).
- **Fix:** the edge-trim routine that computes "where to stop the
  edge body" is slightly off relative to the arrow length. Standard
  fix: compute the boundary intersection point on the ellipse, then
  place the arrow tip exactly there and trim the edge body to
  `tip_pt - arrow_length * direction`.

### M2. Arrow triangle proportions -- elongated in strict, more equilateral in dot

- **Native dot:** Arrow triangles are short and broad (length:width
  approximately 1.4:1 or similar). They look "stout."
- **Dagua strict:** Arrow triangles are LONGER and NARROWER (length:width
  approximately 1.6:1 -- arrow_length=10, arrow_width=7 in styles.py
  yields ratio 10/7 = 1.43). They look "pointy".
- **Theme value:** `arrow_length=10.0, arrow_width=7.0` (lines 941-942)
  yields ratio 1.43, which is reasonable on paper but visually reads
  pointier than dot. dot's effective ratio at default arrowsize=1
  appears closer to 1.0-1.2 (almost equilateral).
- **Fix:** try `arrow_length=8.0, arrow_width=8.0` for a more
  equilateral look, or `arrow_length=9.0, arrow_width=7.5`.

### M3. Cluster border too gray-bright -- dot is closer to faint outline

- **Native dot:** Cluster box borders read as VERY PALE GRAY at this
  rendering size, almost ghost-thin -- maybe `#CCCCCC` or `#D5D5D5` at
  0.5pt.
- **Dagua strict:** Cluster box borders are clearly `#AAAAAA` -- a
  medium gray that reads as DEFINITE not ghost. Stroke also slightly
  thicker (`stroke_width=0.8` per line 976) than dot's effective
  0.5pt.
- **Panels exhibiting:** nested_clusters, deep_nesting_4, transformer_block,
  cluster_showcase, data_pipeline.
- **Fix:** lighten cluster stroke to `#CCCCCC` and/or reduce
  `stroke_width` to `0.5`.

### M4. Cluster fill opacity slightly visible in strict; dot's is essentially invisible

- **Native dot:** Cluster fill is `lightgrey` at very low alpha
  (~0.05-0.10). On nested_clusters LEFT, the inner clusters have a
  barely-perceptible warm-cream tint visible only on careful
  inspection.
- **Dagua strict:** Cluster fill is `#F0F0F0` with `fill_opacity=0.15`
  (lines 974, 985). At rendered size this reads as visible gray
  rectangles -- more "background tint" than dot's invisible-tint.
- **Fix:** reduce `fill_opacity` from 0.15 to 0.07-0.10. On
  transformer_block dot uses an off-white-cream tint that's slightly
  warmer than gray; dagua's `#F0F0F0` is neutral cool gray; this is
  a chromatic difference too.

### M5. Edge label colliding/no occlusion

- **Native dot:** On state_machine.png LEFT, edge labels "retry"
  and "resume" each have their own vertical position with white-
  background occlusion of the underlying edge.
- **Dagua strict:** On state_machine.png MIDDLE, "retry" and "resume"
  labels sit at the same y-coordinate close to each other, almost
  bumping ("retryresume" reads as one word). Labels do appear to have
  white backgrounds but the placement is too close.
- **Panels exhibiting:** state_machine (severe overlap), data_pipeline
  (subtle).
- **Fix:** edge label placement needs to be aware of nearby edge labels
  and offset along-edge to avoid horizontal-y overlap. Or push labels
  to OUTSIDE the curve rather than ON the curve.

### M6. Color saturation drift in colors_showcase

- **Native dot:** Yellow renders as a saturated golden-amber. Orange
  is a warm pure orange. Red is a clear coral-pink (lightcoral).
- **Dagua strict:** Yellow renders slightly more muted. Orange slightly
  desaturated. Red is similar but slightly less pink.
- **Mechanism:** node fills in dot use `fillcolor=red`, etc. -- which
  resolve to ImageMagick X11 named-color values. dagua may be using
  different name-to-RGB tables or applying alpha differently.
- **Fix:** verify the named-color RGB table dagua uses matches X11
  classic palette exactly (R=red, etc., to standard hex).

### M7. Edge body stroke is heavier in strict than dot

- **Native dot:** Edge bodies look like ~1.0pt hairlines -- light but
  visible.
- **Dagua strict:** Edge bodies look ~1.2-1.4pt -- visibly heavier than
  dot's. Even though `width=1.0` (theme line 938) is the same nominal
  value, the rendered weight reads heavier; this is likely a matplotlib
  vs cairo rasterizer difference.
- **Fix:** try `width=0.75` to match the visual weight. Pair with
  the M3 cluster stroke reduction.

---

## Low Departures (visible but minor)

### L1. Title text in dagua panels is bold; in dot it isn't

- The graph TITLE rendered above each panel is bold in dagua
  (`title_font_size=14.0` with implicit weight normal, but the
  rendered glyphs read bold). dot's title text is regular weight.
  Minor since titles are panel-decoration not graph-content.

### L2. Panel margin at top differs

- The vertical spacing between panel header ("Dagua (strict)") and
  the first node varies between dot and strict on multiple panels.
  pipeline.png MIDDLE has noticeably more top whitespace before
  "Input" than dot's LEFT.

### L3. Padding inside ellipse asymmetry on long labels

- On long_labels.png MIDDLE, the long label
  "BatchNormalization2d(128, eps=1e-05, momentum=0.1)" extends to the
  far edges of the ellipse with minimal lateral padding, while dot
  preserves more left-right padding on the same label. This is a
  side-effect of the H3 ellipse aspect issue but distinct in
  presentation.

### L4. Node text vertical centering -- strict labels sit slightly low

- On colors_showcase.png MIDDLE, the labels appear to sit fractionally
  below true vertical center of their ellipses (maybe 1-2px below).
  dot has labels at exact vertical center. Subtle but visible at
  100% zoom.

---

## User-Flagged Issues -- Verification

### "Arrows are wonky"
**CONFIRMED.** Multiple distinct issues compounding under "wonky":
- H6 (back-edges drawn as straight lines through graph body)
- H7 (open arrow forms drawn as filled)
- M1 (tip-to-boundary spacing inconsistent: gap on some, overlap on others)
- M2 (arrow triangle proportions: too elongated in strict)
- M7 (edge body stroke slightly heavier than dot's)

### "Text isn't centered"
**PARTIALLY CONFIRMED.** Specifically:
- L4 (node label vertically slightly low in strict -- 1-2px below true center)
- The user's perception is also colored by H2 (text being too small relative
  to ellipse -- when text is too small AND slightly off-center, the eye reads
  "off-center" more strongly than either alone)
- Edge labels (M5) genuinely overlap/collide on state_machine -- "retry"
  and "resume" sit so close they read as colliding

### "Different font"
**CONFIRMED with root cause identified.** dot's "Times-Roman" PostScript name
resolves to **TeX Gyre Termes** on this Linux system (`fc-match "Times-Roman"`).
dagua's `font_family="Times New Roman"` resolves to **Times New Roman**
(Microsoft Core Fonts). These are physically different fonts with different
glyph shapes, metrics, and stroke contrast. See H1.

### "Cluster bounding boxes look like shit"
**CONFIRMED.** Specific defects:
- H4 (Outer Group cluster top edge passes through node A on nested_clusters)
- H5 (sibling clusters Right/Left Branch overlap; labels collide)
- H8 (cluster border stroke crosses through cluster label text)
- M3 (cluster border too dark gray vs dot's near-invisible)
- M4 (cluster fill opacity too high; visible gray rectangle vs dot's
  ghost-tint)
- L3 (deep_nesting_4 nested boxes have stroke crossing through "Level 1",
  "Level 2", "Level 3", "Level 4 (Core)" labels)

### "etc etc"
Captured by:
- M5 (edge label collision on state_machine)
- M6 (color saturation drift on colors_showcase)
- L1 (title font weight mismatch)
- L2 (panel top margin)
- L3 (long-label padding)
- L4 (vertical text centering)

---

## Acceptable Residual (true rendering-stack floor)

Items that CANNOT be eliminated through theme parameter tuning and ARE
legitimately stack-level differences:

1. **Sub-pixel antialiasing**. Once H1 (font), H2 (font size), H3 (ellipse
   ratio) are fixed, the residual stroke softness from matplotlib's
   raster pipeline vs cairo's will still produce a 0.5px softness
   difference on hairlines. Not fixable without switching renderer
   backend.

2. **B-spline channel routing for edges**. dot uses libspline channel
   routing; dagua uses Bezier/circular arc. Once H6 is fixed (curvature
   floor), there will still be a residual difference in the curve PROFILE
   on long curved edges -- dot's Bsplines have inflection points; dagua's
   arcs do not. This is genuinely stack-level.

3. **Layout topology (algorithmic, not cosmetic)**. dot's Sugiyama rank
   assignment vs dagua's pipeline produce different node x/y on complex
   graphs. EXPLICITLY OUT OF SCOPE per the prompt -- "Layouts are held
   constant" -- so this is not a valid finding for cosmetic audit, but
   listed here for completeness.

4. **Font hinting and metrics on this Linux system**. Even if H1 is
   fixed (use TeX Gyre Termes), matplotlib's font engine measures glyph
   widths slightly differently than Cairo+Pango. Residual width
   mis-measurement on the order of 0.5-1px per long label is unavoidable.

---

## Recommendation

**CONTINUE.** Strongly.

This is not a STOP candidate. There are 8 HIGH-severity defects, several
of which directly correspond to the user's expert complaints. The previous
Sonnet 4.6 audit's "indistinguishable at a glance" verdict was wrong --
specifically because:

1. **H1 (wrong font)** is visible in EVERY panel, not just some. Sonnet
   missed this; it requires comparing actual glyph shapes, not just
   "is there text in roughly the right place."
2. **H4, H5, H8 (cluster box defects on nested_clusters)** are
   first-glance failures on the panel set -- node A pierced by cluster
   top-edge, sibling cluster labels colliding, cluster strokes crossing
   labels. Sonnet either didn't look at nested_clusters carefully or
   confused MIDDLE with another panel.
3. **H6 (back-edges drawn as straight verticals through graph body)** on
   state_machine and multi_cycle is a SEVERE routing failure that no
   careful viewer would call "indistinguishable."
4. **H7 (open arrow forms drawn as filled)** on arrow_types is a literal
   correctness issue -- the rendered shape is wrong, not just stylized
   differently.

### Prioritized next-round fix list

| # | Fix | Effort | Impact |
|---|---|---|---|
| 1 | H1 font_family Times New Roman -> TeX Gyre Termes | trivial (1 line) | massive (every panel) |
| 2 | H2 font_size 10.5 -> ~12.0 | trivial (1 line) | large (every panel) |
| 3 | H3 ellipse aspect ratio: use sqrt(2) circumscription | small (renderer fn) | large |
| 4 | H4 cluster box must not overlap external incoming nodes | medium | large |
| 5 | H5 sibling cluster horizontal padding | medium | large |
| 6 | H6 back-edge curvature absolute floor for long chords | medium | large |
| 7 | H7 named-arrowhead per-shape fill rules | medium | targeted |
| 8 | H8 break cluster border stroke around label OR mask label background | small | large |
| 9 | M1 fix arrow tip-to-boundary trim | small | medium |
| 10 | M3 cluster border #AAAAAA -> #CCCCCC, stroke 0.8 -> 0.5 | trivial | medium |
| 11 | M4 cluster fill_opacity 0.15 -> 0.08 | trivial | medium |
| 12 | M2 arrow proportions 10x7 -> 8x8 or 9x7.5 | trivial | small-medium |

H1, H2, M3, M4, M2 are five trivial parameter changes that will close
~50% of the perceptual gap by themselves. H3, H4, H5, H6, H7, H8 require
renderer changes but are well-defined.

After this round, if all 12 fixes land, then a STOP would be defensible
once the residual is documented as: (a) sub-pixel antialiasing, (b)
B-spline vs arc edge profile, (c) font hinting metric differences.

---

## Confidence

- **High confidence** on H1 (font mismatch root cause): verified at OS
  level via `fc-match` -- objective evidence, not visual inference.
- **High confidence** on H2, H3, H4, H5, H6, H8: visible at-zoom on
  multiple panels, multiple times each.
- **High confidence** on H7: arrow_types panel makes the open/filled
  distinction unambiguous.
- **High confidence** on M3, M4, M5: clear at-zoom.
- **Medium confidence** on M1 (tip-spacing): tight inspection required;
  the gap/overlap difference may be 1-2px which is on the boundary of
  what is a rendering-stack floor.
- **Medium confidence** on M2 (arrow proportions): the 10x7 ratio is
  defensible; the gap may close once the rest of arrows-stack converges.
- **Lower confidence** on L1-L4: visible but bordering on careful-
  inspection territory.
- **Want to re-examine after fixes:** ellipse aspect ratio (H3) is the
  hardest to characterize precisely -- the actual graphviz formula
  needs a code dive into Graphviz's `gvNEATO_layout` ellipse path
  generator to confirm `sqrt(2)` is the right multiplier.

---

## Closing note on the prior STOP verdict

Round 4's audit (REPORT_round_3 + AUDIT_round_4) recommended STOP based
on a panel-by-panel checklist. That checklist captured the round-3
fixes correctly but did NOT enumerate the 8 HIGH-severity defects above
because: (a) the wrong-font issue requires fc-match level verification,
not visual inspection alone; (b) the nested_clusters defects are easily
missed if the panel is small in the audit's display; (c) the back-edge
straight-line issue on state_machine appears subtle in thumbnails but
becomes obvious at zoom; (d) the open-arrow-forms issue on arrow_types
is only visible when comparing each named arrow head individually.

The user's expert complaint was correct. Round 4 should not have been
declared final. Round 5 with the 12 fixes above is the correct path
forward.
