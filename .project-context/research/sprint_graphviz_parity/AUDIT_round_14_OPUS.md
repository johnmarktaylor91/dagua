# Graphviz Theme Parity Audit -- Round 14 (Opus, post-round-13 verification)

## Methodology

- Source dir: `/home/jtaylor/projects/dagua/eval_output/graphviz_theme_round_13/two_way/` (1800x794)
- Panels reviewed (12): node_shapes_showcase, tiny_graph, single_edge, arrow_types, state_machine, pipeline, diamond, balanced_binary_tree, colors_showcase, nested_clusters, cluster_showcase, multi_cycle.
- Column headers verified on every PNG: yes ("Graphviz dot" left, "Dagua (strict)" right).
- Pixel measurement: per-node ellipse width/height at midline (PIL/numpy script). Arrow-head width measured row-by-row in the inter-node band. Star/cylinder zoom crops saved to /tmp for color verification.
- Bar applied: "genuinely identical save for documented rendering-stack residuals." Maximally picky.

## Round 13 Fix Verification (6 items)

### F1 (node size recovery) -- PARTIAL

Round 11 had nodes ~70% of dot. Target was ~10% (i.e. 0.9-1.1). Per-node measurements:

| panel | dot WxH (px) | dagua WxH (px) | W ratio | H ratio |
|---|---|---|---|---|
| tiny_graph "Mid"      | 156x149 | 152x149 | 0.97 | 1.00 |
| tiny_graph "In"       | 132x169 | 140x148 | 1.06 | 0.88 |
| tiny_graph "Out"      | 147x147 | 151x126 | 1.03 | 0.86 |
| single_edge "Source"  | 201x134 | 190x116 | 0.95 | 0.87 |
| single_edge "Sink"    | 155x132 | 148x114 | 0.96 | 0.86 |
| diamond "Start"       | 132x169 | 140x142 | 1.06 | 0.84 |
| diamond "End"         | 147x147 | 150x120 | 1.02 | 0.82 |
| pipeline "Preprocess" | 182x89  | 206x89  | 1.13 | 1.00 |
| pipeline "Output"     | 140x71  | 148x69  | 1.06 | 0.97 |

**Verdict: PARTIAL.** Width recovered well (0.95-1.13, mostly within 10%). **Height undershoots on the small ellipse-only graphs by 12-18%** (tiny_graph, single_edge, diamond). The "In/Source" measurements are inflated on dot's side because the descending arrow tail is touching the node bottom, exaggerating the dot height. But even discounting that, dagua's terminal-row nodes (Out/Sink/End) are 14-18% shorter than dot's. min_height 27->33 was not enough -- needs another small bump (~38) or padding asymmetry (top/bot pad unbalanced because text sits low).

### F2 (star shape) -- FAIL

Round-13 fix: "reverted star compact factor to 2.2x + STAR_INTERIOR_FACTOR; equalized w=h=max(w,h)." Implementer flagged as residual.

What I see on node_shapes_showcase.png:

- **Star color regression: dagua's star is GRAY/light-pen, not solid black.** Pixel sample at the star location: mean RGB 156,156,156 (anti-aliased gray). Dot's star at the same location renders sharper black-on-white edges. The five inner triangle lines that form the pentagonal interior of the star are being drawn in a very light pen (effectively #555-#999) versus dot's full-strength stroke. Possible cause: the equalization to `max(w,h)` plus the 2.2x compact factor is pushing the star outside its anti-aliasing envelope so the rasterizer halves the alpha, OR the STAR_INTERIOR_FACTOR is multiplying alpha down rather than dimensions.
- **Star+cylinder collision: dagua's star and cylinder bounding boxes overlap.** The "star" label text ends up *inside* the cylinder shape from above (visible "star" label clearly intrudes into "cylinder" region). On dot, star and cylinder are vertically separated by clean whitespace. This is a layout-side spacing miss, not a star-shape miss per se, but it is plainly visible.
- Star outline is at least the right outer pentagram silhouette and is approximately square per the equalization, so the geometry fix works.

**Verdict: FAIL on color/visibility (gray pen), FAIL on cylinder overlap, PASS on geometry.**

### F3 (ellipse curved factor) -- PARTIAL

`curved_factor 1.0 -> 1.15` on ellipse. Visible improvement: dagua's ellipses on node_shapes_showcase, tiny_graph, single_edge, diamond, pipeline are now visibly curved (no longer pinched).

However, dagua's ellipses still appear **slightly narrower (more circular) than dot's** especially on multi-character labels: pipeline.png shows dagua's "Preprocess" ellipse just barely fits the label whereas dot's has more horizontal slack. And on tiny_graph, dagua's "In"/"Mid"/"Out" ellipses look more like circles than dot's (which are unmistakably horizontally elongated ellipses). 1.15 may need to push to 1.20-1.25.

**Verdict: PARTIAL.** Better than round 11, still measurably "rounder" than dot.

### F4 (edge label font) -- PARTIAL

`_strict_edge_label_font_size` returns ~11.43pt while node label theme stays at 16.0. Target: edge labels visibly subordinate.

- **arrow_types.png:** Dagua's labels ("normal", "vee", "diamond", "circle") look very close in size to dot's. They are NOT visibly subordinate to node labels in this panel because the node labels themselves are small here. No regression.
- **state_machine.png:** Dagua's edge labels ("restart", "reset", "retry", "resume") are visibly smaller than node labels ("Idle", "Initialize", "Ready", "Running"), matching dot's pattern. F4 IS working here.
- **However**: dagua's edge labels still look slightly LARGER than dot's edge labels in absolute terms on state_machine and arrow_types -- visual eyeball says ~10-15% larger. The 10/14 ratio brings it closer but did not finish the job.

**Verdict: PARTIAL.** Subordination achieved; absolute size still 10-15% above dot.

### F5 (arrow size) -- PARTIAL

`arrow_length 12->14, arrow_width 10->12`. Per-row max-width measurement on tiny_graph arrows:

| arrow | dot max width (px) | dagua max width (px) | width ratio | area ratio |
|---|---|---|---|---|
| In->Mid     | 22 | 20 | 0.91 | 0.71 |
| Mid->Out    | 22 | 20 | 0.91 | 0.94 |

**Verdict: PARTIAL.** Bumping closed some of the gap (round-11 likely 0.75-0.80) but dagua arrowheads still 9% narrower than dot's. Worse, the *area* ratio (filled black mass) of 0.71 on the In->Mid arrow says dagua's arrowhead is 30% less ink than dot's -- the head is also visibly less "chunky" / shorter base-to-tip in side-by-side. tiny_graph arrows clearly show dot's heavier filled triangle vs dagua's slimmer one. Recommend arrow_length 16, arrow_width 13 OR a 1.2x scale on arrow stroke fill.

### F6 (stroke weight) -- PASS (with note)

`node stroke_width 0.75 -> 0.9`.

- pipeline.png: dagua's ellipse stroke now reads as the same line weight as dot's at viewing distance. PASS.
- diamond.png: matches dot. PASS.
- tiny_graph.png: dagua's stroke still reads slightly thinner than dot's, but the gap is much smaller than round 11.
- node_shapes_showcase.png: rect/diamond/hexagon strokes look matched to dot.

**Verdict: PASS.** F6 brought stroke weight into "indistinguishable at glance" range. There may be a tiny residual ~0.05pt below dot but it is not a priority delta.

## Round 13 tally

| Fix | Verdict |
|---|---|
| F1 node size | PARTIAL (height short on small ellipses) |
| F2 star | FAIL (gray pen + cylinder overlap) |
| F3 ellipse curve | PARTIAL (still rounder than dot) |
| F4 edge label font | PARTIAL (subordinate but still ~12% larger absolute) |
| F5 arrow size | PARTIAL (~91% width, ~71-94% area) |
| F6 stroke weight | PASS |

**Score: 1 PASS / 4 PARTIAL / 1 FAIL.**

## Remaining Departures (priority ranked, with measurements)

### P1 -- Star pen rendered light-gray on node_shapes_showcase
Sample mean RGB ~156,156,156 vs dot's sharp black edges. The interior pentagram lines are essentially invisible at viewing distance, making the star look like a five-pointed outline rather than dot's filled pentagram silhouette. Critical visual departure -- "different shape entirely" at glance. Fix: trace the star path code -- equalization + STAR_INTERIOR_FACTOR is producing sub-pixel edges that anti-alias to gray. Either thicken stroke for star specifically or stop dividing dimensions inside the path emitter.

### P2 -- Edge label absolute font size still ~10-15% above dot
F4 brought subordination; absolute size needs another tightening. Suggest dropping multiplier from 10/14 to 9/14 (i.e. ~10.3pt) for edge labels.

### P3 -- Arrow head still 9% narrower / 30% less filled mass on short edges
F5 partial. Bump arrow_length to 16 and arrow_width to 14, OR ensure the arrow fill is a proper closed polygon rather than a stroked outline. The "area ratio 0.71" on tiny_graph In->Mid suggests the arrowhead might be partially open / unfilled.

### P4 -- Small-graph ellipse height undershoot (Sink/Out/End nodes 14-18% shorter than dot)
F1 partial. min_height 33 isn't enough for the terminal nodes in small graphs. Suggest min_height 38, OR add an asymmetric padding (extra top/bottom) so single-line text gets adequate vertical room.

### P5 -- Ellipse aspect ratio still slightly more circular than dot
F3 partial. Bump curved_factor to 1.22 to match dot's wider-than-tall ellipse signature on multi-char labels.

### P6 -- Cylinder shape on node_shapes_showcase
Dagua's cylinder reads as "rect with horizontal mid-line" rather than dot's proper cylinder (rect with curved top + curved bottom). Cylinder rendering needs a separate look. May overlap into H4/H5 deferred bucket if it's a layout-spacing artifact.

### P7 -- arrow_types "tee" arrowhead misaligned
Dagua's tee arrowhead is detached from the edge line by a visible gap (the horizontal bar floats above the edge tip). Dot's tee bar sits flush against edge end. Visible at panel center.

### P8 -- arrow_types "vee" arrowhead degenerate
Dagua's vee on arrow_types is rendered as a barely-visible chevron whereas dot's is a clean V. Outline weight too thin or coordinates collapsed to near-overlap.

### P9 -- arrow_types "diamond" filled mark uneven
Dagua's solid diamond arrow is roughly square in profile, while dot's is taller-than-wide -- shape proportion mismatch on this arrowhead form.

### P10 -- arrow_types "open" arrowhead sized correctly but "none" panel header in dagua appears off-spec (label "none" is rendered above the target node rather than next to the edge stub)
Visible label-position swap between dot's "none" (label below stub) and dagua's "none" (label up near the top "none" source node area).

## New Issues from Round 13

- **NR1 (regression):** Star color is now gray/desaturated (was black in round 11). The compact-factor revert plus equalization introduced a stroke alpha problem.
- **NR2 (regression):** Cylinder shape on node_shapes_showcase is missing its bottom curve (just a rect with line through middle); not flagged in earlier rounds.
- **NR3 (probable side-effect):** F1 size bump pushed the star into the cylinder's row in node_shapes_showcase (overlap visible). Layout spacing did not absorb the F1 size growth -- node_sep needs to scale with min_height/min_width.

## User-Flagged Issues -- Final Status

- **"Arrows are wonky"** -- NOT YET RESOLVED. F5 closed ~half the gap; dagua arrowheads still 9% narrower and 30% less filled than dot on short edges. Tee/vee/diamond arrow variants on arrow_types panel show shape misalignment, detached bars, and proportion errors.
- **"Text isn't centered"** -- RESOLVED for the panels reviewed. Node labels appear vertically centered in their shapes. (No measurement disputed this.)
- **"Different font"** -- RESOLVED (already accepted as font-family residual; no change in round 13 needed).
- **"Cluster bounding boxes look like shit"** -- DEFERRED per H4/H5. nested_clusters.png still shows clusters cutting through nodes ("Outer Group" label is BEHIND node A; "Right Branch" / "Left Branch" labels collide with nodes; clusters do not contain F). Acknowledged outside cosmetic scope.

## Acceptable Residual

- Sub-pixel antialiasing (matplotlib FreeType vs Cairo+Pango).
- Font hinting differences in label edges.
- B-spline routing profile vs Bezier on long curved edges (state_machine "reset" curve, multi_cycle back-edges).
- Layout-side cluster issues (H4/H5).

## Final Recommendation: **CONTINUE**

This is not "genuinely identical save for documented residuals." Specifically:
- The gray-star regression is "different shape entirely" at glance, not a sub-pixel issue.
- The arrow-head deficit is still visible without measuring.
- The edge-label font still reads ~10-15% larger than dot.
- Five PARTIAL items mean five items where the fix landed but undershot.

Recommended Round 14 spec (one round, six narrow fixes):

1. **Star pen fix:** ensure star path is stroked at full theme stroke_width (no alpha multiplication, no dimension division for interior segments). Verify rendered pixel = solid black on white background.
2. **min_height 33 -> 38** OR add 3px asymmetric vertical padding so single-line ellipses match dot's vertical generosity.
3. **curved_factor 1.15 -> 1.22** to widen ellipses on multi-char labels.
4. **arrow_length 14 -> 16, arrow_width 12 -> 14**, and verify arrowhead is filled polygon not stroked outline (audit area-ratio metric).
5. **edge label font ratio 10/14 -> 9.3/14** (give back ~7% to fully drop below dot).
6. **node_sep auto-scale** with min_height so F1 growth doesn't cause star/cylinder overlap on showcase panels.

Optionally, a separate small fix for cylinder shape on node_shapes_showcase (proper top + bottom curve rather than mid-line).

Stop after round 14 only if all six land within pixel-measurement tolerance (size ratio 0.95-1.05, font ratio within 5%, no color regressions).

## Confidence

**HIGH** on the per-fix verdicts (backed by pixel measurements on six panels and direct visual inspection on twelve). **HIGH** on the gray-star regression (pixel sample mean 156,156,156 vs dot's deeper black). **MEDIUM-HIGH** on the recommended numeric bumps -- the magnitudes are extrapolated linearly from the residual deficits and may overshoot slightly; verify on a single iteration before committing.
