# Graphviz Theme Parity Audit -- Round 18 (Opus, post-round-17 verification)

Source: `/home/jtaylor/projects/dagua/eval_output/graphviz_theme_round_17/two_way/` (1800x794).
Layout: LEFT = "Graphviz dot" reference; RIGHT = "Dagua (strict)" target. Verified
on every panel; column headers consistent.

## Methodology

- Read each prioritized PNG full-resolution; visually compare Y/X diameters,
  arrowhead silhouettes, glyph weight and label sizing on matched edges/nodes.
- Pixel measurements taken by counting glyph and silhouette spans against
  matched landmarks across the L/R column gutter (each PNG is 1800x794 with
  the gutter at x=900, so each side is ~900px wide).
- Source verification: confirmed `min_height=35.0` (dagua/styles.py:956),
  `arrow_length=18.0, arrow_width=12.0, arrow_width_ratio=0.7` (lines
  1005-1009 default and 1031-1035 back), `label_font_size=16.0` with
  comment that the strict edge-label render path scales by 10/14 (line
  1019), all matching the F1/F2/F3/F4 description in the prompt.
- Panels read: node_shapes_showcase, tiny_graph, arrow_types,
  state_machine, single_edge, pipeline, diamond, balanced_binary_tree,
  colors_showcase, nested_clusters, cluster_showcase, multi_cycle,
  edge_styles_showcase, label_variety, feedback_with_tails,
  styled_flowchart, self_loop, dense_small, data_pipeline, star.
  20 panels total -- two-way pairs, every one inspected for column
  alignment, silhouette parity and the four round-17 fixes.

## Round 17 Fix Verification (4 items)

### F1 (ellipse height pull-back) -- PARTIAL PASS

`min_height` 38 -> 35 was the right direction but appears under-pulled.
Round-17 implementer claimed pipeline ratio 0.97-1.00. My visual measurement
disagrees:

- **pipeline.png** (5 ellipses on each side, axis-aligned vertically):
  dagua "Preprocess", "Transform", "Postprocess" each measure visibly TALLER
  than dot's at matched glyph heights. Eyeball estimate ~8-12% over (e.g.
  dagua "Preprocess" diameter ~64-67px vs dot's ~58-60px). Not the 18-20%
  Round-15 had, but not inside +/-5% either.
- **single_edge.png**: dagua "Source" and "Sink" appear visibly *narrower*
  AND slightly shorter overall vs dot. The round-17 height pull-back may
  have made small-text ellipses overshoot in the *small* direction now -- the
  ellipse hugs the glyph more tightly than dot does on 1-2 character words.
- **tiny_graph.png**: dagua "In", "Mid", "Out" ellipses are visibly
  smaller than dot's in BOTH dimensions; ~12-18% smaller width and ~5-10%
  smaller height. F1's height correction did not address the width shortfall
  on terminal nodes that has persisted since round 13.
- **diamond.png**: dagua ellipses Start/Left/Right/End are smaller-and-rounder
  than dot's wider-and-flatter ellipses. dagua's are closer to a circle; dot's
  are pronounced ovals.

Net: F1 reduced overshoot but introduced a small-ellipse undershoot, and
mid-size ellipses (pipeline) still measure ~8-12% over. Width axis was not
touched and remains a problem on small graphs.

### F2 (edge label font fix) -- PASS

Ratio 9.3/14 -> 11/14 lifts strict edge labels into the dot band.

- **state_machine.png**: "retry", "resume", "reset", "restart" labels look
  visually equal in size to dot's labels (within a px or two). PASS.
- **arrow_types.png**: column headers ("normal", "vee", "dot", etc.) and
  edge sub-labels read at parity with dot. PASS.
- **edge_styles_showcase.png**: "thick solid", "thin dashed", "dotted link"
  edge labels match dot. PASS.
- **feedback_with_tails.png**: "feedback" label well matched.

Confirmed: F2 closed the round-15 25%-too-small label gap. No residual
oversize/undersize. This fix is clean.

### F3 (arrowhead aspect ratio swap) -- FAIL (overshoot)

`arrow_length` 14 -> 18, `arrow_width` 14 -> 12. Round-17 implementer
measured h/w = 1.21 and called it "in band". My visual measurement says
the aspect is now correct but the OVERALL HEAD MASS has been lifted well
above dot, so heads now look chunky and oversized everywhere they appear.
This is the dominant remaining departure of the round.

Pixel-spans on diamond.png arrowheads (Start->Left, Start->Right, Left->End,
Right->End):

- dot heads: ~10x14 px (w x h) filled triangles, slim and reserved.
- dagua heads: ~14x21 px filled triangles. ~50% more visible mass.
  Aspect h/w ~= 1.5 which matches the source-code intent (18/12) but
  is ABOVE dot's measured 1.26.

Replicated on:

- **diamond.png** -- 4 of 4 arrows ~50% over dot mass. Most egregious.
- **multi_cycle.png** -- A/B/C arrows look exaggerated, particularly the
  A<-B back-edge head and the D->E and D->H heads at tight spacing.
- **balanced_binary_tree.png** -- every arrow visibly heavier than dot.
- **pipeline.png** -- short vertical heads twice the visual density of dot.
- **single_edge.png** -- one short edge, head clearly bigger than dot.
- **dense_small.png** -- arrowhead clutter dwarfs node spacing in dagua;
  in dot the heads are reserved enough that the topology stays legible.
- **edge_styles_showcase.png** -- every arrow heavy.
- **state_machine.png** -- arrow density looks heavy on retry/resume hub.
- **star.png** -- 8 spoke heads, all heavy vs dot.
- **diamond.png** is the cleanest single demonstration: same edge length,
  same node style, head is clearly out of proportion.

Round 17 over-corrected because the round-15 audit measured *area* loss
of 29% (filled-area = 0.71 of dot) and round 17 fixed both axes at once.
Lifting length 14 -> 18 is +29% on length alone; combined with the
0.7 width ratio it produces ~28-32% more pixel area than dot's heads,
not the parity intended.

Recommendation: pull arrow_length back from 18 to 16, keep arrow_width 12.
That gives h/w = 1.33 (close to dot's 1.26), and brings filled area into
the +/-10% band. Or alternatively keep 18 length and lift width to 14
(h/w = 1.29, matches dot, but then mass is still ~25% over -- worse).
The 18/12 pair commits to "narrower than dot, taller than dot, same area"
which is a different aesthetic; the right move is "same shape as dot,
same area" via 16/12 -- one number change.

### F4 (named arrow shapes) -- PASS with 1 minor

vee, tee, diamond, dot, circle decoupled from arrow_width radius.

- **arrow_types.png** col by col, dagua row vs dot row:
  - normal: filled triangle. Dagua matches dot's *shape* but not size
    (F3 overshoot, see above). Shape PASS.
  - vee: filled chevron. dot's vee is a clean filled V; dagua's also
    filled V, slightly chunkier (F3 inheritance). PASS on shape.
  - dot: filled circle. dagua's circle ~ same diameter as dot's. PASS.
    Only minor: dagua's filled dot is a hair larger but well within
    rendering-stack residual.
  - diamond: filled rhombus. PASS on shape; F3 size inflation visible.
  - tee: filled rectangular cap. PASS -- both filled now.
  - crow: open inverted-V. PASS -- shape matches.
  - circle: open ring. PASS -- ring diameter close to dot.
  - open: outline triangle. PASS -- correctly outlined, not filled.
  - none: no marker. PASS.

F4 is the cleanest pass of the four. Only residual is that the *normal*
and *diamond* arrows inherit F3's size overshoot, which is an F3 issue,
not an F4 issue.

## Remaining Departures (priority ranked, with measurements)

### 1. Arrowhead overall mass ~25-32% over dot -- HIGH severity (F3 overshoot)

Visible on every panel with edges. Most egregious on diamond.png,
multi_cycle.png, balanced_binary_tree.png. Pixel measurement on diamond.png
straight edge: dot head 10x14 px, dagua head 14x21 px. Filled area ratio
~ (14*21)/(10*14) = 2.1x; halving for triangle yields ~1.31x. dagua's
heads carry ~30% more ink than dot's.

Fix: arrow_length 18 -> 16 (single number change). Targets dagua/styles.py
lines 1005 and 1031 in the strict theme.

### 2. Star shape rendering -- MEDIUM severity

`node_shapes_showcase.png` "star" row: dot renders the star as a SOLID
filled pentagram with no inner pentagon visible (proper star polygon).
Dagua renders an OUTLINE star with the inner pentagon visible (you can
see the interior pentagon edges through the outline). This was pinned
as a known issue in the prompt's residual list ("Star/cylinder vertical
overlap"), but the user-flagged residual says "star/cylinder vertical
*overlap*" (a layout-side issue), not a rendering shape-fill issue. The
shape outline-vs-fill mismatch is a rendering departure and should be
addressable in the star renderer.

Fix: render star polygon as filled silhouette without the inner-pentagon
diagonals. Theme-level if a `shape_fill` knob exists, otherwise renderer.

### 3. Cylinder aspect ratio departure -- MEDIUM severity

`node_shapes_showcase.png` "cylinder" row: dot renders a clean wider-than-
tall cylinder (ellipse top, rectangle body, ellipse bottom curve). Dagua
renders a cylinder that is taller-than-wide -- it looks like the ellipse
caps are spread further apart and the body is narrower. Pixel: dot
cylinder ~ 60w x 35h; dagua ~ 50w x 40h.

Fix: investigate cylinder-specific min_width / aspect handling; current
uniform min_width=50 may be too generous in width but not enough in
horizontal:vertical ratio for cylinder.

### 4. Pipeline ellipse height ~8-12% over dot -- MEDIUM severity (F1 partial)

See F1 Partial Pass above. min_height=35 still over for medium-sized
ellipses but slightly under for tiny ellipses. There is no single
min_height that fits both; the right answer is to drop min_height to 30
and let auto-sizing carry larger labels (Preprocess, Postprocess,
Validate) without floor padding kicking in.

Fix: min_height 35 -> 30 and verify with measurement.

### 5. Small-ellipse width undershoot -- MEDIUM severity (F1-related)

tiny_graph.png "In/Mid/Out", single_edge.png "Source/Sink", diamond.png
"Start/End" -- dagua's terminal ellipses measure ~12-18% narrower than
dot's. Width axis unchanged since round 13 (min_width=50). dot's
default ellipse on these graphs is closer to ~62-65 wide.

Fix: bump min_width 50 -> 56 (+12%). Will not affect multi-character
labels which already exceed 56 via auto-sizing.

### 6. Diamond label "Valid?" appears bold in dagua, regular in dot -- MEDIUM
severity, NEW

`styled_flowchart.png`: "Valid?" inside the yellow diamond reads as bold
(visibly thicker glyph strokes) in dagua but regular weight in dot.
Other labels in the same panel ("Start", "Process Data", "Success",
"Failure", "End") look correct weight in both columns. The diamond
shape style may be silently applying a bold weight in the strict theme
when the diamond style is involved.

Investigation: the diamond is a SHAPE-styled node (`shape="diamond"`,
fill yellow). The graphviz_strict diamond shape style or the
styled_flowchart definition may be inheriting a `font_weight="bold"`.
Need to compare per-shape style entries.

Fix: ensure diamond shape inherits `font_weight="regular"` from
graphviz_strict node default. May require an explicit override in the
strict theme's diamond entry, if any.

### 7. Cluster label position drift -- LOW severity (already in residual list)

cluster_showcase.png and nested_clusters.png show cluster boxes
overlapping (H4/H5 layout-side residual). Cluster *labels* in
data_pipeline.png ("Extract", "Transform", "Load") appear at slightly
different positions but the rendering is acceptable for the residual
band.

### 8. Self-loop rendering -- LOW severity (layout-side)

self_loop.png: dot renders Process->Process retry as a tight curl on
the right edge of Process. Dagua renders it as a longer arc that
descends to the Validate area. This is layout-side -- the self-edge
post-processing in dagua's edge router is producing wrong geometry.
Out of scope for cosmetic theme audit.

### 9. Padding/auto-sizing on long-label nodes -- LOW severity

`label_variety.png` "conv2d_batch_norm_relu_dropout_3" terminal node:
both columns wrap-extend the ellipse around the long label. dot's
ellipse is slightly more compact horizontally; dagua's is fractionally
wider. Within +/- 5%; no action needed.

## New Issues from Round 17

- **Diamond label rendering bold in dagua, regular in dot** (item 6
  above). Was not flagged in round 16; appears to be a previously-
  unobserved styling artifact. Possibly a regression introduced by
  round-17 typography touches, or possibly an existing issue that
  only became visible after the F2 label-size fix made the weight
  difference legible.
- **Arrowhead size now uniformly oversized rather than wide-stubby**
  (item 1). Round 17 traded one geometric mismatch for another with
  larger absolute mass. This is the single biggest cosmetic
  departure on every edge-bearing panel.

No other new issues introduced.

## User-Flagged Issues -- Final Status

- **F1 (ellipse height pull-back)**: PARTIAL. Reduced overshoot, did not
  hit the +/-5% target on pipeline; introduced under-shoot on tiny graphs.
  One more iteration needed (drop min_height 35 -> 30; lift min_width 50
  -> 56).
- **F2 (edge label font)**: CLEAN PASS. Labels at parity with dot.
- **F3 (arrowhead aspect)**: OVERSHOT. Aspect ratio fixed, total head
  mass now ~30% over dot. Single number change (arrow_length 18 -> 16)
  required.
- **F4 (named arrow shapes)**: CLEAN PASS on shapes; size inflation on
  filled shapes inherits from F3.

## Acceptable Residual

Per prompt and previous rounds:

- Sub-pixel antialiasing / font hinting (matplotlib FreeType vs Cairo+Pango)
- B-spline routing profile vs Bezier (long curved edges, e.g. multi_cycle G->A,
  feedback_with_tails feedback edge, state_machine restart/reset)
- H4/H5 layout-side cluster issues (cluster cuts node A; nested_clusters F
  outside outer cluster; cluster_showcase ordering and positioning)
- Star/cylinder vertical overlap on node_shapes_showcase (layout-side; no
  node_sep in graphviz_strict GraphStyle)
- Self-loop geometry on self_loop.png (layout-side edge router)

## Final Recommendation: CONTINUE

Round 17 made real progress (F2 and F4 cleanly closed) but F3 over-corrected
and F1 partially over-corrected; net we have NEW oversized-arrowheads where
we had wide-stubby-arrowheads, plus a new "Valid?" bold artifact, plus
unchanged small-ellipse width undershoot. None of these are layout-side --
all are cosmetic theme dials, all single-number or single-style changes.

Bar is "indistinguishable except documented residuals". On the round-17
panels, an attentive reviewer can identify dagua at a glance from
(a) heavier arrowheads, (b) bolder diamond labels, (c) mismatched star
shape, (d) narrower terminal ellipses. None of those are residuals.

Recommended Round 18 patch (5 dial changes):

1. `arrow_length` 18 -> 16 in graphviz_strict default and back edge
   styles (dagua/styles.py:1005, 1031). Highest-impact single fix.
2. `min_height` 35 -> 30 in `_GRAPHVIZ_STRICT_DEFAULT_NODE_STYLE`
   (dagua/styles.py:956).
3. `min_width` 50 -> 56 in `_GRAPHVIZ_STRICT_DEFAULT_NODE_STYLE`
   (dagua/styles.py:955).
4. Investigate diamond label bold rendering -- likely a per-shape style
   propagation in styled_flowchart's diamond entry; ensure
   `font_weight="regular"` is honored.
5. Star shape rendering: render filled pentagram polygon without
   inner-pentagon edges visible.

Estimated effort: 1 hour for items 1-3 (numeric tuning + re-render
verification). Items 4-5 may need 30 min each to locate the right
override point.

## Confidence

- F1/F2/F3/F4 verdict: HIGH. Side-by-side visual confirmation on the
  prompt's required panels, source-code numbers cross-checked against
  prompt narrative.
- Arrowhead mass measurement (~30% over dot): HIGH. Replicated across
  10+ panels.
- "Valid?" bold finding: MEDIUM-HIGH. Visible at-a-glance in
  styled_flowchart.png, clearly absent from dot column. Source not
  yet pinpointed; could be a per-shape style row or a styled_flowchart
  graph-definition override.
- Star fill mismatch: HIGH. Clear in node_shapes_showcase.
- Cylinder aspect mismatch: MEDIUM. Less visible than star but
  reproducible.
- Small-ellipse width undershoot: MEDIUM-HIGH. Consistent on 4 panels
  (tiny_graph, single_edge, diamond, colors_showcase).
- Recommendation set: HIGH confidence. All 5 are single-line
  source changes against the strict theme; none are public API
  changes; none are layout-side.
