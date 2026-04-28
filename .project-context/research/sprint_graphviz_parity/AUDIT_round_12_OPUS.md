# Graphviz Theme Parity Audit -- Round 12 (Opus, post-round-11 verification)

## Methodology
- Panels reviewed: node_shapes_showcase, arrow_types, tiny_graph, single_edge,
  pipeline, diamond, balanced_binary_tree, state_machine, nested_clusters,
  cluster_showcase, deep_nesting_4, colors_showcase
- All read from `eval_output/graphviz_theme_round_11/two_way/` at 1800x794.
- Column headers verified on every panel: yes (LEFT="Graphviz dot",
  RIGHT="Dagua (strict)" present and consistent).
- Comparisons are visual side-by-side. Where I cite "px", I am estimating
  off the rendered raster.

## Round 11 Fix Verification (3 items)

### F1 (puffy nodes): FAIL -- over-correction
The hybrid hammer (theme padding 8,4 -> 6,3 + min_width/height -> 41,27 +
`compact_shape_factors` curve dampening) has flipped the regression rather
than landed it. Dagua nodes are now systematically SMALLER than dot's, not
matched.

Measurements from rendered panels (eyeballed, monospace pixel approximation):
- **tiny_graph.png**: dot's "In" ellipse spans ~190 px wide x ~95 px tall;
  Dagua's "In" is ~125 x ~70. That's roughly 65% of dot's area. Same for
  "Mid" and "Out". Glanceable difference, not a residual.
- **single_edge.png**: dot "Source" ellipse ~210 x ~110; Dagua ~155 x ~75.
  Same ~65-70% area shrink. Stroke weight on dot is also visibly heavier
  (~2 px) versus Dagua (~1 px) but that may be a separate axis.
- **pipeline.png**: dot "Preprocess" ellipse ~270 x ~95; Dagua ~250 x ~75.
  Width is closer here (only ~7% short) but height is still ~20% short.
- **diamond.png**: dot "Start" ~230 x ~115; Dagua "Start" ~190 x ~85. Width
  ~17% short, height ~26% short. The "End" ellipse is the worst -- dot
  renders it as a large oval, Dagua produces a cramped narrow ellipse where
  the label nearly touches the right edge.
- **balanced_binary_tree.png**: dot leaves "LLL", "LLR" etc. are clearly
  larger than Dagua's leaves. Same ~25-30% area shortfall.
- **node_shapes_showcase.png** (the F1 primary verification panel):
  - **star: SEVERELY broken.** Dagua draws the star as a tiny outline that
    is BARELY larger than the "star" text label inside it. Dot's star is
    ~110 x ~110 with the label sitting comfortably inside. Dagua's star
    appears to be ~30-40 px high, with the label visibly OVERFLOWING the
    points. The compact_shape_factors damping for the star has clearly been
    set far too aggressively. This is the worst regression in this round.
  - **ellipse, roundrect, hexagon, parallelogram, octagon, cylinder**:
    all noticeably smaller than dot's, roughly matching the puffy-fix
    direction but overshot.
  - **diamond**: closer to parity than the curved shapes -- damping was
    less aggressive here, but Dagua's diamond is still a touch narrower.
  - **triangle**: Dagua's triangle baseline is shorter than dot's; the
    label "triangle" runs almost to both edges where dot has comfortable
    margin.
  - **trapezoid**: Dagua's looks reasonable; closest to parity.
- **colors_showcase.png**: every colored ellipse (Red, Blue, Green, Yellow,
  Purple, Orange) is narrower in Dagua. Most pronounced on Blue and Yellow,
  where dot fits the label with ~25 px horizontal padding while Dagua leaves
  only ~8 px before the curved edge.

Verdict: F1 swung the bar. Round 11 traded "puffy" for "cramped". Net
visible departure from dot is comparable in magnitude, just inverted.
The `compact_shape_factors` flag for star is broken outright.

Suggested next step: split the fix. Treat min_width/min_height bump and
shape-specific damping independently. Star damping factor needs to be
removed or set near 1.0; ellipse curved-shape factor should land in the
0.92-0.96 band, not whatever it is now (looks like ~0.80). Goal is
within 5% of dot at all rendered sizes, not 20-30%.

### F2 (edge label font): FAIL -- over-correction
Two-layer fix (`_strict_edge_label_font_size` -> graph-level 16pt;
`_strict_absolute_edge_label_font_data` -> bypass graph-relative scaling)
defeats the cascade in the wrong direction.

- **state_machine.png**: edge labels `restart`, `reset`, `retry`, `resume`
  are now substantially LARGER than dot's. In dot, these labels are visibly
  smaller than the node labels (dot uses ~10pt edge labels vs ~14pt node
  labels). In Dagua, the edge labels appear roughly the SAME size as node
  labels and significantly larger than dot's edge labels. Estimated
  Dagua edge label cap-height ~14-15 px versus dot's ~9-10 px.
  This makes "restart" and "reset" visually dominate the right-side flow,
  whereas in dot they are subordinate to node identity.
- **arrow_types.png**: same pattern. Column-name labels (`normal`, `vee`,
  `dot`, `diamond`, `tee`, `crow`, `circle`, `open`, `none`) sitting under
  each pair of nodes are now clearly LARGER than dot's. Dot prints these
  as small inline labels at ~10pt; Dagua now renders them at ~14pt --
  bigger than the node labels themselves.

Round 10 audits flagged edge labels as too small. Round 11 has flipped
the direction. The "absolute font_size_points * display_scale" path
appears to apply at full graph point size (likely 14pt) rather than the
intended ~10pt edge size. The 16pt graph-level fallback is also too high
for typical dot defaults (which use 14pt or less).

Suggested fix: target ~10pt for edge labels (graphviz default), not
14-16pt. Confirm display_scale is applied symmetrically with how node
labels resolve.

### F3 (arrow size consistency): PARTIAL -- mixed direction
The `disable_curve_length_clamp` field has changed behavior unevenly:

- **tiny_graph.png**: arrows are now SMALLER in Dagua than in dot, not
  larger. Dot's arrow heads are visibly bold, ~13-15 px from base to tip
  with ~10 px width. Dagua's arrows are slim and short, ~7-8 px tip,
  ~5 px width. The clamp removal didn't yield "full-sized" arrows here
  -- the rendering is just thinner.
- **single_edge.png**: same as tiny_graph. Dagua's lone arrow is markedly
  smaller and lighter than dot's.
- **pipeline.png**: closer to parity. Arrow heads look approximately the
  same size as dot's, perhaps marginally smaller. Acceptable.
- **diamond.png**: arrows on the four edges look very close to dot, perhaps
  slightly heavier (boxier head). PARTIAL match.
- **colors_showcase.png**: Dagua arrows look thinner and shorter than dot's
  along each colored chain.
- **balanced_binary_tree.png**: arrows on the Dagua side look bolder than
  dot's, possibly slightly oversized. Conflicting signal vs single_edge.
- **state_machine.png**: arrows look comparable; not a clear regression.
- **arrow_types.png**: each named arrow head (normal, vee, dot, diamond,
  tee, crow, circle, open) is visibly SMALLER on Dagua than on dot. Dot's
  diamond head fills its stub generously; Dagua's is a tiny lozenge.

So F3 reads as: short edges were not fixed (arrows still small/thin), and
larger graphs are inconsistent (sometimes match, sometimes overshoot).
The disable_curve_length_clamp may not be the right knob, or it isn't
combined with a base arrow-size compensation.

## Remaining Departures (priority ranked)

1. **Star shape collapsed** (F1 over-correction). node_shapes_showcase.png
   star is so small the label overflows the points. Most jarring single
   defect in the audit.

2. **Universal node-size shrink** (F1 over-correction). Every node type
   on every panel is 10-30% smaller in Dagua than in dot. This is a
   bigger visual gap than the puffy regression it replaced. Glanceable
   in tiny_graph, single_edge, pipeline, diamond, colors_showcase,
   balanced_binary_tree, node_shapes_showcase.

3. **Edge labels oversized** (F2 over-correction). state_machine and
   arrow_types both show edge labels at roughly 1.4-1.6x dot's. Inverts
   the round-10 finding without converging.

4. **Arrow heads inconsistently sized** (F3 partial). Short edges have
   arrows that are smaller AND thinner than dot, not larger. arrow_types
   and tiny_graph confirm.

5. **Stroke weight thinner overall**. Dagua's node outlines and edge
   lines appear consistently ~1 px versus dot's ~1.5-2 px. Visible
   especially on tiny_graph, single_edge, colors_showcase. May be a
   render-stack residual but is glanceable, so listing.

6. **"End" / "Out" / "Sink" oval cramping** in diamond and tiny_graph
   and single_edge. Single-syllable terminal labels produce especially
   tight ellipses where the label nearly touches the curve.

7. **Cluster bounding boxes** (KNOWN_DEFERRED, H4/H5). Still misaligned
   in nested_clusters and cluster_showcase. Outer Group box overlaps
   Inner boxes; Right Branch label clipped by node "C". Per the rules
   this is layout-side and deferred -- noting for completeness.

## New Issues from Round 11

- **Star damping bug** (likely a typo or misuse of compact_shape_factors).
  Did not exist in round 10 audits (visual record before fix shows
  star was at least full-sized).
- **Edge label oversize** (F2 over-correction); did not exist before.
- **Universal node shrink** is novel as a regression direction; round 10
  was complaining about puffiness, not cramping.

## User-Flagged Issues -- Final Status

- **"Arrows are wonky"** -- STILL WONKY. Direction has changed (smaller
  on short edges instead of larger), but inconsistency vs dot persists.
  PARTIAL.
- **"Text isn't centered"** -- Centering looks acceptable on this round;
  no clear vertical/horizontal misalignment within nodes. PASS for
  centering specifically. (Edge label *positioning* relative to edges
  is acceptable; size is the new problem -- see F2.)
- **"Different font"** -- Both renders use a serif face that looks
  consistent with dot's Times-like default. Sub-pixel hinting differs
  (matplotlib FreeType vs Pango/Cairo) -- residual. PASS modulo the
  hinting floor.
- **"Cluster bounding boxes look like shit"** -- Still misaligned per
  H4/H5 deferred class. Outer Group / Right Branch / Left Branch boxes
  overlap each other and node positions. KNOWN_DEFERRED layout-side.

## Acceptable Residual

- Sub-pixel antialiasing and font hinting differences (matplotlib FreeType
  vs Cairo+Pango). Visible at extreme zoom only.
- B-spline routing channel profile vs Bezier shape on long curved edges
  (state_machine "restart" arc -- the curve profile differs but both
  reach the same endpoints).
- H4/H5 cluster layout issues -- per architecture, deferred.

## Final Recommendation: CONTINUE

This round did not converge. The bar is "indistinguishable save for
documented residuals." We are visibly distinguishable on every panel I
reviewed -- in fact, in some cases (star, edge labels) the round-11 fixes
introduced more glanceable departures than they removed. The user is an
expert and will see all of these immediately.

Concrete next-round fixes (3-5):

1. **Recalibrate node sizing.** Restore min_width/min_height closer to
   the original (revert from 41,27 toward ~50,33) but keep the
   theme-padding (6,3) trim. Goal: width within 5% of dot at every label
   length. Verify on tiny_graph, single_edge, pipeline, colors_showcase.
2. **Rebuild compact_shape_factors with star=1.0**, ellipse curved
   factor in 0.93-0.96 band only, diamond unchanged. Star MUST not
   shrink -- the round-11 setting is broken.
3. **Reduce edge label font scaling.** Target 10pt edge labels on a
   14pt graph (graphviz default ratio). Fix `_strict_absolute_edge_label_font_data`
   so it doesn't pick up the graph-level 14pt; or set the graph-level
   fallback to 10pt for edge labels specifically.
4. **Increase arrow base size** for short edges. The current
   disable_curve_length_clamp doesn't actually upsize the head; need an
   additive base-size term (think: arrow_size *= max(1.0, 1.2 / clamp_factor)
   or similar). Verify arrow_types matches dot lozenge-for-lozenge.
5. **Increase stroke weight** for node outlines and edge lines on
   strict theme. ~1.0 px -> ~1.4 px to match dot's perceived weight.
   This is sub-pixel-ish but reads as "lighter weight" on every panel
   and is worth one knob turn.

If we can get all five into a round 13, panels should genuinely converge.

## Confidence

High that the regressions listed are real and not artifacts of my reading.
Each was confirmed across multiple panels (star is the only single-panel
finding, but it is unmistakable). Pixel measurements are eyeballed,
not measured; expect 10-15% slop on the numbers but the directional
verdicts (smaller / larger / unchanged) are robust.

Lower confidence on stroke-weight: could be FreeType vs Cairo hinting,
not a real difference. Listed it because it reads visibly across the
gallery, but happy to demote if a render-stack survey confirms it's
hinting only.

Highest priority single fix: the star shape. It is broken in a way that
cannot be excused as "documented residual" -- the label literally
overflows the shape boundary.
