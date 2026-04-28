# Graphviz Theme Parity Audit -- Round 8 (Opus, post-round-7 verification)

## Methodology

- **Panels reviewed (16 total, 12 priority + 4 spot-checks):** long_labels,
  arrow_types, state_machine, diamond, balanced_binary_tree, nested_clusters,
  deep_nesting_4, cluster_showcase, multi_cycle, transformer_block,
  data_pipeline, pipeline (priority 1-12); plus colors_showcase,
  microservices, single_edge, tiny_graph, star, self_loop, medium_mixed,
  grid_4x4 (additional spot-checks for font/stroke regression isolation).
- **Column layout verified:** YES, 2-column LEFT="Graphviz dot" / RIGHT="Dagua
  (strict)" on every panel header inspected. The "Dagua (improved)" column has
  been correctly dropped. Headers checked on long_labels, arrow_types,
  state_machine, diamond, pipeline, single_edge, tiny_graph, colors_showcase,
  multi_cycle, deep_nesting_4, transformer_block.
- **Theme code consulted:** `dagua/styles.py:910-1000` (graphviz_strict block).
  Verified `font_size=12.0`, `edge_label_font_size=12.0`, `font_size_scaling="fixed"`,
  `font_family="TeX Gyre Termes"`, cluster `fill=#F2EFE9`, `fill_opacity=0.10`.
- **Total residual departures:** 18 (4 high, 8 medium, 6 low) PLUS 2
  KNOWN_DEFERRED (H4/H5).
- **Round-7 commit:** aa6f616 (per brief).
- **Inspection level:** at-zoom glyph cap-height comparison; arrow tip vs
  ellipse-boundary inspection; cluster fill perceptibility check; back-edge
  curve magnitude eyeballed; stroke weight ratio LEFT vs RIGHT.

---

## Round 7 Fix Verification (9 items F1-F9)

### F1. sqrt(2) ellipse circumscription (R1+R2) -- **PARTIAL**

- **R1 (long_labels regression):** **PASS.** long_labels.png RIGHT no
  longer shows the Round-6 catastrophic balloon. BatchNormalization2d, Conv2d,
  MultiHeadAttention are now visibly separated. The aspect-cap on extreme
  long-label ellipses prevents runaway height growth.
  - Caveat: dagua's MultiHeadAttention and TransformerEncoderLayer ellipses
    are still WIDER than dot's (extra horizontal padding), but no overlap.
- **R2 (short-label flat ellipses):** **PARTIAL/FAIL.** arrow_types.png RIGHT
  ellipses are STILL VISIBLY FLATTER than dot LEFT.
  - Measurement: dagua's "normal" ellipse approx 100x55 px (W/H ~1.8); dot's
    approx 80x60 px (W/H ~1.3-1.4). Dagua's are still measurably wider.
  - On diamond.png, "Start" RIGHT looks closer to dot's roundness, but
    "Left"/"Right" still read as oblong vs dot's nearly-circular.
  - Codex's chosen fix ("smaller uniform scale + aspect cap") is too
    conservative for short labels. The deviation from full sqrt(2) leaves a
    persistent flatness gap on small-label ellipses.
- **Verdict:** PARTIAL. Long-label regression is closed; short-label
  flatness is reduced but not eliminated.

### F2. crow arrowhead filled (R3) -- **FAIL**

- arrow_types.png "crow" column: dot LEFT shows a small SOLID FILLED triangle
  (a tight upward filled spear). dagua RIGHT shows a THIN OPEN V CHEVRON --
  identical to "vee" appearance, and clearly hollow-stroked. **NOT FILLED.**
- Codex's REPORT_round_7 claims the rebuild to "one compact filled
  Graphviz-style polygon" landed; SVG cross-check verified `fill="black"`
  in dot output. Yet the rendered crop still shows hollow strokes. Either:
  (a) the strict-theme override is not engaging on this fixture
  (b) the polygon is degenerate (zero-area rendered as outline)
  (c) the scale at which crow renders is so small that the fill is
      invisible relative to the stroke width.
- **Verdict:** FAIL. Visual output does not match reported fix.

### F3. open arrowhead filled (R4) -- **PASS (partial)**

- arrow_types.png "open" column: dot LEFT shows a SOLID FILLED triangle.
  dagua RIGHT shows a small filled triangle. **Fill direction is correct.**
- However, dagua's "open" arrow is VISIBLY SMALLER than dot's open arrow
  (about 60-70% the linear size). dot's open arrow looks identical to its
  "normal" arrow; dagua's "open" is smaller and slimmer than its own
  "normal" -- internally inconsistent with how Graphviz aliases open.
- **Verdict:** PASS on fill; PARTIAL on size proportion.

### F4. Edge label font_size 12.0pt (R5) -- **PARTIAL**

- state_machine.png: dagua RIGHT edge labels ("retry", "resume", "restart",
  "reset") read VISIBLY SMALLER than dot LEFT's. dot's labels ~14-16px
  cap-height; dagua's ~10-11px.
- arrow_types.png: dagua RIGHT under-arrow labels ("normal", "vee", "dot",
  etc.) are clearly smaller than dot LEFT's labels. The gap is ~25-30%.
- The styles.py value IS set to 12.0pt (verified line 996). The visual
  shortfall comes from either: (a) matplotlib DPI normalization undershoots
  Graphviz's effective pt-to-px ratio, OR (b) the RESOLVED_FONT mapping is
  still pulling a metrics-different font for measurement. The configuration
  fix landed but the perceptual size still doesn't match dot.
- **Verdict:** PARTIAL. Numerical config raised correctly; visual size
  still ~70-80% of dot.

### F5. Cluster fill #F2EFE9 fill_opacity 0.10 (R6) -- **PASS**

- transformer_block.png: BOTH LEFT and RIGHT show a perceptible warm-cream
  tint inside Multi-Head Attention and Feed-Forward Network clusters.
  Dagua's tint matches dot's character.
- nested_clusters.png: faint warm tint visible in dagua RIGHT clusters.
- deep_nesting_4.png: faint tint visible across nested levels.
- cluster_showcase.png: faint tint on outer clusters in both.
- The 0.08 -> 0.10 + cool-grey -> warm-cream shift correctly hits dot's
  visible-but-subtle perceptual register.
- **Verdict:** PASS.

### F6. Arrow tip-to-ellipse boundary trim (R8) -- **PASS (with one panel residual)**

- diamond.png: arrow tips into "End" now sit ON the ellipse boundary --
  no overlap, no visible gap. Improvement from Round 6 where there was
  a 3-4px bite.
- balanced_binary_tree.png: tips into LL/LR/RL/RR ellipses meet boundary
  cleanly.
- pipeline.png: tips into Preprocess/Transform/Postprocess sit ON
  boundary or with sub-pixel gap (acceptable).
- single_edge.png: tip into "Sink" meets boundary cleanly.
- multi_cycle.png: tips meet boundaries reasonably.
- Residual: on tiny_graph.png the arrow body looks tucked just inside the
  Mid ellipse outline by ~1px (NOT a bite, but base-of-arrow positioning
  could be one px earlier).
- **Verdict:** PASS. The trim is now in spec across the cosmetic suite.

### F7. Edge label collision avoidance (R7) -- **PASS**

- state_machine.png: "retry" and "resume" labels are now visibly
  separated. They no longer read as "retryresume" (Round 6's collision).
  Each label has its own bounding box and they're offset along the y-axis
  by ~12-15px.
- However, the labels still read as SMALLER than dot's labels (this is
  F4 territory, not F7).
- **Verdict:** PASS for collision; the underlying font-size gap is
  separately tracked under F4.

### F8. Back-edge curvature floor (H6 polish) -- **PARTIAL**

- state_machine.png: "restart" Done->Idle and "reset" Error->Idle are now
  visibly curved arcs to the right side of the graph. The 60pt floor is
  larger than the 36pt of Round 5 -- improvement is visible.
- multi_cycle.png: B->A back-edge in dagua RIGHT bows out to the right
  with visible curvature.
- BUT: dot LEFT's back-edges arc MORE WIDELY (further out from the node
  column) than dagua's. dagua's curves are still tighter against the
  body. The 60pt floor is closer to but does not yet match dot's
  preferred side-channel.
- **Verdict:** PARTIAL. Improved over Round 6; still smaller magnitude
  than dot.

### F9. Deepest-cluster label mask padding (H8 polish) -- **PARTIAL**

- deep_nesting_4.png: "Level 4 (Core)" label still appears to be PARTIALLY
  CROSSED in dagua RIGHT. There is a visible vertical line (the edge from
  Inner 2 down to Core) running through the label area, and the cluster
  border on the right side appears to clip near the label.
  - The mask padding 3pt -> 4pt may have helped marginally but isn't
    sufficient for the smallest level-4 cluster.
  - "Level 1", "Level 2", "Level 3" labels all read clean. Only the
    deepest stays half-crossed.
- **Verdict:** PARTIAL. Improvement of upper levels held; deepest level
  still has the regression.

### Round-7 Fix Verification Tally

| Item | Verdict |
|------|---------|
| F1 sqrt(2) ellipse | PARTIAL (R1 closed; R2 short-label flatness lingers) |
| F2 crow filled | FAIL (still hollow) |
| F3 open filled | PASS on fill; PARTIAL on size |
| F4 edge label 12.0pt | PARTIAL (config landed; visual size still small) |
| F5 cluster fill #F2EFE9 0.10 | PASS |
| F6 arrow tip-to-boundary | PASS |
| F7 edge label collision | PASS |
| F8 back-edge curvature 60pt | PARTIAL (closer; still tighter than dot) |
| F9 deepest cluster mask | PARTIAL (deepest still crossed) |

**Tally: 3 PASS / 5 PARTIAL / 1 FAIL / 0 KNOWN_DEFERRED out of 9.**

---

## Status of Round-4 / Round-6 Items Still Alive

| Round-4 ID | Round-6 verdict | Round-8 verdict |
|------------|-----------------|-----------------|
| H1 font family Termes | PASS | PASS (closed) |
| H2 font size 12.0pt | PASS | **REGRESSION** -- visual size now reads smaller |
| H3 ellipse aspect | PARTIAL | PARTIAL (R2 alive) |
| H4 nested clusters cuts node A | KNOWN_DEFERRED | KNOWN_DEFERRED |
| H5 sibling cluster overlap | KNOWN_DEFERRED | KNOWN_DEFERRED |
| H6 back-edge curvature | PARTIAL | PARTIAL (closer) |
| H7 open arrow forms | PARTIAL | F2 crow FAIL; F3 open PASS-fill |
| H8 cluster label mask | MOSTLY PASS | PARTIAL (deepest residual) |
| M1 arrow tip-boundary | FAIL | PASS (closed via F6) |
| M2 arrow proportions | PASS | **POSSIBLE REGRESSION** -- arrows now slimmer than dot |
| M3 cluster border | PASS | **POSSIBLE REGRESSION** -- now reads as DARKER than dot |
| M4 cluster fill_opacity | PARTIAL | PASS (closed via F5) |
| M5 edge label collision | FAIL | PASS (closed via F7) |
| M6 color saturation | PASS | PASS (closed) |
| M7 edge body stroke | PASS | **POSSIBLE REGRESSION** -- now too thin (hairline) |

**Notable regressions inspected:** H2 visual font, M2 arrow proportions,
M3 cluster border weight, M7 edge stroke. See "New Issues" below.

---

## Remaining Departures (priority ranked)

### R1. Node label visual font size systematically smaller than dot (HIGH)
- **Measurement:** single_edge.png "Source"/"Sink" letters in dagua RIGHT are
  ~20-22px cap-height; dot LEFT ~30px. Dagua at ~70% of dot.
- **Most damning panel:** diamond.png -- "Start", "Left", "Right", "End"
  in dagua RIGHT read as roughly half the cap-height of dot LEFT. Severe
  perceptual mismatch.
- **Confirmed across:** single_edge, tiny_graph, diamond, pipeline,
  colors_showcase, nested_clusters, transformer_block, deep_nesting_4,
  long_labels.
- **Mechanism:** styles.py has font_size=12.0 set correctly. The visual
  shortfall is due to (most likely) DPI normalization between
  Graphviz's PostScript rendering at native 96 DPI and matplotlib's
  default 100 DPI bitmap output. dot at fontsize=14 default measures
  ~18-19px; matplotlib at 12pt with 100 DPI measures ~16px nominal but
  the cap-height is closer to ~12-13px.
- **Fix direction:** raise font_size to ~14.0-14.5 in graphviz_strict to
  hit dot's perceived cap-height; OR explicitly correct for the DPI ratio
  (multiply by 100/72 vs Graphviz's 72/96 baseline).

### R2. Arrow proportions slimmer than dot (HIGH)
- single_edge.png: dagua RIGHT arrow tip is THINNER than dot LEFT's stocky
  filled triangle. Width-to-length ratio reads as ~0.6 in dagua vs ~0.85 in
  dot.
- arrow_types.png: every arrow head in dagua RIGHT is visibly slimmer/smaller
  than its counterpart in dot LEFT (normal, vee, diamond, dot, open).
- Round-6 F2 verdict was PASS (M2 arrow proportions). The Round-7 changes
  (radial reclipping) appear to have changed something that re-shrunk
  the arrow rendering.
- **Fix direction:** verify that radial reclipping in arrowheads.py
  preserves the 8x8 (or whatever was nominal) tip dimensions. Currently
  arrow body length appears to be trimmed back along with tip placement.

### R3. F2 crow STILL renders hollow (HIGH)
- arrow_types.png "crow" column: dagua's crow is an OPEN V CHEVRON, not
  a filled triangle.
- Codex reported a code fix and SVG verification of dot's output, but the
  rendered RIGHT panel does not show a filled crow.
- Hypothesis: the strict-theme arrow override path is not engaging when
  the test fixture passes `arrow_fill="hollow"` for crow. The Round-7
  override applied to "open" but not "crow", or the polygon-build path
  for crow has a stroke-only render mode that wins over the fill flag.

### R4. Edge stroke now too thin / hairline (HIGH)
- tiny_graph.png is the smoking gun: dot LEFT shows ~1.2-1.4px ellipse
  outlines and proportional arrow stems. dagua RIGHT shows hairline
  ~0.5px strokes everywhere. The whole render reads as "ghost" compared
  to dot's "drawn".
- single_edge.png same observation: dot's edge body is ~1.2px; dagua's
  edge body is ~0.5-0.7px.
- Round 6 reported M7 edge body stroke as PASS at 0.75pt. After Round 7's
  radial reclipping, strokes appear EVEN THINNER. May be a render-side
  device-pixel rounding issue or the stroke width is being divided by
  DPI scale somewhere.

### R5. Cluster borders now too DARK / heavy (MEDIUM-HIGH)
- microservices.png: dagua RIGHT cluster boxes (API Layer, Service Layer,
  Data Layer, Worker Layer) are visibly DARKER and have THICKER strokes
  than dot LEFT's near-invisible #CCCCCC ghost-thin strokes.
- data_pipeline.png same issue: Transform cluster border is darker in
  dagua RIGHT than dot LEFT.
- deep_nesting_4.png: nested cluster borders in dagua read as solid
  pencil-line, not the wispy ghost in dot.
- May be an antialiasing/stroke-width interaction at certain zoom levels;
  may also be that opacity isn't being applied to the stroke (only the
  fill).
- **Fix direction:** verify cluster stroke also respects opacity OR
  reduce stroke color from #CCCCCC to #DDDDDD to lighten further.

### R6. Vertical text centering still ~2-4px low in ellipses (MEDIUM)
- colors_showcase.png: "Blue", "Green", "Yellow", "Purple", "Orange"
  labels sit visibly below true vertical center of their ellipses.
- diamond.png: "Left", "Right", "End" labels appear slightly low.
- single_edge.png: "Sink" label appears slightly below center.
- This is matplotlib's text-baseline vs Graphviz's PostScript
  text-baseline mismatch. Documented in Round 6 as R11.
- **Fix direction:** apply a baseline-shift correction in mpl.py for
  ellipse-shape labels (about 0.15 * cap_height upward).

### R7. Ellipse aspect ratio still flatter than dot for short labels (MEDIUM)
- arrow_types.png small ellipses: dagua RIGHT W/H ~1.8; dot LEFT W/H ~1.4.
- diamond.png "Left", "Right" ellipses: dagua flatter than dot.
- colors_showcase.png "Red", "Blue", "Green" ellipses: dagua flatter.
- F1 closed the long-label regression but R2 is the persistent
  short-label gap.

### R8. Edge label still smaller than node label and smaller than dot (MEDIUM)
- state_machine.png "retry"/"resume" cap-height ~10-11px in dagua;
  dot LEFT ~14-15px.
- arrow_types.png arrow-name labels (between source and target ellipses)
  are 30% smaller in dagua.
- F4 raised the value to 12.0pt but the visual rendering still
  undershoots.

### R9. Outer cluster (transformer_block) edge crosses cluster stroke (MEDIUM)
- transformer_block.png RIGHT: edges into "Add" exit through Multi-Head
  Attention cluster's bottom-right stroke; edges leaving "Add" punch
  through Feed-Forward Network's right edge. dot LEFT routes around
  cleanly.
- Edge-routing-vs-cluster-bbox issue. Partly layout-side.

### R10. Sibling clusters overlap each other (MEDIUM)
- cluster_showcase.png: Outer Cluster, Nested Inner, Tiny Cluster all
  overlap.
- Same root cause as H4/H5 KNOWN_DEFERRED but on a different fixture.

### R11. self_loop rendering does not match Graphviz (MEDIUM)
- self_loop.png: dot LEFT shows a small clean self-loop on "Process" node
  with "retry" label inside the loop. dagua RIGHT shows the self-loop on
  a different node ("Validate") and the loop shape is different (more of
  an external bow than dot's tight half-circle), with "retry" label
  outside the loop.
- Could be partly layout (graph fixtures use different self-loop
  declarations) but the loop shape itself differs.

### R12. Crow vs. vee indistinguishable in dagua (LOW-MEDIUM)
- Because R3 leaves crow hollow, and vee is also a hollow chevron, the
  two arrow types now look identical in dagua RIGHT but not in dot LEFT.
- Defect on the panel that exists specifically to show arrow types --
  inability to distinguish them is bad signal.

### R13. Arrow size on "open" smaller than "normal" in dagua (LOW)
- arrow_types.png: dot's "open" arrow is the same size as "normal".
  dagua's "open" is smaller than its own "normal".
- The override that made "open" filled apparently shrunk it too.

### R14. Multi-line label vertical line spacing slightly larger in dagua (LOW)
- transformer_block.png "Q Projection"/"K Projection"/"V Projection"
  multi-line labels appear to have larger inter-line gap in dagua than
  dot.
- long_labels.png "Conv2d 3x3, stride=1" two-line label has larger gap
  in dagua.

### R15. Font glyph rendering slightly heavier (faux-bold) in dagua (LOW)
- pipeline.png "Preprocess", colors_showcase.png "Yellow" -- dagua's
  glyphs read as marginally heavier-stroked than dot's. FreeType vs
  Cairo+Pango stack residual; documented but worth keeping on the list.

### R16. Subscript/special characters render differently (LOW)
- long_labels.png "Softmax (sigma) sum exp(x_i)" -- dot LEFT shows the
  subscript "i"; dagua RIGHT shows "x" with apparent space (no visible
  subscript). The "x sub i" Unicode escape may not have a glyph in the
  rendered Termes font on the dagua side.

### R17. Title weight differs between panels (LOW)
- All panels: panel header "long_labels", "arrow_types", etc. is BOLD on
  dagua RIGHT (the bolded title at top); the LEFT side is unlabeled at
  the same level. Cosmetic of the comparison harness, not the graph
  itself, but inconsistent presentation.

### R18. Star spoke nodes overlap each other (LOW, layout-side)
- star.png: dagua RIGHT spokes touch their neighbors; dot LEFT has clear
  gaps. Layout-side spacing.

---

## New Issues Introduced by Round 7

### N1. Edge stroke thickness regressed to hairline
- See R4. Round 6 had M7 PASS at 0.75pt; Round 7 reads as ~0.5pt or
  thinner. Possible regression from the radial reclipping change in
  arrowheads/edges.

### N2. Arrow proportions regressed (slimmer than Round 6)
- See R2. The Round 6 8x8 stocky arrow shape that earned M2 PASS appears
  to have been reduced. Both arrowhead width and overall arrow span
  read smaller in Round 7 than Round 6.

### N3. Cluster border darkness amplified
- See R5. Cluster borders now read DARKER than dot's, possibly because
  the stroke ignored opacity scaling, or because the radial-reclip
  change exposed a stroke-width path that compounds across nested
  clusters.

### N4. crow named arrow now renders hollow despite reported fix
- See R3. Codex reports the fix landed; SVG cross-check verified. But
  the rendered output disagrees. Either the strict-theme override
  doesn't engage on the test fixture, or there's a stroke-only fallback
  path winning.

---

## User-Flagged Issues -- Final Status

- **"Arrows are wonky"** -- **NOT CLOSED.** Round-7 closed F6 (tip-to-boundary)
  and F8 (back-edge curvature, partially), AND F7 (edge label collision).
  But Round 7 INTRODUCED N2 (arrows now slimmer than Round 6), R3 (crow
  still hollow), and R4 (edge strokes too thin/hairline). The "wonky"
  label is still a fair description -- the visible arrow appearance has
  not yet matched dot. Closer than Round 6 on tip-trim and back-edges;
  worse on stroke and proportions.

- **"Text isn't centered"** -- **NOT CLOSED.** Vertical centering
  mismatch (R6) is still visible across colors_showcase, diamond,
  single_edge. The 2-4px vertical drop is consistent. Combined with the
  font-size-too-small visual (R1), labels still don't sit where dot's
  do. Round 6 said "PARTIAL"; Round 8 says "still PARTIAL".

- **"Different font"** -- **MOSTLY CLOSED on family; NOT CLOSED on size.**
  The Termes glyph shapes match (H1 PASS). But the rendered SIZE of
  those glyphs is too small (R1 / N1 of round 6). The user's complaint
  was about the perceptual gestalt of "different font" which includes
  weight and rhythm; the size mismatch perpetuates that gestalt despite
  the family being correct. Half-closed at best.

- **"Cluster bounding boxes look like shit"** -- **PARTIAL.** Cluster
  fill (F5 PASS) and cluster label masking (mostly PASS, F9 only
  partial on deepest level) are real wins. BUT cluster strokes now read
  as too DARK rather than the round-6 light ghost (R5/N3). H4/H5 still
  KNOWN_DEFERRED and remain the most VISIBLE structural cluster
  failures. So: better than Round 6 on fill, regressed on stroke,
  unfixed on overlap. Still readable as "boxes look like shit" on
  panels with sibling cluster overlap.

---

## Acceptable Residual

- Sub-pixel antialiasing of stroked paths (matplotlib FreeType vs Cairo)
- B-spline routing profile vs cubic Bezier (long curved edges have
  different inflection structure)
- Type1 font hinting differences (FreeType vs Cairo+Pango) -- 0.5-1px
  glyph-width drift on long labels
- Layout topology differences on cluster-rich graphs (out of scope)
- Subscript/Unicode special-glyph fallback (R16) -- font-coverage residual

---

## Final Recommendation: **CONTINUE**

Round 7 closed THREE meaningful items (F5 cluster fill, F6 arrow trim,
F7 edge label collision) but introduced FOUR regressions (R3 crow still
hollow despite reported fix; R4 edge stroke now hairline; R2 arrow
proportions slimmer than Round 6; R5 cluster borders darker than Round
6). On NET, panel-by-panel inspection at zoom does NOT yet meet the
"indistinguishable save for documented stack residuals" bar:

1. **R1 (font size visual gap):** ~30% smaller cap-height than dot
   across single_edge, tiny_graph, diamond, pipeline. Pervasive defect
   on EVERY text-bearing panel. Highest-priority fix.
2. **R3 (crow hollow):** the codex-reported fix did not produce visual
   change. Worth investigating the reload/test path before adding more
   code.
3. **R4 (hairline edge strokes):** appears to be a regression vs Round 6.
4. **R2 (arrow slimmer):** appears to be a regression vs Round 6.
5. **R5 (cluster strokes too dark):** appears to be a regression vs
   Round 6.

If recommending CONTINUE, next-round fixes:

- **Fix R1:** raise graphviz_strict font_size from 12.0 to ~14.0; OR
  add explicit DPI correction in `dagua/styles.py` so that 12pt at
  matplotlib's 100 DPI matches dot's perceived 14pt at 96 DPI. Verify
  via cap-height pixel measurement on single_edge.png.
- **Fix R2/R4:** investigate whether the F6 radial-reclip change
  introduced a stroke-width or arrow-length scaling regression. Compare
  Round 6 arrowhead.py and edge-render code paths to Round 7. Run a
  before/after on tiny_graph.png and single_edge.png.
- **Fix R3:** confirm strict-theme override engages on `arrow_fill="hollow"`
  fixtures for crow specifically. Add a regression test that renders
  the arrow_types fixture and asserts crow has fill > 0% pixels-black.
- **Fix R5:** check that cluster stroke respects fill opacity or further
  lighten stroke color (#CCCCCC -> #DDDDDD).
- **Fix R7 (short-label flat ellipses):** Codex's "smaller uniform scale
  + aspect cap" was conservative. Try a wider cap (e.g. axis_scale=1.20
  for short labels with W/H < 1.5) to land closer to dot's roundness on
  arrow_types and colors_showcase.
- **Defer R8-R18** as low-priority/stack-residual.
- **Re-verify after R1+R2+R3+R4+R5 fixes** against single_edge.png and
  tiny_graph.png as primary smoke test panels.

The bar of "indistinguishable save for documented stack residuals" is
NOT yet met. Round 8 fixes are ~5-7 hours total; the work is bounded.

---

## Confidence

- **HIGH confidence** on R1 (font size visual gap): measurable across
  many panels; smoking gun is single_edge.png and tiny_graph.png where
  there is nothing else to confound. Cap-height ratio gap of ~70% is
  not within stack residual.
- **HIGH confidence** on R2 (arrow proportion regression): visible on
  every arrow on every panel. Round 6 didn't show this; Round 7 does.
- **HIGH confidence** on R3 (crow still hollow): visible directly on
  arrow_types.png; the crow column shows an open V chevron.
- **HIGH confidence** on R4 (edge stroke too thin): tiny_graph.png is
  unambiguous.
- **HIGH confidence** on F5 (cluster fill PASS): perceptible warm-cream
  visible on transformer_block, nested_clusters, deep_nesting_4.
- **HIGH confidence** on F6 (tip-to-boundary PASS): inspected on
  diamond, balanced_binary_tree, pipeline, single_edge.
- **HIGH confidence** on F7 (edge label collision PASS): "retry" and
  "resume" in state_machine.png are clearly separated.
- **MEDIUM confidence** on R5 (cluster borders too dark): subjective
  perception; may be antialiasing rather than actual stroke change.
  Recommend pixel-color sampling.
- **MEDIUM confidence** on R6 (vertical centering low): 2-4px is
  borderline-stack-residual.
- **MEDIUM confidence** on F8 (back-edge curvature PARTIAL): visible
  improvement; question is what "enough" means.
- **MEDIUM confidence** on F9 (deepest cluster mask PARTIAL): the
  visible crossing on Level 4 is small and could plausibly be the
  edge-line crossing, not a label-mask failure.
- **LOW confidence** on R11 (self_loop): graph fixture differences
  may be confounding; need to verify same fixture is used.
- **LOW confidence** on R16 (subscript): could be a font-cache issue
  rather than a glyph-coverage one.
