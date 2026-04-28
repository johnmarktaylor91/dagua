# Graphviz Theme Parity Audit -- Round 6 (Opus, post-round-5 verification)

## Methodology

- **Panels reviewed (12 priority):** pipeline, diamond, balanced_binary_tree, state_machine,
  arrow_types, multi_cycle, nested_clusters, deep_nesting_4, cluster_showcase,
  transformer_block, data_pipeline, colors_showcase, microservices, long_labels.
- **Column layout verified:** YES, 2-column LEFT="Graphviz dot" / RIGHT="Dagua (strict)"
  on every panel header inspected. The "Dagua (improved)" column has been dropped
  from the round-5 crops as the brief stated.
- **Total residual departures:** 14 (3 high, 8 medium, 3 low) PLUS 2 known-deferred
  layout-side issues PLUS 1 NEW REGRESSION introduced by the H3 sqrt(2) fix.
- **Theme code consulted:** `dagua/styles.py` (graphviz_strict block);
  `dagua/render/mpl.py` (sqrt(2) ellipse circumscription gate);
  `dagua/render/edges/arrowheads.py` (per-arrow fill rules).
- **Round-5 commit:** 882b970 (per brief).

---

## Round 4 Fix Verification (15 items)

### H1. Font family Times New Roman -> TeX Gyre Termes -- **PASS**

- **Evidence:** pipeline.png, diamond.png, balanced_binary_tree.png, colors_showcase.png
  all show distinctly Termes-style glyphs in MIDDLE matching LEFT. Capital T terminals
  flared on both. Lowercase "g" loop matches. "P" bowl tightness matches.
  `fc-match` in REPORT_round_5 confirms `qtmr.pfb` mapping is wired through both
  rendering AND text measurement (a critical detail -- without `dagua/utils.py`
  patch the metrics would have used the matplotlib fallback). Closed.

### H2. Font size 10.5 -> 12.0pt (cap-height ratio ~0.40) -- **PASS**

- **Evidence:** pipeline.png MIDDLE: "Preprocess" letters now span ~25-27px tall in
  ~70-75px ellipses -> ratio ~0.36 (close to dot's ~0.40). colors_showcase.png MIDDLE:
  "Yellow" reads at the same approximate size as LEFT. Round 4's "labels float in
  ellipses" complaint is GONE on most panels. Slight residual: dagua node labels
  still read marginally smaller than dot's (ratio is ~0.36 vs dot's ~0.40-0.42)
  but the gap is subtle and within stack-residual territory. Closed.

### H3. Ellipse aspect ratio sqrt(2) circumscription -- **PARTIAL with NEW REGRESSION**

- **Evidence GOOD:** pipeline.png MIDDLE: "Preprocess", "Transform", "Postprocess"
  ellipses are now noticeably ROUNDER than round-4's flat squashed shapes -- aspect
  ~2.0 instead of ~3.2. diamond.png MIDDLE: "Start" ellipse is rounder.
  balanced_binary_tree.png MIDDLE: ellipses match dot well.
- **Evidence BAD:** long_labels.png MIDDLE has been catastrophically broken. The
  "BatchNormalization2d(128, eps=1e-05, momentum=0.1)" label ellipse is a HUGE
  balloon that ENGULFS Conv2d above it and MultiHeadAttention below it -- the three
  nodes overlap into one tangled mass. The conditional in `dagua/render/mpl.py`
  (`if width/height > 2.0: height *= sqrt(2)`) explodes for very long single-line
  labels because the label's natural width-to-height ratio is far above 2 and
  multiplying the height by sqrt(2) overshoots when the original width was large.
- **Evidence BAD #2:** arrow_types.png MIDDLE: ellipses are still FLATTER than
  dot's because the short labels ("normal", "vee", "dot", etc.) have width/height
  ratio BELOW 2.0, so the gate at 2.0 means short-label ellipses don't get the
  sqrt(2) treatment AT ALL -- they remain stretched-flat. So there's a discontinuity:
  ellipses with very-short labels are flat; medium ones are good; very long
  ones explode.
- **Verdict:** PARTIAL on the medium-label case, FAIL on extremes. The sqrt(2)
  gate logic (`width/height > 2.0`) is the wrong shape -- it should be a smooth
  function (e.g. `axes_scale = sqrt(2)` always, applied to BOTH semi-axes) or
  width-capped to prevent runaway on long labels.

### H4. nested_clusters Outer Group cuts through node A -- **KNOWN_DEFERRED**

- **Evidence:** nested_clusters.png MIDDLE still shows "Outer Group" cluster
  rectangle's top edge passing through node A. Codex explicitly deferred this
  as a layout-side issue. Confirmed deferred.

### H5. Sibling clusters Right/Left Branch overlap -- **KNOWN_DEFERRED**

- **Evidence:** nested_clusters.png MIDDLE still shows sibling cluster boxes
  overlapping; the "Left Branch" label even has the sibling cluster's right
  stroke cutting through it (the label is partially obscured -- the "h" at end
  of "Branch" is bisected by the next cluster's left stroke). Codex deferred
  this. Confirmed deferred. Note: while the layout cause is upstream, the
  symptom is now WORSE because the lighter (#CCCCCC) stroke makes the overlap
  look more like an unintentional ghost line than a deliberate boundary.

### H6. Back-edge curvature absolute floor -- **PARTIAL**

- **Evidence GOOD:** state_machine.png MIDDLE: "restart" Done->Idle now shows
  visible curvature (right-side arc) instead of round-4's straight vertical.
  multi_cycle.png MIDDLE: A->G back-edge is bowed out to the right. Improvement
  is real and visible.
- **Evidence WEAK:** dagua's curve magnitude is still SMALLER than dot's. On
  state_machine LEFT, "restart" arcs out about 80px from the column; on RIGHT
  it arcs only ~30-40px. dot's "reset" Error->Idle in LEFT is a clean S-curve
  hugging the right margin; dagua's MIDDLE has the back-edges crammed against
  the body of the graph rather than a true side-channel. The 36pt absolute
  floor is a step in the right direction but should probably be larger
  (50-60pt) for graphs of this height, OR the curvature should compute
  side-channel-clearance rather than fixed offset.
- **Verdict:** PARTIAL. Fix worked but didn't go far enough.

### H7. Open arrow forms (vee, open, circle) -- **PARTIAL with one error**

- **Evidence GOOD:** arrow_types.png MIDDLE:
  - vee: open V chevron in BOTH dot and dagua. MATCH.
  - circle: open hollow circle in BOTH (dagua's "circle" -> "odot" mapping).
    MATCH. (Codex's note that native dot reports "circle" unknown is a
    documented Graphviz quirk; the alias-to-odot is sound.)
  - dot: solid black filled circle in BOTH. MATCH.
  - normal, diamond, tee, none: all match.
- **Evidence BAD:**
  - **crow** in dot LEFT renders as a SMALL FILLED triangle/notch (a tight
    upward spear shape). In dagua RIGHT it renders as an OPEN V chevron --
    NOT filled. Codex's report claims "Kept `crow` filled" but the rendered
    output shows hollow strokes for crow. Either (a) the code path is wrong
    or (b) the per-arrow fill flag is being lost somewhere downstream.
  - **open**: dot LEFT renders "open" as a normal-style FILLED triangle
    (nearly identical to "normal"). dagua RIGHT renders "open" as an OPEN
    V chevron (nearly identical to dagua's "vee"). DEPARTURE: dot's
    "open" semantics on this Graphviz version is "fill = same as normal",
    NOT hollow. Dagua interpreted "open" as the open-prefix form when in
    fact the named arrow "open" on this dot version is filled.
- **Verdict:** PARTIAL. 3 of 5 named-open forms are correct; crow is
  inverted (filled in dot, hollow in dagua); "open" is inverted in the
  same direction.

### H8. Cluster border crosses through label text -- **MOSTLY PASS**

- **Evidence GOOD:** deep_nesting_4.png MIDDLE: "Level 1", "Level 2", "Level 3"
  labels are now CLEAR -- no stroke crossing. nested_clusters.png MIDDLE:
  "Outer Group" label is clear; "Right Branch" label is clear.
- **Evidence BAD:** deep_nesting_4.png MIDDLE: "Level 4 (Core)" -- the
  innermost tiny cluster -- still shows what appears to be partial stroke
  through the label (top edge of the inner-inner box clips through the
  bottom of "(Core)"). This may be because the label-mask rectangle is
  too small for the small inner cluster.
- **Verdict:** MOSTLY PASS. Closed for the main cases; one small residual
  on the deepest nesting level.

### M1. Arrow tip-to-boundary spacing -- **FAIL (worse on some panels)**

- **Evidence:** diamond.png MIDDLE: arrow tips into "End" ellipse OVERLAP the
  boundary -- the arrow body and triangle base bite ~3-4px into the ellipse
  outline. balanced_binary_tree.png MIDDLE: similar overlap into LL/LR/RL/RR
  ellipses. pipeline.png MIDDLE: small visible gap (~1-2px) between tip and
  ellipse on some edges (Preprocess, Transform). The change to arrow proportions
  (M2 fix from 10x7 to 8x8) shifted the tip placement, exposing this trim issue
  more visibly. NOT addressed in round 5; explicitly listed as "F11 not
  separately implemented" in REPORT_round_5.
- **Verdict:** FAIL.

### M2. Arrow proportions toward equilateral -- **PASS**

- **Evidence:** pipeline.png MIDDLE, diamond.png MIDDLE: arrowhead triangles are
  now stockier and broader, matching dot's "stout" appearance. The 8x8 ratio
  reads well. Closed.

### M3. Cluster border #AAAAAA -> #CCCCCC, stroke 0.8 -> 0.5 -- **PASS**

- **Evidence:** nested_clusters.png MIDDLE, deep_nesting_4.png MIDDLE,
  cluster_showcase.png MIDDLE: cluster strokes are visibly lighter/thinner than
  round 4. Reads as ghost-thin like dot. Closed.

### M4. Cluster fill_opacity 0.15 -> 0.08 -- **PARTIAL (overshoot)**

- **Evidence:** transformer_block.png MIDDLE, data_pipeline.png MIDDLE,
  microservices.png MIDDLE, deep_nesting_4.png MIDDLE: cluster fill is now
  essentially invisible (white-ish) -- it's gone PAST dot's faint visible
  warm-cream tint to "no tint at all." dot's tint is a real perceptible tone
  on transformer_block; dagua's at 0.08 reads as no-tint white. The change went
  too far in the right direction.
- **Verdict:** PARTIAL. Close but pushed past dot's visible tint. Try 0.10-0.12
  to land on dot's actual tone, or convert to a literal "lightgrey" with
  alpha 0.10.

### M5. Edge label collision on state_machine -- **FAIL**

- **Evidence:** state_machine.png MIDDLE: "retry" and "resume" labels still sit
  at almost the same y-coordinate, almost touching ("retryresume" reads as one
  word). REPORT_round_5 confirms "F12 was not separately implemented." Not
  fixed.
- **Verdict:** FAIL.

### M6. Color saturation drift on colors_showcase -- **PASS**

- **Evidence:** colors_showcase.png MIDDLE: Red, Blue, Green, Yellow, Purple,
  Orange all read at saturation/hue values nearly identical to LEFT. Yellow is
  marginally MORE saturated in dagua than dot, but within the reasonable
  X11-color-table-resolution range. Closed.

### M7. Edge body stroke 1.0 -> 0.75 -- **PASS**

- **Evidence:** pipeline.png MIDDLE, diamond.png MIDDLE: edge strokes read as
  hairlines comparable to dot's. The visual weight is very close. Closed.

### Round-4 Item Summary

| Item | Verdict |
|------|---------|
| H1 font family | **PASS** |
| H2 font size | **PASS** |
| H3 ellipse aspect | **PARTIAL with NEW REGRESSION** |
| H4 cluster cuts node A | **KNOWN_DEFERRED** |
| H5 sibling cluster overlap | **KNOWN_DEFERRED** |
| H6 back-edge curvature | **PARTIAL** |
| H7 open arrow forms | **PARTIAL** (crow + open inverted) |
| H8 cluster label masking | **MOSTLY PASS** (deepest level still clipped) |
| M1 arrow tip-to-boundary | **FAIL** (not attempted) |
| M2 arrow proportions | **PASS** |
| M3 cluster border color/stroke | **PASS** |
| M4 cluster fill_opacity | **PARTIAL** (overshoot to invisible) |
| M5 edge label collision | **FAIL** (not attempted) |
| M6 color saturation | **PASS** |
| M7 edge body stroke | **PASS** |

**Tally: 6 PASS / 5 PARTIAL / 2 FAIL / 2 KNOWN_DEFERRED out of 15.**

---

## Remaining Departures (priority ranked, with measurements)

### R1. NEW REGRESSION: long_labels.png ellipses explode and overlap (HIGH)
- **Mechanism:** the H3 fix `if width/height > 2.0: height *= sqrt(2)` overshoots
  on long single-line labels. "BatchNormalization2d(128, eps=1e-05, momentum=0.1)"
  has width/height >> 2.0; multiplying height by sqrt(2) yields a huge balloon
  that engulfs Conv2d above and MultiHeadAttention below.
- **Visible on:** long_labels.png MIDDLE -- THREE consecutive ellipses overlap.
- **Fix direction:** apply sqrt(2) to BOTH axes uniformly (which is what
  Graphviz actually does), OR cap the height growth. Current asymmetric scaling
  is geometrically wrong.

### R2. arrow_types short-label ellipses still flat (HIGH)
- **Mechanism:** sqrt(2) gate at width/height > 2.0 excludes short-label ellipses
  that started below the gate. arrow_types.png labels ("normal", "vee", "dot")
  produce ellipses with W/H ratio ~1.5-1.8, below 2.0, so they don't get the
  fix.
- **Measurement:** dagua's "normal" ellipse ~95x55px (ratio 1.7). dot's
  "normal" ellipse ~80x55px (ratio 1.45). dagua is wider than dot for the same
  short label.
- **Visible on:** arrow_types.png MIDDLE all 9 small ellipses.
- **Fix direction:** apply sqrt(2) circumscription uniformly (multiply both
  semi-axes), not as a width/height-ratio-conditional fix.

### R3. crow arrowhead inverted: hollow in dagua, filled in dot (MEDIUM-HIGH)
- **Mechanism:** stroke-only refactor in `dagua/render/edges/arrowheads.py`
  apparently caught crow incorrectly. REPORT_round_5 says "Kept `crow` filled"
  but rendered output disagrees.
- **Measurement:** dagua crow at ~25x15px hollow-stroke V; dot crow ~10x5px
  filled triangle.
- **Visible on:** arrow_types.png "crow" column.

### R4. open arrowhead inverted: hollow in dagua, filled in dot (MEDIUM-HIGH)
- **Mechanism:** dagua mapped named "open" to the open-prefix family (hollow);
  but on this Graphviz version (8.0.3) "open" renders as filled (looks like
  "normal").
- **Visible on:** arrow_types.png "open" column.
- **Fix direction:** match Graphviz's actual behavior on the target version.
  If Graphviz says "open == normal" in PostScript on this build, dagua's
  "open" should also be filled. Verify by parsing Graphviz source for arrow
  alias resolution.

### R5. Edge label font size too small (MEDIUM)
- **Measurement:** state_machine.png LEFT "restart" label spans ~16px tall;
  RIGHT spans ~12px tall. arrow_types.png LEFT "normal" arrow label ~11px
  tall; RIGHT "normal" arrow label ~9px tall. Edge labels in dagua are
  systematically ~25-30% smaller than dot's. The H2 node-label font fix to
  12pt may not have been applied to edge labels.
- **Visible on:** state_machine, arrow_types, data_pipeline, microservices.
- **Fix direction:** check that `edge_label_font_size` was raised in the
  same proportion as `font_size`; likely missed in round 5.

### R6. Cluster fill opacity overshot to invisible (MEDIUM)
- **Measurement:** dot's cluster fills on transformer_block produce a clearly
  perceptible warm-cream tint. dagua's at 0.08 reads as no-tint white.
- **Fix direction:** raise `fill_opacity` to 0.10-0.12, AND consider switching
  the fill color from `#F0F0F0` (cool gray) to `lightgrey`/`#D3D3D3`
  resolved to dot's literal default OR even slightly warm
  (e.g. `#F2EFE9`).

### R7. Edge label collision on state_machine (MEDIUM)
- "retry" and "resume" still nearly touching. Not attempted in round 5
  (REPORT_round_5 acknowledges).
- **Fix direction:** along-edge label offset calculation; or push labels off
  the edge curve onto the convex side.

### R8. Arrow tip overlap into node ellipse (MEDIUM)
- diamond.png MIDDLE: arrow tips INTO "End" overlap by ~3-4px past the
  ellipse boundary. balanced_binary_tree.png MIDDLE: similar bite into
  LL/LR/RL/RR. The rest of the strict theme work has converged enough that
  this is now at the top of the residual list. Not attempted in round 5.
- **Fix direction:** ellipse-boundary intersection trim, place tip on
  boundary, body ends at `tip - arrow_length * direction`.

### R9. Outer cluster (transformer_block) edge crosses cluster stroke (MEDIUM)
- transformer_block.png MIDDLE: edges into "Add" exit through the
  Multi-Head Attention cluster's bottom-right stroke; edges leaving "Add" do
  the same with Feed-Forward Network's right edge. dot routes edges around
  the cluster perimeter cleanly. dagua's edges punch through.
- This is partly layout-side but partly an edge-routing-vs-cluster-bbox
  cosmetic concern.

### R10. cluster_showcase: clusters overlap each other (MEDIUM)
- "Outer Cluster" overlaps "Nested Inner"; "Tiny Cluster" overlaps "Medium
  Cluster". Could be partly layout, but the overlap reads as cosmetic
  failure especially at the lighter stroke. Marked separately from
  H4/H5 because it's a different graph topology.

### R11. Vertical text centering in node ellipses still slightly low (LOW)
- colors_showcase.png MIDDLE: "Blue", "Green", "Yellow", "Purple" sit
  visibly below true vertical center -- about 2-3px below ellipse vertical
  midline. dot's labels appear to sit slightly above center. The exact
  centering reference is matplotlib's text-baseline vs Graphviz's PostScript
  text-baseline; these have a known offset that should be corrected via
  baseline shift in the renderer. Round 4 flagged this as L4; round 5 did
  not address.

### R12. Multi-line node label sub-line vertical spacing (LOW)
- transformer_block.png MIDDLE: "Q Projection" / "K Projection" / "V Projection"
  multi-line labels appear to have larger inter-line spacing in dagua than
  dot. Long ellipses now wider AND taller; the line spacing factor reads
  ~1.3 in dagua vs ~1.15 in dot. Subtle but visible at zoom.

### R13. Node label appears slightly heavier (faux-bold) on first observation (LOW)
- pipeline.png MIDDLE: glyph stroke contrast reads slightly heavier than
  LEFT. Possible cause: matplotlib renders Type1 fonts via FreeType with
  its own hinting, which produces marginally bolder strokes than Cairo+Pango.
  This may be in the "stack residual" category but worth noting because
  it manifests on every panel.

### R14. Title weight (panel header) still bold in dagua (LOW)
- All panels: panel title (e.g. "Cluster Showcase") is bold in dagua's
  RIGHT panel header but regular in dot's LEFT panel header. Round 4 L1
  flagged this; round 5 did not address. Minor since titles are
  panel-decoration not graph-content.

---

## New Issues Introduced by Round 5

### N1. long_labels ellipse explosion (HIGH severity)
- See R1 above. The conditional sqrt(2) fix is geometrically wrong.

### N2. Short-label ellipses now visibly inconsistent with mid-label ones
- See R2 above. Within the same panel (arrow_types), the short-label
  ellipses are flat while a longer-label ellipse would have been rounded.
  The discontinuity at width/height = 2.0 is visible to the eye.

### N3. Cluster fill went from too-visible to too-invisible
- See R6 above. The 0.15 -> 0.08 change overshot.

---

## User-Flagged Issues -- Status

- **"Arrows are wonky"** -- LARGELY ADDRESSED but not fully closed. H6
  back-edge curvature is now visible (improvement), H7 open forms are
  mostly correct (vee/circle/dot/normal/diamond/tee/none), M2 arrow
  proportions match dot (PASS), M7 edge stroke matches (PASS). REMAINING
  wonkiness: crow is hollow when it should be filled; "open" is hollow
  when on this version it should be filled (same as normal); back-edge
  curve magnitude is smaller than dot's (still cramped against graph
  body); arrow tips bite into node ellipses (R8). 60-70% of the
  "wonkiness" complaint is closed; 30-40% remains.

- **"Text isn't centered"** -- LARGELY ADDRESSED for size (H2 fixed),
  PARTIALLY remaining for vertical alignment within ellipse (R11). Edge
  labels are systematically too small (R5), which may be reading as
  "off" compared to dot. The big perceptual-fix was H1+H2; remaining
  is the 2-3px vertical baseline offset.

- **"Different font"** -- FULLY ADDRESSED. H1 PASS. TeX Gyre Termes is
  rendering on every panel and the glyph shapes match dot's. The only
  residual is rasterizer-level (FreeType vs Cairo) which is stack
  residual.

- **"Cluster bounding boxes look like shit"** -- PARTIALLY ADDRESSED.
  Cluster strokes are now lighter/thinner (M3 PASS); cluster labels are
  no longer crossed by stroke on the main cases (H8 PASS); however
  H4/H5 are KNOWN_DEFERRED and these are the most VISIBLE quality
  failures. Cluster fill is now too-invisible (R6/N3). Cluster overlap
  on cluster_showcase, microservices, transformer_block is a separate
  layout-side issue (R9/R10). Net: clusters look BETTER than round 4
  but the round-4 user complaint is still partially accurate -- the
  STRUCTURAL bounding-box defects on nested_clusters and friends
  remain.

---

## Acceptable Residual

- Sub-pixel antialiasing of stroked paths (matplotlib FreeType vs Cairo)
- B-spline routing profile vs cubic Bezier (long curved edges have
  different inflection structure)
- Type1 font hinting differences (FreeType vs Cairo+Pango) -- 0.5-1px
  glyph-width drift on long labels
- Layout topology differences on cluster-rich graphs (out of scope)

---

## Final Recommendation: **CONTINUE**

Round 5 made significant progress (6 PASS, 2 KNOWN_DEFERRED, 5 PARTIAL,
2 FAIL out of 15) AND introduced 1 critical new regression (R1, the
long_labels explosion). The bar of "indistinguishable save for documented
rendering-stack residuals" is NOT yet met because:

1. **R1 (long_labels regression)** is a USER-VISIBLE FAILURE -- three
   ellipses overlap/engulf each other. Strictly worse than round 4 on
   this panel.
2. **R2 (short-label ellipses still flat)** demonstrates the sqrt(2) gate
   is geometrically wrong; should be applied uniformly to both axes.
3. **R3, R4 (crow + open arrow inversion)** are correctness failures on
   arrow_types.png. Visible on a panel that exists specifically to test
   arrow types.
4. **R5 (edge label font size systematically smaller)** is a wide-blast
   gap visible on every edge-label-bearing panel.
5. **H4, H5** are KNOWN_DEFERRED but visible cosmetic defects per the
   user's original complaint. While layout-side, they cannot be ignored
   from a "looks like dot" perspective.

The work to close to "documented stack-residual only" is ~5-7 fixes of
~1-1.5 hours total effort:
- Fix R1+R2 by changing sqrt(2) gate to uniform multiplier on both axes
- Fix R3+R4 by parsing actual Graphviz arrow-alias resolution per version
- Fix R5 by raising edge label font size in graphviz_strict
- Fix R6 by raising cluster fill_opacity to ~0.10
- Fix R8 by adding ellipse boundary intersection trim for arrow tips
- (Defer R7, R11, R12, R13, R14, R9, R10 as low-priority/stack-residual)

After these, a STOP would be defensible. Currently we are NOT THERE.

---

## Confidence

- **High confidence** on R1 (long_labels regression): catastrophic visual
  failure on the panel; mechanism understood (asymmetric sqrt(2) on
  height-only).
- **High confidence** on R2 (short-label flat ellipses): same mechanism
  inversed; the gate is unintentionally exclusive.
- **High confidence** on R3 (crow inverted): visible on arrow_types.png
  at zoom. dot's crow is filled; dagua's is hollow.
- **High confidence** on R5 (edge label too small): measured on
  multiple panels.
- **High confidence** on M3, M4, M6, M7: visual + mechanism agreement.
- **High confidence** on H1, H2, M2: cleanly closed.
- **Medium confidence** on R4 (open inverted): depends on what dot
  actually outputs for "open" arrow on this version -- the rendered LEFT
  panel shows it as filled but Graphviz docs are sometimes ambiguous.
  Recommend verification with `dot -Tsvg` direct test.
- **Medium confidence** on R8 (arrow tip overlap): the bite is small
  (~3-4px), borderline-stack-residual; but visible at zoom.
- **Medium confidence** on R6 (fill opacity overshoot): the perception
  of "no tint" vs "faint tint" is somewhat subjective. Recommend a
  pixel-color sample in the rendered output to confirm.
- **Lower confidence** on R11-R14: visible at zoom, well in the
  acceptable-residual zone for most viewers.
- **Want to re-verify after R1+R2 fix:** how the new uniform-sqrt(2)
  ellipses compare to dot on long_labels and arrow_types -- the goal
  is to land BOTH ranges consistently with no discontinuity.
