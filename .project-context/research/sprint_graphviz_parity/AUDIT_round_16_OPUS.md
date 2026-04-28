# Graphviz Theme Parity Audit -- Round 16 (Opus, post-round-15 verification)

## Methodology

- Source dir: `/home/jtaylor/projects/dagua/eval_output/graphviz_theme_round_15/two_way/` (1800x794, two-column).
- Column headers verified on every PNG: yes ("Graphviz dot" left, "Dagua (strict)" right).
- Panels reviewed (12): node_shapes_showcase, tiny_graph, single_edge, arrow_types, state_machine, pipeline, diamond, balanced_binary_tree, colors_showcase, nested_clusters, cluster_showcase, multi_cycle.
- Pixel measurement: per-node ellipse W/H (PIL/numpy bbox detection); per-arrowhead row-width profile (filled-pixel scan); per-edge-label cap-height (right-of-line dark-row tight bbox); column divider auto-detected at x=800 in two-way layout.
- Bar applied: "genuinely identical save for documented rendering-stack residuals." Maximally picky.

## Round 15 Fix Verification (5 items)

### F1 (star black stroke) -- PASS

Round 14 found dagua's star outline rendering as RGB ~156 gray due to centroid-radial inset collapsing perpendicular ribbon at acute apex angles. Round 15 replaced inset with edge-perpendicular miter intersections in `dagua/render/borders/inset.py`.

Pixel audit on node_shapes_showcase.png star bbox:

| metric | DOT | DAGUA round-13 | DAGUA round-15 |
|---|---|---|---|
| pixels < 60 (true black) | 3 / 2940 | (gray ~156) | 58 / 3680 |
| pct < 60 | 0.10% | ~0% | 1.58% |
| outline min intensity | 74 | 156 | **3** |

Dagua's star outline is now SOLID BLACK (in absolute terms even darker than dot, which is fine — "black is black"). The gray-pen regression is gone. Visual confirmation: zoomed star crops show clean dark pentagram outline on both columns, no AA-haze.

**Verdict: PASS** (with minor shape-proportion residual noted in Remaining Departures P2 below).

### F2 (small ellipse height) -- FAIL (over-correction)

Round 14 measured small-graph ellipse heights at 82-88% of dot. min_height bumped 33 -> 38.

Per-panel height ratios (dagua/dot) in Round 15:

| panel | ellipse | dot WxH | dagua WxH | W ratio | H ratio | W/H dot | W/H dagua |
|---|---|---|---|---|---|---|---|
| tiny_graph | In/Mid/Out | 161x106 | 157x118 | 0.97 | **1.11** | 1.52 | **1.33** |
| single_edge | Source | 208x107 | 204x113 | 0.98 | 1.06 | 1.94 | 1.81 |
| single_edge | Sink | 161x106 | 150x111 | 0.93 | 1.05 | 1.52 | **1.35** |
| diamond | Start | 162x80 | 159x90 | 0.98 | **1.13** | 2.02 | **1.77** |
| diamond | End | 161x100 | 155x100 | 0.96 | 1.00 | 1.61 | 1.55 |
| pipeline | Input | 120x69 | 128x82 | 1.07 | **1.19** | 1.74 | **1.56** |
| pipeline | Preprocess | 197x72 | 226x86 | 1.15 | **1.19** | 2.74 | 2.63 |
| pipeline | Transform | 195x72 | 220x86 | 1.13 | **1.19** | 2.71 | 2.56 |
| pipeline | Postprocess | 209x73 | 242x86 | 1.16 | **1.18** | 2.86 | 2.81 |
| pipeline | Output | 143x70 | 158x84 | 1.10 | **1.20** | 2.04 | **1.88** |

Acceptance band was 0.95-1.05 height ratio. **Round 15 dagua heights are 1.05-1.20 -- the small/medium ellipses overshoot by 5-20%, with pipeline.png ellipses uniformly ~19% too tall.** Visually obvious: dagua's ellipses on tiny_graph, single_edge, diamond, pipeline look noticeably puffier / less squished than dot's.

**Verdict: FAIL.** F2 went from 14-18% short to 5-20% tall -- crossed the target band, not landed in it. The min_height 33->38 was a 15% bump; the small ellipses needed ~+10% (to land at min_height ~36) but the multi-character ellipses on pipeline.png didn't need any vertical bump and now look bloated.

### F3 (ellipse curve factor) -- PARTIAL (masked by F2)

curved_factor 1.15 -> 1.22. Goal: widen ellipses on multi-character labels to match dot's W/H signature.

W ratios (dagua/dot) in pipeline.png: 1.07-1.16 -- the ellipses ARE wider in absolute terms. So F3 by itself worked.

BUT W/H aspect ratios:
- DOT pipeline ellipses W/H: 1.74-2.86 (clearly elongated horizontally)
- DAGUA pipeline ellipses W/H: 1.56-2.81 (still rounder than dot)

And on small ellipses (tiny_graph, single_edge Sink, diamond Start), W/H gap is even larger: dot 1.52-2.02 vs dagua 1.33-1.81 -- dagua looks distinctly more circular. **The F2 overshoot dominated F3's contribution: heights grew faster than widths, so the overall shape became MORE circular, not less.**

**Verdict: PARTIAL.** F3 added width but F2 added even more height. Net W/H still 8-13% rounder than dot on multi-char ellipses, 13-15% rounder on small ellipses. To finish F3 cleanly, dagua needs the F2 overshoot rolled back so the curved_factor 1.22 actually widens the rectangle without proportional height gain.

### F4 (arrow chunk) -- PARTIAL (wrong axis fixed)

arrow_width 12 -> 14, arrow_length kept at 14.

Per-row arrowhead profile measurement on tiny_graph (a representative vertical edge):

| panel | head_h (length) | head_max_w (width) | h/w aspect | filled area |
|---|---|---|---|---|
| DOT | 29 | 23 | **1.26 (taller)** | 406 |
| DAGUA | 22 | 25 | **0.88 (wider)** | 330 |

DOT arrowhead is **taller than wide** (h/w=1.26). DAGUA arrowhead is **wider than tall** (h/w=0.88). Width ratio 25/23=1.09 (dagua 9% wider); height ratio 22/29=**0.76** (dagua 24% SHORTER); area ratio 330/406=0.81 (dagua 19% less ink).

Single_edge confirms: DOT 27x24 (h/w 1.13), DAGUA 25x29 (h/w 0.86).

**The width-only bump did the wrong thing.** Round 14 found dagua arrows narrower AND less filled. The fill-area gap was 30%, mostly attributable to LENGTH being short (the triangle is shorter on dagua). Round 15 widened the arrowhead but kept length at 14, producing a SQUAT arrowhead (wide, short, less ink).

Visual confirmation on diamond.png and balanced_binary_tree.png and pipeline.png: dagua's arrowheads look "chunky-but-stubby" -- wider than dot's but visibly less long. On diamond.png in particular, the diagonal arrowheads look out-of-proportion (oversized in width, too short in length).

**Verdict: PARTIAL.** Width landed (slightly over). Length still short. Arrowhead shape now LATERALLY squashed instead of correctly proportioned. arrow_length needs 14 -> 17-18 with arrow_width holding at 14, OR back the width down to 12-13 and bump length to 17.

### F5 (edge label font) -- FAIL (over-correction)

Ratio 10/14 -> 9.3/14, expected ~10.6pt at 14pt base.

Edge label cap heights on arrow_types.png, measured to right of edge line (so source ellipse outline doesn't contaminate):

| arrow type | dot label cap h (px) | dagua label cap h (px) | dagua/dot |
|---|---|---|---|
| normal | 12 | 9 | 0.75 |
| vee | 8 | 6 | 0.75 |
| dot | 11 | 7 | 0.64 |
| diamond | 12 | 9 | 0.75 |
| tee | 8 | 7 | 0.88 |
| crow | 8 | 5 | 0.62 |
| circle | 12 | 9 | 0.75 |
| open | 11 | 9 | 0.82 |
| none | 8 | 6 | 0.75 |

Mean dagua/dot cap-height ratio: **~0.75**. Dagua edge labels are now 25% SMALLER than dot's. Round 14 had them 10-15% LARGER. **Round 15 swung past the target.** The ratio change 10/14 -> 9.3/14 is a ~7% reduction in font size, but apparent rendered output dropped ~32% in cap height -- non-linear because matplotlib font hinting at small pt sizes drops to next pixel-rounded glyph row.

State_machine.png visual check: dagua edge labels ("restart", "reset", "retry", "resume") are still subordinate to node labels (good, that part of F5 holds), but they are now visibly SMALLER than dot's edge labels in the same panel.

**Verdict: FAIL.** Subordination was achieved already at round 13. Round 15 was meant to take the absolute size from ~12.5pt to ~11pt to match dot. Instead, dagua dropped to ~9pt -- noticeably under dot. Need to back off ratio to 10/14 (round 13) or 9.7/14 to land in the 10-11pt band.

## Round 15 tally

| Fix | Verdict |
|---|---|
| F1 star pen | **PASS** (solid black, F1 fully landed) |
| F2 ellipse height | **FAIL** (overshoot 5-20% on small/multi ellipses, was 14-18% short, now 5-20% tall) |
| F3 ellipse curve factor | **PARTIAL** (added width but F2 height dominated; net W/H still rounder than dot) |
| F4 arrow chunk | **PARTIAL** (width landed; length still short, producing wrong h/w aspect 0.88 vs dot's 1.26) |
| F5 edge label font | **FAIL** (over-correction; now 25% smaller than dot, was 10-15% larger) |

**Score: 1 PASS / 2 PARTIAL / 2 FAIL out of 5.**

## Remaining Departures (priority ranked, with measurements)

### P1 -- Edge label font OVERSHOOT (FAIL on F5)
Dagua edge labels 25% smaller in cap-height than dot's. Visible at glance on state_machine.png ("retry", "resume" labels look tiny vs dot's). Recommend backing ratio off to 9.7/14 or 10/14, then re-measuring.

### P2 -- Small/medium ellipse height OVERSHOOT (FAIL on F2)
Pipeline.png ellipses 18-20% taller than dot. Tiny_graph ellipses 11% taller. Visually all dagua ellipses look puffier and rounder than dot. min_height should drop from 38 to 35-36.

### P3 -- Arrow length is short / arrowhead aspect ratio inverted (PARTIAL on F4)
Dagua arrowheads h/w=0.88 (wide-short). Dot's are h/w=1.26 (tall-narrow). Dagua filled area 19-32% less than dot on short edges. Bump arrow_length 14 -> 17, optionally back arrow_width down 14 -> 13. Verify the rendered head is filled triangle, not an outline.

### P4 -- Ellipse W/H aspect still circular vs dot's elliptical (PARTIAL on F3, masked by F2)
Even when F2 is fixed, F3 may still need to push curved_factor higher (1.25-1.30) so multi-character ellipses on pipeline.png and small ellipses on tiny_graph have W/H within 5% of dot's 1.52-2.86 range.

### P5 -- Star shape proportions (post-F1 residual)
Dagua star is 13% wider and 10% taller than dot's, with visibly LONGER, more attenuated points. F1 fixed pen color but the geometric correction also slightly enlarged the bounding box. Reduce the star compact_factor by ~5-7% to align silhouette with dot's classic pentagram proportions.

### P6 -- arrow_types panel "vee" arrowhead is degenerate
Dagua's vee on arrow_types.png renders as a wide flat outline rather than dot's clean filled "V". Visible in zoomed arrow_types crop. Looks like outline-only, not a triangular fill. Likely a head-shape spec issue specific to "vee" arrow type.

### P7 -- arrow_types panel "tee" arrowhead misalignment
Dagua's "tee" appears as a long horizontal bar floating mid-edge (NOT touching where the line ends). Dot's "tee" sits flush at the line tip with a short horizontal bar. Same defect flagged in round 14 P7 -- still present.

### P8 -- arrow_types panel "dot" and "circle" arrowheads OVERSIZED
Dagua's "dot" filled circle is visibly LARGER than dot's; same for "circle" open ring. Likely scaled with the arrow_width 12->14 bump that also affected dot/circle arrow shapes that share that parameter. Decouple "dot/circle" arrow scale from the triangle arrow_width.

### P9 -- diamond.png arrowheads chunky out of proportion vs node size
Dagua's diagonal diamond arrowheads look oversized relative to the small ellipse nodes. This is the F4 width-bump making proportionally wide arrowheads that look unbalanced when nodes are small. Tied to P3 fix.

### P10 -- Star points slightly asymmetric on dagua
On the showcase star crop, dagua's star has very slightly uneven point lengths (one point appears longer than the others). Possibly numerical asymmetry in the new edge-perpendicular miter calculation. Not severe but visible at zoom.

## New Issues from Round 15

- **NR1 (regression):** F2 overshoot. Round 13 had small ellipses 14-18% short; round 15 overshoots to 5-20% tall -- the change crossed the target band rather than landing in it. New: pipeline ellipses, previously OK, are now 18-20% too tall.
- **NR2 (regression):** F5 overshoot. Round 13 edge labels 10-15% too large; round 15 over-corrects to 25% too small. New visual artefact: dagua edge labels look almost too small to read at default zoom on state_machine.png and arrow_types.png.
- **NR3 (residual):** F4 changed arrowhead aspect ratio from "narrow-tall" (closer to dot) to "wide-stubby" (further from dot in shape, even if filled area is closer). The unilateral width bump changed shape rather than scaling proportionally.

## User-Flagged Issues -- Final Status

- **"Arrows are wonky"** -- NOT YET RESOLVED. F4 shifted the deficit but did not close it. Dagua arrowheads are now wide-short instead of narrow-short; visible on diamond.png, balanced_binary_tree.png, pipeline.png as oversized chunky heads. Plus tee/vee/dot/circle arrow type variants on arrow_types.png still wrong shape (P6, P7, P8).
- **"Text isn't centered"** -- RESOLVED. Node labels appear vertically centered in their shapes on all 12 panels. No new issue.
- **"Different font"** -- RESOLVED (font-family residual; no change in round 15 needed). Glyph shapes look comparable.
- **"Cluster bounding boxes look like shit"** -- DEFERRED per H4/H5. nested_clusters.png and cluster_showcase.png still have clusters cutting through nodes, label collisions, and Outer Group containment errors. Acknowledged outside cosmetic scope.

## Acceptable Residual

- Sub-pixel antialiasing (matplotlib FreeType vs Cairo+Pango).
- Font hinting differences in label edges.
- B-spline routing profile vs Bezier on long curved edges (state_machine "reset" curve, multi_cycle back-edges).
- Layout-side cluster issues (H4/H5).
- Star/cylinder vertical overlap on node_shapes_showcase (layout-side; no node_sep in GraphStyle).
- Cylinder mid-line shape regression NR2 from round 14 (deferred per round-15 spec).

## Final Recommendation: **CONTINUE**

This is not "genuinely identical save for documented residuals." Specifically:
- F2 over-corrected (small ellipses 5-20% too tall, pipeline ellipses 18-20% too tall).
- F5 over-corrected (edge labels 25% too small).
- F4 only partial (width landed; length still short, producing wide-stubby heads).
- F1 landed but star geometry slightly off (P5).
- Multiple arrow-type-specific defects on arrow_types.png (P6, P7, P8) still present.

Pickiness applied: the bar is "indistinguishable", and visible at-glance discrepancies remain on every priority panel.

Recommended Round 16 spec (one round, six narrow fixes):

1. **F2 rollback partial:** min_height 38 -> 35-36. Goal: small-graph ellipse height ratio land in 0.97-1.04, not 1.05-1.20.
2. **F5 rollback partial:** edge label ratio 9.3/14 -> 9.8/14 or 10/14. Goal: edge label cap-height ratio land in 0.95-1.05, not 0.62-0.88.
3. **F4 length bump:** arrow_length 14 -> 17. Keep arrow_width at 14. Goal: arrowhead h/w aspect 1.05-1.30 (matching dot's 1.13-1.26), filled area within 5% of dot.
4. **F3 sustain:** keep curved_factor 1.22 (or push to 1.25 if F2 fix doesn't recover the W/H gap). Re-verify W/H ratio after F2 rolls back.
5. **P5 star compact factor -5% to -7%** so dagua star silhouette matches dot's tighter pentagram proportions (W ratio 1.13 -> 1.05; H ratio 1.10 -> 1.02).
6. **Arrow-type-specific defects:** P6 vee should be filled triangle; P7 tee bar should sit flush at line tip; P8 dot/circle marker scale should NOT inflate with general arrow_width.

Verify after round 16 with the same per-panel pixel-measurement script (paste this audit's numbers as the round-15 baseline). Stop only if all five F-items land in 0.95-1.05 ratio band AND P6/P7/P8 are visually fixed.

## Confidence

**HIGH** on the per-fix verdicts (each backed by multi-panel pixel measurements: F1 by 2 metrics on 1 panel; F2 by 10 ellipse measurements across 4 panels; F3 by W/H ratio comparison on 9 ellipses; F4 by 3 arrow profile measurements on 2 panels; F5 by 9 edge label cap-height measurements on 1 panel).

**HIGH** on the recommendation to CONTINUE -- two FAIL-grade overshoots (F2 height, F5 font) and one PARTIAL (F4 length) leave the parity bar clearly unmet.

**MEDIUM-HIGH** on the recommended numeric corrections in the next-round spec -- the magnitudes are interpolated linearly from round-13 + round-15 measurements (for F2 and F5, "split the difference" between the two over-correction directions). Verify on a single dispatch before committing.
