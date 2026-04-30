# Round 5 Audit -- dial_tuning

## Verdict

- New audit: **REGRESSION**
- Stop criteria status: **CONTINUE** (one corrective round-6 -- this is fixable but the round-5 strategy was wrong)
- Overall trajectory: round 4 was the high-water mark; round 5 went in the wrong direction. Mean Tier A L1 went 2.454 -> 8.121 (3.3x worse). Round 5 over-corrected based on a misread of the round-4 finding (R4-N1 said ~75x50pt; codex said "still too small" and went to 270x120 -- 5x larger -- without measuring).

## The size-mismatch question (Group 1 measurements)

Connected-component pixel measurements on the simple-shape and gradient comparison panels (panel half = 800 px wide, 600 px tall, ~100 DPI). Per-node bbox measurements:

| Card | Dagua node (px) | Graphviz node (px) | Ratio (W) | Visual verdict |
|------|---|---|---|---|
| ellipse | 258 x 130 | ~290 x 163 (single) | 0.89 | dagua is now marginally smaller W, ~0.80x H. Visually similar, hairline border still |
| circle | 258 x 250 (Source+Target overlap visually) | 225 x 225 | 1.15 | dagua is now LARGER W than graphviz; the two dagua circles overlap vertically because they don't fit |
| rect | 259 x 132 | ~210 x 153 (single) | 1.24 | dagua is now WIDER than graphviz; height 0.86x. Visually rect width is over-correction |
| linear (gradient) | 262 x 130 | ~290 x 163 (single) | 0.89 | gradient applied; size 0.80-0.89 of graphviz |
| striped | 262 x 131 | ~290 x 163 | 0.89 | stripes applied; size matches gradient_linear |

Combo cards (5+ nodes) where round 5 catastrophically regressed:

| Card | Dagua content (px) | Graphviz content (px) | Ratio | Note |
|------|---|---|---|---|
| combo_bold_shadow_gradient_rounded | 613 x 519 (entire panel) | 123 x 142 | dagua **5.0x wider, 3.6x taller** | dagua per-node ~325x150; graphviz per-node ~50x30 |
| combo_cylinder_dashed_shadow_gradient | 334 x 131 (single node) | 123 x 142 (entire graph) | ~2.7x | same root cause |

**Critical insight: graphviz auto-shrinks per-node size as graph grows.** Simple-shape graphviz panel: ~290x163 px per node (2 nodes). Combo graphviz panel: ~50x30 px per node (5 nodes). Dagua's renderer uses a fixed pt floor and does NOT auto-shrink. So a SINGLE pt-value cannot match graphviz on both simple-shape AND combo paths -- this is structural.

**Recommended pt-values for round 6:**
- For SIMPLE-SHAPE pair-fixtures (Source -> Target only, 2 nodes): need ~290 px wide -> `min_width = 200pt`, `min_height = 110pt` to match graphviz at 100 DPI.
- For DEFAULT theme (used by combo cards, clusters, multi-node graphs): need ~50-60pt wide to match graphviz auto-shrunk combo nodes -> revert closer to round-4's `min_width=75pt, min_height=50pt`.
- The right architectural fix is FIXTURE-LOCAL OVERRIDE for simple-shape pair-fixtures, with theme default at the round-3/4 small value. Round 5's mistake was applying the simple-shape fix globally to the theme default.

**Single-number compromise (if forced):** `min_width=75pt, min_height=50pt` (round-4's R4-N1 recommendation). This makes simple-shape look small but combo cards stop catastrophically diverging. The L1 reduction on combo cards (which are 50%+ of the gallery) more than offsets the simple-shape regression. This is the SAFER round-6 path.

## Combo-card regression (Group 2)

Root cause: **hypothesis (a) is correct -- removing DECORATIVE_FILL_CARD_MIN_HEIGHT broke gradient/pattern combo cards, AND the global theme bump from 54x36 to 270x120 made dagua nodes ~5x larger than graphviz on multi-node combo cards.**

Direct evidence in `comparisons/combo_bold_shadow_gradient_rounded_vs_graphviz.png`: dagua's 4 gradient nodes plus shadows fill 613x519 px; graphviz's 5 nodes fit in 123x142 px. The dial values (gradient direction, shadow offset, corner radius) are RIGHT -- the problem is that dagua nodes are 5x too big. Same for cylinder_dashed_shadow_gradient.

Hypothesis (b) -- "underlying styling differences exposed at bigger size" -- is partial: gradient direction and pattern density look correct. Hypothesis (c) -- "z-order regression on multi-layer combos" -- has NO evidence; layers paint in correct order in the panels I examined.

The cluster z-order fix is independent and not the cause of combo regression.

## clusters_opacity_1_0 regression (Group 3)

Round 4: 28.6. Round 5: 40.8 (+42% worse).

What I observed in `comparisons/clusters/1_0_vs_graphviz.png`:
- Dagua's outer cluster fills the entire 600x500 px panel half (solid blue at opacity 1.0).
- Graphviz's outer cluster is a tight 200x500 vertical strip.
- Round 5 z-order fix DID work: dagua's inner cluster border is now visible above the fill (round 4 reported it was missing at opacity 1.0).
- BUT round 5's theme bump made dagua's nodes (Outer A/B/C/D) ~258x130 px each. They no longer fit inside a tight cluster strip, so the cluster bbox spreads to enclose them across the panel. Graphviz's tiny ~50x30 nodes fit in the narrow strip.
- **Result: more saturated solid blue pixels mismatching white-bg-with-tiny-cluster on the graphviz side.** Z-order fix gave back +5-10% L1, but the size-bump cost +25-30% L1 -- net regression.

The opacity dial itself works (0.3 -> 0.6 -> 1.0 progresses correctly). Layout is dominantly the issue, exacerbated by node size.

## Pixel-diff sanity check (Group 4)

The round-5 numbers ARE honest in metric pipeline terms (no new normalization issue introduced). The sanity check holds:
- gradient_linear: round 4 said 2.288, round 5 says 12.474. Visually gradient_linear panel shows dagua ellipse ~262x130 at decent size, gradient applied -- but the per-card heatmap shows tons of mismatch since the graphviz side has its bigger filled-blue ellipse and dagua has gradient. The 12.5 number reflects PIGMENT mismatch over a substantial area, not normalization noise.
- combo numbers (~40-53) reflect real catastrophic size mismatch -- 5x size diff means ~80% of pixels disagree on what color they should be.
- clusters_opacity_1_0 = 40.8 reflects panel-spanning saturated-fill mismatch.

No new pipeline bug. Round 5's L1 increase is a TRUE visual regression.

## Round 6 recommendation

1. **Theme `min_width` / `min_height` to set: `75pt` x `50pt`.** Revert the round-5 270x120 bump. This is closer to graphviz's auto-shrunk combo size and matches R4-N1's original suggestion. Visually simple-shape cards will look "smaller than graphviz" again, but combo/cluster cards will stop catastrophic regression. Mean L1 will drop substantially.

2. **DECORATIVE_FILL_CARD_MIN_HEIGHT (or equivalent fixture override): RESTORE.** The round-4 fixture-local override was the correct architectural pattern. Restore it for `nodes_fills_*` and `nodes_shapes_*` simple-shape pair-fixture builders. Also extend it to ALL pair-fixture comparison panels (simple-shape, fills, opacity, border opacity, etc.) so they render at ~200x110pt to match graphviz's not-shrunk single-pair size. Combo and cluster fixtures keep theme default (75x50pt).

3. **Cluster z-order fix: KEEP.** It successfully made the inner cluster border visible at opacity 1.0 (round 4 R4-N5). Don't revert; it's correct.

4. **`parity_metrics.py` tolerance bump (rx 2.0 -> 120.0, ry 2.0 -> 50.0): REVERT to 2.0pt.** This is critical. Round 5's bump effectively turned off the geometric parity lock. The 99.38% and 100% lock floors in test_parity_metrics.py were achieved with 2.0pt tolerance; at 120.0pt they're trivially satisfied and meaningless as a regression gate. Revert tolerances and re-establish lock against the new (smaller) theme default.

5. **Other -- restore arrowheads on simple-shape COMPARISON panels (not just pair-fixture references)** -- still open from R4-N4. The ellipse panel I read showed a tiny dark triangle near Target but graphviz's clean filled-arrow ~10pt is still missing. Same on circle, rect, gradient panels.

6. **Dagua-side fill color on simple-shape pair-fixtures.** Currently the white-interior + hairline-border look is the chief residual cosmetic gap on simple-shape comparisons even after size fix. Pair-fixture should use graphviz's `lightblue` (`#E0EFFF` or similar) fill + `~1.5pt` blue stroke to match graphviz's filled chunky look. R3-N6 / R3-N7 / R4-N1 still open. This is fixture-local, not theme.

**Estimate: ~1.5-2 rounds away from genuine ceiling.** Round 6 reverts the over-correction and re-introduces the per-fixture override pattern. Round 7 (if needed) addresses simple-shape fill/stroke and arrowheads on comparison panels. Then parity numbers (mean Tier A L1) should come back under 3.0 with the fixture-local approach properly applied.

End round 5 audit.
