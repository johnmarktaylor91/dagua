# Round 11 Audit -- Maximum Strictness Ceiling Test

## TL;DR

**Verdict: `CONTINUE_ROUND_11`.** The sprint has NOT hit ceiling. There are at least **two systemic, fixable defects** that the round-7/round-10 audits missed and that the metric pipeline is rewarding rather than penalizing. Both can be closed without touching the locked render-path constants and without regressing round-9 wins.

1. **Edge stem is missing on every simple-shape pair-fixture comparison** (16 of 17 nodes_shapes_* and many of the borders/styles parity cards). The arrowhead glyph renders, but the line connecting the two nodes does not. Graphviz draws a clean visible edge in the same fixtures. This is a render-path bug at edge_width=1.0, exposed by the round-7 "simple-shape comparison fill+border+arrow parity" overlay. With width >= 3.0 the stem appears (verified on `edges_styles_width_3_0`); below that it vanishes. **Cause is in `dagua/render/mpl.py`'s edge stem path, NOT in any locked constant.**

2. **Combo cards have illegible, severely truncated labels.** The round-9 density-aware-shrink scales node WIDTH/HEIGHT but does NOT scale label `font_size` proportionally. On 5-node combo cards the nodes shrink to ~25-35px while labels remain at 14pt, causing every text label to be clipped to 3-4 leading characters: "Ingest" -> "nges", "Validate" -> "lida", "Review" -> "evic", "Approve" -> "opro", "Ship" -> "hip". This is true even on the round-9 "wins" `combo_pie_bold` (L1=1.918) and `combo_donut_shadow` (L1=2.056). The L1 metric is rewarding pixel-mass parity at the cost of text legibility -- a genuine fidelity regression that the metric cannot see.

This is exactly the failure mode the sprint summary warned about: "when 'visual verification' disagrees with 'metric verification' repeatedly, suspect the metric pipeline." The round-9 metric-pass is hiding two distinct visual regressions.

## Methodology

I read 16 simple-shape comparison panels (all `nodes_shapes_*_vs_graphviz.png`), 5 worst-tier-A combo cards (`combo_kitchen_sink_5`, `combo_pie_gradient_bold`, `combo_bold_shadow_gradient`, `combo_hexagon_gradient`, `combo_kitchen_sink_1`), 2 round-9 "wins" (`combo_pie_bold`, `combo_donut_shadow`) as controls, plus probes of `nodes_borders_border_opacity_1_0`, `edges_styles_style_solid`, `edges_styles_width_3_0`, and the `nodes_fills_gradient_radial` outlier. I used pixel-grid probes via PIL+numpy on the dagua-side edge-stem corridor (y=270-330, x=380-420) to verify edge presence without relying on visual judgment alone.

Worst-card threshold: at minimum a card must be inspected if (a) it appears in the top-20 worst Tier-A residuals, OR (b) the round-7/round-10 audit certified it as "STOP-ready" / "principled residual" (control sample), OR (c) it is on a fixture path I had reason to suspect from the systemic-bug pattern.

## Per-card classification

### nodes_shapes_box3d (L1 = 3.781) -- `fixable_theme_or_render`

What I see: dagua left has Source box3d (~200x110px) at top, Target box3d at bottom, **NO visible edge line between them**, just a tiny black arrowhead glyph at y=358-368 floating directly above the Target node's top edge. Graphviz right has clean Source -> Target edge with a stroked vertical line and full arrowhead. Pixel probe confirms: dagua y=270-330, x=380-420 is solid white (min=255). Graphviz same region has 162 dark pixels (the visible edge stem).

This is the edge-stem-missing bug. Same defect on every other simple-shape parity card (see cross-card pattern below).

Concrete fix: in `dagua/render/mpl.py`'s edge-rendering path (the function that draws the line stem between two nodes given the EdgeStyle), the regression is masking edge stems at `style.width <= 1.0`. With `style.width = 3.0` (verified on `edges_styles_width_3_0`) the stem renders fine. Investigate the linewidth-to-pixel mapping in matplotlib FancyArrowPatch / Line2D arguments -- a likely culprit is a stale `linewidth=` calc that goes to 0 when the original width is 1.0pt at the active DPI/figsize combo (e.g. integer flooring). Round 11 should add a regression test that draws the canonical 2-node pair fixture at edge width = 1.0 and asserts that pixel column x=panel_center has at least N dark pixels in the gap between Source-bottom and Target-top.

Risk: medium. Touches edge stem rendering. Round-9 wins use 5-node graphs whose edges are at the default width=1.0 and visibly present (see combo_pie_bold image -- arrows + edges visible). So this isn't a universal "all width=1.0 edges are missing" bug. Most likely it's specific to the simple-shape pair fixture path where edge endpoints are computed against the larger `min_width=200, min_height=110` overrides, and the endpoint-port adjustment is over-shrinking the segment to length zero. Investigate `_adjust_port_for_shape()` and `ray_polygon_intersection()` (referenced at scripts/build_gallery_audit.py:1894-1896) -- the comment claims they "now handle shape-aware edge endpoints for all polygon shapes," but the test is by-eye, not by-pixel.

### nodes_shapes_circle (L1 = 3.383) -- `fixable_theme_or_render`

Same edge-stem-missing bug. Pixel probe shows 172 dark pixels in y=270-330,x=380-420, but those are the bottom border of the Source circle and top border of the Target circle, not an edge stem. (The circle fixture renders very large discs because shape=circle is forced to width==height in the parity overlay, expanding to ~250px diameter and consuming most of the inter-node corridor.) Same fix as box3d.

Risk: medium. Same as above.

### nodes_shapes_cylinder (L1 = 3.284) -- `fixable_theme_or_render`

Identical to box3d. Pixel probe: dagua dark count = 0 in edge corridor. Graphviz = 188.

### nodes_shapes_tab (L1 = 3.299) -- `fixable_theme_or_render`

Identical. Dark count: 0 / 162.

### nodes_shapes_double_circle (L1 = 3.166) -- `fixable_theme_or_render`

Identical. 0 / 171.

### nodes_shapes_note (L1 = 3.043) -- `fixable_theme_or_render`

Identical. 0 / 162.

### nodes_shapes_rect (L1 = 3.036) -- `fixable_theme_or_render`

Identical. 0 / 162. The arrowhead at y=360-368 is verifiable via pixel probe: it's 8 dark pixels at x=396-403, narrowing to 2 at y=370 (just above the Target rect's top border at y=370). Edge stem from Source at y=240 down to that arrowhead = pure white.

### nodes_shapes_diamond (L1 = 1.717) -- `fixable_theme_or_render` (NOT a control / NOT a "win")

This card was listed in the brief as "already a relative win, good control." It is NOT. It has the same edge-stem-missing bug. Pixel probe: dagua y=240-360, x=395-405 is uniformly RGB=(255,255,255). Reason the L1 is "low" relative to other shapes: the diamond polygon has thin pointed corners, so the dagua nodes occupy LESS canvas than the rectangular shapes (rect, box3d, tab, etc.), reducing the per-pixel mass that's mismatched against graphviz's compact layout. The L1 metric is rewarding the diamond's geometric inefficiency, not its parity quality.

This is a clear case of metric-vs-visual disagreement. The diamond card has the same defect class as the box3d card, just at lower L1 magnitude.

### nodes_shapes_star (L1 = 1.615) -- `fixable_theme_or_render` (NOT a control / NOT a "win")

Same as diamond. Listed in brief as "best basic shape; control" -- NOT a control. Pixel probe shows dagua y=270-330,x=380-420 has 0 dark pixels. The star's even-thinner geometry depresses L1 further, but the underlying defect (missing edge stem) is identical.

### nodes_fills_gradient_radial (L1 = 9.374) -- `fixable_theme_or_render` (round-10 plan still valid)

Round-10 audit's analysis stands: graphviz competitor not exercising radial gradient + dagua side rendering at full size. The round-10 PROMPT_round_10.md spec wired `style="filled,radial"` + `fillcolor="<fill>:<gradient_color>"` for graphviz's DOT path. That fix is still uncommitted (round 10 hit codex quota and was paused). It should be picked up in round 11.

But there's a NEW finding: even after the round-10 wiring lands, **the radial card will still have the edge-stem-missing bug** (the dagua side will still draw the gradient ellipse pair without a connecting stem because `nodes_fills_*` cards fall through the same `pair` fixture path). A pixel probe of the radial card confirms zero dark pixels in the edge corridor. Until the edge-stem bug is fixed, expect the L1 floor for this card to remain elevated even after the gradient wiring lands.

Concrete fix: same as the round-10 spec for radial gradient wiring + the edge-stem fix (#1).

### nodes_shapes_box3d / cylinder / tab / etc (L1 = 3.0+ family) -- ALL share the same root cause

Cross-card pattern: 16 of 17 simple-shape parity cards (every shape except `circle`, where the node itself is the dark mass in the probe zone) have zero dark pixels in the edge corridor on the dagua side, while graphviz consistently shows 140-190 dark pixels. This is a single defect manifesting across an entire fixture group.

Estimated L1 drop after fix: each card's L1 drops by ~0.4-0.6 (the missing edge contributes ~80-120 dark pixels worth of pixel-mass mismatch over ~1600x600). For 17 affected shape cards plus ~10 affected pair-fixture borders/fills/edges cards, total expected mean Tier A L1 drop: **0.10-0.15**.

Note: this defect coexists with the round-7 audit's "ceiling" verdict because the round-7 audit looked at the comparisons visually and noted "filled comparisons expose small color/weight diffs" (line 204 of dial_tuning_STATE.md) -- but never zoomed in to verify the EDGE itself was present. Mean L1 was rising 2.971 -> 3.417 in round 7, attributed to "visual upgrade not fidelity loss"; in fact part of that rise was the new fill color making the missing-stem more conspicuous.

### combo_kitchen_sink_5 (L1 = 3.617) -- `fixable_theme_or_render`

What I see: dagua left has 5-node tree spanning roughly (200,80)-(600,520). Each node is a TINY ellipse roughly 25-35px wide. The nodes' labels are SEVERELY TRUNCATED -- "Ingest" reads as "nges" (only middle 4 chars visible), "Validate" -> "lida", "Review" -> "evic" / "evie", "Approve" -> "opro", "Ship" -> "hip". The text overflows the node ellipse on both sides and the rendering pipeline crops to the node bbox. Graphviz right has perfect labels in cleanly-sized nodes.

This is the round-9 density-aware-shrink scaling node geometry but NOT scaling label font_size. In `dagua/render/mpl.py:_density_scaled_node_sizes()` (line 972), only `sizes` (the W/H tensor) is multiplied by `factor`; label `font_size` stays at the GraphStyle default (14pt) regardless of node count. When factor = sqrt(0.3/5) ~= 0.245 (the round-9 calibration for 5-node graphs), the nodes shrink to 25% of their base size while labels stay 100%. Result: text overflows.

Concrete fix: in `dagua/render/mpl.py` near line 972, add a parallel font_size scaling pass when density_aware_node_shrink is enabled:

```python
# After: sizes_scaled = sizes * factor
# Apply the same factor (clamped to a higher floor like 0.5) to render-time font_size.
# This is a render-only override; it does not mutate the user-facing NodeStyle.font_size.
font_factor = max(factor, 0.5)  # don't let labels shrink below 50% (readability floor)
# Pass font_factor down to the label-rendering pass; multiply against style.font_size at draw time.
```

Risk: medium. Theme `font_size = 14.0pt` is locked but this is a render-time scalar applied to the resolved value, not the theme value itself. Reverify round-9 wins (combo_pie_bold, combo_donut_shadow, evil_donut_diamond, clusters_opacity_*) after the fix -- their labels should become legible without the L1 spiking. Expected effect: visible improvement in label legibility on every multi-feature combo card; modest L1 changes (likely small +/- because legible text vs illegible squashed glyphs cover roughly the same pixel area).

The font-floor at 0.5 is conservative; 0.6-0.7 might also work and would protect even denser graphs from unreadability. Calibrate empirically.

### combo_pie_gradient_bold (L1 = 3.430) -- `fixable_theme_or_render`

Identical pattern. Same root cause. Same fix applies.

### combo_bold_shadow_gradient (L1 = 3.158) -- `fixable_theme_or_render`

Identical pattern. Same fix.

### combo_hexagon_gradient (L1 = 3.079) -- `fixable_theme_or_render`

Identical pattern. The hexagon shape is even worse for label clipping because the hexagon has angled sides that crop labels both at the ends (for short text) and in the middle of long words.

### combo_kitchen_sink_1 (L1 = 3.121) -- `fixable_theme_or_render`

Identical pattern.

### combo_pie_bold (L1 = 1.918) -- `fixable_theme_or_render` (NOT a "win" -- actively broken)

Round-9 sprint summary lists this card as a "win." Visually it is BROKEN: same illegible-label clipping. The L1 is "low" because the dagua side's miniature pie-fill ellipses happen to have cumulative pixel mass close to graphviz's small flat-fill ellipses, but the visual experience is "graph with unreadable labels" vs "graph with readable labels."

Expected effect of the font-scaling fix: label legibility restored. L1 may rise slightly (2.0 -> 2.3 ish) because the legible-text dagua-side pixel mass differs from graphviz's still-different-rendering text mass. THIS IS GOOD: it surfaces the real-vs-metric gap.

### combo_donut_shadow (L1 = 2.056) -- `fixable_theme_or_render` (same)

Same as combo_pie_bold. Listed as a round-9 "win"; actually broken.

### combo_stadium_gradient (L1 = 2.993) -- `fixable_theme_or_render`

Identical text-clipping bug.

### combo_bevel_shadow_gradient (L1 = 3.113) -- `fixable_theme_or_render`

Identical.

### combo_shadow_gradient (L1 = 3.112) -- `fixable_theme_or_render`

Identical.

### combo_ext_label_hexagon_gradient_bold (L1 = 3.217) -- `fixable_theme_or_render`

Identical. (hexagon variant; same shape-induced cropping aggravates the issue)

### evil_pie_shadow_gradient (L1 = 3.899) -- `principled_residual`

Round-10 audit's analysis was correct here: 1-node graph; canvas-occupancy mismatch; graphviz auto-shrinks 1-node graphs to a tiny ellipse. The font-size fix from #2 won't apply because density_aware_size_factor() returns 1.0 for `node_count <= 2` (line 941 of mpl.py). Single-node text legibility is not affected.

Keep as principled residual.

### combo_trapezoid_gradient (L1 = 3.775) -- both bugs apply

Both #1 (missing edge stem at edge_width=1) AND #2 (illegible labels at density_factor=0.245) apply. Same fixes.

### nodes_borders_border_opacity_1_0 (L1 = 1.566, round-8 win) -- not a regression source

This card uses a HORIZONTAL pair fixture (Default | 1.0 side-by-side), so no vertical-edge stem is needed. Round-8 win is genuine; the missing-stem bug from #1 doesn't surface here.

## Cross-card patterns and root causes

Two systemic findings, ranked by impact:

### A. Edge-stem-missing bug at edge_width <= ~1.0pt

**Manifestation**: 16 of 17 simple-shape parity cards + ~10 borders/fills/edges parity cards (all using the vertical pair-fixture with 2pt nodes), zero dark pixels in the edge corridor on the dagua side. Graphviz consistently shows the edge.

**Pixel-probe evidence**: probed 17 cards; results:
- box3d: dagua=0, graphviz=162
- cylinder: 0/188
- tab: 0/162
- double_circle: 0/171
- note: 0/162
- rect: 0/162
- star: 0/148
- diamond: 0/144
- ellipse: 0/184
- hexagon: 0/143
- pentagon: 0/147
- octagon: 0/168
- triangle: 0/144
- parallelogram: 0/144
- trapezoid: 0/144
- roundrect: 0/168
- circle: 172/122 (the dagua dark pixels are the node BORDER, not an edge stem; probed because the over-large circle invades the corridor)

Plus `edges_styles_style_solid: dagua=0` (same bug on the edges tier of cards), and `edges_styles_width_3_0: dagua=120` (visible -- so the bug is width-dependent).

**Likely cause**: in the matplotlib edge-rendering path, a width-to-pixels conversion is producing a 0-width line at width=1.0pt under the active DPI/figsize combo. The fix is small; recommend examining `dagua/render/mpl.py` for any place `linewidth=` is computed via `int(...)` or `round(...)` rather than passed as a float, OR examining whether `_adjust_port_for_shape()` is shrinking the segment endpoints inside the source/target bounding boxes (resulting in a zero-length segment for which matplotlib draws nothing).

**Verifying the fix**: a regression test that renders the canonical pair-fixture at edge_width = 0.5, 1.0, 1.5, 3.0 and asserts that dark-pixel count along the edge corridor (between source-bottom and target-top, narrow x band centered on the panel) scales monotonically and is ALWAYS positive at width >= 0.5.

### B. Label font-size not scaled with density-aware node shrink

**Manifestation**: every multi-feature combo card with 5 nodes has labels truncated to 3-4 leading characters because text overflows the shrunk node bbox. Confirmed across 8+ combo cards including the round-9 declared "wins" (combo_pie_bold, combo_donut_shadow).

**Likely cause**: `_density_scaled_node_sizes()` at `dagua/render/mpl.py:972` multiplies `sizes` (W/H tensor) by `factor` but does not propagate the same scalar to label `font_size` rendering. The label-rendering path elsewhere reads `style.font_size` directly without multiplying by the active density factor.

**Fix sketch**: thread the `factor` from `_density_scaled_node_sizes()` to the label-drawing pass. At draw time, use `effective_font_size = base_font_size * max(factor, FONT_FLOOR)` where FONT_FLOOR is empirically 0.5-0.7.

**Verifying the fix**: a regression test that renders the 5-node workflow fixture (Ingest -> Validate/Review -> Approve/Ship) with density_aware_node_shrink=True at the graphviz_strict theme and asserts that the rendered "Ingest" label width / node width ratio stays below 1.0 (text doesn't overflow). Currently that ratio is ~3.0 (text 3x wider than node).

## Recommended fix order for codex round 11

| # | Fix | Cards affected | Effort | Impact | Risk |
|---|---|---|---|---|---|
| 1 | Fix edge-stem rendering at width<=1.0pt in `dagua/render/mpl.py` (suspect: port adjustment over-shrinking the segment, or linewidth flooring). Add regression test for canonical pair fixture at width=0.5/1.0/1.5. | All ~26 pair-fixture parity cards (shapes, borders, fills, edges, styles) | 30-90 min | Mean Tier A L1: ~1.785 -> ~1.62 (estimated 0.10-0.15 drop) | Medium. Must verify round-9 wins (5-node combos) still render edges correctly, since their edges also default to width=1.0. The fact that those edges DO render in combo cards (visible in combo_pie_bold etc.) suggests the bug is specific to the pair-fixture path and the fix should be surgical. |
| 2 | Thread density_factor into label font_size at render time in `dagua/render/mpl.py` near the existing `_density_scaled_node_sizes()` call. Empirical FONT_FLOOR = 0.6. | All combo (5-node) cards including round-9 "wins" (~30+ cards). | 30-60 min | L1 may rise slightly but visual/legibility quality dramatically improves. The L1 metric will more honestly reflect the rendering quality. | Medium. Round-9 wins (combo_pie_bold etc.) currently have L1 ~2.0 because of pixel-mass parity at unreadable-text quality. Fix may push L1 to 2.5-3.0 on those cards but readability becomes correct. **This is the right tradeoff -- the metric was lying.** Tell the auditor explicitly to NOT flag the L1 rise as regression; the round-9 "wins" were never genuine. |
| 3 | Pick up round-10's deferred Item D fixture wiring + tier reclassification (radial gradient DOT wiring + 4-card Tier A->C reclass). | nodes_fills_gradient_radial (most), 4 reclassified cards. | 15-30 min | Mean Tier A L1: small additional ~0.1 drop from cleaner radial card + reclassification removing high-L1 cards from the Tier A pool. | None. Fixture-only. |

Total expected drops are dominated by the qualitative win on combo cards (legible text), not L1. The L1 metric will still be in the same ballpark, but the visual quality will be qualitatively better and the metric will start being honest about it.

## Hard guardrails respected

- `_PATTERN_FILL_RESOLUTION`, `_HATCH_PATTERN`, `_MIN_HATCH_LINEWIDTH_POINTS`, `_PIE_GRADIENT_OVERLAY_ALPHA_MULTIPLIER`: NOT touched in any of the recommendations above.
- `dagua/styles.py` node-size defaults (75x50pt) and font_size (14.0pt): NOT touched. Recommendation #2 is a render-time multiplier that doesn't mutate the theme value.
- Density-aware shrink itself: NOT changed; only EXTENDED to also scale label font_size proportionally.
- GRAPHVIZ_STRICT_THEME values: NOT touched. The proposed fixes are entirely in `dagua/render/mpl.py` (edge stem at low width + density-factor passed to label font_size) and `scripts/build_gallery_audit.py` (the existing round-10 Item D wiring).
- Round-9 "wins" are reverified: the L1 metric will likely shift on combo_pie_bold (~1.9 -> 2.5), combo_donut_shadow (~2.1 -> 2.7), evil_donut_diamond (~2.1, mostly unaffected because it's a 1-node fixture with density_factor=1.0). The shift is honesty, not regression.

## Why round 7 / round 9 / round 10 missed this

Three reasons:

1. **The auditor at round-7 declared STOP after looking at "filled comparisons expose small color/weight diffs"** but never zoomed in to the inter-node corridor on the simple-shape pair-fixture cards. Without a pixel-grid probe of the edge corridor, the missing edge stem looks like just a small visual difference; on closer inspection it is a missing FEATURE.

2. **The L1 metric rewards pixel-mass parity, which becomes meaningless when both sides have the same pixel mass via different mechanisms** (dagua: shrunk nodes with overflowing illegible text; graphviz: small nodes with legible text). Round-9's density-aware-shrink "win" was driven by the metric pipeline misinterpreting pixel-mass match as quality match.

3. **Round-10 audit used the same metric to certify fill-pattern as the only remaining residual**, but the L1 mass on combo cards was actually being driven by these two latent bugs, not by canvas-occupancy or fill-pattern style. The round-10 audit's interpretation of L1 mass was honestly wrong.

## What round 11 should NOT do

- Do not chase fill-pattern geometry in the dagua renderer (per round-10 guardrails).
- Do not increase node-size defaults to compensate for label clipping in combos (would regress simple-shape parity).
- Do not disable density-aware shrink (would regress the genuine multi-feature density wins).
- Do not change the L1 calculation to add an SSIM-weighted term (out of scope for a "low-risk dial tweak" round).

## Verdict

**`CONTINUE_ROUND_11`**

Two fixable defects with concrete, low-risk fixes:
1. Edge-stem rendering bug at edge_width <= 1.0pt in `dagua/render/mpl.py` -- surgical fix in the matplotlib edge rendering path; add pixel-probe regression test.
2. Density-aware-shrink missing label scaling in `dagua/render/mpl.py:_density_scaled_node_sizes()` and downstream -- thread the factor through label font_size rendering with FONT_FLOOR=0.6.

Plus round-10's Item D fixture work (radial gradient DOT wiring + 4-card Tier A->C reclass) which was paused mid-dispatch due to codex quota.

Round 11 is genuinely worth running. Round-9's "ceiling" was the L1 metric saying ceiling; the visual reality has at least these two open seams.
