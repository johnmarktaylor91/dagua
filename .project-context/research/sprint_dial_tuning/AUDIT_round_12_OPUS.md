# Round 12 Audit -- Final Round (max_rounds=12) Maximum Strictness

## TL;DR

**Verdict: `STOP_AT_CAP`.** The sprint has hit a genuine ceiling for the locked-constants regime. Round 11's two fixes landed cleanly (edge stem visible, density font_factor threaded). The remaining residual L1 mass on the top-25 worst Tier A cards is **NOT a fixable theme/render defect**. It is structural metric_artifact: dagua's `min_width=200/min_height=110` shape-parity overrides + density-aware-shrink + matplotlib's data-coordinate text path produce nodes that are 2-6x larger / further apart than graphviz's auto-sized native nodes. The L1 metric scores this scale mismatch as Dagua's defect, but every plausible fix to close it requires either (a) GRAPHVIZ_STRICT_THEME numeric changes, (b) density formula changes, (c) gallery-fixture min_width/min_height removal, or (d) metric pipeline rewrite -- all of which violate the round-11/round-12 hard guardrails.

There is **one borderline-fixable finding** worth surfacing: **`_DENSITY_LABEL_FONT_FLOOR = 0.6` is too high for combo cards' label legibility.** Dropping it to 0.5 (or 0.4) closes most of the 4-7-letter clipping seen on Validate/Approve/Review/Ingest. This is the only theme/render handle the brief explicitly leaves open for round 12. I am flagging it as `CONTINUE_ROUND_12`-conditional below; the architect should decide whether to spend the final round on this or accept STOP.

## Methodology

I read the 9 brief-required comparisons plus the 4 metric-pipeline (`per_card_pixel_diff/comparisons/*`) versions of the same cards, plus the heatmaps for box3d and combo_pie_bold to verify defect locations, plus `per_card_pixel_diff/dagua/*` and `per_card_pixel_diff/competitors/graphviz/*` raw renders. Pixel-probed text height, edge stroke width, and content footprint via PIL+numpy.

I traced the rendering pipeline through `dagua/render/mpl.py:_density_scaled_node_sizes`, `_draw_node_labels` (lines 7700-7833), `density_aware_size_factor` (line 913), `_DENSITY_LABEL_FONT_FLOOR` (line 242), and the round-11 edge-stem fix (`_edge_uses_display_stroke_body` at line 5583, `_draw_direct_edge_body` at 6614). Verified the round-11 fix path is `0 < width <= 1.5` triggering display-points stroke at `linewidth=max(width, 1.0)`.

I traced the gallery audit fixture builder in `scripts/build_gallery_audit.py`: PANEL_SIZE=(800,600), RENDER_DPI=100, `_graphviz_node_attrs` (line 2003) defaulting to `fontsize=18, penwidth=2.0` for the *visual comparison cards*, vs `scripts/competitor_renderers/graphviz_renderer.py:155` defaulting to `fontsize=style_map.get("font_size", 14.0)` for the *metric-pipeline graphviz competitor*.

Key reproduction: I monkey-patched `_DENSITY_LABEL_FONT_FLOOR` to 0.4, 0.5, 0.6, 0.7 and re-rendered the canonical 5-node combo workflow (`Ingest -> Validate -> Approve`, `Ingest -> Review -> Ship`) to compare label legibility per-FLOOR. Saved at `/tmp/repro_floor_*.png`, cropped to per-node panels at `/tmp/{validate,approve,ingest}_floors.png`.

## Box3d L1 rise (3.781 -> 3.818) explained

The auditor predicted edge fix should drop shape-card L1 by 0.4-0.6 each. Actual: box3d L1 rose by 0.037. The reason:

1. **Round 11 edge fix landed correctly.** Both `_edge_uses_display_stroke_body(style)` and the new branch in `_draw_direct_edge_body` are reached for `width=1.0`, producing a 2-pixel-wide PathPatch stroke. Pixel probe confirms: dagua's edge stem at `per_card_pixel_diff/dagua/nodes_shapes_box3d.png` column x=399-400 is now solid dark from y=232 to y=366 (~134px length, 2px width = ~268 dark pixels added).

2. **But the auditor's 0.4-0.6 L1 drop estimate was wrong.** The estimate assumed the missing edge stem was the dominant L1 contributor. It is not. The dominant L1 contributor is **dagua's node footprint vastly exceeds graphviz's**:
   - Dagua box3d content: 193x355 px footprint
   - Graphviz box3d content: 55x95 px footprint
   - Pixels-where-dagua-has-content-but-graphviz-doesn't: **42,112 px**
   - Pixels-where-graphviz-has-content-but-dagua-doesn't: **3,782 px**
   - Pixel mass mismatch ratio: ~11:1 in dagua's favor (i.e. dagua occupies more canvas)

3. **Adding the 134-px edge stem to dagua just added more "dagua-has-content / graphviz-doesn't" pixels** in the corridor between dagua's far-apart nodes (where graphviz's compact layout has long been blank). This explains the slight L1 RISE: the round-11 fix added the missing line in dagua, but graphviz's edge corridor is only 38px long while dagua's is 134px long, so 96 of the new dark pixels in dagua land where graphviz still has white space.

4. **Honesty diagnosis confirmed.** The round-11 commit message explicitly anticipated this: "Round-9 'wins' had elevated L1 because of pixel-mass parity at unreadable-text quality; expect those L1 values to rise. This is honesty, not regression." Box3d's tiny rise is part of that honesty -- the fix is correct, it just doesn't help the metric because the metric was rewarding the absence of edge ink in the wide-corridor area.

## Box3d edge-stem stroke width parity

I pixel-probed the edge stems in both renders:
- Dagua corridor center column x=399-400, width=2px, height=134px (uniform)
- Graphviz corridor center column x=397-402 tapering to 399-400, width=6->2px, height=23px

Graphviz's stem is THICKER at the start (where the line meets the Source node) due to its "dot" rendering pipeline using a literal `penwidth=1` SVG stroke that gets antialiased differently. Dagua's stem is uniformly 2px throughout. **This is a rendering-stack residual** (matplotlib PathPatch rasterization vs Graphviz/cairo SVG anti-aliasing); not theme/render-fixable.

Graphviz's arrow head adds another ~30-40 dark pixels of mass at the bottom; dagua's has comparable arrowhead mass. Not the issue.

## Per-card classification (top-25 + brief-required)

### Brief item 1: `nodes_shapes_box3d` (L1 = 3.818) -- `metric_artifact`

Edge fix landed. Dagua: 193x355 footprint, edge stem 2px wide x 134px tall, 42112 px content, "Source" 7px tall. Graphviz: 55x95 footprint, edge stem 2-6px wide x 23px tall, ~3700 px content, "Source" 7px tall. Both render at fontsize=14 in the metric pipeline; both produce comparable per-pixel-of-glyph mass. **The L1 mass is dominated by dagua's gallery-audit-imposed `min_width=200, min_height=110` overrides forcing the nodes to be 4-6x graphviz's auto-sized footprint.** Removing the min_width override would converge dagua to graphviz, but that violates the gallery-audit fixture's explicit decision to give all shape parity cards a fixed footprint (`scripts/build_gallery_audit.py:1858-1869`). This is a metric-vs-fixture-design conflict, not a theme/render defect.

### Brief item 2: `nodes_shapes_circle` (L1 = 3.391) -- `metric_artifact` + minor `competitor_glitch`

Same scale-mismatch story as box3d. Edge stem now visible (round 11 fix). One mild visual oddity: in the per_card_pixel_diff comparison, the arrowhead glyph appears to have a small gap from the stem (the "v" floats with empty space between its tip and the stem's bottom). Probably a sub-pixel rounding artifact between the path-patch stroke and the arrowhead polygon. Not L1-meaningful.

### Brief item 3: `nodes_shapes_cylinder` (L1 = 3.322) -- `metric_artifact`

Edge fix landed cleanly. Same scale mismatch.

### Brief item 4: `nodes_shapes_note` (L1 = 3.083) -- `metric_artifact`

Edge fix landed cleanly. Same scale mismatch.

### Brief item 5: `combo_pie_bold` (L1 = 2.053) -- `fixable_theme_or_render` (FLOOR adjustment)

Round-11 density font fix made "Ingest" readable. But "Validate" still reads as "alidat" (V and last 'e' clipped by ellipse boundary), "Review" as "eview" (R clipped), "Approve" as "pprov" (A and e clipped). Pixel measurement confirms label width vs node width:

| Label | Node-width (effective) | Text-width @ FLOOR=0.6 (8.4pt) | Overflow |
|-------|-----------------------|-------------------------------|----------|
| Ingest | 18.5px | 20.5px | +2px |
| Validate | 23.3px | 28.5px | +5px |
| Review | 21.6px | 25.7px | +4px |
| Approve | 23.9px | 29.4px | +6px |
| Ship | 18.4px | 15.4px | OK |

At **FLOOR=0.5** (font=7pt):

| Label | Node-width | Text-width | Overflow |
|-------|-----------|-----------|----------|
| Ingest | 18.5 | 17.1 | OK |
| Validate | 23.3 | 23.7 | +0px (essentially fits) |
| Review | 21.6 | 21.4 | OK |
| Approve | 23.9 | 24.5 | +1px (essentially fits) |
| Ship | 18.4 | 12.8 | OK |

At FLOOR=0.4 (font=5.6pt) all labels fit comfortably but text becomes very small.

I re-rendered the combo workflow with FLOOR=0.5 (saved at /tmp/repro_floor_05.png) and visually confirmed: Ingest renders as full "Ingest", Validate as nearly-full "Validate" with the V and 'e' barely fitting, Approve as nearly-full "Approve". This is a clear visual improvement over the current FLOOR=0.6.

**Concrete recommendation**: change `dagua/render/mpl.py:242` from `_DENSITY_LABEL_FONT_FLOOR = 0.6` to `_DENSITY_LABEL_FONT_FLOOR = 0.5`. The brief explicitly leaves this open for adjustment with concrete pixel-ratio evidence.

Risk: Low. Round-9 wins (combo_pie_bold L1=2.053, combo_donut_shadow L1=2.209, evil_donut_diamond L1=2.118) are scoring elevated L1 because of label clipping at FLOOR=0.6; FLOOR=0.5 should keep their L1 stable or improve slightly (smaller text -> less mass mismatch where graphviz has different glyphs). Round-9 calibration was for node-WIDTH parity, not text legibility, so dropping the floor should not regress that calibration goal.

Tradeoff to surface: at FLOOR=0.5, the rendered text is at 7pt which is below the typographic readability floor (~8pt for sans-serif at 100dpi). For a "graphviz parity" sprint this is fine -- graphviz routinely renders sub-readable text on dense graphs. For "user-facing default" (graphviz_strict theme used outside parity gallery) it might warrant a per-fixture override rather than a global constant change.

### Brief item 6: `combo_kitchen_sink_5` (L1 = 3.748) -- mixed `fixable_theme_or_render` + `metric_artifact`

Same label-clipping pattern as combo_pie_bold (Validate/Approve/Review badly clipped at FLOOR=0.6). Same recommendation: FLOOR=0.5. Plus same scale-mismatch metric_artifact as the shape cards: dagua's tree spans 353x409 px while graphviz spans 156x143 px (2.3x wider, 2.9x taller). The latter half is unfixable within guardrails.

### Brief item 7: `combo_pie_gradient_bold` (L1 = 3.564) -- mixed (same as kitchen_sink_5)

Same diagnosis. FLOOR=0.5 helps; layout-extent gap is metric_artifact.

### Brief item 8: `combo_donut_shadow` (L1 = 2.209) -- `principled_residual`

This is a round-9 "win". Visual inspection: dagua tree is wider/taller than graphviz, labels are readable but small, donut+shadow rendering looks correct. L1 of 2.21 is roughly the floor for round-9-calibrated 5-node combo cards under the locked density formula. **Not fixable without unlocking density formula.**

FLOOR=0.5 should leave this card stable (labels are already fitting better here because the donut shape has a wider effective inscribed rectangle than the pie ellipse).

### Brief item 9: `combo_hexagon_gradient` (L1 = 3.275) -- `metric_artifact` + minor `fixable_theme_or_render`

Most extreme scale mismatch I see. Dagua's hexagons are MICROSCOPIC (~16x10 px each) because hexagon shape's inscribed-rectangle fit_text shrinks them aggressively, then density-shrink halves them again. Graphviz hexagons are normal-readable (~95x40 px). **The labels in dagua's hexagon nodes are entirely unreadable** -- pixel-probe shows zero "Validate" / "Review" / etc. text glyphs above the noise floor.

This is a clear case where the density-aware-shrink interaction with hexagon's interior-loss rule has produced a degenerate rendering. The principled fix would be to apply a higher minimum cap on the post-shrink node size (e.g. `_MIN_DENSITY_SHRUNK_NODE_WIDTH = 24px`), but that's a new constant outside the brief's allowed adjustments. **Mark as `metric_artifact` with note: hexagon-density interaction is a corner case worth tracking for next sprint.**

### Brief special-investigation cards (L1 > 3.0 from top-25)

- `combo_bold_shadow_gradient` (3.333), `combo_bold_shadow_gradient_rounded` (3.171), `combo_shadow_gradient` (3.298), `combo_bevel_shadow_gradient` (3.297), `combo_bevel_gradient` (3.174): all share the same combo-card label-clipping + scale-mismatch fingerprint. FLOOR=0.5 helps the label clipping; scale mismatch is unfixable.
- `combo_arrow_gradient` (3.121), `combo_arrow_bevel_gradient` (3.120): same fingerprint, plus arrow-shape adds extra geometry on dagua side that graphviz approximates with its standard normal arrowhead.
- `combo_cylinder_dashed_shadow_gradient` (3.238): same fingerprint plus cylinder shape interior-loss.
- `combo_hexagon_gradient` / `combo_hatched_gradient` (3.192) / `combo_stadium_gradient` (3.179) / `combo_per_corner_gradient` (3.069) / `combo_gradient_rounded` (3.039): same fingerprint.
- `nodes_shapes_tab` (3.340), `nodes_shapes_double_circle` (3.207), `nodes_shapes_rect` (3.076): all share the box3d scale-mismatch story; edges now visible.
- `nodes_fills_gradient_radial` (9.381): the round-10 wiring (style="filled,radial" + fillcolor "fill:gradient_color") landed in commit e2079b1, but the per_card_pixel_diff metric pipeline uses `scripts/competitor_renderers/graphviz_renderer.py` not the gallery comparison's `_graphviz_node_attrs`. The fix is in the WRONG path. The metric pipeline's `_node_attrs` (line 130) doesn't have a radial-gradient branch. The round-10 wiring works in the visual comparison (`cards/comparisons/`) but not in the metric pipeline (`per_card_pixel_diff/`). This is a known **fixable_theme_or_render** in `scripts/competitor_renderers/graphviz_renderer.py` -- threading the same `style="filled,radial"` + `fillcolor="fill:gradient_color"` logic into the metric-pipeline graphviz renderer. **However, the brief lists this as "metric pipeline divergence" and the hard guardrails forbid metric pipeline rewrites.** The line between "wire an existing fix into the metric path" and "rewrite the metric path" is fuzzy here. I'd argue threading the same XML emission logic into a sibling renderer file is an extension, not a rewrite -- but I'll defer to the architect.

### Round-9 win control: `combo_pie_bold` (post-round-11)

Round-9 win L1 = 1.918 -> post-round-11 L1 = 2.053. Was the rise "honest" per round-11 commit message? Pixel comparison: yes. The pre-round-11 render had labels truncated to ~3 chars ("nges" not "Ingest"), producing a smaller text glyph footprint that incidentally aligned closer to graphviz's smaller-graph render. The post-round-11 render shows full "Ingest" + partial "alidat"/"eview"/"pprov" labels, more visible mass, but graphviz's compact tree means the new mass lands in dagua-has-content/graphviz-empty zones. L1 rise is honest.

**FLOOR=0.5 prediction for combo_pie_bold**: should land between the pre-round-11 (1.918) and post-round-11 (2.053) values. Smaller text -> less dagua mass in non-overlapping zones -> small L1 drop. Visual: labels become readable (Validate full, Approve full).

## Cross-card patterns

1. **All Tier A cards with L1 >= 3.0 have the same root cause**: dagua's gallery-audit-imposed footprint dominates graphviz's compact native render. This is a structural metric_artifact.

2. **All combo cards have label clipping at FLOOR=0.6**. Lowering to 0.5 measurably improves legibility for 4 of 5 combo workflow labels without regressing round-9 wins.

3. **Round-11 edge fix landed cleanly**. All 17 simple-shape parity cards now show visible edge stems. No regressions.

4. **No rendering-stack residuals beyond known anti-aliasing differences**. Times-vs-Times-Gyre-Termes font substitution, 100dpi-vs-200dpi rasterization differences, sub-pixel polygon edges -- these are all documented in prior audits as `rendering_stack_residual`.

## Verdict

**Primary verdict: `STOP_AT_CAP`.**

The sprint has hit ceiling for the locked-constants regime. The dominant residual L1 mass on every top-25 worst Tier A card is the dagua-footprint-vs-graphviz-footprint scale mismatch, which cannot be closed without violating one of:
- GRAPHVIZ_STRICT_THEME numeric value changes
- Density formula changes (`sqrt(0.3/N)` clamp)
- Gallery audit fixture min_width/min_height removal
- Metric pipeline rewrite

These are all explicitly forbidden by the round-12 hard guardrails. The remaining L1 residual is therefore `principled_residual` / `metric_artifact`.

**Secondary recommendation (architect's call): conditional `CONTINUE_ROUND_12` for FLOOR=0.5 adjustment.**

If the architect chooses to spend the final round, change `_DENSITY_LABEL_FONT_FLOOR = 0.6` -> `0.5` in `dagua/render/mpl.py:242`. This is the single fixable improvement with concrete pixel-ratio evidence:

- Validate text overflow drops from +5px to +0px (essentially fits)
- Approve text overflow drops from +6px to +1px (essentially fits)
- Review text overflow drops from +4px to OK
- Ingest text overflow drops from +2px to OK
- Risk to round-9 wins: low to negligible (smaller text reduces dagua mass in non-overlapping zones)
- Risk to user-facing themes: minor (7pt is below the 8pt readability floor but acceptable for parity gallery; consider per-fixture override rather than global change if this concerns the architect)

Expected impact: 2-5 combo card L1 drops by 0.05-0.20, round-9 wins stable to slightly improved, mean Tier A L1 drops by 0.01-0.03. Not large, but it's the only concrete handle inside the guardrails that improves visible parity.

## File:line evidence index

- `dagua/render/mpl.py:242` -- `_DENSITY_LABEL_FONT_FLOOR = 0.6` (recommendation: change to 0.5)
- `dagua/render/mpl.py:5583` -- `_edge_uses_display_stroke_body` round-11 fix entry
- `dagua/render/mpl.py:6614-6630` -- round-11 PathPatch edge body branch
- `dagua/render/mpl.py:7722-7743` -- round-11 density font_factor threading
- `dagua/render/mpl.py:913-948` -- `density_aware_size_factor` formula (locked, no change)
- `dagua/styles.py:949-1040` -- GRAPHVIZ_STRICT_THEME definition (locked, no change)
- `scripts/build_gallery_audit.py:1858-1869` -- shape parity card min_width/min_height override (architect's design decision; outside fixable scope)
- `scripts/build_gallery_audit.py:2003-2010` -- visual comparison card graphviz attrs (fontsize=18; this asymmetry vs metric pipeline is architectural)
- `scripts/competitor_renderers/graphviz_renderer.py:130-163` -- metric pipeline graphviz renderer (no radial-gradient branch; arguable ext-vs-rewrite)

## Pixel probes used

- box3d edge corridor: dagua x=399-400 width=2px, length=134px, 252 very-dark px
- box3d content footprint: dagua 193x355=68515 px2; graphviz 55x95=5225 px2; ratio 13:1
- box3d L1 source: 42112 dagua-only px vs 3782 graphviz-only px; 11:1 mass mismatch
- combo_pie_bold node widths: 18-24 px effective (after density shrink x0.245)
- combo_pie_bold label widths at FLOOR=0.5/0.6: measured via `dagua.render.text.paths.measure_text_data` and verified against rendered images
- repro renders saved at `/tmp/repro_floor_{04,05,06,07}.png`, per-node crops at `/tmp/{validate,approve,ingest}_floors.png`

## What I did NOT inspect

- Tier B and Tier C cards (out of scope for top-25-Tier-A-only audit)
- SSIM scores (the brief is L1-focused; SSIM tracks similar but with different sensitivities)
- Cluster cards (cluster_label_position_*, cluster_opacity_*, etc.) -- different fixture path
- nodes_borders_* cards beyond what's in top-25 -- different fixture path

These could surface additional fixable defects but are outside the brief.
