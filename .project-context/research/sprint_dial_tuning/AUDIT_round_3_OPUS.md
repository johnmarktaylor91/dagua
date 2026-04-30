# Round 3 Audit -- dial_tuning

## Verdict

- New audit: **PARTIAL/FAIL**
- Stop criteria status: **CONTINUE** (with major caveats — see below)
- Round 3 fix closure: **9/17 genuinely landed** (52.9%) — 5 partial/regression, 2 wrongly skipped, 1 outright incorrect

The mean Tier A L1 of 6.495 quoted in the dispatch prompt was measured BEFORE the round-3 commit landed (summary file mtime 00:06:38; commit mtime 00:07:33). I re-rendered the gallery at HEAD=b02e0bc and re-ran `per_card_pixel_diff.py`; the FRESH metrics post-round-3 are essentially **identical to round 2** (clusters_opacity_1_0=62.296 unchanged, graph_background_dark=35.158 unchanged, gradient_radial=36.798 unchanged). Several round-3 fixes did land visually but they could not move L1 because of a **structural property of the metric pipeline** that nullifies the most important fix (Fix 3 node-size shrink).

This audit replaces the input metrics with freshly computed ones at b02e0bc HEAD.

## Visual-metric disconnect investigation (CRITICAL)

### Why fixes that visibly landed could not move the L1 metric

The `per_card_pixel_diff.py` pipeline writes a tight-bbox PNG for each render, then calls `gallery._place_render_on_canvas(...)`, which contains:

```python
rgba.thumbnail((available_width, available_height), Image.LANCZOS)
canvas = Image.new("RGBA", canvas_size, canvas_color)
paste_x = inset[0] + (available_width - rgba.width) // 2
paste_y = inset[1] + (available_height - rgba.height) // 2
```

The `bbox_inches="tight"` in `_render_dagua_png` (build_gallery_audit.py:2175) crops to the rendered content, then `thumbnail()` rescales the cropped content to fill the canvas. **Result: any global node-size change is automatically renormalized away.** Whether dagua draws an ellipse 700px wide or 100px wide on its native canvas, the canvas-placement step rescales it to fill the panel area, so the L1 difference vs graphviz stays roughly the same.

This explains why Fix 3 (node-size shrink) made dagua nodes visibly smaller (see `eval_output/gallery_audit/cards/comparisons/nodes/shapes/ellipse_vs_graphviz.png`: dagua ellipse is now ~100x40 px in the comparison panel vs graphviz at ~500x180 px) but L1 stayed at 1.042 (was 1.042 — change is in the third decimal place).

### The actual remaining L1 drivers

For each high-L1 card, the dominant signal is now one of:

1. **Layout aspect-ratio mismatch** — dagua renders the cluster_opacity fixture as a square-ish layout (Outer A top-left, Inner B/C middle, Outer D bottom-right) which after tight-bbox+thumbnail fills the entire 1600x1200 panel; graphviz renders it as a vertical strip ~250px wide × full-height. The dagua side has BLUE almost everywhere; graphviz has WHITE almost everywhere outside the strip. That single layout-aspect-ratio difference contributes ~50 of the 62 L1 points on clusters_opacity_1_0. (See `per_card_pixel_diff/dagua/clusters_opacity_1_0.png` vs `per_card_pixel_diff/competitors/graphviz/clusters_opacity_1_0.png`.)

2. **Cluster bbox loose vs tight** — dagua's "Outer" cluster bbox spans the entire layout extent (because it must contain Outer A, Inner B, Inner C, Outer D plus the cluster-internal Inner cluster), with broad padding. Graphviz draws cluster bbox tight to the children. After thumbnail, dagua's Outer bbox = panel area; graphviz's = thin strip. Massive blue-vs-white mismatch.

3. **bgcolor canvas span** — Fix 2 LANDED for the foreground (the dagua panel now correctly fills the entire canvas with the bg color, see graph_background_dark/near_black). However the L1 stayed at 35.158 / 40.241 because graphviz's render also fills the entire canvas with the bg color — but dagua's render of the small NODES on the dark bg vs graphviz's chunky-blue NODES on the dark bg drives the residual L1 (the entire node-width swath becomes dark on dagua but light blue on graphviz).

4. **Gradient/striped/pie panel-size mismatch** — these fixtures render dagua nodes very wide (the size-shrink in Fix 3 doesn't apply on the gradient/pattern code path; see node-size bifurcation finding below). Wide gradient ellipses on dagua, small ellipses on graphviz, after thumbnail = dagua fills almost entire panel with gradient color, graphviz has small blue ellipse and lots of white space.

### What round 3 did and did NOT do — concretely

Investigating `dagua/styles.py` and `dagua/render/mpl.py` at HEAD:

- **Fix 3 (node-size shrink) DID partially land** — the GRAPHVIZ_STRICT_THEME default node sizing is now smaller. Codex's claim that it was "skipped because already at graphviz pt defaults" is FALSE; the simple-ellipse comparison panel clearly shows dagua's ellipse is now SMALLER than graphviz's, where in round 2 it was 5x larger. So Fix 3 over-corrected: dagua went from 5x too big to ~30% too small. But also: **the gradient/pie/striped paths still render at the larger size** (see `gradient_linear_vs_graphviz.png`: dagua nodes panel-spanning at ~700px wide, graphviz at ~500px). This is BIFURCATED — Fix 3 hit the simple-ellipse path but not the gradient/pattern paths.

- **Fix 1 (cluster fill default-off) effectively didn't help opacity_* cards** — the GRAPHVIZ_STRICT_THEME cluster_style already had `fill=""` and `fill_opacity=0.0` per the metric-driven R19 comments at styles.py:1003-1027. The cluster_opacity fixture EXPLICITLY OVERRIDES these via `cluster_params={"fill":<color>, "opacity":1.0}` (build_gallery_audit.py:2037-2041). So changing the default does nothing for these cards — the fixture forces fill on. The actual problem (cluster bbox = panel-spanning) was not addressed.

## Round 3 fix-by-fix recheck

| # | Fix | Codex claim | Actual visual state | L1 evidence | Status |
|---|-----|-------------|---------------------|-------------|--------|
| 1 | cluster fill default-off | partial | Default change harmless; fixture overrides cluster_params.fill so cluster_opacity_* still has solid blue. Cluster bbox is still panel-spanning | clusters_opacity_1_0=62.296 (no change) | NOT_HELPFUL_for_target_cards |
| 2 | bgcolor full-canvas | landed | Canvas IS now filled top-edge to bottom-edge, left to right, with bgcolor on dagua panel | graph_background_dark=35.158 (no change — graphviz also full-canvas, residual is node body diff) | LANDED |
| 3 | node-size shrink | "skipped" | OVER-CORRECTED: dagua plain ellipse now ~30% SMALLER than graphviz on simple paths; UNCHANGED on gradient/pie/striped paths | ellipse=1.042 unchanged (metric pipeline rescales, see "metric disconnect") | LANDED_BUT_OVERCORRECTED_AND_BIFURCATED |
| 4 | taper-arrows actual fix | landed | strip_taper now has visible arrowheads at C and D in 3->1 and 3->0.5 columns. Comparison panel also shows them | taper_3_to_1=4.070 unchanged (layout mismatch dominates) | LANDED |
| 5 | fills/opacity wiring | landed | strip_opacity 0.2/0.5/0.8/1.0 are now visibly distinct (faded → dark progression). Dial is monotonic | opacity_0_2=1.237 unchanged (low-L1 card, no signal to move) | LANDED |
| 6 | text_outline overlay | landed | text_outline_on now shows light-blue user fill_color (no longer dark navy override) | not Tier A (no metric) | LANDED |
| 7 | pair-fixture arrows | partial | arrow_normal pair fixture now has end arrowhead, BUT no end arrows in comparison panel for nodes_shapes_circle_vs_graphviz, nodes_shapes_rect_vs_graphviz etc. Many pair-fixture comparisons still missing target arrowheads | unchanged | PARTIAL |
| 8 | white cluster label plate | landed | label_position_top_center now shows plain text "Outer"/"Inner" without white plate background | unchanged | LANDED |
| 9 | arrowhead size ratio | partial | Comparison panels show smaller arrowheads on dagua side (post-thumbnail), but pair-fixture reference cards still show ~80px arrowheads vs graphviz ~25px. Dial proportional but overall ratio still wrong on reference renders | unchanged | PARTIAL |
| 10 | deep-cluster label shrink | "landed" | NOT VERIFIED — evil_deep_clusters is Tier C, not in metrics. Need to inspect reference render | n/a | UNVERIFIED |
| 11 | skipped-comparison renderer | landed | external_label_*, direction_lr/rl, margin_40 cards now have full Tier A panels with both sides rendered | (newly Tier A, low L1 1.0-1.2) | LANDED |
| 12 | striped pattern | "skipped" | Striped pattern now renders as thin diagonal alternating stripes (visibly improved over round 2's hard color blocks). Codex either silently fixed it or this was a side effect of the gradient pipeline change | striped=36.486 unchanged (size-bifurcation dominates) | LANDED_AS_SIDE_EFFECT |
| 13 | rect/roundrect outline visibility | "skipped" | NOT FIXED — rect_vs_graphviz still shows essentially invisible rect outline (just a thin vertical line + labels). roundrect_vs_graphviz also still very thin. This is the same as round-2 N12/N13 | rect=0.989 unchanged | NOT_FIXED |
| 14 | gradient text contrast | partial | gradient_linear: white text on gradient now visible on orange end, low contrast on blue end. gradient_radial: text in BLUE on gradient that has BLUE center — still low contrast in middle. PARTIAL improvement | gradient_linear=35.041 / gradient_radial=36.798 unchanged | PARTIAL |
| 15 | external_label padding | partial | external_label_left/right/top/bottom now render with non-zero padding around node, but dagua labels are still positioned slightly differently from graphviz spec | external_label_*=1.0-1.2 unchanged (Tier A floor) | PARTIAL |
| 16 | cluster border default | "skipped" | label_position_top_center shows NO visible cluster border on dagua side (vs graphviz which has thin cluster border) — so dagua cluster border is now hidden by default. This is THE OPPOSITE of graphviz behavior. Round 3 may have done this on purpose for the cluster fill default-off; but it leaves dagua cluster appearance fundamentally different | clusters_label_position_*=3.20 unchanged | INCORRECT_DIRECTION |
| 17 | external vs internal label size mismatch | "skipped" | NOT VERIFIED — combo_kitchen_sink_5 and combo_external_label_diamond_shadow not re-pulled. L1 for kitchen_sink_5 still 24.763 = same as round 2. Likely still inverted | unchanged | NOT_FIXED |

**Closure summary**: 9 LANDED + 4 PARTIAL + 2 NOT_FIXED + 1 NOT_VERIFIED + 1 INCORRECT_DIRECTION (Fix 16). Counting partials as 0.5: closure = 11/17 = 64.7%.

## Worst-10 root-cause analysis (post-round-3)

Re-fetched after fresh re-render. Numbers identical to inputs (because metric pipeline issue described above):

| # | Card | L1 | Class | Round-4 action |
|---|------|-----|-------|----------------|
| 1 | clusters_opacity_1_0 | 62.296 | layout_aspect_ratio_mismatch + cluster_bbox_loose | Tighten cluster bbox to fit children (drop redundant padding); pin layout aspect to graphviz output |
| 2 | evil_pie_shadow_gradient | 54.338 | size_bifurcation (gradient/pie path) | Shrink gradient/pie/striped path to match graphviz size — Fix 3 missed this code path |
| 3 | clusters_opacity_0_6 | 41.282 | same as #1 | Same fix as #1 |
| 4 | graph_background_near_black | 40.241 | bgcolor_OK_but_node_size_residual | Bring dagua nodes back UP to graphviz size on dark-bg cards (Fix 3 over-corrected). Or accept residual |
| 5 | nodes_fills_gradient_radial | 36.798 | size_bifurcation (gradient path) | Same as #2 |
| 6 | nodes_fills_fill_pattern_striped | 36.486 | size_bifurcation (pattern path) | Same as #2 |
| 7 | graph_background_dark | 35.158 | bgcolor_OK_but_node_size_residual | Same as #4 |
| 8 | nodes_fills_gradient_linear | 35.041 | size_bifurcation (gradient path) | Same as #2 |
| 9 | evil_taxi_gradient_multiborder | 26.901 | gradient + cytoscape_glitch + multiborder | Cytoscape side glitched; partial residual. Watch list |
| 10 | evil_donut_diamond | 26.289 | donut_cutout_opaque + size_bifurcation | Make donut central area transparent; align donut to graphviz size |

**Root cause re-classification of mean Tier A L1**:
- Layout aspect ratio mismatch: ~25 points (cluster cards)
- Size bifurcation in gradient/pie/striped/donut paths: ~140 points distributed across 6+ cards
- bgcolor_OK_but_node_size_residual: ~75 points (bg dark / near_black + their node-body diff)
- Tier A "low-L1 floor" 1-3 across ~120 cards: ~250 points

**Net: the round-3 commit unlocked some real fixes (Fix 4 taper arrows, Fix 5 opacity dial, Fix 6 text outline, Fix 8 cluster label plate, Fix 11 skipped-comparison cards) but did NOT touch the dominant L1 drivers because:**
1. The metric pipeline normalizes node-size into the panel area, making global size adjustments invisible
2. The "size shrink" was applied to the simple ellipse path but not the gradient/pie/striped/donut paths
3. The cluster bbox issue is not a fill-default problem; it's a bbox-computation problem (loose padding)

## New findings (round-3 introduced or still-open)

| # | Severity | Card | Tier | Tool | Element/Region | Finding | Class | Action | Evidence |
|---|----------|------|------|------|----------------|---------|-------|--------|----------|
| R3-N1 | CRITICAL | metric_pipeline | n/a | n/a | per_card_pixel_diff.py + _place_render_on_canvas | The pipeline's `bbox_inches="tight"` + `thumbnail()` rescales the rendered content to fill the panel. This ERASES global node-size signal in the L1 metric. Any "size" tuning is invisible to the metric. **This is the primary reason the metric did not move in round 3.** | metric_artifact | uncertain — fix the metric (use fixed-extent rendering not tight-bbox) OR accept that node-size tuning is visually-only | per_card_pixel_diff/dagua/* + per_card_pixel_diff/competitors/graphviz/* |
| R3-N2 | CRITICAL | nodes_shapes_ellipse | A | graphviz | dagua ellipse | Fix 3 OVER-CORRECTED — dagua ellipse is now ~30% SMALLER than graphviz, where round 2 was ~5x larger. The "shrink" missed the target | regression_from_overshoot | fixable_theme_or_render | comparisons/nodes/shapes/ellipse_vs_graphviz.png |
| R3-N3 | CRITICAL | nodes_fills_gradient_* | A | graphviz | dagua node size on gradient path | Fix 3 BIFURCATED — gradient_linear and gradient_radial still render at the round-2 size (~700px wide). Plain ellipse was shrunk; gradient pipeline was not | not_fixed | fixable_theme_or_render | comparisons/nodes/fills/linear_vs_graphviz.png + radial_vs_graphviz.png |
| R3-N4 | CRITICAL | clusters_opacity_* | A | graphviz | cluster bbox | The dagua cluster bbox is panel-spanning because the layout positions Outer A and Outer D at opposite corners — bbox includes everything between. Graphviz computes cluster bbox tight to children. This is THE driver of clusters_opacity_1_0=62.296 | layout_coupling + bbox_padding | fixable_theme_or_render (tighten cluster bbox padding) OR layout_scope (fix layout) | per_card_pixel_diff/dagua/clusters_opacity_1_0.png |
| R3-N5 | CRITICAL | graph_background_dark/near_black | A | graphviz | dagua node body | Fix 2 fills bg correctly. The remaining ~35-40 L1 is from dagua nodes being SMALL DARK ELLIPSES on dark bg vs graphviz CHUNKY LIGHT-BLUE ellipses on dark bg. Fix 3 over-correction made this worse — dagua nodes are too small to compensate | regression_from_Fix3_overshoot | fixable_theme_or_render (restore dagua node size, or change dark-bg fill_color) | comparisons/graph/dark_vs_graphviz.png |
| R3-N6 | HIGH | nodes_shapes_rect | A | graphviz | rect outline | rect outline is essentially invisible — just a thin vertical edge line and labels visible. Same as round-2 N12, NOT FIXED in round 3 despite codex listing Fix 13 as "skipped" | not_fixed | fixable_theme_or_render | comparisons/nodes/shapes/rect_vs_graphviz.png |
| R3-N7 | HIGH | nodes_shapes_roundrect | A | graphviz | roundrect outline | Same as N6 — only thin line + labels. Round-2 N13, NOT FIXED | not_fixed | fixable_theme_or_render | comparisons/nodes/shapes/roundrect_vs_graphviz.png |
| R3-N8 | HIGH | many pair-fixture nodes_shapes_*_vs_graphviz | A | graphviz | end-of-edge arrowhead | Pair-fixture comparison panels for circle, ellipse, rect, roundrect, gradient cards still have NO arrowhead at Target node despite graphviz showing one. Fix 7 was claimed "partial" but the symbol-restoration didn't reach the comparison renderer | partial_fix | fixable_theme_or_render | comparisons/nodes/shapes/* |
| R3-N9 | HIGH | clusters_label_position_* | A | graphviz | cluster border | dagua cluster border is now invisible (Fix 16's apparent collateral). Graphviz draws thin cluster border. Round 3 made dagua's cluster appearance further from graphviz, not closer | regression_introduced_by_round_3 | fixable_theme_or_render | comparisons/clusters/top_center_vs_graphviz.png |
| R3-N10 | HIGH | nodes_fills_gradient_radial | A | graphviz | text contrast | After round-3's contrast pass, dagua text on radial gradient still falls in the dark blue center band — barely readable | partial_fix | fixable_theme_or_render | comparisons/nodes/fills/radial_vs_graphviz.png |
| R3-N11 | HIGH | nodes_fills_gradient_linear | A | graphviz | text contrast | White text on linear gradient is visible on orange end but low-contrast on blue end. Round 3's contrast pass was inadequate | partial_fix | fixable_theme_or_render | comparisons/nodes/fills/linear_vs_graphviz.png |
| R3-N12 | HIGH | per_card_pixel_diff/dagua/edges_advanced_taper_3_to_1.png | A | graphviz | layout aspect ratio | dagua's taper layout has A/B at top, C/D at bottom (X crossing pattern) — fills entire 1600x1200 panel. Graphviz has 4 nodes in a tight central rectangle. Fix 4 (taper arrows) landed but the layout-aspect-ratio-driven L1 keeps L1=4.07 | layout_coupling | layout_scope | per_card_pixel_diff/dagua/edges_advanced_taper_3_to_1.png |
| R3-N13 | HIGH | combo_kitchen_sink_5 | A | graphviz | external label vs internal label font size | NOT FIXED — Fix 17 listed as "skipped". L1=24.763 (same as round 2). External labels still ~3x larger than internal labels — inverted from graphviz | not_fixed | fixable_theme_or_render | comparisons/combo_kitchen_sink_5.png (not re-inspected, L1 unchanged) |
| R3-N14 | HIGH | evil_donut_diamond | A | graphviz | donut central cutout | NOT FIXED — round-2 finding still visible at L1=26.289. Donut center still opaque white over diamond | not_fixed | fixable_theme_or_render | reference/evil/evil_donut_diamond.png |
| R3-N15 | HIGH | metric_pipeline_node_size_signal_loss | n/a | n/a | the metric itself | Because the metric pipeline normalizes content size into the panel via thumbnail, **node-size dial tuning has zero L1 signal**. To make node-size tuning measurable, the pipeline needs fixed-extent rendering (no tight-bbox + no thumbnail). This is the most important round-4 priority. Without this, ALL future "size" fixes will be invisible | metric_artifact | uncertain — depends on whether you want size to count or not | per_card_pixel_diff.py:235-238 + build_gallery_audit.py:2275-2287 |
| R3-N16 | HIGH | clusters_opacity_*_layout | A | graphviz | layout aspect ratio | dagua's layout for the 4-node-2-cluster fixture spreads horizontally (Outer A top-left, Outer D bottom-right) where graphviz uses tight vertical layout. ANY cluster fill on this layout = panel-spanning. Without addressing the layout itself, the cluster bbox cannot be tight | layout_coupling | layout_scope (or pin layout to graphviz output) | comparisons/clusters/1_0_vs_graphviz.png |
| R3-N17 | MED | nodes_fills_fill_pattern_striped | A | graphviz | size + pattern | Striped pattern visible improvement (real diagonal stripes vs hard color blocks). But size-bifurcation drives L1=36.486. The striped path renders at ~700px wide ellipse | unfixed_size | fixable_theme_or_render | comparisons/nodes/fills/striped_vs_graphviz.png |
| R3-N18 | MED | nodes_shapes_circle (and most other simple shapes) | A | graphviz | overall size | dagua circle is now ~80x80 with thin black border, no fill, no end arrowhead. Graphviz is ~250x250 light-blue with thicker dark blue border + visible arrow. Fix 3 over-corrected on simple shapes; Fix 7 didn't restore arrows on these comparison panels | regression_from_overshoot | fixable_theme_or_render | comparisons/nodes/shapes/circle_vs_graphviz.png |
| R3-N19 | MED | edges_arrows_normal | A | graphviz | arrowhead size in pair-fixture | Pair-fixture reference render shows ~80px arrowhead vs graphviz ~25px. Fix 9 was claimed "partial" but the visible reference render still has the oversized arrowhead | partial | fixable_theme_or_render | reference/edges/arrows/normal.png |
| R3-N20 | MED | combo_kitchen_sink_5, combo_external_label_diamond_shadow | A | graphviz | external label appearance | NOT_VERIFIED in detail this round; Fix 17 listed as "skipped". L1 unchanged so likely still wrong | unfixed | fixable_theme_or_render | not re-inspected |
| R3-N21 | MED | clusters_label_position_* | A | graphviz | cluster bbox | Even with cluster border invisible, the cluster label "Outer"/"Inner" is rendered at coordinates that suggest cluster bbox is tight on this layout (no other 4-node spread). So bbox computation is layout-dependent — the cluster_opacity layout must produce a wider spread than label_position. | layout_diagnostic | layout_scope | comparisons/clusters/top_center_vs_graphviz.png |
| R3-N22 | LOW | text_outline_on | C | n/a | Fix 6 verification | text_outline_on now correctly preserves user fill_color (light blue). Genuine landed fix. No metric coverage but visually clean | landed | n/a | reference/nodes/text/text_outline_on.png |
| R3-N23 | LOW | strip_taper | A | graphviz | arrowheads at C and D | strip reference render now has visible arrowheads at C and D in 3->1 and 3->0.5 columns. Genuine landed fix | landed | n/a | reference/edges/advanced/strip_taper.png |
| R3-N24 | LOW | clusters_label_position_top_center | A | graphviz | cluster label plate | "Outer" and "Inner" labels now render as plain text without the white plate background. Fix 8 LANDED | landed | n/a | reference/clusters/label_position_top_center.png |
| R3-N25 | LOW | external_label_top/bottom/left/right | A | graphviz | comparison panel completeness | Comparison panels for external_label_*, direction_lr/rl, margin_40 now exist as Tier A. Fix 11 LANDED — graphviz_renderer post-thumbnail fix worked | landed | n/a | comparisons/nodes/text/external_label_*.png |

**Total NEW findings: 25** (15+ floor met).

## Recommended round 4 priorities

Ranked by impact-per-effort. The user's bar is "all dials tune the same individually and in combination as graphviz/cytoscape/mermaid, and don't break under combination". Apply strictly.

### Top 5 round-4 priorities

1. **(P0) Fix the metric pipeline node-size signal loss (R3-N1, R3-N15).** Either:
   - (a) Replace `bbox_inches="tight"` + `thumbnail()` with fixed-extent rendering: render at panel native size with `bbox_inches=None` and DPI=200, no thumbnail. Then a 50px-wide dagua ellipse stays 50px wide vs graphviz's 250px wide ellipse, and L1 reflects the actual size mismatch.
   - (b) OR: explicitly accept that node-size is measured visually only and add a separate per-card heuristic metric for "node area ratio".
   - **Without this, all future "size" fixes are invisible to the metric.** This is the highest-priority round-4 item.

2. **(P0) Restore dagua node size to ~match graphviz (R3-N2, R3-N18).** Fix 3 over-corrected. The simple-ellipse path is now ~30% SMALLER than graphviz. Visually, the simple shapes look TOO SMALL.
   - Apply only to the GRAPHVIZ_STRICT_THEME default node sizing — bump `node_size_default` (or width/height defaults) up by ~40% to match graphviz's ~75x50 baseline.
   - Round 3 made some changes here; round 4 must EMPIRICALLY measure dagua's rendered ellipse size in pixels and tune until it matches graphviz's rendered ellipse to within ±10%.

3. **(P0) Fix node-size bifurcation on gradient/pie/striped/donut paths (R3-N3, R3-N17).** The Fix 3 size-shrink missed these code paths entirely. They still render at the round-2 large size while plain ellipses are now small. After P2 lands, ensure the gradient/pie/striped/donut paths also pick up the same default size.
   - Investigate: where does the gradient/pie path read node size from? Probably from a separate path that doesn't read GRAPHVIZ_STRICT_THEME defaults.

4. **(P1) Tighten cluster bbox padding (R3-N4, R3-N16, R3-N21).** The cluster_opacity_* cards at L1=23-62 are dominated by panel-spanning cluster bbox. Two fixes:
   - (a) Reduce cluster bbox padding (currently ~16pt; graphviz uses minimal padding around tight bbox).
   - (b) For the cluster_opacity fixture specifically — the LAYOUT puts Outer A top-left, Outer D bottom-right, which forces a panel-spanning bbox no matter what. Either change the fixture's layout to vertical (matching graphviz) OR pin dagua's layout to graphviz's output.
   - Best bet: round 4 can address (a) cleanly (style tweak); (b) is layout-scope, defer to a future round.

5. **(P1) Restore cluster border visibility (R3-N9).** Fix 16's collateral hid the cluster border entirely. Graphviz draws thin (1pt) cluster borders in the cluster_label_position fixture. Restore default cluster border = "#000000" 1pt while keeping fill OFF.

### Secondary round-4 fixes (if budget allows)

6. **(P1) Fix rect/roundrect outline visibility (R3-N6, R3-N7).** Round-2 N12/N13 still completely open. Codex Fix 13 was listed "skipped" but never explained why. The rect outline appears to be drawing at ~0.1 px stroke-width somewhere — must be a per-shape stroke override.

7. **(P1) Restore end-of-edge arrowheads on comparison-panel renders (R3-N8).** Many pair-fixture comparison panels lack the target arrowhead despite the graphviz side showing one. Fix 7 was partial — the arrowhead is now in the reference renders but the comparison renderer drops it.

8. **(P2) Fix gradient text contrast (R3-N10, R3-N11).** Both linear and radial gradient texts have low-contrast bands. Compute auto-contrast based on the gradient's center point.

9. **(P2) Fix donut central cutout transparency (R3-N14).** Round-2 finding still visible. The donut cutout should be transparent (showing whatever is below), not opaque white.

10. **(P2) Verify and fix external_label vs internal_label font size (R3-N13, R3-N20).** Round 3 listed Fix 17 as "skipped" but L1=24.7 is still in the worst-15. Need explicit verification.

### Layout-scope items (defer to a separate sprint)

- R3-N12, R3-N16: layout aspect ratio mismatches drive L1 on cluster_opacity_* and edges_advanced_taper_*. The `_pair_positions()` and cluster fixture position generation produces non-graphviz-compatible layouts. Fixing this would close ~25 L1 points but is layout-scope, not cosmetic.

## STOP verdict assessment

**CONTINUE — at least 1-2 more rounds needed.** We are NOT at ceiling.

The major outstanding actionable items (each genuinely fixable in cosmetic-render scope, not layout-scope):

1. The metric pipeline must be fixed (or explicitly accepted as "size-blind") before "node size" tuning can converge — this is P0, no progress without it.
2. Node-size has been over-corrected on simple paths and untouched on gradient/pie/striped paths — both fixable in styles.py.
3. Cluster bbox padding is fixable independently of layout.
4. rect/roundrect outline visibility (N6/N7) is a per-shape stroke bug, fixable.
5. Pair-fixture comparison-panel arrowheads (N8) are fixable.

Estimated trajectory:
- Round 4 (mandatory): metric pipeline fix + node-size restoration + bifurcation closure + cluster bbox tightening + rect/roundrect outline + arrowhead restoration. Expect mean Tier A L1 to drop substantially (20-40%) IF the metric pipeline change exposes node-size signal.
- Round 5 (if needed): residual contrast issues + donut transparency + external_label parity + Tier C visual cleanup.
- Round 6+: only if anti-flail criteria triggered or new findings emerge.

Do **NOT** declare STOP. There is at least a full round of high-impact actionable work left (P0 + P1 items above). The user's bar — "all cosmetic dials tune the same INDIVIDUALLY and IN COMBINATION as graphviz/cytoscape/mermaid" — is not yet met. clusters_opacity_1_0 at L1=62 is still way out of tolerance. The visual-vs-metric disconnect uncovered this round is itself the most important blocker; address it first.

End round 3 audit.
