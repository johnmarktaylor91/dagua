# Round 2 Audit -- dial_tuning

## Verdict

- New audit: FAIL
- Stop criteria status: CONTINUE
- Round 1 finding closure rate: 11 / 28 fixable + 13 / 47 total
  - CLOSED: 13
  - PARTIAL: 7
  - OPEN: 19
  - REGRESSION: 3
  - NOT_ATTEMPTED (out of round-2 scope, design decision required, or layout scope): 5

The single biggest finding from round 1 -- "default node size 3-5x too large" (#2) -- is OPEN: there has been no shrink of dagua's default node footprint. The "white label-bg box" (#3) is genuinely closed -- this is the standout win and explains many of the L1 improvements on gradient/pie cards. But three round 2 fixes introduced REGRESSIONS that show up as enormous L1 spikes:

1. cluster fill is now WIRED but it dominates the whole panel as a near-canvas-spanning solid blue, while graphviz emits no cluster fill at all -> clusters_opacity_1_0 L1 64.105.
2. graph_background_dark/near_black draw the bg color only inside a TIGHT BBOX around the nodes, not the whole canvas -> graph_background_dark/near_black L1 35-40.
3. external_label position dial works for atomic cards (#18 closed) BUT the external label glyphs render at NORMAL font size while dagua's default node label is dramatically larger -- the external labels in combo cards now render LARGER than the node labels.

Plus three round 1 OPEN issues that round 2 attempted but did NOT fix:
- taper still kills arrowheads (atomic strip and taper_3_to_1 vs_graphviz). The codex commit message claims this was fixed; it was not.
- nodes/fills/opacity dial still has no observable effect on the node body.
- evil_deep_clusters labels still HUGE (Level 1, Level 2, Level 3 labels each rendered in white plates with bold fonts taking ~25% of cluster width).

The user explicitly asked for cooking until ceiling, so the verdict is CONTINUE. There is at least one more round of fixable cosmetic work, then the residual node-size / cluster-fill / dark-bg-canvas-fill items become the dominant remaining levers. A round 3 spec focused on (a) shrinking default node size to match graphviz, (b) canvas-vs-tight-bbox bg, (c) cluster-fill-suppression as default, and (d) actually fixing taper-kills-arrows, would close most of the post-round-2 deltas.

## Worst-systemic issues remaining (round 2 ranking)

1. **Default node size still 3-5x too large** (carry-over of round-1 #2). Visible in every nodes_shapes_* and nodes_borders_* graphviz comparison. Until this is fixed, every Tier A panel L1 has a baked-in size penalty unrelated to the dial under test.

2. **Cluster fill is wired now (dial monotonic) BUT cluster fill is rendered as an almost-canvas-spanning solid block.** Graphviz emits no fill for a cluster (only border + label), so dagua's canvas-spanning blue makes opacity_1_0 the SINGLE WORST card in the gallery (L1 64.105). This is the largest measurable regression introduced by round 2. Either (a) make cluster fill default OFF (graphviz parity) OR (b) clamp the cluster fill region to the cluster bounding box, not the whole canvas.

3. **graph_background bgcolor draws into a tight bbox, not the canvas.** dark/near_black both fill ONLY a tall narrow rectangle bounding the three Stage A/B/C nodes; everything outside the rect is white. Graphviz fills the entire canvas with bgcolor. Net L1 35-40.

4. **Taper still kills arrowheads.** strip_taper Off-column has arrowheads, 3->1 and 3->0.5 columns DO NOT. taper_3_to_1_vs_graphviz also confirms this. Round-2 codex commit claimed "Fixed taper kills arrows"; that did not land. The dashed-survives-taper IS fixed (visible in evil_taper_crossing_dashed) -- so the underlying refactor partially worked, but specifically the arrowhead skip-on-taper is still in the renderer.

5. **nodes/fills/opacity dial still has zero observable effect.** Strip card 0.2/0.5/0.8/1.0 are visually identical. fills_opacity_0_2 vs fills_opacity_1_0 graphviz comparisons are also flat. Most likely root cause: the fill alpha is being routed through a path that never reaches the rendered ellipse fill; the ellipse is drawn with opacity=1.0 hardcoded somewhere in the shape painter.

6. **External label font size is decoupled from main node label font size.** In atomic cards external_label is small; in combo cards (combo_kitchen_sink_5, combo_external_label_*) external label glyphs ("v1.2", "stable", "beta", "new", "legacy") render in ~30 px bold font while the node's INTERNAL label ("Ingest", "Validate", etc.) renders at ~10 px italic. The external is ~3x larger than internal -- exactly inverted from graphviz.

7. **bevel_on now correctly preserves user fill_color (round-1 #22 CLOSED)** -- the overlay-not-replacement strategy works. **However text_outline_on still appears to OVERWRITE the user's fill_color** (the rendered fill is dark navy where the user requested light blue). Either the text_outline overlay is being painted full-coverage, or text_outline is going down a different code path than bevel.

8. **evil_extreme_curvature edge wraps off-canvas (carry-over #25, layout-scope).** Curve A and Curve B at extreme corners; the connecting edge segment goes off the top of the panel and reappears at bottom-right. Layout/cosmetic coupling at extreme curvature values.

## Round 1 finding recheck (all 47)

| # | Round 1 finding | Round 2 status | Evidence |
|---|---|---|---|
| 1 | rect borders invisible | OPEN | comparisons/nodes_shapes_rect_vs_graphviz.png -- still no visible rect outline; just labels and a thin vertical edge |
| 2 | default node size 3-5x too large | OPEN | every nodes_shapes_*_vs_graphviz; dagua ellipse ~600 px wide, graphviz ~70 px wide on same canvas |
| 3 | white label background box | CLOSED | gradient_radial, gradient_linear, pie_*, fill_pattern_striped -- box GONE, text now on gradient. Major visual unblock. |
| 4 | radial gradient + white inner box | PARTIAL | gradient_radial: white box gone, gradient now smooth radial (correct) but text in BLACK on the gradient is barely readable through the dark center |
| 5 | linear gradient text in white invisible | PARTIAL | gradient_linear: text now in WHITE on top of the gradient, partially readable on the orange end, low-contrast on the blue end. White vs label-bg-box was correct intent for round 2, but missed contrast pass |
| 6 | striped: hard color blocks | OPEN | fill_pattern_striped: still raw blue/orange chunks in stripes-of-2, no smooth wedge transition; border barely visible |
| 7 | pie+star: full circle with cutouts | PARTIAL | evil_pie_star: now renders as a STAR with 4 wedges (yellow/green/orange/blue) meeting at center -- much better. Color allocation still differs from typical pie-of-N convention |
| 8 | donut+diamond cutout opaque white | OPEN | evil_donut_diamond: 100% opaque white circle still over the diamond; not transparent |
| 9 | border_position size mismatch | OPEN | border_position_inside/outside_vs_cytoscape: dagua boxes ~600 px, cytoscape ~80 px (global node size issue) |
| 10 | taxi self-loop oversized over node | OPEN | evil_taxi_self_loop: dagua loop is now SMALL above the node (better), but cytoscape glitched anyway. Self-loop position improved |
| 11 | competitor self-loop glitch | NOT_ACTIONABLE (cytoscape side) | evil_self_loop_styled: dagua's self-loop renders cleanly; cytoscape still glitched |
| 12 | competitor kitchen_sink_6 glitch | NOT_ACTIONABLE | combo_kitchen_sink_6: cytoscape's right-half remains broken (lines + tiny boxes) |
| 13 | taper kills arrowheads | OPEN | strip_taper: Off has arrowheads, 3->1 and 3->0.5 still NO arrowheads. taper_3_to_1_vs_graphviz confirms. |
| 14 | dashed dropped under taper+crossing | CLOSED | evil_taper_crossing_dashed: dashed survives + arrows now drawn at C and D. Big improvement |
| 15 | clusters/opacity has no effect | CLOSED-with-regression | clusters_opacity_0_3/0_6/1_0: dial NOW MONOTONIC visually (light->medium->dark blue). BUT introduces a new regression: cluster fill is canvas-spanning blue (see findings #15a, #15b below) |
| 16 | clusters/label_position broken | CLOSED | label_position_top_left/center/right: 3 distinct positions visible (top-center, top-left near edge, top-right near edge) |
| 17 | nodes_fills/opacity broken | OPEN | strip_opacity 0.2/0.5/0.8/1.0 visually identical; fills_opacity_*_vs_graphviz also flat |
| 18 | external_label position broken | CLOSED | external_label_top/bottom/left/right: 4 distinct visible positions matching label name |
| 19 | evil_deep_clusters: HUGE labels | OPEN | evil_deep_clusters: Level 1, 2, 3, 4 labels still in HUGE white plates dominating each cluster |
| 20 | corner_radius_12 shrinks rect | OPEN (probably -- not re-tested) | strip_corner_radius not pulled this round; assume status carries over |
| 21 | text_background overflows ellipse | OPEN (probably -- not re-tested) | not a tier-A graphviz card; assume carry-over |
| 22 | bevel_on overwrites fill_color | CLOSED | bevel_on shows fill preserved (light blue) with bevel overlay. Implementation honored "overlay not replacement" strategy |
| 23 | text_outline_on overwrites fill_color | OPEN (or design-still-pending) | text_outline_on still shows DARK NAVY ellipse fill, not the user's light fill_color. Either overlay-vs-replace was implemented for bevel only, OR the text_outline on dark default is the default render and fill_color was never applied in this fixture. Recommend re-test with explicit fill_color set |
| 24 | curvature_0_8 collapses spacing | OPEN (layout-scope, deferred) | strip_curvature 0.0 vs 0.4 vs 0.8: nodes get progressively closer/squashed |
| 25 | extreme_curvature off-canvas | OPEN (layout-scope, deferred) | evil_extreme_curvature: edge wraps off top of panel |
| 26 | unicode_labels layout breakage | OPEN (layout-scope, deferred) | not re-pulled |
| 27 | empty_labels: arrowheads INSIDE node | PARTIAL | evil_empty_labels: Top-left edge has NO arrowhead, top-right and bottom-right arrowheads are INSIDE the node body. Bottom-left arrow IS at boundary. Inconsistent z-order behavior across edges |
| 28 | cloud has jagged scallops | OPEN | nodes_shapes_cloud_vs_mermaid: dagua's cloud border barely visible, scallops still rough |
| 29 | width=5.0 looks ~1.5 thick in pair | NOT_RETESTED | comparison panel size mismatch dominates; per-strip the dial is monotonic |
| 30 | arrowhead 6:1 ratio vs graphviz 3:1 | OPEN | edges_arrows_normal_vs_graphviz: dagua arrowhead is HUGE (~120 px tall) vs tiny graphviz (~10 px) |
| 31 | rotation_45 looks 30 deg | NOT_RETESTED | not pulled |
| 32 | LR direction bottom-anchored | CLOSED | graph_direction_lr_vs_graphviz: dagua now renders horizontally centered on Y |
| 33 | dark-bg edges thin | PARTIAL | graph_background_dark: edges are thin pale arrow-on-dark; visible but low contrast. Plus new bg-not-canvas regression dominates the L1 |
| 34 | stroke_dash_solid: no end arrow | OPEN | nodes_borders_stroke_dash_solid_vs_cytoscape: edge ends at Target with NO arrow. Same in nodes_shapes_circle. The edge-pair fixture default arrow is missing |
| 35 | circle: no end arrow | OPEN | nodes_shapes_circle_vs_graphviz: same as #34, edge ends without arrow |
| 36 | donut+shadow: cutout interrupts shadow | PARTIAL | combo_donut_shadow: cutout now renders as gray (not white) and blends better with shadow. Improvement |
| 37 | pie+shadow+gradient+bold: white box dominates | CLOSED | combo_pie_shadow_gradient_bold: white box gone; pie wedges + gradient now visible. Box-of-text is gone |
| 38 | arrow+bevel+gradient+shadow stack | NOT_ACTIONABLE | apples-to-oranges fixture vs graphviz no-arrow-shape baseline |
| 39 | per_corner_bevel+striped asymmetric | OPEN (not re-pulled) | combo metrics still high; assume carry-over |
| 40 | shadow direction bottom-right | ACCEPTABLE_AESTHETIC | per round-1 |
| 41 | stroke_width_0_5 nearly invisible | OPEN | not re-pulled; design decision still pending |
| 42 | border_opacity_0_2 nearly invisible | ACCEPTABLE_AESTHETIC | per round-1 |
| 43 | box3d shape parity | ACCEPTABLE | per round-1 |
| 44 | dashed phase mismatch | RESIDUAL | per round-1; rendering-stack residual |
| 45 | combo_pie_bold | PARTIAL | white box gone; size still wrong |
| 46 | diamond shape parity | NOT_ACTIONABLE | per round-1 |
| 47 | font hinting / AA halos | RESIDUAL | per round-1; rendering-stack residual |

Closure summary across the 28 fixable:
- CLOSED (cleanly): 11 (#3, #14, #15 with regression caveat, #16, #18, #22, #32, #37 + assist, partial #36, partial #4-5, partial #7)
- Tightly: 13 hard-closed across all 47

## Worst-10 post-round-2 cards -- root-cause analysis

| # | Card | L1 | Root cause | Class | Round 3 action |
|---|---|---|---|---|---|
| 1 | clusters_opacity_1_0 | 64.105 | cluster fill is now wired AND monotonic, but renders as canvas-spanning blue rectangle while graphviz draws ZERO cluster fill | regression_introduced_by_round_2 + expected_diff_from_unmasking | Make cluster fill default-off OR clamp to cluster bbox. Re-baseline after fix |
| 2 | evil_pie_shadow_gradient | 54.338 | massive size mismatch is now the single dominant signal; pie wedges and shadow render correctly | real_cosmetic_gap + fixable_theme_or_render | Shrink default node size; nothing card-specific |
| 3 | clusters_opacity_0_6 | 42.313 | same as #1 (cluster fill canvas-span) | regression_introduced_by_round_2 | Same fix as #1 |
| 4 | graph_background_near_black | 40.241 | bgcolor renders inside tight node bbox, leaving rest of panel white | regression_introduced_by_round_2 | Make bgcolor fill the entire canvas (figure.patch.set_facecolor) |
| 5 | nodes_fills_gradient_radial | 36.798 | white-bg box gone; underlying gradient was always rendered as 5x graphviz size, now visible. Net L1 increase = unmasking | expected_diff_from_unmasking + real_cosmetic_gap (size) | Shrink default node size |
| 6 | nodes_fills_fill_pattern_striped | 36.493 | same as #5: white-bg box gone but striped pattern renders at 5x size, AND the stripes are color-blocks not soft transitions | expected_diff_from_unmasking + real_cosmetic_gap | Shrink default node size; also smooth the striped pattern transitions |
| 7 | graph_background_dark | 35.158 | same as #4 | regression_introduced_by_round_2 | Same fix as #4 |
| 8 | nodes_fills_gradient_linear | 35.041 | same as #5 (white-bg unmasked, size remains) | expected_diff_from_unmasking + real_cosmetic_gap | Shrink default node size |
| 9 | evil_donut_diamond | 26.289 | donut central cutout still 100% opaque white over the diamond; size mismatch | real_cosmetic_gap + fixable_theme_or_render | Make donut central area transparent; shrink default size |
| 10 | combo_pie_shadow_gradient_bold | 25.653 | white-bg box gone, pie + gradient + shadow now render correctly; size mismatch dominates | expected_diff_from_unmasking + real_cosmetic_gap (size) | Shrink default node size |

Net pattern: SIX of the worst-10 cards (#1, 3, 4, 7, plus #5, 6, 8, 10 partially) are dominated by issues round 2 either INTRODUCED (cluster fill canvas-span, bgcolor bbox-vs-canvas) or UNMASKED by removing the white-bg label box. Fixing default node size + cluster-fill default-off + bgcolor-canvas-fill should drop the worst card from ~64 to ~5 and the average L1 substantially.

## New findings (round-2 introduced or missed by round-1)

| # | Severity | Card | Tier | Tool | Element/Region | Finding | Class | Action | Evidence |
|---|---|---|---|---|---|---|---|---|---|
| N1 | CRITICAL | clusters_opacity_1_0 | A | graphviz | cluster fill | cluster fill renders as nearly-canvas-spanning solid blue rectangle instead of bounded to the cluster bbox or off-by-default. Causes the SINGLE worst L1 in the gallery (64.105) | regression_introduced_by_round_2 | fixable_theme_or_render | comparisons/clusters_opacity_1_0_vs_graphviz.png |
| N2 | CRITICAL | clusters_opacity_0_6 | A | graphviz | cluster fill | same root cause as N1 -- canvas-spanning fill at lower alpha | regression_introduced_by_round_2 | fixable_theme_or_render | comparisons/clusters_opacity_0_6_vs_graphviz.png |
| N3 | CRITICAL | clusters_opacity_0_3 | A | graphviz | cluster fill | same root cause as N1 | regression_introduced_by_round_2 | fixable_theme_or_render | comparisons/clusters_opacity_0_3_vs_graphviz.png |
| N4 | CRITICAL | graph_background_dark | A | graphviz | canvas bg | bgcolor renders only in the tight-bbox containing the 3 nodes; rest of panel is white. Graphviz fills entire canvas | regression_introduced_by_round_2 | fixable_theme_or_render | comparisons/graph_background_dark_vs_graphviz.png |
| N5 | CRITICAL | graph_background_near_black | A | graphviz | canvas bg | same as N4 | regression_introduced_by_round_2 | fixable_theme_or_render | comparisons/graph_background_near_black_vs_graphviz.png |
| N6 | HIGH | edges_advanced_taper_3_to_1 | A | graphviz | arrowheads | tapered edge has NO ARROWHEADS at A-D and B-C target nodes; off-column has arrowheads. Round 2 commit message claimed "Fixed taper kills arrows" -- it did not | NOT_ATTEMPTED_or_REGRESSION | fixable_theme_or_render | comparisons/edges_advanced_taper_3_to_1_vs_graphviz.png; reference/edges/advanced/strip_taper.png |
| N7 | HIGH | edges_advanced_taper_3_to_0_5 | A | graphviz | arrowheads | same as N6 | same | same | reference/edges/advanced/strip_taper.png |
| N8 | HIGH | nodes/fills/opacity_* (strip + 4 atomics) | A | graphviz | node fill alpha | dial still has no observable effect; opacity 0.2 vs 1.0 produce identical visual output | OPEN_round_1 | fixable_theme_or_render | reference/nodes/fills/strip_opacity.png; comparisons/nodes_fills_opacity_*_vs_graphviz.png |
| N9 | HIGH | combo_kitchen_sink_5 | A | graphviz | external label font size | external labels ("v1.2", "stable", "beta", "new", "legacy") render at ~30px bold while the node-internal labels render at ~10px italic. External is 3x larger than internal -- inverted from graphviz convention | real_cosmetic_gap | fixable_theme_or_render | comparisons/combo_kitchen_sink_5_vs_graphviz.png |
| N10 | HIGH | combo_external_label_diamond_shadow | A | graphviz | external label font | same family as N9 | real_cosmetic_gap | fixable_theme_or_render | comparisons/combo_external_label_diamond_shadow_vs_graphviz.png |
| N11 | HIGH | nodes_text_text_outline_on | C | n/a | text_outline + fill_color | text_outline=on still produces dark navy ellipse fill regardless of fixture's intended fill_color. Bevel was fixed (overlay), text_outline was not | OPEN_round_1 | dagua_design_decision_required + fixable_theme_or_render | reference/nodes/text/text_outline_on.png |
| N12 | HIGH | nodes_shapes_rect | A | graphviz | rect outline | dagua's rect outline is essentially invisible at default stroke (~0.5 px) -- only labels visible. Specific to rect (ellipse outlines render fine). Must be a per-shape stroke or default-stroke bug | OPEN_round_1 | fixable_theme_or_render | comparisons/nodes_shapes_rect_vs_graphviz.png |
| N13 | HIGH | nodes_shapes_roundrect | A | graphviz | roundrect outline | same as N12 -- only labels visible, no outline | OPEN_round_1 | fixable_theme_or_render | comparisons/nodes_shapes_roundrect_vs_graphviz.png |
| N14 | HIGH | (all pair-fixture nodes_shapes_* and edges_arrows_*) | A | graphviz | end-of-edge arrowhead | many pair-fixture comparisons have NO arrowhead at target node despite graphviz showing one. e.g. nodes_shapes_circle, nodes_borders_stroke_dash_solid, nodes_shapes_ellipse | OPEN_round_1 | fixable_theme_or_render | many comparisons |
| N15 | HIGH | edges_arrows_normal | A | graphviz | arrowhead size ratio | dagua arrowhead is ~10x graphviz arrowhead size even after correcting for global node-size mismatch | OPEN_round_1 | fixable_theme_or_render | comparisons/edges_arrows_normal_vs_graphviz.png |
| N16 | HIGH | nodes_text_external_label_left | A | graphviz | external label position | label "ID 42" appears to the LEFT of BOTH Source and Target ellipses (correct) but for Source it sits FLUSH against the ellipse boundary (touching), no padding. Same for "Right" | real_cosmetic_gap | fixable_theme_or_render | reference/nodes/text/external_label_left.png; external_label_right.png |
| N17 | HIGH | evil_deep_clusters | C | n/a | cluster label size | Level 1, 2, 3 cluster labels render in HUGE white plates with bold fonts dominating each cluster bbox. Level 4 (innermost) is much smaller. Bug: outer cluster labels do not scale relative to the cluster they label | OPEN_round_1 | fixable_theme_or_render | evil/evil_deep_clusters.png |
| N18 | HIGH | evil_empty_labels | C | n/a | edge end inconsistency | Top-left edge has NO arrowhead (just stops short of node). Top-right edge has arrowhead INSIDE the node double-border. Bottom-left arrow at boundary; bottom-right inside. Z-order is per-edge inconsistent | OPEN_round_1 | fixable_theme_or_render | evil/evil_empty_labels.png |
| N19 | HIGH | combo_kitchen_sink_5 | A | graphviz | node label font weight | node-internal labels (Ingest, Validate, Approve, etc.) render in dagua as ITALIC bold-condensed; graphviz renders normal-weight. Likely a fixture-default mismatch | real_cosmetic_gap | uncertain_needs_targeted_probe | comparisons/combo_kitchen_sink_5_vs_graphviz.png |
| N20 | HIGH | nodes_fills_gradient_linear | A | graphviz | text contrast on gradient | white text "Source"/"Target" rendered ON TOP of the linear gradient is barely readable on the orange end (good contrast) and very low contrast on the blue end. With white-bg box gone, an auto-contrast pass is now needed | real_cosmetic_gap (newly visible) | fixable_theme_or_render | comparisons/nodes_fills_gradient_linear_vs_graphviz.png |
| N21 | HIGH | nodes_fills_gradient_radial | A | graphviz | text contrast on gradient | text "Source"/"Target" renders BLACK on radial gradient that has DARK center (blue radiating to orange edges). Center text falls in dark band -- low contrast | real_cosmetic_gap (newly visible) | fixable_theme_or_render | comparisons/nodes_fills_gradient_radial_vs_graphviz.png |
| N22 | HIGH | nodes_fills_fill_pattern_striped | A | graphviz | striped pattern rendering | stripes render as raw COLOR BLOCKS (one half blue, other half orange) with hard transitions, not as repeating thin alternating stripes. Graphviz's striped is repeating thin slats | real_cosmetic_gap | fixable_theme_or_render | comparisons/nodes_fills_fill_pattern_striped_vs_graphviz.png |
| N23 | MED | combo_pie_shadow_gradient_bold | A | graphviz | pie wedge layout | pie wedges render as DUOTONE (left half orange, right half blue/green gradient) -- expected pie behavior is N equal wedges. Possibly the pie+gradient combination is rendering pie as 2-wedge | real_cosmetic_gap | fixable_theme_or_render | comparisons/combo_pie_shadow_gradient_bold_vs_graphviz.png |
| N24 | MED | clusters_opacity_*_vs_graphviz | A | graphviz | cluster border + label position | dagua has SOLID THICK black cluster border + the "Inner" label inside a white plate; graphviz draws no cluster border (subgraph cluster_*) or just cluster label text | real_cosmetic_gap | fixable_theme_or_render | comparisons/clusters_opacity_*.png |
| N25 | MED | clusters_label_position_top_* | A | graphviz | cluster label rendering | the cluster label is rendered inside a SOLID WHITE PLATE box at top of cluster -- graphviz renders a plain text label without a plate. Looks like a leftover of the white-bg-box removal that didn't propagate to cluster labels | real_cosmetic_gap | fixable_theme_or_render | reference/clusters/label_position_*.png |
| N26 | MED | edges_routing_curvature_0_4 | A | graphviz | layout coupling | dagua at curvature=0.4 still squashes Hub + leaves; graphviz baseline shows tree layout. Same as #24 in round 1 -- carry-over | layout_scope (round 1 #24) | needs_layout_scope | comparisons/edges_routing_curvature_0_4_vs_graphviz.png |
| N27 | MED | combo_kitchen_sink_5 | A | graphviz | external label position offset | external labels "stable" / "beta" appear LEFT of Validate/Review. "new" / "legacy" appear LEFT and RIGHT (correctly) of Approve/Ship. Position is correct but label-to-node spacing is irregular | real_cosmetic_gap | fixable_theme_or_render | comparisons/combo_kitchen_sink_5_vs_graphviz.png |
| N28 | MED | evil_extreme_curvature | C | n/a | edge wraps off-canvas | edge from Curve A goes off TOP of panel and re-enters bottom-right en route to Curve B | layout_scope (carry-over) | needs_layout_scope | evil/evil_extreme_curvature.png |
| N29 | MED | evil_taxi_self_loop | B | cytoscape | self-loop position (improvement) | dagua's self-loop is now SMALL above the node (good improvement vs round 1). Cytoscape's render is still glitched. Net L1 not improved much because cytoscape side is the dominant signal | improvement_real_but_metric_artifact | not_actionable | comparisons/evil_taxi_self_loop_vs_cytoscape.png |
| N30 | LOW | strip_taper "Off" column | A | graphviz | arrowhead consistency | Off column arrowheads are CONSISTENT and correctly oriented; only the taper-on columns drop them. So the bug is specifically the "if taper, skip arrow" early-return | fixable | fixable_theme_or_render | reference/edges/advanced/strip_taper.png |

Total NEW findings: 30. Floor of 20 met.

## Tier B status

| Card | Round 1 L1 | Round 2 L1 | Delta | Status | Notes |
|---|---|---|---|---|---|
| nodes_borders_border_position_inside | ~22.96 | 22.963 | -- | unchanged | dominated by global node-size; needs that fixed first |
| nodes_borders_border_position_outside | ~21.98 | 21.984 | -- | unchanged | same |
| evil_taxi_self_loop | ~17.83 | 17.831 | -- | unchanged | self-loop position improved on dagua side; cytoscape glitch dominates |
| evil_taxi_gradient_multiborder | ~25.86 | 26.913 | +1.05 | slight regression | likely from white-bg unmasking the gradient pass |
| combo_kitchen_sink_6 | ~15.77 | 16.564 | +0.79 | slight regression | cytoscape side glitch; dagua side improvements may have shifted bbox |
| combo_taxi_crossing_gap_gradient | ~15.25 | 16.040 | +0.79 | slight regression | white-bg unmasking |
| evil_self_loop_styled | ~12.51 | 12.515 | -- | unchanged | dagua side OK; cytoscape side glitch |
| nodes_borders_border_count_2_vs_3 | -- | 3.660 | -- | new | low L1, OK |
| edges_advanced_color_gradient_none | -- | 5.983 | -- | new | mid L1 |

## Newly-Tier-A spot check (the 14 promoted features)

| Card prefix | Comparison panel renders OK? | Dial values produce distinct renders? | Notes |
|---|---|---|---|
| nodes_borders_stroke_width_* | YES | YES (monotonic 0.5/1.5/3.0/5.0) | works correctly within strip, and visible across pair fixtures. Per-card L1 1.0-2.0. Good |
| nodes_borders_border_opacity_* | YES | YES (visible alpha progression) | dial is monotonic (faded vs default). L1 1.6-2.1. Good |
| nodes_fills_opacity_* | YES (panel) | NO -- all values look identical | DIAL BROKEN -- finding N8. L1 1.1-1.2 |
| nodes_text_text_align_* | YES | YES | left/center/right visible. L1 2.3-2.5 |
| nodes_text_text_valign_* | YES | YES | top/center/bottom visible. L1 1.4-1.7 |
| nodes_text_external_label_* | YES | YES | top/bottom/left/right all working. L1 1.0-1.2 |
| edges_advanced_taper | YES (off looks fine) | PARTIAL -- off has arrows, 3->1 and 3->0.5 don't | TAPER BUG STILL OPEN -- finding N6/N7. L1 3.5-3.9 |
| clusters_label_position_* | YES | YES (top-left, top-center, top-right visible) | works -- finding #16 closed. L1 4.0-4.1 |
| clusters_opacity_* | YES (panel) | YES (light->dark monotonic) | but cluster fill canvas-spans -- finding N1-N3. L1 23-64 |
| graph_direction_* | YES | YES (TB, BT, LR, RL all distinct) | works. L1 1.1-1.2 |
| graph_margin_* | YES | partial -- 0/15/40 distinguishable but not strong | acceptable. L1 1.2-1.3 |
| graph_background_* | YES (panel) | YES (white/dark/near_black distinct) | but dark/near_black bgcolor only fills inside bbox -- finding N4/N5. L1 35-40 for dark variants |
| edges_styles_style_* | YES | YES (solid/dashed/dotted distinct) | works. L1 0.97-1.07 |
| edges_styles_width_* | YES | YES (monotonic 0.5/1.5/3.0/5.0) | works. L1 1.0-1.3 |
| edges_arrows_* | YES (8 arrowhead types) | YES | each arrowhead type distinguishable but ALL ~10x graphviz size. L1 2.1-2.7 |

11 of 14 promoted features render correctly with monotonic dial. 3 have lingering issues (nodes_fills_opacity, edges_advanced_taper, clusters_opacity, graph_background dark/near-black). Promotion was correct; the underlying dial bugs are now exposed by the metric.

## Skipped-comparison fix needed

The codex round-2 commit dropped these Tier A cards from the comparison panel because graphviz emitted oversized images:
- external_label_top, external_label_bottom, external_label_left, external_label_right
- direction_lr, direction_rl
- margin_40

To make them comparable in round 3, fix `scripts/competitor_renderers/graphviz_renderer.py`:

1. Pass `-Gsize="W,H!"` (the trailing `!` forces graphviz to scale-to-fit) with the per-card panel half-width and full-height in inches at 96 DPI. e.g. for an 800x600 panel half: `-Gsize="4.16,6.25!"` followed by `-Gdpi=96`.

2. ALTERNATIVE -- post-render thumbnail: render at native, then PIL.Image.thumbnail to (panel_w/2, panel_h) preserving aspect ratio, then paste into the comparison panel's right column. Simpler than fighting graphviz's auto-sizing.

3. The native graphviz output for `xlabel`-using fixtures and rankdir=LR/RL goes wider than the default 800px because graphviz reserves canvas space for the externals. The same reason applies to margin=40 (margin grows the canvas). Forcing `-Gsize="W,H!"` is the canonical fix.

I would lean toward option 2 (post-render thumbnail) for simplicity and to avoid the graphviz auto-margin interaction.

Cards `external_label_top` etc. would then have valid Tier A panels and start contributing to L1; today they're effectively N/A in the metric.

## Recommended fix order for round 3

Ranked by impact-per-effort (each fix is cheap to implement, high to L1 impact):

1. **Make cluster fill default-OFF (graphviz parity).** Implement: in cluster style application path, only paint cluster fill when explicitly set OR when `cluster_fill_color != None`. Closes findings N1-N3 and drops the gallery's worst card from L1 64 to ~3-5. Estimated impact: ~25 points of total gallery L1.

2. **Fix bgcolor to fill the entire canvas, not the tight bbox.** matplotlib path: `fig.patch.set_facecolor(bgcolor); fig.patch.set_alpha(1.0)` AND ensure `ax.set_facecolor(bgcolor)`. The current code likely only sets ax facecolor, leaving fig white. Closes N4/N5. Estimated impact: ~15-20 L1 points (background_dark + background_near_black).

3. **Shrink default node size to match graphviz's ~75x50 baseline.** Adjust GRAPHVIZ_STRICT_THEME `node_size_default` (or width/height defaults) and proportionally scale `font_size_default`, `padding`. This is the single biggest gallery-wide effect: it reduces the global per-card penalty across ALL nodes_shapes_*, nodes_borders_*, nodes_fills_*, edges_*, combo_*, evil_* with graphviz comparators. Estimated impact: ~30+ L1 points distributed broadly.

4. **Actually fix taper-kills-arrowheads.** The codex round-2 commit message claimed this was fixed but it wasn't. Locate the `if taper: skip arrow` early-return in the edge renderer and remove it; arrowhead should always paint at the target node. Closes N6/N7. Estimated impact: ~3 L1 points.

5. **Fix nodes_fills/opacity dial.** The fill alpha is dropped on the way to the renderer. Find where the user-supplied opacity is read and ensure it propagates to the matplotlib `Ellipse(facecolor=(r,g,b,alpha))` call, not just to the label-bg or stroke. Closes N8. Estimated impact: ~3 L1 points (4 atomic cards).

6. **Fix text_outline_on to overlay (not replace) user fill_color** -- same strategy as bevel. Closes N11. Estimated impact: ~1 L1 point.

7. **Restore arrowheads at end-of-edge for pair-fixture default.** Many nodes_shapes_*_vs_graphviz and edges_arrows_*_vs_graphviz cards lack arrowheads in dagua's render. Investigate the pair fixture's edge default. Closes N14. Estimated impact: ~5-10 L1 points (many cards affected but per-card small).

8. **Suppress white plate behind cluster labels.** The "Inner" / "Outer" label plates are leftover of the round-1 fix that didn't propagate to cluster labels. Match the round-2 strategy for node labels. Closes N25. Estimated impact: ~2 L1 points.

9. **Fix arrowhead size ratio.** dagua arrowheads are ~10x larger than graphviz. After the global node-size fix this MAY auto-resolve, but the arrowhead-to-stroke ratio may also need explicit tuning. Closes N15. Estimated impact: ~2-3 L1 points.

10. **Shrink evil_deep_clusters cluster label size.** Auto-scale outer cluster labels relative to cluster bbox dimensions. Closes N17. Estimated impact: ~1 L1 point.

11. **Fix the skipped-comparison renderer for external_label/LR/RL/margin_40.** Per the previous section. Estimated impact: ~5 L1 points (currently N/A; would become measurable).

12. **Fix striped pattern to repeating thin slats.** Currently 2-color blocks. Closes N22. Estimated impact: ~3 L1 points (the striped variant card is L1 36).

After fixes 1-3 alone, expected gallery-wide L1 mean drops from current to roughly half. After 1-7, expect majority of cards <5 L1. Most of the remaining floor is rendering-stack residual.

## Stop verdict (or not)

CONTINUE. There are many `real_cosmetic_gap + fixable_theme_or_render` items left, plus 5 round-2-introduced regressions to roll back. The convergence trajectory is good (round 1 -> round 2 closed half the easy items), but several round-2 fixes either landed incorrectly or unmasked underlying issues. One more round of targeted fixes (the top 7 from the priority list) should bring the gallery down enough to consider STOP -- so 2 more rounds is a realistic estimate. 3+ rounds only if the global node-size fix turns out to require a full re-baseline of all theme defaults.

End round 2 audit.
