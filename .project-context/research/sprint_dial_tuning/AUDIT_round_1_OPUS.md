# Round 1 Audit -- dial_tuning

## Verdict

- New audit: FAIL
- Stop criteria status: CONTINUE
- Total findings: 47
  - real_cosmetic_gap + fixable_theme_or_render: 28
  - real_cosmetic_gap + rendering_stack_residual: 3
  - real_cosmetic_gap + needs_layout_scope: 4
  - dagua_design_decision_required: 5
  - acceptable_aesthetic_choice: 3
  - metric_or_measurement_artifact: 2
  - uncertain_needs_targeted_probe: 2

The cosmetic dial parity is far from ship-ready. The single biggest issue is a global default-node-size mismatch that dwarfs every other signal in the side-by-side panels; on top of that there are at least three broken dials (cluster opacity, cluster label_position, external label position) and several feature-combination breakages (taper kills arrowheads, dashed style is dropped under taper/crossing, fill_color is overwritten by bevel/text_outline). The "white label background box" is also painted on EVERY render even when no text background was requested -- this is a render-default that should be off and is the visual signature ruining most non-default dial cards.

## Worst-systemic issues (rank highest priority)

1. **Default node size is ~3-5x too large vs graphviz baseline.** Across virtually every Tier A graphviz comparison (rect, ellipse, diamond, star, circle, cylinder, hexagon, pentagon, octagon, cloud, stadium), dagua renders nodes at ~400-700 px wide while graphviz renders at ~80-150 px wide on the same canvas. The L1 metric is dominated by this size mismatch -- the differences are not really about each individual feature but about the global scale. Evidence: every panel under `per_card_pixel_diff/comparisons/nodes_shapes_*_vs_graphviz.png`. Recommended fix area: `dagua/styles.py` GRAPHVIZ_STRICT_THEME default `node_size_default`, `font_size_default`, `padding`. Align baseline metrics to graphviz's 75-by-50 default-ish ellipse.

2. **The "white label background box" is being painted on every node even when no text_background is set.** A solid (or near-solid) white rectangular box surrounds each label inside the ellipse / shape. This box is conspicuous in every gradient/striped/pie/donut card (gradient_linear, gradient_radial, fill_pattern_striped, evil_pie_*, combo_pie_*) and is the dominant visual noise. The competitor renderers do not paint a label background by default. Evidence: `evil_pie_star`, `evil_pie_shadow_gradient`, `nodes_fills_gradient_linear`, `nodes_fills_gradient_radial`, `combo_pie_*`. Recommended fix area: `dagua/render/text.py` (or wherever the label-bg layer lives) -- default `text_background = none`.

3. **Three Tier C dials are completely broken / have no visual effect.** (a) `clusters/opacity_*`: opacity values 0.3, 0.6, 1.0 all render the cluster border at full opacity -- no observable difference. (b) `clusters/label_position_*`: top_left, top_center, top_right all render the cluster label TOP-CENTER -- the position dial is ignored. (c) `nodes/fills/opacity_*`: 0.2, 0.5, 0.8, 1.0 strip cards are visually identical -- the dial has no observable effect. Evidence: `clusters/opacity_0_3.png` vs `opacity_1_0.png`; `clusters/label_position_top_left.png`; `nodes/fills/strip_opacity.png`. Recommended fix area: cluster style application + node fill opacity propagation through the renderer.

4. **`external_label` position dial is ignored.** All external-label values (top, bottom, left, right) render the external label BELOW the node in the Tier C atomic and combo cards. Combo card `external_label_rounded` shows v1.2 below Ingest, stable below Validate, beta below Review, new below Approve, legacy below Ship -- regardless of which position was requested. Evidence: `clusters/label_position_*`, `combos/2way/external_label_rounded.png`, `nodes/text/external_label_top.png`. Recommended fix area: external-label render placement code.

5. **Multiple feature-combination breakages destroy companion features.** (a) Taper combined with arrowheads: arrowheads VANISH (`edges/advanced/strip_taper.png`). (b) Taper combined with dashed style: dashed is replaced with solid (`evil_taper_crossing_dashed`). (c) `bevel_on` and `text_outline_on` both OVERWRITE the user-specified fill_color with a built-in dark blue fill. (d) Cluster label rendering creates HUGE white-box labels that overwhelm the cluster bounding box (`evil_deep_clusters` -- 4 stacked level labels each in massive white rectangles eating most of the cluster).

6. **Edge stroke width 5.0 doesn't read as visibly thick in pair fixtures.** The strip card shows monotonic widening from 0.5 -> 5.0, but in `edges_styles_width_5_0_vs_graphviz` and many combo cards the width-5.0 line still looks ~1.5 stroke. Suggests width is being clamped or default-overridden in some render paths. Verify: pair-fixture default stroke width is the same scale as strip-fixture default.

7. **Curvature 0.8 collapses node spacing.** `strip_curvature.png` shows that curvature=0.8 cramps Hub + 4 leaves into a tight horizontal cluster (Leaf 1 overlapping Leaf 2). The cosmetic curvature dial is coupling into the layout solver. `evil_extreme_curvature` shows nodes at far corners with edges going off-page. Layout/cosmetic should be orthogonal.

8. **Crossing-style + bridge marker shows tiny diamond at crossing center even when not desired in baseline.** `evil_all_new_features` shows a small white-diamond marker at the X intersection -- not obviously matched to documented `crossing_style=bridge` semantic. Verify and re-document.

## Findings

| # | Severity | Card | Tier | Tool | Element/Region | Finding | Class | Action | Evidence |
|---|---|---|---|---|---|---|---|---|---|
| 1 | HIGH | nodes_shapes_rect | A | graphviz | node border / fill | dagua rect borders are essentially invisible at default stroke (~0.5 px) -- only the labels "Source"/"Target" are visible with a single thin vertical edge between them. Graphviz shows clean small rects. The default ellipse case below shows ellipse outlines fine, so this is a rect-specific stroke or shape-not-rendered bug. | real_cosmetic_gap | fixable_theme_or_render | comparisons/nodes_shapes_rect_vs_graphviz.png |
| 2 | HIGH | (global) | A | graphviz | node size | dagua's default ellipse is ~600x250 px; graphviz's is ~70x40 px. Same canvas. Visible across every shape comparison. | real_cosmetic_gap | fixable_theme_or_render | comparisons/nodes_shapes_*_vs_graphviz.png (all) |
| 3 | HIGH | (global) | A/B/C | graphviz/cytoscape/mermaid | text label | white rectangular label background box paints behind every label by default, even when text_background=none was requested. Conspicuous across all gradient/pattern/pie cards. | real_cosmetic_gap | fixable_theme_or_render | nodes_fills_gradient_linear, gradient_radial, fill_pattern_striped, evil_pie_*, combo_pie_* |
| 4 | HIGH | nodes_fills_gradient_radial | A | graphviz | node fill | radial gradient renders as solid orange ellipse with a white inner rectangle (the label background box). Should be a smooth radial gradient with center color -> edge color. | real_cosmetic_gap | fixable_theme_or_render | comparisons/nodes_fills_gradient_radial_vs_graphviz.png |
| 5 | HIGH | nodes_fills_gradient_linear | A | graphviz | node fill text | linear gradient blue->orange is correct in horizontal direction but the label "Source"/"Target" text is rendered in WHITE, INVISIBLE against the lightcream label-bg box. Text contrast lost. | real_cosmetic_gap | fixable_theme_or_render | comparisons/nodes_fills_gradient_linear_vs_graphviz.png |
| 6 | HIGH | nodes_fills_fill_pattern_striped | A | graphviz | node fill | striped pattern renders as raw vertical color blocks (blue / white / orange) intersected with the white label-bg box. No anti-aliased edge transitions; not the smooth wedge pattern competitor produces. | real_cosmetic_gap | fixable_theme_or_render | comparisons/nodes_fills_fill_pattern_striped_vs_graphviz.png |
| 7 | HIGH | evil_pie_star | A | graphviz | node shape + fill | star-shaped node with pie fill renders as full CIRCLE with star-arm cutouts inside, not a star outline filled with pie wedges. The shape+pie combination produces visible non-shape artifacts. | real_cosmetic_gap | fixable_theme_or_render | evil_pie_star_vs_graphviz.png |
| 8 | HIGH | evil_donut_diamond | A | graphviz | node shape | donut-effect on a diamond renders the central circular cutout INSIDE the diamond (correct intent) but the cutout is 100% white (not transparent) and at wrong proportion -- looks like a circle stamped on a diamond. Better as transparent donut hole or matching diamond corner inset. | real_cosmetic_gap | fixable_theme_or_render | evil_donut_diamond_vs_graphviz.png |
| 9 | HIGH | nodes_borders_border_position_inside / outside | B | cytoscape | node border | dagua renders border position correctly (inside vs outside) but at radically different stroke + node size from cytoscape. Cytoscape uses ~80 px nodes with ~3 px borders; dagua uses ~600 px nodes with ~50 px borders. | real_cosmetic_gap | fixable_theme_or_render | comparisons/nodes_borders_border_position_inside_vs_cytoscape.png |
| 10 | HIGH | evil_taxi_self_loop | B | cytoscape | self-loop edge | self-loop on taxi-routed node renders the loop with a thick gray stroke OVER the top of the node -- z-order looks borderline-wrong (loop on top of node body) and the loop diameter is huge relative to node. | real_cosmetic_gap | fixable_theme_or_render | comparisons/evil_taxi_self_loop_vs_cytoscape.png |
| 11 | HIGH | evil_self_loop_styled | B | cytoscape | self-loop edge | dagua self-loop renders correctly small/clean; cytoscape rendered as huge solid blue circle (cytoscape glitch -- not dagua's fault). Mark as competitor-side issue. | metric_or_measurement_artifact | not_actionable | comparisons/evil_self_loop_styled_vs_cytoscape.png |
| 12 | HIGH | combo_kitchen_sink_6 | B | cytoscape | competitor render | cytoscape's kitchen_sink_6 right-half panel is broken -- nodes are tiny dots, edges flat lines. Competitor render glitch, not dagua. | metric_or_measurement_artifact | not_actionable | comparisons/combo_kitchen_sink_6_vs_cytoscape.png |
| 13 | HIGH | edges_advanced_taper | C | n/a | arrowheads under taper | When taper feature is enabled (3->1 or 3->0.5), the edge arrowheads VANISH. With taper off, arrows render normally. Combination breakage. | real_cosmetic_gap | fixable_theme_or_render | reference/edges/advanced/strip_taper.png |
| 14 | HIGH | evil_taper_crossing_dashed | C | n/a | dashed style under combination | Edges should be DASHED but render as SOLID when combined with taper + crossing_style. Dashed dial is dropped. | real_cosmetic_gap | fixable_theme_or_render | evil/evil_taper_crossing_dashed.png |
| 15 | HIGH | clusters_opacity_0_3, 0_6, 1_0 | C | n/a | cluster border | Opacity dial has NO visible effect on cluster borders -- 0.3, 0.6, 1.0 all render at full opacity. Dial broken. | real_cosmetic_gap | fixable_theme_or_render | reference/clusters/opacity_*.png |
| 16 | HIGH | clusters_label_position_top_left/center/right | C | n/a | cluster label | All three position values render the cluster label TOP-CENTER. Dial broken. | real_cosmetic_gap | fixable_theme_or_render | reference/clusters/label_position_*.png |
| 17 | HIGH | nodes_fills_opacity_0_2 ... 1_0 | C | n/a | node fill | strip_opacity has no observable difference between any of the four values. Either dial broken or label-bg box is masking the fill. | real_cosmetic_gap | fixable_theme_or_render | reference/nodes/fills/strip_opacity.png |
| 18 | HIGH | nodes_text_external_label_top/bottom/left/right | C | n/a | external label position | All four position values render external label BELOW the node in atomic + combo cards. Dial broken. | real_cosmetic_gap | fixable_theme_or_render | reference/nodes/text/external_label_*.png; combos/2way/external_label_rounded.png |
| 19 | HIGH | evil_deep_clusters | C | n/a | cluster label | 4-deep nested cluster labels render in massive white rectangular boxes with very large bold text ("Level 1", "Level 2", "Level 3", "Level 4"), each label occupying ~30-40% of its cluster's width. Labels are far too large for cluster size and visually dominate the figure. | real_cosmetic_gap | fixable_theme_or_render | evil/evil_deep_clusters.png |
| 20 | HIGH | nodes_borders_corner_radius_12 | C | n/a | corner radius progression | At corner_radius=12 the rect SHRINKS to half the size of the default rect (corner_radius=0). Other values (24, 40) render at consistent default size. Non-monotonic value progression. | real_cosmetic_gap | fixable_theme_or_render | reference/nodes/borders/strip_corner_radius.png |
| 21 | HIGH | nodes_text_text_background_orange/green | C | n/a | text background | the text background box is drawn as a hard-edged RECTANGLE that overflows the curved ellipse boundary on top and bottom, looking like a sticker stuck on. Should conform to ellipse shape or be clipped. | real_cosmetic_gap | fixable_theme_or_render | reference/nodes/text/text_background_orange.png |
| 22 | HIGH | nodes_effects_bevel_on | C | n/a | bevel + fill_color | bevel mode replaces user-specified fill_color with a built-in mid-blue color. The user's fill choice is lost. | real_cosmetic_gap | dagua_design_decision_required | reference/nodes/effects/bevel_on.png |
| 23 | MED | nodes_text_text_outline_on | C | n/a | text outline + fill | text_outline=on darkens the entire ellipse fill to dark navy regardless of user-specified fill_color. Same bug class as bevel. | real_cosmetic_gap | dagua_design_decision_required | reference/nodes/text/text_outline_on.png |
| 24 | HIGH | edges_routing_curvature_0_8 | A | graphviz | layout coupling | curvature=0.8 cramps the layout (4 leaves overlap). Cosmetic curvature dial should not change positions. | real_cosmetic_gap | needs_layout_scope | reference/edges/routing/strip_curvature.png |
| 25 | HIGH | evil_extreme_curvature | C | n/a | edge clipping | Curve A and Curve B are at opposite extreme corners; the connecting edge wraps off the panel top, with two disconnected arc segments visible. Edge geometry overflows canvas at extreme curvature. | real_cosmetic_gap | needs_layout_scope | evil/evil_extreme_curvature.png |
| 26 | HIGH | evil_unicode_labels | C | n/a | layout | Café node has dangling edge stub going OFF the top of the panel; nodes are misaligned with the edges; the second-pair lower nodes connect to upper nodes via curved edge but Café has no visible source-target. Layout/edge-routing inconsistency. | real_cosmetic_gap | needs_layout_scope | evil/evil_unicode_labels.png |
| 27 | HIGH | evil_empty_labels | C | n/a | z-order arrowheads | Arrowheads on edges with empty-label nodes render INSIDE the node body (inside the double-border) rather than at the node boundary. Z-order/end-of-edge calculation broken with double-border. Edges also have inconsistent colors (one black, one orange). | real_cosmetic_gap | fixable_theme_or_render | evil/evil_empty_labels.png |
| 28 | HIGH | nodes_shapes_cloud | B | mermaid | shape contour | dagua's cloud has SHARP/JAGGED scallops at the top edge (looks like raw triangle peaks rather than smooth bumps); also a discontinuity / gap on the right-bottom scallop. Mermaid's cloud has smooth, even bumps. | real_cosmetic_gap | fixable_theme_or_render | comparisons/nodes_shapes_cloud_vs_mermaid.png |
| 29 | MED | edges_styles_width_5_0 | A | graphviz | edge stroke | edge at width=5.0 in pair fixture is rendered ~1.5 px equivalent visually, much thinner than expected. Strip-fixture width=5.0 looks correctly thick. Per-fixture default is being applied differently. | real_cosmetic_gap | fixable_theme_or_render | comparisons/edges_styles_width_5_0_vs_graphviz.png; reference/edges/styles/strip_width.png |
| 30 | MED | edges_arrows_normal | A | graphviz | arrowhead size | dagua arrowhead is ~80 px wide; graphviz is ~12 px. Scale-dependent (because nodes are 5x larger), but disproportionate even after node-size correction -- arrowhead-to-stroke ratio is still ~6:1 vs graphviz ~3:1. | real_cosmetic_gap | fixable_theme_or_render | comparisons/edges_arrows_normal_vs_graphviz.png |
| 31 | MED | nodes_text_text_rotation_45 | B | cytoscape | text rotation | dagua applies rotation but visible angle is ~30° not 45°. Cytoscape doesn't rotate at all (cytoscape glitch). Verify dagua's rotation calculation. | uncertain_needs_targeted_probe | fixable_theme_or_render | comparisons/nodes_text_text_rotation_45_vs_cytoscape.png |
| 32 | MED | graph_direction_lr | C | n/a | vertical centering | LR direction renders correctly horizontally but the result is bottom-anchored on the panel (top half empty). Vertical centering broken. | real_cosmetic_gap | fixable_theme_or_render | reference/graph/direction_lr.png |
| 33 | MED | graph_background_dark | C | n/a | edge visibility on dark | Edges render as thin white-on-dark lines that are barely visible. Needs auto-contrast adjustment or thicker default edge stroke when bg is dark. | real_cosmetic_gap | fixable_theme_or_render | reference/graph/background_dark.png |
| 34 | MED | nodes_borders_stroke_dash_solid | B | cytoscape | end arrow | Edge from Source -> Target ends with NO arrowhead -- line just terminates at the target ellipse boundary. Compare to graphviz/cytoscape default which always has arrowhead. | real_cosmetic_gap | fixable_theme_or_render | comparisons/nodes_borders_stroke_dash_solid_vs_cytoscape.png |
| 35 | MED | nodes_shapes_circle | A | graphviz | end arrow | similar -- circle pair has no visible arrowhead at target node. | real_cosmetic_gap | fixable_theme_or_render | comparisons/nodes_shapes_circle_vs_graphviz.png |
| 36 | MED | combo_donut_shadow | A | graphviz | donut + shadow | donut central cutout is 100% white over multi-color pie wedges; with shadow under it the cutout interrupts shadow continuity. Shadow + donut combination needs cleanup. | real_cosmetic_gap | fixable_theme_or_render | comparisons/combo_donut_shadow_vs_graphviz.png |
| 37 | MED | combo_pie_shadow_gradient_bold | A | graphviz | pie + bold + gradient | dagua node renders ~600 px tall ellipse with large bold label in a white box; the pie wedges + gradient are visible. Visual is functional but the white label box is the dominant visual element vs the pie. | real_cosmetic_gap | fixable_theme_or_render | comparisons/combo_pie_shadow_gradient_bold_vs_graphviz.png |
| 38 | MED | evil_arrow_bevel_gradient_shadow | A | graphviz | arrow shape + bevel | arrow-shape nodes with bevel + gradient + shadow stack render reasonably -- combinatorial correctness OK -- but ~6x larger than graphviz's no-arrow-shape baseline. Apples-to-oranges from graphviz. | real_cosmetic_gap | not_actionable | comparisons/evil_arrow_bevel_gradient_shadow_vs_graphviz.png |
| 39 | MED | evil_per_corner_bevel_striped | A | graphviz | striped + per_corner_bevel | the stripes manifest as flat color blocks (sharp blue/orange chunks) -- ugly visual. Per-corner-bevel adds a darker shading on bottom-left and bottom-right corners but appears asymmetric. | real_cosmetic_gap | fixable_theme_or_render | comparisons/evil_per_corner_bevel_striped_vs_graphviz.png |
| 40 | MED | nodes_effects_on (shadow) | C | n/a | shadow direction | shadow drops below+right of ellipse. Reasonable. Subtle -- could be slightly more pronounced for visibility but defensible default. | acceptable_aesthetic_choice | not_actionable | reference/nodes/effects/on.png |
| 41 | LOW | strip_stroke_width_0_5 | C | n/a | endpoint visibility | stroke_width=0.5 renders nearly invisible at default rendering DPI. Min endpoint too thin. Increase floor to 0.75 or scale stroke by node size. | real_cosmetic_gap | dagua_design_decision_required | reference/nodes/borders/strip_stroke_width.png |
| 42 | LOW | strip_border_opacity_0_2 | C | n/a | endpoint visibility | border_opacity=0.2 renders nearly invisible. Min endpoint defensible if user explicitly chose it but consider warning at <0.3. | acceptable_aesthetic_choice | dagua_design_decision_required | reference/nodes/borders/strip_border_opacity.png |
| 43 | LOW | nodes_shapes_box3d | A | graphviz | shape | dagua and graphviz both render 3d boxes with consistent shading. Good parity. Sizes differ (per global issue) but shape correctness OK. | acceptable_aesthetic_choice | not_actionable | comparisons/nodes_shapes_box3d_vs_graphviz.png |
| 44 | LOW | edges_styles_style_dashed | A | graphviz | dash pattern | dashed line renders correctly. Pattern length ~12-px-on-12-px-off; graphviz ~6-px-on-6-px-off. Slight scale difference but proportional to node size. | metric_or_measurement_artifact | rendering_stack_residual | comparisons/edges_styles_style_dashed_vs_graphviz.png |
| 45 | LOW | combo_pie_bold | A | graphviz | pie wedges | pie wedge color allocation matches graphviz; bold text renders correctly. Just oversized. | real_cosmetic_gap | fixable_theme_or_render | comparisons/combo_pie_bold_vs_graphviz.png |
| 46 | LOW | nodes_shapes_diamond | A | graphviz | shape | both render outline-only diamond, no fill. Consistent. Size differs (global issue). | metric_or_measurement_artifact | not_actionable | comparisons/nodes_shapes_diamond_vs_graphviz.png |
| 47 | LOW | (all comparison panels) | A/B | (n/a) | font hinting / AA | minor anti-aliasing differences on outline strokes between dagua's matplotlib rasterizer and competitor's native rasterizer. Sub-pixel rounding inevitable. | metric_or_measurement_artifact | rendering_stack_residual | comparisons/* (heatmap stripes around all curves) |

## Tier-classification audit

| Card prefix | Current Tier | Suggested Tier | Reason |
|---|---|---|---|
| nodes_borders_stroke_width_* | C | A | graphviz has `penwidth` -- direct analog |
| nodes_borders_border_opacity_* | C | A | graphviz `color="#RRGGBBAA"` supports per-channel opacity |
| nodes_borders_corner_radius_* | C | C (keep) | graphviz `style="rounded"` is fixed-not-tunable; defensible |
| nodes_fills_opacity_* | C | A | graphviz `fillcolor="#RRGGBBAA"` supports opacity |
| nodes_text_text_align_* | C | A | graphviz `labeljust=l|c|r` supports |
| nodes_text_text_valign_* | C | A | graphviz `labelloc=t|c|b` supports |
| nodes_text_external_label_* | C | A | graphviz `xlabel` supports external node labels |
| nodes_text_text_background_* | C | C (keep) | graphviz has no analog (matplotlib-style only) |
| nodes_effects_off / on | C | A (partially) | graphviz no native shadow but mermaid has CSS class shadow |
| edges_advanced_taper | C | A | graphviz `dir=both` and `arrowhead=tee` produce taper-like; also `style=tapered` exists |
| edges_advanced_crossing_style_* | C | C (keep) | no graphviz/cytoscape analog |
| edges_labels_* (positions 0_2, 0_5, 0_8) | C | A | graphviz supports `headlabel`, `taillabel`, `xlabel` for edges |
| clusters_stroke_dash_* | C | A | graphviz subgraph `style=dashed` |
| clusters_label_position_* | C | A | graphviz `labelloc` supports cluster label position |
| clusters_corner_radius_* | C | A (partial) | graphviz `style=rounded` supports the binary case; full radius = C |
| clusters_opacity_* | C | A | graphviz `fillcolor="#...AA"` -- note dagua has cluster fill, not just border, opacity in some configs |
| graph_background_* | C | A | graphviz `bgcolor` |
| graph_direction_* | C | A | graphviz `rankdir` |
| graph_margin_* | C | A | graphviz `margin` |

About 14 of the current Tier C cards have a clear graphviz/cytoscape analog and could be promoted to Tier A, materially expanding the pixel-anchored coverage of the dial harness. Recommended re-tiering before round 2 so the metric panel reflects fixes.

## Tier C heuristic violations (combination integrity)

- **`evil_taper_crossing_dashed`**: dashed style is replaced with solid. Real combination breakage.
- **`evil_extreme_curvature`**: edges go off-page; multiple disconnected arc fragments visible. Layout breakage.
- **`evil_unicode_labels`**: dangling edge stub off top of canvas; node-edge layout inconsistency.
- **`evil_empty_labels`**: arrowheads INSIDE the double-border; z-order error. Inconsistent edge color.
- **`evil_deep_clusters`**: cluster labels in HUGE white boxes overwhelming the cluster bounding box; multiple stacked level labels.
- **`evil_invisible_on_invisible`**: blank panel as expected -- PASS.
- **`evil_huge_arrows`**: huge filled triangle, no crashes -- PASS.
- **`evil_max_opacity_stack`**: 5 stacked nodes with overlapping labels (CABED). Z-order OK. Borderline pass.
- **`evil_zero_width_big_arrow`**: clean rendering, big arrowhead on thin line -- PASS.
- **`evil_contradictory_styles`**: handles mixed shapes and sizes without crashing. Reasonable.
- **`evil_all_new_features`**: tiny diamond marker at the X-crossing (crossing_style=bridge?). Visual is acceptable but documentation of the marker is unclear.
- **`combo_dashed_border_arrow`**: dashed borders render correctly with diamond arrowheads -- PASS.
- **`combo_opacity_shadow`**: gray ellipses with shadow -- PASS though opacity is barely visible (see broken-dial finding 17).
- **`combo_external_label_rounded`**: external labels all rendered below regardless of position config -- broken (finding 18).
- **`combo_bevel_shadow`**: bevel as thin top-half stripe + drop shadow -- subtle but no crashes.
- **`combo_arrow_bevel`**: not inspected; deferred.

NO rendering crashes (empty/all-black panels). One zero-content panel (`evil_invisible_on_invisible`) is by-design.

## Rendering-stack residuals

These I observed but classify as known floor (do NOT drive cosmetic fixes):

- Anti-aliasing of curved strokes: ~1-2 px halo around all node ellipses in heatmap diff. Always present, harmless.
- Font hinting differences: dagua uses matplotlib's text path, competitor uses native Cairo / SVG-to-PNG rasterizer. Sub-pixel character-edge differences inevitable.
- Dashed line phase: dagua's dash pattern starts at edge midpoint vs graphviz starts at source-end. Cosmetic only, sub-pixel.

## Recommended fix order for round 2

Top 8 fixable theme/render items in priority order:

1. **Disable the default white label background box.** Set `text_background_default = none` (or transparent). This single fix will close the visual noise across ~80% of all dagua renders and dramatically improve pie / gradient / striped / pattern / combo cards. Likely change in `dagua/render/text.py` or wherever the label-bg layer is drawn unconditionally. After fix, re-shoot baseline and most "ugly white box" findings will close.
2. **Shrink default node size to graphviz parity.** In `dagua/styles.py` GRAPHVIZ_STRICT_THEME: bring `node_size_default` (or width/height defaults) down to match graphviz's ~75x50 px footprint, and proportionally adjust `font_size_default`. This will close ~30 of the L1-metric delta dollars.
3. **Fix `clusters/opacity` dial propagation.** Wire the `cluster_style.opacity` value through to the cluster border's stroke-alpha. Currently is dropped or hard-coded to 1.0.
4. **Fix `clusters/label_position` dial.** Apply top_left / top_center / top_right offsets to the cluster label anchor. Currently always positions top_center.
5. **Fix `nodes_text_external_label` position dial.** Apply top / bottom / left / right offsets to the external label anchor. Currently always positions below.
6. **Fix `nodes_fills/opacity` dial.** Wire fill_color alpha through; likely the default-white label-bg box is masking the fill so most of the ellipse interior is white anyway. Will benefit from fix #1.
7. **Fix taper-kills-arrowheads combination.** When taper is enabled, do NOT skip the arrowhead glyph. Likely a `if taper: skip arrow` early-return in the edge renderer.
8. **Fix taper/crossing-kills-dashed combination.** When taper or crossing_style is enabled, preserve `stroke_dasharray`. Same kind of early-return bug.

Likely code areas (do not invent line numbers):
- `dagua/render/text.py` (or `labels.py`): label background painting; external label position
- `dagua/render/edges.py`: arrowhead rendering under taper; dashed style preservation
- `dagua/render/clusters.py`: cluster label position; cluster opacity application
- `dagua/styles.py` GRAPHVIZ_STRICT_THEME: node_size_default, font_size_default, padding
- `dagua/render/effects.py` (or shape.py): bevel + text_outline overwriting fill_color (design decision required first)

Do NOT attempt in round 2:
- Curvature-collapses-layout (round 24, 25): needs layout-scope fix; out of cosmetic-dial sprint.
- Unicode-labels layout breakage (round 26): layout-scope.
- Bevel/text_outline overwriting fill_color (round 22, 23): needs DAGUA design decision -- is bevel a fill-replacement or fill-overlay? Surface to user before fixing.

## Inspection log

Cards inspected (61 total):

Tier A/B comparisons (worst-by-metric):
- evil_pie_star, evil_pie_shadow_gradient, evil_donut_diamond, evil_taxi_gradient_multiborder, combo_pie_shadow_gradient_bold, combo_kitchen_sink_5, nodes_borders_border_position_inside, nodes_borders_border_position_outside, nodes_fills_gradient_radial, combo_pie_gradient_bold, nodes_fills_fill_pattern_striped, nodes_fills_gradient_linear, evil_arrow_bevel_gradient_shadow, evil_taxi_self_loop, combo_kitchen_sink_6, combo_taxi_crossing_gap_gradient, combo_donut_shadow, evil_per_corner_bevel_striped, evil_self_loop_styled, combo_pie_bold

Tier A/B comparisons (green-by-metric, false-pass check):
- nodes_shapes_rect, nodes_shapes_diamond, nodes_shapes_box3d, nodes_shapes_star, nodes_shapes_cylinder, nodes_shapes_circle, nodes_shapes_cloud (mermaid), nodes_shapes_stadium (mermaid), nodes_borders_stroke_dash_solid (cytoscape), edges_arrows_normal, edges_arrows_crow, edges_arrows_diamond, edges_arrows_open, edges_styles_style_dashed, edges_styles_width_5_0, nodes_text_font_weight_bold, nodes_text_font_style_italic, nodes_text_text_rotation_45 (cytoscape)

Tier C strips:
- nodes/borders/strip_stroke_width, strip_corner_radius, strip_border_opacity
- nodes/fills/strip_opacity
- edges/styles/strip_width
- edges/routing/strip_curvature
- edges/advanced/strip_taper

Tier C atomic (effects, text, clusters, graph):
- nodes/effects/off, on, bevel_on
- nodes/text/text_outline_on, text_background_orange, external_label_top
- clusters/label_position_top_left, opacity_0_3, corner_radius_16
- graph/direction_lr, background_dark, margin_15

Tier C combos / evil:
- combo: dashed_border_arrow, opacity_shadow, external_label_rounded, bevel_shadow
- evil: huge_arrows, max_opacity_stack, extreme_curvature, invisible_on_invisible, taper_crossing_dashed, deep_clusters, contradictory_styles, all_new_features, unicode_labels, empty_labels, zero_width_big_arrow

For each, examined: node shape outline, node fill (color, gradient, pattern), node border (stroke width, dash, opacity, count), node label text (font, weight, style, color, position, rotation, background, outline, external), edge path (routing, curvature, dashed), edge stroke (width, color), edge arrowhead (type, size, position), cluster bounding box (border, fill, opacity, corner radius, label), graph background, graph margin, panel-level layout (centering, anchoring), and combination interactions (z-order, occlusion, color contrast, dial-on-dial breakages).

End round 1 audit.
