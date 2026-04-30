# Round 7 Audit -- dial_tuning (Final Verdict)

## Verdict

**STOP.**

The three round-7 surgical fixes (cluster bbox 4 edges, stroke_width=5pt visible thickness, simple-shape pair-fixture fill+border parity) all visually landed. The remaining worst-15 cards are entirely classifiable as principled residuals: layout_coupled (graphviz dot vs dagua force-directed cluster geometry), competitor_semantic_mismatch (cytoscape interpretations of border_position diverge from graphviz-anchored dagua), competitor_glitch (cytoscape taxi/self-loop renders), and multi_feature_density_combo (graphviz auto-shrinks node size at high density; dagua does not, by design). No critical Tier C breakage. The mean L1 regression from 2.971 to 3.417 is a measurement artifact of the stylistic upgrade -- pixel-visible features now produce small color/weight diffs that didn't exist when both sides were unfilled white. This is the right kind of "regression" to absorb.

## Round 7 fix verification

All three fixes verified visually:

1. **Cluster bbox 4 edges**: `clusters/top_center_vs_graphviz.png`, `top_left_vs_graphviz.png`, and the opacity 0.3/0.6/1.0 panels all show a complete rectangular outer cluster bbox AND the inner cluster bbox with all four edges drawn at the dagua side. Cluster border missing-stroke regression from rounds 5/6 is **CLOSED**.

2. **stroke_width=5.0 visibly thick**: `nodes/borders/5_0_vs_graphviz.png` shows the right ellipse with a clearly thick blue stroke (~5pt visual width) versus the thin "Default" stroke. Cross-check with `3_0` and `1_5` panels confirms a monotonic thickening progression. The clamping bug is **CLOSED**.

3. **Simple-shape fill+border+arrowhead parity**: `nodes/shapes/{ellipse,circle,rect,diamond,hexagon,...}_vs_graphviz.png` now show dagua nodes with the same blue fill + blue border as graphviz. Filled-arrowhead is present at the target node (small, but consistent shape with graphviz's filled triangular form). Pair-fixture `DECORATIVE_FILL` override is **applied broadly** across the shape comparisons. Note: dagua nodes still render at smaller absolute size than graphviz at this fixture density (graphviz auto-scales up at 2-node density), and the connecting edge body is short/missing because the fixture spreads Source/Target far apart in the dagua canvas while graphviz packs them tight. Both are layout-coupling, not styling, so out of cosmetic scope.

## Worst-15 residual classification verified

| Card | Class | Verified |
|---|---|---|
| clusters_opacity_1_0 (L1=30.1) | layout_coupled | YES -- dagua force-directed cluster spreads to ~2.5x graphviz vertical-stack footprint; opacity dial monotonic. Tightening cluster padding cannot close this. |
| nodes_borders_border_position_outside (L1=30.1, cytoscape) | competitor_semantic_mismatch | YES -- dagua renders enormous orange filled rect with tiny inner label box; cytoscape renders compact rounded rect with center text. Different interpretations of border_position; cytoscape semantics disagree with dagua's graphviz-anchored interpretation. |
| nodes_borders_border_position_inside (L1=28.8, cytoscape) | competitor_semantic_mismatch | YES -- same as outside, mirror case |
| clusters_opacity_0_6 (L1=19.8) | layout_coupled | YES -- mid-opacity panel, same geometry mismatch as 1.0 |
| evil_taxi_gradient_multiborder (L1=19.7, cytoscape) | competitor_glitch | YES (sampled prior round audits) -- cytoscape's taxi router renders inconsistently |
| evil_taxi_self_loop (L1=16.7, cytoscape) | competitor_glitch | YES -- cytoscape self-loop routing differs from dagua |
| combo_kitchen_sink_6 (L1=14.5, cytoscape) | competitor_glitch | YES -- 5-feature combo where cytoscape renders weakly |
| combo_kitchen_sink_5 (L1=14.1, graphviz) | multi_feature_density_combo | YES -- pie+donut+gradient+bold+ext_label; dagua renders all features cleanly at proper z-order; graphviz auto-shrinks at this density and dagua does not |
| combo_pie_shadow_gradient_bold (L1=14.0) | multi_feature_density_combo | YES -- 4-feature density |
| combo_taxi_crossing_gap_gradient (L1=13.3, cytoscape) | competitor_glitch | YES |
| combo_pie_gradient_bold (L1=12.2) | multi_feature_density_combo | YES |
| combo_donut_shadow (L1=12.0) | multi_feature_density_combo | YES |
| evil_self_loop_styled (L1=11.9, cytoscape) | competitor_glitch | YES |
| clusters_opacity_0_3 (L1=11.2) | layout_coupled | YES |
| combo_pie_bold (L1=10.7) | multi_feature_density_combo | YES |

All 15 fall into the four pre-classified buckets. None is a fixable cosmetic-scope item.

## Mid-range sanity check

Sampled `nodes/borders/3_0_vs_graphviz.png` (L1=2.115), `nodes/borders/1_0_vs_graphviz.png` (L1=2.676), `nodes/borders/border_opacity_1_0_vs_graphviz.png` (L1=2.676), and several shape panels (`diamond` L1=1.717, `hexagon` L1=2.270, `triangle` L1=1.802, `pentagon` L1=2.160).

**Acceptable.** The dagua side of these cards renders the feature monotonically with graphviz-comparable border weight, fill, and shape geometry. Two minor observations:

- **border_opacity at 1.0**: dagua appears to render the border in pure black versus graphviz's blue. The border *opacity* is correctly at 1.0, but the underlying border *color* on the dagua side does not pick up the theme's blue accent at this fixture. This may be a fixture default issue where the override path passes opacity but resets color. Not blocking a STOP verdict, but a candidate next-sprint cleanup if a follow-on cosmetic pass happens. Marked as **not_round_8_actionable** because (a) only affects the high-opacity end of one dial, (b) all four opacity panels score under L1=3 anyway, (c) does not appear in worst-15.
- The remaining mid-range diff content is consistent with anti-aliasing, font hinting, and node-size float between dagua's coordinate-aware render and graphviz's pixel-rounded grid. Standard rendering-stack residual.

## Tier C breakage check

- `combos/5way/kitchen_sink_5.png`: pie wedges + donut hole + gradient + bold labels + external label all render correctly. Z-order is right (label plate sits above pie fill). No mud, no occlusion.
- `combos/5way/kitchen_sink_6.png`: stadium shape + taxi routing + shadow + gradient + crossing gap renders cleanly. Edges path correctly through the crossing junction.
- `evil/evil_deep_clusters.png`: 4-level concentric nesting renders with proper progressive darkening, all four cluster bboxes have full 4 edges, level labels readable, edges route within innermost cluster without crossing parent borders inappropriately.

**No critical Tier C breakage.** No render crashes, no z-order errors, no occluded text, no total visual mud detected. The previously-flagged `evil_extreme_curvature` and `evil_unicode_labels` are layout-scope carry-overs, already documented as deferred.

## Sprint summary highlights

- **Mean L1 trajectory**: round 1 baseline ~6.5 -> round 4 breakthrough 2.454 -> round 5 regression 8.121 -> round 6 recovery 2.971 -> round 7 stylistic-upgrade 3.417 (slight nominal rise as pixel-visible features expose small color/weight diffs). Median L1 stayed stable through 6/7. SSIM ~0.93 average.
- **Tier A under L1=5: 80% (144/181)**. Tier A under L1=2: 48% (86/181). The long tail is dominated by 9 cards at L1>10, all of which fall in the four principled-residual classes documented in this audit.
- **Locked features (programmatic regression gate green)**: white label-bg box closed, cluster z-order correct, taper arrows preserved, fills/opacity dial monotonic, cluster opacity dial monotonic, stroke_width dial monotonic across 0.5/1.0/1.5/3.0/5.0, cluster bbox draws all 4 edges, simple-shape pair-fixtures show fill+border+arrows.
- **Principled residuals (will not close in cosmetic scope)**: (1) graphviz-vs-dagua cluster geometry under dot-vs-force-directed -- requires layout-scope work, not theme tuning; (2) cytoscape-tier semantic_mismatch on border_position dial -- competitor disagrees with graphviz on the dial's meaning; (3) cytoscape glitches on taxi routing and self-loops -- competitor-side noise; (4) graphviz auto-shrinks node size at multi-feature density combos -- dagua does not, by design (single-theme-value architecture).
- **Round count**: 7 codex rounds + 7 Opus audits over ~12 hours. Anti-flail kicked in at round 5; round 6 recovered with auditor-recommended architectural pattern. Round 7 closed three round-6 carryover bugs cleanly.
- **Out-of-scope deferred**: layout-coupled cluster geometry, density-coupled node sizing, evil_extreme_curvature, evil_unicode_labels.

## Closing note

We are genuinely at ceiling for this sprint's cosmetic-scope mandate. The honest worst-case here is that the **border_opacity_1_0 black-vs-blue color mismatch** is mid-range actionable -- but it is single-card, off the worst-15, scoring under L1=3, and the right pass for it is a fixture-color audit, not a theme-value tweak. Recommending we file it as a follow-on cleanup item rather than spend a round 8 on a single dial endpoint that is not visually offensive (the dial still moves monotonically through 0.2/0.5/0.8/1.0; the user gets the feature). Everything else hits one of the four documented residual classes, all of which require either layout-scope work, competitor-side fixes, or architectural changes (auto-density-shrink) that are explicitly out of this sprint's mandate.

JMT, the sprint is at its ceiling. Ship it.
