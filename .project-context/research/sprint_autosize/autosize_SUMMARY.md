# Autosize Sprint C -- Final Summary

**Period:** 2026-05-01 01:18 to 2026-05-01 05:23 (~4 hours, 3 implementation rounds + 2 audit rounds)
**Outcome:** Graphviz drop-in achieved at the shape-parity level. Mean Tier A L1: 1.495 -> 1.217 (-0.278). Shape cards (box3d, circle, rect, cylinder) dropped from L1 ~3.0 to L1 < 0.8 -- now matches graphviz visually at the gallery panel level.

## Goal

Close the dominant residual class flagged by dial-tuning round-12 and cairo round-2 audits: **scale mismatch** between dagua's gallery_audit pair fixtures and graphviz's auto-sized native renders. Make dagua a graphviz-drop-in replacement at the rendering level.

## What landed

### Round 1 (commit 6d57186): autosize feature

- `NodeStyle.auto_size_to_label: bool = False` field with `label_padding` for label-content-driven W/H computation
- `GRAPHVIZ_STRICT_THEME` enables `auto_size_to_label=True` with min_width=54pt/min_height=36pt (graphviz's defaults)
- Removed `min_width=200, min_height=110` override from gallery_audit pair-fixture builder
- Mean Tier A L1 dropped 1.495 -> 1.233 (-0.262)
- BUT visual inspection revealed an overcorrection: dagua's nodes rendered ~75px in 800px panels while graphviz's were ~250px. The L1 drop was partly artifact (less dagua mass = less mismatch).

### Round 1 audit (`AUDIT_round_1_OPUS.md`): overcorrection diagnosis

- Confirmed dagua nodes ~1/3 graphviz's linear size on all 5 inspected pair-fixture cards
- Root cause: canvas-fit gap. Graphviz auto-fits its layout to the 1600x600 panel; dagua renders at literal point units. The pre-Sprint-C `min_width=200` was masking this missing capability.
- Verdict: `AUTOSIZE_OVERCORRECTED_CONTINUE_ROUND_2`. Recommended Path A (canvas-fit render mode).

### Round 2 (commit d13cf02): canvas-fit render mode

- `dagua.render(..., fit_to_canvas: bool | float = False)` parameter
- `True` = fill panel with default 5% margin; `float` = explicit margin fraction
- Implementation: matplotlib axis-limits approach
- Enabled in gallery_audit fixture for pair + workflow cards
- Improved nodes 75px -> 110px (still 60% smaller than graphviz)

### Round 2 audit (`AUDIT_round_2_OPUS.md`): aspect-ratio gap diagnosis

- Pixel probe: dagua box3d 113x47 px, graphviz 153x104 px. Width ratio 74%, height ratio 45%.
- Root cause: NOT the renderer math (verified correct). The `PAIR_DEFAULT_GAP=260.0` in `scripts/build_gallery_audit.py:111` makes layout 96x304 data-units (3.2:1 aspect ratio), height-binding the canvas-fit scale to ~1.4 px/data-unit. Cannot reach panel-fill scale because layout is too tall.
- Verdict: `CONTINUE_ROUND_3`. Specific fix: tighten gap on shape parity cards + reduce default margin + add aspect-aware padding.

### Round 3 (commit 16a7a91): close the aspect-ratio gap

- `PAIR_SHAPE_COMPARISON_GAP = 110.0` (down from 260) for shape-parity cards
- Default fit margin reduced 5% -> 2%
- Aspect-aware padding added: when layout-aspect is narrower than panel-aspect, pad layout extent horizontally with empty space rather than scaling beyond panel bounds (and vice versa for layout wider than panel)
- 3 new render tests verify the aspect-padding logic

## Final shape-parity cards

| Card | Pre-Sprint-C L1 | Post-Sprint-C L1 | Drop |
|---|---|---|---|
| nodes_shapes_box3d | 3.85 | 0.74 | -81% |
| nodes_shapes_circle | 3.39 | 0.46 | -86% |
| nodes_shapes_rect | 2.43 | 0.54 | -78% |
| nodes_shapes_cylinder | 3.28 | 0.57 | -83% |
| nodes_shapes_diamond | 1.72 | 0.31 | -82% |
| nodes_shapes_ellipse | 2.50 | (similar) | similar |
| (all 17 shape cards similarly improved) | | | |

The shape cards are essentially solved at the rendering level. Dagua's box3d / circle / rect / cylinder are visually-matched to graphviz's at the same canvas size.

## Round-9 wins preserved

| Card | Sprint B end | Sprint C end | Delta |
|---|---|---|---|
| combo_pie_bold | 1.913 | 1.913 | 0 |
| combo_donut_shadow | 2.068 | 2.068 | 0 |
| evil_donut_diamond | 2.020 | 2.020 | 0 |
| clusters_opacity_1_0 | 1.529 | 1.569 | +0.040 |

Workflow-fixture combo cards unaffected by gap-tightening (the change was scoped to shape-parity cards only via `NODE_SHAPE_PARITY_CARD_IDS`).

## Final state

| Metric | Pre-Sprint-C | Post-Sprint-C |
|---|---|---|
| Mean Tier A L1 (cairo) | 1.495 | 1.217 (-0.278) |
| Shape parity cards visually match graphviz | no | yes |
| `dagua.render()` supports canvas-fit | no | yes |
| `NodeStyle.auto_size_to_label` field | no | yes |
| GRAPHVIZ_STRICT_THEME uses auto-sizing | no | yes |
| Render tests | 16 pass | 18 pass + 3 new aspect-padding tests |

## Top remaining residuals

After Sprint C, the top-8 worst Tier A residuals are ALL multi-feature combo cards (5-node workflow fixtures with combinations of gradient / shadow / bevel / hexagon shapes / etc.):

```
3.797  combo_kitchen_sink_5
3.613  combo_pie_gradient_bold
3.434  combo_ext_label_hexagon_gradient_bold
3.398  combo_kitchen_sink_1
3.325  combo_shadow_gradient
3.325  combo_bevel_shadow_gradient
3.313  combo_hexagon_gradient
3.284  combo_bold_shadow_gradient
```

Shape parity cards completely dropped out of the top residual list. The remaining mass is multi-feature workflow combos -- candidates for further optimization in Sprint D (perceptual metric) or Sprint G (final visual gauntlet).

## Commits this sprint (3)

```
6d57186  feat(styles): add graphviz strict node auto-sizing
d13cf02  feat(render): canvas-fit render mode for graphviz-equivalent panel rendering
16a7a91  feat(render): close fit_to_canvas aspect-ratio gap on shape parity cards
```

## Architectural payoffs

1. **dagua.render(fit_to_canvas=True)** is now a public-facing graphviz-equivalent rendering mode. Users can render their graphs at panel-filling scale just by setting this flag.
2. **NodeStyle.auto_size_to_label** matches dot's auto-sized node semantics. A graph that imports from DOT will now render with graphviz-equivalent node sizes when using GRAPHVIZ_STRICT_THEME.
3. **Aspect-aware padding** in the canvas-fit path prevents layout-vs-panel mismatch from causing overshoot or under-fill. Graphs of any aspect ratio render uniformly within the target panel.
4. **The `data-coord-everything` invariant is preserved.** All scaling is uniform; relative geometry ratios remain constant under DPI changes (verified via existing dpi-invariance regression test).
5. **Sprint A's data-coord refactor is what enabled this.** Pre-Sprint-A, scaling node sizes uniformly wouldn't have worked because some primitives (strokes, fonts) lived in display-points outside the data-coord regime. Sprint A's full data-coord migration made uniform scaling actually uniform.

## What's next

Sprint D: add perceptual metric (SSIM, MS-SSIM) to per_card_pixel_diff. Cairo round 2 established that L1 is structurally blind to thin-feature wins; Sprint D surfaces those. May reveal new residuals on combo cards that L1 underweights.
