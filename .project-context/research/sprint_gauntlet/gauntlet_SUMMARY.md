# Graphviz-Drop-In Sprint Chain -- Final Summary

**Period:** 2026-04-30 16:50 to 2026-05-01 09:50 (~17 hours)
**Sprints:** 7 (A through G)
**Implementation rounds:** ~15 across the chain
**Audit rounds:** ~10
**Final verdict:** `ACHIEVED_WITH_DOCUMENTED_RESIDUALS`

dagua is now a graphviz-drop-in replacement at the rendering layer. Remaining residuals are in algo_fidelity territory (layout-engine, parallel sprint scope) or fixture refactor scope (gallery_audit harness, not user-facing).

## Trajectory

Mean Tier A L1 across the chain:

| Stage | Mean Tier A L1 | Backend |
|---|---|---|
| Pre-chain (post-dial-tuning round 12) | 1.701 | Agg |
| Sprint A end (data-coord-everything) | 1.515 | Agg |
| Sprint B end (cairo opt-in) | 1.495 | cairo |
| Sprint C end (autosize + canvas-fit) | 1.217 | cairo |
| Sprint D end (SSIM added; no metric change) | 1.217 | cairo |
| Sprint E end (dash/dot visibility) | 1.218 | cairo |
| Sprint F end (override API) | 1.232 | cairo |
| **Sprint G FINAL** | **~1.232** | **cairo** |

Mean SSIM (perceptual): 0.963 (96.3% structural similarity).

Total drop: 1.701 → 1.232 = **-0.469 (28% reduction)**, with the bulk coming from Sprint A's data-coord refactor and Sprint C's autosize+canvas-fit.

## What shipped (per sprint)

### Sprint A: data-coord-everything (4 implementation + 3 audit rounds)

- All render primitives in `dagua/render/` flow through `_compute_display_scale(ax)`
- Reverted Round-11's display-stroke fallback; replaced with data-coord ribbon + `_MIN_VISIBLE_STROKE_POINTS = 2.3` clamp
- Refactored to `Figure(...)` + explicit `FigureCanvasAgg(fig)` (cairo-ready)
- 7-fixture dpi-invariance regression test (locks calibrate-once)
- Differentiable layout claim is now structurally honest

Commits: a0f9678 (R13), bbd4c97 (R14), 042a73d (R15), 3b701a4 (R16)

### Sprint B: cairo opt-in (3 rounds + 1 audit)

- `pip install 'dagua[cairo]'` extra
- Auto-detect default per the cairo policy: cairo if mplcairo installed, else Agg
- `dagua.render(g, pos, backend="agg" | "cairo")` per-call override
- `dagua.set_default_backend(...)` global override
- `_CAIRO_STROKE_WIDTH_SCALE = 0.86` calibration to match Agg ink density on the data-coord ribbon path
- Comparison gallery infrastructure (`scripts/build_backend_comparison_gallery.py`)
- Smoking-gun finding: cairo fixes broken dashed cluster outlines that Agg renders incomplete

Commits: 5b48e16 (R1), cddbba1 (R2), d5af420 (R3)

### Sprint C: autosize + canvas-fit (3 rounds + 2 audits)

- `NodeStyle.auto_size_to_label: bool` (default False; True in GRAPHVIZ_STRICT_THEME)
- min_width=54pt, min_height=36pt floors (graphviz defaults)
- `dagua.render(g, pos, fit_to_canvas: bool | float = False)` -- canvas-fitting render mode
- Aspect-aware padding for layout-vs-panel mismatch
- `PAIR_SHAPE_COMPARISON_GAP = 110` for shape parity cards
- Closes scale-mismatch residual: shape parity cards from L1 ~3 to L1 < 0.8

Commits: 6d57186 (R1), d13cf02 (R2), 16a7a91 (R3)

### Sprint D: SSIM perceptual metric (1 round)

- SSIM + SSIM_loss columns in `per_card_pixel_diff_summary.md`
- `eval_output/perceptual_divergence_report.md` -- L1-vs-perceptual disagreement
- Identifies L1-blind class (perceptually bad, L1-good) AND metric-noise class (L1-bad, perceptually good)
- Cairo ties Agg on mean SSIM (0.963) -- consistent with cairo round-2 audit's structural-blindness finding

Commit: 4b5a951

### Sprint E: dash/dot edge visibility (2 rounds + 1 audit)

- `_MIN_VISIBLE_STROKE_POINTS` enforced on dashed/dotted ribbon construction
- Arrowhead placement decoupled from dash phase (uses analytic edge-vs-Target intersection)
- Closes the L1-blind defect class identified by Sprint D
- Italic "defect" was a graphviz limitation, not dagua bug -- left alone
- Combo card residuals (workflow fixture layout-scale) flagged as algo_fidelity scope

Commit: b2bac8d

### Sprint F: pixel-unit override API (2 rounds)

- 6 `*_override_points` fields on NodeStyle/EdgeStyle/ClusterStyle
  - `NodeStyle.stroke_width_override_points` / `font_size_override_points`
  - `EdgeStyle.width_override_points` / `font_size_override_points`
  - `ClusterStyle.stroke_width_override_points` / `font_size_override_points`
- All default `None`; setting any bypasses data-coord and routes to display-points
- NOT differentiable -- documented in SCALING.md
- Default behavior preserved (162 tests pass, mean L1 unchanged)
- Round 1 broke 11 tests with collapsed default-path branches; Round 2 fixed

Commits: 5a49390 (R1), d7e5617 (R2)

### Sprint G: final visual gauntlet (1 round)

- Comprehensive visual audit of ~22 cards under cairo + autosize + all calibrations
- Verdict: `ACHIEVED_WITH_DOCUMENTED_RESIDUALS`
- Three cairo WINS over graphviz: graphviz fails on dashed edges, dotted edges, dashed clusters; dagua more correct
- Combo card residuals confirmed as layout-engine scope (algo_fidelity territory, not rendering)
- Two L1=10 outliers (`nodes_borders_border_position_inside/outside`) flagged for future NodeStyle public-API work
- ZERO fixable findings in the rendering layer that respect the locked constants and don't violate algo_fidelity territory

## What dagua looks like now (graphviz drop-in evidence)

1. **API parity at the rendering layer.** A graphviz `[shape=box3d, style=dashed, fillcolor=lightblue]` node + edge specification, when imported via dagua's API with GRAPHVIZ_STRICT_THEME, produces visually-matched output.

2. **Auto-sizing semantics.** `NodeStyle.auto_size_to_label=True` (theme default) computes node W/H from label content + padding, with min_width/min_height as floors. Matches graphviz's dot semantics.

3. **Canvas-fit rendering.** `dagua.render(g, pos, fit_to_canvas=True)` scales the layout to fill the target panel with margin. Matches graphviz's `dot -Tpng -Gsize="X,Y!"` behavior.

4. **Cairo rasterizer parity.** Auto-detect cairo backend uses the same rasterizer family as graphviz (cairo). Sub-pixel AA, font hinting, dashed-stroke completeness all match graphviz where the L1 metric undersells them.

5. **Differentiability preserved end-to-end.** ALL render primitives in data coordinates by default. `dagua.layout()` produces positions; `dagua.render()` rasterizes. The optimizer's manifold is the data-coord space. Override fields are explicit opt-out for paper-figure typography use cases (NOT differentiable, documented).

6. **Calibrate-once-correct-everywhere enforced.** dpi-invariance regression test (7 fixtures) trips automatically on any future PR that introduces display-point leakage.

## Documented principled residuals (out of scope per agreed guardrails)

These remain after the chain but are NOT in cosmetic-tuning scope:

1. **Combo workflow fixture layout-scale** (combo_kitchen_sink_5, combo_pie_gradient_bold, etc. at L1 3.3-3.8): dagua's layout solver produces a ~5x wider extent than graphviz's compact dot output on the same 5-node DAG. This is algo_fidelity / layout-engine territory.

2. **Two L1=10 border-position outliers** (nodes_borders_border_position_inside/outside): NodeStyle default `border_position` semantics differ from graphviz. Future NodeStyle public-API work; not a fix per se.

3. **Sub-pixel rasterizer-stack residual**: even with cairo, there's a small floor (~0.5-1.0 L1 per card) from anti-aliasing pixel-level differences between matplotlib/cairo and graphviz/cairo. Identical primitives, slightly different sub-pixel weighting. Closeable only with a unified rasterizer (skia, custom GPU, etc.).

4. **Mean SSIM_loss ~0.037**: ~96.3% perceptual match. Higher quality is theoretically possible but the bar is "indistinguishable to a human at typical zoom" -- which the auditor confirmed dagua-cairo achieves.

## Commits across the chain (15 implementation + 5 docs + 7 sprint summaries)

Implementation:
```
a0f9678  feat(render): round 13 -- replace thin-edge display fallback
bbd4c97  feat(render): round 14 -- fix linewidth leakages with data-coord ribbons
042a73d  feat(render): round 15 data-coordinate residuals
3b701a4  test(render): round 16 -- defense-in-depth dpi-invariance fixtures
5b48e16  feat(render): cairo backend as opt-in matplotlib alternative
cddbba1  feat(scripts): add cairo comparison gallery metrics
d5af420  feat(render): cairo stroke-weight calibration to match Agg ink density
6d57186  feat(styles): add graphviz strict node auto-sizing
d13cf02  feat(render): canvas-fit render mode for graphviz-equivalent panel rendering
16a7a91  feat(render): close fit_to_canvas aspect-ratio gap on shape parity cards
4b5a951  feat(scripts): add SSIM perceptual metric to per_card_pixel_diff
b2bac8d  fix(render): dashed/dotted edge body + arrowhead visibility at thin widths
5a49390  feat(styles): pixel-unit override fields for non-differentiable opt-in
d7e5617  fix(render): restore default render path after override wiring regressions
```

## Lessons learned

1. **Auditor STOP verdicts can be illusory.** This whole 7-sprint chain happened because at each "ceiling," the user pushed past with "but you literally just told me X is fixable." The dial-tuning sprint had the same pattern. Lesson: when the user's intuition says "more is possible," empirically more usually IS possible.

2. **L1 metric is structurally blind to thin-feature wins.** Sprint D added SSIM specifically to surface this. Without it, Sprint E's dashed/dotted visibility fix would have looked like a 0.001 metric improvement instead of the real visibility win it was.

3. **Architectural sequencing matters.** Sprint A's data-coord refactor was the prerequisite for Sprint B's cairo backend (rasterizer-agnostic artist construction), Sprint C's canvas-fit (uniform scaling without data-coord violations), and Sprint F's override API (orthogonal opt-in). Doing them out of order would have produced cascading rework.

4. **Empirical calibration over theoretical prediction.** Sprint B audit predicted `_CAIRO_STROKE_WIDTH_SCALE=1.15`; codex empirically discovered `0.86`. The codebase's specific data-coord ribbon construction goes a different direction than the auditor's mental model. Trust the sweep.

5. **Codex regression discipline matters.** Sprint F Round 1 broke 11 tests by collapsing default-path branches into the override path. The completeness contract caught it; Round 2 fixed it cleanly. Lesson: test-driven discipline + explicit completeness contracts prevent silent regressions.

## Status

**Sprint chain DONE.** dagua is a graphviz-drop-in replacement at the rendering layer. Remaining quality improvements live in:
- Algo_fidelity sprint (parallel; addresses layout-engine convergence to graphviz semantics)
- Future NodeStyle public-API work (border_position alignment)
- Eventually: a unified rasterizer if sub-pixel AA parity matters more than dependency footprint

For now: ship.
