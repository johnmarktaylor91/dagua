# Sprint C Round 1 -- Opus Audit (autosize)

**Auditor:** Opus 4.7 (1M ctx), maximum strictness
**Commit under audit:** 6d57186 (autosize feature + GRAPHVIZ_STRICT_THEME w/ min_width=54pt, min_height=36pt; removal of `min_width=200, min_height=110` override from gallery_audit pair-fixture builder)
**Headline metric:** Tier A mean L1 1.495 -> 1.233 (-0.262)

## Verdict

**`AUTOSIZE_OVERCORRECTED_CONTINUE_ROUND_2`**

The L1 drop is real but **misleading** as a quality signal. Visual parity is markedly *worse* than it was pre-Sprint-C. Across all 5 pair-fixture cards inspected, dagua's nodes are roughly **1/3 the linear size** of graphviz's. Combo cards (round-9 wins) regressed: labels are unreadable squiggles. The metric improved because dagua now puts less pixel mass on the canvas, so there's less to mismatch with graphviz's larger nodes — classic "smaller-everything = smaller-diff" artifact described in the brief's hypothesis.

## Part 1 -- Pair fixture inspection (5 cards)

All five inspected at 1600x600 panel pair (800px per panel side, RENDER_DPI=100, so each panel is 8" x 6").

| # | Card | Dagua node size vs graphviz | Classification |
|---|------|-----------------------------|----------------|
| 1 | nodes/shapes/box3d_vs_graphviz | Dagua "Source"/"Target" boxes ~75x50 px; graphviz ~250x130 px. Dagua label is ~7pt and barely legible; graphviz label is ~24pt and crisp. The 3D depth offset on dagua is 1-2 px and visually invisible; graphviz's is ~12 px and clearly read as a 3D box. Edge stem on dagua is a thin smudge ~2 px tall; graphviz's stem is ~80 px with a clear arrowhead. | **Markedly smaller (over-correction)** |
| 2 | nodes/shapes/circle_vs_graphviz | Dagua circles ~55 px diameter; graphviz ~250 px. Dagua text is ~7pt; graphviz text is ~30pt. Edge stem visible in graphviz (~70 px); near-invisible in dagua. | **Markedly smaller (over-correction)** |
| 3 | nodes/shapes/rect_vs_graphviz | Dagua rectangles ~75x35 px; graphviz ~260x110 px. Dagua label ~7pt; graphviz ~26pt. Stroke width disparity: graphviz's stroke reads as substantial; dagua's is so thin relative to scale that the rectangles look almost weightless. | **Markedly smaller (over-correction)** |
| 4 | nodes/shapes/cylinder_vs_graphviz | Dagua cylinders ~75x40 px; graphviz ~260x130 px. Cylinder ellipse caps are barely distinguishable on dagua side at this scale; on graphviz they are clearly the defining feature of the shape. | **Markedly smaller (over-correction)** |
| 5 | nodes/fills/radial_vs_graphviz | Dagua ellipses ~70x35 px; graphviz ~250x100 px. The radial gradient is *technically* present on dagua but at this size the orange-to-blue transition is compressed into ~30 px and reads as muddy/aliased. On graphviz the gradient has room to breathe and reads cleanly. | **Markedly smaller (over-correction)** |

**Summary:** 5/5 cards show consistent ~3x linear size disparity. This is not noise. It is not "could go either way." It is a uniform under-sizing of dagua nodes versus the graphviz reference at the same canvas size. The Source/Target label is so small in dagua's output that fixture-level QA-by-eye is functionally impossible — you cannot tell whether the *style* renders correctly because the node is too small to inspect.

## Part 2 -- Root cause diagnosis

I read `scripts/build_gallery_audit.py` lines 75-110:

```python
PANEL_HALF_WIDTH = 800        # pair fixture panel pixel width
PANEL_HEIGHT = 600
RENDER_DPI = 100              # dagua side
CARD_DPI = 200                # combo / single-graph cards
PANEL_FIGSIZE = (8.0, 6.0)    # inches at 100 DPI -> 800x600 px
```

And graphviz invocation line 2309: `["dot", "-Gdpi=200", "-Tpng", ...]` — graphviz renders at **200 DPI** with auto-fit to the rendered bounding box, then PIL pastes it into the 800x600 panel (which the panel-compose code scales up to fill the panel content area).

So:

- **Dagua** at min_width=54pt, min_height=36pt: 54/72 in = 0.75 in × 100 DPI = **75 px wide**, 50 px tall. A node literally 75 px wide on an 800 px panel = **9.4% of panel width**. The 5/5 measurements above match this exactly (within a few px of stroke).
- **Graphviz** at the *same authored* width=0.75, height=0.5 (graphviz default): graphviz computes its own canvas, runs auto-fit, exports at dpi=200 → the resulting PNG is auto-scaled by PIL to fit the panel. Net effect: the graphviz rendering fills ~30-35% of panel width regardless of authored point size. The graphviz panel doesn't preserve absolute point dimensions; it preserves *relative bounds* and auto-fits to the panel.

This is **the canvas-fit gap**, not a floor-value problem. graphviz's rendering pipeline includes a "fit to canvas" step; dagua's does not. The pre-Sprint-C `min_width=200pt` override was *masking* this by inflating the dagua side until its literal-point output coincidentally matched graphviz's auto-fit output. Sprint C removed the mask without adding the missing capability.

Verifying the "floor too small?" hypothesis: graphviz's authored default IS 54x36pt (height=0.5in × 72 = 36pt; width=0.75in × 72 = 54pt). Dagua's strict theme matches graphviz's authored value exactly. The disparity is not in the floor; it's in the auto-fit step graphviz applies *after* layout but before output. **Bumping the floor value (Path B) would un-match graphviz's authored defaults and introduce a different kind of inaccuracy** — it would only "work" by coincidence at the gallery's specific 800x600 panel size and would mismatch at any other canvas size.

## Part 3 -- Combo card regression check (round-9 wins)

| # | Card | Status |
|---|------|--------|
| 6 | combo_pie_bold_vs_graphviz | **REGRESSED.** Dagua side renders 5 nodes at the *combo* fixture's density-aware-shrunk size, but on the 1600x1200 card surface they're tiny enough that the labels ("Ingest", "Validate", "Review", "Approve", "Ship") are unreadable. Compare graphviz side: nodes are larger and labels crisp. The 5-node workflow fixture was supposedly unaffected by autosize, but it's clearly now feeding through the same too-small pipeline. |
| 7 | combo_donut_shadow_vs_graphviz | **REGRESSED.** Same pattern — graphviz nodes legible, dagua nodes are tiny ovals with illegible label text. Donut center hole and shadow detail compressed into so few pixels they read as visual noise. |
| 8 | clusters_opacity_1_0_vs_graphviz | **REGRESSED.** Dagua's "Outer A", "Inner B", "Inner C", "Outer D" labels are squiggles at this scale; graphviz's are clean. Cluster geometry (Outer/Inner nesting) is preserved structurally, but the embedded nodes are too small to read. |

The brief's claim that combo cards "should be unaffected by autosize" does not hold. Either (a) the workflow fixture *also* lost a sizing override that wasn't called out in the commit, or (b) the density-aware-shrink logic uses the new min_width as its starting point and shrinks from there — so a smaller starting point produces smaller final nodes. Either way, combo regression is real and is its own bug needing investigation in Round 2.

## Part 4 -- Path forward recommendation

**Path A: add canvas-fitting render mode.** Recommended.

A `fit_to_canvas: bool | tuple[w_in, h_in]` parameter on `dagua.render(...)` that, after layout, computes the bounding box of all nodes (including stroke/halo padding) and applies a *uniform* scale + translation so the bbox fills the target canvas with a configurable margin. Properties:

- **Preserves data-coord-everything:** uniform affine scale is the only operation; no display-points anywhere.
- **Preserves DPI invariance:** the scale is computed in data units, applied in data units, and the same DPI rules apply post-scale.
- **Preserves differentiable layout:** scaling is a downstream render-time operation, not a layout-time operation. Gradients flow through layout exactly as before.
- **Matches graphviz default behavior:** graphviz auto-fits by default, and that's the right "drop-in replacement" semantics.
- **Optional and off-by-default in the public API:** core API stays at literal-point rendering; gallery_audit fixture and any other "compare against graphviz" use case opts in.

Implementation sketch (one rendering op insertion, ~30 LOC):

1. After layout returns positions, compute `(min_x, min_y, max_x, max_y)` over all node bboxes (incl. stroke).
2. Compute scale = min(canvas_w / bbox_w, canvas_h / bbox_h) × (1 - margin_frac).
3. Multiply all node positions, sizes, and edge routing waypoints by `scale`. Translate to center in canvas.
4. Emit at the original canvas size.

Enable in gallery_audit pair-fixture and combo-fixture rendering paths. Keep min_width=54pt for the strict theme (correct as graphviz-authored default).

**Why not Path B (tune up floor values):** Loses graphviz API equivalence. A user who specs a graphviz-style `[width=0.75]` would get a 54pt node from graphviz and a ~150-200pt node from dagua's strict theme. That's a worse drop-in story than "graphviz auto-fits, dagua doesn't (but you can opt into auto-fit)." Path B is also panel-size-specific: 150pt fits 800-px panels but mis-fits other sizes.

**Why not Path C (per-fixture override rollback):** It's the path of least progress. The original 200pt override was always a hack, and re-adding it papers over the issue without giving the public API the missing canvas-fit capability that graphviz users actually expect. Defer this only as a temporary band-aid if Path A turns out to take longer than one round to land.

## Part 5 -- Recommended Round 2 scope

1. Implement Path A: `dagua.render(..., fit_to_canvas=...)` with uniform-scale auto-fit.
2. Add canvas-fit unit tests: bbox respects margin, aspect preserved when canvas aspect != bbox aspect, identity when bbox already matches canvas.
3. Re-render gallery_audit pair fixtures with `fit_to_canvas=True` enabled.
4. Re-render combo cards (round-9 wins) with `fit_to_canvas=True` enabled (or whatever sizing path they use, *with* an investigation of why they regressed when autosize was a node-level change).
5. Re-measure Tier A L1. Expectation: L1 should *increase* slightly versus the artificially-low 1.233 (more pixel mass → more potential for mismatch), then converge somewhere between the pre-Sprint-C 1.495 and the artifact-low 1.233. The "real" improved L1 is the post-canvas-fit number.
6. Visual re-audit (Opus, max strictness) before declaring Sprint C done.

## Hard guardrails honored

No recommendation touches:
- `_PATTERN_FILL_RESOLUTION`, `_HATCH_PATTERN`, `_MIN_HATCH_LINEWIDTH_POINTS`
- `_PIE_GRADIENT_OVERLAY_ALPHA_MULTIPLIER`, `_DENSITY_LABEL_FONT_FLOOR`
- `_MIN_VISIBLE_STROKE_POINTS`, `_CAIRO_STROKE_WIDTH_SCALE`
- `density_aware_size_factor()`
- algo_fidelity territory

The recommended canvas-fit op is a rendering-side scale-and-translate, fully outside fidelity scope.

## Confidence

High. The disparity is uniform across 5/5 pair fixtures and 3/3 combo fixtures, the root cause is mechanically reproducible from the gallery_audit script's RENDER_DPI/PANEL math, and the L1-drop-as-artifact hypothesis is consistent with both the visual evidence (less dagua mass) and the metric direction (better number, worse picture).
