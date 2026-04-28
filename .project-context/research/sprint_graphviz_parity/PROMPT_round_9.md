<task>
Round 9 of cosmetic parity work for `graphviz_strict` theme. Round 7 (commit aa6f616) introduced 4 regressions while landing 3 wins. Round 9 closes the regressions ONLY — narrow scope, surgical fixes.

Read these for context:
- `.project-context/research/sprint_graphviz_parity/AUDIT_round_8_OPUS.md` — the round-8 verification audit driving this round
- `.project-context/research/sprint_graphviz_parity/REPORT_round_7.md` — what you did last round, including the deviations

SCOPE: COSMETIC RENDERING ONLY for `graphviz_strict` theme. No layout changes. Single commit at end. develop branch.

THE FIX LIST (5 items, all bounded)
====================================

### F1 — Node label visual font size REGRESSION (HIGH)
Round-8 audit measures node labels at ~70% of dot's cap-height across single_edge.png, tiny_graph.png, diamond.png, pipeline.png, colors_showcase.png. Despite `font_size=12.0pt` in graphviz_strict, the rendered visual is too small.

Two possible root causes:
- (a) The DPI normalization assumption (matplotlib default 100 dpi vs dot's 96) is wrong on this machine — actual matplotlib effective dpi may be different. Empirically: render a simple 1-node graph at the same canvas size in dot vs dagua, measure label cap-height in pixels. If dot's "Foo" cap-height is N px and dagua's is M px, the size scaling factor is N/M. Apply that to graphviz_strict's font_size (e.g. if N=20, M=14, font_size = 12 * 20/14 ≈ 17pt).
- (b) Matplotlib is rendering at dpi != 96 because of figsize/canvas math elsewhere in the render pipeline.

Fix path:
1. Render `dot` and `dagua` versions of a single-node graph with label "Test" at the same canvas size.
2. Measure label cap-height in pixels in each image (use PIL bbox or eyeball).
3. Compute the scaling factor and update graphviz_strict's `font_size` (and `EdgeStyle.label_font_size` and `GraphStyle.edge_label_font_size`) so the rendered visual matches.

Don't blindly raise to 14pt — measure first.

VERIFY: pipeline.png, diamond.png, colors_showcase.png — node labels should now visually match dot's size.

### F2 — `crow` arrowhead REGRESSION (HIGH, audit FAIL)
Codex round 7 reported `crow` was changed to filled, with `dot -Tsvg` cross-check confirming dot emits `fill="black"`. But the rendered round-7 arrow_types.png MIDDLE panel still shows crow as a HOLLOW V chevron (identical to vee). The fix didn't take visual effect.

Investigate:
1. Render arrow_types.png from current code and read it. Confirm crow IS still hollow.
2. Trace the rendering path: theme → ARROWHEAD_REGISTRY → arrow primitive → fill flag → matplotlib patch.
3. Find where the fill flag is being lost. Common suspects:
   - The "filled" flag in arrowheads.py is set but `arrow_fill` from EdgeStyle is overriding it
   - A late-stage path simplification is dropping the fill
   - The crow primitive emits a stroke-only path even when fill is requested
4. Add a print probe at the point where the crow patch is constructed; render and confirm the fill flag value.
5. Fix the disconnect.

VERIFY: arrow_types.png crow column should show a small filled black triangle/spear, matching dot's panel.

### F3 — Edge body stroke width REGRESSION (HIGH)
Round-8 audit measures edge body stroke at ~0.5px on tiny_graph.png and others — round-6 audit had this as PASS at 0.75pt. Something dropped it.

Find: `EdgeStyle.width` for graphviz_strict default edge style. Currently might be 0.75 in code but rendering thinner. Possible causes:
- Round 7's arrow tip-to-boundary trim (F6) altered the stroke width through some shared path
- The width is being applied at a different unit scale post-trim
- A separate "hairline" mode was introduced

If the value is 0.75 in code but visually thin, increase to 1.0 OR find the regression. If the value got changed, restore to 0.75 (round-6 PASS value) or higher to match dot visually.

VERIFY: tiny_graph.png, single_edge.png, pipeline.png — edge body stroke should match dot's hairline visually (count pixels: dot is typically 1-2px wide).

### F4 — Arrow proportions REGRESSION (MEDIUM-HIGH)
Round-6 audit said M2 (arrow proportions 8×8) was PASS. Round-8 audit says proportions regressed and now look slimmer/more pointy than dot. Likely cause: round-7 F6 (ellipse-boundary trim) computed new arrow positions that effectively shrunk the visible arrow head, OR the `arrow_length`/`arrow_width` values were changed.

Check:
1. `EdgeStyle.arrow_length` and `EdgeStyle.arrow_width` in graphviz_strict default — what are the current values?
2. Did F6's tip-trim logic change how these are interpreted (e.g. arrow_length is now measured from boundary not tip)?
3. Visually: arrow_types.png and pipeline.png — are arrow heads stout/equilateral like dot's?

Fix path: ensure that with current arrow_length/arrow_width, the rendered arrowhead VISUAL footprint matches dot's. May need to bump arrow_length and arrow_width up to compensate for any trim adjustment.

VERIFY: arrow_types.png and pipeline.png — arrow heads should look chunky/equilateral, not pointy/slim.

### F5 — Cluster border darkness REGRESSION (MEDIUM)
Round-6 audit said M3 (cluster border #AAAAAA → #CCCCCC, stroke 0.8 → 0.5) was PASS. Round-8 audit now reports cluster strokes appear too dark. Possible causes:
- The border_opacity field isn't being applied correctly
- Stroke color was changed inadvertently
- A cluster z-order issue is overlaying strokes

Check:
1. Current values: cluster `stroke`, `stroke_width`, `border_opacity` in graphviz_strict ClusterStyle.
2. Render nested_clusters.png and read it — are strokes lighter (matching dot) or darker?
3. If darker: bump stroke to a lighter gray (#DDDDDD) OR reduce border_opacity slightly (0.8) to soften.

VERIFY: nested_clusters.png, deep_nesting_4.png, cluster_showcase.png — cluster strokes should look like ghost-thin light gray, matching dot.

DO NOT TOUCH: `improved` graphviz theme, dagua/layout/, scripts/graphviz_theme_comparison.py.
</task>

<completeness_contract>
Not done until:
1. F1-F5 implemented and visually verified by reading the regenerated panels.
2. `pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"` passes for in-scope files. Pre-existing layout-import errors still out of scope.
3. Re-render gallery: `python scripts/graphviz_theme_comparison.py --output-dir eval_output/graphviz_theme_round_9`.
4. Re-crop to 2-way: same script as round 7 prompt.
5. Visually verify: read pipeline.png, arrow_types.png (crow column!), tiny_graph.png, nested_clusters.png. Confirm each fix landed visually.
6. ONE commit: `feat(theme): graphviz_strict cosmetic round 9 — close round-7 regressions (font size, crow fill, edge stroke, arrow proportions, cluster border)`.
7. REPORT_round_9.md documenting fixes, font measurement (the empirical pixel ratio for F1), crow investigation findings (F2), and any deviations.

Same scope/safety: theme + render + tests only. develop branch. ONE commit.
</completeness_contract>

<verification_loop>
For F1 (font size): MUST measure pixel cap-height in dot and dagua before changing the value. Document the measurement in REPORT.

For F2 (crow): MUST render arrow_types.png AFTER your fix and visually verify crow is filled. Don't trust the SVG cross-check alone — that was misleading last round.

For F4 (arrow proportions): visually verify arrow heads look stout/equilateral.
</verification_loop>

<missing_context_gating>
Default to most reasonable interpretation. Investigate before changing if uncertain. Document deviations in REPORT.
</missing_context_gating>

<action_safety>
Theme + render + tests only. develop branch. ONE commit at end.
</action_safety>

<default_follow_through_policy>
Default to most reasonable low-risk interpretation. Pre-existing test failures unrelated to your changes are not your problem. Keep going. Document deviations.
</default_follow_through_policy>
