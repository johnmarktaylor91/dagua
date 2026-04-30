<task>
Round B2 of overnight autonomous graphviz_strict cosmetic parity. Audit A2 (`/home/jtaylor/projects/dagua/.project-context/research/sprint_graphviz_parity/AUDIT_A2.md`) found 5 HIGH severity `real_cosmetic_gap + fixable_theme_or_render` findings. Implement them surgically.

Read for context:
- `/home/jtaylor/projects/dagua/.project-context/research/sprint_graphviz_parity/AUDIT_A2.md` (drives this round)
- `/home/jtaylor/projects/dagua/.project-context/research/sprint_graphviz_parity/REPORT_B1.md` (what landed last round)

Repo: `/home/jtaylor/projects/dagua` (already on `develop` branch). Single working branch policy.

## Fix list (priority order)

### F1 (HIGHEST) — Figure aspect mismatch (canvas-fill phase 2)
B1 fixed graph margin to 0pt delta on all 45 panels but figure inches/aspect still leaves ~20-30% left/right white bands on tall-narrow panels (pipeline, tiny_graph, single_edge, colors_showcase). This is the dominant SSIM regressor.

Investigation:
1. Look at how dagua sizes its matplotlib figure for graphviz_strict renders. The figsize calculation likely uses fixed proportions or `_compute_figure_size` defaults that don't track content aspect.
2. Native `dot -Tpng` produces an output whose pixel dimensions tightly track content bbox. Dagua should compute figure inches from content bbox at the chosen DPI: `figsize = (content_bbox_w_pt / 72.0, content_bbox_h_pt / 72.0)` plus minimal margin.
3. Likely culprit: figsize calculation in `dagua/render/mpl.py` or wherever `set_size_inches` / `subplots(figsize=...)` is set up. Check if it's using `default_figsize` or computing from content.

Fix: when graphviz_strict is active, set figsize directly from the rendered content bbox (4pt margin on each side already in place from B1). VERIFY by running `python scripts/parity_pixel_diff.py --cases pipeline,tiny_graph,single_edge` and confirming SSIM jumps at least +0.05 on each.

### F2 (HIGHEST) — Arrowhead polygon regressed to flat rhombus
B1 introduced this regression. Every arrow on every panel is now a stubby 4-vertex rhombus instead of dot's clean filled isoceles triangle. Apparent height ~50-60% of dot's.

Investigation:
1. Look at the arrow rendering code in `dagua/render/edges/arrowheads.py` — was the `_normal` (or default triangle) primitive changed in B1?
2. Check `dagua/render/edges/collection.py` — did B1 add an `arrowsize` plumbing path that altered the polygon shape?
3. Compare the rendered polygon vertices to dot's SVG polygon for a `normal` arrow:
   ```
   echo 'digraph { a -> b }' | dot -Tsvg | grep polygon
   ```

Fix: restore the standard 3-vertex isoceles triangle (or 4-vertex with degenerate base point — match dot's exact vertex count and proportions). Verify on tiny_graph.png arrowheads.

### F3 (HIGH) — Per-edge arrowsize attribute still ignored
arrow_types has 4 OOT on arrow_width_pt because dagua flatlines at 7.0pt regardless of per-edge arrowsize attribute. B1 added the field but didn't populate from per-edge attrs.

Investigation: trace dot's per-edge arrowsize through the DOT parser (`dagua/eval/competitors/graphviz_competitor.py` or similar) — find where edge attributes are parsed and ensure `arrowsize` reaches the EdgeStyle.

Fix: when an edge has `arrowsize=N` attribute, multiply the rendered arrow_length and arrow_width by N. Default arrowsize=1.0.

### F4 (HIGH) — Single-line ellipses too circular
Short labels (In, Mid, Out, L1-L5, R1-R5, Red, Blue, Green) render at ~1.2-1.5:1 aspect in dagua vs dot's ~2:1. The 1.28x rx factor (added in B1?) is too low for short labels.

Investigation:
1. Find the ellipse aspect-ratio computation. Likely `compute_node_size` or where `compact_shape_factors` is applied.
2. dot uses sqrt(2) ≈ 1.414 circumscription on text bbox PLUS pads for min ellipse aspect ~2:1. Replicate that.

Fix: when shape is ellipse and the label is single-line, expand width to maintain a minimum 1.8-2.0:1 aspect ratio (matching dot's preference for wide ovals over circles).

VERIFY on tiny_graph.png and colors_showcase.png — single-character or short-word ellipses should be visibly oval, not nearly circular.

### F5 (HIGH) — Long-label ellipses + edge-label font
Two sub-issues:
a) 23 long-label nodes still OOT all-negative; max -13.40pt on MultiHeadAttention. The fixed 1.28x factor undershoots for labels >=10 chars. Try increasing the factor for longer labels OR adding kerning compensation more aggressively.

b) Edge labels in arrow_types over-corrected from "too large" to "too small" (B1 set 11pt; dot uses 14pt = node-label default). Set graphviz_strict's `edge_label_font_size` back to 14.0pt to match node labels.

VERIFY: ellipse_rx_pt should jump to >95% in tolerance; edge labels on arrow_types should match node label sizes.

## Out of scope

- Layout-side cluster geometry (already deferred from F5 in A1)
- Improved `graphviz` theme — only graphviz_strict
- Don't modify scripts/parity_metrics.py or scripts/parity_pixel_diff.py

## Completeness contract

Not done until:
1. F1-F5 attempted; document any infeasible fix in REPORT.
2. Tier-1 tests pass: `pytest tests/test_style.py tests/test_render tests/test_custom_edges.py tests/test_arrowheads.py tests/test_parity_metrics.py -x --tb=short -q`. Update assertions if theme values changed.
3. Re-run `python scripts/parity_metrics.py` and `python scripts/parity_pixel_diff.py` (full 45 panels) — capture before/after numbers.
4. ONE commit on develop: `feat(theme): graphviz_strict round B2 — figure aspect, arrowhead triangle, arrowsize, ellipse aspect, edge label font`.
5. REPORT at `.project-context/research/sprint_graphviz_parity/REPORT_B2.md` with: per-fix outcome, before/after metric numbers, deviations.

## Reply format

Per-fix outcome (F1-F5), before-after summary stats, commit SHA. Max 250 words.
</task>

<missing_context_gating>
Default to most reasonable interpretation. If F2's regression is in code that doesn't allow easy revert (e.g. B1 refactored the arrow primitive), fix forward by re-implementing the correct 3-vertex triangle. Document the path taken in REPORT.
</missing_context_gating>

<action_safety>
Theme + render + tests only. develop branch. ONE commit at end.
</action_safety>
