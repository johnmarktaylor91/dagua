<task>
Round 13 of cosmetic parity work for graphviz_strict theme. Codex is still spent (subscription quota until tomorrow); you continue as the implementer.

Round 11 (commit 225fefd) over-corrected the round-9 regressions:
- F1 (puffy nodes): pulled too hard — nodes now 10-30% smaller than dot; star is BROKEN (label overflows points)
- F2 (edge label font): pulled too hard the other way — labels now 1.4-1.6x dot's size, dominating node labels
- F3 (arrow size on short edges): incomplete — `disable_curve_length_clamp` didn't actually upsize heads on short edges

Read these for context:
- `.project-context/research/sprint_graphviz_parity/AUDIT_round_12_OPUS.md` — the audit driving this round (with measurements)
- `.project-context/research/sprint_graphviz_parity/REPORT_round_5.md`, `REPORT_round_7.md` — historical context

Your last implementation report was inlined in the round-11 chat (no REPORT_round_11.md was written because subagents can't write report files in this harness — that's fine).

SCOPE: COSMETIC RENDERING ONLY. graphviz_strict theme. No layout. No harness. develop branch. ONE commit at end.

THE FIX LIST (5 corrections to round-11 over-corrections + 1 polish)
=====================================================================

### F1 (R13-A) — Node size: bring back from too-small (HIGH)
Round 11 set padding (6.0, 3.0), min_width 41, min_height 27. Audit measures dagua's "In" node at ~125×70 vs dot's ~190×95 (65% of dot's area). Nodes are systematically 10-30% smaller across panels.

Pull back: padding stays (6.0, 3.0) (looks acceptable), but raise min_width 41 → ~50 and min_height 27 → ~33. This compensates without re-introducing puffiness — the audit's recommendation is "pull min_width/min_height back from 41/27 toward ~50/33 (keep padding 6,3)."

Verify on tiny_graph.png, single_edge.png, pipeline.png. Node sizes should now read closer to dot — within ~10% area, not 35% under.

### F2 (R13-B) — Star shape collapsed (HIGH)
Round 11's `compact_shape_factors` flag dampened star's expansion: from `* 2.2` to `* 1.8`, plus skipping the second-pass `STAR_INTERIOR_FACTOR`. Audit says dagua's star is now ~30-40px tall vs dot's ~110px; the "star" label overflows the points entirely.

Fix: revert star compact factor entirely. Star should not be in the compact_shape_factors targets. Either remove star from the compact list, or set its compact multiplier back to the original 2.2 + reinstate STAR_INTERIOR_FACTOR for star specifically.

Verify on node_shapes_showcase.png "star" panel. Label should fit cleanly inside the star points.

### F3 (R13-C) — Ellipse curved-shape factor (MEDIUM)
Round 11 dropped `CURVED_SHAPE_INSCRIBE_FACTOR` from 1.5 to 1.0 in compact mode. Audit says ellipses are now slightly too tight. The audit recommends: "ellipse curved factor 0.93-0.96."

Wait — that seems contradictory (0.93-0.96 < 1.0). Re-read carefully and pick the right direction. If ellipses are TOO SMALL, the multiplier should INCREASE (1.0 → 1.1 or 1.15). If they look right but slightly off, tune in 0.05 increments. Test with node_shapes_showcase.png ellipse panel.

Apply judgment — the audit's exact number may be transposed. Goal: ellipse silhouette matches dot.

### F4 (R13-D) — Edge label font: pull back (HIGH)
Round 11's `_strict_absolute_edge_label_font_data` returns `font_size_points * display_scale` directly using graph_style.edge_label_font_size = 16.0. Audit says these are now 1.4-1.6x dot's, with edge labels visually dominating node labels.

Native dot edge labels appear ~10pt while node labels are ~14pt — edge labels are smaller than node labels, not equal. The round-11 fix made them equal-or-larger.

Fix: target ~10pt for edge labels (dot uses smaller for edges). Either:
- (a) Lower `EdgeStyle.label_font_size` and `GraphStyle.edge_label_font_size` from 16.0 to ~10.0
- (b) Keep 16.0 in theme but apply a 0.625 ratio (10/16) in the strict edge label rendering helper
Either way, the rendered edge label cap-height should be smaller than node label cap-height.

Verify on state_machine.png (transition labels), arrow_types.png (column names). Edge labels should be visually smaller than node labels.

### F5 (R13-E) — Arrow size on short edges (MEDIUM-HIGH)
Round 11's `disable_curve_length_clamp` field bypassed the SHORT_EDGE_HEAD_FRACTION clamp but didn't actually upsize heads. Audit says short edges in tiny_graph/single_edge/arrow_types still have small/thin arrows.

Investigate what's still constraining arrow size on short edges. Possibilities:
- The arrow is positioned ON the edge, so for a short edge the arrow only appears in the small remaining body length
- An additional clamp downstream (in render geometry, not collection)
- The visual size depends on stroke width which got dialed down

Add additive base arrow-size compensation OR find the remaining constraint. Goal: arrow heads on tiny_graph and single_edge should look the same physical size as those on pipeline and colors_showcase.

Verify on tiny_graph.png, single_edge.png — arrow heads should be visibly stout/sized like on pipeline.png.

### F6 (R13-F) — Stroke weight (LOW-MEDIUM, lower confidence)
Audit notes stroke ~1.0px while dot reads ~1.4px on node outlines and edges. Could be FreeType hinting, could be real. Try `_GRAPHVIZ_STRICT_DEFAULT_NODE_STYLE.stroke_width` 0.75 → 0.9 (modest bump). Keep edge body width at 1.0pt unless this clearly under-shoots after node bump.

Verify visually — stroke should match dot's hairline weight.

DO NOT TOUCH: improved theme, dagua/layout/, scripts/graphviz_theme_comparison.py.
</task>

<verification_protocol>
≤5 panel reads total (use two_way 1800×794 crops, NOT three_way).

After all fixes:
1. Render full gallery: `python scripts/graphviz_theme_comparison.py --output-dir eval_output/graphviz_theme_round_13`
2. Crop to two_way (same script as round 11):
   ```python
   from PIL import Image
   import os
   src = 'eval_output/graphviz_theme_round_13/three_way'
   dst = 'eval_output/graphviz_theme_round_13/two_way'
   os.makedirs(dst, exist_ok=True)
   for f in os.listdir(src):
       if f.endswith('.png'):
           img = Image.open(f'{src}/{f}')
           w, h = img.size
           img.crop((0, 0, int(w * 2/3), h)).save(f'{dst}/{f}')
   ```
3. Verify with at most 5 panel reads (must include node_shapes_showcase.png to confirm star + ellipse fix, and tiny_graph.png to confirm node size + arrow fix).
</verification_protocol>

<completeness_contract>
Not done until:
1. F1-F6 all implemented; verified visually within 5-panel budget.
2. Tier 1 tests pass; update test_style.py assertions if theme values changed.
3. Full gallery rendered + cropped.
4. ONE commit on develop: `feat(theme): graphviz_strict cosmetic round 13 — back off round-11 over-corrections (node size, star shape, edge label font, arrow size, stroke weight)`.
5. Reply with summary; report file is OK to inline since subagents can't write report files in this harness.
</completeness_contract>

<missing_context_gating>
Default to most reasonable interpretation. Audit measurements are best estimates — if your visual check disagrees, trust your eyes. Document deviations.
</missing_context_gating>

<action_safety>
Theme + render + tests only. develop. ONE commit at end.
</action_safety>
