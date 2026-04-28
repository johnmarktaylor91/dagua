<task>
Round 7 of cosmetic parity work for `graphviz_strict` theme. Round 5 commit 882b970 closed several high-impact issues (font, font size, cluster strokes, arrow proportions, color saturation, edge stroke weight) but introduced a regression (long_labels ellipse explosion) and left several items partial or unaddressed. The Opus 4.7 round-6 audit (`.project-context/research/sprint_graphviz_parity/AUDIT_round_6_OPUS.md`) drives this round.

Read these for context (mandatory):
- `.project-context/research/sprint_graphviz_parity/AUDIT_round_6_OPUS.md` (the round-6 verification audit — 6 PASS / 5 PARTIAL / 2 FAIL / 2 KNOWN_DEFERRED / 1 NEW REGRESSION)
- `.project-context/research/sprint_graphviz_parity/REPORT_round_5.md` (your prior implementation report)

SCOPE: COSMETIC RENDERING ONLY for `graphviz_strict` theme.
- Layout-side issues (H4 cluster cuts node, H5 sibling cluster overlap) are OUT OF SCOPE per orthogonality rule. Do NOT touch dagua/layout/.
- Do NOT modify scripts/graphviz_theme_comparison.py.
- The IMPROVED `graphviz` theme is deferred. Do not touch it.

THE FIX LIST (priority ordered)
================================

## TIER 1 — CRITICAL (regressions + correctness bugs)

### F1 (R1 + R2): Fix sqrt(2) ellipse circumscription gate
Currently: `if width/height > 2.0: height *= sqrt(2)` in `dagua/render/mpl.py` (or wherever the ellipse-fitting logic lives). This is geometrically WRONG and causes:
- **R1 NEW REGRESSION**: long_labels.png — three ellipses overlap/engulf each other because long single-line labels get height multiplied by sqrt(2) past their natural size.
- **R2**: arrow_types.png short-label ellipses (e.g. "normal", "vee") have W/H below 2.0 so they DON'T get the sqrt(2) treatment and remain flat.

The geometrically-correct fix: graphviz's ellipse circumscription multiplies BOTH semi-axes by sqrt(2) uniformly, so the ellipse circumscribes the label's bounding rectangle (the ellipse passes through the four corners of the box). Remove the `> 2.0` conditional. Apply sqrt(2) to BOTH axes equally:
```
width  = max(label_w + 2*margin, node_min_w) * sqrt(2) / 2  # semi-axis
height = max(label_h + 2*margin, node_min_h) * sqrt(2) / 2  # semi-axis
```
or equivalent using full-axes:
```
ellipse_full_w = max(label_w + 2*margin, ...) * sqrt(2)
ellipse_full_h = max(label_h + 2*margin, ...) * sqrt(2)
```

VERIFY on these panels after the fix:
- pipeline.png (medium-length labels — should still look rounder than round-4 baseline)
- arrow_types.png (short labels — ellipses should now be rounded, no longer flat)
- long_labels.png (long labels — must NOT overlap; the BatchNormalization2d ellipse should be sized but not balloon-sized)
- diamond.png (medium labels — sanity check)

If full sqrt(2) on both axes makes labels too wide visually compared to dot's, try `1.3` or `1.25` as the multiplier. The point is uniformity across all label sizes.

### F2 (R3): `crow` arrowhead must be FILLED, not hollow
Round-5 report claimed "kept crow filled" but the rendered output shows hollow. Investigate. Likely cause: a refactor in `dagua/render/edges/arrowheads.py` reset crow's fill flag.

VERIFICATION PROTOCOL (must do before reporting done):
1. Open the round-7 rendered arrow_types.png after your fix
2. Read it (vision-capable)
3. Confirm crow column shows a small FILLED dark triangle/spear shape (matching dot's panel)
4. Do NOT report done until this is visually confirmed

Cross-check by running `dot -Tsvg` on a small test:
```
echo 'digraph { a -> b [arrowhead="crow"] }' | dot -Tsvg -o /tmp/crow_test.svg
```
Inspect the SVG for the crow path's `fill` attribute. If it's `fill:black` (or no fill attribute, defaulting to black), crow is filled. Match that.

### F3 (R4): `open` arrowhead must be FILLED on Graphviz 8.0.3
Per round-6 audit: dot 8.0.3 renders the named arrow `"open"` as a FILLED triangle (basically same as `normal`), NOT hollow. Dagua currently renders it as hollow V chevron (same as `vee`).

Verification protocol same as F2:
```
echo 'digraph { a -> b [arrowhead="open"] }' | dot -Tsvg -o /tmp/open_test.svg
```
Match whatever dot does on this version. If filled, change dagua's `open` mapping to filled.

VERIFY on arrow_types.png — the "open" column should match dot's appearance.

## TIER 2 — MEDIUM (gaps the user will notice)

### F4 (R5): Edge label font size too small (~25-30% smaller than dot)
Round 5 raised node `font_size` to 12pt but missed edge labels. Find where edge labels' rendering size is determined:
- `EdgeStyle.label_font_size` in graphviz_strict default edge style
- `GraphStyle.edge_label_font_size`
- Any backedge edge style that overrides

Currently likely 10.5pt (the round-3 over-corrected DPI value). Raise to 12.0 to match node labels.

VERIFY on state_machine.png (edge labels "retry"/"resume"/"reset"/"restart") and arrow_types.png (arrow type names like "normal"/"vee").

### F5 (R6): Cluster fill_opacity 0.08 → 0.10
Round 5 dropped 0.15 → 0.08, but Opus confirms this overshot to "no tint at all" — dot's actual cluster tint is faintly visible (warm-cream), 0.08 is invisible. Raise to 0.10. If you want to match the chromatic feel, also try changing the fill color from `#F0F0F0` (cool gray) to a slightly warmer tone like `#F2EFE9` or a lightgrey/`#D3D3D3`. Keep `border_opacity=1.0` (separate field).

### F6 (R8): Arrow tip-to-boundary trim
On diamond.png and balanced_binary_tree.png the arrow tips OVERLAP into the target ellipse by 3-4px. The edge body and arrow base bite into the node. Native dot puts the tip exactly ON the ellipse boundary curve.

Fix: in the edge endpoint computation (likely `dagua/render/edges/collection.py` or `geometry.py`), compute the ellipse-boundary intersection of the edge ray, place arrow tip at that intersection, and trim the edge body to `tip - arrow_length * direction_unit_vector`. Standard parametric ellipse intersection:
```
# ellipse centered at (cx, cy) with semi-axes (a, b)
# edge from (x0, y0) toward (cx, cy)
# intersection: solve ((x0 + t*dx - cx)/a)^2 + ((y0 + t*dy - cy)/b)^2 = 1
# t is the parametric distance from (x0,y0); take the positive root closer to the ellipse boundary
```
Apply for any arrow target that's an ellipse-shaped node. Verify on diamond.png (Start->Left/Right and End approach), balanced_binary_tree.png (any leaf), and pipeline.png (each chain step).

### F7 (R7): Edge label collision avoidance on state_machine
On state_machine.png the labels "retry" and "resume" sit at almost the same y-coordinate, nearly touching ("retryresume" reads as one word). Dot offsets them along-edge or perpendicular.

Add collision detection: when two edge labels' bounding boxes overlap, offset the second one along the edge direction by `label_height + small_padding` so they're stacked vertically. Likely in `dagua/render/edges/labels.py` or wherever label placement happens.

## TIER 3 — POLISH (do if time permits, defer otherwise)

### F8 (H6 polish): Back-edge curvature magnitude — possibly bump floor
Round 5 added 36pt absolute floor; Opus says dot's actual side-channel offset on state_machine looks like ~80px. Try raising the absolute floor to 60pt OR convert the formula to "half the available side-channel width" rather than fixed offset.

### F9 (H8 polish): Deepest-cluster label mask too small
On deep_nesting_4.png the innermost "Level 4 (Core)" cluster's label is partially clipped by stroke. The mask rectangle may be too small for tiny clusters. Check the mask sizing logic — should accommodate any label size, not assume a minimum cluster size.

## SKIP (defer as residual or low-priority)
- R11 vertical centering 2-3px off (sub-pixel territory; matplotlib vs cairo baseline difference)
- R12 multi-line spacing 1.3 vs 1.15
- R13 glyph stroke contrast (FreeType vs Cairo rasterizer; stack residual)
- R14 title font weight (panel decoration, not graph content)
- R9, R10 cluster overlap on transformer_block / cluster_showcase (layout-side)

Two layout-side known-deferred items remain (H4, H5) — explicitly NOT in scope this round.

</task>

<completeness_contract>
Not done until:
1. F1-F7 all implemented and visually verified by reading the regenerated panels.
2. F8, F9 attempted with best-effort or explicitly deferred with justification in REPORT.
3. `pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"` passes for in-scope files. Pre-existing layout-import errors still out of scope.
4. Re-render gallery: `python scripts/graphviz_theme_comparison.py --output-dir eval_output/graphviz_theme_round_7`.
5. Re-crop to 2-way for next audit:
   ```python
   from PIL import Image
   import os
   src = 'eval_output/graphviz_theme_round_7/three_way'
   dst = 'eval_output/graphviz_theme_round_7/two_way'
   os.makedirs(dst, exist_ok=True)
   for f in os.listdir(src):
       if f.endswith('.png'):
           img = Image.open(f'{src}/{f}')
           w, h = img.size
           img.crop((0, 0, int(w * 2/3), h)).save(f'{dst}/{f}')
   ```
6. Visually verify by reading: long_labels.png (R1 fix), arrow_types.png (R2/R3/R4 fixes), state_machine.png (R5/R7 fixes), diamond.png (R8 fix), pipeline.png (sanity check).
7. ONE commit: `feat(theme): graphviz_strict cosmetic round 7 — uniform sqrt(2) ellipse, crow/open arrow fill, edge label size, cluster opacity, arrow tip trim, edge label collision`.
8. REPORT_round_7.md at `.project-context/research/sprint_graphviz_parity/REPORT_round_7.md` documenting fixes, verification (especially the `dot -Tsvg` outputs for F2/F3), test results, deviations.

Same scope/safety: theme + render + tests only. develop branch. ONE commit.
</completeness_contract>

<verification_loop>
For F1 (sqrt(2) gate): MUST re-render and verify on long_labels.png AND arrow_types.png. If long_labels still has overlaps, the geometry is wrong; iterate. If short-label ellipses on arrow_types still flat, gate isn't removed; iterate.

For F2/F3 (arrow fill): mandatory `dot -Tsvg` cross-check before declaring done. Must see the actual rendered MIDDLE panel match LEFT for crow and open columns.

For F6 (arrow tip trim): test on at least 3 panels with different node shapes/edge angles.

For F7 (edge label collision): state_machine.png is the primary test case.
</verification_loop>

<missing_context_gating>
Default to most reasonable interpretation. Investigate before changing if uncertain. Document deviations in REPORT_round_7.md.

If a fix turns out to require renderer-pipeline-level changes you can't do safely (e.g. F6 trim requires layout-aware geometry that crosses the layout boundary), document the issue, do partial fix or no-op, and flag for next round.
</missing_context_gating>

<action_safety>
Theme + render + tests only. develop branch. ONE commit at end.
</action_safety>

<default_follow_through_policy>
Default to most reasonable low-risk interpretation. Pre-existing test failures unrelated to your changes are not your problem. Keep going. Document deviations.
</default_follow_through_policy>
