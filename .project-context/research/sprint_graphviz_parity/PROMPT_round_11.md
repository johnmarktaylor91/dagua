<task>
Round 11 of cosmetic parity work for `graphviz_strict` theme. Codex hit subscription quota; you (Claude Opus 4.7 subagent) are the implementation path going forward.

Round 9 (commit b4ff37d) closed several regressions but introduced a new one: F1's 16pt font bump grew auto-sized node bounding boxes. Round 10 audit (`.project-context/research/sprint_graphviz_parity/AUDIT_round_10_OPUS.md`) recommends 3 targeted fixes.

Read these for context:
- `.project-context/research/sprint_graphviz_parity/AUDIT_round_10_OPUS.md`  (the audit driving this round)
- `.project-context/research/sprint_graphviz_parity/REPORT_round_5.md` and `REPORT_round_7.md` (prior implementation reports — codex's; you can skim)

SCOPE: COSMETIC RENDERING ONLY for `graphviz_strict` theme. No layout. No harness. develop branch. ONE commit at end.

THE FIX LIST (3 items)
=======================

### F1 (R11-A) — Puffy nodes (HIGH)
Round-9's bump from 12pt → 16pt to match dot's visual cap-height inadvertently grew auto-sized node bounding boxes. On node_shapes_showcase.png dagua's diamond is ~2.5× dot's area. The label is the right APPARENT size but the BOUNDING BOX (computed from matplotlib's measured label width × the larger glyph metrics) is too big.

Two clean approaches; pick whichever is simpler:
- **(a) Decouple visual font size from layout font size.** Add a parameter (or use existing infrastructure) so the bounding-box computation uses 12pt-equivalent width while the rendered text uses 16pt. The "layout label width" is `actual_label_width * 12 / 16 = 0.75 * actual_label_width`.
- **(b) Reduce padding/min_width/min_height to compensate.** If the new font_size scales widths by ~1.33×, scale node padding/min sizes by 1/1.33 ≈ 0.75 so ellipses end up the same physical size as before. Padding (8.0, 4.0) → (6.0, 3.0); min_width 54 → 41; min_height 36 → 27. Test on node_shapes_showcase.png.
- **(c) Empirically: actually back off font_size to a smaller value (say 14pt) that's a compromise between matching dot's cap-height and not blowing up ellipse size.** Document the tradeoff in the report.

Investigate which approach is cleanest in dagua's render pipeline. Lean toward (a) if there's a clean place to inject the layout-vs-render font split. Otherwise (b) or (c).

VERIFY on: pipeline.png, diamond.png, node_shapes_showcase.png (THE primary regression case), single_edge.png. Ellipse sizes should match dot's at the same label content.

### F2 (R11-B) — Edge label font size unclosed (HIGH)
F1 in round 9 raised `EdgeStyle.label_font_size` to 16.0 in graphviz_strict, but on arrow_types.png and edge_styles_showcase.png the standalone edge labels (e.g. arrow type names like "normal", "vee") are still ~70% of dot's size.

There must be a separate code path for these standalone edge labels — possibly a `GraphStyle.edge_label_font_size` or a separate "annotation text" field that wasn't bumped. Investigate `dagua/render/edges/labels.py` and `dagua/render/mpl.py` for where these texts are sized.

VERIFY on: arrow_types.png (the arrow type column labels), state_machine.png (transition labels), edge_styles_showcase.png. All edge labels should match dot's apparent size.

### F3 (R11-C) — Arrow size inconsistency (HIGH)
Round 9's arrowhead dimensions (12, 10) work well on pipeline and colors_showcase but over-shoot on those panels (slightly oversized) AND under-shoot on tiny_graph and single_edge (slightly undersized). The variance suggests arrow size is being coupled to something panel-dependent (canvas scale? font size? edge length?) when it should be a constant absolute size.

Investigate `dagua/render/edges/arrowheads.py` and `dagua/render/edges/collection.py` (or wherever arrow scaling happens). Find where the apparent arrow size depends on context, and decouple it. Native dot draws arrowheads at a constant absolute pt size regardless of graph size.

VERIFY on: tiny_graph.png and single_edge.png (should not be undersized) AND pipeline.png and colors_showcase.png (should not be oversized). Arrow heads visible footprint should be constant across panels.

DO NOT TOUCH: improved `graphviz` theme, dagua/layout/, scripts/graphviz_theme_comparison.py.
</task>

<verification_protocol>
For each fix, after implementing:
1. Re-render ONLY the priority panels needed for verification (don't re-render the full gallery during iteration — too slow).
2. Read the relevant TWO_WAY-cropped PNG (use `eval_output/graphviz_theme_round_9/two_way/` as your reference; after re-rendering use the new path under `eval_output/graphviz_theme_round_11/two_way/`). Verify visually.
3. Limit panel reads to NO MORE THAN 5 PNGs total across the whole task to avoid hitting the image-dimension cap. The cropped panels are 1800×794 (under the cap), but cumulative count still matters.
4. After all 3 fixes are in place, render full gallery and crop to two_way for the next audit.

If you need to re-render a single panel ad-hoc:
```python
# in a Python REPL
from scripts.graphviz_theme_comparison import _iter_cases, _render_case, _ensure_output_dirs
from pathlib import Path
import dagua
# pick one case, render it
```
or just do `python scripts/graphviz_theme_comparison.py --output-dir /tmp/round_11_check` and read 1-2 panels.
</verification_protocol>

<completeness_contract>
Not done until:
1. F1, F2, F3 all implemented and visually verified (≤5 panel reads).
2. Tier 1 tests pass: `pytest tests/test_style.py tests/test_render tests/test_custom_edges.py tests/test_arrowheads.py -x --tb=short -q`
3. Update test_style.py assertions if you changed any theme values.
4. Final full gallery: `python scripts/graphviz_theme_comparison.py --output-dir eval_output/graphviz_theme_round_11`
5. Crop to two_way:
   ```python
   from PIL import Image
   import os
   src = 'eval_output/graphviz_theme_round_11/three_way'
   dst = 'eval_output/graphviz_theme_round_11/two_way'
   os.makedirs(dst, exist_ok=True)
   for f in os.listdir(src):
       if f.endswith('.png'):
           img = Image.open(f'{src}/{f}')
           w, h = img.size
           img.crop((0, 0, int(w * 2/3), h)).save(f'{dst}/{f}')
   ```
6. ONE commit on develop: `feat(theme): graphviz_strict cosmetic round 11 — close round-9 regressions (puffy nodes, edge label size, arrow size consistency)`.
7. REPORT_round_11.md documenting fixes, which approach you took for F1, what was wrong with edge labels in F2, what was coupling arrows in F3, deviations.

Same scope/safety as prior rounds.
</completeness_contract>

<missing_context_gating>
Default to most reasonable interpretation. If a fix's approach is fundamentally different from the spec, document and proceed. Don't stall on small ambiguities.
</missing_context_gating>

<action_safety>
Theme + render + tests only. develop branch. ONE commit at end. No pushes.
</action_safety>

<default_follow_through_policy>
Default to most reasonable low-risk interpretation. Pre-existing test failures are not your problem. Keep going.
</default_follow_through_policy>
