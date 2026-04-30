<task>
Round B4 — final cosmetic round before declaring ceiling. Audit A4 (`/home/jtaylor/projects/dagua/.project-context/research/sprint_graphviz_parity/AUDIT_A4.md`) said STOP because the dominant remaining gap (matplotlib-vs-Cairo render-stack) is structurally blocked. But it noted ONE remaining fixable cosmetic gap — B3's 1.2x edge-stroke multiplier was too small to overcome matplotlib AA. This round attempts the stronger fix.

Read for context:
- `/home/jtaylor/projects/dagua/.project-context/research/sprint_graphviz_parity/AUDIT_A4.md`
- `/home/jtaylor/projects/dagua/.project-context/research/sprint_graphviz_parity/REPORT_B3.md`

Repo: `/home/jtaylor/projects/dagua`, branch `develop`. Single working branch. ONE commit at end.

## Single fix

### F1 — Stronger edge stroke crispness
Edges still render visibly gray vs dot's solid black. B3's `_GRAPHVIZ_STRICT_EDGE_WIDTH_RENDER_MULTIPLIER = 1.2` was insufficient.

Try in order, stop when the edges visually match dot's darkness on bipartite_5x5 hi-res:
1. Bump multiplier to **1.5x** (was 1.2x)
2. Add `solid_capstyle="butt"` and `solid_joinstyle="miter"` to edge patches in graphviz_strict
3. Set `alpha=1.0` explicitly (in case AA/blending is dropping it implicitly)
4. If edges STILL gray, try multiplier=1.7x

DO NOT modify the declarative `EdgeStyle.width` (declared as 1.0pt to match dot's SVG); only the render-time visual multiplier.

VERIFY by reading hi-res `/home/jtaylor/projects/dagua/eval_output/parity_pixel_diff/hires/bipartite_5x5/dagua.png` after re-rendering and comparing to `/home/jtaylor/projects/dagua/eval_output/parity_pixel_diff/hires/bipartite_5x5/dot.png`. Edges should look as dark as dot's.

## Out of scope

- Anything else from A4. The dot-rasterizer 26% rx inflation is principled residual, not fixable without breaking declarative parity. Layout-scope items deferred.
- Don't try to re-attempt F3 (long-label kerning) — already reverted twice.

## Completeness contract

Not done until:
1. Edge stroke visually as dark as dot's (verified by reading hi-res panels)
2. Tier-1 tests pass
3. Re-run parity_metrics.py + parity_pixel_diff.py — capture before/after
4. ONE commit on develop: `feat(theme): graphviz_strict round B4 — edge stroke crispness final pass`
5. REPORT at `.project-context/research/sprint_graphviz_parity/REPORT_B4.md` with before/after stats

If even at multiplier=1.7x the edges still appear gray, document as render-stack residual and skip — that confirms ceiling.

## Reply format

Outcome (succeeded / partial / hit-residual), commit SHA, before/after L1 + SSIM. ≤150 words.
</task>

<action_safety>
Theme + render only. develop. ONE commit at end.
</action_safety>
