<task>
Round B3 of overnight autonomous graphviz_strict cosmetic parity. Audit A3 (`/home/jtaylor/projects/dagua/.project-context/research/sprint_graphviz_parity/AUDIT_A3.md`) found that B2's F4 over-corrected the compact-ellipse oval aspect floor, causing 4 cascading downstream issues. Plus an independent finding about edge stroke gray-vs-black.

Read for context:
- `/home/jtaylor/projects/dagua/.project-context/research/sprint_graphviz_parity/AUDIT_A3.md` (drives this round)
- `/home/jtaylor/projects/dagua/.project-context/research/sprint_graphviz_parity/REPORT_B2.md`

Repo: `/home/jtaylor/projects/dagua`, branch `develop`. Single working branch.

## Fix list

### F1 (HIGHEST, ONE-LINE) — Drop oval floor from 1.85 to 1.50
File: `dagua/render/mpl.py:116`
Constant: `_GRAPHVIZ_STRICT_MIN_OVAL_ASPECT`
Current: 1.85 (over-widens short-label ellipses)
Target: 1.50 (matches dot's actual short-label aspect per metric)

Predicted to fix: tiny_graph clipping, bipartite_5x5/ladder/nested_clusters overlaps, star spoke clipping, arrow_types source ellipses oversized. SSIM should jump +0.02-0.04 on 8+ worst panels.

### F2 (HIGH, independent) — Edge stroke renders lighter gray than dot's black
Every panel shows dagua's edges as visibly gray/charcoal while dot's are solid black. Theme color is `#000000` for both — the issue is at matplotlib's AA/linewidth layer.

Investigation:
1. Look at `EdgeStyle.width = 1.0pt` in graphviz_strict — is matplotlib applying it as 1px stroke at the rendered DPI? If so, AA softens it to gray.
2. Compare: dot's PostScript stroke is fully opaque + sharp; matplotlib's antialiased thin line at typical render DPI dilutes the apparent darkness.
3. Possible fixes:
   - (a) Bump `EdgeStyle.width` slightly (1.0 → 1.2 or 1.4) so AA still produces a dark line
   - (b) Set `solid_capstyle="butt"` and `solid_joinstyle="miter"` on edge patches if not already
   - (c) Disable AA selectively via `path_effects` or `set_antialiased(False)` for graphviz_strict edges (might cause aliasing artifacts on diagonals — try (a) first)

Fix path: try (a) first with width=1.2pt. Verify on pipeline.png and bipartite_5x5.png — edges should look as dark as dot's.

### F3 (carry-over from A1, MEDIUM) — Long-label ellipse_rx still narrow
B2 attempted but reverted F5a (long-label rx) because of broad metric regression. The fix needs to be MORE TARGETED: only apply the longer-label compensation to labels above N characters (e.g., >=10), not unconditionally.

Investigation:
1. The kerning gap is per-character. Labels with N chars accumulate ~0.3pt/char of missing kerning vs Cairo.
2. Add a label-length-conditional rx scaling: when `len(label) >= 10`, multiply text_w by `1.0 + (len(label) - 10) * 0.005` before computing ellipse axes.
3. Verify this doesn't regress short labels (which are already in tolerance).

If this still causes broad regression, document and skip — it's a render-stack residual.

### F4 (LOW polish) — None of the above conflict with declarative metric
Re-run `pytest tests/test_parity_metrics.py` after each fix to confirm declarative in-tolerance % stays >= 99% (it was 99.27% post-B2).

## Out of scope

- Layout-side cluster geometry (deferred since A1)
- Improved `graphviz` theme
- scripts/parity_metrics.py and scripts/parity_pixel_diff.py

## Completeness contract

Not done until:
1. F1, F2, F3 attempted; document infeasibility in REPORT.
2. Tier-1 tests pass: `pytest tests/test_style.py tests/test_render tests/test_custom_edges.py tests/test_arrowheads.py tests/test_parity_metrics.py -x --tb=short -q`. Update assertions if values changed.
3. Re-run `python scripts/parity_metrics.py` and `python scripts/parity_pixel_diff.py` (full 45 panels) — capture before/after numbers in REPORT.
4. ONE commit on develop: `feat(theme): graphviz_strict round B3 — oval floor 1.50, edge stroke darker, long-label kerning`.
5. REPORT at `.project-context/research/sprint_graphviz_parity/REPORT_B3.md`.

## Reply format

Per-fix outcome, before/after stats, commit SHA. ≤200 words.
</task>

<missing_context_gating>
F3 (long-label kerning) is the high-risk fix. If your initial implementation regresses metric or pixel parity, revert it and document. Don't keep iterating on F3 if first try fails — F1 and F2 are the high-confidence wins.
</missing_context_gating>

<action_safety>
Theme + render + tests only. develop. ONE commit at end.
</action_safety>
