<task>
Round B1 of overnight autonomous graphviz_strict cosmetic parity. Audit A1 (`/home/jtaylor/projects/dagua/.project-context/research/sprint_graphviz_parity/AUDIT_A1.md`) found 5 HIGH severity `real_cosmetic_gap + fixable_theme_or_render` findings. Implement them surgically.

Read for context (mandatory):
- `/home/jtaylor/projects/dagua/.project-context/research/sprint_graphviz_parity/AUDIT_A1.md` (drives this round)
- `/home/jtaylor/projects/dagua/.project-context/knowledge/visual_tuning_workflow.md` (general approach)

Repo: `/home/jtaylor/projects/dagua` (already on `develop` branch). Single working branch policy.

## Fix list

### F1 (HIGH) — Canvas-fill regression
Dagua does NOT scale its drawing to fill the figure the way `dot` does. The pixel-diff "Background L1" delta on every panel signals huge whitespace bands in dagua's render. Worst on sparse graphs (tiny_graph, bipartite_5x5, single_edge, colors_showcase, pipeline).

Investigation path:
1. Look at `dagua/render/mpl.py` — figure-size and axis bounds computation. Specifically the `tight_layout`, axis padding, and `set_xlim`/`set_ylim` calls.
2. Native dot (`dot -Tpng`) sizes the canvas tightly to content-bbox + small margin (typically 4pt). Dagua's render appears to inflate the canvas with extra whitespace.
3. Likely culprit: `_expand_bounds_for_external_labels` or the GraphStyle.margin application.

Fix: when graphviz_strict is active, match dot's tight content-bbox + 4pt margin convention (or whatever dot's actual SVG `viewBox` minus content-bbox shows).

VERIFY by running `python scripts/parity_pixel_diff.py --cases pipeline,tiny_graph,colors_showcase` and checking that mean L1 RGB / pixel drops substantially.

### F2 (HIGH) — Auto-wrap of long labels
On `long_labels.n3` and `label_variety.n7`, dagua wraps a label that dot keeps single-line. Result: ellipse_ry inflates by 40pt+. Dagua should only break on explicit `\n` / `\l` / `\r` characters, never auto-wrap based on width.

Investigation:
1. Find where text-wrapping happens (`dagua/utils.py` or `dagua/render/text/`).
2. There's likely a `text_wrap` parameter — for graphviz_strict it must be `"none"` and never bypassed.
3. Verify NodeStyle.text_wrap = "none" in graphviz_strict and that no fallback/automatic wrap path overrides it.

### F3 (HIGH) — Systematic ellipse_rx narrowing
138/487 nodes have negative rx delta (mean -3.15pt, max -13.4pt). Co-located with 167 ellipse_aspect_pct out-of-tolerance, also negative.

This is the matplotlib TextToPath glyph-width vs Cairo gap (no kerning compensation in matplotlib's per-glyph sum). Hypothesis test:
1. Use freetype's `font.get_kerning(left, right)` API to measure kerning pairs for "Postprocess" at 14pt qtmr.pfb.
2. If cumulative kerning ≈ observed delta, hypothesis confirmed.

Fix path (try in order):
- (a) Replace `TextToPath.get_text_width_height_descent` calls with a kerning-aware width measurement using freetype directly. See `dagua/utils.py:_tex_gyre_termes_font_path`.
- (b) Alternative: use `matplotlib.text.Text.get_window_extent()` after a temporary render — this DOES honor kerning.
- (c) Fallback: add a per-character kerning correction term scaled with label length.

Verify by re-running parity_metrics.py — ellipse_rx_pt and ellipse_aspect_pct should jump to >90% in tolerance.

### F4 (HIGH) — arrow_types defects
Multiple sub-issues:

a) **arrowsize attribute ignored**: edges e1/e5 in arrow_types have target arrow_width 10.46pt but dagua reports 7.0. Dagua's arrowsize handling needs to multiply arrow_length and arrow_width by the per-edge `arrowsize` attribute (graphviz default = 1.0).

b) **Arrow shapes for `circle`/`open`/`dot`**: visual gaps vs dot. Verify `dot -Tsvg` for each named arrowhead and confirm dagua matches:
```
for shape in circle open dot odot ocircle inv tee crow vee normal diamond box; do
  echo "digraph { a -> b [arrowhead=\"$shape\"] }" | dot -Tsvg -o /tmp/a_$shape.svg
done
```

c) **Edge stroke renders lighter**: dot strokes appear heavier than dagua's at the same nominal 1.0pt. May be matplotlib AA softening; investigate by comparing rendered pixel intensity.

d) **Edge labels are 1.4-1.6x larger/bolder**: edge label rendering uses graph_style.edge_label_font_size which is currently 14pt. Dot's edge labels appear smaller. Investigate the actual SVG declaration on arrow_types panel — maybe dot uses 11pt for edge labels by default. If so, set graphviz_strict's edge_label_font_size accordingly.

### F5 (DEFER as layout-scope) — nested_clusters cluster geometry
Audit classified as uncertain. Most of the issues (node A protrusion, sibling overlap, label-content overlap) ARE layout-scope (cluster bbox sizing during layout). Out of scope for this cosmetic sprint per the orthogonality rule.

DO this for F5: leave the code unchanged but document in REPORT_B1.md that the cluster geometry items are deferred-layout per the architecture rule.

## Out of scope

- Do NOT touch `dagua/layout/`.
- Do NOT modify `scripts/graphviz_theme_comparison.py`.
- Do NOT spend effort on the IMPROVED `graphviz` theme.
- F5 (cluster geometry) is deferred-layout; leave alone.

## Completeness contract

Not done until:
1. F1, F2, F3, F4(a-d) attempted. If any sub-fix turns out to be infeasible (e.g. F3 freetype API doesn't support what's needed), document in REPORT and proceed.
2. Tier-1 tests pass: `pytest tests/test_style.py tests/test_render tests/test_custom_edges.py tests/test_arrowheads.py tests/test_parity_metrics.py -x --tb=short -q`. Update test assertions if theme values changed.
3. Re-run `python scripts/parity_metrics.py` and `python scripts/parity_pixel_diff.py` (full 45 panels) — capture before/after numbers.
4. ONE commit on develop: `feat(theme): graphviz_strict round B1 — canvas fill, label wrap, kerning, arrow defects`.
5. REPORT at `.project-context/research/sprint_graphviz_parity/REPORT_B1.md` with: per-fix outcome, before/after metric numbers, deviations.

## Verification protocol

For each fix, after implementing:
- Re-run parity_metrics + parity_pixel_diff on at most 5 representative panels first; verify the metric or L1 changes in the expected direction.
- Don't move to the next fix until the prior is verified.

After all fixes, full 45-panel run.

## Reply format

Reply with: per-fix outcome (F1/F2/F3/F4 each), before-after summary stats from the metrics/pixel diff, the commit SHA. Max 250 words.
</task>

<missing_context_gating>
If F3 turns out to be infeasible (no clean kerning-aware API in available libraries), document the investigation findings in REPORT_B1.md and skip — that becomes a documented residual rather than a fix. Don't burn cycles on dead ends; the autonomous loop has more rounds to come.
</missing_context_gating>

<action_safety>
Theme + render + tests only. develop branch. ONE commit at end.
</action_safety>
