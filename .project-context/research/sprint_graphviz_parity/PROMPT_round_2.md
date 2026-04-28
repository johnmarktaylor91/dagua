<task>
Round 2 of cosmetic parity work for `graphviz_strict` theme. Read the prior reports for context:
- `.project-context/research/sprint_graphviz_parity/AUDIT_round_1.md` (initial findings)
- `.project-context/research/sprint_graphviz_parity/REPORT_round_1.md` (your prior implementation report)
- `.project-context/research/sprint_graphviz_parity/AUDIT_round_2.md` (latest visual audit)

Round 1 landed 4.5/5 fixes cleanly (commit cfa8e67). Two new issues were introduced and several remain.

THEMES ARE STRICTLY ORTHOGONAL TO LAYOUT. Your scope is COSMETIC ONLY — theme params and render code. Do NOT touch dagua/layout/ or anything that decides node positions. Do NOT modify scripts/graphviz_theme_comparison.py (the harness).

EXACT FIXES TO IMPLEMENT (priority-ranked):

1. **Cluster label renderer override (HIGHEST PRIORITY).** The graphviz_strict ClusterStyle now sets `font_size=10.0`, but the cluster label renderer ignores it and uses a height-based scaling function instead, producing 20-28pt labels in cluster_showcase.png and deep_nesting_4.png. Find the override (likely in `dagua/render/clusters.py` or `dagua/render/mpl.py` — search for cluster-label-rendering code that computes a font size from cluster height). Either:
   - Make the theme's `font_size` value authoritative when explicitly set (don't override it)
   - OR add a config flag in ClusterStyle (e.g. `font_size_scaling: Literal["fixed", "by_height"]`) and set it to "fixed" in graphviz_strict
   Choose whichever is cleaner. Verify on cluster_showcase.png after fix: cluster labels should render at the 10pt declared value, not scaled.

2. **Cluster border invisible (NEW REGRESSION from round 1).** Round 1 dropped cluster opacity to 0.15 to subdue fill, but this also dropped stroke opacity, making cluster borders nearly invisible. Decouple: stroke should remain at full opacity (1.0) while fill is at 0.15. Look at how ClusterStyle.opacity is applied — likely it's a single-channel blend used for both fill and stroke. Either:
   - Add separate `fill_opacity` and `stroke_opacity` fields
   - OR use the existing `border_opacity` field if it exists on ClusterStyle (check)
   - OR change the renderer to apply `opacity` only to fill, with stroke always at 1.0 unless a separate stroke_opacity is set
   Verify on nested_clusters.png and cluster_showcase.png: borders should be clearly visible (medium gray hairline), fills near-transparent.

3. **Stray gray rectangle on complete_k5.png (NEW REGRESSION from round 1).** Round 1 introduced a stray gray background rectangle in the strict panel of complete_k5.png that wasn't there in baseline. Investigate. Likely cause: the cluster fill default with subdued opacity is being applied to graphs that have no clusters (a default cluster being drawn around the whole graph?), or a margin/padding artifact. Fix without re-introducing the background on cluster panels.

4. **Node stroke width.** Reduce `stroke_width` on `_GRAPHVIZ_STRICT_DEFAULT_NODE_STYLE` from 1.3 to 1.0 (the comment in styles.py says "slightly above 1.0 to compensate for AA thinning" but visible result overcorrects per audit). Verify on pipeline.png and diamond.png that node ellipse outlines look like hairlines, not heavy borders.

5. **Back-edge curvature.** Round 1 set the default edge `curvature=0.0` for straight DAG edges, but back-edges (cyclic graphs like state_machine.png and multi_cycle.png) still need some curvature to avoid overlapping with forward-flow nodes. Currently they appear to have curvature ~0.6 producing arcs wider than dot's channel-routed splines. Find where back-edges get their style — likely there's a "back" key in the theme's edge_styles (check). If the strict theme doesn't override the "back" edge style explicitly, add one with `curvature=0.3` (smaller than the default 0.6 but nonzero). Verify on state_machine.png: back-edges should arc tighter, hugging the node column more closely. Note: this is best-effort — exact dot back-edge spline geometry would require reimplementing libspline, which is out of scope (per round-1 audit).

6. **Font sizes — VERIFY before changing.** The round-2 audit reports node font_size 14pt and edge label font_size 14pt look "too large." Native graphviz dot's actual default IS 14pt for both. Before reducing, render a single-graph comparison at native graphviz's default DPI (72) and measure pixel heights. Rules:
   - If pixel-measured node label height matches native dot's at 14pt: leave alone, document as "DPI/scaling artifact, not a real font size diff" in REPORT_round_2.md.
   - If it's actually larger: reduce node `font_size` to 12.0 and edge `label_font_size` to 11.0 in strict theme.
   Use dot's `-Tsvg` output as ground truth — SVG has explicit font-size attributes. Compare with a one-off Python script that renders the same graph through dagua at 72dpi and measures.

ALL FIXES SCOPE: graphviz_strict theme + the rendering code paths it exercises. Do NOT touch the IMPROVED `graphviz` theme. Do NOT touch dagua/layout/. Do NOT touch the comparison harness.
</task>

<completeness_contract>
You are NOT done until:
1. Fixes 1-5 are implemented and visually verified by reading the relevant panels.
2. Fix 6 (fonts) is investigated — either implemented (with pixel-measurement evidence) or documented as not needed.
3. `pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"` passes.
4. Re-run `python scripts/graphviz_theme_comparison.py --output-dir eval_output/graphviz_theme_round_2` and visually verify on cluster_showcase.png, nested_clusters.png, complete_k5.png, pipeline.png, state_machine.png that fixes took effect and no new regressions.
5. ONE commit with message `feat(theme): graphviz_strict cosmetic round 2 — cluster label fix, border opacity, stroke width, back-edge curvature` listing each fix in the body.
6. Report at `.project-context/research/sprint_graphviz_parity/REPORT_round_2.md` with: what fixed, what didn't, font measurement results, any deviations from spec.

Same scope and safety rules as round 1: stay on `develop`, no force-push, single commit at end, theme/render code only.
</completeness_contract>

<verification_loop>
Same pattern as round 1: targeted tier-1 tests during iteration, full tier-2 suite once at end. Re-render the gallery and read PNGs to confirm fixes visually.
</verification_loop>

<missing_context_gating>
Default to most reasonable interpretation. Stop only if:
- A fix's root cause is fundamentally different from what's described and the corrected approach changes scope beyond cosmetic rendering.
- Tests reveal a deeper architectural issue.
Otherwise: keep going, document deviations in REPORT_round_2.md.
</missing_context_gating>

<action_safety>
Same as round 1: theme + render code only. Single working branch `develop`. One commit at end.
</action_safety>

<default_follow_through_policy>
Default to most reasonable low-risk interpretation. Only stop for missing details that change correctness, safety, or irreversibility. Pre-existing test failures unrelated to your changes are not your problem.
</default_follow_through_policy>
