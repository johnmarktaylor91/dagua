<task>
Round 3 of cosmetic parity work for `graphviz_strict` theme — should be the closing round. Read prior context:
- `.project-context/research/sprint_graphviz_parity/AUDIT_round_3.md` (latest visual audit, with PASS/PARTIAL/FAIL verdicts on round-2 fixes and explicit fix recommendations)
- `.project-context/research/sprint_graphviz_parity/REPORT_round_2.md` (your prior implementation report)

The round 3 audit recommends CONTINUE for one more round, then STOP — you are that round.

THEMES ARE STRICTLY ORTHOGONAL TO LAYOUT. Cosmetic-only scope. Do NOT touch dagua/layout/ or scripts/graphviz_theme_comparison.py (the harness).

EXACT FIXES (priority ordered — #1 is most impactful; do them in order, verify each):

1. **CRITICAL — Cluster label `font_size_scaling="fixed"` is not landing.** Round 2 added this field to ClusterStyle and wired it in `dagua/render/mpl.py` `_cluster_font_size_data()`, but the audit's PASS check shows cluster_showcase.png "Large Cluster With Longer Label" still renders at ~22-28pt. Investigate why the new code path isn't being reached for these clusters:
   - Does `_cluster_font_size_data()` get called for top-level/large clusters, or is there a separate code path?
   - Is the round-2 fix only applying when the cluster is small (e.g. when the "fixed" branch fires only below some height threshold)?
   - Is the GRAPHVIZ_STRICT_THEME's ClusterStyle reaching the renderer correctly (the field might be on ClusterStyle but not being deepcopied through some path)?
   Add a temporary debug print at the top of `_cluster_font_size_data()` (and any sibling functions like `_cluster_label_font` or similar), regenerate cluster_showcase.png, see what fires. Then fix so the declared 10pt value is authoritative for ALL cluster sizes when `font_size_scaling="fixed"`. Remove the debug prints before committing. Verify on cluster_showcase.png AND deep_nesting_4.png.

2. **DPI normalization for label fonts.** The audit identified a likely 72dpi-vs-96dpi unit error: dot's SVG declares 14pt at 72dpi (= ~10.5pt at 96dpi), but dagua renders 14pt at 96dpi making node/edge labels visibly oversized. Reduce in graphviz_strict:
   - `_GRAPHVIZ_STRICT_DEFAULT_NODE_STYLE.font_size`: 14.0 → 10.5
   - `EdgeStyle.label_font_size` (in the strict theme's default edge style): 14.0 → 10.5
   - `GraphStyle.edge_label_font_size`: 14.0 → 10.5
   Note: round 2 measured SVG and said 14pt was correct. The visual evidence and the DPI ratio (14 * 72/96 = 10.5) suggests the unit-system mismatch IS the issue. Verify on pipeline.png and balanced_binary_tree.png after the change — labels should fit comfortably inside ellipses with clear horizontal whitespace, matching dot's appearance.

3. **Cluster border color too dark.** Round 2 set `border_opacity=1.0` to fix the round-1 invisible-border regression, but black at full opacity is now too prominent vs dot's light-gray hairline. Two paths, your choice:
   - (a) Change cluster `stroke` color from `"#666666"` to a lighter gray (e.g. `"#999999"` or `"#AAAAAA"`) and keep border_opacity at 1.0
   - (b) Keep stroke at "#666666" but reduce border_opacity to ~0.5
   Either works; (a) is cleaner. Verify on nested_clusters.png — cluster borders should look like light-gray hairlines, not black lines.

4. **Parallel-edge arc sign alternation (renderer logic, not theme value).** On complete_k5.png all parallel arcs between the same node pair fan to one side, making the K5 layout look lopsided. Native dot alternates arc curvature sign (+/-) for successive edges between the same node pair so arcs distribute symmetrically. Find where edge curvature is applied (likely `dagua/render/edges/collection.py` or `geometry.py`) and add: when a graph has multiple edges between the same node pair, alternate the sign of `curvature` for each successive edge. This should be theme-agnostic behavior (apply for ALL themes, but graphviz_strict will exhibit it most cleanly given curvature=0.3 on back-edges and 0.0 on forward edges). Verify on complete_k5.png.

5. **`tee` arrowhead shape.** On arrow_types.png, dagua's `tee` renders as a small filled triangle (same as `normal`/`vee`); native dot's `tee` is a flat horizontal bar perpendicular to the edge (a T-stop). Find the `tee` arrowhead implementation in `dagua/render/edges/arrowheads.py` and either:
   - Implement `tee` as a perpendicular bar primitive
   - Or alias `tee` to an existing bar/perpendicular primitive (verify one exists first)
   Verify on arrow_types.png column labeled "tee".

6. **multi_cycle.png stray gray rectangle.** Same class of bug as the complete_k5 stray rectangle that round 2 fixed. There's a faint light-gray rectangular wash behind multi_cycle's strict panel. Investigate (likely a graph-level cluster default still rendering when no cluster is defined). Apply the same fix pattern as round 2 used for complete_k5.

7. **Node stroke weight 1.0 → 0.75 (minor polish).** Reduce `_GRAPHVIZ_STRICT_DEFAULT_NODE_STYLE.stroke_width` from 1.0 to 0.75. Verify on pipeline.png and diamond.png — node ellipse outlines should be hairlines, not visible thin lines. If 0.75 looks too thin (disappears entirely), use 0.85 instead.

8. **Back-edge curvature 0.3 → 0.20 (minor polish).** In the strict theme's "back" edge style, reduce curvature from 0.3 to 0.20. Verify on state_machine.png and multi_cycle.png — back-arcs should hug the node column tighter.

DO NOT TOUCH: the IMPROVED `graphviz` theme, dagua/layout/, scripts/graphviz_theme_comparison.py.
</task>

<completeness_contract>
Not done until:
1. All 8 fixes implemented and visually verified.
2. Tier-2 tests pass (excluding pre-existing test_classic_drl.py import error which is out of scope per round-2 report).
3. Re-rendered gallery at `eval_output/graphviz_theme_round_3` and visually verified at minimum: cluster_showcase.png, deep_nesting_4.png, pipeline.png, complete_k5.png, arrow_types.png, multi_cycle.png.
4. ONE commit: `feat(theme): graphviz_strict cosmetic round 3 — cluster label scaling fix, DPI font normalization, lighter cluster borders, parallel-arc alternation, tee arrowhead, polish`.
5. REPORT_round_3.md written with what fixed, what didn't, and any deviations.

Same scope/safety rules as prior rounds.
</completeness_contract>

<verification_loop>
Same pattern. For fix #1 (cluster label) you MUST add debug instrumentation and confirm via reading the regenerated cluster_showcase.png that the fix lands — this is the carry-over critical issue.
</verification_loop>

<missing_context_gating>
Default to most reasonable interpretation. Investigate before changing if uncertain about renderer paths. Document deviations in REPORT_round_3.md.
</missing_context_gating>

<action_safety>
Theme + render code only. develop branch. One commit at end.
</action_safety>

<default_follow_through_policy>
Default to most reasonable low-risk interpretation. Pre-existing test failures are not your problem. Keep going.
</default_follow_through_policy>
