<task>
You are implementing a focused round of cosmetic fixes to dagua's `graphviz_strict` theme so it renders panels visually closer to native graphviz `dot`. Themes in dagua are STRICTLY orthogonal to layout algorithms — your scope is COSMETIC ONLY (theme params, render code, arrowhead geometry, edge curvature). Do NOT touch dagua/layout/ or anything that decides node positions. Do NOT modify dagua/eval/.

Repo: /home/jtaylor/projects/dagua
Branch: develop (already checked out, single working branch)
Reference docs: AGENTS.md (quality gates, type hints, NumPy docstrings), .project-context/conventions.md
Audit you are implementing: .project-context/research/sprint_graphviz_parity/AUDIT_round_1.md (read it for full per-panel context)

EXACT FIXES TO IMPLEMENT (5 items; all derived from the round-1 audit):

1. **Edge curvature default for graphviz_strict.** The `graphviz_strict` theme's default EdgeStyle currently inherits a nonzero curvature, so even straight-line DAG edges render as visible bezier arcs. Set `curvature=0.0` (or whatever the EdgeStyle field name is — verify in dagua/styles.py) in GRAPHVIZ_STRICT_THEME's default EdgeStyle. If the rendering pipeline ignores curvature=0 and still bends edges, also verify dagua/render/edges/collection.py treats curvature=0 as a true straight line (single straight segment between endpoints). Compare side-by-side with native dot's straight-rank-separated edges on diamond.png and balanced_binary_tree.png after your fix.

2. **Arrowhead direction is INVERTED.** On every edge in every panel, dagua's arrowhead points toward the SOURCE node instead of the TARGET node. Find where `tangent` (or equivalent direction vector) is passed to `build_arrowhead` (likely in dagua/render/edges/collection.py — search for `build_arrowhead` callers). Either flip the sign of the tangent vector at the call site, or fix the upstream computation so the vector represents "from tip back into edge body" as the docstring presumably requires. Verify on pipeline.png: arrowheads must point DOWN into each receiver node, not UP into each emitter.

3. **Arrowhead size too small.** Currently `arrow_length=7.0, arrow_width=4.5` in graphviz_strict (verify exact numbers in dagua/styles.py). Native dot renders arrowheads at approximately 10pt x 7pt (chunky filled triangles). Bump to `arrow_length=10.0, arrow_width=7.0` in GRAPHVIZ_STRICT_THEME's default EdgeStyle (and any per-edge-type styles in the theme that override). Verify on pipeline.png: arrowheads should look chunky, not pinpoint.

4. **Cluster styling too heavy.** In GRAPHVIZ_STRICT_THEME's default ClusterStyle (or wherever cluster theme params live), reduce label `font_size` from current value to ~10pt, reduce `opacity` from current ~0.6-0.7 to ~0.15, and set `depth_fill_step=0.0` and `depth_stroke_step=0.0` so nested clusters don't progressively darken. Confirm by inspecting on cluster_showcase.png and nested_clusters.png — cluster fills should be near-transparent, labels should be subordinate to node labels.

5. **"circle" arrowhead alias bug.** In dagua/render/edges/arrowheads.py, the alias table maps `"circle"` to `"odot"` (hollow). Native dot renders `circle` as a FILLED dot. Change the alias from `"circle": "odot"` to `"circle": "dot"`. Verify on arrow_types.png: the "circle" arrowhead column should now be a filled black dot, matching dot's panel.

IMPORTANT: graphviz_strict is the literal-match theme; the IMPROVED `graphviz` theme (also in styles.py) intentionally departs from dot. Do NOT touch the IMPROVED theme. Only modify GRAPHVIZ_STRICT_THEME (or whatever the exact name is — verify in styles.py).

Each of these fixes is small (often a single field change). Total diff should be well under 100 lines.
</task>

<completeness_contract>
You are NOT done until:
1. All 5 fixes above are implemented in code.
2. `pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"` (Tier 2) passes — fix any breakage your changes cause. Existing tests for the strict theme may need updating to match new expected values.
3. You re-run the comparison script (`python scripts/graphviz_theme_comparison.py --output-dir eval_output/graphviz_theme_round_1`) and verify by reading the resulting PNGs that the fixes took effect on at least 4 panels: pipeline.png (arrows now chunky and pointing down), diamond.png (edges now straight), arrow_types.png (circle is filled), nested_clusters.png (clusters subdued).
4. You commit your changes with a single conventional-commits message: `feat(theme): graphviz_strict cosmetic round 1 — straight edges, larger correctly-oriented arrowheads, subdued clusters, filled circle arrow`. Include a short body listing each of the 5 fixes.
5. You leave a written report at `.project-context/research/sprint_graphviz_parity/REPORT_round_1.md` summarizing what you did, what tests you ran, any deviations from the spec, and any follow-ups discovered. The architect (Claude) will read this report.

If a fix turns out to require changes beyond what's described (e.g. arrowhead direction is actually correct but a different bug causes the visual symptom), document the deviation clearly in REPORT_round_1.md and proceed with the corrected approach.
</completeness_contract>

<verification_loop>
After each fix:
- Run targeted tests: `pytest tests/test_styles.py tests/test_render -x --tb=short -q` (or whichever tests cover render + styles in this repo)
- Re-run the comparison on a single representative graph if you have time. The script supports running on a subset; if not, render one panel ad-hoc with a small Python snippet to eyeball.

Once all 5 fixes land and tier-1 tests pass:
- Run `pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"` once at the end.
- Run `python scripts/graphviz_theme_comparison.py --output-dir eval_output/graphviz_theme_round_1` to regenerate the gallery.
- Read at least 4 PNGs from eval_output/graphviz_theme_round_1/three_way/ to confirm visual change happened.

If any fix doesn't visually take effect, debug it before reporting done.
</verification_loop>

<missing_context_gating>
Stop and ask only if:
- A field name in styles.py doesn't match what's described (e.g. no `curvature` field). In that case investigate the actual field structure first; only escalate if the underlying concept doesn't exist.
- A test you cannot understand fails after your changes. (Try fixing it first; if root cause unclear after 10 min, document and continue.)

Otherwise: default to most reasonable low-risk interpretation of the spec and keep going. Do NOT stop and ask before each fix.
</missing_context_gating>

<action_safety>
Scope is COSMETIC RENDERING ONLY:
- OK to touch: dagua/styles.py (theme params), dagua/render/ (rendering code), tests/ (update assertions).
- DO NOT touch: dagua/layout/, dagua/eval/, scripts/graphviz_theme_comparison.py (harness), docs/, .project-context/ except your own report file.
- Single working branch is `develop`. Stay on it. Do not create a new branch.
- Make ONE commit at the end. No amends. No force-push. Don't push (Claude will verify locally first).
</action_safety>

<default_follow_through_policy>
Default to the most reasonable low-risk interpretation of any ambiguity and keep going. Only stop for missing details that change correctness, safety, or irreversible actions. Pre-existing test failures unrelated to your changes are NOT your problem — flag them in REPORT_round_1.md and move on.
</default_follow_through_policy>
