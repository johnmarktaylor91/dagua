<task>
Phase 3 of the cluster sprint — render parity polish. Per `/home/jtaylor/projects/dagua/.project-context/research/sprint_clusters/DESIGN.md` §6 Phase 3 + §3 + §4.5–§5.6.

Read DESIGN.md fully first. Especially §3 (current render), §4 (graphviz comparison), §5.6 (render-side cluster path generation).

Repo: `/home/jtaylor/projects/dagua` (already on `develop`). Single working branch.

## Context

Phase 2 (`aed468a`) landed cluster-aware placement for FR/KK/FA2/SFDP. Now the render side needs to match dot's visual cluster behavior. The complaints from prior visual audits were:
- Cluster border stroke crosses through cluster label text
- Cluster labels sit top-left in dagua but top-center in dot (graphviz themes)
- Some themes have heavy cluster fills; dot's are near-transparent

After Phase 2, when dagua produces its own cluster-aware positions (NOT dot's positions), the structural overlap issues should be resolved by placement. This phase focuses on the cosmetic rendering details that still differ from dot regardless of placement.

## Goal

Make dagua's cluster RENDERING visually match dot's:
1. Cluster label centered at top of cluster bbox (graphviz themes)
2. Label background mask is opaque + matches graph background, applied universally
3. Cluster border path is clean — label appears clearly readable, no stroke visible through it
4. Default `ClusterStyle.label_position` is theme-conditional (graphviz themes get top-center)

## Files to touch

### 1. `dagua/styles.py` — ClusterStyle defaults

a. `ClusterStyle.label_position` default stays `"top-left"` (don't change global default — it's a deliberate dagua-style choice). However, in `GRAPHVIZ_STRICT_THEME` and `GRAPHVIZ_THEME` cluster_style, set `label_position="top-center"` to match dot.

b. `ClusterStyle.label_background` default → introduce `"@background"` sentinel string (or new field) that resolves to the graph's `background_color` at render time. Set in graphviz themes' cluster_style.

c. Set `label_background_padding=(4.0, 2.0)` (x, y) and `label_background_opacity=1.0` for the graphviz themes' cluster_style.

### 2. `dagua/render/mpl.py` — universal label background mask

In `_draw_clusters`, the existing graphviz_strict-only label background mask logic should be generalized:

- When `cluster_style.label_background` is set (any theme, any value) → render the rectangle behind the label.
- When the value is `"@background"` (or whatever sentinel) → resolve to `graph.graph_style.background_color`.
- Apply the same `label_background_padding` and `label_background_opacity` as currently used for graphviz_strict.

Effect: in any theme that opts into a label background, the cluster border stroke is masked behind the label cleanly.

### 3. `dagua/render/mpl.py` — top-center label support

If `label_position="top-center"`, position the label at `(cluster_bbox.center_x, cluster_bbox.top + label_inset_y)`. Verify the existing label-anchor logic supports this; if not, extend it.

### 4. (Optional, defer if complex) Path break around label

Per DESIGN.md §5.6 Option B, generate the cluster polygon as four sub-paths with a gap centered on the label of width `label_width + 2*pad`. This is cleaner for SVG export. SKIP for now if the universal background mask (above) gives visually equivalent output for raster targets — dot itself doesn't break the path; it relies on the visual layering. Document the deferral if you skip.

## Verification

1. **New tests** in `tests/test_render/test_cluster_label.py` (new file or extend existing):
   - Render a cluster with `label_position="top-center"` — assert the label's x-anchor is within 2px of cluster bbox center_x.
   - Render a cluster with `label_background="@background"` — assert the rendered output has an opaque rectangle (no transparent pixel) covering the label area within the cluster.
   - Render `nested_clusters` with `graphviz_strict` theme — assert no cluster stroke pixels intersect cluster label patches.

2. **Visual regression on cluster panels**: `python scripts/graphviz_theme_comparison.py --output-dir eval_output/cluster_phase_3_check`.
   - Read 3 cluster panels: nested_clusters, cluster_showcase, transformer_block.
   - Visually verify: labels are top-center for graphviz themes, opaque background, no stroke-through-label.

3. **Existing tests pass**: `pytest tests/test_layout/ tests/test_render/ tests/test_style.py tests/test_parity_metrics.py -x --tb=short -q`. Update assertions if cluster style values changed in graphviz themes.

4. **Parity metric stays >= 99%**: `python scripts/parity_metrics.py`.

## Out of scope

- Phase 4 territory (edge clipping at cluster perimeter)
- Don't change the placement code (Phase 2 territory)
- Don't break the dagua-default theme — `label_position="top-left"` stays the global default

## Completeness contract

Not done until:
1. `ClusterStyle.label_background` resolution to `@background` works.
2. graphviz themes' cluster_style sets `label_position="top-center"`, `label_background="@background"`, `label_background_padding=(4,2)`, `label_background_opacity=1.0`.
3. `_draw_clusters` renders universal label background mask when configured.
4. Top-center label anchor works for `label_position="top-center"`.
5. New tests pass.
6. Cluster panels visually verified (no stroke through label, top-center label in graphviz themes).
7. `pytest tests/` non-slow suite passes (excluding pre-existing test_classic_drl.py import error).
8. ONE commit on develop: `feat(cluster): phase 3 — render parity (top-center label, universal background mask)`.
9. REPORT at `.project-context/research/sprint_clusters/REPORT_phase_3.md`.

## Reply format

Per-step outcome, commit SHA, before/after observations. ≤200 words.
</task>

<missing_context_gating>
If `ClusterStyle.label_background` already accepts a sentinel string elsewhere, use that mechanism rather than inventing a new one. If the existing graphviz_strict implementation is already correct, just generalize it (delete the gate, make it apply to any theme that opts in).
</missing_context_gating>

<action_safety>
develop branch. ONE commit at end.
</action_safety>
