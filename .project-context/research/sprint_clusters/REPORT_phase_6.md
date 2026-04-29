# Phase 6 Corrective Round Report

## Changes

- `dagua/render/mpl.py`: added shared render-cluster bbox computation, parent/child containment enforcement, axes expansion from final bboxes, cluster label z-order above node fills, and sampled edge-body segmentation for foreign-cluster bypass gaps.
- `scripts/graphviz_theme_comparison.py`: made `--use-dagua-placement` request `LayoutConfig(algorithm="fr", cluster_aware=True, steps=80)` so the harness actually exercises the cluster-aware driver instead of falling back through `dagua_native`.
- `tests/test_render/test_edge_cluster_clip.py`: added regression coverage for sibling-cluster edge body span and bypass-edge body splitting.

## Per-Fix Outcome

- G1: Fixed/verified. `deep_nesting_4` renders Level 4 inside Level 3 inside Level 2 inside Level 1. Parent bboxes now expand after child bboxes are known, and axes expansion uses the same final bboxes.
- G2: Fixed/verified. `microservices` Search Service to DB/cache/index edges retain visible bodies between Service Layer and Data Layer perimeters. A regression test now asserts a sibling-cluster edge body spans between both cluster perimeters.
- G3: Fixed/verified. Cluster labels render above node fills (`1.5 + depth * 0.01`), so `nested_clusters` shows a readable `Outer Group` label even when node A overlaps the top band.
- G4: Implemented/partially verified. The renderer now segments body-only edges around foreign cluster interiors; the unit test covers a source-out/target-out bypass crossing. Visual `transformer_block` is improved but remains sensitive to routes that run nearly coincident with a cluster border, so this should stay on the Phase 7 audit list.
- G5: Fixed/verified. The Dagua-placement gallery no longer emits the `dagua_native` fallback warning. Generated:
  - `eval_output/cluster_phase_6_check_dot_positions/`
  - `eval_output/cluster_phase_6_check_dagua_placement/`
  - `eval_output/cluster_phase_6_check/`

## Visual Observations

- `deep_nesting_4`: concentric nesting restored; edge bodies are visible instead of arrowhead-only stubs.
- `microservices`: Search Service fanout bodies are visible to the Data Layer targets.
- `nested_clusters`: `Outer Group` label is no longer fragmented by node A fill; A still overlaps the outer top band under dot-position injection.
- `transformer_block`: bypass clipping is structurally supported and tested; the visual panel still has near-border bypass geometry worth re-auditing.
- `cluster_showcase`: Dagua placement separates major clusters better than dot-position injection.
- `cross_cluster_edges`: Dagua placement does not improve this panel; it remains a placement-quality concern rather than a render bbox cap artifact.

## Assumptions

- Root cluster bboxes keep the Phase 5 root-level placement cap for their own label/min-width expansion, but parent containment may expand a root when required to contain a finalized child bbox. This preserves the stricter semantic invariant from G1.
- For bypass edges, sampled curve segmentation is acceptable for render-time gaps; routing itself remains unchanged.

## Test Results

- `ruff check . --fix`: pass.
- `mypy --follow-imports=silent dagua/cli.py`: pass.
- `pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q`: pass, `270 passed, 6 warnings`.
- `pytest tests/test_render/test_mpl.py::test_deep_cluster_bounds_stay_inside_render_axes tests/test_render/test_edge_cluster_clip.py -x --tb=short -q`: pass, `7 passed`.
- `pytest tests/test_layout/ tests/test_render/ tests/test_style.py tests/test_parity_metrics.py -x --tb=short -q`: pass, `397 passed, 8 warnings`.
- `python scripts/parity_metrics.py`: pass, `cluster_rect_missing` remains `41/41` in tolerance (`100.00%`); overall `99.27%`.
- `pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"`: blocked by existing collection error, `ImportError: cannot import name 'layout_drl' from 'dagua.layout.classic'` in `tests/test_classic_drl.py`.

## Controversial Choices

- `--use-dagua-placement` now uses FR as the cluster-aware inner algorithm because the cluster-aware driver explicitly supports `fr`, `kk`, `fa2`, and `sfdp`, while default `dagua_native` currently warns and falls back to flat placement.
- Bypass gaps are implemented as render-body segmentation rather than route modification or clip-path exclusion. This keeps the fix local to rendering.

## Concerns

- `cross_cluster_edges` remains poor with Dagua placement; that is likely a placement-driver issue.
- `transformer_block` should be re-audited because visually near-border bypasses can still read as continuous when the route hugs the cluster stroke.
- The full non-slow suite remains blocked by the pre-existing missing `layout_drl` export/import.

## Knowledge

- Render-time cluster geometry had three consumers (`_draw_clusters`, edge clipping, axes expansion); they now share final bbox computation to avoid drift.
- The Graphviz comparison harness must not call bare `dagua.layout()` for cluster placement until `dagua_native` supports cluster-aware placement directly.
