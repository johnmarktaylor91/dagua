# Phase 1 Report: Cluster Tree + Placement Bbox Primitive

## Outcomes

- Added `dagua/layout/ops/cluster_geometry.py` with `ClusterTree`, `ClusterLabelMetrics`,
  `ClusterPlacementBox`, tree accessors, and `compute_cluster_placement_bbox`.
- Added optional lazy `LayoutProblem.get_cluster_tree()` memoization with a lock-backed cache.
- Refactored Matplotlib cluster top/bottom bbox prepasses to delegate shared geometry math to
  `compute_cluster_placement_bbox` while preserving the existing render behavior.
- Added regression tests for one cluster, nested clusters, three siblings, four-level nesting,
  bbox formula sanity, helper accessors, and lazy `LayoutProblem` memoization.
- Excluded `cluster_geometry` from ops auto-registration discovery because it is a pure helper
  module, not an op registry module.

## Visual Regression

Rendered Graphviz-strict PNGs from clean `HEAD` into
`eval_output/cluster_phase_1_baseline/` and from the modified tree into
`eval_output/cluster_phase_1_check/`.

| Panel | Mean pixel L1 | Max channel delta | Sum L1 | Outcome |
|---|---:|---:|---:|---|
| `nested_clusters` | 0.000000 | 0 | 0 | Identical |
| `cluster_showcase` | 0.000000 | 0 | 0 | Identical |
| `transformer_block` | 0.000000 | 0 | 0 | Identical |

## Deviations

- Touched `dagua/layout/ops/__init__.py` to keep ops discovery quiet for the new pure helper
  module. This is outside the original file list but required by the existing package import
  validation.
- Used `LayoutProblem.get_cluster_tree()` rather than a property because `cluster_tree` remains an
  optional public dataclass field for backward-compatible construction.

## Test Results

- `ruff check . --fix`: passed.
- `mypy --follow-imports=silent dagua/cli.py`: passed.
- `pytest tests/test_layout/test_cluster_geometry.py tests/test_layout/test_engine.py tests/test_render/ -x --tb=short -q`: passed, 222 tests, 1 Matplotlib open-figure warning.
- `pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q`: passed, 264 tests, 1 existing metrics warning.
- `pytest tests/test_parity_metrics.py -x --tb=short -q`: passed, 1 test.
