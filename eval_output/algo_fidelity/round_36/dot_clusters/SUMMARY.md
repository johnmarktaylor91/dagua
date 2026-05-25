# Round 36 Dot Clusters Summary

## Source Files Read

- `/home/jtaylor/projects/_references/graphviz/lib/dotgen/cluster.c`
- `dagua/layout/ops/pipelines/dagua_native.py`
- `dagua/layout/ops/state.py`
- `dagua/layout/ops/cluster_geometry.py`
- `dagua/config.py`
- `dagua/layout/engine.py`
- `tests/AGENTS.md`
- `dagua/layout/AGENTS.md`

## Implementation Summary

- Added `_DotClusterSkeleton` metadata and a Python/PyTorch port of
  Graphviz `cluster.c:build_skeleton` counters:
  rankleader ranks, adjusted rankleader UF sizes, and skeleton edge counts.
- Added fidelity-only cluster layout helpers in
  `dagua/layout/ops/pipelines/dagua_native.py`.
- Wired the cluster pass behind `fidelity_mode` selectors:
  `True`, `"dot"`, `"graphviz_dot"`, `"graphviz-dot"`,
  `"dot_clusters"`, `"graphviz_dot_clusters"`, and
  `"graphviz-dot-clusters"`.
- Existing behavior is unchanged when `fidelity_mode` is unset.
- The current integration point is a post-native layout pass. It uses
  deterministic dot-style ranks, reserves per-rank cluster skeleton slots,
  and separates sibling cluster boxes. This isolates the cluster component
  without requiring sibling round-36 rank/mincross/position ports to be
  complete first.

## Tests Added

- `tests/test_pipeline_dagua_native.py`
  - Golden vector for `cluster.c:build_skeleton` edge counts and rankleaders.
  - Golden vector for Graphviz's multi-node-rank UF-size decrement.
  - Fidelity cluster layout sibling separation regression.
  - Public `layout_dagua_native_pipeline(..., fidelity_mode="dot_clusters")`
    invocation smoke test.

## Verification

- Passed: `ruff check dagua/layout/ops/pipelines/dagua_native.py tests/test_pipeline_dagua_native.py --fix`
- Passed: `mypy --follow-imports=silent dagua/cli.py`
- Passed: `pytest tests/test_pipeline_dagua_native.py -x --tb=short -q`
- Passed: `pytest tests/test_graph.py -x --tb=short -q`
- Passed: `pytest tests/test_layout/test_cluster_driver.py tests/test_layout/test_cluster_geometry.py -x --tb=short -q`
- Blocked: `ruff check . --fix` reports pre-existing/sibling round-36 issues outside
  this sub-task scope in `dot_rank.py`, `fmmm.py`, `quadtree.py`, `sfdp.py`,
  and `sugiyama.py`.
- Blocked: `pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"`
  stops during collection because `tests/test_classic_drl.py` cannot import
  `layout_drl` from `dagua.layout.classic`.
- Inconclusive: combined `pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q`
  was terminated by the environment before pytest emitted a failure summary;
  split graph and cluster-layout subsets passed.

## Blockers / Interface Assumptions

- Golden capture from the C runtime for full recursive `expand_cluster` output
  is not isolated in Graphviz's public CLI; the unit tests pin the directly
  portable `build_skeleton` counters from `cluster.c` instead.
- Full bit-exact dot output still depends on sibling sub-components:
  rank assignment, mincross ordering, x-position compaction, and virtual-edge
  routing. This sub-task exposes `_build_dot_cluster_skeletons` and
  `_apply_dot_cluster_fidelity_layout` for the integration codex to compose
  with those phases.
- Dagua's pipeline passes clusters as flat node-id membership rather than
  Graphviz `Agraph_t` subgraphs. The implementation normalizes flat and
  nested membership into descendant leaf ids before applying cluster logic.
