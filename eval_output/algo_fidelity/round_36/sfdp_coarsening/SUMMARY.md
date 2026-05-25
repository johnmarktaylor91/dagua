# Round 36 SFDP Coarsening Summary

## Source Files Read

- `/home/jtaylor/projects/_references/graphviz/lib/sfdpgen/Multilevel.c`
  - Read end-to-end. Port target: `maximal_independent_edge_set_heavest_edge_pernode_supernodes_first`, `Multilevel_coarsen_internal`, `Multilevel_coarsen`, and `Multilevel_establish`.
- `/home/jtaylor/projects/_references/graphviz/lib/sfdpgen/Multilevel.h`
  - Read for `MAX_CLUSTER_SIZE = 4` and multilevel payload semantics.
- `/home/jtaylor/projects/_references/graphviz/lib/sparse/SparseMatrix.c`
  - Read `SparseMatrix_decompose_to_supervariables` because `Multilevel.c` delegates the supervariable partition there.
- `/home/jtaylor/projects/_references/graphviz/lib/util/random.c`
  - Read `gv_permutation` to identify the unmatched-node permutation dependency.

## Implementation Summary

- Added opt-in `fidelity_mode` to `build_sfdp_pipeline()` and `layout_sfdp_pipeline()`.
- Added `BuildGraphvizSFDPMatrixHierarchy`, selected only when `fidelity_mode=True`.
- Ported Graphviz's supervariable-first clustering:
  - exact sparse-matrix column-pattern refinement,
  - `MAX_CLUSTER_SIZE = 4` chunking,
  - unmatched-node heavy-edge pairing,
  - singleton tail assignment.
- Ported the matrix coarse graph semantics used by `R * A * P` with diagonal removal into Dagua's `GraphData` representation by summing fine edges between coarse clusters.
- Ported the Graphviz wrapper loop that composes internal coarsening passes until the retained node ratio is at most `0.75`, or until no further internal pass is available.
- Existing SFDP behavior remains the default path; the new hierarchy is only used with `fidelity_mode=True`.

## Tests Added

- Added coverage in `tests/test_pipeline_sfdp.py`:
  - golden supervariable groups for a complete 4-partite graph,
  - golden matrix-coarsened mapping `[0, 0, 1, 1, 2, 2, 3, 3]`,
  - golden coarse complete-graph weights of `4.0` on all six coarse edges,
  - public pipeline-op selection for `fidelity_mode=True`,
  - pipeline extras populated with the Graphviz matrix hierarchy.

## Verification

- `python -m py_compile dagua/layout/ops/pipelines/sfdp.py tests/test_pipeline_sfdp.py`
  - Passed.
- `mypy --follow-imports=silent dagua/cli.py`
  - Passed: `Success: no issues found in 1 source file`.
- `pytest tests/test_pipeline_sfdp.py -x --tb=short -q`
  - Blocked during import by unrelated current worktree state:
    `ModuleNotFoundError: No module named 'dagua.layout.ops.dot_mincross'`.
- `ruff check . --fix`
  - Blocked by unrelated current worktree state:
    `F821 Undefined name use_graphviz_mincross` in `dagua/layout/ops/sugiyama.py:605`.
- `pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"`
  - Blocked during collection by unrelated current worktree state:
    `ImportError: cannot import name 'layout_drl' from 'dagua.layout.classic'`.
- `pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q`
  - Started and produced progress, but did not complete after several minutes and was killed to avoid leaving a long-running session open.

## Blockers / Interface Assumptions

- Golden capture from a live Graphviz binary was not isolated in this subtask. The golden vectors used here are direct consequences of the read C code on a complete multipartite graph where all clustering is performed by supervariables, so the unmatched-node RNG path is not involved.
- Unmatched-node permutation uses Dagua's seeded `torch.Generator` rather than Graphviz's process-global `gv_random` stream. This is an integration assumption for the RNG/sequence-fidelity codex.
- The new matrix coarsening stores only the fine-to-coarse mapping and coarse graph needed by Dagua's current SFDP prolongation. Graphviz's row-normalized `R` matrix is not stored separately because the existing pipeline does not consume interpolation matrices directly.
