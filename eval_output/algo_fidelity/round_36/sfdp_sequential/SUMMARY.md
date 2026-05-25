# Round 36 SFDP Sequential Update

## Source Files Read

- `/home/jtaylor/projects/_references/graphviz/lib/sfdpgen/spring_electrical.c`
  - Read end-to-end.
  - Relevant paths:
    - `spring_electrical_embedding`: default sequential in-place node updates.
    - `spring_electrical_embedding_fast`: batched force accumulation path.
    - `spring_electrical_embedding_slow`: non-quadtree batched debug path.
    - `multilevel_spring_electrical_embedding`: dispatch between slow, fast, and default sequential paths.

## Implementation Summary

- Added `fidelity_mode=True` / `fidelity_mode="graphviz"` dispatch in `dagua/layout/ops/pipelines/sfdp.py`.
- Added `_SFDPGraphvizSequentialStep`, which computes attraction first, then repulsion, normalizes each node force, and immediately writes that node's new coordinate before visiting the next node.
- Added `_apply_graphviz_sequential_refinement`, `_SFDPGraphvizRefineCoarsestLevel`, and `_SFDPGraphvizProlongateAndRefineLevels` so both coarsest and finer levels use the sequential update path under Graphviz fidelity mode.
- Preserved default behavior: `fidelity_mode=False` still uses the existing batched `SFDPRefineCoarsestLevel` and `SFDPProlongateAndRefineLevels` ops.
- Integrated with concurrent Round 36 SFDP sibling work already present in the worktree: Graphviz matrix coarsening and the Graphviz quadtree utility are used only on the fidelity path.

## Tests Added

- `tests/test_pipeline_sfdp_sequential.py`
  - Golden-vector one-step test for the non-quadtree sequential C update order.
  - Pipeline dispatch test proving `fidelity_mode="graphviz"` selects sequential refinement ops.
  - Public wrapper smoke test proving `layout_sfdp_pipeline(..., fidelity_mode="graphviz")` runs end-to-end.

Golden capture note: no instrumented Graphviz C harness was available in this worker. The golden vector is derived directly from the read C loop for a three-node path below the quadtree threshold, where `spring_electrical_embedding` uses exact all-pairs repulsion.

## Verification

- `ruff check dagua/layout/ops/pipelines/sfdp.py tests/test_pipeline_sfdp_sequential.py --fix`
  - PASS: `All checks passed!`
- `mypy --follow-imports=silent dagua/cli.py`
  - PASS: `Success: no issues found in 1 source file`
- `pytest tests/test_pipeline_sfdp.py tests/test_pipeline_sfdp_sequential.py -x --tb=short -q`
  - PASS: `24 passed, 2 warnings in 9.33s`
- `pytest tests/test_graph.py -x --tb=short -q`
  - PASS: `37 passed, 2 warnings in 0.73s`

## Blockers / Interface Assumptions

- `ruff check . --fix` is currently blocked by unrelated Round 36 files in the shared dirty worktree (`dot_rank.py`, `fmmm.py`, `quadtree.py`, `sugiyama.py`). The touched SFDP files pass ruff directly.
- `timeout 300 pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q` timed out without a pytest failure body.
- Final Tier 2 command failed before running this component due unrelated import state: `ImportError: cannot import name 'layout_drl' from 'dagua.layout.classic'`.
- `live_compare` was not run for this component because the comparator cannot currently pass `fidelity_mode="graphviz"` into the `classic_sfdp` competitor variant. Integration codex can wire a Round 36 SFDP variant that sets this parameter.
- I did not stage or commit because `dagua/layout/ops/pipelines/sfdp.py` contains concurrent sibling edits in the same file and the worktree has many unrelated dirty Round 36 changes; staging the file wholesale would capture other workers' changes.
