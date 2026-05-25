# Round 36 Quadtree Summary

## Source Files Read

- `/home/jtaylor/projects/_references/graphviz/lib/sparse/QuadTree.c`
- `/home/jtaylor/projects/_references/graphviz/lib/sparse/QuadTree.h`
- `/home/jtaylor/projects/_references/graphviz/lib/sparse/general.c`
- `/home/jtaylor/projects/_references/graphviz/lib/sparse/general.h`
- `/home/jtaylor/projects/_references/graphviz/lib/sfdpgen/spring_electrical.c`

Note: the prompt path `/home/jtaylor/projects/_references/graphviz/lib/sfdpgen/QuadTree.c`
does not exist in this checkout. The shared Graphviz quadtree implementation is under
`lib/sparse/`.

## Implementation Summary

- Added `dagua/layout/ops/quadtree.py`, a Graphviz-compatible sparse quadtree port.
- Preserved Graphviz-specific behavior:
  - point-list root bounds: max span floor `1e-5`, width scale `0.52`;
  - quadrant bit order and child-center updates;
  - max-level leaf head insertion order;
  - Graphviz's unweighted `average` update semantics;
  - `QuadTree_get_supernodes` traversal/count behavior;
  - two-pass `QuadTree_get_nearest`;
  - symmetric `QuadTree_get_repulsive_force` cell-cell / leaf-leaf accumulation.
- Added public helpers for integration:
  - `GraphvizQuadTree.from_points(...)`
  - `GraphvizQuadTree.get_supernodes(...)`
  - `GraphvizQuadTree.get_repulsive_force(...)`
  - `graphviz_supernode_repulsive_force(...)`
  - `graphviz_spring_electrical_repulsive_forces(...)`
- Wired SFDP Graphviz fidelity-mode sequential refinement to use the Graphviz
  quadtree supernode path when `N >= 45`.
- Registered `quadtree` in the ops module discovery allowlist.

## Tests Added

- `tests/test_ops_quadtree.py`
  - root width / center / quadrant-center golden vectors;
  - max-level leaf order and average golden vector;
  - `get_supernodes` opening and count golden vectors;
  - symmetric repulsive force and normalized counts golden vector;
  - nearest-point lookup golden vector;
  - public repulsive wrapper quadtree-path smoke.

## Verification

- `ruff check dagua/layout/ops/quadtree.py dagua/layout/ops/sfdp.py dagua/layout/ops/pipelines/sfdp.py dagua/layout/ops/__init__.py tests/test_ops_quadtree.py --fix`
  - Pass
- `pytest tests/test_ops_quadtree.py -q`
  - Pass: `6 passed, 2 warnings`
- `pytest tests/test_pipeline_sfdp.py -q`
  - Pass: `21 passed, 2 warnings`
- `pytest tests/test_pipeline_fmmm.py -q`
  - Pass: `20 passed, 2 warnings`
- `mypy --follow-imports=silent dagua/cli.py`
  - Pass
- Direct SFDP fidelity smoke:
  - `layout_sfdp_pipeline(..., num_nodes=50, steps=1, fidelity_mode="graphviz")`
  - Pass: finite `torch.Size([50, 2])` output
- `pytest tests/test_graph.py -q`
  - Pass: `37 passed, 2 warnings`

## Blockers / Interface Assumptions

- Full `ruff check . --fix` is currently blocked by pre-existing line-length
  errors in untracked sibling file `dagua/layout/ops/pipelines/dot_rank.py`.
- Final non-slow suite is currently blocked during collection by an unrelated
  import error: `tests/test_classic_drl.py` cannot import `layout_drl` from
  `dagua.layout.classic`.
- `pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q` did not emit
  a failure, but exceeded a 240s explicit timeout in this shared worktree.
- The utility supports both Graphviz quadtree APIs needed by integration:
  per-node supernode queries for sequential SFDP/FDP-style loops and symmetric
  cell-force accumulation for fast Barnes-Hut loops.
- Commit was not created because the worktree contains concurrent sibling
  sprint edits, including overlapping edits in `dagua/layout/ops/pipelines/sfdp.py`.
  Staging the full file would include unrelated hunks outside this sub-task.
