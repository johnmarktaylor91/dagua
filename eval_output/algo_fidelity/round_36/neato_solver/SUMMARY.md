# Round 36 Neato Solver Summary

## Source Files Read

- `/home/jtaylor/projects/_references/graphviz/lib/neatogen/pca.c`
- `/home/jtaylor/projects/_references/graphviz/lib/neatogen/solve.c`
- `/home/jtaylor/projects/_references/graphviz/lib/neatogen/conjgrad.c`
- `/home/jtaylor/projects/_references/graphviz/lib/neatogen/stress.c`
- `/home/jtaylor/projects/_references/graphviz/lib/neatogen/matrix_ops.c`

Note: the requested `solve.c` reference contains dense Gaussian elimination,
not the neato stress conjugate-gradient loop. The CG implementation used by
neato is in `conjgrad.c`, with the callsite and packed-Laplacian construction
in `stress.c`.

## Implementation Summary

- Added `fidelity_mode="graphviz"` to the stress-majorization pipeline as a new
  opt-in path, leaving the existing default and `graphviz_neato` behavior
  available.
- Implemented Graphviz-style PCA initialization from centered graph-distance
  rows, following `PCA_alloc` projection semantics and neato's smart-init axis
  normalization.
- Implemented Graphviz packed upper-triangle matrix indexing, packed matvec,
  float-centering, float-product/double-accumulated dot products, packed stress
  Laplacian construction, and `conjugate_gradient_mkernel`-style CG updates.
- Added neato pipeline routing so integration can invoke the component with
  `fidelity_mode="graphviz"` while preserving existing boolean fidelity
  post-processing behavior.

## Tests Added

- `tests/test_layout/test_neato_solver_fidelity.py`
  - PCA projection source-derived golden for a four-node path.
  - Packed CG source-derived golden for a centered complete-graph Laplacian.
  - Zero-iteration `fidelity_mode="graphviz"` pipeline check proving PCA init is
    selected.

Direct C golden capture for the PCA helper was infeasible in isolation because
Graphviz `PCA_alloc` depends on private allocation/matrix helpers and randomized
`power_iteration` state inside the full neato setup. The tests therefore pin
deterministic source-derived vectors from the same linear algebra.

## Verification

- `python -m pytest tests/test_layout/test_neato_solver_fidelity.py -q`
  - Passed: 3 passed, 2 warnings.
- `ruff check dagua/layout/ops/pipelines/stress_majorization.py dagua/layout/ops/pipelines/neato.py tests/test_layout/test_neato_solver_fidelity.py --fix`
  - Passed.
- `mypy --follow-imports=silent dagua/cli.py`
  - Passed.
- `pytest tests/test_pipeline_stress_majorization.py tests/test_layout/test_neato.py tests/test_layout/test_neato_solver_fidelity.py -x --tb=short -q`
  - Passed: 18 passed, 2 warnings.
- `pytest tests/test_graph.py -x --tb=short -q`
  - Passed: 37 passed, 2 warnings.
- `pytest tests/test_layout/test_neato.py tests/test_layout/test_neato_solver_fidelity.py tests/test_layout/test_stress_maj_fidelity.py -x --tb=short -q`
  - Passed: 11 passed, 2 warnings.

## Blockers / Interface Assumptions

- Full `ruff check . --fix` is currently blocked by unrelated in-flight sprint
  files (`dot_rank.py`, `quadtree.py`, `sfdp.py`, `sugiyama.py`, and related
  pipeline files).
- `pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"`
  fails during collection on an unrelated `tests/test_classic_drl.py` import of
  missing `dagua.layout.classic.layout_drl`.
- `pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q` terminated
  without failure output in the shared workspace; the component-specific layout
  tests and `tests/test_graph.py` pass separately.
- Live compare was not run because this sub-component is only invokable through
  the new `fidelity_mode="graphviz"` parameter; the existing registered
  `classic_neato` variant does not yet pass that parameter.
- Integration codex should wire `classic_neato` or the desired round-36 variant
  to pass `fidelity_mode="graphviz"` when the PCA + packed-CG solver should be
  active.
