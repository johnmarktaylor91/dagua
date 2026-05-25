# R36 neato_overlap Summary

## Source Files Read

- `/home/jtaylor/projects/_references/graphviz/lib/neatogen/overlap.c`
- `/home/jtaylor/projects/_references/graphviz/lib/neatogen/quad_prog_solve.c`
- `/home/jtaylor/projects/_references/graphviz/lib/neatogen/quad_prog_vpsc.c`
- `/home/jtaylor/projects/_references/graphviz/lib/vpsc/generate-constraints.cpp`
- `/home/jtaylor/projects/_references/graphviz/lib/vpsc/solve_VPSC.cpp`
- `/home/jtaylor/projects/_references/graphviz/lib/vpsc/block.cpp`
- `/home/jtaylor/projects/_references/graphviz/lib/vpsc/blocks.cpp`
- `/home/jtaylor/projects/_references/graphviz/lib/vpsc/constraint.cpp`
- `/home/jtaylor/projects/_references/graphviz/lib/vpsc/variable.cpp`
- Note: the requested path `lib/neatogen/quad_prog_solver.c` does not exist in the
  reference checkout; the available VPSC sources are `quad_prog_solve.c` and
  `quad_prog_vpsc.c`.

## Implementation Summary

- Added opt-in neato overlap fidelity in `dagua/layout/ops/pipelines/neato.py`.
- New public helper: `remove_neato_overlap_fidelity(...)`.
- New `layout_neato_pipeline(...)` options:
  - `fidelity_mode=False` (`"graphviz"` / `"graphviz_neato"` are also accepted
    when combined with the sibling neato solver integration)
  - `overlap_removal=True`
  - `overlap_method="vpsc"` with `"prism"` accepted
  - `overlap_gap=1/9`
- Existing behavior remains unchanged unless `fidelity_mode=True`.
- Ported the Graphviz VPSC rectangle sweep structure:
  - x pass uses the Adaptagrams neighbor-list constraint heuristic
  - y pass uses the second sweep after x coordinates are updated
  - x rectangles use Graphviz's `1.0001` width scale
  - default total gap matches Graphviz 7.0.5's `2 * DFLT_MARGIN / 72 = 1/9`
- Implemented a deterministic Python active-set block merge solver for VPSC
  feasibility projection.
- Added a PRISM-compatible scaling prepass for `overlap_method="prism"`, followed
  by the VPSC cleanup pass.

## Tests Added

- `tests/test_layout/test_neato_overlap.py`
  - two-node VPSC golden spacing from Graphviz 7.0.5 constants
  - no-op behavior when node sizes are unavailable
  - pipeline gate proving overlap removal only runs when `fidelity_mode=True`

## Verification

- `ruff check dagua/layout/ops/pipelines/neato.py tests/test_layout/test_neato_overlap.py --fix`
  passed.
- `mypy --follow-imports=silent dagua/cli.py` passed.
- `pytest tests/test_layout/test_neato.py tests/test_layout/test_neato_overlap.py -x --tb=short -q`
  passed: 5 tests, 2 existing torchlens deprecation warnings.
- `pytest tests/test_graph.py tests/test_layout/test_neato.py tests/test_layout/test_neato_overlap.py -x --tb=short -q`
  passed: 42 tests, 2 existing torchlens deprecation warnings.
- `ruff check . --fix` did not pass because of pre-existing/sibling sprint
  issues in `dagua_native.py`, `stress_majorization.py`, and `sugiyama.py`.
- `pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"`
  failed during collection on unrelated `tests/test_classic_drl.py`:
  `ImportError: cannot import name 'layout_drl' from 'dagua.layout.classic'`.

## Blockers / Interface Assumptions

- Full PRISM bit-exactness is not complete here because Graphviz's PRISM path
  depends on triangulation and `StressMajorizationSmoother_smooth(...)`. This
  subtask ports the invokable overlap component interface plus VPSC constraint
  generation/projection, leaving exact CG/triangulation smoother parity for the
  sibling solver/integration work.
- The VPSC solver currently ports the merge/satisfy projection behavior needed
  for static overlap removal. Graphviz's block-splitting refinement from
  Adaptagrams remains an integration follow-up if strict optimizer parity is
  required on dense multi-constraint cases.
- `node_sizes` is required for overlap removal. When it is `None`, the helper
  returns coordinates unchanged because Graphviz rectangle constraints cannot be
  generated.
- Live compare was not run for this isolated subcomponent; the new behavior is
  only invokable end-to-end when callers pass `fidelity_mode=True`.
