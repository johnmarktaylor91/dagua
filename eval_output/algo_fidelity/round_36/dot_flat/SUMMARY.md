# Round 36 dot_flat Summary

## Source files read

- `/home/jtaylor/projects/_references/graphviz/lib/dotgen/flat.c`
- `dagua/layout/ops/pipelines/dagua_native.py`
- `tests/test_layout/test_dagua_flat.py`
- `dagua/config.py`
- `dagua/layout/ops/state.py`
- `dagua/layout/engine.py`

## Implementation summary

- Added Graphviz-dot flat/self/multi-edge preprocessing helpers in
  `dagua/layout/ops/pipelines/dagua_native.py`.
- Ported the `flat.c::checkFlatAdjacent` blocker rule for same-rank edges:
  normal nodes and labeled virtual nodes block adjacency; unlabeled virtual
  nodes do not.
- Added fidelity-only preprocessing for node placement:
  self-loops are removed from placement constraints, duplicate directed
  multi-edges keep the first representative only, and metadata records
  self-loop ids, duplicate ids, flat edge ids, representative ids, and flat
  adjacency flags for the routing/label integration pass.
- Wired the preprocessing behind `fidelity_mode` aliases `True`, `dot`,
  `graphviz_dot`, `graphviz-dot`, `dot_flat`, `graphviz_dot_flat`, and
  `graphviz-dot-flat`. Default behavior is unchanged.
- Preserved narrow sibling fidelity modes such as `dot_position` by using a
  flat-specific selector for this component.

## Tests added

- `tests/test_layout/test_dot_flat_fidelity.py`
  - Golden blocker vectors for `checkFlatAdjacent` semantics.
  - Self-loop and multi-edge representative filtering vectors.
  - Explicit fidelity-mode selector coverage.
  - Native pipeline smoke test with self-loop, flat, and duplicate edges.

## Verification

- `ruff check dagua/layout/ops/pipelines/dagua_native.py tests/test_layout/test_dot_flat_fidelity.py --fix`
  - Passed.
- `python -m py_compile dagua/layout/ops/pipelines/dagua_native.py tests/test_layout/test_dot_flat_fidelity.py`
  - Passed.
- `mypy --follow-imports=silent dagua/cli.py`
  - Passed.
- `pytest tests/test_layout/test_dot_flat_fidelity.py tests/test_layout/test_dagua_flat.py -x --tb=short -q`
  - Passed: `9 passed, 2 warnings in 90.18s`.
- `ruff check . --fix`
  - Failed on unrelated parallel-sprint edits in `fmmm.py`, `sfdp.py`,
    `stress_majorization.py`, and `sugiyama.py` (undefined names and line
    length outside this sub-task).
- `pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q`
  - Process exited with code `-1` after several minutes without a traceback.
- `pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"`
  - Failed during collection on pre-existing `tests/test_classic_drl.py` import:
    `ImportError: cannot import name 'layout_drl' from 'dagua.layout.classic'`.

## Blockers / interface assumptions

- `dagua_native` currently receives `edge_index`, node sizes, optional ranks,
  and optional weights. It does not receive edge labels, label dimensions,
  ports, or routed splines, so `flat.c::flat_node`, `flat_limits`, and
  adjacent-label `ED_dist` updates cannot be applied end-to-end here.
- This component stores metadata on the prepared config as
  `_dagua_graphviz_dot_flat_metadata` for the integration codex to consume
  when edge-label/routing interfaces are available.
- The duplicate-edge representative policy is directed `(tail, head)` based.
  Reverse edges are treated as separate representatives.

## Live compare

- Not run. This is an internal sub-component; there is no current end-to-end
  route/label surface for flat/self/multi-edge splines in `dagua_native`.
