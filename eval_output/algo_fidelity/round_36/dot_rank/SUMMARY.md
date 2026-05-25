# R36 dot_rank Summary

## Source Files Read

- `/home/jtaylor/projects/_references/graphviz/lib/dotgen/rank.c` (1107 lines)
- `/home/jtaylor/projects/_references/graphviz/lib/dotgen/dotinit.c` (535 lines)
- `/home/jtaylor/projects/_references/graphviz/lib/common/ns.c` (1414 lines)
- Requested `/home/jtaylor/projects/_references/graphviz/lib/dotgen/dot.c` was not present in this reference checkout; `dotinit.c` is the available dot orchestrator equivalent.

## Implementation Summary

- Added `dagua/layout/ops/pipelines/dot_rank.py`.
- Implemented `graphviz_rank_assignment(edges, virtual_node_factory, ...)`:
  - normalizes tensor or tuple edge inputs;
  - supports `minlen` and integer Graphviz-style edge weights;
  - runs a Python port of Graphviz's network-simplex rank assignment (`init_rank`, feasible tight tree construction, cut values, leave/enter edge pivots, and top-bottom balance);
  - returns `{node: rank}` plus `GraphvizVirtualEdge` metadata for long-edge dummy-node chains.
- Wired `fidelity_mode in {"dot", "graphviz_dot", "graphviz"}` in `dagua/layout/ops/pipelines/sugiyama.py` to use the Graphviz ranker for layer assignment.
- Stored Graphviz virtual-edge metadata in Sugiyama state extras for downstream integration; the existing dummy expansion op still materializes the actual dummy nodes used by this pipeline.

## Tests Added

- Added `tests/test_layout/test_dot_rank.py`.
- Golden vectors were captured from Graphviz 7.0.5 using `dot -Tdot` with `graph [phase=1]`.
- Covered:
  - diamond DAG ranks;
  - weighted skip graph ranks;
  - `minlen=3` long-edge virtual nodes;
  - disconnected tensor input with `edge_minlens`;
  - `layout_sugiyama_pipeline(..., fidelity_mode="graphviz")` dispatch.

## Verification

- `ruff check dagua/layout/ops/pipelines/dot_rank.py dagua/layout/ops/sugiyama.py dagua/layout/ops/pipelines/sugiyama.py tests/test_layout/test_dot_rank.py --fix` passed.
- `ruff check . --fix` passed after concurrent sprint files were updated in the shared worktree.
- `mypy --follow-imports=silent dagua/cli.py` passed.
- `pytest tests/test_layout/test_dot_rank.py -q` passed: `5 passed, 2 warnings`.
- `pytest tests/test_graph.py tests/test_layout/test_dot_rank.py tests/test_layout/test_sugiyama_fidelity.py -x --tb=short -q` passed: `48 passed, 2 warnings`.
- `pytest tests/test_pipeline_sugiyama.py tests/test_layout/test_dot_rank.py -x --tb=short -q` failed in existing pipeline parity tests: classic output uses rank spacing `1.0` while direct pipeline output uses dot spacing `72.0` for default calls.
- `pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"` failed during collection on `tests/test_classic_drl.py`: `ImportError: cannot import name 'layout_drl' from 'dagua.layout.classic'`.
- `pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q` exited with code `-1` before producing a failure report in this environment.

## Blockers / Interface Assumptions

- `graphviz_rank_assignment` assumes cyclic input has already passed through Graphviz-style acyclic preprocessing. This matches `rank.c`, where `acyclic(g)` runs before `rank1(g)`.
- Cluster ranksets, `newrank`, and label-rank doubling are not represented in the standalone edge-list interface. The implementation focuses on the flat rank constraints and long-edge virtual-node metadata requested for this sub-component.
- Live compare was not run because this sub-component is not registered as a standalone eval competitor; integration codex can wire the new `graphviz` fidelity mode into a named variant for end-to-end comparison.
