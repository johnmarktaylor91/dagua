# Round 36 dot_position

## Source files read

- `/home/jtaylor/projects/_references/graphviz/lib/dotgen/position.c` (read end-to-end, 1133 lines)
- `/home/jtaylor/projects/_references/graphviz/lib/common/ns.c` (targeted read of `rank`, `rank2`, and `LR_balance` to understand `rank(g, 2, ...)` invoked by `position.c`)

## Implementation summary

- Added a fidelity-gated Graphviz dot x-position component in `dagua/layout/ops/pipelines/dagua_native.py`.
- New helper `_graphviz_dot_x_position_network_simplex(...)` builds the `position.c` auxiliary x-ranking problem:
  - same-rank left-to-right constraints from rounded half-widths plus `nodesep`
  - one slack variable per non-self edge matching `make_edge_pairs`
  - weighted objective equivalent to Graphviz's network-simplex rank objective
  - centered tensor output for Dagua/metric comparison
- Added narrow end-to-end selector `fidelity_mode="graphviz_dot_position"` / `"dot_position"` that tries a simple DAG wrapper:
  - LP rank assignment
  - virtual nodes for long edges
  - median ordering sweeps
  - x-position component
- Default behavior is unchanged unless the narrow position fidelity selector is used.

## Tests added

- `tests/test_pipeline_dagua_native_dot_position.py`
  - complete two-rank golden vector captured from `dot -Tplain`
  - unequal-width same-rank golden vector captured from `dot -Tplain`
  - invalid rank-order validation
  - end-to-end narrow fidelity selector smoke test

## Verification

- `ruff check dagua/layout/ops/pipelines/dagua_native.py tests/test_pipeline_dagua_native_dot_position.py --fix`
  - passed
- `pytest tests/test_pipeline_dagua_native_dot_position.py -x --tb=short -q`
  - `4 passed, 2 warnings`
- `pytest tests/test_layout/test_native_topology_dispatch.py tests/test_pipeline_dagua_native_dot_position.py -x --tb=short -q`
  - `10 passed, 2 warnings`
- `mypy --follow-imports=silent dagua/cli.py`
  - passed
- `pytest tests/test_graph.py -x --tb=short -q`
  - `37 passed, 2 warnings`

## Blockers / interface assumptions

- `live_compare` was not run for this sub-component. The live comparator cannot currently request the new `fidelity_mode="graphviz_dot_position"` variant, and the broader dot integration is being split across sibling Round 36 tasks.
- Full `ruff check . --fix` is blocked by unrelated concurrent Round 36 files (`dot_rank.py`, `quadtree.py`, `sfdp.py`, `fmmm.py`, `sugiyama.py`) with pre-existing lint/type errors in this shared worktree.
- `pytest tests/test_layout/ -x --tb=short -q` was attempted separately after the combined targeted gate exited `-1`; it reached 18% with passing dots but consumed ~56 minutes of CPU without terminal output and was terminated to avoid an unbounded run.
- I did not stage or commit because the worktree contains concurrent sibling edits, including unrelated hunks in `dagua/layout/ops/pipelines/dagua_native.py`; staging the whole file would capture other sub-task work.

## Integration notes

- The component assumes the integration codex will provide final Graphviz-compatible rank assignment, mincross ordering, flat-edge metadata, labels, and clusters. The wrapper included here is intentionally narrow and exists only to make the x-position component invokable in tests.
- The x-position helper accepts rank ordering directly, which is the expected integration boundary for the dot mincross/rank sibling components.
