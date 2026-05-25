# Round 36 fdp_ports Summary

## Source files read

- `/home/jtaylor/projects/_references/graphviz/lib/fdpgen/clusteredges.c`
- `/home/jtaylor/projects/_references/graphviz/lib/fdpgen/clusteredges.h`
- `/home/jtaylor/projects/_references/graphviz/lib/neatogen/neatosplines.c` for `getPath`, `makeSpline`, and `makeObstacle` call semantics
- `dagua/layout/ops/pipelines/fmmm.py`
- `dagua/layout/ops/cluster_geometry.py`
- `dagua/layout/ops/cluster_driver.py`
- `dagua/layout/ops/state.py`

## Implementation summary

- Added fidelity-only fdp compound-edge attachment metadata in `dagua/layout/ops/pipelines/fmmm.py`.
- Ported Graphviz `makeClustObs` expansion math for additive and multiplicative `expand_t`.
- Ported Graphviz `objectList` / `raiseLevel` graph-level obstacle selection for clustered edges.
- Added cluster-boundary attachment point clipping for edges crossing out of or into nested clusters.
- Stored metadata in `SolveState.extras` under:
  - `fmmm_fdp_compound_edge_attachments`
  - `fmmm_fdp_compound_cluster_obstacles`
  - `fmmm_fdp_compound_node_obstacles`
- Preserved default behavior: the new op is appended only when `build_fmmm_pipeline(..., fidelity_mode=True)` is used.

## Tests added

- `tests/test_fmmm_fdp_ports.py`
  - Golden additive `makeClustObs` vector from `clusteredges.c`.
  - Golden multiplicative `makeClustObs` vector from `clusteredges.c`.
  - Nested-cluster `objectList`/`raiseLevel` obstacle walk golden.
  - Sibling-cluster boundary attachment point clipping.
  - Fidelity-only pipeline-op wiring and extras storage.

## Blockers / interface assumptions

- Dagua FMMM currently returns only node coordinates. There is no public edge-route or spline return channel, so this sub-task records attachment/routing seed metadata in `SolveState.extras` for the integration codex to consume.
- Graphviz `Pobsopen`, `Pobspath`, and `Proutespline` visibility-path spline generation are not exposed in this Dagua pipeline component; this port supplies the endpoint and obstacle inputs that those routines consume.
- The public FMMM API does not expose Graphviz `esep` / `expand_t`, so the fidelity op currently records zero-margin obstacles by default. The lower-level helpers support additive and multiplicative expansion for future integration.
- Live compare was not run because this component is not yet invokable end-to-end through rendered edge routes.

## Verification

- `ruff check dagua/layout/ops/pipelines/fmmm.py tests/test_fmmm_fdp_ports.py --fix` passed.
- `mypy --follow-imports=silent dagua/cli.py` passed.
- `pytest tests/test_pipeline_fmmm.py tests/test_fmmm_fdp_ports.py -x --tb=short -q` passed: 25 passed, 2 warnings.
- `pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q` was terminated by the execution harness after a long run without a failure report.
- `pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"` failed during collection before reaching this work: `ImportError: cannot import name 'layout_drl' from 'dagua.layout.classic'`.
- Repo-wide `ruff check . --fix` is currently blocked by unrelated modified files outside this sub-task, including `dagua/layout/ops/pipelines/dagua_native.py`, `dagua/layout/ops/pipelines/stress_majorization.py`, and `dagua/layout/ops/sugiyama.py`.
