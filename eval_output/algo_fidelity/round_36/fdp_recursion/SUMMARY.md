# Round 36 fdp_recursion Summary

## Source files read

- `/home/jtaylor/projects/_references/graphviz/lib/fdpgen/comp.c`
- `/home/jtaylor/projects/_references/graphviz/lib/fdpgen/layout.c`
- `dagua/layout/ops/pipelines/fmmm.py`
- `dagua/layout/ops/cluster_geometry.py`
- `dagua/layout/ops/cluster_driver.py`
- `dagua/layout/ops/state.py`

## Implementation summary

- Added a guarded Graphviz fdp recursion entrypoint:
  `graphviz_fdp_fidelity(...)`.
- Added derived-graph construction for one recursion level:
  child clusters are collapsed first, direct leaves are added next, and
  parent-generated ports are transformed into port nodes.
- Added `findCComp`-style generalized connected components:
  all port-bearing components merge into the first component before remaining
  components are emitted in node order.
- Added `expandCluster`-style port generation:
  incident derived edges are sorted by angle then distance, equal angles are
  separated by Graphviz's two-degree maximum delta, and multi-edge ports keep
  the real-edge ordering/reversal rule.
- Wired `layout_fmmm_pipeline(..., fidelity_mode=True, clusters=...)` to use
  the fdp recursion path. Default behavior remains unchanged when
  `fidelity_mode` is false or no clusters are provided.

## Tests added

- `tests/test_layout/test_fmmm_fdp_recursion.py`
  - derived graph cluster collapse and real-edge grouping
  - port component merge ordering
  - cluster expansion port order and angles
  - default behavior guard
  - public recursion entrypoint finite-output smoke test

## Blockers / interface assumptions

- Golden vectors were captured at the component-contract level from the C
  control flow rather than by executing Graphviz with introspection hooks;
  the public Graphviz CLI does not expose derived graphs or generated ports.
- Graphviz `tLayout`, `xLayout`, and `packGraphs` numerical kernels are still
  represented by Dagua's existing FM^3 solver plus deterministic component
  packing. Integration codex should replace those once the sibling tilepack /
  overlap slices land.
- The public fmmm tensor API has no pin or pre-existing port inputs, so this
  slice ports parent-derived cluster boundary ports only.
- The worktree contains sibling sprint edits in the same fmmm pipeline file;
  commit staging must avoid accidentally capturing those unrelated changes.
