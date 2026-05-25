# Round 36 dot_mincross Summary

## Source files read

- `/home/jtaylor/projects/_references/graphviz/lib/dotgen/mincross.c` read end-to-end.
- Relevant Dagua integration files inspected:
  - `dagua/layout/ops/sugiyama.py`
  - `dagua/layout/ops/pipelines/sugiyama.py`
  - `dagua/layout/ops/pipelines/dagua_native.py`
  - `dagua/layout/ops/ordering.py`

## Implementation summary

- Added `dagua/layout/ops/_dot_mincross.py` with
  `graphviz_mincross(ranks, edges, iterations=24) -> list[list[int]]`.
- Ported the Graphviz dot mincross pass-2 core for already-ranked,
  adjacent-rank graphs:
  - `MC_SCALE = 256` median values.
  - Down/up alternating median passes.
  - Graphviz reverse tie rule (`pass % 4 < 2`).
  - Best crossing-count retention with `Convergence = 0.995` and `MinQuit = 8`.
  - Final non-reverse adjacent transposition.
- Integrated the helper behind `fidelity_mode="dot"` and
  `fidelity_mode="graphviz_dot"` in the Sugiyama pipeline. Default and
  `fidelity_mode="igraph"` paths are unchanged.

## Tests added

- Added `tests/test_layout/test_dot_mincross.py`.
- Golden order vectors were captured from local Graphviz `dot -Tplain`
  (`graphviz version 7.0.5`) for simple ranked cases where dot's rank-build
  phases do not further reorder the fixed source rank.
- Full dot golden capture for arbitrary multi-rank subcomponent inputs is not
  isolated by the CLI because `dot_mincross()` is not externally invokable and
  full `dot` also runs rank construction, flat-edge handling, clusters, and
  coordinate assignment around mincross.

## Blockers / interface assumptions

- The new helper assumes inputs are already in dot-mincross form: ranks are
  known and long edges have been expanded into adjacent-rank virtual-node
  chains. Non-adjacent edges are ignored, matching the interface contract for
  the integration codex to supply dot-rank/dummy-node output.
- Port and flat-edge ordering data are not present in the current Python
  subcomponent interface. The implementation uses Graphviz's default
  no-port-order behavior and documents this as an integration assumption.
- `live_compare` was not run for this subcomponent because the registered
  fidelity competitor variants do not currently expose `fidelity_mode="dot"`
  as a separate runnable engine; the public Sugiyama pipeline smoke test
  exercises the end-to-end hook directly.
