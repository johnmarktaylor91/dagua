# Round 63 LGL Real Port

## Source Lines Ported

- `/home/jtaylor/projects/_references/igraph/src/layout/large_graph.c:156-199`: random/root selection, BFS layer setup, random initialization, bounded grid creation, root insertion, harmonic shell constant.
- `/home/jtaylor/projects/_references/igraph/src/layout/large_graph.c:201-286`: per-layer sphere placement and incident-edge activation.
- `/home/jtaylor/projects/_references/igraph/src/layout/large_graph.c:292-374`: cooling loop, attractive forces, grid-neighbor repulsion, positive-component `maxchange`, and grid move updates.
- `/home/jtaylor/projects/_references/igraph/src/core/grid.c:27-45`: exact bounded grid cell boundary semantics.
- `/home/jtaylor/projects/_references/igraph/src/core/grid.c:94-181`: add/move linked-cell behavior and mutable mass counters.
- `/home/jtaylor/projects/_references/igraph/src/core/grid.c:201-275`: grid iteration and neighbor-pair order.

## Diff Guard

- Confirmed no delegation was added.
- `git diff dagua/layout/ops/pipelines/lgl.py | grep -E "^\\+.*(import igraph|from igraph)" || true`: empty.
- `rg -n "import igraph|from igraph|subprocess|graph\\.layout\\(\"lgl\"|graph\\.layout\\('lgl'" dagua/layout/ops/pipelines/lgl.py dagua/layout/ops/lgl.py`: empty.

## Smoke RMSD

- Before: current task baseline reported approximately `0.17` RMSD against `IgraphLGL`.
- After: max Procrustes RMSD `1.24374864185e-07` across `path3`, `path4`, `path5`, `star8`, `tree7`, and `cycle6_iter10`.

## Final Verdict

Real native port complete for the fidelity path. No `igraph`, `subprocess`, or `graph.layout("lgl")` delegation exists in the scoped LGL implementation files.
