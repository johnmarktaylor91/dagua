# Round 62 Reingold-Tilford Fidelity Summary

## Change

Replaced the `fidelity_mode="igraph"` delegation path with a pure Python port of
igraph 1.0.0 `src/layout/reingold_tilford.c`.

The port covers:

- igraph automatic root selection for `out`, `in`, and `all` modes
- igraph-style synthetic roots for forests and unreachable vertices
- `root` and multi-root `rootlevel` handling
- BFS spanning-tree extraction before tidy-tree placement
- Reingold-Tilford contour threading and final coordinate propagation
- adapter-compatible `50.0` output scaling and optional horizontal axis swap

## Verification

Smoke comparison against python-igraph 1.0.0:

- 2,880 randomized directed graphs, `N=1..12`, modes `out`, `in`, `all`
- explicit multi-root/rootlevel cases across `out`, `in`, and `all`
- maximum absolute coordinate difference: `0.0`

Forbidden-pattern check:

- no `import igraph`
- no `from igraph`
- no `graph.layout("reingold_tilford", ...)`

## Notes

The helper module is internal (`dagua.layout.ops._reingold_tilford`) so it is
not treated as an op registry module. The public pipeline API is unchanged.
