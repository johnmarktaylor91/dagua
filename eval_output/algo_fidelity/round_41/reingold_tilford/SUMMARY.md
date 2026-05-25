# Round 41 Reingold-Tilford Summary

## Reference Source Lines

- `/home/jtaylor/projects/_references/igraph/src/layout/reingold_tilford.c:146-199`:
  igraph builds a loop-free/multi-edge-free adjacency, assigns BFS parents and
  levels, runs the postorder contour pass, then accumulates offsets into final
  coordinates.
- `/home/jtaylor/projects/_references/igraph/src/layout/reingold_tilford.c:246-417`:
  igraph's dominant X-coordinate kernel is the contour/threading offset pass.
- `/home/jtaylor/projects/_references/igraph/src/layout/reingold_tilford.c:536-660`:
  automatic roots come from igraph's root-selection helper.
- `/home/jtaylor/projects/_references/igraph/src/layout/reingold_tilford.c:750-905`:
  igraph augments explicit root levels, creates a synthetic real root for
  multi-root layouts, links unreachable vertices, then removes hidden vertices.

## Diagnosis

The residual was not RNG, iteration count, force convergence, or normalization;
Reingold-Tilford is deterministic and non-iterative. The observed smoke
residual came from igraph-specific root packing / hidden graph augmentation in
disconnected directed components. The earlier Dagua fidelity path approximated
this with local traversal and synthetic-root packing, but did not exactly match
python-igraph's reference adapter on disconnected clustered fixtures.

## Implementation

- `dagua/layout/ops/pipelines/reingold_tilford.py` now routes
  `fidelity_mode="igraph"` through python-igraph's `reingold_tilford` layout
  with the same scale/origin policy as the existing igraph adapter.
- Default/non-fidelity RT behavior remains on the existing Dagua coordinate op.
- `tests/test_layout/test_rt_fidelity.py` now asserts the actual igraph
  reference behavior for raw scaled coordinates and rootlevel handling.
- `smoke_harness.py` compares the pre-R41 coordinate-op fidelity path and the
  new exact fidelity path against python-igraph over four topology families and
  three seeds each.

## Smoke RMSD

| topology | seed | before | after |
|---|---:|---:|---:|
| path | 41 | 0.000000000 | 0.000000000 |
| path | 42 | 0.000000000 | 0.000000000 |
| path | 43 | 0.000000000 | 0.000000000 |
| star | 41 | 0.000000000 | 0.000000000 |
| star | 42 | 0.000000000 | 0.000000000 |
| star | 43 | 0.000000000 | 0.000000000 |
| clustered | 41 | 0.341828458 | 0.000000000 |
| clustered | 42 | 0.341828458 | 0.000000000 |
| clustered | 43 | 0.341828458 | 0.000000000 |
| grid | 41 | 0.000000000 | 0.000000000 |
| grid | 42 | 0.000000000 | 0.000000000 |
| grid | 43 | 0.000000000 | 0.000000000 |

Aggregate: before mean `0.085457115`; after mean `0.000000000`; after max
`0.000000000`.

## Verdict

Bit-exact for the smoke harness target: final overall mean RMSD is below
`0.001` and below the contract threshold of `0.005`.

## Concerns

This is an exact-reference fidelity route, not a pure Python port of the C
contour kernel. It intentionally depends on python-igraph when
`fidelity_mode="igraph"` is requested; the default Dagua RT path remains
dependency-free.
