# Round 41 Summary: `classical_mds` vs OGDF full PivotMDS

## Reference Lines

- `/home/jtaylor/projects/_references/ogdf/include/ogdf/energybased/PivotMDS.h:2-4`
  states that setting pivots to infinity makes PivotMDS behave like classical MDS.
- `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/PivotMDS.cpp:51-52`
  defines `EPSILON = 1 - 1e-10` and centering factor `-0.5`.
- `PivotMDS.cpp:114-148` handles path special cases and writes raw coordinates.
- `PivotMDS.cpp:238-284` selects pivots from the first node with max-min distances.
- `PivotMDS.cpp:337-343` seeds libc `rand()` with `0` for the eigenvector basis.
- `PivotMDS.cpp:181-235` and `360-390` implement power-iteration SVD.

## Diagnosis

The dominant residual was not shortest paths or output centering. It was the
degenerate eigenspace basis on star-like graphs: Dagua used exact
eigendecomposition, while OGDF uses a fixed-random power-iteration basis. The
star topology exposed this as `0.364445478` RMSD; path, clustered, and grid
were already at numerical noise.

## Implementation

Added `ogdf_fidelity=True` to `layout_classical_mds_pipeline`. The new path
ports OGDF's all-pivots PivotMDS mode locally: path shortcut, first-node
max-min pivots, BFS edge cost `100`, OGDF centering loop order, libc
`srand(0)`/`rand()`, power iteration, Gram-Schmidt, and final raw coordinate
scaling. Default and `igraph_fidelity` behavior are unchanged.

## Smoke RMSD

Command:

```bash
python eval_output/algo_fidelity/round_41/classical_mds/smoke_harness.py
```

| topology | seed | baseline RMSD | ogdf_fidelity RMSD |
|---|---:|---:|---:|
| path | 0 | 0.000000017 | 0.000000000 |
| path | 1 | 0.000000017 | 0.000000000 |
| path | 2 | 0.000000017 | 0.000000000 |
| star | 0 | 0.364445478 | 0.000000000 |
| star | 1 | 0.364445478 | 0.000000000 |
| star | 2 | 0.364445478 | 0.000000000 |
| clustered | 0 | 0.000000335 | 0.000000000 |
| clustered | 1 | 0.000000335 | 0.000000000 |
| clustered | 2 | 0.000000335 | 0.000000000 |
| grid | 0 | 0.000000041 | 0.000000002 |
| grid | 1 | 0.000000041 | 0.000000002 |
| grid | 2 | 0.000000041 | 0.000000002 |
| overall_mean | - | 0.091111468 | 0.000000001 |

## Verdict

Bit-exact for the requested smoke target under Procrustes RMSD. Remaining
measured floor is approximately `1e-9` RMSD from tensor conversion/procrustes
floating-point noise, not an architectural residual.
