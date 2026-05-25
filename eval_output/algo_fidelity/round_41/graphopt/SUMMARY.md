# Round 41 GraphOpt Summary

## Reference Source Lines

- `/home/jtaylor/projects/_references/igraph/src/layout/graphopt.c:29` defines `COULOMBS_CONSTANT = 8987500000.0`.
- Lines 77-82 compute Euclidean distance.
- Lines 85-156 decompose and accumulate Coulomb forces.
- Lines 195-238 compute and accumulate Hooke spring forces, with exact `distance == 0.0` skip at lines 216-218.
- Lines 240-285 apply per-axis `force / node_mass` movement clamped by `max_sa_movement`.
- Lines 363-371 select supplied seed matrix or random fallback.
- Lines 374-428 run fixed iterations, node-pair order, edge order, and movement.

## Diagnosis

The requested same-seed smoke comparison does not reproduce the old 0.01-0.05
GraphOpt residual. With the igraph adapter seed matrix passed into both sides,
Dagua is already Procrustes-bit-exact on path, star, clustered, grid, and the
prior round-31 benchmark subset.

The apparent remaining residual in live comparisons is measurement-related:
`algo_fidelity_live_compare.py` computes all seed-pair combinations. Different
GraphOpt seeds are intentionally different layouts, so all-pair summaries retain
non-zero medians even when same-seed pairs match. A round-41 precheck on the old
five-graph subset reported median `0.012208` and worst `0.208465`, but direct
same-seed checks for the same graphs were `~1e-8`.

Per-topology component diagnosis from `smoke_rmsd.csv`:

| Topology | Mean init RMSD | Mean 1-iter RMSD | Mean 500-iter RMSD | Dominant residual |
|---|---:|---:|---:|---|
| path | 7.132e-09 | 7.953e-09 | 6.794e-09 | output frame only |
| star | 7.257e-09 | 7.813e-09 | 7.626e-09 | output frame only |
| clustered | 6.777e-09 | 7.370e-09 | 7.036e-09 | output frame only |
| grid | 7.257e-09 | 7.813e-09 | 6.903e-09 | output frame only |

Raw coordinates differ because python-igraph's returned `Layout` can be in a
different rotation/reflection/translation frame. The fidelity metric is
Procrustes-aligned RMSD, so this is not an algorithmic residual.

## Port Implementation Summary

- Added `eval_output/algo_fidelity/round_41/graphopt/smoke_harness.py`.
- The harness compares four required topologies at seeds 42, 43, and 44 against
  python-igraph 1.0.0 using the same explicit seed matrix as the reference
  adapter.
- Updated `dagua/layout/ops/pipelines/graphopt.py` fidelity notes to replace the
  stale round-33 residual with the round-41 same-seed finding.
- No force-kernel production code was changed: the existing R31/R33 ports already
  cover the dominant igraph semantics for same-seed Procrustes fidelity.

## Before/After Smoke RMSD

The old live all-seed-pair check is included here only as a pre-diagnosis
baseline because it is the source of the apparent residual. The round-41 smoke
harness is the requested same-seed topology-by-seed check.

| Check | Graphs/topologies | Seeds | Mean/median RMSD | Max/worst RMSD |
|---|---:|---:|---:|---:|
| R31 live compare | 5 benchmark graphs | 30 | median 0.043382 | 0.091870 |
| R41 live precheck, all seed pairs | 5 benchmark graphs | 3 | median 0.012208 | 0.208465 |
| R41 same-seed smoke | 4 topologies | 3 | mean 7.090e-09 | 9.951e-09 |

Round-41 same-seed smoke output:

```text
overall_mean_full_rmsd: 7.089731185828302e-09
overall_max_full_rmsd: 9.950840347330515e-09
threshold: 0.005
verdict: pass
```

## Final Verdict

Bit-exact for the requested Procrustes RMSD target. The measured smoke mean is
`7.09e-09`, well below `<0.001` and `<0.005`.

## Concerns

- Raw coordinate-frame RMSD remains large by design; Procrustes removes that
  frame difference.
- All-seed-pair live summaries are useful for distributional equivalence, but
  they should not be interpreted as same-seed implementation residuals.
