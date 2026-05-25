# Round 39 fdp_kernels Summary

## Kernels ported

- `tLayout`: added Graphviz FDP defaults, POSIX `drand48` seeding, random
  rectangle initialization, grid-limited electrical repulsion, edge attraction,
  linear cooling, and temperature-limited position updates in
  `dagua/layout/ops/pipelines/fmmm.py:835`, `dagua/layout/ops/pipelines/fmmm.py:851`,
  and `dagua/layout/ops/pipelines/fmmm.py:1822`.
- `xLayout`: added Graphviz's overlap-counted relaxation loop, overlap and
  non-overlap repulsion constants, edge attraction around node radii, nine
  default tries, and Graphviz default node-size floors in
  `dagua/layout/ops/pipelines/fmmm.py:2130` and
  `dagua/layout/ops/pipelines/fmmm.py:2161`.
- `packGraphs`: reused the R36 `pack.c` bbox polyomino port for both recursive
  and flat weak components; relevant call sites are
  `dagua/layout/ops/pipelines/fmmm.py:1500` and
  `dagua/layout/ops/pipelines/fmmm.py:2528`.

## Smoke RMSD

R38 baseline was reported as `0.21-0.24` on the dropped fdp fidelity variant.
Round 39 smoke used `eval_output/algo_fidelity/round_39/fdp_kernels/smoke_check.py`
with path, two-cluster, and three-cluster topologies across seeds `1, 2, 3`.

| Topology | Seed 1 | Seed 2 | Seed 3 | Mean | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| path | 0.076886609 | 0.023582772 | 0.019508183 | 0.039992521 | 0.076886609 |
| clustered | 0.386046549 | 0.350732185 | 0.347926134 | 0.361568289 | 0.386046549 |
| multi_cluster | 0.300758871 | 0.324869584 | 0.304402497 | 0.310010317 | 0.324869584 |

Additional isolation: flat `tLayout` against Graphviz `fdp -Goverlap=0:`
matched at RMSD `0.000009618` for the path graph, so the flat numerical
kernel is no longer the dominant residual.

## Verdict

Hold. The flat FDP component kernel is below the `<0.10` smoke target, but
clustered recursion remains above target at `0.30-0.39`. The
`classic_fmmm_graphviz_fdp_fidelity` registry variant remains dropped to avoid
shipping a misleading Graphviz-fidelity claim.

## Residual root cause

The remaining error is in clustered fdp recursion semantics: Graphviz's
`expandCluster`/derived-node sizing/final cluster bbox interaction still
produces different intra-cluster geometry even when the component kernel is
Graphviz-like. `packGraphs` is active, but packing cannot fix the internal
cluster layouts being different before packing.
