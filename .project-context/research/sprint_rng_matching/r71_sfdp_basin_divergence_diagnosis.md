# r71 SFDP Basin-Divergence Diagnosis

Date: 2026-06-13
Verdict: B, residual FP-stack basin sensitivity.

## Scope

This diagnosis covers native `classic_sfdp` / `layout_sfdp_pipeline(...,
fidelity_mode="graphviz")` against the seeded `graphviz_sfdp` reference rows in:

- `eval_output/benchmark_100seed_seeded_refs/`
- `eval_output/benchmark_100seed_escalation_final/`
- `eval_output/fidelity_definitive/r71_p1e_seeded_analysis.jsonl`

All Graphviz probes were run with `OMP_NUM_THREADS=1`.

## Initial Placement

Graphviz 7.0.5 does not expose a true iteration-0 SFDP layout through the public
CLI on this install. Passing `-Gmaxiter=0`, `-Gmaxiter=1`, `-Gmaxiter=200`, or
`-Gmaxiter=2000` still prints `maxiter 500` in `sfdp -v`, and the Graphviz 7.0.5
SFDP tune path does not read a `maxiter` graph attribute for SFDP. A direct
zero-iteration position comparison is therefore not externally available.

The seed and RNG path do match:

- `sfdpinit.c::tuneControl` parses `start` through `setSeed(g, INIT_RANDOM, &seed)`.
- `setSeed` only parses/stores the seed; it does not call `srand` before
  multilevel coarsening.
- `spring_electrical.c::spring_electrical_embedding` calls
  `srand(ctrl->random_seed)` immediately before filling random coordinates with
  `drand()`.
- Native `GraphvizRandom` was checked against host libc `srand/rand`; seeds
  1, 42, 100, and 123 matched exactly for the first 10 draws.

Conclusion: no evidence of an initial RNG mismatch. The only missing piece is an
unavailable public Graphviz iteration-0 coordinate dump.

## Coarsening

Graphviz 7.0.5 source uses:

- `Multilevel_control_new`: `minsize=4`, `min_coarsen_factor=0.75`,
  `randomize=TRUE`, `COARSEN_INDEPENDENT_EDGE_SET_HEAVEST_EDGE_PERNODE_SUPERNODES_FIRST`,
  `COARSEN_MODE_FORCEFUL`.
- `maximal_independent_edge_set_heavest_edge_pernode_supernodes_first`: decompose
  supervariables, chunk at `MAX_CLUSTER_SIZE`, then random-permutation heavy-edge
  pair unmatched vertices, then singleton leftovers.
- `Multilevel_coarsen_internal`: builds `P`, `R`, computes `R*A*P`, removes the
  diagonal, and marks the coarse matrix symmetric.
- Because `setSeed` does not call `srand` before coarsening, the coarsening
  permutation starts from libc's default rand state, equivalent to seed 1 on this
  glibc stack.

Native fidelity mode mirrors these choices:

- `BuildGraphvizSFDPMatrixHierarchy` uses `GraphvizRandom(seed=1)` for coarsening.
- `_graphviz_sfdp_cluster_nodes` implements supervariable grouping, max cluster
  size 4, random-permutation heavy-edge pairing, and singleton leftovers.
- `_build_graphviz_matrix_coarse_graph` implements matrix aggregation semantics
  with diagonal removal.
- After coarsening, native resets layout RNG with `GraphvizRandom(seed=problem.seed)`,
  matching Graphviz's later `srand(ctrl->random_seed)`.

Native hierarchy examples from the diagnostic run:

| Graph | Native fidelity hierarchy sizes | Edge counts |
|---|---:|---:|
| `recurrent_feedback_cell` | `[5]` | `[5]` |
| `petersen_10` | `[10, 6]` | `[15, 11]` |
| `parallel_cycles_4x5` | `[20, 12, 8, 4]` | `[20, 12, 4, 0]` |
| `random_dag_50` | `[50, 28, 17, 11, 7, 4]` | `[75, 45, 28, 13, 6, 2]` |
| `heavy_tail_weights_50` | `[50, 28, 16, 10, 7, 5]` | `[74, 50, 35, 18, 10, 5]` |

Conclusion: no code discrepancy found in coarsening structure or RNG staging.

## Force Kernel And Trajectory

Graphviz 7.0.5 source for the non-fast path computes, per vertex and in-place:

- attractive force over CSR neighbors as `-CRK * delta * dist`
- exact repulsion as `KP * delta / pow(dist, 1 - p)`
- per-node force normalization, immediate coordinate writeback
- adaptive cooling from force norm

Native `_SFDPGraphvizSequentialStep` implements the same sequential update order,
same attraction and repulsion exponents, same force normalization, and the same
adaptive-cooling split between coarsest and finer levels. For graphs at or above
the Barnes-Hut threshold, native uses the Graphviz quadtree port; this remains
the expected FP-sensitive boundary because Graphviz depends on libm `sqrt`,
`pow`, and quadtree aggregation order.

Because Graphviz does not expose per-iteration state or honor tiny `maxiter`
values for SFDP in this build, a direct reference energy trace could not be
extracted without rebuilding/instrumenting Graphviz. The native force path has
source-level parity with the reference kernel; remaining divergence is therefore
the documented FP-stack basin sensitivity rather than an actionable port bug.

## Stored Benchmark Evidence

From `r71_p1e_seeded_analysis.jsonl`:

- SFDP rows: 607.
- Non-equivalent, non-insufficient SFDP rows: 277.
- Broad basin-signature rows with `0.75 <= disp <= 1.25`: 91.
- Quality/stress-equivalent among those broad rows: 42.
- Strong signature rows with `0.8 <= disp <= 1.2` and cross/self ratio >= 10: 27.

Examples:

| Graph / engine | disp | E_cross median | E_self q95 | cross/self | W_D | W_R | quality |
|---|---:|---:|---:|---:|---:|---:|---|
| `weighted_karate_34` / `classic_sfdp_default` | 1.179 | 0.207 | 0.007 | 27.8 | 0.305 | 0.259 | no |
| `dense_pair_50` / `classic_sfdp_theta04` | 1.191 | 0.080 | 0.003 | 28.7 | 0.096 | 0.081 | yes |
| `hub_spoke_10x20` / `classic_sfdp_default` | 1.038 | 0.177 | 0.017 | 10.4 | 0.659 | 0.641 | yes |
| `parallel_cycles_4x5` / `classic_sfdp_theta04` | 1.027 | 0.446 | 0.024 | 18.4 | 0.659 | 0.641 | no |
| `chung_lu_150` / `classic_sfdp_default` | 0.980 | 0.053 e_rel | n/a | 4.4 | n/a | n/a | yes |

The signature is not over- or under-convergence: `disp` stays near 1.0, so native
and reference have comparable seed variance. The mismatch is the cross-basin
distance: native and reference ensembles are internally stable but land in
different attractors.

## Residual Label

Use this residual label for remaining SFDP failures:

- `SFDP_FP_BASIN_RESIDUAL_STRESS_EQUIV` for quality/stress-equivalent cases.
- `SFDP_FP_BASIN_RESIDUAL_NON_EQUIV` for the rest.

Do not patch native SFDP for this evidence set unless a future instrumented
Graphviz build exposes a concrete iteration-0, coarsening-level, or per-iteration
force discrepancy.

## External References

- Graphviz 7.0.5 `spring_electrical.c` source:
  `https://gitlab.com/graphviz/graphviz/-/raw/7.0.5/lib/sfdpgen/spring_electrical.c`
- Graphviz 7.0.5 `sfdpinit.c` source:
  `https://gitlab.com/graphviz/graphviz/-/raw/7.0.5/lib/sfdpgen/sfdpinit.c`
- Graphviz 7.0.5 source archive:
  `https://gitlab.com/graphviz/graphviz/-/archive/7.0.5/graphviz-7.0.5.tar.gz`
- Graphviz `startType` documentation:
  `https://graphviz.org/docs/attr-types/startType/`
