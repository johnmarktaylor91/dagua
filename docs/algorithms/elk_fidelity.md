# ELK Layered fidelity verification

Reference: elkjs through `dagua.eval.competitors.elk_competitor.ElkLayered`. One deterministic layout is cached per graph. The production pipeline never invokes Node.

Parameters: `{'elk.algorithm': 'layered', 'elk.direction': 'DOWN', 'elk.spacing.nodeNode': 40, 'elk.layered.spacing.nodeNodeBetweenLayers': 60}`.

Summary: 3/11 bit-exact, 5 close, 3 divergent.

| graph | N | E | layer | order | d_R | anisotropic | max abs diff | class | first divergent phase |
|---|---:|---:|---|---|---:|---:|---:|---|---|
| single_node | 1 | 0 | Y | Y | 0 | 0 | 0 | bit-exact |  |
| small_chain | 6 | 5 | Y | Y | 0 | 0 | 0 | bit-exact |  |
| binary_tree | 11 | 10 | Y | Y | 0.0715855 | 0.0713416 | 48.5734 | close | node placement: Brandes-Koepf balancing/spacing mismatch |
| diamond | 4 | 4 | Y | Y | 1.12814e-17 | 0 | 0 | bit-exact |  |
| grid_5x5 | 25 | 40 | Y | Y | 1.32882e-08 | 1.32007e-08 | 1.52588e-05 | close | node placement: Brandes-Koepf balancing/spacing mismatch |
| org_chart_small | 16 | 15 | Y | N | 1.2151 | 0.96181 | 1054.03 | divergent | crossing minimization: within-layer order mismatch |
| long_skip | 5 | 6 | Y | Y | 0.0781897 | 0.0521634 | 31.1713 | close | node placement: Brandes-Koepf balancing/spacing mismatch |
| disconnected | 5 | 2 | Y | Y | 0.0757369 | 0.0747507 | 40 | close | node placement: Brandes-Koepf balancing/spacing mismatch |
| cycle_4 | 4 | 4 | Y | Y | 0.04762 | 0 | 19.3208 | close | cycle breaking: ELK GREEDY tie semantics not fully ported |
| random_dag_50 | 50 | 90 | Y | N | 1.19948 | 0.923796 | 1757.14 | divergent | crossing minimization: within-layer order mismatch |
| org_chart_deep | 79 | 78 | Y | N | 1.37665 | 0.986329 | 4621.65 | divergent | crossing minimization: within-layer order mismatch |

Named residual: the current native port matches ELK's public coordinate contract and simple layer spacing, but diverges first at ELK's exact layer-sweep/Brandes-Koepf tie semantics on multi-node layers and at ELK GREEDY cycle-breaking ties on cyclic graphs. Edge routing and port extrema are outside this node-position fidelity scope.

## Distributional verification

Reference: elkjs with `elk.randomSeed=1..30`; native ELK uses the same seed set, Java-compatible restart shuffles, and `thoroughness=7`.

| graph | N | E | layers | verdict | variance | procrustes | metric TOST failures |
|---|---:|---:|---|---|---|---|---|
| single_node | 1 | 0 | Y | DISTRIBUTIONAL_EQUIVALENT | match (D 0, R 0) | PASS (between 0 <= band 0.001) |  |
| small_chain | 6 | 5 | Y | DISTRIBUTIONAL_EQUIVALENT | match (D 0, R 0) | PASS (between 0 <= band 0.001) |  |
| binary_tree | 11 | 10 | Y | not_distributional_equivalent | match (D 0.6699, R 0.3984) | FAIL (between 0.738 > band 0.6699) | crossings, stress, edge_length_cv, ordering_inversions |
| diamond | 4 | 4 | Y | DISTRIBUTIONAL_EQUIVALENT | match (D 0.2669, R 0.2609) | PASS (between 0.2627 <= band 0.2669) |  |
| grid_5x5 | 25 | 40 | Y | DISTRIBUTIONAL_EQUIVALENT | match (D 1.952e-16, R 2.872e-16) | PASS (between 1.801e-08 <= band 0.001) |  |
| org_chart_small | 16 | 15 | Y | not_distributional_equivalent | mismatch (D 1.948e-16, R 0.7343) | PASS (between 0.7052 <= band 0.7343) | stress, edge_length_cv, ordering_inversions |
| long_skip | 5 | 6 | Y | not_distributional_equivalent | mismatch (D 3.138e-16, R 0.06737) | FAIL (between 0.09663 > band 0.06737) | crossings, stress |
| disconnected | 5 | 2 | Y | not_distributional_equivalent | mismatch (D 0.1288, R 2.765e-16) | FAIL (between 0.1704 > band 0.1288) | ordering_inversions |
| cycle_4 | 4 | 4 | Y | not_distributional_equivalent | match (D 5.285e-17, R 0) | FAIL (between 0.04762 > band 0.001) |  |
| random_dag_50 | 50 | 90 | Y | not_distributional_equivalent | match (D 0.6252, R 1.025) | FAIL (between 1.112 > band 1.025) | crossings, stress, edge_length_cv, ordering_inversions |
| org_chart_deep | 79 | 78 | Y | not_distributional_equivalent | mismatch (D 4.251e-16, R 0.5082) | FAIL (between 1.242 > band 0.5082) | crossings, stress, edge_length_cv, ordering_inversions |

Interpretation: remaining per-seed mismatches first diverge at `AbstractBarycenterPortDistributor.distributePortsWhileSweeping` generated-port rank/order feedback, after the Java RNG stream has been matched through `LayerSweepCrossingMinimizer.initialize`, `ISweepPortDistributor.create`, `LayerSweepCrossingMinimizer.compareDifferentRandomizedLayouts`, and `BarycenterHeuristic` randomization. The distributional tier is earned only where layers stay exact, scalar TOSTs pass, cross-seed variance is comparable, and the elkjs-vs-native Procrustes cloud sits inside the within-elkjs/native spread band.
