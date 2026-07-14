# ELK Layered fidelity verification

Reference: elkjs through `dagua.eval.competitors.elk_competitor.ElkLayered`. One deterministic layout is cached per graph. The production pipeline never invokes Node.

Parameters: `{'elk.algorithm': 'layered', 'elk.direction': 'DOWN', 'elk.spacing.nodeNode': 40, 'elk.layered.spacing.nodeNodeBetweenLayers': 60}`.

Summary: 2/11 bit-exact, 2 close, 7 divergent.

| graph | N | E | layer | order | d_R | anisotropic | max abs diff | class | first divergent phase |
|---|---:|---:|---|---|---:|---:|---:|---|---|
| single_node | 1 | 0 | Y | Y | 0 | 0 | 0 | bit-exact |  |
| small_chain | 6 | 5 | Y | Y | 0 | 0 | 0 | bit-exact |  |
| binary_tree | 11 | 10 | Y | Y | 0.35078 | 0.335668 | 202.476 | divergent | node placement: Brandes-Koepf balancing/spacing mismatch |
| diamond | 4 | 4 | Y | N | 0.265019 | 0.262682 | 104.109 | divergent | crossing minimization: within-layer order mismatch |
| grid_5x5 | 25 | 40 | Y | Y | 0.218556 | 0.216786 | 170.692 | divergent | node placement: Brandes-Koepf balancing/spacing mismatch |
| org_chart_small | 16 | 15 | Y | N | 1.11039 | 0.920246 | 1054.03 | divergent | crossing minimization: within-layer order mismatch |
| long_skip | 5 | 6 | Y | Y | 0.0751885 | 0.0402415 | 26.7569 | close | node placement: Brandes-Koepf balancing/spacing mismatch |
| disconnected | 5 | 2 | Y | N | 0.127968 | 0.120893 | 119.536 | divergent | crossing minimization: within-layer order mismatch |
| cycle_4 | 4 | 4 | Y | Y | 0.04762 | 0 | 19.3208 | close | cycle breaking: ELK GREEDY tie semantics not fully ported |
| random_dag_50 | 50 | 90 | Y | N | 1.27335 | 0.93803 | 1938.62 | divergent | crossing minimization: within-layer order mismatch |
| org_chart_deep | 79 | 78 | Y | N | 1.38056 | 0.986882 | 4621.65 | divergent | crossing minimization: within-layer order mismatch |

Named residual: the current native port matches ELK's public coordinate contract and simple layer spacing, but diverges first at ELK's exact layer-sweep/Brandes-Koepf tie semantics on multi-node layers and at ELK GREEDY cycle-breaking ties on cyclic graphs. Edge routing and port extrema are outside this node-position fidelity scope.

## Distributional verification

Reference: elkjs with `elk.randomSeed=1..30`; native ELK uses the same seed set, Java-compatible restart shuffles, and `thoroughness=7`.

| graph | N | E | layers | verdict | variance | procrustes | metric TOST failures |
|---|---:|---:|---|---|---|---|---|
| single_node | 1 | 0 | Y | DISTRIBUTIONAL_EQUIVALENT | match (D 0, R 0) | PASS (between 0 <= band 0.001) |  |
| small_chain | 6 | 5 | Y | DISTRIBUTIONAL_EQUIVALENT | match (D 0, R 0) | PASS (between 0 <= band 0.001) |  |
| binary_tree | 11 | 10 | Y | not_distributional_equivalent | match (D 0.6561, R 0.3984) | FAIL (between 0.8258 > band 0.6561) | crossings, stress, edge_length_cv, ordering_inversions |
| diamond | 4 | 4 | Y | not_distributional_equivalent | mismatch (D 0, R 0.2609) | FAIL (between 0.265 > band 0.2609) | stress, edge_length_cv, ordering_inversions |
| grid_5x5 | 25 | 40 | Y | not_distributional_equivalent | match (D 2.002e-16, R 2.872e-16) | FAIL (between 0.2186 > band 0.001) | stress, edge_length_cv |
| org_chart_small | 16 | 15 | Y | not_distributional_equivalent | mismatch (D 2.19e-16, R 0.7343) | PASS (between 0.5062 <= band 0.7343) | ordering_inversions |
| long_skip | 5 | 6 | Y | not_distributional_equivalent | mismatch (D 6.901e-17, R 0.06737) | FAIL (between 0.1256 > band 0.06737) | crossings |
| disconnected | 5 | 2 | Y | not_distributional_equivalent | match (D 2.041e-16, R 2.765e-16) | FAIL (between 0.128 > band 0.001) | ordering_inversions |
| cycle_4 | 4 | 4 | Y | not_distributional_equivalent | match (D 0, R 0) | FAIL (between 0.1177 > band 0.001) | stress, edge_length_cv |
| random_dag_50 | 50 | 90 | Y | not_distributional_equivalent | mismatch (D 0.4383, R 1.025) | FAIL (between 1.155 > band 1.025) | crossings, stress, edge_length_cv |
| org_chart_deep | 79 | 78 | Y | not_distributional_equivalent | mismatch (D 3.596e-19, R 0.5082) | FAIL (between 1.264 > band 0.5082) | crossings, stress, edge_length_cv, ordering_inversions |

Interpretation: the native port is still not expected to be per-seed bit-exact beyond the known ceiling, because elkjs has hidden randomized layer-sweep state. The distributional tier is earned only where layers stay exact, scalar TOSTs pass, cross-seed variance is comparable, and the elkjs-vs-native Procrustes cloud sits inside the within-elkjs/native spread band.
