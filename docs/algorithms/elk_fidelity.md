# ELK Layered fidelity verification

Reference: elkjs through `dagua.eval.competitors.elk_competitor.ElkLayered`. One deterministic layout is cached per graph. The production pipeline never invokes Node.

Parameters: `{'elk.algorithm': 'layered', 'elk.direction': 'DOWN', 'elk.spacing.nodeNode': 40, 'elk.layered.spacing.nodeNodeBetweenLayers': 60}`.

Summary: 2/11 bit-exact, 0 close, 9 divergent.

| graph | N | E | layer | order | d_R | anisotropic | max abs diff | class | first divergent phase |
|---|---:|---:|---|---|---:|---:|---:|---|---|
| single_node | 1 | 0 | Y | Y | 0 | 0 | 0 | bit-exact |  |
| small_chain | 6 | 5 | Y | Y | 0 | 0 | 0 | bit-exact |  |
| binary_tree | 11 | 10 | Y | N | 1.14211 | 0.873507 | 328.522 | divergent | crossing minimization: within-layer order mismatch |
| diamond | 4 | 4 | Y | N | 0.265019 | 0.262682 | 104.109 | divergent | crossing minimization: within-layer order mismatch |
| grid_5x5 | 25 | 40 | Y | Y | 0.218556 | 0.216786 | 170.692 | divergent | node placement: Brandes-Koepf balancing/spacing mismatch |
| org_chart_small | 16 | 15 | Y | N | 1.11039 | 0.920246 | 1054.03 | divergent | crossing minimization: within-layer order mismatch |
| long_skip | 5 | 6 | Y | Y | 0.108233 | 0.0850699 | 41.1713 | divergent | node placement: Brandes-Koepf balancing/spacing mismatch |
| disconnected | 5 | 2 | N | N | 0.51498 | 0.418941 | 239.073 | divergent | cycle breaking / layer assignment: first Y-band mismatch |
| cycle_4 | 4 | 4 | N | N | 0.895517 | 0.799073 | 188 | divergent | cycle breaking / layer assignment: first Y-band mismatch |
| random_dag_50 | 50 | 90 | N | N | 1.15952 | 0.92676 | 1915.54 | divergent | cycle breaking / layer assignment: first Y-band mismatch |
| org_chart_deep | 79 | 78 | Y | N | 1.38056 | 0.986882 | 4621.65 | divergent | crossing minimization: within-layer order mismatch |

Named residual: the current native port matches ELK's public coordinate contract and simple layer spacing, but diverges first at ELK's exact layer-sweep/Brandes-Koepf tie semantics on multi-node layers and at ELK GREEDY cycle-breaking ties on cyclic graphs. Edge routing and port extrema are outside this node-position fidelity scope.
