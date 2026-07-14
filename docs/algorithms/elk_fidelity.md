# ELK Layered fidelity verification

Reference: elkjs through `dagua.eval.competitors.elk_competitor.ElkLayered`. One deterministic layout is cached per graph. The production pipeline never invokes Node.

Parameters: `{'elk.algorithm': 'layered', 'elk.direction': 'DOWN', 'elk.spacing.nodeNode': 40, 'elk.layered.spacing.nodeNodeBetweenLayers': 60}`.

Summary: 2/11 bit-exact, 5 close, 4 divergent.

| graph | N | E | layer | order | d_R | anisotropic | max abs diff | class | first divergent phase |
|---|---:|---:|---|---|---:|---:|---:|---|---|
| single_node | 1 | 0 | Y | Y | 0 | 0 | 0 | bit-exact |  |
| small_chain | 6 | 5 | Y | Y | 0 | 0 | 0 | bit-exact |  |
| binary_tree | 11 | 10 | Y | Y | 0.0715855 | 0.0713416 | 48.5734 | close | node placement: Brandes-Koepf balancing/spacing mismatch |
| diamond | 4 | 4 | Y | N | 0.525365 | 0.458 | 104.109 | divergent | crossing minimization: within-layer order mismatch |
| grid_5x5 | 25 | 40 | Y | Y | 1.32882e-08 | 1.32007e-08 | 1.52588e-05 | close | node placement: Brandes-Koepf balancing/spacing mismatch |
| org_chart_small | 16 | 15 | Y | N | 1.2151 | 0.96181 | 1054.03 | divergent | crossing minimization: within-layer order mismatch |
| long_skip | 5 | 6 | Y | Y | 0.0781897 | 0.0521634 | 31.1713 | close | node placement: Brandes-Koepf balancing/spacing mismatch |
| disconnected | 5 | 2 | Y | Y | 0.0757369 | 0.0747507 | 40 | close | node placement: Brandes-Koepf balancing/spacing mismatch |
| cycle_4 | 4 | 4 | Y | Y | 0.04762 | 0 | 19.3208 | close | cycle breaking: ELK GREEDY tie semantics not fully ported |
| random_dag_50 | 50 | 90 | Y | N | 1.19948 | 0.923796 | 1757.14 | divergent | crossing minimization: within-layer order mismatch |
| org_chart_deep | 79 | 78 | Y | N | 1.37665 | 0.986329 | 4621.65 | divergent | crossing minimization: within-layer order mismatch |

Named residual: the current native port matches ELK's public coordinate contract and simple layer spacing, but diverges first at ELK's exact layer-sweep/Brandes-Koepf tie semantics on multi-node layers and at ELK GREEDY cycle-breaking ties on cyclic graphs. Edge routing and port extrema are outside this node-position fidelity scope.
