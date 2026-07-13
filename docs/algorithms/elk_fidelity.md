# ELK Layered fidelity verification

Reference: elkjs through `dagua.eval.competitors.elk_competitor.ElkLayered`. One deterministic layout is cached per graph. The production pipeline never invokes Node.

Parameters: `{'elk.algorithm': 'layered', 'elk.direction': 'DOWN', 'elk.spacing.nodeNode': 40, 'elk.layered.spacing.nodeNodeBetweenLayers': 60}`.

Summary: 2/11 bit-exact, 0 close, 9 divergent.

| graph | N | E | d_R | anisotropic | max abs diff | class | first divergent phase |
|---|---:|---:|---:|---:|---:|---|---|
| single_node | 1 | 0 | 0 | 0 | 0 | bit-exact |  |
| small_chain | 6 | 5 | 0 | 0 | 0 | bit-exact |  |
| binary_tree | 11 | 10 | 1.14211 | 0.873507 | 328.522 | divergent | crossing minimization / node placement: ELK layer-sweep and BK tie semantics |
| diamond | 4 | 4 | 0.265019 | 0.262682 | 104.109 | divergent | crossing minimization / node placement: ELK layer-sweep and BK tie semantics |
| grid_5x5 | 25 | 40 | 0.218556 | 0.216786 | 170.692 | divergent | crossing minimization / node placement: ELK layer-sweep and BK tie semantics |
| org_chart_small | 16 | 15 | 1.11039 | 0.920246 | 1054.03 | divergent | crossing minimization / node placement: ELK layer-sweep and BK tie semantics |
| long_skip | 5 | 6 | 0.12964 | 0.129367 | 43.5139 | divergent | crossing minimization / node placement: ELK layer-sweep and BK tie semantics |
| disconnected | 5 | 2 | 0.51498 | 0.418941 | 239.073 | divergent | crossing minimization / node placement: ELK layer-sweep and BK tie semantics |
| cycle_4 | 4 | 4 | 0.895204 | 0.800521 | 188 | divergent | cycle breaking: ELK GREEDY tie semantics not fully ported |
| random_dag_50 | 50 | 90 | 1.19733 | 0.931833 | 2211.9 | divergent | crossing minimization / node placement: ELK layer-sweep and BK tie semantics |
| org_chart_deep | 79 | 78 | 1.38056 | 0.986882 | 4621.65 | divergent | crossing minimization / node placement: ELK layer-sweep and BK tie semantics |

Named residual: the current native port matches ELK's public coordinate contract and simple layer spacing, but diverges first at ELK's exact layer-sweep/Brandes-Koepf tie semantics on multi-node layers and at ELK GREEDY cycle-breaking ties on cyclic graphs. Edge routing and port extrema are outside this node-position fidelity scope.
