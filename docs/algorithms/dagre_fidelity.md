# Dagre fidelity verification

Reference: dagre.js 0.8.5 through the existing Node adapter.
One deterministic layout is cached per graph. The production pipeline never invokes Node.

Result: **12/12 similarity-exact**, **0 close**, **0 divergent** at `d_R < 1e-9`.

| graph | N | E | Procrustes d_R | anisotropic d_R | max raw diff | verdict |
|---|---:|---:|---:|---:|---:|---|
| single_node | 1 | 0 | 0.000e+00 | 0.000e+00 | 0.000e+00 | bit-exact |
| small_chain | 6 | 5 | 0.000e+00 | 0.000e+00 | 0.000e+00 | bit-exact |
| binary_tree | 11 | 10 | 3.826e-16 | 0.000e+00 | 0.000e+00 | bit-exact |
| diamond | 4 | 4 | 2.480e-24 | 0.000e+00 | 0.000e+00 | bit-exact |
| grid_5x5 | 25 | 40 | 1.898e-16 | 0.000e+00 | 0.000e+00 | bit-exact |
| org_chart_small | 16 | 15 | 2.190e-16 | 0.000e+00 | 0.000e+00 | bit-exact |
| long_skip | 5 | 6 | 3.788e-16 | 0.000e+00 | 0.000e+00 | bit-exact |
| disconnected | 5 | 2 | 9.714e-17 | 0.000e+00 | 0.000e+00 | bit-exact |
| cycle_4 | 4 | 4 | 8.092e-17 | 0.000e+00 | 0.000e+00 | bit-exact |
| multiedge_adapter | 4 | 5 | 1.828e-25 | 0.000e+00 | 0.000e+00 | bit-exact |
| random_dag_50 | 50 | 90 | 6.456e-16 | 0.000e+00 | 0.000e+00 | bit-exact |
| org_chart_deep | 79 | 78 | 2.038e-16 | 0.000e+00 | 0.000e+00 | bit-exact |

## Stage bisection and variants

The initial reuse probe diverged on 4/60 dense randomized cases. Rank snapshots identified network-simplex as the first-divergent stage: Graphviz's feasible-tree and leaving-edge tie semantics differ from Graphlib/dagre.js. Replacing only that stage with the source-exact Dagre variant closed the probe to 60/60 similarity-exact.

A separate 48-case option matrix (four graph shapes by twelve settings) covered TB/BT/LR/RL, UL/UR/DL/DR, all three rankers, both acyclicers, and non-default nodesep/ranksep/edgesep. All 48 were similarity-exact (`d_R < 1e-9`).

Raw-coordinate translation can differ on cyclic graphs because only node placement, not edge-route extrema, is returned by the headless pipeline. This is a named output-boundary residual; similarity coordinates and all pairwise geometry are exact.
