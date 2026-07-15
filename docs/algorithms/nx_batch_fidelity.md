# NetworkX simple-layout batch fidelity

Reference: `networkx.drawing.layout` from NetworkX 3.6.1.

Pinned fallbacks: bipartite uses BFS parity from node 0 as the explicit node set; multipartite and bfs use BFS-distance layers from node 0, appending disconnected components in node order so every Dagua tensor node has a position.

| Layout | Graph | d_R | max_abs | Class |
| --- | --- | ---: | ---: | --- |
| circular | single_node | 0.000e+00 | 0.000e+00 | bit-exact |
| circular | small_chain | 1.110e-16 | 0.000e+00 | bit-exact |
| circular | binary_tree | 1.405e-16 | 0.000e+00 | bit-exact |
| circular | diamond | 2.586e-16 | 0.000e+00 | bit-exact |
| circular | grid_5x5 | 1.669e-16 | 0.000e+00 | bit-exact |
| circular | org_chart_small | 1.679e-16 | 0.000e+00 | bit-exact |
| circular | long_skip | 2.289e-16 | 0.000e+00 | bit-exact |
| circular | disconnected | 2.289e-16 | 0.000e+00 | bit-exact |
| circular | cycle_4 | 2.586e-16 | 0.000e+00 | bit-exact |
| circular | random_dag_50 | 2.446e-17 | 0.000e+00 | bit-exact |
| circular | org_chart_deep | 1.688e-16 | 0.000e+00 | bit-exact |
| shell | single_node | 0.000e+00 | 0.000e+00 | bit-exact |
| shell | small_chain | 1.117e-16 | 0.000e+00 | bit-exact |
| shell | binary_tree | 3.128e-16 | 0.000e+00 | bit-exact |
| shell | diamond | 2.270e-16 | 0.000e+00 | bit-exact |
| shell | grid_5x5 | 1.945e-16 | 0.000e+00 | bit-exact |
| shell | org_chart_small | 1.662e-16 | 0.000e+00 | bit-exact |
| shell | long_skip | 1.039e-16 | 0.000e+00 | bit-exact |
| shell | disconnected | 1.039e-16 | 0.000e+00 | bit-exact |
| shell | cycle_4 | 2.270e-16 | 0.000e+00 | bit-exact |
| shell | random_dag_50 | 3.198e-16 | 0.000e+00 | bit-exact |
| shell | org_chart_deep | 9.635e-17 | 0.000e+00 | bit-exact |
| spiral | single_node | 0.000e+00 | 0.000e+00 | bit-exact |
| spiral | small_chain | 3.065e-16 | 0.000e+00 | bit-exact |
| spiral | binary_tree | 1.841e-16 | 0.000e+00 | bit-exact |
| spiral | diamond | 5.756e-16 | 0.000e+00 | bit-exact |
| spiral | grid_5x5 | 1.508e-16 | 0.000e+00 | bit-exact |
| spiral | org_chart_small | 3.424e-16 | 0.000e+00 | bit-exact |
| spiral | long_skip | 2.525e-16 | 0.000e+00 | bit-exact |
| spiral | disconnected | 2.525e-16 | 0.000e+00 | bit-exact |
| spiral | cycle_4 | 5.756e-16 | 0.000e+00 | bit-exact |
| spiral | random_dag_50 | 3.453e-16 | 0.000e+00 | bit-exact |
| spiral | org_chart_deep | 2.706e-16 | 0.000e+00 | bit-exact |
| bipartite | single_node | 0.000e+00 | 0.000e+00 | bit-exact |
| bipartite | small_chain | 1.522e-18 | 0.000e+00 | bit-exact |
| bipartite | binary_tree | 2.323e-17 | 0.000e+00 | bit-exact |
| bipartite | diamond | 0.000e+00 | 0.000e+00 | bit-exact |
| bipartite | grid_5x5 | 1.962e-17 | 0.000e+00 | bit-exact |
| bipartite | org_chart_small | 2.550e-17 | 0.000e+00 | bit-exact |
| bipartite | long_skip | 1.875e-18 | 0.000e+00 | bit-exact |
| bipartite | disconnected | 1.875e-18 | 0.000e+00 | bit-exact |
| bipartite | cycle_4 | 0.000e+00 | 0.000e+00 | bit-exact |
| bipartite | random_dag_50 | 3.879e-18 | 0.000e+00 | bit-exact |
| bipartite | org_chart_deep | 2.520e-17 | 0.000e+00 | bit-exact |
| multipartite | single_node | 0.000e+00 | 0.000e+00 | bit-exact |
| multipartite | small_chain | 0.000e+00 | 0.000e+00 | bit-exact |
| multipartite | binary_tree | 8.583e-18 | 0.000e+00 | bit-exact |
| multipartite | diamond | 0.000e+00 | 0.000e+00 | bit-exact |
| multipartite | grid_5x5 | 4.315e-18 | 0.000e+00 | bit-exact |
| multipartite | org_chart_small | 1.997e-17 | 0.000e+00 | bit-exact |
| multipartite | long_skip | 3.476e-18 | 0.000e+00 | bit-exact |
| multipartite | disconnected | 0.000e+00 | 0.000e+00 | bit-exact |
| multipartite | cycle_4 | 0.000e+00 | 0.000e+00 | bit-exact |
| multipartite | random_dag_50 | 4.325e-20 | 0.000e+00 | bit-exact |
| multipartite | org_chart_deep | 6.375e-17 | 0.000e+00 | bit-exact |
| bfs | single_node | 0.000e+00 | 0.000e+00 | bit-exact |
| bfs | small_chain | 0.000e+00 | 0.000e+00 | bit-exact |
| bfs | binary_tree | 8.583e-18 | 0.000e+00 | bit-exact |
| bfs | diamond | 0.000e+00 | 0.000e+00 | bit-exact |
| bfs | grid_5x5 | 4.315e-18 | 0.000e+00 | bit-exact |
| bfs | org_chart_small | 1.997e-17 | 0.000e+00 | bit-exact |
| bfs | long_skip | 3.476e-18 | 0.000e+00 | bit-exact |
| bfs | disconnected | 0.000e+00 | 0.000e+00 | bit-exact |
| bfs | cycle_4 | 0.000e+00 | 0.000e+00 | bit-exact |
| bfs | random_dag_50 | 4.325e-20 | 0.000e+00 | bit-exact |
| bfs | org_chart_deep | 6.375e-17 | 0.000e+00 | bit-exact |
| arf | single_node | 0.000e+00 | 0.000e+00 | bit-exact |
| arf | small_chain | 2.751e-16 | 0.000e+00 | bit-exact |
| arf | binary_tree | 1.102e-16 | 0.000e+00 | bit-exact |
| arf | diamond | 2.375e-16 | 0.000e+00 | bit-exact |
| arf | grid_5x5 | 1.561e-16 | 0.000e+00 | bit-exact |
| arf | org_chart_small | 1.057e-16 | 0.000e+00 | bit-exact |
| arf | long_skip | 6.293e-17 | 0.000e+00 | bit-exact |
| arf | disconnected | 2.058e-16 | 0.000e+00 | bit-exact |
| arf | cycle_4 | 3.801e-16 | 0.000e+00 | bit-exact |
| arf | random_dag_50 | 2.838e-16 | 0.000e+00 | bit-exact |
| arf | org_chart_deep | 2.221e-16 | 0.000e+00 | bit-exact |

## Summary

- `circular`: 11 bit-exact, 0 positional, 0 divergent; max d_R=2.586e-16.
- `shell`: 11 bit-exact, 0 positional, 0 divergent; max d_R=3.198e-16.
- `spiral`: 11 bit-exact, 0 positional, 0 divergent; max d_R=5.756e-16.
- `bipartite`: 11 bit-exact, 0 positional, 0 divergent; max d_R=2.550e-17.
- `multipartite`: 11 bit-exact, 0 positional, 0 divergent; max d_R=6.375e-17.
- `bfs`: 11 bit-exact, 0 positional, 0 divergent; max d_R=6.375e-17.
- `arf`: 11 bit-exact, 0 positional, 0 divergent; max d_R=3.801e-16.
