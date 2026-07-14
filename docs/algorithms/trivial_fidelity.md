# Trivial deterministic layout fidelity

References: igraph-style star angles; documented degree-ring concentric; documented one-level circlepack; Graphviz-osage placeholder uses the same deterministic ordering as production pending a source port; standard BFS arc ordering; NetworkX 3.6.1 planar_layout for Chrobak-Payne planar.

| Layout | Graph | d_R | max_abs | Class | Reason |
| --- | --- | ---: | ---: | --- | --- |
| star | single_node | 0.000e+00 | 0.000e+00 | bit-exact |  |
| star | small_chain | 1.010e-17 | 0.000e+00 | bit-exact |  |
| star | binary_tree | 2.272e-19 | 0.000e+00 | bit-exact |  |
| star | diamond | 3.179e-16 | 0.000e+00 | bit-exact |  |
| star | grid_5x5 | 6.439e-17 | 0.000e+00 | bit-exact |  |
| star | long_skip | 6.123e-17 | 0.000e+00 | bit-exact |  |
| star | disconnected | 6.123e-17 | 0.000e+00 | bit-exact |  |
| star | cycle_4 | 3.179e-16 | 0.000e+00 | bit-exact |  |
| star | random_dag_50 | 4.484e-18 | 0.000e+00 | bit-exact |  |
| star | k5_non_planar | 6.123e-17 | 0.000e+00 | bit-exact |  |
| concentric | single_node | 0.000e+00 | 0.000e+00 | bit-exact |  |
| concentric | small_chain | 0.000e+00 | 0.000e+00 | bit-exact |  |
| concentric | binary_tree | 7.818e-17 | 0.000e+00 | bit-exact |  |
| concentric | diamond | 6.123e-17 | 0.000e+00 | bit-exact |  |
| concentric | grid_5x5 | 6.749e-17 | 0.000e+00 | bit-exact |  |
| concentric | long_skip | 1.170e-16 | 0.000e+00 | bit-exact |  |
| concentric | disconnected | 3.058e-17 | 0.000e+00 | bit-exact |  |
| concentric | cycle_4 | 6.123e-17 | 0.000e+00 | bit-exact |  |
| concentric | random_dag_50 | 3.217e-17 | 0.000e+00 | bit-exact |  |
| concentric | k5_non_planar | 1.010e-17 | 0.000e+00 | bit-exact |  |
| circlepack | single_node | 0.000e+00 | 0.000e+00 | bit-exact |  |
| circlepack | small_chain | 6.262e-17 | 0.000e+00 | bit-exact |  |
| circlepack | binary_tree | 7.889e-17 | 0.000e+00 | bit-exact |  |
| circlepack | diamond | 6.123e-17 | 0.000e+00 | bit-exact |  |
| circlepack | grid_5x5 | 1.063e-16 | 0.000e+00 | bit-exact |  |
| circlepack | long_skip | 2.972e-18 | 0.000e+00 | bit-exact |  |
| circlepack | disconnected | 2.972e-18 | 0.000e+00 | bit-exact |  |
| circlepack | cycle_4 | 6.123e-17 | 0.000e+00 | bit-exact |  |
| circlepack | random_dag_50 | 3.637e-17 | 0.000e+00 | bit-exact |  |
| circlepack | k5_non_planar | 2.972e-18 | 0.000e+00 | bit-exact |  |
| osage | single_node | 0.000e+00 | 0.000e+00 | bit-exact |  |
| osage | small_chain | 0.000e+00 | 0.000e+00 | bit-exact |  |
| osage | binary_tree | 0.000e+00 | 0.000e+00 | bit-exact |  |
| osage | diamond | 0.000e+00 | 0.000e+00 | bit-exact |  |
| osage | grid_5x5 | 0.000e+00 | 0.000e+00 | bit-exact |  |
| osage | long_skip | 0.000e+00 | 0.000e+00 | bit-exact |  |
| osage | disconnected | 0.000e+00 | 0.000e+00 | bit-exact |  |
| osage | cycle_4 | 0.000e+00 | 0.000e+00 | bit-exact |  |
| osage | random_dag_50 | 0.000e+00 | 0.000e+00 | bit-exact |  |
| osage | k5_non_planar | 0.000e+00 | 0.000e+00 | bit-exact |  |
| arc | single_node | 0.000e+00 | 0.000e+00 | bit-exact |  |
| arc | small_chain | 0.000e+00 | 0.000e+00 | bit-exact |  |
| arc | binary_tree | 0.000e+00 | 0.000e+00 | bit-exact |  |
| arc | diamond | 0.000e+00 | 0.000e+00 | bit-exact |  |
| arc | grid_5x5 | 0.000e+00 | 0.000e+00 | bit-exact |  |
| arc | long_skip | 0.000e+00 | 0.000e+00 | bit-exact |  |
| arc | disconnected | 0.000e+00 | 0.000e+00 | bit-exact |  |
| arc | cycle_4 | 0.000e+00 | 0.000e+00 | bit-exact |  |
| arc | random_dag_50 | 0.000e+00 | 0.000e+00 | bit-exact |  |
| arc | k5_non_planar | 0.000e+00 | 0.000e+00 | bit-exact |  |
| planar | single_node | 0.000e+00 | 0.000e+00 | bit-exact |  |
| planar | small_chain | 4.147e-16 | 0.000e+00 | bit-exact |  |
| planar | binary_tree | 2.064e-16 | 0.000e+00 | bit-exact |  |
| planar | diamond | 4.275e-16 | 0.000e+00 | bit-exact |  |
| planar | grid_5x5 | 2.136e-16 | 0.000e+00 | bit-exact |  |
| planar | long_skip | 1.784e-16 | 0.000e+00 | bit-exact |  |
| planar | disconnected | 3.630e-35 | 0.000e+00 | bit-exact |  |
| planar | cycle_4 | 4.275e-16 | 0.000e+00 | bit-exact |  |
| planar | random_dag_50 | N/A | N/A | N/A | G is not planar. |
| planar | k5_non_planar | N/A | N/A | N/A | G is not planar. |

## Summary

- `star`: 10 bit-exact, 0 positional, 0 N/A; max d_R=3.179e-16.
- `concentric`: 10 bit-exact, 0 positional, 0 N/A; max d_R=1.170e-16.
- `circlepack`: 10 bit-exact, 0 positional, 0 N/A; max d_R=1.063e-16.
- `osage`: 10 bit-exact, 0 positional, 0 N/A; max d_R=0.000e+00.
- `arc`: 10 bit-exact, 0 positional, 0 N/A; max d_R=0.000e+00.
- `planar`: 8 bit-exact, 0 positional, 2 N/A; max d_R=4.275e-16.
