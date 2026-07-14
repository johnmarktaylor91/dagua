# Graphviz twopi/circo fidelity verification

Reference: Graphviz 7.0.5 through the Graphviz JSON adapter. One deterministic layout is cached per graph. The production pipelines do not invoke Graphviz.

## twopi

Result: **4/11 similarity-exact**, **0 close**, **7 divergent** at `d_R < 1e-9`.

| graph | N | E | Procrustes d_R | anisotropic d_R | max raw diff | verdict | first divergent stage |
|---|---:|---:|---:|---:|---:|---|---|
| single_node | 1 | 0 | 0.000e+00 | 0.000e+00 | 3.800e+01 | bit-exact | none |
| small_chain | 6 | 5 | 3.050e-17 | 3.050e-17 | 1.610e+02 | bit-exact | none |
| binary_tree | 11 | 10 | 1.011e-05 | 9.876e-06 | 5.930e+02 | divergent | angular wedge/order after BFS rings |
| diamond | 4 | 4 | 2.764e-16 | 3.000e-17 | 3.770e+02 | bit-exact | none |
| grid_5x5 | 25 | 40 | 8.280e-01 | 7.487e-01 | 1.206e+03 | divergent | angular wedge/order after BFS rings |
| org_chart_small | 16 | 15 | 1.540e-05 | 1.170e-05 | 4.490e+02 | divergent | angular wedge/order after BFS rings |
| long_skip | 5 | 6 | 1.174e-05 | 1.173e-05 | 3.288e+02 | divergent | angular wedge/order after BFS rings |
| disconnected | 5 | 2 | 8.181e-01 | 7.398e-01 | 3.365e+02 | divergent | angular wedge/order after BFS rings |
| cycle_4 | 4 | 4 | 2.764e-16 | 3.000e-17 | 3.770e+02 | bit-exact | none |
| random_dag_50 | 50 | 90 | 1.248e+00 | 9.751e-01 | 1.049e+03 | divergent | angular wedge/order after BFS rings |
| org_chart_deep | 79 | 78 | 1.097e-05 | 1.096e-05 | 1.094e+03 | divergent | angular wedge/order after BFS rings |

## circo

Result: **2/11 similarity-exact**, **0 close**, **9 divergent** at `d_R < 1e-9`.

| graph | N | E | Procrustes d_R | anisotropic d_R | max raw diff | verdict | first divergent stage |
|---|---:|---:|---:|---:|---:|---|---|
| single_node | 1 | 0 | 0.000e+00 | 0.000e+00 | 3.800e+01 | bit-exact | none |
| small_chain | 6 | 5 | 8.856e-02 | 8.847e-02 | 7.905e+02 | divergent | block ordering / block-tree placement |
| binary_tree | 11 | 10 | 6.049e-01 | 5.765e-01 | 1.304e+03 | divergent | block ordering / block-tree placement |
| diamond | 4 | 4 | 1.000e+00 | 8.660e-01 | 2.232e+02 | divergent | block ordering / block-tree placement |
| grid_5x5 | 25 | 40 | 1.198e+00 | 9.590e-01 | 1.197e+03 | divergent | block ordering / block-tree placement |
| org_chart_small | 16 | 15 | 9.474e-01 | 8.343e-01 | 8.631e+02 | divergent | block ordering / block-tree placement |
| long_skip | 5 | 6 | 1.370e-05 | 1.111e-05 | 2.280e+02 | divergent | block ordering / block-tree placement |
| disconnected | 5 | 2 | 1.309e+00 | 9.897e-01 | 2.920e+02 | divergent | block ordering / block-tree placement |
| cycle_4 | 4 | 4 | 1.344e-16 | 9.778e-17 | 1.998e+02 | bit-exact | none |
| random_dag_50 | 50 | 90 | 1.344e+00 | 9.951e-01 | 2.609e+03 | divergent | block ordering / block-tree placement |
| org_chart_deep | 79 | 78 | 8.934e-01 | 7.993e-01 | 6.648e+03 | divergent | block ordering / block-tree placement |

## Residual notes

The current twopi implementation matches the prescribed high-level stages: root selection by minimum eccentricity, BFS ring assignment, and subtree leaf-count angular wedges. Residuals identify Graphviz C-source tie/order details in `circleLayout`/`setSubtreeSize` as the remaining first-divergent stage.

The current circo implementation computes Tarjan biconnected components and lays each block on a deterministic circle. Residuals identify Graphviz's `install_in_cc`/`place_node` ordering and block-cutpoint tree arrangement as the remaining first-divergent stage.
