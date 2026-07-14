# Graphviz twopi/circo fidelity verification

Reference: Graphviz 7.0.5 through the Graphviz JSON adapter. One deterministic layout is cached per graph. The production pipelines do not invoke Graphviz.

## twopi

Result: **4/11 similarity-exact**, **5 positional-identical**, **2 divergent**. Thresholds: bit-exact `d_R < 1e-09`, positional `d_R < 1e-03`.

| graph | N | E | Procrustes d_R | anisotropic d_R | max raw diff | verdict | first divergent stage |
|---|---:|---:|---:|---:|---:|---|---|
| single_node | 1 | 0 | 0.000e+00 | 0.000e+00 | 3.800e+01 | bit-exact | none |
| small_chain | 6 | 5 | 3.050e-17 | 3.050e-17 | 1.610e+02 | bit-exact | none |
| binary_tree | 11 | 10 | 1.011e-05 | 9.876e-06 | 5.930e+02 | positional-identical | none |
| diamond | 4 | 4 | 2.764e-16 | 3.000e-17 | 3.770e+02 | bit-exact | none |
| grid_5x5 | 25 | 40 | 1.110e-05 | 1.072e-05 | 1.318e+03 | positional-identical | none |
| org_chart_small | 16 | 15 | 1.540e-05 | 1.170e-05 | 4.490e+02 | positional-identical | none |
| long_skip | 5 | 6 | 1.174e-05 | 1.173e-05 | 3.288e+02 | positional-identical | none |
| disconnected | 5 | 2 | 8.181e-01 | 7.398e-01 | 3.365e+02 | divergent | component packing after radial layout |
| cycle_4 | 4 | 4 | 2.764e-16 | 3.000e-17 | 3.770e+02 | bit-exact | none |
| random_dag_50 | 50 | 90 | 6.249e-01 | 5.891e-01 | 1.120e+03 | divergent | component packing after radial layout |
| org_chart_deep | 79 | 78 | 1.097e-05 | 1.096e-05 | 1.094e+03 | positional-identical | none |

## circo

Result: **4/11 similarity-exact**, **4 positional-identical**, **3 divergent**. Thresholds: bit-exact `d_R < 1e-09`, positional `d_R < 1e-03`.

| graph | N | E | Procrustes d_R | anisotropic d_R | max raw diff | verdict | first divergent stage |
|---|---:|---:|---:|---:|---:|---|---|
| single_node | 1 | 0 | 0.000e+00 | 0.000e+00 | 3.800e+01 | bit-exact | none |
| small_chain | 6 | 5 | 0.000e+00 | 6.536e-17 | 6.825e+02 | bit-exact | none |
| binary_tree | 11 | 10 | 8.971e-05 | 8.874e-05 | 1.029e+03 | positional-identical | none |
| diamond | 4 | 4 | 1.357e-16 | 9.956e-17 | 2.052e+02 | bit-exact | none |
| grid_5x5 | 25 | 40 | 9.299e-01 | 8.022e-01 | 1.253e+03 | divergent | intra-block circular ordering after block discovery |
| org_chart_small | 16 | 15 | 2.387e-04 | 2.387e-04 | 9.606e+02 | positional-identical | none |
| long_skip | 5 | 6 | 1.370e-05 | 1.111e-05 | 2.111e+02 | positional-identical | none |
| disconnected | 5 | 2 | 1.092e+00 | 9.147e-01 | 2.028e+02 | divergent | component packing plus block-tree coordinate placement |
| cycle_4 | 4 | 4 | 1.344e-16 | 9.778e-17 | 1.998e+02 | bit-exact | none |
| random_dag_50 | 50 | 90 | 1.126e+00 | 9.247e-01 | 2.663e+03 | divergent | blockpath ordering before block-tree coordinate placement |
| org_chart_deep | 79 | 78 | 3.392e-05 | 3.217e-05 | 7.527e+03 | positional-identical | none |

## Residual notes

The current twopi implementation matches the prescribed high-level stages: root selection by minimum eccentricity, BFS ring assignment, and subtree leaf-count angular wedges. Connected non-exact residuals are positional-identical at the Graphviz JSON output-precision floor. The two large twopi residuals are named component-packing residuals from Graphviz `pack.c`, a separable post-layout step.

The current circo implementation uses Graphviz-style owned block-cutpoint discovery, `circpos.c` child fan scaling/rotation, and Graphviz inch-to-local node-size conversion for block radii. Simple paths, simple cycles, the long-skip case, binary trees, and org-chart trees are positional-or-better. The remaining circo residuals first diverge in `blockpath.c` ordering for biconnected grids/random DAGs, with disconnected also requiring Graphviz component packing.
