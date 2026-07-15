# d3-dag fidelity verification

Reference: d3-dag 1.2.2 through the Node adapter.
The production pipeline is a Python source port and never invokes Node.

Result: **4/11 bit-exact** (`d_R < 1e-9`), **4 positional** (`d_R < 1e-3`), **2 divergent**, **1 unsupported**.

| graph | N | E | Procrustes d_R | anisotropic d_R | max raw diff | layer | order | first divergent stage | verdict |
|---|---:|---:|---:|---:|---:|---|---|---|---|
| single_node | 1 | 0 | 0.000e+00 | 0.000e+00 | 0.000e+00 | yes | yes | none | bit-exact |
| small_chain | 6 | 5 | 0.000e+00 | 0.000e+00 | 0.000e+00 | yes | yes | none | bit-exact |
| binary_tree | 11 | 10 | 1.072e-07 | 1.072e-07 | 2.954e+02 | yes | yes | solver-floor | positional |
| diamond | 4 | 4 | 0.000e+00 | 0.000e+00 | 6.511e+01 | yes | yes | none | bit-exact |
| grid_5x5 | 25 | 40 | 3.128e-08 | 3.126e-08 | 2.729e+02 | yes | yes | solver-floor | positional |
| org_chart_small | 16 | 15 | 6.647e-08 | 6.640e-08 | 8.317e+02 | yes | yes | solver-floor | positional |
| long_skip | 5 | 6 | 3.907e-08 | 3.455e-08 | 7.629e-06 | yes | yes | solver-floor | positional |
| disconnected | 5 | 2 | 1.508e-16 | 0.000e+00 | 0.000e+00 | yes | yes | none | bit-exact |
| cycle_4 | 4 | 4 | n/a | n/a | n/a | n/a | n/a | d3-dag Sugiyama requires an acyclic graph | unsupported |
| random_dag_50 | 50 | 90 | 6.713e-01 | 6.296e-01 | 1.001e+03 | yes | no | order | divergent |
| org_chart_deep | 79 | 78 | 1.245e+00 | 9.740e-01 | 3.629e+03 | yes | no | order | divergent |

## Stage bisection

The current source port matches d3-dag's deterministic layer LP on all positional-or-better rows. Positional rows are solver-floor residuals below `d_R < 1e-3`; remaining divergent rows first differ in layer/order tie handling.

Cyclic input is reported as an input-domain residual because d3-dag Sugiyama requires a DAG and the reference adapter returns an error.
