# d3-dag fidelity verification

Reference: d3-dag 1.2.2 through the Node adapter.
The production pipeline is a Python source port and never invokes Node.

Result: **3/11 bit-exact**, **4 close**, **3 divergent**, **1 unsupported** at `d_R < 1e-9`.

| graph | N | E | Procrustes d_R | anisotropic d_R | max raw diff | layer | order | first divergent stage | verdict |
|---|---:|---:|---:|---:|---:|---|---|---|---|
| single_node | 1 | 0 | 0.000e+00 | 0.000e+00 | 0.000e+00 | yes | yes | none | bit-exact |
| small_chain | 6 | 5 | 0.000e+00 | 0.000e+00 | 0.000e+00 | yes | yes | none | bit-exact |
| binary_tree | 11 | 10 | 1.072e-07 | 1.072e-07 | 2.954e+02 | yes | no | order | close |
| diamond | 4 | 4 | 0.000e+00 | 0.000e+00 | 6.511e+01 | yes | yes | none | bit-exact |
| grid_5x5 | 25 | 40 | 3.128e-08 | 3.126e-08 | 2.729e+02 | yes | no | order | close |
| org_chart_small | 16 | 15 | 6.647e-08 | 6.640e-08 | 8.317e+02 | yes | no | order | close |
| long_skip | 5 | 6 | 3.907e-08 | 3.455e-08 | 7.629e-06 | yes | no | order | close |
| disconnected | 5 | 2 | 1.404e-01 | 1.379e-01 | 3.500e+01 | yes | no | order | divergent |
| cycle_4 | 4 | 4 | n/a | n/a | n/a | n/a | n/a | d3-dag Sugiyama requires an acyclic graph | unsupported |
| random_dag_50 | 50 | 90 | 7.560e-01 | 6.954e-01 | 1.098e+03 | yes | no | order | divergent |
| org_chart_deep | 79 | 78 | 1.245e+00 | 9.740e-01 | 3.629e+03 | yes | no | order | divergent |

## Stage bisection

The current source port matches d3-dag's deterministic layer LP on the DAG corpus. The remaining non-bit-exact rows first diverge at node-order tie handling: d3-dag's mutable graph iteration order is not identical to the integer-index order used by the headless tensor pipeline.

Cyclic input is reported as an input-domain residual because d3-dag Sugiyama requires a DAG and the reference adapter returns an error.
