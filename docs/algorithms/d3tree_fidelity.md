# d3-hierarchy tree/cluster fidelity verification

Reference: d3-hierarchy through the Node adapter.
Production pipelines are Python source ports and never invoke Node.

## d3_tree

Result: **5/5 bit-exact**, **0 N/A**.

| graph | N | E | max abs diff | first divergent stage | verdict |
|---|---:|---:|---:|---|---|
| single_node | 1 | 0 | 0.000e+00 | none | bit-exact |
| path | 5 | 4 | 0.000e+00 | none | bit-exact |
| star | 6 | 5 | 0.000e+00 | none | bit-exact |
| binary_tree | 7 | 6 | 0.000e+00 | none | bit-exact |
| org_chart | 8 | 7 | 0.000e+00 | none | bit-exact |

## d3_cluster

Result: **5/5 bit-exact**, **0 N/A**.

| graph | N | E | max abs diff | first divergent stage | verdict |
|---|---:|---:|---:|---|---|
| single_node | 1 | 0 | 0.000e+00 | none | bit-exact |
| path | 5 | 4 | 0.000e+00 | none | bit-exact |
| star | 6 | 5 | 0.000e+00 | none | bit-exact |
| binary_tree | 7 | 6 | 0.000e+00 | none | bit-exact |
| org_chart | 8 | 7 | 0.000e+00 | none | bit-exact |

## Stage bisection

All current verification rows are bit-exact. For `d3_tree`, this covers the first walk, apportion/thread shifts, and second walk because the raw coordinates match d3's `tree().nodeSize([1, 1])` output exactly. For `d3_cluster`, this covers the leaf walk and normalization used by d3's default `cluster()`.

Non-tree graphs are converted to a deterministic spanning hierarchy by keeping the first incoming parent for each node, matching the reference adapter's input preparation.
