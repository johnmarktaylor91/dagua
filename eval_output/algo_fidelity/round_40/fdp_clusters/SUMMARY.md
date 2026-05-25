# Round 40 fdp_clusters Summary

## Per-step divergence analysis

- Derived graph construction matched the expected one-level Graphviz shape for the diagnostic graph:
  cluster node first, then direct leaves, with real edges grouped as `(0,1)`, `(0,2)`, and `(1,2)`.
- The largest confirmed Python-side divergence was the recursion ordering around `expandCluster`:
  the previous path ran the combined `tLayout`/`xLayout` kernel before child cluster bboxes existed.
  Graphviz runs `tLayout`, calls `expandCluster`, recursively lays out the child, assigns the child
  bbox as the derived-node size, deletes ports, then runs `xLayout`.
- A second confirmed divergence was the port-aware child `tLayout` path. Graphviz initializes ports
  on the boundary ellipse, initializes adjacent internal nodes from positioned ports, applies the
  stronger port-port repulsion, and constrains nodes to the ellipse during updates.
- Final cluster bbox handling also differed: child cluster layouts now receive the default
  `CL_OFFSET` margin and reserve default top label border space before their bbox is used as the
  parent derived-node size.
- `packGraphs` was not the dominant residual in the one-level harness: the ported pack offset matched
  `_graphviz_tile_pack_offsets` for the diagnostic component.

## Ported components

- Split recursive component layout into Graphviz's actual order:
  `tLayout -> expandCluster/recursive child layout -> delete ports -> xLayout`.
- Preserved generated port angles on derived port nodes.
- Added port-aware `fdp_tLayout` initialization/update behavior for recursive child graphs.
- Added Graphviz default cluster margin and approximate top-label border sizing to recursive
  cluster bboxes.

## Smoke RMSD

Round 39 baseline:

| Topology | Mean | Max |
| --- | ---: | ---: |
| path | 0.039992521 | 0.076886609 |
| clustered | 0.361568289 | 0.386046549 |
| multi_cluster | 0.310010317 | 0.324869584 |

Round 40 after this port, representative run with seeds `1, 2, 3`:

| Topology | Seed 1 | Seed 2 | Seed 3 | Mean | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| one_cluster | 0.017246259 | 0.136345563 | 0.312146136 | 0.155245986 | 0.312146136 |
| path | 0.076886609 | 0.023582772 | 0.019508183 | 0.039992521 | 0.076886609 |
| clustered | 0.328790561 | 0.209861381 | 0.218572593 | 0.252408178 | 0.328790561 |
| multi_cluster | 0.172810725 | 0.166272316 | 0.142147705 | 0.160410249 | 0.172810725 |

Repeated runs showed small Graphviz-side variation in clustered cases despite the same seed
plumbing; the verdict is unchanged because all observed clustered maxima stayed above `0.10`.

## Verdict

Hold. The clustered residual improved materially, but it remains above the `<0.10` ship target.
`classic_fmmm_graphviz_fdp_fidelity` should stay out of `dagua/eval/variants.py`.

## Residual root cause

The remaining mismatch is not packing. The most likely remaining architectural mismatch is cluster
metadata fidelity: the Python tensor path receives only cluster names/membership, while Graphviz's
recursive layout uses fully measured graph label data, exact font metrics, `GD_border`, and CGraph
object iteration/address ordering during the derived-node and port passes. The current port
approximates default cluster label dimensions from the cluster name, which improves derived-node
sizing but is not exact enough to meet the bit-exact target for clustered topologies.

## Verification

- `pytest tests/test_layout/test_fmmm_fdp_recursion.py -x --tb=short -q`
- `python eval_output/algo_fidelity/round_40/fdp_clusters/smoke_check.py`
