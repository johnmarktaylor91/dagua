# Round 43 fdp_clusters Summary

## Diagnosis

Graphviz fdp cluster recursion uses `finalCC` to turn a laid-out child graph into
the derived-node size seen by the parent. That bbox is computed from child node
boxes and child cluster boxes, then expanded by the cluster margin and graph
label border stored in `GD_border`.

The Python recursion path had two remaining fidelity gaps:

- Recursive component `xLayout` was running before child cluster layouts had
  final bboxes, so parent derived-node sizing did not use the child dimensions.
- Cluster bboxes were still raw node extents rather than Graphviz-style
  `finalCC` extents with cluster margin and top label border.

## Ports

- Restored the R40 split ordering:
  `tLayout -> expandCluster -> recursive child layout -> xLayout`.
- Added Graphviz default bbox constants:
  `CL_OFFSET = 8 pt`, top label border `18 pt`, node floors from Graphviz
  source defaults (`0.75 in` width, `0.5 in` height).
- Added bottom-up cluster obstacle bboxes so compound routing metadata uses
  direct leaves plus child cluster boxes rather than flat descendant extents.
- Added the recursive port-aware `tLayout` initialization and update behavior:
  port nodes are placed on the boundary ellipse, adjacent internal nodes seed
  from positioned ports, port-port repulsion is strengthened, and movement is
  clamped to the boundary ellipse.

## Smoke Results

Round 40 reference from `eval_output/algo_fidelity/round_40/fdp_clusters/SUMMARY.md`:

| Topology | Mean | Max |
| --- | ---: | ---: |
| one_cluster | 0.155245986 | 0.312146136 |
| path | 0.039992521 | 0.076886609 |
| clustered | 0.252408178 | 0.328790561 |
| multi_cluster | 0.160410249 | 0.172810725 |

Round 43 after this port, `python eval_output/algo_fidelity/round_40/fdp_clusters/smoke_check.py`:

| Topology | Seed 1 | Seed 2 | Seed 3 | Mean | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| one_cluster | 0.300444180 | 0.123267211 | 0.311203930 | 0.244971774 | 0.311203930 |
| path | 0.076886609 | 0.023582772 | 0.019508183 | 0.039992521 | 0.076886609 |
| clustered | 0.269335653 | 0.195194714 | 0.196027235 | 0.220185867 | 0.269335653 |
| multi_cluster | 0.169847873 | 0.149791494 | 0.144064542 | 0.154567970 | 0.169847873 |

## Verdict

Hold. Clustered RMSD improved from the Round 40 smoke mean of `0.252408178` to
`0.220185867`, and multi-cluster improved from `0.160410249` to `0.154567970`,
but the `<0.05` target was not reached. `classic_fmmm_graphviz_fdp_fidelity`
therefore remains disabled in `dagua/eval/variants.py`.

## Remaining Gap

The residual is no longer explained by default cluster margin, top label border,
derived-node size timing, or boundary-port seeding. The remaining architectural
gap is Cgraph metadata fidelity:

- Dagua receives cluster names and flat membership only; Graphviz lays out
  concrete Cgraph subgraphs with label text, label position flags, object
  allocation/iteration order, and per-object records.
- The Python path uses deterministic tensor order for derived nodes and grouped
  real edges, while Graphviz's `agnode`/`agedge` object order affects same-angle
  port ordering, component traversal, and tie breaks.
- Exact graph label width is still unavailable in the tensor API. The port uses
  the default height/border requested for this round, but `finalCC` also widens
  clusters when the measured label width exceeds content width.
