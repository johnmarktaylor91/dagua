# Round 44 fdp_cgraph Summary

## Source Ports

- Cgraph nodes: Graphviz iterates `g->n_seq` via `agfstnode`/`agnxtnode`
  (`lib/cgraph/node.c:43-55`). The Python fdp fidelity path now builds a
  `_FdpCgraphContext` with node sequence ranks and uses it for direct leaves,
  derived graph construction, and component-local edge tensors
  (`dagua/layout/ops/pipelines/fmmm.py:1245-1325`,
  `dagua/layout/ops/pipelines/fmmm.py:1476-1536`,
  `dagua/layout/ops/pipelines/fmmm.py:1575-1708`).
- Cgraph out-edges: Graphviz restores each subnode `out_seq` and iterates it
  with `dtfirst`/`dtnext` (`lib/cgraph/edge.c:28-56`). Its sequence comparator
  orders by opposite node sequence, then edge sequence (`lib/cgraph/edge.c:388-402`).
  The port sorts outgoing edge ids by target node rank and edge id
  (`dagua/layout/ops/pipelines/fmmm.py:1279-1289`) and preserves that order in
  derived component tensors (`dagua/layout/ops/pipelines/fmmm.py:1800-1831`).
- Cgraph subgraphs: Graphviz iterates `g->g_seq` through
  `agfstsubg`/`agnxtsubg` (`lib/cgraph/subg.c:75-85`). The port records child
  clusters per parent and feeds that order into recursive derived-node creation
  (`dagua/layout/ops/pipelines/fmmm.py:1174-1190`).
- Connected-component node iteration now follows Cgraph subgraph semantics:
  DFS still discovers components through out-edges then in-edges, but component
  node tuples are emitted in sequence order, matching the `n_seq` dictionary
  used by the later layout passes (`dagua/layout/ops/pipelines/fmmm.py:1711-1788`).

## Label Measurement

Graphviz simple labels split on `\n`, `\l`, `\r`, use the max span width, and
accumulate span heights (`lib/common/labels.c:26-54`, `lib/common/labels.c:57-104`).
Cluster graph labels are built with default font attributes, padded with `PAD`,
and stored in `GD_border` for top/bottom label space (`lib/common/input.c:837-891`;
`lib/common/macros.h:27-29`). `finalCC` widens a cluster bbox when
`round(GD_label(rg)->dimen.x)` exceeds content width (`lib/fdpgen/layout.c:123-142`).

The port uses option (a): a minimal in-process Times-Roman 14pt metric table
copied from Graphviz's hard-coded Times metrics (`lib/common/textspan_lut.c:37-48`,
`lib/common/textspan_lut.c:833-840`). It measures cluster labels from
`cluster_labels` when supplied, otherwise from cluster names, and applies the
same `finalCC` width widening plus padded label border
(`dagua/layout/ops/pipelines/fmmm.py:930-1171`,
`dagua/layout/ops/pipelines/fmmm.py:2389-2451`).

## Per-Object Records

Cgraph records are represented by `_FdpObjectRecordStore` with graph, node, and
edge dictionaries keyed by graph/cluster name, node id, and edge id. The shared
Cgraph context attaches root and cluster graph records, original node sequence
records, and original edge tail/head records. Each derived graph also carries
records for derived graph inputs, derived node keys/kinds/members, and derived
edge real-edge backrefs (`dagua/layout/ops/pipelines/fmmm.py:120-164`,
`dagua/layout/ops/pipelines/fmmm.py:1297-1325`,
`dagua/layout/ops/pipelines/fmmm.py:1682-1707`).

## Smoke Results

Round 43 clustered reference from
`eval_output/algo_fidelity/round_43/fdp_clusters/SUMMARY.md`:

| Topology | Seed 1 | Seed 2 | Seed 3 | Mean | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| one_cluster | 0.300444180 | 0.123267211 | 0.311203930 | 0.244971774 | 0.311203930 |
| path | 0.076886609 | 0.023582772 | 0.019508183 | 0.039992521 | 0.076886609 |
| clustered | 0.269335653 | 0.195194714 | 0.196027235 | 0.220185867 | 0.269335653 |
| multi_cluster | 0.169847873 | 0.149791494 | 0.144064542 | 0.154567970 | 0.169847873 |

Round 44, `python eval_output/algo_fidelity/round_40/fdp_clusters/smoke_check.py`:

| Topology | Seed 1 | Seed 2 | Seed 3 | Mean | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| one_cluster | 0.017229474 | 0.126103185 | 0.313781231 | 0.152371296 | 0.313781231 |
| path | 0.076886609 | 0.023582772 | 0.019508183 | 0.039992521 | 0.076886609 |
| clustered | 0.218993471 | 0.185572627 | 0.210558172 | 0.205041423 | 0.218993471 |
| multi_cluster | 0.168563454 | 0.111159528 | 0.127887580 | 0.135870187 | 0.168563454 |

Supplemental unclustered star/grid smoke using the same `smoke_rmsd` helper
(not present in the R40 fdp cluster harness):

| Topology | Seed 1 | Seed 2 | Seed 3 | Mean | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| star | 0.023314397 | 0.015211787 | 0.013352584 | 0.017292922 | 0.023314397 |
| grid | 0.017093853 | 0.099428967 | 0.010244423 | 0.042255748 | 0.099428967 |

## Verdict

Hold. The Cgraph metadata port improved clustered mean RMSD from `0.220185867`
to `0.205041423` and multi-cluster mean from `0.154567970` to `0.135870187`,
but the required clustered RMSD `< 0.05` was not reached. The
`classic_fmmm_graphviz_fdp_fidelity` variant remains disabled.

## Residual

The remaining residual is still chaotic fdp recursion divergence, most visible
in seed-sensitive one-cluster and clustered cases. The new Cgraph order,
label-width, and record metadata closes bookkeeping gaps, but the Python
recursive `tLayout`/`xLayout` path still does not reproduce the exact Graphviz
cluster-internal numerical trajectory across seeds.
