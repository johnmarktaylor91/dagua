# W-E2 Sugiyama Graphviz `dot` fidelity report

## Outcome

This round closes 8 of the 18 clean-HEAD divergent rows and preserves every one of the 74
freshly reference-backed passing rows below `d_R=0.1`. Rank parity, mincross source-order parity,
and the plain x-network inventory are landed. The final positional open set is 10 rows: one
unclustered mincross/x tie (`regular_4_40`) and nine clustered rows. The previously passing
`transformer_layer` remains at `d_R=0.092933`.

The typed Graphviz x model now represents normal, virtual, slack, and border node classes,
saved-edge `ED_to_orig` lineage, cluster label/border widths, and typed separation/containment
edges. It is production-enabled only where its inventory is structurally supported. The partial
recursive cluster skeleton is retained as an instrumentable experimental path, but is not used
for production cluster output because its constraint multiset is not yet equal to Graphviz's.
All 10 cluster outputs are tensor-identical to clean HEAD. They are therefore
quality-identical to the clean-HEAD baseline, but **not** quality-identical to Graphviz according
to the definitive scorer's `quality_identical_raw` field. This distinction is recorded explicitly
in `sugiyama_causes.json`; claiming reference-quality identity would be incorrect.

## Reference and baseline prerequisites

- Full standard-corpus Graphviz reference run:
  `eval_output/benchmark_100seed_r79_sugiyama_refs/`.
- Result: 119/120 reference positions. `ba_5000` exceeded 3,600 seconds; `rgg_2000` completed on
  retry. References use the synthetic
  `graphviz_dot__for__classic_sugiyama_graphviz_fidelity` adapter.
- Clean-HEAD Dagua snapshot:
  `eval_output/benchmark_100seed_r79_sug_snapshot/`.
- Fresh reference-backed classification found 74 passing rows at clean HEAD, rather than the
  older 57-60 estimate that was based on incomplete reference coverage.
- Final all-corpus Dagua run:
  `eval_output/benchmark_100seed_r79_sugiyama_final/`, 117/120 OK; `sbm_8x100`, `rgg_2000`, and
  `ba_5000` exceeded their run timeouts.
- Required 35-seed target rerun:
  `eval_output/benchmark_100seed_r79_sugiyama_final_35seed/`, 630/630 OK.

## Per-row positional result

The before values are recomputed directly from the clean-HEAD tensor snapshot against the new
Graphviz references with free-aspect Procrustes distance. The after values are the definitive
35-seed mode-B rescore in
`eval_output/fidelity_definitive/per_combo_r79_sugiyama.jsonl`.

| Stage | Row | `d_R` before | `d_R` after | Result |
| --- | --- | ---: | ---: | --- |
| rank | `random_dag_200` | 0.955305481737 | 0.006507367330 | closed |
| mincross | `heavy_tail_weights_50` | 0.258385104442 | 0.064531412417 | closed |
| mincross | `regular_4_40` | 0.516723470305 | 0.143523511814 | open |
| mincross | `random_dag_50` | 0.638226332343 | 0.094013776817 | closed |
| mincross | `chung_lu_150` | 0.752311202934 | 0.018085590359 | closed |
| plain x | `braided_feedback_tails` | 0.143450154401 | 0.021334212250 | closed |
| plain x | `regular_3_30` | 0.161538817565 | 0.078782447255 | closed |
| plain x | `planar_60` | 0.225719359584 | 0.007901074760 | closed |
| cluster x | `clustered_longlabel_handoffs` | 0.169678973647 | 0.169678973647 | open; baseline exact |
| cluster x | `clustered_medium_5x20` | 0.147501929349 | 0.147501929349 | open; baseline exact |
| cluster x | `dependency_graph_100` | 0.831705221601 | 0.831705221601 | open; baseline exact |
| cluster x | `interleaved_cluster_crosstalk` | 0.596244626874 | 0.596244626874 | open; baseline exact |
| cluster x | `kitchen_sink_hybrid_net` | 0.846209561316 | 0.846209561316 | open; baseline exact |
| cluster x | `kitchen_sink_platform_graph` | 0.273347193515 | 0.273347193515 | open; baseline exact |
| cluster x | `moe_router_sparse` | 0.326513514388 | 0.326513514388 | open; baseline exact |
| cluster x | `multiscale_skip_cascade` | 0.362020726858 | 0.362020726858 | open; baseline exact |
| cluster x | `transformer_full_4h_2l` | 0.139441651409 | 0.139441651409 | open; baseline exact |
| cluster x | `transformer_layer` | 0.092933397401 | 0.092933397401 | remains passing; baseline exact |

Net result: 8 closures, 9 unchanged cluster divergences, 1 improved but still-open plain row,
and zero passing-row regressions.

## Rank and mincross parity evidence

- Graphviz-compatible feasible-tree construction now uses the source-order heap, recursive
  tight-subtree traversal, and entering-edge DFS order. On the 181-node connected component of
  `random_dag_200`, rank-membership differences fell from 27 nodes to zero.
- Random-DAG rank golden coverage verifies 200 generated DAGs with exact rank membership.
- Graphviz cgraph outgoing scans are by head-node sequence rather than tensor edge order. The
  corrected `_edge_processing_order` makes `heavy_tail_weights_50` pass-0 inventory exact:
  96 crossings in both implementations, with exact final per-rank order.
- Mincross now solves components locally and includes Graphviz's candidate-rank handling during
  transpose.

## Typed x-inventory structural evidence

The production model records `_GraphvizXNodeClass` values `NORMAL`, `VIRTUAL`, `SLACK`, and
`BORDER`; typed edge kinds distinguish rank separation, saved-edge slack, cluster containment,
and keepout/separation constraints. Expanded fast edges retain original-edge IDs so backward
edges are scanned at their original cgraph tail while their virtual chain remains rank-oriented.

For `braided_feedback_tails`, the instrumented plain inventory has exact structural parity:
5 left/right separation constraints, 19 slack nodes, 38 incident slack edges, and an identical
`(weight,minlen)` multiset. Correct original-tail scanning closes its positional distance to
0.021334.

The cluster experiment does **not** pass the structural gate. Exact final auxiliary-graph deltas
for the three fully instrumented probes are:

| Probe | Graphviz nodes/edges | Typed nodes/edges | Graphviz-only `(weight,minlen)` | Dagua-only `(weight,minlen)` |
| --- | ---: | ---: | --- | --- |
| `clustered_longlabel_handoffs` | 35 / 53 | 33 / 49 | `(8,0)x4`, `(63,128)x2`, `(104,128)x1`, `(146,0)x2` | `(1,128)x3`, `(69,0)x2`, `(144,0)x2`, `(315,0)x1` |
| `interleaved_cluster_crosstalk` | 45 / 104 | 39 / 96 | `(8,0)x2`, `(54,128)x1`, `(55,128)x1`, `(57,128)x1`, `(70,0)x5` | `(1,128)x5`, `(69,0)x5`, `(139,0)x3`, `(156,0)x1` |
| `kitchen_sink_platform_graph` | 51 / 113 | 49 / 106 | `(8,0)x8`, `(48,128)x1`, `(56,128)x1`, `(63,128)x1`, `(80,0)x6` | `(1,128)x5`, `(61,0)x3`, `(79,0)x6`, `(189,0)x1` |

The residuals identify missing class-2 recursive cluster-skeleton fast edges, recursive leader
lineage, and label/border minlen reuse. `clustered_medium_5x20` additionally proves the partial
model is cyclic before simplex. Because structural multiset parity is the pre-solve gate, the
partial cluster model is not allowed to replace the regression-clean legacy path.

Dense unclustered graphs with more than two original edges per node also retain the legacy
fast-edge compaction until that inventory is parity-proven. This structural guard prevents the
typed model from regressing `densenet_block`; its final `d_R=0.062078`, below the 0.1 gate.

## Regression proof

- Final rerun: 74/74 clean-HEAD passing rows remain below `d_R=0.1` against the new full reference
  corpus. Maximum passing distance is `transformer_layer`, `d_R=0.092933397401`.
- Tensor comparison: 37/74 final tensors are exact to clean HEAD; 37 changed but remain below the
  fidelity threshold.
- Cluster-specific rerun: 10/10 tensors are exact to clean HEAD.
- Required focused selector: 55 passed, 425 deselected, 3 warnings.
- Final fast suite: 165 passed and 1 xfailed before stopping at the unrelated
  `test_graphviz_base_forwards_timeout`; its monkeypatched helper does not accept the concurrently
  added `graph_attributes` keyword. The failure is outside the touched Sugiyama implementation.
- Ruff on all touched source/tests: passed.
- Mypy strict CLI check: passed.
- Repository-wide Ruff is blocked only by 18 pre-existing errors in unrelated untracked research
  scripts; none were modified.

## Milestones landed on `develop`

1. `45ded57 fix(sugiyama): match dot rank and mincross traversal`
2. `6b5c2d0 fix(sugiyama): model dot x-coordinate inventory`
3. `674830a fix(sugiyama): preserve legacy x fidelity outside typed inventory`
4. `4a69a67 fix(sugiyama): guard dense x inventory compaction`

No protected `fmmm.py`, `sfdp.py`, `spectral.py`, `variants.py`, or `causes_r78.json` file was
modified.

## Final open set

`regular_4_40` remains a plain mincross/x tie at `d_R=0.143524`. The nine clustered rows remain
open for the exact recursive skeleton deltas above. The next correctness step is not coefficient
tuning: it is completing recursive cluster leader creation and class-2 fast-edge compaction until
node count, edge count, and the final `(weight,minlen)` multiset are exact before simplex.
