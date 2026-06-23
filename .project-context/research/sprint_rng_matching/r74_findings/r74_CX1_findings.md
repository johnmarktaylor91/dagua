# R74 CX1 Sugiyama Divergence Findings

Scope: read-only research for 231 `classic_sugiyama*` final-rung-4 combos. Data source: `eval_output/fidelity_definitive_r73/per_combo.json`, filtered to `engine.startswith("classic_sugiyama") and final_rung == "4"`.

## Executive summary

The 231 combos are all mode B deterministic divergences. I found no plausible floating-point chaos floor: the pipeline is dominated by deterministic LP/simplex/order/compaction choices, and every divergent row has `mode == "B"`.

Two large root causes are confirmed:

1. Dagua's igraph fidelity layer LP uses a zero objective and HiGHS, while igraph's GLPK LP uses a nonzero objective `out_strength - in_strength` after removing Eades feedback edges. This explains many igraph-family deterministic-but-different layouts, especially where multiple feasible layerings exist. It should improve fidelity and may improve/canonicalize quality, but the current D/R quality split says most igraph-family divergences are not Dagua-worse by stress/crossings.
2. Dagua's graphviz fidelity path uses dot-style rank simplex and a partial dot mincross port, but then assigns x coordinates with Brandes-Koepf. Graphviz `dot` runs a second network simplex on an auxiliary left-right graph, using node separation constraints, flat-edge constraints, virtual edge pair constraints, and cluster constraints. This is the highest-value quality fix: 59/65 graphviz-fidelity divergent rows are Dagua-worse on stress/crossing thresholds.

## Data split

Filtered count: 231.

By engine:

- `classic_sugiyama_graphviz_fidelity`: 65
- `classic_sugiyama_wide`: 36
- `classic_sugiyama_default`: 34
- `classic_sugiyama_tight`: 34
- `classic_sugiyama_passes4`: 31
- `classic_sugiyama_passes48`: 31

All 231 have `mode == "B"`.

Quality split using `cross_D_mean/cross_R_mean` and `stress_D_mean/stress_R_mean` with conservative thresholds (`D_cross > R_cross + max(1, 0.1R)` or `D_stress > 1.1R`):

- Dagua worse quality: 80 rows.
- Near/equal metrics: 33 rows.
- Dagua equal/better on these two metrics despite divergent verdict: 118 rows.

By engine:

- `classic_sugiyama_graphviz_fidelity`: 59 worse, 6 near/equal. Median stress ratio D/R 2.25; p90 8.85; max 68.63. This is the real quality problem.
- `classic_sugiyama_default`: 3 worse, 6 near/equal, 25 equal/better. Median stress ratio 0.84; median crossing diff -455.
- `classic_sugiyama_passes4`: 5 worse, 4 near/equal, 22 equal/better.
- `classic_sugiyama_passes48`: 4 worse, 5 near/equal, 22 equal/better.
- `classic_sugiyama_tight`: 3 worse, 6 near/equal, 25 equal/better.
- `classic_sugiyama_wide`: 4 worse, 6 near/equal, 26 equal/better.

Interpretation: graphviz-fidelity coordinate/cluster/constraint mismatches drive quality loss; igraph-family mismatches are mostly deterministic different-but-not-worse under stress/crossing.

## Verification questions

### 1. igraph GLPK objective: confirmed nonzero; Dagua zero objective

igraph source evidence:

- `/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:563-565`: GLPK path is entered only under `HAVE_GLPK` and `igraph_is_directed(graph) && no_of_nodes <= 1000`.
- `sugiyama.c:584-586`: igraph first computes and sorts approximate Eades feedback edges.
- `sugiyama.c:589-600`: computes in/out strengths and subtracts feedback-edge weights from source/target degree accumulators. Note line 591 calls `igraph_strength(... IGRAPH_IN ...)` for `outdegs`; by the subsequent line 611 subtraction and objective line 615, the effective source is the released C, not the variable name.
- `sugiyama.c:608-615`: sets `GLP_MIN`, integer columns, lower bound 0, then `glp_set_obj_coef(ip, i, VECTOR(outdegs)[i - 1])` after `igraph_vector_sub(&outdegs, &indegs)` at line 611.
- `sugiyama.c:638-645`: constraints use original or reversed feedback-edge orientation (`x_to - x_from >= 1` or reversed equivalent).
- `sugiyama.c:649-655`: solves GLPK simplex and floors column primals.

Dagua evidence:

- `dagua/layout/ops/sugiyama.py:326-350`: `_igraph_glpk_layer_assignments` claims the objective is effectively zero.
- `sugiyama.py:378-384`: calls `linprog([0.0] * num_nodes, ..., method="highs")`.
- `sugiyama.py:354-373`: builds the same style of inequality constraints but gives the solver no objective beyond feasibility.

Verdict: verified. Dagua is not source-faithful here. The objective should be `out_strength_minus_in_strength` after removing/reversing Eades feedback edges, not all zeros. Because the current objective is degenerate, HiGHS can choose a different feasible layer vector than GLPK simplex. This is deterministic and not FP chaos.

### 2. igraph gating: confirmed; Dagua runs HiGHS for all igraph-mode graphs with SciPy

igraph source evidence:

- `sugiyama.c:563-565`: LP only for directed graphs with `no_of_nodes <= 1000` when GLPK is compiled in.
- `sugiyama.c:661-665`: directed graphs outside the GLPK gate use `igraph_i_feedback_arc_set_eades(... membership)`; undirected graphs use `igraph_i_feedback_arc_set_undirected(... membership)`.
- `sugiyama.c:666-670`: without GLPK, directed graphs use Eades fallback.

Dagua evidence:

- `dagua/layout/ops/sugiyama.py:2250-2263`: for `fidelity_mode == "igraph"`, Dagua always calls `_igraph_glpk_layer_assignments` on non-loop original edges.
- `sugiyama.py:355-358`: only missing SciPy triggers Eades fallback.
- `sugiyama.py:378-384`: if SciPy is present, HiGHS runs regardless of directedness or node count. The pipeline's tensor input does not preserve an explicit `igraph_is_directed` graph flag.
- `dagua/eval/variants.py:920-985`: all igraph-targeted sugiyama variants pass `fidelity_mode: "igraph"` to the same path.

Verdict: verified. For graph sizes above 1000 and for undirected igraph reference cases, Dagua's LP path is simply the wrong algorithm. This can flip all five igraph-family variants for affected graphs.

### 3. Graphviz dot x-coordinate assignment: confirmed network simplex on auxiliary LR graph; Dagua uses Brandes-Koepf

Graphviz source evidence:

- `/home/jtaylor/projects/_references/graphviz/lib/dotgen/position.c:127-148`: `dot_position` sets y coords, creates aux edges, calls `rank(g, 2, nsiter2(g))` with LR balance, then copies `ND_rank` to x in `set_xcoords`.
- `position.c:219-324`: `make_LR_constraints` creates same-rank left-to-right constraints, including node separation, self-edge width, flat-edge label constraints, and flat-edge endpoint constraints.
- `position.c:326-352`: `make_edge_pairs` creates a slack virtual node per original edge and two aux edges using port x deltas and `ED_weight(e)`.
- `position.c:525-531`: `create_aux_edges` runs LR constraints, edge pairs, cluster positioning constraints, and compression before the x simplex.
- `position.c:569-583`: `set_xcoords` assigns `ND_coord(v).x = ND_rank(v)` after the auxiliary rank/simplex solve.
- `/home/jtaylor/projects/_references/graphviz/lib/common/ns.c:940-951`: `rank2` is the network simplex ranker over `ED_minlen` constraints.
- `ns.c:989-1014`: it pivots with `leave_edge`, `enter_edge`, `update`, then applies `LR_balance` for balance code 2.
- `ns.c:778-793`: `LR_balance` reranks zero-cut tree edges by half slack, exactly the left-right balancing step.

Dagua evidence:

- `dagua/layout/ops/sugiyama.py:2596-2633`: `_CoordinateAssignment.apply` says it runs Brandes-Kopf coordinate assignment and calls `_coordinate_assignment`.
- `sugiyama.py:1399-1462`: `_coordinate_assignment` sets y by layer index and delegates x to `_brandes_koepf_x_positions`.
- `sugiyama.py:1465-1555`: `_brandes_koepf_x_positions` runs four BK orientations and median-balances them.
- `dagua/layout/ops/pipelines/sugiyama.py:60-62`: existing docstring already lists Graphviz x-position simplex as a known architectural residual.

Verdict: verified. Dagua has Graphviz-like rank assignment and a mincross helper, but graphviz-fidelity x is not dot's x. This is the top quality/fidelity fix.

About the requested omega weights: I did not find literal `omega` or a 1/2/8 table in `position.c`, `mincross.c`, or `rank.c`. In this checked Graphviz tree, dot x-position edge-pair constraints use `ED_weight(e)` directly at `position.c:345-346`, cluster compaction uses 128 at `position.c:361-364`, compression uses 1000 at `position.c:521-523`, and virtual-node crossing weights use a 1/2/4 table in `mincross.c:1703-1742` (`C_EE=1`, `C_VS=2`, `C_SS=2`, `C_VV=4`). So I verify the auxiliary LR network simplex part, but I refute the exact 1/2/8 omega claim for the source tree I read.

### 4. Mincross completeness gaps: mostly confirmed

Graphviz source evidence:

- `mincross.c:1022-1044`: `init_mincross` runs `class2`, `decompose`, `allocate_ranks`, `ordered_edges`, and global rank setup.
- `mincross.c:1208-1286`: `build_ranks` creates initial orders via BFS/queue traversal from source/sink sides, tries in/out-edge passes, handles clusters via `install_cluster`, and applies an initial transpose.
- `mincross.c:704-746`: the main pass loop tries two initial orderings, tracks best crossings, restores best, and runs final transpose.
- `mincross.c:1046-1131`: flat-edge cycle breaking uses a per-rank flat adjacency matrix and reverses/deletes flat edges when necessary.
- `mincross.c:1310-1408`: flat-edge reordering uses reverse topological sorting and restores intended order; nonconstraint flats are made left-to-right.
- `mincross.c:1410-1488`: `reorder` respects `left2right` constraints and cluster skip behavior, then transposes.
- `mincross.c:1490-1546`: crossing counts include port-order local crossings and `ED_xpenalty` weights.
- `mincross.c:1580-1677`: `medians` uses port-aware `VAL(node, port) = MC_SCALE * ND_order(node) + port.order` and has `flat_mval` for nodes with only flat edges.
- `mincross.c:1703-1742`: virtual edge weights are adjusted by endpoint classes.

Dagua evidence:

- `dagua/layout/ops/_dot_mincross.py:14-91`: helper runs simplified median/transpose on rank lists and adjacent edges.
- `_dot_mincross.py:45-54`: only builds incoming/outgoing adjacency from normalized adjacent edges; no cluster, flat-edge matrix, ordered-edge, or port structures are represented.
- `_dot_mincross.py:64-89`: it does a single initial order, tracks best by crossing count, and final transposes.
- `_dot_mincross.py:249-296`: median values are just scaled neighbor order; no port offsets.
- `_dot_mincross.py:341-385` and `454-498`: crossing counts ignore edge x-penalty, local port crossings, flat-edge constraints, and cluster left-to-right constraints.
- `dagua/layout/ops/sugiyama.py:979-990`: initial Dagua ranks are `sorted(layer)` before calling graphviz_mincross, not Graphviz's `build_ranks` source/sink-seeded orders.

Verdict: confirmed partial port, not complete. The helper captures median/transpose skeleton but misses the ordering substrate that makes dot deterministic and good on clusters, flat edges, ports, and long-edge virtuals.

## Fix avenues, ROI ordered

### A. Port dot x-coordinate assignment simplex for graphviz fidelity

Root cause: Graphviz uses auxiliary LR network simplex for x; Dagua uses BK compaction. Evidence: Graphviz `position.c:127-148`, `219-352`, `525-583`, `ns.c:940-1014`; Dagua `sugiyama.py:1399-1555`.

Fix sketch:

- Add a graphviz x-position mode in `_CoordinateAssignment` when `fidelity_mode in {"dot", "graphviz", "graphviz_dot"}`.
- Reuse the existing network simplex port in `dagua/layout/ops/pipelines/dot_rank.py` as the generic solver, but feed it LR constraints instead of rank constraints.
- Build aux constraints equivalent to `make_LR_constraints` and `make_edge_pairs`: consecutive same-rank node separation, flat-edge label/endpoint constraints where metadata is available, virtual edge pair constraints for original edges, and at least basic cluster keepout if cluster metadata is exposed in benchmark inputs.
- Preserve the current BK path for igraph fidelity; igraph itself uses BK-style four-alignment compaction in `sugiyama.c:858-1030`.

Impact estimate: direct target 65 graphviz-fidelity rows, especially the 59 Dagua-worse rows. Likely flips 45-60 to 3Q or better if rank/mincross are already close; exact bit/layout equivalence needs mincross/cluster work too. Best achievable tier after x-only: mostly 3Q, some 2/3 for simple non-cluster graphs. Effort: high (3-5 days for a useful non-cluster x-simplex, 1-2 weeks with flat labels/clusters/ports). Confidence: high.

### B. Fix igraph GLPK objective and GLPK/Eades gating

Root cause: Dagua uses zero-objective HiGHS unconditionally; igraph uses nonzero GLPK objective only for directed graphs up to 1000 nodes, otherwise Eades/undirected fallback. Evidence: igraph `sugiyama.c:563-665`; Dagua `sugiyama.py:326-389`, `2250-2263`.

Fix sketch:

- In `_igraph_glpk_layer_assignments`, compute feedback edges with existing `_igraph_eades_feedback_edges`, then compute objective coefficients source-faithfully as igraph does after removing feedback-edge contributions. Use edge weights if provided; current call drops `original_weights`, so pass weights through.
- Replace `[0.0] * num_nodes` with those coefficients.
- Add a gate matching igraph: if directedness is known false, use an undirected feedback-layer fallback; if node count > 1000, use `_igraph_eades_layer_assignments` for directed graphs. The current tensor-only API lacks directedness, so the conservative benchmark-path fix can at least gate `num_nodes > 1000` and document that reference variants are directed unless graph metadata says otherwise.
- If GLPK pivot fidelity matters, HiGHS may still differ on degenerate optima. A lexicographic secondary objective can canonicalize, but for igraph fidelity the primary source-faithful move is nonzero objective; a tiny secondary term should be opt-in and benchmark-verified.

Impact estimate: affects all five igraph-targeted variants. Of 166 igraph-family divergent rows, only 19 are Dagua-worse by my thresholds; most are Dagua equal/better/different. Nonzero objective plus >1000 Eades gate could flip perhaps 60-120 igraph-family rows from rung 4 to 3Q/2-like, depending how many divergences are layer degeneracy. Best achievable tier: 2/3 for many small directed graphs if HiGHS optimum matches GLPK enough; 3Q for remaining solver-pivot mismatches. Effort: medium (1-2 days with tests). Confidence: high for root cause, medium for exact flip count.

### C. Complete Graphviz mincross substrate

Root cause: Dagua's dot mincross helper is a median/transpose skeleton, not full dot mincross. Evidence: Graphviz `mincross.c:1022-1044`, `1208-1286`, `704-746`, `1046-1131`, `1310-1408`, `1410-1488`, `1490-1546`, `1580-1677`; Dagua `_dot_mincross.py:14-91`, `249-296`, `341-385`, `454-498`, and `sugiyama.py:979-990`.

Fix sketch:

- Port `build_ranks` initial ordering instead of `sorted(layer)`, including the two initial pass choices and best-cross restoration.
- Represent flat-edge constraints and `left2right` matrix, even if cluster support starts as no-op for graphs without cluster metadata.
- Include port-aware median values and local crossing penalties where port metadata exists; default ports still reduce to current behavior.
- Incorporate virtual edge endpoint class weighting for crossing counts.

Impact estimate: after x-simplex, this is needed for dot fidelity beyond 3Q. Alone, likely improves crossings in 6-15 graphviz rows and may reduce divergence on graph families with flats/ports/clusters. Best achievable tier: 3Q for more graphviz rows; 2/3 only with x-simplex and cluster/rank metadata. Effort: high (1 week+). Confidence: high.

### D. Cluster metadata and Graphviz cluster constraints

Root cause: Graphviz's rank, mincross, and x-position paths treat clusters structurally; Dagua's tensor pipeline mostly lacks this metadata on the benchmark path. Evidence: rank cluster collapse `rank.c:320-326`, `446-457`, `503-519`; mincross cluster fill/order `mincross.c:965-1038`, `1208-1286`, `1410-1437`; position cluster constraints `position.c:354-499`, `1040-1099`.

Fix sketch:

- Plumb cluster/subgraph metadata into `LayoutProblem.graph_data` or extras for classic pipelines.
- Implement cluster collapse/expand for dot rank/mincross and cluster LR constraints for x-position.
- Keep basic non-cluster path separate to avoid destabilizing simple graph cases.

Impact estimate: a subset of the graphviz-fidelity D-worse rows are cluster/label-heavy (`nested_cluster_label_stack`, `clustered_longlabel_handoffs`, `cluster_member_style_stress`, etc.) and show extreme stress ratios. Could flip 10-25 rows after x-simplex/mincross, but difficult without metadata. Best achievable tier: 3Q/2 for cluster graphs if fully ported. Effort: very high (1-2+ weeks). Confidence: medium-high.

### E. Deterministic secondary/lexicographic LP objective for better canonical layering

Hypothesis: after implementing igraph's nonzero primary objective, remaining GLPK/HiGHS degeneracy can be broken deterministically with a lexicographic secondary objective that also favors quality, e.g. minimize total weighted edge span, then layer width variance, then node id order. However, for igraph fidelity, any secondary objective not in the reference risks reducing source fidelity. For Dagua-native quality it may be useful.

Source constraints:

- igraph's primary objective is already a layer-quality/canonical objective (`sugiyama.c:608-615`) and should be implemented first.
- Graphviz rank simplex already optimizes edge length through `ED_minlen`/`ED_weight`; Dagua has a port in `dot_rank.py:104-190`, `414-450`, `494-522`.

Recommended version:

- For `fidelity_mode="igraph"`, use exact primary objective and no invented secondary until measured.
- For non-fidelity/default mode, consider deterministic tiny lexicographic terms only after proving they improve both stress and crossings on a held-out benchmark and do not hurt reference-target variants.

Impact estimate: maybe flips 20-50 igraph-family degenerate rows to 3Q if the secondary approximates GLPK pivot choices; quality improvement uncertain because current igraph-family D layouts are often equal/better by stress/crossings. Best tier: 3Q, not bit/layout exact. Effort: low-medium after objective fix. Confidence: medium-low for fidelity, medium for native quality.

## FP-chaos floor

I see no FP-chaos floor here. All 231 are mode B deterministic; the divergences are caused by discrete choices: zero-vs-nonzero LP objective, wrong solver/gate, BK vs network-simplex x assignment, and incomplete deterministic mincross state. A floor claim would require perturbation experiments, but the source mismatches above are sufficient root causes and should be fixed first.

## Tests/verification recommended for implementation

- Benchmark-path verification only, per shared guardrail. Do not validate only direct pipeline calls.
- Add focused fixtures where igraph's nonzero objective changes the LP optimum from a zero-objective feasible point.
- Add >1000-node directed fixture to verify Eades fallback in igraph mode.
- Add graphviz-dot fixtures with same rank/order but different BK vs LR-simplex x; assert stress improves and coordinates move toward dot.
- Add mincross fixtures for flat edges and port order once those structures are represented.
