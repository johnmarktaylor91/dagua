# r75 sugiyama findings - codex

## 1. Executive summary

- Target set: 129 divergent rows: 58 graphviz-dot family, 71 igraph-family/default.
- CONFIRMED: `binary_tree::classic_sugiyama_graphviz_fidelity` first differs at x-coordinate assignment, not ranking or ordering.
- Graphviz dot does not use Brandes-Koepf for x; it builds an auxiliary constraint graph and runs network simplex with LR balance.
- Dagua's graphviz path has network-simplex ranking and partial mincross, but still uses Dagua dummy expansion and BK x assignment.
- Landing a faithful Graphviz x port plus fuller mincross should address most graphviz-family rows, at most 58/129 of this bucket.
- It cannot fix the 71 igraph-family rows; those need a separate igraph exactness pass.
- Igraph residuals look like exact implementation mismatches, not a different algorithm family: GLPK/LP solver tie behavior, type-1 conflict marking, qsort/tie behavior, and component/dummy indexing.
- Crossings-only rows are not uniformly ordering-stage failures; one benchmark-path probe had identical rank/order and only scale/frame differences.
- Do not blanket-enable graphviz ports for igraph variants or clustered/label-heavy graphviz variants without feature gates.

## 2. Findings ranked by expected combo-count impact

### F1. CONFIRMED: Graphviz-family stress failures first diverge at x-coordinate assignment

Impact: high. Explains the common graphviz pattern where crossings match but stress/NP fail: 23 graphviz rows fail only stress, 16 fail stress+NP, and several small DAGs have matching crossings.

Evidence command:

```bash
MPLCONFIGDIR=/tmp/mpl-r75 python3 scripts/run_benchmark.py \
  --workers 1 --timeout 60 --seeds 1 --seed-start 42 \
  --graphs binary_tree \
  --engines classic_sugiyama_graphviz_fidelity,graphviz_dot__for__classic_sugiyama_graphviz_fidelity \
  --variants --output-dir /tmp/r75_sugiyama_probe

python3 - <<'PY'
import torch
from pathlib import Path
for p in sorted(Path('/tmp/r75_sugiyama_probe/positions').glob('binary_tree__*.pt')):
    pos=torch.load(p,map_location='cpu')
    ys=sorted(set(round(float(v),6) for v in pos[:,1].tolist()))
    ranks=[min(range(len(ys)), key=lambda i: abs(float(pos[n,1])-ys[i])) for n in range(pos.shape[0])]
    orders=[]
    for r in range(len(ys)):
        nodes=[i for i,x in enumerate(ranks) if x==r]
        orders.append(tuple(sorted(nodes,key=lambda n:(float(pos[n,0]),n))))
    print(p.name)
    print('xspan', round(float(pos[:,0].min()),3), round(float(pos[:,0].max()),3), 'yvals', ys)
    print('rank sizes', [len(o) for o in orders])
    print('orders', orders)
PY
```

Output evidence:

```text
[benchmark] Done: 2 total, 2 ok, 0 skipped, 0 errors, 0 timeouts

binary_tree__classic_sugiyama_graphviz_fidelity__seed42.pt
xspan -101.25 101.25 yvals [0.0, 1.0, 2.0, 3.0]
rank sizes [1, 2, 4, 4]
orders [(0,), (1, 2), (3, 4, 5, 6), (7, 8, 9, 10)]

binary_tree__graphviz_dot__for__classic_sugiyama_graphviz_fidelity.pt
xspan 27.0 531.0 yvals [-342.0, -234.0, -126.0, -18.0]
rank sizes [1, 2, 4, 4]
orders [(0,), (1, 2), (3, 4, 5, 6), (7, 8, 9, 10)]
```

Code evidence:

- Dagua graphviz fidelity enables graphviz rank and mincross, then still calls `_CoordinateAssignment`: `dagua/layout/ops/pipelines/sugiyama.py:102-124`.
- Dagua coordinate assignment is BK: `dagua/layout/ops/sugiyama.py:1687-1738`, with four passes at `dagua/layout/ops/sugiyama.py:1780-1831`.
- Graphviz `dot_position()` calls `create_aux_edges()`, `rank(g, 2, nsiter2(g))`, then `set_xcoords()`: `/home/jtaylor/projects/_references/graphviz/lib/dotgen/position.c:127-148`.
- Graphviz rank balance 2 is LR balance in common network simplex: `/home/jtaylor/projects/_references/graphviz/lib/common/ns.c:1007-1015`.
- Graphviz copies the auxiliary simplex rank into x coordinates: `/home/jtaylor/projects/_references/graphviz/lib/dotgen/position.c:569-584`.

Fix sketch:

- Add graphviz-dot x-coordinate ops, not another BK tuning knob.
- Reuse or generalize Dagua's `dot_rank.py` network simplex backend for horizontal constraints.
- Run this only for `fidelity_mode in {"dot", "graphviz", "graphviz_dot"}`.

Expected impact:

- Should fix most graphviz stress-only rows such as `binary_tree`, `bipartite_4_3_4`, `center_port_backedge_hub`, and other rows where crossings already match.
- Overall bucket ceiling for this fix alone is 58/129 rows because igraph-family rows use a different reference.

Risk:

- High if applied outside graphviz variants. BK is correct for igraph-family Sugiyama.
- Medium inside graphviz labels/clusters if the port skips flat-edge and cluster constraints; see F3.

### F2. CONFIRMED: Graphviz x port spec

Impact: high for graphviz-family stress/NP failures.

Reference behavior:

- `dot_position()` first calls `set_ycoords()`, optional concentration/flat handling, `create_aux_edges()`, horizontal network simplex, then `set_xcoords()`: `/home/jtaylor/projects/_references/graphviz/lib/dotgen/position.c:127-153`.
- `create_aux_edges()` is the x auxiliary graph constructor: allocate saved in/out lists, make left-to-right rank constraints, make edge-pair slack nodes, add cluster containment/separation, and optional compression: `/home/jtaylor/projects/_references/graphviz/lib/dotgen/position.c:525-532`.
- Adjacent same-rank node spacing is encoded as aux edges with `len = ND_rw(u) + ND_lw(v) + nodesep`, weight 0: `/home/jtaylor/projects/_references/graphviz/lib/dotgen/position.c:238-267`.
- Flat-edge labels and flat endpoints add extra horizontal constraints and can strengthen existing aux edges: `/home/jtaylor/projects/_references/graphviz/lib/dotgen/position.c:269-321`.
- Every original edge produces a slack node plus two aux edges using head/tail port x offsets and original edge weight: `/home/jtaylor/projects/_references/graphviz/lib/dotgen/position.c:326-350`.
- Clusters add containment/separation constraints through left/right virtual boundary nodes; cluster compaction edge weight is 128: `/home/jtaylor/projects/_references/graphviz/lib/dotgen/position.c:354-499`.
- X is assigned from the post-LR-simplex `ND_rank`: `/home/jtaylor/projects/_references/graphviz/lib/dotgen/position.c:569-584`.

Dagua current behavior:

- The graphviz-fidelity pipeline records only original-node ranks from `graphviz_rank_assignment()`: `dagua/layout/ops/sugiyama.py:323-386`.
- Dagua then expands long edges with generic zero-size dummy nodes: `dagua/layout/ops/sugiyama.py:997-1083`.
- Dagua assigns x by BK, not aux network simplex: `dagua/layout/ops/sugiyama.py:1727-1738`.

Port spec:

1. Add `GraphvizDotPrepareXConstraints` after mincross and before coordinate assignment.
2. Build aux nodes for every original expanded node plus slack nodes per original/virtual edge.
3. Add same-rank LR constraints from ordered ranks:
   - tail = rank[i][j], head = rank[i][j+1]
   - minlen = `round(right_width(left) + left_width(right) + nodesep)`
   - weight = 0
4. Add edge-pair slack constraints:
   - slack -> tail with minlen `max(port_dx, 0) + 1`
   - slack -> head with minlen `max(-port_dx, 0) + 1`
   - weight = Graphviz edge weight
   - for now `port_dx = 0` unless ports are available from adapter DOT.
5. Run the existing network simplex with `balance=2` equivalent. Dagua `dot_rank.py` currently exposes `balance: bool`, so it needs an enum or new helper for LR balance.
6. Set x from aux ranks, not from BK. Restore original y rank values.
7. Stage features:
   - Stage A: no clusters, no labels, no ports.
   - Stage B: flat edge constraints.
   - Stage C: edge labels.
   - Stage D: cluster boundary constraints.

Estimated port size:

- 250-400 LOC for the no-cluster/no-label aux graph and LR simplex reuse.
- 150-250 LOC more for flat-edge/label constraints.
- 250-500 LOC more for cluster containment/separation if required for target rows.

Verification ladder:

- `binary_tree`: rank/order same, x must match after affine translation/axis sign.
- `bipartite_4_3_4`: rank/order/crossing same, stress should collapse.
- `dense_pair_50` and `weighted_karate_34`: exercise edge weights and crossing-sensitive layouts.
- `edge_label_braid`, `clustered_longlabel_handoffs`, `nested_cluster_label_stack`: only after flat/label/cluster stages land.

### F3. CONFIRMED: Dagua graphviz mincross is a partial port

Impact: high for graphviz crossing failures, especially large crossing-heavy graphs.

Reference code:

- Graphviz mincross initializes class2 virtual chains and components before building ranks: `/home/jtaylor/projects/_references/graphviz/lib/dotgen/mincross.c:1022-1040`.
- `class2()` creates virtual chains, merges multi-edges, handles self/flat/backward edges, and increments virtual widths: `/home/jtaylor/projects/_references/graphviz/lib/dotgen/class2.c:68-96`, `/home/jtaylor/projects/_references/graphviz/lib/dotgen/class2.c:130-148`, `/home/jtaylor/projects/_references/graphviz/lib/dotgen/class2.c:155-293`.
- `mincross()` has passes 0, 1, and 2; passes 0/1 build ranks with different initial orders, then pass 2 runs full `MaxIter`: `/home/jtaylor/projects/_references/graphviz/lib/dotgen/mincross.c:690-748`.
- Constants: `MinQuit` with convergence `.995`: `/home/jtaylor/projects/_references/graphviz/lib/dotgen/mincross.c:157-160`, loop use at `/home/jtaylor/projects/_references/graphviz/lib/dotgen/mincross.c:721-738`.
- Local transpose honors `left2right()` flat/cluster constraints and port tie-breaks through `in_cross()`/`out_cross()`: `/home/jtaylor/projects/_references/graphviz/lib/dotgen/mincross.c:557-579`, `/home/jtaylor/projects/_references/graphviz/lib/dotgen/mincross.c:581-617`, `/home/jtaylor/projects/_references/graphviz/lib/dotgen/mincross.c:632-688`.

Dagua code:

- Dagua helper starts from caller-supplied ranks and adjacent edges only: `dagua/layout/ops/_dot_mincross.py:14-91`.
- It filters to adjacent-rank edges and ignores non-adjacent edges instead of owning Graphviz class2 expansion: `dagua/layout/ops/_dot_mincross.py:94-134`.
- It has median/transpose constants but one pass schedule, no pass 0/1 rank rebuild, no cluster/flat `left2right`, and no port x tie-breaks: `dagua/layout/ops/_dot_mincross.py:64-90`, `dagua/layout/ops/_dot_mincross.py:204-246`, `dagua/layout/ops/_dot_mincross.py:340-383`.

Concrete behavioral deltas:

- Initial order: Graphviz rebuilds ranks through `build_ranks(g, pass)` for pass 0 and 1; Dagua starts from `ordered_layers = [sorted(layer)]`: `dagua/layout/ops/sugiyama.py:1255`.
- Virtual width: Graphviz increments widths on merged virtual chains; Dagua dummy nodes are `[0.0, 0.0]`: `dagua/layout/ops/sugiyama.py:1042-1047`.
- Crossing weights: Graphviz multiplies `ED_xpenalty` values in crossing counts; Dagua counts unit crossings: `dagua/layout/ops/_dot_mincross.py:386-451`.
- Constraints: Graphviz blocks swaps through `left2right()`; Dagua transpose only compares before/after crossing counts.
- Iteration schedule: Graphviz uses pass 0/1 with max 4 and pass 2 with `MaxIter`; Dagua uses one loop over `iterations`.

Fix sketch:

- Move class2-like virtual-chain construction into the graphviz variant before mincross.
- Preserve per-edge `xpenalty`, count, and weight fields in an internal record.
- Implement pass 0/1/2 control flow before tuning transpose.
- Add `left2right` flat-order constraints for same-rank edges before clusters.

Expected impact:

- Should explain graphviz rows with crossing failures: `dense_pair_50`, `hub_skip_superfan`, `heavy_tail_weights_50`, `weighted_karate_34`, and the large-graph tail such as `ba_500`.
- Does not explain graphviz rows where rank/order already match and only x differs; those require F2.

Risk:

- High for already bit-exact graphviz cases if class2 expansion is enabled without exact Graphviz input-order and component handling.
- Gate by `fidelity_mode="graphviz"` and verify against existing bit-exact rows after every substage.

### F4. CONFIRMED: Igraph-family rows are not fixed by graphviz ports

Impact: high as a boundary condition: 71/129 rows.

Code evidence:

- Variant registry maps default/tight/wide/pass variants to `igraph_sugiyama` and passes matched `maxiter`, `vgap`, and `hgap`: `dagua/eval/variants.py:920-989`.
- Igraph adapter passes variant kwargs to `ig.layout("sugiyama", **kwargs)` and scales coordinates by 50: `dagua/eval/competitors/igraph_competitor.py:170-188`, `dagua/eval/competitors/igraph_competitor.py:79-99`.
- Dagua igraph mode disables node-size spacing and centering by default: `dagua/layout/ops/pipelines/sugiyama.py:218-255`.
- Dagua packs weak components independently for igraph mode: `dagua/layout/ops/pipelines/sugiyama.py:235-253`, implementation at `dagua/layout/ops/pipelines/sugiyama.py:368-457`.

Benchmark-path probe for a crossings-only row:

```bash
MPLCONFIGDIR=/tmp/mpl-r75 python3 scripts/run_benchmark.py \
  --workers 1 --timeout 60 --seeds 1 --seed-start 42 --seed-refs igraph_sugiyama \
  --graphs multiscale_skip_cascade \
  --engines classic_sugiyama_default,igraph_sugiyama__for__classic_sugiyama_default \
  --variants --output-dir /tmp/r75_sugiyama_igraph_probe
```

Output evidence:

```text
[benchmark] Done: 2 total, 2 ok, 0 skipped, 0 errors, 0 timeouts

multiscale_skip_cascade__classic_sugiyama_default__seed42.pt
rank sizes [1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1]
orders [(0,), (1,), (2,), (3,), (4,), (5,), (6, 9), (7, 10), (11, 8), (12,), (13,), (14,)]

multiscale_skip_cascade__igraph_sugiyama__for__classic_sugiyama_default__seed42.pt
rank sizes [1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1]
orders [(0,), (1,), (2,), (3,), (4,), (5,), (6, 9), (7, 10), (11, 8), (12,), (13,), (14,)]
```

Interpretation:

- This representative crossings-only row did not show rank/order divergence at seed42. The aggregate r74 row still reports crossing D=9/R=8, so crossings-only failures are not safely reducible to "ordering differs"; they need metric-level or geometry tie inspection per graph.

### F5. HYPOTHESIS: Igraph GLPK exactness and solver ties remain a root cause

Impact: medium-high for igraph-family stress/NP rows.

Reference code:

- Igraph vertical placement uses GLPK for directed graphs with <=1000 nodes: `/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:552-675`.
- It disables GLPK presolve and calls `glp_simplex()`: `/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:602-650`.
- It floors `glp_get_col_prim()` into layer membership: `/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:652-656`.

Dagua code:

- Dagua uses scipy `linprog(method="highs")`, not GLPK simplex: `dagua/layout/ops/sugiyama.py:450-492`.
- Dagua computes objective coefficients itself: `dagua/layout/ops/sugiyama.py:495-543`.
- Dagua tests assert the r74 objective assumption: `tests/test_layout/test_sugiyama_fidelity.py:87-105`.

Important source discrepancy:

- In this reference checkout, both `indegs` and `outdegs` are populated with `IGRAPH_IN`: `/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:589-592`.
- Dagua computes out-strength for sources and in-strength for targets: `dagua/layout/ops/sugiyama.py:520-543`.
- Because prior r74 notes say the LP objective was already fixed, I am not calling this a confirmed runtime mismatch until the installed python-igraph source/version is checked. It is a confirmed source-level discrepancy in the provided reference tree.

Cheapest decisive experiment:

- Add a temporary `/tmp` script that constructs a 4-node DAG and compares igraph final y-ranks against Dagua `_igraph_glpk_layer_assignments()` using the benchmark competitor path for `classic_sugiyama_default` and `igraph_sugiyama__for__classic_sugiyama_default`.
- Runtime estimate: 2-5 minutes including benchmark import overhead.

Fix sketch:

- If installed igraph matches source, port the IN/IN plus feedback-subtraction objective exactly.
- Otherwise, keep the current objective but replace HiGHS with a GLPK-compatible deterministic simplex or implement the totally-unimodular network-simplex formulation with GLPK tie order.

Risk:

- Very high. LP rank changes are first-stage changes and can break many existing equivalent/bit-exact igraph rows.
- Gate behind `fidelity_mode="igraph"` and test on known bit-exact controls before broad benchmark runs.

### F6. CONFIRMED/HYPOTHESIS: Igraph BK conflict detection differs from Dagua

Impact: medium for igraph-family crossing/stress rows after ranks/order match.

Reference code:

- Igraph builds the dummy-expanded component and then calls ordering and BK horizontal placement: `/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:379-472`.
- Igraph type-1 conflict detection gathers outgoing neighbors for a layer: `/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:898-910`.
- It then loops `j` and `k` over that gathered count but calls `IGRAPH_FROM(graph, j)` and `IGRAPH_TO(graph, j)`, i.e. edge ids by local ordinal rather than the gathered neighbor/edge list: `/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:912-944`.

Dagua code:

- Dagua implements canonical layer-boundary conflict detection over ordered predecessors: `dagua/layout/ops/sugiyama.py:1914-1973`.
- Dagua vertical alignment is canonical median alignment over transformed layers: `dagua/layout/ops/sugiyama.py:2006-2057`.

Interpretation:

- The code difference is confirmed. Runtime impact is a hypothesis because I did not run an instrumented conflict-dump against installed igraph.
- This matters only once ranks and crossing-minimized order are already close; then BK x changes can alter stress and sometimes straight-segment crossing counts.

Cheapest decisive experiment:

- For `multiscale_skip_cascade` and `regular_4_40`, dump Dagua ignored-edge sets from `_find_type1_conflicts()` and compare to an igraph-C-style ordinal-edge emulation on the same expanded component.
- Runtime estimate: <1 minute as a pure `/tmp` read-only script.

Fix sketch:

- Add an `igraph_exact_conflicts` mode that reproduces the C ordinal-edge behavior, even if it looks wrong.
- Keep the canonical conflict code for non-igraph modes.

Risk:

- Medium. It changes x coordinates only, but crossing counts can move for dense same-rank geometry.

### F7. CONFIRMED: Igraph ordering is closer but still sensitive to qsort/tie behavior

Impact: medium for igraph-family rows with equal barycenters and dense repeated layers.

Reference code:

- Igraph starts with first-seen x order inside each layer: `/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:733-742`.
- It computes incidence barycenters with duplicate neighbors, falling back to current x when isolated: `/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:677-705`.
- It runs down and up passes until no layer order changes or `maxiter` is reached: `/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:749-828`.

Dagua code:

- Dagua igraph mode uses incidence barycenters and stable-stop: `dagua/layout/ops/sugiyama.py:1288-1344`.
- Dagua added an igraph qsort-index imitation for score ties: `dagua/layout/ops/sugiyama.py:1425-1592`.

Remaining hypothesis:

- Dagua's qsort imitation may still differ from the exact igraph utility used by `igraph_vector_sort_ind()` for NaN/equal keys and small partitions.
- Rows like `regular_4_40`, `real_karate_34`, and `weighted_karate_34` have many equal/tied barycenters, making this plausible.

Cheapest decisive experiment:

- Build only the expanded layered graph for one failing graph, record per-pass barycenter arrays, and run Dagua qsort vs an extracted igraph `sort_ind` trace.
- Runtime estimate: 5-10 minutes if using a tiny C harness; <1 minute if the installed Python binding exposes enough sorted index behavior.

## 3. Root causes, fix staging, impact, and risk

### Root cause A: Graphviz x-coordinate assignment is not ported

Fix:

- Implement F2 staged aux-graph + LR-simplex x port.

Expected impact:

- Most graphviz stress-only and stress+NP rows.
- Approximate count: 39 graphviz rows fail stress without crossings, plus some mixed rows after mincross is fixed.

Risk to bit-exact combos:

- Medium for simple graphviz rows if the simplex tie order differs.
- High for labels/clusters if partially ported constraints are applied as if complete.

### Root cause B: Graphviz mincross is partial

Fix:

- Implement class2-like expansion, weighted crossing counts, pass 0/1/2 control flow, and `left2right` constraints in phases.

Expected impact:

- Graphviz crossing failures and large-tail structural failures.
- Should combine with Root cause A before final evaluation.

Risk:

- High. Mincross changes order and crossing counts. Use a ladder and preserve an escape hatch to existing helper during port bring-up.

### Root cause C: Igraph GLPK/LP exactness is not proven

Fix:

- Verify installed igraph objective against source.
- Port exact objective and solver tie behavior, or replace HiGHS with GLPK-compatible behavior.

Expected impact:

- Igraph-family rows where ranks differ first.

Risk:

- Very high because rank is the first stage.

### Root cause D: Igraph BK conflict detection and qsort ties may differ

Fix:

- Add exact igraph conflict and sort-ind modes.

Expected impact:

- Igraph-family rows where ranks/order mostly match but stress/crossing differs.

Risk:

- Medium. Mostly x-stage and order tie changes, but dense graphs can move many crossings.

## 4. Divergence-stage inventory

- `binary_tree::classic_sugiyama_graphviz_fidelity`: CONFIRMED x-coordinate stage. Benchmark-path probe shows same rank sizes and within-rank order; x spans differ.
- `bipartite_4_3_4::classic_sugiyama_graphviz_fidelity`: HYPOTHESIS x-coordinate stage. Target row has equal crossings D=36/R=36 but stress fails; source path matches binary_tree diagnosis.
- `asymmetric_hourglass_hub::classic_sugiyama_graphviz_fidelity`: HYPOTHESIS rank or x stage. Target has stress+NP failure with crossings equal; inspect ranks first.
- `dense_pair_50::classic_sugiyama_graphviz_fidelity`: HYPOTHESIS ordering/mincross first. Crossings differ D=391/R=331; F3 likely.
- `ba_500::classic_sugiyama_graphviz_fidelity`: HYPOTHESIS ordering/mincross first. User-provided example D=22344/R=2805 is too large for x-only if rank/order were equal.
- `multiscale_skip_cascade::classic_sugiyama_default`: CONFIRMED not ordering at seed42 in probe; aggregate crossings-only row remains unexplained by this seed's stage inventory.
- Igraph-family dense/regular/weighted rows: HYPOTHESIS GLPK rank or exact igraph BK/tie behavior. Need per-stage dumps.

## 5. Crossings-only failures

There are 11 crossings-only rows:

- `multiscale_skip_cascade::classic_sugiyama_default`
- `multiscale_skip_cascade::classic_sugiyama_passes48`
- `multiscale_skip_cascade::classic_sugiyama_passes4`
- `multiscale_skip_cascade::classic_sugiyama_tight`
- `multiscale_skip_cascade::classic_sugiyama_wide`
- `random_dag_200::classic_sugiyama_wide`
- `random_dag_50::classic_sugiyama_wide`
- `regular_4_40::classic_sugiyama_default`
- `regular_4_40::classic_sugiyama_tight`
- `regular_4_40::classic_sugiyama_wide`
- `weighted_karate_34::classic_sugiyama_graphviz_fidelity`

Answer: not all are simply ordering-stage divergence with coincidentally matching stress. The seed42 `multiscale_skip_cascade` benchmark probe has identical inferred rank membership and x-order for Dagua and igraph, yet the r75 target row is crossings-only. For the graphviz `weighted_karate_34` row, F3 remains plausible because graphviz mincross has crossing-specific deltas. For `regular_4_40` and `random_dag_*`, the cheapest next check is the same rank/order probe plus exact crossing count on the saved tensors.

## 6. What landing Graphviz x + mincross should fix

If F2 and F3 land faithfully:

- Expected direct coverage: 58/129 target rows, the graphviz-dot family.
- Expected successful reclassification: about 50-55 of those 58 if simple/no-cluster graphs dominate and simplex tie order is exact.
- Overall target fraction: about 39-43% likely fixed, with an absolute ceiling of 45% because 71 rows compare against igraph.
- Large graph tail: likely high impact for graphviz-family large-tail rows, especially large crossing deltas, but only after mincross class2/weighted crossing work lands.

What it cannot fix:

- Any `classic_sugiyama_default`, `classic_sugiyama_passes4`, `classic_sugiyama_passes48`, `classic_sugiyama_tight`, or `classic_sugiyama_wide` row. These compare to igraph and need F5-F7.
- Graphviz rows involving labels/clusters/ports may remain if the first x port skips those constraints.

Graphviz rows at risk if cluster/label/port constraints are skipped:

- `center_port_backedge_hub::classic_sugiyama_graphviz_fidelity`
- `cluster_member_style_stress::classic_sugiyama_graphviz_fidelity`
- `clustered_longlabel_handoffs::classic_sugiyama_graphviz_fidelity`
- `disconnected_label_cycle_collage::classic_sugiyama_graphviz_fidelity`
- `edge_label_braid::classic_sugiyama_graphviz_fidelity`
- `extreme_mixed_width_transformer::classic_sugiyama_graphviz_fidelity`
- `hub_fanout_label_skew::classic_sugiyama_graphviz_fidelity`
- `interleaved_cluster_crosstalk::classic_sugiyama_graphviz_fidelity`
- `mixed_width_labels::classic_sugiyama_graphviz_fidelity`
- `nested_cluster_label_stack::classic_sugiyama_graphviz_fidelity`
- `small_label_storm::classic_sugiyama_graphviz_fidelity`
- `weighted_clusters_3x10::classic_sugiyama_graphviz_fidelity`

## 7. Target combos not individually explained

I explained the graphviz-family rows by root cause class, but I did not individually stage-dump every graph. The following igraph-family rows are not explained by the Graphviz staged port and require separate igraph-stage experiments:

`densenet_block::classic_sugiyama_default`, `densenet_block::classic_sugiyama_passes4`, `densenet_block::classic_sugiyama_passes48`, `densenet_block::classic_sugiyama_tight`, `densenet_block::classic_sugiyama_wide`, `heavy_tail_weights_50::classic_sugiyama_default`, `heavy_tail_weights_50::classic_sugiyama_passes4`, `heavy_tail_weights_50::classic_sugiyama_tight`, `heavy_tail_weights_50::classic_sugiyama_wide`, `hexagonal_lattice_42::classic_sugiyama_default`, `hexagonal_lattice_42::classic_sugiyama_passes4`, `hexagonal_lattice_42::classic_sugiyama_passes48`, `hexagonal_lattice_42::classic_sugiyama_tight`, `hexagonal_lattice_42::classic_sugiyama_wide`, `hub_skip_superfan::classic_sugiyama_default`, `hub_skip_superfan::classic_sugiyama_passes48`, `hub_skip_superfan::classic_sugiyama_tight`, `hub_skip_superfan::classic_sugiyama_wide`, `interleaved_cluster_crosstalk::classic_sugiyama_default`, `interleaved_cluster_crosstalk::classic_sugiyama_passes4`, `interleaved_cluster_crosstalk::classic_sugiyama_passes48`, `interleaved_cluster_crosstalk::classic_sugiyama_tight`, `interleaved_cluster_crosstalk::classic_sugiyama_wide`, `kitchen_sink_platform_graph::classic_sugiyama_default`, `kitchen_sink_platform_graph::classic_sugiyama_passes4`, `kitchen_sink_platform_graph::classic_sugiyama_passes48`, `kitchen_sink_platform_graph::classic_sugiyama_tight`, `kitchen_sink_platform_graph::classic_sugiyama_wide`, `moe_router_sparse::classic_sugiyama_default`, `moe_router_sparse::classic_sugiyama_passes4`, `moe_router_sparse::classic_sugiyama_passes48`, `moe_router_sparse::classic_sugiyama_tight`, `moe_router_sparse::classic_sugiyama_wide`, `dense_pair_50::classic_sugiyama_passes4`, `multiscale_skip_cascade::classic_sugiyama_default`, `multiscale_skip_cascade::classic_sugiyama_passes48`, `multiscale_skip_cascade::classic_sugiyama_passes4`, `multiscale_skip_cascade::classic_sugiyama_tight`, `multiscale_skip_cascade::classic_sugiyama_wide`, `planar_60::classic_sugiyama_default`, `planar_60::classic_sugiyama_passes4`, `planar_60::classic_sugiyama_passes48`, `planar_60::classic_sugiyama_tight`, `planar_60::classic_sugiyama_wide`, `random_dag_200::classic_sugiyama_passes48`, `random_dag_200::classic_sugiyama_wide`, `random_dag_50::classic_sugiyama_wide`, `real_karate_34::classic_sugiyama_default`, `real_karate_34::classic_sugiyama_passes4`, `real_karate_34::classic_sugiyama_tight`, `real_karate_34::classic_sugiyama_wide`, `real_lesmis_77::classic_sugiyama_default`, `regular_3_30::classic_sugiyama_passes4`, `regular_4_40::classic_sugiyama_default`, `regular_4_40::classic_sugiyama_passes4`, `regular_4_40::classic_sugiyama_tight`, `regular_4_40::classic_sugiyama_wide`, `real_lesmis_77::classic_sugiyama_passes48`, `real_lesmis_77::classic_sugiyama_tight`, `real_lesmis_77::classic_sugiyama_wide`, `weighted_clusters_3x10::classic_sugiyama_passes4`, `weighted_clusters_3x10::classic_sugiyama_passes48`, `weighted_karate_34::classic_sugiyama_default`, `width_skew_late_merge::classic_sugiyama_default`, `width_skew_late_merge::classic_sugiyama_passes4`, `width_skew_late_merge::classic_sugiyama_passes48`, `width_skew_late_merge::classic_sugiyama_tight`, `width_skew_late_merge::classic_sugiyama_wide`, `weighted_karate_34::classic_sugiyama_passes4`, `weighted_karate_34::classic_sugiyama_tight`, `weighted_karate_34::classic_sugiyama_wide`.

## 8. Dead code / removable notes

No repo code was modified. I did not identify removable dead code in scope.
