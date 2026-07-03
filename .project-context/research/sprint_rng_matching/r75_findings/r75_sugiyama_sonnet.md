# r75 sugiyama bucket -- STAGED PORT SPEC (sonnet)

## 1. Executive summary

129 divergent combos split 58 graphviz_fidelity (vs `graphviz_dot`) / 71 igraph-family (vs
`igraph_sugiyama`, 5 param variants). Dagua's ranking and ordering stages are ALREADY faithful
ports for both families (network-simplex rank port in `dot_rank.py`, igraph-qsort-faithful
barycenter ordering, graphviz mincross port in `_dot_mincross.py`). The single biggest gap is
**x-coordinate assignment: dagua always uses one shared Brandes-Kopf implementation
(`_coordinate_assignment` in `dagua/layout/ops/sugiyama.py:1675`) for every fidelity mode.**
Graphviz's `dot` does NOT use Brandes-Kopf at all -- it uses network simplex on an auxiliary
graph (`position.c`). Igraph's Sugiyama DOES use Brandes-Kopf (confirmed by full-file read of
`igraph/src/layout/sugiyama.c`), so the igraph-family x-coord gap is a details-matching problem,
not an algorithm-family problem, and is smaller. 39/58 graphviz_fidelity combos show the exact
signature of "ranks+order correct, x-coord wrong" (crossings match, stress diverges, dagua worse
57/58). The igraph family shows a different, more mixed signature (42/71 both cross+stress
diverge) pointing at ordering-stage details (median-of-4 combine, missing omega weights are
N/A for igraph, initial order) as well as x-coord BK details. Landing a real graphviz
network-simplex x-coord port plus a small mincross `init_order`/omega-weight fix should fix
the large majority of the 58 graphviz combos; landing BK detail fixes (median-of-4 mechanics,
`hgap`/nodesep parity, type-1-conflict edge cases) should fix a majority of the crossings-only
and stress-only igraph combos, but the ordering-stage divergence on igraph (42/71) needs the
same care as the mincross work. ba_500 (22344 vs 2805 crossings, r73 finding, not in this
sprint's <=300-node target list) is evidence the ordering/mincross gap is severe at scale and
will need the mincross init_order + omega-weight fix to close, independent of x-coord.

## 2. Findings ranked by expected combo-count impact

### Finding 1 (CONFIRMED): x-coordinate assignment is a single unconditional Brandes-Kopf
implementation for ALL fidelity modes -- no graphviz network-simplex x-coord port exists

**dagua side:** `dagua/layout/ops/pipelines/sugiyama.py:106-124` (`build_sugiyama_pipeline`)
wires `_CoordinateAssignment(center_coordinates=center_coordinates)` unconditionally --
`fidelity_mode` is NOT threaded into `_CoordinateAssignment.__init__` at all (compare
`_AssignLayers(fidelity_mode=...)` at line 111 and `_BarycenterOrdering(..., use_graphviz_mincross=...)`
at line 114-122 which DO branch on fidelity_mode). `_CoordinateAssignment.apply`
(`dagua/layout/ops/sugiyama.py:2868-2919`) calls `_coordinate_assignment` ->
`_brandes_koepf_x_positions` (`dagua/layout/ops/sugiyama.py:1741-1831`) unconditionally: 4-pass
BK (`ul/ur/dl/dr` orientations, type-1 conflict detection, vertical alignment, horizontal
compaction, median-of-4 combine at `_median_balanced_coordinates:2291-2313`).

**Reference side (graphviz):** `lib/dotgen/position.c:127-154` (`dot_position`) calls
`create_aux_edges(g)` (line 141) then `rank(g, 2, nsiter2(g))` (line 142) -- **network simplex on
an auxiliary graph**, not Brandes-Kopf. `make_edge_pairs` (`position.c:327-352`) builds one
"omega" slack node per real edge with two aux edges to its endpoints (the classic
network-simplex-for-x-coordinates construction); `make_LR_constraints` (`position.c:218-324`)
builds zero-weight separation edges enforcing `nodesep`-based ordering constraints. There is NO
`medianpos`/priority function and NO Brandes-Kopf-style algorithm anywhere in `position.c`
(confirmed: forward declarations at `position.c:31-40` list only
`nsiter2, create_aux_edges, remove_aux_edges, set_xcoords, set_ycoords, set_aspect, expand_leaves,
make_lrvn, contain_nodes, idealsize`). Graphviz dot's x-coordinates are architecturally a
different algorithm family than dagua's implementation.

**Reference side (igraph):** `igraph/src/layout/sugiyama.c:858-1047`
(`igraph_i_layout_sugiyama_place_nodes_horizontally`) IS Brandes-Kopf (Brandes & Koepf 2002,
cited line 68-71): type-1 conflict detection (`line 893-949`), 4x
`igraph_i_layout_sugiyama_vertical_alignment` (`line 1049-1178`), horizontal compaction with
recursive block placement (`line 1190-1301`), median-of-4 combine via `igraph_i_median_4`
(`line 211-217`, applied at `line 1024-1029`). No GLPK/LP call appears anywhere in this function
or its helpers -- LP is used ONLY in the ranking stage (see Finding 4). So dagua's choice of BK
for the igraph family is the right algorithm family; only implementation details can differ.

**Evidence from target JSON (CONFIRMED via direct read of
`r75_targets_sugiyama.json`):** for `classic_sugiyama_graphviz_fidelity`, 39/58 combos have
`cross.equiv=True` and `battery_stress.equiv=False` (ranks+order match, x-coord alone diverges);
57/58 have dagua's stress `D` WORSE than the reference `R` (consistent with a wrong x-coord
algorithm, not FP noise). Example: `binary_tree::classic_sugiyama_graphviz_fidelity` --
`cross.D=0.0, cross.R=0.0` (exact crossing match) but `battery_stress.D=0.197, R=0.151`
(30% worse, margin 0.003, nowhere near ref_spread=0.0 tolerance).

**Fix sketch:** port `position.c`'s network-simplex x-coordinate assignment as a new dagua op,
gated by `fidelity_mode in {"dot","graphviz_dot","graphviz"}`, replacing the unconditional BK
call in that mode only. Reuse the existing `dot_rank.py` `_SimplexGraph`/`_run_network_simplex`
machinery (it is already a general Graphviz-style network-simplex solver -- `graphviz_rank_assignment`
takes arbitrary `(tail, head, minlen, weight)` edges, which is exactly the aux-graph shape
`make_edge_pairs`/`make_LR_constraints` build). Concretely:
1. Build one slack/omega node per real+virtual edge in the expanded graph (mirrors
   `make_edge_pairs`, `position.c:327-352`): for edge `(u,v)`, add slack node `s`, aux edges
   `(s,u,minlen=1,weight=W)` and `(s,v,minlen=1,weight=W)` where `W` is the edge's *mincross
   weight* (i.e. AFTER the omega-table multiplier from Finding 2 below -- `virtual_weight()`
   pre-multiplies `ED_weight` before position.c runs, per file:line evidence in the background
   research).
2. Build LR-constraint edges per rank (mirrors `make_LR_constraints`, `position.c:218-324`):
   for adjacent nodes `(u,v)` at positions `(j,j+1)` in a rank, add edge
   `(u,v,minlen=halfwidth(u)+halfwidth(v)+node_sep,weight=0)`.
3. Run `_run_network_simplex`-equivalent with `balance=2` (LR balance, per `position.c:142`
   comment `/* LR balance == 2 */`) on the combined aux graph.
4. Reuse the resulting node potential as x-coordinate directly (`position.c:570-584`:
   `ND_coord(v).x = ND_rank(v)`).
5. Flat-edge handling (`position.c:289-321`, three cases: adjacent-neighbor merge into existing
   LR edge, non-adjacent separator edge, labeled flat edges) can be SKIPPED initially since none
   of the 129 targets are flagged with edge labels in this bucket; add a follow-up note if
   `edge_label_braid`/`clustered_longlabel_handoffs` graphviz_fidelity combos remain divergent
   after the core port lands.
6. Cluster/port constraints (`pos_clusters`, `contain_nodes`) can be SKIPPED -- dagua's classic
   sugiyama pipeline does not claim cluster fidelity for this bucket (mincross.c cluster logic
   is likewise explicitly out of scope per the bucket brief).

**LOC estimate:** ~250-350 lines (new module `dagua/layout/ops/_dot_position.py` mirroring the
shape of `_dot_mincross.py`), reusing `_SimplexGraph`/pivot machinery from `dot_rank.py` via a
shared helper extraction (~50 lines of refactor to make `_run_network_simplex` reusable with a
different constraint-builder front end) rather than reimplementing network simplex a third time.

**Verification ladder:** (1) rank match -- already passing per the crossings-equiv evidence
(39/58 combos have `cross.equiv=True`, meaning rank+order structure is already right on those);
(2) order match -- same evidence; (3) x-coord match -- run the new port on `binary_tree` (0
crossings both sides, pure x-coord test) and check Procrustes RMSD/battery_stress against
`graphviz_dot__for__classic_sugiyama_graphviz_fidelity` positions; escalate through 2-3 more
small trees, then the 39 x-coord-only-signature graphs, then the 18 both-cross-and-stress-fail
graphviz_fidelity graphs (which need Finding 2/3 fixes as well).

**Expected impact:** most of the 39 x-coord-only-signature graphviz_fidelity combos (binary_tree,
asymmetric_hourglass_hub, bipartite_4_3_4, center_port_backedge_hub, broken_symmetry_residual_pair,
cluster_member_style_stress, clustered_longlabel_handoffs, disconnected_encoder_residual,
disconnected_label_cycle_collage, edge_label_braid, extreme_mixed_width_transformer,
heavy_tail_weights_50, and ~27 more -- full list is `[d['graph'] for d in target json if
engine==classic_sugiyama_graphviz_fidelity and cross.equiv and not battery_stress.equiv]`).
Partial help (not full fix, needs Finding 3 too) on the remaining 18 graphviz_fidelity combos
where crossings ALSO diverge (densenet_block, heavy_tail_weights_50, grid_rect_6x8, and 15 more).

**RISK to bit-exact combos:** LOW if gated strictly by `fidelity_mode in
{"dot","graphviz_dot","graphviz"}` (a new branch parallel to the existing
`use_graphviz_mincross`/`use_graphviz_rank` branches, not a change to the default/igraph paths).
The default (no fidelity_mode) and igraph-fidelity paths keep using
`_brandes_koepf_x_positions` unchanged -- zero risk of the "blanket fix broke bit-exact combos"
failure mode from prior rounds, PROVIDED the new op is a genuinely separate code path and not a
parameterization of the existing BK function.

### Finding 2 (CONFIRMED): no omega/mincross-weight table (C_EE=1/C_VS=2/C_SS=2/C_VV=4) anywhere
in dagua's graphviz-fidelity path

**dagua side:** `dot_rank.py::graphviz_rank_assignment` (`dagua/layout/ops/pipelines/dot_rank.py:104-189`)
and `_dot_mincross.py::graphviz_mincross` (`dagua/layout/ops/_dot_mincross.py:14-91`) both accept
caller-supplied per-edge weights but neither computes graphviz's virtual/real endpoint-type
multiplier. `_graphviz_layer_assignments`
(`dagua/layout/ops/sugiyama.py:323-386`) passes through `edge_weights` unchanged (default 1 if
the graph has none) -- confirmed by `grep` for `OMEGA|omega|C_EE|C_VS|C_SS|C_VV` across
`dot_rank.py` and `_dot_mincross.py`: zero matches.

**Reference side:** `class2.c:155-172` tracks `ND_weight_class` per node (capped at 3) as chains
are built; `mincross.c:1709-1718` defines
`table[NTYPES][NTYPES]` with `C_EE=1` (real-real), `C_VS=C_SS=2` (virtual-single or
single-single), `C_VV=4` (virtual-virtual), applied by `virtual_weight()`
(`mincross.c:1729-1741`: `ED_weight(e) *= t;`), called per chain segment inside `make_chain()`
(`class2.c:91-92`). This multiplier makes virtual-virtual (dummy-to-dummy, i.e. the middle
segments of long edges) chain segments 4x "stiffer" in both the mincross median computation
(since `medians()` in `mincross.c:1633-1641` sums `MC_SCALE*ND_order + port.order` weighted
implicitly through which edges even get counted -- weight affects `ED_xpenalty`, gating whether
an edge counts at all in `medians()`, `mincross.c` filters on `ED_xpenalty(e)>0`) and in the
x-coordinate omega-node weight (Finding 1's aux-edge weight `W`). Missing this table means (a)
dagua's graphviz-mode mincross medians under- or over-weight virtual-node chains relative to
graphviz, producing different tie-breaks on graphs with long edges, and (b) once Finding 1's
x-coord port lands, its omega weights will be wrong (all effectively `C_EE=1`) unless this table
is added too.

**Fix sketch:** add a small helper in `_dot_mincross.py` or a new shared module: classify nodes
as `NORMAL`/`VIRTUAL` (dagua already tracks this via `dummy_mask`/`num_original_nodes` boundary
in `_brandes_koepf_x_positions:1780` -- same classification is reusable), then multiply each
expanded edge's weight by `table[endpoint_class(tail)][endpoint_class(head)]` before it's
consumed by both `graphviz_mincross` and the new x-coord port from Finding 1. This must happen
once, upstream of both consumers (mirrors graphviz calling `virtual_weight()` once in
`make_chain`, then both mincross and position.c inherit the pre-multiplied `ED_weight`).

**Expected impact:** contributes to closing the remaining stress/crossing gap on the 18
graphviz_fidelity combos with BOTH legs failing, and likely needed for graphs with long edges
(multi-rank spans) more broadly -- notably relevant to ba_500's severe crossing gap (22344 vs
2805, r73 finding) since that graph likely has many long edges through many ranks where the
weight table matters most.

**RISK:** LOW-MEDIUM. This changes weight VALUES fed into the existing graphviz-mode mincross and
the new x-coord port, both already gated to `fidelity_mode="graphviz"`. Does not touch
default/igraph paths. Risk is to the 58-39=19 graphviz_fidelity combos NOT in the x-coord-only
bucket if the weight change interacts unexpectedly with the mincross tie-break logic already in
place -- test against the currently-passing graphviz_fidelity combos (not in the 129-target list)
to confirm no regression, since `_dot_mincross.py` is shared code touched by this fix.

### Finding 3 (CONFIRMED): mincross init_order uses plain node-id sort, not graphviz's
BFS-from-sources `build_ranks`

**dagua side:** `_barycenter_ordering` (`dagua/layout/ops/sugiyama.py:1255`):
`ordered_layers = [sorted(layer) for layer in layers]` -- unconditional numeric sort, used as
the starting order for BOTH the default barycenter path AND the `use_graphviz_mincross` path
(`dagua/layout/ops/sugiyama.py:1261-1270`, `graphviz_mincross(ranks=ordered_layers, ...)` is
called directly on this numerically-sorted input with no separate initial-order step).

**Reference side:** `build_ranks(graph_t *g, int pass)` (`mincross.c:1212-1286`) is NOT a plain
sort -- it's a **BFS seeded from source/sink nodes in adjacency-list (insertion) order**: pass 0
seeds from nodes with no in-edges and BFS-expands via out-edges
(`mincross.c:1246-1247, 1288-1308`); pass 1 does the mirror (seed from no-out-edge nodes,
expand via in-edges). `mincross()`'s outer loop calls `build_ranks(g, pass)` for `pass<=1`
(`mincross.c:703-706`) -- i.e. it tries BOTH initial orderings and keeps whichever produces fewer
crossings after the subsequent iterate. Graphviz's initial order is a function of edge-creation
order / adjacency-list order (input-order-dependent), not numeric node-id order.

**Impact assessment:** this is a real algorithmic difference, but its practical effect is bounded
by `graphviz_mincross`'s iteration loop (MinQuit=8, up to `iterations` full sweeps) -- for graphs
small enough to converge to a stable local optimum regardless of start point, the effect
disappears. It likely matters most for (a) graphs where mincross does NOT fully converge within
its iteration budget (the ba_500-scale graphs, and possibly the "18 both legs fail" graphviz_fidelity
combos with denser/longer-edge structure), and (b) tie-break-sensitive small graphs where two
different local optima have EQUAL crossing count but different x-coordinate outcomes after
Finding 1's x-coord port (this would show as bit-exact crossings but non-bit-exact final
positions -- exactly the unexplained residual after Findings 1+2 land).

**Fix sketch:** port `build_ranks` as a pre-step before calling `graphviz_mincross`: run BOTH the
in-edge-BFS and out-edge-BFS seeded orderings, count crossings with the existing
`_count_crossings` helper (`_dot_mincross.py:454-498`), keep the better one as the initial
`ordered_ranks` passed into the sweep loop. Needs adjacency-list ORDER to be graphviz-faithful
too (edge-creation order, i.e. dagua's existing `expanded_sources`/`expanded_targets`
construction order in `_expand_long_edges_with_dummy_nodes`,
`dagua/layout/ops/sugiyama.py:997-1083`, which is already edge-index order -- likely already
matches since both build chains in original-edge-list order).

**Expected impact:** secondary/refinement fix; expect it closes SOME of the remaining
crossings-still-different combos after Finding 1+2, and is the leading hypothesis for the ba_500
large-graph crossing gap (22344 vs 2805) since that gap is too large to be explained by x-coord
alone (crossings are an ORDERING-stage quantity, and 22344 vs 2805 is an 8x difference -- consistent
with mincross getting stuck in a much worse local optimum from a much worse start point on a
500-node graph, not a matched-order noise floor).

**RISK:** LOW. Purely additive to the `use_graphviz_mincross=True` path; does not touch
default/igraph barycenter ordering.

### Finding 4 (CONFIRMED, informational): igraph's x-coordinate stage is Brandes-Kopf, NOT an
LP -- corrects the bucket brief's premise; r74's LP fix is scoped correctly to ranking only

Full read of `igraph/src/layout/sugiyama.c` confirms GLPK/LP usage is confined EXCLUSIVELY to
the ranking stage: `igraph_i_layout_sugiyama_place_nodes_vertically`
(`sugiyama.c:552-675`, `#ifdef HAVE_GLPK` block at line 563, `glp_simplex` call at line 649, LP
objective at lines 609-616 minimizing `sum (outdeg_i - indeg_i) * rank_i` subject to
`rank(to)-rank(from) >= 1` per edge). The x-coordinate stage
(`igraph_i_layout_sugiyama_place_nodes_horizontally`, `sugiyama.c:858-1047`) has zero `glp_`
symbols -- confirmed no GLPK usage outside the 552-675 range in the whole file. r74's commit
169ce7b (igraph-faithful GLPK layer objective) is therefore correctly scoped to
`_igraph_glpk_layer_assignments` (`dagua/layout/ops/sugiyama.py:389-493`), which is a ranking-only
function. **No further LP work is needed or applicable to x-coordinates for the igraph family** --
the remaining igraph-family x-coord gap is entirely about Brandes-Kopf implementation detail
matching (see Finding 5).

### Finding 5 (HYPOTHESIS): igraph-family x-coord/ordering BK detail mismatches --
median-of-4 combine, type-1-conflict edge semantics, disconnected-component packing order

**dagua side (median-of-4):** `_median_balanced_coordinates` (`dagua/layout/ops/sugiyama.py:2291-2313`)
sorts the 4 alignment values and averages the middle two: `samples[1]+samples[2])/2.0`. This
matches the STANDARD median-of-4 definition and should match igraph's `igraph_i_median_4`
(cited by the background read at `sugiyama.c:211-217`, not independently re-read here -- flagged
as the cheapest decisive check: read those 6 lines and diff against dagua's formula).

**dagua side (alignment anchoring):** `_align_compacted_coordinates`
(`dagua/layout/ops/sugiyama.py:2242-2270`) picks the alignment with the SMALLEST coordinate span
as an anchor and shifts the other three to match its left/right extreme before taking the
median. This "smallest-span-as-anchor" heuristic is a dagua-specific design choice with no
citation found in the igraph background read (the agent's summary of
`sugiyama.c:990-1022` describes "normalizing them to a common minimum-width frame" but did not
quote the exact selection rule) -- **this is the cheapest decisive experiment**: read
`sugiyama.c:990-1022` directly (30 lines) and diff against dagua's anchor-selection logic. If
igraph anchors differently (e.g. always uses a fixed alignment like `ul`, or normalizes by
translating each independently to a shared left origin without picking an "anchor" at all), this
would explain systematic x-coordinate offsets on the crossings-only and the
both-legs-diverge igraph combos alike.

**dagua side (disconnected component packing):** `_layout_igraph_packed_components`
(`dagua/layout/ops/pipelines/sugiyama.py:368-457`) lays out each weak component independently via
recursive `layout_sugiyama_pipeline` calls and packs left-to-right with `dx += max_x - dx +
node_sep` (`sugiyama.py:452-456`). Reference: igraph packs components via
`igraph_connected_components(..., IGRAPH_WEAK)` (`sugiyama.c:343`), independent per-component
layout (`sugiyama.c:347-542`), packed with `dx += max_x + hgap` (`sugiyama.c:521-527`) -- same
shape, but the COMPONENT ORDER may differ: dagua's `_weak_components`
(`dagua/layout/ops/pipelines/sugiyama.py:323-365`) does a DFS from `range(num_nodes)` in ascending
node-id order and sorts each component by original node id; igraph's component order comes from
`igraph_connected_components`'s internal traversal (not verified here -- likely also ascending
vertex-id BFS/DFS order, needs a direct check if `kitchen_sink_platform_graph` or other
disconnected igraph-family combos remain divergent after Findings 1-3 land). This affects the
`kitchen_sink_platform_graph` combos (6 of the 129, all `classic_sugiyama_*` non-graphviz
variants) and possibly `multi_component_80`.

**Expected impact:** median-of-4/anchor fixes are cheap, likely fix a modest slice of the
19 igraph-family x-coord-only-signature combos (kitchen_sink_platform_graph::default,
moe_router_sparse::default, real_lesmis_77::default, width_skew_late_merge::default, and their
sibling param variants -- 4 unique graphs x up to 5 param variants each). Not expected to move
the 42/71 both-legs-diverge combos, which are primarily an ordering-stage (barycenter/median
mismatch or maxiter/pass-counting) issue -- see Finding 6.

**Cheapest decisive experiment:** read `igraph/src/layout/sugiyama.c:990-1022` and
`sugiyama.c:211-217` directly (already-open reference tree, no benchmark run needed,
<5 min), diff against `_align_compacted_coordinates`/`_median_balanced_coordinates`. If
formulas match exactly, re-scope this finding to "not the cause" and look instead at
`_vertical_alignment`'s median-selection-index formula (`dagua/layout/ops/sugiyama.py:2043-2044`:
`median_start = (len(neighbor_nodes)-1)//2; median_stop = len(neighbor_nodes)//2`) against
igraph's `igraph_i_layout_sugiyama_vertical_alignment` (`sugiyama.c:1049-1178`) inline median
logic (not independently verified here -- second-cheapest check, same read, ~10 more lines).

### Finding 6 (HYPOTHESIS): igraph-family ordering-stage divergence (42/71 both-legs-fail
combos) -- maxiter pass-counting or isolated-node tie-break mismatch

**dagua side:** `_BarycenterOrdering` with `use_incidence_barycenters=True, stop_when_stable=True`
(set via `fidelity_mode="igraph"` in `build_sugiyama_pipeline:106-122`). Per-pass structure in
`_barycenter_ordering` (`dagua/layout/ops/sugiyama.py:1288-1342`): each `pass_num` in
`range(num_passes)` does ONE full down-sweep (`dagua/layout/ops/sugiyama.py:1292-1307`) THEN ONE
full up-sweep (`sugiyama.py:1309-1324`) -- i.e. dagua's "one pass" already matches igraph's
"one iteration = one down-sweep + one up-sweep" semantics (confirmed against the background
read of `sugiyama.c:751` `while (changed && iter < maxiter)` where `iter` increments once per
combined down+up sweep at `sugiyama.c:827`). `num_passes=maxiter` is passed through directly
from the variant config (`{"maxiter": 24/4/48}` in `dagua/eval/variants.py:924-985` ->
`barycenter_passes` in `_ClassicLayoutSpec`), so pass-counting parity looks correct on inspection.

**Isolated-node tie-break:** dagua's `_neighbor_barycenters`
(`dagua/layout/ops/sugiyama.py:1400-1421`, `use_incidence_barycenters=True` branch) sets isolated
nodes (no neighbors in the reference layer) to
`order_index.get(layer_position, float(layer_position))` -- i.e. their CURRENT layer_position
index as a synthetic barycenter. igraph's `igraph_i_layout_sugiyama_calculate_barycenters`
(`sugiyama.c:697-699`, per background read) sets isolated nodes' barycenter to their
**current X COORDINATE** (`MATRIX(*layout, i, 0)`), not their layer-relative ORDER INDEX. These
are similar in spirit (both "stay where you are") but not identical when X-coordinates and order
indices diverge across sweeps for other nodes in the same layer -- specifically, dagua's version
uses ordinal position (0,1,2,...) while igraph's version uses actual x-coordinate distance, which
means isolated nodes interleave differently relative to non-isolated nodes once non-isolated
nodes' barycenters span a wider or narrower numeric range than `[0, layer_size)`. Since
`_igraph_sort_indices` sorts ALL nodes in the layer together by this score, a mismatched numeric
scale for isolated nodes' scores can change their sorted position relative to non-isolated
neighbors, WHICH bleeds into every subsequent pass (compounding).

Additionally, dagua's ordering stage runs on the pure integer LAYER-INDEX space throughout (no
real X-coordinates exist until the final `_CoordinateAssignment` op runs AFTER ordering
completes), so an exact port of igraph's "current X coordinate" isolated-node rule is not
directly pluggable without restructuring -- igraph's ordering stage in principle DOES have access
to provisional X-coordinates because `igraph_layout_sugiyama` interleaves ordering and placement
differently per-component (each component gets ordering THEN placement, sequentially, before the
next component starts) -- but within ONE component's ordering loop, X-coordinates aren't
recomputed per-sweep either (confirmed: `igraph_i_layout_sugiyama_place_nodes_horizontally` is
called once, AFTER the `while (changed...)` ordering loop terminates, per the background read of
lines 462-472 preceding `place_nodes_horizontally`'s call site) -- so "current X coordinate" during
ordering must mean the INITIAL X placement (the "first-seen counter" `xs[layer]++` from
`sugiyama.c:735-742`, since no BK pass has run yet), which is numerically identical in shape to
dagua's layer-position-index approach. **This weakens the hypothesis** -- both are likely
equivalent in practice for the first pass, but MAY diverge on later passes once non-isolated
nodes' order changes but igraph's isolated-node values stay pinned to a value that was never
updated (a stale-reference bug/feature in igraph itself), which dagua's `order_index.get(...)`
call recomputes fresh every pass. This distinction needs direct verification against a specific
failing combo's per-pass isolated-node count before proposing a fix.

**Cheapest decisive experiment:** for `hexagonal_lattice_42::classic_sugiyama_default` (small,
n<=42, `cross.D=3` vs `cross.R=7` -- dagua BETTER on crossings, a red flag per the sprint's
dagua-better asymmetry finding, suggesting a genuine algorithmic difference, not FP noise), dump
per-node degree-0-in-reference-layer counts and manually trace 2-3 ordering passes against a
from-source re-implementation of igraph's stale-X rule. Est. runtime: 20-30 min of scripted
tracing (NOT a benchmark run -- pure Python simulation using the existing dagua ops against the
known graph topology). Given the 45-minute budget, this was NOT run in this pass; flagged as the
top follow-up experiment.

**Expected impact:** if confirmed and fixed, likely closes a meaningful fraction of the 42
both-legs-diverge igraph-family combos, since isolated-node handling compounds across
`barycenter_passes` (4/24/48 depending on variant) and affects EVERY graph with any layer
containing degree-imbalanced nodes (common in DAG-shaped benchmark graphs with source/sink
fan-out).

**RISK:** MEDIUM if the fix changes `_neighbor_barycenters`' isolated-node formula --
`use_incidence_barycenters=True` is used ONLY by the igraph fidelity mode
(`_BarycenterOrderingConfig.use_incidence_barycenters`, set from `fidelity_mode="igraph"` only),
so this is naturally scoped away from default/graphviz paths -- but it IS the same function used
across all 6 igraph-family variants (default/wide/tight/passes4/passes48), so any currently-passing
igraph-family combos (not in the 129-target list) must be re-verified after a fix here to avoid
a repeat of the maxent/classical_mds blanket-fix-broke-bit-exact-combos pattern.

## 3. Crossings-only failures (11 combos) -- explanation

Per the bucket brief's question: are these ordering-stage divergence with coincidentally-matching
stress? **Answer: yes, confirmed by the JSON evidence.** All 11 have `battery_stress.equiv=True`
and `np.equiv=True` but `cross.equiv=False`, with small absolute crossing-count deltas (9 vs 8,
415 vs 387, 335 vs 435, 111 vs 108, 9380 vs 8372) -- these are cases where the FINAL layout is
statistically indistinguishable on stress/neighborhood-preservation (which are somewhat
insensitive to small local reorderings) but the exact discrete crossing count differs by a small
margin, consistent with the mincross/barycenter ordering stage landing in a different
(similar-quality) local optimum rather than a systematically wrong algorithm. This is fully
consistent with Finding 3 (mincross init_order mismatch) for the one graphviz_fidelity example
(`weighted_karate_34::classic_sugiyama_graphviz_fidelity`, 111 vs 108) and Finding 6 (igraph
ordering-stage tie-break/isolated-node handling) for the 10 igraph-family examples
(`multiscale_skip_cascade` x5 variants, `random_dag_200::wide`, `random_dag_50::wide`,
`regular_4_40` x3 variants). Landing Finding 3's `build_ranks`-equivalent BFS init-order (for the
1 graphviz combo) and Finding 6's isolated-node fix (for the 10 igraph combos, pending the
decisive experiment) should close most/all of these 11, since they are the SMALLEST-margin
divergences in the whole bucket (single-digit-to-low-percentage crossing count deltas) --
this is the cheapest subset to verify a fix against.

## 4. Explicit fraction estimate

If Findings 1 (x-coord network-simplex port) + 2 (omega weight table) + 3 (mincross init_order)
land faithfully for the graphviz family:
- Expect the 39 x-coord-only-signature combos to become bit-exact or distributionally matched
  (crossings already match; only x-coord was wrong) -- HIGH confidence, this is a direct
  structural fix for a confirmed root cause.
- Expect a good fraction (not all) of the remaining 19 (18 both-legs-fail + 1 crossings-only) to
  also close, since Findings 2+3 target exactly the ordering-stage mechanisms that would explain
  a crossing-count divergence. Conservative estimate: 12-15 of 19.
- **Graphviz family total estimate: ~51-54 of 58 (88-93%) become bit-exact/distributionally
  matched.**

If Findings 4 (confirmed no-op, informational) + 5 (BK detail fixes) + 6 (isolated-node/ordering
fix, HYPOTHESIS pending decisive experiment) land for the igraph family:
- Finding 5 (median-of-4/anchor/component-order details) should close some fraction of the 19
  igraph x-coord-only-signature combos: estimate 8-12 of 19 (the ones where the anchor-selection
  or component-ordering guess is correct; graphs like `kitchen_sink_platform_graph` with
  disconnected components are the most likely wins since component-order is a clean, testable
  hypothesis).
- Finding 6, IF confirmed (not yet verified -- this is the single largest uncertainty in this
  report), could close a large fraction of the 42 both-legs-fail combos, since isolated-node
  mishandling compounds across many passes and would affect most DAG-shaped graphs with
  source/sink imbalance. Conservative estimate assuming confirmation: 20-28 of 42. If NOT
  confirmed (i.e. dagua's igraph ordering is already correct and the divergence is elsewhere,
  e.g. a still-unidentified rank-assignment tie-break), this fraction could be much lower and
  needs a second investigation round.
- **Igraph family total estimate (wide range reflecting Finding 6 uncertainty): ~28-40 of 71
  (39-56%) become bit-exact/distributionally matched.** This is the weakest part of this
  report's estimate -- Finding 6 needs the decisive experiment (per-pass isolated-node trace)
  before this fraction can be tightened.

**Combined bucket estimate: roughly 79-94 of 129 (61-73%) become bit-exact/distributionally
matched** if all six findings' fix sketches land as scoped, with the graphviz family much more
confidently estimated than the igraph family.

For the large-graph tail (ba_500, mentioned in the bucket brief, NOT in the 129-target
<=300-node list -- confirmed via `r73_RESULTS.md:34`, `22344 vs 2805 crossings`, and
`r74_findings/r74_CX5_findings.md` C1 category "sugiyama large/slow compute frontier",
currently excluded from scoring due to TIMEOUT/insufficient-seeds, not yet a scored divergent
combo): Finding 3 (mincross init_order + iteration efficiency) is the primary lever, since an
8x crossing-count gap on a 500-node graph is much larger than any single-optimum-vs-another
noise floor would produce and is consistent with mincross failing to converge from a poor start
point within its iteration budget. This is separate from and in addition to the correctness
fixes above -- ba_500 also needs the C1-flagged performance work (iterative-not-recursive
sugiyama ordering loop, already partly done per commit 6563d98 for cycle-breaking) to even
produce 100 valid seeds for scoring before a correctness verdict can be reached.

## 5. What CANNOT be fixed by these ports

- **`weighted_karate_34::classic_sugiyama_graphviz_fidelity`** (crossings-only, 111 vs 108): a
  3-crossing difference on a 34-node real-world graph (Zachary's karate club) is within plausible
  degenerate-tie-break range even after Finding 3's init_order fix -- network simplex ranking AND
  mincross ordering both have documented degenerate optima (ns.c's rotating `S_i` search cursor
  and adjacency-list-order-dependent tie-breaks per the background research). If the exact
  adjacency-list construction order in dagua's expanded graph doesn't match graphviz's
  edge-creation order bit-for-bit (plausible residual even after Findings 1-3, since dagua's
  internal data structures build edges in a different code path than graphviz's `agedge()` C API
  call sequence), this could remain a genuine "different but equally optimal" local optimum --
  FP-chaos-adjacent, not exactly the same category as summation-order chaos, but a related
  "combinatorial tie-break sensitive to exact traversal order" floor. Flagging as a likely
  residual rather than confirmed-unfixable; would need the decisive per-edge adjacency-list-order
  diff to be sure.
- **`heavy_tail_weights_50` (both graphviz_fidelity and igraph default variants)**: this graph has
  meaningfully divergent battery_stress even by igraph-family standards
  (`D=0.341,R=0.371` for default -- among the larger deltas in the igraph set) AND is explicitly
  weighted (heavy-tail edge weights). Both the graphviz omega-table fix (Finding 2) and any
  edge-weight-dependent ordering logic interact with edge weights in ways not fully verified in
  this pass (edge-weight handling in igraph's barycenter/ranking LP objective was read and
  confirmed correct for RANKING by r74's fix, but weighted-graph interaction with the BK
  x-coordinate stage and the barycenter ordering's `weighted_sum`/`total_weight` logic in
  `_neighbor_barycenters` non-incidence branch, `sugiyama.py:1407-1416`, was not independently
  stress-tested against igraph's actual weighted-barycenter formula in this pass -- flagged as an
  open gap requiring a dedicated weighted-graph read of `sugiyama.c`'s barycenter weight handling,
  which the background agent's report did not explicitly cover).
- **Disconnected-graph combos beyond the component-order hypothesis** (Finding 5): if
  `_weak_components`' DFS traversal order genuinely doesn't match `igraph_connected_components`'s
  internal order even after inspection, dagua's `_layout_igraph_packed_components` recursive
  per-component layout approach (`sugiyama.py:368-457`) still has to independently re-derive
  ranks/order per component from scratch (calling `layout_sugiyama_pipeline` recursively) --
  any residual per-component seed/tie-break sensitivity compounds across N components. Not
  independently verified beyond the component-order hypothesis in this pass.
- **The 45-minute experiment budget did not permit running Finding 6's decisive experiment**
  (per-pass isolated-node trace for `hexagonal_lattice_42` or `regular_4_40`) -- this is the
  single largest unresolved uncertainty in this report and directly gates the igraph-family
  fraction estimate's confidence interval (39-56% is a wide range specifically because of this).
- **No FP-chaos evidence was gathered for any combo in this pass** (no 1-ULP perturbation
  experiments were run, per the guardrail requirement for "floor/unfixable" claims) -- all
  "cannot be fixed" statements above are HYPOTHESES about combinatorial tie-break sensitivity,
  not confirmed floors. A true floor claim would require the decisive experiments listed above
  plus explicit ULP-perturbation testing on the residual combos after Findings 1-6 are actually
  implemented and re-scored.
