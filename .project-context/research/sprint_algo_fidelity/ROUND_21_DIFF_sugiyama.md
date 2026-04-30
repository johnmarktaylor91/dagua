# Round 21 Adversarial Diff: `classic_sugiyama` vs `igraph_sugiyama`

Date: 2026-04-30
Branch: `develop`
Scope: diagnosis-only, no source edits.

## 1. Files Read

### Dagua implementation and wiring

- `dagua/layout/AGENTS.md:1-122` -- layout architecture, ops pipeline conventions, RNG expectations, and testing notes.
- `dagua/layout/ops/pipelines/sugiyama.py:1-218` -- public pipeline construction and dispatch for `layout_sugiyama_pipeline`.
- `dagua/layout/ops/sugiyama.py:1-1885` -- all private Sugiyama ops: validation, acyclic edge preparation, layer assignment, dummy expansion, barycenter ordering, Brandes-Kopf style coordinate assignment, and optional edge routing.
- `dagua/layout/cycle.py:1-247` -- cycle-breaking helpers used by Dagua Sugiyama through `make_acyclic_robust`.
- `dagua/eval/variants.py:1-260`, `dagua/eval/variants.py:830-930`, `dagua/eval/variants.py:1818-1856` -- variant registry, pairing helpers, Sugiyama variants, and stochasticity flags.
- `dagua/eval/competitors/classic_competitor.py:1-340`, `dagua/eval/competitors/classic_competitor.py:900-980` -- generic classic adapter and `ClassicSugiyama` adapter.
- `dagua/eval/competitors/igraph_competitor.py:1-285` -- igraph adapter, graph conversion, coordinate scaling, and `IgraphSugiyama`.
- `dagua/eval/competitors/base.py` was discovered by search as the registration base, but not needed for this specific algorithm diff beyond the imports cited in `classic_competitor.py:14-18` and `igraph_competitor.py:12`.

### Reference implementation

- `/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:1-1309` -- full C implementation of `igraph_layout_sugiyama`, including comments that describe the intended algorithm, vertical layering, per-component dummy expansion, barycenter ordering, Brandes-Kopf coordinate assignment, routing, and compaction.
- Python wrapper introspection for installed `python-igraph 1.0.0` was used to confirm wrapper defaults: `Graph.layout_sugiyama(graph, layers=None, weights=None, hgap=1, vgap=1, maxiter=100)`. This is runtime API evidence, not a source file. The corresponding C defaults appear indirectly through the C parameter documentation at `/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:265-275`.

### Existing analysis

- `eval_output/fidelity_report/report.md:70-105` -- current mega-run verdict rows, including all five Sugiyama variants.
- `.project-context/research/sprint_algo_fidelity/algo_fidelity_SUMMARY.md:1-70` and `.project-context/research/sprint_algo_fidelity/algo_fidelity_SUMMARY.md:270-292` -- sprint summary and historical Round 3 dot spacing fix.

## 2. Overall Pipeline Structure

### Dagua high-level flow

Dagua's Sugiyama family is a composable ops pipeline. `build_sugiyama_pipeline` wires these operations in order: input validation, spacing storage, node-size resolution, cycle breaking, layer assignment, dummy-node expansion, neighbor structure construction, barycenter sweeps, coordinate assignment, and optional route reconstruction (`dagua/layout/ops/pipelines/sugiyama.py:36-90`). The public `layout_sugiyama_pipeline` then builds a `LayoutProblem`, executes the pipeline, slices/returns original-node positions, and optionally returns trace snapshots or edge routes (`dagua/layout/ops/pipelines/sugiyama.py:93-215`).

Important Dagua flow details:

- Defaults for direct calls are `_DOT_DEFAULT_RANK_CENTER_SEP = 72.0` and `_DOT_DEFAULT_NODE_SEP = 18.0`, unless variant configs override them (`dagua/layout/ops/pipelines/sugiyama.py:29-30`, `dagua/layout/ops/pipelines/sugiyama.py:166-171`).
- The benchmark variants override this to igraph-scale spacing: `rank_sep=1.0`, `node_sep=1.0`, mapped to `vgap=1.0`, `hgap=1.0` on the igraph side (`dagua/eval/variants.py:858-912`).
- The pipeline writes the expanded positions for routes but returns only `expanded_positions[:problem.num_nodes]` as `state.pos` (`dagua/layout/ops/sugiyama.py:1786-1801`).
- Dagua treats `seed` as API compatibility for Sugiyama; `_barycenter_ordering` explicitly deletes it and no RNG is used (`dagua/layout/ops/sugiyama.py:497-512`, `dagua/layout/ops/sugiyama.py:551-558`).

### igraph high-level flow

The reference implementation is `igraph_layout_sugiyama` in C (`/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:278-282`). The implementation comment enumerates the intended full Sugiyama stages: cycle removal, layering, weak-component extraction, compaction/promotion of layering, dummy-node insertion, ordering, Brandes-Kopf horizontal coordinates, and routing (`/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:79-135`).

Actual reference flow:

- Validate layer vector length, resize result, initialize membership and `layer_to_y` (`/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:295-309`).
- If layers are absent, compute vertical layering by `igraph_i_layout_sugiyama_place_nodes_vertically`; otherwise copy caller-supplied layers (`/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:311-320`).
- Normalize layer IDs and build `layer_to_y` as original layer value times `vgap` while eliminating empty layers (`/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:322-340`).
- Compute weakly connected components on the original graph (`/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:342-343`) and process each component separately (`/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:345-550`).
- For each component, remap original node IDs, construct a per-component expanded graph with dummy nodes, ignore same-layer edges, flip upward edges, run barycenter ordering, run Brandes-Kopf coordinate assignment, write real vertices to `res`, optionally write dummy vertices to routing, then shift `dx` for the next component (`/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:348-533`).

### Structural divergences

1. **Component handling is different.** igraph extracts weak components and lays out each separately with an accumulating `dx` offset (`/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:342-347`, `/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:520-527`). Dagua lays out all nodes in one expanded layered graph; `_group_nodes_by_layer` groups every node globally by layer (`dagua/layout/ops/sugiyama.py:223-246`) and `_expand_long_edges_with_dummy_nodes` creates a single expanded graph (`dagua/layout/ops/sugiyama.py:333-394`). This can change disconnected graphs and weakly disconnected DAGs.

2. **Layering is not equivalent.** Dagua uses DFS/greedy cycle breaking, longest-path layering, and a simple successor-based promotion (`dagua/layout/cycle.py:196-247`, `dagua/layout/ops/sugiyama.py:172-220`, `dagua/layout/ops/sugiyama.py:249-305`). igraph uses GLPK/network-simplex-style integer programming for directed graphs with `<=1000` nodes when available, otherwise Eades feedback-arc layering; undirected graphs use a separate undirected feedback path (`/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:552-675`).

3. **Ordering convergence differs.** Dagua runs exactly `barycenter_passes` full down/up sweeps (`dagua/layout/ops/sugiyama.py:558-596`). igraph runs until no order changes or `iter >= maxiter` (`/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:749-828`).

4. **Coordinate assignment is similar in intent but not bit-identical.** Both implement four Brandes-Kopf orientations, align to the narrowest orientation, and take a median (`dagua/layout/ops/sugiyama.py:760-809`; `/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:981-1029`). Dagua centers final coordinates around zero (`dagua/layout/ops/sugiyama.py:1294-1311`); igraph leaves component coordinates in a left-anchored frame and uses `dx` offsets (`/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:477-527`).

## 3. Energy / Loss / Objective

Sugiyama is not a force/energy optimizer in either implementation; the objectives are discrete/heuristic. The relevant "objective" terms are:

### Cycle breaking / layering objective

- Dagua: `make_acyclic_robust` first reverses DFS back edges (`dagua/layout/cycle.py:13-70`, `dagua/layout/cycle.py:227-232`), then falls back to `_greedy_fas` if the result is still cyclic (`dagua/layout/cycle.py:130-193`). `_greedy_fas` scores active nodes by `(out_degree - in_degree, -node_idx)` and flips edges that go against the resulting order (`dagua/layout/cycle.py:162-193`). It ignores edge weights entirely.
- igraph: for directed graphs with GLPK and `no_of_nodes <= 1000`, it finds an approximate feedback edge set, computes in/out strengths after removing feedback edges, and solves an integer linear program minimizing a linear objective over layer variables (`/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:563-616`). It adds one row per edge with `layer[to] - layer[from] >= 1` for normal edges and `<= -1` for feedback edges (`/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:621-646`). Weights affect feedback selection and strength coefficients (`/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:584-600`).

Expected impact: high on cyclic graphs, weighted graphs, and graphs where GLPK is enabled. The installed Python wrapper doc says directed cyclic graphs use Eades feedback arcs then longest-path layering, but the C reference can take the GLPK branch depending on build flags (`/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:563-675`).

### Crossing reduction objective

- Dagua computes weighted neighbor barycenters. For a node, it accumulates `weight * order_index[neighbor]`, divides by total edge weight, and falls back to an unweighted average when total weight is nonpositive (`dagua/layout/ops/sugiyama.py:623-639`). It sorts layers using Python's stable sort with key `barycenters[node]` (`dagua/layout/ops/sugiyama.py:561-579`).
- igraph computes unweighted barycenters from current X/order coordinates and divides by neighbor count, using `IGRAPH_NO_LOOPS` and `IGRAPH_MULTIPLE`, so parallel edges remain multiple neighbor entries but there is no arbitrary float edge-weight multiplier in this ordering stage (`/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:677-712`). Sorting is through `igraph_vector_sort_ind` (`/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:770-775`, `/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:804-809`).

Expected impact: high for weighted graphs if `edge_weights` reaches Dagua; medium for multiedges depending on whether the Dagua graph preserves parallel edges as repeated `edge_index` columns and how weight maps aggregate them. Dagua aggregates weights by neighbor key (`dagua/layout/ops/sugiyama.py:480-494`), while igraph's `IGRAPH_MULTIPLE` keeps multiple neighbor incidences (`/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:692-705`).

### Horizontal coordinate objective

- Dagua performs four orientation passes (`ul`, `ur`, `dl`, `dr`), finds conflicts, aligns vertical blocks, compacts with `_minimum_separation`, aligns all orientations to the minimum span, takes the median, and centers the final coordinates (`dagua/layout/ops/sugiyama.py:760-809`, `dagua/layout/ops/sugiyama.py:1191-1218`, `dagua/layout/ops/sugiyama.py:1220-1311`).
- igraph performs the Brandes-Kopf procedure with four `reverse/align_right` combinations, chooses the minimum-width alignment as anchor, aligns leftmost/rightmost alignments to that anchor, and takes the median of four coordinates (`/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:839-1047`). Its compaction uses exactly `hgap` as the separation between adjacent vertices (`/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:1249-1301`).

Critical coordinate-objective divergence: Dagua includes node widths in the separation, returning `(left_width + right_width) / 2 + node_sep` (`dagua/layout/ops/sugiyama.py:1191-1218`). igraph's C compaction uses `hgap` only (`/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:1284-1294`). In the current benchmarks, igraph receives graph topology only through `_graph_to_igraph`; node sizes are not exported (`dagua/eval/competitors/igraph_competitor.py:53-76`), while Dagua receives `graph.node_sizes` in `ClassicSugiyama.layout` (`dagua/eval/competitors/classic_competitor.py:951-956`). If benchmark graphs have nonzero `node_sizes`, this is a direct residual RMSD source.

## 4. Force / Gradient Computation

No force or gradient loop applies. Dagua uses tensor containers and Python list operations, but no `torch.optim`, autograd, or force accumulation in Sugiyama (`dagua/layout/ops/pipelines/sugiyama.py:73-90`; `dagua/layout/ops/sugiyama.py:558-596`; `dagua/layout/ops/sugiyama.py:662-721`). igraph uses C vectors, matrices, sorting, graph traversal, and optional GLPK simplex/IP-style layering; there is no continuous force/gradient solver in `sugiyama.c` (`/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:552-675`, `/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:719-837`, `/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:858-1301`).

The only continuous numeric formulas are barycenter averages and coordinate compaction. Dagua barycenter formula is `sum(weight * order) / sum(weight)` (`dagua/layout/ops/sugiyama.py:627-635`). igraph barycenter formula is `sum(layout[neighbor, 0]) / m` (`/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:701-705`). Dagua compaction formula is `left_x + (left_width + right_width)/2 + node_sep` when blocks share a sink (`dagua/layout/ops/sugiyama.py:1170-1185`, `dagua/layout/ops/sugiyama.py:1191-1218`). igraph compaction formula is `xs[u] + hgap` (`/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:1288-1294`).

## 5. Initialization

### Dagua

- Initial layer assignment starts from `_longest_path_layering`: Kahn traversal with a min-heap of ready nodes, so lower node IDs are processed first when ties occur (`dagua/layout/ops/sugiyama.py:192-220`).
- Nodes are grouped into layers in original node ID order (`dagua/layout/ops/sugiyama.py:241-246`).
- Expanded layers append dummy nodes during edge iteration order (`dagua/layout/ops/sugiyama.py:344-368`).
- Initial ordering for barycenter sweeps is `ordered_layers = [sorted(layer) for layer in layers]`, which sorts all real and dummy node IDs numerically (`dagua/layout/ops/sugiyama.py:551-555`).
- No RNG initializes positions or tie-breaking; `seed` is deleted in `_barycenter_ordering` (`dagua/layout/ops/sugiyama.py:551-558`).

### igraph

- If layers are not provided, vertical placement is computed before component expansion (`/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:311-320`, `/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:552-675`).
- Layer normalization sorts by layer values and keeps per-layer membership through original IDs (`/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:322-340`).
- Components are processed by original node ID order, remapping component-local IDs in first-seen order (`/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:368-377`).
- Dummy nodes are appended while scanning incident outgoing edges per component (`/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:381-433`).
- Horizontal order initialization sets `layout[i, 0]` to the first-seen index within each layer by iterating expanded vertex IDs from `0` to `no_of_vertices-1` (`/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:733-742`).

Residual divergence: Dagua's `sorted(layer)` can reorder dummy nodes relative to igraph's expanded vertex first-seen order when dummy IDs interleave differently after whole-graph expansion versus per-component expansion. This affects subsequent barycenters even when all high-level parameters match.

## 6. Iteration / Convergence

### Dagua

- `barycenter_passes` default in the pipeline function is `24` (`dagua/layout/ops/pipelines/sugiyama.py:36-43`, `dagua/layout/ops/pipelines/sugiyama.py:93-104`).
- The pipeline always executes exactly `range(num_passes)`; no convergence test exists (`dagua/layout/ops/sugiyama.py:558-596`).
- The benchmark variants map `barycenter_passes` to igraph `maxiter` for 4, 24, and 48 sweeps (`dagua/eval/variants.py:858-912`).

### igraph

- C documentation says `maxiter` is the maximum number of crossing-minimization iterations and `100` is reasonable (`/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:270-272`).
- The installed Python wrapper default is `maxiter=100`.
- The loop is `while (changed && iter < maxiter)`, with `changed` set when any layer order changes (`/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:749-828`).
- In each iteration, igraph performs one downward phase and one upward phase (`/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:754-787`, `/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:789-820`).

Residual divergence: Dagua may continue to sort after igraph would stop. If sorting is stable and all order keys are equal, extra passes can be no-ops, but weighted barycenters and whole-graph component interactions can cause Dagua to keep transforming state where igraph has already converged or vice versa.

## 7. Hyperparameter Alignment Table

| Parameter / behavior | Dagua default / behavior | igraph default / behavior | Match? | Evidence |
|---|---:|---:|---|---|
| Direct-call vertical spacing | `rank_sep=72.0` | `vgap=1.0` in Python wrapper / C param | No for direct calls; variants align | Dagua defaults at `dagua/layout/ops/pipelines/sugiyama.py:29-30`, `dagua/layout/ops/pipelines/sugiyama.py:166-171`; C docs at `/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:265-275`; variants at `dagua/eval/variants.py:858-912` |
| Variant vertical spacing | `rank_sep=1.0`, `0.5`, `2.0` | `vgap=1.0`, `0.5`, `2.0` | Yes | `dagua/eval/variants.py:858-912` |
| Direct-call horizontal spacing | `node_sep=18.0` | `hgap=1.0` in Python wrapper / C param | No for direct calls; variants align | `dagua/layout/ops/pipelines/sugiyama.py:29-30`, `dagua/layout/ops/pipelines/sugiyama.py:166-171`; `/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:267-270`; `dagua/eval/variants.py:858-912` |
| Variant horizontal spacing | `node_sep=1.0`, `0.5`, `2.0` | `hgap=1.0`, `0.5`, `2.0` | Yes in param value, not in coordinate formula when Dagua node sizes nonzero | `dagua/eval/variants.py:858-912`; Dagua formula `dagua/layout/ops/sugiyama.py:1191-1218`; igraph formula `/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:1284-1294` |
| Crossing sweeps default | `barycenter_passes=24` | `maxiter=100` wrapper/default C recommendation | No for base default; variants align max values | `dagua/layout/ops/pipelines/sugiyama.py:39-40`, `dagua/layout/ops/pipelines/sugiyama.py:100-102`; `/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:270-272`; `dagua/eval/variants.py:858-912` |
| Crossing sweep stopping | Fixed pass count | Stop on unchanged order or maxiter | No | `dagua/layout/ops/sugiyama.py:558-596`; `/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:749-828` |
| Seed | Accepted, ignored | No RNG for Sugiyama path; `uses_igraph_rng=False` in adapter | Functionally yes, metadata no | Dagua deletes seed `dagua/layout/ops/sugiyama.py:551-558`; igraph adapter `dagua/eval/competitors/igraph_competitor.py:195-202`; stochasticity flags disagree at `dagua/eval/variants.py:1830-1849` |
| Edge weights in cycle breaking | Ignored by `make_acyclic_robust` | Used for feedback arc selection/strengths | No | `dagua/layout/cycle.py:196-247`; `/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:273-275`, `/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:584-600` |
| Edge weights in barycenters | Weighted average by accumulated edge weight | Unweighted average over neighbor incidences | No | `dagua/layout/ops/sugiyama.py:458-494`, `dagua/layout/ops/sugiyama.py:623-639`; `/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:691-705` |
| Multiedges in barycenters | Parallel edges to same neighbor collapse into one dict key with summed weight | `IGRAPH_MULTIPLE` preserves duplicate neighbor incidences | Partial | `dagua/layout/ops/sugiyama.py:480-494`; `/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:692-705` |
| Self-loops | Marked reversed but retained in `acyclic_edges`; layering can still see them | Ignored in component layered graph and neighbor queries use `IGRAPH_NO_LOOPS` | No / bug risk | `dagua/layout/cycle.py:224-247`; Dagua expansion uses all acyclic edges `dagua/layout/ops/sugiyama.py:1545-1550`; igraph ignores loops at `/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:387-388`, `/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:692-695` |
| Same-layer edges | Dagua's auto layering tries to avoid same-layer edges in acyclic graph; if present in input via future layer override, expansion would add direct edge | igraph explicitly drops same-layer edges from layered graph | Not currently exposed, future risk | `dagua/layout/ops/sugiyama.py:308-394`; `/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:400-403` |
| Disconnected components | One global layout | Per weak component with `dx` packing | No | Dagua `dagua/layout/ops/sugiyama.py:333-394`; igraph `/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:342-347`, `/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:520-527` |
| Coordinate frame | Center final x coordinates around zero | Left-anchored per component plus `dx`; y from `layer_to_y` | No, mostly removed by Procrustes except component packing | Dagua `dagua/layout/ops/sugiyama.py:1294-1311`; igraph `/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:477-527` |
| Numeric dtype | Dagua positions and sizes are `torch.float32` | igraph `igraph_real_t`, typically double | No | Dagua `dagua/layout/ops/sugiyama.py:141-143`, `dagua/layout/ops/sugiyama.py:702-721`; igraph `igraph_real_t` uses throughout e.g. `/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:211-217`, `/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:858-860` |
| Returned dummy coordinates | Hidden unless `return_edge_routes`; state pos sliced to original nodes | Python docs say layout may include dummy rows; adapter truncates to graph nodes | Benchmark effectively aligns original nodes only | Dagua `dagua/layout/ops/sugiyama.py:1797-1801`; igraph adapter truncates `dagua/eval/competitors/igraph_competitor.py:79-99` |
| Coordinate scaling in adapter | Native Dagua units | igraph coordinates multiplied by 50 in adapter | No in raw values, Procrustes mostly absorbs | `dagua/eval/competitors/igraph_competitor.py:79-99` |

## 8. Edge Cases

### Self-loops

Likely Dagua bug. `make_acyclic_robust` detects self-loops, processes non-self-loop edges recursively, then reattaches self-loops unchanged and marks them reversed (`dagua/layout/cycle.py:224-247`). `_AssignLayers` then calls `_longest_path_layering` on the full `acyclic_edges` (`dagua/layout/ops/sugiyama.py:1485-1494`). `_longest_path_layering` increments `in_degree[dst]` for every edge and raises if not all nodes are processed (`dagua/layout/ops/sugiyama.py:192-220`), so a self-loop can leave the graph cyclic from the layering perspective. igraph excludes loops while building the layered graph (`/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:387-388`) and uses `IGRAPH_NO_LOOPS` in barycenter neighbor queries (`/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:692-695`, `/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:1103-1106`).

### Multi-edges

igraph preserves duplicate incidences in neighbor queries by passing `IGRAPH_MULTIPLE` (`/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:692-705`, `/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:1103-1106`). Dagua parent/child adjacency lists keep repeated neighbor entries (`dagua/layout/ops/sugiyama.py:430-455`), but `parent_weights` and `child_weights` aggregate by neighbor key (`dagua/layout/ops/sugiyama.py:480-494`). `_neighbor_barycenters` then loops over the repeated neighbor list and fetches the already-accumulated neighbor weight for each duplicate (`dagua/layout/ops/sugiyama.py:623-639`). For two parallel unweighted edges to the same neighbor, this uses `2` as the weight twice, yielding weighted sum `4*x` and total `4`; the barycenter remains `x`. For mixed parallel edges to several neighbors, duplicates plus aggregate weights can square multiplicity effects relative to igraph's incidence average.

### Disconnected components

igraph lays out weak components independently and packs them along X (`/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:342-347`, `/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:520-527`). Dagua uses one global layering and one coordinate assignment. Isolated nodes all land in layer 0 and are compacted in one row with all other layer-0 nodes (`dagua/layout/ops/sugiyama.py:201-220`, `dagua/layout/ops/sugiyama.py:223-246`, `dagua/layout/ops/sugiyama.py:662-721`). This is likely the largest remaining divergence for graphs with disconnected components after Procrustes.

### Weighted edges

The igraph API accepts weights and uses them in cycle breaking/layering only according to C docs (`/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:273-275`) and C code (`/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:584-600`). The Dagua igraph adapter exports edge weights to `igraph` as an edge attribute (`dagua/eval/competitors/igraph_competitor.py:70-76`), but `IgraphSugiyama.layout_kwargs = {}` and variants do not pass `weights`, so python-igraph may not consume the attribute unless wrapper defaults infer it; based on signature, `weights=None` by default. Dagua's classic adapter passes no `edge_weights` in `ClassicSugiyama.layout` (`dagua/eval/competitors/classic_competitor.py:951-956`), while the generic `_ClassicBase` variant path can pass only variant params, not graph weights unless the pipeline receives them through a direct call (`dagua/eval/competitors/classic_competitor.py:54-97`). In direct API use, Dagua can weight barycenters (`dagua/layout/ops/sugiyama.py:458-494`, `dagua/layout/ops/sugiyama.py:623-639`), diverging from igraph ordering.

### Empty graph

Dagua handles `num_nodes == 0` in grouping and coordinate assignment (`dagua/layout/ops/sugiyama.py:238-246`, `dagua/layout/ops/sugiyama.py:708-721`). `edge_index` with no edges returns an empty reversed mask (`dagua/layout/ops/sugiyama.py:166-169`). igraph resizes the result to `no_of_nodes x 2`, handles empty membership as zero layers, and vertical placement returns zero membership for no edges (`/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:299-309`, `/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:152-156`, `/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:552-560`). Empty graph behavior should be equivalent for `N=0`; for `N>0,E=0`, component packing differs because igraph treats every isolated vertex as its own weak component, while Dagua puts all nodes in a single layer-0 row.

## 9. Numerical Precision

- Dagua resolves node sizes to CPU `torch.float32` (`dagua/layout/ops/sugiyama.py:126-143`) and creates final `positions` as `torch.float32` (`dagua/layout/ops/sugiyama.py:702-721`). Expanded edge weights are also `torch.float32` (`dagua/layout/ops/sugiyama.py:385-388`).
- Dagua's intermediate x-coordinate calculations are Python floats in lists (`dagua/layout/ops/sugiyama.py:724-809`, `dagua/layout/ops/sugiyama.py:1038-1100`) before conversion to `torch.float32` (`dagua/layout/ops/sugiyama.py:720`).
- igraph uses `igraph_real_t` in coordinate math (`/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:211-217`, `/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:858-860`, `/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:990-1029`). In normal igraph builds this is double precision.
- Summation order differs in barycenter calculation. Dagua iterates `neighbors_by_node[node]` list order from `edge_index` construction and then dict weights (`dagua/layout/ops/sugiyama.py:623-639`). igraph iterates the neighbor vector returned by `igraph_neighbors` (`/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:691-705`).

For sub-percent residuals on simple DAGs, precision is likely low impact after Procrustes. For tie-heavy layers, tiny ordering differences near equal barycenters can be amplified into discrete swaps.

## 10. RNG Semantics

Dagua's Sugiyama `seed` does not produce any torch sequence because no torch RNG is used. `_BarycenterOrderingConfig` stores `seed` for API compatibility (`dagua/layout/ops/sugiyama.py:53-70`), the pipeline forwards it (`dagua/layout/ops/pipelines/sugiyama.py:81-85`, `dagua/layout/ops/pipelines/sugiyama.py:187-194`), but `_barycenter_ordering` executes `del seed` (`dagua/layout/ops/sugiyama.py:551-558`).

The igraph adapter only routes seeds through `_igraph_rng_seed` when `uses_igraph_rng` is true (`dagua/eval/competitors/igraph_competitor.py:18-50`, `dagua/eval/competitors/igraph_competitor.py:177-180`). `IgraphSugiyama` does not set `uses_igraph_rng`, so it inherits `False` (`dagua/eval/competitors/igraph_competitor.py:102-109`, `dagua/eval/competitors/igraph_competitor.py:195-202`). The reference C Sugiyama implementation has no RNG calls in the read source.

Answer to the specific question: no, Dagua's torch seed cannot produce the same sequence as the reference RNG because neither side uses a stochastic sequence for Sugiyama, and Dagua does not touch the torch RNG in this pipeline. The benchmark metadata incorrectly marks `classic_sugiyama` as stochastic while marking `igraph_sugiyama` as deterministic (`dagua/eval/variants.py:1830-1849`), and every Sugiyama variant is also marked `is_stochastic=True` (`dagua/eval/variants.py:858-912`). That metadata can cause unnecessary multi-seed repeats, but it should not change coordinates.

## 11. Edge-Case Bugs

1. **Self-loops are not actually filtered before Dagua layering.** Dagua cycle handling comments say the caller can filter self-loops (`dagua/layout/cycle.py:202-206`), but `_PrepareAcyclicEdges` stores full `acyclic_edges` (`dagua/layout/ops/sugiyama.py:1441-1447`) and `_AssignLayers` runs `_longest_path_layering` on them (`dagua/layout/ops/sugiyama.py:1485-1494`). A self-loop will keep its node with positive in-degree and can trigger `ValueError("graph must be acyclic after back-edge reversal")` (`dagua/layout/ops/sugiyama.py:217-220`). igraph ignores loops in the layered graph (`/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:387-388`).

2. **Dagua's multiedge weight map likely overweights duplicate neighbors.** Dagua adjacency preserves duplicates (`dagua/layout/ops/sugiyama.py:451-454`) and weight maps aggregate duplicates (`dagua/layout/ops/sugiyama.py:490-494`), then `_neighbor_barycenters` iterates duplicate neighbors and applies aggregate weight each time (`dagua/layout/ops/sugiyama.py:623-639`). This can square multiplicity effects for mixed-neighbor multiedges. igraph counts duplicate neighbor incidences once per incidence through `IGRAPH_MULTIPLE` (`/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:692-705`).

3. **`igraph` type-1 conflict loop appears suspicious in reference C, but Dagua intentionally differs.** igraph builds `neis1` as a vector of neighbor vertex IDs for edges from a layer to the next (`/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:898-910`), then loops `j` and `k` over `n = size(neis1)` but calls `IGRAPH_FROM(graph, j)` and `IGRAPH_TO(graph, j)` rather than indexing an edge ID from `neis1` (`/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:912-943`). This is either a reference quirk/bug or a misleading variable use. Dagua uses a boundary-based type-1 conflict detection (`dagua/layout/ops/sugiyama.py:892-951`), which may be more like the Brandes-Kopf paper but will not bit-match igraph's exact behavior if igraph's loop is as written.

4. **Layering can diverge massively on cyclic graphs.** Dagua's DFS first pass and greedy fallback are topology-only (`dagua/layout/cycle.py:13-70`, `dagua/layout/cycle.py:130-193`), while igraph's vertical placement uses weighted feedback arcs and possibly GLPK (`/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:552-675`). This is not a bug for a Dagua-style Sugiyama, but it is a fidelity bug for `igraph_sugiyama`.

5. **Dagua and igraph use different coordinate frames for disconnected components.** Dagua centers the whole graph (`dagua/layout/ops/sugiyama.py:1294-1311`), while igraph packs components by `dx += max_x + hgap` (`/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:520-527`). Procrustes cannot fully hide different relative component placement.

6. **Benchmark stochastic metadata is wrong for Dagua Sugiyama.** `classic_sugiyama` is flagged stochastic (`dagua/eval/variants.py:1830`) even though `seed` is ignored (`dagua/layout/ops/sugiyama.py:551-558`). Variant entries also mark Sugiyama stochastic (`dagua/eval/variants.py:858-912`). This wastes runs and could confuse TOST/floor interpretation.

7. **Dagua's direct defaults are not igraph defaults.** Direct `layout_sugiyama_pipeline` defaults are graphviz-dot point spacing (`dagua/layout/ops/pipelines/sugiyama.py:29-30`, `dagua/layout/ops/pipelines/sugiyama.py:166-171`), while igraph defaults are unit spacing. The variant registry aligns values for the benchmark (`dagua/eval/variants.py:858-912`), but base `classic_sugiyama` through `_CLASSIC_LAYOUT_SPECS` has empty default params (`dagua/eval/competitors/classic_competitor.py:174-178`), so base-vs-base comparisons may use graphviz-like spacing unless variants are used.

## 12. Ranked Fix List

1. **Filter self-loops before Dagua Sugiyama layering and expanded graph construction.**
   - Expected RMSD / pass-rate impact: high on self-loop fixtures; prevents hard failures and matches igraph loop exclusion.
   - Evidence: self-loops reattached in `make_acyclic_robust` (`dagua/layout/cycle.py:224-247`), Dagua layers full acyclic edge set (`dagua/layout/ops/sugiyama.py:1485-1494`), igraph ignores loops in layered graph and neighbor queries (`/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:387-388`, `/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:692-695`).
   - Proposed fix size: small, about 20-40 LOC in `dagua/layout/ops/sugiyama.py` plus regression tests.

2. **Add an `igraph_fidelity` component-packing path for weak components.**
   - Expected impact: high for disconnected/isolated-node graphs; likely reduces residual RMSD beyond the current strong-equivalent rows.
   - Evidence: igraph computes weak components and lays each out separately (`/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:342-347`) then offsets by `dx` (`/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:520-527`); Dagua expands and orders globally (`dagua/layout/ops/sugiyama.py:333-394`, `dagua/layout/ops/sugiyama.py:551-596`).
   - Proposed fix size: medium-large, about 120-220 LOC if added as an opt-in fidelity mode to avoid changing existing graphviz-dot behavior.

3. **Match igraph's early-stop semantics for barycenter ordering.**
   - Expected impact: medium on tie-heavy graphs and low on already-converged simple DAGs.
   - Evidence: Dagua fixed loop (`dagua/layout/ops/sugiyama.py:558-596`); igraph `while (changed && iter < maxiter)` (`/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:749-828`).
   - Proposed fix size: small-medium, about 40-80 LOC to track old/new layer equality after each phase or full pass.

4. **Align barycenter weighting/multiedge semantics with igraph.**
   - Expected impact: medium-high on multiedge and weighted fixtures; low on simple unweighted graphs.
   - Evidence: Dagua weighted maps and weighted barycenters (`dagua/layout/ops/sugiyama.py:458-494`, `dagua/layout/ops/sugiyama.py:623-639`); igraph unweighted incidence averaging with `IGRAPH_MULTIPLE` (`/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:691-705`).
   - Proposed fix size: medium, about 60-120 LOC and tests for parallel-edge barycenter ordering. If preserving Dagua's weighted extension matters, gate it behind a parameter.

5. **Add an igraph-compatible layering mode for cyclic directed graphs.**
   - Expected impact: high for cyclic graphs; possibly no effect for pure DAG benchmark subsets.
   - Evidence: Dagua DFS/greedy cycle breaking (`dagua/layout/cycle.py:13-70`, `dagua/layout/cycle.py:130-193`, `dagua/layout/cycle.py:196-247`); igraph GLPK/Eades vertical placement (`/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:552-675`).
   - Proposed fix size: large. Eades-only approximation could be 150-250 LOC; GLPK parity would be larger and introduces dependency/build concerns.

6. **Use igraph separation formula in fidelity comparisons when node sizes are not part of the reference input.**
   - Expected impact: medium if benchmark graphs have nonzero `node_sizes`; low otherwise.
   - Evidence: Classic Dagua passes `graph.node_sizes` (`dagua/eval/competitors/classic_competitor.py:951-956`), igraph adapter does not export sizes (`dagua/eval/competitors/igraph_competitor.py:53-76`), Dagua separation includes widths (`dagua/layout/ops/sugiyama.py:1191-1218`), igraph uses `hgap` only (`/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:1284-1294`).
   - Proposed fix size: small-medium, about 30-70 LOC if a `use_node_sizes_for_spacing` flag is added and defaulted carefully.

7. **Make base adapter defaults explicit for igraph-vs-igraph fidelity mode.**
   - Expected impact: medium for base `classic_sugiyama` vs `igraph_sugiyama` runs, lower for variant-mode runs.
   - Evidence: `_CLASSIC_LAYOUT_SPECS["classic_sugiyama"].default_params = {}` (`dagua/eval/competitors/classic_competitor.py:174-178`), direct pipeline defaults are graphviz point units (`dagua/layout/ops/pipelines/sugiyama.py:29-30`), variants align unit spacing (`dagua/eval/variants.py:858-912`).
   - Proposed fix size: small, about 10-30 LOC in adapter/variants, but must avoid regressing the Round 3 graphviz-dot spacing goal.

8. **Correct stochasticity metadata for Sugiyama.**
   - Expected impact: low on coordinates, medium on evaluation efficiency and interpretation.
   - Evidence: Dagua seed ignored (`dagua/layout/ops/sugiyama.py:551-558`), igraph adapter deterministic (`dagua/eval/competitors/igraph_competitor.py:195-202`), metadata says `classic_sugiyama=True`, `igraph_sugiyama=False` (`dagua/eval/variants.py:1830-1849`).
   - Proposed fix size: tiny, about 5-15 LOC plus expected report/cache implications.

9. **Audit and possibly emulate igraph's exact type-1 conflict behavior.**
   - Expected impact: low-medium on dense long-edge DAGs with many dummy nodes; low on simple graphs.
   - Evidence: Dagua boundary conflict implementation (`dagua/layout/ops/sugiyama.py:892-951`); igraph pairwise ignored-edge implementation (`/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:893-945`).
   - Proposed fix size: medium, about 80-160 LOC; needs fixture-level comparison because the C loop appears suspicious.

10. **Delay final centering or make it optional for raw igraph coordinate parity.**
    - Expected impact: low under Procrustes, but helpful for raw coordinate diagnostics and route parity.
    - Evidence: Dagua centers (`dagua/layout/ops/sugiyama.py:1294-1311`); igraph writes `layout + dx` directly (`/home/jtaylor/projects/_references/igraph/src/layout/sugiyama.c:477-527`).
    - Proposed fix size: small, about 20-40 LOC.

## 13. Recommended Round 22+ Fix Scope

Recommended bundle for one follow-up round:

1. **Self-loop filtering before layering/expansion.** This is the clearest correctness bug and has direct igraph evidence. It is small, testable, and does not require changing the public algorithm identity.
2. **Early-stop barycenter mode.** This is a contained fidelity improvement and should be easy to validate against simple order traces.
3. **Multiedge barycenter semantics.** Add targeted tests for parallel-edge bundles and align the fidelity path with igraph's incidence average.
4. **Stochastic metadata correction.** Mark `classic_sugiyama` and Sugiyama variants deterministic unless there is an external reason to keep multi-seed scheduling.
5. **Do not tackle full igraph GLPK layering in the same round.** That is the largest remaining architectural mismatch and should be a separate, explicitly scoped round after the smaller deterministic issues are removed.

Conservative Round 22 acceptance target: keep all current Sugiyama variants `strong_equivalent` in `eval_output/fidelity_report/report.md:92-96`, add regression coverage for self-loops and multiedges, and reduce or preserve median RMSD for `sugiyama_default`, `sugiyama_passes4`, and `sugiyama_passes48` (`eval_output/fidelity_report/report.md:92-94`).

## Current Verdict Context

The mega-run already reports all five Sugiyama variants as `strong_equivalent`: default, passes4, passes48, tight, and wide (`eval_output/fidelity_report/report.md:92-96`). Historical sprint context says the graphviz-dot/Sugiyama spacing fix moved median RMSD from `0.342` to `0.019` in Round 3 (`.project-context/research/sprint_algo_fidelity/algo_fidelity_SUMMARY.md:27-35`, `.project-context/research/sprint_algo_fidelity/algo_fidelity_SUMMARY.md:47-53`). This diagnosis therefore treats the current differences as residual fidelity risks rather than evidence that the family is globally divergent.

The highest-confidence residuals are not RNG-related. They are structural: component packing, self-loop handling, cyclic layering, multiedge/weight semantics, and exact Brandes-Kopf conflict/compaction details.
