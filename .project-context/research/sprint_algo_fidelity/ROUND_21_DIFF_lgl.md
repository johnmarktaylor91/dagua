# Round 21 Diff: LGL (`classic_lgl` vs `igraph_lgl`)

Diagnosis-only round. No dagua source changes were made. The current mega-run verdict is **weak_equivalent**, not strong equivalent: `eval_output/fidelity_report/report.md:50-54` lists all five LGL variants as weak, and `eval_output/fidelity_report/data/algorithm_summary.csv:41-45` reports medians around `0.135-0.144` with scale ratios around `0.023`.

## 1. Files Read

Dagua implementation and wiring:

- `dagua/layout/ops/lgl.py:1-546` -- current composable LGL ops.
- `dagua/layout/ops/pipelines/lgl.py:1-206` -- pipeline builder and public `layout_lgl_pipeline`.
- `dagua/layout/_archive/classic/lgl.py:1-540` -- archived monolithic LGL translation, read as historical context because it mirrors the current ops structure.
- `dagua/eval/variants.py:1365-1418` -- LGL variant registry.
- `dagua/eval/variants.py:1820-1853` -- stochasticity registry for `classic_lgl` and `igraph_lgl`.
- `dagua/eval/competitors/classic_competitor.py:153-248` -- base dispatch spec for `classic_lgl`.
- `dagua/eval/competitors/classic_competitor.py:1570-1628` -- shared `_quick_classic` runner.
- `dagua/eval/competitors/classic_competitor.py:1666-1681` -- explicit `ClassicLGL` adapter.
- `dagua/eval/competitors/igraph_competitor.py:18-50` -- igraph RNG seeding context manager.
- `dagua/eval/competitors/igraph_competitor.py:53-76` -- dagua graph to igraph conversion.
- `dagua/eval/competitors/igraph_competitor.py:79-99` -- igraph layout coordinate scaling.
- `dagua/eval/competitors/igraph_competitor.py:102-184` -- generic igraph adapter execution.
- `dagua/eval/competitors/igraph_competitor.py:275-286` -- `IgraphLGL` adapter.
- `dagua/eval/competitors/dagua_competitor.py:1-97` -- read to confirm this family uses the classic adapter, not native `dagua`.

Reference implementation:

- `/home/jtaylor/projects/_references/igraph/src/layout/large_graph.c:1-389` -- actual `igraph_layout_lgl` implementation; the hint `large_graph_layout.c` does not exist.
- `/home/jtaylor/projects/_references/igraph/include/igraph_layout.h:76-79` -- public C signature.
- `/home/jtaylor/projects/_references/igraph/tests/unit/igraph_layout_lgl.c:1-108` -- upstream default-parameter and disconnected-graph tests.

Existing fidelity context:

- `eval_output/fidelity_report/report.md:50-54` -- LGL verdict rows.
- `eval_output/fidelity_report/data/algorithm_summary.csv:41-45` -- LGL aggregate metrics.
- `eval_output/fidelity_report/data/per_graph_detail.csv:4097-4105` -- sampled LGL per-graph rows and anomaly fields.
- `.project-context/research/sprint_algo_fidelity/algo_fidelity_SUMMARY.md:37-45` -- sprint context for phase-2 families.
- `.project-context/research/sprint_algo_fidelity/algo_fidelity_SUMMARY.md:111-130` -- stochastic-floor methodology.

## 2. Overall Pipeline Structure

The dagua pipeline is a four-op composition:

1. `LGLPrepareState` resolves parameters, builds undirected adjacency and spring edges, chooses the root, and stores metadata in `state.extras` (`dagua/layout/ops/lgl.py:83-168`).
2. `LGLInitializePositions` creates an initial random square cloud and overwrites the root position with zero (`dagua/layout/ops/lgl.py:180-216`).
3. `LGLLayeredRefinement` runs BFS shell growth and repeated FR-style refinement (`dagua/layout/ops/lgl.py:232-505`).
4. `LGLFinalizePositions` casts to float32 and output device (`dagua/layout/ops/lgl.py:518-536`).

The public wrapper validates inputs, handles the zero-node case, builds `LayoutProblem`, records whether the root was random, applies the pipeline, and returns final positions (`dagua/layout/ops/pipelines/lgl.py:149-203`). The classic competitor uses this function through `_quick_classic`: `classic_lgl` maps to `dagua.layout.ops.pipelines.lgl:layout_lgl_pipeline` in the spec (`dagua/eval/competitors/classic_competitor.py:244-248`) and in `ClassicLGL.layout` (`dagua/eval/competitors/classic_competitor.py:1666-1681`).

The reference is one monolithic C function. It validates scalar parameters, picks a root, runs `igraph_bfs_simple`, warns on disconnected graphs, initializes `res` with `igraph_layout_random`, initializes a mutable `igraph_2dgrid_t`, inserts the root at `(0, 0)`, then loops over BFS layers, placing the next shell and refining the grid in-place (`/home/jtaylor/projects/_references/igraph/src/layout/large_graph.c:88-389`).

High-level match: both use BFS shell growth plus FR-style attraction/repulsion. High-level divergence: dagua reconstructs sparse cell buckets from positions each iteration (`dagua/layout/ops/lgl.py:433-487`), while igraph maintains an `igraph_2dgrid_t` whose positions are the authoritative layout state (`large_graph.c:189-195`, `large_graph.c:263-264`, `large_graph.c:364`). That is not only a data-structure detail; it affects iteration order, pair enumeration, boundary behavior, and exact coordinate updates.

## 3. Energy / Loss / Objective

There is no global minimized scalar loss in either implementation. Both implement direct force updates. The equivalent implicit terms are:

| Term | Dagua | igraph | Match |
| --- | --- | --- | --- |
| Edge attraction | `magnitude = distance_matrix.square() / frk`, with direction from source to target and equal/opposite updates (`dagua/layout/ops/lgl.py:417-431`) | `force = dist * dist / frk`; `from -= xd*force`, `to += xd*force` (`large_graph.c:307-325`) | Mostly yes |
| Repulsion | `magnitude = frk^2 * (1/d - d^2/repulserad)` if pair distance `< cellsize` (`dagua/layout/ops/lgl.py:459-471`, `dagua/layout/ops/lgl.py:475-487`) | Same `force = frk * frk * (1.0 / dist - dist * dist / repulserad)` if `dist < cellsize` (`large_graph.c:327-347`) | Formula yes |
| Temperature cap | Clip displacement norm to `temperature` (`dagua/layout/ops/lgl.py:489-494`) | Clip `sqrt(fx^2+fy^2)` to `t` (`large_graph.c:354-364`) | Yes |
| Cooling schedule | `maxdelta * (((maxiter - iteration) / maxiter) ** coolexp)` (`dagua/layout/ops/lgl.py:412-414`) | `maxdelta * pow((maxit - it) / (igraph_real_t) maxit, coolexp)` (`large_graph.c:292-296`) | Yes |

Material objective divergences:

- **Weights:** igraph explicitly says this function does not use weights (`large_graph.c:146-154`). Dagua accepts `edge_weights` and scales attraction by weights when provided (`dagua/layout/ops/lgl.py:133-149`, `dagua/layout/ops/lgl.py:427-429`). The classic adapter forwards `graph.edge_weights` to all classic functions when present (`dagua/eval/competitors/classic_competitor.py:1606-1609`), while the igraph adapter only sets `g.es["weight"]` (`dagua/eval/competitors/igraph_competitor.py:74-75`) and `IgraphLGL` does not pass `weights` in `layout_kwargs` or `variant_param_names` (`dagua/eval/competitors/igraph_competitor.py:275-286`). For weighted benchmark graphs, this is a real objective mismatch.
- **Output-scale objective is not comparable pre-Procrustes:** dagua returns raw coordinates in radius roughly `sqrt(N^2/pi)` and float32 (`dagua/layout/ops/lgl.py:204-215`, `dagua/layout/ops/lgl.py:531-535`). The igraph competitor multiplies layout coordinates by `50.0` (`dagua/eval/competitors/igraph_competitor.py:94-99`). The fidelity report shows mean scale ratios around `0.023`, and flags `scale_ratio_out_of_range` for LGL rows (`algorithm_summary.csv:41-45`, `per_graph_detail.csv:4097-4105`). Procrustes absorbs scale for shape tests, but metric-surface differences and absolute downstream aesthetics are affected.

## 4. Force / Gradient Computation

Attraction direction and signs are aligned. Dagua computes `delta = positions[source] - positions[target]`, normalizes to `direction`, then subtracts contribution from source and adds to target (`dagua/layout/ops/lgl.py:420-431`). igraph computes `xd = res[from] - res[to]`, normalizes, then subtracts from `from` and adds to `to` (`large_graph.c:313-324`).

Repulsion formula is aligned, but pair enumeration is not guaranteed to be:

- igraph iterates via `igraph_2dgrid_next` and `igraph_2dgrid_next_nei` (`large_graph.c:329-331`). Those functions encapsulate a grid traversal order not replicated by dagua.
- dagua sorts cell keys lexicographically (`dagua/layout/ops/lgl.py:444-450`) and loops offsets in `_LGL_BUCKET_NEIGHBORHOOD = (-1, 0, 1)` (`dagua/layout/ops/lgl.py:23-24`, `dagua/layout/ops/lgl.py:447-449`). This is deterministic but not necessarily igraph's grid order.
- igraph mutates the grid with `igraph_2dgrid_move` while applying node movements (`large_graph.c:354-364`). Dagua applies movements to `positions` and rebuilds buckets only at the top of the next iteration (`dagua/layout/ops/lgl.py:433-442`, `dagua/layout/ops/lgl.py:489-494`). If igraph's grid object uses updated coordinates immediately for move bookkeeping and future queries, pair state remains equivalent only if the move does not affect force enumeration until next iteration. This should be verified against `core/grid.c`; it was not read in this round.

The most suspicious force-level bug is the convergence check. igraph updates `maxchange` only when `fx > maxchange` or `fy > maxchange`, without absolute value (`large_graph.c:365-370`). Dagua uses absolute x/y movement (`dagua/layout/ops/lgl.py:495-501`). If all movement components are negative or the largest movement is negative, igraph may stop earlier while dagua continues. This is likely intentional preservation of igraph's historical semantics in the reference, not mathematically correct behavior. For fidelity, dagua should mimic the reference if exactness matters.

## 5. Initialization

Reference:

- Root: if `proot < 0`, igraph uses `RNG_INTEGER(0, no_of_nodes - 1)` (`large_graph.c:156-161`).
- Initial layout: `igraph_layout_random(graph, res)` then `igraph_matrix_scale(res, sqrt(area / M_PI))` (`large_graph.c:184-187`). This means the random layout is generated by igraph's RNG and whatever range `igraph_layout_random` uses.
- Root coordinate is set in the grid by adding root at `(0, 0)` (`large_graph.c:194-195`). The matrix had already been initialized randomly, and the grid call is the authoritative coordinate update path.

Dagua:

- Root: `random.Random(problem.seed).randrange(num_nodes)` when no root is configured (`dagua/layout/ops/lgl.py:153-155`).
- Initial layout: a separate `random.Random(problem.seed)` draws uniform coordinates in `[-radius, radius]` for each axis (`dagua/layout/ops/lgl.py:204-213`).
- Root coordinate is overwritten in `positions` (`dagua/layout/ops/lgl.py:214-215`).

Important RNG detail: dagua consumes the random root once in `LGLPrepareState` (`dagua/layout/ops/lgl.py:153-155`), but `LGLInitializePositions` starts a new RNG at the same seed and does not consume the root draw before initial positions (`dagua/layout/ops/lgl.py:204-213`). In `LGLLayeredRefinement`, dagua starts another RNG and conditionally consumes one `randrange` to mimic the root draw before shell random vectors (`dagua/layout/ops/lgl.py:318-321`). This means shell-vector RNG is at least intentionally offset, but the initial random layout is not on the same stream as reference because igraph uses one global RNG sequence under `_igraph_rng_seed` (`dagua/eval/competitors/igraph_competitor.py:18-50`, `large_graph.c:156-186`).

## 6. Iteration / Convergence

Layer loop:

- igraph computes `no_of_layers = igraph_vector_int_size(&layers) - 1` (`large_graph.c:170`) and loops `for (actlayer = 1; actlayer < no_of_layers; actlayer++)` for both harmonic sum and shell refinement (`large_graph.c:197-201`).
- dagua stores layers as `layers[depth]` and loops `for layer_index in range(len(layers) - 1)` (`dagua/layout/ops/lgl.py:322-324`), where `next_depth = layer_index + 1` (`dagua/layout/ops/lgl.py:340`).

This is a likely off-by-one divergence. If igraph `layers` is a boundary vector, `no_of_layers` and `actlayer < no_of_layers` may skip the deepest boundary by design while still using `VECTOR(layers)[actlayer + 1]` (`large_graph.c:222-264`). Dagua's `range(len(layers) - 1)` processes every explicit next-layer list. The naming makes the semantics hard to compare one-to-one, but the line evidence is suspicious enough to rank highly.

Iterations:

- Both default to `maxiter=150` (`dagua/layout/ops/lgl.py:33-34`, `dagua/layout/ops/pipelines/lgl.py:26-33`, `large_graph.c:57-58`).
- Variant registry aligns default, iter50, iter300, cool1, and cool2 on both sides (`dagua/eval/variants.py:1365-1418`).
- Both stop when `maxchange <= epsilon` in effect, but the exact epsilon differs syntactically: dagua default `1.0e-5` (`dagua/layout/ops/lgl.py:64-69`) and igraph `10e-6` (`large_graph.c:209-211`), which equals `1.0e-5`. The bigger divergence is absolute vs positive-only maxchange as noted above (`dagua/layout/ops/lgl.py:495-501`, `large_graph.c:365-370`).

Progress/interruption exists only in igraph (`large_graph.c:240`, `large_graph.c:276`, `large_graph.c:298-300`, `large_graph.c:312`) and has no mathematical impact unless interruption changes completion.

## 7. Hyperparameter Alignment Table

| Parameter | Dagua default / behavior | Reference default / behavior | Match? | Notes |
| --- | --- | --- | --- | --- |
| `maxiter` / `maxit` | `150` (`dagua/layout/ops/lgl.py:33-34`, `dagua/layout/ops/pipelines/lgl.py:26-33`) | recommended `150` (`large_graph.c:57-58`); test passes `150` (`igraph_layout_lgl.c:40-47`) | Y | Variants align (`variants.py:1365-1396`). |
| `maxdelta` | defaults to `num_nodes` (`dagua/layout/ops/lgl.py:109-112`) | recommended number of vertices (`large_graph.c:59-61`); test passes `vc` (`igraph_layout_lgl.c:40-47`) | Y | Dagua allows zero? pipeline only validates maxiter/coolexp; op can resolve 0 only for empty graph. igraph rejects `<=0` for non-empty (`large_graph.c:124-126`). |
| `area` | defaults to `num_nodes ** 2` (`dagua/layout/ops/lgl.py:113-115`) | recommended vertices squared (`large_graph.c:62-64`); test passes `vc * vc` (`igraph_layout_lgl.c:40-47`) | Y | |
| `coolexp` | `1.5` (`dagua/layout/ops/lgl.py:39-40`) | recommended `1.5` (`large_graph.c:65-66`) | Y | Variants align (`variants.py:1398-1418`). |
| `repulserad` | defaults to `area * max(num_nodes, 1)` (`dagua/layout/ops/lgl.py:116-120`) | recommended `area * vertices` (`large_graph.c:67-69`); tests use `vc * vc * vc` (`igraph_layout_lgl.c:40-47`) | Y for non-empty | Empty handled separately in both (`dagua/layout/ops/pipelines/lgl.py:175-177`, `large_graph.c:111-117`). |
| `cellsize` | defaults to `area ** 0.25` (`dagua/layout/ops/lgl.py:121-123`) | fourth root of area (`large_graph.c:70-73`); tests use `sqrt(sqrt(vc))` when default area is `vc^2` (`igraph_layout_lgl.c:40-47`) | Y | |
| `root` | `None` means Python random root (`dagua/layout/ops/lgl.py:153-155`) | negative means igraph RNG root (`large_graph.c:156-161`) | N in RNG semantics | Same concept, different RNG and different default sentinel API. |
| `edge_weights` | accepted and scales attraction (`dagua/layout/ops/lgl.py:133-149`, `dagua/layout/ops/lgl.py:427-429`) | not used; TODO says weights are not handled (`large_graph.c:146-154`) | N | Highest-impact weighted-graph mismatch. |
| Directionality | dagua converts all edges to undirected lower/upper pairs (`dagua/layout/ops/lgl.py:139-147`) | BFS uses `IGRAPH_ALL` and incident edges use `IGRAPH_ALL` (`large_graph.c:168`, `large_graph.c:277`) | Mostly Y | Dagua loses original `from/to` ordering in spring list; signs are symmetric so shape should not change. |
| Self-loops | dagua skips at graph build (`dagua/layout/ops/lgl.py:139-141`) | `igraph_incident(..., IGRAPH_LOOPS)` sees loops (`large_graph.c:277`), but loop attraction has `dist=0`, force 0 (`large_graph.c:313-325`) | Mostly Y | Loop may still duplicate an edge id in the active edge list; no force if same endpoint. |
| Multi-edges | dagua keeps multiplicity (`dagua/layout/ops/lgl.py:146-149`) | igraph pushes edge ids; multiple edges are separate ids (`large_graph.c:272-285`) | Y | |
| Output dtype | final `torch.float32` (`dagua/layout/ops/lgl.py:531-535`) | `igraph_real_t`, typically double, then Python adapter stores into default `torch.zeros` float32 (`dagua/eval/competitors/igraph_competitor.py:94-99`) | Mixed | Both benchmark tensors end up float32, but internal precision differs. |
| Output scale | raw dagua LGL coordinates | igraph adapter multiplies by `50.0` (`dagua/eval/competitors/igraph_competitor.py:96-98`) | N | Report flags scale out of range (`per_graph_detail.csv:4097-4105`). |

## 8. Edge Cases

Self-loops:

- Dagua drops self-loops before adjacency/spring construction (`dagua/layout/ops/lgl.py:139-141`).
- igraph includes loops in incident edge discovery (`large_graph.c:277`), but loop attraction computes zero distance and therefore zero normalized direction and zero force (`large_graph.c:313-325`). Practical force impact is near zero, but active-edge list length and iteration overhead can differ.

Multi-edges:

- Dagua intentionally preserves multiplicity in `spring_edges` (`dagua/layout/ops/lgl.py:146-149`).
- igraph appends every incident edge id whose other endpoint is in the grid (`large_graph.c:272-285`). Multi-edge force multiplicity should match.

Disconnected components:

- igraph warns when BFS does not reach all nodes (`large_graph.c:172-176`); upstream unit test expects this warning (`igraph_layout_lgl.c:86-101`).
- Dagua does not warn. It omits unreached nodes from `layers` but still initializes random positions for all nodes (`dagua/layout/ops/lgl.py:274-295`, `dagua/layout/ops/lgl.py:401-403`). Unreached nodes never become `placed`, never join `refinement_nodes`, and never get edge forces unless incident to a reached node, which cannot happen by definition for a disconnected component.
- The fidelity data includes disconnected graphs in LGL anomaly lists (`algorithm_summary.csv:41-45`), so this is not theoretical.

Weighted edges:

- As above, dagua uses weights; igraph LGL does not (`dagua/layout/ops/lgl.py:133-149`, `dagua/layout/ops/lgl.py:427-429`, `large_graph.c:146-154`). Weighted benchmark rows such as `weighted_chain_20`, `weighted_clusters_3x10`, and `weighted_karate_34` are listed among LGL anomaly graphs (`algorithm_summary.csv:41-45`).

Empty graph:

- Dagua returns empty `[0,2]` float32 tensor before parameter resolution (`dagua/layout/ops/pipelines/lgl.py:175-177`).
- igraph resizes to `0 x 2` and skips parameter checks for null graphs (`large_graph.c:111-117`). Upstream tests cover this case (`igraph_layout_lgl.c:36-49`). Match.

Invalid parameters:

- igraph rejects negative `maxit`, non-positive `maxdelta`, `area`, `coolexp`, `repulserad`, and `cellsize` (`large_graph.c:119-142`).
- Dagua public wrapper rejects negative `maxiter`, non-positive `coolexp`, invalid `edge_index`, and bad `edge_weights` length (`dagua/layout/ops/pipelines/lgl.py:149-164`), but it does not reject non-positive explicit `maxdelta`, `area`, `repulserad`, or `cellsize` before the ops use them. Negative area can break `math.sqrt(area / math.pi)` in initialization (`dagua/layout/ops/lgl.py:204-205`) rather than producing an igraph-like validation error.

## 9. Numerical Precision

Dagua uses Python floats and torch float64 internally for positions, forces, and weights (`dagua/layout/ops/lgl.py:204-213`, `dagua/layout/ops/lgl.py:406-414`, `dagua/layout/ops/lgl.py:433-494`), then casts to float32 (`dagua/layout/ops/lgl.py:531-535`). The reference uses `igraph_real_t`, normally double in igraph builds, for all coordinate and force values (`large_graph.c:88-92`, `large_graph.c:108-109`, `large_graph.c:203-214`, `large_graph.c:311-342`). The Python adapter then copies into `torch.zeros(num_nodes, 2)` without dtype override, which is default float32 (`dagua/eval/competitors/igraph_competitor.py:94-99`).

The bigger numerical difference is summation/update order:

- Dagua vectorizes attractive forces with `index_add_` (`dagua/layout/ops/lgl.py:417-431`), then loops sorted repulsion cells (`dagua/layout/ops/lgl.py:444-487`).
- igraph loops active edge ids in insertion order (`large_graph.c:307-325`) and grid neighbor pairs in grid iterator order (`large_graph.c:327-349`).
- Dagua computes center of mass as a torch mean over currently placed nodes (`dagua/layout/ops/lgl.py:328-336`). igraph asks the grid for its center (`large_graph.c:240-242`). If grid center uses all inserted coordinates with its own order, low-order bits differ; if grid state differs, high-level shell placement differs.

The report's `scale_ratio_std` is tiny but non-zero for all LGL variants (`algorithm_summary.csv:41-45`), consistent with stable but systematically different absolute scaling and shape.

## 10. RNG Semantics

Dagua's torch seed does **not** produce the same sequence as the reference's RNG. In fact, LGL does not use torch RNG for its random draws; it uses Python `random.Random(problem.seed)` for root, initialization, and shell vectors (`dagua/layout/ops/lgl.py:153-155`, `dagua/layout/ops/lgl.py:204-213`, `dagua/layout/ops/lgl.py:318-321`, `dagua/layout/ops/lgl.py:374-381`).

The igraph adapter sets python-igraph's global RNG to `random.Random(seed)` when `uses_igraph_rng=True` (`dagua/eval/competitors/igraph_competitor.py:18-50`, `dagua/eval/competitors/igraph_competitor.py:177-178`). This makes igraph consume Python RNG values through python-igraph's RNG adapter, but the sequence of calls is still reference-defined:

1. Root draw through `RNG_INTEGER` (`large_graph.c:156-159`).
2. Random layout through `igraph_layout_random` (`large_graph.c:184-186`).
3. Shell vectors through `RNG_UNIF(-1, 1)` (`large_graph.c:256-259`).

Dagua splits those into separate RNG instances:

1. Root draw in prepare (`dagua/layout/ops/lgl.py:153-155`).
2. Initial layout starts again from the seed, so it does not follow root consumption (`dagua/layout/ops/lgl.py:204-213`).
3. Shell RNG starts again from the seed, then manually consumes one root draw when root was random (`dagua/layout/ops/lgl.py:318-321`).

Therefore even if python-igraph's `random.Random` adapter and dagua's Python `random.Random` share a generator implementation, the streams are not globally identical. The initial random cloud especially diverges from igraph because the root draw is not consumed first in dagua initialization. Additionally, igraph's `igraph_layout_random` range and call count must match dagua's `uniform(-radius, radius)` exactly for parity; the line evidence only confirms igraph random layout plus scale (`large_graph.c:184-187`), not the same base range.

## 11. Edge-Case Bugs / Suspicious Divergences

1. **Weighted-edge mismatch.** Dagua scales attraction by `edge_weights`; igraph LGL ignores weights by design (`dagua/layout/ops/lgl.py:427-429`, `large_graph.c:146-154`). This is an objective mismatch, not stochastic noise.

2. **Layer loop boundary likely differs.** Dagua loops over every `len(layers)-1` next layer (`dagua/layout/ops/lgl.py:322-324`). igraph loops `actlayer = 1; actlayer < no_of_layers` after `no_of_layers = size(layers)-1` (`large_graph.c:170`, `large_graph.c:197-201`). The indexing is not trivially equivalent because igraph's `layers` vector is a boundary vector from `igraph_bfs_simple`, while dagua's `layers` is a list per depth. This needs a micrograph trace.

3. **First-shell angle denominator is probably different.** Dagua uses total next-layer children as denominator and starts index at `0`: `angle = 2*pi*layer_child_index / total_layer_children` (`dagua/layout/ops/lgl.py:340-371`). igraph uses `phi = 2 * M_PI / (VECTOR(layers)[2] - 1) * (j - 1)` (`large_graph.c:250-254`). If `VECTOR(layers)[2]` is the exclusive boundary for depth-2 and root occupies index 0, denominator may equal first-shell count. If not, this is an off-by-one. It deserves a dedicated trace because first-shell placement controls the entire basin.

4. **Convergence positive-only vs absolute maxchange.** Dagua uses absolute movement components (`dagua/layout/ops/lgl.py:495-501`); igraph uses only positive `fx`/`fy` comparisons (`large_graph.c:365-370`). This can change early stopping.

5. **RNG stream split.** Dagua uses separate Python RNG instances for root, initial positions, and shell vectors (`dagua/layout/ops/lgl.py:153-155`, `dagua/layout/ops/lgl.py:204-213`, `dagua/layout/ops/lgl.py:318-321`); igraph uses one reference RNG stream through root, random layout, and shell vectors (`large_graph.c:156-186`, `large_graph.c:256-259`).

6. **Initial random layout not reference-identical.** Dagua draws `uniform(-radius, radius)` directly (`dagua/layout/ops/lgl.py:204-213`). igraph calls `igraph_layout_random` then scales by `sqrt(area / M_PI)` (`large_graph.c:184-187`). If `igraph_layout_random` draws `[0,1]`, `[-1,1]`, or a graph-specific shape, this changes initial grid center and all later shells.

7. **Absolute scale mismatch in adapter.** igraph competitor multiplies by `50.0` (`dagua/eval/competitors/igraph_competitor.py:96-98`); dagua returns raw positions. The report flags `scale_ratio_out_of_range` in LGL rows (`per_graph_detail.csv:4097-4105`). Procrustes shape can pass while downstream non-Procrustes metrics diverge.

8. **Disconnected graph behavior differs in diagnostics and probably layout state.** igraph warns and keeps unreached random coordinates (`large_graph.c:172-176`); dagua silently leaves unreached initialized coordinates outside active refinement (`dagua/layout/ops/lgl.py:274-295`, `dagua/layout/ops/lgl.py:401-403`).

9. **Grid semantics are approximated, not ported.** Dagua bucket rebuild is simpler (`dagua/layout/ops/lgl.py:433-487`); igraph relies on `igraph_2dgrid_t` (`large_graph.c:189-195`, `large_graph.c:329-331`, `large_graph.c:364`). Neighbor inclusion at cell boundaries and iteration order can differ.

10. **Parameter validation mismatch.** igraph rejects non-positive explicit `maxdelta`, `area`, `repulserad`, and `cellsize` (`large_graph.c:124-142`); dagua public wrapper does not validate these beyond `coolexp` and `maxiter` (`dagua/layout/ops/pipelines/lgl.py:149-164`).

## 12. Ranked Fix List

1. **Disable edge-weight influence for `classic_lgl` fidelity mode.**
   Evidence: dagua weight path (`dagua/layout/ops/lgl.py:133-149`, `dagua/layout/ops/lgl.py:427-429`), igraph no-weight TODO (`large_graph.c:146-154`), adapter forwards weights (`dagua/eval/competitors/classic_competitor.py:1606-1609`).
   Expected RMSD impact: high on weighted graphs; low elsewhere.
   Fix size: small, 10-25 lines. Add a flag such as `use_edge_weights=False` defaulting to igraph-compatible behavior for LGL, or stop `_quick_classic` forwarding weights for `layout_lgl_pipeline`.

2. **Make RNG one-stream compatible with igraph call order.**
   Evidence: dagua separate RNGs (`dagua/layout/ops/lgl.py:153-155`, `dagua/layout/ops/lgl.py:204-213`, `dagua/layout/ops/lgl.py:318-321`), igraph root/random-layout/shell sequence (`large_graph.c:156-186`, `large_graph.c:256-259`).
   Expected RMSD impact: high for small/medium stochastic layouts; may reduce median if basin selection aligns.
   Fix size: medium, 40-90 lines. Thread one RNG object through prepare/init/refine or store consumed initial coordinates in extras. Exact python-igraph RNG parity may still require matching `igraph_layout_random` semantics.

3. **Trace and align layer boundary indexing.**
   Evidence: dagua `range(len(layers)-1)` (`dagua/layout/ops/lgl.py:322-324`), igraph `actlayer < no_of_layers` with boundary vector (`large_graph.c:170`, `large_graph.c:197-201`).
   Expected RMSD impact: high if off-by-one confirmed; especially trees and chains.
   Fix size: medium, 30-80 lines plus tests. Start with a micrograph trace for path, star, and binary tree.

4. **Align first-shell angular formula exactly.**
   Evidence: dagua denominator `total_layer_children` and index `0` (`dagua/layout/ops/lgl.py:340-371`), igraph denominator `VECTOR(layers)[2] - 1` and multiplier `j - 1` (`large_graph.c:250-254`).
   Expected RMSD impact: medium-high on all connected graphs because shell 1 sets global orientation and basin.
   Fix size: small-medium, 15-40 lines after layer semantics are confirmed.

5. **Mimic igraph `maxchange` sign behavior.**
   Evidence: dagua absolute movement (`dagua/layout/ops/lgl.py:495-501`), igraph positive-only component comparison (`large_graph.c:365-370`).
   Expected RMSD impact: medium; mostly affects early-stop iteration count.
   Fix size: small, 5-15 lines, but should be guarded by a fidelity-mode comment because it intentionally preserves a reference quirk.

6. **Add igraph-compatible initial random layout path.**
   Evidence: dagua uniform square cloud (`dagua/layout/ops/lgl.py:204-213`), igraph `igraph_layout_random` plus scale (`large_graph.c:184-187`).
   Expected RMSD impact: medium; may be mostly stochastic-floor if refinement dominates.
   Fix size: medium, 30-60 lines once `igraph_layout_random` range is confirmed from `layout_random.c`.

7. **Normalize or remove the `50.0` adapter scale for shape-plus-metric fidelity.**
   Evidence: igraph adapter scale (`dagua/eval/competitors/igraph_competitor.py:94-99`), report scale ratios (`algorithm_summary.csv:41-45`).
   Expected RMSD impact: low for Procrustes RMSD, high for absolute metric parity.
   Fix size: small, but cross-family risk: `_igraph_pos_to_tensor` is shared by all igraph competitors. Use per-adapter scale if changed.

8. **Mirror igraph disconnected warning / semantics.**
   Evidence: igraph warning (`large_graph.c:172-176`), upstream test (`igraph_layout_lgl.c:86-101`), dagua no warning in BFS path (`dagua/layout/ops/lgl.py:274-295`).
   Expected RMSD impact: low to medium on disconnected benchmark graphs.
   Fix size: small for warning; medium if trying to match exact unreached-node grid/matrix semantics.

9. **Port or inspect `igraph_2dgrid_t` for exact pair order and boundary handling.**
   Evidence: reference grid API usage (`large_graph.c:189-195`, `large_graph.c:329-331`, `large_graph.c:364`) vs dagua sorted sparse buckets (`dagua/layout/ops/lgl.py:433-487`).
   Expected RMSD impact: low-medium, mostly residual after larger levers.
   Fix size: large, 150-300 lines if ported faithfully; small if only boundary/order tweaks are needed.

## 13. Recommended Round 22+ Fix Scope

Recommended one-round bundle:

1. Add a targeted LGL fidelity mode or direct default change so edge weights do not affect `classic_lgl` attraction. This is the clearest objective mismatch and low-risk.
2. Build a tiny trace harness for path/star/binary-tree comparing dagua layer lists and igraph boundary-vector assumptions. Use it to decide whether the `range(len(layers)-1)` and first-shell denominator need adjustment.
3. Align the convergence quirk to igraph's positive-only `maxchange` inside fidelity mode.
4. Do **not** port `igraph_2dgrid_t` yet. The grid rewrite is higher cost and should wait until the cheap objective/RNG/layer fixes are measured.

Expected outcome: weighted graph anomalies should improve immediately; small connected graph RMSD may drop if layer/angle semantics are confirmed and fixed. The family may remain `weak_equivalent` because stochastic RNG parity and igraph grid order are deeper than one follow-up round, but these levers are the most defensible next steps.

## Verification

No tests were run because this was diagnosis-only and no source behavior was changed. Verification for this round is that this report exists at `.project-context/research/sprint_algo_fidelity/ROUND_21_DIFF_lgl.md` and exceeds 10 KB with line references throughout.
