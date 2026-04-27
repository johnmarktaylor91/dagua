Torch seeding is centralized in [`engine.py`](/home/jtaylor/projects/dagua/dagua/layout/engine.py): `layout()` calls `torch.manual_seed(config.seed)` for all runs and additionally `torch.cuda.manual_seed(config.seed)` when the resolved device is `"cuda"`. None of these files accept a `torch.Generator`; every Torch random draw uses the global device RNG, and only [`constraints.py`](/home/jtaylor/projects/dagua/dagua/layout/constraints.py) also uses Python `random`, which `engine.py` does not seed.

## [constraints.py](/home/jtaylor/projects/dagua/dagua/layout/constraints.py)

Core data: `pos[N,2]`; `edge_index[2,E]`; `node_sizes[N,2]`; `LayerIndex(node_to_layer, layer_offsets, sorted_nodes, num_layers)`; `EdgeBatchLike(src,tgt,dx,dy,dist_sq)`; `SampledNodeLike(active_idx,sampled)`; `clusters: dict[name -> members|nested dict]`; `cluster_parents: dict[str, str|None]`; flex tensors (`pin_indices`, `pin_targets`, `pin_weights`, `pin_mask`) and `align_groups: list[(indices, weight, axis)]`.

Engine integration: `_layout_inner()` in [`engine.py`](/home/jtaylor/projects/dagua/dagua/layout/engine.py) binds every public loss into `loss_fns` once, then evaluates them through `_compute_loss_term()`, `_backward_standard_loss_terms()`, hybrid CPU `_hybrid_loss()`, tiled GPU, or subset-GPU executors. Direct call mapping: `w_dag -> dag_ordering_loss`, `w_attract -> edge_attraction_loss`, `w_repel -> repulsion_loss`, `w_overlap -> overlap_avoidance_loss`, `w_cluster -> cluster_compactness_loss` and `w_cluster*0.5 -> cluster_separation_loss`, `w_cluster_contain -> cluster_containment_loss`, `w_crossing -> crossing_loss`, `w_straightness -> edge_straightness_loss`, `w_length_variance -> edge_length_variance_loss`, `w_spacing -> spacing_consistency_loss(target_gap=node_sep)`, `w_fanout -> fanout_distribution_loss(step=current_step_ref[0])`, `w_back_edge -> back_edge_compactness_loss`, flex losses to `position_pin_loss` / `alignment_loss` / `flex_spacing_loss`, and `project_hard_pins()` immediately after `optimizer.step()`.

1. `dag_ordering_loss`: 1) choose `src,tgt` from `edge_ctx` or `_non_self_edges(edge_index)`; 2) early-return on empty; 3) compute vertical margin `(h_src+h_tgt)/2 + rank_sep*0.5`; 4) penalize `relu(src_y - tgt_y + margin)`; 5) mean. Tunables: `rank_sep: float = 50.0`, `edge_ctx: EdgeBatchLike | None = None`.
2. `edge_attraction_loss`: 1) reuse `dx,dy,dist_sq` from `edge_ctx` or compute them; 2) initialize cap=`1`; 3) for edges with `dist_sq < 9.0`, limit effective attraction to `dist/3`; 4) return `x_bias*mean(dx^2*cap) + mean(dy^2*cap)`. Tunables: `x_bias: float = 4.0`, `edge_ctx: EdgeBatchLike | None = None`.
3. `edge_straightness_loss`: 1) reuse or compute `dx`; 2) return `mean(dx^2)`. Tunables: `edge_ctx: EdgeBatchLike | None = None`.
4. `edge_length_variance_loss`: 1) reuse or compute `dist_sq`; 2) convert to lengths `sqrt(dist_sq + 1e-8)`; 3) return variance. Tunables: `edge_ctx: EdgeBatchLike | None = None`.
5. `repulsion_loss`: dispatcher. Order is exact pairwise if `N <= threshold`, then sampled-context RVS if `sampled_ctx` exists, then RVS if `N > rvs_threshold and layer_index`, then layer-local scatter if `layer_index`, else global negative sampling. Tunables: `threshold: int = 2000`, `sample_k: int = 128`, `layer_index: LayerIndex | None = None`, `node_sizes: Tensor | None = None`, `rvs_threshold: int = 5000`, `rvs_nn_k: int = 20`, `sampled_ctx: SampledNodeLike | None = None`.
6. `_repulsion_exact`: all-pairs `diff`, `dist_sq+1e-4`, mask diagonal, optional size factor `((w1+w2)*(h1+h2))/mean`, mean inverse distance. Required inputs only.
7. `_repulsion_sampled`: sample `k=min(sample_k,N-1)` global negatives with self-skip trick, compute inverse-distance mean. Required tunable: `sample_k`.
8. `_repulsion_scatter`: use `LayerIndex` to sample `K` nodes from same/adjacent layers, mask self-pairs, compute optional size-aware inverse distance, normalize by valid pair count. Required tunables: `sample_k`, `node_sizes|None`.
9. `_repulsion_rvs`: sample active set `n_active=min(max(N**0.75,min(N,256)),1_000_000)`, random adjacent-layer samples `n_random=max(N**0.25,4)`, same-layer pseudo-NN samples `K_nn=min(nn_k,N-1)`, concatenate, mask self-pairs, compute size-aware inverse distance, average. Required tunables: `sample_k`, `nn_k`, `node_sizes|None`.
10. `_repulsion_rvs_from_context`: split shared `sampled_ctx.sampled` into same-layer and adjacent-layer slices, keep up to `nn_k` same-layer neighbors plus `n_random=max(N**0.25,4)` randoms, then compute the same size-aware inverse-distance mean. Required tunables: `nn_k`, `node_sizes|None`.
11. `overlap_avoidance_loss`: dispatcher. Order is shared sampled-context path, then exact if `N <= 500`, then active-subset same-layer path if `N > rvs_threshold and layer_index`, then same-layer scatter if `layer_index`, else grid fallback. Tunables: `padding: float = 2.0`, `layer_index: LayerIndex | None = None`, `rvs_threshold: int = 100000`, `sampled_ctx: SampledNodeLike | None = None`, `debug_callback: Callable[[str],None] | None = None`.
12. `_overlap_exact`: all-pairs `dx_abs`, `dy_abs`, min separations from half-widths/half-heights plus padding, overlap area `relu(min_dx-dx)*relu(min_dy-dy)`, drop diagonal, mean.
13. `_overlap_scatter`: sample `K=min(128,N-1)` same-layer neighbors, compute bbox overlap against sampled peers, mask self-pairs, normalize by valid samples.
14. `_overlap_active_subset`: choose `n_active=min(max(N**0.75,min(N,256)),1_000_000)`, sample `K=min(64,N-1)` same-layer peers for those actives, compute bbox overlap, mask self-pairs, normalize.
15. `_overlap_active_subset_from_context`: reuse first `min(64, sampled.shape[1])` sampled columns from shared context and compute the same bbox overlap.
16. `_overlap_grid_vectorized`: derive `cell_size=max(max_w,max_h)+padding` with floor at `1.0`; hash cells; sort by cell; batch small cells (`<=64`) for vectorized all-pairs; iterate large cells individually; cap processed cells at `1000`; cap per-large-cell sample at `200`; return average overlap.
17. `crossing_loss`: chooses `_crossing_loss_fallback` when `layer_assignments is None` or `E < 20`, else `_crossing_loss_layered`. Tunables: `alpha: float = 5.0`, `max_pairs: int = 2000`, `layer_assignments: list[int] | Tensor | None = None`. Engine overrides `max_pairs=500` and anneals `alpha` from `3.0` to `10.0`.
18. `_crossing_loss_fallback`: optionally subsample edges if pair count exceeds `max_pairs`; use all upper-triangle pairs for `n<=200` else random `i,j`; compute `sigmoid(-alpha * dx_src * dx_tgt)` and sum.
19. `_crossing_loss_layered`: orient every edge upward by layer, discard non-forward edges, decompose long edges into adjacent-layer virtual segments, optionally subsample segments if total exceeds `max(num_edges*4,50000)`, build same-layer segment pairs exactly or stochastically, compare segment endpoint x-order, sum sigmoid proxy.
20. `cluster_compactness_loss`: resolve each cluster to leaf indices, compute cluster centroid, average mean squared distance to centroid across clusters. Tunables: none besides required `clusters`, `device`.
21. `cluster_separation_loss`: build cluster list `(name, idx, parent)`; if `cluster_parents` exists, only compare siblings; otherwise sample up to `50` cluster pairs for `>50` clusters or enumerate all pairs; compute padded bounding boxes and add overlap area. Tunables: `padding: float = 10.0`, `device: torch.device | None = None`, `cluster_parents: dict[str, str|None] | None = None`.
22. `cluster_containment_loss`: for each `(child,parent)` in `cluster_parents`, compute child and padded parent bboxes, penalize `relu(parent_min-child_min)^2 + relu(child_max-parent_max)^2`, average over valid pairs. Tunables: `padding: float = 18.0`, `device: torch.device | None = None`.
23. `position_pin_loss`: select pinned rows, square distance to targets, multiply by per-axis weights and masks, divide by constrained-axis count. Tunables: required tensors only; no defaults.
24. `alignment_loss`: for each `(indices, weight, axis)` group with at least two nodes, compute variance around group mean on that axis, weight it, average across groups. Tunables: required `align_groups`; no defaults.
25. `flex_spacing_loss`: early-return unless `layer_index` exists and `weight > 0`; call `spacing_consistency_loss(..., target_gap=target_sep)`; multiply by `weight`. Tunables: required `target_sep: float`, `weight: float`.
26. `project_hard_pins`: `torch.no_grad()`, gather pinned positions, replace masked axes with targets via `torch.where`, write back in place. Tunables: required pin tensors only.
27. `spacing_consistency_loss`: if `N > 100_000_000` call `_spacing_consistency_loss_layerlocal`; else sort by `layer*1e8 + x`, keep consecutive same-layer pairs, compute edge-to-edge gaps, penalize `(gap-target_gap)^2`. Tunables: `target_gap: float = 25.0`.
28. `_spacing_consistency_loss_layerlocal`: iterate layers, x-sort nodes within each layer, compute consecutive gaps using widths, accumulate squared deviation from `target_gap`, divide by pair count.
29. `fanout_distribution_loss`: skip most full-edge evaluations on `N > 1_000_000` unless `step % 5 == 0` or edges are already sampled; group outgoing edges by source; keep hubs with `degree >= degree_threshold`; compute child angles around each hub; sort by `(hub_id, angle)`; compare angular gaps to ideal `2π/degree`; average per hub. Tunables: `degree_threshold: int = 5`, `edge_ctx: EdgeBatchLike | None = None`, `step: int = 0`, `edge_is_sampled: bool = False`.
30. `back_edge_compactness_loss`: reuse or compute `dx,dy`; mark back edges as `dy > 0` (target above source in screen coords); return mean `dx^2` for those edges. Tunables: `edge_ctx: EdgeBatchLike | None = None`.

Shared classic steps: `edge_attraction_loss` is the differentiable analogue of classic `SpringAttraction` from FR/GEM/FMMM/GraphOpt/FA2; `repulsion_loss` is the differentiable analogue of classic `CoulombRepulsion` with large-graph sampling strategies instead of Barnes-Hut; `crossing_loss` targets the same aesthetic as Sugiyama crossing minimization; the rest are native-engine-only in the current catalog.

Hardcoded-but-parameterizable items: `rank_sep*0.5` margin factor; attraction near-edge cutoff `dist_sq < 9.0`, cap fraction `1/3`, max local force `1.0`; repulsion/overlap epsilons `1e-4`; overlap exact threshold `500`; overlap sample caps `128` and `64`; RVS caps `256`, `1_000_000`, `N**0.75`, `N**0.25`; grid caps `1000` cells, `64` batched nodes, `200` large-cell nodes; crossing fallback threshold `E < 20`, pair sampling cutoff `n > 200`, layered segment cap `max(E*4,50000)`; cluster pair sample cap `50`; spacing huge-graph cutoff `100_000_000` and composite-key scale `1e8`; fanout skip threshold `1_000_000` and decimation interval `5`. Every one of these is still buried in code, not config.

RNG: Torch RNG draws occur in `_repulsion_sampled` (`torch.randint[N,k]`), `_repulsion_scatter` (`torch.rand[N,K]`), `_repulsion_rvs` (`torch.randint[A]`, `torch.rand[A,n_random]`, `torch.rand[A,K_nn]`), `_overlap_scatter` (`torch.rand[N,K]`), `_overlap_active_subset` (`torch.randint[A]`, `torch.rand[A,K]`), `_overlap_grid_vectorized` (`torch.randperm` when sampling cells/nodes), `_crossing_loss_fallback` (`torch.randperm` for edge subsampling and `torch.randint` for pair sampling), and `_crossing_loss_layered` (`torch.randperm` for segment subsampling or `torch.rand` for per-layer pair sampling). Python RNG is consumed only in `cluster_separation_loss` when `cluster_parents` is absent and `num_clusters > 50`; it samples sibling pairs with repeated `random.randint` calls and is not controlled by `config.seed`.

Catalog status: every public loss in this file has a direct DESIGN.md `LOSS` mapping already. What is missing is not coverage but op packaging: no `LossOp` subclasses, no frozen config dataclasses, and no `HardPinProjection` op wrapper even though the algorithmic logic exists.

## [layers.py](/home/jtaylor/projects/dagua/dagua/layout/layers.py)

This file does not compute DAG layers. It only builds an index over already-computed layer assignments.

Core data: `LayerIndex.node_to_layer[N]`, `layer_offsets[L+1]`, `sorted_nodes[N]`, `num_layers: int`.

Engine integration: `_layout_inner()` calls `build_layer_index()` in four cases: to normalize `prebuilt_layer_index` onto the resident device, to build from caller-supplied `layer_assignments`, to build from `longest_path_layering(...)` results, and to build a CPU mirror for hybrid mode.

1. `build_layer_index`: 1) coerce `layer_assignments` to `int32` if possible else `long`; 2) compute `num_layers=max(layer)+1`; 3) estimate CUDA sort scratch as `num_nodes*8*3`; 4) if CUDA sort is enabled and free VRAM exceeds `required_bytes/0.6`, run stable CUDA argsort, otherwise stable CPU argsort; 5) compute per-layer counts with `bincount`; 6) cumulative-sum into `layer_offsets`; 7) return `LayerIndex`. Tunables: `device: str = "cpu"`, `verbose: bool = False`, `progress: Callable[[str],None] | None = None`, `enable_cuda_sort: bool = True`.
2. `_log_layer_index_status`: emit a status line only when `verbose` is true. Required inputs only.
3. `_layer_index_gpu_sort_required_bytes`: returns `num_nodes * 8 * 3`. No tunables.
4. `_cuda_layer_argsort`: move to CUDA if needed, run stable `argsort`, move permutation back to `output_device`. Required inputs only.

Shared classic steps: only `BuildLayerIndex`, and DESIGN.md marks it as engine-only. The classic `LongestPathLayering` op exists, but not in this file.

Hardcoded-but-parameterizable items: GPU scratch estimate `24 bytes/node`; VRAM safety factor `0.6`; dtype cutoff at `torch.iinfo(torch.int32).max`.

RNG: none.

Catalog status: mapped op present is `BuildLayerIndex`. Missing from this file relative to the layering catalog are `LongestPathLayering`, `LayerPromotion`, and `InsertDummyNodes`.

## [projection.py](/home/jtaylor/projects/dagua/dagua/layout/projection.py)

Core data: `pos[N,2]`, `node_sizes[N,2]`, optional `LayerIndex`, plus temporary sorted indices, overlap masks, and per-node displacement accumulators.

Engine integration: `_layout_inner()` calls `project_overlaps()` after `optimizer.step()` on an adaptive interval and again once at the end. Runtime args are always `padding=2.0`, `layer_index=layer_index`, and `iterations` chosen from graph size; no direct `engine.py` call into `HardPinProjection` happens here.

1. `project_overlaps`: 1) early-return for `N <= 1`; 2) if CUDA + layered + `N > 500`, estimate required bytes `N*72` and decide whether to keep projection on GPU or copy to CPU; 3) enter `torch.no_grad()`; 4) dispatch to `_project_exact`, `_project_sweep_cuda`, `_project_sweep_streaming`, `_project_sweep`, or `_project_grid`; 5) on CUDA OOM, empty cache, retry on CPU, copy result back. Tunables: `padding: float = 2.0`, `iterations: int = 10`, `layer_index: LayerIndex | None = None`.
2. `_project_exact`: iterate up to `iterations`; compute all-pairs `dx,dy`, min separations, overlap masks, break if no overlaps, then resolve each overlapping upper-triangle pair along the smaller-overlap axis, splitting displacement symmetrically. Required inputs only.
3. `_project_sweep`: per iteration, sort by `(layer,x)` via `_layer_sorted_x_indices`, find consecutive same-layer pairs, push overlapping consecutive pairs by `0.25 * overlap_x`, optionally also second neighbors by `0.125 * overlap_x` when `N <= 100_000`. Required inputs only.
4. `_layer_sorted_x_indices`: stable x-sort, then stable sort that order by layer. Required inputs only.
5. `_project_sweep_cuda`: CUDA version of the same sweep and second-neighbor logic.
6. `_project_sweep_streaming`: iterate layers one at a time for `N > 100_000_000`, x-sort inside the layer, push consecutive overlaps with local scatter buffers, break when no layer moved.
7. `_project_grid`: derive grid cell size from max node size plus padding, hash nodes into cells, sort by cell key, iterate up to `10000` occupied multi-node cells, randomly subsample to `200` nodes per large cell, compute pairwise bbox overlaps inside the cell, and push pairs apart along the smaller-overlap axis by a quarter-overlap step.
8. `_copy_layer_index_to_cpu` and `_run_projection_impl` are dispatch helpers, not distinct ops.

Shared classic steps: `OverlapProjection` is the direct DESIGN mapping and is currently engine-only.

Hardcoded-but-parameterizable items: dispatch thresholds `500` and `100_000_000`; GPU estimate `72 bytes/node`; VRAM fraction `0.60`; push factors `0.25` and `0.125`; grid `cell_size >= 1.0`; `10000` max cells; `200` max nodes per large cell.

RNG: only `_project_grid` consumes RNG, via `torch.randperm(m)` when a cell has more than `200` nodes. That means the module is deterministic only if the global Torch RNG is seeded; the docstring’s unconditional “deterministically” claim is not literally true on that path.

Catalog status: `OverlapProjection` is present. `HardPinProjection` exists algorithmically, but in `constraints.py`, not here. `BoundaryClamp`, `MovementClamp`, and `MonotoneSafeguard` from the broader project catalog are absent from this file.

## [init_placement.py](/home/jtaylor/projects/dagua/dagua/layout/init_placement.py)

Core data: `edge_index[2,E]`, `node_sizes[N,2]`, `layers[N]`, Python `layer_groups: dict[int,list[int]]`, `children_of`/`parents_of` dicts on the small-graph path, tensor `counts`, `offsets`, `sorted_by_layer`, and `order[N]` on the vectorized path.

Engine integration: `_layout_inner()` calls `init_positions()` unless a warm-start `init_pos` is supplied. The result becomes the optimizer’s initial `pos`.

1. `init_positions`: 1) compute layers with `longest_path_layering`; 2) choose `_init_positions_vectorized` if `N > 100`, otherwise Python path; 3) Python path groups nodes by layer; 4) if edges exist, build adjacency and run alternating forward/backward barycenter passes with mean/median centers for `num_passes=min(max(15,N//5),40)`; 5) optionally run `_transpose_heuristic` for `<=500` nodes (8 passes) or `<=2000` (3 passes); 6) assign centered x/y coordinates from node widths, `node_sep`, and `rank_sep`; 7) run `_spread_fanout_children`. Tunables: `node_sep: float = 25.0`, `rank_sep: float = 50.0`, `device: str = "cpu"`, `verbose: bool = False`.
2. `_init_positions_vectorized`: 1) call `_choose_init_device`; 2) build `counts`, `offsets`, and `sorted_by_layer`; 3) try `_spectral_order` for large, not-too-dense graphs (`N > 10000`, `N <= spectral_cap`, `0 < E < 10N`); 4) map spectral scores to within-layer order with `_spectral_to_layer_order`, else use `_barycenter_order`; 5) assign `y = layer * rank_sep`; 6) assign `x = (order - layer_width/2) * (avg_width + node_sep)`; 7) run `_spread_fanout_children`; 8) move result back if compute device differed. Required tunables: `node_sep`, `rank_sep`, `device`.
3. `_choose_init_device`: estimate edge/node/work-buffer bytes and choose CPU when VRAM budget will not fit. Required inputs only.
4. `_spectral_order`: 1) symmetrize the directed graph; 2) build degree vector; 3) build sparse Laplacian `L = D - A`; 4) seed `torch.lobpcg` with `X0 = torch.randn(N,2)`; 5) solve for two smallest eigenpairs; 6) return the second eigenvector; 7) return `None` on failure. Required inputs only.
5. `_spectral_to_layer_order`: normalize spectral coordinates into `[0,N)`, build composite key `layer*(N+1)+spectral_norm`, stable-sort globally, and assign sequential within-layer order.
6. `_barycenter_order`: initialize order as within-layer index, precompute in/out degree, then run 12 tensorized forward/backward barycenter passes using `scatter_add_` and composite-key re-sorts.
7. `_transpose_heuristic`: repeatedly swap adjacent nodes inside each layer when `_count_local_crossings` decreases.
8. `_count_local_crossings`: compare the relative order of two nodes against their parents/children in adjacent layers and count inversions.
9. `_spread_fanout_children`: detect hubs with `out_degree >= 8`, compute required child span, widen it by `1.5x`, preserve child left-right order, and redistribute children evenly around the hub x-coordinate.
10. `_update_node_order`: flatten `layer_groups` into a monotonically increasing float order map.

Shared classic steps: `longest_path_layering` is the Sugiyama/engine/multilevel layering step; `_barycenter_order` is `BarycenterSweep`; `_transpose_heuristic` is `TransposeHeuristic`; `_spectral_order` and `_spectral_to_layer_order` are the engine’s `SpectralOrder`/`FiedlerVector` path; deterministic coordinate assignment corresponds to `DeterministicInit`.

Hardcoded-but-parameterizable items: vectorized cutoff `N > 100`; spectral threshold `N > 10000`; edge-density gate `E < 10N`; spectral caps `50_000_000` or `2_000_000` under VRAM pressure; Python barycenter passes `15..40`; tensor barycenter passes fixed at `12`; transpose passes `8` or `3`; lobpcg iterations `min(30, max(10, 60 - N//1_000_000))`; fanout hub threshold `8` and widening factor `1.5`.

RNG: only `_spectral_order` explicitly consumes randomness, through `torch.randn(N, 2, device=device)` for the initial LOBPCG guess. All other paths are deterministic given input ordering.

Catalog status: this file contains the substance of `SpectralOrder`, `BarycenterSweep`, `TransposeHeuristic`, and a deterministic coordinate assignment, but they are still fused into one monolithic initializer. Missing as separate cataloged ops are `RandomUniformInit`, `RandomNormalInit`, `CircularInit`, `MDS/PivotMDSInit`, `XavierInit`, `FromAlgorithmInit`, and a standalone coordinate op.

## [graph_classify.py](/home/jtaylor/projects/dagua/dagua/layout/graph_classify.py)

Core data: `GraphFamily`, `GraphStructure(family, num_components, max_degree, num_layers, avg_layer_width, is_planar_hint)`, degree tensor, union-find parent/rank arrays, and optional `layer_assignments[N]`.

Engine integration: `_layout_inner()` calls `classify_graph()` once after initialization/layer-index setup. The result is used immediately to apply `_override_for_tree(config)` and to cap chain steps at `50`.

1. `_compute_degree`: scatter-add undirected degrees from both source and target endpoints on CPU. Required inputs only.
2. `_find_root`: union-find path compression lookup. Required inputs only.
3. `_count_components_and_acyclic`: if `E > N-1`, bail out with `(1, False)`; otherwise union-find across edges, mark self-loops and repeated-root joins as cyclic, and return `(component_count, is_acyclic)`.
4. `_resolve_layer_assignments`: return provided layers as CPU `long`, or compute `longest_path_layering(edge_index.cpu(), num_nodes, device=("cuda" if available else "cpu"))`, then move result back to CPU.
5. `_analyze_layers`: `bincount` layers, count non-empty layers, compute `avg_layer_width = num_nodes / num_layers`.
6. `classify_graph`: 1) if `N > 10_000_000`, short-circuit to `GENERAL` and optionally keep layer stats; 2) otherwise compute degree and max degree; 3) compute components and acyclicity; 4) derive `is_tree`, `is_forest`, `is_chain`; 5) resolve layers and derive `is_bipartite_dag` (`num_layers==2`) and `is_wide_layered` (`avg_layer_width >= 100` and `num_layers <= max(N/100,1)`); 6) derive planar hint from `E < 3N-6`; 7) assign family priority `CHAIN > TREE > BIPARTITE_DAG > WIDE_LAYERED > FOREST > GENERAL`. Tunables: only optional `layer_assignments`; no threshold config exists.

Shared classic steps: `ClassifyGraph` is the direct mapping; `_count_components_and_acyclic` implements the `DetectComponents` substance that DESIGN.md also lists for LGL.

Hardcoded-but-parameterizable items: large-graph short-circuit at `10_000_000`; dense-graph acyclicity bailout when `E > N-1`; bipartite DAG heuristic `num_layers == 2`; wide-layered heuristic `avg_layer_width >= 100` and `num_layers <= N/100`; planar hint `E < 3N-6`.

RNG: none.

Catalog status: `ClassifyGraph` exists in substance, but its promised `threshold` config does not. `GraphFamily.GRID` is defined but unreachable; `classify_graph()` never returns it.

## [cycle.py](/home/jtaylor/projects/dagua/dagua/layout/cycle.py)

Core data: adjacency lists `node -> [(child, edge_idx)]`, node colors `WHITE/GRAY/BLACK`, `back_edge_mask[E]`, `children` lists for Kahn validation, greedy FAS `incoming_edges`/`outgoing_edges`, `order`, and `reversed_mask[E]`.

Engine integration: there is no direct import from `engine.py`. The only engine-side hook is `layout()` calling `graph._prepare_for_layout()` in [`graph.py`](/home/jtaylor/projects/dagua/dagua/graph.py), and that method imports `detect_back_edges()` plus `make_acyclic()` before `_layout_inner()` runs. Classic Sugiyama imports `make_acyclic_robust()` directly.

1. `detect_back_edges`: 1) convert edge tensors to Python lists; 2) build adjacency with edge indices; 3) compute in-degree and visit sources first; 4) run iterative DFS with `WHITE/GRAY/BLACK` coloring; 5) mark an edge as back when it reaches a `GRAY` child; 6) return `BoolTensor[E]`. Required inputs only.
2. `make_acyclic`: clone `edge_index` and swap `src/tgt` wherever `back_edge_mask` is true. Required inputs only.
3. `_is_acyclic`: run Kahn’s algorithm; return `processed == num_nodes`.
4. `_greedy_fas`: build incoming/outgoing edge index lists, repeatedly pick the active node maximizing `(in_degree - out_degree, -node_idx)`, append it to an order, deactivate its incident edges, then reverse every edge whose source appears after its target in that order.
5. `make_acyclic_robust`: 1) run `detect_back_edges`; 2) reverse those edges with `make_acyclic`; 3) if the result is acyclic by `_is_acyclic`, return it; 4) otherwise run `_greedy_fas` on the already-flipped graph and return its result.

Shared classic steps: `DetectCycles` and `MakeAcyclic` are direct mappings and are also used by classic Sugiyama.

Hardcoded-but-parameterizable items: DFS visit order is “all zero in-degree nodes first, then the rest”; the greedy fallback score is fixed to `(in_degree - out_degree, -node_idx)`; `make_acyclic_robust()` always uses DFS first and greedy second with no exposed method selector.

RNG: none.

Catalog status: the cataloged behavior exists, but the DESIGN.md `method(dfs,greedy)` configuration does not; the strategy is hardwired to DFS then greedy fallback.

## Cross-Reference

`constraints.py`: `_non_self_edges` -> helper for edge-based loss ops; `dag_ordering_loss` -> `DagOrderingLoss`; `edge_attraction_loss` -> `EdgeAttractionLoss`; `edge_straightness_loss` -> `EdgeStraightnessLoss`; `edge_length_variance_loss` -> `EdgeLengthVarianceLoss`; `repulsion_loss` -> `RepulsionLoss`; `_repulsion_exact`/`_repulsion_sampled`/`_repulsion_scatter`/`_repulsion_rvs`/`_repulsion_rvs_from_context` -> internal execution strategies for `RepulsionLoss`; `overlap_avoidance_loss` -> `OverlapAvoidanceLoss`; `_overlap_exact`/`_overlap_scatter`/`_overlap_active_subset`/`_overlap_active_subset_from_context`/`_overlap_grid_vectorized` -> internal execution strategies for `OverlapAvoidanceLoss`; `crossing_loss` -> `CrossingLoss`; `_crossing_loss_fallback`/`_crossing_loss_layered` -> internal execution strategies for `CrossingLoss`; `_resolve_cluster_members` -> helper for cluster ops; `cluster_compactness_loss` -> `ClusterCompactnessLoss`; `cluster_separation_loss` -> `ClusterSeparationLoss`; `cluster_containment_loss` -> `ClusterContainmentLoss`; `position_pin_loss` -> `PositionPinLoss`; `alignment_loss` -> `AlignmentLoss`; `flex_spacing_loss` -> `FlexSpacingLoss`; `project_hard_pins` -> `HardPinProjection`; `_spacing_consistency_loss_layerlocal` -> large-graph execution strategy for `SpacingConsistencyLoss`; `spacing_consistency_loss` -> `SpacingConsistencyLoss`; `fanout_distribution_loss` -> `FanoutDistributionLoss`; `back_edge_compactness_loss` -> `BackEdgeCompactnessLoss`.

`layers.py`: `_log_layer_index_status`/`_layer_index_gpu_sort_required_bytes`/`_cuda_layer_argsort` -> helpers for `BuildLayerIndex`; `build_layer_index` -> `BuildLayerIndex`. Missing cataloged layering ops in this file: `LongestPathLayering`, `LayerPromotion`, `InsertDummyNodes`.

`projection.py`: `_copy_layer_index_to_cpu`/`_run_projection_impl` -> projection helpers; `project_overlaps` -> `OverlapProjection`; `_project_exact`/`_project_sweep`/`_layer_sorted_x_indices`/`_project_sweep_cuda`/`_project_sweep_streaming`/`_project_grid` -> internal execution strategies for `OverlapProjection`. Missing in this file: `HardPinProjection` wrapper, plus other project-family ops outside native engine.

`init_placement.py`: `init_positions` -> fused `DeterministicInit` pipeline using `LongestPathLayering`, `BarycenterSweep`, optional `TransposeHeuristic`, and coordinate assignment; `_init_positions_vectorized` -> fused engine-init pipeline; `_choose_init_device` -> helper; `_spectral_order` -> `FiedlerVector` / `SpectralOrder`; `_spectral_to_layer_order` -> `SpectralOrder` postprocessing; `_barycenter_order` -> `BarycenterSweep`; `_transpose_heuristic` -> `TransposeHeuristic`; `_count_local_crossings` -> helper for `TransposeHeuristic`; `_spread_fanout_children` -> no separate cataloged op today; `_update_node_order` -> helper.

`graph_classify.py`: `_compute_degree` -> helper for `ClassifyGraph`; `_find_root` -> helper for `DetectComponents`; `_count_components_and_acyclic` -> substance of `DetectComponents`; `_resolve_layer_assignments`/`_analyze_layers` -> helpers for `ClassifyGraph`; `classify_graph` -> `ClassifyGraph`. Missing: threshold config and any real `GRID` classification op/branch.

`cycle.py`: `detect_back_edges` -> `DetectCycles`; `make_acyclic` -> `MakeAcyclic`; `_is_acyclic` -> helper; `_greedy_fas` -> greedy fallback strategy inside `MakeAcyclic`; `make_acyclic_robust` -> fused `DetectCycles + MakeAcyclic` pipeline with fallback. Missing: exposed `method(dfs,greedy)` config promised by DESIGN.md.

Codex session ID: 019d4fcf-7b52-78e2-99ed-11d315d8f603
Resume in Codex: codex resume 019d4fcf-7b52-78e2-99ed-11d315d8f603
