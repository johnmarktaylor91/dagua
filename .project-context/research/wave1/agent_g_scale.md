## [multilevel.py](/home/jtaylor/projects/dagua/dagua/layout/multilevel.py)

**Execution order**
1. `engine.layout()` calls `multilevel_layout()` when `n > config.multilevel_threshold` ([engine.py:914](/home/jtaylor/projects/dagua/dagua/layout/engine.py)).
2. `multilevel_layout()` normalizes device, resolves progress path, reseeds with `torch.manual_seed(config.seed)`, classifies the graph, and takes a tree/chain fast path if possible.
3. It restores a precomputed hierarchy if present, otherwise `build_hierarchy()` computes longest-path layers once and repeatedly calls `coarsen_once()` until `num_nodes <= min_nodes`, `max_levels` is hit, or reduction falls below `30%`.
4. `coarsen_once()` chooses streaming for `N` or `E` above `100_000_000`; otherwise it computes per-node adjacency signatures, degrees, skip statistics, hub thresholds, layer-local ordering, pair/triple compatibility, then scans assignments into coarse groups.
5. Both coarsening paths build coarse node sizes with `scatter_reduce_(amax)` and remap/deduplicate coarse edges.
6. `multilevel_layout()` lays out the coarsest level via `_layout_inner()`, then walks levels back to fine: reload finer graph if offloaded, choose refinement execution mode, prolong positions by indexed copy plus jitter, apply huge-graph overrides, then refine with `_layout_inner()`.
7. Final output is direction-transformed and temporary offload artifacts are cleaned.

**Configurable parameters**
- Direct function params: `build_hierarchy(min_nodes:int=2000, max_levels:int=20, device:str="cpu", progress:Optional[Callable]=None, cluster_ids:Optional[Tensor]=None, initial_layer_assignments:Optional[Tensor]=None, layer_assignments_callback:Optional[Callable]=None, level_callback:Optional[Callable]=None, offload_to_disk:bool=True)`, `coarsen_once(device:str="cpu", cluster_ids:Optional[Tensor]=None, progress:Optional[Callable]=None)`, `prolong_positions(device:str="cpu")`.
- `LayoutConfig` fields read here: `multilevel_min_nodes:int=2000`, `multilevel_coarse_steps:int=100`, `multilevel_refine_steps:int=25`, `multilevel_threshold:int=20000`, `device:str="cpu"`, `seed:Optional[int]=42`, `verbose:bool=False`, `direction:str="TB"`, `lr:float=0.05`, `node_sep:float=28.0`, `rank_sep:float=50.0`, `w_dag:float=10.0`, `w_attract:float=2.0`, `w_attract_x_bias:float=2.4`, `w_repel:float=0.1`, `w_overlap:float=5.0`, `w_crossing:float=1.8`, `w_straightness:float=2.2`, `w_length_variance:float=0.7`, `exact_repulsion_threshold:int=2000`, `negative_sample_k:int=128`, `per_loss_backward:str="auto"`, `gradient_checkpointing:str="auto"`, `hybrid_device:str="auto"`, `execution_mode:Literal["auto","standard","subset_gpu"]="auto"`, `subset_gpu_threshold:int=10_000_000`, `optimizer_fallback:Literal["auto","adam","sgd"]="auto"`, `num_workers:int=0`, `edge_batch_size:int=0`, `edge_random_fraction:float=0.2`, `overlap_check_interval:int=0`, `offload_to_disk:bool=True`.

**Data structures**
- `CoarseLevel`: `edge_index`, `node_sizes`, `num_nodes`, `fine_to_coarse`, `num_fine`, `fine_layer_assignments`, `coarse_layer_assignments`, `offload_path`, `offload_dir`.
- Streaming coarsen state: `min_neighbor`, `layer_counts`, `layer_offsets`, `coarse_offsets`, block tuples `(layer_start, layer_end, node_start, node_end)`, bucketed unique edge hashes.
- Non-streaming coarsen state: `min_parent`, `min_child`, `in_degree`, `out_degree`, `skip_degree`, `mean_span`, `global_order`, `hub_threshold_per_layer`, NumPy `pair_ok`/`triple_ok`/`group_ids`.
- Refinement state: original-graph checkpoint path, per-level `fine_ei_cpu`/`fine_sizes_cpu`, `fine_pos`, sample-cap override, progress metadata.

**Engine callers**
- `engine.layout()` is the only direct caller from `engine.py`; it dispatches the entire large-graph solve to `multilevel_layout()` ([engine.py:914](/home/jtaylor/projects/dagua/dagua/layout/engine.py)).
- `multilevel.py` then calls back into engine internals: `_layout_inner()` for coarsest layout and every refinement stage, `_auto_edge_batch_size()`, `_resolve_progress_file_path()`, `_apply_direction()`, and GPU memory estimators.

**Memory management**
- Hierarchy build is CPU-resident by default; streaming coarsen borrows CUDA only for guarded scatter/sort/dedup scratch.
- Previous levels can be serialized with `_offload_level_to_disk()` and reloaded on demand with `_reload_level_from_disk()`.
- The original graph is separately offloaded before deep coarsening/refinement on `n > 10_000_000`.
- The code aggressively drops references (`graph._edge_index_tensor`, `graph.node_sizes`, `fine_to_coarse`, earlier `edge_index`/`node_sizes`), calls `gc.collect()`, `malloc_trim(0)`, and `torch.cuda.empty_cache()` on fallback paths.
- Final-level sampling and overlap/crossing amortization are reduced under RAM pressure.

**Hardcoded likely-params**
- Module constants: streaming threshold `100_000_000`, dedup bucket target `150_000_000`, hub percentile `90.0`, hub floor `8`, skip anchor `(degree=2, span=1.5)`, final sample cap `1_000_000`, CPU edge batch caps `500_000..20_000_000`, refinement VRAM fraction `0.40`, streaming VRAM fractions `0.70/0.60`.
- Policy constants embedded in flow: offload cutoff `10_000_000`, coarse-step caps `30/20`, huge-refine cutoff `50_000_000`, subset-GPU force at `200_000_000`, final-step caps `18/12/8`, reduction stop factor `0.7`, prolong jitter `5.0`, progress file cutoff `1_000_000`, hardcoded locker path `/mnt/locker/jt3295/dagua_bench_large`.

**RNG**
- This file calls `torch.manual_seed(config.seed)` once at `multilevel_layout()` entry.
- The active prolongation path consumes RNG via `torch.randn(level.num_fine, 2, device=device)` on GPU prolongation or `torch.randn(level.num_fine, 2)` on CPU prolongation; `prolong_positions()` also does this but is not used by the active V-cycle.
- No coarsening step is random.
- Because `engine.layout()` already seeds before calling this module, the multilevel path resets the global RNG again to the same seed; afterward, refinement configs pass `seed=None`, so all later draws consume the evolving global generator state rather than reseeding per level.

**Catalog mapping**
- Present in `DESIGN.md`: `LayerAwareCoarsen`, `DirectMapping`, `DiskOffload`, `DiskReload`, `GarbageCollect`, `MultilevelVCycle`.
- Missing or only implicit: streaming segmented-sort assignment, chunked min-neighbor scatter, bucketed edge dedup, refinement execution-policy selection, sample-cap override, original-graph offload/reload, and huge-final-level subset-GPU policy.

## [tiled_compute.py](/home/jtaylor/projects/dagua/dagua/layout/tiled_compute.py)

**Execution order**
1. `_layout_inner()` instantiates `TiledGPUCompute` in standard execution mode when tiled-GPU heuristics say full CUDA residency is unsafe but tiling is viable ([engine.py:1533](/home/jtaylor/projects/dagua/dagua/layout/engine.py)).
2. `__init__` copies edges/node sizes to CPU, computes VRAM budget, edge batch size, tile size, tile ranges, tile-local edges, cross-tile edges, and total non-self edge count.
3. `compute_step()` splits loss terms into node-only vs cross-tile edge terms.
4. For each tile: gather positions/sizes to CUDA, optionally build tile-local layer index, run node losses, then iterate tile-local edge batches, building `_TileEdgeContext` for each batch.
5. For cross-tile edges: unique active nodes, gather only those nodes, remap edges locally, run edge losses, scatter gradients back.
6. Accumulated CPU gradient buffer is stored into `positions.grad`.

**Configurable parameters**
- Constructor params: `device:str="cuda"`, `vram_budget:Optional[int]=None`.
- Runtime params: `_iter_edge_batches(batch_size:Optional[int]=None)`, `_backward_loss_group(scale:float=1.0)`.
- No `LayoutConfig` fields are read inside this module; all policy comes from engine-side dispatch plus module constants.

**Data structures**
- `_TileEdgeContext`: `src`, `tgt`, `dx`, `dy`, `dist_sq`.
- `TiledGPUCompute` runtime state: `tile_starts`, `tile_ends`, `tile_edges`, `cross_edges`, `layer_assignments`, `cross_tile_loss_mask`, `current_edge_index`, `current_edge_ctx`, `last_unweighted_loss`, CPU `grad_buffer`.

**Engine callers**
- `_layout_inner()` constructs the object, installs `layer_assignments`, sets `cross_tile_loss_mask`, and calls `compute_step()`; if tiled execution OOMs, engine resets runtime state and falls back to CPU losses for that step ([engine.py:2432](/home/jtaylor/projects/dagua/dagua/layout/engine.py)).

**Memory management**
- Positions stay on CPU; only one tile or one active cross-edge subset is moved to CUDA at a time.
- CPU RAM guard checks `psutil.virtual_memory().available`.
- Frequent `torch.cuda.synchronize()` calls surface transfer OOMs early.
- `torch.cuda.empty_cache()` is called after each tile and each cross-edge batch.
- Gradients are copied back immediately and CUDA leaf grads are cleared.

**Hardcoded likely-params**
- `_BYTES_PER_TILE_NODE=64`, `_CUDA_CONTEXT_OVERHEAD_BYTES=500_000_000`, `_EDGE_BATCH_BYTES=256`, `_EDGE_BUDGET_RATIO=0.30`, `_CROSS_EDGE_HEADROOM_RATIO=0.80`, `_MIN_TILE_SIZE=1_000_000`, `_MIN_EDGE_BATCH=100_000`, `_fit_tile_size_to_vram()` safe-limit factor `0.85`, CPU RAM headroom `1.3`.

**RNG**
- None. Tiled execution is deterministic modulo CUDA kernel nondeterminism in downstream loss ops.

**Catalog mapping**
- `DESIGN.md` explicitly treats tiled GPU as a `LossGroup` execution strategy, not a standalone op.
- Closest cataloged ops used internally: `BuildLayerIndex` for `_tile_layer_index`, `BuildEdgeBatchCtx` for `_edge_context`.
- Missing explicit ops if you want first-class migration: tile partitioning, cross-tile active-node gather/remap, gradient scatter-back.

## [subset_gpu.py](/home/jtaylor/projects/dagua/dagua/layout/subset_gpu.py)

**Execution order**
1. `_layout_inner()` enters subset-GPU mode, builds `SubsetGPULossTerm` objects with `EdgeAccessPattern`, `SampledAccessPattern`, or `GlobalAccessPattern`, then constructs `SubsetGPUExecutor` ([engine.py:2326](/home/jtaylor/projects/dagua/dagua/layout/engine.py)).
2. `compute_step()` allocates or reuses a full-size CPU gradient buffer, resets scalar loss totals, and precomputes shared edge/sampled remaps when possible.
3. It iterates active loss terms: shared-edge path, shared-sampled path, global CPU path, or per-term subset path.
4. Each subset path gathers `pos_local` and `node_sizes_local` to the execution device, temporarily swaps engine refs to local edges/sample indices, evaluates the loss, takes `autograd.grad`, and scatters gradients into the global CPU buffer with `index_add_`.
5. Periodic progress heartbeats fire every 30 seconds.
6. The final CPU gradient buffer is returned to engine and assigned to `pos.grad`.

**Configurable parameters**
- `SampledAccessPattern(active_row_cap:Optional[int]=None)`.
- `SubsetGPUExecutor(verbose:bool=False)`.
- `compute_step(verbose:Optional[bool]=None, progress_callback:Optional[Callable[[float],None]]=None)`.
- No direct `LayoutConfig` reads here; engine decides subset mode, loss access patterns, and sampled contexts.

**Data structures**
- `PreparedSubsetData`, `SharedEdgeSubsetData`, `SharedSampledSubsetData`.
- Access-pattern markers: `GlobalAccessPattern`, `EdgeAccessPattern`, `SampledAccessPattern`, `UnionAccessPattern`.
- `SubsetGPULossTerm`.
- Executor mutable refs: `batch_edges_ref`, `edge_ctx_ref`, `sampled_ctx_ref`, plus reusable `_grad_buffer`.

**Engine callers**
- Only `_layout_inner()` calls this module directly. It wires engine-owned mutable refs into the executor so existing loss lambdas can run unchanged on local subsets.

**Memory management**
- Full `pos` and `node_sizes` remain CPU-resident.
- Shared remap paths estimate VRAM and refuse CUDA if required bytes exceed `60%` of free VRAM.
- Local remaps use int32 indices below `2**31` to halve index memory.
- Shared edge/sampled gathers are reused across multiple terms within a step.
- OOM on shared remaps falls back to per-term execution and clears CUDA cache.

**Hardcoded likely-params**
- `_PROGRESS_LOG_INTERVAL_SECONDS=30.0`, `_SHARED_EDGE_REMAP_VRAM_FRACTION=0.60`, `_INT32_INDEX_LIMIT=2**31`.
- Shared-edge reuse requires identical edge-batch object identity, not semantic equality.

**RNG**
- None. Access-pattern extraction, remapping, and gradient scattering are deterministic.

**Catalog mapping**
- `DESIGN.md` treats subset-GPU as execution-plan state (`subset_gpu_threshold`) plus `LossGroup` behavior, not a separate op.
- Closest cataloged ops consumed or supported: `BuildEdgeBatchCtx`, `RefreshSampledNodeCtx`, `LossGroup`.
- Missing explicit ops if desired: subset gather, local-index remap, shared sampled remap, gradient scatter-back.

## [cuda_kernels.py](/home/jtaylor/projects/dagua/dagua/layout/cuda_kernels.py)

**Execution order**
1. `is_available()` checks CUDA and attempts lazy compilation.
2. `_load_csr_kernel()` compiles and caches an inline CUDA extension exposing `csr_scatter`.
3. `build_csr_cuda()` validates CUDA tensors, computes `out_degree`, prefix-sums to `csr_offsets`, clones `write_pos`, dispatches the kernel, and returns `(csr_offsets, csr_targets)`.

**Configurable parameters**
- `build_csr_cuda(num_nodes:int)` and `is_available()` expose no tunable defaults.
- Kernel launch uses a hardcoded `threads=256`; module name is hardcoded to `dagua_csr_cuda`.

**Data structures**
- Global cached module `_csr_module`.
- CSR outputs: `csr_offsets[num_nodes+1]`, `csr_targets[E]`.
- Temporary buffer: `write_pos = csr_offsets[:-1].clone()`.

**Engine callers**
- None in `engine.py`.
- Practical caller is `dagua.utils`’ CUDA CSR path, not the layout engine dispatcher.

**Memory management**
- Lazy-compilation cache avoids recompiling the extension.
- Input tensors are made contiguous before kernel launch.
- No explicit offload, GC, or cache clearing.

**Hardcoded likely-params**
- Kernel thread block size `256`, extension name `dagua_csr_cuda`, atomic-add implementation choice, separate int32 vs int64 kernels.

**RNG**
- None.

**Catalog mapping**
- No direct op in `DESIGN.md`.
- Closest existing catalog concept is `BuildAdjacency`; otherwise this is missing low-level kernel infrastructure.

## [edge_optimization.py](/home/jtaylor/projects/dagua/dagua/layout/edge_optimization.py)

**Execution order**
1. `optimize_edges()` early-exits on `E==0` or `edge_opt_steps < 0`; `edge_opt_steps == 0` auto-scales to `min(100, max(20, E*2))`.
2. It extracts endpoints and initial control points from `BezierCurve` objects, detaches positions/node sizes to CPU float tensors, and reads six loss weights from config.
3. It precomputes cluster boxes and per-node cluster membership, plus source/target edge lists and fixed `t_samples`.
4. Each optimization step evaluates Bezier sample points, accumulates losses in this order: edge-edge crossing, edge-node crossing, angular resolution, curvature consistency, curvature penalty, edge-cluster crossing.
5. If differentiable, it backprops, zeroes non-finite grads, clips grad norm to `50.0`, steps Adam, and repairs non-finite control points with linear-interpolation fallbacks.
6. Final control points are converted back to `BezierCurve`; if final `cp` is non-finite, the original curves are returned.

**Configurable parameters**
- Direct args: `graph:Optional[object]=None`, `trace=None`.
- Config fields with defaults/types from `LayoutConfig`: `edge_opt_steps:int=0`, `edge_opt_lr:float=0.1`, `w_edge_crossing:float=5.0`, `w_edge_node_crossing:float=10.0`, `w_edge_angular_res:float=2.0`, `w_edge_curvature_consistency:float=1.0`, `w_edge_curvature_penalty:float=0.5`, `w_edge_cluster_crossing:float=8.0`.

**Data structures**
- Endpoints tensor `[E,2,2]`, control points tensor `[E,2,2]`, Bezier sample points `[E,T,2]`.
- Cluster data: `cluster_bboxes[C,4]`, `node_cluster_mask[N,C]`.
- Edge endpoint lists: `src_list`, `tgt_list`.

**Engine callers**
- None in `engine.py`.
- This module is called from the higher-level public draw/export pipeline, not from the layout engine.

**Memory management**
- Entire routine is CPU-resident.
- No offload/checkpoint/GC behavior.
- Stability guards reduce wasted work: skip on `E==0`, skip on `steps<0`, repair non-finite grads/control points, and return original curves if final state diverges.

**Hardcoded likely-params**
- `T=10` samples, `max_pairs=5000`, crossing `sharpness=10.0`, node-crossing `safety_margin=3.0`, node sample cap `500`, angular target `pi/8`, grad clip `50.0`, fallback control-point mixes `0.667/0.333`.

**RNG**
- No local seeding.
- `_edge_crossing_loss()` consumes global CPU RNG via `torch.randint()` when `E*(E-1)/2 > 5000`.
- `_edge_node_crossing_loss()` consumes global CPU RNG via `torch.randperm()` when `N > 500`.
- If `w_edge_crossing <= 0`, `w_edge_node_crossing <= 0`, `E <= 1`, or `N <= 500`, those draws do not happen.

## [ops/DESIGN.md](/home/jtaylor/projects/dagua/dagua/layout/ops/DESIGN.md)

Relevant catalog entries:
- `LayerAwareCoarsen`, `DirectMapping`, `MultilevelVCycle`, `DiskOffload`, `DiskReload`, `GarbageCollect`.
- `BuildEdgeBatchCtx`, `RefreshSampledNodeCtx`, `LossGroup`.
- `BezierControlPointOpt`.
- Explicit note: tiled GPU is a `LossGroup` execution strategy, not its own op; `subset_gpu_threshold` lives in `ExecutionPlan`.

## Cross-Reference

| Function group | DESIGN.md op | Status |
|---|---|---|
| `multilevel_layout`, `build_hierarchy` | `MultilevelVCycle` | Direct match |
| `coarsen_once`, `_coarsen_once_streaming`, `_match_scan_python`, `_build_match_scan`, `_compute_hub_thresholds`, `_build_streaming_min_neighbor`, `_stable_argsort_on_device`, `_build_streaming_assignment_blocks`, `_streaming_block_group_ids`, `_streaming_block_order`, `_assign_streaming_coarse_groups`, `_deduplicate_streaming_coarse_edges` | `LayerAwareCoarsen` | Direct match; streaming helpers are implementation details not separately cataloged |
| `prolong_positions` and the inlined prolongation inside `multilevel_layout` | `DirectMapping` | Direct match; helper is currently unused |
| `_offload_level_to_disk` | `DiskOffload` | Direct match |
| `_reload_level_from_disk` | `DiskReload` | Direct match |
| `_cleanup_offloaded_hierarchy` plus explicit `gc.collect()/malloc_trim()/empty_cache()` sites | `GarbageCollect` | Direct match |
| `_select_refinement_execution`, `_positions_fit_cuda_refinement`, `_available_ram_bytes`, `_auto_cpu_edge_batch_size`, `_apply_large_final_level_execution_overrides`, `_scaled_final_refine_steps`, `_scaled_sample_cap`, `_scaled_amortization`, `_apply_final_level_memory_guard`, `_temporary_sample_cap` | No exact op | Missing policy/utility ops for execution planning and guarded refinement |
| `TiledGPUCompute.compute_step`, `_backward_loss_group` | `LossGroup` | Match by DESIGN note: tiled execution strategy |
| `TiledGPUCompute._tile_layer_index` | `BuildLayerIndex` | Closest match |
| `TiledGPUCompute._edge_context` | `BuildEdgeBatchCtx` | Closest match |
| `_compute_tile_size`, `_partition_edges_by_tile`, `_compute_edge_batch_size`, `_compute_cross_edge_batch_size`, `_accumulate_leaf_grad`, `_reset_runtime_state` | No exact op | Missing tiling utility ops |
| `SubsetGPUExecutor.compute_step` | `LossGroup` | Match by execution strategy |
| `EdgeAccessPattern`, `_build_local_edge_context`, `_local_refs`, `_shared_edge_refs`, `_prepare_shared_edge_data`, `_accumulate_shared_edge_grad` | `BuildEdgeBatchCtx` | Closest match |
| `SampledAccessPattern`, `_shared_sampled_refs`, `_prepare_shared_sampled_data`, `_accumulate_shared_sampled_grad` | `RefreshSampledNodeCtx` | Closest match |
| `GlobalAccessPattern`, `UnionAccessPattern`, `PreparedSubsetData`, `SharedEdgeSubsetData`, `SharedSampledSubsetData`, `_prepare_grad_buffer`, `_accumulate_global_grad`, `_accumulate_subset_grad`, `_record_loss`, `_loss_grad` | `LossGroup` | Internal execution detail, not separately cataloged |
| `_load_csr_kernel`, `build_csr_cuda`, `is_available` | Closest: `BuildAdjacency` | Missing low-level kernel op in catalog |
| `optimize_edges` | `BezierControlPointOpt` | Direct match |
| `_evaluate_bezier_batch`, `_edge_crossing_loss`, `_edge_node_crossing_loss`, `_port_angular_resolution_loss`, `_bezier_derivatives_batch`, `_curvature_consistency_loss`, `_curvature_penalty_loss`, `_edge_cluster_crossing_loss`, `_build_cluster_data_for_edge_opt` | `BezierControlPointOpt` | Internal sub-terms; catalog intentionally keeps them collapsed into one op |

Codex session ID: 019d4fcf-9426-77d1-ad41-09115af94b08
Resume in Codex: codex resume 019d4fcf-9426-77d1-ad41-09115af94b08
