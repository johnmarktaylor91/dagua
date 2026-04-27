## Scope

This checkout’s native engine is 3,274 lines, not “~1200”. I read [engine.py](/home/jtaylor/projects/dagua/dagua/layout/engine.py#L870), [config.py](/home/jtaylor/projects/dagua/dagua/config.py#L27), [multilevel.py](/home/jtaylor/projects/dagua/dagua/layout/multilevel.py#L1449), [init_placement.py](/home/jtaylor/projects/dagua/dagua/layout/init_placement.py#L23), [projection.py](/home/jtaylor/projects/dagua/dagua/layout/projection.py#L49), [constraints.py](/home/jtaylor/projects/dagua/dagua/layout/constraints.py#L1301), and [DESIGN.md](/home/jtaylor/projects/dagua/dagua/layout/ops/DESIGN.md#L227).

## 1. Computational Steps, In Execution Order

### Top-level `layout()`
1. Build default `LayoutConfig` if absent.
2. Force `graph.compute_node_sizes()`.
3. Resolve effective device:
   - downgrade `cuda -> cpu` for `N < 1000`
   - downgrade if CUDA unavailable.
4. Early-return zero/one-node layouts.
5. Seed PyTorch if `config.seed is not None` ([engine.py](/home/jtaylor/projects/dagua/dagua/layout/engine.py#L902)).
6. Call `graph._prepare_for_layout()` so the engine sees a DAG.
7. Branch:
   - `N > multilevel_threshold`: call `multilevel_layout()`.
   - else: direct `_layout_inner(...)`.
8. Optional relaxation pass:
   - rerun `_layout_inner(...)` with `w_dag=0`, `steps=relax_steps`, `lr *= 0.5`.
9. Apply direction transform (`TB/BT/LR/RL`).
10. Cache layout on the graph.
11. `finally`: restore graph state via `graph._restore_after_layout()`.

### Direct `_layout_inner()`
1. Resolve execution mode: `standard` vs `subset_gpu`.
2. Normalize `node_sizes` to `[N,2]`.
3. Force `edge_index` to CPU in `subset_gpu`.
4. Resolve progress file path for very large runs.
5. Apply adaptive spacing if enabled.
6. Initialize positions:
   - warm start if `init_pos` supplied
   - else `init_positions(...)`.
7. Trace initial positions if trace sink exists.
8. Build/prebuild `LayerIndex`:
   - reuse provided index
   - or build from explicit `layer_assignments`
   - or compute `longest_path_layering()` then build.
9. Compute adaptive runtime controls:
   - `num_edges`
   - `edge_batch`
   - `overlap_interval`
   - `steps` (`config.steps` or auto).
10. Classify graph structure unless `skip_classification=True`.
11. Override weights for `TREE`/`CHAIN`; cap chain steps at 50.
12. Decide whether edges stay resident or stream from CPU.
13. Optionally activate tiled GPU loss evaluation.
14. Resolve memory strategy:
   - standard combined backward
   - per-loss backward
   - checkpointing
   - hybrid CPU/GPU.
15. Optionally create thread pool for hybrid heavy losses.
16. Build CPU mirrors for node sizes/layers/edges if hybrid.
17. Turn `pos` into the sole learnable parameter and create optimizer.
18. Build static loss function list once.
19. Preallocate batch buffers.
20. Optimization loop per step:
   1. Compute normalized time `t`.
   2. Zero gradients.
   3. Select edge batch:
      - random sampled batch
      - or contiguous wrapped chunk
      - or full edge set.
   4. Drop self-loops from active batch.
   5. Build `EdgeBatchContext` unless per-loss backward is active.
   6. Refresh `SampledNodeContext` on its cadence.
   7. Compute annealed weights.
   8. Materialize active loss term list.
   9. Execute losses via one of:
      - `SubsetGPUExecutor`
      - tiled GPU
      - per-loss backward
      - combined backward.
   10. Clip gradient norm to `100.0`.
   11. `optimizer.step()`.
   12. Hard-pin projection.
   13. Periodic hard overlap projection.
   14. Trace step positions.
   15. Progress logging / JSON snapshots.
   16. Early-stop check on unweighted loss stall.
21. Final forced progress snapshot.
22. Delete optimizer and clear grad.
23. Final aggressive overlap projection.
24. Trace final positions.
25. Shutdown hybrid executor.
26. Return detached `pos`.

## 2. Configurable Parameters, Defaults, Types

From [config.py](/home/jtaylor/projects/dagua/dagua/config.py#L27):

### Used directly by the engine path
- `node_sep: float = 28.0`
- `rank_sep: float = 50.0`
- `direction: str = "TB"`
- `steps: int = 0`
- `lr: float = 0.05`
- `device: str = "cpu"`
- `seed: Optional[int] = 42`
- `adaptive_spacing: bool = True`
- `verbose: bool = False`
- `w_dag: float = 10.0`
- `w_attract: float = 2.0`
- `w_attract_x_bias: float = 2.4`
- `w_repel: float = 0.1`
- `w_overlap: float = 5.0`
- `w_cluster: float = 1.0`
- `w_cluster_contain: float = 2.0`
- `w_crossing: float = 1.8`
- `w_straightness: float = 2.2`
- `w_length_variance: float = 0.7`
- `w_spacing: float = 0.3`
- `w_fanout: float = 0.3`
- `w_back_edge: float = 0.3`
- `exact_repulsion_threshold: int = 2000`
- `negative_sample_k: int = 128`
- `multilevel_threshold: int = 20000`
- `multilevel_min_nodes: int = 2000`
- `multilevel_coarse_steps: int = 100`
- `multilevel_refine_steps: int = 25`
- `rvs_threshold: int = 100000`
- `rvs_nn_k: int = 20`
- `per_loss_backward: str = "auto"`
- `gradient_checkpointing: str = "auto"`
- `hybrid_device: str = "auto"`
- `execution_mode: Literal["auto","standard","subset_gpu"] = "auto"`
- `subset_gpu_threshold: int = 10_000_000`
- `optimizer_fallback: Literal["auto","adam","sgd"] = "auto"`
- `num_workers: int = 0`
- `edge_batch_size: int = 0`
- `overlap_check_interval: int = 0`
- `repel_amortize_interval: int = 2`
- `repel_amortize_threshold: int = 10_000_000`
- `edge_random_fraction: float = 0.2`
- `relax_steps: int = 0`
- `offload_to_disk: bool = True`
- `flex: Optional[LayoutFlex] = None`

### Present in `LayoutConfig` but not wired in `engine.py`
- `w_align: float = 5.0`
- `fanout_amortize_interval: int = 3`
- `fanout_amortize_threshold: int = 10_000_000`
- `edge_opt_steps`, `edge_opt_lr`, `w_edge_*`

### Hidden / non-dataclass config hooks
- `progress_path`: read via `getattr(config, "progress_path", None)` for progress JSON.
- `_dagua_crossing_interval_override`: private multilevel override for crossing cadence.

## 3. Data Structures Used

### Core tensors
- `pos: Tensor[N,2]`
  - the only optimized parameter.
- `edge_index: Tensor[2,E]`
  - directed edge list.
- `node_sizes: Tensor[N,2]`
  - width/height, normalized from `[N]` or `[N,1]`.
- `layer_assignments_raw: Tensor[N] | list[int] | None`

### `LayerIndex`
From [layers.py](/home/jtaylor/projects/dagua/dagua/layout/layers.py#L15):
- `node_to_layer: Tensor[N]`
- `layer_offsets: Tensor[L+1]`
- `sorted_nodes: Tensor[N]`
- `num_layers: int`

### Cached edge context
From [engine.py](/home/jtaylor/projects/dagua/dagua/layout/engine.py#L110):
- `src: Tensor[B]`
- `tgt: Tensor[B]`
- `dx: Tensor[B]`
- `dy: Tensor[B]`
- `dist_sq: Tensor[B]`

### Cached sampled-node context
From [engine.py](/home/jtaylor/projects/dagua/dagua/layout/engine.py#L121) and [engine.py](/home/jtaylor/projects/dagua/dagua/layout/engine.py#L502):
- `active_idx: Tensor[A]`
- `sampled: Tensor[A, k_same + n_random]`
  - first block: same-layer samples
  - second block: adjacent-layer random samples.

### Flex cache dict
From [engine.py](/home/jtaylor/projects/dagua/dagua/layout/engine.py#L3081):
- `has_soft_pins: bool`
- `has_hard_pins: bool`
- `pin_indices: Tensor[P]`
- `pin_targets: Tensor[P,2]`
- `pin_weights: Tensor[P,2]`
- `soft_pin_mask: Tensor[P,2]`
- `hard_pin_mask: Tensor[P,2]`
- `align_groups: list[(Tensor[G], float, axis)]`
- `flex_node_sep: float | None`
- `flex_node_sep_weight: float`

### Multilevel hierarchy
From [multilevel.py](/home/jtaylor/projects/dagua/dagua/layout/multilevel.py#L302):
- `CoarseLevel.edge_index: Tensor[2,E_c] | None`
- `CoarseLevel.node_sizes: Tensor[N_c,2] | None`
- `CoarseLevel.num_nodes: int`
- `CoarseLevel.fine_to_coarse: Tensor[N_fine] | None`
- `CoarseLevel.num_fine: int`
- `CoarseLevel.fine_layer_assignments: Tensor[N_fine] | None`
- `CoarseLevel.coarse_layer_assignments: Tensor[N_c] | None`
- `CoarseLevel.offload_path/offload_dir`

### Important absence
- No persistent `forces` buffer exists in the real engine. The implementation is loss-based/autograd-based, not force-accumulation-based.

## 4. Optimizer / Solver Used And How

From [engine.py](/home/jtaylor/projects/dagua/dagua/layout/engine.py#L995):
- Default optimizer: `torch.optim.Adam([pos], lr=config.lr)`.
- Fallbacks:
  - `sgd_nesterov`: `SGD(lr=config.lr*3, momentum=0.9, nesterov=True)`
  - `sgd`: `SGD(lr=config.lr*5)`
- Only `pos` is optimized.
- No LR decay scheduler.
- Gradient clipping: `clip_grad_norm_([pos], 100.0)`.
- Backward modes:
  - combined backward
  - per-loss backward
  - checkpointed heavy losses
  - hybrid CPU heavy-loss backward via `_GradBridge`
  - subset-GPU executor
  - tiled GPU executor.

## 5. Convergence Criteria

From [engine.py](/home/jtaylor/projects/dagua/dagua/layout/engine.py#L2615):
- Primary stop: fixed `steps` or auto steps:
  - `<=10: 50`
  - `<=50: 100`
  - `<=200: 150`
  - `<=500: 200`
  - `<=2000: 250`
  - `<=5000: 300`
  - `<=10000: 400`
  - else `500`
- Early stopping uses **unweighted** loss.
- After `step > 10`:
  - `rel_threshold = 5e-4` if `N <= 200`, else `1e-4`
  - `stall_limit = 3` if `N <= 200`, else `5`
  - if `N <= 5000`, effective stall limit becomes `3`
- Break once relative change stays below threshold for the stall limit.

## 6. Steps Shared With Classic Algorithms

- `LongestPathLayering`: Sugiyama-style layering.
- `Barycenter` / median / transpose ordering in initialization: Sugiyama crossing-reduction family.
- Spectral/Fiedler initialization: spectral layout family.
- Attraction + repulsion + overlap penalties: force-directed / energy-minimization family.
- Crossing minimization as a sampled proxy: Sugiyama-style crossing reduction.
- Multilevel coarsen / prolong / refine: SFDP/FMMM-style multilevel layout.
- Hard projection after gradient step: projected optimization / constraint projection.

## 7. Hardcoded Things That Could Be Parameters

High-impact hardcoded behavior knobs include:
- Adaptive spacing scales and breakpoints: `1.3 / 1.0 / 0.85 / 0.7`.
- Auto step schedule thresholds.
- Tree override zeros `w_crossing`, `w_straightness`, `w_length_variance`.
- Chain step cap `50`.
- Crossing:
  - `max_pairs=500`
  - interval auto `10/5/3`
  - alpha schedule `3 -> 10`
  - warmup window `30%`.
- `w_cluster_sep = 0.5 * w_cluster`.
- `spacing_is_heavy` threshold `N > 1_000_000`.
- `subset_gpu` forced at `N >= 50_000_000`.
- `tiled_gpu` threshold `N >= 50_000_000`.
- Gradient clip `100.0`.
- Per-step overlap projection padding `2.0`.
- Per-step overlap projection iterations `5 / 3 / 2`.
- Final projection iterations `5 / 10 / 20 / 10 / 3`.
- Sampled-context formulas:
  - `n_active = min(max(N^0.75, min(N,256)), cap)`
  - `n_random = max(N^0.25, 4)`
  - `k_same >= 64`.
- Multilevel:
  - hierarchy `max_levels=20`
  - stop coarsening if reduction < 30%
  - prolong jitter std `5.0`
  - offload threshold `10_000_000`
  - huge-level thresholds `50M / 100M / 200M / 500M`
  - final sample caps `1M / 500k / 200k / 100k`.
- `fanout_amortize_*` exists in config but is not actually used.

## 8. Exact Random Seed Consumption

### General rule
- No call passes an explicit `torch.Generator`.
- All random draws use PyTorch’s default generator for the device of the op.
- Explicit seed calls in code:
  - `layout()`: `torch.manual_seed(config.seed)` and, if `device=="cuda"`, `torch.cuda.manual_seed(config.seed)` ([engine.py](/home/jtaylor/projects/dagua/dagua/layout/engine.py#L902))
  - `multilevel_layout()`: `torch.manual_seed(config.seed)` again ([multilevel.py](/home/jtaylor/projects/dagua/dagua/layout/multilevel.py#L2004))

### Direct layout path, in order
1. Seed calls above.
2. If init uses spectral ordering:
   - `_spectral_order()` does `torch.randn(N,2, device=...)` for `lobpcg` initial vectors ([init_placement.py](/home/jtaylor/projects/dagua/dagua/layout/init_placement.py#L318)).
3. Each optimizer step:
   - If active edge batching uses random remixes on that step:
     - `torch.randint(0, num_edges, (edge_batch,), out=perm_buf)`
     - `perm_buf` is CPU, so this draw is from the default CPU generator.
   - If sampled context refresh runs:
     1. `torch.randint(0, num_nodes, (n_active,), device=sampled_device)`
     2. `torch.rand(n_active, k_same, device=sampled_device)`
     3. `torch.rand(n_active, n_random, device=sampled_device)`
   - If there is **no shared sampled context**, randomness moves into loss functions in this order:
     - repulsion
     - overlap
     - crossing
     because that is the loss execution order.
   - If overlap projection falls back to grid mode, it may use `torch.randperm(...)` to subsample crowded cells/nodes.
4. Relaxation pass does **not** reseed; it continues from the current generator state.

### Multilevel path, in order
1. `layout()` seeds.
2. `multilevel_layout()` reseeds with `torch.manual_seed(config.seed)`.
3. Coarsest solve may consume spectral-init randomness if it is not warm-started and spectral init is selected.
4. Before every refine-level `_layout_inner()` call, prolongation injects jitter:
   - GPU prolong: `torch.randn(level.num_fine, 2, device=device).mul_(5.0)`
   - CPU prolong: `torch.randn(level.num_fine, 2).mul_(5.0)`
5. Refine-level `_layout_inner()` calls are warm-started (`init_pos=pos`), so they skip initialization RNG.
6. Final huge refine levels may force `edge_random_fraction=1.0`, making the edge-batch `torch.randint(...)` happen every step.

### Important branch-specific random consumers in losses
- Repulsion:
  - sampled global path: `randint`
  - scatter path: `rand`
  - RVS path: `randint`, then `rand`, then `rand`
  - shared sampled-context path: no additional random draws.
- Overlap:
  - scatter path: `rand`
  - active subset path: `randint`, then `rand`
  - shared sampled-context path: no additional random draws.
- Crossing:
  - fallback path may use `randperm` and then `randint`
  - layered path may use `randperm` and/or `rand`.

## 9. Multilevel V-cycle Structure

From [multilevel.py](/home/jtaylor/projects/dagua/dagua/layout/multilevel.py#L1956):

### Coarsen
1. Compute full-graph layering once.
2. Repeatedly `coarsen_once(...)` until:
   - `num_nodes <= multilevel_min_nodes`
   - or `max_levels == 20`
   - or node reduction is too weak (`current_n > 0.7 * previous_n`).
3. `coarsen_once(...)` does:
   - layer-local ordering
   - adjacency feature extraction (`min_neighbor`, `min_parent`, `min_child`, degrees, skip degree, mean span)
   - hub thresholding
   - pair/triple compatibility tests
   - variable-stride match scan
   - `fine_to_coarse`
   - coarse node sizes via `amax`
   - coarse edge remap + dedup.
4. For `N > 100M` or `E > 100M`, switch to streaming coarsening.

### Base solve
1. Solve only the coarsest graph with `_layout_inner(...)`.
2. Use `steps = multilevel_coarse_steps`, but cap:
   - `>50k coarse nodes -> 20`
   - `>10k coarse nodes -> 30`
3. Use `lr = config.lr * 2`.
4. Skip classification (`skip_classification=True`).

### Prolong
1. For each level from coarsest back to finest:
   - gather coarse positions by `fine_to_coarse`
   - add Gaussian jitter with std `5.0`
   - choose CPU/GPU prolong depending on VRAM.

### Refine
1. Determine per-level execution mode and optimizer fallback.
2. Build a level-specific config.
3. Apply huge-graph overrides:
   - tighter step caps
   - sample-cap override
   - projected crossing/projection cadences
   - possible `subset_gpu`
   - possible `SGD`.
4. Call `_layout_inner(...)` warm-started with the prolonged positions.
5. Free mappings and offloaded tensors aggressively.

This is a one-way V-cycle, not a recursive multigrid revisit.

## 10. All Loss Functions And Weights / Schedules

From [engine.py](/home/jtaylor/projects/dagua/dagua/layout/engine.py#L1673):

| Loss | Active when | Effective weight |
|---|---|---|
| `dag_ordering_loss` | `w_dag > 0` | `w_dag * (1 - 0.5*t)` |
| `edge_attraction_loss` | `w_attract > 0` | `w_attract` |
| `repulsion_loss` | `w_repel > 0` | `w_repel * (1 + 2*t)` |
| `overlap_avoidance_loss` | `w_overlap > 0` | `w_overlap * (1 + t)` |
| `cluster_compactness_loss` | `w_cluster > 0 and clusters` | `w_cluster` |
| `cluster_separation_loss` | `w_cluster > 0 and clusters` | `0.5 * w_cluster` |
| `cluster_containment_loss` | `w_cluster_contain > 0 and cluster_parents` | `w_cluster_contain` |
| `crossing_loss` | `w_crossing > 0 and E >= 4` | `w_crossing * min(t/0.3, 1)` and only every `crossing_interval`; actual evaluated term is also multiplied by `crossing_interval` |
| `edge_straightness_loss` | `w_straightness > 0` | `w_straightness * (1 + 0.5*t)` |
| `edge_length_variance_loss` | `w_length_variance > 0` | constant |
| `spacing_consistency_loss` | `w_spacing > 0 and layer_index` | constant |
| `fanout_distribution_loss` | `w_fanout > 0` | constant |
| `back_edge_compactness_loss` | `w_back_edge > 0` | constant |
| `position_pin_loss` | soft pins exist | `1.0 * pin_weights/mask inside loss` |
| `alignment_loss` | flex align groups exist | `1.0 * group.weight inside loss` |
| `flex_spacing_loss` | flex node_sep exists | `1.0 * flex weight inside loss` |

Notes:
- Crossing `alpha` is annealed `3.0 -> 10.0` over first 30% of steps.
- Repulsion amortization is implemented.
- Fanout amortization config exists but is not implemented in the engine.

## 11. Edge Batching And Sampled Node Context Mechanics

### Edge batching
From [engine.py](/home/jtaylor/projects/dagua/dagua/layout/engine.py#L2940) and [engine.py](/home/jtaylor/projects/dagua/dagua/layout/engine.py#L2139):
- `edge_batch_size > 0` forces exact batch size.
- Else:
  - small graphs may use all edges (`0`)
  - CPU uses tiered fixed sizes
  - CUDA auto-sizes from free VRAM.
- Random vs contiguous:
  - `edge_random_fraction = 0.2` means random every 5th step.
  - `1.0` means random every step.
  - `0.0` means always contiguous wrapped chunks.
- After batching, self-loops are removed.
- If not in per-loss-backward mode, `EdgeBatchContext` caches `src/tgt/dx/dy/dist_sq`.

### Sampled node context
From [engine.py](/home/jtaylor/projects/dagua/dagua/layout/engine.py#L502):
- Built only when:
  - not tiled compute
  - `layer_index` exists
  - `w_repel > 0` or `w_overlap > 0`.
- Sizes:
  - `n_active = min(max(N^0.75, min(N,256)), auto_cap)`
  - `n_random = max(N^0.25, 4)`
  - `k_same = max(64, min(rvs_nn_k, N-1))`
- Contents:
  - `active_idx`: random active nodes
  - `same_sampled`: uniform same-layer samples
  - `random_sampled`: uniform samples from adjacent layers `[layer-1, layer+1]`
- Refresh cadence:
  - every step normally
  - every `repel_amortize_interval` steps if amortized repulsion is active.
- Additional caps:
  - per-loss GPU budget cap
  - subset-GPU transfer cap.
- `SubsetGPUExecutor` caches a derived sampled access pattern across steps until the context object changes.

## 12. Overlap Projection And Hard Pin Projection Details

### Hard pins
From [constraints.py](/home/jtaylor/projects/dagua/dagua/layout/constraints.py#L1377):
- Runs immediately after `optimizer.step()`.
- For selected pinned nodes:
  - `projected = where(pin_mask, pin_targets, current)`
  - writes back in-place.
- Soft pins are not projected; they stay in `position_pin_loss`.

### Overlap projection
From [projection.py](/home/jtaylor/projects/dagua/dagua/layout/projection.py#L49):
- Dispatch:
  - `N <= 500`: exact all-pairs
  - layered CUDA sweep if layered + GPU + VRAM okay
  - layered streaming sweep if layered + `N > 100M`
  - layered CPU sweep otherwise
  - grid fallback if no layer info.
- Engine usage:
  - during loop: padding `2.0`, interval auto or configured, iterations `5/3/2`
  - final pass: padding `2.0`, iterations `5/10/20/10/3`.
- Exact projector:
  - detect pair overlaps in x/y
  - push along the smaller-overlap axis.
- Sweep projector:
  - stable sort by `(layer, x)`
  - check consecutive same-layer pairs
  - push by `0.25 * overlap`
  - second-neighbor pass with `0.125 * overlap` for `N <= 100k`.
- Grid projector:
  - spatial hash by cell
  - cap processed cells / nodes with random subsampling in crowded cells.

## Catalog Assessment Against `DESIGN.md`

### Cataloged ops that clearly match real engine code
- `ClassifyGraph`
- `LongestPathLayering`
- `BuildLayerIndex`
- `LayerAwareCoarsen`
- `StreamingCoarsen`
- `DirectMapping` (implemented inline with jitter)
- All listed engine loss ops
- `AdamStep`
- `OptimizerStep`
- `ClipGradNorm`
- `OverlapProjection`
- `HardPinProjection`
- `WeightAnnealing`
- `BuildEdgeBatchCtx`
- `RefreshSampledNodeCtx`
- `StallCount`
- `DirectionTransform`
- `DiskOffload`
- `DiskReload`
- `GarbageCollect`
- `VRAMGuard`
- `ProgressReport`
- `Conditional`
- `LossGroup`
- `MultilevelVCycle`
- `EarlyBreak`

### Cataloged but only partially matching
- `BuildEdgeBatchCtx`: real code also chooses random vs contiguous sampling and strips self-loops.
- `RefreshSampledNodeCtx`: real code also budget-caps active rows for GPU/subset-GPU.
- `WeightAnnealing`: real code includes weight schedules, crossing alpha schedule, and interval compensation.
- `DirectMapping`: real code is direct gather plus Gaussian jitter, not a standalone op call.

### Missing from the catalog but present in real engine behavior
- `AdaptiveSpacing`
- `ResolveExecutionMode`
- `ResolveMemoryStrategy`
- `ResolveFlexIds`
- `PrepareFlexData`
- `OverrideTreeWeights`
- `RelaxPass`
- `WarmStartInit`
- `HybridLossBridge`
- `EdgeStreamingDecision`
- `SampleCapOverride` for huge multilevel refinement

### Ops that should probably be split
- `WeightAnnealing`
  - split into `LossWeightSchedule` and `CrossingSchedule`
- `BuildEdgeBatchCtx`
  - split into `SelectEdgeBatch` and `BuildEdgeBatchCtx`
- `RefreshSampledNodeCtx`
  - split into `SizeSampledContext` and `SampleLayerNeighborhoods`

### Ops that should probably stay merged
- `OverlapProjection`
  - exact/sweep/grid/streaming are kernel variants, not distinct algorithmic steps
- `LossGroup`
  - current engine’s combined/per-loss/checkpoint/hybrid/subset/tiled execution modes belong here, not as separate ops

## Cross-Reference

| Real engine step | DESIGN.md op(s) | Match | Note |
|---|---|---|---|
| `graph._prepare_for_layout()` before solve | `DetectCycles`, `MakeAcyclic` | Partial | Happens outside `_layout_inner`, via graph wrapper |
| Graph family classification | `ClassifyGraph` | Yes | Direct path only; skipped in multilevel refine |
| Init layering in init placement | `LongestPathLayering` | Yes | Also reused in multilevel hierarchy build |
| Build per-layer index | `BuildLayerIndex` | Yes | Direct and refine paths |
| Init positions | `DeterministicInit`, `SpectralInit`, `BarycenterSweep`, `TransposeHeuristic` | Partial | Real code is a composite helper, not separate ops |
| Tree/chain weight override | example `OverrideTreeWeights()` | Missing from table | Real code has it |
| Edge batch selection + cached edge deltas | `BuildEdgeBatchCtx` | Partial | Real step also samples/streams/filters |
| Shared sampled active set | `RefreshSampledNodeCtx` | Partial | Real step also budget-caps |
| Annealed loss weights | `WeightAnnealing` | Yes | Real code also anneals crossing alpha |
| Loss evaluation/backward | `LossGroup` | Yes | Includes combined/per-loss/checkpoint/hybrid/subset/tiled modes |
| Optimizer update | `OptimizerStep`, `AdamStep`, `SGDNesterovStep` | Yes | Direct uses Adam; multilevel refine may downgrade |
| Gradient clipping | `ClipGradNorm` | Yes | Fixed max norm `100` |
| Hard pin snap | `HardPinProjection` | Yes | Exact post-step projection |
| Hard overlap push-apart | `OverlapProjection` | Yes | Method chosen by size/device/layer info |
| Early stop on stalled unweighted loss | `StallCount`, `EarlyBreak` | Yes | Exact implementation exists inline |
| Multilevel hierarchy build | `LayerAwareCoarsen`, `StreamingCoarsen`, `MultilevelVCycle` | Yes | Real implementation is concrete |
| Prolong coarse -> fine with jitter | `DirectMapping` | Partial | Inline, with hardcoded Gaussian jitter |
| Final direction transform | `DirectionTransform` | Yes | Applied after direct solve and after multilevel |
| Progress JSON / verbose progress | `ProgressReport` | Yes | Concrete implementation exists |
| Disk offload / reload / cleanup | `DiskOffload`, `DiskReload`, `GarbageCollect` | Yes | Multilevel only |
| Execution-mode / memory-policy selection | `VRAMGuard` | Partial | Present, but more granular than catalog |

No code changes were made; no tests were run.

Codex session ID: 019d4fcd-e011-74c3-9a05-7d25e920ae87
Resume in Codex: codex resume 019d4fcd-e011-74c3-9a05-7d25e920ae87
