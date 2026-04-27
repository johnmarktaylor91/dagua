# Sprint 20 Agent D: Runtime Performance and GPU Acceleration

## TL;DR

- The runtime regression is concentrated in the large layered DAG path. In my single-thread CPU run, `dependency_500` took **17.35s** under the op timer and **22.77s** under `cProfile`; `random_dag_200` took **5.69s** under the op timer and **5.08s** under `cProfile`.
- Dummy-node expansion is the multiplier, not the direct cost. On `dependency_500`, `InsertDummyNodes` itself took only **0.0245s**, but it expanded the active state from **500** to **4,922** nodes. The downstream sprint-19 ordering/BK passes then ran over the expanded graph.
- The top sprint-19 CPU-locked hot path is `TransposeHeuristic`: **6.91s** in the op timer and **13.40s cumulative** in `cProfile` on `dependency_500`, with **78,448** calls to `_count_local_crossings()`. This is the first target.
- The differentiable core is still a major baseline cost. `loss_group` took **8.13s** on `dependency_500` and **5.29s** on `random_dag_200`. In `cProfile`, exact repulsion, exact overlap, crossing loss, projection, and backward dominate this block.
- `LayoutConfig(device="cuda")` does not skip the sprint-19 CPU hot paths. The native adapter initially moves tensors to CUDA, but dummy gating, dummy insertion, median sweep, transpose, BK, and component detection repeatedly call `.to(device="cpu")`, `.cpu()`, `.tolist()`, or `.numpy()`.
- Per-component decomposition is sequential. `layout_dagua_native_pipeline()` loops over `torch.unique(component_ids).tolist()` and calls `_run_native_problem()` once per component. There is no batched component solve or packed GPU launch.
- At a projected 10k original-node graph with mean edge span 4, physical dummy expansion to about 40k active nodes is not primarily a tensor-memory disaster by itself. The likely breakage is CPU time and Python-object churn in expanded ordering/BK. The worst peak-memory risk remains any accidental full-expanded all-pairs op; the current overlap/loss projectors mostly slice back to original nodes, which is good.

## Method

I read the sprint-20 context and the requested files:

- `.project-context/research/sprint_20_mega_sprint/CONTEXT.md`
- `dagua/layout/ops/pipelines/dagua_native.py`
- `dagua/layout/ops/ordering.py`
- `dagua/layout/ops/coordinate.py`
- `dagua/layout/ops/layering.py`
- `dagua/layout/ops/preprocess.py`

I first tried a default-thread op timer over all five requested graphs. The machine was heavily oversubscribed by other sprint research jobs, and the run timed out after completing only `org_chart_deep`. To get stable real measurements rather than a thread-pool fight, I reran the same workload with:

```bash
PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" python ...
torch.set_num_threads(1)
```

All numbers below are therefore **single-thread CPU** measurements unless stated otherwise. They are still real timings against the current code and the requested graphs, but they should not be treated as the final production wall time. The op ranking is robust because all ops in each graph were measured in the same process. CUDA was unavailable: `torch 2.8.0+cu128`, `torch.cuda.is_available() == False`.

The fixture loader itself took about **51-55s** under contention. I exclude that from per-graph totals. The timer monkeypatched `Pipeline.apply()` in-process and recorded inclusive wall time for each op. I then restored the original executor and ran `cProfile` on the two slowest graphs, `dependency_500` and `random_dag_200`, after a small warm-up layout to keep first-use optimizer setup out of the profiles.

## Per-op Time Breakdown

### Graph Totals

| graph | kind | original nodes | edges | timed total |
|---|---|---:|---:|---:|
| `org_chart_deep` | small/deep DAG | 79 | 78 | 0.908s |
| `random_dag_200` | medium DAG | 383 | 300 | 5.695s |
| `dependency_500` | large sparse DAG | 500 | 1,470 | 17.349s |
| `small_world_100` | cyclic/no hierarchy | 100 | 200 | 0.0268s |
| `disconnected_label_cycle_collage` | disconnected tiny | 7 | 6 | 0.303s |

`small_world_100` is fast because the current pipeline exits after only four gradient steps on the flat/cyclic path. That speed is not a quality endorsement; it matches the sprint-20 context that this graph is a structural loss.

### `org_chart_deep`

`org_chart_deep` is not the runtime problem. The first measured graph paid one-time optimizer setup:

| op | calls | time |
|---|---:|---:|
| `CreateOptimizer` | 1 | 0.866s |
| `gradient_core` | 1 | 0.0171s |
| `loss_group` | 4 | 0.0146s |
| `barycenter_reorder` | 1 | 0.00736s |
| `native_engine_init` | 1 | 0.00593s |
| `median_sweep` | 1 | 0.00425s |
| `transpose_heuristic` | 1 | 0.00260s |

BK was gated off or returned immediately: `brandes_koepf_horizontal_refine` took **0.00005s**.

### `random_dag_200`

The medium DAG did not use dummy expansion in this run. It is dominated by the differentiable loop:

| op | calls | time |
|---|---:|---:|
| `gradient_core` | 1 | 5.546s |
| `loss_group` | 200 | 5.292s |
| `periodic_overlap_projection` | 200 | 0.131s |
| `transpose_heuristic` | 1 | 0.107s |
| `OptimizerStep` | 200 | 0.053s |
| `ClipGradNorm` | 200 | 0.040s |
| `median_sweep` | 1 | 0.0150s |
| `barycenter_reorder` | 1 | 0.0144s |

`cProfile` confirms the same picture:

| function | cumulative time |
|---|---:|
| `base.py:400(Repeat.apply)` | 4.715s |
| `base.py:658(LossGroup.apply)` | 4.460s |
| `torch._C._EngineBase.run_backward` | 1.760s |
| `loss_engine.py:289(_exact_repulsion_loss)` | 1.259s |
| `loss_engine.py:327(_exact_overlap_loss)` | 0.632s |
| `ordering.py:1011(TransposeHeuristic.apply)` | 0.277s |
| `init_placement.py:488(_transpose_heuristic)` | 0.262s |
| `constraints.py:975(_crossing_loss_layered)` | 0.213s |

### `dependency_500`

This is the important profile. The graph starts with 500 nodes and 1,470 edges. Dummy insertion expands the active state to **4,922** nodes. The final output strips back to 500 nodes.

| op | calls | active size | time |
|---|---:|---:|---:|
| `gradient_core` | 1 | 4,922 | 9.379s |
| `loss_group` | 200 | 4,922 state, original-node loss view | 8.126s |
| `transpose_heuristic` | 1 | 4,922 | 6.912s |
| `periodic_overlap_projection` | 200 | original-node view | 1.100s |
| `brandes_koepf_horizontal_refine` | 1 | 4,922 | 0.565s |
| `median_sweep` | 1 | 4,922 | 0.294s |
| `activate_expanded_graph_state` | 1 | 4,922 | 0.093s |
| `barycenter_reorder` | 1 | 4,922 | 0.046s |
| `native_engine_init` | 1 | 500 | 0.027s |
| `insert_dummy_nodes` | 1 | metadata only | 0.0245s |

The key cProfile excerpt:

| function | cumulative time |
|---|---:|
| `engine.py:883(layout)` | 22.774s |
| `dagua_native.py:1180(layout_dagua_native_pipeline)` | 22.770s |
| `ordering.py:1011(TransposeHeuristic.apply)` | 13.537s |
| `init_placement.py:488(_transpose_heuristic)` | 13.399s |
| `init_placement.py:522(_count_local_crossings)` | 12.921s |
| `base.py:400(Repeat.apply)` | 7.980s |
| `base.py:658(LossGroup.apply)` | 6.858s |
| `torch._C._EngineBase.run_backward` | 2.694s |
| `loss_engine.py:289(_exact_repulsion_loss)` | 1.916s |
| `loss_engine.py:327(_exact_overlap_loss)` | 1.029s |
| `projection.py:192(_project_exact)` | 0.938s |
| `coordinate.py:1511(BrandesKoepfHorizontalRefine.apply)` | 0.628s |
| `ordering.py:912(MedianSweep.apply)` | 0.448s |
| `coordinate.py:58(_brandes_koepf_x_positions)` | 0.330s |

The transpose internals explain why dummy expansion hurts so much. `init_placement.py:488-570` runs adjacent swaps, and each proposed swap rebuilds dictionaries in `_count_local_crossings()`:

- line 496: pass loop
- line 498: layer loop
- line 503: adjacent-pair loop
- lines 535, 541, and 557: dictionary comprehensions rebuilt for local crossing counts
- lines 549-552 and 565-568: nested parent/child crossing checks

On the `dependency_500` cProfile, `_count_local_crossings()` was called **78,448** times, and the three dict comprehensions under it consumed about **12.23s** combined. This is the sharpest sprint-19-specific speed target.

### `small_world_100`

The cyclic graph total was **0.0268s**:

| op | calls | time |
|---|---:|---:|
| `gradient_core` | 1 | 0.0195s |
| `loss_group` | 4 | 0.0178s |
| `native_engine_init` | 1 | 0.00441s |
| `force_2d_init_if_flat` | 1 | 0.00012s |

Median, transpose, barycenter, and BK are effectively skipped or no-op for this graph.

### `disconnected_label_cycle_collage`

This graph decomposes into three small component solves:

| op | calls | time |
|---|---:|---:|
| `gradient_core` | 3 | 0.294s |
| `loss_group` | 104 | 0.251s |
| `OptimizerStep` | 104 | 0.0157s |
| `ClipGradNorm` | 104 | 0.0132s |
| `periodic_overlap_projection` | 104 | 0.00435s |
| `barycenter_reorder` | 3 | 0.00127s |
| `native_engine_init` | 3 | 0.00121s |

This confirms decomposition is not free. For tiny disconnected graphs, each component pays repeated pipeline, optimizer, and convergence overhead. The total is still small, but this is exactly the kind of work a batched component executor could collapse.

## Sprint-19 Ablation Timing

I ran a timing-only flag ablation on the two DAGs:

| graph/config | time |
|---|---:|
| `dependency_500` current | 17.9439s |
| `dependency_500` no dummy | 10.4786s |
| `dependency_500` no median/transpose | 11.2777s |
| `dependency_500` no BK | 18.6996s |
| `dependency_500` no dummy + no median/transpose + no BK | 9.0425s |
| `random_dag_200` current | 6.0843s |
| `random_dag_200` no dummy | 5.6622s |
| `random_dag_200` no median/transpose | 4.9435s |
| `random_dag_200` no BK | 5.5906s |
| `random_dag_200` no dummy + no median/transpose + no BK | 5.0096s |

The dependency graph result is the clearest runtime proof: dummy expansion costs about **7.47s** indirectly, and median/transpose costs about **6.67s**. BK was small in the op timer and noisy in the ablation; turning it off did not speed this single run up.

## CPU-locked Hot Paths

Ranked by measured wall time and source risk:

1. **`TransposeHeuristic` over expanded dummy graphs.** Code refs: `ordering.py:1011-1078`, `init_placement.py:488-570`. Measured: **6.91s op timer**, **13.40s cProfile cumulative** on `dependency_500`. This is pure Python control flow, list mutation, dict rebuilds, `.tolist()` edge traversal, and nested loops.

2. **Differentiable `loss_group` core.** Code refs: `base.py:658`, `loss_engine.py:289-324`, `loss_engine.py:327-390`, `loss_engine.py:695-786`, `loss_engine.py:805-870`. Measured: **8.13s** on `dependency_500`, **5.29s** on `random_dag_200`. This is not a sprint-19-only op, but dummy expansion keeps edge-centric losses and the optimizer on a larger active state. The original-node guard in `_visible_original_pos()` prevents the worst all-pairs blowup for node-box losses, but backward and exact paths remain meaningful CPU cost at this benchmark scale.

3. **`MedianSweep` and ordering materialization.** Code refs: `ordering.py:912-988`, especially the repeated `_node_order_map()` rebuilds at `ordering.py:959-975`. Measured: **0.294s op timer**, **0.448s cProfile cumulative** on `dependency_500`. Smaller than transpose, but it uses the same CPU list/dict model and grows with expanded nodes.

4. **`BrandesKoepfHorizontalRefine`.** Code refs: `coordinate.py:58-135`, `coordinate.py:760-823`, `coordinate.py:1511-1588`. Measured: **0.565s op timer**, **0.628s cProfile cumulative** on `dependency_500`. It is four-pass Python BK with CPU lists, conflict sets, recursive compaction, and CPU tensor materialization.

5. **Dummy insertion/activation.** Code refs: `layering.py:278-365`, `layering.py:725-757`, `layering.py:777-805`. Measured direct cost on `dependency_500`: **0.0245s** insert and **0.093s** activate. This is not the direct hot path at 500 nodes, but it creates the expanded-state multiplier that makes later ops slow.

6. **Component detection and component decomposition.** Code refs: `preprocess.py:485-530`, `preprocess.py:1338-1382`, `dagua_native.py:1272-1328`. Measured direct cost was small in these graphs, but it is CPU-only union-find plus a sequential solve loop.

## GPU Regression Audit

The native adapter does start on the requested device:

- `dagua_native.py:1206-1210` resolves `requested_device`, falling back to CPU only if CUDA is unavailable.
- `dagua_native.py:1216-1229` moves node sizes, edges, init positions, edge weights, and layer assignments to `target_device`.

But sprint-19 paths do not preserve device-generic execution:

- Native dummy gating forces CPU layer and edge tensors: `dagua_native.py:117-122`, `dagua_native.py:146`.
- Dummy insertion validates and expands on CPU: `layering.py:117`, `layering.py:320`, `layering.py:395`, `layering.py:436`, `layering.py:484`.
- Median/transpose validation and ordering derive CPU tensors: `ordering.py:63`, `ordering.py:93`, `ordering.py:118`, `ordering.py:269`, `ordering.py:985`, `ordering.py:1075`.
- Transpose adjacency construction traverses `edge_index.t().tolist()` through `_layered_neighbors_from_edges()` (`ordering.py:569`, called by `ordering.py:1048-1052`).
- BK converts all active layers, ordering, edges, x-ordering, and node sizes to CPU: `coordinate.py:55`, `coordinate.py:694`, `coordinate.py:721`, `coordinate.py:746`, `coordinate.py:909`, plus `edge_index.t().tolist()` at `coordinate.py:808`.
- Component detection forces CPU edges and Python union-find: `preprocess.py:506-530`.
- Component decomposition serializes IDs through `.tolist()` and solves each component one at a time: `dagua_native.py:1286-1316`.

So the answer to the question is: **no, `LayoutConfig(device="cuda")` would not skip these CPU hot paths.** It would run the gradient core and some tensor ops on CUDA, then repeatedly synchronize and copy structural data back to CPU for the sprint-19 discrete passes. That silently violates the original "same code on CPU or CUDA via `device=`" principle for the default pipeline.

There is already some CUDA-aware code in overlap projection. `projection.py:119-187` decides whether a CUDA sweep projection can fit in VRAM and falls back to CPU if not. That is the right pattern: explicit stage decision, bounded memory estimate, and fallback. The sprint-19 ops need the same treatment instead of unconditional CPU materialization.

## Batching Opportunities

Components are sequential today. The control flow is:

1. `DetectComponents().apply(...)` computes labels (`dagua_native.py:1272-1280`).
2. `_should_decompose_components(...)` decides whether to split.
3. `for component_id in torch.unique(component_ids, sorted=True).tolist():` iterates components (`dagua_native.py:1286`).
4. Each component calls `_extract_component_problem(...)`, prepares a child config, and runs `_run_native_problem(...)` independently (`dagua_native.py:1291-1316`).
5. `_tile_component_positions(...)` packs results afterward.

That means each non-singleton component pays Python function overhead, config preparation, optimizer creation, pipeline setup, and its own gradient loop. For `disconnected_label_cycle_collage`, three components triggered three gradient cores and 104 total loss-group calls for only seven original nodes.

A batched component kernel should look like a packed disconnected graph solve:

- Keep `pos` as one `[N_total, 2]` tensor.
- Keep `edge_index` as one concatenated edge tensor with component-local offsets already applied.
- Keep `component_ptr` and `edge_ptr` arrays, CSR-style, to delimit components.
- For gradient losses, operate on all components in one launch while masking cross-component pair interactions. Repulsion/overlap can be component-local for decomposed solves, or use a block-diagonal component mask.
- For ordering, store `layer_ptr`, `layer_nodes`, `node_to_layer`, and `ordering` for all components in one packed layer CSR. Median scores can be computed as segmented reductions. Transpose should not run as Python adjacent-swap loops; it should compute candidate delta crossings from precomputed neighbor position arrays and apply non-conflicting swaps in parallel by parity, like an odd-even transposition pass.
- For BK, run the four orientations over packed layers with per-component offsets. The algorithm has dependencies along layers and blocks, so it is less GPU-friendly than median scoring, but it can still avoid Python object construction by using flat arrays for `root`, `align`, `sink`, `shift`, `x`, and `pos_of`.

This is not just a CUDA idea. Even on CPU, a packed component solve would remove repeated optimizer/pipeline setup for tiny components and avoid serial Python loops over components.

## Memory Budget at 10k Original Nodes, Mean Edge Span 4

The prompt's projection is about **40k total active nodes**. Interpreting mean span 4 as three dummy vertices per long edge, a 10k-edge graph would add about 30k dummy nodes and 40k expanded edge segments. If the graph has more than 10k edges, scale the dummy count linearly by `sum(max(span - 1, 0))`.

Physical tensor memory for 40k active nodes is modest:

- `pos [40k, 2] float32`: about 0.3 MB.
- Adam moments for `pos`: two more tensors, about 0.6 MB.
- `layers` and `ordering [40k] int64`: about 0.6 MB total.
- Expanded `edge_index [2, 40k] int64`: about 0.6 MB.

The bad memory is not those tensors. The bad memory is accidental all-pairs work or Python object duplication:

- Exact repulsion uses `diff = pos.unsqueeze(0) - pos.unsqueeze(1)` and `dist_sq [N,N]` (`loss_engine.py:314-316`). At N=40k, `diff` alone would be about **12.8 GB**, `dist_sq` about **6.4 GB**, plus masks and autograd. The current loss path slices to original nodes for dummy-expanded state and switches to sampled repulsion above 2,000 original nodes (`loss_engine.py:741-746`, `loss_engine.py:752-786`), so this specific all-pairs disaster should not happen for a 10k original-node graph.
- Exact overlap has the same shape problem (`loss_engine.py:327-390`) and switches to sampled overlap above 2,000 original nodes (`loss_engine.py:832-870`).
- Overlap projection takes the original-node view when state is dummy-expanded (`project.py:195-224`, `project.py:374-389`). That avoids projecting all dummy nodes against each other, which is important.
- Transpose/BK memory is mostly Python lists, dicts, and sets over expanded nodes and edges. At 40k active nodes, the concern is CPU time and allocator churn more than raw tensor RAM. `edge_paths` in dummy expansion also keeps a Python list per original edge (`layering.py:316-343`), so it duplicates path metadata in high-overhead Python objects.

What breaks first at 10k/40k is likely **`TransposeHeuristic` wall time**, not VRAM. It was already 6.91s in the op timer and 13.40s under cProfile at only 4,922 active nodes. A naive extrapolation by expanded node count alone gives roughly 8x; by adjacent-swap/local-neighbor work it could be worse. This path needs either a hard size gate or a different implementation before 40k active nodes is acceptable.

## Top-3 Speedup Targets

1. **Replace or gate `TransposeHeuristic`.** Immediate conservative implementation: skip transpose when `expanded_num_nodes > 2_000` or when estimated local-swap work exceeds a budget, and keep median/BK. Better implementation: precompute neighbor ranks once per pass, compute crossing-delta for adjacent pairs without rebuilding dictionaries, and apply odd/even non-conflicting swaps. Expected payoff on `dependency_500`: up to **6-13s** depending on profiler overhead, with risk to crossing-rate quality.

2. **Avoid physical dummy nodes during optimization.** Use virtual dummy paths for ordering/routing, but keep gradient optimization on original nodes. The current loss/project code already tries to use original-node views; the expanded `state.pos` mainly exists for ordering and BK. A virtual representation would remove the 4,922-node active state and let discrete passes consume compact edge-span metadata. Expected payoff on `dependency_500`: the ablation says `insert_dummy_nodes=False` drops runtime from **17.94s** to **10.48s**, about **42%** faster. Quality risk is high because dummy expansion was added for long-edge routing.

3. **Batch components and packed layer ops.** Sequential decomposition is architecturally clean but runtime-hostile for many tiny components. Introduce a packed `ComponentBatch`/`LayerBatch` representation and run one optimizer over all component positions with segmented losses. Expected payoff is small on the five measured graphs except disconnected tiny cases, but it matters for many-component real graphs and restores the GPU story.

## Big-bet Proposals

**Batched segmented ordering kernel.** Build a new packed-layer primitive that stores `layer_ptr`, `layer_nodes`, `node_to_layer`, `ordering`, `parent_ptr`, `parents`, `child_ptr`, and `children`. Median sweep becomes segmented median or approximate median over neighbor ranks. Transpose becomes a bounded number of odd/even passes with vectorized delta scores. This is the biggest direct answer to the sprint-19 regression.

**Virtual dummy DAG mode.** Keep dummy nodes out of `state.pos` and store each long edge as `(src, dst, src_layer, dst_layer, span)`. Crossing reduction and routing can use virtual segment endpoints and layer slots without creating trainable dummy positions. If final edge rendering needs bend points, interpolate or solve them after original-node layout. This sacrifices some precision in dummy-dummy interactions, but current losses already ignore dummy nodes for node-box terms.

**CUDA graph or captured gradient core only after discrete ops are fixed.** Capturing the 200-step gradient loop could reduce Python overhead, but it will not solve CPU transpose, median, BK, or component serialization. Do this later, after the default pipeline no longer bounces structural state to CPU.

**Topology-dispatch performance budgets.** The default pipeline should carry an execution budget: if dummy expansion grows active nodes by more than, say, 4x, run median only, skip transpose, and use BK only if layer count and width predict benefit. This is an engineering guardrail, not a research algorithm.

## Risks and Regression Analysis

The protected sprint-20 wins include `org_chart_deep`, `random_dag_200`, and `dependency_500`. Blindly disabling sprint-19 features risks losing crossing and long-edge quality on exactly those DAGs. The ablation numbers are runtime-only; they do not prove it is safe to turn off dummy nodes or transpose by default.

The safest rollout is:

1. Add measurement-only counters first: original nodes, expanded nodes, expanded edges, layer count, max layer width, estimated transpose work, and whether BK applied.
2. Gate transpose by estimated work, not just graph name. Use protected wins as required regression tests.
3. Optimize transpose internals before deleting behavior. The cProfile shows dictionary rebuilds, not the concept of adjacent swaps, are the immediate problem.
4. Keep dummy expansion for DAGs where long-edge mass is modest and quality wins are proven. For extreme expansion, switch to virtual dummies or median-only.

## Implementation Order

1. Instrument expansion ratio and per-op timings in a reusable debug trace. This is low risk and makes future regressions visible.
2. Replace `_count_local_crossings()` dictionary rebuilds with cached layer position arrays and neighbor rank arrays. This keeps the algorithm but attacks the measured 12.9s cProfile hotspot.
3. Add a conservative transpose budget gate for expanded graphs until the vectorized version lands.
4. Move median sweep and BK inputs to flat tensor/CSR structures that can stay on CUDA.
5. Build packed component execution after the single-component hot paths are fixed.
6. Consider CUDA graph capture for `gradient_core` once the pipeline no longer syncs to CPU between structural passes.

## Knowledge

- `dependency_500` expands from **500** to **4,922** active nodes in the current native pipeline.
- `InsertDummyNodes` is cheap directly; the runtime regression is the expanded graph hitting Python ordering/BK.
- `TransposeHeuristic` is the top sprint-19 hot path by wall time and by cProfile.
- The current dummy-expanded loss/project code often slices back to original nodes, which avoids the worst N-expanded all-pairs memory blowup.
- CUDA support is partial: the differentiable tensor core can use the requested device, but sprint-19 discrete passes force CPU structural data.
