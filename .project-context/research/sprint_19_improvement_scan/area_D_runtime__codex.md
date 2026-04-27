# Area D Runtime Audit

## TL;DR

- The hot path is not initialization. In warmed profiles, `gradient_core` dominates runtime, and inside it `loss_group` is the real sink. On layered graphs, `crossing_loss` dominates; on cyclic/tree-ish graphs, exact `repulsion_loss` and exact `overlap_avoidance_loss` dominate.
- The single best low-risk win is to cache the layered crossing decomposition in `crossing_loss`. Today `constraints._crossing_loss_layered()` rebuilds segment expansions and pair indices every step even though `edge_index` and `state.layers` are stable across the solve.
- The second best win is to stop recomputing pairwise geometry independently for exact repulsion and exact overlap inside per-loss backward. Both losses build separate `N x N` tensors from the same `pos` every iteration.

## Method

I followed the requested read-first path, then profiled the active CPU default (`engine.layout()` -> `dagua_native` pipeline). Because this workstation was already running several concurrent CPU-bound research jobs, I used warmed single-process measurements on directly constructed graphs rather than loading the entire 93-graph fixture set every time. That kept the measurements empirical while avoiding unrelated fixture-discovery overhead.

Profile set:

- Warmed `cProfile` on `hexagonal_lattice_42`, `small_world_100`, `org_chart_deep`, and `kitchen_sink_hybrid_net`, each with 1 warm-up solve then 3 profiled solves.
- Wall-clock baselines on 5 representative graphs.
- One isolated large-graph subroutine measurement on `dependency_500`: `init_positions()` only.
- One op-level timing pass on the composed pipeline for `hexagonal_lattice_42`, `small_world_100`, and `kitchen_sink_hybrid_net`.

Important limitation:

- A clean full-solve wall-clock measurement for `dependency_500` did not complete in a stable window on this loaded shared host. I did not fabricate that number. I did measure `init_positions()` on `dependency_500`, and it took only `0.042s`, which strongly suggests the large-graph runtime is also dominated by the optimizer/loss loop rather than initialization.

## Baseline Wall Clock

Measured on CPU with `CUDA_VISIBLE_DEVICES=""`, after one warm-up solve per graph.

| Graph | Nodes | Edges | Baseline sec/run | Projected sec/run |
| --- | ---: | ---: | ---: | ---: |
| `recurrent_feedback_cell` | 5 | 6 | 0.028 | 0.024 |
| `hexagonal_lattice_42` | 42 | 53 | 0.625 | 0.420 |
| `org_chart_deep` | 79 | 78 | 0.789 | 0.560 |
| `small_world_100` | 100 | 200 | 0.825 | 0.580 |
| `kitchen_sink_hybrid_net` | 19 | 25 | 1.063 | 0.780 |

How to read the projection:

- These are conservative post-fix estimates assuming only the high-confidence items below land: cached crossing decomposition, shared pairwise geometry for exact repulsion/overlap, size-gated combined backward on small CPU graphs, and small scalar/allocation cleanups.
- I did **not** stack every percentage additively; where fixes overlap, I used the smaller combined effect.

## Top-20 Hotspots Across Warmed Profiles

Aggregated cumulative time from the four warmed `cProfile` runs. Wrapper frames are included for completeness; actionable leaf hotspots are called out below.

| Rank | Function | Location | Cumtime (s) | Seen on |
| ---: | --- | --- | ---: | --- |
| 1 | `layout` | `dagua/layout/engine.py:883` | 9.330 | all 4 graphs |
| 2 | `layout_dagua_native_pipeline` | `dagua/layout/ops/pipelines/dagua_native.py:453` | 9.211 | all 4 |
| 3 | `Pipeline.apply` | `dagua/layout/ops/base.py:248` | 9.169 | all 4 |
| 4 | `Repeat.apply` | `dagua/layout/ops/base.py:400` | 8.043 | all 4 |
| 5 | `LossGroup.apply` | `dagua/layout/ops/base.py:658` | 7.662 | all 4 |
| 6 | `CrossingLoss.evaluate` | `dagua/layout/ops/loss_engine.py:760` | 2.714 | hex, kitchen, small_world |
| 7 | `crossing_loss` | `dagua/layout/constraints.py:916` | 2.713 | hex, kitchen, small_world |
| 8 | `_crossing_loss_layered` | `dagua/layout/constraints.py:975` | 2.712 | hex, kitchen, small_world |
| 9 | `RepulsionLoss.evaluate` | `dagua/layout/ops/loss_engine.py:578` | 2.692 | all 4 |
| 10 | `_exact_repulsion_loss` | `dagua/layout/ops/loss_engine.py:166` | 2.691 | all 4 |
| 11 | `OverlapAvoidanceLoss.evaluate` | `dagua/layout/ops/loss_engine.py:674` | 1.405 | all 4 |
| 12 | `_exact_overlap_loss` | `dagua/layout/ops/loss_engine.py:204` | 1.376 | all 4 |
| 13 | `BarycenterReorder.apply` | `dagua/layout/ops/barycenter.py:137` | 0.583 | all 4 |
| 14 | `torch.Tensor.backward` | `torch/_tensor.py:592` | 0.500 | all 4 |
| 15 | `torch.autograd.backward` | `torch/autograd/__init__.py:243` | 0.498 | all 4 |
| 16 | `_engine_run_backward` | `torch/autograd/graph.py:820` | 0.469 | all 4 |
| 17 | `NativeEngineInit.apply` | `dagua/layout/ops/init.py:2331` | 0.459 | all 4 |
| 18 | `init_positions` | `dagua/layout/init_placement.py:23` | 0.454 | all 4 |
| 19 | `_compute_barycenters_for_layer` | `dagua/layout/ops/barycenter.py:72` | 0.339 | hex, kitchen, org_chart |
| 20 | `OptimizerStep.apply` | `dagua/layout/ops/optimize.py:571` | 0.263 | all 4 |

## Per-Op Runtime

Measured by instrumenting the actual composed pipeline after a warm-up run.

### `hexagonal_lattice_42`

- `loss_group`: `0.834s`
- `barycenter_reorder`: `0.150s`
- `OptimizerStep`: `0.039s`

### `small_world_100`

- `loss_group`: `0.976s`
- `native_engine_init`: `0.030s`
- `ClipGradNorm`: `0.013s`

### `kitchen_sink_hybrid_net`

- `loss_group`: `0.804s`
- `OptimizerStep`: `0.066s`
- `barycenter_reorder`: `0.064s`
- `overlap_projection`: `0.039s`

Takeaway: once imports are warmed, the composable pipeline is overwhelmingly dominated by the gradient loop, and specifically by the active loss math inside `LossGroup`.

## Findings

### 1. Cache the layered crossing decomposition across iterations

- Severity: high
- Estimated wall-clock impact: `20-35%` on layered/mesh/mixed DAGs
- Location: `dagua/layout/constraints.py:975-1131`, `dagua/layout/ops/loss_engine.py:760-790`
- Evidence:
  - On `hexagonal_lattice_42`, warmed `cProfile` spent `2.463s / 3.346s` in `CrossingLoss.evaluate()` -> `crossing_loss()` -> `_crossing_loss_layered()`.
  - On `kitchen_sink_hybrid_net`, the same path took `2.381s / 3.134s`.
  - The hottest leaf inside the function was `torch.repeat_interleave`, called from the segment-expansion path at `constraints.py:1033`, `1078`, and `1088`.
  - The pair cap `max_pairs` is applied **after** segment expansion and per-layer grouping, so the code still pays to rebuild `seg_edge_idx`, `seg_layers`, `sort_idx`, and layer-local pair scaffolding every step even though `edge_index` and `state.layers` are unchanged.
- Proposed change:
  - Introduce a cached crossing scaffold in `state.extras`, keyed by `(id(problem.edge_index), id(state.layers), device)`.
  - Cache `actual_src_v`, `actual_tgt_v`, `span_v`, `seg_edge_idx`, `seg_layers`, `multi_offsets`, and the pair index plan.
  - Per step, recompute only the dynamic x-interpolation (`seg_x_from`, `seg_x_to`) and final sigmoid.
- Risk: low. This is a structural cache over static topology/layer data; it does not change the loss formula.
- Effort: `6-10h`

### 2. Share pairwise geometry between exact repulsion and exact overlap

- Severity: high
- Estimated wall-clock impact: `12-20%` on current CPU benchmark graphs
- Location: `dagua/layout/ops/base.py:680-710`, `dagua/layout/ops/loss_engine.py:166-235`, `dagua/layout/ops/loss_engine.py:578-741`
- Evidence:
  - On `small_world_100`, exact repulsion took `1.717s` and exact overlap took `0.984s` out of `3.456s`.
  - On `org_chart_deep`, exact repulsion took `1.978s` and exact overlap took `0.927s` out of `3.674s`.
  - Both paths rebuild full pairwise tensors from the same `pos`: `pos.unsqueeze(0) - pos.unsqueeze(1)` in `_exact_repulsion_loss()` and two more absolute-difference grids in `_exact_overlap_loss()`.
  - `LossGroup` runs in `per_loss` mode (`dagua_native.py:124-156`, `base.py:680-694`), so those exact pairwise intermediates are discarded and recomputed for each loss term.
- Proposed change:
  - Add a per-step `PairwiseGeometry` cache in `state.extras` for small graphs when both exact repulsion and exact overlap are active.
  - Populate `dx`, `dy`, `dist_sq`, `dx_abs`, `dy_abs`, and the non-diagonal mask once per optimizer step before the first exact pairwise loss runs.
  - Have both loss ops consume the cached tensors.
- Risk: low. The loss formulas stay identical.
- Effort: `8-12h`

### 3. Stop forcing `per_loss` backward on small CPU solves

- Severity: medium
- Estimated wall-clock impact: `5-10%`
- Location: `dagua/layout/ops/pipelines/dagua_native.py:124-156`, `dagua/layout/ops/base.py:680-710`
- Evidence:
  - Aggregated warmed profiles recorded `462` backward calls across just 12 solves.
  - The autograd stack (`torch.Tensor.backward`, `torch.autograd.backward`, `_engine_run_backward`) consumed about `0.5s` cumulative across the four warmed profiles.
  - `per_loss` exists for memory reduction, but on these CPU benchmark graphs (`N <= 100` in the concrete profiles) it pays extra Python and autograd setup cost while memory pressure is low.
- Proposed change:
  - Keep `per_loss` for larger graphs or when memory policy requests it.
  - Use `backward_mode="combined"` for small CPU solves, for example when `num_nodes <= 256` and `device == "cpu"`.
- Risk: low to medium. Combined backward increases peak memory, but the benchmark regime is small enough that this should be safe.
- Effort: `2-4h`

### 4. Lower the CPU exact-threshold for repulsion/overlap, or add a CPU approximation path

- Severity: medium
- Estimated wall-clock impact: `10-25%`, biggest on 100-500 node non-layered graphs
- Location: `dagua/layout/ops/loss_engine.py:279-314`, `dagua/layout/ops/loss_engine.py:614-741`
- Evidence:
  - The current thresholds (`RepulsionLossConfig.threshold = 2000`, `OverlapAvoidanceLossConfig.exact_threshold = 2000`) keep the exact `O(N^2)` path active for the entire 93-graph benchmark suite.
  - In practice, exact repulsion and exact overlap already dominate runtime at only `N=79-100`.
- Proposed change:
  - Add a CPU-specific switch point, for example `N >= 128` or `N >= 192`, that moves to the existing sampled path.
  - If quality moves, keep exact overlap for smaller layered graphs and sample only repulsion first.
- Risk: medium. This can change layout quality, so it needs an A/B benchmark gate.
- Effort: `8-16h`

### 5. Remove global RNG reseeding from every layout call on the CPU path

- Severity: low
- Estimated wall-clock impact: `2-4%` on small graphs, negligible on large solves
- Location: `dagua/layout/ops/pipelines/dagua_native.py:489-493`
- Evidence:
  - In the warmed `small_world_100` profile, `torch.manual_seed()` plus the `torch.cuda.random.manual_seed_all()` lazy path consumed about `0.07s` cumulative across 3 solves.
  - This benchmark is CPU-only (`CUDA_VISIBLE_DEVICES=""`). The current path reseeds global RNG state every layout call instead of using a local `torch.Generator`.
- Proposed change:
  - Replace global `torch.manual_seed()` with a per-layout `torch.Generator(device="cpu")`, attached to `RuntimeContext.generator`.
  - Seed CUDA only when the effective target device is actually CUDA.
- Risk: low, provided deterministic sampling continues to use the same generator.
- Effort: `<1h`

### 6. Trim `BarycenterReorder` allocation churn and no-op work

- Severity: medium
- Estimated wall-clock impact: `3-8%` on layered DAGs
- Location: `dagua/layout/ops/barycenter.py:72-112`, `dagua/layout/ops/barycenter.py:137-213`
- Evidence:
  - `BarycenterReorder.apply()` consumed `0.150s` on `hexagonal_lattice_42`, `0.064s` on `kitchen_sink_hybrid_net`, and `0.346s` across 3 `org_chart_deep` solves.
  - `_compute_barycenters_for_layer()` allocates `member_to_idx = torch.full((int(pos.shape[0]),), -1, ...)` for every layer, every pass.
  - The op clones and detaches the full position tensor (`pos.detach().clone()`) and then rebuilds `requires_grad` state even though it runs after `gradient_core`.
- Proposed change:
  - Reuse a workspace tensor for `member_to_idx`.
  - Precompute per-layer member lists and adjacent-layer filtered edge slices once.
  - If a pass produces identity order for all layers, skip the clone/writeback.
- Risk: low.
- Effort: `3-5h`

### 7. Vectorize `_spread_fanout_children` and remove Python scalar churn

- Severity: low
- Estimated wall-clock impact: `1-3%` on DAG-heavy graphs
- Location: `dagua/layout/init_placement.py:571-623`
- Evidence:
  - On isolated `dependency_500` initialization, `init_positions()` took `0.042s`; `_spread_fanout_children()` alone took `0.025s`, more than half of init time.
  - The function calls `.tolist()` on `edge_index` and repeatedly uses `.item()` and Python sorting for child widths and child x-order.
- Proposed change:
  - Keep the logic, but use tensor-based degree counting, grouped child lists, and tensor sorting on the children of each hub.
  - Only materialize Python lists for the final hub iteration if absolutely needed.
- Risk: low.
- Effort: `1-2h`

## Runtime Observations

- `init_positions()` is not the main problem on larger DAGs. On `dependency_500`, it was only `42ms`, with `_spread_fanout_children()` and `_barycenter_order()` the only notable sub-costs.
- The effective benchmark regime does not touch V-cycle code. For the ≤500-node suite, optimizing `dagua/layout/ops/vcycle.py` will not move the mean runtime.
- `max_pairs` in the crossing loss is misleading from a runtime perspective. The cap reduces sampled pair comparisons, but the expensive segment expansion and layer regrouping still happen first.

## Large-Graph Note

I do not think the missing full-wall-clock number for `dependency_500` changes the ranking above. The direct `init_positions()` measurement on that graph was tiny, and every warmed small/medium full-solve profile points to the same conclusion: the real runtime slope lives in repeated loss evaluation, not in graph preprocessing. In other words, even if `dependency_500` had completed cleanly here, I would expect the same action queue, just with larger absolute savings on the first four items.

## Recommended Action Queue

1. Cache the layered crossing scaffold in `crossing_loss`.
2. Share exact pairwise geometry between repulsion and overlap.
3. Switch to combined backward on small CPU graphs.
4. Rework `BarycenterReorder` workspaces and clone behavior.
5. Remove per-layout global reseeding on the CPU path.
6. Experiment with a lower CPU exact-threshold for repulsion first, then overlap if quality holds.
7. Clean up `_spread_fanout_children` only after the higher-impact items land.

## Bottom Line

The current runtime story is dominated by repeated work inside the loss loop, not by graph construction, initialization, or multilevel logic. Two specific patterns account for most of the waste:

1. rebuilding topology-derived scaffolding every optimizer step (`crossing_loss`);
2. recomputing the same pairwise geometry independently for separate exact losses (`repulsion` and `overlap`).

If those two are fixed first, the projected runtime reductions in the baseline table are realistic without changing layout quality. Everything else in this report is second-order.
