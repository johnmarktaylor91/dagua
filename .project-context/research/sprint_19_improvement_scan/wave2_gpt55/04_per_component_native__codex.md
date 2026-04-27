# Per-Component Layout Decomposition For `dagua_native`

## Section 1 -- Design

The right place to add disconnected-component decomposition is not inside the pure op chain in `dagua/layout/ops/pipelines/dagua_native.py`, and not at the public `dagua.layout()` graph-object entry point in `dagua/layout/engine.py`. The lowest-friction insertion point is the tensor adapter `layout_dagua_native_pipeline()` in `dagua/layout/ops/pipelines/dagua_native.py`. That function already does the heavy lifting that matters here: it resolves config, normalizes node sizes, prepares `LayoutProblem` and `SolveState`, resolves flex constraints, and then dispatches into the default native pipeline. Component decomposition is orchestration over whole problems, not one more state mutation within a single problem, so it belongs at that adapter boundary.

### Component detection

Use the existing `DetectComponents` op, backed by the shared union-find implementation in `dagua/layout/ops/preprocess.py`, not `networkx`.

Reasons:

1. The code already exists and is directly aligned with Dagua’s weak-component semantics.
2. It operates on `torch.Tensor` edge data and matches the current pipeline state model.
3. It avoids a new dependency edge and avoids duplicating the component-label algorithm in a second place.

Concretely, the adapter should instantiate a temporary `SolveState`, run `DetectComponents().apply(problem, state, ctx)`, and read `state.component_ids`. That keeps the decomposition logic consistent with any future ops that also depend on `component_ids`.

### Decomposition gate

The decomposition branch should be taken only when all of the following are true:

1. `config.decompose_components` is `True`.
2. `problem.num_nodes >= 2`.
3. `problem.structure.num_components > 1`, or `component_ids.max() > 0` if structure is absent.
4. `problem.clusters` is empty or `None`.
5. `problem.flex` has no pins.

The pin gate is mandatory per the task. I would implement it as: skip when `problem.flex` is present and `problem.flex.pin_indices` is not `None` and has length greater than zero. That is stricter and safer than trying to infer whether pins span components. For this first patch, “any pins means no decomposition” is the correct conservative interpretation.

For alignment groups, I would not add a second global gate. Instead, when a group is fully contained within one component, keep it in that child problem; when a group spans multiple components, drop it from the child problems because the semantics no longer survive tiling. That is conservative, localized, and does not widen the skip condition beyond the explicit pin requirement.

### Recursion strategy

Run the same native pipeline recursively per component, but do not recurse through public `layout()` in `engine.py`.

The clean structure is:

1. Factor the current “tree fast-path or `build_dagua_pipeline(...).apply(...)`” body inside `layout_dagua_native_pipeline()` into a private helper such as `_run_native_problem(problem, state, ctx, prepared_config) -> torch.Tensor`.
2. When decomposition is enabled, build one child `LayoutProblem` per component and call that helper on each child problem.
3. Tile the returned child positions back into a full `[N, 2]` parent tensor.

This is still “same pipeline recursion” in the sense that every component gets the exact same `dagua_native` logic: tree fast-path, `NativeEngineInit`, `Force2DInitIfFlat`, gradient core, barycenter polish, overlap projection, and per-component `AspectRatioFit`. What it avoids is repeated graph-object prep, direction transforms, layout caching, and flex re-resolution through `engine.layout()`. It also lets us pass filtered `FlexConstraints` directly instead of reconstituting `LayoutFlex`.

I would explicitly avoid a cut-down sub-pipeline. The whole point of this change is to make disconnected graphs benefit from the same future improvements as connected ones. If Wave 2 lands Brandes-Koepf, dummy nodes, or transpose, the component path should inherit that automatically.

### Subproblem construction

Each component subproblem should be an induced subgraph with local node IDs:

1. Extract the parent node index list for one component.
2. Build a `global_to_local` tensor of shape `[N_parent]` initialized to `-1`.
3. Filter edges whose endpoints are in that component.
4. Relabel `sub_edge_index = global_to_local[parent.edge_index[:, mask]]`.
5. Slice `node_sizes`, `edge_weights`, and optional initial positions by component membership.
6. Rebuild `problem.structure` for the child by calling the normal classifier on `sub_edge_index`.

If an initial position tensor is present, normalize it before the child solve by subtracting the component centroid or bbox minimum. Passing absolute parent-space offsets into the child solve would contaminate bbox measurements and aspect-ratio logic.

### Tiling

Use bbox-preserving tiling after all child solves complete. The algorithm I would ship is:

1. For each child position tensor, shift it so its bbox minimum corner is at `(0, 0)`.
2. Compute `width_i`, `height_i`, and padded dimensions:
   `tile_w_i = max(width_i, node_sep) + gap`
   `tile_h_i = max(height_i, node_sep) + gap`
3. Sort components by descending node count, then descending bbox area.
4. Evaluate candidate column counts `cols in [1, num_components]`.
5. For each `cols`, place components row-major, tracking per-row max height and row width.
6. Score each candidate by closeness to a packing target aspect plus a small area penalty.
7. Pick the best-scoring layout, then place the original unpadded component positions at those offsets.
8. Re-center the final full position tensor around the origin.

I would not hard-code a single-row tiler. That works for `disconnected_label_cycle_collage`, but it is the wrong default for graphs like `multi_component_80` where seven components would become an excessively wide strip.

I would also not target the current global `AspectRatioFit` default of `0.25` during tiling. That target is intentionally very tall, and using it at the packing stage would collapse many disconnected graphs into a single vertical stack again. The better approach is:

1. pack into a near-square outer bbox first,
2. then apply one final outer `AspectRatioFit`.

That keeps the tiler from reproducing the “everything in one column” pathology while still letting the global aesthetic policy act after the components are composed.

### Orientation

Each component should be laid out independently in its own coordinate system. After the child solve:

1. compute the child bbox,
2. shift the child to local origin using bbox min corner,
3. tile by translating the whole child geometry.

Do not rotate or mirror components during tiling. The child solve has already chosen a local orientation consistent with direction and graph family. Tiling should be pure translation only.

### Spacing

Use:

`gap = resolved_node_sep * 2.0`

That matches the task’s requested `node_sep * tile_pad_factor` policy and is easy to reason about. `2.0` is a sensible first default because it is:

1. large enough to visually separate tiny components,
2. small enough that a singleton component next to a 499-node component does not dominate the outer bbox,
3. consistent with existing component-gap intuition in the tree coordinate ops, which also use a 2x sibling-spacing default.

I would keep this as a private constant for the first patch, not a public config field. The only public rollback knob needed here is `decompose_components`.

## Section 2 -- Exact code patches

### a) `build_dagua_pipeline` in `dagua_native.py`

I would leave `build_dagua_pipeline()` itself structurally unchanged. It should stay a pure composed-op definition. The decomposition branch should sit above it in `layout_dagua_native_pipeline()`.

That means `build_dagua_pipeline()` keeps returning the same pipeline:

- `NativeEngineInit`
- `Force2DInitIfFlat`
- stress prep if enabled
- annealing
- optimizer
- gradient core
- `BarycenterReorder`
- `OverlapProjection`
- `AspectRatioFit`
- `ClusterGridArrange`

The key architectural decision is: do not teach the pipeline builder how to construct subproblems.

### b) Adapter-level wrapper in `dagua_native.py`

This is the core patch.

Add one public config field in `dagua/config.py`:

```python
decompose_components: bool = True
```

Then refactor `dagua/layout/ops/pipelines/dagua_native.py` roughly as follows:

```python
def _run_native_problem(
    problem: LayoutProblem,
    state: SolveState,
    ctx: RuntimeContext,
    config: LayoutConfig,
) -> torch.Tensor:
    """Run the native pipeline for one already-prepared problem."""
    structure = problem.structure
    if (
        structure is not None
        and getattr(structure, "family", None) == GraphFamily.TREE
        and getattr(config, "use_tree_fast_path", True)
        and problem.num_nodes > 0
    ):
        rt_state = ReingoldTilfordTree(ReingoldTilfordTreeConfig()).apply(problem, state, ctx)
        if rt_state.pos is not None:
            return rt_state.pos.detach()

    final_state = build_dagua_pipeline(config).apply(problem, state, ctx)
    if final_state.pos is None:
        raise RuntimeError("dagua_native pipeline did not produce final positions.")
    return final_state.pos.detach()
```

Then add the decomposition helpers:

```python
_COMPONENT_TILE_PAD_FACTOR = 2.0


def _has_pins(flex: Optional[FlexConstraints]) -> bool:
    """Return whether the prepared flex constraints contain any pins."""
    if flex is None or flex.pin_indices is None:
        return False
    return int(flex.pin_indices.numel()) > 0


def _should_decompose_components(
    problem: LayoutProblem,
    config: LayoutConfig,
) -> bool:
    """Return whether this problem should be split into weak components."""
    if not getattr(config, "decompose_components", True):
        return False
    if problem.num_nodes < 2:
        return False
    if problem.clusters:
        return False
    if _has_pins(problem.flex):
        return False
    structure = problem.structure
    if structure is not None and getattr(structure, "num_components", 1) <= 1:
        return False
    return True
```

The subproblem extractor:

```python
def _extract_component_problem(
    parent_problem: LayoutProblem,
    parent_state: SolveState,
    component_nodes: torch.Tensor,
    component_ids: torch.Tensor,
) -> tuple[LayoutProblem, SolveState, torch.Tensor]:
    """Build one relabeled child problem and return its parent index map."""
    device = parent_problem.edge_index.device
    parent_indices = component_nodes.to(device=device, dtype=torch.long)
    local_index = torch.full((parent_problem.num_nodes,), -1, dtype=torch.long, device=device)
    local_index[parent_indices] = torch.arange(parent_indices.shape[0], device=device)

    edge_index = parent_problem.edge_index
    if edge_index.numel() == 0:
        edge_mask = torch.zeros((0,), dtype=torch.bool, device=device)
    else:
        edge_mask = component_ids[edge_index[0]] == component_ids[parent_indices[0]]
    sub_edge_index = local_index[edge_index[:, edge_mask]]
    sub_node_sizes = (
        None if parent_problem.node_sizes is None else parent_problem.node_sizes[parent_indices]
    )
    sub_edge_weights = (
        None if parent_problem.edge_weights is None else parent_problem.edge_weights[edge_mask]
    )
    sub_init_pos = None
    if parent_state.pos is not None:
        sub_init_pos = parent_state.pos[parent_indices].clone()
        sub_init_pos -= sub_init_pos.mean(dim=0, keepdim=True)

    sub_problem = LayoutProblem(
        edge_index=sub_edge_index,
        num_nodes=int(parent_indices.shape[0]),
        node_sizes=sub_node_sizes,
        direction=parent_problem.direction,
        structure=classify_graph(sub_edge_index, int(parent_indices.shape[0])),
        flex=_subset_flex(parent_problem.flex, parent_indices, local_index),
        edge_weights=sub_edge_weights,
        seed=parent_problem.seed,
    )
    return sub_problem, SolveState(pos=sub_init_pos), parent_indices
```

And the tiler:

```python
def _tile_component_positions(
    component_results: list[tuple[torch.Tensor, torch.Tensor]],
    node_sep: float,
) -> torch.Tensor:
    """Tile independently solved component layouts into one parent tensor."""
    if not component_results:
        return torch.zeros((0, 2), dtype=torch.float32)

    gap = max(float(node_sep) * _COMPONENT_TILE_PAD_FACTOR, 1.0)
    packed = []
    for parent_indices, pos in component_results:
        x_min = float(pos[:, 0].min().item())
        x_max = float(pos[:, 0].max().item())
        y_min = float(pos[:, 1].min().item())
        y_max = float(pos[:, 1].max().item())
        local = pos.clone()
        local[:, 0] -= x_min
        local[:, 1] -= y_min
        width = max(x_max - x_min, node_sep)
        height = max(y_max - y_min, node_sep)
        packed.append((parent_indices, local, width, height))

    packed.sort(key=lambda item: (-int(item[0].numel()), -(item[2] * item[3])))
    cols = _choose_component_grid(packed, gap=gap)
    offsets = _row_major_offsets(packed, cols=cols, gap=gap)

    total_nodes = sum(int(idx.numel()) for idx, _, _, _ in packed)
    out = torch.zeros((total_nodes, 2), dtype=packed[0][1].dtype, device=packed[0][1].device)
    for (parent_indices, local, _, _), (ox, oy) in zip(packed, offsets):
        out[parent_indices, 0] = local[:, 0] + ox
        out[parent_indices, 1] = local[:, 1] + oy

    out -= out.mean(dim=0, keepdim=True)
    return out
```

Finally, branch in `layout_dagua_native_pipeline()` after `problem/state/ctx` creation and before the normal single-problem execution:

```python
component_state = DetectComponents().apply(problem, SolveState(), ctx)
component_ids = component_state.component_ids
if component_ids is not None and _should_decompose_components(problem, prepared_config):
    unique_components = torch.unique(component_ids, sorted=True)
    component_results = []
    for component_id in unique_components.tolist():
        component_nodes = torch.nonzero(component_ids == component_id, as_tuple=False).squeeze(1)
        child_problem, child_state, parent_indices = _extract_component_problem(
            problem,
            state,
            component_nodes,
            component_ids,
        )
        child_pos = _run_native_problem(child_problem, child_state, ctx, prepared_config)
        component_results.append((parent_indices, child_pos))

    tiled = _tile_component_positions(
        component_results,
        node_sep=float(getattr(prepared_config, "_dagua_native_node_sep", prepared_config.node_sep)),
    )
    outer_state = SolveState(pos=tiled)
    outer_state = AspectRatioFit(AspectRatioFitConfig()).apply(problem, outer_state, ctx)
    if outer_state.pos is None:
        raise RuntimeError("component tiling did not produce positions")
    return outer_state.pos.detach()

return _run_native_problem(problem, state, ctx, prepared_config)
```

### c) New op if needed: `TileComponents`

I do not recommend a new registered op for the first patch.

Why:

1. Tiling is only meaningful after recursive per-component solves have returned.
2. Registered ops operate on one `LayoutProblem` and one `SolveState`; component extraction creates many child problems.
3. If implemented as an op, recursion would need to tunnel child problems through `state.extras`, which is exactly the kind of orchestration leakage the op boundary is supposed to avoid.

The right abstraction is a private helper in `dagua_native.py`, not a public op in `ops/postprocess.py`.

### d) Integration with existing `DetectComponents`

Use `DetectComponents` once at the parent problem level. `state.component_ids` remains the canonical representation. No extra `extras["components"]` field is needed.

That keeps the patch aligned with the existing state schema:

- detection writes `state.component_ids`,
- orchestration reads it immediately,
- child problems are then fully separate.

## Section 3 -- Interaction with other wave-2 patches

Wave-2 patches #1 through #3 become strictly cleaner under this design because they run inside the child solve with no special-case code.

### #1 Brandes-Koepf

If Brandes-Koepf is added later in the default pipeline, it will run per component automatically because each component calls the same `_run_native_problem(...)` helper. That is exactly what we want. BK should never try to coordinate x-compaction across disconnected components.

### #2 Dummy nodes

Dummy-node insertion is also naturally per component. Long-edge splitting should happen inside each connected subproblem, not across the tiled composition layer. The decomposition wrapper therefore reduces the state surface for dummy nodes rather than expanding it.

### #3 Median / transpose

Same story. Crossing-reduction heuristics are meaningful only inside a component’s layered subgraph. The wrapper means these heuristics run on smaller, cleaner problems and do not waste effort comparing neighbor barycenters across unrelated components.

### #5 Aspect ratio

This patch should deliberately apply aspect ratio twice at different scales:

1. each child solve gets the normal per-component `AspectRatioFit`,
2. the tiled parent layout gets one outer `AspectRatioFit`.

That yields the right separation of concerns:

- child AR shapes the internal geometry of a component,
- outer AR shapes the composition of the page.

If Wave 2 patch #5 makes the target aspect topology-aware, the child solves inherit that immediately because they use the same prepared config and classifier. The outer pass should still run, but it should use the same config so the entire page composition stays consistent with the updated policy.

## Section 4 -- Regression safety

### Single-component graphs

This must be a strict no-op. The branch should be skipped entirely when `num_components <= 1`. That preserves current winners like `random_dag_200` bit-for-bit aside from negligible control-flow overhead.

### Giant-component graphs with small satellites

`dependency_500` is the important safety case: one large component plus one singleton. The decomposition design above does not change the geometry of the 499-node solve at all; it simply solves it in isolation, solves the singleton trivially, and tiles the results with bounded padding.

The singleton should not inflate the outer bbox materially if the tiler uses:

`max(component_extent, node_sep)`

for minimal tile size rather than a huge default floor. That keeps the one-node component visible without letting it dominate the packed row width.

### Clustered graphs

Skip decomposition entirely when `problem.clusters` is populated.

That is the safest first cut for two reasons:

1. cluster membership already introduces a higher-level partitioning pressure,
2. tiling components above clusters would create ambiguous semantics for cluster separation and containment.

This is also consistent with the current `AspectRatioFit` logic, which already opts out on cluster-aware graphs for similar “don’t blindly reuse global geometry policy” reasons.

### Empty graph, one-node graph, all singletons

The wrapper should preserve current early exits:

- empty graph: return `torch.zeros((0, 2))`,
- one node: return `torch.zeros((1, 2))`.

For all-singleton graphs, decomposition is still correct. Each child solve is trivial, and the tiler becomes the whole layout algorithm. That is acceptable and arguably desirable: an edgeless graph is exactly a component-packing problem.

## Section 5 -- Tests

I would add or update the following tests.

### Unit: `DetectComponents`

File: `tests/test_ops_preprocess.py`

Expand the existing component test to a 7-node, 3-component case matching the requested shape:

- edges `(0, 1), (1, 2)` for component A,
- edge `(3, 4)` for component B,
- edge `(5, 6)` or a two-node disconnected pair for component C.

Expected labels should be stable and monotonic, for example `[0, 0, 0, 1, 1, 2, 2]`.

### Integration: `disconnected_label_cycle_collage`

File: `tests/test_layout/test_engine.py` or a new native-pipeline regression file.

Test recipe:

1. load the named graph from `dagua.eval.graphs`,
2. run `layout()` or `layout_dagua_native_pipeline()` with default config,
3. compute metrics using the existing benchmark metric helpers,
4. assert composite score exceeds a fixed regression threshold above the current 62.08 baseline.

I would set the first threshold around `>= 70.0`, not `>= 75.19`, so the test is robust to small host-level drift while still proving the decomposition landed meaningful value.

### Regression: `random_dag_200`

Use a two-config comparison:

1. `LayoutConfig(decompose_components=True)`,
2. `LayoutConfig(decompose_components=False)`.

Because the graph has one component, the two outputs should match within a very small tolerance, and the composite score should be unchanged within floating-point noise. That is a direct guard against accidentally routing all graphs through the tiler.

### Edge cases

Add three edge-case tests:

1. empty graph returns shape `[0, 2]`,
2. one-node graph returns exactly one position at origin,
3. all-singletons graph returns non-overlapping tiled positions with more than one unique coordinate.

### Test commands

For the eventual implementation, the quality-gate sequence should be:

1. `ruff check . --fix`
2. `mypy --follow-imports=silent dagua/cli.py`
3. `pytest tests/test_layout/ -x --tb=short -q`
4. final once: `pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"`

## Section 6 -- Expected impact

These are estimates, not measured post-patch numbers.

### `disconnected_label_cycle_collage`

Current: `62.08`

Expected after decomposition: `73` to `76`, with `~75` as the target band.

Why this graph should move hard:

1. its three components stop sharing one coordinate system,
2. the 3-cycle no longer collapses onto one global y level,
3. edge straightness should recover toward ELK’s behavior immediately.

This is the headline beneficiary of the patch, and matching or nearly matching ELK’s `75.19` is realistic.

### `disconnected_encoder_residual`

Current: `85.59`

Expected after decomposition: `86` to `88`.

This graph is already near-tied, so the gain here is modest. The main effect is reduced cross-component repulsion noise, not a fundamental change in internal ordering quality.

### `multi_component_80`

Current: `74.78`

Expected after decomposition: `76` to `78`.

Seven components are enough that independent solves plus packing should improve bbox usage, overlap pressure, and possibly some edge-length variance. I would not expect a dramatic jump because the graph is already near-tied, but a low-single-digit gain is plausible.

### `dependency_500`

Current: `51.96`

Expected after decomposition: `53` to `56`.

This should help, but it is not the full fix for `dependency_500`. The singleton component is likely contributing to wasted bbox and pressure, so removing it from the main solve should lift quality slightly. The large gap here is still dominated by other issues already identified in Wave 2: dummy nodes, coordinate assignment, and overlap behavior on deep DAGs.

### Overall suite impact

I would model the near-term mean composite lift as roughly `+0.2` to `+0.4` on the 93-graph benchmark.

Reasoning:

1. the patch directly targets a small disconnected bucket, not the full suite,
2. one graph in that bucket (`disconnected_label_cycle_collage`) is a very large loser and should recover materially,
3. several other disconnected or partially disconnected graphs should get small lifts,
4. single-component graphs should remain unchanged.

The median graph should not move. This is a tail-improvement patch.

## Section 7 -- Rollback

Add:

```python
decompose_components: bool = True
```

to `LayoutConfig`.

That is the rollback switch.

Why this is sufficient:

1. the branch is entirely above the existing native pipeline,
2. disabling the flag restores the old one-problem execution path,
3. no downstream op needs to know whether decomposition happened.

I would not add more public knobs in the first patch. If rollout shows packing-quality sensitivity, the pad factor and packing heuristic can remain private constants until there is evidence they need to be user-tunable.

## Recommendation

Implement this in `dagua/layout/ops/pipelines/dagua_native.py` as an adapter-level recursive wrapper, not as an in-pipeline op and not as a public `engine.layout()` rewrite. Reuse `DetectComponents`, gate on `no pins` and `no clusters`, recurse through the same native pipeline helper per component, then tile with bbox-preserving row-major packing and one final outer `AspectRatioFit`. That gives the exact behavior Wave 1 identified, contains regression risk, and composes cleanly with the other Wave-2 layered-layout fixes.
