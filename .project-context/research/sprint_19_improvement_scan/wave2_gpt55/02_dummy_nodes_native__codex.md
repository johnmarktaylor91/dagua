# Dummy Nodes In `dagua_native`

## Section 1 -- Design

### Gate

Insert dummy nodes in the default native pipeline only when all of the following are true:

1. `LayoutConfig.enable_native_dummy_nodes` is `True`.
2. The graph is acyclic: `structure.is_acyclic`.
3. The graph is connected: `structure.num_components == 1`.
4. The graph is not flat: `structure.num_layers > 1`.
5. At least one edge spans two or more layers under the resolved longest-path layering:
   `max(layer[target] - layer[source]) >= 2`.

That is the conservative gate the current codebase can support without changing the public API. It matches the Sprint 19 finding: the dummy-node expansion is only valuable on DAG-like graphs with real long edges. Cyclic, disconnected, and effectively single-layer graphs should skip it entirely.

I verified the extraction assumption: `dagua/layout/ops/layering.py:317-370` initializes `next_dummy_index = num_original_nodes`, appends every dummy after the original node block, and concatenates dummy sizes after the original `node_sizes`. So `pos[:num_original_nodes]` is a valid restore path.

### What “dummy-aware” means downstream

For native, “dummy-aware” should not mean “treat dummy nodes as real boxes everywhere.” It should mean:

- Ordering ops operate on the expanded graph. That includes `BarycenterReorder` now, and `MedianSweep` / `TransposeHeuristic` / `BrandesKopf4Pass` once Wave 2 patches #1 and #3 land.
- Edge-centric gradient losses operate on expanded edges. Those are the losses that need the new degrees of freedom: `DagOrderingLoss`, `EdgeAttractionLoss`, `CrossingLoss`, `EdgeStraightnessLoss`, and `EdgeLengthVarianceLoss`.
- Box-centric constraints do **not** operate on dummies. Repulsion, overlap loss, overlap projection, and spacing consistency should continue to use only the original node block. Dummy nodes are routing artifacts with zero visual area; letting them participate in box packing adds noise and can push real nodes away from phantom geometry.
- Gradient optimization still owns dummy coordinates while the expanded state is active. Dummies exist in `state.pos`, receive gradients from the expanded edge losses, and are then removed by `StripDummyNodes`.
- Final output remains `pos[N_original, 2]`. Expanded positions are internal only.

That split is the important design choice. It keeps the metric-relevant part of dummy nodes alive, but does not pretend that invisible routing points are legitimate label boxes.

### Strategy choice

I recommend **Strategy A**: expand up-front, optimize the expanded graph, then extract original nodes at the end.

Reason:

- The research finding is specifically about `edge_length_cv` and `edge_straightness`, both of which are driven during native by differentiable edge losses, not just by the final ordering polish.
- If native keeps optimizing only the original stretched edge, the main loss source is still present during most of the solve, and the final ordering pass is trying to repair a geometry the optimizer never modeled correctly.
- The current code already has the pieces needed for a constrained Strategy A:
  - `InsertDummyNodes` already materializes an `ExpandedGraph`.
  - `CreateOptimizer` can optimize whatever tensor lives in `state.pos`.
  - `StripDummyNodes` already exists.
  - The only plumbing that needs widening is “which edge set do edge losses read?” and “which node block do box constraints read?”

This is the smallest change that actually gives dummy nodes real degrees of freedom inside native.

Assumption I am making explicitly: cluster-heavy DAGs can stay on the original path initially if they prove noisy under expanded-edge optimization, but I would not hard-gate them off in the first patch. The original-node-only box constraints are enough to keep the change conservative.

## Section 2 -- Exact code patches

### a) Gate + `InsertDummyNodes` in the native pipeline

#### [dagua/config.py:124-140] add a public kill-switch

```python
    # Sprint 19 Wave 2: expand long-span DAG edges into dummy-node chains
    # inside the default native pipeline. This is intentionally a public
    # kill-switch so head-to-head regressions can be isolated without
    # reverting the whole patch.
    enable_native_dummy_nodes: bool = True

    # Multilevel coarsening (default: N > 20K)
    multilevel_threshold: int = 20000
```

#### [dagua/layout/resolve.py:1-120, 276-370] add the gate helpers and annotate the config

```python
from dagua.utils import longest_path_layering


def _resolve_native_layer_assignments(
    edge_index: torch.Tensor,
    num_nodes: int,
    layer_assignments: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Return the layer tensor used by native DAG gating.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    layer_assignments : torch.Tensor | None
        Optional caller-supplied layer assignments.

    Returns
    -------
    torch.Tensor | None
        CPU ``torch.long`` layer assignments when available.
    """
    if layer_assignments is not None:
        return layer_assignments.detach().to(device="cpu", dtype=torch.long)
    if num_nodes == 0 or edge_index.numel() == 0:
        return None
    resolved = longest_path_layering(edge_index.detach().cpu(), num_nodes, device="cpu")
    if isinstance(resolved, torch.Tensor):
        return resolved.to(device="cpu", dtype=torch.long)
    return torch.tensor(resolved, dtype=torch.long)


def _has_long_edges(
    edge_index: torch.Tensor,
    layer_assignments: Optional[torch.Tensor],
) -> bool:
    """Return whether any edge spans at least two layers.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity with shape ``[2, E]``.
    layer_assignments : torch.Tensor | None
        Layer assignments with shape ``[N]``.

    Returns
    -------
    bool
        ``True`` when at least one edge span is ``>= 2``.
    """
    if layer_assignments is None or edge_index.numel() == 0:
        return False
    edge_cpu = edge_index.detach().to(device="cpu", dtype=torch.long)
    src = edge_cpu[0]
    dst = edge_cpu[1]
    spans = layer_assignments[dst] - layer_assignments[src]
    return bool((spans >= 2).any().item())


def _use_native_dummy_nodes(
    config: LayoutConfig,
    structure: Optional[GraphStructure],
    edge_index: torch.Tensor,
    layer_assignments: Optional[torch.Tensor],
) -> bool:
    """Return whether the native pipeline should activate dummy nodes.

    Parameters
    ----------
    config : LayoutConfig
        User-visible layout configuration.
    structure : GraphStructure | None
        Resolved graph classification.
    edge_index : torch.Tensor
        Graph connectivity with shape ``[2, E]``.
    layer_assignments : torch.Tensor | None
        Resolved layer assignments with shape ``[N]``.

    Returns
    -------
    bool
        ``True`` when the graph is a connected non-flat DAG with long edges.
    """
    if not getattr(config, "enable_native_dummy_nodes", True):
        return False
    if structure is None:
        return False
    if not bool(getattr(structure, "is_acyclic", True)):
        return False
    if int(getattr(structure, "num_components", 1)) != 1:
        return False
    if int(getattr(structure, "num_layers", 0)) <= 1:
        return False
    return _has_long_edges(edge_index=edge_index, layer_assignments=layer_assignments)


def prepare_pipeline_config(
    config: LayoutConfig,
    num_nodes: int,
    edge_index: torch.Tensor,
    device: str,
    layer_assignments: Optional[torch.Tensor],
    prebuilt_layer_index: Optional[Any],
    graph_structure: Optional[GraphStructure],
    skip_classification: bool,
) -> LayoutConfig:
    """Resolve native-engine pipeline settings for one problem instance.

    Produces a shallow config copy annotated with resolved private pipeline
    metadata (prefixed with ``_dagua_native_``) that ``build_dagua_pipeline``
    consumes.
    """
    effective_config = copy.copy(config)
    structure: Optional[GraphStructure] = None
    resolved_layer_assignments = _resolve_native_layer_assignments(
        edge_index=edge_index,
        num_nodes=num_nodes,
        layer_assignments=layer_assignments,
    )
    if not skip_classification:
        structure = graph_structure
        if structure is None:
            structure = classify_graph(
                edge_index,
                num_nodes,
                layer_assignments=resolved_layer_assignments,
            )
        if structure.family in {GraphFamily.TREE, GraphFamily.CHAIN}:
            effective_config = override_for_tree(effective_config)
        if structure.family == GraphFamily.CHAIN:
            auto_steps = auto_layout_steps(num_nodes)
            resolved_steps = min(
                effective_config.steps if effective_config.steps > 0 else auto_steps,
                50,
            )
        else:
            resolved_steps = (
                effective_config.steps
                if effective_config.steps > 0
                else auto_layout_steps(num_nodes)
            )
    else:
        resolved_steps = (
            effective_config.steps if effective_config.steps > 0 else auto_layout_steps(num_nodes)
        )

    resolved_node_sep = effective_config.node_sep
    resolved_rank_sep = effective_config.rank_sep
    if effective_config.adaptive_spacing:
        resolved_node_sep, resolved_rank_sep = adaptive_spacing(
            num_nodes=num_nodes,
            base_node_sep=resolved_node_sep,
            base_rank_sep=resolved_rank_sep,
        )

    stall_limit, rel_threshold = stall_config(num_nodes=num_nodes)
    setattr(effective_config, "_dagua_native_steps", resolved_steps)
    setattr(effective_config, "_dagua_native_node_sep", resolved_node_sep)
    setattr(effective_config, "_dagua_native_rank_sep", resolved_rank_sep)
    setattr(effective_config, "_dagua_native_device", device)
    setattr(effective_config, "_dagua_native_verbose", effective_config.verbose)
    setattr(effective_config, "_dagua_native_layer_assignments", resolved_layer_assignments)
    setattr(effective_config, "_dagua_native_prebuilt_layer_index", prebuilt_layer_index)
    setattr(
        effective_config,
        "_dagua_native_use_dummy_nodes",
        _use_native_dummy_nodes(
            config=effective_config,
            structure=structure,
            edge_index=edge_index,
            layer_assignments=resolved_layer_assignments,
        ),
    )
    setattr(
        effective_config,
        "_dagua_native_overlap_interval",
        overlap_interval(num_nodes=num_nodes, config=effective_config),
    )
    setattr(
        effective_config,
        "_dagua_native_final_projection_iterations",
        final_projection_iterations(num_nodes=num_nodes),
    )
    setattr(effective_config, "_dagua_native_stall_limit", stall_limit)
    setattr(effective_config, "_dagua_native_rel_threshold", rel_threshold)
    setattr(effective_config, "_dagua_native_crossing_alpha", 3.0)
    setattr(effective_config, "_dagua_native_optimizer_type", "adam")
    setattr(effective_config, "structure", structure)
    setattr(effective_config, "_dagua_native_structure", structure)
    setattr(effective_config, "_dagua_native_num_nodes", num_nodes)
```

#### [dagua/layout/ops/pipelines/dagua_native.py:16-69, 360-450] insert the expansion stage

```python
from dagua.layout.ops.layering import ActivateExpandedGraphState, InsertDummyNodes
from dagua.layout.ops.postprocess import AspectRatioFit, AspectRatioFitConfig, StripDummyNodes


def build_dagua_pipeline(config: LayoutConfig) -> Pipeline:
    """Build the composable native-engine pipeline from a resolved config."""
    resolved_steps = int(
        getattr(config, "_dagua_native_steps", config.steps if config.steps > 0 else 0),
    )
    resolved_node_sep = float(getattr(config, "_dagua_native_node_sep", config.node_sep))
    resolved_rank_sep = float(getattr(config, "_dagua_native_rank_sep", config.rank_sep))
    resolved_device = str(getattr(config, "_dagua_native_device", config.device))
    resolved_verbose = bool(getattr(config, "_dagua_native_verbose", config.verbose))
    resolved_use_dummy_nodes = bool(getattr(config, "_dagua_native_use_dummy_nodes", False))
    overlap_interval = int(getattr(config, "_dagua_native_overlap_interval", 5))
    final_projection_iterations = int(
        getattr(config, "_dagua_native_final_projection_iterations", 10),
    )
    stall_limit = int(getattr(config, "_dagua_native_stall_limit", 5))
    rel_threshold = float(getattr(config, "_dagua_native_rel_threshold", 1.0e-4))
    optimizer_type = str(getattr(config, "_dagua_native_optimizer_type", "adam"))
    losses = build_loss_ops(
        config=config,
        node_sep=resolved_node_sep,
        rank_sep=resolved_rank_sep,
    )
    weight_config = InitAnnealingScheduleConfig(
        w_dag=config.w_dag,
        w_attract=config.w_attract,
        w_repel=config.w_repel,
        w_overlap=config.w_overlap,
        w_cluster=config.w_cluster,
        w_cluster_contain=config.w_cluster_contain,
        w_crossing=config.w_crossing,
        w_straightness=config.w_straightness,
        w_length_variance=config.w_length_variance,
        w_spacing=config.w_spacing,
        w_fanout=config.w_fanout,
        w_back_edge=config.w_back_edge,
        w_stress=getattr(config, "w_stress", 0.0),
    )

    native_ops: list = [
        FixedSteps(FixedStepsConfig(n=resolved_steps)),
        NativeEngineInit(
            NativeEngineInitConfig(
                node_sep=resolved_node_sep,
                rank_sep=resolved_rank_sep,
                device=resolved_device,
                verbose=resolved_verbose,
                layer_assignments=getattr(config, "_dagua_native_layer_assignments", None),
                prebuilt_layer_index=getattr(
                    config,
                    "_dagua_native_prebuilt_layer_index",
                    None,
                ),
            ),
        ),
        Force2DInitIfFlat(Force2DInitIfFlatConfig()),
    ]
    if resolved_use_dummy_nodes:
        native_ops.extend(
            [
                InsertDummyNodes(),
                ActivateExpandedGraphState(),
            ]
        )
    native_ops.extend(
        [
            *_stress_pivot_prep(config),
            InitAnnealingSchedule(weight_config),
            CreateOptimizer(
                CreateOptimizerConfig(
                    optimizer_type=optimizer_type,
                    lr=config.lr,
                    target="pos",
                    key="default",
                ),
            ),
            build_gradient_core(
                losses=losses,
                steps=resolved_steps,
                overlap_interval=overlap_interval,
                stall_limit=stall_limit,
                rel_threshold=rel_threshold,
            ),
            BarycenterReorder(BarycenterReorderConfig()),
            OverlapProjection(
                OverlapProjectionConfig(
                    padding=2.0,
                    iterations=final_projection_iterations,
                ),
            ),
            StripDummyNodes(),
            AspectRatioFit(AspectRatioFitConfig()),
            ClusterGridArrange(ClusterGridArrangeConfig()),
        ]
    )
    return Pipeline(native_ops, name="dagua_native_pipeline")
```

### b) Expand `state.pos`/`state.layers`, and make ordering/BK consume expanded edges

#### [dagua/layout/ops/layering.py:630+] add an activation op after `InsertDummyNodes`

```python
def _expanded_layers_to_tensor(
    expanded_layers: list[list[int]],
    num_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    """Convert layered node lists into a dense per-node layer tensor.

    Parameters
    ----------
    expanded_layers : list[list[int]]
        Node ids grouped by layer.
    num_nodes : int
        Expanded node count.
    device : torch.device
        Output device.

    Returns
    -------
    torch.Tensor
        Layer tensor with shape ``[N_expanded]``.
    """
    layers = torch.full((num_nodes,), -1, dtype=torch.long)
    for layer_index, nodes in enumerate(expanded_layers):
        if not nodes:
            continue
        layers[torch.tensor(nodes, dtype=torch.long)] = layer_index
    if bool((layers < 0).any().item()):
        raise ValueError("expanded_graph.layers must cover every expanded node exactly once")
    return layers.to(device=device)


def _seed_expanded_positions(
    pos: torch.Tensor,
    edge_paths: list[list[int]],
    expanded_num_nodes: int,
) -> torch.Tensor:
    """Interpolate dummy-node coordinates along each expanded edge chain.

    Parameters
    ----------
    pos : torch.Tensor
        Original-node positions with shape ``[N_original, 2]``.
    edge_paths : list[list[int]]
        Dummy-expanded node chains for each original edge.
    expanded_num_nodes : int
        Total expanded node count.

    Returns
    -------
    torch.Tensor
        Expanded positions with shape ``[N_expanded, 2]``.
    """
    expanded = torch.zeros((expanded_num_nodes, 2), dtype=pos.dtype, device=pos.device)
    expanded[: pos.shape[0]] = pos
    for path in edge_paths:
        if len(path) <= 2:
            continue
        start = pos[path[0]]
        end = pos[path[-1]]
        denom = float(len(path) - 1)
        for step, node in enumerate(path[1:-1], start=1):
            alpha = float(step) / denom
            expanded[node] = start + (end - start) * alpha
    return expanded


@register_op
class ActivateExpandedGraphState(Op):
    """Promote ``state.pos`` and ``state.layers`` to the dummy-expanded graph."""

    name: ClassVar[str] = "activate_expanded_graph_state"
    category: ClassVar[OpCategory] = OpCategory.LAYERING
    reads: ClassVar[Tuple[str, ...]] = ("pos", "layers", "layer_index", "extras.expanded_graph")
    writes: ClassVar[Tuple[str, ...]] = (
        "pos",
        "layers",
        "layer_index",
        "extras.original_layers",
        "extras.original_layer_index",
    )
    requires: ClassVar[Tuple[str, ...]] = ("pos", "extras.expanded_graph")

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Expand the active solve state to match ``extras['expanded_graph']``.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State whose ``pos`` / ``layers`` / ``layer_index`` match the expanded graph.
        """
        del problem, ctx
        if state.pos is None:
            raise ValueError("ActivateExpandedGraphState requires state.pos.")
        expanded_graph = state.extras.get("expanded_graph")
        if expanded_graph is None:
            return state

        state.extras["original_layers"] = None if state.layers is None else state.layers.clone()
        state.extras["original_layer_index"] = state.layer_index

        expanded_pos = _seed_expanded_positions(
            pos=state.pos.detach(),
            edge_paths=expanded_graph.edge_paths,
            expanded_num_nodes=expanded_graph.num_nodes,
        )
        expanded_layers = _expanded_layers_to_tensor(
            expanded_layers=expanded_graph.layers,
            num_nodes=expanded_graph.num_nodes,
            device=state.pos.device,
        )
        state.pos = expanded_pos.requires_grad_(state.pos.requires_grad)
        state.layers = expanded_layers
        state.layer_index = build_layer_index(expanded_layers, device=str(state.pos.device))
        state.ordering = None
        return state
```

#### [dagua/layout/ops/barycenter.py:72-213] make the reorder pass use expanded edges when active

```python
def _active_edge_index(problem: LayoutProblem, state: SolveState, pos: torch.Tensor) -> torch.Tensor:
    """Return the edge tensor active for barycenter reordering."""
    expanded_graph = state.extras.get("expanded_graph")
    if expanded_graph is not None and int(getattr(expanded_graph, "num_nodes", -1)) == int(pos.shape[0]):
        return expanded_graph.edge_index.to(device=pos.device, dtype=torch.long)
    return problem.edge_index.to(device=pos.device, dtype=torch.long)


@register_op
@dataclass
class BarycenterReorder(Op):
    """Sugiyama barycenter polish pass over the active layered layout."""

    name: ClassVar[str] = "barycenter_reorder"
    category: ClassVar[OpCategory] = OpCategory.PROJECT
    reads: ClassVar[Tuple[str, ...]] = ("pos", "layers", "layer_index", "extras.expanded_graph")
    writes: ClassVar[Tuple[str, ...]] = ("pos",)
    requires: ClassVar[Tuple[str, ...]] = ("pos",)

    config: BarycenterReorderConfig = field(default_factory=BarycenterReorderConfig)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        del ctx
        if not self.config.enabled:
            return state
        if state.pos is None or state.layers is None:
            return state
        if state.layer_index is None or state.layer_index.num_layers < 2:
            return state

        pos = state.pos
        layers = state.layers.to(device=pos.device, dtype=torch.long)
        layer_index = state.layer_index
        edge_index = _active_edge_index(problem=problem, state=state, pos=pos)
        src_all = edge_index[0]
        tgt_all = edge_index[1]
        src_layer = layers[src_all]
        tgt_layer = layers[tgt_all]
        pos_new = pos.detach().clone()

        num_layers = layer_index.num_layers
        iterations = max(self.config.iterations, 0)
        min_size = max(self.config.min_layer_size, 2)

        for it in range(iterations):
            direction_up = (it % 2) == 0
            layer_order = range(1, num_layers) if direction_up else range(num_layers - 1, -1, -1)
            for k in layer_order:
                members = layer_index.nodes_in_layer(k).to(device=pos_new.device)
                if members.numel() < min_size:
                    continue
                if direction_up:
                    adj_layer_idx = k - 1
                    mask = (tgt_layer == k) & (src_layer == adj_layer_idx)
                    neighbour_src = tgt_all[mask]
                    neighbour_dst = src_all[mask]
                else:
                    adj_layer_idx = k + 1
                    if adj_layer_idx >= num_layers:
                        continue
                    mask = (src_layer == k) & (tgt_layer == adj_layer_idx)
                    neighbour_src = src_all[mask]
                    neighbour_dst = tgt_all[mask]
                barycenters = _compute_barycenters_for_layer(
                    pos_new,
                    members,
                    mask,
                    neighbour_src,
                    neighbour_dst,
                )
                order = torch.argsort(barycenters, stable=True)
                current_x = pos_new[members, 0]
                sorted_x, _ = torch.sort(current_x)
                new_member_order = members[order]
                pos_new[new_member_order, 0] = sorted_x

        state.pos = pos_new.detach().requires_grad_(pos.requires_grad)
        return state
```

#### [dagua/layout/ops/ordering.py:38-124, 648-837] validate against the active expanded graph

The same active-graph rule should be applied to `BarycenterSweep`, `MedianSweep`, and `TransposeHeuristic`, otherwise Wave 2 patch #3 will still validate against `problem.num_nodes` and silently fall back to original-node orderings. The drop-in helper is:

```python
def _active_ordering_graph(
    problem: LayoutProblem,
    state: SolveState,
) -> tuple[int, torch.Tensor]:
    """Return the node count and edge tensor used by ordering ops."""
    expanded_graph = state.extras.get("expanded_graph")
    if (
        expanded_graph is not None
        and state.layers is not None
        and int(getattr(expanded_graph, "num_nodes", -1)) == int(state.layers.shape[0])
    ):
        return expanded_graph.num_nodes, expanded_graph.edge_index
    return problem.num_nodes, problem.edge_index
```

Then each ordering op replaces `problem.num_nodes` and `problem.edge_index` with the values from `_active_ordering_graph(...)`. That is the only change needed for the Wave 2 median+transpose patch to become dummy-aware.

#### [dagua/layout/ops/coordinate.py:1249-1323] make BK use expanded nodes once Wave 2 patch #1 lands

```python
@register_op
class BrandesKopf4Pass(Op):
    """Assign layered coordinates via four-pass Brandes-Kopf horizontal compaction."""

    name: ClassVar[str] = "brandes_kopf_4pass"
    category: ClassVar[OpCategory] = OpCategory.COORDINATE
    reads: ClassVar[Tuple[str, ...]] = ("layers", "ordering", "extras.expanded_graph")
    writes: ClassVar[Tuple[str, ...]] = ("pos", "extras.expanded_positions")
    requires: ClassVar[Tuple[str, ...]] = ("layers", "ordering")
    access_pattern: ClassVar[str] = "global"

    def __init__(self, config: Optional[BrandesKopf4PassConfig] = None) -> None:
        self.config = config or BrandesKopf4PassConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        del ctx

        expanded_graph = state.extras.get("expanded_graph")
        active_num_nodes = problem.num_nodes
        active_edge_index = problem.edge_index
        active_node_sizes = _resolve_node_sizes(problem.node_sizes, problem.num_nodes)
        if (
            expanded_graph is not None
            and state.layers is not None
            and int(getattr(expanded_graph, "num_nodes", -1)) == int(state.layers.shape[0])
        ):
            active_num_nodes = expanded_graph.num_nodes
            active_edge_index = expanded_graph.edge_index
            active_node_sizes = expanded_graph.node_sizes

        layers_cpu = _validate_layers(state.layers, active_num_nodes)
        ordering_cpu = _validate_ordering(state.ordering, active_num_nodes)
        edge_index_cpu = _validate_edge_index(active_edge_index, active_num_nodes)
        ordered_layers = _ordered_layers_from_state(layers_cpu, ordering_cpu)
        parents, children = _layered_neighbors_from_edges(
            edge_index=edge_index_cpu,
            layers=layers_cpu,
            num_nodes=active_num_nodes,
        )

        positions = torch.zeros((active_num_nodes, _POSITION_OUTPUT_DIM), dtype=torch.float32)
        if active_num_nodes > 0:
            x_coordinates = _brandes_koepf_x_positions(
                layers=ordered_layers,
                parents=parents,
                children=children,
                node_sizes=active_node_sizes,
                num_nodes=active_num_nodes,
                num_original_nodes=problem.num_nodes,
                node_sep=self.config.node_sep,
            )
            positions[:, 0] = torch.tensor(x_coordinates, dtype=torch.float32)
            positions[:, 1] = layers_cpu.to(dtype=torch.float32) * self.config.rank_sep

        device_positions = positions.to(device=_target_device(problem, state))
        state.extras["expanded_positions"] = device_positions
        state.pos = device_positions
        return state
```

### c) Let the gradient core optimize dummy positions, but keep box constraints original-only

#### [dagua/layout/ops/loss_engine.py:37-80, 394-790, 965-1092] add active-edge helpers

```python
from typing import Any, ClassVar, Optional, Tuple


def _expanded_graph_for_state(state: SolveState, pos: torch.Tensor) -> Optional[Any]:
    """Return the active expanded graph when ``state.pos`` matches it."""
    expanded_graph = state.extras.get("expanded_graph")
    if expanded_graph is None:
        return None
    if int(getattr(expanded_graph, "num_nodes", -1)) != int(pos.shape[0]):
        return None
    return expanded_graph


def _active_edge_index(problem: LayoutProblem, state: SolveState, pos: torch.Tensor) -> torch.Tensor:
    """Return the edge tensor used by edge-centric losses."""
    expanded_graph = _expanded_graph_for_state(state=state, pos=pos)
    if expanded_graph is not None:
        return expanded_graph.edge_index.to(device=pos.device, dtype=torch.long)
    return problem.edge_index.to(device=pos.device, dtype=torch.long)


def _visible_original_pos(problem: LayoutProblem, state: SolveState, pos: torch.Tensor) -> torch.Tensor:
    """Return the original-node position block for box-centric losses."""
    expanded_graph = _expanded_graph_for_state(state=state, pos=pos)
    if expanded_graph is None:
        return pos
    return pos[: problem.num_nodes]


def _visible_original_layer_index(state: SolveState, pos: torch.Tensor) -> Optional[object]:
    """Return the layer index used by original-node spacing/projection logic."""
    expanded_graph = _expanded_graph_for_state(state=state, pos=pos)
    if expanded_graph is None:
        return state.layer_index
    return state.extras.get("original_layer_index", state.layer_index)


@register_op
@dataclass(frozen=True)
class DagOrderingLoss(LossOp):
    ...
    def evaluate(self, problem: LayoutProblem, state: SolveState, ctx: RuntimeContext) -> torch.Tensor:
        del ctx
        pos = _require_pos(state)
        node_sizes = (
            state.extras["expanded_graph"].node_sizes.to(device=pos.device, dtype=pos.dtype)
            if _expanded_graph_for_state(state, pos) is not None
            else _require_node_sizes(problem)
        )
        return dag_ordering_loss(
            pos,
            _active_edge_index(problem, state, pos),
            node_sizes,
            rank_sep=self.config.rank_sep,
            edge_ctx=state.edge_batch_context,
        )


@register_op
@dataclass(frozen=True)
class EdgeAttractionLoss(LossOp):
    ...
    def evaluate(self, problem: LayoutProblem, state: SolveState, ctx: RuntimeContext) -> torch.Tensor:
        del ctx
        pos = _require_pos(state)
        return edge_attraction_loss(
            pos,
            _active_edge_index(problem, state, pos),
            x_bias=self.config.x_bias,
            edge_ctx=state.edge_batch_context,
        )


@register_op
@dataclass(frozen=True)
class EdgeStraightnessLoss(LossOp):
    ...
    def evaluate(self, problem: LayoutProblem, state: SolveState, ctx: RuntimeContext) -> torch.Tensor:
        del ctx
        pos = _require_pos(state)
        return edge_straightness_loss(
            pos,
            _active_edge_index(problem, state, pos),
            edge_ctx=state.edge_batch_context,
        )


@register_op
@dataclass(frozen=True)
class EdgeLengthVarianceLoss(LossOp):
    ...
    def evaluate(self, problem: LayoutProblem, state: SolveState, ctx: RuntimeContext) -> torch.Tensor:
        del ctx
        pos = _require_pos(state)
        return edge_length_variance_loss(
            pos,
            _active_edge_index(problem, state, pos),
            edge_ctx=state.edge_batch_context,
        )


@register_op
@dataclass(frozen=True)
class CrossingLoss(LossOp):
    ...
    def evaluate(self, problem: LayoutProblem, state: SolveState, ctx: RuntimeContext) -> torch.Tensor:
        del ctx
        pos = _require_pos(state)
        return crossing_loss(
            pos,
            _active_edge_index(problem, state, pos),
            alpha=self.config.alpha,
            max_pairs=self.config.max_pairs,
            layer_assignments=state.layers,
        )


@register_op
@dataclass(frozen=True)
class RepulsionLoss(LossOp):
    ...
    def evaluate(self, problem: LayoutProblem, state: SolveState, ctx: RuntimeContext) -> torch.Tensor:
        del ctx
        pos = _visible_original_pos(problem, state, _require_pos(state))
        node_sizes = problem.node_sizes
        num_nodes = pos.shape[0]
        ...


@register_op
@dataclass(frozen=True)
class OverlapAvoidanceLoss(LossOp):
    ...
    def evaluate(self, problem: LayoutProblem, state: SolveState, ctx: RuntimeContext) -> torch.Tensor:
        del ctx
        pos = _visible_original_pos(problem, state, _require_pos(state))
        node_sizes = _require_node_sizes(problem)
        num_nodes = pos.shape[0]
        ...


@register_op
@dataclass(frozen=True)
class SpacingConsistencyLoss(LossOp):
    ...
    def evaluate(self, problem: LayoutProblem, state: SolveState, ctx: RuntimeContext) -> torch.Tensor:
        del ctx
        pos = _visible_original_pos(problem, state, _require_pos(state))
        node_sizes = _require_node_sizes(problem)
        return spacing_consistency_loss(
            pos,
            node_sizes,
            _visible_original_layer_index(state, _require_pos(state)),
            target_gap=self.config.target_gap,
        )


@register_op
@dataclass(frozen=True)
class FanoutDistributionLoss(LossOp):
    ...
    def evaluate(self, problem: LayoutProblem, state: SolveState, ctx: RuntimeContext) -> torch.Tensor:
        del ctx
        pos = _visible_original_pos(problem, state, _require_pos(state))
        return fanout_distribution_loss(
            pos,
            problem.edge_index,
            degree_threshold=self.config.degree_threshold,
            edge_ctx=state.edge_batch_context,
            step=state.step,
            edge_is_sampled=state.edge_batch_context is not None,
        )


@register_op
@dataclass(frozen=True)
class BackEdgeCompactnessLoss(LossOp):
    ...
    def evaluate(self, problem: LayoutProblem, state: SolveState, ctx: RuntimeContext) -> torch.Tensor:
        del ctx
        pos = _visible_original_pos(problem, state, _require_pos(state))
        return back_edge_compactness_loss(
            pos,
            problem.edge_index,
            edge_ctx=state.edge_batch_context,
        )
```

That is the key behavior split: dummies receive gradients from edge losses, but box/cluster/fanout losses stay tied to the visible original graph.

#### [dagua/layout/ops/project.py:208-343] keep overlap projection original-only

```python
def _visible_original_positions(
    problem: LayoutProblem,
    state: SolveState,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, Optional[object]]:
    """Return the original-node position view and its layer index."""
    expanded_graph = state.extras.get("expanded_graph")
    if expanded_graph is None or int(getattr(expanded_graph, "num_nodes", -1)) != int(positions.shape[0]):
        return positions, state.layer_index
    return positions[: problem.num_nodes], state.extras.get("original_layer_index", state.layer_index)


@register_op
@dataclass(frozen=True)
class OverlapProjection(Op):
    ...
    def apply(self, problem: LayoutProblem, state: SolveState, ctx: RuntimeContext) -> SolveState:
        del ctx
        positions = _require_positions(state=state, op_name=self.name)
        if problem.node_sizes is None or problem.node_sizes.numel() == 0:
            return state
        visible_positions, layer_index = _visible_original_positions(problem, state, positions)
        node_sizes = problem.node_sizes.to(device=visible_positions.device, dtype=visible_positions.dtype)
        project_overlaps(
            pos=visible_positions,
            node_sizes=node_sizes,
            padding=self.config.padding,
            iterations=self.config.iterations,
            layer_index=layer_index,
        )
        state.pos = positions
        return state


@register_op
@dataclass(frozen=True)
class PeriodicOverlapProjection(Op):
    ...
    def apply(self, problem: LayoutProblem, state: SolveState, ctx: RuntimeContext) -> SolveState:
        del ctx
        positions = _require_positions(state=state, op_name=self.name)
        if problem.node_sizes is None or problem.node_sizes.numel() == 0:
            return state
        ...
        visible_positions, layer_index = _visible_original_positions(problem, state, positions)
        node_sizes = problem.node_sizes.to(device=visible_positions.device, dtype=visible_positions.dtype)
        project_overlaps(
            pos=visible_positions,
            node_sizes=node_sizes,
            padding=self.config.padding,
            iterations=iterations,
            layer_index=layer_index,
        )
        state.pos = positions
        return state
```

### d) Strip dummies before user-visible postprocessing and defensively slice on return

#### [dagua/layout/ops/postprocess.py:1091-1137] restore original layer state when stripping

```python
@register_op
class StripDummyNodes(Op):
    """Remove dummy-node coordinates introduced by layered graph expansion."""

    name: ClassVar[str] = "strip_dummy_nodes"
    category: ClassVar[OpCategory] = OpCategory.POSTPROCESS
    reads: ClassVar[Tuple[str, ...]] = (
        "pos",
        "layers",
        "ordering",
        "extras.expanded_graph",
        "extras.original_layers",
        "extras.original_layer_index",
    )
    writes: ClassVar[Tuple[str, ...]] = ("pos", "layers", "layer_index", "ordering")
    requires: ClassVar[Tuple[str, ...]] = ("pos",)
    access_pattern: ClassVar[str] = "global"

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        _ = ctx
        positions = _require_positions(state=state, op_name=self.name)
        expanded_graph = state.extras.get(_EXPANDED_GRAPH_KEY)
        if expanded_graph is not None:
            expanded_num_nodes = getattr(expanded_graph, "num_nodes", positions.shape[0])
            if expanded_num_nodes < problem.num_nodes:
                raise ValueError(
                    "expanded_graph.num_nodes cannot be smaller than problem.num_nodes"
                )
        visible_nodes = min(problem.num_nodes, positions.shape[0])
        state.pos = positions[:visible_nodes].clone()
        original_layers = state.extras.get("original_layers")
        original_layer_index = state.extras.get("original_layer_index")
        if isinstance(original_layers, torch.Tensor):
            state.layers = original_layers[:visible_nodes].clone()
            state.layer_index = original_layer_index
        elif state.layers is not None and state.layers.shape[0] != visible_nodes:
            state.layers = state.layers[:visible_nodes].clone()
        if state.ordering is not None and state.ordering.shape[0] != visible_nodes:
            state.ordering = state.ordering[:visible_nodes].clone()
        return state
```

#### [dagua/layout/ops/pipelines/dagua_native.py:557-560] final extraction guard

```python
    final_state = build_dagua_pipeline(prepared_config).apply(problem, state, ctx)
    if final_state.pos is None:
        raise RuntimeError("dagua_native pipeline did not produce final positions.")
    result = final_state.pos.detach()
    if result.shape[0] > num_nodes:
        result = result[:num_nodes]
    return result
```

### e) Guard behavior summary

The complete skip guard is:

```python
if not config.enable_native_dummy_nodes:
    skip
if structure is None or not structure.is_acyclic:
    skip
if structure.num_components != 1:
    skip
if structure.num_layers <= 1:
    skip
if no edge span >= 2:
    skip
```

That is the right answer for “cyclic OR disconnected OR flat, skip dummy insertion.”

## Section 3 -- Interaction with concurrent Wave 2 patches

Dependency order should be:

1. **Wave 2 #4 component decomposition** first, at the outer orchestration layer. Dummy-node gating must run per component, not on the pre-packed disconnected supergraph.
2. **Dummy-node insertion** next. Every later layered phase should see the expanded edge set, not the original long edges.
3. **Wave 2 #3 median+transpose ordering** after dummy insertion. Median and transpose are only worth doing on the expanded graph because the whole point is to expose intermediate long-edge segments.
4. **Wave 2 #1 Brandes-Koepf coordinate assignment** after ordering. BK should read the expanded ordering and expanded node sizes, then leave `state.pos` expanded until `StripDummyNodes`.
5. **`StripDummyNodes`** immediately after the last dummy-aware layered phase.
6. **Wave 2 #5 topology-aware aspect ratio** last, on original positions only. Aspect-ratio fit should not see dummy routing points, or it will overestimate width/height from internal chain geometry.

So the order inside a connected DAG component should be:

`NativeEngineInit -> InsertDummyNodes -> ActivateExpandedGraphState -> gradient_core -> median/transpose -> BK -> OverlapProjection(original only) -> StripDummyNodes -> AspectRatioFit -> ClusterGridArrange`

If #1 lands before this patch, `BrandesKopf4Pass` must take the coordinate patch above at the same time. If #3 lands first but still validates against `problem.num_nodes`, it will silently ignore dummies; the `_active_ordering_graph(...)` helper avoids that failure mode.

## Section 4 -- Regression safety

Top-5 current dagua wins should stay safe for structural reasons:

- `random_dag_200`: likely neutral-to-positive. Random DAGs usually contain skip edges, so some improvement in straightness/CV is plausible, but the gate still skips if the sampled layering collapses to mostly unit spans.
- `org_chart_deep`: mostly a no-op. Org-chart trees are dominated by unit-span parent-child edges; if there are no spans `>= 2`, the new gate never fires.
- `random_dag_50`: same story as `random_dag_200`, but smaller. Some mild gain is possible; a regression is unlikely because the old path is still used when no long edges exist.
- `hub_fanout_label_skew`: likely a no-op. This family is wide and shallow, so `structure.num_layers` can be >1, but most edges are one-rank fanout edges, not long skips.
- `org_chart_1_5_4_8`: effectively a no-op. This graph is tree-like and already one of Dagua’s strongest cases.

The important safety property is: **graphs with zero long edges are unchanged**. The helper in `resolve.py` checks actual layer spans, not just “is DAG.” That prevents the native pipeline from paying extra machinery on already-good layered graphs.

The other safety property is negative gating:

- Cyclic graph: no expansion.
- Disconnected graph: no expansion until per-component decomposition exists.
- Flat graph: no expansion.

Those three cases are where dummy nodes would otherwise create expensive internal state without giving the solver any legitimate layered structure to exploit.

## Section 5 -- Tests

I would add one unit test in [tests/test_ops_layering.py](/home/jtaylor/projects/dagua/tests/test_ops_layering.py) and three integration/regression tests in [tests/test_layout/test_engine.py](/home/jtaylor/projects/dagua/tests/test_layout/test_engine.py).

### [tests/test_ops_layering.py] unit test for the helper

```python
from dagua.layout.ops.layering import _expand_long_edges_with_dummy_nodes


def test_expand_long_edges_with_dummy_nodes_builds_expected_layer_chain() -> None:
    """A three-rank skip edge should become a full dummy chain."""
    edge_index = torch.tensor([[0, 0], [1, 3]], dtype=torch.long)
    layers = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    node_sizes = torch.ones((4, 2), dtype=torch.float32)

    expanded, _ = _expand_long_edges_with_dummy_nodes(
        edge_index=edge_index,
        layer_assignments=layers,
        node_sizes=node_sizes,
        num_original_nodes=4,
        dummy_size=(0.0, 0.0),
        edge_weights=None,
    )

    assert expanded.num_nodes == 6
    assert expanded.edge_paths == [[0, 1], [0, 4, 5, 3]]
    assert expanded.layers == [[0], [1, 4], [2, 5], [3]]
    assert torch.equal(
        expanded.edge_index,
        torch.tensor([[0, 0, 4, 5], [1, 4, 5, 3]], dtype=torch.long),
    )
```

### [tests/test_layout/test_engine.py] integration + regression + metric tests

```python
from dagua.eval.graphs import get_test_graphs
from dagua.layout.resolve import prepare_pipeline_config
from dagua.metrics import full


def _named_eval_graph(name: str) -> DaguaGraph:
    """Return a benchmark graph by name."""
    for tg in get_test_graphs():
        if tg.name == name:
            tg.graph.compute_node_sizes()
            return tg.graph
    raise AssertionError(f"missing graph fixture: {name}")


@pytest.mark.slow
def test_native_dummy_nodes_reduce_hexagonal_lattice_edge_length_cv() -> None:
    """Dummy-node splitting should reduce long-edge CV on a planar DAG."""
    graph = _named_eval_graph("hexagonal_lattice_42")
    off_cfg = LayoutConfig(seed=42, steps=150, enable_native_dummy_nodes=False)
    on_cfg = LayoutConfig(seed=42, steps=150, enable_native_dummy_nodes=True)

    off_pos = layout(graph, off_cfg)
    on_pos = layout(graph, on_cfg)

    off_metrics = full(off_pos, graph.edge_index, node_sizes=graph.node_sizes)
    on_metrics = full(on_pos, graph.edge_index, node_sizes=graph.node_sizes)

    assert on_metrics["edge_length_cv"] + 0.03 < off_metrics["edge_length_cv"]


def test_native_dummy_nodes_skip_cyclic_graph() -> None:
    """The native dummy-node gate must stay off for cyclic inputs."""
    graph = DaguaGraph.from_edge_list([("a", "b"), ("b", "c"), ("c", "a")])
    graph.compute_node_sizes()
    prepared = prepare_pipeline_config(
        config=LayoutConfig(enable_native_dummy_nodes=True, seed=42),
        num_nodes=graph.num_nodes,
        edge_index=graph.edge_index,
        device="cpu",
        layer_assignments=None,
        prebuilt_layer_index=None,
        graph_structure=None,
        skip_classification=False,
    )
    assert getattr(prepared, "_dagua_native_use_dummy_nodes", False) is False


@pytest.mark.slow
def test_native_dummy_nodes_reduce_dependency_edge_straightness() -> None:
    """Expanded long edges should lower mean deviation on a deep dependency DAG."""
    graph = _named_eval_graph("dependency_500")
    off_cfg = LayoutConfig(seed=42, steps=180, enable_native_dummy_nodes=False)
    on_cfg = LayoutConfig(seed=42, steps=180, enable_native_dummy_nodes=True)

    off_pos = layout(graph, off_cfg)
    on_pos = layout(graph, on_cfg)

    off_metrics = full(off_pos, graph.edge_index, node_sizes=graph.node_sizes)
    on_metrics = full(on_pos, graph.edge_index, node_sizes=graph.node_sizes)

    assert on_metrics["edge_straightness_mean_deg"] <= (
        off_metrics["edge_straightness_mean_deg"] - 4.0
    )
```

The Tier 1/Tier 2 commands for the real patch should be the project-standard ones:

- `ruff check . --fix`
- `mypy --follow-imports=silent dagua/cli.py`
- `pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q`
- Final once: `pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"`

## Section 6 -- Expected impact

These are realistic patch-level targets, not best-case theoretical ceilings:

| Graph | Current | Expected after | Why |
|---|---:|---:|---|
| `hexagonal_lattice_42` | `82.42` | `84.8` to `85.6` | Main gain is `edge_length_cv`: splitting skip edges turns one long segment into near-unit spans. A `CV` drop of `0.08` is worth about `+1.6` composite by `20 * (1 - CV)`. Straightness can plausibly improve another `5-7°`, worth about `+1.1` to `+1.6`. |
| `sierpinski_42` | `78.35` | `80.6` to `81.6` | Same mechanism as the hexagonal lattice, but this graph is less regular and more crossing-sensitive, so I expect a slightly smaller CV win and a modest straightness win. |
| `dense_pair_50` | `71.81` | `72.5` to `73.3` | Dense DAGs do not get the full dummy-node benefit because many edges are already local and crossings dominate. The improvement here is mostly on the few true skip edges plus cleaner barycenter behavior. |
| `dependency_500` | `51.96` | `54.5` to `56.0` | This is the highest-value target. Deep sparse dependency graphs pay both the CV penalty and the straightness penalty today. A `CV` drop around `0.09-0.12` is `+1.8` to `+2.4`; a `6-9°` straightness drop is another `+1.3` to `+2.0`. |
| `extreme_mixed_width_transformer` | `73.82` | `75.8` to `76.8` | Mixed-width transformer motifs have obvious long skip edges. Dummy chains give ordering and the edge losses intermediate control points, which should reduce the current “one stretched diagonal skip” failure mode. |

The metric-formula logic is straightforward:

- `edge_length_cv` contributes `20 * max(0, 1 - CV)`. Every `0.05` CV reduction is about `+1.0` composite.
- `edge_straightness_mean_deg` contributes `10 * max(0, 1 - deg / 45)`. Every `4.5°` drop is about `+1.0` composite.
- `crossing_rate` is secondary here, but splitting long edges into adjacent-layer segments gives the reorder passes more local structure, so small spillover gains are plausible once Wave 2 #3 lands.

So the Sprint 19 expectation of “+2 to +4 composite on the targeted DAG losses” is consistent with the actual formulas.

## Section 7 -- Rollback and verification

### Kill-switch

The public rollback switch should be:

```python
LayoutConfig(enable_native_dummy_nodes=False)
```

That is better than an environment variable because it is visible in tests, benchmarks, and per-graph diagnostics.

### Verification command

Use an explicit on/off head-to-head on the target graphs:

```bash
CUDA_VISIBLE_DEVICES="" python - <<'PY'
from dagua.config import LayoutConfig
from dagua.eval.graphs import get_test_graphs
from dagua.layout.engine import layout as engine_layout
from dagua.metrics import full, composite

targets = {
    "hexagonal_lattice_42",
    "sierpinski_42",
    "dependency_500",
    "extreme_mixed_width_transformer",
}
graphs = {tg.name: tg.graph for tg in get_test_graphs() if tg.name in targets}

for name in sorted(graphs):
    g = graphs[name]
    g.compute_node_sizes()
    print(f"\n== {name} ==")
    for enabled in (False, True):
        cfg = LayoutConfig(seed=42, enable_native_dummy_nodes=enabled)
        pos = engine_layout(g, cfg)
        m = full(pos, g.edge_index, node_sizes=g.node_sizes)
        print(
            f"enabled={enabled}  composite={composite(m):.2f}  "
            f"edge_length_cv={m['edge_length_cv']:.4f}  "
            f"edge_straightness_mean_deg={m['edge_straightness_mean_deg']:.2f}"
        )
PY
```

Then spot-check the competitor gap on one graph with the existing single-graph diagnostic:

```bash
CUDA_VISIBLE_DEVICES="" python /tmp/diag_single.py dependency_500
```

### Rollback plan

If the patch regresses a protected family:

1. Flip `enable_native_dummy_nodes` to `False` in the benchmark config and confirm the regression disappears.
2. If the regression is only on disconnected graphs, keep the code and tighten the gate until Wave 2 #4 lands.
3. If the regression is on shallow DAGs with no real long-edge headroom, tighten the span gate from `>= 2` to `>= 3`.

The lowest-risk rollback is therefore “disable by config, keep code in tree,” not “rip the feature back out.”
