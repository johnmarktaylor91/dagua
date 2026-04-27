# Wave 2 / Task 01: Brandes-Koepf In `dagua_native`

Two source-level caveats materially shape this plan.

First, `dagua/layout/graph_classify.py` is safe to reuse for `family` and rough `num_layers`, but not for a DAG gate by itself. `GraphStructure.is_acyclic` is populated from the undirected shadow graph in `_count_components_and_acyclic()` (`graph_classify.py:91-145`), so a directed diamond DAG already reports `False`. `num_components` is also intentionally lossy when `E > N - 1`: the helper returns `(1, False)` early at `graph_classify.py:113-115`, which hides real multi-component cases such as the current `dependency_500` generator (exact weak components `[499, 1]`). Second, the Wave 1 report baselines in `CONTEXT.md` and `area_B_loss_buckets__claude.md` do not exactly match a fresh local re-score of the cached `variant_bench_full` position files with the current `dagua.metrics.full()` implementation. I therefore treat the published Wave 1 numbers as the baseline-of-record for score targets, and use the metric formulas in `dagua/metrics.py:1153-1212` only to budget conservative deltas.

## Section 1 -- Scope and design choice

The exact change is: keep the default `dagua_native` pipeline intact, but insert a new x-only post-ordering refinement op after `BarycenterReorder` and before `OverlapProjection`/`AspectRatioFit`. That op reconstructs the current within-layer order from the native pipeline’s `state.pos[:, 0]`, runs the existing four-pass Brandes-Koepf horizontal compaction from `dagua/layout/ops/coordinate.py`, and writes only the new x coordinates back. It does not route the whole graph through the `sugiyama` pipeline, does not change y coordinates, and does not introduce dummy nodes yet; that separation is deliberate because Wave 2 task #2 is already carrying dummy-node expansion, and replacing native DAG layout wholesale with `sugiyama` now would duplicate layering/order work and entangle this change with in-flight ranking and component patches.

The gate should be conservative. The correct decision tree is:

1. If the graph is a `TREE` or `CHAIN`, do not run BK.
2. Else if the current layout has fewer than 6 layers, do not run BK.
3. Else if the exact weak-component sizes are neither `[N]` nor `[N - 1, 1]`, do not run BK.
4. Else if any edge violates the current layering order (`layers[src] >= layers[dst]`), do not run BK.
5. Else run BK horizontal compaction on the current native ordering and preserve y.

That means BK is applied to:

- `hexagonal_lattice_42`
- `sierpinski_42`
- `dense_pair_50`
- `extreme_mixed_width_transformer`
- `dependency_500`

`dependency_500` only qualifies because the exact weak components are `[499, 1]`: one dominant DAG plus one isolated tail node. The gate intentionally allows that singleton-tail case without claiming full disconnected-component support.

BK is not applied to:

- Cyclic or back-edge-bearing graphs such as `small_world_100` and `recurrent_feedback_cell`
- Trees and chains, which already bypass `dagua_native` through the Reingold-Tilford fast path in `dagua_native.py:529-556`
- Shallow / flat DAGs with fewer than 6 layers, including `hub_fanout_label_skew`
- Multi-component DAGs that are not just “one main component plus one isolate”, including `random_dag_50` and `random_dag_200`

I am not recommending a direct “route all DAGs to `sugiyama`” patch here. The current `sugiyama` pipeline (`dagua/layout/ops/pipelines/sugiyama.py:59-76`) is the right long-term layered bucket, but it currently owns dummy expansion, barycenter ordering, and coordinate assignment as one monolithic alternative pipeline. For Wave 2 task #1, the lowest-risk merge is a narrow `dagua_native` insertion: `NativeEngineInit -> gradient_core -> BarycenterReorder -> BK-x-refine -> OverlapProjection -> AspectRatioFit`. That keeps native y placement, avoids early interaction with dummy-node collapse, and still lands the highest-leverage missing phase called out in Wave 1 Finding A3.

## Section 2 -- Exact code patch(es)

### Patch A

File: `/home/jtaylor/projects/dagua/dagua/layout/ops/coordinate.py`

Insertion point: immediately after `_layered_neighbors_from_edges()` and before `BrandesKopf4PassConfig` (current anchor starts at `def _layered_neighbors_from_edges(` around `coordinate.py:774`).

New code:

```python
from dagua.layout.graph_classify import (
    GraphFamily,
    GraphStructure as TopologyGraphStructure,
    classify_graph,
)

_BRANDES_KOEPF_APPLIED_KEY = "brandes_koepf_horizontal_refine_applied"


def _weak_component_sizes(edge_index: torch.Tensor, num_nodes: int) -> List[int]:
    """Return exact weak-component sizes for the directed graph.

    Parameters
    ----------
    edge_index : torch.Tensor
        CPU edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes ``N``.

    Returns
    -------
    list of int
        Weak-component sizes sorted descending.
    """
    if num_nodes == 0:
        return []

    adjacency: List[List[int]] = [[] for _ in range(num_nodes)]
    for source, target in edge_index.t().tolist():
        adjacency[source].append(target)
        adjacency[target].append(source)

    seen = [False] * num_nodes
    sizes: List[int] = []
    for start in range(num_nodes):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        component_size = 0
        while stack:
            node = stack.pop()
            component_size += 1
            for neighbor in adjacency[node]:
                if seen[neighbor]:
                    continue
                seen[neighbor] = True
                stack.append(neighbor)
        sizes.append(component_size)
    sizes.sort(reverse=True)
    return sizes


def _has_strict_forward_layering(edge_index: torch.Tensor, layers: torch.Tensor) -> bool:
    """Return whether all edges advance strictly in the current layering.

    Parameters
    ----------
    edge_index : torch.Tensor
        CPU edge tensor with shape ``[2, E]``.
    layers : torch.Tensor
        CPU layer assignments with shape ``[N]``.

    Returns
    -------
    bool
        ``True`` when every edge satisfies ``layers[src] < layers[dst]``.
    """
    if edge_index.numel() == 0:
        return False
    layer_deltas = layers[edge_index[1]] - layers[edge_index[0]]
    return bool(torch.all(layer_deltas > 0).item())


def _ordering_from_current_x(layers: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Build an in-layer ordering from the current x coordinates.

    Parameters
    ----------
    layers : torch.Tensor
        CPU layer assignments with shape ``[N]``.
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.

    Returns
    -------
    torch.Tensor
        CPU ordering tensor with shape ``[N]``.
    """
    if layers.numel() == 0:
        return torch.zeros((0,), dtype=torch.long)

    pos_cpu = pos.detach().to(device="cpu", dtype=torch.float32)
    max_layer = int(layers.max().item())
    ordering = torch.zeros((layers.shape[0],), dtype=torch.long)
    for layer_index in range(max_layer + 1):
        layer_nodes = torch.where(layers == layer_index)[0].tolist()
        layer_nodes.sort(key=lambda node_idx: (float(pos_cpu[node_idx, 0].item()), node_idx))
        for position, node_idx in enumerate(layer_nodes):
            ordering[node_idx] = position
    return ordering


def _resolve_topology_structure(
    structure: Optional[TopologyGraphStructure],
    edge_index: torch.Tensor,
    layers: torch.Tensor,
    num_nodes: int,
) -> TopologyGraphStructure:
    """Return the topology structure used by the BK gate.

    Parameters
    ----------
    structure : TopologyGraphStructure | None
        Optional pre-classified structure from pipeline config.
    edge_index : torch.Tensor
        CPU edge tensor with shape ``[2, E]``.
    layers : torch.Tensor
        CPU layer assignments with shape ``[N]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    TopologyGraphStructure
        Structure used for family-based gating.
    """
    if structure is not None:
        return structure
    return classify_graph(edge_index=edge_index, num_nodes=num_nodes, layer_assignments=layers)


def _should_apply_brandes_koepf_refine(
    structure: TopologyGraphStructure,
    edge_index: torch.Tensor,
    layers: torch.Tensor,
    num_nodes: int,
    min_layers: int,
) -> bool:
    """Return whether the BK horizontal refine should run.

    Parameters
    ----------
    structure : TopologyGraphStructure
        Classified graph structure. Only the stable fields ``family`` and
        the existence of a layering are trusted here.
    edge_index : torch.Tensor
        CPU edge tensor with shape ``[2, E]``.
    layers : torch.Tensor
        CPU layer assignments with shape ``[N]``.
    num_nodes : int
        Number of graph nodes.
    min_layers : int
        Minimum layer count required for BK to be worthwhile.

    Returns
    -------
    bool
        ``True`` when the graph is a safe BK candidate.

    Notes
    -----
    ``classify_graph().is_acyclic`` and ``classify_graph().num_components`` are
    intentionally *not* used here because they are conservative forest-oriented
    signals, not exact directed-DAG / weak-component checks on general DAGs.
    """
    if num_nodes == 0:
        return False
    if structure.family in {GraphFamily.TREE, GraphFamily.CHAIN}:
        return False

    num_layers = int(layers.max().item()) + 1 if layers.numel() > 0 else 0
    if num_layers < min_layers:
        return False

    component_sizes = _weak_component_sizes(edge_index=edge_index, num_nodes=num_nodes)
    if component_sizes not in ([num_nodes], [num_nodes - 1, 1]):
        return False

    return _has_strict_forward_layering(edge_index=edge_index, layers=layers)


@dataclass(frozen=True)
class BrandesKoepfHorizontalRefineConfig:
    """Configuration for :class:`BrandesKoepfHorizontalRefine`.

    Parameters
    ----------
    node_sep : float, default=1.0
        Horizontal separation used by the BK compaction pass.
    min_layers : int, default=6
        Minimum number of layers required before BK runs.
    enabled : bool, default=True
        Master kill-switch used for rollback and A/B tests.
    structure : TopologyGraphStructure | None, default=None
        Optional pre-classified graph structure from pipeline resolution.
    """

    node_sep: float = 1.0
    min_layers: int = 6
    enabled: bool = True
    structure: Optional[TopologyGraphStructure] = None


@register_op
@dataclass(frozen=True)
class BrandesKoepfHorizontalRefine(Op):
    """Reassign x coordinates with Brandes-Koepf while preserving y.

    The native pipeline already owns y placement and within-layer crossing
    reduction. This op only reconstructs the current left-to-right order,
    runs BK compaction on that order, and writes the new x coordinates back.
    """

    config: BrandesKoepfHorizontalRefineConfig = field(
        default_factory=BrandesKoepfHorizontalRefineConfig
    )

    name: ClassVar[str] = "brandes_koepf_horizontal_refine"
    category: ClassVar[OpCategory] = OpCategory.COORDINATE
    reads: ClassVar[Tuple[str, ...]] = ("pos", "layers")
    writes: ClassVar[Tuple[str, ...]] = (
        "pos",
        "ordering",
        f"extras.{_BRANDES_KOEPF_APPLIED_KEY}",
    )
    requires: ClassVar[Tuple[str, ...]] = ("pos", "layers")
    access_pattern: ClassVar[str] = "global"

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Apply Brandes-Koepf x compaction to the current layered layout.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state with ``pos`` and ``layers`` populated.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this deterministic op.

        Returns
        -------
        SolveState
            Updated state with BK-refined ``pos[:, 0]`` when the gate passes.
        """
        del ctx

        state.extras[_BRANDES_KOEPF_APPLIED_KEY] = False
        if not self.config.enabled or state.pos is None or state.layers is None:
            return state

        layers_cpu = _validate_layers(state.layers, problem.num_nodes)
        edge_index_cpu = _validate_edge_index(problem.edge_index, problem.num_nodes)
        structure = _resolve_topology_structure(
            structure=self.config.structure,
            edge_index=edge_index_cpu,
            layers=layers_cpu,
            num_nodes=problem.num_nodes,
        )
        if not _should_apply_brandes_koepf_refine(
            structure=structure,
            edge_index=edge_index_cpu,
            layers=layers_cpu,
            num_nodes=problem.num_nodes,
            min_layers=max(self.config.min_layers, 2),
        ):
            return state

        ordering_cpu = _ordering_from_current_x(layers=layers_cpu, pos=state.pos)
        ordered_layers = _ordered_layers_from_state(layers=layers_cpu, ordering=ordering_cpu)
        parents, children = _layered_neighbors_from_edges(
            edge_index=edge_index_cpu,
            layers=layers_cpu,
            num_nodes=problem.num_nodes,
        )
        node_sizes_cpu = _resolve_node_sizes(problem.node_sizes, problem.num_nodes)

        requires_grad = bool(state.pos.requires_grad)
        updated_pos = state.pos.detach().clone()
        updated_pos[:, 0] = torch.tensor(
            _brandes_koepf_x_positions(
                layers=ordered_layers,
                parents=parents,
                children=children,
                node_sizes=node_sizes_cpu,
                num_nodes=problem.num_nodes,
                num_original_nodes=problem.num_nodes,
                node_sep=self.config.node_sep,
            ),
            dtype=updated_pos.dtype,
            device=updated_pos.device,
        )
        state.pos = updated_pos.requires_grad_(requires_grad)
        state.ordering = ordering_cpu.to(device=_target_device(problem, state), dtype=torch.long)
        state.extras[_BRANDES_KOEPF_APPLIED_KEY] = True
        return state
```

Rationale: this is the narrowest possible BK integration. It reuses the existing BK internals in `coordinate.py`, preserves the native y solution, and adds exact local gate logic where `GraphStructure.is_acyclic` / `num_components` are not reliable enough for this purpose.

### Patch B

File: `/home/jtaylor/projects/dagua/dagua/layout/resolve.py`

Insertion point: inside `prepare_pipeline_config()`, immediately after `setattr(effective_config, "_dagua_native_structure", structure)` (current anchor at `resolve.py:347-348`).

New code:

```python
    setattr(effective_config, "structure", structure)
    setattr(effective_config, "_dagua_native_structure", structure)
    if not hasattr(effective_config, "_dagua_native_enable_brandes_koepf"):
        setattr(effective_config, "_dagua_native_enable_brandes_koepf", True)
    if not hasattr(effective_config, "_dagua_native_brandes_koepf_min_layers"):
        setattr(effective_config, "_dagua_native_brandes_koepf_min_layers", 6)
```

Rationale: the BK refine needs a default-on feature flag and a stable `min_layers` threshold, both of which should be overrideable in tests and rollback experiments without adding new public API surface in Sprint 19.

### Patch C

File: `/home/jtaylor/projects/dagua/dagua/layout/ops/pipelines/dagua_native.py`

Insertion point: import section near the top, plus the main non-V-cycle `Pipeline(...)` body that currently ends `BarycenterReorder -> OverlapProjection -> AspectRatioFit` (current anchor `dagua_native.py:421-439`).

New code:

```python
from dagua.layout.ops.coordinate import (
    BrandesKoepfHorizontalRefine,
    BrandesKoepfHorizontalRefineConfig,
)

    return Pipeline(
        [
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
            BrandesKoepfHorizontalRefine(
                BrandesKoepfHorizontalRefineConfig(
                    node_sep=resolved_node_sep,
                    min_layers=int(
                        getattr(config, "_dagua_native_brandes_koepf_min_layers", 6)
                    ),
                    enabled=bool(
                        getattr(config, "_dagua_native_enable_brandes_koepf", True)
                    ),
                    structure=getattr(config, "_dagua_native_structure", None),
                ),
            ),
            OverlapProjection(
                OverlapProjectionConfig(
                    padding=2.0,
                    iterations=final_projection_iterations,
                ),
            ),
            AspectRatioFit(AspectRatioFitConfig()),
            ClusterGridArrange(ClusterGridArrangeConfig()),
        ],
        name="dagua_native_pipeline",
    )
```

Rationale: the ordering is the key. `BarycenterReorder` stays the crossing-reduction phase, BK becomes the horizontal coordinate assignment phase, and `AspectRatioFit` remains last so it rescales a compacted layout instead of fighting the compaction.

### No patch in `graph_classify.py`

I am explicitly *not* changing the semantics of `GraphStructure.is_acyclic` or the early-exit behavior in `_count_components_and_acyclic()` in this task, even though they are the reason the BK gate cannot trust those fields. Changing the classifier globally would have wider consequences for loss gating and algorithm dispatch. For this sprint item, the correct conservative move is to keep the classifier stable and do the exact DAG/component checks locally inside the BK refine op.

## Section 3 -- Regression safety analysis

The top-five current wins are the regression budget. The proposed gate is shaped to skip four of the five completely and to skip the fifth (`hub_fanout_label_skew`) until the ordering stack is stronger.

`random_dag_200` (+27): do not apply BK. The current generator produces 202 weak components with sizes headed by `[181, 2, 1, ...]`; that is far outside the allowed `[N]` or `[N - 1, 1]` component patterns. Running BK across dozens of isolates would turn this into accidental component-packing work, which is Wave 2 task #4, not this task.

`org_chart_deep` (+23): do not apply BK. It is a `TREE` under `classify_graph()`, and `layout_dagua_native_pipeline()` already exits through Reingold-Tilford before `build_dagua_pipeline()` runs. The proposed BK insertion therefore cannot touch it unless the existing tree fast path regresses independently.

`random_dag_50` (+22): do not apply BK. Exact weak-component sizes are `[45, 2, 1, ...]`, again failing the connected / singleton-tail gate. This graph’s current win is a good example of why the gate cannot be “all DAGs”: structurally it is a highly fragmented DAG bucket, not a single coherent layered drawing problem.

`hub_fanout_label_skew` (+16): do not apply BK in this first merge. It is a connected DAG-like graph, but only 5 layers deep. That is exactly the kind of shallow graph where the native path is already winning and where BK can over-compact width before Wave 2 task #3 (median + transpose ordering) lands. The `min_layers=6` threshold is there primarily to keep this graph out of scope for now.

`org_chart_1_5_4_8` (+16): do not apply BK. Like `org_chart_deep`, it is a tree and stays on the existing exact tree path.

This is the practical consequence of the gate:

- The current large wins are preserved because the BK op does not run on them.
- The loss graphs named in Wave 1 mostly do get BK: `hexagonal_lattice_42`, `sierpinski_42`, `dense_pair_50`, `extreme_mixed_width_transformer`.
- `dependency_500` gets BK only because its exact component pattern is `[499, 1]`, not because `GraphStructure.num_components` says so.

If one of the skipped winners later becomes a new BK candidate after subsequent wave patches, the merge order should be: land stronger ordering first, then relax the gate. In particular, the first candidate to reconsider is `hub_fanout_label_skew`, and only after the median/transpose patch is in place.

## Section 4 -- Tests to add

I would add one new file: `/home/jtaylor/projects/dagua/tests/test_pipeline_dagua_native_bk.py`.

```python
from __future__ import annotations

import copy

import pytest
import torch

from dagua.config import LayoutConfig
from dagua.eval.graphs import _make_hexagonal_lattice_graph
from dagua.layout.ops.pipelines.dagua_native import build_dagua_pipeline
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState
from dagua.layout.resolve import normalize_node_sizes, prepare_pipeline_config
from dagua.metrics import composite, full


def _run_native_pipeline(
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    config: LayoutConfig,
) -> SolveState:
    """Execute the native pipeline and return the final solve state."""
    num_nodes = int(node_sizes.shape[0])
    prepared = prepare_pipeline_config(
        config=copy.copy(config),
        num_nodes=num_nodes,
        edge_index=edge_index,
        device="cpu",
        layer_assignments=None,
        prebuilt_layer_index=None,
        graph_structure=None,
        skip_classification=False,
    )
    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=normalize_node_sizes(node_sizes=node_sizes, device=torch.device("cpu")),
        direction=prepared.direction,
        seed=int(prepared.seed),
    )
    state = SolveState()
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu", optimizer_type="adam"))
    return build_dagua_pipeline(prepared).apply(problem, state, ctx)


def test_brandes_koepf_gate_only_applies_to_deep_connected_dag_like_graphs() -> None:
    edge_index = torch.tensor(
        [[0, 0, 1, 2, 3, 4, 5], [1, 2, 3, 3, 4, 5, 6]],
        dtype=torch.long,
    )
    node_sizes = torch.full((7, 2), 20.0, dtype=torch.float32)
    enabled_config = LayoutConfig(seed=42, steps=12)
    setattr(enabled_config, "_dagua_native_enable_brandes_koepf", True)
    setattr(enabled_config, "_dagua_native_brandes_koepf_min_layers", 6)

    deep_state = _run_native_pipeline(edge_index=edge_index, node_sizes=node_sizes, config=enabled_config)

    assert deep_state.extras["brandes_koepf_horizontal_refine_applied"] is True

    shallow_edge_index = torch.tensor(
        [[0, 0, 1, 2], [1, 2, 3, 4]],
        dtype=torch.long,
    )
    shallow_node_sizes = torch.full((5, 2), 20.0, dtype=torch.float32)
    shallow_state = _run_native_pipeline(
        edge_index=shallow_edge_index,
        node_sizes=shallow_node_sizes,
        config=enabled_config,
    )

    assert shallow_state.extras["brandes_koepf_horizontal_refine_applied"] is False


def test_native_pipeline_brandes_koepf_refine_aligns_rank_blocks() -> None:
    edge_index = torch.tensor(
        [[0, 0, 1, 2, 3, 4, 5], [1, 2, 3, 3, 4, 5, 6]],
        dtype=torch.long,
    )
    node_sizes = torch.full((7, 2), 20.0, dtype=torch.float32)
    config = LayoutConfig(seed=42, steps=12)
    setattr(config, "_dagua_native_enable_brandes_koepf", True)
    setattr(config, "_dagua_native_brandes_koepf_min_layers", 6)

    result = _run_native_pipeline(edge_index=edge_index, node_sizes=node_sizes, config=config)

    assert result.pos is not None
    assert result.extras["brandes_koepf_horizontal_refine_applied"] is True
    assert result.pos[3, 0].item() == pytest.approx(result.pos[4, 0].item(), abs=1.0e-4)
    assert result.pos[4, 0].item() == pytest.approx(result.pos[5, 0].item(), abs=1.0e-4)
    assert result.pos[5, 0].item() == pytest.approx(result.pos[6, 0].item(), abs=1.0e-4)


def test_cyclic_graph_does_not_trigger_brandes_koepf_refine() -> None:
    edge_index = torch.tensor(
        [[0, 1, 2, 2], [1, 2, 0, 3]],
        dtype=torch.long,
    )
    node_sizes = torch.full((4, 2), 20.0, dtype=torch.float32)
    config = LayoutConfig(seed=42, steps=12)
    setattr(config, "_dagua_native_enable_brandes_koepf", True)
    setattr(config, "_dagua_native_brandes_koepf_min_layers", 3)

    result = _run_native_pipeline(edge_index=edge_index, node_sizes=node_sizes, config=config)

    assert result.pos is not None
    assert torch.isfinite(result.pos).all()
    assert result.extras["brandes_koepf_horizontal_refine_applied"] is False


def test_hexagonal_lattice_42_composite_improves_with_brandes_koepf() -> None:
    graph = _make_hexagonal_lattice_graph(rows=6, cols=7)
    node_sizes = torch.full((graph.num_nodes, 2), 20.0, dtype=torch.float32)

    without_bk = LayoutConfig(seed=42, steps=20)
    setattr(without_bk, "_dagua_native_enable_brandes_koepf", False)
    setattr(without_bk, "_dagua_native_brandes_koepf_min_layers", 6)

    with_bk = LayoutConfig(seed=42, steps=20)
    setattr(with_bk, "_dagua_native_enable_brandes_koepf", True)
    setattr(with_bk, "_dagua_native_brandes_koepf_min_layers", 6)

    baseline_state = _run_native_pipeline(
        edge_index=graph.edge_index,
        node_sizes=node_sizes,
        config=without_bk,
    )
    refined_state = _run_native_pipeline(
        edge_index=graph.edge_index,
        node_sizes=node_sizes,
        config=with_bk,
    )

    assert baseline_state.pos is not None
    assert refined_state.pos is not None

    baseline_score = composite(full(baseline_state.pos, graph.edge_index, node_sizes=node_sizes))
    refined_score = composite(full(refined_state.pos, graph.edge_index, node_sizes=node_sizes))

    assert refined_score >= baseline_score + 1.0
```

Why this exact test shape:

- The first test proves the gate logic instead of only proving the op exists.
- The second proves the intended BK behavioral signature: a merged vertical block gets a single x column.
- The third is the required cycle regression guard.
- The fourth is the minimum benchmark-facing proof that the patch is buying real composite score on a named Wave 1 loss graph rather than merely moving coordinates around.

## Section 5 -- Expected impact and verification plan

The score budgeting below uses the published Wave 1 baselines and the composite formula in `dagua/metrics.py:1168-1199`. A few conversion constants matter:

- `edge_length_cv`: every `0.05` reduction is `+1.0` composite
- `edge_straightness_mean_deg`: every `4.5` degree reduction is `+1.0` composite
- `crossing_rate`: every `0.01` reduction is `+1.0` composite
- `angular_res_mean_deg`: every `8` degree increase is `+1.0` composite

Conservative expected post-patch composites:

- `hexagonal_lattice_42`: `82.42 -> 84.5`
  - Budget: `edge_length_cv` down by ~`0.08` (`+1.6`), `crossing_rate` down by ~`0.003` (`+0.3`), `angular_res_mean_deg` up ~`1.5-2.0` degrees (`+0.2` to `+0.25`).
  - Why not bigger: Wave 1 already showed that most of this graph’s remaining gap is topology-blind aspect ratio, not ordering alone.

- `sierpinski_42`: `78.35 -> 80.3`
  - Budget: `edge_length_cv` down ~`0.07` (`+1.4`), `crossing_rate` down ~`0.002` (`+0.2`), `angular_res_mean_deg` up ~`2.0` degrees (`+0.25`).

- `dense_pair_50`: `71.81 -> 74.7`
  - Budget: `edge_straightness_mean_deg` down ~`9` degrees (`+2.0`), `crossing_rate` down ~`0.007` (`+0.7`), `edge_length_cv` down ~`0.01` (`+0.2`).
  - This is the graph where BK should pay off most directly because the Wave 1 loss is already concentrated in straightness and crossing pressure rather than aspect ratio.

- `dependency_500`: `51.96 -> 55.0`
  - Budget: `edge_length_cv` down ~`0.05` (`+1.0`), `edge_straightness_mean_deg` down ~`4` degrees (`+0.9`), `crossing_rate` down ~`0.011` (`+1.1`).
  - This assumes the current graph shape remains the same `[499, 1]` dominant-plus-isolate pattern. If that benchmark graph evolves into a genuinely disconnected multi-component DAG, this patch should stop applying and the score expectation should be revised downward until Wave 2 task #4 merges.

- `extreme_mixed_width_transformer`: `73.82 -> 75.1`
  - Budget: `edge_length_cv` down ~`0.03` (`+0.6`), `edge_straightness_mean_deg` down ~`2` degrees (`+0.4`), residual crossing cleanup from better x-compaction ~`+0.3`.
  - The one-crossing elimination here still belongs mostly to Wave 2 task #3; BK alone is not the whole answer.

Verification command after implementation:

```bash
for g in \
  hexagonal_lattice_42 \
  sierpinski_42 \
  dense_pair_50 \
  dependency_500 \
  extreme_mixed_width_transformer
do
  CUDA_VISIBLE_DEVICES="" python /tmp/diag_single.py "$g"
done
```

Then run the suite-level comparison once:

```bash
CUDA_VISIBLE_DEVICES="" python /tmp/h2h2.py
```

Rollback plan if the mean regresses:

1. Flip the new private default in `prepare_pipeline_config()` by setting `_dagua_native_enable_brandes_koepf = False`.
2. Re-run the five target graphs plus the five current winners.
3. If the regression disappears, keep the kill-switch off and tighten the gate rather than reverting unrelated code.

That is why Patch B matters: it gives Sprint 19 a one-line rollback without deleting the op or disturbing future Wave 2 merges.

## Section 6 -- Ordering / composition with the other wave-2 patches

Wave 2 task #2, dummy-node long-edge splitting: must run before BK once it lands. Brandes-Koepf is strongest on a dummy-expanded layered DAG because it can align whole long-edge chains to one column. The patch in this report intentionally runs BK on the original graph only. After task #2 merges, `BrandesKoepfHorizontalRefine` should switch from `(problem.edge_index, state.layers)` to the expanded graph stored in `state.extras["expanded_graph"]`, then collapse back to original-node positions afterward.

Wave 2 task #3, median + transpose crossing reduction: should run immediately before BK. BK should never be asked to “fix” a bad layer order; it should only assign coordinates for an already decent order. Merge order recommendation: insert median/transpose between `BarycenterReorder` and `BrandesKoepfHorizontalRefine`, or replace `BarycenterReorder` entirely with the stronger ordering stack while leaving BK in the same relative position.

Wave 2 task #4, per-component decomposition: should wrap around BK rather than fight it. Once component decomposition exists, BK should run inside each component sub-pipeline, not across the packed global graph. The gate in this report is intentionally strict so that task #4 can later relax it cleanly instead of untangling a too-eager global BK pass.

Wave 2 task #5, topology-aware aspect ratio: stays after BK. This ordering should not change. BK computes relative x compaction inside the layered structure; aspect-ratio fit is a whole-layout affine rescale. If task #5 moves earlier than BK, it will just create a larger coordinate field for BK to compact again.

Recommended merge order across the layered bucket:

1. This BK patch lands first as the narrow x-only refine.
2. Median/transpose ordering lands next and feeds a better order into the same BK stage.
3. Dummy-node expansion lands after that and upgrades BK from original-node compaction to true Sugiyama-style long-edge alignment.
4. Component decomposition wraps the whole bucket afterward.
5. Topology-aware aspect-ratio remains the last affine stage.

That sequence keeps each patch locally understandable:

- ordering improves the order
- BK improves the coordinates for that order
- dummy nodes improve the graph BK sees
- component decomposition changes the subproblem boundary
- aspect ratio rescales the already-corrected layout
