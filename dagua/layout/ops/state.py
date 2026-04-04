"""Layout pipeline state objects.

Three-part state model:

LayoutProblem
    Immutable structural inputs: graph topology, node sizes, cluster
    membership, user constraints, direction. Created once, never modified
    by ops. Passed as read-only context.

SolveState
    Mutable working state: positions, hierarchy levels, cached precomputes
    (distances, layers, contexts), convergence counters, annealing state.
    Ops read and write this freely.

RuntimeContext
    Execution infrastructure: device plan, memory policy, trace/progress
    sinks, checkpoint handles, RNG generators. Not layout-specific — manages
    HOW ops execute, not WHAT they compute.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import torch


@dataclass(frozen=True)
class GraphStructure:
    """Classification of graph topology.

    Computed once via graph classification and used to select algorithm
    fast-paths such as tree, chain, or wide-layered layouts.

    Attributes
    ----------
    family : str
        Structural family: general, tree, forest, chain, wide_layered, grid.
    num_nodes : int
        Node count.
    num_edges : int
        Edge count.
    max_degree : int
        Maximum node degree.
    depth : int
        Longest path length for DAG-like graphs.
    max_layer_width : int
        Widest layer in a topological layering.
    num_layers : int
        Number of distinct layers.
    avg_layer_width : float
        Mean layer width.
    is_dag : bool
        Whether the graph is acyclic.
    num_components : int
        Number of connected components.
    """

    family: str
    num_nodes: int
    num_edges: int
    max_degree: int
    depth: int
    max_layer_width: int
    num_layers: int = 0
    avg_layer_width: float = 0.0
    is_dag: bool = True
    num_components: int = 1


@dataclass(frozen=True)
class FlexConstraints:
    """User-specified layout constraints.

    These inputs are prepared before execution and treated as read-only by
    operations so they can be safely shared across pipeline stages.

    Attributes
    ----------
    pin_indices : torch.Tensor | None
        Indices of pinned nodes.
    pin_targets : torch.Tensor | None
        Target positions for pinned nodes with shape ``[P, 2]``.
    pin_weights : torch.Tensor | None
        Per-pin strength weights.
    soft_pin_mask : torch.Tensor | None
        Boolean mask for soft, loss-based pins.
    hard_pin_mask : torch.Tensor | None
        Boolean mask for hard, projection-based pins.
    align_groups : list[Any] | None
        Alignment group specifications.
    flex_node_sep : float | None
        Target node separation.
    flex_node_sep_weight : float | None
        Strength of the node separation constraint.
    """

    pin_indices: Optional[torch.Tensor] = None
    pin_targets: Optional[torch.Tensor] = None
    pin_weights: Optional[torch.Tensor] = None
    soft_pin_mask: Optional[torch.Tensor] = None
    hard_pin_mask: Optional[torch.Tensor] = None
    align_groups: Optional[List[Any]] = None
    flex_node_sep: Optional[float] = None
    flex_node_sep_weight: Optional[float] = None


@dataclass
class LayoutProblem:
    """Immutable problem definition for layout.

    Created once from user inputs. Operations read this structure but should
    not mutate it. The object holds graph topology, node metadata, user
    constraints, and optional topology classification.

    Attributes
    ----------
    edge_index : torch.Tensor
        Edge list with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    node_sizes : torch.Tensor | None
        Node bounding boxes with shape ``[N, 2]``.
    node_labels : list[str] | None
        Node label strings.
    direction : str
        Layout direction such as ``TB``, ``BT``, ``LR``, or ``RL``.
    clusters : dict[str, Any] | None
        Cluster membership mapping.
    cluster_parents : dict[str, str] | None
        Cluster hierarchy mapping.
    structure : GraphStructure | None
        Topology classification, usually computed lazily.
    flex : FlexConstraints | None
        User constraints such as pins and alignments.
    seed : int
        Base random seed for reproducible planning.
    """

    edge_index: torch.Tensor
    num_nodes: int
    node_sizes: Optional[torch.Tensor] = None
    node_labels: Optional[List[str]] = None
    direction: str = "BT"
    clusters: Optional[Dict[str, Any]] = None
    cluster_parents: Optional[Dict[str, str]] = None
    structure: Optional[GraphStructure] = None
    flex: Optional[FlexConstraints] = None
    edge_weights: Optional[torch.Tensor] = None
    seed: int = 42

    # NOTE: LayoutProblem is intentionally NOT frozen -- some fields
    # (structure) are lazily computed.  However, ops SHOULD treat it
    # as read-only.  edge_weights lives here (not in config) because
    # 15 of 25 classic algorithms consume it.


@dataclass
class HierarchyLevel:
    """One level in a multilevel coarsening hierarchy.

    Tensor payloads are nullable so a level can be offloaded to disk for
    billion-node workflows. When offloaded, ``offload_path`` points to the
    serialized payload and tensor fields may be set to ``None``.

    Attributes
    ----------
    num_nodes : int
        Nodes at this coarsened level.
    num_fine : int
        Nodes at the finer child level.
    edge_index : torch.Tensor | None
        Coarsened edges with shape ``[2, E_coarse]``.
    node_sizes : torch.Tensor | None
        Coarsened node sizes.
    fine_to_coarse : torch.Tensor | None
        Mapping from fine to coarse node IDs with shape ``[N_fine]``.
    fine_layer_assignments : torch.Tensor | None
        Layer assignments at the finer level.
    coarse_layer_assignments : torch.Tensor | None
        Layer assignments at this level.
    offload_path : Path | None
        Disk path for the serialized payload.
    offload_dir : Path | None
        Parent directory for offload files.
    payload_fields : tuple[str, ...]
        Names of fields included in the offload payload.
    """

    num_nodes: int
    num_fine: int
    edge_index: Optional[torch.Tensor] = None
    node_sizes: Optional[torch.Tensor] = None
    fine_to_coarse: Optional[torch.Tensor] = None
    fine_layer_assignments: Optional[torch.Tensor] = None
    coarse_layer_assignments: Optional[torch.Tensor] = None
    cluster_ids: Optional[torch.Tensor] = None
    offload_path: Optional[Path] = None
    offload_dir: Optional[Path] = None
    payload_fields: Tuple[str, ...] = ()


@dataclass
class AnnealingSchedule:
    """Annealing configuration and current optimization snapshot.

    Schedule functions map ``(step, total_steps)`` to the current weight so
    the optimization loop can update weights without mutating the schedule
    definition itself.

    Attributes
    ----------
    weight_fns : dict[str, Callable[[int, int], float]]
        Per-weight schedule functions.
    current_weights : dict[str, float]
        Snapshot of current weight values for the active step.
    crossing_alpha : float
        Current crossing loss exponent.
    crossing_step_counter : int
        Steps since the last crossing computation.
    crossing_interval : int
        How often to compute crossing loss.
    crossing_weight_scale : float
        Multiplicative compensation for skipped crossing steps.
    """

    weight_fns: Dict[str, Callable[[int, int], float]] = field(default_factory=dict)
    current_weights: Dict[str, float] = field(default_factory=dict)
    crossing_alpha: float = 3.0
    crossing_step_counter: int = 0
    crossing_interval: int = 1
    crossing_weight_scale: float = 1.0


@dataclass
class SolveState:
    """Mutable working state for layout operations.

    This object holds positions, hierarchy data, cached precomputes, and
    convergence counters. ``pos`` is optional because hierarchy construction
    can happen before position initialization.

    Attributes
    ----------
    pos : torch.Tensor | None
        Current node positions with shape ``[N, 2]``.
    layers : torch.Tensor | None
        DAG layer assignments with shape ``[N]``.
    layer_index : Any | None
        Precomputed layer index helper.
    back_edge_mask : torch.Tensor | None
        Boolean mask of reversed back-edges with shape ``[E]``.
    hierarchy : list[HierarchyLevel] | None
        Multilevel coarsening hierarchy from finest to coarsest.
    adjacency : list[list[int]] | None
        Cached undirected adjacency list for graph traversals.
    adjacency_weighted : list[list[tuple[int, float]]] | None
        Cached weighted adjacency list.
    dense_adjacency : torch.Tensor | None
        Dense adjacency matrix with shape ``[N, N]``.
    distance_matrix : torch.Tensor | None
        All-pairs shortest paths with shape ``[N, N]``.
    pivot_indices : torch.Tensor | None
        Pivot node IDs with shape ``[P]``.
    pivot_distances : torch.Tensor | None
        Pivot-to-node distances with shape ``[P, N]``.
    component_ids : torch.Tensor | None
        Connected component labels with shape ``[N]``.
    degree : torch.Tensor | None
        Node degree with shape ``[N]``.
    laplacian : Any | None
        Cached graph Laplacian object.
    affinity_matrix : torch.Tensor | None
        Affinity matrix used by embedding-style initializers.
    spring_lengths : torch.Tensor | None
        Target edge lengths for spring layouts.
    spring_strengths : torch.Tensor | None
        Spring constants for spring layouts.
    sampled_node_context : Any | None
        Per-step sampled node context.
    edge_batch_context : Any | None
        Per-step edge batching context.
    annealing : AnnealingSchedule | None
        Annealing schedule state.
    optimizer : Any | None
        Optimizer instance (e.g. ``torch.optim.Adam``). Created by an
        init-phase op after ``pos`` is allocated. Stored here because
        the optimizer holds a reference to ``pos`` and carries stateful
        running averages (Adam m1/m2, SGD momentum).
    step : int
        Current optimization step.
    total_steps : int
        Total planned optimization steps.
    prev_loss : float
        Previous total loss, used for convergence checks.
    stall_count : int
        Consecutive steps without improvement.
    ops_applied : list[str]
        Bounded provenance history of recently applied operations.
    forces : torch.Tensor | None
        Force accumulation buffer with shape ``[N, 2]``. Used by
        non-differentiable force-directed layouts (FR, GEM, etc.).
        Force ops write into this buffer; a displacement op reads it,
        clamps, and applies to ``pos``.
    old_forces : torch.Tensor | None
        Previous iteration's forces with shape ``[N, 2]``. Needed by
        FA2's adaptive speed controller which compares swinging between
        consecutive force vectors.
    temperature : float | None
        Global temperature scalar for simulated-annealing-style layouts
        (FR, SFDP, GEM, Davidson-Harel, DRL, FMMM, native engine).
    ordering : torch.Tensor | None
        Within-layer node ordering indices with shape ``[N]``. Used by
        Sugiyama crossing minimization and init_placement barycenter.
    edge_routes : Any | None
        Polyline or Bezier curve routes per edge. Produced by Sugiyama
        dummy-node route reconstruction or edge optimization.
    converged : bool
        Flag set by convergence-checking ops or ``EarlyBreak``. When
        ``True``, ``Repeat`` stops its iteration loop.
    extras : dict[str, Any]
        Escape hatch for algorithm-specific transient state that does
        not belong in the core schema. Key convention: ``algo_field``
        (e.g., ``gem_local_temperatures``, ``tsne_velocity``,
        ``drl_density_grid``, ``sgd2_crossing_state``).
    """

    pos: Optional[torch.Tensor] = None
    layers: Optional[torch.Tensor] = None
    layer_index: Optional[Any] = None
    back_edge_mask: Optional[torch.Tensor] = None
    hierarchy: Optional[List[HierarchyLevel]] = None
    adjacency: Optional[List[List[int]]] = None
    adjacency_weighted: Optional[List[List[Tuple[int, float]]]] = None
    dense_adjacency: Optional[torch.Tensor] = None
    distance_matrix: Optional[torch.Tensor] = None
    pivot_indices: Optional[torch.Tensor] = None
    pivot_distances: Optional[torch.Tensor] = None
    component_ids: Optional[torch.Tensor] = None
    degree: Optional[torch.Tensor] = None
    laplacian: Optional[Any] = None
    affinity_matrix: Optional[torch.Tensor] = None
    spring_lengths: Optional[torch.Tensor] = None
    spring_strengths: Optional[torch.Tensor] = None
    sampled_node_context: Optional[Any] = None
    edge_batch_context: Optional[Any] = None
    # -- Optimization bookkeeping ----------------------------------------
    annealing: Optional[AnnealingSchedule] = None
    optimizer: Optional[Any] = None
    step: int = 0
    total_steps: int = 100
    prev_loss: float = float("inf")
    stall_count: int = 0
    ops_applied: List[str] = field(default_factory=list)

    # -- Force accumulation (non-differentiable force-directed layouts) ---
    forces: Optional[torch.Tensor] = None
    old_forces: Optional[torch.Tensor] = None

    # -- Temperature (global cooling for FR, SFDP, GEM, DH, DRL, FMMM) --
    temperature: Optional[float] = None
    force_area: Optional[float] = None

    # -- Ordering (within-layer node ordering for Sugiyama, init, etc.) --
    ordering: Optional[torch.Tensor] = None

    # -- Edge routes (polylines/curves from Sugiyama or edge optimization)
    edge_routes: Optional[Any] = None

    # -- Convergence flag (set by EarlyBreak or convergence ops) ---------
    converged: bool = False

    # -- Escape hatch for algorithm-specific transient state -------------
    # Guidelines:
    #   - Use for state needed by only 1-2 algorithms
    #   - Key convention: "algo_field" (e.g. "gem_local_temperatures")
    #   - Ops should declare extras keys in reads/writes metadata
    extras: Dict[str, Any] = field(default_factory=dict)


class TraceSink(Protocol):
    """Protocol for trace and progress sinks.

    Implementations receive position snapshots, log messages, and
    operation boundary events. The ``op_start`` / ``op_end`` pair
    fires at each ``Pipeline`` step boundary when ``trace_between``
    is enabled, providing natural "video frame" points.
    """

    def snapshot(self, pos: torch.Tensor, step: int) -> None:
        """Record a position snapshot.

        Parameters
        ----------
        pos : torch.Tensor
            Position tensor with shape ``[N, 2]``.
        step : int
            Optimization step for the snapshot.

        Returns
        -------
        None
            This method records side effects only.
        """

    def log(self, message: str) -> None:
        """Record a progress message.

        Parameters
        ----------
        message : str
            Human-readable progress or diagnostic message.

        Returns
        -------
        None
            This method records side effects only.
        """

    def op_start(self, op_name: str, step: int) -> None:
        """Called before an operation runs inside a Pipeline.

        Parameters
        ----------
        op_name : str
            Name of the operation about to execute.
        step : int
            Current optimization step counter.

        Returns
        -------
        None
            This method records side effects only.
        """

    def op_end(self, op_name: str, step: int) -> None:
        """Called after an operation completes inside a Pipeline.

        Parameters
        ----------
        op_name : str
            Name of the operation that just completed.
        step : int
            Current optimization step counter.

        Returns
        -------
        None
            This method records side effects only.
        """


@dataclass
class ExecutionPlan:
    """Resolved execution strategy.

    The plan separates execution axes such as device placement, optimizer
    choice, and streaming policy instead of collapsing everything into a
    single mode enum.

    Attributes
    ----------
    device : str
        Primary compute device.
    edge_device : str
        Edge tensor residency, which may differ from ``device``.
    mode : str
        Execution mode such as ``standard`` or ``tiled_gpu``.
    optimizer_type : str
        Optimizer name such as ``adam`` or ``sgd``.
    per_loss_backward : bool
        Whether each loss term backpropagates independently.
    gradient_checkpointing : bool
        Whether gradient checkpointing is enabled.
    hybrid_cpu_gpu : bool
        Whether losses are split across CPU and GPU.
    edges_on_cpu : bool
        Whether edges are streamed from CPU memory.
    edge_batch_size : int
        Edge batch size for streaming.
    sampled_node_cap : int
        Maximum sampled nodes per step.
    vram_budget_mb : float
        Available VRAM budget in megabytes.
    ram_budget_gb : float
        Available RAM budget in gigabytes.
    worker_threads : int
        Thread count for hybrid execution.
    """

    device: str = "cpu"
    edge_device: str = "cpu"
    mode: str = "standard"
    optimizer_type: str = "adam"
    per_loss_backward: bool = False
    gradient_checkpointing: bool = False
    hybrid_cpu_gpu: bool = False
    edges_on_cpu: bool = False
    edge_batch_size: int = 0
    sampled_node_cap: int = 0
    vram_budget_mb: float = 0.0
    ram_budget_gb: float = 0.0
    worker_threads: int = 0
    subset_gpu_threshold: int = 10_000_000


@dataclass
class MemoryPolicy:
    """Memory management policy for large-scale runs.

    Attributes
    ----------
    offload_dir : Path | None
        Directory for serialized hierarchy levels.
    ram_threshold_gb : float
        Spill to disk when RSS exceeds this threshold.
    gc_between_levels : bool
        Whether to run garbage collection between V-cycle levels.
    offload_original_graph : bool
        Whether to offload the original graph during a V-cycle.
    original_graph_path : Path | None
        Serialized original-graph payload path.
    checkpoint_dir : Path | None
        Directory for checkpoint files.
    """

    offload_dir: Optional[Path] = None
    ram_threshold_gb: float = 100.0
    gc_between_levels: bool = True
    offload_original_graph: bool = False
    original_graph_path: Optional[Path] = None
    checkpoint_dir: Optional[Path] = None


class NullTraceSink:
    """Default trace sink that discards snapshots, logs, and op events."""

    def snapshot(self, pos: torch.Tensor, step: int) -> None:
        """Discard a position snapshot.

        Parameters
        ----------
        pos : torch.Tensor
            Position tensor with shape ``[N, 2]``.
        step : int
            Optimization step for the snapshot.

        Returns
        -------
        None
            This sink intentionally performs no work.
        """
        return None

    def log(self, message: str) -> None:
        """Discard a progress message.

        Parameters
        ----------
        message : str
            Human-readable progress or diagnostic message.

        Returns
        -------
        None
            This sink intentionally performs no work.
        """
        return None

    def op_start(self, op_name: str, step: int) -> None:
        """Discard an operation start event.

        Parameters
        ----------
        op_name : str
            Name of the operation about to execute.
        step : int
            Current optimization step counter.

        Returns
        -------
        None
            This sink intentionally performs no work.
        """
        return None

    def op_end(self, op_name: str, step: int) -> None:
        """Discard an operation end event.

        Parameters
        ----------
        op_name : str
            Name of the operation that just completed.
        step : int
            Current optimization step counter.

        Returns
        -------
        None
            This sink intentionally performs no work.
        """
        return None


@dataclass
class RuntimeContext:
    """Execution infrastructure for layout operations.

    The runtime context describes how operations run: device placement,
    memory policy, trace sinks, checkpoint locations, and RNG state.
    It remains orthogonal to the actual layout problem and mutable solve state.

    Attributes
    ----------
    plan : ExecutionPlan
        Resolved execution strategy.
    memory : MemoryPolicy
        Memory management policy.
    trace_sink : TraceSink
        Callback sink for snapshots and progress logs.
    progress_file : Path | None
        External progress file such as ``progress.json``.
    generator : torch.Generator | None
        RNG generator for reproducible sampling.
    log_prefix : str
        Prefix applied to log messages.
    """

    plan: ExecutionPlan = field(default_factory=ExecutionPlan)
    memory: MemoryPolicy = field(default_factory=MemoryPolicy)
    trace_sink: TraceSink = field(default_factory=NullTraceSink)
    progress_file: Optional[Path] = None
    generator: Optional[torch.Generator] = None
    log_prefix: str = "[dagua]"
