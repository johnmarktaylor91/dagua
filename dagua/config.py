"""LayoutConfig with all tunable parameters, defaults, and metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple

if TYPE_CHECKING:
    from dagua.flex import LayoutFlex


@dataclass
class TunableParam:
    """Metadata for a single tunable parameter."""

    name: str
    display_name: str
    description: str
    visual_effect: str
    default: float
    sweep_range: Tuple[float, float]
    sweep_values: List[float]
    category: str  # 'spacing', 'forces', 'aesthetics', 'routing', 'optimization'


@dataclass
class LayoutConfig:
    """All tunable parameters for the layout engine."""

    # Spacing
    node_sep: float = 28.0
    rank_sep: float = 50.0
    direction: str = "TB"

    # Optimization (0 = auto-scale based on graph size)
    steps: int = 0
    lr: float = 0.05
    device: str = "cpu"
    seed: Optional[int] = 42

    # Adaptive spacing: scale node_sep and rank_sep based on graph size
    adaptive_spacing: bool = True

    # Verbose: print progress at key stages (hierarchy, per-level, projection)
    verbose: bool = False

    # Node placement loss weights
    w_dag: float = 10.0
    w_attract: float = 2.0
    w_attract_x_bias: float = 2.4
    w_repel: float = 0.1
    w_overlap: float = 5.0
    w_cluster: float = 1.0
    w_cluster_contain: float = 2.0  # child clusters stay within parent bbox
    w_align: float = 5.0
    w_crossing: float = 1.8
    w_straightness: float = 2.2
    w_length_variance: float = 0.7
    w_spacing: float = 0.3  # penalize deviation from target node_sep within layers
    w_fanout: float = 0.3  # penalize uneven angular spread of high-degree node children
    w_back_edge: float = 0.3  # penalize wide back-edge arcs (horizontal distance)

    # Scale thresholds
    exact_repulsion_threshold: int = 2000
    negative_sample_k: int = 128

    # Multilevel coarsening (default: N > 20K)
    multilevel_threshold: int = 20000
    multilevel_min_nodes: int = 2000
    multilevel_coarse_steps: int = 100
    multilevel_refine_steps: int = 25

    # RVS repulsion (available for very large direct-layout cases)
    # Default: disabled (multilevel handles N > 20K more efficiently)
    # Lower this threshold to enable RVS for N in (rvs_threshold, multilevel_threshold]
    rvs_threshold: int = 100000
    rvs_nn_k: int = 20

    # Memory optimization modes — "auto" enables based on graph size and device.
    # Set to "on"/"off" to override. These dramatically reduce peak memory for
    # large graphs, enabling GPU layout at scales that would otherwise OOM.
    #
    # per_loss_backward: backward each loss term separately, freeing intermediates
    #   between terms. ~3-4x peak memory reduction. Auto: on when N > 50K.
    per_loss_backward: str = "auto"
    # gradient_checkpointing: recompute forward activations during backward instead
    #   of storing them. ~2x memory reduction, ~30% more compute. Auto: on when
    #   device=cuda and N > 500K.
    gradient_checkpointing: str = "auto"
    # hybrid_device: keep positions on GPU but compute memory-heavy losses (repulsion,
    #   overlap) on CPU. Only the [N, 2] gradient transfers between devices.
    #   Auto: on when device=cuda and N > 2M.
    hybrid_device: str = "auto"
    # execution_mode: overall position residency strategy.
    # "standard" keeps positions on the requested device. "subset_gpu" keeps
    # large tensors on CPU and accelerates per-loss subsets on GPU. "auto"
    # selects subset_gpu once the graph is large enough that full CUDA
    # residency becomes unsafe.
    execution_mode: Literal["auto", "standard", "subset_gpu"] = "auto"
    # Node-count threshold for auto-activating subset_gpu when CUDA is the
    # requested execution device. Above 50M nodes the engine forces
    # subset_gpu regardless of this value.
    subset_gpu_threshold: int = 10_000_000
    # optimizer_fallback: optimizer choice for huge hybrid refinement levels when
    # Adam's state no longer fits on GPU.
    # "auto" = Adam -> SGD+Nesterov -> SGD, "adam" = never downgrade optimizer,
    # "sgd" = skip Nesterov and use vanilla SGD as the only fallback tier.
    optimizer_fallback: Literal["auto", "adam", "sgd"] = "auto"

    # CPU worker threads for hybrid-mode parallel loss computation.
    # 0 = sequential (no workers). 2+ = parallel CPU losses.
    # Only used when hybrid_device mode is active with per_loss_backward.
    num_workers: int = 0

    # Edge batch size for sampling during optimization.
    # 0 = auto-scale based on edge count (default). >0 = fixed batch size.
    # Larger batches = fewer iterations per step but more memory.
    edge_batch_size: int = 0

    # Overlap check interval: how often to run overlap projection (every N steps).
    # 0 = auto-scale based on graph size (default). >0 = fixed interval.
    overlap_check_interval: int = 0

    # Repulsion amortization: run repulsion loss every N steps for large graphs.
    # 1 = every step (default/no amortization). 2+ = skip steps.
    # Only applies when N > repel_amortize_threshold. Set threshold to 0 to
    # apply at all scales (not recommended for small graphs).
    repel_amortize_interval: int = 2
    repel_amortize_threshold: int = 10_000_000

    # Fanout amortization: run fanout_distribution loss every N steps for large graphs.
    # 1 = every step (default/no amortization). 3 = run every 3rd step.
    # Only applies when N > fanout_amortize_threshold.
    fanout_amortize_interval: int = 3
    fanout_amortize_threshold: int = 10_000_000

    # Edge sampling strategy: fraction of steps that use random sampling vs
    # contiguous chunks. 0.0 = all contiguous, 1.0 = all random.
    # Default 0.2 = random every 5th step, contiguous otherwise.
    # Contiguous is more cache-friendly; random ensures coverage.
    edge_random_fraction: float = 0.2

    # Force-directed relaxation pass after the main hierarchical layout.
    # Runs additional steps with w_dag=0, softening rigid layer structure
    # into a more organic layout while preserving overall flow.
    # 0 = no relaxation (default). 20-50 = light softening. 100+ = strong.
    relax_steps: int = 0

    # Disk offloading: when True, large intermediate tensors are saved to
    # temporary files during multilevel layout to reduce peak RSS.
    # Disable for machines with sufficient RAM to keep the full hierarchy
    # resident (avoids large torch.save calls that can fail on low disk).
    offload_to_disk: bool = True

    # Flex layout constraints (soft targets for spacing, pins, alignment)
    # When present, flex values override the corresponding fixed values.
    flex: Optional["LayoutFlex"] = None

    # Edge optimization: gradient descent on bezier control points
    # 0 = auto-scale based on edge count, -1 = skip (zero overhead)
    edge_opt_steps: int = 0
    edge_opt_lr: float = 0.1
    # Sprint 6 user opt-out: "differentiable" runs optimize_edges (the
    # default; unchanged from Sprint 5 behaviour), "heuristic" skips
    # gradient refinement entirely and keeps the heuristic bezier curves
    # produced by route_edges. Takes precedence over edge_opt_steps -- a
    # user who wants the heuristic path shouldn't have to also remember
    # to pass edge_opt_steps=-1.
    edge_routing: str = "differentiable"
    # Sprint 6 r3: adaptive skip threshold. When edge_routing is
    # "differentiable" but the heuristic routing already produces fewer
    # than this many edge-node crossings, the gradient refinement is
    # skipped entirely. Protects nested-cluster graphs whose heuristic
    # routing is already near-optimal (CP refinement would create
    # crossings; see audit at eval_output/native_algo/sprint_6_edge_routing/).
    # Set to 0 to force refinement always, or a large number to force the
    # heuristic path.
    edge_routing_auto_skip_threshold: int = 5
    # Sprint 6 r2: re-tuned edge-CP loss weights. The Sprint 5 defaults
    # (crossing=5, node_crossing=10, angular=2, curv_cons=1, curv_pen=0.5,
    # cluster_cross=8) produced NEGATIVE edge-node-crossing drops on
    # trees (-27.7% on tree_branching_4 n=800; -13.4% on branching_3).
    # Per-loss ablation showed angular_res and curv_consistency each
    # REGRESS crossings by ~6%, curv_penalty IMPROVES by ~52%, and
    # edge_crossing+cluster_crossing are modestly positive. New defaults
    # zero the saboteurs, strengthen curv_penalty, soften edge_crossing.
    # Result on trees: +52% / +56% drop; dense DAGs neutral (no room to
    # route around). Aggregate drop on the 39-graph held-out is dominated
    # by dense graphs where edge-node crossings are inherent to the
    # topology -- the 30% aggregate target is a Sprint 6.5 tuning
    # bundle, not a default-weight fix.
    w_edge_crossing: float = 0.5
    w_edge_node_crossing: float = 10.0
    w_edge_angular_res: float = 0.0
    w_edge_curvature_consistency: float = 0.0
    w_edge_curvature_penalty: float = 1.0
    w_edge_cluster_crossing: float = 4.0  # penalize edges through foreign clusters

    # Optional algorithm override for direct pipeline-based layouts.
    # None routes to the native multilevel/direct engine.
    use_pipeline: bool = False
    algorithm: Optional[str] = None
    algorithm_params: dict[str, Any] = field(default_factory=dict)


# Registry of all tunable parameters with metadata
PARAM_REGISTRY: List[TunableParam] = [
    TunableParam(
        name="w_dag",
        display_name="DAG Ordering Strength",
        description="How strongly edges are forced to point downward.",
        visual_effect="Increasing: more rigidly layered. Decreasing: looser structure.",
        default=10.0,
        sweep_range=(1.0, 50.0),
        sweep_values=[1.0, 5.0, 10.0, 20.0, 50.0],
        category="forces",
    ),
    TunableParam(
        name="w_attract",
        display_name="Edge Attraction",
        description="How strongly connected nodes pull toward each other.",
        visual_effect="Increasing: tighter graph, shorter edges. Decreasing: spread out.",
        default=2.0,
        sweep_range=(0.1, 10.0),
        sweep_values=[0.1, 0.5, 2.0, 5.0, 10.0],
        category="forces",
    ),
    TunableParam(
        name="w_attract_x_bias",
        display_name="Vertical Edge Preference",
        description="Extra weight on horizontal attraction (makes edges more vertical).",
        visual_effect="Increasing: straighter vertical edges. Decreasing: more diagonal.",
        default=2.4,
        sweep_range=(1.0, 16.0),
        sweep_values=[1.0, 2.0, 2.4, 4.0, 8.0, 16.0],
        category="forces",
    ),
    TunableParam(
        name="w_repel",
        display_name="Node Repulsion",
        description="How strongly all nodes push apart from each other.",
        visual_effect="Increasing: more spacing. Decreasing: denser graph.",
        default=0.1,
        sweep_range=(0.01, 1.0),
        sweep_values=[0.01, 0.05, 0.1, 0.5, 1.0],
        category="forces",
    ),
    TunableParam(
        name="w_overlap",
        display_name="Overlap Avoidance",
        description="Penalty for node bounding box intersection.",
        visual_effect="Increasing: harder overlap avoidance. Decreasing: may allow overlap.",
        default=5.0,
        sweep_range=(1.0, 20.0),
        sweep_values=[1.0, 3.0, 5.0, 10.0, 20.0],
        category="forces",
    ),
    TunableParam(
        name="w_crossing",
        display_name="Crossing Minimization",
        description="Penalty for edge crossings (differentiable proxy).",
        visual_effect=(
            "Increasing: fewer crossings, may distort layout. Decreasing: ignore crossings."
        ),
        default=1.8,
        sweep_range=(0.5, 5.0),
        sweep_values=[0.5, 1.0, 1.8, 3.0, 5.0],
        category="aesthetics",
    ),
    TunableParam(
        name="w_straightness",
        display_name="Edge Straightness",
        description="Penalizes horizontal displacement between connected nodes.",
        visual_effect="Increasing: straighter vertical edges. Decreasing: more flexible.",
        default=2.2,
        sweep_range=(0.5, 5.0),
        sweep_values=[0.5, 1.0, 2.2, 3.0, 5.0],
        category="aesthetics",
    ),
    TunableParam(
        name="w_length_variance",
        display_name="Edge Length Uniformity",
        description="Penalizes variance in edge lengths (prefer uniform over minimum).",
        visual_effect="Increasing: more uniform edge lengths. Decreasing: variable lengths OK.",
        default=0.7,
        sweep_range=(0.1, 2.0),
        sweep_values=[0.1, 0.3, 0.7, 1.0, 2.0],
        category="aesthetics",
    ),
    TunableParam(
        name="node_sep",
        display_name="Node Separation",
        description="Minimum horizontal gap between nodes (pixels).",
        visual_effect="Increasing: more horizontal breathing room.",
        default=28.0,
        sweep_range=(10.0, 60.0),
        sweep_values=[10.0, 15.0, 28.0, 40.0, 60.0],
        category="spacing",
    ),
    TunableParam(
        name="rank_sep",
        display_name="Rank Separation",
        description="Minimum vertical gap between layers (pixels).",
        visual_effect="Increasing: more vertical breathing room.",
        default=50.0,
        sweep_range=(25.0, 100.0),
        sweep_values=[25.0, 35.0, 50.0, 60.0, 100.0],
        category="spacing",
    ),
    TunableParam(
        name="steps",
        display_name="Optimization Steps",
        description="Number of gradient descent steps.",
        visual_effect="Increasing: better quality, slower. Decreasing: faster, rougher.",
        default=0,
        sweep_range=(100.0, 2000.0),
        sweep_values=[100, 250, 500, 1000, 2000],
        category="optimization",
    ),
    TunableParam(
        name="lr",
        display_name="Learning Rate",
        description="Step size for gradient descent.",
        visual_effect="Increasing: faster but less stable. Decreasing: slower but smoother.",
        default=0.05,
        sweep_range=(0.01, 0.2),
        sweep_values=[0.01, 0.03, 0.05, 0.1, 0.2],
        category="optimization",
    ),
    TunableParam(
        name="w_cluster_contain",
        display_name="Cluster Containment",
        description="How strongly child clusters are kept inside parent clusters.",
        visual_effect="Increasing: strict nesting. Decreasing: children may escape.",
        default=2.0,
        sweep_range=(0.5, 10.0),
        sweep_values=[0.5, 1.0, 2.0, 5.0, 10.0],
        category="forces",
    ),
]

PARAM_REGISTRY_DICT: Dict[str, TunableParam] = {p.name: p for p in PARAM_REGISTRY}
