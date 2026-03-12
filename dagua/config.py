"""LayoutConfig with all tunable parameters, defaults, and metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

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
    node_sep: float = 25.0
    rank_sep: float = 45.0
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
    w_attract_x_bias: float = 2.0
    w_repel: float = 0.1
    w_overlap: float = 5.0
    w_cluster: float = 1.0
    w_cluster_contain: float = 2.0  # child clusters stay within parent bbox
    w_align: float = 5.0
    w_crossing: float = 1.5
    w_straightness: float = 2.0
    w_length_variance: float = 0.5
    w_spacing: float = 0.3  # penalize deviation from target node_sep within layers
    w_fanout: float = 0.3  # penalize uneven angular spread of high-degree node children
    w_back_edge: float = 0.3  # penalize wide back-edge arcs (horizontal distance)

    # Scale thresholds
    exact_repulsion_threshold: int = 2000
    negative_sample_k: int = 128

    # Multilevel coarsening (Tier 3: N > multilevel_threshold)
    multilevel_threshold: int = 50000
    multilevel_min_nodes: int = 2000
    multilevel_coarse_steps: int = 100
    multilevel_refine_steps: int = 25

    # RVS repulsion (available for very large direct-layout cases)
    # Default: disabled (multilevel handles N > 50K more efficiently)
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

    # CPU worker threads for hybrid-mode parallel loss computation.
    # 0 = sequential (no workers). 2+ = parallel CPU losses.
    # Only used when hybrid_device mode is active with per_loss_backward.
    num_workers: int = 0

    # Flex layout constraints (soft targets for spacing, pins, alignment)
    # When present, flex values override the corresponding fixed values.
    flex: Optional["LayoutFlex"] = None

    # Edge optimization: gradient descent on bezier control points
    # 0 = auto-scale based on edge count, -1 = skip (zero overhead)
    edge_opt_steps: int = 0
    edge_opt_lr: float = 0.1
    w_edge_crossing: float = 5.0
    w_edge_node_crossing: float = 10.0
    w_edge_angular_res: float = 2.0
    w_edge_curvature_consistency: float = 1.0
    w_edge_curvature_penalty: float = 0.5
    w_edge_cluster_crossing: float = 8.0  # penalize edges through foreign clusters


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
        default=2.0,
        sweep_range=(1.0, 16.0),
        sweep_values=[1.0, 2.0, 4.0, 8.0, 16.0],
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
        visual_effect="Increasing: fewer crossings, may distort layout. Decreasing: ignore crossings.",
        default=1.5,
        sweep_range=(0.5, 5.0),
        sweep_values=[0.5, 1.0, 1.5, 3.0, 5.0],
        category="aesthetics",
    ),
    TunableParam(
        name="w_straightness",
        display_name="Edge Straightness",
        description="Penalizes horizontal displacement between connected nodes.",
        visual_effect="Increasing: straighter vertical edges. Decreasing: more flexible.",
        default=2.5,
        sweep_range=(0.5, 5.0),
        sweep_values=[0.5, 1.0, 2.0, 3.0, 5.0],
        category="aesthetics",
    ),
    TunableParam(
        name="w_length_variance",
        display_name="Edge Length Uniformity",
        description="Penalizes variance in edge lengths (prefer uniform over minimum).",
        visual_effect="Increasing: more uniform edge lengths. Decreasing: variable lengths OK.",
        default=0.5,
        sweep_range=(0.1, 2.0),
        sweep_values=[0.1, 0.3, 0.5, 1.0, 2.0],
        category="aesthetics",
    ),
    TunableParam(
        name="node_sep",
        display_name="Node Separation",
        description="Minimum horizontal gap between nodes (pixels).",
        visual_effect="Increasing: more horizontal breathing room.",
        default=25.0,
        sweep_range=(10.0, 60.0),
        sweep_values=[10.0, 15.0, 25.0, 40.0, 60.0],
        category="spacing",
    ),
    TunableParam(
        name="rank_sep",
        display_name="Rank Separation",
        description="Minimum vertical gap between layers (pixels).",
        visual_effect="Increasing: more vertical breathing room.",
        default=45.0,
        sweep_range=(25.0, 100.0),
        sweep_values=[25.0, 35.0, 45.0, 60.0, 100.0],
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
