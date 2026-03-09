"""LayoutConfig with all tunable parameters, defaults, and metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


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
    rank_sep: float = 50.0
    direction: str = "TB"

    # Optimization
    steps: int = 500
    lr: float = 0.05
    device: str = "cpu"
    seed: Optional[int] = 42

    # Node placement loss weights
    w_dag: float = 10.0
    w_attract: float = 2.0
    w_attract_x_bias: float = 4.0
    w_repel: float = 0.1
    w_overlap: float = 5.0
    w_cluster: float = 1.0
    w_align: float = 5.0
    w_crossing: float = 0.5
    w_straightness: float = 1.0
    w_length_variance: float = 0.5

    # Edge routing
    edge_routing: str = "bezier"

    # Scale thresholds
    exact_repulsion_threshold: int = 5000
    negative_sample_k: int = 128


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
        default=4.0,
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
        default=0.5,
        sweep_range=(0.1, 2.0),
        sweep_values=[0.1, 0.3, 0.5, 1.0, 2.0],
        category="aesthetics",
    ),
    TunableParam(
        name="w_straightness",
        display_name="Edge Straightness",
        description="Penalizes horizontal displacement between connected nodes.",
        visual_effect="Increasing: straighter vertical edges. Decreasing: more flexible.",
        default=1.0,
        sweep_range=(0.1, 5.0),
        sweep_values=[0.1, 0.5, 1.0, 2.0, 5.0],
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
        default=50.0,
        sweep_range=(25.0, 100.0),
        sweep_values=[25.0, 35.0, 50.0, 75.0, 100.0],
        category="spacing",
    ),
    TunableParam(
        name="steps",
        display_name="Optimization Steps",
        description="Number of gradient descent steps.",
        visual_effect="Increasing: better quality, slower. Decreasing: faster, rougher.",
        default=500,
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
]

PARAM_REGISTRY_DICT: Dict[str, TunableParam] = {p.name: p for p in PARAM_REGISTRY}
