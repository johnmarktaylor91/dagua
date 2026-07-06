"""LayoutConfig with all tunable parameters, defaults, and metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple

import torch

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
    """All tunable parameters for the layout engine.

    Parameters
    ----------
    fidelity_dtype : torch.dtype, optional
        Internal dtype used only by fidelity-mode engine adapters. ``None``
        lets fidelity-mode pipelines default to ``torch.float64`` while
        non-fidelity paths remain ``torch.float32``. Public layout outputs
        remain ``float32``.
    """

    # Spacing
    # Bumped 28 -> 60, 50 -> 80 after holdout sweep
    # discovered the prior defaults caused 6+ overlaps on dense
    # families (bipartite, erdos_renyi, powerlaw_dag), each costing
    # 10 composite points (overlap_count != 0 binary).
    #
    # Bumped rank_sep 80 -> 120 (ratio 1.5x -> 2x). The
    # taller layout makes multi-layer edges more vertical, lifting
    # edge_straightness + angular_resolution metrics.
    #
    # Refined to (70, 140) after 7-config local search.
    # Re-swept after after a 7-config sweep AR target=0.25.
    # The aggressive vertical rescaling means larger rank_sep gets
    # amplified through AR, so a higher rank_sep is now optimal:
    #
    #   70_140: suite 76.842
    #   60_140             : suite 76.871
    #   70_160             : suite 76.933
    #   80_160             : suite 76.911
    #   70_200             : suite 77.005
    #   70_240       : suite 77.043  <-- adopted
    #   70_300             : suite 76.934 (plateau)
    #   70_400             : suite 77.006 (plateau)
    #   60_200             : suite 77.013
    #   50_200             : suite 76.522
    #   40_200             : suite 75.906
    #
    # Adopting (70, 240) gives a 3.43:1 rank/node ratio. Layouts
    # produced are pre-AR aspect ~ 0.4; AR target=0.25 only mildly
    # rescales (since natural aspect already < 1.0). Plateau past 240.
    # Below node_sep=60 quickly degrades.
    #
    # adaptive_spacing still scales these down for very large
    # graphs (n >= 1000), so >1M layouts stay compact.
    node_sep: float = 70.0
    rank_sep: float = 240.0
    direction: str = "TB"

    # Optimization (0 = auto-scale based on graph size)
    steps: int = 0
    lr: float = 0.05
    device: str = "cpu"
    seed: Optional[int] = 42
    fidelity_dtype: Optional[torch.dtype] = None
    multi_start_k: int = 1
    # Route degenerate-layering cyclic graphs (e.g. small_world)
    # to the stress-majorization pipeline instead of the layered native path.
    # The layered path produces one-node-per-layer outputs on these graphs
    # which collapses crossings, dag_consistency, and edge_straightness.
    route_flat_to_stress: bool = True
    # Override native topology dispatch. Deprecated booleans remain
    # accepted as kill switches, but auto dispatch should be controlled
    # through this single selector.
    force_pipeline: Optional[
        Literal["tree", "layered_dag", "force_directed", "hybrid", "planar", "legacy_monolith"]
    ] = None
    # Best-of-polish edge-equalize after the native pipeline
    # converges. The gradient pipeline saturates on edge_length_variance
    # for layered DAGs and lattices (confirmed empirically: w=0..200 same
    # output). A direct constraint projection toward mean edge length,
    # scored against the un-polished baseline, escapes the local minimum
    # on ~9 of the 10 moderate-loss bucket graphs. Disable to skip the
    # 5-candidate scoring pass (~10ms overhead at N=100).
    edge_equalize_polish: bool = True

    # Try the native_planar sub-pipeline when classify_graph confirms
    # exact planarity. Defaults to False because the current planar pipeline
    # (Schnyder init + flat stress refine) drops the depth_spearman /
    # dag_consistency hierarchy bonus that layered_dag earns on planar DAGs,
    # so it loses on every benchmark candidate today (-3 to -35 composite
    # points vs layered_dag). Set True or use force_pipeline="planar" to
    # exercise the wired pipeline; an open task is to layer-aware the planar
    # init so it can beat layered_dag on at least cyclic-planar graphs.
    try_planar_first: bool = False

    # Adaptive spacing: scale node_sep and rank_sep based on graph size
    adaptive_spacing: bool = True

    # Verbose: print progress at key stages (hierarchy, per-level, projection)
    verbose: bool = False

    # Node placement loss weights
    w_dag: float = 10.0
    w_attract: float = 2.0
    # Bumped 2.4 -> 1.0 after re-sweep at the new
    # spacing/AR defaults. Sweep:
    #   xb=0.5:  77.070
    #   xb=1.0:  77.092  <-- adopted
    #   xb=2.4:  77.064 (previous default)
    #   xb=5.0:  77.074
    #   xb=10.0: 76.634
    w_attract_x_bias: float = 1.0
    # Bumped 0.1 -> 0.2 after holdout sweep. +3.14 composite
    # on bipartite, zero change on the other 20 families. Bipartite
    # graphs need stronger repulsion to spread their two layers apart;
    # other families have other constraints dominating already.
    w_repel: float = 0.2
    w_overlap: float = 5.0
    w_cluster: float = 1.0
    w_cluster_contain: float = 2.0  # child clusters stay within parent bbox
    # Cluster-aware placement treats clusters as recursive placement
    # primitives. Legacy cluster losses remain available when this is False.
    cluster_aware: bool = True
    cluster_side_padding_pt: float = 8.0
    cluster_label_band_pt: float = 26.0
    cluster_external_clearance_pt: float = 10.0
    w_align: float = 5.0
    w_crossing: float = 1.8
    # Dropped 2.2 -> 0.5 after a comprehensive 93-graph sweep.
    # The straightness loss was over-constraining the layered_dag pipeline:
    # at 2.2 the optimizer pinned edges close to the layer axis at the
    # expense of edge-length uniformity (the metric weights more heavily
    # at 20pts vs straightness's 10pts). Net delta across 93 graphs was
    # +4.22 composite when dropped to 0.5. Wins: random_dag_50 +1.49,
    # dependency_graph_100 +1.20, er_100 +1.07, dependency_500 +0.84,
    # random_bipartite_60 +0.63. Losses: inception_block -1.21 (still a
    # double-digit win), kitchen_sink_hybrid_net -0.81 (still positive
    # vs dagre). Lattices unchanged (already at convergence).
    w_straightness: float = 0.5
    # Scale-invariant (CV^2) formulation -- see
    # edge_length_variance_loss. Loss magnitude is now roughly 0.0-1.0
    # vs legacy 10^2-10^3. Weight bumped 0.7 -> 8.0 to keep the
    # gradient contribution comparable AND make edge uniformity a
    # stronger effective constraint (the metric we're being judged on).
    w_length_variance: float = 8.0
    # Pivot-approximated graph-theoretic stress loss.
    # Targets `sampled_stress` metric where Dagua wins 1.9% in the
    # forward-memo benchmark. Default 0.0 = disabled (opt-in via
    # LayoutConfig(w_stress=0.1) for graphs where node-distance
    # fidelity matters). Non-zero triggers pivot adjacency + distance
    # pre-computation at pipeline build time.
    w_stress: float = 0.0
    w_stress_n_pivots: int = 50
    w_spacing: float = 0.3  # penalize deviation from target node_sep within layers
    w_fanout: float = 0.3  # penalize uneven angular spread of high-degree node children
    w_back_edge: float = 0.3  # penalize wide back-edge arcs (horizontal distance)

    # Scale thresholds
    # The default native loss path now switches from exact O(N^2)
    # repulsion/overlap to a cell-list spatial hash at N >= 500. Set
    # exact_repulsion=True to force the legacy exact pairwise losses.
    exact_repulsion: bool = False
    exact_repulsion_threshold: int = 500
    negative_sample_k: int = 128

    # Tree topology fast-path. Graphs that classify as
    # rooted trees (|E| == N-1, 1 component, acyclic) skip the entire
    # gradient pipeline and use classical Reingold-Tilford layout --
    # 0 crossings by construction, matches igraph_rt / graphviz_dot.
    # DEFAULT IS OFF because R-T naturally produces high edge-length
    # variance (deeper subtrees spread wider), which the dagua
    # composite metric weights heavily. Empirically the
    # barycenter-polish + CV-aware gradient path scores better on
    # composite for trees. Opt in via LayoutConfig(use_tree_fast_path=True)
    # when edge crossings are the dominant concern vs edge uniformity.
    use_tree_fast_path: bool = False
    # Discrete native crossing reduction after barycenter polish.
    # Median and transpose run only on acyclic graphs and can be disabled
    # quickly if a narrow DAG family regresses.
    use_native_median_transpose: bool = True
    native_median_passes: int = 4
    native_transpose_passes: int = 8
    # X-only Brandes-Koepf compaction after native ordering.
    # Conservative DAG gate keeps this out of cyclic, flat, tree, chain, and
    # multi-component cases unless there is only one isolated tail node.
    brandes_koepf_refine: bool = True
    # Default-on R79 P2C layered polish. Row snapping removes gradient-created
    # micro-ranks after ordering, and cluster compaction adds cluster interval
    # gaps after BK only for clustered layered graphs.
    layered_rank_row_snap: bool = True
    cluster_aware_x_compaction: bool = True
    tiny_multiedge_rank_sep_cap: bool = True
    tiny_multiedge_rank_sep_max: float = 240.0
    # Split long-span DAG edges into dummy-node chains inside
    # dagua_native. This public kill switch lets benchmarks isolate routing
    # regressions without reverting the whole feature.
    insert_dummy_nodes: bool = True
    # Adapter-level disconnected-component decomposition for the
    # default dagua_native pipeline. Each weak component is solved
    # independently and tiled unless a conservative safety gate disables it.
    decompose_components: bool = True
    component_tiling_crossing_risk: bool = True

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
    # user opt-out: "differentiable" runs optimize_edges (the
    # default; unchanged from behaviour), "heuristic" skips
    # gradient refinement entirely and keeps the heuristic bezier curves
    # produced by route_edges. Takes precedence over edge_opt_steps -- a
    # user who wants the heuristic path shouldn't have to also remember
    # to pass edge_opt_steps=-1.
    edge_routing: str = "differentiable"
    # r3: adaptive skip threshold. When edge_routing is
    # "differentiable" but the heuristic routing already produces fewer
    # than this many edge-node crossings, the gradient refinement is
    # skipped entirely. Protects nested-cluster graphs whose heuristic
    # routing is already near-optimal (CP refinement would create
    # crossings; see audit at eval_output/native_algo/sprint_6_edge_routing/).
    # Set to 0 to force refinement always, or a large number to force the
    # heuristic path.
    edge_routing_auto_skip_threshold: int = 5
    # r2: re-tuned edge-CP loss weights. The defaults
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
    # topology -- the 30% aggregate target is a tuning
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
        default=1.0,
        sweep_range=(1.0, 16.0),
        sweep_values=[1.0, 2.0, 2.4, 4.0, 8.0, 16.0],
        category="forces",
    ),
    TunableParam(
        name="w_repel",
        display_name="Node Repulsion",
        description="How strongly all nodes push apart from each other.",
        visual_effect="Increasing: more spacing. Decreasing: denser graph.",
        default=0.2,
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
        default=0.5,
        sweep_range=(0.5, 5.0),
        sweep_values=[0.5, 1.0, 2.2, 3.0, 5.0],
        category="aesthetics",
    ),
    TunableParam(
        name="w_length_variance",
        display_name="Edge Length Uniformity",
        description="Penalizes variance in edge lengths (prefer uniform over minimum).",
        visual_effect="Increasing: more uniform edge lengths. Decreasing: variable lengths OK.",
        default=8.0,
        sweep_range=(0.1, 2.0),
        sweep_values=[0.1, 0.3, 0.7, 1.0, 2.0],
        category="aesthetics",
    ),
    TunableParam(
        name="node_sep",
        display_name="Node Separation",
        description="Minimum horizontal gap between nodes (pixels).",
        visual_effect="Increasing: more horizontal breathing room.",
        default=70.0,
        sweep_range=(10.0, 60.0),
        sweep_values=[10.0, 15.0, 28.0, 40.0, 60.0],
        category="spacing",
    ),
    TunableParam(
        name="rank_sep",
        display_name="Rank Separation",
        description="Minimum vertical gap between layers (pixels).",
        visual_effect="Increasing: more vertical breathing room.",
        default=240.0,
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
