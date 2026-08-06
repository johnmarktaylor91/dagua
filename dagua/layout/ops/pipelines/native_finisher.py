"""Bounded W5 differentiable finisher for native layout candidates."""

from __future__ import annotations

import inspect
import json
import logging
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from dagua.config import LayoutConfig
from dagua.eval.ruler_v3 import (
    C4_CLEARANCE_BAND_NODE_DIAGONALS,
    WHITESPACE_CROWDING_DECAY,
    WHITESPACE_RATIO_HI,
    WHITESPACE_RATIO_LO,
    WHITESPACE_SPRAWL_DECAY,
    _structure_area_floor,
)
from dagua.eval.ruler_v3_groups import evaluate_conditional_groups
from dagua.layout.ops.cluster_geometry import (
    break_cluster_parent_cycles,
    build_cluster_geometry_profile,
)
from dagua.layout.ops.pipelines.native_budget import (
    DETERMINISTIC_BUDGET_ATTR,
    PROCESS_DEADLINE_ATTR,
    charge,
    remaining_dwu,
    remaining_process_s,
    remaining_wall_s,
    wall_reserve_exhausted,
)
from dagua.layout.ops.pipelines.native_cost_model import (
    estimate_native_work_cost,
    estimate_v3_referee_cost,
)
from dagua.layout.ops.pipelines.native_shape_geometry import (
    NativeShapeGeometry,
    pairwise_shape_signed_gap,
)
from dagua.layout.ops.pipelines.native_surrogates import (
    angular_resolution_loss,
    barrier_floor_loss,
    crossing_angle_loss,
    depth_order_score_surrogate,
    edge_length_cv_loss,
    gabriel_intrusion_loss,
    overlap_hinge_loss,
    path_continuity_loss,
    signed_flow_score_surrogate,
    soft_crossing_loss,
    soft_knn_neighborhood_loss,
)
from dagua.layout.projection import project_overlaps

DEGENERACY_CHAMPION_INELIGIBLE_FLAGS = frozenset(
    {"DEGENERATE_SCALE", "SPRAWL_COLLAPSE", "COINCIDENT_COLLAPSE"}
)
_LOGGER = logging.getLogger(__name__)
_ABSOLUTE_DEADLINE_RESERVE_S = 5.0
_MIN_BENCHMARK_REMAINING_S = 30.0
_FINISHER_SCORE_RESERVE_S = 2.0
_DEFAULT_FINISHER_SLICE_S = 4.0
_MIN_FINISHER_ENTRY_S = 1.0
_MAX_W5_SPEND_S = 20.0
_TOTAL_BUDGET_FRACTION = 0.10
_W5_ACCEPT_MARGIN = 0.05
_W5_LAYERED_READING_EPS = 0.01
_W5_LAYERED_SHAPE_EPS = 0.08
_W5_LAYERED_SHAPE_MIN_SCORE = 0.90
_PREDICTED_COST_LATE_ENTRY_REMAINING_S = 90.0
_PREDICTED_COST_RETURN_RESERVE_S = 2.0
_MEASURED_COST_MAX_SEEDS = 2
_MEASURED_COST_MAX_CHECKPOINTS = 2
_MEASURED_COST_TINY_MAX_CHECKPOINTS = 4
_MEASURED_COST_TINY_REFEREE_S = 0.08
_MEASURED_COST_TINY_MAX_N = 64
_MEASURED_COST_TINY_STEPS = 96
_TINY_ROW_CONTINUATION_CAP_S = 2.0
_TINY_ROW_DETERMINISTIC_STEP_COST_S = 0.0437
_TINY_ROW_DETERMINISTIC_REFEREE_COST_S = 0.019
_MEASURED_COST_DEFAULT_REFEREE_S = 1.20
_MEASURED_COST_SURROGATE_STEPS = 4
_W5_STRESS_MAX_SOURCES = 200
_W5_STRESS_MAX_PAIRS = 100_000
_W5_NEIGHBORHOOD_RADII = (2, 3)
_W5_NEIGHBORHOOD_MAX_TRIPLETS = 16_384
_W5_SOFT_BBOX_TAU_NODE_DIAG = 0.25
_W5_C5_SOFT_BBOX_WEIGHT = 8.0
_W5_C4_CLEARANCE_BAND_WEIGHT = 10.0
_W5_PASS1_STRESS_WEIGHT = 12.0
_W5_PASS2_STRESS_BASE_WEIGHT = 16.0
_W5_PASS2_STRESS_HEADROOM_WEIGHT = 48.0
_W5_PASS1_CONTRASTIVE_WEIGHT = 8.0
_W5_PASS2_CONTRASTIVE_WEIGHT = 12.0
_W5_CONTRASTIVE_MARGIN_NODE_DIAG = 0.45
_W5_SCALE_SEARCH_EVALS = 6
_W5_SCALE_SEARCH_LARGE_EVALS = 3
_W5_SCALE_SEARCH_LARGE_N = 300
_W5_SCALE_SEARCH_MIN = 0.50
_W5_SCALE_SEARCH_MAX = 2.40
_W5_TERMINAL_GLOBAL_SCALE_MULTIPLIERS = (0.85, 0.9, 0.95, 1.05, 1.1, 1.2, 1.35)
_W5_TERMINAL_ANISO_ASPECT_MIN = 6.0
_W5_TERMINAL_ANISO_STRONG_MULTIPLIERS = (
    (0.9, 2.0),
    (0.8, 4.0),
    (0.75, 6.0),
    (0.7, 8.0),
)
_W5_TERMINAL_ANISO_MILD_MIN_NODES = 128
_W5_TERMINAL_ANISO_MILD_ASPECT_MIN = 0.75
_W5_TERMINAL_ANISO_MILD_ASPECT_MAX = 1.50
_W5_TERMINAL_ANISO_MILD_RADIAL_CV_MAX = 0.16
_W5_TERMINAL_ANISO_MILD_INNER_RADIUS_FRACTION = 0.50
_W5_TERMINAL_ANISO_MILD_INNER_MASS_MAX = 0.02
_W5_TERMINAL_ANISO_MILD_BASE_SCALES = (0.85, 0.90, 0.95, 1.0)
_W5_TERMINAL_ANISO_MILD_MULTIPLIERS = (
    (0.90, 1.05),
    (0.90, 1.10),
    (0.90, 1.15),
    (0.92, 1.05),
    (0.92, 1.10),
    (0.92, 1.15),
    (0.94, 1.05),
    (0.94, 1.10),
    (0.94, 1.15),
    (0.96, 1.05),
    (0.96, 1.10),
    (0.96, 1.15),
    (0.98, 1.05),
    (0.98, 1.10),
    (0.98, 1.15),
)
_W5_TERMINAL_ANISO_MILD_MAX_PAIRS = 64
_W5_TERMINAL_SCALE_TIE_EPS = 1.0e-6
_W5_SMALL_N_ANNEAL_MAX_NODES = 50
_W5_SMALL_N_ANNEAL_TRIALS = 800
_W5_SMALL_N_ANNEAL_SEED = 42
_W5_SMALL_N_ANNEAL_SIGMA_HI_FRACTION = 0.35
_W5_SMALL_N_ANNEAL_SIGMA_LO_FRACTION = 0.015
_W5_SMALL_N_ANNEAL_SINGLE_NODE_PROBABILITY = 0.70
_W5_SMALL_N_ANNEAL_TIE_EPS = 1.0e-6
_CONTINUOUS_FACET_POLISH_TIE_EPS = 1.0e-6
_CONTINUOUS_FACET_POLISH_STEP_FRACTIONS = (0.04, 0.02, 0.008, 0.003)
_CONTINUOUS_FACET_POLISH_DIRECTIONS = (
    (1.0, 0.0),
    (-1.0, 0.0),
    (0.0, 1.0),
    (0.0, -1.0),
    (1.0, 1.0),
    (1.0, -1.0),
    (-1.0, 1.0),
    (-1.0, -1.0),
)
_DEEP_TREE_RANK_WARP_STRENGTHS = (1.25, 2.0, 2.75, 3.5)
_DEEP_TREE_RANK_WARP_POWER = 1.8
_DEEP_TREE_MAX_RANK_BAND_STD = 1.0e-4
_DEEP_TREE_MAX_CHILD_CENTROID_OFFSET_RATIO = 0.10
_W5_SMACOF_STRESS_MAX_NODES = 256
_W5_SMACOF_STRESS_MAX_EDGES = 1_024
_W5_SMACOF_STRESS_ITERATIONS = (20, 40, 60)
_W5_SMACOF_STRESS_OUTPUT_SCALES = (1.0, 0.9, 1.1)
_W5_SMACOF_STRESS_TIE_EPS = 1.0e-6
_W5_SMACOF_STRESS_MIN_DISTANCE = 1.0e-9
_DISABLE_W5_ENV = "DAGUA_NATIVE_DISABLE_W5"
_GRAPH_NAME_ATTR = "_dagua_native_graph_name"
_W5_PROJECTION_ITERATIONS = 20
_CLUSTER_TIGHTEN_MIN_SIBLING_GAP_FACTOR = 0.55
_CLUSTER_TIGHTEN_MILD_FACTORS = (0.99, 0.92)
_CLUSTER_TIGHTEN_STRONG_FACTORS = (0.40, 0.50, 0.65)
_CLUSTER_TIGHTEN_MAX_NODES = 2_000
_CLUSTER_BOX_ESCAPE_GUTTER_FACTORS = (0.35, 0.70)
_CLUSTER_BOX_ESCAPE_MAX_PASSES = 3
_CLUSTER_SEPARATE_PUSH_FACTORS = (1.15, 1.30)
_CLUSTER_SEPARATE_MAX_NODES = 500
_CLUSTER_SEPARATE_MAX_ROOT_CLUSTERS = 8


@dataclass(frozen=True)
class ClusterTighteningCandidate:
    """Deterministic terminal candidate for declared clustered graphs.

    Parameters
    ----------
    name : str
        Stable candidate label.
    pos : torch.Tensor
        Tightened positions with shape ``[N, 2]``.
    gate_reason : str
        Structural reason the candidate generator fired.
    cluster_count : int
        Number of non-empty declared clusters.
    max_depth : int
        Maximum declared nesting depth among valid clusters.
    """

    name: str
    pos: torch.Tensor
    gate_reason: str
    cluster_count: int
    max_depth: int


def _runtime_telemetry_payload(
    *,
    config: Optional[LayoutConfig],
    wall_s: Optional[float],
    process_s: Optional[float],
    use_deterministic_costs: bool,
) -> dict[str, Optional[float] | int | str | bool]:
    """Return environment metadata for native timing telemetry.

    Parameters
    ----------
    config : LayoutConfig, optional
        Prepared layout configuration carrying device and budget metadata.
    wall_s : float, optional
        Wall-clock seconds for the measured work.
    process_s : float, optional
        Process CPU seconds for the measured work.
    use_deterministic_costs : bool
        Whether this record used deterministic tiny-row cost units.

    Returns
    -------
    dict[str, float | int | str | bool | None]
        JSON-ready runtime metadata for cross-machine interpretation.
    """
    cpu_wall_ratio = None
    if wall_s is not None and process_s is not None and wall_s > 0.0:
        cpu_wall_ratio = float(process_s) / float(wall_s)
    return {
        "use_deterministic_costs": bool(use_deterministic_costs),
        "cpu_wall_ratio": cpu_wall_ratio,
        "torch_num_threads": int(torch.get_num_threads()),
        "device": str(getattr(config, "device", "unknown")) if config is not None else "unknown",
    }


@dataclass(frozen=True)
class W5PhaseTiming:
    """Per-seed W5 phase timing telemetry.

    Parameters
    ----------
    seed : str
        Seed family label.
    mode : str
        Routed W5 mode for this seed.
    pass_id : int
        Surrogate pass identifier, ``1`` for the existing objective and ``2``
        for the honest-aligned continuation.
    route_s : float
        Wall-clock seconds spent choosing the route.
    optimize_s : float
        Wall-clock seconds spent in surrogate descent.
    viability_s : float
        Wall-clock seconds spent projecting and checking checkpoints.
    score_s : float
        Wall-clock seconds spent in honest scoring for checkpoints.
    """

    seed: str
    mode: str
    pass_id: int
    route_s: float
    optimize_s: float
    viability_s: float
    score_s: float


@dataclass(frozen=True)
class W5ScorePair:
    """Directed, undirected, and optional V3 composites from one evaluation.

    Parameters
    ----------
    directed : float
        Score from the hierarchy-gated directed composite.
    undirected : float
        Score from the frozen common undirected composite.
    v3 : float, optional
        Runtime-restricted V3 tiered headline score when available.
    c5_whitespace_ratio : float, optional
        Runtime-restricted V3 C5 whitespace ratio when available.
    c4_clearance_penalty : float, optional
        Runtime-restricted V3 C4 clearance penalty when available.
    c4_clearance_contact_pairs : int, optional
        Runtime-restricted V3 C4 clearance contact count when available.
    g1_directed_flow : float, optional
        Frozen V3 ``G1_directed_flow`` score used by declared-layered
        finisher preservation guards.
    g1_depth_order : float, optional
        Frozen V3 ``G1_depth_order`` diagnostic score used by declared-layered
        finisher preservation guards.
    g4_layered_parent_centering : float, optional
        Frozen V3 ``G4_layered_parent_centering`` score used by deep-tree
        visual-shape preservation guards.
    g4_layered_subtree_congruence : float, optional
        Frozen V3 ``G4_layered_subtree_congruence`` score used by deep-tree
        visual-shape preservation guards.
    champion_ineligibility_flags : frozenset[str], optional
        Frozen V3 row flags that disqualify a candidate from champion
        selection. ``None`` preserves callers without V3 flag payloads.
    """

    directed: float
    undirected: float
    v3: Optional[float] = None
    c5_whitespace_ratio: Optional[float] = None
    c4_clearance_penalty: Optional[float] = None
    c4_clearance_contact_pairs: Optional[int] = None
    g1_directed_flow: Optional[float] = None
    g1_depth_order: Optional[float] = None
    g4_layered_parent_centering: Optional[float] = None
    g4_layered_subtree_congruence: Optional[float] = None
    champion_ineligibility_flags: Optional[frozenset[str]] = None


def _finite_v3_facet_score(v3_result: Any, code: str) -> Optional[float]:
    """Return a finite frozen V3 facet score when it is available.

    Parameters
    ----------
    v3_result : object
        Runtime-restricted V3 result exposing a ``facets`` mapping.
    code : str
        V3 facet code to read.

    Returns
    -------
    float or None
        Finite facet score when present; otherwise ``None``.
    """
    facet = getattr(v3_result, "facets", {}).get(code)
    value = getattr(facet, "score", None)
    if value is None:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


def w5_score_pair_from_v3_result(directed: float, undirected: float, v3_result: Any) -> W5ScorePair:
    """Build a W5 score pair with runtime V3 facet telemetry.

    Parameters
    ----------
    directed : float
        Directed frozen-ruler composite score.
    undirected : float
        Undirected frozen-ruler composite score.
    v3_result : object
        Runtime-restricted V3 result exposing ``scores`` and ``facets``.

    Returns
    -------
    W5ScorePair
        Score pair carrying the tiered headline score plus C4/C5 facet fields
        used to gate global-scale line search.
    """
    c4_meta = getattr(v3_result.facets.get("C4"), "metadata", {})
    c5_meta = getattr(v3_result.facets.get("C5"), "metadata", {})
    clearance_pairs = c4_meta.get("clearance_contact_pairs")
    return W5ScorePair(
        directed=directed,
        undirected=undirected,
        v3=float(v3_result.scores["tiered"]),
        c5_whitespace_ratio=float(c5_meta["whitespace_ratio"]),
        c4_clearance_penalty=float(c4_meta["clearance_penalty"]),
        c4_clearance_contact_pairs=None if clearance_pairs is None else int(clearance_pairs),
        g1_directed_flow=_finite_v3_facet_score(v3_result, "G1_directed_flow"),
        g1_depth_order=_finite_v3_facet_score(v3_result, "G1_depth_order"),
        g4_layered_parent_centering=_finite_v3_facet_score(
            v3_result,
            "G4_layered_parent_centering",
        ),
        g4_layered_subtree_congruence=_finite_v3_facet_score(
            v3_result,
            "G4_layered_subtree_congruence",
        ),
        champion_ineligibility_flags=frozenset(str(flag) for flag in v3_result.flags)
        & DEGENERACY_CHAMPION_INELIGIBLE_FLAGS,
    )


@dataclass(frozen=True)
class W5HonestAxes:
    """Honest per-axis W5 routing signals from the frozen metrics ruler.

    Parameters
    ----------
    flow : float, optional
        Honest ``directed_flow_score`` in ``[0, 1]`` when available.
    depth : float, optional
        Honest ``depth_order_score`` in ``[0, 1]`` when available.
    ksm : float, optional
        Honest ``ksm_score`` in ``[0, 1]`` when available.
    edge_length : float, optional
        Honest ``edge_length_deviation_score`` in ``[0, 1]`` when available.
    """

    flow: Optional[float] = None
    depth: Optional[float] = None
    ksm: Optional[float] = None
    edge_length: Optional[float] = None


def w5_honest_axes_from_metrics(numeric: dict[str, Any]) -> W5HonestAxes:
    """Extract route-safe honest axes from a ``full()`` metric dictionary.

    Parameters
    ----------
    numeric : dict[str, Any]
        Metric payload returned by ``dagua.metrics.full``.

    Returns
    -------
    W5HonestAxes
        Finite honest axis scores used by W5 routing and barrier weighting.
    """

    def finite_float(key: str) -> Optional[float]:
        """Return a finite float from ``numeric`` or ``None``.

        Parameters
        ----------
        key : str
            Metric key to read.

        Returns
        -------
        float or None
            Finite metric value when present.
        """
        value = numeric.get(key)
        if value is None:
            return None
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if math.isfinite(result) else None

    return W5HonestAxes(
        flow=finite_float("directed_flow_score"),
        depth=finite_float("depth_order_score"),
        ksm=finite_float("ksm_score"),
        edge_length=finite_float("edge_length_deviation_score"),
    )


def _valid_cluster_members(
    clusters: Optional[Mapping[str, Sequence[int]]],
    num_nodes: int,
) -> dict[str, tuple[int, ...]]:
    """Return non-empty declared cluster memberships.

    Parameters
    ----------
    clusters : Mapping[str, Sequence[int]] or None
        Declared cluster membership keyed by cluster name.
    num_nodes : int
        Number of layout nodes.

    Returns
    -------
    dict[str, tuple[int, ...]]
        Sorted in-range node indices keyed by stable string cluster names.
    """
    if not clusters:
        return {}
    valid: dict[str, tuple[int, ...]] = {}
    for name, members in clusters.items():
        indices = sorted({int(index) for index in members if 0 <= int(index) < num_nodes})
        if indices:
            valid[str(name)] = tuple(indices)
    return valid


def _cluster_depth_lookup(
    cluster_names: Sequence[str],
    cluster_parents: Optional[Mapping[str, Optional[str]]],
) -> dict[str, int]:
    """Return declared nesting depths for valid clusters.

    Parameters
    ----------
    cluster_names : Sequence[str]
        Valid cluster names.
    cluster_parents : Mapping[str, Optional[str]] or None
        Optional declared parent lookup.

    Returns
    -------
    dict[str, int]
        Depth per cluster, with roots at zero.
    """
    names = set(cluster_names)
    parents = {
        name: (
            str(cluster_parents.get(name))
            if cluster_parents is not None and cluster_parents.get(name) in names
            else None
        )
        for name in cluster_names
    }
    # User metadata may contain parent cycles; without this guard the parent
    # walk below never bottoms out (drywell R2-B1 F-1, the 4th copy of the
    # WP05-F02 pattern -- previously a RecursionError here silently aborted
    # the whole terminal W5 pass through dagua_native's blanket except).
    parents = break_cluster_parent_cycles(parents)
    depths: dict[str, int] = {}

    # Iterative chain walk (mirrors coordinate._cluster_depths after drywell
    # B2-F01): the memoize-after-recurse closure this replaces descended one
    # frame per nesting level and crashed at ~1000 levels of valid linear
    # nesting whenever the root sorted last. Walk up to the nearest memoized
    # ancestor (or a root), then assign depths ancestor-first -- the exact
    # memoization insertion order the recursion produced.
    for cluster_name in cluster_names:
        chain: list[str] = []
        current: Optional[str] = cluster_name
        while current is not None and current not in depths:
            chain.append(current)
            current = parents[current]
        next_depth = 0 if current is None else depths[current] + 1
        for name in reversed(chain):
            depths[name] = next_depth
            next_depth += 1
    return depths


def _weak_component_count(edge_index: torch.Tensor, num_nodes: int) -> int:
    """Return the undirected connected-component count.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``. Edges are treated as undirected for
        component packing eligibility.
    num_nodes : int
        Number of layout nodes.

    Returns
    -------
    int
        Number of weak connected components.
    """
    if num_nodes <= 0:
        return 0
    adjacency: list[list[int]] = [[] for _ in range(num_nodes)]
    cpu_edges = edge_index.detach().to(device="cpu", dtype=torch.long)
    if cpu_edges.numel() > 0:
        for source, target in cpu_edges.t().tolist():
            source_index = int(source)
            target_index = int(target)
            if (
                0 <= source_index < num_nodes
                and 0 <= target_index < num_nodes
                and source_index != target_index
            ):
                adjacency[source_index].append(target_index)
                adjacency[target_index].append(source_index)

    seen = [False] * num_nodes
    component_count = 0
    for start in range(num_nodes):
        if seen[start]:
            continue
        component_count += 1
        stack = [start]
        seen[start] = True
        while stack:
            node = stack.pop()
            for neighbor in adjacency[node]:
                if not seen[neighbor]:
                    seen[neighbor] = True
                    stack.append(neighbor)
    return component_count


def _cluster_tightening_gate(
    edge_index: torch.Tensor,
    clusters: Optional[Mapping[str, Sequence[int]]],
    cluster_parents: Optional[Mapping[str, Optional[str]]],
    num_nodes: int,
) -> tuple[bool, str, dict[str, tuple[int, ...]], dict[str, int]]:
    """Return whether component-sensitive cluster compaction may run.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``. Compaction is only eligible for
        single-component graphs because it changes intra-component geometry.
    clusters : Mapping[str, Sequence[int]] or None
        Declared cluster membership keyed by cluster name.
    cluster_parents : Mapping[str, Optional[str]] or None
        Optional declared parent lookup.
    num_nodes : int
        Number of layout nodes.

    Returns
    -------
    tuple[bool, str, dict[str, tuple[int, ...]], dict[str, int]]
        Gate verdict, reason, valid members, and nesting depth lookup.
    """
    members = _valid_cluster_members(clusters, num_nodes)
    depths = _cluster_depth_lookup(tuple(sorted(members)), cluster_parents)
    max_depth = max(depths.values(), default=0)
    if num_nodes > _CLUSTER_TIGHTEN_MAX_NODES:
        return False, "too_large", members, depths
    if _weak_component_count(edge_index, num_nodes) != 1:
        return False, "disconnected_components", members, depths
    if len(members) >= 2:
        return True, "multi_cluster", members, depths
    if max_depth >= 1:
        return True, "nested_cluster", members, depths
    return False, "no_declared_multi_or_nested_cluster", members, depths


def _cluster_box_escape_gate(
    clusters: Optional[Mapping[str, Sequence[int]]],
    cluster_parents: Optional[Mapping[str, Optional[str]]],
    num_nodes: int,
) -> tuple[bool, str, dict[str, tuple[int, ...]], dict[str, int]]:
    """Return whether rigid cluster-box escape candidates may run.

    Parameters
    ----------
    clusters : Mapping[str, Sequence[int]] or None
        Declared cluster membership keyed by cluster name.
    cluster_parents : Mapping[str, Optional[str]] or None
        Optional declared parent lookup.
    num_nodes : int
        Number of layout nodes.

    Returns
    -------
    tuple[bool, str, dict[str, tuple[int, ...]], dict[str, int]]
        Gate verdict, reason, valid members, and nesting depth lookup. Unlike
        compaction, this gate deliberately ignores graph connectivity because
        each escape move is a rigid whole-cluster translation.
    """
    members = _valid_cluster_members(clusters, num_nodes)
    depths = _cluster_depth_lookup(tuple(sorted(members)), cluster_parents)
    if num_nodes > _CLUSTER_TIGHTEN_MAX_NODES:
        return False, "too_large_escape", members, depths
    if len(members) >= 2:
        return True, "multi_cluster_escape", members, depths
    return False, "no_declared_multi_cluster", members, depths


def _cluster_bounds(
    pos: torch.Tensor,
    node_sizes: torch.Tensor,
    members: Sequence[int],
) -> tuple[float, float, float, float]:
    """Return a rendered AABB for a cluster's descendants.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    node_sizes : torch.Tensor
        Node-size tensor with shape ``[N, 2]``.
    members : Sequence[int]
        Descendant node indices.

    Returns
    -------
    tuple[float, float, float, float]
        Bounds as ``(left, right, bottom, top)``.
    """
    idx = torch.tensor(tuple(members), dtype=torch.long, device=pos.device)
    lower = pos[idx] - node_sizes[idx] * 0.5
    upper = pos[idx] + node_sizes[idx] * 0.5
    return (
        float(lower[:, 0].min().item()),
        float(upper[:, 0].max().item()),
        float(lower[:, 1].min().item()),
        float(upper[:, 1].max().item()),
    )


def _compact_cluster_descendants(
    pos: torch.Tensor,
    members_by_name: Mapping[str, tuple[int, ...]],
    depths: Mapping[str, int],
    factor: float,
    target_depth: Optional[int] = None,
) -> torch.Tensor:
    """Compact each cluster's descendants toward their current centroid.

    Parameters
    ----------
    pos : torch.Tensor
        Input positions with shape ``[N, 2]``.
    members_by_name : Mapping[str, tuple[int, ...]]
        Valid cluster memberships.
    depths : Mapping[str, int]
        Declared cluster depths.
    factor : float
        Centroid shrink factor in ``(0, 1]``.
    target_depth : int or None, optional
        When set, only clusters at this declared nesting depth are compacted.

    Returns
    -------
    torch.Tensor
        Compacted position tensor.
    """
    out = pos.detach().clone()
    for name in sorted(members_by_name, key=lambda item: (-depths.get(item, 0), item)):
        if target_depth is not None and depths.get(name, 0) != target_depth:
            continue
        members = members_by_name[name]
        if len(members) <= 1:
            continue
        idx = torch.tensor(members, dtype=torch.long, device=out.device)
        center = out[idx].mean(dim=0, keepdim=True)
        out[idx] = center + (out[idx] - center) * float(factor)
    return out


def _separate_sibling_clusters(
    pos: torch.Tensor,
    node_sizes: torch.Tensor,
    members_by_name: Mapping[str, tuple[int, ...]],
    depths: Mapping[str, int],
    cluster_parents: Optional[Mapping[str, Optional[str]]],
    min_gap: float,
) -> torch.Tensor:
    """Repel same-parent cluster boxes along the horizontal packing axis.

    Parameters
    ----------
    pos : torch.Tensor
        Input positions with shape ``[N, 2]``.
    node_sizes : torch.Tensor
        Node-size tensor with shape ``[N, 2]``.
    members_by_name : Mapping[str, tuple[int, ...]]
        Valid cluster memberships.
    depths : Mapping[str, int]
        Declared cluster depths.
    cluster_parents : Mapping[str, Optional[str]] or None
        Optional declared parent lookup.
    min_gap : float
        Minimum rendered gap between sibling cluster AABBs.

    Returns
    -------
    torch.Tensor
        Position tensor with sibling boxes separated.
    """
    out = pos.detach().clone()
    names = set(members_by_name)
    for depth in sorted(set(depths.values()), reverse=True):
        siblings: dict[Optional[str], list[str]] = {}
        for name in members_by_name:
            if depths.get(name, 0) != depth:
                continue
            raw_parent = cluster_parents.get(name) if cluster_parents is not None else None
            parent = str(raw_parent) if raw_parent in names else None
            siblings.setdefault(parent, []).append(name)
        for sibling_names in siblings.values():
            if len(sibling_names) <= 1:
                continue
            sibling_names.sort(
                key=lambda name: (_cluster_bounds(out, node_sizes, members_by_name[name])[0], name)
            )
            right_edge: Optional[float] = None
            for name in sibling_names:
                left, right, _, _ = _cluster_bounds(out, node_sizes, members_by_name[name])
                if right_edge is not None and left < right_edge + min_gap:
                    shift = right_edge + min_gap - left
                    idx = torch.tensor(members_by_name[name], dtype=torch.long, device=out.device)
                    out[idx, 0] += float(shift)
                    left += shift
                    right += shift
                right_edge = right if right_edge is None else max(right_edge, right)
    return out


def _bounds_area(bounds: tuple[float, float, float, float]) -> float:
    """Return the non-negative area of an ``(x_min, y_min, x_max, y_max)`` box.

    Parameters
    ----------
    bounds : tuple[float, float, float, float]
        Axis-aligned bounds.

    Returns
    -------
    float
        Box area, clamped at zero for malformed or degenerate boxes.
    """
    return max(0.0, float(bounds[2]) - float(bounds[0])) * max(
        0.0,
        float(bounds[3]) - float(bounds[1]),
    )


def _bounds_intersection_area(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    """Return the intersection area of two axis-aligned boxes.

    Parameters
    ----------
    left : tuple[float, float, float, float]
        First bounds as ``(x_min, y_min, x_max, y_max)``.
    right : tuple[float, float, float, float]
        Second bounds as ``(x_min, y_min, x_max, y_max)``.

    Returns
    -------
    float
        Strictly positive only when both axes overlap, including containment.
    """
    width = min(float(left[2]), float(right[2])) - max(float(left[0]), float(right[0]))
    height = min(float(left[3]), float(right[3])) - max(float(left[1]), float(right[1]))
    if width <= 0.0 or height <= 0.0:
        return 0.0
    return width * height


def _minimal_escape_shift(
    mover_bounds: tuple[float, float, float, float],
    obstacle_bounds: tuple[float, float, float, float],
    gutter: float,
) -> tuple[int, float]:
    """Return the smallest axis-aligned translation that clears an obstacle.

    Parameters
    ----------
    mover_bounds : tuple[float, float, float, float]
        Bounds of the cluster that will move.
    obstacle_bounds : tuple[float, float, float, float]
        Bounds of the sibling cluster that stays fixed for this pair scan.
    gutter : float
        Extra separation to enforce after clearing the sibling box.

    Returns
    -------
    tuple[int, float]
        Axis index and signed displacement in placement units.
    """
    x_left = float(obstacle_bounds[0]) - float(gutter) - float(mover_bounds[2])
    x_right = float(obstacle_bounds[2]) + float(gutter) - float(mover_bounds[0])
    x_shift = x_left if abs(x_left) <= abs(x_right) else x_right
    y_down = float(obstacle_bounds[1]) - float(gutter) - float(mover_bounds[3])
    y_up = float(obstacle_bounds[3]) + float(gutter) - float(mover_bounds[1])
    y_shift = y_down if abs(y_down) < abs(y_up) else y_up
    if abs(x_shift) <= abs(y_shift):
        return 0, x_shift
    return 1, y_shift


def _escape_cluster_boxes(
    pos: torch.Tensor,
    node_sizes: torch.Tensor,
    clusters: Mapping[str, Sequence[int]],
    cluster_parents: Optional[Mapping[str, Optional[str]]],
    gutter: float,
) -> torch.Tensor:
    """Rigidly translate smaller sibling cluster boxes until intersections clear.

    Parameters
    ----------
    pos : torch.Tensor
        Input positions with shape ``[N, 2]``.
    node_sizes : torch.Tensor
        Node-size tensor with shape ``[N, 2]``.
    clusters : Mapping[str, Sequence[int]]
        Declared cluster membership keyed by cluster name.
    cluster_parents : Mapping[str, Optional[str]] or None
        Optional declared parent lookup.
    gutter : float
        Post-escape sibling-box gutter in placement units.

    Returns
    -------
    torch.Tensor
        Position tensor after at most three derive-and-translate passes.
    """
    out = pos.detach().clone()
    for _ in range(_CLUSTER_BOX_ESCAPE_MAX_PASSES):
        profile = build_cluster_geometry_profile(
            out,
            node_sizes,
            {},
            clusters,
            cluster_parents,
        )
        if profile is None:
            break
        moved = False
        for left_name, right_name in profile.sibling_pairs:
            left_box = profile.boxes[left_name]
            right_box = profile.boxes[right_name]
            if _bounds_intersection_area(left_box.bounds, right_box.bounds) <= 0.0:
                continue
            left_key = (_bounds_area(left_box.bounds), left_name)
            right_key = (_bounds_area(right_box.bounds), right_name)
            if left_key <= right_key:
                mover_box, obstacle_box = left_box, right_box
            else:
                mover_box, obstacle_box = right_box, left_box
            axis, shift = _minimal_escape_shift(mover_box.bounds, obstacle_box.bounds, gutter)
            members = tuple(sorted(mover_box.descendants))
            if not members:
                continue
            idx = torch.tensor(members, dtype=torch.long, device=out.device)
            out[idx, axis] += float(shift)
            moved = True
        if not moved:
            break
    return out


def _root_cluster_names(
    members_by_name: Mapping[str, tuple[int, ...]],
    cluster_parents: Optional[Mapping[str, Optional[str]]],
) -> tuple[str, ...]:
    """Return declared root cluster names in deterministic order.

    Parameters
    ----------
    members_by_name : Mapping[str, tuple[int, ...]]
        Valid cluster memberships.
    cluster_parents : Mapping[str, Optional[str]] or None
        Optional declared parent lookup.

    Returns
    -------
    tuple[str, ...]
        Root cluster names whose parent is absent or not a valid cluster.
    """
    names = set(members_by_name)
    roots: list[str] = []
    for name in sorted(members_by_name):
        raw_parent = cluster_parents.get(name) if cluster_parents is not None else None
        if raw_parent not in names:
            roots.append(name)
    return tuple(roots)


def _push_root_clusters_from_dominant(
    pos: torch.Tensor,
    members_by_name: Mapping[str, tuple[int, ...]],
    root_names: Sequence[str],
    dominant_name: str,
    factor: float,
) -> torch.Tensor:
    """Push non-dominant root clusters away from the dominant root centroid.

    Parameters
    ----------
    pos : torch.Tensor
        Input positions with shape ``[N, 2]``.
    members_by_name : Mapping[str, tuple[int, ...]]
        Valid cluster memberships.
    root_names : Sequence[str]
        Root cluster names to consider.
    dominant_name : str
        Root cluster with the largest member count.
    factor : float
        Multiplicative centroid distance factor for non-dominant roots.

    Returns
    -------
    torch.Tensor
        Position tensor after rigid root-cluster translations.
    """
    out = pos.detach().clone()
    dominant_idx = torch.tensor(members_by_name[dominant_name], dtype=torch.long, device=out.device)
    dominant_center = out[dominant_idx].mean(dim=0, keepdim=True)
    for name in sorted(root_names):
        if name == dominant_name:
            continue
        members = members_by_name[name]
        if not members:
            continue
        idx = torch.tensor(members, dtype=torch.long, device=out.device)
        center = out[idx].mean(dim=0, keepdim=True)
        out[idx] += (center - dominant_center) * (float(factor) - 1.0)
    return out


def _compact_single_cluster_descendants(
    pos: torch.Tensor,
    members: Sequence[int],
    factor: float,
) -> torch.Tensor:
    """Compact one cluster's descendants around their current centroid.

    Parameters
    ----------
    pos : torch.Tensor
        Input positions with shape ``[N, 2]``.
    members : Sequence[int]
        Descendant node indices for the cluster being compacted.
    factor : float
        Centroid shrink factor in ``(0, 1]``.

    Returns
    -------
    torch.Tensor
        Position tensor with only the selected descendants compacted.
    """
    out = pos.detach().clone()
    if len(members) <= 1:
        return out
    idx = torch.tensor(tuple(members), dtype=torch.long, device=out.device)
    center = out[idx].mean(dim=0, keepdim=True)
    out[idx] = center + (out[idx] - center) * float(factor)
    return out


def build_cluster_tightening_candidates(
    incumbent_pos: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    clusters: Optional[Mapping[str, Sequence[int]]],
    cluster_parents: Optional[Mapping[str, Optional[str]]],
) -> tuple[ClusterTighteningCandidate, ...]:
    """Build terminal cluster-tightening candidates under the structural gate.

    Parameters
    ----------
    incumbent_pos : torch.Tensor
        Incumbent positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]`` used to exclude disconnected graphs
        from component-packing-sensitive tightening.
    node_sizes : torch.Tensor
        Node-size tensor with shape ``[N, 2]``.
    clusters : Mapping[str, Sequence[int]] or None
        Declared cluster membership.
    cluster_parents : Mapping[str, Optional[str]] or None
        Optional declared nesting metadata.

    Returns
    -------
    tuple[ClusterTighteningCandidate, ...]
        Candidate positions. Empty means the structural gate did not fire.
    """
    escape_enabled, escape_reason, escape_members, escape_depths = _cluster_box_escape_gate(
        clusters,
        cluster_parents,
        int(incumbent_pos.shape[0]),
    )
    compact_enabled, compact_reason, compact_members, compact_depths = _cluster_tightening_gate(
        edge_index,
        clusters,
        cluster_parents,
        int(incumbent_pos.shape[0]),
    )
    if not escape_enabled and not compact_enabled:
        return ()
    work_pos = incumbent_pos.detach().to(device="cpu", dtype=torch.float32)
    work_sizes = node_sizes.detach().to(device="cpu", dtype=torch.float32)
    mean_diag = (
        float(torch.linalg.norm(work_sizes, dim=1).mean().item()) if work_sizes.numel() else 1.0
    )
    min_gap = max(mean_diag * _CLUSTER_TIGHTEN_MIN_SIBLING_GAP_FACTOR, 1.0e-6)
    candidates: list[ClusterTighteningCandidate] = []

    def append_candidate(
        label: str,
        candidate_pos: torch.Tensor,
        gate_reason: str,
        members_by_name: Mapping[str, tuple[int, ...]],
        depths: Mapping[str, int],
    ) -> None:
        """Append one finite candidate after overlap projection fallback.

        Parameters
        ----------
        label : str
            Stable candidate label suffix.
        candidate_pos : torch.Tensor
            Candidate positions with shape ``[N, 2]``.
        gate_reason : str
            Structural gate reason for telemetry.
        members_by_name : Mapping[str, tuple[int, ...]]
            Valid cluster memberships for metadata.
        depths : Mapping[str, int]
            Declared cluster depths for metadata.

        Returns
        -------
        None
            The candidate is appended to ``candidates`` in-place.
        """
        separated = candidate_pos.detach().clone()
        if _overlap_count(separated, work_sizes) > _overlap_count(work_pos, work_sizes):
            project_overlaps(
                separated,
                work_sizes,
                iterations=4,
                convergent=True,
            )
        if separated.numel() > 0:
            separated -= separated.mean(dim=0, keepdim=True)
        candidates.append(
            ClusterTighteningCandidate(
                name=f"cluster_tighten_{label}",
                pos=separated.to(device=incumbent_pos.device, dtype=incumbent_pos.dtype),
                gate_reason=gate_reason,
                cluster_count=len(members_by_name),
                max_depth=max(depths.values(), default=0),
            )
        )

    escaped_by_gutter: dict[float, torch.Tensor] = {}
    if escape_enabled and clusters is not None:
        for gutter_factor in _CLUSTER_BOX_ESCAPE_GUTTER_FACTORS:
            escaped = _escape_cluster_boxes(
                work_pos,
                work_sizes,
                clusters,
                cluster_parents,
                max(mean_diag * gutter_factor, 1.0e-6),
            )
            escaped_by_gutter[gutter_factor] = escaped
            if torch.equal(escaped, work_pos):
                continue
            append_candidate(
                f"cluster_box_escape_g{gutter_factor:.2f}",
                escaped,
                escape_reason,
                escape_members,
                escape_depths,
            )

    if (
        escape_enabled
        and int(work_pos.shape[0]) <= _CLUSTER_SEPARATE_MAX_NODES
        and len(_root_cluster_names(escape_members, cluster_parents))
        <= _CLUSTER_SEPARATE_MAX_ROOT_CLUSTERS
    ):
        root_names = _root_cluster_names(escape_members, cluster_parents)
        if len(root_names) >= 2:
            dominant_name = min(root_names, key=lambda name: (-len(escape_members[name]), name))
            escape_base = escaped_by_gutter.get(_CLUSTER_BOX_ESCAPE_GUTTER_FACTORS[0], work_pos)
            for push_factor in _CLUSTER_SEPARATE_PUSH_FACTORS:
                pushed = _push_root_clusters_from_dominant(
                    escape_base,
                    escape_members,
                    root_names,
                    dominant_name,
                    push_factor,
                )
                append_candidate(
                    f"cluster_separate_push_{push_factor:.2f}",
                    pushed,
                    escape_reason,
                    escape_members,
                    escape_depths,
                )
            compacted_dominant = _compact_single_cluster_descendants(
                escape_base,
                escape_members[dominant_name],
                0.85,
            )
            pushed_compacted = _push_root_clusters_from_dominant(
                compacted_dominant,
                escape_members,
                root_names,
                dominant_name,
                _CLUSTER_SEPARATE_PUSH_FACTORS[0],
            )
            append_candidate(
                "cluster_box_escape_compact_dominant_0.85_push_1.15",
                pushed_compacted,
                escape_reason,
                escape_members,
                escape_depths,
            )
            compacted_stronger = _compact_single_cluster_descendants(
                escape_base,
                escape_members[dominant_name],
                0.75,
            )
            append_candidate(
                "cluster_box_escape_compact_dominant_0.75",
                compacted_stronger,
                escape_reason,
                escape_members,
                escape_depths,
            )

    if compact_enabled:
        max_depth = max(compact_depths.values(), default=0)
        variants: list[tuple[str, tuple[int, ...], float, float]] = []
        if int(work_pos.shape[0]) <= _CLUSTER_SEPARATE_MAX_NODES:
            for factor in _CLUSTER_TIGHTEN_STRONG_FACTORS:
                variants.extend(
                    [
                        (f"root_compact_strong_f{factor:.2f}", (0,), factor, 0.0),
                        (f"deepest_compact_strong_f{factor:.2f}", (max_depth,), factor, 0.0),
                    ]
                )
        variants.extend(
            [
                (
                    "root_deepest_compact_f0.99",
                    (0, max_depth),
                    _CLUSTER_TIGHTEN_MILD_FACTORS[0],
                    0.0,
                ),
                ("root_compact_f0.99", (0,), _CLUSTER_TIGHTEN_MILD_FACTORS[0], 0.0),
                ("deepest_compact_f0.99", (max_depth,), _CLUSTER_TIGHTEN_MILD_FACTORS[0], 0.0),
                ("root_compact_stronger_f0.92", (0,), _CLUSTER_TIGHTEN_MILD_FACTORS[1], 0.0),
                (
                    "deepest_compact_stronger_f0.92",
                    (max_depth,),
                    _CLUSTER_TIGHTEN_MILD_FACTORS[1],
                    0.0,
                ),
                ("sibling_gap_f0.99", (0,), _CLUSTER_TIGHTEN_MILD_FACTORS[0], min_gap * 0.1),
            ]
        )
        seen: set[tuple[str, tuple[int, ...], float, float]] = set()
        for label, target_depths, factor, sibling_gap in variants:
            key = (label, target_depths, factor, sibling_gap)
            if key in seen:
                continue
            seen.add(key)
            separated = work_pos.detach().clone()
            for target_depth in target_depths:
                separated = _compact_cluster_descendants(
                    separated,
                    compact_members,
                    compact_depths,
                    factor,
                    target_depth=target_depth,
                )
            if sibling_gap > 0.0:
                separated = _separate_sibling_clusters(
                    separated,
                    work_sizes,
                    compact_members,
                    compact_depths,
                    cluster_parents,
                    sibling_gap,
                )
            append_candidate(
                label,
                separated,
                compact_reason,
                compact_members,
                compact_depths,
            )
    return tuple(candidates)


@dataclass(frozen=True)
class W5Seed:
    """Warm-start position for the W5 finisher.

    Parameters
    ----------
    name : str
        Stable seed family label.
    pos : torch.Tensor
        Seed positions with shape ``[N, 2]``.
    """

    name: str
    pos: torch.Tensor


@dataclass(frozen=True)
class W5CostPlan:
    """Modeled W5 work bounds admitted by the shared ledger.

    Parameters
    ----------
    seeds : int
        Number of finite seeds to run.
    steps : int
        Maximum optimizer steps per seed.
    checkpoints : int
        Maximum honest-scored checkpoints per seed, capped at two.
    measured_step_s : float
        Modeled wall-second cost for one optimizer step. The historical field
        name is retained for telemetry compatibility.
    warmup_s : float
        Shadow wall-clock warmup observed by the surrogate probe.
    referee_s : float
        Modeled wall-second cost reserved for one checkpoint's honest scoring,
        including the deterministic global scale search evaluations.
    budget_s : float
        Wall-clock seconds available under the shared W5 spend cap.
    budget_usable_s : float
        Wall-clock seconds available after the explicit return reserve.
    predicted_s : float
        Conservative wall-clock cost estimate for the admitted plan.
    shadow_step_s : float, optional
        Measured surrogate step wall seconds, retained for calibration audits
        and never used for plan sizing.
    shadow_warmup_s : float, optional
        Measured surrogate first-step wall seconds, retained for calibration
        audits and never used for plan sizing.
    scale_search_evals : int
        Number of honest referee evaluations reserved per checkpoint for the
        deterministic global scale line search.
    """

    seeds: int
    steps: int
    checkpoints: int
    measured_step_s: float
    warmup_s: float
    referee_s: float
    budget_s: float
    budget_usable_s: float
    predicted_s: float
    shadow_step_s: Optional[float] = None
    shadow_warmup_s: Optional[float] = None
    scale_search_evals: int = _W5_SCALE_SEARCH_EVALS


@dataclass(frozen=True)
class W5StepMeasurement:
    """Measured W5 surrogate cost components.

    Parameters
    ----------
    step_s : float
        Wall-clock seconds for one post-warmup optimizer step.
    warmup_s : float
        Wall-clock seconds charged once for the first optimizer step.
    """

    step_s: float
    warmup_s: float


@dataclass(frozen=True)
class W5Checkpoint:
    """Honest-scored W5 checkpoint telemetry.

    Parameters
    ----------
    seed : str
        Seed family label.
    mode : str
        Finisher mode.
    pass_id : int
        Surrogate pass identifier, ``1`` for the existing objective and ``2``
        for the honest-aligned continuation.
    step : int
        Optimization step represented by the checkpoint.
    surrogate_delta : float
        Start loss minus checkpoint loss, so positive means the surrogate improved.
    honest_delta : float
        Directed honest score minus the incumbent/current-winner directed score.
    undirected_honest_delta : float
        Undirected honest score minus the incumbent/current-winner undirected score.
    honest_score_pair : W5ScorePair
        Directed and undirected honest scores for the checkpoint.
    accepted : bool
        Whether this checkpoint cleared the honest W5 accept margin.
    reason : str
        Accept or reject reason.
    pass_spend_s : float
        Wall-clock seconds spent in this seed/mode/pass through the checkpoint
        scoring event.
    legacy_tallied_sole_failure : bool
        Whether the old V3-branch tallied-axis conjunct was the only legacy
        reason this checkpoint would have been rejected.
    """

    seed: str
    mode: str
    pass_id: int
    step: int
    surrogate_delta: float
    honest_delta: float
    undirected_honest_delta: float
    honest_score_pair: W5ScorePair
    accepted: bool
    reason: str
    pass_spend_s: float
    legacy_tallied_sole_failure: bool = False


@dataclass(frozen=True)
class W5Candidate:
    """Accepted W5 candidate.

    Parameters
    ----------
    name : str
        Candidate label.
    pos : torch.Tensor
        Candidate positions with shape ``[N, 2]``.
    score_pair : W5ScorePair
        Directed and undirected honest scores.
    mode : str
        Finisher mode that produced the candidate.
    """

    name: str
    pos: torch.Tensor
    score_pair: W5ScorePair
    mode: str


@dataclass(frozen=True)
class W5FinisherResult:
    """Anytime winner result for one W5 finisher invocation.

    Parameters
    ----------
    winner_pos : torch.Tensor
        Current W5 winner, initialized from the incumbent.
    incumbent_score_pair : W5ScorePair
        Directed and undirected scores for the entry incumbent.
    winner_score_pair : W5ScorePair
        Directed and undirected scores for ``winner_pos``.
    winner_name : str
        Incumbent or accepted checkpoint label.
    deadline_returned : bool
        Whether W5 returned early because deadline or budget was exhausted.
    accepted : tuple[W5Candidate, ...]
        Accepted W5 checkpoints.
    rejected : tuple[W5Checkpoint, ...]
        Rejected W5 checkpoints.
    checkpoints : tuple[W5Checkpoint, ...]
        Honest-scored checkpoints, accepted or rejected.
    mode : str
        Routed mode, or ``"skip"`` when no seed ran.
    steps : int
        Total optimizer steps completed across seeds.
    skipped_reason : str, optional
        Reason the finisher skipped all work.
    slice_s : float, optional
        Work slice granted to this invocation.
    spent_s : float
        Wall-clock seconds spent in this invocation.
    process_spent_s : float
        Process CPU seconds spent in this invocation.
    remaining_entry_s : float, optional
        Benchmark seconds remaining at entry.
    remaining_exit_s : float, optional
        Benchmark seconds remaining at return.
    node_count : int
        Number of graph nodes.
    edge_count : int
        Number of graph edges.
    is_semantically_directed : bool
        Routed directedness flag.
    declared_hierarchical : bool
        Routed hierarchy flag.
    direction_is_declared : bool
        Whether directedness came from a user/config declaration.
    graph_name : str, optional
        Benchmark graph name when the driver supplied one on the config.
    incumbent_axes : W5HonestAxes, optional
        Honest per-axis incumbent scores used to route W5.
    phase_timings_s : tuple[W5PhaseTiming, ...]
        Per-seed route/optimize/viability/score timing records.
    viability_counts : dict[str, int]
        Counts for viability outcomes, including projection outcomes.
    viability_drop_counts : dict[str, int]
        Counts for pre-score viability drop reasons.
    cost_plan : W5CostPlan, optional
        Measured-cost admission math, when the terminal W5 path used it.
    """

    winner_pos: torch.Tensor
    incumbent_score_pair: W5ScorePair
    winner_score_pair: W5ScorePair
    winner_name: str
    deadline_returned: bool
    accepted: tuple[W5Candidate, ...]
    rejected: tuple[W5Checkpoint, ...]
    checkpoints: tuple[W5Checkpoint, ...]
    mode: str
    steps: int
    skipped_reason: Optional[str] = None
    slice_s: Optional[float] = None
    spent_s: float = 0.0
    process_spent_s: float = 0.0
    remaining_entry_s: Optional[float] = None
    remaining_exit_s: Optional[float] = None
    node_count: int = 0
    edge_count: int = 0
    is_semantically_directed: bool = False
    declared_hierarchical: bool = False
    direction_is_declared: bool = False
    graph_name: Optional[str] = None
    incumbent_axes: Optional[W5HonestAxes] = None
    phase_timings_s: tuple[W5PhaseTiming, ...] = ()
    viability_counts: dict[str, int] = field(default_factory=dict)
    viability_drop_counts: dict[str, int] = field(default_factory=dict)
    cost_plan: Optional[W5CostPlan] = None


@dataclass(frozen=True)
class W5StressSample:
    """Fixed differentiable stress sample for W5 pass 2.

    Parameters
    ----------
    sources : torch.Tensor
        Source node indices with shape ``[P]``.
    targets : torch.Tensor
        Target node indices with shape ``[P]``.
    graph_distances : torch.Tensor
        Positive graph distances with shape ``[P]``.
    """

    sources: torch.Tensor
    targets: torch.Tensor
    graph_distances: torch.Tensor


@dataclass(frozen=True)
class W5NeighborhoodSample:
    """Fixed contrastive neighborhood sample for W5 C3 guidance.

    Parameters
    ----------
    anchors : torch.Tensor
        Anchor node indices with shape ``[P]``.
    positives : torch.Tensor
        Positive graph-neighborhood target indices with shape ``[P]``.
    negatives : torch.Tensor
        Negative graph-neighborhood target indices with shape ``[P]``.
    radii : torch.Tensor
        Graph-distance radius for each triplet with shape ``[P]``.
    """

    anchors: torch.Tensor
    positives: torch.Tensor
    negatives: torch.Tensor
    radii: torch.Tensor


@dataclass(frozen=True)
class W5ScaleSearchResult:
    """Best deterministic global-scale candidate from one checkpoint.

    Parameters
    ----------
    pos : torch.Tensor
        Best scaled positions with shape ``[N, 2]``.
    score_pair : W5ScorePair
        Honest score pair for ``pos``.
    raw_score_pair : W5ScorePair
        Honest score pair for the unscaled checkpoint.
    scale : float
        Global scale factor applied around the checkpoint centroid.
    evals : int
        Number of honest score evaluations used by the search.
    keepalive : tuple[torch.Tensor, ...]
        Scored position tensors retained until W5 exits so id-keyed scorer
        caches cannot collide with recycled Python object ids.
    """

    pos: torch.Tensor
    score_pair: W5ScorePair
    raw_score_pair: W5ScorePair
    scale: float
    evals: int
    keepalive: tuple[torch.Tensor, ...] = ()


@dataclass(frozen=True)
class W5GlobalScaleSweepCandidate:
    """One scored terminal global-scale sweep candidate.

    Parameters
    ----------
    scale : float
        Uniform multiplier applied around the incumbent centroid.
    score_pair : W5ScorePair, optional
        Restricted-V3-backed score pair when scoring completed.
    referee_key : tuple[int, float]
        Severe-G6 referee key for the scaled candidate.
    selected : bool
        Whether this candidate became the terminal scale-sweep winner.
    reason : str
        Stable accept or reject reason for telemetry and tests.
    scale_x : float, default=1.0
        X-axis multiplier applied around the incumbent centroid.
    scale_y : float, default=1.0
        Y-axis multiplier applied around the incumbent centroid.
    """

    scale: float
    score_pair: Optional[W5ScorePair]
    referee_key: Tuple[int, float]
    selected: bool
    reason: str
    scale_x: float = 1.0
    scale_y: float = 1.0


@dataclass(frozen=True)
class W5GlobalScaleSweepResult:
    """Result of the terminal post-W5 scale sweep.

    Parameters
    ----------
    winner_pos : torch.Tensor
        Incumbent positions or the best accepted scaled positions with shape
        ``[N, 2]``.
    winner_score_pair : W5ScorePair
        Score pair for ``winner_pos``.
    winner_scale : float
        Accepted global scale multiplier. ``1.0`` means the incumbent won.
        For anisotropic winners this is the geometric mean of the axis
        multipliers and exists for compatibility with the prior uniform-only
        telemetry.
    winner_scale_x : float, default=1.0
        Accepted X-axis multiplier.
    winner_scale_y : float, default=1.0
        Accepted Y-axis multiplier.
    selected : bool
        Whether any scaled candidate strictly improved restricted V3.
    candidates : tuple[W5GlobalScaleSweepCandidate, ...]
        Candidate telemetry in evaluation order.
    keepalive : tuple[torch.Tensor, ...]
        Scored tensors retained so id-keyed scorer caches cannot collide with
        recycled Python object ids during the sweep.
    """

    winner_pos: torch.Tensor
    winner_score_pair: W5ScorePair
    winner_scale: float
    selected: bool
    candidates: tuple[W5GlobalScaleSweepCandidate, ...]
    keepalive: tuple[torch.Tensor, ...] = ()
    winner_scale_x: float = 1.0
    winner_scale_y: float = 1.0


@dataclass(frozen=True)
class W5SmallNAnnealCandidate:
    """One scored terminal small-N anneal perturbation.

    Parameters
    ----------
    trial : int
        Zero-based trial index in the deterministic anneal schedule.
    sigma : float
        Gaussian perturbation standard deviation in layout units.
    changed_nodes : int
        Number of node positions perturbed in this trial.
    score_pair : W5ScorePair, optional
        Restricted-V3-backed score pair when scoring completed.
    referee_key : tuple[int, float]
        Severe-G6 referee key for the candidate.
    selected : bool
        Whether this candidate became the anneal winner.
    reason : str
        Stable accept or reject reason for telemetry and tests.
    """

    trial: int
    sigma: float
    changed_nodes: int
    score_pair: Optional[W5ScorePair]
    referee_key: Tuple[int, float]
    selected: bool
    reason: str


@dataclass(frozen=True)
class W5SmallNAnnealResult:
    """Result of the terminal small-N V3-surrogate anneal.

    Parameters
    ----------
    winner_pos : torch.Tensor
        Incumbent positions or the best accepted annealed positions with shape
        ``[N, 2]``.
    winner_score_pair : W5ScorePair
        Score pair for ``winner_pos``.
    selected : bool
        Whether any perturbation strictly improved restricted V3.
    trials_completed : int
        Number of perturbation trials scored or attempted.
    accepted_count : int
        Number of strict V3-improving perturbations accepted during anneal.
    skipped_reason : str, optional
        Structural or budget reason the annealer skipped before trial work.
    candidates : tuple[W5SmallNAnnealCandidate, ...]
        Candidate telemetry in evaluation order.
    keepalive : tuple[torch.Tensor, ...]
        Scored tensors retained so id-keyed scorer caches cannot collide with
        recycled Python object ids during anneal.
    """

    winner_pos: torch.Tensor
    winner_score_pair: W5ScorePair
    selected: bool
    trials_completed: int
    accepted_count: int
    skipped_reason: Optional[str]
    candidates: tuple[W5SmallNAnnealCandidate, ...]
    keepalive: tuple[torch.Tensor, ...] = ()


@dataclass(frozen=True)
class W5SMACOFStressCandidate:
    """One scored terminal SMACOF stress-polish candidate.

    Parameters
    ----------
    iterations : int
        Number of Guttman-transform SMACOF updates used for the candidate.
    output_scale : float
        Uniform multiplier applied around the SMACOF output centroid before
        scoring. ``1.0`` is the unscaled stress-polish result.
    score_pair : W5ScorePair, optional
        Restricted-V3-backed score pair when scoring completed.
    referee_key : tuple[int, float]
        Severe-G6 referee key for the candidate.
    selected : bool
        Whether this candidate became the SMACOF stress-polish winner.
    reason : str
        Stable accept or reject reason for telemetry and tests.
    """

    iterations: int
    output_scale: float
    score_pair: Optional[W5ScorePair]
    referee_key: Tuple[int, float]
    selected: bool
    reason: str


@dataclass(frozen=True)
class W5SMACOFStressResult:
    """Result of the terminal full-pair SMACOF stress-polish arm.

    Parameters
    ----------
    winner_pos : torch.Tensor
        Incumbent positions or the best accepted SMACOF candidate with shape
        ``[N, 2]``.
    winner_score_pair : W5ScorePair
        Score pair for ``winner_pos``.
    selected : bool
        Whether any SMACOF candidate strictly improved restricted V3.
    skipped_reason : str, optional
        Structural or budget reason the arm skipped before scoring candidates.
    candidates : tuple[W5SMACOFStressCandidate, ...]
        Candidate telemetry in evaluation order.
    keepalive : tuple[torch.Tensor, ...]
        Scored tensors retained so id-keyed scorer caches cannot collide with
        recycled Python object ids during the arm.
    """

    winner_pos: torch.Tensor
    winner_score_pair: W5ScorePair
    selected: bool
    skipped_reason: Optional[str]
    candidates: tuple[W5SMACOFStressCandidate, ...]
    keepalive: tuple[torch.Tensor, ...] = ()


@dataclass(frozen=True)
class W5ContinuousFacetPolishCandidate:
    """One accepted continuous-facet local-search move.

    Parameters
    ----------
    pass_id : int
        One-based coordinate-descent pass that accepted the move.
    node : int
        Node index moved by the accepted candidate.
    step : float
        Layout-unit displacement magnitude.
    direction : tuple[float, float]
        Unit-grid direction applied to the moved node.
    score_pair : W5ScorePair
        Referee-backed score pair for the accepted candidate.
    surrogate_loss : float
        Continuous C9/C7/C3 surrogate objective value after the move.
    referee_key : tuple[int, float]
        Severe-G6 referee key for the accepted candidate.
    g4_layered_parent_centering : float, optional
        Deep-tree layered parent-centering score for the accepted candidate.
    g4_layered_subtree_congruence : float, optional
        Deep-tree layered subtree-congruence score for the accepted candidate.
    """

    pass_id: int
    node: int
    step: float
    direction: tuple[float, float]
    score_pair: W5ScorePair
    surrogate_loss: float
    referee_key: Tuple[int, float]
    g4_layered_parent_centering: Optional[float] = None
    g4_layered_subtree_congruence: Optional[float] = None


@dataclass(frozen=True)
class W5ContinuousFacetPolishResult:
    """Result of the terminal C9/C7/C3 continuous-facet polish.

    Parameters
    ----------
    winner_pos : torch.Tensor
        Incumbent positions or the best accepted polished positions with shape
        ``[N, 2]``.
    winner_score_pair : W5ScorePair
        Referee-backed score pair for ``winner_pos``.
    selected : bool
        Whether at least one strict V3-improving move was accepted.
    skipped_reason : str, optional
        Structural, budget, or scoring reason when no work ran.
    gate_reason : str
        Structural gate label for telemetry.
    passes_completed : int
        Number of coordinate passes attempted.
    evaluations : int
        Number of candidate positions honestly scored.
    accepted : tuple[W5ContinuousFacetPolishCandidate, ...]
        Accepted strict V3-improving moves in chronological order.
    keepalive : tuple[torch.Tensor, ...]
        Scored tensors retained so id-keyed scorer caches cannot collide with
        recycled Python object ids during local search.
    start_surrogate_loss : float, optional
        Continuous surrogate loss on entry.
    winner_surrogate_loss : float, optional
        Continuous surrogate loss for ``winner_pos``.
    """

    winner_pos: torch.Tensor
    winner_score_pair: W5ScorePair
    selected: bool
    skipped_reason: Optional[str]
    gate_reason: str
    passes_completed: int
    evaluations: int
    accepted: tuple[W5ContinuousFacetPolishCandidate, ...]
    keepalive: tuple[torch.Tensor, ...] = ()
    start_surrogate_loss: Optional[float] = None
    winner_surrogate_loss: Optional[float] = None


def is_worker_timeout_like_exception(exc: Exception) -> bool:
    """Return whether ``exc`` is the benchmark worker timeout signal.

    Parameters
    ----------
    exc : Exception
        Exception caught by optional polish or W5 code.

    Returns
    -------
    bool
        ``True`` when the worker alarm raised the exception.
    """
    return type(exc).__name__ == "_WorkerLayoutTimeoutError" or (
        "worker layout timeout exceeded" in str(exc)
    )


def candidate_introduces_champion_ineligible_flag(
    candidate_flags: Optional[frozenset[str]],
    incumbent_flags: Optional[frozenset[str]],
) -> bool:
    """Return whether a candidate newly carries a champion-ineligible flag.

    Parameters
    ----------
    candidate_flags : frozenset[str], optional
        Frozen V3 champion-ineligibility flags carried by the candidate.
        ``None`` disables this guard for legacy callers.
    incumbent_flags : frozenset[str], optional
        Frozen V3 champion-ineligibility flags carried by the incumbent.

    Returns
    -------
    bool
        ``True`` when both flag payloads are populated and the candidate has a
        frozen champion-ineligibility flag absent from the incumbent.
    """
    if candidate_flags is None or incumbent_flags is None:
        return False
    return bool((candidate_flags - incumbent_flags) & DEGENERACY_CHAMPION_INELIGIBLE_FLAGS)


def _layered_preservation_required(
    *,
    is_semantically_directed: bool,
    declared_hierarchical: bool,
    direction_is_declared: bool,
) -> bool:
    """Return whether a finisher candidate must preserve layered reading.

    Parameters
    ----------
    is_semantically_directed : bool
        Whether the graph's edge direction has semantic meaning.
    declared_hierarchical : bool
        Whether the row declares hierarchy metadata used by the frozen ruler.
    direction_is_declared : bool
        Whether semantic direction came from explicit user/config metadata.

    Returns
    -------
    bool
        ``True`` for declared-layered rows; ``False`` for inferred,
        undirected, clustered-undirected, geometric, and cyclic-feedback rows.
    """
    return bool(is_semantically_directed and direction_is_declared and declared_hierarchical)


def _layered_reading_preserved(
    candidate: W5ScorePair,
    incumbent: W5ScorePair,
    eps: float = _W5_LAYERED_READING_EPS,
) -> bool:
    """Return whether frozen V3 layered-reading facets did not regress.

    Parameters
    ----------
    candidate : W5ScorePair
        Candidate score pair carrying frozen V3 G1 facet telemetry.
    incumbent : W5ScorePair
        Incumbent score pair carrying frozen V3 G1 facet telemetry.
    eps : float, default=_W5_LAYERED_READING_EPS
        Allowed numerical slack before a facet regression rejects a candidate.

    Returns
    -------
    bool
        ``True`` when the candidate preserves both available layered-reading
        facets. Missing facet payloads keep legacy/direct unit callers active.
    """
    for candidate_value, incumbent_value in (
        (candidate.g1_directed_flow, incumbent.g1_directed_flow),
        (candidate.g1_depth_order, incumbent.g1_depth_order),
    ):
        if candidate_value is None or incumbent_value is None:
            continue
        if float(candidate_value) < float(incumbent_value) - float(eps):
            return False
    return True


def _layered_shape_preserved(
    candidate: W5ScorePair,
    incumbent: W5ScorePair,
    eps: float = _W5_LAYERED_SHAPE_EPS,
    min_score: float = _W5_LAYERED_SHAPE_MIN_SCORE,
) -> bool:
    """Return whether layered tree-shape facets remain visually plausible.

    Parameters
    ----------
    candidate : W5ScorePair
        Candidate score pair carrying frozen V3 G4 facet telemetry.
    incumbent : W5ScorePair
        Incumbent score pair carrying frozen V3 G4 facet telemetry.
    eps : float, default=_W5_LAYERED_SHAPE_EPS
        Allowed score drop relative to the current incumbent.
    min_score : float, default=_W5_LAYERED_SHAPE_MIN_SCORE
        Absolute lower bound for G4 tree-shape facets when both payloads are
        available.

    Returns
    -------
    bool
        ``True`` when parent centering and repeated-subtree congruence stay
        within the small visual-trade window expected for deep tree fan polish.
        Missing payloads keep non-V3 and legacy unit callers active.
    """
    for candidate_value, incumbent_value in (
        (candidate.g4_layered_parent_centering, incumbent.g4_layered_parent_centering),
        (candidate.g4_layered_subtree_congruence, incumbent.g4_layered_subtree_congruence),
    ):
        if candidate_value is None or incumbent_value is None:
            continue
        floor = max(float(min_score), float(incumbent_value) - float(eps))
        if float(candidate_value) < floor:
            return False
    return True


def _layered_shape_values_preserved(
    candidate: tuple[Optional[float], Optional[float]],
    incumbent: tuple[Optional[float], Optional[float]],
    eps: float = _W5_LAYERED_SHAPE_EPS,
    min_score: float = _W5_LAYERED_SHAPE_MIN_SCORE,
) -> bool:
    """Return whether raw layered-shape facet values pass the visual guard.

    Parameters
    ----------
    candidate : tuple[float or None, float or None]
        Candidate parent-centering and subtree-congruence scores.
    incumbent : tuple[float or None, float or None]
        Incumbent parent-centering and subtree-congruence scores.
    eps : float, default=_W5_LAYERED_SHAPE_EPS
        Allowed score drop relative to the current incumbent.
    min_score : float, default=_W5_LAYERED_SHAPE_MIN_SCORE
        Absolute lower bound for available tree-shape facets.

    Returns
    -------
    bool
        ``True`` when every available facet stays within the allowed visual
        trade window. Missing values preserve legacy and non-tree callers.
    """
    for candidate_value, incumbent_value in zip(candidate, incumbent):
        if candidate_value is None or incumbent_value is None:
            continue
        floor = max(float(min_score), float(incumbent_value) - float(eps))
        if float(candidate_value) < floor:
            return False
    return True


def w5_legacy_tallied_sole_failure(
    candidate: W5ScorePair,
    incumbent: W5ScorePair,
    margin: float = _W5_ACCEPT_MARGIN,
    *,
    candidate_referee_key: Tuple[int, float] = (1, -0.0),
    incumbent_referee_key: Tuple[int, float] = (1, -0.0),
    tallied_axis: Optional[str] = None,
) -> bool:
    """Return whether only the removed legacy tallied-axis veto would reject.

    Parameters
    ----------
    candidate : W5ScorePair
        Candidate directed, undirected, V3, and optional flag scores.
    incumbent : W5ScorePair
        Incumbent directed, undirected, V3, and optional flag scores.
    margin : float, default=0.05
        Acceptance margin.
    candidate_referee_key : tuple[int, float], default=(1, -0.0)
        Severe-G6 eligibility prefix for the candidate.
    incumbent_referee_key : tuple[int, float], default=(1, -0.0)
        Severe-G6 eligibility prefix for the incumbent.
    tallied_axis : str, optional
        Legacy composite axis previously used as the V3 non-regression veto.

    Returns
    -------
    bool
        ``True`` when the V3 margin and new flag guard pass, but the removed
        tallied-axis conjunct would have rejected the candidate.
    """
    if candidate_referee_key != incumbent_referee_key:
        return False
    candidate_v3 = candidate.v3
    incumbent_v3 = incumbent.v3
    if (
        candidate_v3 is None
        or incumbent_v3 is None
        or not math.isfinite(float(candidate_v3))
        or not math.isfinite(float(incumbent_v3))
        or float(candidate_v3) <= float(incumbent_v3) + margin
        or candidate_introduces_champion_ineligible_flag(
            candidate.champion_ineligibility_flags,
            incumbent.champion_ineligibility_flags,
        )
    ):
        return False
    axis = "directed" if tallied_axis == "directed" else "undirected"
    candidate_tallied = candidate.directed if axis == "directed" else candidate.undirected
    incumbent_tallied = incumbent.directed if axis == "directed" else incumbent.undirected
    return (
        not math.isfinite(candidate_tallied)
        or not math.isfinite(incumbent_tallied)
        or candidate_tallied < incumbent_tallied - margin
    )


def w5_dominates(
    candidate: W5ScorePair,
    incumbent: W5ScorePair,
    margin: float = _W5_ACCEPT_MARGIN,
    *,
    candidate_referee_key: Tuple[int, float] = (1, -0.0),
    incumbent_referee_key: Tuple[int, float] = (1, -0.0),
    tallied_axis: Optional[str] = None,
    preserve_layered_reading: bool = False,
    layered_reading_eps: float = _W5_LAYERED_READING_EPS,
) -> bool:
    """Return whether ``candidate`` beats ``incumbent`` under the accept gate.

    Parameters
    ----------
    candidate : W5ScorePair
        Candidate directed and undirected scores.
    incumbent : W5ScorePair
        Current winner directed and undirected scores.
    margin : float, default=0.05
        Required improvement in both score components.
    candidate_referee_key : tuple[int, float], default=(1, -0.0)
        Severe-G6 eligibility prefix for the candidate. The neutral default
        preserves the historical score-only W5 gate.
    incumbent_referee_key : tuple[int, float], default=(1, -0.0)
        Severe-G6 eligibility prefix for the incumbent/current winner.
    tallied_axis : str, optional
        Legacy composite axis retained for telemetry compatibility.
    preserve_layered_reading : bool, default=False
        Whether to reject V3 improvements that degrade frozen G1 layered
        reading facets beyond ``layered_reading_eps``.
    layered_reading_eps : float, default=_W5_LAYERED_READING_EPS
        Allowed numerical slack for layered-reading facet comparisons.

    Returns
    -------
    bool
        ``True`` only when both finite components clear ``margin``.
    """
    if preserve_layered_reading and not _layered_reading_preserved(
        candidate,
        incumbent,
        layered_reading_eps,
    ):
        return False
    if candidate_referee_key != incumbent_referee_key:
        return candidate_referee_key > incumbent_referee_key
    candidate_v3 = candidate.v3
    incumbent_v3 = incumbent.v3
    if (
        candidate_v3 is not None
        and incumbent_v3 is not None
        and math.isfinite(float(candidate_v3))
        and math.isfinite(float(incumbent_v3))
    ):
        if (
            candidate.champion_ineligibility_flags is None
            or incumbent.champion_ineligibility_flags is None
        ):
            _LOGGER.debug(
                "w5_dominates V3 branch received missing champion ineligibility flags",
                extra={
                    "candidate_flags_missing": candidate.champion_ineligibility_flags is None,
                    "incumbent_flags_missing": incumbent.champion_ineligibility_flags is None,
                    "tallied_axis": tallied_axis,
                },
            )
        if candidate_introduces_champion_ineligible_flag(
            candidate.champion_ineligibility_flags,
            incumbent.champion_ineligibility_flags,
        ):
            return False
        return float(candidate_v3) > float(incumbent_v3) + margin
    return (
        math.isfinite(candidate.directed)
        and math.isfinite(candidate.undirected)
        and candidate.directed > incumbent.directed + margin
        and candidate.undirected > incumbent.undirected + margin
    )


def _w5_dominates_with_axis(
    candidate: W5ScorePair,
    incumbent: W5ScorePair,
    margin: float,
    *,
    candidate_referee_key: Tuple[int, float],
    incumbent_referee_key: Tuple[int, float],
    tallied_axis: str,
    preserve_layered_reading: bool = False,
    layered_reading_eps: float = _W5_LAYERED_READING_EPS,
) -> bool:
    """Call the W5 dominance gate with legacy monkeypatch compatibility.

    Parameters
    ----------
    candidate : W5ScorePair
        Candidate score pair.
    incumbent : W5ScorePair
        Incumbent score pair.
    margin : float
        Acceptance margin.
    candidate_referee_key : tuple[int, float]
        Severe-G6 prefix for the candidate.
    incumbent_referee_key : tuple[int, float]
        Severe-G6 prefix for the incumbent.
    tallied_axis : str
        Legacy composite axis for the V3 non-regression guardrail.
    preserve_layered_reading : bool, default=False
        Whether declared-layered rows must preserve frozen G1 reading facets.
    layered_reading_eps : float, default=_W5_LAYERED_READING_EPS
        Allowed numerical slack for layered-reading facet comparisons.

    Returns
    -------
    bool
        Dominance decision.
    """
    try:
        return w5_dominates(
            candidate,
            incumbent,
            margin,
            candidate_referee_key=candidate_referee_key,
            incumbent_referee_key=incumbent_referee_key,
            tallied_axis=tallied_axis,
            preserve_layered_reading=preserve_layered_reading,
            layered_reading_eps=layered_reading_eps,
        )
    except TypeError as exc:
        if (
            "tallied_axis" not in str(exc)
            and "preserve_layered_reading" not in str(exc)
            and "layered_reading_eps" not in str(exc)
        ):
            raise
        return w5_dominates(
            candidate,
            incumbent,
            margin,
            candidate_referee_key=candidate_referee_key,
            incumbent_referee_key=incumbent_referee_key,
        )


def _remaining_s(config: Optional[LayoutConfig]) -> Optional[float]:
    """Return remaining benchmark seconds when a deadline is known.

    Parameters
    ----------
    config : LayoutConfig, optional
        Prepared layout configuration.

    Returns
    -------
    float or None
        Remaining seconds, or ``None`` outside benchmark deadline mode.
    """
    return remaining_wall_s(config)


def _w5_first_score_epilogue_has_budget(config: Optional[LayoutConfig]) -> bool:
    """Return whether one terminal checkpoint score has deterministic budget.

    Parameters
    ----------
    config : LayoutConfig, optional
        Prepared layout configuration carrying benchmark deadline metadata.

    Returns
    -------
    bool
        ``True`` when no benchmark budget exists, or when the deterministic
        process-time budget can fit one referee score with a safety margin.
        Wall-clock only vetoes when the hard return reserve is already gone.
    """
    if wall_reserve_exhausted(config, _ABSOLUTE_DEADLINE_RESERVE_S):
        return False
    referee_s = float(
        getattr(config, "_dagua_native_w5_referee_cost_s", _MEASURED_COST_DEFAULT_REFEREE_S)
    )
    referee_s = max(1.0e-6, referee_s)
    process_remaining = _process_remaining_s(config)
    return process_remaining is None or process_remaining > 2.0 * referee_s + 1.0


def _w5_first_score_epilogue_has_wall_headroom(config: Optional[LayoutConfig]) -> bool:
    """Return whether one terminal checkpoint score may run.

    Parameters
    ----------
    config : LayoutConfig, optional
        Prepared layout configuration carrying benchmark deadline metadata.

    Returns
    -------
    bool
        Alias for the deterministic epilogue budget gate, kept for existing
        tests and callers that monkeypatch the historical helper name.
    """
    return _w5_first_score_epilogue_has_budget(config)


def _stack_graph_name() -> Optional[str]:
    """Return a benchmark graph name discovered from active driver frames.

    Returns
    -------
    str or None
        Graph name from ``scripts/run_benchmark.py``/``dagua.eval.benchmark``
        locals when W5 is executing under those drivers, otherwise ``None``.
    """
    frame = inspect.currentframe()
    current = None if frame is None else frame.f_back
    try:
        while current is not None:
            locals_by_name: dict[str, Any] = current.f_locals
            work_item = locals_by_name.get("work_item")
            graph_name = getattr(work_item, "graph_name", None)
            if graph_name is not None:
                return str(graph_name)
            test_graph = locals_by_name.get("test_graph")
            graph_name = getattr(test_graph, "name", None)
            if graph_name is not None:
                return str(graph_name)
            benchmark_graph = locals_by_name.get("bg")
            graph_name = getattr(getattr(benchmark_graph, "test_graph", None), "name", None)
            if graph_name is not None:
                return str(graph_name)
            current = current.f_back
        return None
    finally:
        del frame
        del current


def _graph_name(config: Optional[LayoutConfig]) -> Optional[str]:
    """Return the benchmark graph name when it was attached to ``config``.

    Parameters
    ----------
    config : LayoutConfig, optional
        Prepared layout configuration.

    Returns
    -------
    str or None
        Stable graph name, or ``None`` outside benchmark/name-aware callers.
    """
    name = getattr(config, _GRAPH_NAME_ATTR, None) if config is not None else None
    if name is not None:
        return str(name)
    return _stack_graph_name()


def _process_remaining_s(config: Optional[LayoutConfig]) -> Optional[float]:
    """Return process-time seconds remaining for optional W5 admission gates.

    Wall-clock deadline checks remain the hard return guard. This helper
    gives optional work a CPU-time ruler so sibling-worker contention does
    not change late-entry or predicted-cost admission after the process
    deadline is initialized.

    Parameters
    ----------
    config : LayoutConfig, optional
        Prepared layout configuration carrying benchmark deadline metadata.

    Returns
    -------
    float or None
        Process CPU seconds remaining, or ``None`` without benchmark budget
        metadata.
    """
    return remaining_process_s(config)


def _w5_disabled_by_env() -> bool:
    """Return whether W5 is disabled by the diagnostic environment flag.

    Returns
    -------
    bool
        ``True`` when ``DAGUA_NATIVE_DISABLE_W5`` is set to a truthy value.
    """
    raw = os.environ.get(_DISABLE_W5_ENV, "")
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def _w5_spend_cap_s(config: Optional[LayoutConfig], remaining: Optional[float]) -> float:
    """Return the per-layout accumulated W5 deterministic spend cap.

    Parameters
    ----------
    config : LayoutConfig, optional
        Prepared layout configuration carrying benchmark budget metadata.
    remaining : float, optional
        Remaining deterministic benchmark seconds, used as a fallback budget
        outside normal benchmark configuration.

    Returns
    -------
    float
        Maximum total process seconds W5 may spend for this layout invocation.
    """
    if config is None or (
        remaining is None and not hasattr(config, "_dagua_native_total_budget_s")
    ):
        return _DEFAULT_FINISHER_SLICE_S
    fallback_budget = _DEFAULT_FINISHER_SLICE_S if remaining is None else remaining
    total_budget = float(getattr(config, "_dagua_native_total_budget_s", fallback_budget))
    return min(_MAX_W5_SPEND_S, _TOTAL_BUDGET_FRACTION * total_budget)


def _w5_spent_s(config: Optional[LayoutConfig], started_perf: Optional[float] = None) -> float:
    """Return accumulated W5 seconds, including this invocation when supplied.

    Parameters
    ----------
    config : LayoutConfig, optional
        Prepared layout configuration carrying accumulated W5 runtime.
    started_perf : float, optional
        ``time.perf_counter()`` value for the active invocation.

    Returns
    -------
    float
        Accumulated W5 wall-clock seconds.
    """
    previous = float(getattr(config, "_dagua_native_w5_spent_s", 0.0))
    if started_perf is None:
        return previous
    return previous + max(0.0, time.perf_counter() - started_perf)


def _w5_process_spent_s(
    config: Optional[LayoutConfig],
    started_process: Optional[float] = None,
) -> float:
    """Return accumulated W5 process seconds, including this invocation.

    Parameters
    ----------
    config : LayoutConfig, optional
        Prepared layout configuration carrying accumulated process runtime.
    started_process : float, optional
        ``time.process_time()`` value for the active invocation.

    Returns
    -------
    float
        Accumulated W5 process seconds.
    """
    previous = float(getattr(config, "_dagua_native_w5_process_spent_s", 0.0))
    if started_process is None:
        return previous
    return previous + max(0.0, time.process_time() - started_process)


def _use_tiny_row_deterministic_w5_costs(
    config: Optional[LayoutConfig],
    node_count: int,
) -> bool:
    """Return whether tiny-row W5 admission should use fixed cost units.

    Parameters
    ----------
    config : LayoutConfig, optional
        Prepared layout configuration carrying optional benchmark metadata.
    node_count : int
        Number of layout nodes in the W5 seed.

    Returns
    -------
    bool
        ``True`` for tiny benchmark-budgeted rows where measured process-time
        step costs are too small and noisy to be a stable admission unit.
    """
    if config is None or int(node_count) > _MEASURED_COST_TINY_MAX_N:
        return False
    if remaining_dwu(config) is not None:
        return True
    return (
        getattr(config, DETERMINISTIC_BUDGET_ATTR, None) is not None
        and getattr(config, PROCESS_DEADLINE_ATTR, None) is not None
    )


def _use_deterministic_w5_costs(config: Optional[LayoutConfig], node_count: int) -> bool:
    """Return whether W5 cost planning is independent of wall-clock timings.

    Parameters
    ----------
    config : LayoutConfig, optional
        Prepared native configuration carrying optional deterministic budget
        metadata.
    node_count : int
        Number of layout nodes in the W5 seed.

    Returns
    -------
    bool
        ``True`` when a modeled-work ledger is installed, or when the legacy
        tiny-row deterministic process-budget path is active.
    """
    if remaining_dwu(config) is not None:
        return True
    return _use_tiny_row_deterministic_w5_costs(config, node_count)


def _native_device_class(config: Optional[LayoutConfig]) -> str:
    """Return the native cost-model device class for a W5 config.

    Parameters
    ----------
    config : LayoutConfig, optional
        Prepared native configuration carrying a device string.

    Returns
    -------
    str
        ``"cuda"`` for CUDA devices, otherwise ``"cpu"``.
    """
    device = str(getattr(config, "device", "cpu")) if config is not None else "cpu"
    return "cuda" if device.startswith("cuda") else "cpu"


def _charge_w5_owner_plan(
    config: Optional[LayoutConfig],
    node_count: int,
    edge_count: int,
    mode: str,
    cost_plan: W5CostPlan,
) -> None:
    """Charge the deterministic W5 owner package for an admitted plan.

    Parameters
    ----------
    config : LayoutConfig, optional
        Prepared native configuration carrying an optional ledger.
    node_count : int
        Number of layout nodes.
    edge_count : int
        Number of layout edges.
    mode : str
        Routed W5 mode for the first admitted plan.
    cost_plan : W5CostPlan
        Deterministic W5 plan containing seeds, steps, and checkpoints.

    Returns
    -------
    None
        The function charges W5 generation when installed and marks the plan so
        repeated local checks do not double-debit it. Referee evaluations are
        charged by the runtime V3 scorer as they happen.
    """
    if config is None or bool(getattr(config, "_dagua_native_w5_owner_charged", False)):
        return
    problem = {"num_nodes": int(node_count), "num_edges": int(edge_count)}
    cost = estimate_native_work_cost(
        problem,
        "w5",
        {
            "mode": mode,
            "steps": int(cost_plan.steps),
            "seeds": int(cost_plan.seeds),
            "checkpoints": int(cost_plan.checkpoints) * int(cost_plan.scale_search_evals),
        },
        _native_device_class(config),
    )
    charge(config, cost.generation_dwu, "mandatory_w5_owner")
    setattr(config, "_dagua_native_w5_owner_charged", True)


def w5_predicted_skip_reason(
    node_count: int,
    edge_count: int,
    config: Optional[LayoutConfig],
) -> Optional[str]:
    """Return a predicted-cost skip reason before W5 does expensive work.

    Parameters
    ----------
    node_count : int
        Number of graph nodes.
    edge_count : int
        Number of graph edges.
    config : LayoutConfig, optional
        Prepared layout configuration carrying benchmark deadline metadata.

    Returns
    -------
    str or None
        Skip reason when W5 should preserve the incumbent without running, or
        ``None`` when W5 may proceed.
    """
    del node_count, edge_count
    if _w5_disabled_by_env():
        return "disabled_by_env"
    if bool(getattr(config, "_dagua_native_w5_measured_sizing", False)):
        return None
    remaining = _process_remaining_s(config)
    if remaining is not None and remaining < _PREDICTED_COST_LATE_ENTRY_REMAINING_S:
        return "predicted_cost_late_entry"
    return None


def _finisher_slice_s(config: Optional[LayoutConfig]) -> Optional[float]:
    """Return the bounded W5 work slice, or ``None`` when it must skip.

    Parameters
    ----------
    config : LayoutConfig, optional
        Prepared layout configuration.

    Returns
    -------
    float or None
        Available finisher seconds before the return reserve.
    """
    if _w5_disabled_by_env():
        return None
    process_remaining = _process_remaining_s(config)
    if process_remaining is None and _remaining_s(config) is None:
        return _DEFAULT_FINISHER_SLICE_S
    if wall_reserve_exhausted(config, _ABSOLUTE_DEADLINE_RESERVE_S):
        return None
    if process_remaining is None:
        return None
    if process_remaining < _MIN_BENCHMARK_REMAINING_S:
        return None
    available = max(0.0, float(process_remaining) - _ABSOLUTE_DEADLINE_RESERVE_S)
    if available < _MIN_FINISHER_ENTRY_S + _FINISHER_SCORE_RESERVE_S:
        return None
    spend_cap = _w5_spend_cap_s(config, process_remaining)
    ledger_remaining = remaining_dwu(config)
    spent = 0.0 if ledger_remaining is not None else _w5_process_spent_s(config)
    remaining_w5_budget = spend_cap - spent
    if remaining_w5_budget < _MIN_FINISHER_ENTRY_S + _FINISHER_SCORE_RESERVE_S:
        return None
    return max(0.0, min(remaining_w5_budget, available))


def make_w5_skip_result(
    *,
    incumbent_pos: torch.Tensor,
    incumbent_score_pair: Optional[W5ScorePair],
    reason: str,
    edge_index: Optional[torch.Tensor] = None,
    config: Optional[LayoutConfig] = None,
    is_semantically_directed: bool = False,
    declared_hierarchical: bool = False,
    direction_is_declared: bool = False,
) -> W5FinisherResult:
    """Build a telemetry-friendly W5 skip result without doing W5 pre-work.

    Parameters
    ----------
    incumbent_pos : torch.Tensor
        Incumbent positions with shape ``[N, 2]``.
    incumbent_score_pair : W5ScorePair, optional
        Precomputed incumbent score pair, when available without extra work.
    reason : str
        Skip reason to report.
    edge_index : torch.Tensor, optional
        Edge tensor with shape ``[2, E]``.
    config : LayoutConfig, optional
        Prepared layout configuration.
    is_semantically_directed : bool, default=False
        Routed directedness flag.
    declared_hierarchical : bool, default=False
        Routed hierarchy flag.
    direction_is_declared : bool, default=False
        Whether directedness was explicitly declared.

    Returns
    -------
    W5FinisherResult
        Incumbent winner result with skip metadata.
    """
    fallback_score = incumbent_score_pair or W5ScorePair(float("nan"), float("nan"))
    return W5FinisherResult(
        winner_pos=incumbent_pos,
        incumbent_score_pair=fallback_score,
        winner_score_pair=fallback_score,
        winner_name="incumbent",
        deadline_returned=reason in {"no_budget", "deadline"},
        accepted=(),
        rejected=(),
        checkpoints=(),
        mode="skip",
        steps=0,
        skipped_reason=reason,
        slice_s=None,
        spent_s=0.0,
        remaining_entry_s=_remaining_s(config),
        remaining_exit_s=_remaining_s(config),
        node_count=int(incumbent_pos.shape[0]),
        edge_count=(
            int(edge_index.shape[1]) if edge_index is not None and edge_index.ndim == 2 else 0
        ),
        is_semantically_directed=is_semantically_directed,
        declared_hierarchical=declared_hierarchical,
        direction_is_declared=direction_is_declared,
        graph_name=_graph_name(config),
    )


def _dedupe_seeds(seeds: Sequence[W5Seed], max_seeds: int = 3) -> list[W5Seed]:
    """Return finite, non-duplicate W5 seeds.

    Parameters
    ----------
    seeds : Sequence[W5Seed]
        Candidate warm starts.
    max_seeds : int, default=3
        Maximum seeds to keep.

    Returns
    -------
    list[W5Seed]
        Deduplicated finite seeds.
    """
    kept: list[W5Seed] = []
    for seed in seeds:
        if len(kept) >= max_seeds or not bool(torch.isfinite(seed.pos).all().item()):
            continue
        if any(
            torch.allclose(seed.pos, existing.pos, atol=1.0e-5, rtol=1.0e-5) for existing in kept
        ):
            continue
        kept.append(seed)
    return kept


def _longest_path_depth(
    edge_index: torch.Tensor,
    node_count: int,
    device: torch.device,
) -> torch.Tensor:
    """Return longest-path depths, falling back to zeros on cyclic inputs.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    node_count : int
        Number of nodes.
    device : torch.device
        Output device.

    Returns
    -------
    torch.Tensor
        Depth tensor with shape ``[N]``.
    """
    try:
        from dagua.utils import longest_path_layering

        depth = longest_path_layering(edge_index.detach().to(device="cpu"), node_count)
        if not isinstance(depth, torch.Tensor):
            depth = torch.as_tensor(depth, dtype=torch.long)
        return depth.to(device=device, dtype=torch.long)
    except Exception:  # noqa: BLE001 -- cyclic semantic graphs use flow-only barriers
        return torch.zeros(node_count, dtype=torch.long, device=device)


def _route_mode(
    seed_pos: torch.Tensor,
    edge_index: torch.Tensor,
    topo_depth: torch.Tensor,
    *,
    is_semantically_directed: bool,
    declared_hierarchical: bool,
    direction_is_declared: bool,
    honest_axes: Optional[W5HonestAxes] = None,
) -> str:
    """Choose the W5 descent mode from honest incumbent axes.

    Parameters
    ----------
    seed_pos : torch.Tensor
        Seed positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    topo_depth : torch.Tensor
        Depth tensor with shape ``[N]``.
    is_semantically_directed : bool
        Whether edge direction has semantic meaning.
    declared_hierarchical : bool
        Whether the honest ruler treats the graph as hierarchical.
    direction_is_declared : bool
        Whether semantic direction came from user/config metadata.
    honest_axes : W5HonestAxes, optional
        Frozen-ruler incumbent axis scores. When absent, W5 falls back to the
        old surrogate route for compatibility with direct unit callers.

    Returns
    -------
    str
        One of ``"x_only"``, ``"barrier_2d"``, or ``"undirected_2d_sampled"``.
    """
    if not is_semantically_directed or not direction_is_declared:
        return "undirected_2d_sampled"
    if not declared_hierarchical:
        return "barrier_2d"
    # The mode decision must use the same honest axes as the accept ruler:
    # a high surrogate flow self-report cannot route a flow-deficient
    # incumbent into x_only, where y-motion is frozen and the flow gap is
    # unreachable. The dominance gate below remains the monotone safety rail.
    if honest_axes is not None:
        flow = 0.0 if honest_axes.flow is None else float(honest_axes.flow)
        depth = 0.0 if honest_axes.depth is None else float(honest_axes.depth)
    else:
        flow = float(signed_flow_score_surrogate(seed_pos, edge_index).detach().item())
        depth = float(depth_order_score_surrogate(seed_pos, topo_depth).detach().item())
    node_count = int(seed_pos.shape[0])
    if flow >= 0.95 and depth >= 0.95 and (node_count <= 64 or node_count >= 250):
        return "x_only"
    return "barrier_2d"


def _mode_ladder(mode: str, *, is_semantically_directed: bool) -> tuple[str, ...]:
    """Return the W5 mode ladder for one seed.

    Parameters
    ----------
    mode : str
        Initially routed mode.
    is_semantically_directed : bool
        Whether edge direction has semantic meaning.

    Returns
    -------
    tuple[str, ...]
        Modes to try in order. The second directed pass is only useful for
        ``x_only`` failures; it gives the same seed y-motion without changing
        the monotone accept gate.
    """
    if is_semantically_directed and mode == "x_only":
        return ("x_only", "barrier_2d")
    return (mode,)


def _overlap_count(
    pos: torch.Tensor,
    node_sizes: torch.Tensor,
    shape_geometry: Optional[NativeShapeGeometry] = None,
) -> int:
    """Count exact overlapping boxes for W5 regression rejection.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    node_sizes : torch.Tensor
        Node-size tensor with shape ``[N, 2]``.
    shape_geometry : NativeShapeGeometry, optional
        Optional non-box shape descriptors.

    Returns
    -------
    int
        Number of overlapping unordered node pairs.
    """
    if pos.shape[0] <= 1:
        return 0
    if shape_geometry is not None:
        gaps = pairwise_shape_signed_gap(
            pos.detach().to(device="cpu", dtype=torch.float32),
            node_sizes.detach().to(device="cpu", dtype=torch.float32),
            shape_geometry.to(device=torch.device("cpu"), dtype=torch.float32),
            max_nodes=int(pos.shape[0]),
        )
        return int((gaps < 0.0).sum().item())
    work_pos = pos.detach().to(device="cpu", dtype=torch.float32)
    work_sizes = node_sizes.detach().to(device="cpu", dtype=torch.float32)
    dx = (work_pos[:, None, 0] - work_pos[None, :, 0]).abs()
    dy = (work_pos[:, None, 1] - work_pos[None, :, 1]).abs()
    min_dx = (work_sizes[:, None, 0] + work_sizes[None, :, 0]) * 0.5
    min_dy = (work_sizes[:, None, 1] + work_sizes[None, :, 1]) * 0.5
    overlap = (dx < min_dx) & (dy < min_dy)
    overlap.fill_diagonal_(False)
    return int(overlap.triu(diagonal=1).sum().item())


def _is_degenerate(pos: torch.Tensor, node_sizes: torch.Tensor) -> bool:
    """Return whether positions are non-finite or collapsed.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    node_sizes : torch.Tensor
        Node-size tensor with shape ``[N, 2]``.

    Returns
    -------
    bool
        ``True`` when the candidate is not safe to score.
    """
    if not bool(torch.isfinite(pos).all().item()):
        return True
    if pos.shape[0] <= 1:
        return False
    extent = pos.detach().amax(dim=0) - pos.detach().amin(dim=0)
    min_extent = float(node_sizes.detach().to(dtype=torch.float32).mean().item()) * 0.1
    return float(extent.max().item()) <= max(1.0e-6, min_extent)


def _project_checkpoint_for_viability(
    checkpoint_pos: torch.Tensor,
    node_sizes: torch.Tensor,
    shape_geometry: Optional[NativeShapeGeometry] = None,
) -> torch.Tensor:
    """Return an overlap-projected checkpoint for W5 viability checks.

    Parameters
    ----------
    checkpoint_pos : torch.Tensor
        Candidate checkpoint positions with shape ``[N, 2]``.
    node_sizes : torch.Tensor
        Node-size tensor with shape ``[N, 2]``.
    shape_geometry : NativeShapeGeometry, optional
        Optional non-box shape descriptors.

    Returns
    -------
    torch.Tensor
        Projected checkpoint positions with shape ``[N, 2]``.
    """
    projected = checkpoint_pos.detach().clone().to(dtype=torch.float32)
    sizes = node_sizes.detach().to(device=projected.device, dtype=projected.dtype)
    project_overlaps(
        projected,
        sizes,
        iterations=_W5_PROJECTION_ITERATIONS,
        convergent=True,
    )
    if shape_geometry is not None:
        _project_shape_overlaps(projected, sizes, shape_geometry)
    return projected


def _project_shape_overlaps(
    pos: torch.Tensor,
    node_sizes: torch.Tensor,
    shape_geometry: NativeShapeGeometry,
    *,
    iterations: int = _W5_PROJECTION_ITERATIONS,
) -> None:
    """Resolve true-shape overlaps with deterministic pairwise radial shifts.

    Parameters
    ----------
    pos : torch.Tensor
        Mutable position tensor with shape ``[N, 2]``.
    node_sizes : torch.Tensor
        Node sizes with shape ``[N, 2]``.
    shape_geometry : NativeShapeGeometry
        Per-node shape descriptors.
    iterations : int, default=_W5_PROJECTION_ITERATIONS
        Maximum repair sweeps.

    Returns
    -------
    None
        ``pos`` is modified in place.
    """
    if int(pos.shape[0]) < 2:
        return
    geometry = shape_geometry.to(device=pos.device, dtype=pos.dtype)
    for _ in range(iterations):
        moved = False
        for left in range(int(pos.shape[0]) - 1):
            for right in range(left + 1, int(pos.shape[0])):
                gap = pairwise_shape_signed_gap(
                    pos[[left, right]],
                    node_sizes[[left, right]],
                    NativeShapeGeometry(kind_codes=geometry.kind_codes[[left, right]]),
                    max_nodes=2,
                )
                if gap.numel() == 0 or float(gap[0].item()) >= 0.0:
                    continue
                delta = pos[right] - pos[left]
                distance = torch.linalg.vector_norm(delta).clamp_min(1.0e-6)
                if float(distance.item()) <= 1.0e-5:
                    angle = float(left * 92821 + right * 68917) * 0.0001
                    direction = pos.new_tensor((math.cos(angle), math.sin(angle)))
                else:
                    direction = delta / distance
                shift = direction * ((-gap[0] + 1.0e-4) * 0.5)
                pos[left] -= shift
                pos[right] += shift
                moved = True
        if not moved:
            break


def _closed_over_all_pairs_dist(score_fn: Callable[[torch.Tensor], W5ScorePair]) -> Optional[Any]:
    """Return ``all_pairs_dist`` transitively captured by the honest scorer.

    Parameters
    ----------
    score_fn : Callable[[torch.Tensor], W5ScorePair]
        Honest scoring closure built by the terminal native path.

    Returns
    -------
    object or None
        Existing all-pairs distance matrix from the scorer closure. ``None``
        means pass 2 must omit the stress term rather than compute APSP here.
    """

    def search_closure(fn: object, visited: set[int]) -> Optional[Any]:
        """Search one closure chain for the APSP matrix without invoking code.

        Parameters
        ----------
        fn : object
            Candidate function-valued object whose closure may capture
            ``all_pairs_dist``.
        visited : set[int]
            Object identities already scanned, used to guard closure cycles.

        Returns
        -------
        object or None
            Captured APSP matrix when present.
        """
        fn_id = id(fn)
        if fn_id in visited:
            return None
        visited.add(fn_id)
        code = getattr(fn, "__code__", None)
        closure = getattr(fn, "__closure__", None)
        if code is None or closure is None:
            return None
        function_cells: list[object] = []
        for name, cell in zip(code.co_freevars, closure):
            try:
                value = cell.cell_contents
            except ValueError:
                continue
            if name == "all_pairs_dist":
                return value
            if callable(value) and getattr(value, "__closure__", None) is not None:
                function_cells.append(value)
        for value in function_cells:
            nested = search_closure(value, visited)
            if nested is not None:
                return nested
        return None

    return search_closure(score_fn, set())


def _build_w5_stress_sample(
    edge_index: torch.Tensor,
    node_count: int,
    all_pairs_dist: Optional[Any],
    device: torch.device,
) -> Optional[W5StressSample]:
    """Build a fixed capped stress sample from an existing APSP matrix.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    node_count : int
        Number of graph nodes.
    all_pairs_dist : object, optional
        Precomputed all-pairs graph distances. No sample is built when this is
        absent, preserving the no-new-APSP invariant.
    device : torch.device
        Device for returned index and target tensors.

    Returns
    -------
    W5StressSample or None
        Deterministic detached stress sample, capped at ``_W5_STRESS_MAX_PAIRS``.
    """
    if all_pairs_dist is None or node_count < 2 or edge_index.numel() == 0:
        return None
    try:
        from dagua.metrics import _deterministic_sample_indices, _stratified_graph_pairs

        n_targets = max(1, _W5_STRESS_MAX_PAIRS // _W5_STRESS_MAX_SOURCES)
        sources, targets, graph_distances = _stratified_graph_pairs(
            edge_index.detach().to(device="cpu", dtype=torch.long),
            int(node_count),
            _W5_STRESS_MAX_SOURCES,
            n_targets,
            all_pairs_dist=all_pairs_dist,
        )
        if sources.size == 0:
            return None
        if sources.size > _W5_STRESS_MAX_PAIRS:
            keep = _deterministic_sample_indices(int(sources.size), _W5_STRESS_MAX_PAIRS)
            sources = sources[keep]
            targets = targets[keep]
            graph_distances = graph_distances[keep]
        source_tensor = torch.as_tensor(sources, dtype=torch.long, device=device).detach()
        target_tensor = torch.as_tensor(targets, dtype=torch.long, device=device).detach()
        distance_tensor = torch.as_tensor(
            graph_distances,
            dtype=torch.float32,
            device=device,
        ).detach()
    except Exception:  # noqa: BLE001 -- pass-2 stress is optional candidate guidance
        return None
    if source_tensor.numel() == 0:
        return None
    return W5StressSample(
        sources=source_tensor,
        targets=target_tensor,
        graph_distances=distance_tensor.clamp_min(1.0),
    )


def _build_w5_neighborhood_sample(
    all_pairs_dist: Optional[Any],
    node_count: int,
    device: torch.device,
) -> Optional[W5NeighborhoodSample]:
    """Build deterministic multi-radius contrastive graph-neighborhood triplets.

    Parameters
    ----------
    all_pairs_dist : object, optional
        Precomputed all-pairs graph distances. No sample is built when this is
        absent, preserving the W5 no-new-APSP invariant.
    node_count : int
        Number of graph nodes.
    device : torch.device
        Device for returned index tensors.

    Returns
    -------
    W5NeighborhoodSample or None
        Fixed positive/negative triplets for radii 2 and 3, capped for bounded
        per-step cost.
    """
    if all_pairs_dist is None or node_count < 3:
        return None
    try:
        distances = torch.as_tensor(all_pairs_dist, dtype=torch.float32, device="cpu")
    except Exception:  # noqa: BLE001 -- contrastive C3 guidance is optional
        return None
    if distances.ndim != 2 or int(distances.shape[0]) != node_count:
        return None
    finite = torch.isfinite(distances)
    anchors: list[int] = []
    positives: list[int] = []
    negatives: list[int] = []
    radii: list[int] = []
    per_radius_cap = max(1, _W5_NEIGHBORHOOD_MAX_TRIPLETS // len(_W5_NEIGHBORHOOD_RADII))
    for radius in _W5_NEIGHBORHOOD_RADII:
        radius_count = 0
        for anchor in range(node_count):
            row = distances[anchor]
            positive_mask = finite[anchor] & (row > 0.0) & (row <= float(radius))
            negative_mask = finite[anchor] & (row > float(radius))
            positive_targets = torch.nonzero(positive_mask, as_tuple=False).flatten().tolist()
            negative_targets = torch.nonzero(negative_mask, as_tuple=False).flatten().tolist()
            if not positive_targets or not negative_targets:
                continue
            negative_count = len(negative_targets)
            for local_index, positive in enumerate(positive_targets):
                if radius_count >= per_radius_cap:
                    break
                negative = negative_targets[(anchor + local_index) % negative_count]
                anchors.append(anchor)
                positives.append(int(positive))
                negatives.append(int(negative))
                radii.append(int(radius))
                radius_count += 1
            if radius_count >= per_radius_cap:
                break
    if not anchors:
        return None
    return W5NeighborhoodSample(
        anchors=torch.as_tensor(anchors, dtype=torch.long, device=device).detach(),
        positives=torch.as_tensor(positives, dtype=torch.long, device=device).detach(),
        negatives=torch.as_tensor(negatives, dtype=torch.long, device=device).detach(),
        radii=torch.as_tensor(radii, dtype=torch.float32, device=device).detach(),
    )


def _mean_node_diag_tensor(pos: torch.Tensor, node_sizes: torch.Tensor) -> torch.Tensor:
    """Return mean node diagonal as a differentiable-device scalar.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]`` used only for device/dtype.
    node_sizes : torch.Tensor
        Node-size tensor with shape ``[N, 2]``.

    Returns
    -------
    torch.Tensor
        Scalar mean node diagonal clamped away from zero.
    """
    if node_sizes.numel() == 0:
        return pos.new_tensor(1.0)
    sizes = node_sizes.to(device=pos.device, dtype=pos.dtype)
    return torch.linalg.vector_norm(sizes, dim=1).mean().clamp_min(1.0e-6)


def _soft_bbox_area_band_loss(
    pos: torch.Tensor,
    node_sizes: torch.Tensor,
    structure_floor: Optional[float],
) -> torch.Tensor:
    """Penalize C5 visual-area ratios outside the V3 asymmetric log band.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    node_sizes : torch.Tensor
        Node-size tensor with shape ``[N, 2]``.
    structure_floor : float, optional
        V3 topology-aware structure area floor for the graph.

    Returns
    -------
    torch.Tensor
        Scalar loss that is zero inside the V3 C5 area band.
    """
    if pos.shape[0] == 0 or node_sizes.numel() == 0 or structure_floor is None:
        return pos.new_zeros(())
    if not math.isfinite(float(structure_floor)) or float(structure_floor) <= 0.0:
        return pos.new_zeros(())
    sizes = node_sizes.to(device=pos.device, dtype=pos.dtype)
    lower = pos - sizes * 0.5
    upper = pos + sizes * 0.5
    tau = (_W5_SOFT_BBOX_TAU_NODE_DIAG * _mean_node_diag_tensor(pos, sizes)).clamp_min(1.0e-6)
    soft_right = tau * torch.logsumexp(upper[:, 0] / tau, dim=0)
    soft_top = tau * torch.logsumexp(upper[:, 1] / tau, dim=0)
    soft_left = -tau * torch.logsumexp(-lower[:, 0] / tau, dim=0)
    soft_bottom = -tau * torch.logsumexp(-lower[:, 1] / tau, dim=0)
    area = (soft_right - soft_left).clamp_min(1.0e-6) * (soft_top - soft_bottom).clamp_min(1.0e-6)
    floor_tensor = pos.new_tensor(float(structure_floor)).clamp_min(1.0e-6)
    ratio = (area / floor_tensor).clamp_min(1.0e-12)
    crowd_distance = torch.relu(torch.log(pos.new_tensor(WHITESPACE_RATIO_LO) / ratio))
    sprawl_distance = torch.relu(torch.log(ratio / pos.new_tensor(WHITESPACE_RATIO_HI)))
    return (
        float(WHITESPACE_CROWDING_DECAY) * crowd_distance.square()
        + float(WHITESPACE_SPRAWL_DECAY) * sprawl_distance.square()
    )


def _box_pair_signed_gap(
    pos: torch.Tensor,
    node_sizes: torch.Tensor,
    max_nodes: int,
) -> torch.Tensor:
    """Return deterministic sampled AABB signed gaps for node pairs.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    node_sizes : torch.Tensor
        Node-size tensor with shape ``[N, 2]``.
    max_nodes : int
        Maximum nodes included in the all-pairs sample.

    Returns
    -------
    torch.Tensor
        Pairwise signed gaps with shape ``[P]``; negative values indicate
        overlap on both axes.
    """
    node_count = int(pos.shape[0])
    if node_count < 2:
        return pos.new_empty(0)
    if node_count > max_nodes:
        sample = torch.linspace(
            0,
            node_count - 1,
            steps=max_nodes,
            dtype=torch.float32,
            device=pos.device,
        ).round()
        sample = sample.to(dtype=torch.long).clamp(0, node_count - 1)
        work_pos = pos[sample]
        work_sizes = node_sizes.to(device=pos.device, dtype=pos.dtype)[sample]
    else:
        work_pos = pos
        work_sizes = node_sizes.to(device=pos.device, dtype=pos.dtype)
    left, right = torch.triu_indices(
        int(work_pos.shape[0]),
        int(work_pos.shape[0]),
        offset=1,
        device=pos.device,
    )
    if left.numel() == 0:
        return pos.new_empty(0)
    gap_xy = (
        torch.abs(work_pos[right] - work_pos[left]) - (work_sizes[left] + work_sizes[right]) * 0.5
    )
    positive_gap = torch.clamp(gap_xy, min=0.0)
    separated = (gap_xy > 0.0).any(dim=1)
    separated_gap = torch.linalg.vector_norm(positive_gap, dim=1)
    overlap_gap = torch.max(gap_xy, dim=1).values
    return torch.where(separated, separated_gap, overlap_gap)


def _clearance_band_hinge_loss(
    pos: torch.Tensor,
    node_sizes: torch.Tensor,
    shape_geometry: Optional[NativeShapeGeometry],
    max_nodes: int = 512,
) -> torch.Tensor:
    """Penalize sub-clearance node pairs before they become C4 overlaps.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    node_sizes : torch.Tensor
        Node-size tensor with shape ``[N, 2]``.
    shape_geometry : NativeShapeGeometry, optional
        Optional non-box shape descriptors for signed shape gaps.
    max_nodes : int, default=512
        Maximum nodes included in deterministic pair sampling.

    Returns
    -------
    torch.Tensor
        Scalar clearance-band hinge normalized by mean node diagonal.
    """
    if int(pos.shape[0]) < 2 or node_sizes.numel() == 0:
        return pos.new_zeros(())
    if shape_geometry is None:
        signed_gap = _box_pair_signed_gap(pos, node_sizes, max_nodes)
    else:
        signed_gap = pairwise_shape_signed_gap(
            pos,
            node_sizes,
            shape_geometry,
            max_nodes=max_nodes,
        )
    if signed_gap.numel() == 0:
        return pos.new_zeros(())
    mean_diag = _mean_node_diag_tensor(pos, node_sizes)
    band = max(0.5, float(C4_CLEARANCE_BAND_NODE_DIAGONALS)) * mean_diag
    hinge = torch.relu(band - signed_gap.to(device=pos.device, dtype=pos.dtype))
    return torch.nan_to_num(hinge.square().mean() / mean_diag.square(), nan=0.0, posinf=10.0)


def _contrastive_neighborhood_loss(
    pos: torch.Tensor,
    node_sizes: torch.Tensor,
    neighborhood_sample: Optional[W5NeighborhoodSample],
) -> torch.Tensor:
    """Return multi-radius C3 contrastive neighborhood preservation loss.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    node_sizes : torch.Tensor
        Node-size tensor with shape ``[N, 2]``.
    neighborhood_sample : W5NeighborhoodSample, optional
        Precomputed graph-distance triplets for radii 2 and 3.

    Returns
    -------
    torch.Tensor
        Scalar hinge loss requiring graph-near positives to be geometrically
        closer than graph-far negatives by a node-scale margin.
    """
    if neighborhood_sample is None or neighborhood_sample.anchors.numel() == 0:
        return pos.new_zeros(())
    anchors = neighborhood_sample.anchors.to(device=pos.device, dtype=torch.long)
    positives = neighborhood_sample.positives.to(device=pos.device, dtype=torch.long)
    negatives = neighborhood_sample.negatives.to(device=pos.device, dtype=torch.long)
    radii = neighborhood_sample.radii.to(device=pos.device, dtype=pos.dtype)
    positive_dist = torch.linalg.vector_norm(pos[anchors] - pos[positives], dim=1)
    negative_dist = torch.linalg.vector_norm(pos[anchors] - pos[negatives], dim=1)
    mean_diag = _mean_node_diag_tensor(pos, node_sizes)
    margin = _W5_CONTRASTIVE_MARGIN_NODE_DIAG * mean_diag * (radii / 2.0).clamp_min(1.0)
    hinge = torch.relu(positive_dist - negative_dist + margin)
    normalizer = (negative_dist.detach().mean().clamp_min(mean_diag.detach())).square()
    return torch.nan_to_num(hinge.square().mean() / normalizer, nan=0.0, posinf=10.0)


def _increment_count(counts: dict[str, int], key: str) -> None:
    """Increment ``key`` in a telemetry count dictionary.

    Parameters
    ----------
    counts : dict[str, int]
        Mutable telemetry count dictionary.
    key : str
        Count key to increment.

    Returns
    -------
    None
        ``counts`` is updated in place.
    """
    counts[key] = int(counts.get(key, 0)) + 1


def _surrogate_loss(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    topo_depth: torch.Tensor,
    mode: str,
    floors: dict[str, float],
    shape_geometry: Optional[NativeShapeGeometry] = None,
) -> torch.Tensor:
    """Evaluate the composite-weighted W5 surrogate objective.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    node_sizes : torch.Tensor
        Node-size tensor with shape ``[N, 2]``.
    topo_depth : torch.Tensor
        Depth tensor with shape ``[N]``.
    mode : str
        Routed W5 mode.
    floors : dict[str, float]
        Incumbent floor values for barrier terms.
    shape_geometry : NativeShapeGeometry, optional
        Optional non-box shape descriptors.

    Returns
    -------
    torch.Tensor
        Scalar loss to minimize.
    """
    crossing = soft_crossing_loss(pos, edge_index)
    flow_score = signed_flow_score_surrogate(pos, edge_index)
    depth_score = depth_order_score_surrogate(pos, topo_depth)
    overlap_loss = overlap_hinge_loss(pos, node_sizes, shape_geometry=shape_geometry)
    knn_loss = soft_knn_neighborhood_loss(pos, edge_index)
    edge_cv_loss = edge_length_cv_loss(pos, edge_index)
    common_scale = 0.75 if mode in {"x_only", "barrier_2d"} else 1.0
    loss = (
        common_scale * 20.0 * crossing
        + common_scale * 13.0 * overlap_loss
        + common_scale * 12.0 * knn_loss
        + common_scale * 7.0 * edge_cv_loss
        + common_scale * 5.0 * gabriel_intrusion_loss(pos, edge_index)
        + common_scale * 5.0 * crossing_angle_loss(pos, edge_index)
        + common_scale * 4.0 * angular_resolution_loss(pos, edge_index)
        + common_scale * 4.0 * path_continuity_loss(pos, edge_index)
    )
    if mode in {"x_only", "barrier_2d"}:
        loss = loss + 16.0 * (1.0 - flow_score) + 9.0 * (1.0 - depth_score)
    if mode == "barrier_2d":
        honest_flow = floors.get("honest_flow")
        flow_headroom = 0.0 if honest_flow is None else max(0.0, 1.0 - float(honest_flow))
        honest_ksm = floors.get("honest_ksm")
        ksm_floor_weight = 24.0 if honest_ksm is None else 24.0 + 24.0 * float(honest_ksm)
        edge_cv_weight = 10.0 + 42.0 * flow_headroom
        loss = (
            loss
            + (24.0 + 96.0 * flow_headroom) * (1.0 - flow_score)
            + 64.0 * barrier_floor_loss(flow_score, floors.get("flow"))
            + 36.0 * barrier_floor_loss(depth_score, floors.get("depth"))
            + 20.0 * torch.relu(crossing - floors.get("crossing_loss", crossing.detach())).square()
            + 20.0
            * torch.relu(overlap_loss - floors.get("overlap_loss", overlap_loss.detach())).square()
            + ksm_floor_weight
            * torch.relu(knn_loss - floors.get("knn_loss", knn_loss.detach())).square()
            + edge_cv_weight * edge_cv_loss
        )
    return torch.nan_to_num(loss, nan=1.0e6, posinf=1.0e6, neginf=1.0e6)


def _stress_gain_loss(pos: torch.Tensor, stress_sample: Optional[W5StressSample]) -> torch.Tensor:
    """Return a scale-fitted sampled stress loss for W5 pass 2.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    stress_sample : W5StressSample, optional
        Fixed source/target graph-distance sample.

    Returns
    -------
    torch.Tensor
        Scalar normalized stress residual. Zero is returned when no fixed
        sample is available.
    """
    if stress_sample is None or stress_sample.sources.numel() == 0:
        return pos.new_zeros(())
    geometric = torch.linalg.vector_norm(
        pos[stress_sample.sources] - pos[stress_sample.targets],
        dim=1,
    )
    targets = stress_sample.graph_distances.to(device=pos.device, dtype=pos.dtype)
    denominator = torch.dot(targets, targets).clamp_min(1.0e-12)
    scale = torch.dot(geometric, targets) / denominator
    reference = scale * targets
    residual = geometric - reference
    normalizer = geometric.detach().square().mean().clamp_min(1.0e-12)
    return residual.square().mean() / normalizer


def _edge_length_l1_deviation_loss(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Return MAD/mean edge-length deviation matching the honest metric shape.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.

    Returns
    -------
    torch.Tensor
        Scalar relative mean absolute deviation over edge lengths.
    """
    if edge_index.numel() == 0:
        return pos.new_zeros(())
    lengths = torch.linalg.vector_norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1)
    mean_length = lengths.mean().clamp_min(1.0e-12)
    return torch.abs(lengths - mean_length).mean() / mean_length.detach().clamp_min(1.0e-12)


def _aligned_surrogate_loss(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    topo_depth: torch.Tensor,
    mode: str,
    floors: dict[str, float],
    stress_sample: Optional[W5StressSample],
    neighborhood_sample: Optional[W5NeighborhoodSample],
    pass_id: int,
    shape_geometry: Optional[NativeShapeGeometry] = None,
) -> torch.Tensor:
    """Evaluate the W5 objective with V3-facet-aligned additive terms.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    node_sizes : torch.Tensor
        Node-size tensor with shape ``[N, 2]``.
    topo_depth : torch.Tensor
        Depth tensor with shape ``[N]``.
    mode : str
        Routed W5 mode.
    floors : dict[str, float]
        Incumbent floor values for barrier terms.
    stress_sample : W5StressSample, optional
        Fixed sampled graph-distance pairs for C1 stress guidance.
    neighborhood_sample : W5NeighborhoodSample, optional
        Fixed graph-neighborhood triplets for C3 contrastive guidance.
    pass_id : int
        Surrogate pass identifier. Pass 1 uses moderate C1/C3 weights; pass 2
        increases the honest-aligned continuation terms.
    shape_geometry : NativeShapeGeometry, optional
        Optional non-box shape descriptors.

    Returns
    -------
    torch.Tensor
        Scalar loss to minimize during pass 2.
    """
    loss = _surrogate_loss(
        pos,
        edge_index,
        node_sizes,
        topo_depth,
        mode,
        floors,
        shape_geometry,
    )
    honest_ksm = floors.get("honest_ksm")
    ksm_headroom = 0.0 if honest_ksm is None else max(0.0, 1.0 - float(honest_ksm))
    if pass_id == 1:
        stress_weight = _W5_PASS1_STRESS_WEIGHT
        contrastive_weight = _W5_PASS1_CONTRASTIVE_WEIGHT
    else:
        stress_weight = (
            _W5_PASS2_STRESS_BASE_WEIGHT + _W5_PASS2_STRESS_HEADROOM_WEIGHT * ksm_headroom
        )
        contrastive_weight = _W5_PASS2_CONTRASTIVE_WEIGHT
    edge_l1_weight = 6.0
    loss = (
        loss
        + _W5_C5_SOFT_BBOX_WEIGHT
        * _soft_bbox_area_band_loss(pos, node_sizes, floors.get("c5_structure_floor"))
        + _W5_C4_CLEARANCE_BAND_WEIGHT * _clearance_band_hinge_loss(pos, node_sizes, shape_geometry)
        + stress_weight * _stress_gain_loss(pos, stress_sample)
        + contrastive_weight
        * _contrastive_neighborhood_loss(
            pos,
            node_sizes,
            neighborhood_sample,
        )
        + edge_l1_weight * _edge_length_l1_deviation_loss(pos, edge_index)
    )
    return torch.nan_to_num(loss, nan=1.0e6, posinf=1.0e6, neginf=1.0e6)


def _checkpoint_steps(desired_steps: int, max_checkpoints: int) -> set[int]:
    """Return deterministic checkpoint steps for a bounded optimizer pass.

    Parameters
    ----------
    desired_steps : int
        Planned optimizer steps for the pass.
    max_checkpoints : int
        Maximum number of checkpoints to score.

    Returns
    -------
    set[int]
        Pass-local step indices to checkpoint.
    """
    if max_checkpoints <= 0:
        return set()
    if max_checkpoints == 1:
        return {max(1, int(desired_steps))}
    return {
        int(desired_steps)
        if index == max_checkpoints
        else max(1, int(desired_steps) * index // max_checkpoints)
        for index in range(1, max_checkpoints + 1)
    }


def _pass_loss(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    topo_depth: torch.Tensor,
    mode: str,
    floors: dict[str, float],
    pass_id: int,
    stress_sample: Optional[W5StressSample],
    neighborhood_sample: Optional[W5NeighborhoodSample],
    shape_geometry: Optional[NativeShapeGeometry] = None,
) -> torch.Tensor:
    """Evaluate the selected W5 pass loss.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    node_sizes : torch.Tensor
        Node-size tensor with shape ``[N, 2]``.
    topo_depth : torch.Tensor
        Depth tensor with shape ``[N]``.
    mode : str
        Routed W5 mode.
    floors : dict[str, float]
        Incumbent floor values for barrier terms.
    pass_id : int
        Surrogate pass identifier.
    stress_sample : W5StressSample, optional
        Fixed sampled graph-distance pairs for stress guidance.
    neighborhood_sample : W5NeighborhoodSample, optional
        Fixed graph-neighborhood triplets for contrastive C3 guidance.
    shape_geometry : NativeShapeGeometry, optional
        Optional non-box shape descriptors.

    Returns
    -------
    torch.Tensor
        Scalar loss for the requested pass.
    """
    return _aligned_surrogate_loss(
        pos,
        edge_index,
        node_sizes,
        topo_depth,
        mode,
        floors,
        stress_sample,
        neighborhood_sample,
        pass_id,
        shape_geometry,
    )


def _optimize_seed(
    seed: W5Seed,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    topo_depth: torch.Tensor,
    mode: str,
    deadline: float,
    honest_axes: Optional[W5HonestAxes] = None,
    max_steps: Optional[int] = None,
    max_checkpoints: int = _MEASURED_COST_MAX_CHECKPOINTS,
    step_timing_hook: Optional[Callable[[int, float], None]] = None,
    pass_id: int = 1,
    stress_sample: Optional[W5StressSample] = None,
    neighborhood_sample: Optional[W5NeighborhoodSample] = None,
    shape_geometry: Optional[NativeShapeGeometry] = None,
) -> tuple[torch.Tensor, int, float, list[tuple[int, torch.Tensor, float]]]:
    """Run one bounded W5 descent from ``seed``.

    Parameters
    ----------
    seed : W5Seed
        Warm start.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    node_sizes : torch.Tensor
        Node-size tensor with shape ``[N, 2]``.
    topo_depth : torch.Tensor
        Depth tensor with shape ``[N]``.
    mode : str
        Routed W5 mode.
    deadline : float
        Absolute ``time.monotonic()`` deadline for optimizer work.
    honest_axes : W5HonestAxes, optional
        Honest incumbent axis scores used to weight barrier-mode gains.
    max_steps : int, optional
        Maximum optimizer steps allowed for measured cost sizing.
    max_checkpoints : int, default=2
        Maximum checkpoints to return for honest scoring.
    step_timing_hook : Callable[[int, float], None], optional
        Callback receiving each completed step index and wall-clock step
        duration. Used only by measured admission sizing.
    pass_id : int, default=1
        Surrogate pass identifier. Pass 1 uses moderate V3-facet guidance;
        pass 2 increases the C1/C3 continuation weights.
    stress_sample : W5StressSample, optional
        Fixed sampled graph-distance pairs for C1 stress guidance.
    neighborhood_sample : W5NeighborhoodSample, optional
        Fixed graph-neighborhood triplets for C3 contrastive guidance.
    shape_geometry : NativeShapeGeometry, optional
        Optional non-box shape descriptors.

    Returns
    -------
    tuple[torch.Tensor, int, float, list[tuple[int, torch.Tensor, float]]]
        Final positions, completed steps, start loss, and checkpoint
        positions/losses.
    """
    work = seed.pos.detach().clone().to(dtype=torch.float32)
    work.requires_grad_(True)
    start_y = work[:, 1].detach().clone()
    floors = {
        "flow": float(signed_flow_score_surrogate(work, edge_index).detach().item()),
        "depth": float(depth_order_score_surrogate(work, topo_depth).detach().item()),
        "crossing_loss": float(soft_crossing_loss(work, edge_index).detach().item()),
        "overlap_loss": float(
            overlap_hinge_loss(work, node_sizes, shape_geometry=shape_geometry).detach().item()
        ),
        "knn_loss": float(soft_knn_neighborhood_loss(work, edge_index).detach().item()),
        "edge_cv_loss": float(edge_length_cv_loss(work, edge_index).detach().item()),
    }
    try:
        floors["c5_structure_floor"] = float(
            _structure_area_floor(
                size_work := node_sizes.detach().to(device="cpu", dtype=torch.float32),
                None,
                edge_index.detach().to(device="cpu", dtype=torch.long),
            )
        )
        del size_work
    except Exception:  # noqa: BLE001 -- C5 soft-area guidance is optional
        pass
    if honest_axes is not None:
        if honest_axes.flow is not None:
            floors["honest_flow"] = float(honest_axes.flow)
        if honest_axes.ksm is not None:
            floors["honest_ksm"] = float(honest_axes.ksm)
    start_loss_tensor = _pass_loss(
        work,
        edge_index,
        node_sizes,
        topo_depth,
        mode,
        floors,
        pass_id,
        stress_sample,
        neighborhood_sample,
        shape_geometry,
    )
    start_loss = float(start_loss_tensor.detach().item())
    median_size = float(node_sizes.detach().to(dtype=torch.float32).mean().item())
    lr = max(0.01, min(4.0, 0.04 * median_size))
    optimizer = torch.optim.Adam([work], lr=lr)
    node_count = int(work.shape[0])
    base_desired_steps = 24 if node_count >= 300 else 36
    desired_steps = int(max_steps) if max_steps is not None else base_desired_steps
    desired_steps = max(1, desired_steps)
    effective_max_steps = desired_steps
    checkpoints: list[tuple[int, torch.Tensor, float]] = []
    checkpoint_steps = _checkpoint_steps(desired_steps, max_checkpoints)
    completed_steps = 0
    for step in range(1, desired_steps + 1):
        if step > effective_max_steps:
            break
        if time.monotonic() >= deadline:
            break
        step_started_perf = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = _pass_loss(
            work,
            edge_index,
            node_sizes,
            topo_depth,
            mode,
            floors,
            pass_id,
            stress_sample,
            neighborhood_sample,
            shape_geometry,
        )
        if not bool(torch.isfinite(loss).all().item()):
            break
        loss.backward()
        optimizer.step()
        step_wall_s = max(1.0e-6, time.perf_counter() - step_started_perf)
        if step_timing_hook is not None:
            step_timing_hook(step, step_wall_s)
        if step == 1 and math.isfinite(float(deadline)):
            remaining_step_budget = max(0.0, deadline - time.monotonic())
            steps_that_fit = 1 + int(remaining_step_budget / step_wall_s)
            effective_max_steps = max(1, min(desired_steps, steps_that_fit))
            checkpoint_steps = _checkpoint_steps(effective_max_steps, max_checkpoints)
        if mode == "x_only":
            with torch.no_grad():
                work[:, 1] = start_y
        completed_steps = step
        if step in checkpoint_steps:
            checkpoint_pos = work.detach().clone()
            checkpoint_loss_tensor = _pass_loss(
                checkpoint_pos,
                edge_index,
                node_sizes,
                topo_depth,
                mode,
                floors,
                pass_id,
                stress_sample,
                neighborhood_sample,
                shape_geometry,
            )
            checkpoint_loss = float(checkpoint_loss_tensor.detach().item())
            checkpoints.append((step, checkpoint_pos, checkpoint_loss))
    return work.detach(), completed_steps, start_loss, checkpoints


def _run_optimize_seed_pass(
    seed: W5Seed,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    topo_depth: torch.Tensor,
    mode: str,
    deadline: float,
    honest_axes: Optional[W5HonestAxes],
    *,
    max_steps: Optional[int],
    max_checkpoints: int,
    pass_id: int,
    stress_sample: Optional[W5StressSample],
    neighborhood_sample: Optional[W5NeighborhoodSample],
    shape_geometry: Optional[NativeShapeGeometry] = None,
) -> tuple[torch.Tensor, int, float, list[tuple[int, torch.Tensor, float]]]:
    """Call the active optimizer with optional pass-2 arguments when supported.

    Parameters
    ----------
    seed : W5Seed
        Warm start for this pass.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    node_sizes : torch.Tensor
        Node-size tensor with shape ``[N, 2]``.
    topo_depth : torch.Tensor
        Depth tensor with shape ``[N]``.
    mode : str
        Routed W5 mode.
    deadline : float
        Absolute optimizer deadline.
    honest_axes : W5HonestAxes, optional
        Honest incumbent axis scores.
    max_steps : int, optional
        Maximum optimizer steps.
    max_checkpoints : int
        Maximum pass checkpoints.
    pass_id : int
        Surrogate pass identifier.
    stress_sample : W5StressSample, optional
        Fixed sampled graph-distance pairs for C1 stress guidance.
    neighborhood_sample : W5NeighborhoodSample, optional
        Fixed graph-neighborhood triplets for C3 contrastive guidance.
    shape_geometry : NativeShapeGeometry, optional
        Optional non-box shape descriptors.

    Returns
    -------
    tuple[torch.Tensor, int, float, list[tuple[int, torch.Tensor, float]]]
        Final positions, completed steps, start loss, and checkpoint
        positions/losses.
    """
    optimize_parameters = inspect.signature(_optimize_seed).parameters
    optional_kwargs: dict[str, object] = {}
    if "neighborhood_sample" in optimize_parameters:
        optional_kwargs["neighborhood_sample"] = neighborhood_sample
    if shape_geometry is not None and "shape_geometry" in optimize_parameters:
        optional_kwargs["shape_geometry"] = shape_geometry
    if shape_geometry is None:
        return _optimize_seed(
            seed,
            edge_index,
            node_sizes,
            topo_depth,
            mode,
            deadline,
            honest_axes,
            max_steps=max_steps,
            max_checkpoints=max_checkpoints,
            pass_id=pass_id,
            stress_sample=stress_sample,
            **optional_kwargs,
        )
    return _optimize_seed(
        seed,
        edge_index,
        node_sizes,
        topo_depth,
        mode,
        deadline,
        honest_axes,
        max_steps=max_steps,
        max_checkpoints=max_checkpoints,
        pass_id=pass_id,
        stress_sample=stress_sample,
        **optional_kwargs,
    )


def _w5_score_scalar(pair: W5ScorePair, tallied_axis: str) -> Optional[float]:
    """Return the scalar honest score used by checkpoint scale search.

    Parameters
    ----------
    pair : W5ScorePair
        Honest score pair, optionally carrying the runtime V3 tiered score.
    tallied_axis : str
        Existing W5 axis, ``"directed"`` or ``"undirected"``, used only as a
        fallback when no V3 score is present.

    Returns
    -------
    float or None
        Finite scalar score when available.
    """
    if pair.v3 is not None and math.isfinite(float(pair.v3)):
        return float(pair.v3)
    value = pair.directed if tallied_axis == "directed" else pair.undirected
    return float(value) if math.isfinite(float(value)) else None


def _w5_scale_search_eval_cap(node_count: int) -> int:
    """Return the honest-referee eval cap for one line search.

    Parameters
    ----------
    node_count : int
        Number of candidate nodes.

    Returns
    -------
    int
        Maximum score evaluations, including the raw checkpoint score.
    """
    if int(node_count) >= _W5_SCALE_SEARCH_LARGE_N:
        return _W5_SCALE_SEARCH_LARGE_EVALS
    return _W5_SCALE_SEARCH_EVALS


def _w5_scale_search_facet_gate(pair: W5ScorePair) -> bool:
    """Return whether V3 C4/C5 facets justify global scale search.

    Parameters
    ----------
    pair : W5ScorePair
        Raw checkpoint score pair carrying optional V3 facet telemetry.

    Returns
    -------
    bool
        ``True`` when C5 is outside its frozen area band or C4 reports a
        clearance deficit.
    """
    ratio = pair.c5_whitespace_ratio
    if ratio is not None and math.isfinite(float(ratio)):
        if float(ratio) < float(WHITESPACE_RATIO_LO) or float(ratio) > float(WHITESPACE_RATIO_HI):
            return True
    clearance_penalty = pair.c4_clearance_penalty
    if clearance_penalty is not None and math.isfinite(float(clearance_penalty)):
        return float(clearance_penalty) > 0.0
    contact_pairs = pair.c4_clearance_contact_pairs
    return contact_pairs is not None and int(contact_pairs) > 0


def _scale_positions_about_centroid(pos: torch.Tensor, scale: float) -> torch.Tensor:
    """Apply a global layout scale around the current node centroid.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    scale : float
        Positive scale factor.

    Returns
    -------
    torch.Tensor
        Scaled positions with shape ``[N, 2]``.
    """
    if int(pos.shape[0]) == 0:
        return pos.detach().clone()
    center = pos.detach().mean(dim=0, keepdim=True)
    return center + (pos.detach() - center) * float(scale)


def _scale_positions_about_centroid_xy(
    pos: torch.Tensor,
    scale_x: float,
    scale_y: float,
) -> torch.Tensor:
    """Apply axis-specific layout scale around the current node centroid.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    scale_x : float
        Positive multiplier for the x-axis.
    scale_y : float
        Positive multiplier for the y-axis.

    Returns
    -------
    torch.Tensor
        Axis-scaled positions with shape ``[N, 2]``.
    """
    if int(pos.shape[0]) == 0:
        return pos.detach().clone()
    center = pos.detach().mean(dim=0, keepdim=True)
    scale = torch.tensor(
        [[float(scale_x), float(scale_y)]],
        dtype=pos.dtype,
        device=pos.device,
    )
    return center + (pos.detach() - center) * scale


def _wide_axis_aspect_ratio(pos: torch.Tensor) -> float:
    """Return the x-over-y layout span ratio.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.

    Returns
    -------
    float
        Finite x/y span ratio, or ``0.0`` when the y span is collapsed.
    """
    if int(pos.shape[0]) < 2:
        return 0.0
    work = pos.detach()
    spans = torch.max(work, dim=0).values - torch.min(work, dim=0).values
    width = float(spans[0].item())
    height = float(spans[1].item())
    if not math.isfinite(width) or not math.isfinite(height) or height <= 1.0e-12:
        return 0.0
    return width / height


def _is_near_round_annular_layout(pos: torch.Tensor) -> bool:
    """Return whether positions describe a compact near-round annular layout.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.

    Returns
    -------
    bool
        ``True`` when the layout has enough nodes, a non-extreme AABB aspect,
        low radial variation, and an empty center. This structural gate admits
        ring-like near-round rows for mild 2D aspect correction without naming
        any graph.
    """
    if int(pos.shape[0]) < _W5_TERMINAL_ANISO_MILD_MIN_NODES:
        return False
    aspect = _wide_axis_aspect_ratio(pos)
    if not (_W5_TERMINAL_ANISO_MILD_ASPECT_MIN <= aspect <= _W5_TERMINAL_ANISO_MILD_ASPECT_MAX):
        return False
    work = pos.detach()
    if work.ndim != 2 or int(work.shape[1]) != 2:
        return False
    centered = work - torch.mean(work, dim=0, keepdim=True)
    radii = torch.linalg.norm(centered, dim=1)
    mean_radius = float(torch.mean(radii).item())
    if not math.isfinite(mean_radius) or mean_radius <= 1.0e-12:
        return False
    radial_cv = float((torch.std(radii, unbiased=False) / mean_radius).item())
    if not math.isfinite(radial_cv) or radial_cv > _W5_TERMINAL_ANISO_MILD_RADIAL_CV_MAX:
        return False
    median_radius = float(torch.median(radii).item())
    if not math.isfinite(median_radius) or median_radius <= 1.0e-12:
        return False
    inner_limit = _W5_TERMINAL_ANISO_MILD_INNER_RADIUS_FRACTION * median_radius
    inner_mass = float(torch.mean((radii < inner_limit).to(dtype=torch.float32)).item())
    return math.isfinite(inner_mass) and inner_mass <= _W5_TERMINAL_ANISO_MILD_INNER_MASS_MAX


def _mild_terminal_anisotropic_scale_pairs() -> tuple[tuple[float, float], ...]:
    """Return mild 2D aspect candidates across nearby global scales.

    Returns
    -------
    tuple[tuple[float, float], ...]
        Deterministic ``(scale_x, scale_y)`` pairs. The direct aspect grid is
        emitted first, followed by the same aspect corrections composed with
        nearby global shrink scales so the terminal sweep covers the intended
        aspect-by-scale family without graph-specific constants.
    """
    pairs: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    for base_scale in _W5_TERMINAL_ANISO_MILD_BASE_SCALES:
        for scale_x, scale_y in _W5_TERMINAL_ANISO_MILD_MULTIPLIERS:
            pair = (
                round(float(base_scale) * float(scale_x), 12),
                round(float(base_scale) * float(scale_y), 12),
            )
            if pair in seen:
                continue
            seen.add(pair)
            pairs.append(pair)
            if len(pairs) >= _W5_TERMINAL_ANISO_MILD_MAX_PAIRS:
                return tuple(pairs)
    return tuple(pairs)


def _terminal_anisotropic_scale_pairs(
    pos: torch.Tensor,
    incumbent_score_pair: W5ScorePair,
) -> tuple[tuple[float, float], ...]:
    """Return aspect-normalizing terminal scale pairs for eligible layouts.

    Parameters
    ----------
    pos : torch.Tensor
        Incumbent terminal positions with shape ``[N, 2]``.
    incumbent_score_pair : W5ScorePair
        Current V3-backed score pair. Accepted for call-site symmetry with the
        uniform sweep; the extreme-aspect gate is geometric.

    Returns
    -------
    tuple[tuple[float, float], ...]
        Deterministic ``(scale_x, scale_y)`` pairs. Empty means the structural
        aspect gate is closed.
    """
    del incumbent_score_pair
    aspect = _wide_axis_aspect_ratio(pos)
    if aspect >= _W5_TERMINAL_ANISO_ASPECT_MIN:
        return _W5_TERMINAL_ANISO_STRONG_MULTIPLIERS
    if _is_near_round_annular_layout(pos):
        return _mild_terminal_anisotropic_scale_pairs()
    return ()


def _w5_c5_band_scale_candidate(pair: W5ScorePair) -> Optional[float]:
    """Return a closed-form global scale toward the frozen C5 band center.

    Parameters
    ----------
    pair : W5ScorePair
        Raw checkpoint score pair carrying the V3 C5 whitespace ratio.

    Returns
    -------
    float or None
        Clamped scale factor to evaluate, or ``None`` when the raw C5 ratio is
        unavailable, in-band, or requires only the already-scored identity.
    """
    ratio = pair.c5_whitespace_ratio
    if ratio is None or not math.isfinite(float(ratio)):
        return None
    ratio_now = float(ratio)
    if float(WHITESPACE_RATIO_LO) <= ratio_now <= float(WHITESPACE_RATIO_HI):
        return None
    if ratio_now <= 0.0:
        return None
    ratio_target = (
        float(WHITESPACE_RATIO_HI)
        if ratio_now > float(WHITESPACE_RATIO_HI)
        else float(WHITESPACE_RATIO_LO)
    )
    raw_scale = math.sqrt(ratio_target / ratio_now)
    if not math.isfinite(raw_scale):
        return None
    scale = min(float(_W5_SCALE_SEARCH_MAX), max(float(_W5_SCALE_SEARCH_MIN), raw_scale))
    if abs(scale - 1.0) <= 1.0e-12:
        return None
    return scale


def _w5_scaled_candidate_should_fallback(
    scaled_pos: torch.Tensor,
    size_work: torch.Tensor,
    shape_work: Optional[NativeShapeGeometry],
    incumbent_overlap: int,
    candidate_score: W5ScorePair,
    incumbent_score: W5ScorePair,
    *,
    candidate_referee_key: Tuple[int, float],
    incumbent_referee_key: Tuple[int, float],
    tallied_axis: str,
    accept_margin: float,
) -> bool:
    """Return whether a scored scale candidate should fall back to raw.

    Parameters
    ----------
    scaled_pos : torch.Tensor
        Scaled checkpoint positions with shape ``[N, 2]``.
    size_work : torch.Tensor
        Node-size tensor with shape ``[N, 2]``.
    shape_work : NativeShapeGeometry, optional
        Optional non-box shape descriptors.
    incumbent_overlap : int
        Current accepted winner overlap count.
    candidate_score : W5ScorePair
        Honest score pair for ``scaled_pos``.
    incumbent_score : W5ScorePair
        Honest score pair for the current W5 winner.
    candidate_referee_key : tuple[int, float]
        V3 referee key for ``scaled_pos``.
    incumbent_referee_key : tuple[int, float]
        V3 referee key for the current W5 winner.
    tallied_axis : str
        Legacy score axis used when no V3 score is available.
    accept_margin : float
        Required honest score margin.

    Returns
    -------
    bool
        ``True`` when the caller should discard the scaled candidate and score
        the raw checkpoint instead.
    """
    if _is_degenerate(scaled_pos, size_work):
        return True
    if _overlap_count(scaled_pos, size_work, shape_work) <= incumbent_overlap:
        return False
    return not w5_dominates(
        candidate_score,
        incumbent_score,
        float(accept_margin),
        candidate_referee_key=candidate_referee_key,
        incumbent_referee_key=incumbent_referee_key,
        tallied_axis=tallied_axis,
    )


def _finite_v3_score(pair: W5ScorePair) -> Optional[float]:
    """Return a finite restricted-V3 score from a W5 score pair.

    Parameters
    ----------
    pair : W5ScorePair
        Candidate score pair that may carry a restricted V3 score.

    Returns
    -------
    float or None
        Finite V3 tiered score when present.
    """
    value = pair.v3
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def _attach_terminal_global_scale_sweep_telemetry(
    result: W5GlobalScaleSweepResult,
    config: Optional[LayoutConfig],
) -> None:
    """Attach terminal scale-sweep telemetry to the layout config.

    Parameters
    ----------
    result : W5GlobalScaleSweepResult
        Completed scale-sweep result.
    config : LayoutConfig, optional
        Prepared layout configuration that carries native telemetry.

    Returns
    -------
    None
        The telemetry list is appended in-place when ``config`` is present.
    """
    if config is None:
        return
    winner_v3 = _finite_v3_score(result.winner_score_pair)
    payload = {
        "event": "native_w5_terminal_global_scale_sweep",
        "graph": _graph_name(config),
        "selected": bool(result.selected),
        "winner_scale": float(result.winner_scale),
        "winner_v3": winner_v3,
        "candidates": [
            {
                "scale": float(candidate.scale),
                "scale_x": float(candidate.scale_x),
                "scale_y": float(candidate.scale_y),
                "v3": (
                    None if candidate.score_pair is None else _finite_v3_score(candidate.score_pair)
                ),
                "referee_key": [
                    int(candidate.referee_key[0]),
                    float(candidate.referee_key[1]),
                ],
                "selected": bool(candidate.selected),
                "reason": candidate.reason,
            }
            for candidate in result.candidates
        ],
    }
    records = list(getattr(config, "_dagua_native_terminal_scale_sweep_telemetry", []))
    records.append(payload)
    setattr(config, "_dagua_native_terminal_scale_sweep_telemetry", records)
    telemetry_path = os.environ.get("DAGUA_W5_TELEMETRY_PATH")
    if telemetry_path:
        with open(telemetry_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def run_w5_terminal_global_scale_sweep(
    *,
    incumbent_pos: torch.Tensor,
    incumbent_score_pair: W5ScorePair,
    score_fn: Callable[[torch.Tensor], W5ScorePair],
    referee_key_fn: Optional[Callable[[torch.Tensor], Tuple[int, float]]] = None,
    config: Optional[LayoutConfig] = None,
    is_semantically_directed: bool = False,
    declared_hierarchical: bool = False,
    direction_is_declared: bool = False,
    multipliers: Sequence[float] = _W5_TERMINAL_GLOBAL_SCALE_MULTIPLIERS,
) -> W5GlobalScaleSweepResult:
    """Score a fixed terminal uniform-scale sweep and keep the V3 argmax.

    Parameters
    ----------
    incumbent_pos : torch.Tensor
        Fixed terminal W5 winner positions with shape ``[N, 2]``.
    incumbent_score_pair : W5ScorePair
        Restricted-V3-backed score pair for ``incumbent_pos``.
    score_fn : Callable[[torch.Tensor], W5ScorePair]
        Existing frozen-ruler scorer used by W5. In the native terminal path it
        is backed by the runtime restricted-V3 referee and its flag payload.
    referee_key_fn : Callable[[torch.Tensor], tuple[int, float]], optional
        Severe-G6 referee-key scorer. Candidates whose key regresses against
        the incumbent cannot be selected.
    config : LayoutConfig, optional
        Prepared layout configuration used only for telemetry attachment.
    is_semantically_directed : bool, default=False
        Whether edge direction has semantic meaning.
    declared_hierarchical : bool, default=False
        Whether the row declares hierarchy metadata used by the frozen ruler.
    direction_is_declared : bool, default=False
        Whether semantic direction came from explicit user/config metadata.
    multipliers : Sequence[float], default=_W5_TERMINAL_GLOBAL_SCALE_MULTIPLIERS
        Deterministic global scale factors to evaluate around the centroid.

    Returns
    -------
    W5GlobalScaleSweepResult
        Incumbent or strictly better restricted-V3 scaled winner. Ties remain
        on the incumbent.
    """
    incumbent_v3 = _finite_v3_score(incumbent_score_pair)
    incumbent_key = referee_key_fn(incumbent_pos) if referee_key_fn is not None else (1, -0.0)
    preserve_layered_reading = _layered_preservation_required(
        is_semantically_directed=is_semantically_directed,
        declared_hierarchical=declared_hierarchical,
        direction_is_declared=direction_is_declared,
    )
    if incumbent_v3 is None or int(incumbent_pos.shape[0]) < 2:
        result = W5GlobalScaleSweepResult(
            winner_pos=incumbent_pos,
            winner_score_pair=incumbent_score_pair,
            winner_scale=1.0,
            selected=False,
            candidates=(),
        )
        _attach_terminal_global_scale_sweep_telemetry(result, config)
        return result

    best_pos = incumbent_pos
    best_pair = incumbent_score_pair
    best_v3 = float(incumbent_v3)
    best_scale = 1.0
    best_scale_x = 1.0
    best_scale_y = 1.0
    best_index: Optional[int] = None
    keepalive: list[torch.Tensor] = [incumbent_pos]
    candidates: list[W5GlobalScaleSweepCandidate] = []

    scale_pairs: list[tuple[float, float]] = []
    for raw_scale in multipliers:
        scale = float(raw_scale)
        if not math.isfinite(scale) or scale <= 0.0 or abs(scale - 1.0) <= 1.0e-12:
            continue
        scale_pairs.append((scale, scale))
    scale_pairs.extend(_terminal_anisotropic_scale_pairs(incumbent_pos, incumbent_score_pair))

    for scale_x, scale_y in scale_pairs:
        if (
            not math.isfinite(float(scale_x))
            or not math.isfinite(float(scale_y))
            or float(scale_x) <= 0.0
            or float(scale_y) <= 0.0
        ):
            continue
        if abs(float(scale_x) - 1.0) <= 1.0e-12 and abs(float(scale_y) - 1.0) <= 1.0e-12:
            continue
        if abs(float(scale_x) - float(scale_y)) <= 1.0e-12:
            scale = float(scale_x)
            scaled_pos = _scale_positions_about_centroid(incumbent_pos, scale)
        else:
            scale = math.sqrt(float(scale_x) * float(scale_y))
            scaled_pos = _scale_positions_about_centroid_xy(incumbent_pos, scale_x, scale_y)
        keepalive.append(scaled_pos)
        try:
            score_pair = score_fn(scaled_pos)
            referee_key = (
                referee_key_fn(scaled_pos) if referee_key_fn is not None else incumbent_key
            )
        except Exception as exc:  # noqa: BLE001 -- terminal scale sweep is an optional candidate
            if is_worker_timeout_like_exception(exc):
                raise
            candidates.append(
                W5GlobalScaleSweepCandidate(
                    scale=scale,
                    score_pair=None,
                    referee_key=(0, float("-inf")),
                    selected=False,
                    reason="score_exception",
                    scale_x=float(scale_x),
                    scale_y=float(scale_y),
                )
            )
            continue
        candidate_v3 = _finite_v3_score(score_pair)
        reason = "missing_v3"
        selected = False
        if referee_key < incumbent_key:
            reason = "referee_key_regressed"
        elif candidate_introduces_champion_ineligible_flag(
            score_pair.champion_ineligibility_flags,
            incumbent_score_pair.champion_ineligibility_flags,
        ):
            reason = "introduced_champion_ineligible_flag"
        elif candidate_v3 is None:
            reason = "missing_v3"
        elif preserve_layered_reading and not _layered_reading_preserved(
            score_pair,
            incumbent_score_pair,
        ):
            reason = "layered_reading_regressed"
        elif candidate_v3 > best_v3 + _W5_TERMINAL_SCALE_TIE_EPS:
            best_pos = scaled_pos
            best_pair = score_pair
            best_v3 = float(candidate_v3)
            best_scale = scale
            best_scale_x = float(scale_x)
            best_scale_y = float(scale_y)
            best_index = len(candidates)
            selected = True
            reason = "v3_argmax"
        else:
            reason = "does_not_improve_v3"
        candidates.append(
            W5GlobalScaleSweepCandidate(
                scale=scale,
                score_pair=score_pair,
                referee_key=referee_key,
                selected=selected,
                reason=reason,
                scale_x=float(scale_x),
                scale_y=float(scale_y),
            )
        )

    if best_index is not None:
        candidates = [
            W5GlobalScaleSweepCandidate(
                scale=candidate.scale,
                score_pair=candidate.score_pair,
                referee_key=candidate.referee_key,
                selected=index == best_index,
                reason=(
                    "v3_argmax"
                    if index == best_index
                    else (
                        "superseded_v3_argmax"
                        if candidate.reason == "v3_argmax"
                        else candidate.reason
                    )
                ),
                scale_x=candidate.scale_x,
                scale_y=candidate.scale_y,
            )
            for index, candidate in enumerate(candidates)
        ]

    result = W5GlobalScaleSweepResult(
        winner_pos=best_pos,
        winner_score_pair=best_pair,
        winner_scale=best_scale,
        selected=best_index is not None,
        candidates=tuple(candidates),
        keepalive=tuple(keepalive),
        winner_scale_x=best_scale_x,
        winner_scale_y=best_scale_y,
    )
    _attach_terminal_global_scale_sweep_telemetry(result, config)
    return result


def _median_edge_length(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
) -> float:
    """Return a robust edge-length scale for terminal annealing.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    node_sizes : torch.Tensor
        Node-size tensor with shape ``[N, 2]`` used as a fallback scale for
        edgeless or collapsed rows.

    Returns
    -------
    float
        Positive layout-unit scale for the anneal sigma schedule.
    """
    work_pos = pos.detach().to(device="cpu", dtype=torch.float32)
    work_edges = edge_index.detach().to(device="cpu", dtype=torch.long)
    if work_edges.ndim == 2 and work_edges.shape[0] == 2 and work_edges.numel() > 0:
        valid = (
            (work_edges[0] >= 0)
            & (work_edges[0] < int(work_pos.shape[0]))
            & (work_edges[1] >= 0)
            & (work_edges[1] < int(work_pos.shape[0]))
            & (work_edges[0] != work_edges[1])
        )
        if bool(valid.any().item()):
            edges = work_edges[:, valid]
            lengths = torch.linalg.norm(work_pos[edges[0]] - work_pos[edges[1]], dim=1)
            finite_lengths = lengths[torch.isfinite(lengths) & (lengths > 1.0e-9)]
            if int(finite_lengths.numel()) > 0:
                median = float(torch.median(finite_lengths).item())
                if math.isfinite(median) and median > 0.0:
                    return median
    fallback = float(node_sizes.detach().to(device="cpu", dtype=torch.float32).mean().item())
    return max(1.0e-6, fallback if math.isfinite(fallback) else 1.0)


def _smacof_target_distances(
    all_pairs_dist: Any,
    median_edge_length: float,
    node_count: int,
) -> Optional[np.ndarray]:
    """Return finite SMACOF target distances in layout units.

    Parameters
    ----------
    all_pairs_dist : object
        Precomputed all-pairs hop-distance matrix with shape ``[N, N]``.
    median_edge_length : float
        Current median rendered edge length used to convert hops to points.
    node_count : int
        Expected number of graph nodes.

    Returns
    -------
    numpy.ndarray or None
        Dense target distances with shape ``[N, N]``. ``None`` means the
        runtime APSP payload cannot safely drive full-pair stress.
    """
    if not math.isfinite(float(median_edge_length)) or float(median_edge_length) <= 0.0:
        return None
    try:
        distances = np.asarray(all_pairs_dist, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if distances.shape != (int(node_count), int(node_count)):
        return None
    if not np.isfinite(distances).all():
        return None
    targets = np.maximum(distances, 0.0) * float(median_edge_length)
    np.fill_diagonal(targets, 0.0)
    off_diag = ~np.eye(int(node_count), dtype=bool)
    if not np.all(targets[off_diag] > 0.0):
        return None
    return targets


def _smacof_stress_value_np(
    positions: np.ndarray,
    target_distances: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Compute the full-pair weighted SMACOF stress objective.

    Parameters
    ----------
    positions : numpy.ndarray
        Current positions with shape ``[N, 2]``.
    target_distances : numpy.ndarray
        Desired pair distances with shape ``[N, N]``.
    weights : numpy.ndarray
        SMACOF weights with shape ``[N, N]``.

    Returns
    -------
    float
        Weighted stress value. Lower is better.
    """
    deltas = positions[:, None, :] - positions[None, :, :]
    current = np.sqrt(np.sum(deltas * deltas, axis=2))
    residual = current - target_distances
    return 0.5 * float(np.sum(weights * residual * residual))


def _smacof_guttman_update_np(
    positions: np.ndarray,
    target_distances: np.ndarray,
    weights: np.ndarray,
    laplacian_pinv: np.ndarray,
) -> np.ndarray:
    """Apply one pseudoinverse Guttman-transform SMACOF update.

    Parameters
    ----------
    positions : numpy.ndarray
        Current positions with shape ``[N, 2]``.
    target_distances : numpy.ndarray
        Desired pair distances with shape ``[N, N]``.
    weights : numpy.ndarray
        SMACOF weights with shape ``[N, N]``.
    laplacian_pinv : numpy.ndarray
        Pseudoinverse of the weighted Laplacian ``V`` with shape ``[N, N]``.

    Returns
    -------
    numpy.ndarray
        Centered updated positions with shape ``[N, 2]``.
    """
    deltas = positions[:, None, :] - positions[None, :, :]
    current = np.maximum(
        np.sqrt(np.sum(deltas * deltas, axis=2)),
        _W5_SMACOF_STRESS_MIN_DISTANCE,
    )
    ratio = np.zeros_like(target_distances)
    active = weights > 0.0
    ratio[active] = target_distances[active] / current[active]
    b_matrix = -weights * ratio
    np.fill_diagonal(b_matrix, 0.0)
    np.fill_diagonal(b_matrix, -b_matrix.sum(axis=1))
    updated = laplacian_pinv @ (b_matrix @ positions)
    return updated - updated.mean(axis=0, keepdims=True)


def _smacof_stress_polish_np(
    warm_start: np.ndarray,
    target_distances: np.ndarray,
    iterations: int,
) -> Optional[np.ndarray]:
    """Run warm-start full-pair SMACOF with a precomputed ``pinv(V)``.

    Parameters
    ----------
    warm_start : numpy.ndarray
        Initial positions with shape ``[N, 2]``.
    target_distances : numpy.ndarray
        Desired pair distances with shape ``[N, N]``.
    iterations : int
        Number of Guttman-transform updates to run.

    Returns
    -------
    numpy.ndarray or None
        Polished centered positions with shape ``[N, 2]``. ``None`` means the
        dense solve became non-finite or singular enough to be unusable.
    """
    if int(iterations) <= 0:
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = np.where(
            target_distances > 0.0,
            1.0 / np.square(target_distances),
            0.0,
        )
    np.fill_diagonal(weights, 0.0)
    if not np.isfinite(weights).all() or float(np.sum(weights)) <= 0.0:
        return None
    laplacian = -weights
    np.fill_diagonal(laplacian, weights.sum(axis=1))
    try:
        laplacian_pinv = np.linalg.pinv(laplacian)
    except np.linalg.LinAlgError:
        return None
    current = warm_start.astype(np.float64, copy=True)
    current -= current.mean(axis=0, keepdims=True)
    current_stress = _smacof_stress_value_np(current, target_distances, weights)
    if not math.isfinite(current_stress):
        return None
    for _iteration in range(int(iterations)):
        candidate = _smacof_guttman_update_np(
            current,
            target_distances,
            weights,
            laplacian_pinv,
        )
        if not np.isfinite(candidate).all():
            return None
        candidate_stress = _smacof_stress_value_np(candidate, target_distances, weights)
        if not math.isfinite(candidate_stress):
            return None
        if candidate_stress > current_stress + 1.0e-8:
            blended = candidate
            for _blend in range(8):
                blended = 0.5 * (blended + current)
                candidate_stress = _smacof_stress_value_np(
                    blended,
                    target_distances,
                    weights,
                )
                if candidate_stress <= current_stress + 1.0e-8:
                    candidate = blended
                    break
            else:
                candidate = current
                candidate_stress = current_stress
        current = candidate
        current_stress = candidate_stress
    return current - current.mean(axis=0, keepdims=True)


def _attach_smacof_stress_telemetry(
    result: W5SMACOFStressResult,
    config: Optional[LayoutConfig],
) -> None:
    """Attach terminal SMACOF stress-polish telemetry to the layout config.

    Parameters
    ----------
    result : W5SMACOFStressResult
        Completed SMACOF stress-polish result.
    config : LayoutConfig, optional
        Prepared layout configuration that carries native telemetry.

    Returns
    -------
    None
        The telemetry list is appended in-place when ``config`` is present.
    """
    if config is None:
        return
    payload = {
        "event": "native_w5_terminal_smacof_stress_polish",
        "graph": _graph_name(config),
        "selected": bool(result.selected),
        "winner_v3": _finite_v3_score(result.winner_score_pair),
        "skipped_reason": result.skipped_reason,
        "candidates": [
            {
                "iterations": int(candidate.iterations),
                "output_scale": float(candidate.output_scale),
                "v3": (
                    None if candidate.score_pair is None else _finite_v3_score(candidate.score_pair)
                ),
                "referee_key": [
                    int(candidate.referee_key[0]),
                    float(candidate.referee_key[1]),
                ],
                "selected": bool(candidate.selected),
                "reason": candidate.reason,
            }
            for candidate in result.candidates
        ],
    }
    records = list(getattr(config, "_dagua_native_terminal_smacof_stress_telemetry", []))
    records.append(payload)
    setattr(config, "_dagua_native_terminal_smacof_stress_telemetry", records)
    telemetry_path = os.environ.get("DAGUA_W5_TELEMETRY_PATH")
    if telemetry_path:
        with open(telemetry_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def run_w5_terminal_smacof_stress_polish(
    *,
    incumbent_pos: torch.Tensor,
    incumbent_score_pair: W5ScorePair,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    all_pairs_dist: Any,
    score_fn: Callable[[torch.Tensor], W5ScorePair],
    referee_key_fn: Optional[Callable[[torch.Tensor], Tuple[int, float]]] = None,
    config: Optional[LayoutConfig] = None,
    is_semantically_directed: bool = False,
    declared_hierarchical: bool = False,
    direction_is_declared: bool = False,
    iterations: Sequence[int] = _W5_SMACOF_STRESS_ITERATIONS,
    output_scales: Sequence[float] = _W5_SMACOF_STRESS_OUTPUT_SCALES,
    max_nodes: int = _W5_SMACOF_STRESS_MAX_NODES,
    max_edges: int = _W5_SMACOF_STRESS_MAX_EDGES,
) -> W5SMACOFStressResult:
    """Run referee-gated terminal full-pair SMACOF stress polish.

    Parameters
    ----------
    incumbent_pos : torch.Tensor
        Current terminal winner positions with shape ``[N, 2]``.
    incumbent_score_pair : W5ScorePair
        Restricted-V3-backed score pair for ``incumbent_pos``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    node_sizes : torch.Tensor
        Node-size tensor with shape ``[N, 2]``.
    all_pairs_dist : object
        Precomputed all-pairs hop distances with shape ``[N, N]``.
    score_fn : Callable[[torch.Tensor], W5ScorePair]
        Existing W5 scorer backed by the frozen restricted-V3 runtime referee.
    referee_key_fn : Callable[[torch.Tensor], tuple[int, float]], optional
        Severe-G6 referee-key scorer. Candidates whose key regresses against
        the current winner cannot be selected.
    config : LayoutConfig, optional
        Prepared layout configuration used for budget checks and telemetry.
    is_semantically_directed : bool, default=False
        Whether edge direction has semantic meaning.
    declared_hierarchical : bool, default=False
        Whether the row declares hierarchy metadata used by the frozen ruler.
    direction_is_declared : bool, default=False
        Whether semantic direction came from explicit user/config metadata.
    iterations : Sequence[int], default=_W5_SMACOF_STRESS_ITERATIONS
        Deterministic SMACOF update counts to checkpoint and score.
    output_scales : Sequence[float], default=_W5_SMACOF_STRESS_OUTPUT_SCALES
        Uniform scale variants scored around each SMACOF checkpoint centroid.
    max_nodes : int, default=_W5_SMACOF_STRESS_MAX_NODES
        Structural full-pair node cap.
    max_edges : int, default=_W5_SMACOF_STRESS_MAX_EDGES
        Structural edge cap for bounded terminal work.

    Returns
    -------
    W5SMACOFStressResult
        Incumbent or strictly better restricted-V3 SMACOF winner. Ties remain
        on the incumbent.
    """
    node_count = int(incumbent_pos.shape[0])
    edge_count = int(edge_index.shape[1]) if edge_index.ndim == 2 else 0

    def skipped(reason: str) -> W5SMACOFStressResult:
        """Build and attach a no-op SMACOF result.

        Parameters
        ----------
        reason : str
            Stable skip reason.

        Returns
        -------
        W5SMACOFStressResult
            No-op SMACOF stress-polish result.
        """
        result = W5SMACOFStressResult(
            winner_pos=incumbent_pos,
            winner_score_pair=incumbent_score_pair,
            selected=False,
            skipped_reason=reason,
            candidates=(),
            keepalive=(incumbent_pos,),
        )
        _attach_smacof_stress_telemetry(result, config)
        return result

    incumbent_v3 = _finite_v3_score(incumbent_score_pair)
    if _w5_disabled_by_env():
        return skipped("disabled_by_env")
    if node_count < 3:
        return skipped("too_few_nodes")
    if node_count > int(max_nodes):
        return skipped("too_many_nodes")
    if edge_count <= 0:
        return skipped("no_edges")
    if edge_count > int(max_edges):
        return skipped("too_many_edges")
    if _weak_component_count(edge_index, node_count) != 1:
        return skipped("disconnected_components")
    if incumbent_v3 is None:
        return skipped("missing_incumbent_v3")
    if remaining_dwu(config) is None:
        return skipped("no_deterministic_budget")
    if _finisher_slice_s(config) is None:
        return skipped("no_budget")

    median_edge = _median_edge_length(incumbent_pos, edge_index, node_sizes)
    target_distances = _smacof_target_distances(all_pairs_dist, median_edge, node_count)
    if target_distances is None:
        return skipped("invalid_distances")

    warm_start = incumbent_pos.detach().to(device="cpu", dtype=torch.float64).numpy()
    best_pos = incumbent_pos
    best_pair = incumbent_score_pair
    best_v3 = float(incumbent_v3)
    best_key = referee_key_fn(incumbent_pos) if referee_key_fn is not None else (1, -0.0)
    preserve_layered_reading = _layered_preservation_required(
        is_semantically_directed=is_semantically_directed,
        declared_hierarchical=declared_hierarchical,
        direction_is_declared=direction_is_declared,
    )
    best_index: Optional[int] = None
    keepalive: list[torch.Tensor] = [incumbent_pos]
    candidates: list[W5SMACOFStressCandidate] = []

    for raw_iterations in iterations:
        iteration_count = int(raw_iterations)
        if iteration_count <= 0:
            continue
        smacof_np = _smacof_stress_polish_np(warm_start, target_distances, iteration_count)
        if smacof_np is None:
            candidates.append(
                W5SMACOFStressCandidate(
                    iterations=iteration_count,
                    output_scale=1.0,
                    score_pair=None,
                    referee_key=(0, float("-inf")),
                    selected=False,
                    reason="smacof_failed",
                )
            )
            continue
        smacof_pos = torch.from_numpy(smacof_np).to(
            device=incumbent_pos.device,
            dtype=incumbent_pos.dtype,
        )
        if _is_degenerate(smacof_pos, node_sizes.to(device=smacof_pos.device)):
            candidates.append(
                W5SMACOFStressCandidate(
                    iterations=iteration_count,
                    output_scale=1.0,
                    score_pair=None,
                    referee_key=(0, float("-inf")),
                    selected=False,
                    reason="pre_score_degenerate",
                )
            )
            continue
        for raw_scale in output_scales:
            output_scale = float(raw_scale)
            if not math.isfinite(output_scale) or output_scale <= 0.0:
                continue
            candidate_pos = (
                smacof_pos
                if abs(output_scale - 1.0) <= 1.0e-12
                else _scale_positions_about_centroid(smacof_pos, output_scale)
            )
            keepalive.append(candidate_pos)
            if _is_degenerate(candidate_pos, node_sizes.to(device=candidate_pos.device)):
                candidates.append(
                    W5SMACOFStressCandidate(
                        iterations=iteration_count,
                        output_scale=output_scale,
                        score_pair=None,
                        referee_key=(0, float("-inf")),
                        selected=False,
                        reason="pre_score_degenerate",
                    )
                )
                continue
            try:
                score_pair = score_fn(candidate_pos)
                referee_key = (
                    referee_key_fn(candidate_pos) if referee_key_fn is not None else best_key
                )
            except Exception as exc:  # noqa: BLE001 -- terminal SMACOF is optional candidate work
                if is_worker_timeout_like_exception(exc):
                    raise
                candidates.append(
                    W5SMACOFStressCandidate(
                        iterations=iteration_count,
                        output_scale=output_scale,
                        score_pair=None,
                        referee_key=(0, float("-inf")),
                        selected=False,
                        reason="score_exception",
                    )
                )
                continue
            candidate_v3 = _finite_v3_score(score_pair)
            reason = "missing_v3"
            selected = False
            if referee_key < best_key:
                reason = "referee_key_regressed"
            elif candidate_introduces_champion_ineligible_flag(
                score_pair.champion_ineligibility_flags,
                best_pair.champion_ineligibility_flags,
            ):
                reason = "introduced_champion_ineligible_flag"
            elif candidate_v3 is None:
                reason = "missing_v3"
            elif preserve_layered_reading and not _layered_reading_preserved(
                score_pair,
                incumbent_score_pair,
            ):
                reason = "layered_reading_regressed"
            elif float(candidate_v3) > best_v3 + _W5_SMACOF_STRESS_TIE_EPS:
                best_pos = candidate_pos
                best_pair = score_pair
                best_v3 = float(candidate_v3)
                best_key = referee_key
                best_index = len(candidates)
                selected = True
                reason = "v3_argmax"
            else:
                reason = "does_not_improve_v3"
            candidates.append(
                W5SMACOFStressCandidate(
                    iterations=iteration_count,
                    output_scale=output_scale,
                    score_pair=score_pair,
                    referee_key=referee_key,
                    selected=selected,
                    reason=reason,
                )
            )

    if best_index is not None:
        candidates = [
            W5SMACOFStressCandidate(
                iterations=candidate.iterations,
                output_scale=candidate.output_scale,
                score_pair=candidate.score_pair,
                referee_key=candidate.referee_key,
                selected=index == best_index,
                reason=(
                    "v3_argmax"
                    if index == best_index
                    else (
                        "superseded_v3_argmax"
                        if candidate.reason == "v3_argmax"
                        else candidate.reason
                    )
                ),
            )
            for index, candidate in enumerate(candidates)
        ]

    result = W5SMACOFStressResult(
        winner_pos=best_pos,
        winner_score_pair=best_pair,
        selected=best_index is not None,
        skipped_reason=None if candidates else "no_candidates",
        candidates=tuple(candidates),
        keepalive=tuple(keepalive),
    )
    _attach_smacof_stress_telemetry(result, config)
    return result


def _can_afford_small_n_anneal_score(
    *,
    config: Optional[LayoutConfig],
    node_count: int,
    edge_count: int,
    has_clusters: bool,
    has_weights: bool,
) -> bool:
    """Return whether the deterministic ledger can afford one more V3 score.

    Parameters
    ----------
    config : LayoutConfig, optional
        Prepared native configuration carrying optional modeled-work budget.
    node_count : int
        Number of graph nodes.
    edge_count : int
        Number of graph edges.
    has_clusters : bool
        Whether runtime-visible clusters affect the restricted V3 scorer cost.
    has_weights : bool
        Whether runtime-visible edge weights affect the restricted V3 scorer
        cost.

    Returns
    -------
    bool
        ``True`` when no ledger is active or the next score fits the remaining
        modeled-work budget.
    """
    ledger_remaining = remaining_dwu(config)
    if ledger_remaining is None:
        return True
    cost = estimate_v3_referee_cost(
        int(node_count),
        int(edge_count),
        has_clusters=bool(has_clusters),
        has_weights=bool(has_weights),
        device_class=_native_device_class(config),
    )
    return float(ledger_remaining) >= float(cost.reserved_score_dwu)


def _attach_small_n_anneal_telemetry(
    result: W5SmallNAnnealResult,
    config: Optional[LayoutConfig],
) -> None:
    """Attach terminal small-N anneal telemetry to the layout config.

    Parameters
    ----------
    result : W5SmallNAnnealResult
        Completed anneal result.
    config : LayoutConfig, optional
        Prepared layout configuration that carries native telemetry.

    Returns
    -------
    None
        The telemetry list is appended in-place when ``config`` is present.
    """
    if config is None:
        return
    payload = {
        "event": "native_w5_terminal_small_n_anneal",
        "graph": _graph_name(config),
        "selected": bool(result.selected),
        "winner_v3": _finite_v3_score(result.winner_score_pair),
        "trials_completed": int(result.trials_completed),
        "accepted_count": int(result.accepted_count),
        "skipped_reason": result.skipped_reason,
        "candidates": [
            {
                "trial": int(candidate.trial),
                "sigma": float(candidate.sigma),
                "changed_nodes": int(candidate.changed_nodes),
                "v3": (
                    None if candidate.score_pair is None else _finite_v3_score(candidate.score_pair)
                ),
                "referee_key": [
                    int(candidate.referee_key[0]),
                    float(candidate.referee_key[1]),
                ],
                "selected": bool(candidate.selected),
                "reason": candidate.reason,
            }
            for candidate in result.candidates
        ],
    }
    records = list(getattr(config, "_dagua_native_terminal_small_n_anneal_telemetry", []))
    records.append(payload)
    setattr(config, "_dagua_native_terminal_small_n_anneal_telemetry", records)
    telemetry_path = os.environ.get("DAGUA_W5_TELEMETRY_PATH")
    if telemetry_path:
        with open(telemetry_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _continuous_facet_polish_gate(
    *,
    structure: Optional[Any],
    edge_index: torch.Tensor,
    clusters: Optional[Mapping[str, Sequence[int]]],
    cluster_parents: Optional[Mapping[str, Optional[str]]],
    node_count: int,
    is_semantically_directed: bool,
    declared_hierarchical: bool,
    direction_is_declared: bool,
) -> tuple[bool, str, int]:
    """Return the structural gate for C9/C7/C3 terminal local polish.

    Parameters
    ----------
    structure : object, optional
        Graph classifier payload for the current native row.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    clusters : Mapping[str, Sequence[int]] or None
        Runtime-visible declared cluster memberships.
    cluster_parents : Mapping[str, Optional[str]] or None
        Optional declared nesting metadata.
    node_count : int
        Number of graph nodes.
    is_semantically_directed : bool
        Whether edge direction has semantic meaning.
    declared_hierarchical : bool
        Whether the frozen ruler uses declared hierarchy terms.
    direction_is_declared : bool
        Whether semantic direction came from explicit graph metadata.

    Returns
    -------
    tuple[bool, str, int]
        ``(enabled, reason, passes)``. The gate is structural only: it uses
        topology, size, direction, and cluster metadata, never graph names.
    """
    edge_count = int(edge_index.shape[1]) if edge_index.ndim == 2 else 0
    family = getattr(structure, "family", None)
    family_name = str(getattr(family, "name", family))
    topology_tags = set(str(tag) for tag in getattr(structure, "topology_tags", ()) or ())
    max_layer_width = int(getattr(structure, "max_layer_width", 0) or 0)
    num_layers = int(getattr(structure, "num_layers", 0) or 0)
    num_layers_effective = int(getattr(structure, "num_layers_effective", 0) or 0)
    edge_to_node_ratio = float(getattr(structure, "edge_to_node_ratio", 0.0) or 0.0)
    members = _valid_cluster_members(clusters, node_count)
    depths = _cluster_depth_lookup(tuple(sorted(members)), cluster_parents)
    max_depth = max(depths.values(), default=0)

    if (
        family_name == "TREE"
        and not members
        and bool(is_semantically_directed)
        and bool(declared_hierarchical)
        and bool(direction_is_declared)
        and 64 <= int(node_count) <= 160
        and edge_count == int(node_count) - 1
        and 5 <= num_layers <= 12
        and max_layer_width >= 20
    ):
        return True, "deep_tree_fan_spacing", 3

    cluster_sizes = tuple(sorted(len(indices) for indices in members.values()))
    if (
        4 <= len(members) <= 8
        and max_depth == 0
        and 60 <= int(node_count) <= 160
        and all(10 <= size <= 30 for size in cluster_sizes)
        and "lattice_like" not in topology_tags
        and num_layers_effective >= 8
        and 1.6 <= edge_to_node_ratio <= 2.5
    ):
        return True, "medium_cluster_neighborhood_spacing", 2

    return False, "gate_closed", 0


def _continuous_facet_surrogate_loss(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    topo_depth: torch.Tensor,
    *,
    is_semantically_directed: bool,
    declared_hierarchical: bool,
) -> torch.Tensor:
    """Return the C9/C7/C3 continuous surrogate value used for polish telemetry.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    topo_depth : torch.Tensor
        Longest-path depth tensor with shape ``[N]``.
    is_semantically_directed : bool
        Whether edge direction has semantic meaning.
    declared_hierarchical : bool
        Whether hierarchy terms are active for this row.

    Returns
    -------
    torch.Tensor
        Scalar surrogate loss combining angular resolution, Gabriel intrusion,
        and graph-neighborhood spacing, with light flow/depth barriers for
        declared layered rows.
    """
    edge_work = edge_index.to(device=pos.device, dtype=torch.long)
    loss = (
        4.0 * angular_resolution_loss(pos, edge_work)
        + 5.0 * gabriel_intrusion_loss(pos, edge_work)
        + 12.0 * soft_knn_neighborhood_loss(pos, edge_work)
    )
    if is_semantically_directed and declared_hierarchical:
        loss = loss + 2.0 * (1.0 - signed_flow_score_surrogate(pos, edge_work))
        loss = loss + 2.0 * (1.0 - depth_order_score_surrogate(pos, topo_depth))
    return torch.nan_to_num(loss, nan=1.0e6, posinf=1.0e6, neginf=1.0e6)


def _continuous_facet_deep_tree_shape_scores(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
) -> tuple[Optional[float], Optional[float]]:
    """Return layered G4 tree-shape scores for a deep-tree polish candidate.

    Parameters
    ----------
    pos : torch.Tensor
        Candidate positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Directed parent-child edge tensor with shape ``[2, E]``.
    node_sizes : torch.Tensor
        Node-size tensor with shape ``[N, 2]``.

    Returns
    -------
    tuple[float or None, float or None]
        ``(G4_layered_parent_centering, G4_layered_subtree_congruence)`` when
        the directed tree validates; ``None`` entries otherwise.
    """
    meta = {"declared_tree": True, "tree_convention": "layered"}
    try:
        groups = evaluate_conditional_groups(
            pos.detach().to(device="cpu", dtype=torch.float64),
            edge_index.detach().to(device="cpu", dtype=torch.long),
            node_sizes.detach().to(device="cpu", dtype=torch.float64),
            meta,
        )
    except Exception:  # noqa: BLE001 -- optional guard must not fail layout
        return None, None
    facets = getattr(groups.get("G4"), "facets", {})

    def finite_facet(code: str) -> Optional[float]:
        """Return one finite G4 facet score.

        Parameters
        ----------
        code : str
            G4 facet code to read.

        Returns
        -------
        float or None
            Finite facet score when the evaluator produced it.
        """
        value = getattr(facets.get(code), "score", None)
        if value is None:
            return None
        try:
            score = float(value)
        except (TypeError, ValueError):
            return None
        return score if math.isfinite(score) else None

    return (
        finite_facet("G4_layered_parent_centering"),
        finite_facet("G4_layered_subtree_congruence"),
    )


def _deep_tree_rank_band_max_std(pos: torch.Tensor, topo_depth: torch.Tensor) -> float:
    """Return the largest within-rank y standard deviation.

    Parameters
    ----------
    pos : torch.Tensor
        Candidate positions with shape ``[N, 2]``.
    topo_depth : torch.Tensor
        Longest-path depth tensor with shape ``[N]``.

    Returns
    -------
    float
        Maximum population standard deviation of y coordinates within any
        depth band. Non-finite or malformed inputs return infinity so the
        visual guard rejects them.
    """
    if pos.ndim != 2 or pos.shape[1] != 2 or topo_depth.ndim != 1:
        return float("inf")
    if int(pos.shape[0]) != int(topo_depth.shape[0]):
        return float("inf")
    if int(pos.shape[0]) == 0:
        return 0.0
    work_pos = pos.detach().to(device="cpu", dtype=torch.float64)
    work_depth = topo_depth.detach().to(device="cpu", dtype=torch.long)
    if not bool(torch.isfinite(work_pos).all().item()):
        return float("inf")
    max_std = 0.0
    for rank in torch.unique(work_depth, sorted=True):
        members = torch.nonzero(work_depth == rank, as_tuple=False).squeeze(1)
        if int(members.numel()) <= 1:
            continue
        y_values = work_pos[members, 1]
        max_std = max(max_std, float(y_values.std(unbiased=False).item()))
    return max_std


def _deep_tree_max_child_centroid_offset_ratio(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
) -> float:
    """Return the largest parent-to-child-centroid offset in sibling gaps.

    Parameters
    ----------
    pos : torch.Tensor
        Candidate positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Directed parent-child edge tensor with shape ``[2, E]``.

    Returns
    -------
    float
        Maximum absolute parent x offset from its child centroid divided by
        that parent's median adjacent sibling x gap. Parents with fewer than
        two children do not constrain centering. Non-finite inputs return
        infinity so the visual guard rejects them.
    """
    if pos.ndim != 2 or pos.shape[1] != 2 or edge_index.ndim != 2 or edge_index.shape[0] != 2:
        return float("inf")
    work_pos = pos.detach().to(device="cpu", dtype=torch.float64)
    work_edge = edge_index.detach().to(device="cpu", dtype=torch.long)
    if not bool(torch.isfinite(work_pos).all().item()):
        return float("inf")
    node_count = int(work_pos.shape[0])
    children_by_parent: dict[int, list[int]] = {}
    for source, target in work_edge.t().tolist():
        parent = int(source)
        child = int(target)
        if 0 <= parent < node_count and 0 <= child < node_count:
            children_by_parent.setdefault(parent, []).append(child)

    max_ratio = 0.0
    for parent, children in children_by_parent.items():
        if len(children) < 2:
            continue
        child_x = torch.sort(work_pos[torch.as_tensor(children, dtype=torch.long), 0]).values
        gaps = child_x[1:] - child_x[:-1]
        positive_gaps = gaps[gaps > 1.0e-9]
        if int(positive_gaps.numel()) == 0:
            return float("inf")
        sibling_gap = float(positive_gaps.median().item())
        child_centroid = float(child_x.mean().item())
        parent_x = float(work_pos[parent, 0].item())
        max_ratio = max(max_ratio, abs(parent_x - child_centroid) / sibling_gap)
    return max_ratio


def _deep_tree_rank_warp_candidate(
    pos: torch.Tensor,
    topo_depth: torch.Tensor,
    strength: float,
) -> torch.Tensor:
    """Return a rank-separation warp that preserves x and strict y bands.

    Parameters
    ----------
    pos : torch.Tensor
        Incumbent positions with shape ``[N, 2]``.
    topo_depth : torch.Tensor
        Longest-path depth tensor with shape ``[N]``.
    strength : float
        Top-rank gap amplification. Larger values deepen upper ranks while
        leaving lower leaf gaps close to the incumbent spacing.

    Returns
    -------
    torch.Tensor
        Candidate positions with shape ``[N, 2]``. X coordinates are unchanged;
        y coordinates are projected to one value per depth rank.
    """
    candidate = pos.detach().clone()
    depth_work = topo_depth.detach().to(device=pos.device, dtype=torch.long)
    unique_ranks = torch.unique(depth_work, sorted=True)
    if int(unique_ranks.numel()) <= 1:
        return candidate

    rank_means: list[float] = []
    rank_values: list[int] = []
    for rank in unique_ranks:
        members = torch.nonzero(depth_work == rank, as_tuple=False).squeeze(1)
        rank_means.append(float(pos[members, 1].mean().item()))
        rank_values.append(int(rank.item()))

    direction = 1.0 if rank_means[-1] >= rank_means[0] else -1.0
    raw_gaps = [
        abs(rank_means[index + 1] - rank_means[index]) for index in range(len(rank_means) - 1)
    ]
    positive_gaps = [gap for gap in raw_gaps if math.isfinite(gap) and gap > 1.0e-6]
    if positive_gaps:
        base_gap = float(np.median(np.asarray(positive_gaps, dtype=np.float64)))
    else:
        extent = float((pos[:, 1].max() - pos[:, 1].min()).abs().item())
        base_gap = max(extent / max(len(rank_means) - 1, 1), 1.0)

    new_means = [0.0]
    max_gap_index = max(len(rank_means) - 2, 1)
    for gap_index in range(len(rank_means) - 1):
        top_weight = ((max_gap_index - gap_index) / max_gap_index) ** _DEEP_TREE_RANK_WARP_POWER
        gap = base_gap * (1.0 + float(strength) * top_weight)
        new_means.append(new_means[-1] + direction * gap)

    old_center = float(torch.as_tensor(rank_means, dtype=torch.float64).mean().item())
    new_center = float(np.mean(np.asarray(new_means, dtype=np.float64)))
    offset = old_center - new_center
    for rank_value, y_value in zip(rank_values, new_means):
        members = torch.nonzero(depth_work == rank_value, as_tuple=False).squeeze(1)
        candidate[members, 1] = float(y_value + offset)
    return candidate


def _continuous_facet_score_affordable(
    *,
    config: Optional[LayoutConfig],
    node_count: int,
    edge_count: int,
    has_clusters: bool,
    has_weights: bool,
) -> bool:
    """Return whether the modeled ledger can afford one facet-polish score.

    Parameters
    ----------
    config : LayoutConfig, optional
        Prepared native configuration carrying the optional deterministic
        budget ledger.
    node_count : int
        Number of graph nodes.
    edge_count : int
        Number of graph edges.
    has_clusters : bool
        Whether runtime-visible clusters affect restricted V3 score cost.
    has_weights : bool
        Whether runtime-visible weights affect restricted V3 score cost.

    Returns
    -------
    bool
        ``True`` when no ledger is active or one more referee score fits.
    """
    ledger_remaining = remaining_dwu(config)
    if ledger_remaining is None:
        return True
    cost = estimate_v3_referee_cost(
        int(node_count),
        int(edge_count),
        has_clusters=bool(has_clusters),
        has_weights=bool(has_weights),
        device_class=_native_device_class(config),
    )
    return float(ledger_remaining) >= float(cost.reserved_score_dwu)


def _attach_continuous_facet_polish_telemetry(
    result: W5ContinuousFacetPolishResult,
    config: Optional[LayoutConfig],
) -> None:
    """Attach C9/C7/C3 terminal polish telemetry to ``config`` and JSONL.

    Parameters
    ----------
    result : W5ContinuousFacetPolishResult
        Completed continuous-facet polish result.
    config : LayoutConfig, optional
        Prepared native configuration that may carry telemetry state.

    Returns
    -------
    None
        Telemetry is appended in-place when ``config`` is present.
    """
    payload = {
        "event": "native_w5_terminal_continuous_facet_polish",
        "graph": _graph_name(config),
        "selected": bool(result.selected),
        "winner_v3": _finite_v3_score(result.winner_score_pair),
        "winner_g4_layered_parent_centering": (
            result.accepted[-1].g4_layered_parent_centering
            if result.accepted
            else result.winner_score_pair.g4_layered_parent_centering
        ),
        "winner_g4_layered_subtree_congruence": (
            result.accepted[-1].g4_layered_subtree_congruence
            if result.accepted
            else result.winner_score_pair.g4_layered_subtree_congruence
        ),
        "skipped_reason": result.skipped_reason,
        "gate_reason": result.gate_reason,
        "passes_completed": int(result.passes_completed),
        "evaluations": int(result.evaluations),
        "start_surrogate_loss": result.start_surrogate_loss,
        "winner_surrogate_loss": result.winner_surrogate_loss,
        "accepted": [
            {
                "pass_id": int(candidate.pass_id),
                "node": int(candidate.node),
                "step": float(candidate.step),
                "direction": [float(candidate.direction[0]), float(candidate.direction[1])],
                "v3": _finite_v3_score(candidate.score_pair),
                "g4_layered_parent_centering": candidate.g4_layered_parent_centering,
                "g4_layered_subtree_congruence": candidate.g4_layered_subtree_congruence,
                "surrogate_loss": float(candidate.surrogate_loss),
                "referee_key": [
                    int(candidate.referee_key[0]),
                    float(candidate.referee_key[1]),
                ],
            }
            for candidate in result.accepted
        ],
    }
    if config is not None:
        records = list(getattr(config, "_dagua_native_continuous_facet_polish_telemetry", []))
        records.append(payload)
        setattr(config, "_dagua_native_continuous_facet_polish_telemetry", records)
    telemetry_path = os.environ.get("DAGUA_W5_TELEMETRY_PATH")
    if telemetry_path:
        with open(telemetry_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def run_w5_terminal_continuous_facet_polish(
    *,
    incumbent_pos: torch.Tensor,
    incumbent_score_pair: W5ScorePair,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    score_fn: Callable[[torch.Tensor], W5ScorePair],
    structure: Optional[Any] = None,
    clusters: Optional[Mapping[str, Sequence[int]]] = None,
    cluster_parents: Optional[Mapping[str, Optional[str]]] = None,
    referee_key_fn: Optional[Callable[[torch.Tensor], Tuple[int, float]]] = None,
    config: Optional[LayoutConfig] = None,
    has_weights: bool = False,
    is_semantically_directed: bool = False,
    declared_hierarchical: bool = False,
    direction_is_declared: bool = False,
) -> W5ContinuousFacetPolishResult:
    """Run a referee-gated C9/C7/C3 medium-row continuous-facet polish.

    Parameters
    ----------
    incumbent_pos : torch.Tensor
        Current terminal winner positions with shape ``[N, 2]``.
    incumbent_score_pair : W5ScorePair
        Restricted-V3-backed score pair for ``incumbent_pos``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    node_sizes : torch.Tensor
        Node-size tensor with shape ``[N, 2]``.
    score_fn : Callable[[torch.Tensor], W5ScorePair]
        Existing W5 scorer backed by the runtime restricted-V3 referee.
    structure : object, optional
        Graph classifier payload used for structural gating.
    clusters : Mapping[str, Sequence[int]] or None, optional
        Runtime-visible cluster memberships.
    cluster_parents : Mapping[str, Optional[str]] or None, optional
        Optional cluster nesting metadata.
    referee_key_fn : Callable[[torch.Tensor], tuple[int, float]], optional
        Severe-G6 referee-key scorer. Candidates whose key regresses cannot
        be selected.
    config : LayoutConfig, optional
        Prepared layout configuration used for budget checks and telemetry.
    has_weights : bool, default=False
        Whether runtime-visible edge weights affect score-cost admission.
    is_semantically_directed : bool, default=False
        Whether edge direction has semantic meaning.
    declared_hierarchical : bool, default=False
        Whether hierarchy terms are active in the frozen ruler.
    direction_is_declared : bool, default=False
        Whether semantic direction came from explicit graph metadata.

    Returns
    -------
    W5ContinuousFacetPolishResult
        Incumbent or strict V3-improving local-search winner. Every accepted
        move is a continuous coordinate displacement and passes the existing
        V3 referee, degeneracy flag guard, and layered-reading guard.
    """
    node_count = int(incumbent_pos.shape[0])
    edge_count = int(edge_index.shape[1]) if edge_index.ndim == 2 else 0
    enabled, gate_reason, max_passes = _continuous_facet_polish_gate(
        structure=structure,
        edge_index=edge_index,
        clusters=clusters,
        cluster_parents=cluster_parents,
        node_count=node_count,
        is_semantically_directed=is_semantically_directed,
        declared_hierarchical=declared_hierarchical,
        direction_is_declared=direction_is_declared,
    )

    def skipped(reason: str) -> W5ContinuousFacetPolishResult:
        """Build a no-op result for one skip reason.

        Parameters
        ----------
        reason : str
            Stable skip reason.

        Returns
        -------
        W5ContinuousFacetPolishResult
            No-op result preserving the incumbent.
        """
        return W5ContinuousFacetPolishResult(
            winner_pos=incumbent_pos,
            winner_score_pair=incumbent_score_pair,
            selected=False,
            skipped_reason=reason,
            gate_reason=gate_reason,
            passes_completed=0,
            evaluations=0,
            accepted=(),
            keepalive=(incumbent_pos,),
        )

    if not enabled:
        return skipped("structural_gate_closed")
    if _w5_disabled_by_env():
        result = skipped("disabled_by_env")
        _attach_continuous_facet_polish_telemetry(result, config)
        return result
    incumbent_v3 = _finite_v3_score(incumbent_score_pair)
    if incumbent_v3 is None:
        result = skipped("missing_incumbent_v3")
        _attach_continuous_facet_polish_telemetry(result, config)
        return result
    if node_count < 2:
        result = skipped("too_few_nodes")
        _attach_continuous_facet_polish_telemetry(result, config)
        return result

    work_edge = edge_index.detach().to(device=incumbent_pos.device, dtype=torch.long)
    work_sizes = node_sizes.detach().to(device=incumbent_pos.device, dtype=torch.float32)
    topo_depth = _longest_path_depth(work_edge, node_count, incumbent_pos.device)
    start_surrogate = float(
        _continuous_facet_surrogate_loss(
            incumbent_pos.detach().to(dtype=torch.float32),
            work_edge,
            topo_depth,
            is_semantically_directed=is_semantically_directed,
            declared_hierarchical=declared_hierarchical,
        )
        .detach()
        .item()
    )
    extent = incumbent_pos.detach().amax(dim=0) - incumbent_pos.detach().amin(dim=0)
    diagonal = float(torch.linalg.vector_norm(extent).item())
    if not math.isfinite(diagonal) or diagonal <= 0.0:
        result = skipped("invalid_extent")
        _attach_continuous_facet_polish_telemetry(result, config)
        return result

    best_pos = incumbent_pos
    best_pair = incumbent_score_pair
    best_v3 = float(incumbent_v3)
    best_key = referee_key_fn(incumbent_pos) if referee_key_fn is not None else (1, -0.0)
    preserve_layered_reading = _layered_preservation_required(
        is_semantically_directed=is_semantically_directed,
        declared_hierarchical=declared_hierarchical,
        direction_is_declared=direction_is_declared,
    )
    preserve_layered_shape = preserve_layered_reading and gate_reason == "deep_tree_fan_spacing"
    incumbent_layered_shape = (
        incumbent_score_pair.g4_layered_parent_centering,
        incumbent_score_pair.g4_layered_subtree_congruence,
    )
    if preserve_layered_shape and (
        incumbent_layered_shape[0] is None or incumbent_layered_shape[1] is None
    ):
        incumbent_layered_shape = _continuous_facet_deep_tree_shape_scores(
            incumbent_pos,
            work_edge,
            work_sizes,
        )
    accepted: list[W5ContinuousFacetPolishCandidate] = []
    keepalive: list[torch.Tensor] = [incumbent_pos]
    evaluations = 0
    passes_completed = 0
    has_clusters = bool(_valid_cluster_members(clusters, node_count))

    if preserve_layered_shape:
        incumbent_centroid_offset = _deep_tree_max_child_centroid_offset_ratio(
            incumbent_pos,
            work_edge,
        )
        for strength in _DEEP_TREE_RANK_WARP_STRENGTHS:
            if not _continuous_facet_score_affordable(
                config=config,
                node_count=node_count,
                edge_count=edge_count,
                has_clusters=has_clusters,
                has_weights=has_weights,
            ):
                break
            candidate_pos = _deep_tree_rank_warp_candidate(best_pos, topo_depth, strength)
            if _is_degenerate(candidate_pos, work_sizes):
                continue
            candidate_rank_std = _deep_tree_rank_band_max_std(candidate_pos, topo_depth)
            if candidate_rank_std > _DEEP_TREE_MAX_RANK_BAND_STD:
                continue
            candidate_centroid_offset = _deep_tree_max_child_centroid_offset_ratio(
                candidate_pos,
                work_edge,
            )
            if candidate_centroid_offset > _DEEP_TREE_MAX_CHILD_CENTROID_OFFSET_RATIO:
                continue
            if candidate_centroid_offset > max(
                _DEEP_TREE_MAX_CHILD_CENTROID_OFFSET_RATIO,
                incumbent_centroid_offset + 1.0e-6,
            ):
                continue
            keepalive.append(candidate_pos)
            try:
                candidate_pair = score_fn(candidate_pos)
                candidate_key = (
                    referee_key_fn(candidate_pos) if referee_key_fn is not None else best_key
                )
            except Exception as exc:  # noqa: BLE001 -- optional polish candidate
                if is_worker_timeout_like_exception(exc):
                    raise
                continue
            evaluations += 1
            candidate_v3 = _finite_v3_score(candidate_pair)
            if candidate_key < best_key:
                continue
            if candidate_introduces_champion_ineligible_flag(
                candidate_pair.champion_ineligibility_flags,
                incumbent_score_pair.champion_ineligibility_flags,
            ):
                continue
            if candidate_v3 is None or float(candidate_v3) <= best_v3:
                continue
            if preserve_layered_reading and not _layered_reading_preserved(
                candidate_pair,
                incumbent_score_pair,
            ):
                continue
            candidate_layered_shape = (
                candidate_pair.g4_layered_parent_centering,
                candidate_pair.g4_layered_subtree_congruence,
            )
            if candidate_layered_shape[0] is None or candidate_layered_shape[1] is None:
                candidate_layered_shape = _continuous_facet_deep_tree_shape_scores(
                    candidate_pos,
                    work_edge,
                    work_sizes,
                )
            if not _layered_shape_preserved(candidate_pair, incumbent_score_pair):
                continue
            if not _layered_shape_values_preserved(
                candidate_layered_shape,
                incumbent_layered_shape,
            ):
                continue
            candidate_surrogate = float(
                _continuous_facet_surrogate_loss(
                    candidate_pos.detach().to(dtype=torch.float32),
                    work_edge,
                    topo_depth,
                    is_semantically_directed=is_semantically_directed,
                    declared_hierarchical=declared_hierarchical,
                )
                .detach()
                .item()
            )
            if float(candidate_v3) <= best_v3:
                continue
            best_pos = candidate_pos
            best_pair = candidate_pair
            best_v3 = float(candidate_v3)
            best_key = candidate_key
            accepted = [
                W5ContinuousFacetPolishCandidate(
                    pass_id=1,
                    node=-1,
                    step=float(strength),
                    direction=(0.0, 0.0),
                    score_pair=best_pair,
                    surrogate_loss=candidate_surrogate,
                    referee_key=best_key,
                    g4_layered_parent_centering=candidate_layered_shape[0],
                    g4_layered_subtree_congruence=candidate_layered_shape[1],
                )
            ]

        selected = (
            bool(accepted) and best_v3 > float(incumbent_v3) + _CONTINUOUS_FACET_POLISH_TIE_EPS
        )
        result = W5ContinuousFacetPolishResult(
            winner_pos=best_pos if selected else incumbent_pos,
            winner_score_pair=best_pair if selected else incumbent_score_pair,
            selected=selected,
            skipped_reason=None if selected else "no_rank_warp_improved_v3",
            gate_reason=gate_reason,
            passes_completed=1,
            evaluations=evaluations,
            accepted=tuple(accepted if selected else ()),
            keepalive=tuple(keepalive),
            start_surrogate_loss=start_surrogate,
            winner_surrogate_loss=accepted[-1].surrogate_loss if accepted else start_surrogate,
        )
        _attach_continuous_facet_polish_telemetry(result, config)
        return result

    for pass_id in range(1, int(max_passes) + 1):
        improved_this_pass = False
        passes_completed = pass_id
        for node in range(node_count):
            if wall_reserve_exhausted(config, _ABSOLUTE_DEADLINE_RESERVE_S):
                break
            for step_fraction in _CONTINUOUS_FACET_POLISH_STEP_FRACTIONS:
                step = float(step_fraction) * diagonal
                local_winner: (
                    tuple[
                        torch.Tensor,
                        W5ScorePair,
                        float,
                        Tuple[int, float],
                        tuple[float, float],
                        float,
                        tuple[Optional[float], Optional[float]],
                    ]
                    | None
                ) = None
                for direction in _CONTINUOUS_FACET_POLISH_DIRECTIONS:
                    if not _continuous_facet_score_affordable(
                        config=config,
                        node_count=node_count,
                        edge_count=edge_count,
                        has_clusters=has_clusters,
                        has_weights=has_weights,
                    ):
                        break
                    candidate_pos = best_pos.detach().clone()
                    candidate_pos[node, 0] += float(direction[0]) * step
                    candidate_pos[node, 1] += float(direction[1]) * step
                    if _is_degenerate(candidate_pos, work_sizes):
                        continue
                    keepalive.append(candidate_pos)
                    try:
                        candidate_pair = score_fn(candidate_pos)
                        candidate_key = (
                            referee_key_fn(candidate_pos)
                            if referee_key_fn is not None
                            else best_key
                        )
                    except Exception as exc:  # noqa: BLE001 -- optional polish candidate
                        if is_worker_timeout_like_exception(exc):
                            raise
                        continue
                    evaluations += 1
                    candidate_v3 = _finite_v3_score(candidate_pair)
                    if candidate_key < best_key:
                        continue
                    if candidate_introduces_champion_ineligible_flag(
                        candidate_pair.champion_ineligibility_flags,
                        incumbent_score_pair.champion_ineligibility_flags,
                    ):
                        continue
                    if candidate_v3 is None or float(candidate_v3) <= best_v3:
                        continue
                    if preserve_layered_reading and not _layered_reading_preserved(
                        candidate_pair,
                        incumbent_score_pair,
                    ):
                        continue
                    candidate_layered_shape = (
                        candidate_pair.g4_layered_parent_centering,
                        candidate_pair.g4_layered_subtree_congruence,
                    )
                    if preserve_layered_shape and (
                        candidate_layered_shape[0] is None or candidate_layered_shape[1] is None
                    ):
                        candidate_layered_shape = _continuous_facet_deep_tree_shape_scores(
                            candidate_pos,
                            work_edge,
                            work_sizes,
                        )
                    if preserve_layered_shape and not _layered_shape_preserved(
                        candidate_pair,
                        incumbent_score_pair,
                    ):
                        continue
                    if preserve_layered_shape and not _layered_shape_values_preserved(
                        candidate_layered_shape,
                        incumbent_layered_shape,
                    ):
                        continue
                    if local_winner is not None and float(candidate_v3) <= local_winner[2]:
                        continue
                    candidate_surrogate = float(
                        _continuous_facet_surrogate_loss(
                            candidate_pos.detach().to(dtype=torch.float32),
                            work_edge,
                            topo_depth,
                            is_semantically_directed=is_semantically_directed,
                            declared_hierarchical=declared_hierarchical,
                        )
                        .detach()
                        .item()
                    )
                    local_winner = (
                        candidate_pos,
                        candidate_pair,
                        float(candidate_v3),
                        candidate_key,
                        (float(direction[0]), float(direction[1])),
                        candidate_surrogate,
                        candidate_layered_shape,
                    )
                if local_winner is None:
                    continue
                (
                    best_pos,
                    best_pair,
                    best_v3,
                    best_key,
                    best_direction,
                    best_surrogate,
                    best_layered_shape,
                ) = local_winner
                accepted.append(
                    W5ContinuousFacetPolishCandidate(
                        pass_id=pass_id,
                        node=node,
                        step=step,
                        direction=best_direction,
                        score_pair=best_pair,
                        surrogate_loss=best_surrogate,
                        referee_key=best_key,
                        g4_layered_parent_centering=best_layered_shape[0],
                        g4_layered_subtree_congruence=best_layered_shape[1],
                    )
                )
                improved_this_pass = True
                break
        if not improved_this_pass:
            break

    selected = bool(accepted) and best_v3 > float(incumbent_v3) + _CONTINUOUS_FACET_POLISH_TIE_EPS
    result = W5ContinuousFacetPolishResult(
        winner_pos=best_pos if selected else incumbent_pos,
        winner_score_pair=best_pair if selected else incumbent_score_pair,
        selected=selected,
        skipped_reason=None if selected else "no_move_improved_v3",
        gate_reason=gate_reason,
        passes_completed=passes_completed,
        evaluations=evaluations,
        accepted=tuple(accepted if selected else ()),
        keepalive=tuple(keepalive),
        start_surrogate_loss=start_surrogate,
        winner_surrogate_loss=accepted[-1].surrogate_loss if accepted else start_surrogate,
    )
    _attach_continuous_facet_polish_telemetry(result, config)
    return result


def run_w5_terminal_small_n_anneal(
    *,
    incumbent_pos: torch.Tensor,
    incumbent_score_pair: W5ScorePair,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    score_fn: Callable[[torch.Tensor], W5ScorePair],
    referee_key_fn: Optional[Callable[[torch.Tensor], Tuple[int, float]]] = None,
    config: Optional[LayoutConfig] = None,
    trials: int = _W5_SMALL_N_ANNEAL_TRIALS,
    seed: int = _W5_SMALL_N_ANNEAL_SEED,
    max_nodes: int = _W5_SMALL_N_ANNEAL_MAX_NODES,
    has_clusters: bool = False,
    has_weights: bool = False,
    is_semantically_directed: bool = False,
    declared_hierarchical: bool = False,
    direction_is_declared: bool = False,
) -> W5SmallNAnnealResult:
    """Run seeded terminal small-N Gaussian annealing scored by restricted V3.

    Parameters
    ----------
    incumbent_pos : torch.Tensor
        Current terminal winner positions with shape ``[N, 2]``.
    incumbent_score_pair : W5ScorePair
        Restricted-V3-backed score pair for ``incumbent_pos``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    node_sizes : torch.Tensor
        Node-size tensor with shape ``[N, 2]``.
    score_fn : Callable[[torch.Tensor], W5ScorePair]
        Existing W5 scorer backed by the frozen restricted-V3 runtime referee.
    referee_key_fn : Callable[[torch.Tensor], tuple[int, float]], optional
        Severe-G6 referee-key scorer. Candidates whose key regresses against
        the current winner cannot be selected.
    config : LayoutConfig, optional
        Prepared layout configuration used for budget checks and telemetry.
    trials : int, default=_W5_SMALL_N_ANNEAL_TRIALS
        Maximum number of deterministic perturbation trials.
    seed : int, default=_W5_SMALL_N_ANNEAL_SEED
        Fixed anneal random seed.
    max_nodes : int, default=_W5_SMALL_N_ANNEAL_MAX_NODES
        Structural small-N gate.
    has_clusters : bool, default=False
        Whether runtime-visible clusters should be included in the score-cost
        guard.
    has_weights : bool, default=False
        Whether runtime-visible edge weights should be included in the
        score-cost guard.
    is_semantically_directed : bool, default=False
        Whether edge direction has semantic meaning.
    declared_hierarchical : bool, default=False
        Whether the row declares hierarchy metadata used by the frozen ruler.
    direction_is_declared : bool, default=False
        Whether semantic direction came from explicit user/config metadata.

    Returns
    -------
    W5SmallNAnnealResult
        Incumbent or strictly better restricted-V3 perturbation winner. Ties
        remain on the incumbent.
    """
    node_count = int(incumbent_pos.shape[0])
    edge_count = int(edge_index.shape[1]) if edge_index.ndim == 2 else 0

    def skipped(reason: str) -> W5SmallNAnnealResult:
        """Build and attach a no-op anneal result.

        Parameters
        ----------
        reason : str
            Stable skip reason.

        Returns
        -------
        W5SmallNAnnealResult
            No-op anneal result.
        """
        result = W5SmallNAnnealResult(
            winner_pos=incumbent_pos,
            winner_score_pair=incumbent_score_pair,
            selected=False,
            trials_completed=0,
            accepted_count=0,
            skipped_reason=reason,
            candidates=(),
            keepalive=(incumbent_pos,),
        )
        _attach_small_n_anneal_telemetry(result, config)
        return result

    incumbent_v3 = _finite_v3_score(incumbent_score_pair)
    if _w5_disabled_by_env():
        return skipped("disabled_by_env")
    if node_count < 2:
        return skipped("too_few_nodes")
    if node_count > int(max_nodes):
        return skipped("too_many_nodes")
    if incumbent_v3 is None:
        return skipped("missing_incumbent_v3")
    if remaining_dwu(config) is None:
        return skipped("no_deterministic_budget")
    slice_s = _finisher_slice_s(config)
    if slice_s is None:
        return skipped("no_budget")

    deadline = float("inf")
    median_edge = _median_edge_length(incumbent_pos, edge_index, node_sizes)
    sigma_hi = _W5_SMALL_N_ANNEAL_SIGMA_HI_FRACTION * median_edge
    sigma_lo = _W5_SMALL_N_ANNEAL_SIGMA_LO_FRACTION * median_edge
    if not math.isfinite(sigma_hi) or not math.isfinite(sigma_lo) or sigma_hi <= 0.0:
        return skipped("invalid_sigma")

    py_rng = random.Random(int(seed))
    torch_rng = torch.Generator(device="cpu").manual_seed(int(seed))
    best_pos = incumbent_pos
    best_pair = incumbent_score_pair
    best_v3 = float(incumbent_v3)
    best_key = referee_key_fn(incumbent_pos) if referee_key_fn is not None else (1, -0.0)
    preserve_layered_reading = _layered_preservation_required(
        is_semantically_directed=is_semantically_directed,
        declared_hierarchical=declared_hierarchical,
        direction_is_declared=direction_is_declared,
    )
    best_index: Optional[int] = None
    accepted_count = 0
    keepalive: list[torch.Tensor] = [incumbent_pos]
    candidates: list[W5SmallNAnnealCandidate] = []

    for trial in range(max(0, int(trials))):
        if (
            wall_reserve_exhausted(config, _ABSOLUTE_DEADLINE_RESERVE_S)
            or time.monotonic() >= deadline
        ):
            break
        if not _can_afford_small_n_anneal_score(
            config=config,
            node_count=node_count,
            edge_count=edge_count,
            has_clusters=has_clusters,
            has_weights=has_weights,
        ):
            break
        frac = float(trial) / float(max(int(trials) - 1, 1))
        sigma = sigma_hi * (sigma_lo / sigma_hi) ** frac
        changed_nodes = (
            1
            if py_rng.random() < _W5_SMALL_N_ANNEAL_SINGLE_NODE_PROBABILITY
            else max(1, node_count // 3)
        )
        sampled = py_rng.sample(range(node_count), int(changed_nodes))
        index = torch.tensor(sampled, dtype=torch.long, device=incumbent_pos.device)
        candidate_pos = best_pos.detach().clone()
        noise = torch.randn((int(changed_nodes), 2), generator=torch_rng, dtype=torch.float32)
        candidate_pos[index] += noise.to(
            device=candidate_pos.device,
            dtype=candidate_pos.dtype,
        ) * float(sigma)
        keepalive.append(candidate_pos)

        if _is_degenerate(candidate_pos, node_sizes.to(device=candidate_pos.device)):
            candidates.append(
                W5SmallNAnnealCandidate(
                    trial=trial,
                    sigma=float(sigma),
                    changed_nodes=int(changed_nodes),
                    score_pair=None,
                    referee_key=(0, float("-inf")),
                    selected=False,
                    reason="pre_score_degenerate",
                )
            )
            continue
        try:
            score_pair = score_fn(candidate_pos)
            referee_key = referee_key_fn(candidate_pos) if referee_key_fn is not None else best_key
        except Exception as exc:  # noqa: BLE001 -- terminal anneal is optional candidate work
            if is_worker_timeout_like_exception(exc):
                raise
            candidates.append(
                W5SmallNAnnealCandidate(
                    trial=trial,
                    sigma=float(sigma),
                    changed_nodes=int(changed_nodes),
                    score_pair=None,
                    referee_key=(0, float("-inf")),
                    selected=False,
                    reason="score_exception",
                )
            )
            continue

        candidate_v3 = _finite_v3_score(score_pair)
        reason = "missing_v3"
        selected = False
        if referee_key < best_key:
            reason = "referee_key_regressed"
        elif candidate_introduces_champion_ineligible_flag(
            score_pair.champion_ineligibility_flags,
            best_pair.champion_ineligibility_flags,
        ):
            reason = "introduced_champion_ineligible_flag"
        elif candidate_v3 is None:
            reason = "missing_v3"
        elif preserve_layered_reading and not _layered_reading_preserved(
            score_pair,
            incumbent_score_pair,
        ):
            reason = "layered_reading_regressed"
        elif float(candidate_v3) > best_v3 + _W5_SMALL_N_ANNEAL_TIE_EPS:
            best_pos = candidate_pos
            best_pair = score_pair
            best_v3 = float(candidate_v3)
            best_key = referee_key
            best_index = len(candidates)
            accepted_count += 1
            selected = True
            reason = "v3_argmax"
        else:
            reason = "does_not_improve_v3"
        candidates.append(
            W5SmallNAnnealCandidate(
                trial=trial,
                sigma=float(sigma),
                changed_nodes=int(changed_nodes),
                score_pair=score_pair,
                referee_key=referee_key,
                selected=selected,
                reason=reason,
            )
        )

    if best_index is not None:
        candidates = [
            W5SmallNAnnealCandidate(
                trial=candidate.trial,
                sigma=candidate.sigma,
                changed_nodes=candidate.changed_nodes,
                score_pair=candidate.score_pair,
                referee_key=candidate.referee_key,
                selected=index == best_index,
                reason=(
                    "v3_argmax"
                    if index == best_index
                    else (
                        "superseded_v3_argmax"
                        if candidate.reason == "v3_argmax"
                        else candidate.reason
                    )
                ),
            )
            for index, candidate in enumerate(candidates)
        ]

    result = W5SmallNAnnealResult(
        winner_pos=best_pos,
        winner_score_pair=best_pair,
        selected=best_index is not None,
        trials_completed=len(candidates),
        accepted_count=accepted_count,
        skipped_reason=None if candidates else "no_trials",
        candidates=tuple(candidates),
        keepalive=tuple(keepalive),
    )
    _attach_small_n_anneal_telemetry(result, config)
    return result


def _honest_scale_line_search(
    checkpoint_pos: torch.Tensor,
    score_fn: Callable[[torch.Tensor], W5ScorePair],
    tallied_axis: str,
    *,
    deadline: float,
    config: Optional[LayoutConfig],
    keepalive: list[torch.Tensor],
) -> W5ScaleSearchResult:
    """Run deterministic global-scale search scored by the honest referee.

    Parameters
    ----------
    checkpoint_pos : torch.Tensor
        Candidate checkpoint positions with shape ``[N, 2]``.
    score_fn : Callable[[torch.Tensor], W5ScorePair]
        Existing honest W5 scorer. In the native path this is backed by
        ``score_v3_runtime_result`` and its DWU charge/cache layer.
    tallied_axis : str
        Existing W5 axis used as fallback for legacy callers without V3 scores.
    deadline : float
        Absolute ``time.monotonic()`` deadline for non-measured W5 work.
    config : LayoutConfig, optional
        Prepared layout configuration used for hard wall-reserve checks.
    keepalive : list[torch.Tensor]
        Run-level tensor retention list used by id-keyed scorer caches.

    Returns
    -------
    W5ScaleSearchResult
        Best scored scale candidate. Legacy non-V3 score pairs return the raw
        checkpoint after one score evaluation.
    """

    def can_afford_next_referee_eval() -> bool:
        """Return whether the ledger can afford one more V3 referee score.

        Returns
        -------
        bool
            ``True`` when no ledger is active, or when the current remaining
            deterministic budget covers the same N-keyed charge used by the
            runtime V3 scorer.
        """
        ledger_remaining = remaining_dwu(config)
        if ledger_remaining is None:
            return True
        eval_cost = estimate_v3_referee_cost(
            int(checkpoint_pos.shape[0]),
            0,
            has_clusters=False,
            has_weights=False,
            device_class=_native_device_class(config),
        ).reserved_score_dwu
        return float(ledger_remaining) >= float(eval_cost)

    scored_keepalive = [checkpoint_pos]
    keepalive.append(checkpoint_pos)
    raw_pair = score_fn(checkpoint_pos)
    raw_scalar = _w5_score_scalar(raw_pair, tallied_axis)
    max_evals = _w5_scale_search_eval_cap(int(checkpoint_pos.shape[0]))
    if (
        raw_pair.v3 is None
        or raw_scalar is None
        or max_evals <= 1
        or not _w5_scale_search_facet_gate(raw_pair)
    ):
        return W5ScaleSearchResult(
            pos=checkpoint_pos,
            score_pair=raw_pair,
            raw_score_pair=raw_pair,
            scale=1.0,
            evals=1,
            keepalive=tuple(scored_keepalive),
        )
    best_pos = checkpoint_pos
    best_pair = raw_pair
    best_scalar = raw_scalar
    best_scale = 1.0
    evals = 1
    left = float(_W5_SCALE_SEARCH_MIN)
    right = float(_W5_SCALE_SEARCH_MAX)
    inv_phi = (math.sqrt(5.0) - 1.0) * 0.5

    def score_scale(scale: float) -> Optional[float]:
        """Score one scale and update the best candidate when it improves.

        Parameters
        ----------
        scale : float
            Global scale factor to evaluate.

        Returns
        -------
        float or None
            Finite scalar honest score for the scaled candidate.
        """
        nonlocal best_pair, best_pos, best_scalar, best_scale, evals
        if wall_reserve_exhausted(config, _ABSOLUTE_DEADLINE_RESERVE_S) or (
            math.isfinite(float(deadline)) and time.monotonic() >= deadline
        ):
            return None
        if not can_afford_next_referee_eval():
            return None
        scaled_pos = _scale_positions_about_centroid(checkpoint_pos, scale)
        scored_keepalive.append(scaled_pos)
        keepalive.append(scaled_pos)
        pair = score_fn(scaled_pos)
        evals += 1
        scalar = _w5_score_scalar(pair, tallied_axis)
        if scalar is not None and scalar > best_scalar:
            best_pos = scaled_pos
            best_pair = pair
            best_scalar = scalar
            best_scale = float(scale)
        return scalar

    analytic_scale = _w5_c5_band_scale_candidate(raw_pair)
    if analytic_scale is not None:
        score_scale(analytic_scale)

    x1 = right - inv_phi * (right - left)
    x2 = left + inv_phi * (right - left)
    score1: Optional[float] = None
    score2: Optional[float] = None
    while evals < max_evals:
        if score1 is None:
            score1 = score_scale(x1)
            if score1 is None:
                break
            continue
        if score2 is None:
            score2 = score_scale(x2)
            if score2 is None:
                break
            continue
        if score1 < score2:
            left = x1
            x1 = x2
            score1 = score2
            x2 = left + inv_phi * (right - left)
            score2 = None
        else:
            right = x2
            x2 = x1
            score2 = score1
            x1 = right - inv_phi * (right - left)
            score1 = None
    return W5ScaleSearchResult(
        pos=best_pos,
        score_pair=best_pair,
        raw_score_pair=raw_pair,
        scale=best_scale,
        evals=evals,
        keepalive=tuple(scored_keepalive),
    )


def _measure_one_surrogate_step_s(
    seed: W5Seed,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    topo_depth: torch.Tensor,
    mode: str,
    honest_axes: Optional[W5HonestAxes],
    measurement_budget_s: float,
    shape_geometry: Optional[NativeShapeGeometry] = None,
) -> W5StepMeasurement:
    """Measure wall-clock cost for one steady-state W5 surrogate step.

    Parameters
    ----------
    seed : W5Seed
        Finite warm start used as the measurement surrogate.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    node_sizes : torch.Tensor
        Node-size tensor with shape ``[N, 2]``.
    topo_depth : torch.Tensor
        Depth tensor with shape ``[N]``.
    mode : str
        Routed W5 optimization mode.
    honest_axes : W5HonestAxes, optional
        Honest incumbent axes used by barrier weighting.
    measurement_budget_s : float
        Wall-clock seconds available for the surrogate probe.
    shape_geometry : NativeShapeGeometry, optional
        Optional non-box shape descriptors.

    Returns
    -------
    W5StepMeasurement
        Positive wall-clock seconds for steady-state step cost and one-time
        first-step warmup.
    """
    step_times_s: list[float] = []

    def timing_hook(_step: int, duration_s: float) -> None:
        """Record one measured optimizer step duration.

        Parameters
        ----------
        _step : int
            Completed step index, unused by the averaging logic.
        duration_s : float
            Wall-clock duration for the completed step.

        Returns
        -------
        None
            ``step_times_s`` is appended in place.
        """
        step_times_s.append(duration_s)

    if shape_geometry is None:
        _optimize_seed(
            seed,
            edge_index,
            node_sizes,
            topo_depth,
            mode,
            time.monotonic() + max(1.0e-6, measurement_budget_s),
            honest_axes,
            max_steps=_MEASURED_COST_SURROGATE_STEPS,
            max_checkpoints=0,
            step_timing_hook=timing_hook,
        )
    else:
        _optimize_seed(
            seed,
            edge_index,
            node_sizes,
            topo_depth,
            mode,
            time.monotonic() + max(1.0e-6, measurement_budget_s),
            honest_axes,
            max_steps=_MEASURED_COST_SURROGATE_STEPS,
            max_checkpoints=0,
            step_timing_hook=timing_hook,
            shape_geometry=shape_geometry,
        )
    if not step_times_s:
        return W5StepMeasurement(step_s=1.0e-6, warmup_s=0.0)
    warmup_s = max(0.0, step_times_s[0])
    if len(step_times_s) == 1:
        step_s = warmup_s
        warmup_s = 0.0
    else:
        steady_times_s = step_times_s[1:]
        step_s = sum(steady_times_s) / float(len(steady_times_s))
    return W5StepMeasurement(step_s=max(1.0e-6, step_s), warmup_s=warmup_s)


def _measured_cost_plan(
    *,
    seeds: Sequence[W5Seed],
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    topo_depth: torch.Tensor,
    routed_mode: str,
    slice_s: float,
    config: Optional[LayoutConfig],
    started_perf: float,
    started_process: float,
    remaining_entry: Optional[float],
    honest_axes: Optional[W5HonestAxes],
    shape_geometry: Optional[NativeShapeGeometry] = None,
) -> Optional[W5CostPlan]:
    """Return a modeled W5 plan that fits the deterministic ledger cap.

    Parameters
    ----------
    seeds : Sequence[W5Seed]
        Finite deduplicated W5 seeds.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    node_sizes : torch.Tensor
        Node-size tensor with shape ``[N, 2]``.
    topo_depth : torch.Tensor
        Depth tensor with shape ``[N]``.
    routed_mode : str
        First routed W5 mode.
    slice_s : float
        Work slice admitted by deadline gates.
    config : LayoutConfig, optional
        Prepared configuration carrying spend and referee measurements.
    started_perf : float
        ``time.perf_counter()`` value captured at W5 entry.
    started_process : float
        ``time.process_time()`` value captured at W5 entry.
    remaining_entry : float, optional
        Benchmark process seconds remaining at W5 entry.
    honest_axes : W5HonestAxes, optional
        Honest incumbent axes used by barrier weighting.
    shape_geometry : NativeShapeGeometry, optional
        Optional non-box shape descriptors.

    Returns
    -------
    W5CostPlan or None
        Admitted plan, or ``None`` when one seed and one checkpoint cannot fit.
        The live surrogate probe is retained as shadow telemetry and does not
        affect the returned plan shape or modeled costs.
    """
    node_count = int(seeds[0].pos.shape[0])
    edge_count = int(edge_index.shape[1]) if edge_index.ndim == 2 else 0
    measurement_budget_s = max(1.0e-6, min(float(slice_s), _MEASURED_COST_SURROGATE_STEPS * 0.25))
    step_measurement = _measure_one_surrogate_step_s(
        seeds[0],
        edge_index,
        node_sizes,
        topo_depth,
        routed_mode,
        honest_axes,
        measurement_budget_s,
        shape_geometry,
    )
    del started_perf, started_process
    ledger_remaining = remaining_dwu(config)
    cap_remaining = _w5_spend_cap_s(config, remaining_entry)
    if ledger_remaining is not None:
        cap_remaining = min(cap_remaining, float(ledger_remaining))
    budget_s = max(0.0, min(float(slice_s), cap_remaining))
    usable_s = budget_s - _PREDICTED_COST_RETURN_RESERVE_S
    base_steps = 24 if node_count >= 300 else 36
    base_seeds = min(3, len(seeds))
    base_checkpoints = _MEASURED_COST_MAX_CHECKPOINTS
    max_checkpoints = base_checkpoints

    def build_plan(seed_count: int, steps: int, checkpoints: int) -> W5CostPlan:
        """Build a model-denominated plan payload for candidate work bounds.

        Parameters
        ----------
        seed_count : int
            Number of W5 seeds to run.
        steps : int
            Optimizer steps per seed.
        checkpoints : int
            Honest checkpoints per seed.

        Returns
        -------
        W5CostPlan
            Cost plan priced solely by the frozen W5 cost model.
        """
        scale_search_evals = _w5_scale_search_eval_cap(node_count)
        reserved_checkpoints = int(checkpoints) * scale_search_evals
        cost = estimate_native_work_cost(
            {"num_nodes": node_count, "num_edges": edge_count},
            "w5",
            {
                "mode": routed_mode,
                "steps": steps,
                "seeds": seed_count,
                "checkpoints": reserved_checkpoints,
            },
            _native_device_class(config),
        )
        predicted_s = cost.generation_dwu + cost.reserved_score_dwu
        step_volume = max(1, steps * max(seed_count, 1))
        referee_volume = max(1, checkpoints * max(seed_count, 1))
        step_s = cost.generation_dwu / float(step_volume)
        referee_s = cost.reserved_score_dwu / float(referee_volume)
        return W5CostPlan(
            seeds=seed_count,
            steps=steps,
            checkpoints=checkpoints,
            measured_step_s=step_s,
            warmup_s=step_measurement.warmup_s,
            referee_s=referee_s,
            budget_s=budget_s,
            budget_usable_s=usable_s,
            predicted_s=predicted_s,
            shadow_step_s=step_measurement.step_s,
            shadow_warmup_s=step_measurement.warmup_s,
            scale_search_evals=scale_search_evals,
        )

    minimum_plan = build_plan(1, 1, 1)
    if usable_s < minimum_plan.predicted_s:
        if config is not None:
            setattr(config, "_dagua_native_w5_cost_plan", minimum_plan)
        return None

    base_plan = build_plan(base_seeds, base_steps, base_checkpoints)
    if base_plan.predicted_s <= usable_s:
        if node_count <= _MEASURED_COST_TINY_MAX_N:
            for steps in range(_MEASURED_COST_TINY_STEPS, base_steps - 1, -1):
                for checkpoints in range(
                    _MEASURED_COST_TINY_MAX_CHECKPOINTS,
                    base_checkpoints - 1,
                    -1,
                ):
                    raised_plan = build_plan(base_seeds, steps, checkpoints)
                    if raised_plan.predicted_s <= usable_s:
                        if config is not None:
                            setattr(config, "_dagua_native_w5_cost_plan", raised_plan)
                        return raised_plan
        if config is not None:
            setattr(config, "_dagua_native_w5_cost_plan", base_plan)
        return base_plan

    max_seeds = min(_MEASURED_COST_MAX_SEEDS, base_seeds)
    for steps in range(base_steps, 0, -1):
        for checkpoints in range(max_checkpoints, 0, -1):
            for seed_count in range(max_seeds, 0, -1):
                candidate_plan = build_plan(seed_count, steps, checkpoints)
                if candidate_plan.predicted_s <= usable_s:
                    if config is not None:
                        setattr(config, "_dagua_native_w5_cost_plan", candidate_plan)
                    return candidate_plan
    return None


def run_w5_finisher(
    *,
    incumbent_pos: torch.Tensor,
    incumbent_score_pair: W5ScorePair,
    seeds: Sequence[W5Seed],
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    score_fn: Callable[[torch.Tensor], W5ScorePair],
    is_semantically_directed: bool,
    declared_hierarchical: bool,
    direction_is_declared: bool = False,
    config: Optional[LayoutConfig] = None,
    accept_margin: float = _W5_ACCEPT_MARGIN,
    incumbent_axes: Optional[W5HonestAxes] = None,
    shape_geometry: Optional[NativeShapeGeometry] = None,
    referee_key_fn: Optional[Callable[[torch.Tensor], Tuple[int, float]]] = None,
) -> W5FinisherResult:
    """Run the W5 finisher and return the anytime honest winner.

    Parameters
    ----------
    incumbent_pos : torch.Tensor
        Incumbent positions with shape ``[N, 2]``.
    incumbent_score_pair : W5ScorePair
        Directed and undirected honest scores for ``incumbent_pos``.
    seeds : Sequence[W5Seed]
        Warm starts already generated by the native contest/polish path.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    node_sizes : torch.Tensor
        Node-size tensor with shape ``[N, 2]``.
    score_fn : Callable[[torch.Tensor], W5ScorePair]
        Dual-ruler honest scorer.
    is_semantically_directed : bool
        Whether edge direction has semantic meaning.
    declared_hierarchical : bool
        Whether the honest ruler uses directed hierarchy terms.
    direction_is_declared : bool, default=False
        Whether semantic direction came from user/config metadata.
    config : LayoutConfig, optional
        Prepared native configuration carrying optional benchmark deadline.
    accept_margin : float, default=0.05
        Required improvement over the current winner in both score components.
    incumbent_axes : W5HonestAxes, optional
        Honest incumbent axes from the same ``full()`` metrics pass that
        produced ``incumbent_score_pair``.
    shape_geometry : NativeShapeGeometry, optional
        Optional non-box shape descriptors for overlap loss and viability.
    referee_key_fn : Callable[[torch.Tensor], tuple[int, float]], optional
        Severe-G6 referee-key scorer. When omitted, all candidates use the
        neutral prefix and W5 remains the historical dual-composite gate.

    Returns
    -------
    W5FinisherResult
        Anytime winner plus telemetry for all honest-scored checkpoints.
    """
    started_perf = time.perf_counter()
    started_process = time.process_time()
    remaining_entry = _process_remaining_s(config)
    slice_s = _finisher_slice_s(config)
    node_count = int(incumbent_pos.shape[0])
    edge_count = int(edge_index.shape[1]) if edge_index.ndim == 2 else 0
    tallied_axis = (
        "directed" if is_semantically_directed and declared_hierarchical else "undirected"
    )
    preserve_layered_reading = _layered_preservation_required(
        is_semantically_directed=is_semantically_directed,
        declared_hierarchical=declared_hierarchical,
        direction_is_declared=direction_is_declared,
    )
    predicted_skip_reason = w5_predicted_skip_reason(node_count, edge_count, config)
    cost_plan: Optional[W5CostPlan] = None
    incumbent_referee_key = (
        referee_key_fn(incumbent_pos) if referee_key_fn is not None else (1, -0.0)
    )
    if slice_s is None and predicted_skip_reason != "disabled_by_env":
        predicted_skip_reason = None
    if predicted_skip_reason is not None:
        slice_s = None

    def finish(
        *,
        winner_pos: torch.Tensor,
        winner_score_pair: W5ScorePair,
        winner_name: str,
        deadline_returned: bool,
        accepted: list[W5Candidate],
        rejected: list[W5Checkpoint],
        checkpoints: list[W5Checkpoint],
        phase_timings: list[W5PhaseTiming],
        viability_counts: dict[str, int],
        viability_drop_counts: dict[str, int],
        mode: str,
        steps: int,
        skipped_reason: Optional[str],
    ) -> W5FinisherResult:
        """Finalize result metadata and update config W5 spend.

        Parameters
        ----------
        winner_pos : torch.Tensor
            Current winner tensor with shape ``[N, 2]``.
        winner_score_pair : W5ScorePair
            Directed and undirected scores for ``winner_pos``.
        winner_name : str
            Current winner label.
        deadline_returned : bool
            Whether the return was forced by a deadline or exhausted budget.
        accepted : list[W5Candidate]
            Accepted checkpoint candidates.
        rejected : list[W5Checkpoint]
            Rejected checkpoint telemetry.
        checkpoints : list[W5Checkpoint]
            All honest-scored checkpoint telemetry.
        phase_timings : list[W5PhaseTiming]
            Per-seed phase timing records.
        viability_counts : dict[str, int]
            Viability outcome counts.
        viability_drop_counts : dict[str, int]
            Pre-score viability drop counts by reason.
        mode : str
            Last routed mode or ``"skip"``.
        steps : int
            Optimizer steps completed.
        skipped_reason : str, optional
            Reason no checkpoint was accepted or no work ran.

        Returns
        -------
        W5FinisherResult
            Finalized W5 result.
        """
        winner_referee_key = (
            referee_key_fn(winner_pos) if referee_key_fn is not None else incumbent_referee_key
        )
        if winner_pos is not incumbent_pos and not _w5_dominates_with_axis(
            winner_score_pair,
            incumbent_score_pair,
            float(accept_margin),
            candidate_referee_key=winner_referee_key,
            incumbent_referee_key=incumbent_referee_key,
            tallied_axis=tallied_axis,
            preserve_layered_reading=preserve_layered_reading,
        ):
            winner_pos = incumbent_pos
            winner_score_pair = incumbent_score_pair
            winner_name = "incumbent"
            skipped_reason = "clamped_to_incumbent"
        spent_s = max(0.0, time.perf_counter() - started_perf)
        process_spent_s = max(0.0, time.process_time() - started_process)
        if config is not None:
            previous_spent = float(getattr(config, "_dagua_native_w5_spent_s", 0.0))
            setattr(config, "_dagua_native_w5_spent_s", previous_spent + spent_s)
            previous_process_spent = float(getattr(config, "_dagua_native_w5_process_spent_s", 0.0))
            setattr(
                config,
                "_dagua_native_w5_process_spent_s",
                previous_process_spent + process_spent_s,
            )
        return W5FinisherResult(
            winner_pos=winner_pos,
            incumbent_score_pair=incumbent_score_pair,
            winner_score_pair=winner_score_pair,
            winner_name=winner_name,
            deadline_returned=deadline_returned,
            accepted=tuple(accepted),
            rejected=tuple(rejected),
            checkpoints=tuple(checkpoints),
            mode=mode,
            steps=steps,
            skipped_reason=skipped_reason,
            slice_s=slice_s,
            spent_s=spent_s,
            process_spent_s=process_spent_s,
            remaining_entry_s=remaining_entry,
            remaining_exit_s=_remaining_s(config),
            node_count=node_count,
            edge_count=edge_count,
            is_semantically_directed=is_semantically_directed,
            declared_hierarchical=declared_hierarchical,
            direction_is_declared=direction_is_declared,
            graph_name=_graph_name(config),
            incumbent_axes=incumbent_axes,
            phase_timings_s=tuple(phase_timings),
            viability_counts=dict(viability_counts),
            viability_drop_counts=dict(viability_drop_counts),
            cost_plan=cost_plan,
        )

    if slice_s is None:
        return finish(
            winner_pos=incumbent_pos,
            winner_score_pair=incumbent_score_pair,
            winner_name="incumbent",
            deadline_returned=predicted_skip_reason in {None, "predicted_cost_late_entry"},
            accepted=[],
            rejected=[],
            checkpoints=[],
            phase_timings=[],
            viability_counts={},
            viability_drop_counts={},
            mode="skip",
            steps=0,
            skipped_reason=predicted_skip_reason or "no_budget",
        )
    use_measured_cost = bool(getattr(config, "_dagua_native_w5_measured_sizing", False))
    kept_seeds = _dedupe_seeds(
        seeds,
        max_seeds=3,
    )
    if not kept_seeds:
        return finish(
            winner_pos=incumbent_pos,
            winner_score_pair=incumbent_score_pair,
            winner_name="incumbent",
            deadline_returned=False,
            accepted=[],
            rejected=[],
            checkpoints=[],
            phase_timings=[],
            viability_counts={},
            viability_drop_counts={},
            mode="skip",
            steps=0,
            skipped_reason="no_finite_seed",
        )
    # The anytime wall deadline only exists in wall-deadline benchmark mode
    # (an installed hard wall deadline). The deterministic default path runs
    # the fixed step/checkpoint plan with an infinite deadline so the output
    # can never depend on machine load (wall-clock robustness invariant).
    deadline = time.monotonic() + slice_s if _remaining_s(config) is not None else float("inf")
    edge_work = edge_index.detach().to(device=kept_seeds[0].pos.device, dtype=torch.long)
    size_work = node_sizes.detach().to(device=kept_seeds[0].pos.device, dtype=torch.float32)
    shape_work = (
        shape_geometry.to(device=kept_seeds[0].pos.device, dtype=torch.float32)
        if shape_geometry is not None
        else None
    )
    topo_depth = _longest_path_depth(
        edge_work,
        int(kept_seeds[0].pos.shape[0]),
        kept_seeds[0].pos.device,
    )
    stress_sample: Optional[W5StressSample] = None
    neighborhood_sample: Optional[W5NeighborhoodSample] = None
    guidance_samples_ready = False
    max_steps: Optional[int] = None
    max_checkpoints = _MEASURED_COST_MAX_CHECKPOINTS
    if use_measured_cost:
        first_mode = _route_mode(
            kept_seeds[0].pos.detach().to(device=edge_work.device, dtype=torch.float32),
            edge_work,
            topo_depth,
            is_semantically_directed=is_semantically_directed,
            declared_hierarchical=declared_hierarchical,
            direction_is_declared=direction_is_declared,
            honest_axes=incumbent_axes,
        )
        cost_plan = _measured_cost_plan(
            seeds=kept_seeds,
            edge_index=edge_work,
            node_sizes=size_work,
            topo_depth=topo_depth,
            routed_mode=first_mode,
            slice_s=float(slice_s),
            config=config,
            started_perf=started_perf,
            started_process=started_process,
            remaining_entry=remaining_entry,
            honest_axes=incumbent_axes,
            shape_geometry=shape_work,
        )
        if cost_plan is None:
            cost_plan = getattr(config, "_dagua_native_w5_cost_plan", None)
            return finish(
                winner_pos=incumbent_pos,
                winner_score_pair=incumbent_score_pair,
                winner_name="incumbent",
                deadline_returned=True,
                accepted=[],
                rejected=[],
                checkpoints=[],
                phase_timings=[],
                viability_counts={},
                viability_drop_counts={},
                mode="skip",
                steps=0,
                skipped_reason="predicted_cost_measured",
            )
        _charge_w5_owner_plan(
            config,
            node_count,
            edge_count,
            first_mode,
            cost_plan,
        )
        kept_seeds = kept_seeds[: cost_plan.seeds]
        max_steps = cost_plan.steps
        max_checkpoints = cost_plan.checkpoints
    winner_pos = incumbent_pos
    winner_score_pair = incumbent_score_pair
    winner_name = "incumbent"
    winner_referee_key = incumbent_referee_key
    accepted: list[W5Candidate] = []
    rejected: list[W5Checkpoint] = []
    checkpoints: list[W5Checkpoint] = []
    phase_timings: list[W5PhaseTiming] = []
    viability_counts: dict[str, int] = {}
    viability_drop_counts: dict[str, int] = {}
    scale_search_keepalive: list[torch.Tensor] = []
    steps_total = 0
    routed_mode = "skip"
    incumbent_overlap = _overlap_count(incumbent_pos, size_work, shape_work)
    deadline_returned = False
    first_score_epilogue_attempted = False
    for seed in kept_seeds:
        seed_accepted_entry = len(accepted)
        if wall_reserve_exhausted(config, _ABSOLUTE_DEADLINE_RESERVE_S):
            deadline_returned = True
            break
        route_started = time.perf_counter()
        mode = _route_mode(
            seed.pos.detach().to(device=edge_work.device, dtype=torch.float32),
            edge_work,
            topo_depth,
            is_semantically_directed=is_semantically_directed,
            declared_hierarchical=declared_hierarchical,
            direction_is_declared=direction_is_declared,
            honest_axes=incumbent_axes,
        )
        route_s = max(0.0, time.perf_counter() - route_started)
        for ladder_index, mode in enumerate(
            _mode_ladder(mode, is_semantically_directed=is_semantically_directed)
        ):
            if wall_reserve_exhausted(config, _ABSOLUTE_DEADLINE_RESERVE_S):
                deadline_returned = True
                break
            routed_mode = mode
            mode_seed = (
                W5Seed(f"{seed.name}_incumbent", winner_pos)
                if len(accepted) > seed_accepted_entry
                else seed
            )
            pass_seed = mode_seed
            for pass_id in (1, 2):
                if deadline_returned:
                    break
                if wall_reserve_exhausted(config, _ABSOLUTE_DEADLINE_RESERVE_S):
                    deadline_returned = True
                    break
                if not guidance_samples_ready:
                    all_pairs_dist = _closed_over_all_pairs_dist(score_fn)
                    stress_sample = _build_w5_stress_sample(
                        edge_work,
                        int(kept_seeds[0].pos.shape[0]),
                        all_pairs_dist,
                        kept_seeds[0].pos.device,
                    )
                    neighborhood_sample = _build_w5_neighborhood_sample(
                        all_pairs_dist,
                        int(kept_seeds[0].pos.shape[0]),
                        kept_seeds[0].pos.device,
                    )
                    guidance_samples_ready = True
                optimize_s = 0.0
                viability_s = 0.0
                score_s = 0.0
                try:
                    optimize_started = time.perf_counter()
                    final_pos, steps, start_loss, scored_points = _run_optimize_seed_pass(
                        pass_seed,
                        edge_work,
                        size_work,
                        topo_depth,
                        mode,
                        float("inf") if use_measured_cost else deadline - _FINISHER_SCORE_RESERVE_S,
                        incumbent_axes,
                        max_steps=max_steps if use_measured_cost else None,
                        max_checkpoints=max_checkpoints,
                        pass_id=pass_id,
                        stress_sample=stress_sample,
                        neighborhood_sample=neighborhood_sample,
                        shape_geometry=shape_work,
                    )
                    optimize_s = max(0.0, time.perf_counter() - optimize_started)
                except Exception as exc:  # noqa: BLE001 -- W5 is optional candidate generation
                    if is_worker_timeout_like_exception(exc):
                        raise
                    _LOGGER.warning("W5 finisher seed %s failed", seed.name, exc_info=True)
                    phase_timings.append(
                        W5PhaseTiming(
                            seed=seed.name,
                            mode=mode,
                            pass_id=pass_id,
                            route_s=route_s if ladder_index == 0 and pass_id == 1 else 0.0,
                            optimize_s=optimize_s,
                            viability_s=viability_s,
                            score_s=score_s,
                        )
                    )
                    continue
                steps_total += steps
                if not scored_points or scored_points[-1][0] != steps:
                    final_loss = float(
                        _pass_loss(
                            final_pos,
                            edge_work,
                            size_work,
                            topo_depth,
                            mode,
                            {},
                            pass_id,
                            stress_sample,
                            neighborhood_sample,
                            shape_work,
                        )
                        .detach()
                        .item()
                    )
                    scored_points.append((steps, final_pos, final_loss))
                for step, checkpoint_pos, checkpoint_loss in scored_points[:max_checkpoints]:
                    epilogue_scoring = False
                    if wall_reserve_exhausted(config, _ABSOLUTE_DEADLINE_RESERVE_S) or (
                        not use_measured_cost and time.monotonic() >= deadline
                    ):
                        deadline_returned = True
                        if (
                            checkpoints
                            or first_score_epilogue_attempted
                            or not scored_points
                            or not _w5_first_score_epilogue_has_wall_headroom(config)
                        ):
                            break
                        step, checkpoint_pos, checkpoint_loss = scored_points[-1]
                        first_score_epilogue_attempted = True
                        epilogue_scoring = True
                    viability_started = time.perf_counter()
                    checkpoint_overlap = _overlap_count(checkpoint_pos, size_work, shape_work)
                    if checkpoint_overlap > incumbent_overlap:
                        _increment_count(viability_counts, "projected_overlap_candidate")
                        if shape_work is None:
                            checkpoint_pos = _project_checkpoint_for_viability(
                                checkpoint_pos,
                                size_work,
                            )
                        else:
                            checkpoint_pos = _project_checkpoint_for_viability(
                                checkpoint_pos,
                                size_work,
                                shape_work,
                            )
                        projected_overlap = _overlap_count(checkpoint_pos, size_work, shape_work)
                        if projected_overlap <= incumbent_overlap:
                            _increment_count(viability_counts, "projection_resolved_overlap")
                    if _is_degenerate(checkpoint_pos, size_work):
                        _increment_count(viability_counts, "drop_degenerate")
                        _increment_count(viability_drop_counts, "degenerate")
                        viability_s += max(0.0, time.perf_counter() - viability_started)
                        if epilogue_scoring:
                            break
                        continue
                    if _overlap_count(checkpoint_pos, size_work, shape_work) > incumbent_overlap:
                        _increment_count(viability_counts, "scored_overlap_regressed")
                    else:
                        _increment_count(viability_counts, "scored_viable")
                    viability_s += max(0.0, time.perf_counter() - viability_started)
                    try:
                        score_started = time.perf_counter()
                        score_pos = checkpoint_pos.to(device=edge_index.device, dtype=torch.float32)
                        raw_score_pos = score_pos
                        scale_search = _honest_scale_line_search(
                            score_pos,
                            score_fn,
                            tallied_axis,
                            deadline=float("inf") if use_measured_cost else deadline,
                            config=config,
                            keepalive=scale_search_keepalive,
                        )
                        score_pos = scale_search.pos
                        scaled_checkpoint_pos = scale_search.pos.to(
                            device=checkpoint_pos.device,
                            dtype=checkpoint_pos.dtype,
                        )
                        honest = scale_search.score_pair
                        if abs(float(scale_search.scale) - 1.0) > 1.0e-12:
                            scaled_referee_key = (
                                referee_key_fn(score_pos)
                                if referee_key_fn is not None
                                else winner_referee_key
                            )
                            if _w5_scaled_candidate_should_fallback(
                                scaled_checkpoint_pos,
                                size_work,
                                shape_work,
                                incumbent_overlap,
                                honest,
                                winner_score_pair,
                                candidate_referee_key=scaled_referee_key,
                                incumbent_referee_key=winner_referee_key,
                                tallied_axis=tallied_axis,
                                accept_margin=float(accept_margin),
                            ):
                                _increment_count(viability_counts, "scaled_viability_fallback")
                                score_pos = raw_score_pos
                                honest = scale_search.raw_score_pair
                            else:
                                checkpoint_pos = scaled_checkpoint_pos
                        else:
                            checkpoint_pos = scaled_checkpoint_pos
                        score_s += max(0.0, time.perf_counter() - score_started)
                    except Exception as exc:
                        score_s += max(0.0, time.perf_counter() - score_started)
                        if is_worker_timeout_like_exception(exc):
                            raise
                        _increment_count(viability_counts, "drop_score_exception")
                        if epilogue_scoring:
                            break
                        continue
                    if not math.isfinite(honest.directed) or not math.isfinite(honest.undirected):
                        _increment_count(viability_counts, "drop_nonfinite_score")
                        if epilogue_scoring:
                            break
                        continue
                    directed_delta = honest.directed - winner_score_pair.directed
                    undirected_delta = honest.undirected - winner_score_pair.undirected
                    surrogate_delta = start_loss - checkpoint_loss
                    checkpoint_referee_key = (
                        referee_key_fn(score_pos)
                        if referee_key_fn is not None
                        else winner_referee_key
                    )
                    is_accepted = _w5_dominates_with_axis(
                        honest,
                        winner_score_pair,
                        float(accept_margin),
                        candidate_referee_key=checkpoint_referee_key,
                        incumbent_referee_key=winner_referee_key,
                        tallied_axis=tallied_axis,
                        preserve_layered_reading=preserve_layered_reading,
                    )
                    legacy_tallied_sole_failure = w5_legacy_tallied_sole_failure(
                        honest,
                        winner_score_pair,
                        float(accept_margin),
                        candidate_referee_key=checkpoint_referee_key,
                        incumbent_referee_key=winner_referee_key,
                        tallied_axis=tallied_axis,
                    )
                    reason = "dominates" if is_accepted else "does_not_dominate_both"
                    checkpoint = W5Checkpoint(
                        seed=seed.name,
                        mode=mode,
                        pass_id=pass_id,
                        step=int(step),
                        surrogate_delta=float(surrogate_delta),
                        honest_delta=float(directed_delta),
                        undirected_honest_delta=float(undirected_delta),
                        honest_score_pair=honest,
                        accepted=is_accepted,
                        reason=reason,
                        pass_spend_s=float(optimize_s + viability_s + score_s),
                        legacy_tallied_sole_failure=legacy_tallied_sole_failure,
                    )
                    checkpoints.append(checkpoint)
                    if is_accepted:
                        name = f"w5_p{pass_id}_{mode}_{seed.name}_{step}"
                        winner_pos = checkpoint_pos.to(
                            device=incumbent_pos.device,
                            dtype=incumbent_pos.dtype,
                        )
                        winner_score_pair = honest
                        winner_name = name
                        winner_referee_key = checkpoint_referee_key
                        accepted_candidate = W5Candidate(
                            name=name,
                            pos=winner_pos,
                            score_pair=honest,
                            mode=mode,
                        )
                        accepted.append(accepted_candidate)
                        incumbent_overlap = min(
                            incumbent_overlap,
                            _overlap_count(winner_pos, size_work, shape_work),
                        )
                    else:
                        rejected.append(checkpoint)
                    if epilogue_scoring:
                        break
                phase_timings.append(
                    W5PhaseTiming(
                        seed=seed.name,
                        mode=mode,
                        pass_id=pass_id,
                        route_s=route_s if ladder_index == 0 and pass_id == 1 else 0.0,
                        optimize_s=optimize_s,
                        viability_s=viability_s,
                        score_s=score_s,
                    )
                )
                pass_seed_pos = winner_pos
                pass_seed = W5Seed(f"{seed.name}_p{pass_id}", pass_seed_pos)
                if deadline_returned:
                    break
            if deadline_returned:
                break
    skipped = None if accepted else ("no_checkpoint_improved" if checkpoints else "no_checkpoint")
    return finish(
        winner_pos=winner_pos,
        winner_score_pair=winner_score_pair,
        winner_name=winner_name,
        deadline_returned=deadline_returned,
        accepted=accepted,
        rejected=rejected,
        checkpoints=checkpoints,
        phase_timings=phase_timings,
        viability_counts=viability_counts,
        viability_drop_counts=viability_drop_counts,
        mode=routed_mode,
        steps=steps_total,
        skipped_reason=skipped,
    )


def log_w5_telemetry(result: W5FinisherResult, config: Optional[LayoutConfig]) -> None:
    """Emit and attach structured W5 finisher telemetry.

    Parameters
    ----------
    result : W5FinisherResult
        Finisher result to report.
    config : LayoutConfig, optional
        Config receiving ``_dagua_native_w5_telemetry`` when available.

    Returns
    -------
    None
        Telemetry is logged and optionally stored on ``config``.
    """

    def pair_payload(pair: W5ScorePair) -> dict[str, float]:
        """Convert a score pair to a JSON-serializable payload.

        Parameters
        ----------
        pair : W5ScorePair
            Directed and undirected score pair.

        Returns
        -------
        dict[str, float]
            JSON-ready score fields.
        """
        return {"directed": float(pair.directed), "undirected": float(pair.undirected)}

    def axes_payload(axes: Optional[W5HonestAxes]) -> Optional[dict[str, Optional[float]]]:
        """Convert honest route axes to a JSON-serializable payload.

        Parameters
        ----------
        axes : W5HonestAxes, optional
            Honest incumbent axes used by the route.

        Returns
        -------
        dict[str, float | None] or None
            JSON-ready axis payload.
        """
        if axes is None:
            return None
        return {
            "flow": axes.flow,
            "depth": axes.depth,
            "ksm": axes.ksm,
            "edge_length": axes.edge_length,
        }

    payload = {
        "event": "native_w5_finisher",
        "graph_name": result.graph_name,
        "node_count": result.node_count,
        "edge_count": result.edge_count,
        "mode": result.mode,
        "steps": result.steps,
        "skipped_reason": result.skipped_reason,
        "deadline_returned": result.deadline_returned,
        "direction_is_declared": result.direction_is_declared,
        "direction_is_inferred": not result.direction_is_declared,
        "is_semantically_directed": result.is_semantically_directed,
        "declared_hierarchical": result.declared_hierarchical,
        "slice_s": result.slice_s,
        "spent_s": result.spent_s,
        "process_spent_s": result.process_spent_s,
        "remaining_entry_s": result.remaining_entry_s,
        "remaining_exit_s": result.remaining_exit_s,
        "measured_step_s": (None if result.cost_plan is None else result.cost_plan.measured_step_s),
        "shadow_measured_step_s": (
            None if result.cost_plan is None else result.cost_plan.shadow_step_s
        ),
        "shadow_measured_warmup_s": (
            None if result.cost_plan is None else result.cost_plan.shadow_warmup_s
        ),
        "warmup_s": None if result.cost_plan is None else result.cost_plan.warmup_s,
        "referee_s": None if result.cost_plan is None else result.cost_plan.referee_s,
        "budget_usable_s": (None if result.cost_plan is None else result.cost_plan.budget_usable_s),
        "predicted_s": None if result.cost_plan is None else result.cost_plan.predicted_s,
        "plan_seeds": None if result.cost_plan is None else result.cost_plan.seeds,
        "plan_steps": None if result.cost_plan is None else result.cost_plan.steps,
        "plan_checkpoints": None if result.cost_plan is None else result.cost_plan.checkpoints,
        "phase_timings_s": [
            {
                "seed": timing.seed,
                "mode": timing.mode,
                "pass_id": timing.pass_id,
                "route_s": timing.route_s,
                "optimize_s": timing.optimize_s,
                "viability_s": timing.viability_s,
                "score_s": timing.score_s,
            }
            for timing in result.phase_timings_s
        ],
        "viability_counts": result.viability_counts,
        "viability_drop_counts": result.viability_drop_counts,
        "winner_name": result.winner_name,
        "incumbent_score_pair": pair_payload(result.incumbent_score_pair),
        "incumbent_axes": axes_payload(result.incumbent_axes),
        "winner_score_pair": pair_payload(result.winner_score_pair),
        "accepted": [candidate.name for candidate in result.accepted],
        "legacy_tallied_sole_failure_checkpoint_count": sum(
            1 for checkpoint in result.checkpoints if checkpoint.legacy_tallied_sole_failure
        ),
        "rejected": [
            {
                "seed": checkpoint.seed,
                "mode": checkpoint.mode,
                "pass_id": checkpoint.pass_id,
                "step": checkpoint.step,
                "reason": checkpoint.reason,
                "legacy_tallied_sole_failure": checkpoint.legacy_tallied_sole_failure,
            }
            for checkpoint in result.rejected
        ],
        "checkpoints": [
            {
                "seed": checkpoint.seed,
                "mode": checkpoint.mode,
                "pass_id": checkpoint.pass_id,
                "step": checkpoint.step,
                "surrogate_delta": checkpoint.surrogate_delta,
                "directed_honest_delta": checkpoint.honest_delta,
                "undirected_honest_delta": checkpoint.undirected_honest_delta,
                "honest_score_pair": pair_payload(checkpoint.honest_score_pair),
                "accepted": checkpoint.accepted,
                "reason": checkpoint.reason,
                "legacy_tallied_sole_failure": checkpoint.legacy_tallied_sole_failure,
                "pass_spend_s": checkpoint.pass_spend_s,
            }
            for checkpoint in result.checkpoints
        ],
    }
    payload.update(
        _runtime_telemetry_payload(
            config=config,
            wall_s=result.spent_s,
            process_s=result.process_spent_s,
            use_deterministic_costs=_use_deterministic_w5_costs(config, result.node_count),
        )
    )
    if config is not None:
        existing = list(getattr(config, "_dagua_native_w5_telemetry", []))
        existing.append(payload)
        setattr(config, "_dagua_native_w5_telemetry", existing)
    telemetry_path = os.environ.get("DAGUA_W5_TELEMETRY_PATH")
    if telemetry_path:
        with open(telemetry_path, "a", encoding="utf-8") as handle:
            if config is not None:
                for record in getattr(config, "_dagua_native_cluster_tightening_telemetry", []):
                    handle.write(json.dumps({"stage": "cluster_tightening", **record}) + "\n")
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
    print("native_w5_finisher " + json.dumps(payload, sort_keys=True), flush=True)
    _LOGGER.info("Native W5 finisher telemetry %s", json.dumps(payload, sort_keys=True))


__all__ = [
    "W5Candidate",
    "W5Checkpoint",
    "W5ContinuousFacetPolishCandidate",
    "W5ContinuousFacetPolishResult",
    "W5FinisherResult",
    "W5GlobalScaleSweepCandidate",
    "W5GlobalScaleSweepResult",
    "W5HonestAxes",
    "W5PhaseTiming",
    "W5ScorePair",
    "W5Seed",
    "W5SMACOFStressCandidate",
    "W5SMACOFStressResult",
    "W5SmallNAnnealCandidate",
    "W5SmallNAnnealResult",
    "ClusterTighteningCandidate",
    "build_cluster_tightening_candidates",
    "candidate_introduces_champion_ineligible_flag",
    "DEGENERACY_CHAMPION_INELIGIBLE_FLAGS",
    "is_worker_timeout_like_exception",
    "log_w5_telemetry",
    "make_w5_skip_result",
    "run_w5_finisher",
    "run_w5_terminal_continuous_facet_polish",
    "run_w5_terminal_global_scale_sweep",
    "run_w5_terminal_smacof_stress_polish",
    "run_w5_terminal_small_n_anneal",
    "w5_honest_axes_from_metrics",
    "w5_dominates",
    "w5_legacy_tallied_sole_failure",
    "w5_score_pair_from_v3_result",
]
