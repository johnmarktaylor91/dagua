"""Undirected-portfolio native route (r80-S4).

Semantically-undirected graphs (social/community/SBM/small-world/mesh/
scale-free families) are where the layered native default loses most of its
benchmark comparisons to external force engines. This route runs a small
candidate CONTEST instead of betting on one pipeline:

- Candidate A (incumbent): whatever ``_choose_native_pipeline_baseline``
  would have selected if this route did not exist, run through the normal
  native path (including its own polish battery). This guarantees the route
  can never do worse than the pre-portfolio router wherever selection is
  honest.
- Candidate B: dagua's own bit-faithful sfdp reimplementation on the same
  problem tensors, finished with size-aware overlap projection.
- Candidate C: dagua's own bit-faithful neato reimplementation + projection,
  gated by the quality knob (see ``_neato_in_contest``).
- Candidate D (r80-S9, cluster-aware graphs only): the recursive
  ``ClusterAwareDriver`` running an sfdp inner pipeline, so clustered-
  undirected graphs get a candidate that structurally places cluster
  hierarchy levels instead of relying solely on the composite's cluster-
  separation term (see ``_cluster_aware_sfdp_candidate``).
- Candidate E (r80-S9, weighted graphs only): the native-stress core with
  Dijkstra/pivot target distances built from similarity-transformed weights
  (``weight_transform="inverse"``) instead of the default distance
  semantics, for community/social weighted families where a heavy edge
  means "close" (see ``_weighted_similarity_candidate``).
- Candidate F (r81-P1.5): the native-stress core with target distances scaled
  into the point units used by node boxes (see ``_stress_points_candidate``).

All candidates are scored with the SAME honest composite the benchmark
harness uses for undirected rows (``metrics.full`` + ``composite_auto``
with ``is_semantically_directed=False``); argmax wins, ties go to the
incumbent. Challenger candidates additionally pass a degeneracy guard
(collapsed layouts with near-zero edge lengths or a bounding box smaller
than the nodes it must contain are rejected before the contest) so a
pathological challenger score can never launder a broken layout past the
incumbent.

No external layout binaries are invoked anywhere in this module -- the
sfdp/neato pipelines are the fidelity-campaign reimplementations
(``dagua/layout/ops/pipelines/sfdp.py`` / ``neato.py``), pure PyTorch.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Dict, Optional, Tuple

import torch

from dagua.config import LayoutConfig
from dagua.layout.ops.base import Op, Pipeline
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op
from dagua.layout.projection import project_overlaps

if TYPE_CHECKING:  # pragma: no cover - typing only
    from dagua.layout.aesthetics import AestheticProfile

# Above this size the contest is skipped and the incumbent runs alone.
# Documented cap: the Stage-1 probe only produced candidate data up to 500
# nodes (see .project-context/research/r79_native/P8_PORTFOLIO_PROBE.md);
# probe data for larger graphs would be needed before raising this.
MAX_CONTEST_NODES = 1500

# Candidate C (neato) participates when the public quality knob resolves to
# at least this value ("high" alias = 0.75)...
NEATO_QUALITY_THRESHOLD = 0.75
# ...OR at balanced quality throughout the measured contest range. The
# iteration schedule below bounds larger SMACOF solves instead of excluding
# the graph families where neato is the reference winner.
NEATO_BALANCED_NODE_CAP = MAX_CONTEST_NODES

# Candidate refinement schedule. The faithful 500-step SFDP solve costs
# 9-20s through 150 nodes on the r81 CPU probe. Above that knee, 150 steps
# keeps the measured 500-node solve near the 60s default envelope; explicit
# high quality retains the full reference-fidelity budget.
FULL_REFINEMENT_NODE_CAP = 150
FULL_REFINEMENT_STEPS = 500
BALANCED_LARGE_REFINEMENT_STEPS = 150
NEATO_FULL_ITERATIONS = 200
NEATO_MEDIUM_NODE_CAP = 250
NEATO_BALANCED_MEDIUM_ITERATIONS = 40
NEATO_BALANCED_LARGE_ITERATIONS = 4

_LOGGER = logging.getLogger(__name__)

# Degeneracy guard thresholds (see _candidate_is_degenerate).
DEGENERACY_MIN_EDGE_TO_DIAGONAL_RATIO = 0.5
DEGENERACY_MIN_BBOX_TO_NODE_AREA_RATIO = 0.5
# Reject challenger layouts that fling ISOLATED (degree-0) nodes far from the
# layout centroid. Scoped to isolated nodes only: the r80 gate sweep proved a
# global max/median radius test also rejects legitimately-dispersed structure
# (multi_component_80 -11.0, er_500 real win -> loss -4.9, scale_free_ba_120
# -1.9). Non-isolated spread is legitimate layout structure and is not judged.
# Threshold 8.0: the pathology class is ORDER-OF-MAGNITUDE fling, not
# peripheral placement. Measured on the r80 store: legitimate isolate
# placements reach 5.4x median at most (er_500 periphery 0.5-4.8x,
# multi_component_80 tiles 2.8-2.9x), while the pathological
# random_bipartite_60 fling starts at 15.1x (measured 15-21x). 8x sits in
# the measured separation gap with margin on both sides.
DEGENERACY_MAX_ISOLATED_SPREAD_RATIO = 8.0


@dataclass(frozen=True)
class UndirectedPortfolioRouteConfig:
    """Frozen op-config for the undirected portfolio contest.

    Parameters
    ----------
    layout_config : LayoutConfig, optional
        Prepared native layout configuration used to run the incumbent and
        resolve quality/time budgets. ``None`` falls back to defaults.
    """

    layout_config: Optional[LayoutConfig] = field(default=None)


def _candidate_is_degenerate(
    pos: torch.Tensor,
    node_sizes: Optional[torch.Tensor],
    edge_index: torch.Tensor,
) -> Tuple[bool, str]:
    """Return whether a challenger layout is geometrically collapsed.

    Three symptoms are checked, any one rejects the candidate BEFORE the
    composite contest (composite terms like edge-length uniformity can score
    a geometrically broken layout deceptively well):

    1. Mean edge length below ``DEGENERACY_MIN_EDGE_TO_DIAGONAL_RATIO`` times
       the mean node bounding-box diagonal -- edges shorter than half a node
       mean the drawing cannot visually separate its endpoints.
    2. Layout bounding-box area below
       ``DEGENERACY_MIN_BBOX_TO_NODE_AREA_RATIO`` times the summed node-box
       area -- the canvas is smaller than the nodes it must contain, so
       overlap is unavoidable.
    3. Any ISOLATED (degree-0) node sits further than
       ``DEGENERACY_MAX_ISOLATED_SPREAD_RATIO`` times the median centroid
       distance from the layout centroid -- edge-based composite terms are
       blind to edgeless nodes, so a flung isolate can make the metrics call
       an illegible corner blob a win (random_bipartite_60 pathology).
       Connected-node spread is NOT judged: multi-component tilings and
       ER-periphery layouts legitimately exceed a global max/median radius
       test (r80 gate sweep regressions).

    Parameters
    ----------
    pos : torch.Tensor
        Candidate positions with shape ``[N, 2]``.
    node_sizes : torch.Tensor, optional
        Node bounding boxes with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.

    Returns
    -------
    tuple[bool, str]
        ``(is_degenerate, reason)``. ``reason`` is empty when healthy.
    """
    n = int(pos.shape[0])
    if n <= 1 or node_sizes is None or node_sizes.numel() == 0:
        return False, ""
    sizes = node_sizes.to(dtype=pos.dtype)
    if sizes.ndim == 1:
        sizes = sizes.unsqueeze(1).expand(-1, 2)

    mean_diagonal = float(torch.linalg.vector_norm(sizes, dim=1).mean().item())
    if edge_index.numel() > 0 and mean_diagonal > 0.0:
        deltas = pos[edge_index[1]] - pos[edge_index[0]]
        mean_edge_length = float(torch.linalg.vector_norm(deltas, dim=1).mean().item())
        if mean_edge_length < DEGENERACY_MIN_EDGE_TO_DIAGONAL_RATIO * mean_diagonal:
            return True, (
                f"mean edge length {mean_edge_length:.2f} < "
                f"{DEGENERACY_MIN_EDGE_TO_DIAGONAL_RATIO} x mean node diagonal "
                f"{mean_diagonal:.2f}"
            )

    bbox_extent = pos.max(dim=0).values - pos.min(dim=0).values
    # Include node extents so a single-row layout is not falsely zero-area.
    bbox_area = float(
        ((bbox_extent[0] + sizes[:, 0].mean()) * (bbox_extent[1] + sizes[:, 1].mean())).item()
    )
    total_node_area = float((sizes[:, 0] * sizes[:, 1]).sum().item())
    if total_node_area > 0.0 and bbox_area < (
        DEGENERACY_MIN_BBOX_TO_NODE_AREA_RATIO * total_node_area
    ):
        return True, (
            f"bbox area {bbox_area:.1f} < {DEGENERACY_MIN_BBOX_TO_NODE_AREA_RATIO} x "
            f"total node-box area {total_node_area:.1f}"
        )

    # Isolated-node fling check. Judged for degree-0 nodes ONLY: a global
    # max/median test over all nodes also rejected legitimately-dispersed
    # candidates (multi-component tilings, ER periphery) in the r80 gate
    # sweep. Not applicable (ratio 0.0) when there are no isolates, every
    # node is isolated, or the median distance is zero (true collapse is
    # already covered by checks 1-2). Normally pre-empted by the
    # _repair_flung_isolates repair path; kept as a backstop should a
    # repaired (or unrepairable single-component) candidate still fling.
    spread_ratio = _max_isolated_spread_ratio(pos, edge_index)
    if spread_ratio > DEGENERACY_MAX_ISOLATED_SPREAD_RATIO:
        return True, (
            f"isolated-node centroid spread {spread_ratio:.1f}x median > "
            f"{DEGENERACY_MAX_ISOLATED_SPREAD_RATIO}x"
        )
    return False, ""


def _max_isolated_spread_ratio(pos: torch.Tensor, edge_index: torch.Tensor) -> float:
    """Return the worst isolated-node centroid-distance / median-distance ratio.

    The exact quantity the isolated-fling guard and the repair trigger
    evaluate. Returns ``0.0`` when the check does not apply: no isolated
    (degree-0) nodes, ALL nodes isolated (no connected core exists to be far
    from), or zero median distance (true collapse is covered by the other
    degeneracy checks).

    Parameters
    ----------
    pos : torch.Tensor
        Positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.

    Returns
    -------
    float
        Max isolated-node centroid distance divided by the median centroid
        distance over all nodes, or ``0.0`` when not applicable.
    """
    n = int(pos.shape[0])
    if n <= 1:
        return 0.0
    isolated_mask = torch.ones(n, dtype=torch.bool)
    if edge_index.numel() > 0:
        isolated_mask[edge_index.reshape(-1).to(dtype=torch.long)] = False
    if not bool(isolated_mask.any()) or bool(isolated_mask.all()):
        return 0.0
    centroid = pos.mean(dim=0, keepdim=True)
    centroid_distances = torch.linalg.vector_norm(pos - centroid, dim=1)
    median_distance = float(torch.median(centroid_distances).item())
    if median_distance <= 0.0:
        return 0.0
    return float(centroid_distances[isolated_mask].max().item()) / median_distance


def _repair_flung_isolates(
    pos: torch.Tensor,
    problem: LayoutProblem,
    node_sep: float,
) -> torch.Tensor:
    """Repair isolated-node fling by re-tiling components; no-op otherwise.

    r80 round 4: packing is a REPAIR, not a default. Unconditional challenger
    packing regressed er_500 (-4.9, honest win lost) and multi_component_80
    (-11.0) whose isolates sat at a legitimate 2.8-4.8x median -- rewriting
    healthy layouts let the composite mildly prefer the original moderate
    spread. A candidate keeps its raw layout byte-identical UNLESS the
    isolated-fling trigger fires (any degree-0 node beyond
    ``DEGENERACY_MAX_ISOLATED_SPREAD_RATIO`` x median centroid distance), in
    which case each weak component keeps its raw internal geometry and the
    components are re-tiled adjacent with the shared
    ``_tile_component_positions`` tiler. Repair-then-rescore: the contest
    referee sees the repaired version.

    Parameters
    ----------
    pos : torch.Tensor
        Raw challenger positions with shape ``[N, 2]``.
    problem : LayoutProblem
        Parent layout problem (edge_index / num_nodes are read).
    node_sep : float
        Node separation passed through to the shared component tiler.

    Returns
    -------
    torch.Tensor
        ``pos`` unchanged when the trigger does not fire, else the repaired
        full-layout positions with shape ``[N, 2]``.
    """
    if _max_isolated_spread_ratio(pos, problem.edge_index) <= (
        DEGENERACY_MAX_ISOLATED_SPREAD_RATIO
    ):
        return pos

    from dagua.layout.ops.coordinate import _weak_components
    from dagua.layout.ops.pipelines._native_shared import _tile_component_positions
    from dagua.layout.ops.postprocess import AspectRatioFit, AspectRatioFitConfig

    components = _weak_components(
        problem.edge_index.detach().to(device="cpu", dtype=torch.long),
        int(problem.num_nodes),
    )
    if len(components) <= 1:
        return pos
    component_results: list[tuple[torch.Tensor, torch.Tensor]] = []
    for component_nodes_list in components:
        component_nodes = torch.tensor(
            component_nodes_list,
            dtype=torch.long,
            device=pos.device,
        )
        component_results.append((component_nodes, pos[component_nodes]))
    tiled_positions = _tile_component_positions(component_results, node_sep=node_sep)
    fit_state = AspectRatioFit(AspectRatioFitConfig()).apply(
        problem,
        SolveState(pos=tiled_positions),
        RuntimeContext(),
    )
    if fit_state.pos is None:
        raise RuntimeError("isolate-fling repair did not produce positions.")
    repaired = fit_state.pos.detach()
    # The guard judges isolate radius against the all-node median radius, so
    # make the repair target that exact geometry after component tiling.
    # Recompute because moving isolates also shifts the all-node centroid.
    isolated_mask = torch.ones(int(problem.num_nodes), dtype=torch.bool, device=repaired.device)
    if problem.edge_index.numel() > 0:
        isolated_mask[
            problem.edge_index.reshape(-1).to(device=repaired.device, dtype=torch.long)
        ] = False
    target_ratio = DEGENERACY_MAX_ISOLATED_SPREAD_RATIO * 0.95
    for _iteration in range(4):
        centroid = repaired.mean(dim=0, keepdim=True)
        distances = torch.linalg.vector_norm(repaired - centroid, dim=1)
        median_distance = float(torch.median(distances).item())
        if median_distance <= 0.0:
            break
        limit = target_ratio * median_distance
        far_mask = isolated_mask & (distances > limit)
        if not bool(far_mask.any()):
            break
        vectors = repaired[far_mask] - centroid
        repaired[far_mask] = centroid + vectors * (limit / distances[far_mask]).unsqueeze(1)
    return repaired


def _build_cluster_ids(problem: LayoutProblem) -> Optional[torch.Tensor]:
    """Reconstruct per-node cluster ids from problem cluster metadata.

    Mirrors ``DaguaGraph.cluster_ids`` (deepest assignment wins, indices
    follow sorted cluster-name order) so the layout-time composite sees the
    same cluster-separation term the benchmark scorer sees. Nested-dict
    cluster values fall back to leaf collection.

    Parameters
    ----------
    problem : LayoutProblem
        Problem carrying optional ``clusters`` metadata.

    Returns
    -------
    torch.Tensor | None
        Cluster ids with shape ``[N]`` or ``None`` when no clusters exist.
    """
    clusters = problem.clusters
    if not clusters or problem.num_nodes == 0:
        return None
    try:
        from dagua.utils import collect_cluster_leaves

        parents = problem.cluster_parents or {}

        def _depth(name: str) -> int:
            depth = 0
            current: Optional[str] = name
            seen = set()
            while current is not None and current not in seen:
                seen.add(current)
                current = parents.get(current)
                depth += 1
            return depth

        ids = torch.full((problem.num_nodes,), -1, dtype=torch.long)
        node_depth = [-1] * problem.num_nodes
        cluster_name_list = sorted(clusters.keys())
        name_to_idx = {name: index for index, name in enumerate(cluster_name_list)}
        for name in cluster_name_list:
            members = clusters[name]
            if isinstance(members, dict):
                members = collect_cluster_leaves(members)
            depth = _depth(name)
            for node_idx in members:
                node_int = int(node_idx)
                if 0 <= node_int < problem.num_nodes and depth > node_depth[node_int]:
                    ids[node_int] = name_to_idx[name]
                    node_depth[node_int] = depth
        return ids
    except Exception:  # noqa: BLE001 -- scoring must not crash the solve
        return None


def _score_undirected_candidate(
    pos: torch.Tensor,
    problem: LayoutProblem,
    cluster_ids: Optional[torch.Tensor],
    aesthetic_profile: Optional["AestheticProfile"] = None,
) -> float:
    """Score one candidate with the benchmark's honest undirected composite.

    Uses ``metrics.full`` (tier the benchmark uses for graphs under its full
    cutoff -- the contest node cap keeps us in that regime) and
    ``composite_auto(..., is_semantically_directed=False)``. ``full`` is
    self-deterministic for fixed positions (sampled crossing rate seeds its
    own generator), so selection is reproducible.

    r80-S8: when ``aesthetic_profile`` is ``None`` (the default, unset knob)
    this calls ``composite_auto`` exactly as before -- no wrapper, no extra
    float ops, bit-identical to pre-knob behavior. When a profile is
    resolved, every candidate in the contest is scored with
    ``dagua.layout.aesthetics.reweighted_composite`` and that SAME profile
    object (see ``layout_native_undirected_portfolio``), which is required
    for contest fairness.

    Parameters
    ----------
    pos : torch.Tensor
        Candidate positions with shape ``[N, 2]``.
    problem : LayoutProblem
        Problem carrying topology and node sizes.
    cluster_ids : torch.Tensor, optional
        Optional per-node cluster ids for the cluster-separation term.
    aesthetic_profile : AestheticProfile, optional
        Resolved aesthetic-priority profile shared by every candidate in
        the current contest. ``None`` preserves the exact pre-knob scoring
        path.

    Returns
    -------
    float
        Higher-is-better undirected composite score.
    """
    from dagua.metrics import composite_auto, full

    metrics = full(
        pos.detach().to(device="cpu", dtype=torch.float32),
        problem.edge_index.detach().to(device="cpu"),
        node_sizes=(
            None
            if problem.node_sizes is None
            else problem.node_sizes.detach().to(device="cpu", dtype=torch.float32)
        ),
        cluster_ids=cluster_ids,
        direction=problem.direction,
    )
    numeric = {
        key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))
    }
    if aesthetic_profile is None:
        return float(composite_auto(numeric, is_semantically_directed=False))

    from dagua.layout.aesthetics import reweighted_composite

    return reweighted_composite(numeric, is_directed=False, profile=aesthetic_profile)


# Convergent-cleanup pass budget for challenger candidates. The convergent
# exact projector early-exits at zero overlaps or on measured stagnation,
# so this ceiling is only consumed on hard overlap fields; the contest cap
# (MAX_CONTEST_NODES) bounds the per-pass O(N^2) cost.
CHALLENGER_PROJECTION_ITERATIONS = 200
PRISM_ZERO_MAX_ITERATIONS = 4
PRISM_SCALE_MARGIN = 1.001


def _candidate_refinement_steps(config: Optional[LayoutConfig], num_nodes: int) -> int:
    """Return the quality-scaled force refinement budget.

    Parameters
    ----------
    config : LayoutConfig, optional
        Prepared public layout configuration.
    num_nodes : int
        Number of nodes in the contest problem.

    Returns
    -------
    int
        SFDP refinement steps for the candidate solve.
    """
    if (
        num_nodes <= FULL_REFINEMENT_NODE_CAP
        or _resolved_quality(config) >= NEATO_QUALITY_THRESHOLD
    ):
        return FULL_REFINEMENT_STEPS
    return BALANCED_LARGE_REFINEMENT_STEPS


def _neato_iterations(config: Optional[LayoutConfig], num_nodes: int) -> int:
    """Return the quality-scaled neato SMACOF iteration budget.

    Parameters
    ----------
    config : LayoutConfig, optional
        Prepared public layout configuration.
    num_nodes : int
        Number of nodes in the contest problem.

    Returns
    -------
    int
        Maximum SMACOF iterations for the candidate solve.
    """
    if _resolved_quality(config) >= NEATO_QUALITY_THRESHOLD:
        return NEATO_FULL_ITERATIONS
    if num_nodes <= FULL_REFINEMENT_NODE_CAP:
        return NEATO_FULL_ITERATIONS
    if num_nodes <= NEATO_MEDIUM_NODE_CAP:
        return NEATO_BALANCED_MEDIUM_ITERATIONS
    return NEATO_BALANCED_LARGE_ITERATIONS


def _overlap_pairs(pos: torch.Tensor, node_sizes: torch.Tensor) -> torch.Tensor:
    """Return upper-triangle pairs whose axis-aligned node boxes overlap.

    Parameters
    ----------
    pos : torch.Tensor
        Positions with shape ``[N, 2]``.
    node_sizes : torch.Tensor
        Node sizes with shape ``[N, 2]``.

    Returns
    -------
    torch.Tensor
        Overlapping node-index pairs with shape ``[K, 2]``.
    """
    deltas = torch.abs(pos.unsqueeze(1) - pos.unsqueeze(0))
    required = (node_sizes.unsqueeze(1) + node_sizes.unsqueeze(0)) * 0.5
    overlaps = (deltas[..., 0] < required[..., 0]) & (deltas[..., 1] < required[..., 1])
    return torch.nonzero(torch.triu(overlaps, diagonal=1), as_tuple=False)


def _scale_past_residual_overlaps(
    pos: torch.Tensor,
    node_sizes: torch.Tensor,
    overlap_pairs: torch.Tensor,
) -> torch.Tensor:
    """Uniformly scale a layout just past its remaining overlap pairs.

    Parameters
    ----------
    pos : torch.Tensor
        Positions with shape ``[N, 2]``.
    node_sizes : torch.Tensor
        Node sizes with shape ``[N, 2]``.
    overlap_pairs : torch.Tensor
        Residual overlapping pairs with shape ``[K, 2]``.

    Returns
    -------
    torch.Tensor
        Topology-preserving uniformly scaled positions.
    """
    source = overlap_pairs[:, 0]
    target = overlap_pairs[:, 1]
    deltas = torch.abs(pos[target] - pos[source])
    required = (node_sizes[target] + node_sizes[source]) * 0.5
    ratios = required / torch.clamp(deltas, min=torch.finfo(pos.dtype).eps)
    # A pair stops overlapping as soon as either axis clears. Only residual
    # pairs determine the smallest global scale bump, preserving all angles.
    pair_scales = torch.min(ratios, dim=1).values
    scale = float(torch.max(pair_scales).item()) * PRISM_SCALE_MARGIN
    centered = pos - pos.mean(dim=0, keepdim=True)
    return centered * scale + pos.mean(dim=0, keepdim=True)


def _project_candidate_prism(pos: torch.Tensor, problem: LayoutProblem) -> torch.Tensor:
    """Apply native PRISM cleanup and iterate residual overlaps to zero.

    Parameters
    ----------
    pos : torch.Tensor
        Candidate positions in points with shape ``[N, 2]``.
    problem : LayoutProblem
        Problem carrying topology and node sizes.

    Returns
    -------
    torch.Tensor
        PRISM-cleaned positions in points with shape ``[N, 2]``.
    """
    if problem.node_sizes is None or problem.node_sizes.numel() == 0:
        return pos
    from dagua.layout.ops.pipelines.fmmm import _graphviz_fdp_prism_overlap

    points_per_inch = 72.0
    projected = _graphviz_fdp_prism_overlap(
        positions=pos.detach().to(dtype=torch.float64) / points_per_inch,
        edge_index=problem.edge_index.detach().to(device="cpu", dtype=torch.long),
        node_sizes=problem.node_sizes.detach().to(device="cpu", dtype=torch.float64),
    )
    projected = (projected * points_per_inch).to(device=pos.device, dtype=torch.float32)
    sizes = problem.node_sizes.to(device=projected.device, dtype=projected.dtype)
    for _iteration in range(PRISM_ZERO_MAX_ITERATIONS):
        pairs = _overlap_pairs(projected, sizes)
        if pairs.numel() == 0:
            break
        projected = _scale_past_residual_overlaps(projected, sizes, pairs)
    return projected


def _project_candidate(
    pos: torch.Tensor,
    problem: LayoutProblem,
    convergent: bool = False,
) -> torch.Tensor:
    """Apply size-aware overlap projection to one challenger candidate.

    Two cleanup variants exist and NEITHER dominates (r80-S2b petersen_10
    bisect, P7_PROJECTOR_EVIDENCE.md):

    - ``convergent=False``: the legacy projector call the S4 portfolio
      shipped with (default padding/iterations). Its last-write-wins pushes
      stall on dense overlap fields, but its trajectory produced the
      trunk's flagship wins (petersen_10 79.0, weighted_karate_34 69.5,
      weighted_clusters_3x10 68.1 -- all legacy-cleaned neato candidates).
    - ``convergent=True``: the accumulate+damp+deadlock-re-lay projector
      with a generous early-exit ceiling. Provably reaches zero overlaps
      on dense cliques the legacy path stalls on (P3B2 forensics) and
      produced the S2b sweep gains (planar_60 +19.9, regular_4_40 +15.4,
      weighted_clusters sfdp +21.4 over legacy).

    The contest therefore scores BOTH variants as separate candidates --
    never replacing one with the other -- and lets the honest-composite
    referee choose (the S2b regression came from replacing the legacy
    variant instead of adding the convergent one alongside it).

    Parameters
    ----------
    pos : torch.Tensor
        Candidate positions with shape ``[N, 2]``.
    problem : LayoutProblem
        Problem carrying node sizes.
    convergent : bool, default=False
        Select the convergent cleanup variant.

    Returns
    -------
    torch.Tensor
        Overlap-projected positions (new tensor).
    """
    if problem.node_sizes is None or problem.node_sizes.numel() == 0:
        return pos
    projected = pos.detach().clone().to(dtype=torch.float32)
    node_sizes = problem.node_sizes.to(device=projected.device, dtype=projected.dtype)
    if convergent:
        project_overlaps(
            projected,
            node_sizes,
            iterations=CHALLENGER_PROJECTION_ITERATIONS,
            convergent=True,
        )
    else:
        project_overlaps(projected, node_sizes)
    return projected


def _resolved_quality(config: Optional[LayoutConfig]) -> float:
    """Return the normalized [0, 1] quality value from a prepared config.

    Parameters
    ----------
    config : LayoutConfig, optional
        Prepared layout configuration.

    Returns
    -------
    float
        Normalized quality; 0.5 (balanced) when unavailable.
    """
    if config is None:
        return 0.5
    try:
        return float(getattr(config, "quality", 0.5))
    except (TypeError, ValueError):
        return 0.5


def _neato_in_contest(config: Optional[LayoutConfig], num_nodes: int) -> bool:
    """Return whether candidate C (neato + projection) joins the contest.

    Two admission paths:

    1. Quality >= high (0.75): neato always joins (up to the contest cap).
    2. Balanced/lower quality with ``num_nodes <= NEATO_BALANCED_NODE_CAP``:
       the Stage-1 probe (P8_PORTFOLIO_PROBE.md) shows every balanced-quality
       contest neato ever wins sits at n <= 80, where its SMACOF loop
       epsilon-exits in <= ~8s; on larger graphs it costs 40-150s and never
       beat the sfdp/incumbent winner in any probe row. The cap keeps the
       neato wins (karate, lattices, grids, petersen, multi-component)
       inside the default wall-time envelope and leaves the slow never-wins
       region to the explicit quality knob.

    Parameters
    ----------
    config : LayoutConfig, optional
        Prepared layout configuration.
    num_nodes : int
        Number of nodes in the current problem.

    Returns
    -------
    bool
        ``True`` when candidate C should run.
    """
    if _resolved_quality(config) >= NEATO_QUALITY_THRESHOLD:
        return True
    return num_nodes <= NEATO_BALANCED_NODE_CAP


# r80-S9 Deliverable 2: weighted-similarity Dijkstra-target transform. A
# 3-graph mini-probe (r79_weighted_small_world_120, r79_weighted_community_
# 4x18, real_lesmis_77; see P12_SQUEEZE.md) compared "inverse" (1/w) against
# an ad hoc 1/sqrt(w) transform on the raw and legacy-projected candidate
# tiers (the convergent-projector tier washed out the difference -- 200
# damped passes converge to the same overlap-free arrangement regardless of
# the small-scale stress differences between transforms). "inverse" won 2 of
# 3 graphs and never lost by more than 1.3 points on the graph it lost,
# while both transforms beat the untransformed (today's default) distance
# semantics on every graph. "inverse" is also the transform preprocess.py
# already implements (BuildAdjacencyConfig.weight_transform), so no new
# transform code is needed.
WEIGHTED_SIMILARITY_TRANSFORM = "inverse"


def _cluster_aware_sfdp_candidate(
    problem: LayoutProblem,
    config: LayoutConfig,
    ctx: RuntimeContext,
) -> Optional[torch.Tensor]:
    """Run the recursive cluster-aware driver with an sfdp inner pipeline.

    r80-S9 Deliverable 1, candidate B: clustered-undirected graphs (e.g. the
    ``r79_undirected_sbm_*`` community corpus) reach this contest today
    (the S4-era diagnosis that "the cluster driver preempts routing" no
    longer applies for the ``dagua_native``/default algorithm -- verified
    empirically, see P12_SQUEEZE.md), but candidate A (the incumbent) comes
    from the FLAT native path with cluster-separation LOSS terms only, and
    candidate B (flat sfdp, added below via ``_add_challenger``) also never
    places clusters structurally -- both rely entirely on the scoring
    composite's cluster term to reward containment after the fact. This
    candidate instead PLACES each cluster hierarchy level with dagua's sfdp
    reimplementation via the existing recursive ``ClusterAwareDriver``
    (``dagua/layout/ops/cluster_driver.py`` -- the same machinery
    ``dagua.layout.engine._layout_cluster_aware_pipeline`` uses for the
    algorithms it natively supports; ``"dagua_native"``/``None`` is not one
    of them, which is why clustered-undirected graphs never got this
    candidate before). Returns ``None`` when there are no clusters on this
    problem or the driver cannot be built.

    Parameters
    ----------
    problem : LayoutProblem
        Prepared layout problem, expected to carry cluster metadata.
    config : LayoutConfig
        Prepared native configuration.
    ctx : RuntimeContext
        Shared execution context.

    Returns
    -------
    torch.Tensor or None
        Candidate positions, or ``None`` when clusters are absent or the
        driver could not run.
    """
    if not problem.clusters:
        return None
    from dagua.layout.engine import _build_cluster_inner_pipeline
    from dagua.layout.ops.cluster_driver import ClusterAwareDriver

    inner_pipeline = _build_cluster_inner_pipeline("sfdp", config)
    if inner_pipeline is None:
        return None
    driver = ClusterAwareDriver(
        inner_pipeline=inner_pipeline.ops,
        # No DaguaGraph is available inside this headless contest to merge
        # per-graph cluster_style.padding overrides the way
        # engine._effective_cluster_side_padding does for the top-level
        # cluster driver -- this candidate uses the raw config padding
        # knobs. Documented limitation (P12_SQUEEZE.md): only affects this
        # one candidate's geometry among several scored in the contest.
        side_padding_pt=float(getattr(config, "cluster_side_padding_pt", 8.0)),
        label_band_pt=float(getattr(config, "cluster_label_band_pt", 26.0)),
        external_clearance_pt=float(getattr(config, "cluster_external_clearance_pt", 10.0)),
        cluster_compactness_weight=float(getattr(config, "w_cluster", 1.0)),
    )
    driver_state = driver.apply(problem, SolveState(), ctx)
    return driver_state.pos


def _weighted_similarity_candidate(
    problem: LayoutProblem,
    seed: int,
) -> Optional[torch.Tensor]:
    """Run the native-stress core with weights treated as similarities.

    r80-S9 Deliverable 2: for declared-undirected weighted graphs, the
    default Dijkstra/pivot target-distance costs use edge weights AS
    distances (``weight_transform="none"``) -- but for community/social
    weighted families (this contest only ever runs for declared-undirected
    graphs, exactly the family P3B2_STRESS_FORENSICS.md Ranked Fix 4 is
    about) a heavier weight usually means a STRONGER/closer relationship,
    not a longer one. This candidate reruns the native-stress core with
    ``weight_transform="inverse"`` (``1 / w``, see
    ``WEIGHTED_SIMILARITY_TRANSFORM`` for the mini-probe that picked it)
    so heavy edges pull their endpoints together. Purely additive: it is
    ONE MORE contest candidate, never a change to default weight handling
    (``NativeStressConfig.weight_transform`` defaults to ``"none"``
    everywhere else). Returns ``None`` when the problem carries no edge
    weights.

    Parameters
    ----------
    problem : LayoutProblem
        Prepared layout problem.
    seed : int
        Deterministic seed shared with the rest of the contest.

    Returns
    -------
    torch.Tensor or None
        Candidate positions, or ``None`` when there are no edge weights.
    """
    if problem.edge_weights is None:
        return None
    from dagua.layout.ops.pipelines.native_stress import (
        NativeStressConfig,
        layout_native_stress_pipeline,
    )

    return layout_native_stress_pipeline(
        edge_index=problem.edge_index,
        num_nodes=int(problem.num_nodes),
        node_sizes=problem.node_sizes,
        edge_weights=problem.edge_weights,
        seed=seed,
        config=NativeStressConfig(weight_transform=WEIGHTED_SIMILARITY_TRANSFORM, seed=seed),
    )


def _stress_points_candidate(problem: LayoutProblem, seed: int) -> torch.Tensor:
    """Run native stress with target distances expressed in points.

    Parameters
    ----------
    problem : LayoutProblem
        Prepared undirected layout problem.
    seed : int
        Deterministic seed shared with the rest of the contest.

    Returns
    -------
    torch.Tensor
        Candidate positions with shape ``[N, 2]``.
    """
    from dagua.layout.ops.pipelines.native_stress import (
        NativeStressConfig,
        layout_native_stress_pipeline,
    )

    return layout_native_stress_pipeline(
        edge_index=problem.edge_index,
        num_nodes=int(problem.num_nodes),
        node_sizes=problem.node_sizes,
        edge_weights=problem.edge_weights,
        seed=seed,
        config=NativeStressConfig(target_unit="points", seed=seed),
    )


def layout_native_undirected_portfolio(
    problem: LayoutProblem,
    state: SolveState,
    ctx: RuntimeContext,
    config: LayoutConfig,
) -> torch.Tensor:
    """Run the undirected portfolio contest for one prepared problem.

    Parameters
    ----------
    problem : LayoutProblem
        Prepared layout problem (whole graph or one weak component).
    state : SolveState
        Incoming solve state; candidate runs receive cloned copies.
    ctx : RuntimeContext
        Shared execution context.
    config : LayoutConfig
        Prepared native configuration with ``_dagua_native_*`` metadata.

    Returns
    -------
    torch.Tensor
        Selected positions with shape ``[N, 2]``.
    """
    # Late import avoids a circular import with dagua_native (which imports
    # this module lazily at its two dispatch points).
    from dagua.layout.ops.pipelines.dagua_native import _run_native_problem

    def _run_incumbent() -> torch.Tensor:
        # Candidate A must be EXACTLY today's default output. Re-enter the
        # router with the portfolio branch suppressed via a private attr --
        # NOT via force_pipeline, because several polish stages (edge
        # equalize best-of-polish, component-tiling polish) are gated on
        # force_pipeline being None and would silently weaken the incumbent.
        incumbent_config = copy.copy(config)
        setattr(incumbent_config, "_dagua_native_suppress_portfolio", True)
        incumbent_state = SolveState(pos=None if state.pos is None else state.pos.detach().clone())
        return _run_native_problem(problem, incumbent_state, ctx, incumbent_config)

    incumbent_pos = _run_incumbent()

    # Contest predicate: documented caps. Above MAX_CONTEST_NODES the probe
    # has no candidate data; with an explicit wall-clock budget the extra
    # candidate solves would silently blow it.
    n = int(problem.num_nodes)
    if n > MAX_CONTEST_NODES or getattr(config, "time_budget_s", None) is not None:
        return incumbent_pos

    # r80-S8: the aesthetic profile was resolved ONCE in
    # prepare_pipeline_config and stashed on this (already-prepared) config.
    # Reusing that exact object -- rather than re-resolving here -- is what
    # guarantees every candidate in this contest is scored under the
    # identical profile (fairness). `None` when the knob is unset.
    aesthetic_profile: Optional["AestheticProfile"] = getattr(
        config, "_dagua_native_aesthetic_profile", None
    )

    cluster_ids = _build_cluster_ids(problem)
    scores: Dict[str, float] = {}
    positions: Dict[str, torch.Tensor] = {}

    # Candidate A: the incumbent is ALWAYS eligible (degeneracy guard applies
    # to challengers only).
    positions["incumbent"] = incumbent_pos
    scores["incumbent"] = _score_undirected_candidate(
        incumbent_pos, problem, cluster_ids, aesthetic_profile
    )

    # P3 geometry challengers derive from the exact incumbent and bypass
    # projection so their measured transforms reach the honest referee intact.
    from dagua.layout.ops.pipelines.dagua_native import (
        _collinear_dodge,
        _unshear_bimodal_edges,
    )

    for name, candidate in (
        (
            "collinear_dodge_0.10",
            _collinear_dodge(incumbent_pos, problem.edge_index, delta=0.10),
        ),
        (
            "collinear_dodge_0.15",
            _collinear_dodge(incumbent_pos, problem.edge_index, delta=0.15),
        ),
        ("unshear", _unshear_bimodal_edges(incumbent_pos, problem.edge_index)),
    ):
        if candidate is None or not bool(torch.isfinite(candidate).all().item()):
            continue
        candidate_score = _score_undirected_candidate(
            candidate, problem, cluster_ids, aesthetic_profile
        )
        if candidate_score > scores["incumbent"] + 0.1:
            positions[name] = candidate
            scores[name] = candidate_score

    seed = int(problem.seed) if problem.seed is not None else 42
    challenger_node_sep = float(getattr(config, "_dagua_native_node_sep", config.node_sep))

    def _add_challenger(name: str, raw_pos: torch.Tensor) -> None:
        # Repair, not default (r80 round 4): the candidate keeps its raw
        # layout byte-identical unless the isolated-fling trigger fires, in
        # which case the flung singletons are re-tiled adjacent to the core
        # before projection and the referee scores the repaired version.
        # Applied at this shared entry so every challenger family (sfdp,
        # neato, cluster_sfdp, weighted_similarity) gets the same backstop.
        raw_pos = _repair_flung_isolates(raw_pos, problem, challenger_node_sep)
        # Both cleanup variants enter the contest as separate candidates --
        # never replace one with the other (r80-S2b bisect: replacing the
        # legacy variant with the convergent one silently removed the
        # trunk's petersen/karate/wclusters flagship candidates from the
        # pool; neither variant dominates). The degeneracy guard applies to
        # each variant independently.
        for suffix, convergent in (("", False), ("_convergent", True)):
            projected = _project_candidate(raw_pos, problem, convergent=convergent)
            degenerate, reason = _candidate_is_degenerate(
                projected,
                problem.node_sizes,
                problem.edge_index,
            )
            if degenerate:
                _LOGGER.info("Rejected undirected candidate %s%s: %s", name, suffix, reason)
                continue
            positions[name + suffix] = projected
            scores[name + suffix] = _score_undirected_candidate(
                projected, problem, cluster_ids, aesthetic_profile
            )
        prism_projected = _project_candidate_prism(raw_pos, problem)
        degenerate, reason = _candidate_is_degenerate(
            prism_projected,
            problem.node_sizes,
            problem.edge_index,
        )
        if degenerate:
            _LOGGER.info("Rejected undirected candidate %s_prism: %s", name, reason)
        else:
            positions[name + "_prism"] = prism_projected
            scores[name + "_prism"] = _score_undirected_candidate(
                prism_projected, problem, cluster_ids, aesthetic_profile
            )

    # Candidate B: our graphviz-fidelity sfdp reimplementation. The contest
    # owns a quality-scaled nonzero budget because LayoutConfig.steps=0 means
    # automatic at the public API, not zero refinement for this challenger.
    try:
        from dagua.layout.ops.pipelines.sfdp import layout_sfdp_pipeline

        # Raw full-problem solve (round 4): per-component packed solving was
        # tried and regressed healthy multi-component candidates; any
        # isolate fling in this raw output is repaired conditionally inside
        # _add_challenger.
        sfdp_pos = layout_sfdp_pipeline(
            edge_index=problem.edge_index,
            num_nodes=n,
            node_sizes=problem.node_sizes,
            steps=_candidate_refinement_steps(config, n),
            seed=seed,
            edge_weights=problem.edge_weights,
            fidelity_mode="graphviz",
        )
        _add_challenger("sfdp", sfdp_pos)
        if problem.edge_weights is not None:
            sfdp_unweighted_pos = layout_sfdp_pipeline(
                edge_index=problem.edge_index,
                num_nodes=n,
                node_sizes=problem.node_sizes,
                steps=_candidate_refinement_steps(config, n),
                seed=seed,
                edge_weights=None,
                fidelity_mode="graphviz",
            )
            _add_challenger("sfdp_unweighted", sfdp_unweighted_pos)
    except Exception:  # noqa: BLE001 -- a failed challenger never sinks the solve
        _LOGGER.warning("SFDP undirected challenger failed", exc_info=True)

    # Candidate C: our neato reimplementation + projection, quality-gated.
    if _neato_in_contest(config, n):
        try:
            from dagua.layout.ops.pipelines.neato import layout_neato_pipeline

            # Raw full-problem solve (round 4); isolate fling repaired
            # conditionally inside _add_challenger.
            neato_pos = layout_neato_pipeline(
                edge_index=problem.edge_index,
                num_nodes=n,
                node_sizes=problem.node_sizes,
                seed=seed,
                edge_weights=problem.edge_weights,
                maxiter=_neato_iterations(config, n),
                fidelity_mode="graphviz",
                overlap_removal=False,
            )
            _add_challenger("neato", neato_pos)
            if problem.edge_weights is not None:
                neato_unweighted_pos = layout_neato_pipeline(
                    edge_index=problem.edge_index,
                    num_nodes=n,
                    node_sizes=problem.node_sizes,
                    seed=seed,
                    edge_weights=None,
                    maxiter=_neato_iterations(config, n),
                    fidelity_mode="graphviz",
                    overlap_removal=False,
                )
                _add_challenger("neato_unweighted", neato_unweighted_pos)
        except Exception:  # noqa: BLE001
            _LOGGER.warning("neato undirected challenger failed", exc_info=True)

    # Candidate D (r80-S9 Deliverable 1): cluster-aware sfdp driver, only
    # for problems that actually carry cluster metadata. Adds a candidate
    # that structurally places cluster hierarchy levels instead of relying
    # on the composite's cluster-separation term alone (see
    # _cluster_aware_sfdp_candidate). Never replaces the incumbent or the
    # flat sfdp/neato challengers above.
    if problem.clusters:
        try:
            cluster_sfdp_pos = _cluster_aware_sfdp_candidate(problem, config, ctx)
        except Exception:  # noqa: BLE001 -- a failed challenger never sinks the solve
            _LOGGER.warning("cluster-SFDP undirected challenger failed", exc_info=True)
            cluster_sfdp_pos = None
        if cluster_sfdp_pos is not None:
            _add_challenger("cluster_sfdp", cluster_sfdp_pos)

    # Candidate E (r80-S9 Deliverable 2): weighted-similarity native-stress
    # core, only for problems that carry edge weights. Adds a candidate
    # whose Dijkstra/pivot target distances treat weights as similarities
    # (see _weighted_similarity_candidate). Never changes default weight
    # handling anywhere else.
    if problem.edge_weights is not None:
        try:
            weighted_pos = _weighted_similarity_candidate(problem, seed)
        except Exception:  # noqa: BLE001
            _LOGGER.warning("weighted-similarity undirected challenger failed", exc_info=True)
            weighted_pos = None
        if weighted_pos is not None:
            _add_challenger("weighted_similarity", weighted_pos)

    # Candidate F (r81-P1.5): point-unit native stress uses the existing
    # quality-scaled stress schedule. It is additive and contest-scored, so
    # graphs where hop-unit or force candidates are stronger remain unchanged.
    try:
        stress_points_pos = _stress_points_candidate(problem, seed)
    except Exception:  # noqa: BLE001 -- a failed challenger never sinks the solve
        _LOGGER.warning("point-unit stress undirected challenger failed", exc_info=True)
    else:
        _add_challenger("stress_points", stress_points_pos)

    # Argmax selection; strict inequality means ties go to the incumbent.
    best_name = "incumbent"
    for name, score in scores.items():
        if name != "incumbent" and score > scores[best_name]:
            best_name = name
    _LOGGER.info(
        "Undirected contest candidates=%s winner=%s",
        ", ".join(f"{name}:{score:.3f}" for name, score in scores.items()),
        best_name,
    )
    return positions[best_name]


@register_op
@dataclass(frozen=True)
class UndirectedPortfolioRoute(Op):
    """Run the undirected candidate contest and select the honest winner."""

    config: UndirectedPortfolioRouteConfig = field(default_factory=UndirectedPortfolioRouteConfig)

    name: ClassVar[str] = "undirected_portfolio_route"
    category: ClassVar[OpCategory] = OpCategory.CONTROL
    reads: ClassVar[Tuple[str, ...]] = ("pos",)
    writes: ClassVar[Tuple[str, ...]] = ("pos",)
    requires: ClassVar[Tuple[str, ...]] = ()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run the portfolio contest and write the winning positions.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        SolveState
            State with the contest winner's positions.
        """
        layout_config = self.config.layout_config or LayoutConfig()
        state.pos = layout_native_undirected_portfolio(
            problem=problem,
            state=state,
            ctx=ctx,
            config=layout_config,
        )
        return state


def build_native_undirected_portfolio_pipeline(config: LayoutConfig) -> Pipeline:
    """Build the undirected-portfolio route as a one-op pipeline.

    Follows the existing top-level-route precedent: the route is a
    registered op composed into a named pipeline, so pipeline-level callers
    (``build_dagua_pipeline``) and the direct ``_run_native_problem`` branch
    share one implementation.

    Parameters
    ----------
    config : LayoutConfig
        Prepared native configuration.

    Returns
    -------
    Pipeline
        Single-op pipeline running the candidate contest.
    """
    return Pipeline(
        [UndirectedPortfolioRoute(UndirectedPortfolioRouteConfig(layout_config=config))],
        name="native_undirected_portfolio",
    )


__all__ = [
    "MAX_CONTEST_NODES",
    "NEATO_BALANCED_NODE_CAP",
    "NEATO_QUALITY_THRESHOLD",
    "WEIGHTED_SIMILARITY_TRANSFORM",
    "UndirectedPortfolioRoute",
    "UndirectedPortfolioRouteConfig",
    "build_native_undirected_portfolio_pipeline",
    "layout_native_undirected_portfolio",
]
