"""Stress-seeded clustered candidate for the native undirected portfolio."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import torch

from dagua.layout.ops.state import LayoutProblem
from dagua.layout.projection import project_overlaps

ARM_S_CANDIDATE_PREFIX = "arm_s_stress"
ARM_S_PRIOR_S = 90.0
ARM_S_FULL_REFEREE_PRIOR_S = 55.0
ARM_S_STRESS_STEPS = 30
ARM_S_STRESS_SEED = 42
ARM_S_SCALE_MULTIPLIERS = (1.6, 2.5, 4.0, 10.0)
ARM_S_STRICT_WIN_REFERENCE = 77.316999
ARM_S_ACCEPTANCE_MARGIN = 0.05
ARM_S_MIN_NEIGHBORHOOD_PRESERVATION = 0.30
ARM_S_MIN_KSM = 0.6516
ARM_S_MIN_NESTING = 0.7353
PROJECTION_MAX_MEAN_DIAGONAL_FACTOR = 0.25
ARM_S_PROJECTION_ITERATIONS = 40
ARM_S_HEAL_PASSES = 16
ARM_S_HEAL_EPSILON = 1.0e-3


@dataclass(frozen=True)
class ArmSProjectionTelemetry:
    """Last-mile projection movement for one Arm S candidate.

    Parameters
    ----------
    max_displacement : float
        Maximum per-node movement caused by the last-mile projection.
    mean_node_diagonal : float
        Mean node-box diagonal used to normalize the cosmetic bound.
    displacement_ratio : float
        ``max_displacement / mean_node_diagonal``.
    """

    max_displacement: float
    mean_node_diagonal: float
    displacement_ratio: float


@dataclass(frozen=True)
class ArmSCandidate:
    """Arm S portfolio candidate payload.

    Parameters
    ----------
    name : str
        Stable candidate name.
    positions : torch.Tensor
        Candidate positions with shape ``[N, 2]``.
    scale_multiplier : float
        Deterministic median-edge calibration multiplier.
    pre_projection_overlap_count : int
        Exact overlap count after bounded heal and before projection.
    projection : ArmSProjectionTelemetry
        Last-mile projection displacement telemetry.
    """

    name: str
    positions: torch.Tensor
    scale_multiplier: float
    pre_projection_overlap_count: int
    projection: ArmSProjectionTelemetry


@dataclass(frozen=True)
class ArmSProxyScore:
    """Cheap Arm S ladder score used before the honest referee.

    Parameters
    ----------
    candidate_name : str
        Stable name of the scale-ladder candidate.
    score : float
        Higher-is-better proxy score.
    overlap_count : int
        Exact node-box overlap count used as a hard proxy penalty.
    """

    candidate_name: str
    score: float
    overlap_count: int


@dataclass(frozen=True)
class ArmSAdmissionReport:
    """Named-floor admission result for Arm S.

    Parameters
    ----------
    passed : bool
        Whether every strict Arm S gate passed.
    failures : tuple[str, ...]
        Machine-readable gate failures.
    overlap_count : int
        Exact final node overlap count from the scored metric payload.
    """

    passed: bool
    failures: tuple[str, ...]
    overlap_count: int


def _mean_node_diagonal(node_sizes: torch.Tensor) -> torch.Tensor:
    """Return the mean node-box diagonal clamped away from zero.

    Parameters
    ----------
    node_sizes : torch.Tensor
        Node-size tensor with shape ``[N, 2]``.

    Returns
    -------
    torch.Tensor
        Scalar mean diagonal in layout units.
    """
    return torch.linalg.vector_norm(node_sizes, dim=1).mean().clamp_min(1.0)


def _median_edge_length(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Return median Euclidean edge length for calibration.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.

    Returns
    -------
    torch.Tensor
        Scalar median edge length clamped away from zero.
    """
    if edge_index.ndim != 2 or edge_index.shape[1] == 0:
        return torch.ones((), device=pos.device, dtype=pos.dtype)
    edges = edge_index.to(device=pos.device, dtype=torch.long)
    lengths = torch.linalg.vector_norm(pos[edges[0]] - pos[edges[1]], dim=1)
    return lengths.median().clamp_min(1.0e-6)


def calibrate_arm_s_scale(
    seed_pos: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    multiplier: float,
) -> torch.Tensor:
    """Scale a stress seed to a deterministic median-edge target.

    Parameters
    ----------
    seed_pos : torch.Tensor
        Stress seed positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    node_sizes : torch.Tensor
        Node sizes with shape ``[N, 2]``.
    multiplier : float
        Target median edge length as a multiple of mean node diagonal.

    Returns
    -------
    torch.Tensor
        Scaled ``float32`` positions with shape ``[N, 2]``.
    """
    work = seed_pos.detach().to(device="cpu", dtype=torch.float32)
    sizes = node_sizes.detach().to(device="cpu", dtype=torch.float32)
    center = work.mean(dim=0, keepdim=True)
    target = float(multiplier) * _mean_node_diagonal(sizes)
    scale = target / _median_edge_length(work, edge_index.detach().to(device="cpu"))
    return ((work - center) * scale + center).to(dtype=torch.float32)


def exact_arm_s_overlap_count(pos: torch.Tensor, node_sizes: torch.Tensor) -> int:
    """Count strict axis-aligned node-box overlaps exactly.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    node_sizes : torch.Tensor
        Node sizes with shape ``[N, 2]``.

    Returns
    -------
    int
        Number of unordered overlapping node pairs.
    """
    if pos.shape[0] <= 1:
        return 0
    work_pos = pos.detach().to(device="cpu", dtype=torch.float32)
    sizes = node_sizes.detach().to(device="cpu", dtype=torch.float32)
    dx = (work_pos[:, None, 0] - work_pos[None, :, 0]).abs()
    dy = (work_pos[:, None, 1] - work_pos[None, :, 1]).abs()
    min_dx = (sizes[:, None, 0] + sizes[None, :, 0]) * 0.5
    min_dy = (sizes[:, None, 1] + sizes[None, :, 1]) * 0.5
    overlap = (dx < min_dx) & (dy < min_dy)
    overlap.fill_diagonal_(False)
    return int(overlap.triu(diagonal=1).sum().item())


def _overlap_pairs(pos: torch.Tensor, node_sizes: torch.Tensor) -> torch.Tensor:
    """Return strict overlapping node pairs on CPU.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    node_sizes : torch.Tensor
        Node sizes with shape ``[N, 2]``.

    Returns
    -------
    torch.Tensor
        Long tensor with shape ``[P, 2]`` containing unordered overlap pairs.
    """
    work_pos = pos.detach().to(device="cpu", dtype=torch.float32)
    sizes = node_sizes.detach().to(device="cpu", dtype=torch.float32)
    dx = (work_pos[:, None, 0] - work_pos[None, :, 0]).abs()
    dy = (work_pos[:, None, 1] - work_pos[None, :, 1]).abs()
    min_dx = (sizes[:, None, 0] + sizes[None, :, 0]) * 0.5
    min_dy = (sizes[:, None, 1] + sizes[None, :, 1]) * 0.5
    overlap = (dx < min_dx) & (dy < min_dy)
    overlap.fill_diagonal_(False)
    return torch.triu(overlap, diagonal=1).nonzero()


def bounded_arm_s_overlap_heal(pos: torch.Tensor, node_sizes: torch.Tensor) -> torch.Tensor:
    """Resolve residual seed overlaps while preserving the global embedding.

    Parameters
    ----------
    pos : torch.Tensor
        Scaled stress-seed positions with shape ``[N, 2]``.
    node_sizes : torch.Tensor
        Node sizes with shape ``[N, 2]``.

    Returns
    -------
    torch.Tensor
        Healed positions with shape ``[N, 2]``.
    """
    work = pos.detach().to(device="cpu", dtype=torch.float32).clone()
    sizes = node_sizes.detach().to(device="cpu", dtype=torch.float32)
    for _ in range(ARM_S_HEAL_PASSES):
        pairs = _overlap_pairs(work, sizes)
        if pairs.numel() == 0:
            break
        for left_raw, right_raw in pairs.tolist():
            left = int(left_raw)
            right = int(right_raw)
            delta = work[left] - work[right]
            min_sep = (sizes[left] + sizes[right]) * 0.5
            penetration = min_sep - delta.abs()
            if penetration[0] <= 0.0 or penetration[1] <= 0.0:
                continue
            if float(penetration[0].item()) <= float(penetration[1].item()):
                axis = 0
            else:
                axis = 1
            sign = 1.0 if float(delta[axis].item()) >= 0.0 else -1.0
            shift = sign * (float(penetration[axis].item()) + ARM_S_HEAL_EPSILON) * 0.5
            work[left, axis] += shift
            work[right, axis] -= shift
    return work


def _project_last_mile(
    pos: torch.Tensor,
    node_sizes: torch.Tensor,
) -> tuple[torch.Tensor, ArmSProjectionTelemetry]:
    """Apply the existing convergent projector and measure movement.

    Parameters
    ----------
    pos : torch.Tensor
        Healed positions with shape ``[N, 2]``.
    node_sizes : torch.Tensor
        Node sizes with shape ``[N, 2]``.

    Returns
    -------
    tuple[torch.Tensor, ArmSProjectionTelemetry]
        Projected positions and movement telemetry.
    """
    sizes = node_sizes.detach().to(device="cpu", dtype=torch.float32)
    before = pos.detach().to(device="cpu", dtype=torch.float32)
    projected = before.clone()
    project_overlaps(
        projected,
        sizes,
        iterations=ARM_S_PROJECTION_ITERATIONS,
        convergent=True,
    )
    max_displacement = float(torch.linalg.vector_norm(projected - before, dim=1).max().item())
    mean_diagonal = float(_mean_node_diagonal(sizes).item())
    return projected.to(dtype=torch.float32), ArmSProjectionTelemetry(
        max_displacement=max_displacement,
        mean_node_diagonal=mean_diagonal,
        displacement_ratio=max_displacement / max(mean_diagonal, 1.0),
    )


def arm_s_proxy_score(
    candidate: ArmSCandidate,
    problem: LayoutProblem,
    cluster_ids: Optional[torch.Tensor] = None,
) -> ArmSProxyScore:
    """Score one Arm S scale variant with cheap local metrics.

    Parameters
    ----------
    candidate : ArmSCandidate
        Scale-ladder candidate to proxy-score.
    problem : LayoutProblem
        Prepared native layout problem.
    cluster_ids : torch.Tensor, optional
        Optional per-node cluster ids for cheap cluster separation metrics.

    Returns
    -------
    ArmSProxyScore
        Proxy payload. The honest admission still happens later with
        ``metrics.full``; this score only chooses which scale gets that pass.
    """
    if problem.node_sizes is None:
        return ArmSProxyScore(
            candidate_name=candidate.name,
            score=float("-inf"),
            overlap_count=0,
        )
    from dagua.metrics import (
        cluster_silhouette_score,
        composite_auto,
        isotonic_stress,
        neighborhood_preservation,
        quick,
    )

    cpu_pos = candidate.positions.detach().to(device="cpu", dtype=torch.float32)
    cpu_edges = problem.edge_index.detach().to(device="cpu")
    cpu_sizes = problem.node_sizes.detach().to(device="cpu", dtype=torch.float32)
    numeric = quick(
        cpu_pos,
        cpu_edges,
        node_sizes=cpu_sizes,
        direction=problem.direction,
        seed=ARM_S_STRESS_SEED,
    )
    numeric.update(
        isotonic_stress(
            cpu_pos,
            cpu_edges,
            int(problem.num_nodes),
            n_sources=32,
            n_targets=128,
        )
    )
    numeric.update(
        neighborhood_preservation(
            cpu_pos,
            cpu_edges,
            int(problem.num_nodes),
            n_samples=512,
        )
    )
    if cluster_ids is not None:
        numeric.update(cluster_silhouette_score(cpu_pos, cluster_ids.detach().to(device="cpu")))
    overlap_count = exact_arm_s_overlap_count(cpu_pos, cpu_sizes)
    numeric["overlap_count"] = float(overlap_count)
    proxy = float(composite_auto(numeric, is_semantically_directed=False))
    if overlap_count:
        proxy -= 1000.0 * float(overlap_count)
    return ArmSProxyScore(
        candidate_name=candidate.name,
        score=proxy,
        overlap_count=overlap_count,
    )


def build_arm_s_stress_candidates(problem: LayoutProblem) -> dict[str, ArmSCandidate]:
    """Build cluster-gated Arm S stress-seeded portfolio candidates.

    Parameters
    ----------
    problem : LayoutProblem
        Prepared native layout problem.

    Returns
    -------
    dict[str, ArmSCandidate]
        Candidate payloads keyed by stable name, or an empty dictionary when
        the row is not clustered or lacks node boxes.
    """
    if not problem.clusters or problem.node_sizes is None:
        return {}
    from dagua.layout.ops.pipelines.stress_sgd import layout_stress_sgd_pipeline

    seed_pos = layout_stress_sgd_pipeline(
        problem.edge_index.detach().to(device="cpu"),
        int(problem.num_nodes),
        node_sizes=None,
        steps=ARM_S_STRESS_STEPS,
        seed=ARM_S_STRESS_SEED,
    )
    if not isinstance(seed_pos, torch.Tensor) or not bool(torch.isfinite(seed_pos).all().item()):
        return {}
    candidates: dict[str, ArmSCandidate] = {}
    for multiplier in ARM_S_SCALE_MULTIPLIERS:
        scaled = calibrate_arm_s_scale(
            seed_pos,
            problem.edge_index,
            problem.node_sizes,
            multiplier,
        )
        healed = bounded_arm_s_overlap_heal(scaled, problem.node_sizes)
        pre_projection_overlaps = exact_arm_s_overlap_count(healed, problem.node_sizes)
        projected, projection = _project_last_mile(healed, problem.node_sizes)
        name = f"{ARM_S_CANDIDATE_PREFIX}_k{multiplier:g}"
        candidates[name] = ArmSCandidate(
            name=name,
            positions=projected,
            scale_multiplier=float(multiplier),
            pre_projection_overlap_count=pre_projection_overlaps,
            projection=projection,
        )
    return candidates


def select_arm_s_proxy_finalist(
    candidates: Mapping[str, ArmSCandidate],
    problem: LayoutProblem,
    cluster_ids: Optional[torch.Tensor] = None,
) -> tuple[Optional[ArmSCandidate], dict[str, ArmSProxyScore]]:
    """Select the single Arm S ladder variant that reaches full scoring.

    Parameters
    ----------
    candidates : Mapping[str, ArmSCandidate]
        Full deterministic scale ladder keyed by candidate name.
    problem : LayoutProblem
        Prepared native layout problem.
    cluster_ids : torch.Tensor, optional
        Optional per-node cluster ids for cheap cluster separation metrics.

    Returns
    -------
    tuple[ArmSCandidate | None, dict[str, ArmSProxyScore]]
        The best proxy candidate, or ``None`` when no finite candidate exists,
        plus proxy telemetry for every evaluated ladder member.
    """
    proxy_scores: dict[str, ArmSProxyScore] = {}
    best: Optional[ArmSCandidate] = None
    best_score = float("-inf")
    for name, candidate in candidates.items():
        score = arm_s_proxy_score(candidate, problem, cluster_ids)
        proxy_scores[name] = score
        if score.score > best_score:
            best = candidate
            best_score = score.score
    return best, proxy_scores


def build_arm_s_stress_finalist(
    problem: LayoutProblem,
    cluster_ids: Optional[torch.Tensor] = None,
) -> tuple[Optional[ArmSCandidate], dict[str, ArmSCandidate], dict[str, ArmSProxyScore]]:
    """Build the Arm S ladder and return only its cheap-proxy finalist.

    Parameters
    ----------
    problem : LayoutProblem
        Prepared native layout problem.
    cluster_ids : torch.Tensor, optional
        Optional per-node cluster ids for cheap cluster separation metrics.

    Returns
    -------
    tuple[ArmSCandidate | None, dict[str, ArmSCandidate], dict[str, ArmSProxyScore]]
        Single finalist selected by the cheap proxy, the complete ladder
        payload for telemetry, and per-scale proxy scores.
    """
    candidates = build_arm_s_stress_candidates(problem)
    finalist, proxy_scores = select_arm_s_proxy_finalist(candidates, problem, cluster_ids)
    return finalist, candidates, proxy_scores


def evaluate_arm_s_admission(
    candidate: ArmSCandidate,
    metrics: Mapping[str, float],
    *,
    extended_score: float,
) -> ArmSAdmissionReport:
    """Evaluate strict Arm S named floors against scored metrics.

    Parameters
    ----------
    candidate : ArmSCandidate
        Candidate payload being considered for admission.
    metrics : Mapping[str, float]
        Numeric metrics from the exact full ruler.
    extended_score : float
        Candidate extended composite from the same exact metric pass.

    Returns
    -------
    ArmSAdmissionReport
        Gate result and named failures.
    """
    failures: list[str] = []
    overlap_count = int(metrics.get("overlap_count", 0))
    if extended_score <= ARM_S_STRICT_WIN_REFERENCE + ARM_S_ACCEPTANCE_MARGIN:
        failures.append("strict_win")
    if overlap_count != 0:
        failures.append(f"node_overlaps:{overlap_count}")
    if (
        float(metrics.get("neighborhood_preservation_score", 0.0))
        < ARM_S_MIN_NEIGHBORHOOD_PRESERVATION
    ):
        failures.append("neighborhood_preservation_floor")
    if float(metrics.get("ksm_score", 0.0)) < ARM_S_MIN_KSM:
        failures.append("ksm_floor")
    if float(metrics.get("cluster_nesting_fidelity_score", 0.0)) < ARM_S_MIN_NESTING:
        failures.append("nesting_floor")
    if candidate.projection.displacement_ratio > PROJECTION_MAX_MEAN_DIAGONAL_FACTOR + 1.0e-6:
        failures.append("projection_bound")
    return ArmSAdmissionReport(
        passed=not failures,
        failures=tuple(failures),
        overlap_count=overlap_count,
    )


__all__ = [
    "ARM_S_ACCEPTANCE_MARGIN",
    "ARM_S_CANDIDATE_PREFIX",
    "ARM_S_FULL_REFEREE_PRIOR_S",
    "ARM_S_PRIOR_S",
    "ARM_S_STRICT_WIN_REFERENCE",
    "ArmSAdmissionReport",
    "ArmSCandidate",
    "ArmSProjectionTelemetry",
    "ArmSProxyScore",
    "arm_s_proxy_score",
    "bounded_arm_s_overlap_heal",
    "build_arm_s_stress_finalist",
    "build_arm_s_stress_candidates",
    "calibrate_arm_s_scale",
    "evaluate_arm_s_admission",
    "exact_arm_s_overlap_count",
    "select_arm_s_proxy_finalist",
]
