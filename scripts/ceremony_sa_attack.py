"""GG-3 simulated-annealing fooling attack for the frozen V3 ruler."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from dagua.eval.ruler_v3 import RulerV3Result, score_core_v3

AGGREGATE_TOLERANCE_FRACTION = 0.05
SHAPE_DISTANCE_THRESHOLD = 0.35
DEFAULT_ITERATIONS = 450
DEFAULT_RESTARTS = 4
HIGH_FACET_FLOOR = 0.8
FOOLED_FACET_MAX_DROP = 0.05
OBJECTIVE_PENALTY = 10.0
MIN_BASELINE_SCORE = 1.0e-12


@dataclass(frozen=True)
class ScoreConfig:
    """Deterministic V3 scoring budget for attack evaluations.

    Parameters
    ----------
    crossing_samples : int
        Edge-pair sample budget forwarded to ``score_core_v3``.
    neighborhood_samples : int
        Neighborhood-preservation row budget forwarded to ``score_core_v3``.
    stress_sources : int
        KSM source budget forwarded to ``score_core_v3``.
    stress_targets : int
        KSM per-source target budget forwarded to ``score_core_v3``.
    seed : int
        Frozen scorer seed used for sampled V3 primitives.
    """

    crossing_samples: int = 50_000
    neighborhood_samples: int = 512
    stress_sources: int = 96
    stress_targets: int = 256
    seed: int = 0


@dataclass(frozen=True)
class AttackConfig:
    """Simulated-annealing control parameters.

    Parameters
    ----------
    iterations : int
        SA iterations per restart.
    restarts : int
        Number of independently seeded restarts per family.
    aggregate_tolerance_fraction : float
        Allowed absolute aggregate drift as a fraction of the good baseline.
    shape_distance_threshold : float
        Normalized Procrustes RMSD at which a drawing is deemed materially
        changed after optimal similarity alignment.
    initial_temperature : float
        Initial Metropolis temperature.
    final_temperature : float
        Final Metropolis temperature.
    """

    iterations: int = DEFAULT_ITERATIONS
    restarts: int = DEFAULT_RESTARTS
    aggregate_tolerance_fraction: float = AGGREGATE_TOLERANCE_FRACTION
    shape_distance_threshold: float = SHAPE_DISTANCE_THRESHOLD
    initial_temperature: float = 1.0
    final_temperature: float = 0.02


@dataclass(frozen=True)
class ProbeFamily:
    """Frozen probe graph used for one GG-3 family attack.

    Parameters
    ----------
    family : str
        Stable family label.
    pos : torch.Tensor
        Good baseline positions with shape ``[N, 2]``.
    edges : torch.Tensor
        Edge index with shape ``[2, E]``.
    sizes : torch.Tensor
        Node sizes with shape ``[N, 2]``.
    meta : Mapping[str, Any]
        Declared graph metadata forwarded to V3 conditional groups.
    label_sizes : Optional[torch.Tensor], optional
        Optional label sizes with shape ``[N, 2]``.
    label_offsets : Optional[torch.Tensor], optional
        Optional label offsets with shape ``[N, 2]``.
    edge_length_targets : Optional[torch.Tensor], optional
        Optional declared edge-length targets with shape ``[E]``.
    """

    family: str
    pos: torch.Tensor
    edges: torch.Tensor
    sizes: torch.Tensor
    meta: Mapping[str, Any]
    label_sizes: Optional[torch.Tensor] = None
    label_offsets: Optional[torch.Tensor] = None
    edge_length_targets: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class AttackResult:
    """Published outcome for one family attack.

    Parameters
    ----------
    family : str
        Stable family label.
    baseline_score : float
        Good-layout V3 tiered aggregate.
    best_score : float
        V3 tiered aggregate of the best valid adversarial drawing.
    best_shape_distance : float
        Largest valid normalized Procrustes RMSD found by the attack.
    aggregate_delta_fraction : float
        Absolute aggregate drift as a fraction of the baseline.
    blocked : bool
        Whether the attack found a materially changed drawing inside the
        aggregate tolerance.
    fooled_facets : Tuple[str, ...]
        Facets that stayed high on a blocking morph.
    """

    family: str
    baseline_score: float
    best_score: float
    best_shape_distance: float
    aggregate_delta_fraction: float
    blocked: bool
    fooled_facets: Tuple[str, ...]


def _sizes(count: int, width: float = 0.2, height: Optional[float] = None) -> torch.Tensor:
    """Return uniform probe node boxes.

    Parameters
    ----------
    count : int
        Number of nodes.
    width : float, optional
        Box width.
    height : Optional[float], optional
        Box height. Defaults to ``width``.

    Returns
    -------
    torch.Tensor
        Node sizes with shape ``[N, 2]``.
    """
    box_height = width if height is None else height
    return torch.tensor([[width, box_height] for _index in range(count)], dtype=torch.float64)


def _tensor(coords: Sequence[Tuple[float, float]]) -> torch.Tensor:
    """Create a float64 position tensor from coordinate pairs.

    Parameters
    ----------
    coords : Sequence[Tuple[float, float]]
        Coordinate pairs.

    Returns
    -------
    torch.Tensor
        Position tensor with shape ``[N, 2]``.
    """
    return torch.tensor(coords, dtype=torch.float64)


def _edges(pairs: Sequence[Tuple[int, int]]) -> torch.Tensor:
    """Create a canonical edge-index tensor.

    Parameters
    ----------
    pairs : Sequence[Tuple[int, int]]
        Directed edge pairs.

    Returns
    -------
    torch.Tensor
        Edge index with shape ``[2, E]``.
    """
    return torch.tensor(pairs, dtype=torch.long).t().contiguous()


def build_probe_families() -> Tuple[ProbeFamily, ...]:
    """Build the frozen GG-3 probe set.

    Returns
    -------
    Tuple[ProbeFamily, ...]
        One representative probe for each required family.
    """
    tree_edges = _edges(((0, 1), (0, 2), (0, 3), (3, 4), (3, 5)))
    tree_pos = _tensor(((0.0, 0.0), (-2.0, 1.0), (0.0, 1.0), (2.0, 1.0), (1.5, 2.0), (2.5, 2.0)))
    tree_meta: Dict[str, object] = {
        "declared_tree": True,
        "root": 0,
        "tree_convention": "layered",
    }

    dag_edges = _edges(((0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (3, 5)))
    dag_pos = _tensor(((0.0, 0.0), (-1.4, 1.5), (1.4, 1.5), (0.0, 3.0), (-1.0, 4.5), (1.0, 4.5)))
    dag_meta: Dict[str, object] = {
        "declared_hierarchical": True,
        "flow_direction": "TB",
        "topological_depth": [0, 1, 1, 2, 3, 3],
    }

    clustered_edges = _edges(((0, 1), (1, 2), (3, 4), (4, 5), (2, 6), (6, 3), (5, 7), (7, 0)))
    clustered_pos = _tensor(
        (
            (-3.0, -0.45),
            (-3.25, 0.15),
            (-2.65, 0.35),
            (3.0, -0.45),
            (3.25, 0.15),
            (2.65, 0.35),
            (-0.7, 0.0),
            (0.7, 0.0),
        )
    )
    clustered_meta: Dict[str, object] = {
        "clusters": {"left": [0, 1, 2, 6], "right": [3, 4, 5, 7]},
        "cluster_labels": {"left": "Left", "right": "Right"},
    }

    generic_pos = _tensor(
        (
            (0.0, 0.0),
            (2.0, 0.0),
            (4.0, 0.0),
            (0.0, 2.0),
            (2.0, 2.0),
            (4.0, 2.0),
            (0.0, 4.0),
            (2.0, 4.0),
            (4.0, 4.0),
        )
    )
    generic_edges = _edges(
        (
            (0, 1),
            (1, 2),
            (3, 4),
            (4, 5),
            (6, 7),
            (7, 8),
            (0, 3),
            (3, 6),
            (1, 4),
            (4, 7),
            (2, 5),
            (5, 8),
            (0, 4),
            (4, 8),
        )
    )

    weighted_lengths = (1.0, 2.0, 4.0, 8.0, 12.0)
    weighted_pos = _tensor(
        tuple(
            (0.0, 0.0) if index == 0 else (weighted_lengths[index - 1], 0.08 * index)
            for index in range(len(weighted_lengths) + 1)
        )
    )
    weighted_edges = _edges(tuple((0, target) for target in range(1, len(weighted_lengths) + 1)))
    weighted_meta: Dict[str, object] = {
        "edge_weights": list(weighted_lengths),
        "weight_mode": "distance",
    }
    weighted_targets = torch.tensor(weighted_lengths, dtype=torch.float64)

    ported_edges = _edges(((0, 1), (0, 2), (1, 3), (2, 3)))
    ported_pos = _tensor(((0.0, 0.0), (4.0, -1.0), (4.0, 1.0), (8.0, 0.0)))
    ports: List[Dict[str, object]] = [
        {"edge": 0, "endpoint": "source", "side": "E", "order": 0},
        {"edge": 1, "endpoint": "source", "side": "E", "order": 1},
        {"edge": 0, "endpoint": "target", "side": "W"},
        {"edge": 1, "endpoint": "target", "side": "W"},
        {"edge": 2, "endpoint": "source", "side": "E"},
        {"edge": 3, "endpoint": "source", "side": "E"},
        {"edge": 2, "endpoint": "target", "side": "W", "order": 0},
        {"edge": 3, "endpoint": "target", "side": "W", "order": 1},
    ]
    ported_meta: Dict[str, object] = {
        "ports": ports,
        "flow_direction": "LR",
        "routing_declared": True,
        "route_paths": [
            [(0.0, 0.0), (2.0, -1.0), (4.0, -1.0)],
            [(0.0, 0.0), (2.0, 1.0), (4.0, 1.0)],
            [(4.0, -1.0), (6.0, -1.0), (8.0, 0.0)],
            [(4.0, 1.0), (6.0, 1.0), (8.0, 0.0)],
        ],
        "routed_labels": ["a", "b", "c", "d"],
        "label_positions": [(2.0, -1.0), (2.0, 1.0), (6.0, -1.0), (6.0, 1.0)],
    }

    return (
        ProbeFamily("tree", tree_pos, tree_edges, _sizes(6), tree_meta),
        ProbeFamily("dag", dag_pos, dag_edges, _sizes(6), dag_meta),
        ProbeFamily("clustered", clustered_pos, clustered_edges, _sizes(8), clustered_meta),
        ProbeFamily("generic_force", generic_pos, generic_edges, _sizes(9), {}),
        ProbeFamily(
            "weighted",
            weighted_pos,
            weighted_edges,
            _sizes(6),
            weighted_meta,
            edge_length_targets=weighted_targets,
        ),
        ProbeFamily("ported", ported_pos, ported_edges, _sizes(4), ported_meta),
    )


def probe_by_family(family: str) -> ProbeFamily:
    """Return one frozen probe by family name.

    Parameters
    ----------
    family : str
        Family label.

    Returns
    -------
    ProbeFamily
        Matching probe family.
    """
    probes = {probe.family: probe for probe in build_probe_families()}
    return probes[family]


def _score_probe(
    probe: ProbeFamily,
    pos: torch.Tensor,
    score_config: ScoreConfig,
) -> RulerV3Result:
    """Score one candidate drawing with the frozen V3 aggregate.

    Parameters
    ----------
    probe : ProbeFamily
        Family probe supplying graph inputs and declarations.
    pos : torch.Tensor
        Candidate positions with shape ``[N, 2]``.
    score_config : ScoreConfig
        Deterministic V3 scoring budget.

    Returns
    -------
    RulerV3Result
        Full V3 score result.
    """
    return score_core_v3(
        pos,
        probe.edges,
        probe.sizes,
        label_sizes=probe.label_sizes,
        label_offsets=probe.label_offsets,
        edge_length_targets=probe.edge_length_targets,
        graph_meta=probe.meta,
        crossing_samples=score_config.crossing_samples,
        neighborhood_samples=score_config.neighborhood_samples,
        stress_sources=score_config.stress_sources,
        stress_targets=score_config.stress_targets,
        seed=score_config.seed,
    )


def procrustes_shape_distance(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    """Return normalized Procrustes RMSD after optimal similarity alignment.

    Parameters
    ----------
    reference : torch.Tensor
        Good baseline positions with shape ``[N, 2]``.
    candidate : torch.Tensor
        Candidate positions with shape ``[N, 2]``.

    Returns
    -------
    float
        RMSD divided by the baseline RMS spread. Values near zero are
        similarity-equivalent; values above the frozen GG-3 threshold are
        materially different shapes.
    """
    ref = reference.detach().cpu().numpy().astype(np.float64)
    cand = candidate.detach().cpu().numpy().astype(np.float64)
    ref_centered = ref - ref.mean(axis=0, keepdims=True)
    cand_centered = cand - cand.mean(axis=0, keepdims=True)
    ref_spread = math.sqrt(float(np.mean(np.sum(ref_centered * ref_centered, axis=1))))
    cand_energy = float(np.sum(cand_centered * cand_centered))
    if ref_spread <= 0.0 or cand_energy <= 0.0:
        return 0.0
    u_matrix, singular_values, vt_matrix = np.linalg.svd(cand_centered.T @ ref_centered)
    rotation = u_matrix @ vt_matrix
    scale = float(np.sum(singular_values) / cand_energy)
    aligned = scale * (cand_centered @ rotation)
    residual = aligned - ref_centered
    rmsd = math.sqrt(float(np.mean(np.sum(residual * residual, axis=1))))
    return rmsd / ref_spread


def _aggregate_delta_fraction(score: float, baseline_score: float) -> float:
    """Return aggregate drift as a fraction of the baseline.

    Parameters
    ----------
    score : float
        Candidate V3 tiered aggregate.
    baseline_score : float
        Good-layout V3 tiered aggregate.

    Returns
    -------
    float
        Absolute fractional aggregate drift.
    """
    denominator = max(MIN_BASELINE_SCORE, abs(baseline_score))
    return abs(score - baseline_score) / denominator


def _objective(
    shape_distance: float,
    aggregate_delta_fraction: float,
    attack_config: AttackConfig,
) -> float:
    """Return the adversarial SA objective with a hard tolerance penalty.

    Parameters
    ----------
    shape_distance : float
        Normalized Procrustes RMSD.
    aggregate_delta_fraction : float
        Absolute aggregate drift fraction.
    attack_config : AttackConfig
        Attack thresholds.

    Returns
    -------
    float
        Value to maximize during SA.
    """
    excess = aggregate_delta_fraction - attack_config.aggregate_tolerance_fraction
    if excess <= 0.0:
        return shape_distance
    excess_ratio = excess / max(MIN_BASELINE_SCORE, attack_config.aggregate_tolerance_fraction)
    return shape_distance - OBJECTIVE_PENALTY * (1.0 + excess_ratio * excess_ratio)


def _temperature(step: int, attack_config: AttackConfig) -> float:
    """Return geometric SA temperature for one step.

    Parameters
    ----------
    step : int
        Zero-based SA step.
    attack_config : AttackConfig
        Attack schedule.

    Returns
    -------
    float
        Positive temperature.
    """
    if attack_config.iterations <= 1:
        return attack_config.final_temperature
    ratio = step / float(attack_config.iterations - 1)
    return (
        attack_config.initial_temperature
        * (attack_config.final_temperature / attack_config.initial_temperature) ** ratio
    )


def _baseline_spread(pos: torch.Tensor) -> float:
    """Return RMS spread of a baseline drawing.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.

    Returns
    -------
    float
        RMS distance from centroid.
    """
    centered = pos - pos.mean(dim=0, keepdim=True)
    return float(torch.sqrt(torch.mean(torch.sum(centered * centered, dim=1))).item())


def _proposal_scale(reference: torch.Tensor, temperature: float) -> float:
    """Return the step scale for a proposal at one temperature.

    Parameters
    ----------
    reference : torch.Tensor
        Good baseline positions with shape ``[N, 2]``.
    temperature : float
        Current SA temperature.

    Returns
    -------
    float
        Coordinate perturbation scale.
    """
    spread = max(1.0e-6, _baseline_spread(reference))
    return spread * (0.03 + 0.55 * temperature)


def _propose_positions(
    current: torch.Tensor,
    reference: torch.Tensor,
    rng: np.random.Generator,
    temperature: float,
) -> torch.Tensor:
    """Create one adversarial node-position proposal.

    Parameters
    ----------
    current : torch.Tensor
        Current positions with shape ``[N, 2]``.
    reference : torch.Tensor
        Good baseline positions with shape ``[N, 2]``.
    rng : numpy.random.Generator
        Deterministic random generator.
    temperature : float
        Current SA temperature.

    Returns
    -------
    torch.Tensor
        Proposed positions with shape ``[N, 2]``.
    """
    proposal = current.detach().clone()
    count = int(proposal.shape[0])
    scale = _proposal_scale(reference, temperature)
    operation = float(rng.random())
    if operation < 0.58:
        node = int(rng.integers(0, count))
        delta = torch.tensor(rng.normal(0.0, scale, size=2), dtype=torch.float64)
        proposal[node] += delta
    elif operation < 0.78 and count >= 2:
        first, second = rng.choice(count, size=2, replace=False)
        first_index = int(first)
        second_index = int(second)
        tmp = proposal[first_index].clone()
        proposal[first_index] = proposal[second_index]
        proposal[second_index] = tmp
        noise = torch.tensor(rng.normal(0.0, 0.15 * scale, size=(2, 2)), dtype=torch.float64)
        proposal[[first_index, second_index]] += noise
    elif operation < 0.9:
        mask = rng.random(count) < 0.5
        if not bool(mask.any()):
            mask[int(rng.integers(0, count))] = True
        center = reference.mean(dim=0)
        axis = int(rng.integers(0, 2))
        selected = torch.tensor(mask, dtype=torch.bool)
        proposal[selected, axis] = 2.0 * center[axis] - proposal[selected, axis]
        proposal[selected] += torch.tensor(
            rng.normal(0.0, 0.08 * scale, size=(int(mask.sum()), 2)),
            dtype=torch.float64,
        )
    else:
        subset_size = int(rng.integers(1, count + 1))
        subset = rng.choice(count, size=subset_size, replace=False)
        shift = torch.tensor(rng.normal(0.0, scale, size=2), dtype=torch.float64)
        proposal[torch.tensor(subset, dtype=torch.long)] += shift
    return proposal


def _facet_scores(result: RulerV3Result) -> Dict[str, float]:
    """Extract finite facet scores from a V3 result.

    Parameters
    ----------
    result : RulerV3Result
        Full V3 score result.

    Returns
    -------
    Dict[str, float]
        Finite facet scores keyed by facet code.
    """
    scores: Dict[str, float] = {}
    for code, facet in result.facets.items():
        if facet.score is None:
            continue
        score = float(facet.score)
        if math.isfinite(score):
            scores[code] = score
    return scores


def fooled_facets(
    baseline_result: RulerV3Result,
    candidate_result: RulerV3Result,
) -> Tuple[str, ...]:
    """Return facets that stayed high on an adversarial candidate.

    Parameters
    ----------
    baseline_result : RulerV3Result
        Good-layout score result.
    candidate_result : RulerV3Result
        Candidate score result.

    Returns
    -------
    Tuple[str, ...]
        Facet codes whose candidate score remained high and barely dropped.
    """
    baseline = _facet_scores(baseline_result)
    candidate = _facet_scores(candidate_result)
    held: List[str] = []
    for code, candidate_score in candidate.items():
        baseline_score = baseline.get(code)
        if baseline_score is None:
            continue
        if (
            candidate_score >= HIGH_FACET_FLOOR
            and baseline_score - candidate_score <= FOOLED_FACET_MAX_DROP
        ):
            held.append(code)
    return tuple(sorted(held))


def run_family_attack(
    probe: ProbeFamily,
    *,
    seed: int,
    attack_config: AttackConfig = AttackConfig(),
    score_config: ScoreConfig = ScoreConfig(),
) -> AttackResult:
    """Run the GG-3 SA morph attack against one probe family.

    Parameters
    ----------
    probe : ProbeFamily
        Frozen family probe.
    seed : int
        Base deterministic seed for the SA process.
    attack_config : AttackConfig, optional
        SA budget and thresholds.
    score_config : ScoreConfig, optional
        Deterministic V3 scoring budget.

    Returns
    -------
    AttackResult
        Best valid morph and PASS/BLOCK verdict for the family.
    """
    baseline_result = _score_probe(probe, probe.pos, score_config)
    baseline_score = float(baseline_result.scores["tiered"])
    best_valid_result = baseline_result
    best_valid_shape = 0.0
    best_valid_score = baseline_score
    rng = np.random.default_rng(seed)

    for _restart in range(attack_config.restarts):
        current = probe.pos.detach().clone()
        current_result = baseline_result
        current_score = baseline_score
        current_shape = 0.0
        current_delta = 0.0
        current_objective = _objective(current_shape, current_delta, attack_config)
        for step in range(attack_config.iterations):
            temperature = _temperature(step, attack_config)
            proposed = _propose_positions(current, probe.pos, rng, temperature)
            proposed_result = _score_probe(probe, proposed, score_config)
            proposed_score = float(proposed_result.scores["tiered"])
            proposed_shape = procrustes_shape_distance(probe.pos, proposed)
            proposed_delta = _aggregate_delta_fraction(proposed_score, baseline_score)
            proposed_objective = _objective(proposed_shape, proposed_delta, attack_config)
            objective_gain = proposed_objective - current_objective
            accept = objective_gain >= 0.0 or rng.random() < math.exp(
                max(-700.0, objective_gain / max(1.0e-12, temperature))
            )
            if accept:
                current = proposed
                current_result = proposed_result
                current_score = proposed_score
                current_shape = proposed_shape
                current_delta = proposed_delta
                current_objective = proposed_objective
            if (
                proposed_delta <= attack_config.aggregate_tolerance_fraction
                and proposed_shape > best_valid_shape
            ):
                best_valid_result = proposed_result
                best_valid_shape = proposed_shape
                best_valid_score = proposed_score
            if (
                current_delta <= attack_config.aggregate_tolerance_fraction
                and current_shape > best_valid_shape
            ):
                best_valid_result = current_result
                best_valid_shape = current_shape
                best_valid_score = current_score

    best_delta = _aggregate_delta_fraction(best_valid_score, baseline_score)
    blocked = (
        best_delta <= attack_config.aggregate_tolerance_fraction
        and best_valid_shape >= attack_config.shape_distance_threshold
    )
    return AttackResult(
        family=probe.family,
        baseline_score=baseline_score,
        best_score=best_valid_score,
        best_shape_distance=best_valid_shape,
        aggregate_delta_fraction=best_delta,
        blocked=blocked,
        fooled_facets=fooled_facets(baseline_result, best_valid_result) if blocked else (),
    )


def run_all_attacks(
    *,
    seed: int,
    families: Optional[Iterable[str]] = None,
    attack_config: AttackConfig = AttackConfig(),
    score_config: ScoreConfig = ScoreConfig(),
) -> Tuple[AttackResult, ...]:
    """Run GG-3 attacks for the selected families.

    Parameters
    ----------
    seed : int
        Base deterministic seed.
    families : Optional[Iterable[str]], optional
        Family names to run. Defaults to all frozen probes.
    attack_config : AttackConfig, optional
        SA budget and thresholds.
    score_config : ScoreConfig, optional
        Deterministic V3 scoring budget.

    Returns
    -------
    Tuple[AttackResult, ...]
        Per-family attack outcomes in frozen probe order.
    """
    selected = set(families) if families is not None else None
    results: List[AttackResult] = []
    for index, probe in enumerate(build_probe_families()):
        if selected is not None and probe.family not in selected:
            continue
        results.append(
            run_family_attack(
                probe,
                seed=seed + 10_003 * index,
                attack_config=attack_config,
                score_config=score_config,
            )
        )
    return tuple(results)


def format_results_table(results: Sequence[AttackResult]) -> str:
    """Format the ceremony PASS/BLOCK table.

    Parameters
    ----------
    results : Sequence[AttackResult]
        Per-family attack outcomes.

    Returns
    -------
    str
        Markdown table with the required verdict columns.
    """
    lines = [
        "| family | best shape dist | aggregate delta | verdict | fooled facets |",
        "|---|---:|---:|---|---|",
    ]
    for result in results:
        verdict = "BLOCK" if result.blocked else "PASS"
        facets = ", ".join(result.fooled_facets) if result.fooled_facets else "-"
        lines.append(
            f"| {result.family} | {result.best_shape_distance:.4f} | "
            f"{100.0 * result.aggregate_delta_fraction:.2f}% | {verdict} | {facets} |"
        )
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the ceremony runner.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    family_names = tuple(probe.family for probe in build_probe_families())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="base deterministic SA seed")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--restarts", type=int, default=DEFAULT_RESTARTS)
    parser.add_argument("--shape-threshold", type=float, default=SHAPE_DISTANCE_THRESHOLD)
    parser.add_argument(
        "--aggregate-tolerance",
        type=float,
        default=AGGREGATE_TOLERANCE_FRACTION,
        help="allowed aggregate drift fraction",
    )
    parser.add_argument(
        "--family",
        choices=family_names,
        action="append",
        help="family to attack; repeat to run multiple families",
    )
    return parser.parse_args()


def main() -> int:
    """Run the GG-3 ceremony attack from the command line.

    Returns
    -------
    int
        Process exit code. ``1`` means at least one family blocked.
    """
    args = _parse_args()
    attack_config = AttackConfig(
        iterations=int(args.iterations),
        restarts=int(args.restarts),
        aggregate_tolerance_fraction=float(args.aggregate_tolerance),
        shape_distance_threshold=float(args.shape_threshold),
    )
    results = run_all_attacks(
        seed=int(args.seed),
        families=args.family,
        attack_config=attack_config,
        score_config=ScoreConfig(),
    )
    print(format_results_table(results))
    print()
    print(
        "thresholds: "
        f"aggregate_tolerance={attack_config.aggregate_tolerance_fraction:.2%}, "
        f"shape_distance={attack_config.shape_distance_threshold:.2f}, "
        f"budget={attack_config.restarts}x{attack_config.iterations}"
    )
    return 1 if any(result.blocked for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
