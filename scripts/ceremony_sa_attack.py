"""GG-3 simulated-annealing fooling attack for the frozen V3 ruler."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from dagua.eval.ruler_v3 import (
    FAMILY_MARGIN_ALLOWANCES,
    RulerV3Facet,
    RulerV3Result,
    angle_weighted_crossing_score,
    material_hold_ineligible,
    referee_eligibility_key,
    score_core_v3,
    severe_g6_breach,
    sol_declared_weight_subcontract,
    tier1_measurement_weight,
    tier1_only_drop,
    tier1_tradeoff_flags,
    with_tier1_tradeoff_flag,
)
from dagua.eval.ruler_v3 import (
    severe_g6_floor_breach as severe_g6_pair_floor_breach,
)
from dagua.eval.ruler_v3_groups import G6_SEVERE_FLOOR, G6_SOFT_FLOOR

AGGREGATE_TOLERANCE_FRACTION = 0.05
GG3_BLOCK_AGGREGATE_DELTA_FRACTION = 0.02
SHAPE_DISTANCE_THRESHOLD = 0.35
PRIMARY_FAITHFULNESS_DROP_THRESHOLD = 0.05
SOL_PRIMARY_FAITHFULNESS_DROP_THRESHOLD = 0.08
SOL_AGGREGATE_DELTA_TIERED_POINTS = 3.0
PRIMARY_FAITHFULNESS_FACETS = ("C1", "C3", "G6_weighted_ksm")
ATTACK_THE_FIX_OBJECTIVE_MODES = (
    "g6_subsevere_corridor",
    "clustered_g2_push",
    "c2_perpendicularize",
    "c4_g6_band_edge",
    "eligibility_surface_abuse",
    "free_clause_dodging",
    "pool_level_eligibility",
    "non_g6_financed_corridor",
    "weighted_margin_tail",
    "deramped_knife_edge_corridor",
    "clustered_softmin_repush",
)
RETARGETED_OBJECTIVE_MODES = (
    "tier1_loss",
    "g6_damage",
    *ATTACK_THE_FIX_OBJECTIVE_MODES,
)
# FROZEN -- pre-registered before the deciding measurement; do not post-hoc ratchet.
TIER1_ONLY_MATERIALITY_DROP = 5.0
DEGENERATE_PCA_RATIO = 0.05
DEGENERATE_HUB_ARC_DEG = 10.0
DEGENERATE_TIER1_TOLERANCE = 1.0
DEGENERATE_C3_IMPROVE = 0.05
GG3_VERDICT_BLOCK = "BLOCK"
GG3_VERDICT_PASS_WITH_T1_TRADEOFF = "PASS_WITH_T1_TRADEOFF"
GG3_VERDICT_PASS_DEGENERATE_ESCAPE = "PASS_DEGENERATE_ESCAPE"
GG3_VERDICT_PASS = "PASS"
DEFAULT_ITERATIONS = 450
DEFAULT_RESTARTS = 4
DEFAULT_BATTERY_SEEDS_PER_FAMILY = 3
HIGH_FACET_FLOOR = 0.8
FOOLED_FACET_MAX_DROP = 0.05
OBJECTIVE_PENALTY = 10.0
MIN_BASELINE_SCORE = 1.0e-12
DEFAULT_DIAGNOSTICS_DIR = Path.home() / ".claude/research/dagua/megasprint/gg3_diagnostics"
DEFAULT_ATTACK_THE_FIX_RESULTS_PATH = (
    Path.home() / ".claude/research/dagua/megasprint/fresh_attack_results.md"
)
DEFAULT_ATTACK_THE_FIX_ARTIFACT_DIR = (
    Path.home() / ".claude/research/dagua/megasprint/fresh_attack_artifacts"
)
PHASE1_CLEAN_ROWS_STORE = (
    Path.home() / ".claude/research/dagua/megasprint/PHASE1_CLEAN_REAL_ROWS_V3.json"
)
MARGIN_AUDIT_FALLBACK = 12.2
MARGIN_AUDIT_FAMILIES = ("tree", "dag", "clustered", "generic_force", "weighted", "ported")
# Fable's pre-registered 1.0 bar is deliberately lower than Sol's 1.5 note,
# biasing the structural backstop toward review/block on unknown valves.
TWO_LAYOUT_BUYBACK_BAR = 1.0
OLD_SURVIVOR_CLOSURE_ROWS = (
    ("g6_subsevere_corridor", "weighted"),
    ("clustered_g2_push", "clustered"),
    ("c4_g6_band_edge", "weighted"),
    ("free_clause_dodging", "weighted"),
    ("free_clause_dodging", "clustered"),
    ("eligibility_surface_abuse", "weighted"),
)


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
    objective_mode : str
        SA objective selector. ``"shape"`` preserves the legacy shape-targeted
        attack, ``"tier1_loss"`` retargets the search to T1-only loss under
        the frozen GG-3 2% tiered hold, and ``"g6_damage"`` stresses weighted
        G6 damage inside the degeneracy-escape geometry.
    """

    iterations: int = DEFAULT_ITERATIONS
    restarts: int = DEFAULT_RESTARTS
    aggregate_tolerance_fraction: float = AGGREGATE_TOLERANCE_FRACTION
    shape_distance_threshold: float = SHAPE_DISTANCE_THRESHOLD
    initial_temperature: float = 1.0
    final_temperature: float = 0.02
    objective_mode: str = "shape"


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
    objective_mode : str
        Objective mode used for this attack leg.
    seed : int
        Deterministic SA seed used for this attack leg.
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
    primary_faithfulness_drop : float
        Maximum C1/C3/G6 weighted-KSM score drop, ignoring inapplicable facets.
    sol_variant_blocked : bool
        Whether the same morph blocks under Sol's stricter sensitivity
        thresholds.
    tier1_tradeoff : bool
        Whether W7 flags the morph as an aggregate-held T1 tradeoff.
    gate_verdict : str
        Structured joint GG-3 verdict label.
    tier1_only_drop : float
        Baseline-minus-morph ``tier1_only`` score movement in points.
    max_blockregion_tier1_only_drop : float
        Maximum raw ``tier1_only`` drop encountered among candidates that were
        simultaneously shape-material, within the frozen 2% tiered hold, and
        outside the degeneracy escape.
    blockregion_shape_distance : float
        Shape distance of the candidate that produced
        ``max_blockregion_tier1_only_drop``.
    blockregion_aggregate_delta_fraction : float
        Tiered aggregate drift fraction of the candidate that produced
        ``max_blockregion_tier1_only_drop``.
    blockregion_primary_faithfulness_drop : float
        Primary faithfulness drop of the candidate that produced
        ``max_blockregion_tier1_only_drop``.
    blockregion_two_layout_buyback : float
        Two-layout buyback measured on the same block-region candidate that
        produced ``max_blockregion_tier1_only_drop``.
    blockregion_row_severe_g6_floor_breach : bool
        Whether the same block-region candidate that produced
        ``max_blockregion_tier1_only_drop`` breached the severe-G6 floor.
    blockregion_survives_condition5 : bool
        Whether the best block-region candidate satisfies the frozen
        condition-5 survival predicate without a severe-G6 or degeneracy block.
    blockregion_severe_g6_floor_breach : bool
        Whether any block-region candidate breached the hard G6 floor.
    buyback_headroom : float
        Reporting-only tier2/3 linear buyback corridor,
        ``sum effective_weight * (1 - score)``.
    severe_g6_floor_breach : bool
        Whether a G6 facet crossed the hard pre-registered floor after a
        material baseline-relative drop.
    degenerate_escape : bool
        Whether the morph qualifies for the narrow weighted-degeneracy escape.
    fooled_facets : Tuple[str, ...]
        Facets that stayed high on a blocking morph.
    best_positions : torch.Tensor
        Position tensor with shape ``[N, 2]`` for the best valid morph.
    blockregion_positions : torch.Tensor
        Position tensor with shape ``[N, 2]`` for the best block-region T1 row,
        or the baseline positions when no block-region row was found.
    """

    family: str
    objective_mode: str
    seed: int
    baseline_score: float
    best_score: float
    best_shape_distance: float
    aggregate_delta_fraction: float
    blocked: bool
    primary_faithfulness_drop: float
    sol_variant_blocked: bool
    tier1_tradeoff: bool
    gate_verdict: str
    tier1_only_drop: float
    max_blockregion_tier1_only_drop: float
    blockregion_shape_distance: float
    blockregion_aggregate_delta_fraction: float
    blockregion_primary_faithfulness_drop: float
    blockregion_two_layout_buyback: float
    blockregion_row_severe_g6_floor_breach: bool
    blockregion_survives_condition5: bool
    blockregion_severe_g6_floor_breach: bool
    buyback_headroom: float
    two_layout_buyback: float
    buyback_pass_through: float
    tiered_drop: float
    margin_audit_flag: bool
    margin_audit_margin: float
    margin_audit_allowance: float
    severe_g6_floor_breach: bool
    degenerate_escape: bool
    fooled_facets: Tuple[str, ...]
    gaining_facets: Tuple[str, ...]
    best_positions: torch.Tensor
    blockregion_positions: torch.Tensor


@dataclass(frozen=True)
class AttackTheFixResult:
    """Published result for one attack-the-fix objective/family leg.

    Parameters
    ----------
    objective : str
        Named adversarial objective.
    family : str
        Probe family attacked by the objective.
    seeds_run : int
        Number of independent SA seeds included in the row.
    max_in_band_tier1_drop : float
        Maximum same-row T1-only drop inside the condition-5 shape and tiered
        band before clause/degen disqualification.
    max_buyback : float
        Same-row two-layout buyback for the row that set
        ``max_in_band_tier1_drop``.
    best_shape_distance : float
        Same-row shape distance for the row that set
        ``max_in_band_tier1_drop``.
    severe_g6_floor_breach : bool
        Whether any seed found a severe-G6 floor breach in the block region.
    clause_blocked : bool
        Whether the max-achievable row was blocked by a non-severe
        unconditional clause such as pair-aware material-hold ineligibility.
    survives_condition5 : bool
        Whether any seed found a clause-silent non-degenerate condition-5
        survivor.
    best_seed : int
        Seed for the row that set ``max_in_band_tier1_drop``.
    best_result : AttackResult
        Full per-seed result for artifact and audit use.
    """

    objective: str
    family: str
    seeds_run: int
    max_in_band_tier1_drop: float
    max_buyback: float
    best_shape_distance: float
    severe_g6_floor_breach: bool
    clause_blocked: bool
    survives_condition5: bool
    best_seed: int
    best_result: AttackResult


@dataclass(frozen=True)
class C2AngleRidingCheck:
    """Fixed-vs-archived C2' teeth check outcome.

    Parameters
    ----------
    old_delta : float
        Archived pre-fix C2' delta for two perpendicular crossings versus one
        shallow crossing.
    fixed_delta : float
        Current fixed C2' delta for the same construction.
    added_crossings : int
        Crossing-count increase in the perpendicularized construction.
    old_attack_found : bool
        Whether the archived formula rewards the added-crossing angle ride.
    fixed_attack_loses : bool
        Whether the fixed formula rejects the same added-crossing ride.
    """

    old_delta: float
    fixed_delta: float
    added_crossings: int
    old_attack_found: bool
    fixed_attack_loses: bool


@dataclass(frozen=True)
class CandidateMeasurement:
    """Measured attack-candidate state used for objective and gate tracking.

    Parameters
    ----------
    positions : torch.Tensor
        Candidate positions with shape ``[N, 2]``.
    result : RulerV3Result
        V3 score result for ``positions``.
    score : float
        Candidate tiered score.
    shape_distance : float
        Candidate shape distance from the baseline.
    aggregate_delta_fraction : float
        Absolute tiered aggregate drift fraction from the baseline.
    primary_faithfulness_drop : float
        Maximum primary faithfulness facet drop.
    objective : float
        SA objective value used for search acceptance.
    gate_verdict : GG3GateVerdict
        Per-candidate GG-3 verdict under the frozen gate.
    buyback_headroom : float
        Reporting-only tier2/3 buyback headroom for this candidate.
    """

    positions: torch.Tensor
    result: RulerV3Result
    score: float
    shape_distance: float
    aggregate_delta_fraction: float
    primary_faithfulness_drop: float
    objective: float
    gate_verdict: GG3GateVerdict
    buyback_headroom: float
    two_layout_buyback: float


@dataclass(frozen=True)
class GG3GateVerdict:
    """Joint GG-3 gate verdict and its blocking predicates.

    Parameters
    ----------
    verdict : str
        One of ``BLOCK``, ``PASS_WITH_T1_TRADEOFF``,
        ``PASS_DEGENERATE_ESCAPE``, or ``PASS``.
    blocked : bool
        Whether the joint gate blocks the freeze ceremony.
    tier1_tradeoff : bool
        Whether W7 emitted its score-neutral tradeoff flag.
    tier1_only_drop : float
        Baseline-minus-morph ``tier1_only`` movement in score points.
    severe_g6_floor_breach : bool
        Whether a G6 facet dropped through the hard floor.
    degenerate_escape : bool
        Whether the morph satisfies the narrow baseline-degeneracy waiver.
    shape_changed : bool
        Whether Procrustes shape distance exceeds the materiality threshold.
    aggregate_held : bool
        Whether the tiered aggregate stayed within the 2% GG-3 hold band.
    material_tier1_loss : bool
        Whether primary-facet loss and ``tier1_only`` point loss both exceed
        the pre-registered materiality bars.
    """

    verdict: str
    blocked: bool
    tier1_tradeoff: bool
    tier1_only_drop: float
    severe_g6_floor_breach: bool
    degenerate_escape: bool
    shape_changed: bool
    aggregate_held: bool
    material_tier1_loss: bool
    two_layout_buyback: float
    material_buyback: bool


@dataclass(frozen=True)
class FacetDiff:
    """Facet comparison used by the GG-3 diagnostics report.

    Parameters
    ----------
    code : str
        Facet code.
    baseline_score : Optional[float]
        Baseline facet score, or ``None`` when inapplicable.
    morph_score : Optional[float]
        Morph facet score, or ``None`` when inapplicable.
    baseline_applicable : bool
        Whether the facet applied on the baseline drawing.
    morph_applicable : bool
        Whether the facet applied on the morph drawing.
    weight : float
        Morph-side effective tiered weight used by the attack aggregate.
    drop : Optional[float]
        Baseline minus morph score when both are applicable.
    """

    code: str
    baseline_score: Optional[float]
    morph_score: Optional[float]
    baseline_applicable: bool
    morph_applicable: bool
    weight: float
    drop: Optional[float]


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


def primary_faithfulness_drop(
    baseline_result: RulerV3Result,
    candidate_result: RulerV3Result,
) -> float:
    """Return the largest primary faithfulness drop for GG-3.

    Parameters
    ----------
    baseline_result : RulerV3Result
        Baseline V3 result.
    candidate_result : RulerV3Result
        Candidate V3 result.

    Returns
    -------
    float
        Maximum baseline-minus-candidate score over C1, C3, and G6 weighted
        KSM, ignoring inapplicable or missing facet scores.
    """
    drops: List[float] = []
    for code in PRIMARY_FAITHFULNESS_FACETS:
        baseline = baseline_result.facets.get(code)
        candidate = candidate_result.facets.get(code)
        if baseline is None or candidate is None:
            continue
        if baseline.score is None or candidate.score is None:
            continue
        if not baseline.applicable or not candidate.applicable:
            continue
        drops.append(float(baseline.score) - float(candidate.score))
    return max(drops) if drops else 0.0


def _score_value(result: RulerV3Result, score_name: str) -> float:
    """Return a named composite score as a finite float.

    Parameters
    ----------
    result : RulerV3Result
        V3 result containing composite views.
    score_name : str
        Name of the score to read.

    Returns
    -------
    float
        Composite score value.
    """
    return float(result.scores[score_name])


def _facet_score(result: RulerV3Result, code: str) -> Optional[float]:
    """Return an applicable facet score, or ``None`` when absent.

    Parameters
    ----------
    result : RulerV3Result
        V3 score result.
    code : str
        Facet code to inspect.

    Returns
    -------
    Optional[float]
        Finite score for an applicable facet, otherwise ``None``.
    """
    facet = result.facets.get(code)
    if facet is None or facet.score is None or not facet.applicable:
        return None
    score = float(facet.score)
    return score if math.isfinite(score) else None


def _facet_metadata_float(
    result: RulerV3Result,
    code: str,
    key: str,
    default: float = 0.0,
) -> float:
    """Return a numeric facet metadata field.

    Parameters
    ----------
    result : RulerV3Result
        V3 result to inspect.
    code : str
        Facet code.
    key : str
        Metadata key.
    default : float, optional
        Value returned when the facet or metadata field is absent.

    Returns
    -------
    float
        Finite metadata value, otherwise ``default``.
    """
    facet = result.facets.get(code)
    if facet is None:
        return default
    value = facet.metadata.get(key, default)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _tier1_only_drop(
    baseline_result: RulerV3Result,
    candidate_result: RulerV3Result,
) -> float:
    """Return baseline-minus-candidate ``tier1_only`` score movement.

    Parameters
    ----------
    baseline_result : RulerV3Result
        Baseline V3 result.
    candidate_result : RulerV3Result
        Candidate V3 result.

    Returns
    -------
    float
        Positive values mean T1-only faithfulness dropped.
    """
    return tier1_only_drop(baseline_result, candidate_result)


def _pca_minor_major_ratio(pos: torch.Tensor) -> float:
    """Return the PCA minor/major standard-deviation ratio for positions.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.

    Returns
    -------
    float
        ``sqrt(min_eigenvalue / max_eigenvalue)`` of the mean-centered
        coordinate covariance. Degenerate zero-spread drawings return ``0``.
    """
    points = pos.detach().cpu().numpy().astype(np.float64)
    if points.shape[0] < 2:
        return 0.0
    centered = points - points.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / float(max(1, points.shape[0] - 1))
    eigenvalues = np.linalg.eigvalsh(covariance)
    major = max(float(eigenvalues[-1]), 0.0)
    minor = max(float(eigenvalues[0]), 0.0)
    if major <= 0.0:
        return 0.0
    return math.sqrt(minor / major)


def _minimal_circular_arc_degrees(angles: Sequence[float]) -> float:
    """Return the smallest circular arc covering all angles.

    Parameters
    ----------
    angles : Sequence[float]
        Incident-edge directions in degrees on ``[0, 360)``.

    Returns
    -------
    float
        Width in degrees of the smallest arc containing every direction.
    """
    if len(angles) <= 1:
        return 0.0
    sorted_angles = sorted(float(angle) % 360.0 for angle in angles)
    wrapped = [*sorted_angles, sorted_angles[0] + 360.0]
    largest_gap = max(wrapped[index + 1] - wrapped[index] for index in range(len(sorted_angles)))
    return 360.0 - largest_gap


def _incident_edge_arc_degrees(pos: torch.Tensor, edges: torch.Tensor) -> float:
    """Return the narrowest qualifying high-degree incident-edge arc.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    edges : torch.Tensor
        Edge index tensor with shape ``[2, E]``.

    Returns
    -------
    float
        Minimum arc width among nodes of degree at least three, or
        ``inf`` when no such hub exists.
    """
    count = int(pos.shape[0])
    directions: List[List[float]] = [[] for _index in range(count)]
    edge_pairs = edges.detach().cpu().t().tolist() if edges.numel() else []
    coordinates = pos.detach().cpu().numpy().astype(np.float64)
    for raw_source, raw_target in edge_pairs:
        source = int(raw_source)
        target = int(raw_target)
        delta_st = coordinates[target] - coordinates[source]
        delta_ts = coordinates[source] - coordinates[target]
        if float(np.linalg.norm(delta_st)) > 0.0:
            directions[source].append(math.degrees(math.atan2(delta_st[1], delta_st[0])) % 360.0)
        if float(np.linalg.norm(delta_ts)) > 0.0:
            directions[target].append(math.degrees(math.atan2(delta_ts[1], delta_ts[0])) % 360.0)
    arcs = [
        _minimal_circular_arc_degrees(node_directions)
        for node_directions in directions
        if len(node_directions) >= 3
    ]
    return min(arcs) if arcs else math.inf


def _baseline_shape_degenerate(pos: torch.Tensor, edges: torch.Tensor) -> bool:
    """Return whether baseline geometry has the weighted escape signature.

    Parameters
    ----------
    pos : torch.Tensor
        Baseline positions with shape ``[N, 2]``.
    edges : torch.Tensor
        Edge index tensor with shape ``[2, E]``.

    Returns
    -------
    bool
        ``True`` for a globally near-collinear baseline with a degree-three
        hub whose incident edge rays are compressed into a 10-degree arc.
    """
    # The waiver is intentionally geometric and narrow: a long path can have a
    # tiny PCA ratio, but without a collapsed high-degree hub it is not the
    # weighted star escape that both labs agreed to exempt.
    return (
        _pca_minor_major_ratio(pos) < DEGENERATE_PCA_RATIO
        and _incident_edge_arc_degrees(pos, edges) <= DEGENERATE_HUB_ARC_DEG
    )


def _degenerate_escape(
    baseline_result: RulerV3Result,
    candidate_result: RulerV3Result,
    baseline_pos: torch.Tensor,
    edges: torch.Tensor,
    severe_g6_floor_breach: bool,
) -> bool:
    """Return whether a morph qualifies for the weighted degeneracy waiver.

    Parameters
    ----------
    baseline_result : RulerV3Result
        Baseline V3 result.
    candidate_result : RulerV3Result
        Candidate V3 result.
    baseline_pos : torch.Tensor
        Baseline positions with shape ``[N, 2]``.
    edges : torch.Tensor
        Edge index tensor with shape ``[2, E]``.
    severe_g6_floor_breach : bool
        Precomputed G6 floor breach predicate.

    Returns
    -------
    bool
        ``True`` only for the narrow agreed weighted mis-flag escape.
    """
    baseline_c3 = _facet_score(baseline_result, "C3")
    candidate_c3 = _facet_score(candidate_result, "C3")
    if baseline_c3 is None or candidate_c3 is None:
        return False
    baseline_tier1 = _score_value(baseline_result, "tier1_only")
    candidate_tier1 = _score_value(candidate_result, "tier1_only")
    return (
        _baseline_shape_degenerate(baseline_pos, edges)
        and candidate_tier1 >= baseline_tier1 - DEGENERATE_TIER1_TOLERANCE
        and candidate_c3 - baseline_c3 >= DEGENERATE_C3_IMPROVE
        and not severe_g6_floor_breach
    )


def _gg3_gate_verdict(
    *,
    shape_distance: float,
    aggregate_delta_fraction: float,
    faithfulness_drop: float,
    baseline_result: RulerV3Result,
    candidate_result: RulerV3Result,
    baseline_pos: torch.Tensor,
    edges: torch.Tensor,
) -> GG3GateVerdict:
    """Return the converged joint GG-3 gate verdict.

    Parameters
    ----------
    shape_distance : float
        Normalized Procrustes RMSD.
    aggregate_delta_fraction : float
        Absolute tiered-aggregate drift as a baseline fraction.
    faithfulness_drop : float
        Maximum primary faithfulness facet drop.
    baseline_result : RulerV3Result
        Baseline V3 result.
    candidate_result : RulerV3Result
        Candidate V3 result.
    baseline_pos : torch.Tensor
        Baseline positions with shape ``[N, 2]``.
    edges : torch.Tensor
        Edge index tensor with shape ``[2, E]``.

    Returns
    -------
    GG3GateVerdict
        Structured BLOCK/PASS variant plus the score-neutral W7 flag.
    """
    shape_changed = shape_distance >= SHAPE_DISTANCE_THRESHOLD
    aggregate_held = aggregate_delta_fraction <= GG3_BLOCK_AGGREGATE_DELTA_FRACTION
    tier1_drop = _tier1_only_drop(baseline_result, candidate_result)
    severe_g6_floor_breach = severe_g6_pair_floor_breach(baseline_result, candidate_result)
    degenerate_escape = _degenerate_escape(
        baseline_result,
        candidate_result,
        baseline_pos,
        edges,
        severe_g6_floor_breach,
    )
    material_tier1_loss = (
        faithfulness_drop >= PRIMARY_FAITHFULNESS_DROP_THRESHOLD
        and tier1_drop >= TIER1_ONLY_MATERIALITY_DROP
    )
    buyback = two_layout_buyback_decomposition(baseline_result, candidate_result)["buyback"]
    material_buyback = buyback >= TWO_LAYOUT_BUYBACK_BAR
    pair_eligibility = referee_eligibility_key(
        candidate_result,
        baseline=baseline_result,
        shape_distance=shape_distance,
        aggregate_delta_fraction=aggregate_delta_fraction,
        two_layout_buyback=buyback,
    )
    material_hold_ineligible = pair_eligibility[0] == 0 and not severe_g6_breach(candidate_result)
    blocked = (
        severe_g6_floor_breach
        or (
            shape_changed
            and aggregate_held
            and material_tier1_loss
            and material_buyback
            and not degenerate_escape
        )
        or (material_hold_ineligible and not degenerate_escape)
    )
    tier1_tradeoff = bool(tier1_tradeoff_flags(baseline_result, candidate_result))
    if blocked:
        verdict = GG3_VERDICT_BLOCK
    elif shape_changed and aggregate_held and degenerate_escape:
        verdict = GG3_VERDICT_PASS_DEGENERATE_ESCAPE
    elif (
        shape_changed
        and aggregate_held
        and faithfulness_drop >= PRIMARY_FAITHFULNESS_DROP_THRESHOLD
        and tier1_tradeoff
    ):
        verdict = GG3_VERDICT_PASS_WITH_T1_TRADEOFF
    else:
        verdict = GG3_VERDICT_PASS
    return GG3GateVerdict(
        verdict=verdict,
        blocked=blocked,
        tier1_tradeoff=tier1_tradeoff,
        tier1_only_drop=tier1_drop,
        severe_g6_floor_breach=severe_g6_floor_breach,
        degenerate_escape=degenerate_escape,
        shape_changed=shape_changed,
        aggregate_held=aggregate_held,
        material_tier1_loss=material_tier1_loss,
        two_layout_buyback=buyback,
        material_buyback=material_buyback,
    )


def _sol_variant_blocked(
    *,
    shape_distance: float,
    aggregate_delta_points: float,
    faithfulness_drop: float,
) -> bool:
    """Return whether Sol's GG-3 sensitivity variant blocks.

    Parameters
    ----------
    shape_distance : float
        Normalized Procrustes RMSD.
    aggregate_delta_points : float
        Absolute tiered-aggregate movement in 0--100 score points.
    faithfulness_drop : float
        Maximum primary faithfulness facet drop.

    Returns
    -------
    bool
        Sensitivity-check verdict using 0.08 primary drop and 3.0 tiered-point
        aggregate thresholds.
    """
    return (
        shape_distance >= SHAPE_DISTANCE_THRESHOLD
        and faithfulness_drop >= SOL_PRIMARY_FAITHFULNESS_DROP_THRESHOLD
        and aggregate_delta_points <= SOL_AGGREGATE_DELTA_TIERED_POINTS
    )


def _objective_hold_fraction(attack_config: AttackConfig) -> float:
    """Return the mode-specific aggregate hold band used by the objective.

    Parameters
    ----------
    attack_config : AttackConfig
        Attack thresholds and objective mode.

    Returns
    -------
    float
        Aggregate hold fraction for objective penalties and best-valid
        publishing.
    """
    if attack_config.objective_mode in RETARGETED_OBJECTIVE_MODES:
        return GG3_BLOCK_AGGREGATE_DELTA_FRACTION
    return attack_config.aggregate_tolerance_fraction


def _buyback_headroom(result: RulerV3Result) -> float:
    """Return reporting-only tier2/3 linear buyback headroom.

    Parameters
    ----------
    result : RulerV3Result
        Candidate V3 result.

    Returns
    -------
    float
        Sum of ``effective_weight * (1 - score)`` over applicable tier2/3
        facets. This is intentionally not normalized because the converged
        diagnostic requested the raw linear corridor.
    """
    headroom = 0.0
    for facet in result.facets.values():
        if int(facet.tier) <= 1 or facet.score is None or not bool(facet.applicable):
            continue
        score = min(1.0, max(0.0, float(facet.score)))
        headroom += float(facet.effective_weight) * (1.0 - score)
    return headroom


def _tier1_weight_share(result: RulerV3Result) -> float:
    """Return the Tier-1 share of applicable tiered weight.

    Parameters
    ----------
    result : RulerV3Result
        Baseline V3 result.

    Returns
    -------
    float
        ``W1 / Wtotal`` for applicable tiered facets.
    """
    total = 0.0
    tier1 = 0.0
    for facet in result.facets.values():
        if facet.score is None or not bool(facet.applicable):
            continue
        weight = tier1_measurement_weight(facet)
        total += weight
        if int(facet.tier) == 1:
            tier1 += weight
    return 0.0 if total <= 0.0 else tier1 / total


def two_layout_buyback_decomposition(
    baseline_result: RulerV3Result,
    candidate_result: RulerV3Result,
) -> Dict[str, float]:
    """Return the converged GG-3 two-layout buyback decomposition.

    Parameters
    ----------
    baseline_result : RulerV3Result
        Good-layout baseline score.
    candidate_result : RulerV3Result
        Morph score.

    Returns
    -------
    Dict[str, float]
        ``tier1_drop``, ``tier1_weight_share``, ``pass_through``,
        ``tiered_drop``, and ``buyback`` in 0--100 score points.
    """
    tier1_drop = _tier1_only_drop(baseline_result, candidate_result)
    tiered_drop = _score_value(baseline_result, "tiered") - _score_value(candidate_result, "tiered")
    tier1_weight_share = _tier1_weight_share(baseline_result)
    pass_through = tier1_drop * tier1_weight_share
    # Buyback is the net Tier-2/3 gain needed to hold the tiered score against
    # the linear pass-through loss from Tier 1. It is not the older one-layout
    # "buyback headroom" corridor and must stay baseline-vs-morph differential.
    return {
        "tier1_drop": tier1_drop,
        "tier1_weight_share": tier1_weight_share,
        "pass_through": pass_through,
        "tiered_drop": tiered_drop,
        "buyback": pass_through - tiered_drop,
    }


def _margin_family_for_row(row: Mapping[str, Any]) -> str:
    """Return the GG-3 probe-family bucket for one honest-corpus row.

    Parameters
    ----------
    row : Mapping[str, Any]
        Frozen clean-row payload from ``PHASE1_CLEAN_ROWS_STORE``.

    Returns
    -------
    str
        One of the six battery family labels used by the score-neutral
        margin-audit tripwire.
    """
    graph = str(row.get("graph", "")).lower()
    meta_raw = row.get("graph_meta", {})
    meta = meta_raw if isinstance(meta_raw, Mapping) else {}
    if "ports" in meta or "routing_declared" in meta or "route_paths" in meta or "port" in graph:
        return "ported"
    if "edge_weights" in meta or "weight_mode" in meta or "weighted" in graph:
        return "weighted"
    if bool(meta.get("declared_tree")) or "tree" in graph:
        return "tree"
    if (
        "clusters" in meta
        or "cluster_parents" in meta
        or "planted_partition" in meta
        or "cluster" in graph
        or "community" in graph
    ):
        return "clustered"
    if (
        bool(meta.get("declared_hierarchical"))
        or meta.get("flow_direction") in {"TB", "BT", "LR", "RL"}
        or "dag" in graph
        or "dependency" in graph
        or "citation" in graph
    ):
        return "dag"
    return "generic_force"


def _load_margin_envelopes() -> Tuple[Dict[str, float], str]:
    """Load score-neutral margin audit envelopes from the frozen corpus store.

    Parameters
    ----------
    None

    Returns
    -------
    Tuple[Dict[str, float], str]
        Per-family allowances and a provenance note.
    """
    try:
        payload = json.loads(PHASE1_CLEAN_ROWS_STORE.read_text(encoding="utf-8"))
        rows = payload["rows"]
        by_family: Dict[str, List[float]] = {}
        all_margins: List[float] = []
        for row in rows:
            graph_key = str(row["graph"])
            family = _margin_family_for_row(row)
            tiered = float(row.get("v3_tiered", row["metrics"]["v3_scores"]["tiered"]))
            tier1_only = float(row.get("v3_tier1_only", row["metrics"]["v3_scores"]["tier1_only"]))
            margin = tiered - tier1_only
            by_family.setdefault(graph_key, []).append(margin)
            by_family.setdefault(family, []).append(margin)
            all_margins.append(margin)
        envelopes = {
            family: float(np.percentile(margins, 95)) + 1.0
            for family, margins in by_family.items()
            if margins
        }
        envelopes.update(FAMILY_MARGIN_ALLOWANCES)
        return envelopes, "frozen_corpus_p95_plus_1_with_deramped_family_refreeze"
    except Exception as error:  # pragma: no cover - exercised only when the local store is absent.
        return (
            {"__fallback__": MARGIN_AUDIT_FALLBACK},
            f"NOTE-FOR-REVIEW: margin envelope store unreadable ({error!r}); using fallback",
        )


MARGIN_AUDIT_ENVELOPES, MARGIN_AUDIT_NOTE = _load_margin_envelopes()


def _margin_audit(result: RulerV3Result, family: str) -> Tuple[bool, float, float]:
    """Return the score-neutral margin tripwire state for one row.

    Parameters
    ----------
    result : RulerV3Result
        V3 result to audit.
    family : str
        Family or graph key used to look up ``A_f``.

    Returns
    -------
    Tuple[bool, float, float]
        Flag, observed margin, and allowance.
    """
    margin = _score_value(result, "tiered") - _score_value(result, "tier1_only")
    allowance = MARGIN_AUDIT_ENVELOPES.get(
        family,
        MARGIN_AUDIT_ENVELOPES.get("__fallback__", MARGIN_AUDIT_FALLBACK),
    )
    return margin > allowance, margin, allowance


def _measure_candidate(
    probe: ProbeFamily,
    positions: torch.Tensor,
    *,
    attack_config: AttackConfig,
    score_config: ScoreConfig,
    baseline_result: RulerV3Result,
    baseline_score: float,
) -> CandidateMeasurement:
    """Score and gate one candidate drawing.

    Parameters
    ----------
    probe : ProbeFamily
        Frozen family probe.
    positions : torch.Tensor
        Candidate positions with shape ``[N, 2]``.
    attack_config : AttackConfig
        Attack thresholds and objective mode.
    score_config : ScoreConfig
        Deterministic V3 scoring budget.
    baseline_result : RulerV3Result
        Baseline V3 result for ``probe.pos``.
    baseline_score : float
        Baseline tiered score.

    Returns
    -------
    CandidateMeasurement
        Complete measurement for objective selection and constrained gate
        co-tracking.
    """
    candidate_result = _score_probe(probe, positions, score_config)
    candidate_score = float(candidate_result.scores["tiered"])
    shape_distance = procrustes_shape_distance(probe.pos, positions)
    aggregate_delta = _aggregate_delta_fraction(candidate_score, baseline_score)
    faithfulness_drop = primary_faithfulness_drop(baseline_result, candidate_result)
    objective = _objective(
        shape_distance,
        aggregate_delta,
        faithfulness_drop,
        attack_config,
        baseline_result=baseline_result,
        candidate_result=candidate_result,
        baseline_pos=probe.pos,
        edges=probe.edges,
    )
    gate_verdict = _gg3_gate_verdict(
        shape_distance=shape_distance,
        aggregate_delta_fraction=aggregate_delta,
        faithfulness_drop=faithfulness_drop,
        baseline_result=baseline_result,
        candidate_result=candidate_result,
        baseline_pos=probe.pos,
        edges=probe.edges,
    )
    return CandidateMeasurement(
        positions=positions.detach().clone(),
        result=candidate_result,
        score=candidate_score,
        shape_distance=shape_distance,
        aggregate_delta_fraction=aggregate_delta,
        primary_faithfulness_drop=faithfulness_drop,
        objective=objective,
        gate_verdict=gate_verdict,
        buyback_headroom=_buyback_headroom(candidate_result),
        two_layout_buyback=gate_verdict.two_layout_buyback,
    )


def _is_blockregion_candidate(measurement: CandidateMeasurement) -> bool:
    """Return whether a candidate belongs to the GG-3 BLOCK-eligible region.

    Parameters
    ----------
    measurement : CandidateMeasurement
        Candidate measurement to classify.

    Returns
    -------
    bool
        ``True`` when the candidate is shape-material, inside the frozen 2%
        tiered hold, and outside the degenerate escape.
    """
    return (
        measurement.shape_distance >= SHAPE_DISTANCE_THRESHOLD
        and measurement.aggregate_delta_fraction <= GG3_BLOCK_AGGREGATE_DELTA_FRACTION
        and not measurement.gate_verdict.degenerate_escape
    )


def _blockregion_decision_blocks(
    max_blockregion_tier1_only_drop: float,
    blockregion_material_buyback: bool,
    blockregion_severe_g6_floor_breach: bool,
) -> bool:
    """Return the aggregate GG-3 decision for one attack leg.

    Parameters
    ----------
    max_blockregion_tier1_only_drop : float
        Maximum raw tier1-only drop observed inside the block region.
    blockregion_material_buyback : bool
        Whether any block-region candidate simultaneously met the T1 drop and
        two-layout buyback bars.
    blockregion_severe_g6_floor_breach : bool
        Whether any block-region candidate breached the hard G6 floor.

    Returns
    -------
    bool
        ``True`` when the deciding quantity trips the material T1 bar or the
        severe G6 floor breach trips.
    """
    return (
        max_blockregion_tier1_only_drop >= TIER1_ONLY_MATERIALITY_DROP
        and blockregion_material_buyback
    ) or blockregion_severe_g6_floor_breach


def _survives_condition5(
    baseline_result: RulerV3Result,
    measurement: CandidateMeasurement,
) -> bool:
    """Return whether one measured row survives frozen condition 5.

    Parameters
    ----------
    baseline_result : RulerV3Result
        Baseline V3 result for the attacked probe.
    measurement : CandidateMeasurement
        Same-row candidate measurement to classify.

    Returns
    -------
    bool
        ``True`` only when the row satisfies the shape, 2% tiered band,
        T1-only materiality, two-layout buyback, severe-G6 clause, and
        degenerate-escape predicates exactly as pre-registered.
    """
    return (
        _is_blockregion_candidate(measurement)
        and measurement.gate_verdict.tier1_only_drop >= TIER1_ONLY_MATERIALITY_DROP
        and measurement.two_layout_buyback >= TWO_LAYOUT_BUYBACK_BAR
        and not measurement.gate_verdict.severe_g6_floor_breach
        and not measurement.gate_verdict.degenerate_escape
        and not _pair_material_hold_clause_blocked(baseline_result, measurement)
    )


def _pair_material_hold_clause_blocked(
    baseline_result: RulerV3Result,
    measurement: CandidateMeasurement,
) -> bool:
    """Return whether pair-aware material-hold ineligibility blocks a candidate.

    Parameters
    ----------
    baseline_result : RulerV3Result
        Baseline V3 result for the attacked probe.
    measurement : CandidateMeasurement
        Candidate measurement to classify.

    Returns
    -------
    bool
        ``True`` when the broad pair-aware ineligibility prefix fires while the
        severe-G6 floor does not.
    """
    return not measurement.gate_verdict.severe_g6_floor_breach and material_hold_ineligible(
        baseline_result,
        measurement.result,
        shape_distance=measurement.shape_distance,
        aggregate_delta_fraction=measurement.aggregate_delta_fraction,
        two_layout_buyback=measurement.two_layout_buyback,
    )


def _objective(
    shape_distance: float,
    aggregate_delta_fraction: float,
    faithfulness_drop: float,
    attack_config: AttackConfig,
    *,
    baseline_result: Optional[RulerV3Result] = None,
    candidate_result: Optional[RulerV3Result] = None,
    baseline_pos: Optional[torch.Tensor] = None,
    edges: Optional[torch.Tensor] = None,
) -> float:
    """Return the adversarial SA objective with a hard tolerance penalty.

    Parameters
    ----------
    shape_distance : float
        Normalized Procrustes RMSD.
    aggregate_delta_fraction : float
        Absolute aggregate drift fraction.
    faithfulness_drop : float
        Maximum primary faithfulness score drop.
    attack_config : AttackConfig
        Attack thresholds.
    baseline_result : Optional[RulerV3Result], optional
        Baseline result, required by retargeted objective modes.
    candidate_result : Optional[RulerV3Result], optional
        Candidate result, required by retargeted objective modes.
    baseline_pos : Optional[torch.Tensor], optional
        Baseline positions with shape ``[N, 2]`` for degeneracy gating.
    edges : Optional[torch.Tensor], optional
        Edge index with shape ``[2, E]`` for degeneracy gating.

    Returns
    -------
    float
        Value to maximize during SA.
    """
    mode = attack_config.objective_mode
    if mode not in {"shape", "tier1_loss", "g6_damage", *ATTACK_THE_FIX_OBJECTIVE_MODES}:
        raise ValueError(f"unknown GG-3 objective mode: {mode!r}")
    hold_fraction = _objective_hold_fraction(attack_config)
    excess = aggregate_delta_fraction - hold_fraction
    penalty = 0.0
    if excess > 0.0:
        excess_ratio = excess / max(MIN_BASELINE_SCORE, hold_fraction)
        penalty = OBJECTIVE_PENALTY * (1.0 + excess_ratio * excess_ratio)
    if mode == "tier1_loss":
        if baseline_result is None or candidate_result is None:
            raise ValueError("tier1_loss objective requires baseline_result and candidate_result")
        # This is the deciding GG-3 retarget: attack T1-only loss directly
        # while enforcing the same 2% tiered-hold band used by the BLOCK gate.
        hold_bonus = 0.0 if excess > 0.0 else 0.1 * (1.0 - aggregate_delta_fraction / hold_fraction)
        return _tier1_only_drop(baseline_result, candidate_result) + hold_bonus - penalty
    if mode in ATTACK_THE_FIX_OBJECTIVE_MODES:
        if (
            baseline_result is None
            or candidate_result is None
            or baseline_pos is None
            or edges is None
        ):
            raise ValueError(
                f"{mode} objective requires baseline_result, candidate_result, "
                "baseline_pos, and edges"
            )
        tier1_drop = _tier1_only_drop(baseline_result, candidate_result)
        buyback = two_layout_buyback_decomposition(baseline_result, candidate_result)["buyback"]
        shape_bonus = min(1.0, shape_distance / max(1.0e-12, SHAPE_DISTANCE_THRESHOLD))
        severe_pair = severe_g6_pair_floor_breach(baseline_result, candidate_result)
        degenerate = _degenerate_escape(
            baseline_result,
            candidate_result,
            baseline_pos,
            edges,
            severe_pair,
        )
        clause_penalty = OBJECTIVE_PENALTY if severe_pair else 0.0
        degenerate_penalty = OBJECTIVE_PENALTY if degenerate else 0.0
        material_core = tier1_drop + 0.35 * buyback + 0.5 * shape_bonus
        pair_material = material_hold_ineligible(
            baseline_result,
            candidate_result,
            shape_distance=shape_distance,
            aggregate_delta_fraction=aggregate_delta_fraction,
            two_layout_buyback=buyback,
        )
        declared_weight_contract = sol_declared_weight_subcontract(
            baseline_result,
            candidate_result,
        )
        margin = _score_value(candidate_result, "tiered") - _score_value(
            candidate_result,
            "tier1_only",
        )
        weighted_allowance = FAMILY_MARGIN_ALLOWANCES["weighted"]
        if mode == "g6_subsevere_corridor":
            g6_floor_debt = 0.0
            ramp_edge_bonus = 0.0
            for code in ("G6_weighted_ksm", "G6_local_weight_monotonicity"):
                score = _facet_score(candidate_result, code)
                if score is None:
                    continue
                g6_floor_debt += max(0.0, G6_SEVERE_FLOOR - score)
                if G6_SEVERE_FLOOR <= score < G6_SOFT_FLOOR:
                    ramp_edge_bonus += G6_SOFT_FLOOR - score
            return (
                material_core
                + 8.0 * ramp_edge_bonus
                - 80.0 * g6_floor_debt
                - penalty
                - clause_penalty
                - degenerate_penalty
            )
        if mode == "clustered_g2_push":
            g2_financing = 0.0
            for code in ("C10", "G2_cluster_hac_ari", "G2_cluster_compactness_log_band"):
                baseline_score = _facet_score(baseline_result, code)
                candidate_score = _facet_score(candidate_result, code)
                if baseline_score is not None and candidate_score is not None:
                    g2_financing += max(0.0, candidate_score - baseline_score)
            return (
                material_core + 3.0 * g2_financing - penalty - clause_penalty - degenerate_penalty
            )
        if mode == "c2_perpendicularize":
            baseline_c2 = _facet_score(baseline_result, "C2") or 0.0
            candidate_c2 = _facet_score(candidate_result, "C2") or 0.0
            baseline_count = _facet_metadata_float(baseline_result, "C2", "crossing_count")
            candidate_count = _facet_metadata_float(candidate_result, "C2", "crossing_count")
            added_crossings = max(0.0, candidate_count - baseline_count)
            return (
                12.0 * (candidate_c2 - baseline_c2)
                + 0.5 * added_crossings
                + material_core
                - penalty
                - clause_penalty
                - degenerate_penalty
            )
        if mode == "c4_g6_band_edge":
            c4_score = _facet_score(candidate_result, "C4")
            c4_edge_bonus = 0.0 if c4_score is None else max(0.0, 1.0 - c4_score)
            g6_ramp_bonus = 0.0
            for code in ("G6_weighted_ksm", "G6_local_weight_monotonicity"):
                score = _facet_score(candidate_result, code)
                if score is not None and G6_SEVERE_FLOOR <= score < G6_SOFT_FLOOR:
                    g6_ramp_bonus += G6_SOFT_FLOOR - score
            return (
                material_core
                + 2.0 * c4_edge_bonus
                + 6.0 * g6_ramp_bonus
                - penalty
                - clause_penalty
                - degenerate_penalty
            )
        if mode == "eligibility_surface_abuse":
            eligibility = referee_eligibility_key(candidate_result)
            breach_bonus = 1.0 - float(severe_g6_breach(candidate_result))
            least_breach_bonus = eligibility[1]
            return material_core + breach_bonus + least_breach_bonus - penalty - degenerate_penalty
        if mode == "free_clause_dodging":
            return material_core - penalty - clause_penalty - degenerate_penalty
        if mode == "pool_level_eligibility":
            eligibility = referee_eligibility_key(
                candidate_result,
                baseline=baseline_result,
                shape_distance=shape_distance,
                aggregate_delta_fraction=aggregate_delta_fraction,
                two_layout_buyback=buyback,
            )
            least_breach_bonus = eligibility[1]
            pool_pressure = 2.0 if pair_material else 0.0
            return (
                material_core
                + pool_pressure
                + least_breach_bonus
                + 0.02 * _score_value(candidate_result, "tiered")
                - penalty
                - clause_penalty
                - degenerate_penalty
            )
        if mode == "non_g6_financed_corridor":
            g6_damage = 0.0
            for code in ("G6_weighted_ksm", "G6_local_weight_monotonicity"):
                baseline_score = _facet_score(baseline_result, code)
                candidate_score = _facet_score(candidate_result, code)
                if baseline_score is not None and candidate_score is not None:
                    g6_damage = max(g6_damage, max(0.0, baseline_score - candidate_score))
            non_g6_buyback = buyback + 3.0 * max(0.0, 0.03 - g6_damage)
            return (
                tier1_drop
                + 0.8 * non_g6_buyback
                + 0.5 * shape_bonus
                - 40.0 * max(0.0, g6_damage - 0.03)
                - (OBJECTIVE_PENALTY if declared_weight_contract else 0.0)
                - penalty
                - clause_penalty
                - degenerate_penalty
            )
        if mode == "weighted_margin_tail":
            margin_below_allowance = max(0.0, weighted_allowance - margin)
            margin_above_allowance = max(0.0, margin - weighted_allowance)
            return (
                material_core
                + 2.0 * min(1.0, margin_below_allowance)
                - 10.0 * margin_above_allowance
                - penalty
                - clause_penalty
                - degenerate_penalty
            )
        if mode == "deramped_knife_edge_corridor":
            gap_to_bar = abs(tier1_drop - TIER1_ONLY_MATERIALITY_DROP)
            g6_ramp_bonus = 0.0
            for code in ("G6_weighted_ksm", "G6_local_weight_monotonicity"):
                score = _facet_score(candidate_result, code)
                if score is not None and G6_SEVERE_FLOOR <= score < G6_SOFT_FLOOR:
                    g6_ramp_bonus += G6_SOFT_FLOOR - score
            return (
                tier1_drop
                + 0.5 * buyback
                + 6.0 * g6_ramp_bonus
                + 1.0 / (1.0 + gap_to_bar)
                - penalty
                - clause_penalty
                - degenerate_penalty
            )
        if mode == "clustered_softmin_repush":
            clustered_financing = 0.0
            for code in ("C10", "G2_cluster_hac_ari", "G2_cluster_compactness_log_band"):
                baseline_score = _facet_score(baseline_result, code)
                candidate_score = _facet_score(candidate_result, code)
                if baseline_score is not None and candidate_score is not None:
                    clustered_financing += max(0.0, candidate_score - baseline_score)
            allowance = FAMILY_MARGIN_ALLOWANCES["clustered"]
            softmin_tail = 1.0 / (1.0 + abs(margin - allowance))
            return (
                material_core
                + 4.0 * clustered_financing
                + softmin_tail
                - penalty
                - clause_penalty
                - degenerate_penalty
            )
    if mode == "g6_damage":
        if (
            baseline_result is None
            or candidate_result is None
            or baseline_pos is None
            or edges is None
        ):
            raise ValueError(
                "g6_damage objective requires baseline_result, candidate_result, "
                "baseline_pos, and edges"
            )
        damage = 0.0
        for code in ("G6_weighted_ksm", "G6_local_weight_monotonicity"):
            baseline_score = _facet_score(baseline_result, code)
            candidate_score = _facet_score(candidate_result, code)
            if baseline_score is not None and candidate_score is not None:
                damage = max(damage, baseline_score - candidate_score)
        geometry_penalty = 0.0
        # The G6-damage probe is allowed to search for floor breaches, so this
        # constraint omits the "not severe floor breach" part of the waiver.
        if not _baseline_shape_degenerate(baseline_pos, edges):
            geometry_penalty += OBJECTIVE_PENALTY
        baseline_c3 = _facet_score(baseline_result, "C3")
        candidate_c3 = _facet_score(candidate_result, "C3")
        if baseline_c3 is None or candidate_c3 is None:
            geometry_penalty += OBJECTIVE_PENALTY
        elif candidate_c3 - baseline_c3 < DEGENERATE_C3_IMPROVE:
            geometry_penalty += OBJECTIVE_PENALTY * (
                1.0 + DEGENERATE_C3_IMPROVE - (candidate_c3 - baseline_c3)
            )
        if (
            _score_value(candidate_result, "tier1_only")
            < _score_value(baseline_result, "tier1_only") - DEGENERATE_TIER1_TOLERANCE
        ):
            geometry_penalty += OBJECTIVE_PENALTY
        return damage + 0.05 * shape_distance - penalty - geometry_penalty
    if excess <= 0.0:
        aggregate_hold_bonus = max(
            0.0,
            1.0 - aggregate_delta_fraction / hold_fraction,
        )
        return shape_distance + 0.25 * max(0.0, faithfulness_drop) + 0.1 * aggregate_hold_bonus
    return shape_distance + 0.25 * max(0.0, faithfulness_drop) - penalty


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


def gaining_facets(
    baseline_result: RulerV3Result,
    candidate_result: RulerV3Result,
) -> Tuple[str, ...]:
    """Return facets that gained score on an adversarial candidate.

    Parameters
    ----------
    baseline_result : RulerV3Result
        Good-layout score result.
    candidate_result : RulerV3Result
        Candidate score result.

    Returns
    -------
    Tuple[str, ...]
        Facet codes whose candidate score increased versus baseline.
    """
    baseline = _facet_scores(baseline_result)
    candidate = _facet_scores(candidate_result)
    gained = [
        code
        for code, candidate_score in candidate.items()
        if baseline.get(code) is not None and candidate_score > float(baseline[code]) + 1.0e-12
    ]
    return tuple(sorted(gained))


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
    baseline_gate_verdict = _gg3_gate_verdict(
        shape_distance=0.0,
        aggregate_delta_fraction=0.0,
        faithfulness_drop=0.0,
        baseline_result=baseline_result,
        candidate_result=baseline_result,
        baseline_pos=probe.pos,
        edges=probe.edges,
    )
    baseline_objective = _objective(
        0.0,
        0.0,
        0.0,
        attack_config,
        baseline_result=baseline_result,
        candidate_result=baseline_result,
        baseline_pos=probe.pos,
        edges=probe.edges,
    )
    best_valid = CandidateMeasurement(
        positions=probe.pos.detach().clone(),
        result=baseline_result,
        score=baseline_score,
        shape_distance=0.0,
        aggregate_delta_fraction=0.0,
        primary_faithfulness_drop=0.0,
        objective=baseline_objective,
        gate_verdict=baseline_gate_verdict,
        buyback_headroom=_buyback_headroom(baseline_result),
        two_layout_buyback=baseline_gate_verdict.two_layout_buyback,
    )
    baseline_measurement = best_valid
    best_blockregion: Optional[CandidateMeasurement] = None
    best_blockregion_material_buyback: Optional[CandidateMeasurement] = None
    best_blockregion_severe_g6: Optional[CandidateMeasurement] = None
    blockregion_severe_g6_floor_breach = False
    blockregion_material_buyback = False
    best_valid_hold_fraction = _objective_hold_fraction(attack_config)
    # NOTE-FOR-REVIEW: retargeted modes publish only from the frozen 2% hold
    # band, while shape mode keeps the legacy 5% exploratory tolerance. The
    # separate block-region tracker below is the ceremony decision quantity and
    # deliberately ignores the hold_bonus used only to guide SA search.
    rng = np.random.default_rng(seed)

    for _restart in range(attack_config.restarts):
        current = baseline_measurement
        for step in range(attack_config.iterations):
            temperature = _temperature(step, attack_config)
            proposed_positions = _propose_positions(current.positions, probe.pos, rng, temperature)
            proposed = _measure_candidate(
                probe,
                proposed_positions,
                attack_config=attack_config,
                score_config=score_config,
                baseline_result=baseline_result,
                baseline_score=baseline_score,
            )
            if _is_blockregion_candidate(proposed):
                proposed_severe_g6 = proposed.gate_verdict.severe_g6_floor_breach
                proposed_material_buyback = (
                    proposed.gate_verdict.tier1_only_drop >= TIER1_ONLY_MATERIALITY_DROP
                    and proposed.two_layout_buyback >= TWO_LAYOUT_BUYBACK_BAR
                )
                blockregion_severe_g6_floor_breach = (
                    blockregion_severe_g6_floor_breach or proposed_severe_g6
                )
                blockregion_material_buyback = (
                    blockregion_material_buyback or proposed_material_buyback
                )
                if (
                    best_blockregion is None
                    or proposed.gate_verdict.tier1_only_drop
                    > best_blockregion.gate_verdict.tier1_only_drop
                ):
                    best_blockregion = proposed
                if proposed_material_buyback and (
                    best_blockregion_material_buyback is None
                    or proposed.gate_verdict.tier1_only_drop
                    > best_blockregion_material_buyback.gate_verdict.tier1_only_drop
                ):
                    best_blockregion_material_buyback = proposed
                if proposed_severe_g6 and (
                    best_blockregion_severe_g6 is None
                    or proposed.primary_faithfulness_drop
                    > best_blockregion_severe_g6.primary_faithfulness_drop
                ):
                    best_blockregion_severe_g6 = proposed
            objective_gain = proposed.objective - current.objective
            accept = objective_gain >= 0.0 or rng.random() < math.exp(
                max(-700.0, objective_gain / max(1.0e-12, temperature))
            )
            if accept:
                current = proposed
            if (
                proposed.aggregate_delta_fraction <= best_valid_hold_fraction
                and proposed.objective > best_valid.objective
            ):
                best_valid = proposed
            if (
                current.aggregate_delta_fraction <= best_valid_hold_fraction
                and current.objective > best_valid.objective
            ):
                best_valid = current

    max_blockregion_tier1_only_drop = (
        0.0 if best_blockregion is None else best_blockregion.gate_verdict.tier1_only_drop
    )
    blockregion_shape_distance = (
        0.0 if best_blockregion is None else best_blockregion.shape_distance
    )
    blockregion_aggregate_delta_fraction = (
        0.0 if best_blockregion is None else best_blockregion.aggregate_delta_fraction
    )
    blockregion_primary_faithfulness_drop = (
        0.0 if best_blockregion is None else best_blockregion.primary_faithfulness_drop
    )
    blockregion_two_layout_buyback = (
        0.0 if best_blockregion is None else best_blockregion.two_layout_buyback
    )
    blockregion_row_severe_g6_floor_breach = (
        False if best_blockregion is None else best_blockregion.gate_verdict.severe_g6_floor_breach
    )
    blockregion_survives_condition5 = (
        False
        if best_blockregion is None
        else _survives_condition5(baseline_result, best_blockregion)
    )
    blockregion_positions = (
        probe.pos.detach().clone()
        if best_blockregion is None
        else best_blockregion.positions.detach().clone()
    )
    aggregate_drop_points = max(0.0, baseline_score - best_valid.score)
    gate_verdict = best_valid.gate_verdict
    aggregate_decision_blocked = _blockregion_decision_blocks(
        max_blockregion_tier1_only_drop,
        blockregion_material_buyback,
        blockregion_severe_g6_floor_breach,
    )
    published_verdict = GG3_VERDICT_BLOCK if aggregate_decision_blocked else gate_verdict.verdict
    reporting_candidate = best_valid
    if aggregate_decision_blocked:
        reporting_candidate = (
            best_blockregion_material_buyback
            or best_blockregion_severe_g6
            or best_blockregion
            or best_valid
        )
        gate_verdict = reporting_candidate.gate_verdict
    sol_variant_blocked = _sol_variant_blocked(
        shape_distance=best_valid.shape_distance,
        aggregate_delta_points=aggregate_drop_points,
        faithfulness_drop=best_valid.primary_faithfulness_drop,
    )
    buyback_decomposition = two_layout_buyback_decomposition(
        baseline_result,
        reporting_candidate.result,
    )
    margin_flag, margin, margin_allowance = _margin_audit(reporting_candidate.result, probe.family)
    return AttackResult(
        family=probe.family,
        objective_mode=attack_config.objective_mode,
        seed=seed,
        baseline_score=baseline_score,
        best_score=best_valid.score,
        best_shape_distance=best_valid.shape_distance,
        aggregate_delta_fraction=best_valid.aggregate_delta_fraction,
        blocked=aggregate_decision_blocked,
        primary_faithfulness_drop=best_valid.primary_faithfulness_drop,
        sol_variant_blocked=sol_variant_blocked,
        tier1_tradeoff=gate_verdict.tier1_tradeoff,
        gate_verdict=published_verdict,
        tier1_only_drop=gate_verdict.tier1_only_drop,
        max_blockregion_tier1_only_drop=max_blockregion_tier1_only_drop,
        blockregion_shape_distance=blockregion_shape_distance,
        blockregion_aggregate_delta_fraction=blockregion_aggregate_delta_fraction,
        blockregion_primary_faithfulness_drop=blockregion_primary_faithfulness_drop,
        blockregion_two_layout_buyback=blockregion_two_layout_buyback,
        blockregion_row_severe_g6_floor_breach=blockregion_row_severe_g6_floor_breach,
        blockregion_survives_condition5=blockregion_survives_condition5,
        blockregion_severe_g6_floor_breach=blockregion_severe_g6_floor_breach,
        buyback_headroom=reporting_candidate.buyback_headroom,
        two_layout_buyback=buyback_decomposition["buyback"],
        buyback_pass_through=buyback_decomposition["pass_through"],
        tiered_drop=buyback_decomposition["tiered_drop"],
        margin_audit_flag=margin_flag,
        margin_audit_margin=margin,
        margin_audit_allowance=margin_allowance,
        severe_g6_floor_breach=gate_verdict.severe_g6_floor_breach,
        degenerate_escape=gate_verdict.degenerate_escape,
        fooled_facets=fooled_facets(baseline_result, reporting_candidate.result)
        if aggregate_decision_blocked
        else (),
        gaining_facets=gaining_facets(baseline_result, reporting_candidate.result),
        best_positions=best_valid.positions,
        blockregion_positions=blockregion_positions,
    )


def _jsonable(value: Any) -> Any:
    """Convert scorer metadata to JSON-compatible primitives.

    Parameters
    ----------
    value : Any
        Metadata value emitted by V3 scoring or probe declarations.

    Returns
    -------
    Any
        JSON-serializable representation preserving diagnostic content.
    """
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _tiered_contributions(result: RulerV3Result) -> Dict[str, float]:
    """Return per-facet contributions to the tiered 0--100 composite.

    Parameters
    ----------
    result : RulerV3Result
        Full V3 score result.

    Returns
    -------
    Dict[str, float]
        Tiered contribution keyed by facet code. Contributions sum to
        ``result.scores["tiered"]`` up to floating-point roundoff.
    """
    total_weight = sum(
        float(facet.effective_weight)
        for facet in result.facets.values()
        if facet.score is not None and float(facet.effective_weight) > 0.0
    )
    if total_weight <= 0.0:
        return {code: 0.0 for code in result.facets}
    contributions: Dict[str, float] = {}
    for code, facet in result.facets.items():
        if facet.score is None or float(facet.effective_weight) <= 0.0:
            contributions[code] = 0.0
            continue
        contributions[code] = 100.0 * float(facet.effective_weight) * float(facet.score)
        contributions[code] /= total_weight
    return contributions


def _facet_records(result: RulerV3Result) -> List[Dict[str, Any]]:
    """Return the full diagnostic facet records for one score result.

    Parameters
    ----------
    result : RulerV3Result
        Full V3 score result.

    Returns
    -------
    List[Dict[str, Any]]
        JSON-ready facet records sorted by facet code.
    """
    contributions = _tiered_contributions(result)
    records: List[Dict[str, Any]] = []
    for code in sorted(result.facets):
        facet = result.facets[code]
        records.append(
            {
                "code": facet.code,
                "name": facet.name,
                "score": None if facet.score is None else float(facet.score),
                "applicable": bool(facet.applicable),
                "applicability_reason": facet.applicability_reason,
                "tier": int(facet.tier),
                "base_weight": float(facet.base_weight),
                "effective_weight": float(facet.effective_weight),
                "tiered_contribution": float(contributions[code]),
                "metadata": _jsonable(facet.metadata),
            }
        )
    return records


def _save_probe_artifact(
    path: Path,
    probe: ProbeFamily,
    positions: torch.Tensor,
    *,
    role: str,
    score_result: RulerV3Result,
    shape_distance: float,
) -> None:
    """Save positions and graph inputs needed to re-score or re-render.

    Parameters
    ----------
    path : pathlib.Path
        Output ``.pt`` path.
    probe : ProbeFamily
        Frozen family probe.
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    role : str
        Artifact role, either ``"baseline"`` or ``"morph"``.
    score_result : RulerV3Result
        Score result for ``positions``.
    shape_distance : float
        Procrustes shape distance from the baseline drawing.

    Returns
    -------
    None
    """
    torch.save(
        {
            "family": probe.family,
            "role": role,
            "positions": positions.detach().cpu(),
            "edges": probe.edges.detach().cpu(),
            "sizes": probe.sizes.detach().cpu(),
            "label_sizes": None if probe.label_sizes is None else probe.label_sizes.detach().cpu(),
            "label_offsets": None
            if probe.label_offsets is None
            else probe.label_offsets.detach().cpu(),
            "edge_length_targets": None
            if probe.edge_length_targets is None
            else probe.edge_length_targets.detach().cpu(),
            "meta": _jsonable(probe.meta),
            "scores": dict(score_result.scores),
            "shape_distance": float(shape_distance),
        },
        path,
    )


def _render_comparison_png(
    path: Path,
    probe: ProbeFamily,
    baseline_pos: torch.Tensor,
    morph_pos: torch.Tensor,
    baseline_result: RulerV3Result,
    morph_result: RulerV3Result,
    morph_shape_distance: float,
) -> None:
    """Render a side-by-side baseline-vs-morph comparison PNG.

    Parameters
    ----------
    path : pathlib.Path
        Output PNG path.
    probe : ProbeFamily
        Frozen family probe supplying edges.
    baseline_pos : torch.Tensor
        Baseline positions with shape ``[N, 2]``.
    morph_pos : torch.Tensor
        Morph positions with shape ``[N, 2]``.
    baseline_result : RulerV3Result
        Baseline score result.
    morph_result : RulerV3Result
        Morph score result.
    morph_shape_distance : float
        Procrustes shape distance for the morph.

    Returns
    -------
    None
    """
    import matplotlib.pyplot as plt

    baseline = baseline_pos.detach().cpu().numpy()
    morph = morph_pos.detach().cpu().numpy()
    combined = np.vstack([baseline, morph])
    min_xy = combined.min(axis=0)
    max_xy = combined.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1.0)
    pad = 0.12 * span
    xlim = (float(min_xy[0] - pad[0]), float(max_xy[0] + pad[0]))
    ylim = (float(min_xy[1] - pad[1]), float(max_xy[1] + pad[1]))
    edge_pairs = probe.edges.detach().cpu().t().tolist() if probe.edges.numel() else []

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    panels = (
        ("baseline", baseline, baseline_result, 0.0),
        ("best morph", morph, morph_result, morph_shape_distance),
    )
    for axis, (title, positions, result, shape_distance) in zip(axes, panels):
        for source, target in edge_pairs:
            xs = [positions[int(source), 0], positions[int(target), 0]]
            ys = [positions[int(source), 1], positions[int(target), 1]]
            axis.plot(xs, ys, color="#6b7280", linewidth=1.1, alpha=0.85, zorder=1)
        axis.scatter(
            positions[:, 0],
            positions[:, 1],
            s=80,
            color="#2563eb",
            edgecolors="#111827",
            linewidths=0.8,
            zorder=2,
        )
        for index, (x_value, y_value) in enumerate(positions):
            axis.text(
                float(x_value),
                float(y_value),
                str(index),
                ha="center",
                va="center",
                fontsize=8,
            )
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(*xlim)
        axis.set_ylim(*ylim)
        axis.set_title(f"{title}\ntiered={result.scores['tiered']:.2f}, shape={shape_distance:.3f}")
        axis.grid(True, color="#e5e7eb", linewidth=0.6)
    fig.suptitle(probe.family)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _facet_diffs(
    baseline_result: RulerV3Result,
    morph_result: RulerV3Result,
) -> Tuple[FacetDiff, ...]:
    """Compare baseline and morph facet scores.

    Parameters
    ----------
    baseline_result : RulerV3Result
        Baseline score result.
    morph_result : RulerV3Result
        Morph score result.

    Returns
    -------
    Tuple[FacetDiff, ...]
        Facet comparisons sorted by code.
    """
    diffs: List[FacetDiff] = []
    for code in sorted(set(baseline_result.facets) | set(morph_result.facets)):
        baseline = baseline_result.facets[code]
        morph = morph_result.facets[code]
        baseline_score = None if baseline.score is None else float(baseline.score)
        morph_score = None if morph.score is None else float(morph.score)
        drop = None
        if baseline_score is not None and morph_score is not None:
            drop = baseline_score - morph_score
        diffs.append(
            FacetDiff(
                code=code,
                baseline_score=baseline_score,
                morph_score=morph_score,
                baseline_applicable=bool(baseline.applicable),
                morph_applicable=bool(morph.applicable),
                weight=float(morph.effective_weight),
                drop=drop,
            )
        )
    return tuple(diffs)


def _format_optional_score(value: Optional[float]) -> str:
    """Format an optional facet score for tabular diagnostics.

    Parameters
    ----------
    value : Optional[float]
        Score value, or ``None`` when inapplicable.

    Returns
    -------
    str
        Human-readable score cell.
    """
    return "NA" if value is None else f"{value:.4f}"


def _format_facet_table(diffs: Sequence[FacetDiff]) -> str:
    """Format a baseline-vs-morph facet comparison table.

    Parameters
    ----------
    diffs : Sequence[FacetDiff]
        Per-facet comparison rows.

    Returns
    -------
    str
        Markdown table with score, applicability, and weight columns.
    """
    lines = [
        "| facet | base | morph | drop | base appl | morph appl | morph weight |",
        "|---|---:|---:|---:|---|---|---:|",
    ]
    for diff in diffs:
        drop = "NA" if diff.drop is None else f"{diff.drop:.4f}"
        lines.append(
            f"| {diff.code} | {_format_optional_score(diff.baseline_score)} | "
            f"{_format_optional_score(diff.morph_score)} | {drop} | "
            f"{diff.baseline_applicable} | {diff.morph_applicable} | {diff.weight:.4f} |"
        )
    return "\n".join(lines)


def _decomposition_summary(
    result: AttackResult,
    baseline_result: RulerV3Result,
    morph_result: RulerV3Result,
) -> str:
    """Return the aggregate-hold explanation for one family.

    Parameters
    ----------
    result : AttackResult
        Attack outcome containing aggregate drift and shape distance.
    baseline_result : RulerV3Result
        Baseline score result.
    morph_result : RulerV3Result
        Morph score result.

    Returns
    -------
    str
        One-line decomposition summary.
    """
    diffs = _facet_diffs(baseline_result, morph_result)
    gaining = [diff.code for diff in diffs if diff.drop is not None and diff.drop < 0.0]
    dropped = [
        diff.code for diff in diffs if diff.drop is not None and diff.drop > FOOLED_FACET_MAX_DROP
    ]
    gaining_codes = set(gaining)
    gaining_weight = sum(
        diff.weight for diff in diffs if diff.code in gaining_codes and diff.morph_applicable
    )
    applicability_changes = sum(
        1 for diff in diffs if diff.baseline_applicable != diff.morph_applicable
    )
    gaining_text = ", ".join(gaining) if gaining else "none"
    dropped_text = ", ".join(dropped) if dropped else "none"
    return (
        f"aggregate held within {100.0 * result.aggregate_delta_fraction:.2f}% because "
        f"facets {{{gaining_text}}} (total weight {gaining_weight:.4f}) gained against "
        f"facets {{{dropped_text}}} dropping; buyback={result.two_layout_buyback:.4f}, "
        f"pass-through={result.buyback_pass_through:.4f}; "
        f"{applicability_changes} facets changed applicability."
    )


def _faithfulness_readability_summary(diffs: Sequence[FacetDiff]) -> str:
    """Summarize whether faithfulness dropped while readability held.

    Parameters
    ----------
    diffs : Sequence[FacetDiff]
        Per-facet comparison rows.

    Returns
    -------
    str
        Plain-language C1/C3 versus C2/C4/C5/C6 summary.
    """
    by_code = {diff.code: diff for diff in diffs}
    faithfulness = []
    readability = []
    for code in ("C1", "C3"):
        diff = by_code.get(code)
        if diff is None or diff.drop is None:
            faithfulness.append(f"{code}=NA")
        else:
            faithfulness.append(f"{code} drop {diff.drop:.4f}")
    for code in ("C2", "C4", "C5", "C6"):
        diff = by_code.get(code)
        if diff is None or diff.drop is None or diff.morph_score is None:
            readability.append(f"{code}=NA")
        else:
            held = diff.morph_score >= HIGH_FACET_FLOOR and diff.drop <= FOOLED_FACET_MAX_DROP
            readability.append(f"{code} {'HELD' if held else 'DROPPED'} ({diff.morph_score:.4f})")
    return f"faithfulness: {', '.join(faithfulness)}; readability: {', '.join(readability)}"


def _write_family_diagnostics(
    output_dir: Path,
    probe: ProbeFamily,
    result: AttackResult,
    score_config: ScoreConfig,
) -> Tuple[RulerV3Result, RulerV3Result]:
    """Write diagnostics artifacts for one family.

    Parameters
    ----------
    output_dir : pathlib.Path
        Directory receiving ``.pt``, ``.json``, and ``.png`` artifacts.
    probe : ProbeFamily
        Frozen family probe.
    result : AttackResult
        Attack outcome with best morph positions.
    score_config : ScoreConfig
        Deterministic V3 scoring budget.

    Returns
    -------
    Tuple[RulerV3Result, RulerV3Result]
        Baseline and independently recomputed morph score results.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_result = _score_probe(probe, probe.pos, score_config)
    morph_result = with_tier1_tradeoff_flag(
        baseline_result,
        _score_probe(probe, result.best_positions, score_config),
    )
    morph_shape = procrustes_shape_distance(probe.pos, result.best_positions)
    baseline_path = output_dir / f"{probe.family}_baseline.pt"
    morph_path = output_dir / f"{probe.family}_morph.pt"
    json_path = output_dir / f"{probe.family}_facets.json"
    png_path = output_dir / f"{probe.family}_compare.png"
    _save_probe_artifact(
        baseline_path,
        probe,
        probe.pos,
        role="baseline",
        score_result=baseline_result,
        shape_distance=0.0,
    )
    _save_probe_artifact(
        morph_path,
        probe,
        result.best_positions,
        role="morph",
        score_result=morph_result,
        shape_distance=morph_shape,
    )
    _render_comparison_png(
        png_path,
        probe,
        probe.pos,
        result.best_positions,
        baseline_result,
        morph_result,
        morph_shape,
    )
    payload = {
        "family": probe.family,
        "attack": {
            "objective_mode": result.objective_mode,
            "seed": result.seed,
            "gate_verdict": result.gate_verdict,
            "blocked": result.blocked,
            "tier1_only_drop": result.tier1_only_drop,
            "max_blockregion_tier1_only_drop": result.max_blockregion_tier1_only_drop,
            "blockregion_shape_distance": result.blockregion_shape_distance,
            "blockregion_aggregate_delta_fraction": result.blockregion_aggregate_delta_fraction,
            "blockregion_primary_faithfulness_drop": (result.blockregion_primary_faithfulness_drop),
            "blockregion_two_layout_buyback": result.blockregion_two_layout_buyback,
            "blockregion_row_severe_g6_floor_breach": result.blockregion_row_severe_g6_floor_breach,
            "blockregion_survives_condition5": result.blockregion_survives_condition5,
            "blockregion_severe_g6_floor_breach": result.blockregion_severe_g6_floor_breach,
            "buyback_headroom": result.buyback_headroom,
            "two_layout_buyback": result.two_layout_buyback,
            "buyback_pass_through": result.buyback_pass_through,
            "tiered_drop": result.tiered_drop,
            "gaining_facets": list(result.gaining_facets),
            "margin_audit": {
                "flag": result.margin_audit_flag,
                "margin": result.margin_audit_margin,
                "allowance": result.margin_audit_allowance,
                "source": MARGIN_AUDIT_NOTE,
            },
        },
        "baseline": {
            "scores": dict(baseline_result.scores),
            "coverage": dict(baseline_result.coverage),
            "flags": list(baseline_result.flags),
            "facets": _facet_records(baseline_result),
        },
        "morph": {
            "scores": dict(morph_result.scores),
            "coverage": dict(morph_result.coverage),
            "flags": list(morph_result.flags),
            "shape_distance": morph_shape,
            "facets": _facet_records(morph_result),
        },
        "decomposition": _decomposition_summary(result, baseline_result, morph_result),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return baseline_result, morph_result


def _assert_deterministic_positions(
    probe: ProbeFamily,
    *,
    seed: int,
    attack_config: AttackConfig,
    score_config: ScoreConfig,
    expected: torch.Tensor,
) -> None:
    """Assert same-seed attack replay reproduces the best morph positions.

    Parameters
    ----------
    probe : ProbeFamily
        Frozen family probe.
    seed : int
        Family-specific deterministic attack seed.
    attack_config : AttackConfig
        SA budget and thresholds.
    score_config : ScoreConfig
        Deterministic V3 scoring budget.
    expected : torch.Tensor
        First-run best morph positions with shape ``[N, 2]``.

    Returns
    -------
    None
    """
    replay = run_family_attack(
        probe,
        seed=seed,
        attack_config=attack_config,
        score_config=score_config,
    )
    if not torch.equal(replay.best_positions, expected):
        raise AssertionError(f"{probe.family} same-seed morph positions changed")


def run_diagnostics(
    *,
    seed: int,
    families: Optional[Iterable[str]] = None,
    attack_config: AttackConfig = AttackConfig(),
    score_config: ScoreConfig = ScoreConfig(),
    output_dir: Path = DEFAULT_DIAGNOSTICS_DIR,
) -> Tuple[AttackResult, ...]:
    """Run GG-3 diagnostics and write requested artifacts.

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
    output_dir : pathlib.Path, optional
        Artifact directory.

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
        family_seed = seed + 10_003 * index
        result = run_family_attack(
            probe,
            seed=family_seed,
            attack_config=attack_config,
            score_config=score_config,
        )
        _assert_deterministic_positions(
            probe,
            seed=family_seed,
            attack_config=attack_config,
            score_config=score_config,
            expected=result.best_positions,
        )
        baseline_result, morph_result = _write_family_diagnostics(
            output_dir,
            probe,
            result,
            score_config,
        )
        diffs = _facet_diffs(baseline_result, morph_result)
        independent_delta = abs(float(morph_result.scores["tiered"]) - result.best_score)
        if independent_delta > 1.0e-9:
            raise AssertionError(
                f"{probe.family} tiered score drift: attack={result.best_score}, "
                f"recomputed={morph_result.scores['tiered']}"
            )
        print(f"\n## {probe.family}")
        print(_format_facet_table(diffs))
        print(_decomposition_summary(result, baseline_result, morph_result))
        print(_faithfulness_readability_summary(diffs))
        print(
            "sanity: attack tiered equals independent score_core_v3 recompute "
            f"({result.best_score:.10f})"
        )
        results.append(result)
    return tuple(results)


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


def _required_battery_modes(probe: ProbeFamily) -> Tuple[str, ...]:
    """Return the required GG-3 battery objective modes for one family.

    Parameters
    ----------
    probe : ProbeFamily
        Frozen family probe.

    Returns
    -------
    Tuple[str, ...]
        Objective modes required before a freeze-facing PASS may be reported.
    """
    modes = ["shape", "tier1_loss"]
    if probe.family == "weighted" or _baseline_shape_degenerate(probe.pos, probe.edges):
        modes.append("g6_damage")
    return tuple(modes)


def run_gg3_battery(
    *,
    seed: int,
    families: Optional[Iterable[str]] = None,
    attack_config: AttackConfig = AttackConfig(),
    score_config: ScoreConfig = ScoreConfig(),
    seeds_per_family: int = DEFAULT_BATTERY_SEEDS_PER_FAMILY,
) -> Tuple[AttackResult, ...]:
    """Run the required freeze-facing GG-3 attack battery.

    Parameters
    ----------
    seed : int
        Base deterministic seed.
    families : Optional[Iterable[str]], optional
        Family names to run. Defaults to all frozen probes.
    attack_config : AttackConfig, optional
        SA budget and thresholds used for each leg.
    score_config : ScoreConfig, optional
        Deterministic V3 scoring budget.
    seeds_per_family : int, optional
        Number of independent seeds per family/mode leg. The ceremony schema
        requires at least three so a single lucky retarget cannot certify PASS.

    Returns
    -------
    Tuple[AttackResult, ...]
        Attack outcomes for every required family/mode/seed leg.
    """
    if seeds_per_family < DEFAULT_BATTERY_SEEDS_PER_FAMILY:
        raise ValueError("GG-3 battery requires at least three seeds per family")
    selected = set(families) if families is not None else None
    results: List[AttackResult] = []
    for family_index, probe in enumerate(build_probe_families()):
        if selected is not None and probe.family not in selected:
            continue
        for mode_index, objective_mode in enumerate(_required_battery_modes(probe)):
            mode_config = replace(attack_config, objective_mode=objective_mode)
            for seed_index in range(seeds_per_family):
                # Widely spaced deterministic offsets keep legs reproducible
                # while avoiding accidental seed reuse across family/mode axes.
                family_seed = seed + 10_003 * family_index + 1_000_003 * mode_index
                family_seed += 10_000_019 * seed_index
                results.append(
                    run_family_attack(
                        probe,
                        seed=family_seed,
                        attack_config=mode_config,
                        score_config=score_config,
                    )
                )
    return tuple(results)


def gg3_battery_passed(
    results: Sequence[AttackResult],
    *,
    families: Optional[Iterable[str]] = None,
    seeds_per_family: int = DEFAULT_BATTERY_SEEDS_PER_FAMILY,
) -> bool:
    """Return whether a complete GG-3 battery permits a freeze PASS.

    Parameters
    ----------
    results : Sequence[AttackResult]
        Battery attack outcomes.
    families : Optional[Iterable[str]], optional
        Family names expected in the battery. Defaults to all frozen probes.
    seeds_per_family : int, optional
        Required independent seed count for each family/mode leg.

    Returns
    -------
    bool
        ``True`` only when every required mode/seed is present and no result's
        constrained deciding quantity or severe G6 floor breach blocks.
    """
    selected = set(families) if families is not None else None
    expected: Dict[Tuple[str, str], int] = {}
    for probe in build_probe_families():
        if selected is not None and probe.family not in selected:
            continue
        for mode in _required_battery_modes(probe):
            expected[(probe.family, mode)] = seeds_per_family
    observed: Dict[Tuple[str, str], int] = {}
    for result in results:
        key = (result.family, result.objective_mode)
        observed[key] = observed.get(key, 0) + 1
    complete = all(observed.get(key, 0) >= count for key, count in expected.items())
    return complete and not any(result.blocked for result in results)


def _best_attack_the_fix_leg(results: Sequence[AttackResult]) -> AttackResult:
    """Return the best report row, preferring actual survivors.

    Parameters
    ----------
    results : Sequence[AttackResult]
        Per-seed attack results for one objective/family pair.

    Returns
    -------
    AttackResult
        Best surviving row by T1-only drop when a survivor exists; otherwise
        the best block-region row by T1-only drop.
    """
    if not results:
        raise ValueError("cannot select a best attack-the-fix leg from no results")
    survivor_results = [result for result in results if result.blockregion_survives_condition5]
    candidates = survivor_results or list(results)
    return max(
        candidates,
        key=lambda result: (
            result.max_blockregion_tier1_only_drop,
            result.blockregion_two_layout_buyback,
            result.blockregion_shape_distance,
        ),
    )


def run_attack_the_fix_battery(
    *,
    seed: int,
    attack_config: AttackConfig = AttackConfig(),
    score_config: ScoreConfig = ScoreConfig(),
    seeds_per_family: int = DEFAULT_BATTERY_SEEDS_PER_FAMILY,
) -> Tuple[AttackTheFixResult, ...]:
    """Run the named post-fix adversarial objectives.

    Parameters
    ----------
    seed : int
        Base deterministic seed.
    attack_config : AttackConfig, optional
        SA budget used for each attack leg.
    score_config : ScoreConfig, optional
        Deterministic V3 scoring budget.
    seeds_per_family : int, optional
        Independent seeds per objective/family pair. Must be at least three for
        the pre-registered attack-the-fix round.

    Returns
    -------
    Tuple[AttackTheFixResult, ...]
        Aggregated max-achievable rows for every named objective/family pair.
    """
    if seeds_per_family < DEFAULT_BATTERY_SEEDS_PER_FAMILY:
        raise ValueError("attack-the-fix battery requires at least three seeds per objective")
    families_by_objective = {
        "g6_subsevere_corridor": ("weighted",),
        "clustered_g2_push": ("clustered",),
        "c2_perpendicularize": ("generic_force",),
        "c4_g6_band_edge": ("weighted", "generic_force"),
        "eligibility_surface_abuse": ("weighted",),
        "free_clause_dodging": tuple(probe.family for probe in build_probe_families()),
        "pool_level_eligibility": ("weighted", "generic_force"),
        "non_g6_financed_corridor": ("weighted", "clustered", "generic_force"),
        "weighted_margin_tail": ("weighted",),
        "deramped_knife_edge_corridor": ("weighted",),
        "clustered_softmin_repush": ("clustered",),
    }
    probe_by_name = {probe.family: probe for probe in build_probe_families()}
    aggregated: List[AttackTheFixResult] = []
    for objective_index, objective in enumerate(ATTACK_THE_FIX_OBJECTIVE_MODES):
        for family_index, family in enumerate(families_by_objective[objective]):
            probe = probe_by_name[family]
            legs: List[AttackResult] = []
            intensive = {
                "pool_level_eligibility",
                "deramped_knife_edge_corridor",
                "clustered_softmin_repush",
            }
            mode_config = replace(
                attack_config,
                objective_mode=objective,
                restarts=max(attack_config.restarts, attack_config.restarts + 2)
                if objective in intensive
                else attack_config.restarts,
            )
            for seed_index in range(seeds_per_family):
                family_seed = seed + 2_000_003 * objective_index + 200_003 * family_index
                family_seed += 10_000_019 * seed_index
                legs.append(
                    run_family_attack(
                        probe,
                        seed=family_seed,
                        attack_config=mode_config,
                        score_config=score_config,
                    )
                )
            best = _best_attack_the_fix_leg(legs)
            aggregated.append(
                AttackTheFixResult(
                    objective=objective,
                    family=family,
                    seeds_run=len(legs),
                    max_in_band_tier1_drop=best.max_blockregion_tier1_only_drop,
                    max_buyback=best.blockregion_two_layout_buyback,
                    best_shape_distance=best.blockregion_shape_distance,
                    severe_g6_floor_breach=best.blockregion_row_severe_g6_floor_breach,
                    clause_blocked=(
                        best.max_blockregion_tier1_only_drop >= TIER1_ONLY_MATERIALITY_DROP
                        and best.blockregion_two_layout_buyback >= TWO_LAYOUT_BUYBACK_BAR
                        and not best.blockregion_row_severe_g6_floor_breach
                        and not best.blockregion_survives_condition5
                    ),
                    survives_condition5=any(leg.blockregion_survives_condition5 for leg in legs),
                    best_seed=best.seed,
                    best_result=best,
                )
            )
    return tuple(aggregated)


def format_attack_the_fix_table(results: Sequence[AttackTheFixResult]) -> str:
    """Format the attack-the-fix max-achievable results table.

    Parameters
    ----------
    results : Sequence[AttackTheFixResult]
        Aggregated attack-the-fix rows.

    Returns
    -------
    str
        Markdown table requested by the pre-freeze confidence gate.
    """
    lines = [
        "| objective | family | seeds | max in-band T1 drop | max buyback | "
        "best shape | severe-G6? | clause-blocked? | SURVIVING condition-5 hold? | best seed |",
        "|---|---|---:|---:|---:|---:|---|---|---|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result.objective} | {result.family} | {result.seeds_run} | "
            f"{result.max_in_band_tier1_drop:.4f} | {result.max_buyback:.4f} | "
            f"{result.best_shape_distance:.4f} | "
            f"{'yes' if result.severe_g6_floor_breach else 'no'} | "
            f"{'yes' if result.clause_blocked else 'no'} | "
            f"{'YES' if result.survives_condition5 else 'no'} | {result.best_seed} |"
        )
    return "\n".join(lines)


def _c2_counterexample_layouts() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the archived one-shallow versus two-perpendicular C2' probe.

    Parameters
    ----------
    None

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Edge index, one-shallow layout, and two-perpendicular layout.
    """
    edges = torch.tensor([[0, 2, 4], [1, 3, 5]], dtype=torch.long)
    one_shallow = torch.tensor(
        [
            [0.8353381566711482, -1.0993437673106923],
            [1.3500512486651401, -0.7385221705893695],
            [-1.6376227294973686, 0.8472125848570669],
            [-0.3236051039641243, 1.0049758493947827],
            [-1.7353746109554686, 0.8531447547286918],
            [0.23303430345011902, 1.044852294675133],
        ],
        dtype=torch.float64,
    )
    two_perpendicular = torch.tensor(
        [
            [0.2910327299230801, -0.8581254109036337],
            [0.15620328357839547, 0.8183790749837612],
            [1.017022075537967, 0.37468995055029247],
            [-0.2584253542881093, 0.26551331656227545],
            [0.3173615316656587, -0.7338447788947672],
            [0.17999323883164786, 0.7824427852386611],
        ],
        dtype=torch.float64,
    )
    return edges, one_shallow, two_perpendicular


def _archived_prefix_c2_score(measurement: Mapping[str, float]) -> float:
    """Return the archived pre-fix C2' score for one fixed measurement.

    Parameters
    ----------
    measurement : Mapping[str, float]
        Current C2 measurement dictionary containing crossing cost and capacity.

    Returns
    -------
    float
        Pre-fix score using weighted crossing cost over crossing capacity.
    """
    denominator = max(1.0, float(measurement["crossing_c_max"]))
    cost = float(measurement["crossing_angle_weighted_cost"])
    return max(0.0, min(1.0, 1.0 - math.sqrt(min(1.0, cost / denominator))))


def run_c2_angle_riding_teeth_check() -> C2AngleRidingCheck:
    """Run the old-vs-fixed C2' perpendicularization teeth check.

    Parameters
    ----------
    None

    Returns
    -------
    C2AngleRidingCheck
        Numeric proof that the archived objective wins pre-fix and loses now.
    """
    edges, one_shallow, two_perpendicular = _c2_counterexample_layouts()
    shallow = angle_weighted_crossing_score(one_shallow, edges)
    perpendicular = angle_weighted_crossing_score(two_perpendicular, edges)
    old_delta = _archived_prefix_c2_score(perpendicular) - _archived_prefix_c2_score(shallow)
    fixed_delta = float(perpendicular["edge_crossing_score"]) - float(
        shallow["edge_crossing_score"]
    )
    added_crossings = int(perpendicular["crossing_count"]) - int(shallow["crossing_count"])
    return C2AngleRidingCheck(
        old_delta=old_delta,
        fixed_delta=fixed_delta,
        added_crossings=added_crossings,
        old_attack_found=old_delta > 0.0 and added_crossings > 0,
        fixed_attack_loses=fixed_delta < 0.0 and added_crossings > 0,
    )


def eligibility_surface_abuse_check() -> bool:
    """Return whether synthetic least-breach abuse can beat compliance.

    Parameters
    ----------
    None

    Returns
    -------
    bool
        ``True`` when a breaching high-score candidate can outrank a compliant
        low-score candidate under the exported referee key.
    """
    compliant = _synthetic_key_result(tiered=1.0, g6_weighted_ksm=0.56)
    shallow_breach = _synthetic_key_result(tiered=100.0, g6_weighted_ksm=0.54)
    deep_breach = _synthetic_key_result(tiered=100.0, g6_weighted_ksm=0.30)
    compliant_keyed = (referee_eligibility_key(compliant), compliant.scores["tiered"])
    shallow_keyed = (referee_eligibility_key(shallow_breach), shallow_breach.scores["tiered"])
    deep_keyed = (referee_eligibility_key(deep_breach), deep_breach.scores["tiered"])
    return shallow_keyed > compliant_keyed or deep_keyed > shallow_keyed


def pool_level_eligibility_check() -> bool:
    """Return whether least-breach fallback outranks compliant candidates.

    Parameters
    ----------
    None

    Returns
    -------
    bool
        ``True`` when either a mixed compliant/breaching pool selects a
        breacher, or an all-breach pool fails to select the least-breach row.
    """
    compliant = _synthetic_key_result(tiered=1.0, g6_weighted_ksm=0.56)
    shallow_breach = _synthetic_key_result(tiered=100.0, g6_weighted_ksm=0.54)
    deep_breach = _synthetic_key_result(tiered=200.0, g6_weighted_ksm=0.30)

    mixed_winner = max(
        (compliant, shallow_breach),
        key=lambda result: (referee_eligibility_key(result), result.scores["tiered"]),
    )
    all_breach_winner = max(
        (shallow_breach, deep_breach),
        key=lambda result: (referee_eligibility_key(result), result.scores["tiered"]),
    )
    return mixed_winner is not compliant or all_breach_winner is not shallow_breach


def _synthetic_key_result(*, tiered: float, g6_weighted_ksm: float) -> RulerV3Result:
    """Create a compact weighted result for eligibility-key abuse checks.

    Parameters
    ----------
    tiered : float
        Synthetic tiered score.
    g6_weighted_ksm : float
        Synthetic applicable G6 weighted-KSM facet score.

    Returns
    -------
    RulerV3Result
        Minimal result accepted by the shared severe-G6 helpers.
    """
    facet = RulerV3Facet(
        code="G6_weighted_ksm",
        name="weighted_shortest_path_isotonic_ksm",
        tier=1,
        score=g6_weighted_ksm,
        base_weight=4.0,
        effective_weight=4.0,
        applicable=True,
        applicability_reason="synthetic",
        metadata={},
    )
    return RulerV3Result(
        facets={"G6_weighted_ksm": facet},
        scores={"tiered": tiered, "equal": tiered, "tier1_only": tiered},
        flags=tuple(),
        applicability={"G6_weighted_ksm": True},
        coverage={"applicable_facets": 1, "total_facets": 1, "tier1_applicable_facets": 1},
        metadata={},
    )


def save_attack_the_fix_survivor_artifacts(
    results: Sequence[AttackTheFixResult],
    *,
    output_dir: Path,
    score_config: ScoreConfig = ScoreConfig(),
) -> Tuple[Path, ...]:
    """Save artifacts for every surviving condition-5 hold.

    Parameters
    ----------
    results : Sequence[AttackTheFixResult]
        Aggregated attack-the-fix rows.
    output_dir : pathlib.Path
        Directory receiving survivor artifacts.
    score_config : ScoreConfig, optional
        Deterministic V3 scorer configuration.

    Returns
    -------
    Tuple[pathlib.Path, ...]
        Paths written for survivor artifacts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    probes = {probe.family: probe for probe in build_probe_families()}
    written: List[Path] = []
    for result in results:
        if not result.survives_condition5:
            continue
        probe = probes[result.family]
        baseline_result = _score_probe(probe, probe.pos, score_config)
        morph_result = with_tier1_tradeoff_flag(
            baseline_result,
            _score_probe(probe, result.best_result.blockregion_positions, score_config),
        )
        prefix = f"{result.objective}_{result.family}_{result.best_seed}"
        pt_path = output_dir / f"{prefix}.pt"
        png_path = output_dir / f"{prefix}_compare.png"
        torch.save(
            {
                "objective": result.objective,
                "family": result.family,
                "seed": result.best_seed,
                "positions": result.best_result.blockregion_positions,
                "baseline_positions": probe.pos,
                "edges": probe.edges,
                "scores": dict(morph_result.scores),
                "facets": _facet_records(morph_result),
            },
            pt_path,
        )
        _render_comparison_png(
            png_path,
            probe,
            probe.pos,
            result.best_result.blockregion_positions,
            baseline_result,
            morph_result,
            result.best_shape_distance,
        )
        written.extend((pt_path, png_path))
    return tuple(written)


def write_attack_the_fix_report(
    results: Sequence[AttackTheFixResult],
    *,
    output_path: Path,
    c2_check: C2AngleRidingCheck,
    eligibility_abuse_found: bool,
    pool_abuse_found: bool,
    survivor_artifacts: Sequence[Path],
) -> None:
    """Write the attack-the-fix markdown report.

    Parameters
    ----------
    results : Sequence[AttackTheFixResult]
        Aggregated attack rows.
    output_path : pathlib.Path
        Markdown path to write.
    c2_check : C2AngleRidingCheck
        Old-vs-fixed C2' teeth-check outcome.
    eligibility_abuse_found : bool
        Whether the synthetic eligibility abuse check found a key failure.
    pool_abuse_found : bool
        Whether the synthetic pool-level eligibility check found fallback abuse.
    survivor_artifacts : Sequence[pathlib.Path]
        Artifact paths written for surviving holds.

    Returns
    -------
    None
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ramp_rows = [row for row in results if row.objective == "g6_subsevere_corridor"]
    clustered_rows = [row for row in results if row.objective == "clustered_g2_push"]
    ramp_profitable = any(row.survives_condition5 for row in ramp_rows)
    clustered_crossed = any(row.survives_condition5 for row in clustered_rows)
    max_clustered = max((row.max_in_band_tier1_drop for row in clustered_rows), default=0.0)
    old_rows = [
        row for row in results if (row.objective, row.family) in set(OLD_SURVIVOR_CLOSURE_ROWS)
    ]
    old_survivors_closed = old_rows and not any(row.survives_condition5 for row in old_rows)
    broad_predicate_rows = [
        row
        for row in results
        if row.objective == "non_g6_financed_corridor"
        and row.clause_blocked
        and not row.severe_g6_floor_breach
    ]
    headline = "BARREN" if not any(row.survives_condition5 for row in results) else "NOT BARREN"
    artifact_lines = (
        "\n".join(f"- {path}" for path in survivor_artifacts) if survivor_artifacts else "- none"
    )
    text = "\n\n".join(
        (
            "# Fresh Attack Results",
            f"**Headline:** {headline}.",
            format_attack_the_fix_table(results),
            "## Deciders\n"
            f"- RAMP DECIDER: {'profitable corridor' if ramp_profitable else 'barren'}.\n"
            f"- CLUSTERED DECIDER: {'crossed 5.0' if clustered_crossed else 'did not cross 5.0'} "
            f"(max clustered T1 drop {max_clustered:.4f}).\n"
            f"- OLD-SURVIVOR-CLOSED: {'yes' if old_survivors_closed else 'NO'}.\n"
            f"- BROAD-PREDICATE EVIDENCE: {len(broad_predicate_rows)} non-G6 rows clause-blocked.",
            "## Teeth Checks\n"
            "- E1 weighted severe-G6 hold: checked by archived E1 parity/unit test and severe "
            "path diagnostics.\n"
            f"- C2' pre-fix angle riding old_delta={c2_check.old_delta:.6f}, "
            f"fixed_delta={c2_check.fixed_delta:.6f}, "
            f"added_crossings={c2_check.added_crossings}, "
            f"old_found={'yes' if c2_check.old_attack_found else 'no'}, "
            f"fixed_loses={'yes' if c2_check.fixed_attack_loses else 'no'}.\n"
            "- Eligibility least-breach abuse found: "
            f"{'yes' if eligibility_abuse_found else 'no'}.\n"
            "- Pool-level fallback abuse found: "
            f"{'yes' if pool_abuse_found else 'no'}.",
            f"## Survivor Artifacts\n{artifact_lines}",
        )
    )
    output_path.write_text(text + "\n", encoding="utf-8")


def format_battery_summary(
    results: Sequence[AttackResult],
    *,
    families: Optional[Iterable[str]] = None,
    seeds_per_family: int = DEFAULT_BATTERY_SEEDS_PER_FAMILY,
) -> str:
    """Format a freeze-facing GG-3 battery completeness and decision summary.

    Parameters
    ----------
    results : Sequence[AttackResult]
        Battery attack outcomes.
    families : Optional[Iterable[str]], optional
        Family names expected in the battery. Defaults to all frozen probes.
    seeds_per_family : int, optional
        Required independent seed count for each family/mode leg.

    Returns
    -------
    str
        Markdown table that cannot say PASS when a required mode is missing or
        any constrained deciding quantity trips.
    """
    selected = set(families) if families is not None else None
    by_key: Dict[Tuple[str, str], List[AttackResult]] = {}
    for result in results:
        by_key.setdefault((result.family, result.objective_mode), []).append(result)
    lines = [
        "| family | objective mode | seeds run | max block-region T1 drop | "
        "max two-layout buyback | G6 floor breach | decision |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for probe in build_probe_families():
        if selected is not None and probe.family not in selected:
            continue
        for mode in _required_battery_modes(probe):
            legs = by_key.get((probe.family, mode), [])
            max_drop = max(
                (leg.max_blockregion_tier1_only_drop for leg in legs),
                default=0.0,
            )
            severe = any(leg.blockregion_severe_g6_floor_breach for leg in legs)
            max_buyback = max((leg.blockregion_two_layout_buyback for leg in legs), default=0.0)
            complete = len(legs) >= seeds_per_family
            blocked = any(leg.blocked for leg in legs)
            if not complete:
                decision = "REQUIRED_MISSING"
            elif blocked:
                decision = GG3_VERDICT_BLOCK
            else:
                decision = GG3_VERDICT_PASS
            lines.append(
                f"| {probe.family} | {mode} | {len(legs)} | {max_drop:.4f} | "
                f"{max_buyback:.4f} | {'yes' if severe else 'no'} | {decision} |"
            )
    return "\n".join(lines)


def _fable_verdict(result: AttackResult) -> str:
    """Return the Fable GG-3 verdict label for one attack result.

    Parameters
    ----------
    result : AttackResult
        Per-family attack outcome.

    Returns
    -------
    str
        Structured joint GG-3 verdict label.
    """
    return result.gate_verdict


def _sol_verdict(result: AttackResult) -> str:
    """Return Sol's sensitivity-check GG-3 verdict label.

    Parameters
    ----------
    result : AttackResult
        Per-family attack outcome.

    Returns
    -------
    str
        ``BLOCK``, ``PASS-with-signal``, or ``PASS`` under Sol thresholds.
    """
    if result.sol_variant_blocked:
        return "BLOCK"
    aggregate_drop_points = max(0.0, result.baseline_score - result.best_score)
    signaled = (
        result.best_shape_distance >= SHAPE_DISTANCE_THRESHOLD
        and result.primary_faithfulness_drop >= SOL_PRIMARY_FAITHFULNESS_DROP_THRESHOLD
        and aggregate_drop_points > SOL_AGGREGATE_DELTA_TIERED_POINTS
        and result.aggregate_delta_fraction <= AGGREGATE_TOLERANCE_FRACTION
        and result.tier1_tradeoff
    )
    return "PASS-with-signal" if signaled else "PASS"


def format_results_table(
    results: Sequence[AttackResult],
    *,
    sensitivity_variant: str = "fable",
) -> str:
    """Format the ceremony PASS/BLOCK table.

    Parameters
    ----------
    results : Sequence[AttackResult]
        Per-family attack outcomes.
    sensitivity_variant : str, optional
        Threshold variant to format, either ``"fable"`` or ``"sol"``.

    Returns
    -------
    str
        Markdown table with the required verdict columns.
    """
    if sensitivity_variant not in {"fable", "sol"}:
        raise ValueError(f"unknown GG-3 sensitivity variant: {sensitivity_variant}")
    lines = [
        "| family | objective | seed | best shape dist | primary faith drop | aggregate delta | "
        "tiered delta | T1 drop | max block-region T1 drop | block-region shape | "
        "block-region agg | block-region primary | block-region buyback | two-layout buyback | "
        "buyback pass-through | "
        "margin audit | W7 | verdict | gaining facets |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for result in results:
        verdict = _fable_verdict(result) if sensitivity_variant == "fable" else _sol_verdict(result)
        facets = ", ".join(result.gaining_facets) if result.gaining_facets else "-"
        signed_tiered_drop = result.baseline_score - result.best_score
        margin_text = (
            f"{'yes' if result.margin_audit_flag else 'no'} "
            f"({result.margin_audit_margin:.2f}/{result.margin_audit_allowance:.2f})"
        )
        lines.append(
            f"| {result.family} | {result.objective_mode} | {result.seed} | "
            f"{result.best_shape_distance:.4f} | "
            f"{result.primary_faithfulness_drop:.4f} | "
            f"{100.0 * result.aggregate_delta_fraction:.2f}% | "
            f"{signed_tiered_drop:.2f} | "
            f"{result.tier1_only_drop:.4f} | "
            f"{result.max_blockregion_tier1_only_drop:.4f} | "
            f"{result.blockregion_shape_distance:.4f} | "
            f"{100.0 * result.blockregion_aggregate_delta_fraction:.2f}% | "
            f"{result.blockregion_primary_faithfulness_drop:.4f} | "
            f"{result.blockregion_two_layout_buyback:.4f} | "
            f"{result.two_layout_buyback:.4f} | "
            f"{result.buyback_pass_through:.4f} | "
            f"{margin_text} | "
            f"{'yes' if result.tier1_tradeoff else 'no'} | {verdict} | {facets} |"
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
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="write GG-3 per-facet diagnostics artifacts and print decomposition tables",
    )
    parser.add_argument(
        "--diagnostics-dir",
        type=Path,
        default=DEFAULT_DIAGNOSTICS_DIR,
        help="directory for diagnostics .pt, .json, and .png artifacts",
    )
    parser.add_argument(
        "--objective-mode",
        choices=("battery", "attack_the_fix", "shape", "tier1_loss", "g6_damage")
        + ATTACK_THE_FIX_OBJECTIVE_MODES,
        default="battery",
        help="SA objective mode; default runs the required freeze-facing GG-3 battery",
    )
    parser.add_argument(
        "--battery-seeds",
        type=int,
        default=DEFAULT_BATTERY_SEEDS_PER_FAMILY,
        help="independent seeds per family/mode leg for --objective-mode battery",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=DEFAULT_ATTACK_THE_FIX_RESULTS_PATH,
        help="markdown output path for --objective-mode attack_the_fix",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=DEFAULT_ATTACK_THE_FIX_ARTIFACT_DIR,
        help="surviving-hold artifact directory for --objective-mode attack_the_fix",
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
    objective_mode = str(args.objective_mode)
    if objective_mode in {"battery", "attack_the_fix"} and bool(args.diagnostics):
        raise ValueError(
            "aggregate diagnostics would overwrite per-family artifacts; use a single mode"
        )
    attack_config = AttackConfig(
        iterations=int(args.iterations),
        restarts=int(args.restarts),
        aggregate_tolerance_fraction=float(args.aggregate_tolerance),
        shape_distance_threshold=float(args.shape_threshold),
        objective_mode=(
            "shape" if objective_mode in {"battery", "attack_the_fix"} else objective_mode
        ),
    )
    if objective_mode == "battery":
        results = run_gg3_battery(
            seed=int(args.seed),
            families=args.family,
            attack_config=attack_config,
            score_config=ScoreConfig(),
            seeds_per_family=int(args.battery_seeds),
        )
    elif objective_mode == "attack_the_fix":
        attack_results = run_attack_the_fix_battery(
            seed=int(args.seed),
            attack_config=attack_config,
            score_config=ScoreConfig(),
            seeds_per_family=int(args.battery_seeds),
        )
        c2_check = run_c2_angle_riding_teeth_check()
        eligibility_abuse_found = eligibility_surface_abuse_check()
        pool_abuse_found = pool_level_eligibility_check()
        artifacts = save_attack_the_fix_survivor_artifacts(
            attack_results,
            output_dir=Path(args.artifact_dir),
            score_config=ScoreConfig(),
        )
        write_attack_the_fix_report(
            attack_results,
            output_path=Path(args.results_path),
            c2_check=c2_check,
            eligibility_abuse_found=eligibility_abuse_found,
            pool_abuse_found=pool_abuse_found,
            survivor_artifacts=artifacts,
        )
        print("ATTACK-THE-FIX results")
        print(format_attack_the_fix_table(attack_results))
        print()
        print(
            "teeth: "
            f"C2_old_delta={c2_check.old_delta:.6f}, "
            f"C2_fixed_delta={c2_check.fixed_delta:.6f}, "
            f"old_found={'yes' if c2_check.old_attack_found else 'no'}, "
            f"fixed_loses={'yes' if c2_check.fixed_attack_loses else 'no'}, "
            f"eligibility_abuse={'yes' if eligibility_abuse_found else 'no'}, "
            f"pool_abuse={'yes' if pool_abuse_found else 'no'}"
        )
        print(f"results_path={Path(args.results_path)}")
        print(f"survivor_artifacts={len(artifacts)}")
        return 1 if any(result.survives_condition5 for result in attack_results) else 0
    elif bool(args.diagnostics):
        results = run_diagnostics(
            seed=int(args.seed),
            families=args.family,
            attack_config=attack_config,
            score_config=ScoreConfig(),
            output_dir=Path(args.diagnostics_dir),
        )
    else:
        results = run_all_attacks(
            seed=int(args.seed),
            families=args.family,
            attack_config=attack_config,
            score_config=ScoreConfig(),
        )
    if objective_mode == "battery":
        print("GG-3 required battery summary")
        print(
            format_battery_summary(
                results,
                families=args.family,
                seeds_per_family=int(args.battery_seeds),
            )
        )
        print()
    print("Fable GG-3 criterion")
    print(format_results_table(results, sensitivity_variant="fable"))
    print()
    print("Sol GG-3 sensitivity variant")
    print(format_results_table(results, sensitivity_variant="sol"))
    print()
    print(
        "thresholds: "
        f"search_aggregate_tolerance={attack_config.aggregate_tolerance_fraction:.2%}, "
        f"block_aggregate_tolerance={GG3_BLOCK_AGGREGATE_DELTA_FRACTION:.2%}, "
        f"primary_faithfulness_drop={PRIMARY_FAITHFULNESS_DROP_THRESHOLD:.2f}, "
        f"sol_primary_faithfulness_drop={SOL_PRIMARY_FAITHFULNESS_DROP_THRESHOLD:.2f}, "
        f"sol_aggregate_delta={SOL_AGGREGATE_DELTA_TIERED_POINTS:.1f}pt, "
        f"shape_distance={attack_config.shape_distance_threshold:.2f}, "
        f"objective_mode={objective_mode}, "
        f"battery_seeds={int(args.battery_seeds)}, "
        f"budget={attack_config.restarts}x{attack_config.iterations}"
    )
    if objective_mode == "battery":
        passed = gg3_battery_passed(
            results,
            families=args.family,
            seeds_per_family=int(args.battery_seeds),
        )
        return 0 if passed else 1
    return 1 if any(result.blocked for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
