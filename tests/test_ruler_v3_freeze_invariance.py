"""Freeze-ceremony pure-function gates for the V3 ruler."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pytest
import torch

import dagua.eval.ruler_v3 as ruler_v3
import dagua.eval.ruler_v3_groups as ruler_v3_groups
from dagua.eval.ruler_v3 import score_core_v3, whitespace_sprawl_score
from dagua.eval.ruler_v3_frozen import FROZEN_CONSTANTS
from dagua.eval.ruler_v3_groups import CANONICAL_NODE_HEIGHT_REF

ALPHAS = (0.02, 0.1, 1.0, 10.0, 50.0)
PURE_POSITION_FACETS = {
    "C1",
    "C2",
    "C3",
    "C6",
    "C7",
    "C8",
    "C9",
    "C10",
    "G1_directed_flow",
    "G1_depth_order",
    "G3_community_ari",
    "G4_layered_depth_layering",
    "G4_layered_parent_centering",
    "G4_layered_sibling_order",
    "G4_layered_subtree_congruence",
    "G4_layered_small_subtree_distribution",
    "G4_layered_width_ratio",
    "G4_radial_depth_monotonicity",
    "G4_radial_angular_allocation",
    "G4_radial_angular_overlap",
    "G4_radial_circular_order",
    "G4_radial_subtree_congruence",
    "G4_radial_small_subtree_distribution",
    "G4_radial_area_ratio",
    "G6_weighted_ksm",
    "G6_local_weight_monotonicity",
}
SIZE_SENSITIVE_FACETS = {
    "C4",
    "C5",
    "G2_cluster_exclusion",
    "G2_cluster_sibling_overlap",
    "G2_cluster_nesting_fidelity",
    "G2_cluster_edge_intrusion",
    "G2_cluster_label_occlusion",
    "G2_cluster_compactness_log_band",
    "G4_layered_contour_separation",
    "G7_port_hard_compliance",
    "G7_routed_curve_quality",
    "G7_routed_bend_terminal_economy",
}
LOW_INFORMATION_DEFORMATION_FACETS = {
    "G2_cluster_nesting_fidelity",
}


@dataclass(frozen=True)
class Probe:
    """Input bundle for one pure-function ruler probe.

    Parameters
    ----------
    name : str
        Stable probe name used in matrix output.
    pos : torch.Tensor
        Node positions with shape ``[N, 2]``.
    edges : torch.Tensor
        Edge index with shape ``[2, E]``.
    sizes : torch.Tensor
        Node sizes with shape ``[N, 2]``.
    meta : Mapping[str, Any]
        Declared input metadata.
    label_sizes : Optional[torch.Tensor]
        Optional label sizes with shape ``[N, 2]``.
    label_offsets : Optional[torch.Tensor]
        Optional label offsets with shape ``[N, 2]``.
    """

    name: str
    pos: torch.Tensor
    edges: torch.Tensor
    sizes: torch.Tensor
    meta: Mapping[str, Any]
    label_sizes: Optional[torch.Tensor] = None
    label_offsets: Optional[torch.Tensor] = None


def _sizes(count: int, height: float = 1.0) -> torch.Tensor:
    """Return uniform probe node boxes.

    Parameters
    ----------
    count : int
        Number of nodes.
    height : float, optional
        Box height and width.

    Returns
    -------
    torch.Tensor
        Node sizes with shape ``[N, 2]``.
    """
    return torch.full((count, 2), height, dtype=torch.float64)


def _score(probe: Probe) -> Mapping[str, Any]:
    """Score one probe with small deterministic budgets.

    Parameters
    ----------
    probe : Probe
        Probe input bundle.

    Returns
    -------
    Mapping[str, Any]
        Facet records keyed by code.
    """
    return score_core_v3(
        probe.pos,
        probe.edges,
        probe.sizes,
        label_sizes=probe.label_sizes,
        label_offsets=probe.label_offsets,
        graph_meta=probe.meta,
        crossing_samples=100_000,
        neighborhood_samples=256,
        stress_sources=64,
        stress_targets=256,
    ).facets


def _scaled(probe: Probe, alpha: float, *, positions_only: bool) -> Probe:
    """Return a unit-scaled or position-only-scaled copy of a probe.

    Parameters
    ----------
    probe : Probe
        Baseline probe.
    alpha : float
        Scale factor.
    positions_only : bool
        Whether to leave node sizes and label extents fixed.

    Returns
    -------
    Probe
        Scaled probe.
    """
    scale_sizes = 1.0 if positions_only else alpha
    return Probe(
        name=f"{probe.name}@{alpha}",
        pos=probe.pos * alpha,
        edges=probe.edges,
        sizes=probe.sizes * scale_sizes,
        meta=_scale_metadata(probe.meta, alpha, scale_geometry=not positions_only),
        label_sizes=None if probe.label_sizes is None else probe.label_sizes * scale_sizes,
        label_offsets=None if probe.label_offsets is None else probe.label_offsets * alpha,
    )


def _scale_metadata(
    meta: Mapping[str, Any],
    alpha: float,
    *,
    scale_geometry: bool,
) -> Mapping[str, Any]:
    """Scale geometric metadata used by G5/G7 probes.

    Parameters
    ----------
    meta : Mapping[str, Any]
        Declared metadata.
    alpha : float
        Position scale factor.
    scale_geometry : bool
        Whether route and previous-frame coordinates should scale as unit geometry.

    Returns
    -------
    Mapping[str, Any]
        Scaled metadata.
    """
    if not scale_geometry:
        return dict(meta)
    scaled = dict(meta)
    if "route_paths" in scaled:
        scaled["route_paths"] = [
            [(alpha * float(x_value), alpha * float(y_value)) for x_value, y_value in route]
            for route in scaled["route_paths"]
        ]
    if "label_positions" in scaled:
        scaled["label_positions"] = [
            (alpha * float(point[0]), alpha * float(point[1])) if point is not None else None
            for point in scaled["label_positions"]
        ]
    previous = scaled.get("previous")
    if isinstance(previous, Mapping) and "positions" in previous:
        prev = dict(previous)
        prev["positions"] = [
            [alpha * float(point[0]), alpha * float(point[1])] for point in prev["positions"]
        ]
        scaled["previous"] = prev
    return scaled


def _facet_value(facets: Mapping[str, Any], code: str) -> float:
    """Return a finite facet score.

    Parameters
    ----------
    facets : Mapping[str, Any]
        Facet records keyed by code.
    code : str
        Facet code.

    Returns
    -------
    float
        Facet value.
    """
    score = facets[code].score
    assert score is not None
    return float(score)


def _assert_exactish(base: float, scaled: float) -> None:
    """Assert exact or pinned tiny-relative equality.

    Parameters
    ----------
    base : float
        Baseline score.
    scaled : float
        Scaled score.

    Returns
    -------
    None
    """
    if scaled == base:
        return
    assert abs(scaled - base) <= 1e-12 * max(1.0, abs(base))


def _print_matrix(title: str, rows: Sequence[Tuple[str, str, str]]) -> None:
    """Print a compact PASS/FAIL matrix for freeze evidence.

    Parameters
    ----------
    title : str
        Matrix title.
    rows : Sequence[Tuple[str, str, str]]
        Matrix rows as ``(facet, probe, status)``.

    Returns
    -------
    None
    """
    print(title)
    for facet, probe, status in rows:
        print(f"{facet:40s} {probe:24s} {status}")


def _all_freeze_probes() -> Tuple[Probe, ...]:
    """Return probes that cover every C1-C10 and G1-G7 facet.

    Parameters
    ----------
    None

    Returns
    -------
    Tuple[Probe, ...]
        Probe set.
    """
    return (
        _core_probe(),
        _g1_probe(0),
        _cluster_probe(),
        _community_probe(),
        _tree_probe(radial=False),
        _tree_probe(radial=True),
        _temporal_probe(current_shift=0.2, previous_quality=0.98, graph_change=0.042),
        _weighted_probe([1.0, 2.0, 4.0, 8.0]),
        _ported_probe(routed=True),
    )


def _core_probe() -> Probe:
    """Create a core-facet probe with label geometry.

    Parameters
    ----------
    None

    Returns
    -------
    Probe
        Core probe.
    """
    pos = torch.tensor(
        [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0], [8.0, 4.0], [12.0, 4.0]],
        dtype=torch.float64,
    )
    edges = torch.tensor([[0, 1, 2, 3, 0, 1, 3, 4], [1, 2, 3, 0, 2, 3, 4, 5]])
    labels = torch.full((6, 2), 0.4, dtype=torch.float64)
    offsets = torch.tensor([[0.0, 0.7]] * 6, dtype=torch.float64)
    return Probe("core", pos, edges.to(torch.long), _sizes(6, 0.5), {}, labels, offsets)


def _cluster_probe(scale: float = 1.0) -> Probe:
    """Create a declared-cluster probe.

    Parameters
    ----------
    scale : float, optional
        Position scale factor.

    Returns
    -------
    Probe
        Cluster probe.
    """
    pos = scale * torch.tensor(
        [[-2.0, -0.2], [-2.2, 0.2], [-1.8, 0.0], [2.0, -0.2], [2.2, 0.2], [1.8, 0.0]],
        dtype=torch.float64,
    )
    edges = torch.tensor([[0, 1, 3, 4], [1, 2, 4, 5]], dtype=torch.long)
    meta = {
        "clusters": {"left": [0, 1, 2], "right": [3, 4, 5]},
        "cluster_labels": {"left": "Left", "right": "Right"},
    }
    return Probe("cluster", pos, edges, _sizes(6, 0.2), meta)


def _community_probe() -> Probe:
    """Create a planted-community probe.

    Parameters
    ----------
    None

    Returns
    -------
    Probe
        Community probe.
    """
    pos = torch.tensor(
        [[-3.0, -0.2], [-3.1, 0.1], [-2.8, 0.0], [3.0, -0.1], [3.2, 0.2], [2.9, 0.1]],
        dtype=torch.float64,
    )
    edges = torch.tensor([[0, 1, 3, 4], [1, 2, 4, 5]], dtype=torch.long)
    return Probe("community", pos, edges, _sizes(6, 0.2), {"planted_partition": [0, 0, 0, 1, 1, 1]})


def _tree_probe(*, radial: bool) -> Probe:
    """Create a declared rooted-tree probe.

    Parameters
    ----------
    radial : bool
        Whether to use radial tree convention.

    Returns
    -------
    Probe
        Tree probe.
    """
    edges = torch.tensor([[0, 0, 0, 3, 3], [1, 2, 3, 4, 5]], dtype=torch.long)
    if radial:
        coords = [(0.0, 0.0), (1.5, 1.2), (0.0, 2.0), (-1.5, 1.2), (-1.8, 2.2), (-2.4, 1.8)]
        convention = "radial"
        name = "tree_radial"
    else:
        coords = [(0.0, 0.0), (-2.0, 1.0), (0.0, 1.0), (2.0, 1.0), (1.5, 2.0), (2.5, 2.0)]
        convention = "layered"
        name = "tree_layered"
    return Probe(
        name,
        torch.tensor(coords, dtype=torch.float64),
        edges,
        _sizes(6, 0.2),
        {"declared_tree": True, "root": 0, "tree_convention": convention},
    )


def _temporal_probe(
    *,
    current_shift: float,
    previous_quality: float,
    graph_change: float,
) -> Probe:
    """Create a declared temporal probe.

    Parameters
    ----------
    current_shift : float
        Current-frame displacement applied to one node.
    previous_quality : float
        Declared previous static quality.
    graph_change : float
        Declared graph-change magnitude.

    Returns
    -------
    Probe
        Temporal probe.
    """
    previous = torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]])
    current = previous.clone().to(dtype=torch.float64)
    current[3, 0] += current_shift
    edges = torch.tensor([[0, 1, 2], [1, 3, 3]], dtype=torch.long)
    meta = {
        "node_ids": ["a", "b", "c", "d"],
        "previous": {
            "positions": previous.tolist(),
            "node_ids": ["a", "b", "c", "d"],
            "quality": previous_quality,
            "best_static_v3_core": 1.0,
            "graph_change": graph_change,
        },
    }
    return Probe("temporal", current, edges, _sizes(4, 0.2), meta)


def _weighted_probe(lengths: Sequence[float]) -> Probe:
    """Create a declared weighted-distance star probe.

    Parameters
    ----------
    lengths : Sequence[float]
        Leaf distances from the center.

    Returns
    -------
    Probe
        Weighted probe.
    """
    coords = [(0.0, 0.0)] + [
        (float(length), float(index) * 0.05) for index, length in enumerate(lengths)
    ]
    edges = torch.tensor([[0 for _ in lengths], list(range(1, len(lengths) + 1))], dtype=torch.long)
    meta = {"edge_weights": [1.0, 2.0, 4.0, 8.0], "weight_mode": "distance"}
    return Probe(
        "weighted", torch.tensor(coords, dtype=torch.float64), edges, _sizes(len(coords), 0.2), meta
    )


def _ported_probe(*, routed: bool, wrong_side: bool = False) -> Probe:
    """Create a declared port/routing probe.

    Parameters
    ----------
    routed : bool
        Whether to include route paths.
    wrong_side : bool, optional
        Whether to declare the source side incorrectly.

    Returns
    -------
    Probe
        Ported probe.
    """
    pos = torch.tensor([[0.0, 0.0], [4.0, -1.0], [4.0, 1.0], [8.0, 0.0]], dtype=torch.float64)
    edges = torch.tensor([[0, 0, 1, 2], [1, 2, 3, 3]], dtype=torch.long)
    side = "W" if wrong_side else "E"
    ports: List[Dict[str, object]] = [
        {"edge": 0, "endpoint": "source", "side": side, "order": 0},
        {"edge": 1, "endpoint": "source", "side": side, "order": 1},
        {"edge": 0, "endpoint": "target", "side": "W"},
        {"edge": 1, "endpoint": "target", "side": "W"},
        {"edge": 2, "endpoint": "source", "side": "E"},
        {"edge": 3, "endpoint": "source", "side": "E"},
        {"edge": 2, "endpoint": "target", "side": "W", "order": 0},
        {"edge": 3, "endpoint": "target", "side": "W", "order": 1},
    ]
    meta: Dict[str, object] = {"ports": ports, "flow_direction": "LR"}
    if routed:
        meta["routing_declared"] = True
        meta["route_paths"] = [
            [(0.0, 0.0), (2.0, -1.0), (4.0, -1.0)],
            [(0.0, 0.0), (2.0, 1.0), (4.0, 1.0)],
            [(4.0, -1.0), (6.0, -1.0), (8.0, 0.0)],
            [(4.0, 1.0), (6.0, 1.0), (8.0, 0.0)],
        ]
        meta["routed_labels"] = ["a", "b", "c", "d"]
        meta["label_positions"] = [(2.0, -1.0), (2.0, 1.0), (6.0, -1.0), (6.0, 1.0)]
    return Probe("ported", pos, edges, _sizes(4, CANONICAL_NODE_HEIGHT_REF), meta)


def test_gg5_unit_invariance_all_facets(capsys: pytest.CaptureFixture[str]) -> None:
    """Assert unit invariance for all applicable core and group facets.

    Parameters
    ----------
    capsys : pytest.CaptureFixture[str]
        Pytest capture fixture used only to keep matrix output available.

    Returns
    -------
    None
    """
    del capsys
    rows: List[Tuple[str, str, str]] = []
    seen: set[str] = set()
    for probe in _all_freeze_probes():
        base = _score(probe)
        for code, facet in base.items():
            if not facet.applicable or facet.score is None:
                continue
            seen.add(code)
            base_score = float(facet.score)
            for alpha in ALPHAS:
                scaled = _score(_scaled(probe, alpha, positions_only=False))
                _assert_exactish(base_score, _facet_value(scaled, code))
            rows.append((code, probe.name, "PASS"))
    _print_matrix("GG-5a unit invariance matrix", sorted(rows))
    assert {"C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10"}.issubset(seen)
    assert any(code.startswith("G1_") for code in seen)
    assert any(code.startswith("G2_") for code in seen)
    assert any(code.startswith("G3_") for code in seen)
    assert any(code.startswith("G4_") for code in seen)
    assert any(code.startswith("G5_") for code in seen)
    assert any(code.startswith("G6_") for code in seen)
    assert any(code.startswith("G7_") for code in seen)


def test_gg5_position_only_scale_contract(capsys: pytest.CaptureFixture[str]) -> None:
    """Assert pure geometry is invariant and size-anchored facets move.

    Parameters
    ----------
    capsys : pytest.CaptureFixture[str]
        Pytest capture fixture used only to keep matrix output available.

    Returns
    -------
    None
    """
    del capsys
    rows: List[Tuple[str, str, str]] = []
    moved: set[str] = set()
    invariant: set[str] = set()
    for probe in _all_freeze_probes():
        base = _score(probe)
        for code, facet in base.items():
            if not facet.applicable or facet.score is None:
                continue
            base_score = float(facet.score)
            values = [
                _facet_value(_score(_scaled(probe, alpha, positions_only=True)), code)
                for alpha in ALPHAS
            ]
            if code in PURE_POSITION_FACETS:
                for value in values:
                    _assert_exactish(base_score, value)
                invariant.add(code)
                rows.append((code, probe.name, "PASS invariant"))
            elif code in SIZE_SENSITIVE_FACETS and any(
                abs(value - base_score) > 1e-9 for value in values
            ):
                moved.add(code)
                rows.append((code, probe.name, "PASS sensitive"))
    _print_matrix("GG-5b position-only scale matrix", sorted(rows))
    assert {"C1", "C2", "C3", "C6", "C7", "C8", "C9", "C10"}.issubset(invariant)
    assert {"C4", "C5"}.issubset(moved)
    assert any(code.startswith("G2_") for code in moved)


def test_gg5_c5_whitespace_crater_response() -> None:
    """Assert C5 craters under the y-gap 1800-vs-36 signature.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    probe = _grid_probe(side=4, gap=36.0)
    scores = []
    ratios = []
    for alpha in (1.0, 10.0, 50.0):
        result = _score(_scaled(probe, alpha, positions_only=True))
        scores.append(_facet_value(result, "C5"))
        ratios.append(float(result["C5"].metadata["whitespace_ratio"]))
    print("GG-5c C5 crater matrix")
    print("alpha score ratio")
    for alpha, score, ratio in zip((1.0, 10.0, 50.0), scores, ratios):
        print(f"{alpha:4.0f} {score:.12g} {ratio:.12g}")
    assert scores[0] > scores[1] > scores[2]
    assert scores[1] < 0.5
    assert scores[2] < 0.05
    assert ratios[2] / ratios[0] > 1000.0


def _grid_probe(side: int, gap: float) -> Probe:
    """Create a directed grid probe.

    Parameters
    ----------
    side : int
        Number of nodes per side.
    gap : float
        Center spacing.

    Returns
    -------
    Probe
        Grid probe.
    """
    coords = [(float(x) * gap, float(y) * gap) for y in range(side) for x in range(side)]
    edges = []
    for y_index in range(side):
        for x_index in range(side):
            node = y_index * side + x_index
            if x_index + 1 < side:
                edges.append((node, node + 1))
            if y_index + 1 < side:
                edges.append((node, node + side))
    return Probe(
        "grid",
        torch.tensor(coords, dtype=torch.float64),
        torch.tensor(edges, dtype=torch.long).t().contiguous(),
        _sizes(side * side, 1.0),
        {},
    )


def test_f5_c5_form_selection(capsys: pytest.CaptureFixture[str]) -> None:
    """Compare primary and fallback C5 forms on legit sprawl and crater probes.

    Parameters
    ----------
    capsys : pytest.CaptureFixture[str]
        Pytest capture fixture used only to keep matrix output available.

    Returns
    -------
    None
    """
    del capsys
    rows = []
    legit = (
        ("deep_tree", _deep_tree_probe(7, 3.0)),
        ("long_chain", _chain_probe(30, 3.0)),
        ("radial_star", _radial_star_probe(24)),
    )
    craters = (
        ("grid_10x", _scaled(_grid_probe(4, 36.0), 10.0, positions_only=True)),
        ("grid_50x", _scaled(_grid_probe(4, 36.0), 50.0, positions_only=True)),
    )
    for name, probe in (*legit, *craters):
        meta = whitespace_sprawl_score(probe.pos, probe.edges, probe.sizes)
        primary = float(meta["whitespace_sprawl_score"])
        fallback = float(meta["whitespace_edge_length_node_diag_score"])
        rows.append((name, primary, fallback))
    print("F5 C5 form comparison")
    print("probe primary fallback")
    for name, primary, fallback in rows:
        print(f"{name:12s} {primary:.12g} {fallback:.12g}")
    legit_rows = rows[: len(legit)]
    crater_rows = rows[len(legit) :]
    assert all(primary >= 0.99 for _name, primary, _fallback in legit_rows)
    assert min(primary for _name, primary, _fallback in crater_rows) < min(
        fallback for _name, _primary, fallback in crater_rows
    )
    assert FROZEN_CONSTANTS["c5_primary_form"].value == "structure_floor_area_band"


def _chain_probe(count: int, gap: float) -> Probe:
    """Create a chain probe.

    Parameters
    ----------
    count : int
        Node count.
    gap : float
        Center spacing.

    Returns
    -------
    Probe
        Chain probe.
    """
    pos = torch.stack((torch.arange(count, dtype=torch.float64) * gap, torch.zeros(count)), dim=1)
    edges = torch.tensor([list(range(count - 1)), list(range(1, count))], dtype=torch.long)
    return Probe("chain", pos, edges, _sizes(count, 1.0), {})


def _deep_tree_probe(depth: int, gap: float) -> Probe:
    """Create a deep skinny tree probe.

    Parameters
    ----------
    depth : int
        Number of depth steps.
    gap : float
        Vertical gap.

    Returns
    -------
    Probe
        Deep tree probe.
    """
    count = depth + 1
    pos = torch.stack(
        (torch.zeros(count, dtype=torch.float64), torch.arange(count, dtype=torch.float64) * gap),
        dim=1,
    )
    edges = torch.tensor([list(range(count - 1)), list(range(1, count))], dtype=torch.long)
    return Probe("deep_tree", pos, edges, _sizes(count, 1.0), {})


def _radial_star_probe(leaves: int) -> Probe:
    """Create a radial star probe.

    Parameters
    ----------
    leaves : int
        Leaf count.

    Returns
    -------
    Probe
        Radial star probe.
    """
    radius = leaves / (2.0 * math.pi)
    coords = [(0.0, 0.0)]
    for index in range(leaves):
        theta = 2.0 * math.pi * index / leaves
        coords.append((radius * math.cos(theta), radius * math.sin(theta)))
    edges = torch.tensor([[0 for _ in range(leaves)], list(range(1, leaves + 1))], dtype=torch.long)
    return Probe(
        "radial_star", torch.tensor(coords, dtype=torch.float64), edges, _sizes(leaves + 1, 1.0), {}
    )


def test_f10_manifest_covers_required_constants() -> None:
    """Assert the frozen constants manifest contains the Part-A required records.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    required = {
        "tier_multipliers",
        "c2_size_decay",
        "c2_angle_weighted_crossing_cost",
        "c3_graph_distance_radii",
        "c4_smooth_clearance_band",
        "overlap_severity_saturation",
        "overlap_packing_fill_gate",
        "overlap_contact_shrinkage",
        "c5_primary_form",
        "c5_band_and_decay",
        "c6_crossing_angle_ideal_degrees",
        "diagnostic_core_weights",
        "canonical_node_height_ref",
        "deterministic_seed_default",
        "core_sampling_budgets",
        "hac_rule",
        "g5_quality_band",
        "gg3_buyback_gate",
        "margin_audit_tripwire",
        "g7_port_and_route_thresholds",
        "frac_acyclic_grading",
    }
    assert required.issubset(FROZEN_CONSTANTS)
    print("F10 frozen constants manifest")
    print("name value source")
    for key in sorted(FROZEN_CONSTANTS):
        record = FROZEN_CONSTANTS[key]
        print(f"{record.name}: {record.value!r} [{record.source}]")


def test_f10_manifest_values_match_live_constants() -> None:
    """Assert each frozen manifest value is bound to the live scorer constant.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    score_defaults = score_core_v3.__kwdefaults__ or {}
    c3_defaults = ruler_v3.multi_radius_neighborhood_preservation.__kwdefaults__ or {}
    g2_slots = ruler_v3_groups.GROUP_REGISTRY["G2"].tier_slots
    live_values: Dict[str, Any] = {
        "tier_multipliers": {
            "T1": ruler_v3.TIER_1_WEIGHT,
            "T2": ruler_v3.TIER_2_WEIGHT,
            "T3": ruler_v3.TIER_3_WEIGHT,
        },
        "c2_size_decay": {
            "N_mid": ruler_v3.CROSSING_DECAY_N_MID,
            "s": ruler_v3.CROSSING_DECAY_S,
            "floor_multiplier": 0.5,
        },
        "c2_angle_weighted_crossing_cost": {
            "min": ruler_v3.C2_ANGLE_COST_MIN,
            "max": ruler_v3.C2_ANGLE_COST_MAX,
            "best_angle_degrees": 90.0,
            "total_refinement_step_fraction": ruler_v3.C2_ANGLE_REFINEMENT_STEP_FRACTION,
        },
        "c3_graph_distance_radii": tuple(c3_defaults["radii"]),
        "c5_primary_form": "structure_floor_area_band",
        "c5_band_and_decay": {
            "content_multiplier": ruler_v3.WHITESPACE_CONTENT_MULTIPLIER,
            "structure_separation": ruler_v3.WHITESPACE_STRUCTURE_SEPARATION,
            "ratio_lo": ruler_v3.WHITESPACE_RATIO_LO,
            "ratio_hi": ruler_v3.WHITESPACE_RATIO_HI,
            "crowding_decay": ruler_v3.WHITESPACE_CROWDING_DECAY,
            "sprawl_decay": ruler_v3.WHITESPACE_SPRAWL_DECAY,
            "sprawl_ratio_factor": ruler_v3.SPRAWL_RATIO_FACTOR,
        },
        "c5_fallback_edge_length_band": {
            "ratio_lo": ruler_v3.EDGE_LENGTH_RATIO_LO,
            "ratio_hi": ruler_v3.EDGE_LENGTH_RATIO_HI,
        },
        "c6_crossing_angle_ideal_degrees": 90.0,
        "diagnostic_core_weights": {
            "C2": ruler_v3.CORE_WEIGHT_OVERRIDES["C2"],
            "C6": ruler_v3.CORE_DIAGNOSTIC_WEIGHTS["C6"],
            "C8": ruler_v3.CORE_DIAGNOSTIC_WEIGHTS["C8"],
        },
        "c4_smooth_clearance_band": {
            "clearance_band_node_diagonals": ruler_v3.C4_CLEARANCE_BAND_NODE_DIAGONALS,
            "label_inclusive_when_supplied": True,
        },
        "overlap_severity_saturation": ruler_v3.OVERLAP_SEVERITY_SATURATION,
        "overlap_packing_fill_gate": {
            "fill_lo": ruler_v3.OVERLAP_PACKING_FILL_LO,
            "fill_hi": ruler_v3.OVERLAP_PACKING_FILL_HI,
        },
        "overlap_contact_shrinkage": ruler_v3.OVERLAP_CONTACT_SHRINKAGE,
        "canonical_node_height_ref": ruler_v3_groups.CANONICAL_NODE_HEIGHT_REF,
        "default_node_box": {
            "width": ruler_v3.DEFAULT_NODE_WIDTH,
            "height": ruler_v3.DEFAULT_NODE_HEIGHT,
        },
        "degenerate_scale_and_occlusion_flags": {
            "degenerate_scale_ratio": ruler_v3.DEGENERATE_SCALE_RATIO,
            "occlusion_floor_threshold": ruler_v3.OCCLUSION_FLOOR_THRESHOLD,
        },
        "sprawl_collapse_c4_max": ruler_v3.SPRAWL_COLLAPSE_C4_MAX,
        "coincident_collapse_radius": ruler_v3.COINCIDENT_COLLAPSE_RADIUS,
        "coincident_collapse_fraction": ruler_v3.COINCIDENT_COLLAPSE_FRACTION,
        "deterministic_seed_default": score_defaults["seed"],
        "core_sampling_budgets": {
            "stress_sources": score_defaults["stress_sources"],
            "stress_targets": score_defaults["stress_targets"],
            "crossing_samples": score_defaults["crossing_samples"],
            "neighborhood_samples": score_defaults["neighborhood_samples"],
        },
        "g2_slot_weights": {
            "slot_a_total": ruler_v3_groups.G2_SLOT_A_TOTAL,
            "slot_b_total": ruler_v3_groups.G2_SLOT_B_TOTAL,
            "slot_a_split": tuple(
                int(slot.weight_within_slot * ruler_v3_groups.G2_SLOT_A_TOTAL)
                for slot in g2_slots[:4]
            ),
            "slot_b_split": tuple(
                int(slot.weight_within_slot * ruler_v3_groups.G2_SLOT_B_TOTAL)
                for slot in g2_slots[4:]
            ),
        },
        "hac_rule": {
            "linkage": "average",
            "metric": "euclidean",
            "cluster_count": "number_of_unique_declared_labels",
        },
        "g2_compactness_band": {
            "plateau_hi": ruler_v3_groups.COMPACTNESS_RATIO_PLATEAU_HI,
            "log_decay": ruler_v3_groups.COMPACTNESS_LOG_DECAY,
            "legible_spacing_guard": ruler_v3_groups.COMPACTNESS_LEGIBLE_SPACING_GUARD,
            "legible_spacing_eps": ruler_v3_groups.COMPACTNESS_LEGIBLE_SPACING_EPS,
        },
        "g4_slot_weights_and_extent_band": {
            "slot_a_total": ruler_v3_groups.G4_SLOT_A_TOTAL,
            "slot_b_total": ruler_v3_groups.G4_SLOT_B_TOTAL,
            "extent_ratio_lo": ruler_v3_groups.TREE_RATIO_PLATEAU_LO,
            "extent_ratio_hi": ruler_v3_groups.TREE_RATIO_PLATEAU_HI,
            "extent_log_decay": ruler_v3_groups.TREE_RATIO_LOG_DECAY,
        },
        "g5_quality_band": {
            "delta": ruler_v3_groups.G5_QUALITY_BAND_DELTA,
            "change_epsilon": ruler_v3_groups.G5_CHANGE_EPSILON,
            "zero_change_scale": ruler_v3_groups.G5_ZERO_CHANGE_SCALE,
        },
        "g6_weighted_budgets_and_cv": {
            "weight_cv_saturation": ruler_v3_groups.WEIGHT_CV_SATURATION,
            "nondegenerate_weight_cv": ruler_v3_groups.NONDEGENERATE_WEIGHT_CV,
            "local_node_budget": ruler_v3_groups.LOCAL_WEIGHT_MONOTONICITY_NODE_BUDGET,
            "weighted_stress_sources": ruler_v3_groups.G6_WEIGHTED_STRESS_SOURCE_BUDGET,
            "weighted_stress_targets": ruler_v3_groups.G6_WEIGHTED_STRESS_TARGET_BUDGET,
            "severe_floor": ruler_v3_groups.G6_SEVERE_FLOOR,
            "severe_floor_drop": ruler_v3.G6_FLOOR_DROP,
            "soft_floor": ruler_v3_groups.G6_SOFT_FLOOR,
            "severe_weight_multiplier": ruler_v3_groups.G6_SEVERE_WEIGHT_MULTIPLIER,
            "severe_breach_facets": ruler_v3.SEVERE_G6_FACETS,
        },
        "severe_g6_breach_flag": ruler_v3.SEVERE_G6_BREACH_FLAG,
        "gg3_buyback_gate": {
            "tier1_materiality": ruler_v3.PAIR_MATERIAL_HOLD_TIER1_DROP,
            "buyback_bar": ruler_v3.PAIR_MATERIAL_HOLD_BUYBACK,
            "hold_band_fraction": ruler_v3.PAIR_MATERIAL_HOLD_AGGREGATE_FRACTION,
        },
        "margin_audit_tripwire": {
            "headline": "tiered_linear",
            "family_envelope": "adjudication_instrument_only",
            "fallback": ruler_v3.FAMILY_MARGIN_ALLOWANCES["__fallback__"],
            "softmin_tau": ruler_v3.FAMILY_SOFTMIN_TAU,
            "de_ramped_tier1_instrument": True,
            "audit_column": "tiered_capped",
            "hold_instrument_column": "tiered_hold_instrument",
            "A_f": {
                "weighted": ruler_v3.FAMILY_MARGIN_ALLOWANCES["weighted"],
                "clustered": ruler_v3.FAMILY_MARGIN_ALLOWANCES["clustered"],
                "dag": ruler_v3.FAMILY_MARGIN_ALLOWANCES["dag"],
                "generic_force": ruler_v3.FAMILY_MARGIN_ALLOWANCES["generic_force"],
                "tree": ruler_v3.FAMILY_MARGIN_ALLOWANCES["tree"],
                "ported": ruler_v3.FAMILY_MARGIN_ALLOWANCES["ported"],
            },
        },
        "g7_port_and_route_thresholds": {
            "severe_port_cap": ruler_v3_groups.G7_SEVERE_PORT_CAP,
            "side_cosine_threshold": 2**-0.5,
            "route_sample_canonical_distance": ruler_v3_groups.G7_ROUTE_SAMPLE_CANONICAL_DISTANCE,
            "terminal_separation_fraction": ruler_v3_groups.G7_TERMINAL_SEPARATION_FRACTION,
        },
        "frac_acyclic_grading": (
            "effective_weight *= fraction_of_hierarchy_edges_after_declared_feedback_removal"
        ),
        "freeze_probe_thresholds": {
            "unit_alphas": ALPHAS,
            "position_scale_alphas": ALPHAS,
            "continuous_relative_tolerance": 1e-12,
            "crater_10x_max_score": 0.5,
            "crater_50x_max_score": 0.05,
            "legit_sprawl_min_score": 0.99,
        },
    }

    assert set(live_values) == set(FROZEN_CONSTANTS)
    for key, value in live_values.items():
        assert FROZEN_CONSTANTS[key].value == value


def test_gg4_deformation_monotonicity_sweep(capsys: pytest.CaptureFixture[str]) -> None:
    """Assert every oracle decays under progressive deformation probes.

    Parameters
    ----------
    capsys : pytest.CaptureFixture[str]
        Pytest capture fixture used only to keep matrix output available.

    Returns
    -------
    None
    """
    del capsys
    sweeps = _deformation_sweeps()
    rows: List[Tuple[str, str, str]] = []
    detail_rows: List[Tuple[str, int, float]] = []
    failures = []
    for facet, probes in sweeps.items():
        scores = [_facet_value(_score(probe), facet) for probe in probes]
        monotone = all(a + 1e-9 >= b for a, b in zip(scores, scores[1:]))
        decays = scores[0] > scores[-1] + 1e-9
        flat = max(scores) - min(scores) <= 1e-9
        low_information = facet in LOW_INFORMATION_DEFORMATION_FACETS and monotone and flat
        status = (
            "PASS"
            if monotone and decays
            else "LOW-INFORMATION flat-not-inverting"
            if low_information
            else f"FAIL {scores}"
        )
        rows.append((facet, probes[0].name, status))
        detail_rows.extend((facet, step, score) for step, score in enumerate(scores))
        if not ((monotone and decays) or low_information):
            failures.append((facet, scores))
    _print_matrix("GG-4 deformation monotonicity matrix", sorted(rows))
    print("GG-4 deformation sweep scores")
    print("facet step score")
    for facet, step, score in sorted(detail_rows):
        print(f"{facet:40s} {step:d} {score:.12g}")
    assert not failures


def _deformation_sweeps() -> Mapping[str, Tuple[Probe, ...]]:
    """Return four-step deformation sweeps keyed by facet.

    Parameters
    ----------
    None

    Returns
    -------
    Mapping[str, Tuple[Probe, ...]]
        Deformation probes keyed by facet code.
    """
    return {
        "C1": tuple(
            _path_order_probe(order)
            for order in ([0, 1, 2, 3, 4], [0, 1, 3, 2, 4], [0, 3, 1, 2, 4], [4, 2, 0, 3, 1])
        ),
        "C2": tuple(_crossing_count_probe(step) for step in range(4)),
        "C3": tuple(
            _path_order_probe(order)
            for order in ([0, 1, 2, 3, 4], [0, 1, 3, 2, 4], [0, 3, 1, 2, 4], [4, 2, 0, 3, 1])
        ),
        "C4": tuple(_overlap_probe(gap) for gap in (4.0, 0.9, 0.45, 0.05)),
        "C5": tuple(
            _scaled(_grid_probe(4, 36.0), alpha, positions_only=True)
            for alpha in (1.0, 5.0, 10.0, 50.0)
        ),
        "C6": tuple(_crossing_angle_probe(angle) for angle in (90.0, 60.0, 30.0, 10.0)),
        "C7": tuple(_gabriel_probe(offset) for offset in (3.0, 1.0, 0.3, 0.0)),
        "C8": tuple(_path_bend_probe(offset) for offset in (0.0, 0.5, 1.5, 4.0)),
        "C9": tuple(_star_angle_probe(width) for width in (6.0, 2.0, 0.6, 0.1)),
        "C10": tuple(
            _length_cv_probe(lengths)
            for lengths in ([4, 4, 4, 4], [2, 4, 4, 4], [1, 3, 6, 8], [1, 2, 8, 16])
        ),
        "G1_directed_flow": tuple(_g1_probe(step) for step in range(4)),
        "G1_depth_order": tuple(_g1_probe(step) for step in range(4)),
        "G2_cluster_exclusion": tuple(_g2_deformation(step) for step in range(4)),
        "G2_cluster_sibling_overlap": tuple(_g2_deformation(step) for step in range(4)),
        "G2_cluster_edge_intrusion": tuple(_g2_deformation(step) for step in range(4)),
        "G2_cluster_hac_ari": tuple(_g2_deformation(step) for step in range(4)),
        "G2_cluster_nesting_fidelity": tuple(
            _g2_nesting_escape_deformation(step) for step in range(4)
        ),
        "G2_cluster_compactness_log_band": tuple(
            _g2_compactness_deformation(step) for step in range(4)
        ),
        "G3_community_ari": tuple(_g3_deformation(step) for step in range(4)),
        "G4_layered_depth_bands": tuple(_g4_layered_deformation(step) for step in range(4)),
        "G4_layered_parent_centering": tuple(_g4_layered_deformation(step) for step in range(4)),
        "G4_layered_sibling_order": tuple(_g4_layered_deformation(step) for step in range(4)),
        "G4_layered_contour_separation": tuple(
            _g4_layered_contour_deformation(step) for step in range(4)
        ),
        "G4_radial_depth_monotonicity": tuple(
            _g4_radial_depth_deformation(step) for step in range(4)
        ),
        "G4_radial_angular_allocation": tuple(_g4_radial_deformation(step) for step in range(4)),
        "G4_radial_angular_overlap": tuple(_g4_radial_deformation(step) for step in range(4)),
        "G4_radial_circular_order": tuple(
            _g4_radial_circular_order_deformation(step) for step in range(4)
        ),
        "G5_temporal_stability": tuple(_g5_deformation(step) for step in range(4)),
        "G6_weighted_ksm": tuple(
            _weighted_probe(lengths)
            for lengths in ([1, 2, 4, 8], [1, 2, 8, 4], [8, 4, 2, 1], [8, 4, 2, 1])
        ),
        "G6_local_weight_monotonicity": tuple(
            _weighted_probe(lengths)
            for lengths in ([1, 2, 4, 8], [1, 2, 8, 4], [8, 4, 2, 1], [8, 4, 2, 1])
        ),
        "G7_port_hard_compliance": tuple(_g7_deformation(step) for step in range(4)),
        "G7_routed_curve_quality": tuple(_g7_deformation(step) for step in range(4)),
        "G7_routed_bend_terminal_economy": tuple(_g7_deformation(step) for step in range(4)),
    }


def _path_order_probe(order: Sequence[int]) -> Probe:
    """Create a path graph with a declared drawing order.

    Parameters
    ----------
    order : Sequence[int]
        Node ids from left to right.

    Returns
    -------
    Probe
        Path order probe.
    """
    coords = [(0.0, 0.0)] * len(order)
    for x_index, node in enumerate(order):
        coords[node] = (float(x_index), 0.0)
    return _path_probe_from_coords(coords)


def _path_probe_from_coords(coords: Sequence[Tuple[float, float]]) -> Probe:
    """Create a path probe from coordinates.

    Parameters
    ----------
    coords : Sequence[Tuple[float, float]]
        Node coordinates.

    Returns
    -------
    Probe
        Path probe.
    """
    count = len(coords)
    edges = torch.tensor([list(range(count - 1)), list(range(1, count))], dtype=torch.long)
    return Probe(
        "path_deform", torch.tensor(coords, dtype=torch.float64), edges, _sizes(count, 0.2), {}
    )


def _crossing_count_probe(step: int) -> Probe:
    """Create a graph with progressively more crossings.

    Parameters
    ----------
    step : int
        Deformation step.

    Returns
    -------
    Probe
        Crossing-count probe.
    """
    pos = torch.tensor(
        [(0.0, 0.0), (1.0, 0.2 * step), (2.0, 0.0), (3.0, 0.2 * step)], dtype=torch.float64
    )
    if step == 0:
        edges = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)
    else:
        pos = torch.tensor([(0.0, 0.0), (0.0, 2.0), (2.0, 2.0), (2.0, 0.0)], dtype=torch.float64)
        edge_sets = (
            [(0, 1), (2, 3)],
            [(0, 2), (1, 3)],
            [(0, 2), (1, 3), (0, 3)],
            [(0, 2), (1, 3), (0, 3)],
        )
        edges = torch.tensor(edge_sets[step], dtype=torch.long).t().contiguous()
    return Probe("crossings", pos, edges, _sizes(4, 0.1), {})


def _overlap_probe(gap: float) -> Probe:
    """Create a progressive node-overlap probe.

    Parameters
    ----------
    gap : float
        Center spacing.

    Returns
    -------
    Probe
        Overlap probe.
    """
    pos = torch.tensor([(0.0, 0.0), (gap, 0.0), (2.0 * gap, 0.0)], dtype=torch.float64)
    edges = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    return Probe("overlap", pos, edges, _sizes(3, 1.0), {})


def _crossing_angle_probe(angle_degrees: float) -> Probe:
    """Create one crossing with a controlled angle.

    Parameters
    ----------
    angle_degrees : float
        Crossing angle in degrees.

    Returns
    -------
    Probe
        Crossing-angle probe.
    """
    theta = math.radians(angle_degrees)
    pos = torch.tensor(
        [
            (-1.0, 0.0),
            (1.0, 0.0),
            (-math.cos(theta), -math.sin(theta)),
            (math.cos(theta), math.sin(theta)),
        ],
        dtype=torch.float64,
    )
    edges = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)
    return Probe("crossing_angle", pos, edges, _sizes(4, 0.1), {})


def _gabriel_probe(offset: float) -> Probe:
    """Create an edge-node lens probe.

    Parameters
    ----------
    offset : float
        Third-node offset from the edge midpoint.

    Returns
    -------
    Probe
        Gabriel probe.
    """
    pos = torch.tensor([(-2.0, 0.0), (2.0, 0.0), (0.0, offset)], dtype=torch.float64)
    edges = torch.tensor([[0], [1]], dtype=torch.long)
    return Probe("gabriel", pos, edges, _sizes(3, 0.1), {})


def _path_bend_probe(offset: float) -> Probe:
    """Create a degree-2 path continuity probe.

    Parameters
    ----------
    offset : float
        Bend offset at middle path nodes.

    Returns
    -------
    Probe
        Path-continuity probe.
    """
    coords = [(0.0, 0.0), (1.0, offset), (2.0, -offset), (3.0, offset), (4.0, 0.0)]
    return _path_probe_from_coords(coords)


def _star_angle_probe(width: float) -> Probe:
    """Create an angular-resolution star probe.

    Parameters
    ----------
    width : float
        Fan width.

    Returns
    -------
    Probe
        Star probe.
    """
    angles = [0.0, 90.0, 180.0, 270.0] if width >= 6.0 else [0.0, width, 180.0, 270.0]
    pos = torch.tensor(
        [(0.0, 0.0)]
        + [(math.cos(math.radians(angle)), math.sin(math.radians(angle))) for angle in angles],
        dtype=torch.float64,
    )
    edges = torch.tensor([[0, 0, 0, 0], [1, 2, 3, 4]], dtype=torch.long)
    return Probe("star_angle", pos, edges, _sizes(5, 0.1), {})


def _length_cv_probe(lengths: Sequence[float]) -> Probe:
    """Create an edge-length CV probe.

    Parameters
    ----------
    lengths : Sequence[float]
        Star edge lengths.

    Returns
    -------
    Probe
        Length-CV probe.
    """
    return _weighted_probe(lengths)


def _g1_probe(step: int) -> Probe:
    """Create a directed-flow deformation probe.

    Parameters
    ----------
    step : int
        Deformation step.

    Returns
    -------
    Probe
        G1 probe.
    """
    ys = ([0.0, 2.0, 4.0, 6.0], [0.0, 2.0, 1.0, 3.0], [3.0, 2.0, 1.0, 0.0], [3.0, 2.0, 1.0, 0.0])[
        step
    ]
    pos = torch.tensor([(0.0, y_value) for y_value in ys], dtype=torch.float64)
    edges = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    meta = {
        "declared_hierarchical": True,
        "flow_direction": "TB",
        "topological_depth": [0, 1, 2, 3],
    }
    return Probe("g1", pos, edges, _sizes(4, 0.2), meta)


def _g2_deformation(step: int) -> Probe:
    """Create a declared-cluster deformation probe.

    Parameters
    ----------
    step : int
        Deformation step.

    Returns
    -------
    Probe
        G2 probe.
    """
    coords = (
        [[-2.0, -0.2], [-2.2, 0.2], [-1.8, 0.0], [2.0, -0.2], [2.2, 0.2], [1.8, 0.0]],
        [[-2.0, -0.2], [-2.2, 0.2], [0.0, 0.0], [2.0, -0.2], [2.2, 0.2], [0.2, 0.0]],
        [[-2.0, -0.2], [2.2, 0.2], [-1.8, 0.0], [2.0, -0.2], [-2.2, 0.2], [1.8, 0.0]],
        [[-2.0, -0.2], [2.2, 0.2], [-1.8, 0.0], [2.0, -0.2], [-2.2, 0.2], [1.8, 0.0]],
    )[step]
    base = _cluster_probe()
    return Probe("g2", torch.tensor(coords, dtype=torch.float64), base.edges, base.sizes, base.meta)


def _g2_compactness_deformation(step: int) -> Probe:
    """Create an intra-cluster compactness deformation.

    Parameters
    ----------
    step : int
        Deformation step.

    Returns
    -------
    Probe
        G2 compactness probe.
    """
    scale = (2.0, 3.0, 20.0, 100.0)[step]
    return _cluster_probe(scale=scale)


def _g2_nesting_escape_deformation(step: int) -> Probe:
    """Create a nested-cluster child-escape deformation probe.

    Parameters
    ----------
    step : int
        Deformation step.

    Returns
    -------
    Probe
        Nested G2 probe.
    """
    escape = (0.0, 8.0, 40.0, 160.0)[step]
    pos = torch.tensor(
        [
            [-1.0 + escape, -0.2],
            [1.0 + escape, 0.2],
            [-3.0, 0.0],
            [3.0, 0.0],
        ],
        dtype=torch.float64,
    )
    edges = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    meta = {
        "clusters": {
            "parent": [0, 1, 2, 3],
            "child": [0, 1],
        },
        "cluster_parents": {"parent": None, "child": "parent"},
        "cluster_labels": {"parent": "Parent", "child": "Child"},
    }
    return Probe("g2_nesting_escape", pos, edges, _sizes(4, 0.5), meta)


def _g3_deformation(step: int) -> Probe:
    """Create a planted-community deformation probe.

    Parameters
    ----------
    step : int
        Deformation step.

    Returns
    -------
    Probe
        G3 probe.
    """
    coords = (
        [[-3.0, -0.2], [-3.1, 0.1], [-2.8, 0.0], [3.0, -0.1], [3.2, 0.2], [2.9, 0.1]],
        [[-3.0, -0.2], [-3.1, 0.1], [2.8, 0.0], [3.0, -0.1], [3.2, 0.2], [-2.9, 0.1]],
        [[-3.0, -0.2], [3.1, 0.1], [-2.8, 0.0], [3.0, -0.1], [-3.2, 0.2], [2.9, 0.1]],
        [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.15, 0.0], [0.25, 0.0], [0.35, 0.0]],
    )[step]
    base = _community_probe()
    return Probe("g3", torch.tensor(coords, dtype=torch.float64), base.edges, base.sizes, base.meta)


def _g4_layered_deformation(step: int) -> Probe:
    """Create a layered-tree deformation probe.

    Parameters
    ----------
    step : int
        Deformation step.

    Returns
    -------
    Probe
        Layered G4 probe.
    """
    coords = (
        [(0.0, 0.0), (-2.0, 1.0), (0.0, 1.0), (2.0, 1.0), (1.5, 2.0), (2.5, 2.0)],
        [(0.8, 0.0), (-2.0, 1.0), (2.8, 1.0), (2.0, 1.0), (2.7, 1.3), (2.5, 2.0)],
        [(3.0, 2.0), (2.0, 1.0), (1.0, 0.5), (0.0, 0.0), (-1.0, 0.0), (-2.0, 0.0)],
        [(3.0, 2.0), (2.0, 1.0), (1.0, 0.5), (0.0, 0.0), (-1.0, 0.0), (-2.0, 0.0)],
    )[step]
    base = _tree_probe(radial=False)
    return Probe(
        "g4_layered", torch.tensor(coords, dtype=torch.float64), base.edges, base.sizes, base.meta
    )


def _g4_layered_contour_deformation(step: int) -> Probe:
    """Create a layered sibling-subtree overlap deformation probe.

    Parameters
    ----------
    step : int
        Deformation step.

    Returns
    -------
    Probe
        Layered contour-separation probe.
    """
    right_center = (5.0, -1.5, -4.0, -5.0)[step]
    coords = [
        (0.0, -2.0),
        (-5.0, 0.0),
        (-6.0, 1.0),
        (-4.0, 1.0),
        (right_center, 0.0),
        (right_center - 1.0, 1.0),
        (right_center + 1.0, 1.0),
    ]
    edges = torch.tensor([[0, 0, 1, 1, 4, 4], [1, 4, 2, 3, 5, 6]], dtype=torch.long)
    meta = {"declared_tree": True, "root": 0, "tree_convention": "layered"}
    return Probe(
        "g4_layered_contour",
        torch.tensor(coords, dtype=torch.float64),
        edges,
        _sizes(7, 2.0),
        meta,
    )


def _g4_radial_deformation(step: int) -> Probe:
    """Create a radial-tree deformation probe.

    Parameters
    ----------
    step : int
        Deformation step.

    Returns
    -------
    Probe
        Radial G4 probe.
    """
    radius = 3.0
    perfect_angles = [0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0]
    coords = (
        [(0.0, 0.0), *[(radius * math.cos(a), radius * math.sin(a)) for a in perfect_angles]],
        [(0.0, 0.0), (3.0, -0.05), (3.0, 0.0), (-1.5, -2.6)],
        [(0.0, 0.0), (3.0, -0.05), (3.0, 0.0), (3.0, 0.05)],
        [(0.0, 0.0), (3.0, -0.05), (3.0, 0.0), (3.0, 0.05)],
    )[step]
    edges = torch.tensor([[0, 0, 0], [1, 2, 3]], dtype=torch.long)
    meta = {"declared_tree": True, "root": 0, "tree_convention": "radial"}
    return Probe(
        "g4_radial", torch.tensor(coords, dtype=torch.float64), edges, _sizes(4, 0.2), meta
    )


def _g4_radial_depth_deformation(step: int) -> Probe:
    """Create a radial radius-depth inversion deformation probe.

    Parameters
    ----------
    step : int
        Deformation step.

    Returns
    -------
    Probe
        Radial depth-monotonicity probe.
    """
    inner_radius = (1.0, 3.5, 5.5, 7.5)[step]
    coords = [
        (0.0, 0.0),
        (inner_radius, 0.0),
        (0.0, 2.0),
        (0.0, 3.0),
        (0.0, 4.0),
        (0.0, 5.0),
    ]
    edges = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long)
    meta = {"declared_tree": True, "root": 0, "tree_convention": "radial"}
    return Probe(
        "g4_radial_depth", torch.tensor(coords, dtype=torch.float64), edges, _sizes(6, 0.2), meta
    )


def _g4_radial_circular_order_deformation(step: int) -> Probe:
    """Create a radial sibling circular-order deformation probe.

    Parameters
    ----------
    step : int
        Deformation step.

    Returns
    -------
    Probe
        Radial circular-order probe.
    """
    drawn_orders = (
        [1, 2, 3, 4, 5],
        [1, 2, 4, 3, 5],
        [1, 4, 2, 5, 3],
        [5, 4, 3, 2, 1],
    )
    radius = 4.0
    coords = [(0.0, 0.0)] * 6
    for rank, node in enumerate(drawn_orders[step]):
        angle = 2.0 * math.pi * rank / 5.0
        coords[node] = (radius * math.cos(angle), radius * math.sin(angle))
    edges = torch.tensor([[0, 0, 0, 0, 0], [1, 2, 3, 4, 5]], dtype=torch.long)
    meta = {"declared_tree": True, "root": 0, "tree_convention": "radial"}
    return Probe(
        "g4_radial_order", torch.tensor(coords, dtype=torch.float64), edges, _sizes(6, 0.2), meta
    )


def _g5_deformation(step: int) -> Probe:
    """Create a temporal deformation probe.

    Parameters
    ----------
    step : int
        Deformation step.

    Returns
    -------
    Probe
        G5 probe.
    """
    args = ((0.2, 0.98, 0.042), (0.45, 0.90, 0.02), (0.0, 0.70, 0.40), (0.0, 0.70, 0.40))[step]
    return _temporal_probe(current_shift=args[0], previous_quality=args[1], graph_change=args[2])


def _g7_deformation(step: int) -> Probe:
    """Create a port/routing deformation probe.

    Parameters
    ----------
    step : int
        Deformation step.

    Returns
    -------
    Probe
        G7 probe.
    """
    probe = _ported_probe(routed=True, wrong_side=step >= 2)
    meta = dict(probe.meta)
    routes = list(meta["route_paths"])
    if step == 1:
        routes[1] = [(0.0, 0.0), (2.0, -0.8), (4.0, 1.0)]
    elif step >= 2:
        routes = [
            [(0.0, 0.0), (-1.0, 0.0), (4.0, -1.0)],
            [(0.0, 0.0), (-1.0, 0.0), (4.0, 1.0)],
            [(4.0, -1.0), (4.0, 2.0), (8.0, 0.0)],
            [(4.0, 1.0), (4.0, -2.0), (8.0, 0.0)],
        ]
    if step == 3:
        routes = [[(0.0, 0.0), (-3.0, 0.0), (4.0, -1.0)] for _ in range(4)]
    meta["route_paths"] = routes
    return Probe("g7", probe.pos, probe.edges, probe.sizes, meta)
