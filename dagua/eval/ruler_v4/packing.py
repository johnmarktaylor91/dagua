"""Multi-component packing, planar-face, and channel-contrast facets."""

# Contract identity docstrings intentionally keep each title and digest on one line.
# ruff: noqa: E501

from __future__ import annotations

import math
from typing import List, Tuple

import torch

from dagua.eval.ruler_v4._util import (
    bounded,
    components,
    mean_result,
    proper_intersection,
    route_segments,
)
from dagua.eval.ruler_v4.frames import RobustFrame, robust_frame
from dagua.eval.ruler_v4.scene import FacetResult, Scene, na_result


def _component_frames(scene: Scene) -> List[Tuple[List[int], RobustFrame]]:
    """Build robust frames for simple-support components.

    Parameters
    ----------
    scene : Scene
        Validated graph scene.

    Returns
    -------
    list[tuple[list[int], RobustFrame]]
        Canonical members and their robust frames.
    """

    return [
        (members, robust_frame(scene.positions[members], scene.intrinsic_unit))
        for members in components(scene)
    ]


def U38(scene: Scene) -> FacetResult:
    """Multi-component packing. Frozen SHA-256: 1a0023e6c0515714dbda83f97a6a57e1c4ecbf7e27f15a6650c674ca9f0f68b1."""

    frames = _component_frames(scene)
    if len(frames) < 2:
        return na_result("single_component")
    clearances = []
    proportionality = []
    component_areas = []
    component_masses = []
    for members, frame in frames:
        component_areas.append(frame.area)
        component_masses.append(len(members))
    for index, (_, left) in enumerate(frames):
        for _, right in frames[index + 1 :]:
            delta = torch.abs(left.center - right.center) - (left.half_extents + right.half_extents)
            outside = float(torch.linalg.vector_norm(torch.clamp(delta, min=0.0)))
            penetration = max(0.0, -max(float(delta[0]), float(delta[1])))
            clearances.append(bounded(penetration + math.exp(-outside / scene.intrinsic_unit)))
    total_area = sum(component_areas)
    total_mass = sum(component_masses)
    for area, mass in zip(component_areas, component_masses):
        proportionality.append(abs(area / total_area - mass / total_mass))
    global_frame = robust_frame(scene.positions, scene.intrinsic_unit)
    occupied = sum(component_areas) / global_frame.area
    pack = bounded(max(0.0, 0.25 - occupied) / 0.25)
    values = {
        "U38.L_clear": sum(clearances) / len(clearances),
        "U38.L_pack": pack,
        "U38.L_prop": min(1.0, sum(proportionality)),
    }
    return mean_result(
        "U38", values, {"component_count": len(frames), "occupied_fraction": occupied}
    )


def _crossing_count(scene: Scene) -> int:
    """Count proper nonincident route crossings.

    Parameters
    ----------
    scene : Scene
        Validated routed scene.

    Returns
    -------
    int
        Exact segment-pair crossing count.
    """

    segments = route_segments(scene)
    count = 0
    for index, (route_a, _, start_a, end_a) in enumerate(segments):
        edge_a = scene.graph.edges[scene.routes[route_a].edge_index]
        for route_b, _, start_b, end_b in segments[index + 1 :]:
            if route_a == route_b:
                continue
            edge_b = scene.graph.edges[scene.routes[route_b].edge_index]
            if set(edge_a) & set(edge_b):
                continue
            count += int(proper_intersection(start_a, end_a, start_b, end_b))
    return count


def U41(scene: Scene) -> FacetResult:
    """Planarity and face quality. Frozen SHA-256: b26cdb05f3d09cda123b5fd6becbc8d7ebdaca931b89d0f2d2ee1e2d0f0f518a."""

    certificate = scene.graph.planarity_certificate
    if certificate is None:
        return na_result("no_input_planarity_certificate")
    if not bool(certificate.get("planar", False)):
        return na_result("graph_not_certified_planar")
    face_opportunity = max(1, scene.edge_count - scene.node_count + len(components(scene)))
    crossings = _crossing_count(scene)
    # The canonical arrangement is event-aware: crossings become dummy vertices.
    # Phase 1 publishes bounded convexity and area proxies while preserving F0 as an
    # input-only normalizer; exact DCEL construction remains independent of composition.
    convexity = bounded(crossings / face_opportunity)
    lengths = []
    for route in scene.routes:
        lengths.extend(
            torch.linalg.vector_norm(route.points[1:] - route.points[:-1], dim=1).tolist()
        )
    if lengths:
        tensor = torch.tensor(lengths, dtype=torch.float64)
        balance = bounded(
            float(torch.std(tensor, unbiased=False) / torch.clamp(torch.mean(tensor), min=1e-12))
        )
    else:
        balance = 0.0
    values = {"U41.L_conv": convexity, "U41.L_area": balance}
    return mean_result(
        "U41",
        values,
        {"F0": face_opportunity, "crossing_pairs_k": crossings, "certificate_verified": True},
    )


def U42(scene: Scene) -> FacetResult:
    """Encoding fidelity & contrast. Frozen SHA-256: 7ee672a532e6032cab87ca1ffe35156192c1a19edf384a4db5e33803d389c94e."""

    return na_result("no_declared_channels")
