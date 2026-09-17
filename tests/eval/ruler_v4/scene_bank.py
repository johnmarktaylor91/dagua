"""Deterministic stratified pilot scene bank for 6.3 certification.

V4_SPEC_r4 6.3 requires conformance gates on a frozen stratified bank,
PER CLASS AND SCALE. The frozen production bank is a P5 calibration
artifact (DISCREPANCIES entry 39); this module is the pilot-time
implementation of the same stratification contract: deterministic
seeded construction, structural classes crossed with size bands, and a
published bank digest so any two runs can prove they scored identical
bytes. Every cell carries one graph drawn at graded quality levels
(constructed layout, mild/moderate/severe seeded noise, seeded
scramble) so rankings inside a cell have real content.

Nothing here is score-visible: the bank feeds certification harnesses
and tests only.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import torch

from dagua.eval.ruler_v4.ingestion import ingest
from dagua.eval.ruler_v4.scene import (
    DrawingScene,
    GraphSemantics,
    ObservationProfile,
    Route,
    Scene,
    StyleContract,
    ValidScene,
)

BANK_SEED = 20260815
# Pilot-time size bands: exact facet evaluation is superlinear in node
# count (measured ~145 s/scene at 20 nodes), so the pilot's large band is
# 32 nodes; the production bank's larger bands are P5-owned (entry 39).
SCALE_BANDS: Mapping[str, int] = {"small": 8, "medium": 20, "large": 32}
CLASS_NAMES: Tuple[str, ...] = ("chain", "tree", "layered_dag", "clustered")
VARIANT_NAMES: Tuple[str, ...] = (
    "constructed",
    "rescaled",
    "noise_mild",
    "noise_moderate",
    "noise_severe",
    "scrambled",
)
_NOISE_SIGMAS: Mapping[str, float] = {
    "noise_mild": 0.05,
    "noise_moderate": 0.20,
    "noise_severe": 0.60,
}


@dataclass(frozen=True)
class BankCell:
    """One stratified cell: a graph drawn at graded quality levels.

    Parameters
    ----------
    class_name : str
        Structural class stratum.
    scale_name : str
        Size band stratum.
    scenes : tuple[Scene, ...]
        Validated scenes in ``VARIANT_NAMES`` order.
    """

    class_name: str
    scale_name: str
    scenes: Tuple[Scene, ...]


def _chain_semantics(count: int) -> Tuple[GraphSemantics, torch.Tensor]:
    """Build a directed path with layered tree semantics and a good layout."""

    edges = tuple((index, index + 1) for index in range(count - 1))
    graph = GraphSemantics(
        node_ids=tuple(f"n{index}" for index in range(count)),
        edges=edges,
        directed=True,
        node_labels=tuple(f"n{index}" for index in range(count)),
        edge_labels=tuple(None for _ in edges),
        ranks=tuple(range(count)),
        roots=(0,),
        feedback=tuple(False for _ in edges),
        edge_weights=tuple(1.0 + (index % 3) for index in range(len(edges))),
        weight_semantics="distance_cost",
        required_primitives=frozenset({"nodes", "routes"}),
        tree_parents=(None, *range(count - 1)),
        tree_depths=tuple(range(count)),
        tree_layout="layered",
        flow_axis=(1.0, 0.0),
    )
    positions = torch.stack(
        [torch.tensor([2.0 * index, 0.0], dtype=torch.float64) for index in range(count)]
    )
    return graph, positions


def _tree_semantics(count: int) -> Tuple[GraphSemantics, torch.Tensor]:
    """Build a balanced binary tree with a layered layout."""

    parents: List[int | None] = [None]
    for index in range(1, count):
        parents.append((index - 1) // 2)
    edges = tuple((parent, child) for child, parent in enumerate(parents) if parent is not None)
    depths = [0] * count
    for index in range(1, count):
        depths[index] = depths[parents[index]] + 1
    graph = GraphSemantics(
        node_ids=tuple(f"n{index}" for index in range(count)),
        edges=edges,
        directed=True,
        node_labels=tuple(f"n{index}" for index in range(count)),
        edge_labels=tuple(None for _ in edges),
        ranks=tuple(depths),
        roots=(0,),
        feedback=tuple(False for _ in edges),
        edge_weights=tuple(1.0 + (index % 2) for index in range(len(edges))),
        weight_semantics="distance_cost",
        required_primitives=frozenset({"nodes", "routes"}),
        tree_parents=tuple(parents),
        tree_depths=tuple(depths),
        tree_layout="layered",
        flow_axis=(0.0, -1.0),
        ordered_children={
            parent: tuple(child for child, p in enumerate(parents) if p == parent)
            for parent in range(count)
            if any(p == parent for p in parents[1:])
        },
    )
    max_depth = max(depths)
    per_depth: Dict[int, int] = {}
    coordinates = []
    for index in range(count):
        offset = per_depth.get(depths[index], 0)
        per_depth[depths[index]] = offset + 1
        width = sum(1 for d in depths if d == depths[index])
        coordinates.append(
            [2.0 * (offset - (width - 1) / 2.0), -2.0 * depths[index] + 0.0 * max_depth]
        )
    positions = torch.tensor(coordinates, dtype=torch.float64)
    return graph, positions


def _layered_dag_semantics(count: int) -> Tuple[GraphSemantics, torch.Tensor]:
    """Build a two-track diamond ladder joined at both ends."""

    interior = count - 2
    track_length = interior // 2
    upper = list(range(1, 1 + track_length))
    lower = list(range(1 + track_length, 1 + 2 * track_length))
    sink = 1 + 2 * track_length
    edges: List[Tuple[int, int]] = [(0, upper[0]), (0, lower[0])]
    for track in (upper, lower):
        edges.extend((track[index], track[index + 1]) for index in range(len(track) - 1))
    edges.extend(((upper[-1], sink), (lower[-1], sink)))
    used = sink + 1
    ranks = [0] * used
    for position, node in enumerate(upper):
        ranks[node] = position + 1
    for position, node in enumerate(lower):
        ranks[node] = position + 1
    ranks[sink] = track_length + 1
    graph = GraphSemantics(
        node_ids=tuple(f"n{index}" for index in range(used)),
        edges=tuple(edges),
        directed=True,
        node_labels=tuple(f"n{index}" for index in range(used)),
        edge_labels=tuple(None for _ in edges),
        ranks=tuple(ranks),
        roots=(0,),
        feedback=tuple(False for _ in edges),
        edge_weights=tuple(1.0 + (index % 4) for index in range(len(edges))),
        weight_semantics="distance_cost",
        required_primitives=frozenset({"nodes", "routes"}),
        flow_axis=(1.0, 0.0),
    )
    coordinates = [[0.0, 0.0]]
    for position, node in enumerate(upper):
        coordinates.append([2.0 * (position + 1), 2.0])
    for position, node in enumerate(lower):
        coordinates.append([2.0 * (position + 1), -2.0])
    coordinates.append([2.0 * (track_length + 1), 0.0])
    positions = torch.tensor(coordinates, dtype=torch.float64)
    return graph, positions


def _clustered_semantics(count: int) -> Tuple[GraphSemantics, torch.Tensor]:
    """Build a two-cluster layered graph under one parent cluster."""

    half = count // 2
    left = tuple(range(half))
    right = tuple(range(half, count))
    edges: List[Tuple[int, int]] = []
    for block in (left, right):
        edges.extend((block[index], block[index + 1]) for index in range(len(block) - 1))
    edges.append((left[-1], right[0]))
    ranks = list(range(count))
    graph = GraphSemantics(
        node_ids=tuple(f"n{index}" for index in range(count)),
        edges=tuple(edges),
        directed=True,
        node_labels=tuple(f"n{index}" for index in range(count)),
        edge_labels=tuple(None for _ in edges),
        clusters={
            "left": left,
            "right": right,
            "parent": tuple(range(count)),
        },
        cluster_parents={"left": "parent", "right": "parent"},
        ranks=tuple(ranks),
        roots=(0,),
        feedback=tuple(False for _ in edges),
        edge_weights=tuple(1.0 + (index % 3) for index in range(len(edges))),
        weight_semantics="distance_cost",
        required_primitives=frozenset({"nodes", "routes"}),
    )
    coordinates = []
    for position in range(half):
        coordinates.append([2.0 * position, 1.5])
    for position in range(count - half):
        coordinates.append([2.0 * (half + position), -1.5])
    positions = torch.tensor(coordinates, dtype=torch.float64)
    return graph, positions


_BUILDERS = {
    "chain": _chain_semantics,
    "tree": _tree_semantics,
    "layered_dag": _layered_dag_semantics,
    "clustered": _clustered_semantics,
}


def _ingest_variant(graph: GraphSemantics, positions: torch.Tensor) -> Scene:
    """Ingest one variant with chord routes and visible labels."""

    routes = tuple(
        Route(index, torch.stack((positions[source], positions[target])))
        for index, (source, target) in enumerate(graph.edges)
    )
    drawing = DrawingScene(positions, routes, ("nodes", "routes", "node_labels"))
    result = ingest(
        graph,
        drawing,
        StyleContract(),
        ObservationProfile(visible_channels=frozenset({"nodes", "routes", "node_labels"})),
    )
    assert isinstance(result, ValidScene), f"bank variant failed ingestion: {result}"
    return result.scene


def build_cell(class_name: str, scale_name: str, *, seed: int = BANK_SEED) -> BankCell:
    """Build one stratified cell deterministically.

    Parameters
    ----------
    class_name : str
        One of ``CLASS_NAMES``.
    scale_name : str
        One of ``SCALE_BANDS``.
    seed : int
        Bank seed; the per-cell stream is derived from it and the strata
        names, so cells are independent and reproducible in isolation.

    Returns
    -------
    BankCell
        Validated scenes in ``VARIANT_NAMES`` order.
    """

    count = SCALE_BANDS[scale_name]
    graph, constructed = _BUILDERS[class_name](count)
    cell_seed = int.from_bytes(
        hashlib.sha256(f"{seed}:{class_name}:{scale_name}".encode()).digest()[:4],
        "big",
    )
    generator = torch.Generator().manual_seed(cell_seed)
    unit = 2.0
    scenes = []
    for variant in VARIANT_NAMES:
        if variant == "constructed":
            positions = constructed.clone()
        elif variant == "rescaled":
            positions = constructed * 1.15
        elif variant in _NOISE_SIGMAS:
            noise = torch.randn(constructed.shape, generator=generator, dtype=torch.float64) * (
                unit * _NOISE_SIGMAS[variant]
            )
            positions = constructed + noise
        else:
            spread = constructed.abs().max()
            positions = (
                torch.rand(constructed.shape, generator=generator, dtype=torch.float64) * 2.0 - 1.0
            ) * spread
        scenes.append(_ingest_variant(graph, positions))
    return BankCell(class_name=class_name, scale_name=scale_name, scenes=tuple(scenes))


def build_bank(
    *,
    classes: Sequence[str] = CLASS_NAMES,
    scales: Sequence[str] = tuple(SCALE_BANDS),
    seed: int = BANK_SEED,
) -> Tuple[BankCell, ...]:
    """Build the stratified pilot bank.

    Parameters
    ----------
    classes : sequence[str]
        Class strata to include.
    scales : sequence[str]
        Size bands to include.
    seed : int
        Bank seed.

    Returns
    -------
    tuple[BankCell, ...]
        Cells in class-major, scale-minor order.
    """

    return tuple(
        build_cell(class_name, scale_name, seed=seed)
        for class_name in classes
        for scale_name in scales
    )


def bank_digest(cells: Sequence[BankCell]) -> str:
    """Digest the bank's positions and structure for freeze evidence.

    Parameters
    ----------
    cells : sequence[BankCell]
        Bank cells.

    Returns
    -------
    str
        Hex sha256 over every cell's strata names, edges, and position bytes.
    """

    digest = hashlib.sha256()
    for cell in cells:
        digest.update(cell.class_name.encode())
        digest.update(cell.scale_name.encode())
        for scene in cell.scenes:
            digest.update(str(scene.graph.edges).encode())
            digest.update(scene.positions.numpy().tobytes())
    return digest.hexdigest()
