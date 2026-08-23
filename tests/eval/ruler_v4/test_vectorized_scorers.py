"""Byte-identity proof batteries for the review-gated U07/U11 rewrites."""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import replace
from typing import Callable, Iterator, List, Mapping, Sequence, Tuple

import pytest
import torch

import dagua.eval.ruler_v4.edges as edge_scorers
from dagua.eval.ruler_v4.ingestion import ingest
from dagua.eval.ruler_v4.scene import (
    BoxGeometry,
    DrawingScene,
    FacetResult,
    GraphSemantics,
    ObservationProfile,
    Route,
    Scene,
    StyleContract,
    ValidScene,
)
from tests.eval.ruler_v4 import test_edges as edge_repros
from tests.eval.ruler_v4 import test_review_repros as review_repros
from tests.eval.ruler_v4.scene_bank import build_cell

_ORIGINAL_BATTERY_DIGESTS: Mapping[str, Mapping[str, int]] = {
    "U07": {
        "chain/small/0": 15457632458606232243,
        "chain/small/1": 15457632458606232243,
        "chain/small/2": 15457632458606232243,
        "chain/small/3": 15457632458606232243,
        "chain/small/4": 15457632458606232243,
        "chain/small/5": 7678898722713208920,
        "tree/small/0": 12366900145529334792,
        "tree/small/1": 12366900145529334792,
        "tree/small/2": 12366900145529334792,
        "tree/small/3": 12366900145529334792,
        "tree/small/4": 12366900145529334792,
        "tree/small/5": 1878759187743097314,
        "layered_dag/small/0": 16762222588643782406,
        "layered_dag/small/1": 16762222588643782406,
        "layered_dag/small/2": 16762222588643782406,
        "layered_dag/small/3": 16762222588643782406,
        "layered_dag/small/4": 16762222588643782406,
        "layered_dag/small/5": 16155087074124745301,
        "clustered/small/0": 15457632458606232243,
        "clustered/small/1": 15457632458606232243,
        "clustered/small/2": 15457632458606232243,
        "clustered/small/3": 15457632458606232243,
        "clustered/small/4": 15457632458606232243,
        "clustered/small/5": 10773904841176886084,
        "tree/medium/0": 9088114420855241413,
        "tree/medium/1": 9088114420855241413,
    },
    "U11": {
        "chain/small/0": 7794219267888681994,
        "chain/small/1": 7200975527765057918,
        "chain/small/2": 16196212422780242095,
        "chain/small/3": 3735137058512483680,
        "chain/small/4": 12993733983297059731,
        "chain/small/5": 10403042640911387736,
        "tree/small/0": 7567941043595894019,
        "tree/small/1": 17101318361319821185,
        "tree/small/2": 11406908958048032978,
        "tree/small/3": 14543777768219467364,
        "tree/small/4": 14535077126193554206,
        "tree/small/5": 9178400721551426798,
        "layered_dag/small/0": 743181308804139293,
        "layered_dag/small/1": 420297013962806196,
        "layered_dag/small/2": 4689497278629056508,
        "layered_dag/small/3": 13390474873582667563,
        "layered_dag/small/4": 11769578341839241002,
        "layered_dag/small/5": 7387651480238903971,
        "clustered/small/0": 11902197159327323931,
        "clustered/small/1": 16637724374517472788,
        "clustered/small/2": 12168869733823871583,
        "clustered/small/3": 10235446861437516230,
        "clustered/small/4": 7368500844737615811,
        "clustered/small/5": 3567712910036849227,
        "tree/medium/0": 17817149473593092251,
        "tree/medium/1": 17387112117542496452,
    },
}
_SIZE_SEED_DIGESTS: Mapping[str, Tuple[int, ...]] = {
    "U07": (
        3396005990594195180,
        13486203249651600114,
        12776645543701673038,
        6189586592573420951,
        13844938689633676007,
        18029185080214005738,
        16034266918722305864,
        3396005990594195180,
    ),
    "U11": (
        5191705964977090974,
        9006656863334427611,
        4902857091215116058,
        6531804921733594192,
        11512628660717821398,
        18233026032596245623,
        16852044401704073119,
        5191705964977090974,
    ),
}
_SIZE_BANDS: Tuple[Tuple[str, int], ...] = (
    ("le30", 24),
    ("31-100", 64),
    ("101-300", 192),
    ("301-1000", 800),
    ("1001-3000", 2000),
)


def _byte_tree(value: object) -> object:
    """Convert published values into a type-preserving IEEE-754 byte tree.

    Parameters
    ----------
    value : object
        Facet result field or recursively nested raw diagnostic.

    Returns
    -------
    object
        JSON-safe structure with every float represented by its exact bytes.
    """

    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return {"tensor_float": struct.pack(">d", float(value.detach())).hex()}
        return {"tensor": [_byte_tree(item) for item in value.detach().tolist()]}
    if isinstance(value, bool):
        return {"bool": value}
    if isinstance(value, int):
        return {"int": value}
    if isinstance(value, float):
        return {"float": struct.pack(">d", value).hex()}
    if isinstance(value, Mapping):
        return {"mapping": [[str(key), _byte_tree(item)] for key, item in sorted(value.items())]}
    if isinstance(value, tuple):
        return {"tuple": [_byte_tree(item) for item in value]}
    if isinstance(value, list):
        return {"list": [_byte_tree(item) for item in value]}
    if value is None or isinstance(value, str):
        return value
    raise TypeError(f"unsupported byte-tree value: {type(value).__name__}")


def _facet_digest(result: FacetResult) -> int:
    """Hash every published field of one exact facet result.

    Parameters
    ----------
    result : FacetResult
        Exact scorer output.

    Returns
    -------
    int
        First 64 SHA-256 bits of the type- and byte-preserving canonical record.
    """

    payload = {
        "state": result.state.value,
        "value": _byte_tree(result.value),
        "reason": result.reason,
        "subterms": _byte_tree(result.subterms),
        "raw": _byte_tree(result.raw),
        "temporal_headline": _byte_tree(result.temporal_headline),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


def _original_battery() -> Iterator[Tuple[str, Scene]]:
    """Yield the exact 26 scene identities used by the 71f84569 proof.

    Yields
    ------
    tuple[str, Scene]
        Stable scene id and deterministic pilot-bank scene.
    """

    for class_name in ("chain", "tree", "layered_dag", "clustered"):
        cell = build_cell(class_name, "small")
        for index, scene in enumerate(cell.scenes):
            yield f"{class_name}/small/{index}", scene
    medium = build_cell("tree", "medium")
    for index in (0, 1):
        yield f"tree/medium/{index}", medium.scenes[index]


def _size_band_scene(node_count: int, seed: int) -> Scene:
    """Build one deterministic crossing drawing with remote spectator nodes.

    Parameters
    ----------
    node_count : int
        Total nodes, including isolated spectators.
    seed : int
        Drawing variant in ``range(8)``.

    Returns
    -------
    Scene
        Validated scene with 12 crossing routes and up to 1,976 spectators.
    """

    edge_count = 12
    positions: List[Tuple[float, float]] = []
    for index in range(edge_count):
        shift = float(((seed + 3) * (index + 5)) % 7 - 3) / 8.0
        positions.append((-100.0, 10.0 * (index - 5.5) + shift))
    for index in range(edge_count):
        shift = float(((seed + 5) * (index + 2)) % 7 - 3) / 8.0
        positions.append((100.0, -10.0 * (index - 5.5) + shift))
    positions.extend(
        (10_000.0 + 3.0 * index, 20.0 * ((index + seed) % 11))
        for index in range(node_count - 2 * edge_count)
    )
    position_tensor = torch.tensor(positions, dtype=torch.float64)
    edges = tuple((index, edge_count + index) for index in range(edge_count))
    graph = GraphSemantics(
        node_ids=tuple(f"n{index}" for index in range(node_count)),
        edges=edges,
        directed=True,
        node_labels=tuple(None for _ in range(node_count)),
        edge_labels=tuple(None for _ in edges),
        feedback=tuple(False for _ in edges),
        flow_axis=(1.0, 0.0),
        required_primitives=frozenset({"nodes", "routes"}),
    )
    routes = tuple(
        Route(index, torch.stack((position_tensor[source], position_tensor[target])))
        for index, (source, target) in enumerate(edges)
    )
    drawing = DrawingScene(position_tensor, routes, ("nodes", "routes"))
    ingested = ingest(
        graph,
        drawing,
        StyleContract(),
        ObservationProfile(visible_channels=frozenset({"nodes", "routes"})),
    )
    assert isinstance(ingested, ValidScene)
    return ingested.scene


def _size_band_battery() -> Iterator[Tuple[str, Scene]]:
    """Yield 40 drawings across every frozen size band.

    Yields
    ------
    tuple[str, Scene]
        Stable size-band/seed id and validated scene. The final 16 drawings
        carry 800 or 2,000 nodes.
    """

    for band_name, node_count in _SIZE_BANDS:
        for seed in range(8):
            yield f"{band_name}/{node_count}/{seed}", _size_band_scene(node_count, seed)


def _translated_scene(scene: Scene, offset: Tuple[float, float]) -> Scene:
    """Translate every producer and derived coordinate in a validated scene.

    Parameters
    ----------
    scene : Scene
        Validated source scene.
    offset : tuple[float, float]
        Extreme-coordinate translation applied without changing scene scale.

    Returns
    -------
    Scene
        Coordinate-translated scene with the original semantic hashes and unit.
    """

    translation = torch.tensor(offset, dtype=torch.float64)

    def translate_box(box: BoxGeometry) -> BoxGeometry:
        """Translate one derived axis-aligned box.

        Parameters
        ----------
        box : BoxGeometry
            Derived primitive box.

        Returns
        -------
        BoxGeometry
            Box translated by the scene offset.
        """

        return replace(box, center=box.center + translation)

    return replace(
        scene,
        positions=scene.positions + translation,
        routes=tuple(replace(route, points=route.points + translation) for route in scene.routes),
        node_boxes=tuple(translate_box(box) for box in scene.node_boxes),
        node_label_boxes=tuple(translate_box(box) for box in scene.node_label_boxes),
        edge_label_boxes=tuple(translate_box(box) for box in scene.edge_label_boxes),
        cluster_label_boxes={
            key: translate_box(box) for key, box in scene.cluster_label_boxes.items()
        },
    )


def _scalar_crossing_pairs(
    segments: Sequence[Tuple[int, int, torch.Tensor, torch.Tensor]],
) -> List[Tuple[int, int]]:
    """Return event-bearing segment pairs through the shipped scalar predicate.

    Parameters
    ----------
    segments : sequence[tuple[int, int, torch.Tensor, torch.Tensor]]
        Flattened synthetic route segments.

    Returns
    -------
    list[tuple[int, int]]
        Scalar event pairs in nested-sweep order.
    """

    pairs: List[Tuple[int, int]] = []
    for left_index, (left_edge, _, left_start, left_end) in enumerate(segments):
        for right_index in range(left_index + 1, len(segments)):
            right_edge, _, right_start, right_end = segments[right_index]
            if (
                left_edge != right_edge
                and edge_scorers._segment_event_point(left_start, left_end, right_start, right_end)
                is not None
            ):
                pairs.append((left_index, right_index))
    return pairs


def test_vectorized_scorers_default_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the review-adopted vectorized scorers as the shipped default path."""

    assert edge_scorers.VECTORIZED_EXACT_SCORERS is True

    original_events = edge_scorers._crossing_events_vectorized
    event_calls: List[int] = []

    def record_events(scene: Scene, gamma: float) -> List[object]:
        """Record that the default U07 call reaches the adopted implementation."""

        event_calls.append(1)
        return original_events(scene, gamma)

    original_baseline = edge_scorers._route_baseline
    baseline_flags: List[bool] = []

    def record_baseline_flag(
        scene: Scene, route: Route, *, vectorized: bool = False
    ) -> Tuple[object, object]:
        """Record the U11 gate value before calling the real baseline."""

        baseline_flags.append(vectorized)
        return original_baseline(scene, route, vectorized=vectorized)

    scene = _size_band_scene(24, 0)
    monkeypatch.setattr(edge_scorers, "_crossing_events_vectorized", record_events)
    monkeypatch.setattr(edge_scorers, "_route_baseline", record_baseline_flag)
    edge_scorers.U07(scene)
    edge_scorers.U11(scene)
    assert event_calls
    assert baseline_flags and all(baseline_flags)


def test_vectorized_u07_pair_decisions_match_scalar_random_battery() -> None:
    """Pin transversal, collinear, endpoint, and degenerate U07 decisions."""

    generator = torch.Generator().manual_seed(20260823)
    segments: List[Tuple[int, int, torch.Tensor, torch.Tensor]] = []
    for index in range(96):
        points = torch.randint(-8, 9, (2, 2), generator=generator, dtype=torch.int64).to(
            torch.float64
        )
        if index % 11 == 0:
            points[1] = points[0]
        segments.append((index // 2, index % 2, points[0], points[1]))
    assert edge_scorers._crossing_candidate_pairs_vectorized(segments) == _scalar_crossing_pairs(
        segments
    )


@pytest.mark.parametrize("scorer_name", ("U07", "U11"))
def test_original_26_scene_vectorized_hex_battery(
    monkeypatch: pytest.MonkeyPatch, scorer_name: str
) -> None:
    """Match shipped hashes on the full 71f84569 26-scene battery."""

    monkeypatch.setattr(edge_scorers, "VECTORIZED_EXACT_SCORERS", True)
    scorer = getattr(edge_scorers, scorer_name)
    actual = {scene_id: _facet_digest(scorer(scene)) for scene_id, scene in _original_battery()}
    assert actual == _ORIGINAL_BATTERY_DIGESTS[scorer_name]


@pytest.mark.parametrize("scorer_name", ("U07", "U11"))
def test_40_scene_all_size_band_vectorized_hex_battery(
    monkeypatch: pytest.MonkeyPatch, scorer_name: str
) -> None:
    """Match once-banked shipped hashes through the 2,000-node band."""

    monkeypatch.setattr(edge_scorers, "VECTORIZED_EXACT_SCORERS", True)
    scorer = getattr(edge_scorers, scorer_name)
    actual = {scene_id: _facet_digest(scorer(scene)) for scene_id, scene in _size_band_battery()}
    expected = {
        f"{band_name}/{node_count}/{seed}": _SIZE_SEED_DIGESTS[scorer_name][seed]
        for band_name, node_count in _SIZE_BANDS
        for seed in range(8)
    }
    assert actual == expected


def test_extreme_coordinate_u11_vectorized_full_record_hex_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep U11 scalar/vector records identical beyond the AABB ULP threshold."""

    scenes = dict(_original_battery())
    extreme_cases = (
        ("chain/small/4", (0.0, 2.0e14)),
        ("tree/small/5", (0.0, 2.0e14)),
        ("clustered/small/4", (0.0, 2.0e14)),
        ("tree/medium/0", (0.0, 2.0e14)),
        ("tree/medium/1", (0.0, 2.0e14)),
        ("chain/small/4", (0.0, 3.0e14)),
        ("tree/small/5", (0.0, 3.0e14)),
        ("tree/small/4", (1.3e15, 0.0)),
    )
    translated = [_translated_scene(scenes[scene_id], offset) for scene_id, offset in extreme_cases]
    monkeypatch.setattr(edge_scorers, "VECTORIZED_EXACT_SCORERS", False)
    scalar = [_facet_digest(edge_scorers.U11(scene)) for scene in translated]
    monkeypatch.setattr(edge_scorers, "VECTORIZED_EXACT_SCORERS", True)
    vectorized = [_facet_digest(edge_scorers.U11(scene)) for scene in translated]
    assert vectorized == scalar


@pytest.mark.parametrize(
    "repro",
    (
        edge_repros.test_u07_crossing_free_routes_have_exact_zero_defect,
        edge_repros.test_u07_remote_perpendicular_crossing_pins_guarded_normalization,
        edge_repros.test_u11_straight_route_hits_all_five_anchored_zeros,
        edge_repros.test_u11_terminal_disk_clears_coincident_nonterminal_box,
        edge_repros.test_u11_tangentless_terminal_pair_is_not_best_case,
        edge_repros.test_u11_distant_tangentless_terminal_pair_earns_its_separation,
        review_repros.test_u11_scores_plain_declared_rank_dag,
        review_repros.test_u11_through_path_is_not_merge_identity,
        review_repros.test_u11_angle_factor_is_monotone_on_the_oriented_range,
        review_repros.test_u11_grid_of_paths_scores_interior_nodes_clean,
    ),
)
def test_banked_u07_u11_repros_on_vectorized_path(
    monkeypatch: pytest.MonkeyPatch, repro: Callable[[], None]
) -> None:
    """Run every banked U07/U11 exact-value repro through the review path."""

    monkeypatch.setattr(edge_scorers, "VECTORIZED_EXACT_SCORERS", True)
    repro()
