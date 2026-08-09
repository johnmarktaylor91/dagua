"""Tests for the planar-certificate contest arm (sprint2 W1-B)."""

from __future__ import annotations

import hashlib
import importlib
from typing import Any, Dict, cast

import pytest
import torch

from dagua.config import LayoutConfig
from dagua.layout.graph_classify import classify_graph
from dagua.layout.ops.pipelines.native_planar_arm import (
    PLANAR_ARM_MAX_NODES,
    PLANAR_ARM_OUTER_FACES,
    _select_outer_faces,
    build_planar_arm_candidates,
    planar_arm_admitted,
    planar_candidate_requires_certificate,
)
from dagua.layout.ops.planar_polish import exact_crossing_count
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState


def _wheel_problem(spokes: int = 8) -> LayoutProblem:
    """Return a wheel-graph problem (planar, 3-connected, single component)."""
    edges = []
    for spoke in range(1, spokes + 1):
        edges.append((0, spoke))
        edges.append((spoke, 1 + spoke % spokes))
    edge_index = torch.tensor(sorted(set(edges)), dtype=torch.long).t().contiguous()
    num_nodes = spokes + 1
    structure = classify_graph(edge_index, num_nodes)
    return LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=torch.full((num_nodes, 2), 20.0),
        structure=cast(Any, structure),
        seed=42,
    )


def _k5_problem() -> LayoutProblem:
    """Return a K5 problem (non-planar: the gate must stay closed)."""
    edges = [(u, v) for u in range(5) for v in range(u + 1, 5)]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    structure = classify_graph(edge_index, 5)
    return LayoutProblem(
        edge_index=edge_index,
        num_nodes=5,
        node_sizes=torch.full((5, 2), 20.0),
        structure=cast(Any, structure),
        seed=42,
    )


def _grid_dag_problem(width: int = 3, height: int = 3) -> LayoutProblem:
    """Return a planar acyclic grid problem (edges point right and down)."""
    edges = []
    for y in range(height):
        for x in range(width):
            node = y * width + x
            if x + 1 < width:
                edges.append((node, node + 1))
            if y + 1 < height:
                edges.append((node, node + width))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    num_nodes = width * height
    structure = classify_graph(edge_index, num_nodes)
    return LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=torch.full((num_nodes, 2), 20.0),
        structure=cast(Any, structure),
        seed=42,
    )


def _k33_dag_problem() -> LayoutProblem:
    """Return a declared-directed acyclic K3,3 problem (non-planar gate row)."""
    edges = [(u, v) for u in range(3) for v in range(3, 6)]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    structure = classify_graph(edge_index, 6)
    return LayoutProblem(
        edge_index=edge_index,
        num_nodes=6,
        node_sizes=torch.full((6, 2), 20.0),
        structure=cast(Any, structure),
        seed=42,
    )


def _planar_family_must_certify(name: str) -> bool:
    """Return whether a registered contest candidate name claims the certificate."""
    return name.startswith("planar_") and not name.startswith("planar_seeded_stress")


def _sha256(tensor: torch.Tensor) -> str:
    """Return the SHA-256 of a tensor's float32 CPU bytes."""
    return hashlib.sha256(
        tensor.detach().to(device="cpu", dtype=torch.float32).numpy().tobytes()
    ).hexdigest()


def test_gate_opens_on_exact_planar_and_closes_on_k5() -> None:
    """Gate requires exact planarity with a cached embedding."""
    wheel = _wheel_problem()
    wheel_structure = cast(Any, wheel.structure)
    assert wheel_structure is not None and wheel_structure.is_planar is True
    assert planar_arm_admitted(wheel)
    k5 = _k5_problem()
    k5_structure = cast(Any, k5.structure)
    assert k5_structure is not None and k5_structure.is_planar is False
    assert not planar_arm_admitted(k5)


def test_gate_closes_without_embedding_or_above_cap() -> None:
    """Euler-hint-only planarity (no embedding) and oversize rows fail closed."""
    from types import SimpleNamespace

    wheel = _wheel_problem()
    hint_only = SimpleNamespace(is_planar=True, planar_embedding=None, num_components=1)
    problem = LayoutProblem(
        edge_index=wheel.edge_index,
        num_nodes=wheel.num_nodes,
        structure=cast(Any, hint_only),
        seed=42,
    )
    assert not planar_arm_admitted(problem)
    oversized = LayoutProblem(
        edge_index=wheel.edge_index,
        num_nodes=PLANAR_ARM_MAX_NODES + 1,
        structure=wheel.structure,
        seed=42,
    )
    assert not planar_arm_admitted(oversized)


def test_builder_emits_parity_floor_and_zero_crossing_polish() -> None:
    """The arm yields FPP raw + polished variants with exact zero crossings."""
    problem = _wheel_problem()
    candidates = build_planar_arm_candidates(problem, node_sep=40.0)
    assert "planar_fpp" in candidates
    assert "planar_fpp_polished" in candidates
    for name, pos in candidates.items():
        assert bool(torch.isfinite(pos).all().item()), name
        if planar_candidate_requires_certificate(name):
            assert exact_crossing_count(pos, problem.edge_index) == 0, name


def test_builder_is_deterministic() -> None:
    """Two builder runs produce byte-identical candidates."""
    problem = _wheel_problem()
    first = build_planar_arm_candidates(problem, node_sep=40.0)
    second = build_planar_arm_candidates(problem, node_sep=40.0)
    assert sorted(first) == sorted(second)
    for name in first:
        assert torch.equal(first[name], second[name]), name


def test_outer_face_matrix_covers_both_embeddings() -> None:
    """W1B-2: 3-4 outer faces x FPP AND Schnyder drawers, all certified.

    The spec's matrix (PLAN_RANKED section 4, O19) requires the outer-face
    enumeration crossed with BOTH embedding drawers. FPP cells that duplicate
    the default-face base drawing byte-identically are deduplicated, so the
    base + face cells must jointly cover at least three distinct faces per
    drawer.
    """
    problem = _wheel_problem()
    candidates = build_planar_arm_candidates(problem, node_sep=40.0)
    fpp_cells = [name for name in candidates if name.startswith("planar_fpp_f")]
    schnyder_cells = [name for name in candidates if name.startswith("planar_schnyder_f")]
    assert len(fpp_cells) + ("planar_fpp_polished" in candidates) >= 3, sorted(candidates)
    assert len(schnyder_cells) >= 3, sorted(candidates)
    matrix = fpp_cells + schnyder_cells
    for name in matrix:
        assert exact_crossing_count(candidates[name], problem.edge_index) == 0, name
    for index, first in enumerate(matrix):
        for second in matrix[index + 1 :]:
            assert not torch.equal(candidates[first], candidates[second]), (first, second)


def test_outer_face_matrix_covers_bridge_rows() -> None:
    """The matrix still produces cells when every embedding face is non-simple."""
    edges = [(0, 1), (1, 2), (0, 2), (0, 3), (1, 4)]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    structure = classify_graph(edge_index, 5)
    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=5,
        node_sizes=torch.full((5, 2), 20.0),
        structure=cast(Any, structure),
        seed=42,
    )
    assert planar_arm_admitted(problem)
    candidates = build_planar_arm_candidates(problem, node_sep=40.0)
    assert any(name.startswith("planar_fpp_f") for name in candidates), sorted(candidates)
    assert any(name.startswith("planar_schnyder_f") for name in candidates), sorted(candidates)


def test_select_outer_faces_fills_to_limit_from_stable_ranking() -> None:
    """Face selection tops up from the size ranking past the representatives."""
    faces = [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [0, 1, 2, 3, 4, 5]]
    degrees = [3] * 6
    selected = _select_outer_faces(faces, degrees, PLANAR_ARM_OUTER_FACES)
    assert len(selected) == min(PLANAR_ARM_OUTER_FACES, len(faces))
    canonical = {tuple(sorted(face)) for face in selected}
    assert len(canonical) == len(selected)


def test_gate_closed_row_never_builds_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    """On a non-planar row the arm code never runs (byte-inert contract)."""
    from dagua.layout.ops.pipelines.native_undirected import (
        layout_native_undirected_portfolio,
    )

    arm_module = importlib.import_module("dagua.layout.ops.pipelines.native_planar_arm")

    def _must_not_run(*args: object, **kwargs: object) -> dict[str, torch.Tensor]:
        raise AssertionError("planar arm built candidates on a gate-closed row")

    monkeypatch.setattr(arm_module, "build_planar_arm_candidates", _must_not_run)
    problem = _k5_problem()
    result = layout_native_undirected_portfolio(
        problem,
        SolveState(),
        RuntimeContext(),
        LayoutConfig(seed=42),
    )
    assert bool(torch.isfinite(result).all().item())


def test_planar_arm_fires_inside_undirected_contest(monkeypatch: pytest.MonkeyPatch) -> None:
    """On a gated planar row the contest builds and registers the arm."""
    from dagua.layout.ops.pipelines.native_undirected import (
        layout_native_undirected_portfolio,
    )

    arm_module = importlib.import_module("dagua.layout.ops.pipelines.native_planar_arm")
    real_builder = arm_module.build_planar_arm_candidates
    seen: list[str] = []

    def _recording_builder(*args: object, **kwargs: object) -> dict[str, torch.Tensor]:
        candidates = real_builder(*args, **kwargs)
        seen.extend(sorted(candidates))
        return candidates

    monkeypatch.setattr(arm_module, "build_planar_arm_candidates", _recording_builder)
    problem = _wheel_problem()
    result = layout_native_undirected_portfolio(
        problem,
        SolveState(),
        RuntimeContext(),
        LayoutConfig(seed=42),
    )
    assert bool(torch.isfinite(result).all().item())
    assert "planar_fpp" in seen
    assert any(name.endswith("_polished") for name in seen)


def _capture_registered_positions(
    monkeypatch: pytest.MonkeyPatch,
) -> Dict[str, torch.Tensor]:
    """Route marketplace telemetry into a dict capturing the candidate registry."""
    native_undirected = importlib.import_module("dagua.layout.ops.pipelines.native_undirected")
    captured: Dict[str, torch.Tensor] = {}
    real_telemetry = native_undirected._log_marketplace_telemetry

    def _capturing_telemetry(*args: object, **kwargs: object) -> None:
        positions = kwargs.get("positions")
        if isinstance(positions, dict):
            captured.update(positions)
        real_telemetry(*args, **kwargs)

    monkeypatch.setattr(native_undirected, "_log_marketplace_telemetry", _capturing_telemetry)
    return captured


def test_undirected_contest_registers_only_certified_planar_variants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """W1B-1: no registered planar-family variant may carry crossings.

    Reproduces the reviewer's case: the overlap projectors run AFTER the
    guarded polish, and on the wheel fixture their legacy/convergent outputs
    carry 8 and 11 crossings on the parent commit. Every registered
    planar-family candidate (any suffix, seeded-stress exempt) must now hold
    the exact zero-crossing certificate.
    """
    from dagua.layout.ops.pipelines.native_undirected import (
        layout_native_undirected_portfolio,
    )

    captured = _capture_registered_positions(monkeypatch)
    problem = _wheel_problem()
    layout_native_undirected_portfolio(
        problem,
        SolveState(),
        RuntimeContext(),
        LayoutConfig(seed=42),
    )
    planar_names = [name for name in captured if _planar_family_must_certify(name)]
    assert planar_names, sorted(captured)
    for name in planar_names:
        crossings = exact_crossing_count(captured[name], problem.edge_index)
        assert crossings == 0, f"{name} registered with {crossings} crossings"


def test_directed_contest_registers_only_certified_planar_variants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """W1B-1 + W1B-3: the directed contest registers the arm, all certified."""
    from dagua.layout.ops.pipelines.native_directed import (
        layout_native_directed_portfolio,
    )

    captured = _capture_registered_positions(monkeypatch)
    problem = _grid_dag_problem()
    assert planar_arm_admitted(problem)
    layout_native_directed_portfolio(
        problem,
        SolveState(),
        RuntimeContext(),
        LayoutConfig(seed=42),
    )
    planar_names = [name for name in captured if _planar_family_must_certify(name)]
    assert planar_names, sorted(captured)
    for name in planar_names:
        crossings = exact_crossing_count(captured[name], problem.edge_index)
        assert crossings == 0, f"{name} registered with {crossings} crossings"


# Gate-closed golden bytes, captured on the pre-packet parent 34c42d00 (and
# verified identical on fcb85d37): the planar arm must stay byte-inert on
# rows where the structural gate is closed.
_GOLDEN_UNDIRECTED_K5_SHA256 = (
    "1ec653585599b72ec4df8c04bd84f41545f724b8f622421da2f36ea81eb2e982"  # pragma: allowlist secret
)
_GOLDEN_DIRECTED_K33_SHA256 = (
    "ba5312d450473a42cdee6cfb4a771281f50f577f985dc57968e875ad17e3771d"  # pragma: allowlist secret
)


def test_gate_closed_undirected_row_byte_identical_golden() -> None:
    """Non-planar undirected K5 reproduces the pre-packet bytes exactly."""
    from dagua.layout.ops.pipelines.native_undirected import (
        layout_native_undirected_portfolio,
    )

    problem = _k5_problem()
    result = layout_native_undirected_portfolio(
        problem,
        SolveState(),
        RuntimeContext(),
        LayoutConfig(seed=42),
    )
    assert _sha256(result) == _GOLDEN_UNDIRECTED_K5_SHA256


def test_gate_closed_directed_row_byte_identical_golden() -> None:
    """Non-planar directed K3,3 reproduces the pre-packet bytes exactly."""
    from dagua.layout.ops.pipelines.native_directed import (
        layout_native_directed_portfolio,
    )

    problem = _k33_dag_problem()
    result = layout_native_directed_portfolio(
        problem,
        SolveState(),
        RuntimeContext(),
        LayoutConfig(seed=42),
    )
    assert _sha256(result) == _GOLDEN_DIRECTED_K33_SHA256
