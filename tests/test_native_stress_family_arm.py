"""Tests for the stress-family contest arms (sprint2 W2-2)."""

from __future__ import annotations

import hashlib
import importlib
from typing import Any, cast

import pytest
import torch

from dagua.config import LayoutConfig
from dagua.layout.graph_classify import classify_graph
from dagua.layout.ops.pipelines.native_stress_family_arm import (
    LOW_LAYERING_MIN_AVG_LAYER_WIDTH,
    MAXENT_SEED_BANK,
    STRESS_FAMILY_MAX_NODES,
    STRESS_SGD_SEED_BANK,
    _is_connected,
    build_stress_family_candidates,
    stress_family_arm_admitted,
    stress_family_candidate_prefix,
    stress_family_directed_admitted,
    stress_family_parity_floor,
)
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState


def _problem(edges: list[tuple[int, int]], num_nodes: int) -> LayoutProblem:
    """Return a LayoutProblem with real classifier output for the edges."""
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    structure = classify_graph(edge_index, num_nodes)
    return LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=torch.full((num_nodes, 2), 20.0),
        structure=cast(Any, structure),
        seed=42,
    )


def _chords_problem() -> LayoutProblem:
    """Return a small connected undirected-style row (gate open)."""
    edges = [(i, i + 1) for i in range(11)] + [(0, 5), (3, 9), (2, 7)]
    return _problem(edges, 12)


def _wide_dag_problem() -> LayoutProblem:
    """Return a shallow two-layer DAG (K3,3 orientation: avg width 3.0)."""
    return _problem([(u, v) for u in range(3) for v in range(3, 6)], 6)


def _chain_dag_problem() -> LayoutProblem:
    """Return a deep chain DAG (avg layer width 1.0: directed gate closed)."""
    return _problem([(i, i + 1) for i in range(7)], 8)


def _disjoint_k5s_problem() -> LayoutProblem:
    """Return two disjoint K5s: the classifier component fast path lies here."""
    edges = [(u, v) for u in range(5) for v in range(u + 1, 5)]
    edges += [(5 + u, 5 + v) for u in range(5) for v in range(u + 1, 5)]
    return _problem(edges, 10)


def _k5_tournament_problem() -> LayoutProblem:
    """Return a K5 tournament: connected but deep chain-like layering."""
    return _problem([(u, v) for u in range(5) for v in range(u + 1, 5)], 5)


def _sha256(tensor: torch.Tensor) -> str:
    """Return the SHA-256 of a tensor's float32 CPU bytes."""
    return hashlib.sha256(
        tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy().tobytes()
    ).hexdigest()


def test_is_connected_exact_on_dense_disconnected_inputs() -> None:
    """Union-find sees through the classifier's E > N-1 component fast path."""
    assert _is_connected(_chords_problem().edge_index, 12)
    assert not _is_connected(_disjoint_k5s_problem().edge_index, 10)
    assert not _is_connected(torch.tensor([(0, 1)], dtype=torch.long).t(), 3)
    assert not _is_connected(torch.zeros((2, 0), dtype=torch.long), 2)


def test_gate_opens_on_small_connected_rows() -> None:
    """The shared gate admits small connected rows inside the band."""
    assert stress_family_arm_admitted(_chords_problem())
    assert stress_family_arm_admitted(_wide_dag_problem())


def test_gate_closes_on_disconnected_missing_structure_and_size() -> None:
    """Disconnected, unclassified, tiny, and oversized rows all fail closed."""
    assert not stress_family_arm_admitted(_disjoint_k5s_problem())
    sparse_disconnected = _problem([(0, 1), (1, 2)], 5)
    assert not stress_family_arm_admitted(sparse_disconnected)
    triangle = _problem([(0, 1), (1, 2), (0, 2)], 3)
    assert not stress_family_arm_admitted(triangle)
    unclassified = _chords_problem()
    unclassified.structure = None
    assert not stress_family_arm_admitted(unclassified)
    band_break = STRESS_FAMILY_MAX_NODES + 1
    long_path = _problem([(i, i + 1) for i in range(band_break - 1)], band_break)
    assert not stress_family_arm_admitted(long_path)


def test_directed_gate_requires_low_layering() -> None:
    """Wide/cyclic layering opens the directed gate; deep chains stay closed."""
    wide = _wide_dag_problem()
    assert float(getattr(wide.structure, "avg_layer_width", 0.0)) >= (
        LOW_LAYERING_MIN_AVG_LAYER_WIDTH
    )
    assert stress_family_directed_admitted(wide)
    assert not stress_family_directed_admitted(_chain_dag_problem())
    assert not stress_family_directed_admitted(_k5_tournament_problem())
    cyclic = _problem([(0, 1), (1, 2), (2, 3), (3, 0)], 4)
    assert not bool(getattr(cyclic.structure, "is_directed_acyclic", True))
    assert stress_family_directed_admitted(cyclic)


def test_builder_emits_expected_candidates_deterministically() -> None:
    """The builder emits the frozen candidate set, byte-identical across runs."""
    problem = _chords_problem()
    first = build_stress_family_candidates(problem, node_sep=40.0)
    second = build_stress_family_candidates(problem, node_sep=40.0)
    expected = {f"stress_sgd_k_seed{seed}" for seed in STRESS_SGD_SEED_BANK}
    expected |= {f"maxent_stress_seed{seed}" for seed in MAXENT_SEED_BANK}
    expected.add("elk_stress_arm")
    assert set(first) == expected
    assert set(second) == expected
    for name, pos in first.items():
        assert pos.shape == (12, 2)
        assert bool(torch.isfinite(pos).all().item())
        assert torch.equal(pos, second[name]), name


def test_candidate_names_never_claim_sgd2() -> None:
    """D2 staged call: the arm is stress_sgd_k, never conflated with sgd2."""
    problem = _chords_problem()
    candidates = build_stress_family_candidates(problem, node_sep=40.0)
    for name in candidates:
        assert "sgd2" not in name, name
        assert stress_family_candidate_prefix(name) is not None, name
    parity = [name for name in candidates if stress_family_parity_floor(name)]
    assert sorted(parity) == [
        "elk_stress_arm",
        f"maxent_stress_seed{MAXENT_SEED_BANK[0]}",
        f"stress_sgd_k_seed{STRESS_SGD_SEED_BANK[0]}",
    ]


def test_candidates_rescaled_into_node_units() -> None:
    """Unit-scale pipeline output is similarity-rescaled for the contest.

    stress-SGD natively emits ~unit-length edges, which the shared degeneracy
    guard would reject against 20-point node boxes (the W1-A t-FDP lesson).
    After the rescale the median edge must sit at the node-box diagonal plus
    node_sep, far above the raw unit scale.
    """
    problem = _chords_problem()
    candidates = build_stress_family_candidates(problem, node_sep=40.0)
    pos = candidates[f"stress_sgd_k_seed{STRESS_SGD_SEED_BANK[0]}"]
    src, dst = problem.edge_index[0], problem.edge_index[1]
    median_edge = float((pos[src] - pos[dst]).norm(dim=1).median().item())
    expected = float(torch.full((2,), 20.0).norm().item()) + 40.0
    assert median_edge == pytest.approx(expected, rel=0.05)


def test_marketplace_family_labels() -> None:
    """Registered variants roll up to the three declared telemetry families."""
    from dagua.layout.ops.pipelines.native_undirected import _marketplace_family

    assert _marketplace_family("stress_sgd_k_seed42") == "stress_sgd_k"
    assert _marketplace_family("stress_sgd_k_seed1379_raw") == "stress_sgd_k"
    assert _marketplace_family("maxent_stress_seed7_prism") == "maxent_stress"
    assert _marketplace_family("elk_stress_arm_convergent") == "elk_stress_arm"


def test_gate_closed_undirected_row_never_builds(monkeypatch: pytest.MonkeyPatch) -> None:
    """On a disconnected row the arm code never runs (byte-inert contract)."""
    from dagua.layout.ops.pipelines.native_undirected import (
        layout_native_undirected_portfolio,
    )

    arm_module = importlib.import_module("dagua.layout.ops.pipelines.native_stress_family_arm")

    def _must_not_run(*args: object, **kwargs: object) -> dict[str, torch.Tensor]:
        raise AssertionError("stress-family arm built candidates on a gate-closed row")

    monkeypatch.setattr(arm_module, "build_stress_family_candidates", _must_not_run)
    result = layout_native_undirected_portfolio(
        _disjoint_k5s_problem(),
        SolveState(),
        RuntimeContext(),
        LayoutConfig(seed=42),
    )
    assert bool(torch.isfinite(result).all().item())


def test_gate_closed_directed_row_never_builds(monkeypatch: pytest.MonkeyPatch) -> None:
    """On a deep-chain DAG the directed contest never runs the arm."""
    from dagua.layout.ops.pipelines.native_directed import (
        layout_native_directed_portfolio,
    )

    arm_module = importlib.import_module("dagua.layout.ops.pipelines.native_stress_family_arm")

    def _must_not_run(*args: object, **kwargs: object) -> dict[str, torch.Tensor]:
        raise AssertionError("stress-family arm built candidates on a gate-closed row")

    monkeypatch.setattr(arm_module, "build_stress_family_candidates", _must_not_run)
    result = layout_native_directed_portfolio(
        _chain_dag_problem(),
        SolveState(),
        RuntimeContext(),
        LayoutConfig(seed=42),
    )
    assert bool(torch.isfinite(result).all().item())


def test_arm_fires_inside_undirected_contest(monkeypatch: pytest.MonkeyPatch) -> None:
    """On a gated undirected row the contest builds and registers the arm."""
    from dagua.layout.ops.pipelines.native_undirected import (
        layout_native_undirected_portfolio,
    )

    arm_module = importlib.import_module("dagua.layout.ops.pipelines.native_stress_family_arm")
    real_builder = arm_module.build_stress_family_candidates
    seen: list[str] = []

    def _recording_builder(*args: object, **kwargs: object) -> dict[str, torch.Tensor]:
        candidates = real_builder(*args, **kwargs)
        seen.extend(sorted(candidates))
        return candidates

    monkeypatch.setattr(arm_module, "build_stress_family_candidates", _recording_builder)
    result = layout_native_undirected_portfolio(
        _chords_problem(),
        SolveState(),
        RuntimeContext(),
        LayoutConfig(seed=42),
    )
    assert bool(torch.isfinite(result).all().item())
    assert f"stress_sgd_k_seed{STRESS_SGD_SEED_BANK[0]}" in seen
    assert "elk_stress_arm" in seen


def test_arm_fires_inside_directed_contest(monkeypatch: pytest.MonkeyPatch) -> None:
    """On a low-layering DAG the directed contest builds and registers the arm."""
    from dagua.layout.ops.pipelines.native_directed import (
        layout_native_directed_portfolio,
    )

    arm_module = importlib.import_module("dagua.layout.ops.pipelines.native_stress_family_arm")
    real_builder = arm_module.build_stress_family_candidates
    seen: list[str] = []

    def _recording_builder(*args: object, **kwargs: object) -> dict[str, torch.Tensor]:
        candidates = real_builder(*args, **kwargs)
        seen.extend(sorted(candidates))
        return candidates

    monkeypatch.setattr(arm_module, "build_stress_family_candidates", _recording_builder)
    problem = _wide_dag_problem()
    assert stress_family_directed_admitted(problem)
    result = layout_native_directed_portfolio(
        problem,
        SolveState(),
        RuntimeContext(),
        LayoutConfig(seed=42),
    )
    assert bool(torch.isfinite(result).all().item())
    assert f"maxent_stress_seed{MAXENT_SEED_BANK[0]}" in seen


# Gate-closed golden bytes, captured on the pre-packet parent 0ee2db36: rows
# where the stress-family gate is closed must stay byte-identical. Both
# fixtures are also closed for the W1-A/W1-B arms, so the goldens pin the
# whole gate-closed path, not a lucky overlap.
_GOLDEN_UNDIRECTED_DISJOINT_K5S_SHA256 = (
    "5d7078d63636835b3d866a4dc6613c2936aa821737515f222fc9a343ed5c190f"  # pragma: allowlist secret
)
_GOLDEN_DIRECTED_K5_TOURNAMENT_SHA256 = (
    "ed8454005c92a5a836592fe3f79592c0c399fbf836db7106010580ab0338f278"  # pragma: allowlist secret
)


def test_gate_closed_undirected_row_byte_identical_golden() -> None:
    """Disconnected undirected row reproduces the pre-packet bytes exactly."""
    from dagua.layout.ops.pipelines.native_undirected import (
        layout_native_undirected_portfolio,
    )

    result = layout_native_undirected_portfolio(
        _disjoint_k5s_problem(),
        SolveState(),
        RuntimeContext(),
        LayoutConfig(seed=42),
    )
    assert _sha256(result) == _GOLDEN_UNDIRECTED_DISJOINT_K5S_SHA256


def test_gate_closed_directed_row_byte_identical_golden() -> None:
    """Deep-layered directed row reproduces the pre-packet bytes exactly."""
    from dagua.layout.ops.pipelines.native_directed import (
        layout_native_directed_portfolio,
    )

    result = layout_native_directed_portfolio(
        _k5_tournament_problem(),
        SolveState(),
        RuntimeContext(),
        LayoutConfig(seed=42),
    )
    assert _sha256(result) == _GOLDEN_DIRECTED_K5_TOURNAMENT_SHA256
