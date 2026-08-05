"""Regression tests for GRIP, omega, and tidy reference adapters."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import torch

from dagua.eval.competitors import get_competitor, native_reference_competitor
from dagua.eval.variants import base_pairings, variant_pairings
from dagua.graph import DaguaGraph


def _edge_index(edges: list[tuple[int, int]]) -> torch.Tensor:
    """Build an edge-index tensor.

    Parameters
    ----------
    edges : list[tuple[int, int]]
        Directed edge pairs.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, E]``.
    """
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def test_grip_omega_tidy_references_are_registered_and_paired() -> None:
    """Reference and reimplementation competitors should be paired both ways."""
    expected = {
        "grip_reimpl": "grip_reference",
        "omega_reimpl": "omega_reference",
        "tidy_reimpl": "tidy_reference",
    }
    base = base_pairings()
    variants = variant_pairings()
    for reimplementation, reference in expected.items():
        assert get_competitor(reimplementation) is not None
        assert get_competitor(reference) is not None
        assert base[reimplementation] == [reference]
        variant_name = f"{reimplementation}_default"
        original_name = f"{reference}__for__{variant_name}"
        assert variants[variant_name] == [original_name]
        assert variants[original_name] == [variant_name]


@pytest.mark.parametrize(
    ("name", "edge_index", "num_nodes", "variant_params"),
    [
        (
            "grip_reference",
            _edge_index([(0, 1), (1, 2), (2, 3)]),
            4,
            {"rounds": 2, "final_rounds": 2, "init_vertices": 3, "dim": 2},
        ),
        (
            "omega_reference",
            _edge_index([(0, 1), (1, 2), (2, 3), (0, 2)]),
            4,
            {"k": 2, "sgd_iterations": 2, "unit_edge_length": 1.0},
        ),
        (
            "tidy_reference",
            _edge_index([(0, 1), (0, 2), (1, 3)]),
            4,
            {"parent_child_margin": 7.0, "peer_margin": 5.0},
        ),
    ],
)
def test_grip_omega_tidy_reference_smoke(
    name: str,
    edge_index: torch.Tensor,
    num_nodes: int,
    variant_params: dict[str, float | int],
) -> None:
    """Built native references should return finite coordinates."""
    competitor = get_competitor(name)
    assert competitor is not None
    if not competitor.available():
        pytest.skip(f"{name} binary is not available")
    node_sizes = (
        torch.full((num_nodes, 2), 10.0, dtype=torch.float64) if name == "tidy_reference" else None
    )
    graph = DaguaGraph.from_edge_index(edge_index, num_nodes, node_sizes=node_sizes)
    result = competitor.layout_with_variant(graph, seed=11, variant_params=variant_params)
    assert result.error is None
    assert result.pos is not None
    assert result.pos.shape == (num_nodes, 2)
    assert torch.isfinite(result.pos).all()


def _fake_tidy_run(command: list[str], timeout: float) -> "subprocess.CompletedProcess[str]":
    """Emulate the tidy binary: echo POSITIONS for the input file's node ids.

    Parameters
    ----------
    command : list[str]
        Adapter command vector; ``command[2]`` is the component input path.
    timeout : float
        Ignored subprocess timeout.

    Returns
    -------
    subprocess.CompletedProcess[str]
        Fake completed process with per-node POSITIONS stdout.
    """
    del timeout
    input_lines = Path(command[2]).read_text().splitlines()
    node_ids = [int(line.split()[0]) for line in input_lines[1:]]
    stdout = f"POSITIONS {len(node_ids)}\n" + "\n".join(
        f"{node} {float(node)} {2.0 * node}" for node in node_ids
    )
    return subprocess.CompletedProcess(args=command, returncode=0, stdout=stdout, stderr="")


def test_tidy_reference_rootless_cyclic_graph_is_error_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rootless (cyclic) graph must be an explicit error, not silent zeros."""
    monkeypatch.setattr(
        native_reference_competitor,
        "_resolve_binary",
        lambda *args, **kwargs: Path("/nonexistent/tidy_fake"),
    )
    competitor = native_reference_competitor.TidyReferenceCompetitor()
    graph = DaguaGraph.from_edge_index(_edge_index([(0, 1), (1, 2), (2, 0)]), 3)

    result = competitor.layout(graph)

    assert result.pos is None
    assert result.error is not None
    assert "forest-like" in result.error


def test_tidy_reference_partial_cycle_is_error_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cycle nodes unreachable from any root must not stay silently at (0, 0)."""
    monkeypatch.setattr(
        native_reference_competitor,
        "_resolve_binary",
        lambda *args, **kwargs: Path("/nonexistent/tidy_fake"),
    )
    competitor = native_reference_competitor.TidyReferenceCompetitor()
    # Component A: rooted tree 0 -> 1. Component B: 2-cycle 2 <-> 3 (rootless).
    graph = DaguaGraph.from_edge_index(_edge_index([(0, 1), (2, 3), (3, 2)]), 4)

    result = competitor.layout(graph)

    assert result.pos is None
    assert result.error is not None
    assert "forest-like" in result.error


def test_tidy_reference_multi_root_forest_covers_all_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multi-root forests must lay out every component instead of erroring.

    The adapter previously parsed each component's stdout against the WHOLE
    graph's node range, so the first component of any two-root forest raised
    ``reference omitted coordinates`` and the row became an error.
    """
    monkeypatch.setattr(
        native_reference_competitor,
        "_resolve_binary",
        lambda *args, **kwargs: Path("/nonexistent/tidy_fake"),
    )
    monkeypatch.setattr(native_reference_competitor, "_run_subprocess", _fake_tidy_run)
    competitor = native_reference_competitor.TidyReferenceCompetitor()
    # Two trees: {0 -> 1, 0 -> 2} and {3 -> 4}.
    graph = DaguaGraph.from_edge_index(_edge_index([(0, 1), (0, 2), (3, 4)]), 5)

    result = competitor.layout(graph)

    assert result.error is None
    assert result.pos is not None
    assert result.pos.shape == (5, 2)
    assert torch.isfinite(result.pos).all()
    # Second component must have been placed (its y values come from the fake
    # binary, which never emits (0, 0) for node 4).
    assert result.pos[4, 1].item() == pytest.approx(8.0)


@pytest.mark.skipif(
    not native_reference_competitor.TidyReferenceCompetitor().available(),
    reason="tidy reference binary is not available",
)
def test_tidy_reference_multi_root_forest_real_binary() -> None:
    """The real tidy binary should succeed on a two-root forest."""
    competitor = native_reference_competitor.TidyReferenceCompetitor()
    node_sizes = torch.full((5, 2), 10.0, dtype=torch.float64)
    graph = DaguaGraph.from_edge_index(
        _edge_index([(0, 1), (0, 2), (3, 4)]), 5, node_sizes=node_sizes
    )

    result = competitor.layout(graph)

    assert result.error is None
    assert result.pos is not None
    assert result.pos.shape == (5, 2)
    assert torch.isfinite(result.pos).all()
    # Components must not overlap at the origin: node 3's tree is offset.
    assert result.pos[3, 0].item() != pytest.approx(result.pos[0, 0].item())
