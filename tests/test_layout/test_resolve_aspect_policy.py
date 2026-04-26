"""Regression tests for topology-aware native aspect policy."""

from __future__ import annotations

import pytest
import torch

from dagua.config import LayoutConfig
from dagua.eval.graphs import (
    _make_hexagonal_lattice_graph,
    _make_sierpinski_graph,
    _random_dag,
    make_org_chart,
)
from dagua.graph import DaguaGraph
from dagua.layout import layout
from dagua.layout.graph_classify import classify_graph
from dagua.layout.resolve import prepare_pipeline_config


def _resolved_config(graph: DaguaGraph) -> LayoutConfig:
    """Return the native pipeline config resolved for ``graph``.

    Parameters
    ----------
    graph : DaguaGraph
        Graph to classify and resolve.

    Returns
    -------
    LayoutConfig
        Shallow config copy with ``_dagua_native_*`` private attrs populated.
    """
    return prepare_pipeline_config(
        config=LayoutConfig(algorithm="dagua_native", seed=42, device="cpu"),
        num_nodes=graph.num_nodes,
        edge_index=graph.edge_index,
        device="cpu",
        layer_assignments=None,
        prebuilt_layer_index=None,
        graph_structure=None,
        skip_classification=False,
    )


def _bbox_aspect(pos: torch.Tensor) -> float:
    """Return the width/height aspect ratio of a position tensor.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor shaped ``[N, 2]``.

    Returns
    -------
    float
        Bounding-box width divided by height.
    """
    width = float((pos[:, 0].max() - pos[:, 0].min()).item())
    height = float((pos[:, 1].max() - pos[:, 1].min()).item())
    return width / max(height, 1.0e-6)


def test_lattice_and_planar_families_resolve_topology_targets() -> None:
    """Lattice-like and planar DAG benchmark families should leave the default path."""
    hexagonal = _make_hexagonal_lattice_graph(rows=6, cols=7)
    sierpinski = _make_sierpinski_graph(depth=3)

    hex_structure = classify_graph(hexagonal.edge_index, hexagonal.num_nodes)
    sierpinski_structure = classify_graph(sierpinski.edge_index, sierpinski.num_nodes)

    assert "lattice_like" in hex_structure.topology_tags
    assert getattr(_resolved_config(hexagonal), "_dagua_native_target_aspect") == pytest.approx(
        0.05,
    )
    assert getattr(
        _resolved_config(hexagonal),
        "_dagua_native_rank_sep_multiplier",
    ) == pytest.approx(1.0)
    assert "planar_dag" in sierpinski_structure.topology_tags
    assert "lattice_like" not in sierpinski_structure.topology_tags
    assert getattr(_resolved_config(sierpinski), "_dagua_native_target_aspect") == pytest.approx(
        0.45,
    )


def test_random_dag_200_keeps_tall_default_target() -> None:
    """The top-win random DAG family should keep the Sprint 18h 0.25 target."""
    graph = _random_dag(200, 300, seed=42)
    structure = classify_graph(graph.edge_index, graph.num_nodes)
    resolved = _resolved_config(graph)

    assert structure.topology_tags == ()
    assert getattr(resolved, "_dagua_native_target_aspect") == pytest.approx(0.25)
    assert getattr(resolved, "_dagua_native_rank_sep_multiplier") == pytest.approx(1.0)


def test_tree_family_keeps_tall_default_target() -> None:
    """Tree-like top-win families should not receive planar aspect tags."""
    graph = make_org_chart(6, (1, 3, 5, 10, 20, 40), seed=42)
    structure = classify_graph(graph.edge_index, graph.num_nodes)
    resolved = _resolved_config(graph)

    assert structure.topology_tags == ()
    assert getattr(resolved, "_dagua_native_target_aspect") == pytest.approx(0.25)


def test_hexagonal_lattice_layout_moves_to_lattice_aspect() -> None:
    """The native pipeline should emit hexagonal lattice positions near the tag target.

    Sprint-22c added a dot-mimic LP polish primitive whose chosen LP
    coordinates can override the lattice-aspect snap when the LP solution
    scores higher composite. To keep this test focused on the
    aspect-snap mechanism (rather than the post-hoc polish picker) we
    disable ``edge_equalize_polish`` and assert against the un-polished
    gradient output. Using ``algorithm=None`` (the default routing path)
    is required because ``algorithm="dagua_native"`` does not forward
    the user-facing config to the pipeline, so the polish flag is not
    honored on the explicit-algorithm path.
    """
    graph = _make_hexagonal_lattice_graph(rows=6, cols=7)
    pos = layout(
        graph,
        LayoutConfig(
            steps=20,
            seed=42,
            device="cpu",
            edge_equalize_polish=False,
        ),
    )

    aspect = _bbox_aspect(pos)

    assert 0.04 <= aspect <= 0.12
    assert abs(aspect - 0.05) < abs(aspect - 0.25)
