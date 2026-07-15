"""Pipeline pins for trivial deterministic geometric layouts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import dagua
from dagua.config import LayoutConfig
from dagua.layout.ops.pipelines import PIPELINE_REGISTRY, get_pipeline_function
from dagua.layout.ops.pipelines.arc import build_arc_pipeline, layout_arc_pipeline
from dagua.layout.ops.pipelines.circlepack import (
    build_circlepack_pipeline,
    layout_circlepack_pipeline,
)
from dagua.layout.ops.pipelines.concentric import (
    build_concentric_pipeline,
    layout_concentric_pipeline,
)
from dagua.layout.ops.pipelines.osage import build_osage_pipeline, layout_osage_pipeline
from dagua.layout.ops.pipelines.star import build_star_pipeline, layout_star_pipeline


def _path_edges() -> torch.Tensor:
    """Return a fixed four-node path edge tensor.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, 3]``.
    """
    return torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)


@pytest.mark.parametrize(
    ("algorithm", "module", "function", "builder", "op_name"),
    [
        (
            "star",
            "dagua.layout.ops.pipelines.star",
            "layout_star_pipeline",
            build_star_pipeline,
            "networkx_simple_layout",
        ),
        (
            "concentric",
            "dagua.layout.ops.pipelines.concentric",
            "layout_concentric_pipeline",
            build_concentric_pipeline,
            "networkx_simple_layout",
        ),
        (
            "circlepack",
            "dagua.layout.ops.pipelines.circlepack",
            "layout_circlepack_pipeline",
            build_circlepack_pipeline,
            "networkx_simple_layout",
        ),
        (
            "arc",
            "dagua.layout.ops.pipelines.arc",
            "layout_arc_pipeline",
            build_arc_pipeline,
            "networkx_simple_layout",
        ),
        (
            "osage",
            "dagua.layout.ops.pipelines.osage",
            "layout_osage_pipeline",
            build_osage_pipeline,
            "graphviz_osage_array_layout",
        ),
    ],
)
def test_trivial_geometric_pipeline_is_registered(
    algorithm: str,
    module: str,
    function: str,
    builder: object,
    op_name: str,
) -> None:
    """Register the public trivial-geometric algorithm.

    Parameters
    ----------
    algorithm : str
        Registered layout name.
    module : str
        Expected pipeline module.
    function : str
        Expected pipeline function name.
    builder : object
        Pipeline builder callable.
    op_name : str
        Expected single operation name.

    Returns
    -------
    None
        Registry lookups must resolve the entrypoint.
    """
    assert PIPELINE_REGISTRY[algorithm] == (module, function)
    assert get_pipeline_function(algorithm.upper()).__name__ == function
    assert [operation.name for operation in builder().ops] == [op_name]


def test_star_matches_igraph_reference() -> None:
    """Match igraph's star-layout coordinates when the center is pinned.

    Returns
    -------
    None
        Dagua's source port must match igraph's deterministic angles.
    """
    ig = pytest.importorskip("igraph")
    expected = np.asarray(
        ig.Graph.Star(5, center=0, mode="undirected").layout_star(center=0).coords
    )

    actual = layout_star_pipeline(
        torch.tensor([[0, 0, 0, 0], [1, 2, 3, 4]], dtype=torch.long),
        5,
        center=0,
    ).numpy()

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-12)


def test_arc_places_bfs_order_on_x_axis() -> None:
    """Pin the standard arc-diagram BFS/input ordering.

    Returns
    -------
    None
        Nodes should lie on y=0 in BFS order.
    """
    actual = layout_arc_pipeline(_path_edges(), 4).numpy()

    np.testing.assert_allclose(actual[:, 1], np.zeros(4), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(actual[:, 0], np.linspace(-1.0, 1.0, 4), rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("algorithm", "pipeline"),
    [
        ("star", layout_star_pipeline),
        ("concentric", layout_concentric_pipeline),
        ("circlepack", layout_circlepack_pipeline),
        ("arc", layout_arc_pipeline),
        ("osage", layout_osage_pipeline),
    ],
)
def test_layout_config_algorithm_trivial_geometric_dispatches(
    algorithm: str,
    pipeline: object,
) -> None:
    """Exercise public engine dispatch for new trivial-geometric algorithms.

    Parameters
    ----------
    algorithm : str
        Layout algorithm name.
    pipeline : object
        Direct pipeline function for the same algorithm.

    Returns
    -------
    None
        Dispatch must return finite coordinates.
    """
    del pipeline
    graph = dagua.DaguaGraph.from_edge_list([("a", "b"), ("b", "c")])
    positions = dagua.layout(graph, LayoutConfig(algorithm=algorithm))

    assert positions.shape == (3, 2)
    assert torch.isfinite(positions).all()


def test_trivial_geometric_pipelines_have_no_external_binary_delegation() -> None:
    """Guard production trivial-geometric pipelines against binary delegation.

    Returns
    -------
    None
        Production source must not call NetworkX, Graphviz, Node, or competitor
        adapters.
    """
    root = Path(__file__).parents[1]
    source_paths = [
        root / "dagua" / "layout" / "ops" / "networkx_simple.py",
        root / "dagua" / "layout" / "ops" / "pipelines" / "star.py",
        root / "dagua" / "layout" / "ops" / "pipelines" / "concentric.py",
        root / "dagua" / "layout" / "ops" / "pipelines" / "circlepack.py",
        root / "dagua" / "layout" / "ops" / "pipelines" / "arc.py",
        root / "dagua" / "layout" / "ops" / "pipelines" / "osage.py",
    ]
    source = "\n".join(path.read_text() for path in source_paths)
    forbidden = (
        "subprocess",
        "competitor",
        "GraphvizOsage",
        "import networkx",
        '__import__("networkx")',
        "__import__('networkx')",
        "nx.",
    )
    for token in forbidden:
        assert token not in source
    assert "graphviz_competitor" not in source
    assert "dagre_competitor" not in source
