"""Shared test fixtures — sample graphs, common assertions."""

import pytest
import torch

from dagua.graph import DaguaGraph
from dagua.config import LayoutConfig


@pytest.fixture
def simple_chain():
    """Simple 5-node chain: a→b→c→d→e."""
    return DaguaGraph.from_edge_list([
        ("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"),
    ])


@pytest.fixture
def diamond_graph():
    """Diamond: a→b, a→c, b→d, c→d."""
    return DaguaGraph.from_edge_list([
        ("a", "b"), ("a", "c"), ("b", "d"), ("c", "d"),
    ])


@pytest.fixture
def skip_graph():
    """Chain with skip connection: a→b→c→d, a→d."""
    return DaguaGraph.from_edge_list([
        ("a", "b"), ("b", "c"), ("c", "d"), ("a", "d"),
    ])


@pytest.fixture
def wide_graph():
    """Wide parallel: input→b1,b2,b3,b4→output."""
    return DaguaGraph.from_edge_list([
        ("input", "b1"), ("input", "b2"), ("input", "b3"), ("input", "b4"),
        ("b1", "output"), ("b2", "output"), ("b3", "output"), ("b4", "output"),
    ])


@pytest.fixture
def clustered_graph():
    """Graph with clusters."""
    g = DaguaGraph.from_edge_list([
        ("input", "enc1"), ("enc1", "enc2"),
        ("enc2", "dec1"), ("dec1", "dec2"),
        ("dec2", "output"),
    ])
    g.add_cluster("encoder", [1, 2], label="Encoder")
    g.add_cluster("decoder", [3, 4], label="Decoder")
    return g


@pytest.fixture
def empty_graph():
    """Graph with no nodes or edges."""
    return DaguaGraph()


@pytest.fixture
def single_node_graph():
    """Graph with one node, no edges."""
    g = DaguaGraph()
    g.add_node("alone")
    return g


@pytest.fixture
def default_config():
    """Default layout config."""
    return LayoutConfig()


@pytest.fixture
def fast_config():
    """Fast layout config for quick tests."""
    return LayoutConfig(steps=50, lr=0.1)
