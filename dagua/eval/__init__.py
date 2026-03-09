"""Evaluation and aesthetic tuning subpackage.

Provides test graph collection, metrics, parameter sweeps,
Graphviz comparison, and report generation.
"""

from dagua.eval.graphs import get_test_graphs, TestGraph
from dagua.eval.compare import compare_with_graphviz

__all__ = ["get_test_graphs", "TestGraph", "compare_with_graphviz"]
