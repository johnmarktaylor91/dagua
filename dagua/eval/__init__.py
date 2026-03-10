"""Evaluation and aesthetic tuning subpackage.

Provides test graph collection, metrics, parameter sweeps,
Graphviz comparison, competitive benchmarking, and report generation.
"""

from dagua.eval.graphs import get_test_graphs, get_scale_suite, TestGraph
from dagua.eval.compare import compare_with_graphviz
from dagua.eval.benchmark import run_benchmark, BenchmarkResult

__all__ = [
    "get_test_graphs",
    "get_scale_suite",
    "TestGraph",
    "compare_with_graphviz",
    "run_benchmark",
    "BenchmarkResult",
]
