"""Evaluation and aesthetic tuning subpackage.

Provides test graph collection, metrics, parameter sweeps,
Graphviz comparison, competitive benchmarking, and report generation.
"""

from dagua.eval.benchmark import BenchmarkResult, run_benchmark
from dagua.eval.compare import compare_with_graphviz, layout_all
from dagua.eval.graphs import TestGraph, get_scale_suite, get_test_graphs

__all__ = [
    "get_test_graphs",
    "get_scale_suite",
    "TestGraph",
    "compare_with_graphviz",
    "layout_all",
    "run_benchmark",
    "BenchmarkResult",
]
