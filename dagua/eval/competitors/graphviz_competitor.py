"""Graphviz competitor adapters — dot, sfdp, neato, fdp engines."""

from __future__ import annotations

import shutil
import time
from typing import TYPE_CHECKING

from dagua.eval.competitors.base import CompetitorBase, CompetitorResult, register

if TYPE_CHECKING:
    from dagua.graph import DaguaGraph


def _graphviz_available() -> bool:
    return shutil.which("dot") is not None


class _GraphvizBase(CompetitorBase):
    """Base class for Graphviz engine variants."""

    engine: str = "dot"

    def layout(self, graph: DaguaGraph, timeout: float = 300.0) -> CompetitorResult:
        from dagua.graphviz_utils import layout_with_graphviz

        start = time.perf_counter()
        try:
            pos = layout_with_graphviz(graph, engine=self.engine)
            elapsed = time.perf_counter() - start
            return CompetitorResult(
                name=self.name, pos=pos, runtime_seconds=elapsed
            )
        except Exception as e:
            elapsed = time.perf_counter() - start
            return CompetitorResult(
                name=self.name, pos=None, runtime_seconds=elapsed, error=str(e)
            )

    def available(self) -> bool:
        return _graphviz_available()


@register
class GraphvizDot(_GraphvizBase):
    name = "graphviz_dot"
    engine = "dot"
    max_nodes = 5_000


@register
class GraphvizSfdp(_GraphvizBase):
    name = "graphviz_sfdp"
    engine = "sfdp"
    max_nodes = 100_000


@register
class GraphvizNeato(_GraphvizBase):
    name = "graphviz_neato"
    engine = "neato"
    max_nodes = 2_000


@register
class GraphvizFdp(_GraphvizBase):
    name = "graphviz_fdp"
    engine = "fdp"
    max_nodes = 5_000
