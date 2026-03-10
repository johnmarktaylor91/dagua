"""Dagua competitor adapter — wraps dagua.layout() for uniform benchmarking."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from dagua.eval.competitors.base import CompetitorBase, CompetitorResult, register

if TYPE_CHECKING:
    from dagua.graph import DaguaGraph


@register
class DaguaCompetitor(CompetitorBase):
    name = "dagua"
    max_nodes = 100_000_000  # no practical limit

    def layout(self, graph: DaguaGraph, timeout: float = 300.0) -> CompetitorResult:
        from dagua.config import LayoutConfig
        from dagua.layout import layout

        config = LayoutConfig(device="cpu", verbose=False)

        start = time.perf_counter()
        try:
            pos = layout(graph, config)
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
        return True
