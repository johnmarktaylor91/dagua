"""Scale strategy protocol definitions."""

from __future__ import annotations

from typing import Any, Protocol

import torch

from dagua.config import LayoutConfig
from dagua.layout.scale.sketch import TopologySketch


class ScaleStrategyProtocol(Protocol):
    """Protocol implemented by above-gate scale layout strategies."""

    def layout(
        self,
        graph: Any,
        config: LayoutConfig,
        sketch: TopologySketch,
        *,
        trace: Any = None,
    ) -> torch.Tensor:
        """Compute a scale layout for ``graph``.

        Parameters
        ----------
        graph : Any
            Prepared graph-like layout input.
        config : LayoutConfig
            User layout configuration.
        sketch : TopologySketch
            Topology sketch used by the router.
        trace : Any, optional
            Optional trace sink.

        Returns
        -------
        torch.Tensor
            Position tensor with shape ``[N, 2]``.
        """
        ...


__all__ = ["ScaleStrategyProtocol"]
