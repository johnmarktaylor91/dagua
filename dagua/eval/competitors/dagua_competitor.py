"""Dagua competitor adapter — wraps dagua.layout() for uniform benchmarking."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

from dagua.eval.competitors.base import CompetitorBase, CompetitorResult, register
from dagua.eval.runtime_env import suspend_torchlens_decoration

if TYPE_CHECKING:
    from dagua.graph import DaguaGraph


@register
class DaguaCompetitor(CompetitorBase):
    """Benchmark adapter for dagua's native layout engine."""

    name = "dagua"
    max_nodes = 100_000_000  # no practical limit

    @property
    def device(self) -> str:
        """Return the preferred execution device for the current runtime.

        Returns
        -------
        str
            ``"cuda"`` when CUDA is available, otherwise ``"cpu"``.
        """
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run dagua layout on the best available device.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Unused compatibility parameter for the competitor interface.
        seed : int | None, default=None
            Random seed forwarded to ``LayoutConfig``. ``None`` preserves the
            adapter's historical default of ``42``.

        Returns
        -------
        CompetitorResult
            Layout result with positions moved back to CPU for downstream
            metrics and artifact serialization.
        """
        del timeout

        from dagua.config import LayoutConfig
        from dagua.layout import layout

        config = LayoutConfig(
            device=self.device,
            verbose=False,
            seed=seed if seed is not None else 42,
        )

        start = time.perf_counter()
        try:
            with suspend_torchlens_decoration():
                pos = layout(graph, config)
            elapsed = time.perf_counter() - start
            return CompetitorResult(
                name=self.name,
                pos=pos.detach().cpu(),
                runtime_seconds=elapsed,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - start
            return CompetitorResult(
                name=self.name,
                pos=None,
                runtime_seconds=elapsed,
                error=str(exc),
            )

    def available(self) -> bool:
        """Report that dagua is always available in the local environment.

        Returns
        -------
        bool
            Always ``True`` for the in-repo engine adapter.
        """
        return True
