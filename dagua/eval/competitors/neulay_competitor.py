"""NeuLay competitor adapter."""

from __future__ import annotations

import importlib
import time
from typing import TYPE_CHECKING, Any, Callable, Optional

from dagua.eval.competitors.base import CompetitorBase, CompetitorResult, register

if TYPE_CHECKING:
    from dagua.graph import DaguaGraph


def _pyg_available() -> bool:
    """Return whether PyTorch Geometric is importable.

    Returns
    -------
    bool
        ``True`` when PyTorch Geometric imports successfully.
    """
    try:
        import torch_geometric  # noqa: F401
    except Exception:
        return False
    return True


def _load_upstream_neulay() -> Optional[Callable[..., Any]]:
    """Load an installed upstream NeuLay entry point when available.

    Returns
    -------
    Callable[..., Any] | None
        Callable reference entry point, or ``None`` when no upstream package is
        installed in the environment.

    Notes
    -----
    This adapter must never fall back to Dagua's own implementation because it
    exists to benchmark an independent reference.
    """
    if not _pyg_available():
        return None

    module_names = ("neulay", "NeuLay")
    function_names = ("layout_neulay", "layout")
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        for function_name in function_names:
            candidate = getattr(module, function_name, None)
            if callable(candidate):
                return candidate
    return None


@register
class NeuLayReference(CompetitorBase):
    """Competitor adapter for an independently installed NeuLay reference."""

    name = "neulay"
    max_nodes = 20_000

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run the upstream NeuLay reference implementation.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Unused compatibility parameter for the benchmark interface.
        seed : int | None, default=None
            Random seed for the stochastic solver.

        Returns
        -------
        CompetitorResult
            Layout result and runtime metadata.
        """
        del timeout

        start = time.perf_counter()
        upstream_layout = _load_upstream_neulay()
        if upstream_layout is None:
            elapsed = time.perf_counter() - start
            return CompetitorResult(
                name=self.name,
                pos=None,
                runtime_seconds=elapsed,
                error="upstream NeuLay reference is not installed",
            )

        try:
            pos = upstream_layout(
                graph.edge_index,
                graph.num_nodes,
                node_sizes=graph.node_sizes,
                seed=42 if seed is None else seed,
                steps=20_000,
                gcn_steps=2_000,
                use_gcn=True,
            )
            elapsed = time.perf_counter() - start
            return CompetitorResult(name=self.name, pos=pos, runtime_seconds=elapsed)
        except Exception as exc:
            elapsed = time.perf_counter() - start
            return CompetitorResult(
                name=self.name,
                pos=None,
                runtime_seconds=elapsed,
                error=str(exc),
            )

    def available(self) -> bool:
        """Report whether an upstream NeuLay reference can be executed.

        Returns
        -------
        bool
            ``True`` only when PyTorch Geometric and an upstream NeuLay package
            are both importable.
        """
        return _load_upstream_neulay() is not None
