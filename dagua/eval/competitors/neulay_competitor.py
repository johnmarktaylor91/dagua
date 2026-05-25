"""NeuLay competitor adapter."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional

from dagua.eval.competitors.base import CompetitorBase, CompetitorResult, register

if TYPE_CHECKING:
    import torch

    from dagua.graph import DaguaGraph

# Round 34 recovery: the installable ``neulay``/``NeuLay`` packages are absent,
# and the script clone no longer contains an importable ``NeuLay-2.py`` file.
# Point the reference competitor at the side-effect-free wrapper recovered from
# the monolithic port that was previously factored from that script.


def _load_upstream_neulay() -> Optional[Callable[..., "torch.Tensor"]]:
    """Load the recovered NeuLay reference entry point when available.

    Returns
    -------
    Callable[..., torch.Tensor] | None
        Callable reference entry point, or ``None`` when the recovered wrapper
        cannot be imported.

    Notes
    -----
    The function name is preserved for tests and older monkeypatches, but the
    entry point is the recovered script wrapper rather than an installed package.
    """
    try:
        from dagua.eval.competitors.neulay_wrapper import layout_neulay_reference
    except Exception:
        return None
    if callable(layout_neulay_reference):
        return layout_neulay_reference
    return None


@register
class NeuLayReference(CompetitorBase):
    """Competitor adapter for an independently installed NeuLay reference."""

    name = "neulay"
    max_nodes = 20_000
    variant_param_names = frozenset({"gcn_steps", "lr", "radius", "steps", "use_gcn"})

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run the recovered NeuLay reference implementation.

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
        return self.layout_with_variant(graph, timeout=timeout, seed=seed, variant_params=None)

    def layout_with_variant(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
        variant_params: Optional[Mapping[str, Any]] = None,
    ) -> CompetitorResult:
        """Run the recovered NeuLay reference implementation.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Unused compatibility parameter for the benchmark interface.
        seed : int | None, default=None
            Random seed for the stochastic solver.
        variant_params : Mapping[str, Any] | None, default=None
            Optional NeuLay parameter overrides.

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
            layout_kwargs: dict[str, Any] = {
                "seed": 42 if seed is None else seed,
                "steps": 20_000,
                "gcn_steps": 2_000,
                "use_gcn": True,
                "lr": 0.1,
                "radius": 0.4,
            }
            if variant_params is not None:
                layout_kwargs.update(dict(variant_params))

            pos = upstream_layout(
                graph.edge_index,
                graph.num_nodes,
                node_sizes=graph.node_sizes,
                **layout_kwargs,
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
        """Report whether the recovered NeuLay reference can be executed.

        Parameters
        ----------
        None
            This method reads no caller-supplied parameters.

        Returns
        -------
        bool
            ``True`` when the recovered wrapper can be imported.
        """
        return _load_upstream_neulay() is not None
