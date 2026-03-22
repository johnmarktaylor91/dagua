"""ForceAtlas2 reference competitor adapter."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Mapping, Optional, Type

import torch

from dagua.eval.competitors.base import CompetitorBase, CompetitorResult, register

if TYPE_CHECKING:
    from dagua.graph import DaguaGraph


def _load_forceatlas2() -> Type[Any]:
    """Load the ForceAtlas2 implementation from either supported package name.

    Returns
    -------
    type[Any]
        ``ForceAtlas2`` class from ``fa2`` or ``fa2_modified``.

    Raises
    ------
    ImportError
        If neither supported package is importable.
    """
    try:
        from fa2 import ForceAtlas2

        return ForceAtlas2
    except ImportError:
        from fa2_modified import ForceAtlas2

        return ForceAtlas2


def _fa2_available() -> bool:
    """Return whether the ForceAtlas2 reference dependency is usable.

    Returns
    -------
    bool
        ``True`` when both ``networkx`` and a supported ForceAtlas2 package
        are importable.
    """
    try:
        import networkx  # noqa: F401

        _load_forceatlas2()
    except Exception:
        return False
    return True


@register
class FA2Reference(CompetitorBase):
    """Competitor adapter for the reference ForceAtlas2 implementation."""

    name = "fa2_ref"
    max_nodes = 20_000
    variant_param_names = frozenset(
        {
            "barnesHutOptimize",
            "barnesHutTheta",
            "dissuadeHubs",
            "gravity",
            "iterations",
            "linLogMode",
            "outboundAttractionDistribution",
            "scalingRatio",
            "strongGravityMode",
        }
    )

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run ForceAtlas2 through its NetworkX adapter.

        Parameters
        ----------
        graph : DaguaGraph
            Input graph to lay out.
        timeout : float, optional
            Unused adapter timeout in seconds. Included for interface
            compatibility with the benchmark harness.
        seed : int | None, default=None
            Random seed for the NumPy-backed initial positions used by the
            reference implementation. ``None`` keeps the library default.

        Returns
        -------
        CompetitorResult
            Layout result with positions shaped ``[N, 2]`` on CPU, or an error
            payload if the third-party engine fails.
        """
        return self.layout_with_variant(graph, timeout=timeout, seed=seed, variant_params=None)

    def layout_with_variant(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
        variant_params: Optional[Mapping[str, Any]] = None,
    ) -> CompetitorResult:
        """Run ForceAtlas2 through its NetworkX adapter.

        Parameters
        ----------
        graph : DaguaGraph
            Input graph to lay out.
        timeout : float, optional
            Unused adapter timeout in seconds. Included for interface
            compatibility with the benchmark harness.
        seed : int | None, default=None
            Random seed for the NumPy-backed initial positions used by the
            reference implementation. ``None`` keeps the library default.

        Returns
        -------
        CompetitorResult
            Layout result with positions shaped ``[N, 2]`` on CPU, or an error
            payload if the third-party engine fails.
        """
        import networkx as nx

        start = time.perf_counter()
        try:
            if seed is not None:
                import numpy as np

                # The reference library samples initial positions from NumPy's
                # global RNG rather than accepting a seed argument directly.
                np.random.seed(seed)

            if graph.num_nodes <= 1:
                pos = torch.zeros((graph.num_nodes, 2), dtype=torch.float32)
                elapsed = time.perf_counter() - start
                return CompetitorResult(name=self.name, pos=pos, runtime_seconds=elapsed)

            forceatlas2_cls = _load_forceatlas2()
            edge_index = graph.edge_index.cpu().numpy()
            weights = graph.edge_weights
            nx_graph = nx.Graph()
            nx_graph.add_nodes_from(range(graph.num_nodes))
            for edge_idx in range(edge_index.shape[1]):
                source = int(edge_index[0, edge_idx])
                target = int(edge_index[1, edge_idx])
                if source != target:
                    if weights is not None:
                        nx_graph.add_edge(
                            source,
                            target,
                            weight=float(weights[edge_idx].item()),
                        )
                    else:
                        nx_graph.add_edge(source, target)

            engine_kwargs: dict[str, Any] = {
                "outboundAttractionDistribution": True,
                "edgeWeightInfluence": 1.0,
                "jitterTolerance": 1.0,
                "barnesHutOptimize": True,
                "barnesHutTheta": 1.2,
                "scalingRatio": 2.0,
                "strongGravityMode": False,
                "gravity": 1.0,
                "verbose": False,
            }
            layout_kwargs: dict[str, Any] = {"pos": None, "iterations": 100}
            if variant_params is not None:
                for key, value in dict(variant_params).items():
                    if key == "iterations":
                        layout_kwargs["iterations"] = value
                    else:
                        engine_kwargs[key] = value

            layout_engine = forceatlas2_cls(**engine_kwargs)
            positions = layout_engine.forceatlas2_networkx_layout(
                nx_graph,
                **layout_kwargs,
            )

            pos = torch.zeros((graph.num_nodes, 2), dtype=torch.float32)
            for node_id, (x_coord, y_coord) in positions.items():
                pos[int(node_id), 0] = float(x_coord)
                pos[int(node_id), 1] = float(y_coord)

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
        """Report whether ForceAtlas2 is usable in the current environment.

        Returns
        -------
        bool
            ``True`` when a supported ForceAtlas2 package is importable.
        """
        return _fa2_available()
