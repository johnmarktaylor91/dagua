"""ForceAtlas2 reference competitor adapter."""

from __future__ import annotations

import inspect
import random
import time
from typing import TYPE_CHECKING, Any, Mapping, Optional, Set, Type

import torch

from dagua.eval.competitors.base import CompetitorBase, CompetitorResult, register

if TYPE_CHECKING:
    from dagua.graph import DaguaGraph

_FA2_REFERENCE_PACKAGE_ORDER = ("fa2", "fa2_modified")


def _load_forceatlas2() -> Type[Any]:
    """Load the explicitly preferred ForceAtlas2 reference implementation.

    Returns
    -------
    type[Any]
        ``ForceAtlas2`` class from ``fa2`` first, falling back to
        ``fa2_modified`` only when the maintained package is unavailable.

    Raises
    ------
    ImportError
        If neither supported package is importable.
    """
    last_error: Optional[ImportError] = None
    for package_name in _FA2_REFERENCE_PACKAGE_ORDER:
        try:
            module = __import__(package_name, fromlist=["ForceAtlas2"])
            return module.ForceAtlas2
        except ImportError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise ImportError("No ForceAtlas2 reference package configured.")


def _accepted_init_params(cls: Type[Any]) -> Set[str]:
    """Return the set of parameter names accepted by a class __init__.

    Parameters
    ----------
    cls : type
        Class whose ``__init__`` signature to inspect.

    Returns
    -------
    set[str]
        Parameter names excluding ``self``.
    """
    try:
        sig = inspect.signature(cls.__init__)
        return {name for name in sig.parameters if name != "self"}
    except (ValueError, TypeError):
        return set()


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

                # The maintained FA2 package initializes from both Python's
                # RNG and NumPy's global RNG, so we must seed both for parity.
                random.seed(seed)
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
            if seed is not None:
                # Newer ``fa2`` releases initialize from random.Random(self.seed);
                # global RNG seeding above only covers older reference packages.
                engine_kwargs["seed"] = seed
            layout_kwargs: dict[str, Any] = {"pos": None, "iterations": 100}
            if variant_params is not None:
                for key, value in dict(variant_params).items():
                    if key == "iterations":
                        layout_kwargs["iterations"] = value
                    else:
                        engine_kwargs[key] = value

            # Filter engine_kwargs to only parameters the library actually
            # accepts.  This guards against variant definitions that reference
            # params the installed library version does not support (e.g.
            # dissuadeHubs was never in fa2_modified despite appearing in some
            # FA2 documentation).
            accepted = _accepted_init_params(forceatlas2_cls)
            if accepted:
                engine_kwargs = {k: v for k, v in engine_kwargs.items() if k in accepted}

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
