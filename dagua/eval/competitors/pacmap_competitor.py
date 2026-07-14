"""PaCMAP reference adapter on graph shortest-path distance features."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Mapping, Optional

import torch

from dagua.eval.competitors.base import CompetitorBase, CompetitorResult, register
from dagua.layout.ops.pipelines.tsne_graph import _graph_geodesic_distances

if TYPE_CHECKING:
    from dagua.graph import DaguaGraph


@register
class PaCMAPGraph(CompetitorBase):
    """Competitor adapter for PaCMAP on graph geodesic features."""

    name = "pacmap"
    max_nodes = 5_000
    variant_param_names = frozenset(
        {"n_neighbors", "MN_ratio", "FP_ratio", "lr", "num_iters", "init"}
    )

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run reference PaCMAP on graph distance features.

        Parameters
        ----------
        graph : DaguaGraph
            Input graph to lay out.
        timeout : float, default=300.0
            Unused adapter timeout in seconds.
        seed : int, optional
            Random seed forwarded to PaCMAP.

        Returns
        -------
        CompetitorResult
            Layout result containing CPU positions or an error payload.
        """
        return self.layout_with_variant(graph, timeout=timeout, seed=seed, variant_params=None)

    def layout_with_variant(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
        variant_params: Optional[Mapping[str, object]] = None,
    ) -> CompetitorResult:
        """Run reference PaCMAP on graph distance features.

        Parameters
        ----------
        graph : DaguaGraph
            Input graph to lay out.
        timeout : float, default=300.0
            Unused adapter timeout in seconds.
        seed : int, optional
            Random seed forwarded to PaCMAP.
        variant_params : Mapping[str, object], optional
            PaCMAP constructor or ``fit_transform`` overrides.

        Returns
        -------
        CompetitorResult
            Layout result containing CPU positions or an error payload.
        """
        start = time.perf_counter()
        try:
            from pacmap import PaCMAP

            if graph.num_nodes <= 1:
                elapsed = time.perf_counter() - start
                return CompetitorResult(
                    name=self.name,
                    pos=torch.zeros((graph.num_nodes, 2), dtype=torch.float32),
                    runtime_seconds=elapsed,
                )

            params: dict[str, object] = {
                "n_components": 2,
                "n_neighbors": 10,
                "MN_ratio": 0.5,
                "FP_ratio": 2.0,
                "lr": 1.0,
                "num_iters": (100, 100, 250),
                "random_state": seed if seed is not None else 42,
                "knn_backend": "faiss",
                "verbose": False,
            }
            init: object = "pca"
            if variant_params is not None:
                params.update(
                    {key: value for key, value in variant_params.items() if key != "init"}
                )
                init = variant_params.get("init", init)

            distances = _graph_geodesic_distances(
                graph.edge_index,
                graph.num_nodes,
                graph.edge_weights,
            )
            coordinates = PaCMAP(**params).fit_transform(distances, init=init)
            elapsed = time.perf_counter() - start
            return CompetitorResult(
                name=self.name,
                pos=torch.tensor(coordinates, dtype=torch.float32),
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
        """Report whether the PaCMAP package is importable.

        Returns
        -------
        bool
            ``True`` when ``pacmap.PaCMAP`` imports successfully.
        """
        try:
            from pacmap import PaCMAP  # noqa: F401
        except Exception:
            return False
        return True
