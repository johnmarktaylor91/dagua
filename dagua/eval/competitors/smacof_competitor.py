"""scikit-learn SMACOF competitor adapters."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

import torch

from dagua.eval.competitors.base import CompetitorBase, CompetitorResult, register
from dagua.layout.ops.graph_utils import shortest_path_distances

if TYPE_CHECKING:
    from dagua.graph import DaguaGraph


@register
class SklearnSmacofNonmetric(CompetitorBase):
    """Reference adapter for sklearn nonmetric SMACOF on graph geodesics."""

    name = "sklearn_smacof_nonmetric"
    max_nodes = 2_000
    variant_param_names = frozenset({"eps", "max_iter", "normalized_stress"})

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run sklearn ``smacof(metric=False)`` on graph geodesic distances.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Unused compatibility timeout parameter.
        seed : int | None, default=None
            Random seed forwarded to sklearn. ``None`` uses ``42``.

        Returns
        -------
        CompetitorResult
            Layout result and timing information.
        """
        del timeout

        from sklearn.manifold import smacof

        start = time.perf_counter()
        try:
            distances = shortest_path_distances(
                edge_index=graph.edge_index,
                num_nodes=graph.num_nodes,
                edge_weights=graph.edge_weights,
            )
            positions, _stress, _n_iter = smacof(
                distances,
                metric=False,
                n_components=2,
                init=None,
                n_init=1,
                max_iter=300,
                eps=1.0e-6,
                random_state=42 if seed is None else seed,
                normalized_stress=False,
            )
            elapsed = time.perf_counter() - start
            return CompetitorResult(
                name=self.name,
                pos=torch.tensor(positions, dtype=torch.float32),
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
