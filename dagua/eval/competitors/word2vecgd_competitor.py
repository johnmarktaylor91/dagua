"""Word2VecGD adapter using the native graphv_nn-style implementation."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Mapping, Optional

import torch

from dagua.eval.competitors.base import CompetitorBase, CompetitorResult, register
from dagua.layout.ops.pipelines.word2vecgd import layout_word2vecgd_pipeline

if TYPE_CHECKING:
    from dagua.graph import DaguaGraph


@register
class Word2VecGDReference(CompetitorBase):
    """Adapter for the graphv_nn-inspired Word2VecGD layout."""

    name = "word2vecgd"
    max_nodes = 5_000
    variant_param_names = frozenset(
        {
            "embedding_dim",
            "num_walks",
            "walk_length",
            "window",
            "epochs",
            "negative_samples",
            "embedding_lr",
            "layout_lr",
            "steps",
        }
    )

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run the Word2VecGD adapter.

        Parameters
        ----------
        graph : DaguaGraph
            Input graph to lay out.
        timeout : float, default=300.0
            Unused adapter timeout in seconds.
        seed : int, optional
            Random seed.

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
        """Run the native Word2VecGD implementation through adapter shape.

        Parameters
        ----------
        graph : DaguaGraph
            Input graph to lay out.
        timeout : float, default=300.0
            Unused adapter timeout in seconds.
        seed : int, optional
            Random seed.
        variant_params : Mapping[str, object], optional
            Pipeline parameter overrides.

        Returns
        -------
        CompetitorResult
            Layout result containing CPU positions or an error payload.
        """
        del timeout
        start = time.perf_counter()
        try:
            params = dict(variant_params or {})
            positions = layout_word2vecgd_pipeline(
                graph.edge_index,
                graph.num_nodes,
                node_sizes=graph.node_sizes,
                seed=seed if seed is not None else 42,
                **params,
            )
            elapsed = time.perf_counter() - start
            return CompetitorResult(
                name=self.name,
                pos=positions.detach().to(device="cpu", dtype=torch.float32),
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
        """Report adapter availability.

        Returns
        -------
        bool
            Always ``True`` because this adapter is self-contained.
        """
        return True
