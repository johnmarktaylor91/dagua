"""Dagua pipeline reimplementation competitors for fidelity benchmarks."""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Optional

import torch

from dagua.eval.competitors.base import CompetitorBase, CompetitorResult, register
from dagua.layout.ops.pipelines import get_pipeline_function

if TYPE_CHECKING:
    from dagua.graph import DaguaGraph


@dataclass(frozen=True)
class PipelineReimplementationSpec:
    """Metadata for one Dagua pipeline benchmark adapter.

    Parameters
    ----------
    name : str
        Registered competitor name.
    pipeline_name : str
        Key in ``PIPELINE_REGISTRY``.
    max_nodes : int
        Graph-size cap used by the benchmark scheduler.
    default_params : Mapping[str, Any]
        Keyword arguments forwarded to the pipeline for reference fidelity.
    """

    name: str
    pipeline_name: str
    max_nodes: int
    default_params: Mapping[str, Any]


class PipelineReimplementationCompetitor(CompetitorBase):
    """Benchmark adapter for a Dagua reimplementation pipeline."""

    spec: PipelineReimplementationSpec
    supports_clusters = False

    def __init__(self) -> None:
        """Initialize registration metadata from the class spec."""
        self.name = self.spec.name
        self.max_nodes = self.spec.max_nodes
        self.variant_param_names = self._variant_param_names()

    def _variant_param_names(self) -> frozenset[str]:
        """Return pipeline keyword names supported as variant overrides.

        Returns
        -------
        frozenset[str]
            Callable keyword names excluding graph inputs and seed plumbing.
        """
        function = get_pipeline_function(self.spec.pipeline_name)
        signature = inspect.signature(function)
        common_params = {"edge_index", "num_nodes", "node_sizes", "edge_weights", "seed"}
        names = {
            name
            for name, parameter in signature.parameters.items()
            if name not in common_params
            and parameter.kind
            in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
        }
        return frozenset(names)

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run the configured Dagua reimplementation pipeline.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Unused compatibility timeout; benchmark workers enforce wall-clock
            limits outside this in-process adapter.
        seed : int | None, default=None
            Optional benchmark seed forwarded when the pipeline accepts one.

        Returns
        -------
        CompetitorResult
            Layout positions on CPU or an error payload.
        """
        return self.layout_with_variant(graph, timeout=timeout, seed=seed, variant_params=None)

    def layout_with_variant(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
        variant_params: Optional[Mapping[str, Any]] = None,
    ) -> CompetitorResult:
        """Run the pipeline with optional variant parameter overrides.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Unused compatibility timeout.
        seed : int | None, default=None
            Optional benchmark seed.
        variant_params : Mapping[str, Any] | None, default=None
            Additional pipeline keyword arguments.

        Returns
        -------
        CompetitorResult
            Layout positions on CPU or an error payload.
        """
        del timeout
        function = get_pipeline_function(self.spec.pipeline_name)
        params = dict(self.spec.default_params)
        if variant_params is not None:
            params.update(dict(variant_params))

        signature = inspect.signature(function)
        if "edge_weights" in signature.parameters and graph.edge_weights is not None:
            params.setdefault("edge_weights", graph.edge_weights)
        if "seed" in signature.parameters:
            params.setdefault("seed", 42 if seed is None else int(seed))

        start = time.perf_counter()
        try:
            result = function(
                graph.edge_index,
                graph.num_nodes,
                node_sizes=graph.node_sizes,
                **params,
            )
            positions = result[0] if isinstance(result, tuple) else result
            if not isinstance(positions, torch.Tensor):
                raise TypeError(f"{self.spec.pipeline_name} returned {type(positions).__name__}")
            elapsed = time.perf_counter() - start
            return CompetitorResult(
                name=self.name,
                pos=positions.detach().cpu(),
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


def _register_pipeline_reimplementation(
    name: str,
    pipeline_name: str,
    max_nodes: int,
    default_params: Optional[Mapping[str, Any]] = None,
) -> None:
    """Register one pipeline-backed reimplementation competitor.

    Parameters
    ----------
    name : str
        Competitor registry name.
    pipeline_name : str
        Pipeline registry key.
    max_nodes : int
        Benchmark graph-size cap.
    default_params : Mapping[str, Any] | None, default=None
        Pipeline keyword defaults.

    Returns
    -------
    None
        Registration happens through the shared competitor registry.
    """
    spec = PipelineReimplementationSpec(
        name=name,
        pipeline_name=pipeline_name,
        max_nodes=max_nodes,
        default_params={} if default_params is None else dict(default_params),
    )
    class_name = "".join(part.capitalize() for part in name.replace("-", "_").split("_"))
    competitor_cls = type(
        f"{class_name}Competitor",
        (PipelineReimplementationCompetitor,),
        {"__doc__": f"Dagua reimplementation adapter for ``{pipeline_name}``.", "spec": spec},
    )
    register(competitor_cls)


_PIPELINE_REIMPLEMENTATIONS: tuple[tuple[str, str, int, Mapping[str, Any]], ...] = (
    ("dagre_reimpl", "dagre", 1_500, {"nodesep": 40.0, "ranksep": 60.0, "edgesep": 20.0}),
    ("elk_layered_reimpl", "elk", 15_000, {}),
    ("elk_force_reimpl", "elk_force", 15_000, {}),
    ("elk_stress_reimpl", "elk_stress", 15_000, {}),
    ("elk_mrtree_reimpl", "elk_mrtree", 15_000, {}),
    ("elk_radial_reimpl", "elk_radial", 15_000, {}),
    ("d3force_reimpl", "d3force", 5_000, {"ticks": 300}),
    ("d3_tree_reimpl", "d3_tree", 10_000, {}),
    ("d3_tree_radial_reimpl", "d3_tree_radial", 10_000, {}),
    ("d3_cluster_reimpl", "d3_cluster", 10_000, {}),
    ("d3_cluster_radial_reimpl", "d3_cluster_radial", 10_000, {}),
    ("circo_reimpl", "circo", 10_000, {}),
    ("twopi_reimpl", "twopi", 10_000, {}),
    ("osage_reimpl", "osage", 10_000, {}),
    ("ogdf_balloon_reimpl", "balloon", 100_000, {}),
    ("ogdf_bertault_reimpl", "bertault", 10_000, {}),
    ("ogdf_fpp_reimpl", "fpp", 100_000, {}),
    ("ogdf_schnyder_reimpl", "schnyder", 100_000, {}),
    ("ogdf_sugiyama_reimpl", "sugiyama", 20_000, {}),
    ("nx_circular_reimpl", "circular", 100_000, {}),
    ("nx_shell_reimpl", "shell", 100_000, {}),
    ("nx_spiral_reimpl", "spiral", 100_000, {}),
    ("nx_bipartite_reimpl", "bipartite", 100_000, {}),
    ("nx_multipartite_reimpl", "multipartite", 100_000, {}),
    ("nx_bfs_reimpl", "bfs", 100_000, {}),
    ("nx_arf_reimpl", "arf", 10_000, {}),
    ("nx_planar_reimpl", "planar", 100_000, {}),
    ("backbone_reimpl", "backbone", 2_000, {"keep": 0.2}),
    (
        "sparse_stress_reimpl",
        "sparse_stress",
        5_000,
        {"pivots": 8, "steps": 20, "sampler": "kmeans", "mds_pivots": 8},
    ),
    ("isom_reimpl", "isom", 5_000, {}),
    ("smacof_nonmetric_reimpl", "smacof_nonmetric", 2_000, {}),
    ("coregd_reimpl", "coregd", 100_000, {}),
    ("smartgd_reimpl", "smartgd", 5_000, {}),
    ("deepgd_reimpl", "deepgd", 5_000, {}),
    ("tfdp_reimpl", "tfdp", 2_000, {"max_iter": 300}),
    ("pacmap_reimpl", "pacmap", 5_000, {}),
    ("word2vecgd_reimpl", "word2vecgd", 5_000, {}),
    ("drgraph_reimpl", "drgraph", 10_000_000, {}),
    ("largevis_reimpl", "largevis", 10_000_000, {}),
    ("openord_reimpl", "openord", 5_000, {}),
    ("grip_reimpl", "grip", 100_000, {}),
    ("omega_reimpl", "omega", 100_000, {}),
    ("tidy_reimpl", "tidy", 100_000, {}),
    ("mulment_reimpl", "mulment", 100_000, {}),
    ("nnpnet_reimpl", "nnpnet", 100_000, {}),
)

for _name, _pipeline_name, _max_nodes, _default_params in _PIPELINE_REIMPLEMENTATIONS:
    _register_pipeline_reimplementation(
        name=_name,
        pipeline_name=_pipeline_name,
        max_nodes=_max_nodes,
        default_params=_default_params,
    )


__all__ = [
    "PipelineReimplementationCompetitor",
    "PipelineReimplementationSpec",
]
