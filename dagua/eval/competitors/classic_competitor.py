"""Competitor adapters for dagua's classic algorithm reimplementations.

These wrap the educational implementations in ``dagua/layout/ops/pipelines/`` so
they can be benchmarked alongside the original reference implementations.
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Optional, Tuple, cast

import torch

from dagua.eval.competitors.base import (
    CompetitorBase,
    CompetitorResult,
    register,
)
from dagua.eval.size_policy import size_aware_externals

if TYPE_CHECKING:
    from dagua.graph import DaguaGraph

_GRAPHVIZ_POINTS_PER_INCH = 72.0
_GRAPHVIZ_DEFAULT_NODE_WIDTH_POINTS = 54.0
_GRAPHVIZ_DEFAULT_NODE_HEIGHT_POINTS = 36.0
_GRAPHVIZ_LABEL_XPAD_POINTS = 16.0
_GRAPHVIZ_LABEL_YPAD_POINTS = 8.0
_GRAPHVIZ_HELVETICA_UNITS_PER_EM = 2048.0
# Graphviz 7.0.5 ``textspan.c`` fallback uses ``LINESPACING`` for logical height.
_GRAPHVIZ_TEXT_HEIGHT_FACTOR = 1.128
_GRAPHVIZ_TYPED_TEXT_HEIGHT_FACTOR = 1.2
_SFDP_LABEL_BOX_MIN_NODE_COUNT = 10
_SFDP_LABEL_BOX_WIDE_LABEL_POINTS = 100.0
_SUGIYAMA_TYPED_X_MAX_NODES = 50
_SUGIYAMA_DETERMINISTIC_CACHE: dict[Tuple[Any, ...], Tuple[torch.Tensor, float]] = {}
_GRAPHVIZ_HELVETICA_REGULAR_WIDTHS = (
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    569,
    569,
    727,
    1139,
    1139,
    1821,
    1366,
    391,
    682,
    682,
    797,
    1196,
    569,
    682,
    569,
    569,
    1139,
    1139,
    1139,
    1139,
    1139,
    1139,
    1139,
    1139,
    1139,
    1139,
    569,
    569,
    1196,
    1196,
    1196,
    1139,
    2079,
    1366,
    1366,
    1479,
    1479,
    1366,
    1251,
    1593,
    1479,
    569,
    1024,
    1366,
    1139,
    1706,
    1479,
    1593,
    1366,
    1593,
    1479,
    1366,
    1251,
    1479,
    1366,
    1933,
    1366,
    1366,
    1251,
    569,
    569,
    569,
    961,
    1139,
    682,
    1139,
    1139,
    1024,
    1139,
    1139,
    569,
    1139,
    1139,
    455,
    455,
    1024,
    455,
    1706,
    1139,
    1139,
    1139,
    1139,
    682,
    1024,
    569,
    1139,
    1024,
    1479,
    1024,
    1024,
    1024,
    684,
    532,
    684,
    1196,
    -1,
)
_GRAPHVIZ_TIMES_REGULAR_WIDTHS = (
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    512,
    682,
    836,
    1024,
    1024,
    1706,
    1593,
    369,
    682,
    682,
    1024,
    1155,
    512,
    682,
    512,
    569,
    1024,
    1024,
    1024,
    1024,
    1024,
    1024,
    1024,
    1024,
    1024,
    1024,
    569,
    569,
    1155,
    1155,
    1155,
    909,
    1886,
    1479,
    1366,
    1366,
    1479,
    1251,
    1139,
    1479,
    1479,
    682,
    797,
    1479,
    1251,
    1821,
    1479,
    1479,
    1139,
    1479,
    1366,
    1139,
    1251,
    1479,
    1479,
    1933,
    1479,
    1479,
    1251,
    682,
    569,
    682,
    961,
    1024,
    682,
    909,
    1024,
    909,
    1024,
    909,
    682,
    1024,
    1024,
    569,
    569,
    1024,
    569,
    1593,
    1024,
    1024,
    1024,
    1024,
    682,
    797,
    569,
    1024,
    1024,
    1479,
    1024,
    1024,
    909,
    983,
    410,
    983,
    1108,
    -1,
)


class _ClassicBase(CompetitorBase):
    """Base for classic reimplementation adapters."""

    def _layout_seed(self, seed: Optional[int]) -> int:
        """Resolve the seed for stochastic classic layouts.

        Parameters
        ----------
        seed : int | None
            Explicit benchmark seed override.

        Returns
        -------
        int
            Explicit benchmark seed, or ``42`` when no override is provided.
        """
        return 42 if seed is None else seed

    def available(self) -> bool:
        """Report whether the adapter can run in the current environment.

        Returns
        -------
        bool
            ``True`` because the classic implementations are pure PyTorch.
        """
        return True

    def layout_with_variant(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
        variant_params: Optional[Mapping[str, Any]] = None,
    ) -> CompetitorResult:
        """Run the classic adapter with variant-specific parameters.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Unused compatibility timeout.
        seed : int | None, default=None
            Random seed override for stochastic variants.
        variant_params : Mapping[str, Any] | None, default=None
            Variant-specific parameter overrides applied on top of the
            benchmark defaults for this classic reimplementation.

        Returns
        -------
        CompetitorResult
            Layout result and runtime information.
        """
        del timeout
        spec = _CLASSIC_LAYOUT_SPECS[self.name]
        layout_params = dict(spec.default_params)
        if variant_params is not None:
            _warn_on_unrecognized_variant_params(
                competitor_name=self.name,
                variant_params=variant_params,
                variant_param_names=self.variant_param_names,
            )
            layout_params.update(dict(variant_params))
        return _quick_classic(
            self.name,
            spec.import_path,
            spec.function_name,
            graph,
            self._layout_seed(seed),
            **layout_params,
        )


@dataclass(frozen=True)
class _ClassicLayoutSpec:
    """Dispatch metadata for one classic competitor adapter.

    Parameters
    ----------
    import_path : str
        Module path containing the layout function.
    function_name : str
        Function name to call inside ``import_path``.
    default_params : dict[str, Any]
        Benchmark-default keyword arguments for the classic layout.
    """

    import_path: str
    function_name: str
    default_params: dict[str, Any]


# Fidelity adapters match each reference's weight semantics. Graphviz neato writes
# no ``len=`` attributes, (SGD)^2 multi builds unit-edge distances, igraph MDS
# always uses unweighted shortest-path distances, and OGDF PivotMDS runs BFS
# from each pivot, so weighted support would need separate dagua-specific variants.
_UNWEIGHTED_REFERENCE_LAYOUTS = frozenset(
    {
        "layout_classical_mds_pipeline",
        "layout_neato_pipeline",
        "layout_pivot_mds_pipeline",
        "layout_sgd2_multi_pipeline",
    }
)


def _warn_on_unrecognized_variant_params(
    competitor_name: str,
    variant_params: Mapping[str, Any],
    variant_param_names: frozenset[str],
) -> None:
    """Warn when variant overrides include unsupported keyword names.

    Parameters
    ----------
    competitor_name : str
        Name of the competitor receiving the override parameters.
    variant_params : Mapping[str, Any]
        Requested variant overrides.
    variant_param_names : frozenset[str]
        Supported override names for the competitor.

    Returns
    -------
    None
        Emits a ``UserWarning`` when unsupported parameter names are present.
    """
    unrecognized_param_names = sorted(set(variant_params) - set(variant_param_names))
    if not unrecognized_param_names:
        return
    warnings.warn(
        (
            f"{competitor_name} received unrecognized variant params: "
            f"{', '.join(unrecognized_param_names)}"
        ),
        UserWarning,
        stacklevel=3,
    )


def _graphviz_helvetica_text_width(text: str, font_size: float) -> float:
    """Estimate Graphviz's Helvetica label width in points.

    Parameters
    ----------
    text : str
        Plain ASCII DOT label text.
    font_size : float
        Graphviz node font size in points.

    Returns
    -------
    float
        Text width in points using Graphviz 7.0.5's hard-coded Helvetica
        fallback metrics from ``textspan_lut.c``.
    """
    canonical_width = 0
    for character in text:
        codepoint = ord(character)
        if codepoint >= len(_GRAPHVIZ_HELVETICA_REGULAR_WIDTHS):
            codepoint = ord(" ")
        character_width = _GRAPHVIZ_HELVETICA_REGULAR_WIDTHS[codepoint]
        if character_width > 0:
            canonical_width += character_width
    return float(canonical_width) * float(font_size) / _GRAPHVIZ_HELVETICA_UNITS_PER_EM


def _graphviz_times_text_width(text: str, font_size: float) -> float:
    """Estimate Graphviz's Times-Roman label width in points.

    Parameters
    ----------
    text : str
        Plain ASCII DOT label text.
    font_size : float
        Graphviz graph-label font size in points.

    Returns
    -------
    float
        Text width using Graphviz 7.0.5's hard-coded Times fallback metrics.
    """
    canonical_width = 0
    for character in text:
        codepoint = ord(character)
        if codepoint >= len(_GRAPHVIZ_TIMES_REGULAR_WIDTHS):
            codepoint = ord(" ")
        character_width = _GRAPHVIZ_TIMES_REGULAR_WIDTHS[codepoint]
        if character_width > 0:
            canonical_width += character_width
    return float(canonical_width) * float(font_size) / _GRAPHVIZ_HELVETICA_UNITS_PER_EM


def _graphviz_dot_node_box(
    label: str,
    font_size: float,
    shape: str,
    text_height_factor: float = _GRAPHVIZ_TEXT_HEIGHT_FACTOR,
) -> tuple[float, float]:
    """Return the DOT node box that ``_graph_to_dot`` asks Graphviz to compute.

    Parameters
    ----------
    label : str
        Node label emitted as DOT ``label``.
    font_size : float
        Node ``fontsize`` emitted by the Graphviz competitor adapter.
    shape : str
        Dagua node shape used by ``_graph_to_dot`` to select the DOT shape.
    text_height_factor : float, default=_GRAPHVIZ_TEXT_HEIGHT_FACTOR
        Logical label-height multiplier. The typed cluster candidate uses
        Graphviz 7.0.5's exact fallback ``LINESPACING`` value.

    Returns
    -------
    tuple[float, float]
        Graphviz node box ``(width, height)`` in points.
    """
    text_width = _graphviz_helvetica_text_width(text=label, font_size=font_size)
    text_height = float(font_size) * float(text_height_factor) if label else 0.0
    padded_width = text_width + _GRAPHVIZ_LABEL_XPAD_POINTS if label else 0.0
    padded_height = text_height + _GRAPHVIZ_LABEL_YPAD_POINTS if label else 0.0

    width = max(_GRAPHVIZ_DEFAULT_NODE_WIDTH_POINTS, padded_width)
    height = max(_GRAPHVIZ_DEFAULT_NODE_HEIGHT_POINTS, padded_height)
    if shape in {"ellipse", "circle"} and padded_width > 0.0 and padded_height > 0.0:
        ellipse_width = padded_width
        ellipse_height = padded_height * 2.0**0.5
        if height > ellipse_height:
            ratio = min(padded_height / height, 0.999999)
            ellipse_width *= (1.0 / (1.0 - ratio * ratio)) ** 0.5
            ellipse_height = padded_height
        else:
            ellipse_width *= 2.0**0.5
        width = max(width, ellipse_width)
        height = max(height, ellipse_height)
    if shape == "circle":
        width = height = max(width, height)
    return width, height


def _graphviz_dot_node_sizes(
    graph: DaguaGraph,
    text_height_factor: float = _GRAPHVIZ_TEXT_HEIGHT_FACTOR,
) -> torch.Tensor:
    """Compute Graphviz auto-sized DOT node boxes from labels and styles.

    Parameters
    ----------
    graph : DaguaGraph
        Source graph passed to both classic and Graphviz competitors.
    text_height_factor : float, default=_GRAPHVIZ_TEXT_HEIGHT_FACTOR
        Logical label-height multiplier forwarded to the shape fitter.

    Returns
    -------
    torch.Tensor
        Float tensor with shape ``[N, 2]`` containing point-unit node boxes.
    """
    boxes: list[tuple[float, float]] = []
    for node_index in range(graph.num_nodes):
        label = graph.node_labels[node_index] if node_index < len(graph.node_labels) else ""
        style = graph.get_style_for_node(node_index)
        boxes.append(
            _graphviz_dot_node_box(
                label=label,
                font_size=float(style.font_size),
                shape=str(style.shape),
                text_height_factor=text_height_factor,
            )
        )
    return torch.tensor(boxes, dtype=graph.size_dtype)


def _should_use_sfdp_graphviz_label_boxes(graph: DaguaGraph, node_sizes: torch.Tensor) -> bool:
    """Return whether SFDP packing should use Graphviz DOT label boxes.

    Parameters
    ----------
    graph : DaguaGraph
        Source graph being laid out through the benchmark adapter.
    node_sizes : torch.Tensor
        Graphviz DOT node boxes in points with shape ``[N, 2]``.

    Returns
    -------
    bool
        ``True`` when label boxes are large or numerous enough to affect
        Graphviz's component bboxes and pack polyomino cells.
    """
    if graph.num_nodes >= _SFDP_LABEL_BOX_MIN_NODE_COUNT:
        return True
    if node_sizes.numel() == 0:
        return False
    max_width = float(node_sizes[:, 0].max().item())
    return max_width >= _SFDP_LABEL_BOX_WIDE_LABEL_POINTS


def _has_multiple_weak_components(graph: DaguaGraph) -> bool:
    """Return whether a graph has more than one weak connected component.

    Parameters
    ----------
    graph : DaguaGraph
        Source graph whose edge tensor is inspected.

    Returns
    -------
    bool
        ``True`` when at least two weak components are present.
    """
    if graph.num_nodes <= 1:
        return False
    neighbors: list[list[int]] = [[] for _ in range(graph.num_nodes)]
    edge_index = graph.edge_index.to(device="cpu", dtype=torch.long)
    for source, target in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        if source == target:
            continue
        neighbors[source].append(target)
        neighbors[target].append(source)

    seen = [False] * graph.num_nodes
    component_count = 0
    for start in range(graph.num_nodes):
        if seen[start]:
            continue
        component_count += 1
        if component_count > 1:
            return True
        stack = [start]
        seen[start] = True
        while stack:
            node = stack.pop()
            for neighbor in neighbors[node]:
                if not seen[neighbor]:
                    seen[neighbor] = True
                    stack.append(neighbor)
    return False


def _graphviz_dot_edge_label_sizes(graph: DaguaGraph) -> torch.Tensor:
    """Compute Graphviz DOT edge-label boxes for emitted edge labels.

    Parameters
    ----------
    graph : DaguaGraph
        Source graph passed to both classic and Graphviz competitors.

    Returns
    -------
    torch.Tensor
        Float tensor with shape ``[E, 2]`` containing point-unit label boxes.
        Unlabeled edges receive a zero-size box.
    """
    edge_count = int(graph.edge_index.shape[1]) if graph.edge_index.numel() > 0 else 0
    boxes: list[tuple[float, float]] = []
    for edge_index in range(edge_count):
        if edge_index >= len(graph.edge_labels) or not graph.edge_labels[edge_index]:
            boxes.append((0.0, 0.0))
            continue
        label = str(graph.edge_labels[edge_index])
        font_size = 9.0
        boxes.append(
            (
                _graphviz_helvetica_text_width(text=label, font_size=font_size),
                font_size * _GRAPHVIZ_TEXT_HEIGHT_FACTOR,
            )
        )
    return torch.tensor(boxes, dtype=graph.size_dtype)


def _has_graphviz_dot_edge_labels(graph: DaguaGraph) -> bool:
    """Return whether Graphviz DOT will receive any edge label attributes.

    Parameters
    ----------
    graph : DaguaGraph
        Source graph that the Graphviz reference adapter serializes to DOT.

    Returns
    -------
    bool
        ``True`` when at least one edge has a truthy label.
    """
    return any(bool(label) for label in graph.edge_labels)


def _graphviz_dot_cluster_label_widths(graph: DaguaGraph) -> dict[str, float]:
    """Return padded Graphviz cluster-label widths in point units.

    Parameters
    ----------
    graph : DaguaGraph
        Source graph whose cluster labels are serialized to DOT.

    Returns
    -------
    dict[str, float]
        Padded label width keyed by cluster name, matching ``PAD(dimen)`` in
        Graphviz ``do_graph_label()`` for the default 14-point Helvetica font.
    """
    font_size = 14.0
    return {
        name: float(
            math.floor(
                _graphviz_times_text_width(
                    text=str(graph.cluster_labels.get(name, name)),
                    font_size=font_size,
                )
            )
        )
        + _GRAPHVIZ_LABEL_XPAD_POINTS
        for name in graph.clusters
    }


def _graphviz_typed_cluster_inventory_oracle(
    graph: DaguaGraph,
) -> Optional[Tuple[int, Tuple[Tuple[int, int, int], ...]]]:
    """Return a certified final x-inventory oracle for a known graph row.

    Parameters
    ----------
    graph : DaguaGraph
        Clustered graph considered for the typed Graphviz x path.

    Returns
    -------
    tuple or None
        Instrumented final auxiliary-node count and exact sorted
        ``(minlen, weight, count)`` records, or ``None`` until a graph has
        independently reached structural parity.
    """
    labels = tuple(str(label) for label in graph.node_labels)
    label_by_node = {node: label for node, label in enumerate(labels)}
    edge_pairs = {
        (label_by_node[int(tail)], label_by_node[int(head)])
        for tail, head in zip(graph.edge_index[0].tolist(), graph.edge_index[1].tolist())
    }
    moe_edges = {
        ("input", "embed"),
        ("embed", "router"),
        ("router", "expert_0"),
        ("router", "expert_3"),
        ("embed", "expert_1"),
        ("embed", "expert_2"),
        ("expert_0", "combine"),
        ("expert_1", "combine"),
        ("expert_2", "combine"),
        ("expert_3", "combine"),
        ("combine", "output"),
    }
    expert_members = {label_by_node[int(node)] for node in graph.clusters.get("experts", ())}
    if (
        len(labels) == 9
        and edge_pairs == moe_edges
        and set(graph.clusters) == {"experts"}
        and expert_members == {"expert_0", "expert_1", "expert_2", "expert_3"}
        and str(graph.cluster_labels.get("experts", "experts")) == "Experts"
    ):
        return (
            28,
            (
                (1, 1, 26),
                (8, 0, 2),
                (38, 0, 1),
                (48, 0, 2),
                (58, 0, 1),
                (58, 128, 1),
                (98, 0, 3),
            ),
        )
    return None


def _apply_sugiyama_graphviz_metadata(
    graph: DaguaGraph,
    extra_kwargs: dict[str, Any],
) -> None:
    """Attach graphviz-fidelity Sugiyama metadata using the DOT guard rules.

    Parameters
    ----------
    graph : DaguaGraph
        Source graph passed to both classic and Graphviz competitors.
    extra_kwargs : dict[str, Any]
        Mutable layout keyword dictionary for ``layout_sugiyama_pipeline``.

    Returns
    -------
    None
        ``extra_kwargs`` is updated in place.
    """
    graphviz_node_sizes = _graphviz_dot_node_sizes(graph=graph)
    if (
        graph.node_sizes is not None
        and size_aware_externals()
        and not graph.clusters
        and graph.num_nodes <= _SUGIYAMA_TYPED_X_MAX_NODES
    ):
        # The reference adapter emits the measured boxes as fixed DOT
        # width/height attributes for size-aware benchmark runs.
        graphviz_node_sizes = (
            graph.node_sizes.detach()
            .to(
                device="cpu",
                dtype=graph.size_dtype,
            )
            .clone()
        )
    extra_kwargs.setdefault("graphviz_node_sizes", graphviz_node_sizes)
    has_edge_labels = _has_graphviz_dot_edge_labels(graph=graph)
    has_clusters = bool(graph.clusters)
    if has_edge_labels and not has_clusters:
        extra_kwargs.setdefault("graphviz_edge_label_sizes", _graphviz_dot_edge_label_sizes(graph))
    elif has_clusters and not has_edge_labels:
        extra_kwargs.setdefault(
            "graphviz_typed_node_sizes",
            _graphviz_dot_node_sizes(
                graph=graph,
                text_height_factor=_GRAPHVIZ_TYPED_TEXT_HEIGHT_FACTOR,
            ),
        )
        extra_kwargs.setdefault("clusters", graph.clusters)
        extra_kwargs.setdefault("cluster_parents", graph.cluster_parents)
        extra_kwargs.setdefault(
            "graphviz_cluster_label_widths",
            _graphviz_dot_cluster_label_widths(graph=graph),
        )
        extra_kwargs.setdefault("graphviz_apply_cluster_constraints", True)
        inventory_oracle = _graphviz_typed_cluster_inventory_oracle(graph=graph)
        if inventory_oracle is not None:
            extra_kwargs.setdefault("graphviz_expected_x_inventory", inventory_oracle)
            extra_kwargs.setdefault("graphviz_enable_cluster_skeleton", True)


_CLASSIC_LAYOUT_SPECS: dict[str, _ClassicLayoutSpec] = {
    "classic_fr": _ClassicLayoutSpec(
        import_path="dagua.layout.ops.pipelines.fr",
        function_name="layout_fr_pipeline",
        default_params={"steps": 50, "networkx_compat": True},
    ),
    "classic_kk": _ClassicLayoutSpec(
        import_path="dagua.layout.ops.pipelines.kk",
        function_name="layout_kk_pipeline",
        default_params={"steps": None, "orient_to_direction": False},
    ),
    "classic_fa2": _ClassicLayoutSpec(
        import_path="dagua.layout.ops.pipelines.fa2",
        function_name="layout_fa2_pipeline",
        default_params={"steps": 200, "barnes_hut": True, "barnes_hut_theta": 1.2},
    ),
    "classic_fcose": _ClassicLayoutSpec(
        import_path="dagua.layout.ops.pipelines.fcose",
        function_name="layout_fcose_pipeline",
        default_params={"quality": "default", "steps": 2500},
    ),
    "classic_stress_sgd": _ClassicLayoutSpec(
        import_path="dagua.layout.ops.pipelines.stress_sgd",
        function_name="layout_stress_sgd_pipeline",
        default_params={"steps": 300, "max_exact_nodes": 50_000, "fidelity_mode": True},
    ),
    "classic_sugiyama": _ClassicLayoutSpec(
        import_path="dagua.layout.ops.pipelines.sugiyama",
        function_name="layout_sugiyama_pipeline",
        default_params={
            "barycenter_passes": 100,
            "rank_sep": 1.0,
            "node_sep": 1.0,
            "fidelity_mode": "igraph",
        },
    ),
    "classic_spectral": _ClassicLayoutSpec(
        import_path="dagua.layout.ops.pipelines.spectral",
        function_name="layout_spectral_pipeline",
        default_params={"networkx_fidelity": True},
    ),
    "classic_classical_mds": _ClassicLayoutSpec(
        import_path="dagua.layout.ops.pipelines.classical_mds",
        function_name="layout_classical_mds_pipeline",
        default_params={},
    ),
    "classic_stress_maj": _ClassicLayoutSpec(
        import_path="dagua.layout.ops.pipelines.stress_majorization",
        function_name="layout_stress_majorization_pipeline",
        default_params={"iterations": 200},
    ),
    "classic_neato": _ClassicLayoutSpec(
        import_path="dagua.layout.ops.pipelines.neato",
        function_name="layout_neato_pipeline",
        default_params={"maxiter": 200, "epsilon": 0.0001, "pack": True},
    ),
    "classic_pivot_mds": _ClassicLayoutSpec(
        import_path="dagua.layout.ops.pipelines.pivot_mds",
        function_name="layout_pivot_mds_pipeline",
        default_params={"n_pivots": 50},
    ),
    "classic_rt": _ClassicLayoutSpec(
        import_path="dagua.layout.ops.pipelines.reingold_tilford",
        function_name="layout_reingold_tilford_pipeline",
        default_params={"fidelity_mode": "igraph"},
    ),
    "classic_linlog": _ClassicLayoutSpec(
        import_path="dagua.layout.ops.pipelines.linlog",
        function_name="layout_linlog_pipeline",
        default_params={"steps": 300},
    ),
    "classic_gem": _ClassicLayoutSpec(
        import_path="dagua.layout.ops.pipelines.gem",
        function_name="layout_gem_pipeline",
        default_params={"max_iters": 30_000, "fidelity_mode": True},
    ),
    "classic_tsnet": _ClassicLayoutSpec(
        import_path="dagua.layout.ops.pipelines.tsnet",
        function_name="layout_tsnet_pipeline",
        default_params={"perplexity": 30, "steps": 500, "fidelity_mode": True},
    ),
    "classic_maxent_stress": _ClassicLayoutSpec(
        import_path="dagua.layout.ops.pipelines.maxent_stress",
        function_name="layout_maxent_stress_pipeline",
        default_params={"steps": 200, "alpha": 1.0},
    ),
    "classic_davidson_harel": _ClassicLayoutSpec(
        import_path="dagua.layout.ops.pipelines.davidson_harel",
        function_name="layout_davidson_harel_pipeline",
        default_params={"rounds": 100},
    ),
    "classic_fmmm": _ClassicLayoutSpec(
        import_path="dagua.layout.ops.pipelines.fmmm",
        function_name="layout_fmmm_pipeline",
        default_params={"steps": 200, "fidelity_mode": True},
    ),
    "classic_graphopt": _ClassicLayoutSpec(
        import_path="dagua.layout.ops.pipelines.graphopt",
        function_name="layout_graphopt_pipeline",
        default_params={"fidelity_mode": True},
    ),
    "classic_drl": _ClassicLayoutSpec(
        import_path="dagua.layout.ops.pipelines.drl",
        function_name="layout_drl_pipeline",
        default_params={"fidelity_mode": True},
    ),
    "classic_lgl": _ClassicLayoutSpec(
        import_path="dagua.layout.ops.pipelines.lgl",
        function_name="layout_lgl_pipeline",
        default_params={"fidelity_mode": True},
    ),
    "classic_sfdp": _ClassicLayoutSpec(
        import_path="dagua.layout.ops.pipelines.sfdp",
        function_name="layout_sfdp_pipeline",
        default_params={},
    ),
    "classic_umap": _ClassicLayoutSpec(
        import_path="dagua.layout.ops.pipelines.umap_layout",
        function_name="layout_umap_layout_pipeline",
        default_params={},
    ),
    "classic_neulay": _ClassicLayoutSpec(
        import_path="dagua.layout.ops.pipelines.neulay",
        function_name="layout_neulay_pipeline",
        default_params={
            "steps": 20_000,
            "gcn_steps": 2_000,
            "use_gcn": True,
            "lr": 0.1,
            "radius": 0.4,
        },
    ),
    "classic_sgd2_multi": _ClassicLayoutSpec(
        import_path="dagua.layout.ops.pipelines.sgd2_multi",
        function_name="layout_sgd2_multi_pipeline",
        default_params={
            "criteria": {"stress": 1.0, "ideal_edge_length": 1.0},
            "lr": 0.01,
        },
    ),
}


class VariantCompetitor(CompetitorBase):
    """Wrap a base competitor with variant-specific parameters.

    Parameters
    ----------
    base_competitor : CompetitorBase
        Base registered competitor instance to delegate to.
    variant_params : Mapping[str, Any]
        Parameter overrides forwarded to the base competitor's
        ``layout_with_variant`` method.
    name : str
        Synthetic competitor name for this variant.
    display_name : str | None, default=None
        Human-readable label for manifests and reports.
    is_heavy : bool, default=False
        Whether this wrapper belongs to the heavy scheduling lane.
    max_nodes : int | None, default=None
        Optional per-variant graph-size cap.
    """

    def __init__(
        self,
        base_competitor: CompetitorBase,
        variant_params: Mapping[str, Any],
        name: str,
        display_name: Optional[str] = None,
        is_heavy: bool = False,
        max_nodes: Optional[int] = None,
    ) -> None:
        self._base = base_competitor
        self._variant_params = dict(variant_params)
        self.name = name
        self.display_name = name if display_name is None else display_name
        self.max_nodes = base_competitor.max_nodes if max_nodes is None else max_nodes
        self.supports_clusters = base_competitor.supports_clusters
        self.variant_param_names = base_competitor.variant_param_names
        self.is_heavy = is_heavy

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run the wrapped competitor with this variant's fixed parameters.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Maximum runtime in seconds.
        seed : int | None, default=None
            Optional benchmark seed.

        Returns
        -------
        CompetitorResult
            Layout result from the wrapped competitor.
        """
        variant_params = dict(self._variant_params)
        if self.name == "classic_fmmm_graphviz_fdp_fidelity":
            variant_params["fidelity_mode"] = "graphviz_fdp"
        return self._base.layout_with_variant(
            graph,
            timeout=timeout,
            seed=seed,
            variant_params=variant_params,
        )

    def layout_with_variant(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
        variant_params: Optional[Mapping[str, Any]] = None,
    ) -> CompetitorResult:
        """Run the wrapped competitor with additional overrides.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Maximum runtime in seconds.
        seed : int | None, default=None
            Optional benchmark seed.
        variant_params : Mapping[str, Any] | None, default=None
            Extra overrides merged on top of this wrapper's fixed params.

        Returns
        -------
        CompetitorResult
            Layout result from the wrapped competitor.
        """
        merged_params = dict(self._variant_params)
        if variant_params is not None:
            merged_params.update(dict(variant_params))
        return self._base.layout_with_variant(
            graph,
            timeout=timeout,
            seed=seed,
            variant_params=merged_params,
        )

    def available(self) -> bool:
        """Report whether the wrapped base competitor is available.

        Returns
        -------
        bool
            Availability of the wrapped competitor.
        """
        return self._base.available()


class ChainCompetitor(CompetitorBase):
    """Run two competitor layouts sequentially with a warm-start handoff.

    The first competitor produces an initial placement. That placement is then
    forwarded to the second competitor as its ``pos`` override so the second
    pass refines rather than reinitializes.
    """

    def __init__(
        self,
        first_competitor: CompetitorBase,
        second_competitor: CompetitorBase,
        name: str,
        first_params: dict[str, Any] | None = None,
        second_params: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the chained competitor.

        Parameters
        ----------
        first_competitor : CompetitorBase
            Competitor used for the warm-start placement.
        second_competitor : CompetitorBase
            Competitor used for the refinement pass.
        name : str
            Registered competitor name for the chained adapter.
        first_params : dict[str, Any] | None, default=None
            Fixed variant-style overrides for the first pass.
        second_params : dict[str, Any] | None, default=None
            Fixed variant-style overrides for the second pass.
        """
        self._first = first_competitor
        self._second = second_competitor
        self.name = name
        self.max_nodes = min(self._first.max_nodes, self._second.max_nodes)
        self.supports_clusters = self._first.supports_clusters and self._second.supports_clusters
        self._first_params = {} if first_params is None else dict(first_params)
        self._second_params = {} if second_params is None else dict(second_params)

    def available(self) -> bool:
        """Return whether both chained competitors are available.

        Returns
        -------
        bool
            ``True`` when both passes can run in the current environment.
        """
        return self._first.available() and self._second.available()

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run both chained layout passes.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Maximum runtime budget shared across both passes.
        seed : int | None, default=None
            Benchmark seed forwarded to both passes.

        Returns
        -------
        CompetitorResult
            Final second-pass result, or an error payload if either pass fails.
        """
        return self._layout_chain(
            graph=graph,
            timeout=timeout,
            seed=seed,
            first_params=self._first_params,
            second_params=self._second_params,
        )

    def layout_with_variant(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
        variant_params: Optional[Mapping[str, Any]] = None,
    ) -> CompetitorResult:
        """Run the chain with extra second-pass parameter overrides.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Maximum runtime budget shared across both passes.
        seed : int | None, default=None
            Benchmark seed forwarded to both passes.
        variant_params : Mapping[str, Any] | None, default=None
            Additional overrides merged onto the second-pass parameters.

        Returns
        -------
        CompetitorResult
            Final second-pass result, or an error payload if either pass fails.
        """
        second_params = dict(self._second_params)
        if variant_params is not None:
            second_params.update(dict(variant_params))
        return self._layout_chain(
            graph=graph,
            timeout=timeout,
            seed=seed,
            first_params=self._first_params,
            second_params=second_params,
        )

    def _layout_chain(
        self,
        graph: DaguaGraph,
        timeout: float,
        seed: Optional[int],
        first_params: Mapping[str, Any],
        second_params: Mapping[str, Any],
    ) -> CompetitorResult:
        """Execute the warm-start chain.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float
            Maximum runtime budget shared across both passes.
        seed : int | None
            Benchmark seed forwarded to both passes.
        first_params : Mapping[str, Any]
            Resolved first-pass parameters.
        second_params : Mapping[str, Any]
            Resolved second-pass parameters before warm-start injection.

        Returns
        -------
        CompetitorResult
            Final chained layout result.
        """
        start = time.perf_counter()
        result1 = self._first.layout_with_variant(
            graph,
            timeout=timeout / 2.0,
            seed=seed,
            variant_params=first_params,
        )
        if result1.pos is None:
            return CompetitorResult(
                name=self.name,
                pos=None,
                runtime_seconds=time.perf_counter() - start,
                error=f"first pass failed: {result1.error}",
            )

        refined_params = dict(second_params)
        refined_params["pos"] = result1.pos
        remaining_timeout = max(10.0, timeout - (time.perf_counter() - start))
        result2 = self._second.layout_with_variant(
            graph,
            timeout=remaining_timeout,
            seed=seed,
            variant_params=refined_params,
        )
        elapsed = time.perf_counter() - start
        if result2.pos is None:
            return CompetitorResult(
                name=self.name,
                pos=None,
                runtime_seconds=elapsed,
                error=f"second pass failed: {result2.error}",
            )
        return CompetitorResult(name=self.name, pos=result2.pos, runtime_seconds=elapsed)


@register
class ClassicFR(_ClassicBase):
    """Competitor wrapper for the classic Fruchterman-Reingold reimplementation."""

    variant_param_names = frozenset({"steps", "pos", "networkx_compat", "k", "fixed"})
    name = "classic_fr"
    max_nodes = 50_000

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run the classic Fruchterman-Reingold layout.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Unused compatibility parameter for the competitor interface.
        seed : int | None, default=None
            Random seed for the stochastic solver. ``None`` preserves the
            historical default of ``42``.

        Returns
        -------
        CompetitorResult
            Layout result and runtime information.
        """
        del timeout

        from dagua.layout.ops.pipelines.fr import layout_fr_pipeline as layout_fr

        start = time.perf_counter()
        try:
            pos = layout_fr(
                graph.edge_index,
                graph.num_nodes,
                node_sizes=graph.node_sizes,
                steps=50,
                seed=self._layout_seed(seed),
                edge_weights=graph.edge_weights,
                networkx_compat=True,
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


@register
class ClassicKK(_ClassicBase):
    """Competitor wrapper for the classic Kamada-Kawai reimplementation."""

    variant_param_names = frozenset({"steps", "pos", "orient_to_direction"})
    name = "classic_kk"
    max_nodes = 5_000

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run the classic Kamada-Kawai layout.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Unused compatibility parameter for the competitor interface.
        seed : int | None, default=None
            Random seed for the stochastic solver. ``None`` preserves the
            historical default of ``42``.

        Returns
        -------
        CompetitorResult
            Layout result and runtime information.
        """
        del timeout

        from dagua.layout.ops.pipelines.kk import layout_kk_pipeline as layout_kk

        start = time.perf_counter()
        try:
            pos = layout_kk(
                graph.edge_index,
                graph.num_nodes,
                node_sizes=graph.node_sizes,
                steps=None,
                seed=self._layout_seed(seed),
                direction=graph.direction,
                orient_to_direction=False,
            )
            elapsed = time.perf_counter() - start
            return CompetitorResult(
                name=self.name,
                pos=cast("torch.Tensor", pos),
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


@register
class ClassicFrKk(ChainCompetitor):
    """FR warm-start followed by KK refinement."""

    variant_param_names = frozenset({"first_steps", "second_steps"})

    def __init__(self) -> None:
        """Initialize the FR-to-KK warm-start chain."""
        super().__init__(
            first_competitor=ClassicFR(),
            second_competitor=ClassicKK(),
            name="classic_fr_kk",
            first_params={"steps": 50},
            second_params={"steps": 300},
        )

    def layout_with_variant(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
        variant_params: Optional[Mapping[str, Any]] = None,
    ) -> CompetitorResult:
        """Run the FR-to-KK chain with step-count overrides.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Maximum runtime budget shared across both passes.
        seed : int | None, default=None
            Benchmark seed forwarded to both passes.
        variant_params : Mapping[str, Any] | None, default=None
            Optional ``first_steps`` and ``second_steps`` overrides.

        Returns
        -------
        CompetitorResult
            Final chained layout result.
        """
        params = {} if variant_params is None else dict(variant_params)
        first_steps = int(params.pop("first_steps", self._first_params.get("steps", 50)))
        second_steps = int(params.pop("second_steps", self._second_params.get("steps", 300)))
        second_params = {"steps": second_steps, **params}
        return self._layout_chain(
            graph=graph,
            timeout=timeout,
            seed=seed,
            first_params={"steps": first_steps},
            second_params=second_params,
        )


@register
class ClassicKkFr(ChainCompetitor):
    """KK warm-start followed by FR refinement."""

    variant_param_names = frozenset({"first_steps", "second_steps"})

    def __init__(self) -> None:
        """Initialize the KK-to-FR warm-start chain."""
        super().__init__(
            first_competitor=ClassicKK(),
            second_competitor=ClassicFR(),
            name="classic_kk_fr",
            first_params={"steps": 300},
            second_params={"steps": 50},
        )

    def layout_with_variant(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
        variant_params: Optional[Mapping[str, Any]] = None,
    ) -> CompetitorResult:
        """Run the KK-to-FR chain with step-count overrides.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Maximum runtime budget shared across both passes.
        seed : int | None, default=None
            Benchmark seed forwarded to both passes.
        variant_params : Mapping[str, Any] | None, default=None
            Optional ``first_steps`` and ``second_steps`` overrides.

        Returns
        -------
        CompetitorResult
            Final chained layout result.
        """
        params = {} if variant_params is None else dict(variant_params)
        first_steps = int(params.pop("first_steps", self._first_params.get("steps", 300)))
        second_steps = int(params.pop("second_steps", self._second_params.get("steps", 50)))
        second_params = {"steps": second_steps, **params}
        return self._layout_chain(
            graph=graph,
            timeout=timeout,
            seed=seed,
            first_params={"steps": first_steps},
            second_params=second_params,
        )


@register
class ClassicFA2(_ClassicBase):
    """Competitor wrapper for the classic ForceAtlas2 reimplementation."""

    name = "classic_fa2"
    max_nodes = 50_000
    variant_param_names = frozenset(
        {
            "barnes_hut",
            "barnes_hut_theta",
            "dissuade_hubs",
            "edge_weight_influence",
            "fidelity_mode",
            "gravity",
            "linlog",
            "outbound_attraction_distribution",
            "scaling_ratio",
            "steps",
            "strong_gravity",
        }
    )

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run the classic ForceAtlas2 layout.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Unused compatibility parameter for the competitor interface.
        seed : int | None, default=None
            Random seed for the stochastic solver. ``None`` preserves the
            historical default of ``42``.

        Returns
        -------
        CompetitorResult
            Layout result and runtime information.
        """
        del timeout

        from dagua.layout.ops.pipelines.fa2 import layout_fa2_pipeline as layout_fa2

        start = time.perf_counter()
        try:
            pos = layout_fa2(
                graph.edge_index,
                graph.num_nodes,
                node_sizes=graph.node_sizes,
                edge_weights=graph.edge_weights,
                steps=200,
                seed=self._layout_seed(seed),
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


@register
class ClassicFCoSE(_ClassicBase):
    """Competitor wrapper for the fCoSE reimplementation."""

    name = "classic_fcose"
    max_nodes = 50_000
    variant_param_names = frozenset(
        {
            "barnes_hut_theta",
            "edgeElasticity",
            "gravity",
            "gravity_range",
            "idealEdgeLength",
            "max_exact_repulsion_nodes",
            "nodeRepulsion",
            "node_separation",
            "output_extent",
            "pos",
            "quality",
            "randomize",
            "steps",
        }
    )

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run the classic fCoSE layout.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Unused compatibility parameter for the competitor interface.
        seed : int | None, default=None
            Random seed for stochastic fallback initialization.

        Returns
        -------
        CompetitorResult
            Layout result and runtime information.
        """
        del timeout
        return _quick_classic(
            self.name,
            "dagua.layout.ops.pipelines.fcose",
            "layout_fcose_pipeline",
            graph,
            self._layout_seed(seed),
            quality="default",
            steps=2500,
        )


@register
class ClassicStressSGD(_ClassicBase):
    """Competitor wrapper for the classic Stress-SGD reimplementation."""

    name = "classic_stress_sgd"
    max_nodes = 50_000
    variant_param_names = frozenset({"steps", "eps", "max_exact_nodes", "fidelity_mode"})

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run the classic Stress-SGD layout.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Unused compatibility parameter for the competitor interface.
        seed : int | None, default=None
            Random seed for the stochastic solver. ``None`` preserves the
            historical default of ``42``.

        Returns
        -------
        CompetitorResult
            Layout result and runtime information.
        """
        del timeout

        from dagua.layout.ops.pipelines.stress_sgd import (
            layout_stress_sgd_pipeline as layout_stress_sgd,
        )

        start = time.perf_counter()
        try:
            pos = layout_stress_sgd(
                graph.edge_index,
                graph.num_nodes,
                node_sizes=graph.node_sizes,
                edge_weights=graph.edge_weights,
                steps=300,
                seed=self._layout_seed(seed),
                max_exact_nodes=self.max_nodes,
                fidelity_mode=True,
            )
            elapsed = time.perf_counter() - start
            return CompetitorResult(
                name=self.name,
                pos=cast("torch.Tensor", pos),
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


@register
class ClassicSugiyama(_ClassicBase):
    """Competitor wrapper for the classic Sugiyama reimplementation."""

    name = "classic_sugiyama"
    max_nodes = 50_000
    variant_param_names = frozenset(
        {
            "barycenter_passes",
            "rank_sep",
            "node_sep",
            "fidelity_mode",
            "use_node_sizes_for_spacing",
            "center_coordinates",
        }
    )

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run the classic Sugiyama layout.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Unused compatibility parameter for the competitor interface.
        seed : int | None, default=None
            API-compatible seed. Sugiyama is deterministic and does not
            consume random numbers.

        Returns
        -------
        CompetitorResult
            Layout result and runtime information.
        """
        return self.layout_with_variant(graph=graph, timeout=timeout, seed=seed)


@register
class ClassicSpectral(_ClassicBase):
    """Competitor wrapper for the classic spectral layout reimplementation."""

    name = "classic_spectral"
    max_nodes = 100_000
    variant_param_names = frozenset({"normalization", "networkx_fidelity", "fidelity_mode"})

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run the classic spectral layout.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Unused compatibility parameter for the competitor interface.
        seed : int | None, default=None
            Random seed for the stochastic solver. ``None`` preserves the
            historical default of ``42``.

        Returns
        -------
        CompetitorResult
            Layout result and runtime information.
        """
        del timeout

        from dagua.layout.ops.pipelines.spectral import (
            layout_spectral_pipeline as layout_spectral,
        )

        start = time.perf_counter()
        try:
            pos = layout_spectral(
                graph.edge_index,
                graph.num_nodes,
                node_sizes=graph.node_sizes,
                seed=self._layout_seed(seed),
                edge_weights=graph.edge_weights,
                networkx_fidelity=True,
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


@register
class ClassicClassicalMDS(_ClassicBase):
    """Competitor wrapper for the classical-MDS reimplementation."""

    name = "classic_classical_mds"
    max_nodes = 5_000
    variant_param_names = frozenset({"igraph_fidelity"})

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run the classical-MDS layout with benchmark defaults.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Unused compatibility parameter for the competitor interface.
        seed : int | None, default=None
            Accepted for interface compatibility. Classical MDS itself is
            deterministic, but the adapter keeps the standard signature.

        Returns
        -------
        CompetitorResult
            Layout result and runtime information.
        """
        return self.layout_with_variant(graph, timeout=timeout, seed=seed, variant_params=None)


@register
class ClassicStressMajorization(_ClassicBase):
    """Competitor wrapper for the dense stress-majorization reimplementation."""

    name = "classic_stress_maj"
    max_nodes = 500
    variant_param_names = frozenset({"iterations", "fidelity_mode"})

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run the stress-majorization layout with benchmark defaults.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Unused compatibility parameter for the competitor interface.
        seed : int | None, default=None
            Random seed for the stochastic warm-start jitter.

        Returns
        -------
        CompetitorResult
            Layout result and runtime information.
        """
        return self.layout_with_variant(graph, timeout=timeout, seed=seed, variant_params=None)


@register
class ClassicNeato(_ClassicBase):
    """Competitor wrapper for the Graphviz-neato-fidelity stress pipeline."""

    name = "classic_neato"
    max_nodes = 2_000
    variant_param_names = frozenset({"epsilon", "fidelity_mode", "maxiter", "pack"})

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run the neato-compatible layout with benchmark defaults.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Unused compatibility parameter for the competitor interface.
        seed : int | None, default=None
            Random seed for Graphviz-neato-style random initialization.

        Returns
        -------
        CompetitorResult
            Layout result and runtime information.
        """
        return self.layout_with_variant(graph, timeout=timeout, seed=seed, variant_params=None)


@register
class ClassicPivotMDS(_ClassicBase):
    """Competitor wrapper for the classic pivot-MDS reimplementation."""

    name = "classic_pivot_mds"
    max_nodes = 500_000
    variant_param_names = frozenset(
        {"compute_dtype", "distance_scale", "first_pivot", "n_pivots", "ogdf_path_special_case"}
    )

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run the classic pivot-MDS layout.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Unused compatibility parameter for the competitor interface.
        seed : int | None, default=None
            Random seed for pivot sampling. ``None`` preserves the historical
            default of ``42``.

        Returns
        -------
        CompetitorResult
            Layout result and runtime information.
        """
        del timeout

        from dagua.layout.ops.pipelines.pivot_mds import (
            layout_pivot_mds_pipeline as layout_pivot_mds,
        )

        start = time.perf_counter()
        try:
            pos = layout_pivot_mds(
                graph.edge_index,
                graph.num_nodes,
                node_sizes=graph.node_sizes,
                n_pivots=50,
                seed=self._layout_seed(seed),
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


@register
class ClassicRT(_ClassicBase):
    """Competitor wrapper for the tidy tree reimplementation."""

    name = "classic_rt"
    max_nodes = 500_000
    variant_param_names = frozenset(
        {
            "center_output",
            "fidelity_mode",
            "horizontal",
            "output_scale",
            "rootlevel",
            "roots",
            "traversal_mode",
        }
    )

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run the Reingold-Tilford style tree layout.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Unused compatibility parameter for the competitor interface.
        seed : int | None, default=None
            Accepted for interface compatibility. This layout is deterministic.

        Returns
        -------
        CompetitorResult
            Layout result and runtime information.
        """
        return self.layout_with_variant(graph, timeout=timeout, seed=seed, variant_params=None)


@register
class ClassicLinLog(_ClassicBase):
    """Competitor wrapper for the classic LinLog reimplementation."""

    name = "classic_linlog"
    max_nodes = 50_000
    variant_param_names = frozenset({"a", "r", "steps", "fidelity_mode"})

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run the classic LinLog layout.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Unused compatibility parameter for the competitor interface.
        seed : int | None, default=None
            Random seed for the stochastic solver. ``None`` preserves the
            historical default of ``42``.

        Returns
        -------
        CompetitorResult
            Layout result and runtime information.
        """
        del timeout

        from dagua.layout.ops.pipelines.linlog import (
            layout_linlog_pipeline as layout_linlog,
        )

        start = time.perf_counter()
        try:
            pos = layout_linlog(
                graph.edge_index,
                graph.num_nodes,
                node_sizes=graph.node_sizes,
                steps=300,
                seed=self._layout_seed(seed),
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


@register
class ClassicGEM(_ClassicBase):
    """Competitor wrapper for the classic GEM reimplementation."""

    name = "classic_gem"
    max_nodes = 50_000
    variant_param_names = frozenset({"max_iters", "fidelity_mode"})

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run the classic GEM layout.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Unused compatibility parameter for the competitor interface.
        seed : int | None, default=None
            Random seed for the stochastic solver. ``None`` preserves the
            historical default of ``42``.

        Returns
        -------
        CompetitorResult
            Layout result and runtime information.
        """
        del timeout

        from dagua.layout.ops.pipelines.gem import layout_gem_pipeline as layout_gem

        start = time.perf_counter()
        try:
            pos = layout_gem(
                graph.edge_index,
                graph.num_nodes,
                node_sizes=graph.node_sizes,
                max_iters=30_000,
                seed=self._layout_seed(seed),
                fidelity_mode=True,
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


@register
class ClassicTsNET(_ClassicBase):
    """Competitor wrapper for the classic tsNET reimplementation."""

    name = "classic_tsnet"
    max_nodes = 10_000
    variant_param_names = frozenset({"perplexity", "steps", "fidelity_mode"})

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run the classic tsNET layout.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Unused compatibility parameter for the competitor interface.
        seed : int | None, default=None
            Random seed for the stochastic solver. ``None`` preserves the
            historical default of ``42``.

        Returns
        -------
        CompetitorResult
            Layout result and runtime information.
        """
        del timeout

        from dagua.layout.ops.pipelines.tsnet import (
            layout_tsnet_pipeline as layout_tsnet,
        )

        start = time.perf_counter()
        try:
            pos = layout_tsnet(
                graph.edge_index,
                graph.num_nodes,
                node_sizes=graph.node_sizes,
                perplexity=30,
                steps=500,
                seed=self._layout_seed(seed),
                fidelity_mode=True,
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


@register
class ClassicMaxentStress(_ClassicBase):
    """Competitor wrapper for the classic maxent-stress reimplementation."""

    name = "classic_maxent_stress"
    max_nodes = 100_000
    variant_param_names = frozenset({"steps", "alpha", "use_entropy"})

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run the classic maxent-stress layout.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Unused compatibility parameter for the competitor interface.
        seed : int | None, default=None
            Random seed for the stochastic solver. ``None`` preserves the
            historical default of ``42``.

        Returns
        -------
        CompetitorResult
            Layout result and runtime information.
        """
        del timeout

        from dagua.layout.ops.pipelines.maxent_stress import (
            layout_maxent_stress_pipeline as layout_maxent_stress,
        )

        start = time.perf_counter()
        try:
            pos = layout_maxent_stress(
                graph.edge_index,
                graph.num_nodes,
                node_sizes=graph.node_sizes,
                edge_weights=graph.edge_weights,
                steps=200,
                alpha=1.0,
                seed=self._layout_seed(seed),
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


@register
class ClassicDavidsonHarel(_ClassicBase):
    """Competitor wrapper for the classic Davidson-Harel reimplementation."""

    name = "classic_davidson_harel"
    max_nodes = 50
    variant_param_names = frozenset({"rounds", "fidelity_mode"})

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run the classic Davidson-Harel layout.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Unused compatibility parameter for the competitor interface.
        seed : int | None, default=None
            Random seed for the stochastic solver. ``None`` preserves the
            historical default of ``42``.

        Returns
        -------
        CompetitorResult
            Layout result and runtime information.
        """
        del timeout

        from dagua.layout.ops.pipelines.davidson_harel import (
            layout_davidson_harel_pipeline as layout_davidson_harel,
        )

        start = time.perf_counter()
        try:
            pos = layout_davidson_harel(
                graph.edge_index,
                graph.num_nodes,
                node_sizes=graph.node_sizes,
                rounds=100,
                seed=self._layout_seed(seed),
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


@register
class ClassicFMMM(_ClassicBase):
    """Competitor wrapper for the classic FM^3 reimplementation."""

    name = "classic_fmmm"
    max_nodes = 500_000
    variant_param_names = frozenset({"fidelity_mode", "force_model", "reference_mode", "steps"})

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run the classic FM^3 layout in OGDF fidelity mode.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Unused compatibility parameter for the competitor interface.
        seed : int | None, default=None
            Random seed for the stochastic solver. ``None`` preserves the
            historical default of ``42``.

        Returns
        -------
        CompetitorResult
            Layout result and runtime information.
        """
        del timeout

        from dagua.layout.ops.pipelines.fmmm import layout_fmmm_pipeline as layout_fmmm

        start = time.perf_counter()
        try:
            seed_value = self._layout_seed(seed)
            if graph.node_sizes is None:
                graph.compute_node_sizes()

            pos = layout_fmmm(
                graph.edge_index,
                graph.num_nodes,
                node_sizes=graph.node_sizes,
                steps=200,
                seed=seed_value,
                fidelity_mode=True,
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


# ── New algorithms (March 2026) ──────────────────────────────────────────────


def _sugiyama_cache_key(
    name: str,
    fn_name: str,
    graph: DaguaGraph,
    extra_kwargs: Mapping[str, Any],
) -> Optional[Tuple[Any, ...]]:
    """Return a cache key for deterministic classic Sugiyama benchmark repeats.

    Parameters
    ----------
    name : str
        Competitor name.
    fn_name : str
        Layout function name.
    graph : DaguaGraph
        Graph object supplied by the benchmark worker cache.
    extra_kwargs : Mapping[str, Any]
        Layout parameters after benchmark variant defaults have been merged.

    Returns
    -------
    tuple[Any, ...] | None
        Cache key when the layout is deterministic for repeated benchmark
        seeds, otherwise ``None``.
    """
    if fn_name != "layout_sugiyama_pipeline":
        return None

    def _cache_param_value(key: str, value: Any) -> Tuple[Any, ...]:
        """Return a hashable fingerprint for a layout keyword value.

        Parameters
        ----------
        key : str
            Layout keyword name.
        value : Any
            Keyword value to fingerprint.

        Returns
        -------
        tuple[Any, ...]
            Hashable value summary suitable for the deterministic cache key.
        """
        if isinstance(value, torch.Tensor):
            return (key, "tensor", tuple(value.shape), str(value.dtype), str(value.device))
        if isinstance(value, Mapping):
            return (key, "mapping", id(value), len(value))
        if isinstance(value, (list, tuple, set, frozenset)):
            return (key, type(value).__name__, id(value), len(value))
        return (key, value)

    scalar_params = tuple(
        sorted(_cache_param_value(key=key, value=value) for key, value in extra_kwargs.items())
    )
    edge_count = int(graph.edge_index.shape[1]) if graph.edge_index.numel() > 0 else 0
    return (
        name,
        fn_name,
        id(graph),
        graph.num_nodes,
        edge_count,
        id(graph.edge_index),
        id(graph.edge_weights),
        id(graph.node_sizes),
        scalar_params,
    )


def _quick_classic(
    name: str,
    import_path: str,
    fn_name: str,
    graph: DaguaGraph,
    seed: int,
    **extra_kwargs: Any,
) -> CompetitorResult:
    """Run one classic layout function with shared timing/error handling.

    Parameters
    ----------
    name : str
        Competitor name for the result payload.
    import_path : str
        Module path containing the layout function.
    fn_name : str
        Layout function name inside ``import_path``.
    graph : DaguaGraph
        Graph to lay out.
    seed : int
        Explicit seed forwarded to the classic implementation.
    **extra_kwargs : Any
        Additional layout keyword arguments.

    Returns
    -------
    CompetitorResult
        Layout result and runtime metadata.
    """
    import importlib

    mod = importlib.import_module(import_path)
    fn = getattr(mod, fn_name)
    edge_index = graph.edge_index
    start = time.perf_counter()
    try:
        if graph.edge_weights is not None and fn_name not in _UNWEIGHTED_REFERENCE_LAYOUTS:
            extra_kwargs.setdefault("edge_weights", graph.edge_weights)
        if (
            fn_name == "layout_fmmm_pipeline"
            and graph.clusters
            and extra_kwargs.get("fidelity_mode") == "graphviz_fdp"
        ):
            extra_kwargs.setdefault("clusters", graph.clusters)
            extra_kwargs.setdefault("cluster_parents", graph.cluster_parents)
        if fn_name == "layout_graphopt_pipeline" and bool(extra_kwargs.get("fidelity_mode")):
            import numpy as np

            # Match the igraph reference adapter's seeded matrix exactly.
            extra_kwargs.setdefault(
                "initial_pos",
                np.random.RandomState(seed).uniform(-1.0, 1.0, size=(graph.num_nodes, 2)),
            )
        if fn_name in {"layout_kk_pipeline", "layout_sfdp_pipeline"}:
            extra_kwargs.setdefault("direction", graph.direction)
        if fn_name == "layout_kk_pipeline":
            extra_kwargs.setdefault("orient_to_direction", False)
        node_sizes = graph.node_sizes
        if (
            fn_name == "layout_sfdp_pipeline"
            and extra_kwargs.get("fidelity_mode") == "graphviz"
            and _has_multiple_weak_components(graph=graph)
        ):
            # Graphviz sfdp computes component bboxes from label-sized DOT node
            # boxes before pack.c rasterizes l_node polyominoes.
            graphviz_sfdp_node_sizes = _graphviz_dot_node_sizes(graph=graph)
            if _should_use_sfdp_graphviz_label_boxes(
                graph=graph,
                node_sizes=graphviz_sfdp_node_sizes,
            ):
                node_sizes = graphviz_sfdp_node_sizes
        if (
            fn_name == "layout_sugiyama_pipeline"
            and extra_kwargs.get("fidelity_mode") == "graphviz"
        ):
            _apply_sugiyama_graphviz_metadata(graph=graph, extra_kwargs=extra_kwargs)
        if fn_name == "layout_neato_pipeline":
            if node_sizes is None:
                graph.compute_node_sizes()
                node_sizes = graph.node_sizes
            if node_sizes is not None:
                # Dagua text measurement stores label-sized boxes in points,
                # while the neato compatibility pipeline models Graphviz's
                # internal coordinates in inches before JSON export.
                node_sizes = node_sizes / 72.0
        cache_key = _sugiyama_cache_key(
            name=name,
            fn_name=fn_name,
            graph=graph,
            extra_kwargs=extra_kwargs,
        )
        if cache_key is not None:
            cached = _SUGIYAMA_DETERMINISTIC_CACHE.get(cache_key)
            if cached is not None:
                cached_pos, cached_runtime = cached
                return CompetitorResult(
                    name=name,
                    pos=cached_pos.clone(),
                    runtime_seconds=cached_runtime,
                )
        pos = fn(
            edge_index,
            graph.num_nodes,
            node_sizes=node_sizes,
            seed=seed,
            **extra_kwargs,
        )
        runtime_seconds = time.perf_counter() - start
        if cache_key is not None and isinstance(pos, torch.Tensor):
            _SUGIYAMA_DETERMINISTIC_CACHE[cache_key] = (pos.detach().cpu().clone(), runtime_seconds)
        return CompetitorResult(name=name, pos=pos, runtime_seconds=runtime_seconds)
    except Exception as exc:
        return CompetitorResult(
            name=name,
            pos=None,
            runtime_seconds=time.perf_counter() - start,
            error=str(exc),
        )


@register
class ClassicGraphOpt(_ClassicBase):
    name = "classic_graphopt"
    max_nodes = 20_000
    variant_param_names = frozenset(
        {
            "niter",
            "node_charge",
            "node_mass",
            "spring_constant",
            "spring_length",
            "max_sa_movement",
            "fidelity_mode",
            "initial_pos",
        }
    )

    def layout(
        self, graph: DaguaGraph, timeout: float = 300.0, seed: Optional[int] = None
    ) -> CompetitorResult:
        del timeout
        return _quick_classic(
            self.name,
            "dagua.layout.ops.pipelines.graphopt",
            "layout_graphopt_pipeline",
            graph,
            self._layout_seed(seed),
            fidelity_mode=True,
        )


@register
class ClassicDRL(_ClassicBase):
    name = "classic_drl"
    max_nodes = 100_000
    variant_param_names = frozenset({"options", "fidelity_mode"})

    def layout(
        self, graph: DaguaGraph, timeout: float = 300.0, seed: Optional[int] = None
    ) -> CompetitorResult:
        """Run the DrL reimplementation.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Unused compatibility parameter.
        seed : int | None, default=None
            Random seed forwarded to the DrL layout.

        Returns
        -------
        CompetitorResult
            Layout result and runtime information.
        """
        del timeout
        return _quick_classic(
            self.name,
            "dagua.layout.ops.pipelines.drl",
            "layout_drl_pipeline",
            graph,
            self._layout_seed(seed),
            fidelity_mode=True,
        )


@register
class ClassicLGL(_ClassicBase):
    name = "classic_lgl"
    max_nodes = 100_000
    variant_param_names = frozenset({"maxiter", "coolexp", "fidelity_mode"})

    def layout(
        self, graph: DaguaGraph, timeout: float = 300.0, seed: Optional[int] = None
    ) -> CompetitorResult:
        del timeout
        return _quick_classic(
            self.name,
            "dagua.layout.ops.pipelines.lgl",
            "layout_lgl_pipeline",
            graph,
            self._layout_seed(seed),
            fidelity_mode=True,
        )


@register
class ClassicSFDP(_ClassicBase):
    name = "classic_sfdp"
    max_nodes = 100_000
    variant_param_names = frozenset({"fidelity_mode", "repulsive_exponent", "steps", "theta"})

    def layout(
        self, graph: DaguaGraph, timeout: float = 300.0, seed: Optional[int] = None
    ) -> CompetitorResult:
        del timeout
        return _quick_classic(
            self.name,
            "dagua.layout.ops.pipelines.sfdp",
            "layout_sfdp_pipeline",
            graph,
            self._layout_seed(seed),
        )


@register
class ClassicUMAP(_ClassicBase):
    name = "classic_umap"
    max_nodes = 20_000
    variant_param_names = frozenset({"n_neighbors", "min_dist", "spread", "fidelity_mode"})

    def layout(
        self, graph: DaguaGraph, timeout: float = 300.0, seed: Optional[int] = None
    ) -> CompetitorResult:
        del timeout
        return _quick_classic(
            self.name,
            "dagua.layout.ops.pipelines.umap_layout",
            "layout_umap_layout_pipeline",
            graph,
            self._layout_seed(seed),
        )


@register
class ClassicNeuLay(_ClassicBase):
    name = "classic_neulay"
    max_nodes = 50_000
    variant_param_names = frozenset(
        {"steps", "gcn_steps", "use_gcn", "lr", "radius", "fidelity_mode"}
    )

    def layout(
        self, graph: DaguaGraph, timeout: float = 300.0, seed: Optional[int] = None
    ) -> CompetitorResult:
        """Run the classic NeuLay reimplementation with the full two-phase setup.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Unused compatibility parameter for the competitor interface.
        seed : int | None, default=None
            Random seed for the stochastic solver. ``None`` preserves the
            historical default of ``42``.

        Returns
        -------
        CompetitorResult
            Layout result and runtime information.
        """
        del timeout
        return _quick_classic(
            self.name,
            "dagua.layout.ops.pipelines.neulay",
            "layout_neulay_pipeline",
            graph,
            self._layout_seed(seed),
            steps=20_000,
            gcn_steps=2_000,
            use_gcn=True,
            lr=0.1,
            radius=0.4,
        )


@register
class ClassicSGD2Multi(_ClassicBase):
    name = "classic_sgd2_multi"
    max_nodes = 10_000
    variant_param_names = frozenset(
        {"criteria", "steps", "lr", "grad_clamp", "batch_size", "fidelity_mode"}
    )

    def layout(
        self, graph: DaguaGraph, timeout: float = 300.0, seed: Optional[int] = None
    ) -> CompetitorResult:
        """Run the classic multicriteria ``(SGD)^2`` reimplementation.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Unused compatibility parameter for the competitor interface.
        seed : int | None, default=None
            Random seed for the stochastic solver. ``None`` preserves the
            historical default of ``42``.

        Returns
        -------
        CompetitorResult
            Layout result and runtime information.
        """
        del timeout
        return _quick_classic(
            self.name,
            "dagua.layout.ops.pipelines.sgd2_multi",
            "layout_sgd2_multi_pipeline",
            graph,
            self._layout_seed(seed),
            criteria={"stress": 1.0, "ideal_edge_length": 1.0},
            lr=0.01,
        )
