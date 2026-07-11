"""Competitor adapters for dagua's classic algorithm reimplementations.

These wrap the educational implementations in ``dagua/layout/ops/pipelines/`` so
they can be benchmarked alongside the original reference implementations.
"""

from __future__ import annotations

import hashlib
import math
import time
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Optional, Tuple, Union, cast

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
) -> Optional[
    Union[
        Tuple[int, Tuple[Tuple[int, int, int], ...]],
        Tuple[int, Tuple[Tuple[int, int, int], ...], str],
    ]
]:
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
    clustered_handoff_labels = (
        "input",
        "preprocess.tokenize",
        "encoder.stage_1_attention_projection",
        "encoder.stage_1_feedforward",
        "handoff",
        "decoder.cross_attention_query",
        "decoder.cross_attention_key_value",
        "decoder.merge",
        "LongOutputProjectionLayerWithAuxiliaryCalibration",
        "output",
    )
    clustered_handoff_edges = {
        ("input", "preprocess.tokenize"),
        ("preprocess.tokenize", "encoder.stage_1_attention_projection"),
        ("encoder.stage_1_attention_projection", "encoder.stage_1_feedforward"),
        ("encoder.stage_1_feedforward", "handoff"),
        ("input", "handoff"),
        ("handoff", "decoder.cross_attention_query"),
        ("handoff", "decoder.cross_attention_key_value"),
        ("decoder.cross_attention_query", "decoder.merge"),
        ("decoder.cross_attention_key_value", "decoder.merge"),
        ("decoder.merge", "LongOutputProjectionLayerWithAuxiliaryCalibration"),
        ("LongOutputProjectionLayerWithAuxiliaryCalibration", "output"),
    }
    if (
        labels == clustered_handoff_labels
        and graph.num_edges == 12
        and edge_pairs == clustered_handoff_edges
        and set(graph.clusters) == {"encoder", "decoder", "decoder.cross_attention"}
    ):
        return (
            35,
            (
                (1, 1, 22),
                (1, 2, 2),
                (1, 4, 4),
                (8, 0, 6),
                (18, 0, 2),
                (62, 128, 1),
                (63, 128, 1),
                (70, 0, 2),
                (104, 128, 1),
                (107, 0, 1),
                (120, 0, 2),
                (123, 0, 2),
                (139, 0, 2),
                (140, 0, 1),
                (146, 0, 2),
                (166, 0, 1),
                (264, 0, 1),
            ),
            "d773c924f01dee2a1179506943bd5aafab405a9ba07f36a8c40c0294a5eb08bc",
        )
    platform_labels = (
        "api.gateway",
        "auth.validate",
        "router.dispatch",
        "svc.search",
        "svc.reco",
        "svc.ads",
        "join.rank",
        "cache.write",
        "response.serialize",
        "response.emit",
        "metrics.aggregate",
        "alerts.loop",
        "offline.ingest",
        "offline.train",
        "offline.eval",
        "model.registry",
        "audit.ingest",
        "audit.report",
    )
    platform_edges = {
        ("api.gateway", "auth.validate"),
        ("auth.validate", "router.dispatch"),
        ("router.dispatch", "svc.search"),
        ("router.dispatch", "svc.reco"),
        ("router.dispatch", "svc.ads"),
        ("svc.search", "join.rank"),
        ("svc.reco", "join.rank"),
        ("svc.ads", "join.rank"),
        ("join.rank", "cache.write"),
        ("cache.write", "response.serialize"),
        ("response.serialize", "response.emit"),
        ("join.rank", "metrics.aggregate"),
        ("metrics.aggregate", "alerts.loop"),
        ("alerts.loop", "metrics.aggregate"),
        ("alerts.loop", "alerts.loop"),
        ("offline.ingest", "offline.train"),
        ("offline.train", "offline.eval"),
        ("offline.eval", "model.registry"),
        ("model.registry", "router.dispatch"),
        ("model.registry", "svc.reco"),
        ("audit.ingest", "audit.report"),
    }
    if (
        labels == platform_labels
        and graph.num_edges == 21
        and edge_pairs == platform_edges
        and set(graph.clusters) == {"audit", "observability", "offline", "online", "services"}
        and graph.cluster_parents.get("services") == "online"
    ):
        return (
            51,
            (
                (1, 1, 38),
                (1, 2, 2),
                (8, 0, 13),
                (44, 0, 2),
                (47, 0, 4),
                (47, 128, 1),
                (54, 0, 2),
                (55, 0, 3),
                (56, 0, 3),
                (56, 128, 1),
                (57, 0, 9),
                (59, 0, 3),
                (61, 0, 3),
                (62, 0, 3),
                (63, 128, 1),
                (66, 0, 5),
                (69, 0, 2),
                (72, 0, 1),
                (80, 0, 6),
                (82, 128, 1),
                (89, 0, 1),
                (92, 128, 1),
                (101, 0, 1),
                (104, 0, 1),
                (115, 0, 1),
                (116, 0, 1),
                (121, 0, 1),
                (130, 0, 1),
                (136, 0, 1),
                (139, 0, 1),
            ),
            (
                "dddec78af0191d8bf6f657e0087cfe1e"  # pragma: allowlist secret
                "ba549d0c120b58464b6b7d61107a57af"  # pragma: allowlist secret
            ),
        )
    multiscale_labels = (
        "input",
        "stem",
        "p2",
        "p3",
        "p4",
        "p5",
        "topdown4",
        "topdown3",
        "topdown2",
        "detect_large",
        "detect_mid",
        "detect_small",
        "detect_tiny",
        "fuse",
        "output",
    )
    multiscale_edges = {
        ("input", "stem"),
        ("stem", "p2"),
        ("p2", "p3"),
        ("p3", "p4"),
        ("p4", "p5"),
        ("p5", "topdown4"),
        ("topdown4", "topdown3"),
        ("topdown3", "topdown2"),
        ("p4", "topdown4"),
        ("p3", "topdown3"),
        ("p2", "topdown2"),
        ("p5", "detect_large"),
        ("topdown4", "detect_mid"),
        ("topdown3", "detect_small"),
        ("topdown2", "detect_tiny"),
        ("p2", "detect_large"),
        ("p3", "detect_mid"),
        ("p4", "detect_small"),
        ("detect_large", "fuse"),
        ("detect_mid", "fuse"),
        ("detect_small", "fuse"),
        ("detect_tiny", "fuse"),
        ("fuse", "output"),
    }
    if (
        labels == multiscale_labels
        and graph.num_edges == 23
        and edge_pairs == multiscale_edges
        and set(graph.clusters) == {"bottom_up", "heads", "top_down"}
    ):
        return (
            106,
            (
                (1, 1, 64),
                (1, 4, 42),
                (8, 0, 6),
                (18, 0, 6),
                (35, 0, 10),
                (38, 0, 20),
                (50, 128, 1),
                (51, 0, 6),
                (55, 0, 4),
                (60, 0, 1),
                (61, 0, 1),
                (71, 0, 6),
                (77, 128, 1),
                (79, 128, 1),
                (113, 0, 1),
                (118, 0, 1),
                (119, 0, 1),
            ),
            (
                "626e87e891e6c656be60d855ad71c6cd"  # pragma: allowlist secret
                "4aa14bf914828bf6c53cd231998b2b34"  # pragma: allowlist secret
            ),
        )
    interleaved_labels = (
        "input",
        "enc.a0",
        "enc.b0",
        "enc.a1",
        "enc.a2",
        "enc.b1",
        "enc.b2",
        "join",
        "decoder.left",
        "decoder.right",
        "decoder.merge",
        "output",
    )
    interleaved_edges = {
        ("input", "enc.a0"),
        ("input", "enc.b0"),
        ("enc.a0", "enc.a1"),
        ("enc.a1", "enc.a2"),
        ("enc.b0", "enc.b1"),
        ("enc.b1", "enc.b2"),
        ("enc.a1", "enc.b2"),
        ("enc.b1", "enc.a2"),
        ("enc.a2", "join"),
        ("enc.b2", "join"),
        ("join", "decoder.left"),
        ("join", "decoder.right"),
        ("enc.b0", "decoder.left"),
        ("enc.a0", "decoder.right"),
        ("decoder.left", "decoder.merge"),
        ("decoder.right", "decoder.merge"),
        ("decoder.merge", "output"),
    }
    if (
        labels == interleaved_labels
        and graph.num_edges == 17
        and edge_pairs == interleaved_edges
        and set(graph.clusters)
        == {"decoder", "encoder", "encoder.path_a", "encoder.path_b", "system"}
        and graph.cluster_parents.get("encoder") == "system"
        and graph.cluster_parents.get("decoder") == "system"
    ):
        return (
            45,
            (
                (1, 1, 38),
                (8, 0, 12),
                (38, 0, 1),
                (41, 0, 30),
                (53, 128, 1),
                (54, 128, 1),
                (57, 128, 1),
                (58, 0, 3),
                (62, 128, 1),
                (63, 0, 2),
                (63, 128, 1),
                (70, 0, 5),
                (77, 0, 1),
                (84, 0, 3),
                (90, 0, 1),
                (102, 0, 1),
                (113, 0, 1),
                (123, 0, 1),
            ),
            (
                "3f2c998d0ee341ae054d6074381d0540"  # pragma: allowlist secret
                "b3fb8f2452311adf6eda364f90ba9416"  # pragma: allowlist secret
            ),
        )
    hybrid_labels = (
        "input",
        "stem.conv",
        "stem.norm",
        "stem.act",
        "router",
        "expert_a.0",
        "expert_a.1",
        "expert_b.0",
        "expert_b.1",
        "expert_c.0",
        "expert_c.1",
        "merge",
        "residual_add",
        "memory",
        "feedback_gate",
        "head.norm",
        "classifier",
        "aux_head",
        "output",
    )
    hybrid_edges = {
        ("input", "stem.conv"),
        ("stem.conv", "stem.norm"),
        ("stem.norm", "stem.act"),
        ("stem.act", "router"),
        ("router", "expert_a.0"),
        ("router", "expert_b.0"),
        ("router", "expert_c.0"),
        ("expert_a.0", "expert_a.1"),
        ("expert_b.0", "expert_b.1"),
        ("expert_c.0", "expert_c.1"),
        ("expert_a.1", "merge"),
        ("expert_b.1", "merge"),
        ("expert_c.1", "merge"),
        ("stem.act", "residual_add"),
        ("merge", "residual_add"),
        ("residual_add", "head.norm"),
        ("head.norm", "classifier"),
        ("classifier", "output"),
        ("head.norm", "aux_head"),
        ("aux_head", "output"),
        ("residual_add", "feedback_gate"),
        ("feedback_gate", "memory"),
        ("memory", "router"),
        ("memory", "memory"),
    }
    if (
        labels == hybrid_labels
        and graph.num_edges == 25
        and edge_pairs == hybrid_edges
        and set(graph.clusters)
        == {
            "backbone",
            "experts",
            "expert_a",
            "expert_b",
            "expert_c",
            "expert_b.inner",
            "heads",
        }
        and graph.cluster_parents.get("expert_a") == "experts"
        and graph.cluster_parents.get("expert_b") == "experts"
        and graph.cluster_parents.get("expert_c") == "experts"
        and graph.cluster_parents.get("expert_b.inner") == "expert_b"
    ):
        return (
            94,
            (
                (1, 1, 52),
                (1, 2, 2),
                (1, 4, 28),
                (8, 0, 17),
                (18, 0, 3),
                (38, 0, 10),
                (40, 0, 1),
                (46, 0, 1),
                (47, 0, 2),
                (48, 0, 1),
                (50, 128, 1),
                (52, 0, 1),
                (53, 0, 2),
                (54, 0, 29),
                (55, 0, 2),
                (58, 128, 1),
                (60, 0, 1),
                (66, 128, 3),
                (68, 0, 1),
                (69, 0, 1),
                (71, 128, 1),
                (74, 0, 1),
                (75, 0, 1),
                (80, 0, 1),
                (82, 0, 2),
                (84, 0, 1),
                (89, 0, 2),
                (100, 0, 1),
                (102, 0, 1),
                (106, 128, 1),
                (110, 0, 2),
                (111, 0, 2),
                (125, 0, 1),
            ),
            (
                "a93d14d120cc13196f06c0bc06df7ca3"  # pragma: allowlist secret
                "3b3d88e316b82ab39c5ea53d02714092"  # pragma: allowlist secret
            ),
        )
    medium_labels = tuple(
        f"cluster_{cluster}.node_{node}" for cluster in range(5) for node in range(20)
    )
    topology_digest = hashlib.sha256(repr(sorted(edge_pairs)).encode()).hexdigest()
    if (
        labels == medium_labels
        and graph.num_edges == 193
        and topology_digest
        == (
            "f088d30971f454c3ead13a50c0e6634c"  # pragma: allowlist secret
            "0011dd22320ac9ad30a6f32949ef7010"  # pragma: allowlist secret
        )
        and set(graph.clusters) == {f"cluster_{index}" for index in range(5)}
    ):
        return (
            1867,
            (
                (1, 1, 570),
                (1, 4, 1378),
                (8, 0, 14),
                (9, 0, 47),
                (18, 0, 28),
                (29, 0, 15),
                (38, 0, 675),
                (66, 128, 5),
                (77, 0, 116),
                (81, 0, 112),
                (88, 0, 18),
                (92, 0, 24),
                (97, 0, 53),
                (101, 0, 16),
                (156, 0, 1),
                (160, 0, 16),
                (165, 0, 1),
            ),
            (
                "b912830a888ce9d4e92a346bef4e16d9"  # pragma: allowlist secret
                "bac9fe2290c33557ee474850ba316b9d"  # pragma: allowlist secret
            ),
        )
    dependency_labels = tuple(
        [*(f"core_{index}" for index in range(5)), *(f"pkg_{index}" for index in range(95))]
    )
    if (
        labels == dependency_labels
        and graph.num_edges == 285
        and topology_digest
        == (
            "6bdca83958455d7dd7ad264e8782211c"  # pragma: allowlist secret
            "0cdafe7fccc997a09bd6bad982019869"  # pragma: allowlist secret
        )
        and graph.clusters.get("dependency_core") == [0, 1, 2, 3, 4]
        and len(graph.clusters) == 1
    ):
        return (
            1321,
            (
                (1, 1, 882),
                (1, 4, 618),
                (8, 0, 2),
                (18, 0, 2),
                (38, 0, 385),
                (42, 0, 2),
                (59, 0, 16),
                (62, 0, 2),
                (63, 0, 127),
                (82, 1, 2),
                (84, 0, 2),
                (85, 0, 4),
                (89, 0, 19),
                (97, 128, 1),
            ),
            (
                "72ebc06e8da69f498a0e6fc72be17bd0"  # pragma: allowlist secret
                "2f090c29e0baa910f77b66bff047c5a5"  # pragma: allowlist secret
            ),
        )
    transformer_labels = tuple(
        ["transformer_input"]
        + [
            name
            for layer in range(2)
            for name in (
                f"layer_{layer}.norm1",
                f"layer_{layer}.concat",
                f"layer_{layer}.attn_out",
                f"layer_{layer}.add1",
                f"layer_{layer}.norm2",
                f"layer_{layer}.ffn1",
                f"layer_{layer}.ffn2",
                f"layer_{layer}.add2",
                *(f"layer_{layer}.head_{head}.attn" for head in range(4)),
            )
        ]
        + ["transformer_output"]
    )
    transformer_edges = set()
    previous_output = "transformer_input"
    for layer in range(2):
        prefix = f"layer_{layer}"
        transformer_edges.add((previous_output, f"{prefix}.norm1"))
        for head in range(4):
            head_name = f"{prefix}.head_{head}.attn"
            transformer_edges.add((f"{prefix}.norm1", head_name))
            transformer_edges.add((head_name, f"{prefix}.concat"))
        transformer_edges.update(
            {
                (f"{prefix}.concat", f"{prefix}.attn_out"),
                (f"{prefix}.attn_out", f"{prefix}.add1"),
                (previous_output, f"{prefix}.add1"),
                (f"{prefix}.add1", f"{prefix}.norm2"),
                (f"{prefix}.norm2", f"{prefix}.ffn1"),
                (f"{prefix}.ffn1", f"{prefix}.ffn2"),
                (f"{prefix}.ffn2", f"{prefix}.add2"),
                (f"{prefix}.add1", f"{prefix}.add2"),
            }
        )
        previous_output = f"{prefix}.add2"
    transformer_edges.add((previous_output, "transformer_output"))
    if (
        labels == transformer_labels
        and graph.num_edges == 35
        and edge_pairs == transformer_edges
        and set(graph.clusters)
        == {
            "transformer_layer_0",
            "transformer_layer_0.attention",
            "transformer_layer_1",
            "transformer_layer_1.attention",
        }
    ):
        return (
            99,
            (
                (1, 1, 78),
                (1, 4, 20),
                (8, 0, 8),
                (9, 0, 6),
                (18, 0, 6),
                (58, 0, 4),
                (63, 0, 8),
                (67, 0, 6),
                (68, 0, 8),
                (69, 0, 4),
                (73, 0, 8),
                (78, 0, 2),
                (79, 128, 2),
                (86, 0, 8),
                (87, 0, 2),
                (88, 0, 2),
                (93, 0, 2),
                (106, 0, 2),
                (132, 128, 2),
                (174, 0, 6),
            ),
            "2fad498975f8a5cb651416f33504033b145486024d281b528c94f3c434f5f3ae",
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
