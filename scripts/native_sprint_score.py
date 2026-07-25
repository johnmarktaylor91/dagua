"""Score native saved positions against field saved positions for R8 Event A.

The scorer keeps the frozen old-108 corpus as the old table source, optionally
extends it with the exact 13 R8 nested-cluster graphs, and evaluates saved
position tensors under two rulers:

* old: current metrics with the six R8 cluster terms disabled;
* extended: current metrics with those terms active.

Legacy field-cache reuse is only allowed for non-cluster graphs under the
pinned G2 bridge guard. Clustered graphs are always scored from saved positions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

from dagua.eval.benchmark import _declares_hierarchy
from dagua.eval.graphs import TestGraph, get_test_graphs, is_semantically_directed
from dagua.eval.ruler_v3 import (
    SEVERE_G6_BREACH_FLAG,
    RulerV3Result,
    referee_eligibility_key,
    score_core_v3,
    severe_g6_breach,
)
from dagua.eval.size_policy import set_size_aware_externals
from dagua.metrics import DEGENERATE_SCALE_RATIO, composite_auto, full
from dagua.render.mpl import _density_scaled_node_sizes, _layout_extent_pt

TIE_BAND = 0.5
RULER_SCHEMA = "r8-cluster-extended-v1"
DEGENERACY_CHAMPION_INELIGIBLE_FLAGS = frozenset(
    {"DEGENERATE_SCALE", "SPRAWL_COLLAPSE", "COINCIDENT_COLLAPSE"}
)
RAW_CACHE_FILENAME = "R8_EVENTA_RAW_SCORES_V1.json"
LEGACY_FIELD_CACHE_SHA256 = (
    "019c777c64b914e7b402d981378bb6aa4e3216000f5921bbc0184f8f0fe936ee"  # pragma: allowlist secret
)
LEGACY_FIELD_CACHE_PAIR_COUNT = 11_163
CLUSTER_SCORE_KEYS = (
    "cluster_exclusion_score",
    "cluster_sibling_overlap_score",
    "cluster_nesting_fidelity_score",
    "cluster_edge_intrusion_score",
    "cluster_label_occlusion_score",
    "cluster_compactness_score",
)
R8_GRAPH_NAMES = (
    "r8_nested_chain_depth8_directed",
    "r8_nested_balanced_3x3x4",
    "r8_nested_mixed_direct_leaf",
    "r8_nested_cross_edges_ladder",
    "r8_nested_edge_labels_compound",
    "r8_nested_wide_labels_shapes",
    "r8_nested_undirected_communities_depth3",
    "r8_nested_sbm_overlap_trap",
    "r8_nested_parent_child_backedges",
    "r8_nested_disconnected_forest",
    "r8_nested_singleton_deep",
    "r8_nested_scale_1k_budget",
    "r8_nested_lr_direction",
)
_CLASSICAL_ENGINES = (
    "dagre",
    "elk_layered",
    "graphviz_dot",
    "graphviz_neato",
    "graphviz_sfdp",
    "igraph_kamada_kawai",
    "igraph_sugiyama",
    "nx_spring",
)
_CORPUS_RE = re.compile(
    r"^(?P<graph>.+)__(?P<engine>" + "|".join(_CLASSICAL_ENGINES) + r")(?P<variant>.*)\.pt$"
)
_FULL_POLICY = {
    "profile": "full",
    "stress_sources": 200,
    "stress_targets": 1000,
    "crossing_samples": 1_000_000,
    "neighborhood_samples": 5000,
    "cluster_score_keys": CLUSTER_SCORE_KEYS,
}
_WORKER_GRAPHS: Dict[str, TestGraph] = {}
_WORKER_SIGNATURE = ""
_PLANTED_PARTITION_GRAPH_NAMES = {
    "r79_weighted_community_4x18",
    "r79_undirected_sbm_low_mix_4x25",
    "r79_undirected_sbm_mid_mix_5x20",
    "r79_undirected_sbm_high_mix_3x30",
    "sbm_5x40",
    "weighted_clusters_3x10",
}
_RENDER_POINTS_PER_NATIVE_UNIT = 72.0
_NATIVE_UNIT_ENGINES = frozenset(
    {
        "backbone",
        "backbone_reimpl",
        "classic_fr_kk",
        "classic_kk",
        "classic_linlog",
        "classic_sgd2_multi",
        "classic_spectral",
        "classic_sugiyama",
        "d3_cluster_radial_reimpl",
        "d3_cluster_reimpl",
        "d3_tree_radial_reimpl",
        "d3_tree_reimpl",
        "d3hierarchy",
        "dot",
        "linlog",
        "nx_arf",
        "nx_arf_reimpl",
        "nx_bfs",
        "nx_bfs_reimpl",
        "nx_bipartite",
        "nx_bipartite_reimpl",
        "nx_circular",
        "nx_circular_reimpl",
        "nx_kamada_kawai",
        "nx_multipartite",
        "nx_multipartite_reimpl",
        "nx_planar",
        "nx_planar_reimpl",
        "nx_shell",
        "nx_shell_reimpl",
        "nx_spectral",
        "nx_spectral_random_walk",
        "nx_spiral",
        "nx_spiral_reimpl",
        "sgd2_multi_ref",
        "sparse_stress",
    }
)


@dataclass(frozen=True)
class ScoringUnitNormalization:
    """Rendered-unit payload used by corpus scoring."""

    positions: torch.Tensor
    node_sizes: torch.Tensor
    position_scale: float
    node_size_scale: float
    source_span: float
    rendered_span: float
    node_diag_mean: float
    flags: Tuple[str, ...]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line options.

    Parameters
    ----------
    argv : Optional[Sequence[str]], optional
        Explicit argument sequence, or ``None`` for ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Parsed options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--field-dir",
        type=Path,
        required=True,
        action="append",
        help="Benchmark output dir of a field baseline store; repeatable.",
    )
    parser.add_argument(
        "--native-dir",
        type=Path,
        required=True,
        action="append",
        help="Benchmark output dir of a native run store; repeatable.",
    )
    parser.add_argument(
        "--corpus-positions",
        type=Path,
        default=Path("/home/jtaylor/projects/dagua/eval_output/r81_regate2/positions"),
        help="Frozen r81 positions dir that defines the shared 108-graph corpus.",
    )
    parser.add_argument(
        "--include-r8",
        action="store_true",
        help="Append the exact 13 R8 nested-cluster graphs to the extended table.",
    )
    parser.add_argument(
        "--legacy-field-cache",
        type=Path,
        default=None,
        help="Pinned old field cache, usable only for non-cluster G2 bridge rows.",
    )
    parser.add_argument(
        "--score-cache",
        "--field-cache",
        dest="score_cache",
        type=Path,
        default=Path(RAW_CACHE_FILENAME),
        help="Versioned raw score cache. The deprecated --field-cache alias points here.",
    )
    parser.add_argument("--old-output", type=Path, required=True)
    parser.add_argument("--extended-output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) // 2))
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    """Compute a file SHA-256 digest.

    Parameters
    ----------
    path : Path
        File to hash.

    Returns
    -------
    str
        Hex SHA-256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_hash(payload: Any) -> str:
    """Hash a JSON-compatible payload with canonical ordering.

    Parameters
    ----------
    payload : Any
        JSON-compatible object.

    Returns
    -------
    str
        Hex SHA-256 digest.
    """
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def corpus_names(corpus_positions: Path) -> List[str]:
    """Return the exact frozen 108-graph shared corpus.

    Parameters
    ----------
    corpus_positions : Path
        Directory holding the frozen r81 classical position tensors.

    Returns
    -------
    List[str]
        Sorted shared corpus names.
    """
    names = {
        match.group("graph")
        for path in corpus_positions.glob("*.pt")
        if (match := _CORPUS_RE.match(path.name)) is not None
    }
    if len(names) != 108:
        raise RuntimeError(f"expected the frozen 108-graph corpus, found {len(names)}")
    return sorted(names)


def selected_names(corpus_positions: Path, include_r8: bool) -> Tuple[List[str], List[str]]:
    """Build ordered old and extended graph-name lists.

    Parameters
    ----------
    corpus_positions : Path
        Frozen old-108 corpus positions directory.
    include_r8 : bool
        Whether to append the exact R8 graph tuple.

    Returns
    -------
    Tuple[List[str], List[str]]
        Ordered old-108 names and ordered table names.
    """
    old_names = corpus_names(corpus_positions)
    if len(old_names) != 108:
        raise RuntimeError(f"expected exactly 108 old graph names, found {len(old_names)}")
    if not include_r8:
        return old_names, old_names
    overlap = sorted(set(old_names) & set(R8_GRAPH_NAMES))
    if overlap:
        raise RuntimeError(f"R8 graph(s) overlap frozen 108 corpus: {overlap}")
    extended = [*old_names, *R8_GRAPH_NAMES]
    if len(set(extended)) != 121:
        raise RuntimeError(f"expected 121 unique extended graph names, found {len(set(extended))}")
    return old_names, extended


def load_run_tasks(
    run_dirs: Sequence[Path], names: Sequence[str], engine_filter: Optional[str] = None
) -> List[Tuple[str, str, List[str]]]:
    """Collect scoreable ``(graph, engine, position paths)`` tasks.

    Parameters
    ----------
    run_dirs : Sequence[Path]
        Benchmark output dirs containing ``results.json`` and ``positions/``.
    names : Sequence[str]
        Corpus names to keep.
    engine_filter : Optional[str], optional
        Keep only this engine when set.

    Returns
    -------
    List[Tuple[str, str, List[str]]]
        Graph name, engine name, and deduplicated absolute tensor paths.
    """
    wanted = set(names)
    grouped: Dict[Tuple[str, str], set[str]] = defaultdict(set)
    for run_dir in run_dirs:
        results = json.loads((run_dir / "results.json").read_text())
        for record in results.values():
            if record.get("status") != "ok" or not record.get("positions_file"):
                continue
            graph = str(record["graph_name"])
            engine = str(record["engine_name"])
            if graph not in wanted:
                continue
            if engine_filter is not None and engine != engine_filter:
                continue
            path = (run_dir / str(record["positions_file"])).resolve()
            if path.exists():
                grouped[(graph, engine)].add(str(path))
    return [(graph, engine, sorted(paths)) for (graph, engine), paths in sorted(grouped.items())]


def result_file_hashes(run_dirs: Sequence[Path]) -> Dict[str, str]:
    """Hash every source ``results.json`` file.

    Parameters
    ----------
    run_dirs : Sequence[Path]
        Benchmark output dirs.

    Returns
    -------
    Dict[str, str]
        Absolute results path to SHA-256 digest.
    """
    return {
        str((run_dir / "results.json").resolve()): sha256_file(run_dir / "results.json")
        for run_dir in run_dirs
    }


def task_position_hashes(tasks: Sequence[Tuple[str, str, List[str]]]) -> Dict[str, str]:
    """Hash every candidate position file referenced by score tasks.

    Parameters
    ----------
    tasks : Sequence[Tuple[str, str, List[str]]]
        Score tasks.

    Returns
    -------
    Dict[str, str]
        Absolute position path to SHA-256 digest.
    """
    paths = sorted({path for _, _, task_paths in tasks for path in task_paths})
    return {path: sha256_file(Path(path)) for path in paths}


def build_graph_map(
    names: Sequence[str], r8_names: Sequence[str] = R8_GRAPH_NAMES
) -> Dict[str, TestGraph]:
    """Build current corpus graphs and select the requested names.

    Parameters
    ----------
    names : Sequence[str]
        Requested graph names.
    r8_names : Sequence[str], optional
        R8 graph names that must reconstruct with cluster metadata.

    Returns
    -------
    Dict[str, TestGraph]
        Test graphs keyed by name, with measured node sizes.
    """
    wanted = set(names)
    selected = {
        graph.name: graph for graph in get_test_graphs(max_nodes=1000) if graph.name in wanted
    }
    missing = sorted(wanted - set(selected))
    if missing:
        raise RuntimeError(f"current corpus cannot reconstruct frozen graph(s): {missing}")
    missing_cluster_r8 = [
        name for name in r8_names if name in wanted and not selected[name].graph.clusters
    ]
    if missing_cluster_r8:
        raise RuntimeError(f"R8 graph(s) reconstructed without clusters: {missing_cluster_r8}")
    for test_graph in selected.values():
        test_graph.graph.compute_node_sizes()
    return selected


def graph_hashes(graphs: Mapping[str, TestGraph], names: Sequence[str]) -> Dict[str, str]:
    """Hash canonical graph JSON for the requested graph names.

    Parameters
    ----------
    graphs : Mapping[str, TestGraph]
        Reconstructed graph map.
    names : Sequence[str]
        Ordered graph names.

    Returns
    -------
    Dict[str, str]
        Graph name to canonical JSON digest.
    """
    return {name: canonical_json_hash(graphs[name].graph.to_json()) for name in names}


def scoring_signature() -> str:
    """Build the score signature from policy constants and ruler source files.

    Returns
    -------
    str
        Canonical scoring-policy digest.
    """
    root = Path(__file__).resolve().parents[1]
    payload = {
        "ruler_schema": RULER_SCHEMA,
        "signature_bump": "overlap-fold-v3",
        "policy": _FULL_POLICY,
        "source_hashes": {
            "scripts/native_sprint_score.py": sha256_file(
                root / "scripts" / "native_sprint_score.py"
            ),
            "dagua/metrics.py": sha256_file(root / "dagua" / "metrics.py"),
            "dagua/layout/ops/cluster_geometry.py": sha256_file(
                root / "dagua" / "layout" / "ops" / "cluster_geometry.py"
            ),
            "dagua/eval/ruler_v3.py": sha256_file(root / "dagua" / "eval" / "ruler_v3.py"),
            "dagua/eval/ruler_v3_frozen.py": sha256_file(
                root / "dagua" / "eval" / "ruler_v3_frozen.py"
            ),
            "dagua/eval/ruler_v3_groups.py": sha256_file(
                root / "dagua" / "eval" / "ruler_v3_groups.py"
            ),
        },
    }
    return canonical_json_hash(payload)


def cache_header(
    old_names: Sequence[str],
    extended_names: Sequence[str],
    graphs: Mapping[str, TestGraph],
    field_dirs: Sequence[Path],
    native_dirs: Sequence[Path],
    tasks: Sequence[Tuple[str, str, List[str]]],
    signature: str,
) -> Dict[str, Any]:
    """Create the versioned raw-cache header.

    Parameters
    ----------
    old_names : Sequence[str]
        Ordered frozen old-108 names.
    extended_names : Sequence[str]
        Ordered output graph names.
    graphs : Mapping[str, TestGraph]
        Reconstructed graph map.
    field_dirs : Sequence[Path]
        Field result stores.
    native_dirs : Sequence[Path]
        Native result stores.
    tasks : Sequence[Tuple[str, str, List[str]]]
        All score tasks represented in the cache.
    signature : str
        Current scoring signature.

    Returns
    -------
    Dict[str, Any]
        Header payload.
    """
    position_hashes = task_position_hashes(tasks)
    return {
        "ruler_schema": RULER_SCHEMA,
        "scoring_signature": signature,
        "old_names": list(old_names),
        "extended_names": list(extended_names),
        "graph_hashes": graph_hashes(graphs, extended_names),
        "source_results_hashes": result_file_hashes([*field_dirs, *native_dirs]),
        "position_file_hashes": position_hashes,
        "position_file_hash_aggregate": canonical_json_hash(position_hashes),
    }


def init_worker(graphs: Dict[str, TestGraph], signature: str) -> None:
    """Initialize process-local graph state.

    Parameters
    ----------
    graphs : Dict[str, TestGraph]
        Reconstructed corpus graphs.
    signature : str
        Current scoring signature.

    Returns
    -------
    None
    """
    global _WORKER_GRAPHS, _WORKER_SIGNATURE
    _WORKER_GRAPHS = graphs
    _WORKER_SIGNATURE = signature
    torch.set_num_threads(1)
    set_size_aware_externals(True)


def normalized_metric_value(value: Any) -> Any:
    """Convert metric values to JSON-safe scalar/list payloads.

    Parameters
    ----------
    value : Any
        Metric value returned by ``full``.

    Returns
    -------
    Any
        JSON-compatible value.
    """
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): normalized_metric_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalized_metric_value(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            return value
    return value


def normalized_metrics(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize a metrics dictionary for JSON persistence.

    Parameters
    ----------
    metrics : Mapping[str, Any]
        Raw metric payload.

    Returns
    -------
    Dict[str, Any]
        JSON-compatible metrics dictionary.
    """
    return {key: normalized_metric_value(value) for key, value in metrics.items()}


def _position_span(positions: torch.Tensor) -> float:
    """Return the maximum center-coordinate span.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]`` in the source layout units.

    Returns
    -------
    float
        Maximum of x-span and y-span, or ``0.0`` for an empty tensor.
    """
    if positions.numel() == 0:
        return 0.0
    spans = positions.amax(dim=0) - positions.amin(dim=0)
    return float(spans.max().item())


def _rendered_node_sizes(graph: Any, positions: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """Return node boxes in the same point-like units used by rendering.

    Parameters
    ----------
    graph : Any
        Graph object exposing ``node_sizes``, ``num_nodes``, and ``graph_style``.
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]`` after dtype normalization.

    Returns
    -------
    tuple[torch.Tensor, float]
        Renderer-equivalent node sizes and the render-time density shrink
        multiplier.

    Raises
    ------
    RuntimeError
        If the graph has no measured node sizes.
    """
    if graph.node_sizes is None:
        raise RuntimeError("graph.node_sizes is None")
    style = graph.graph_style
    enabled = bool(getattr(style, "density_aware_node_shrink", False))
    position_array = positions.detach().cpu().numpy()
    return _density_scaled_node_sizes(
        graph.node_sizes.to(dtype=torch.float32),
        int(graph.num_nodes),
        _layout_extent_pt(position_array),
        enabled,
    )


def _engine_position_scale(engine: str) -> float:
    """Return the render-point scale for an engine's stored coordinate unit.

    Parameters
    ----------
    engine : str
        Engine identifier from the benchmark result store.

    Returns
    -------
    float
        ``72.0`` for stores whose native coordinates are graph-display inches
        or unit-box coordinates, otherwise ``1.0`` for point-scale stores.
    """
    return _RENDER_POINTS_PER_NATIVE_UNIT if engine in _NATIVE_UNIT_ENGINES else 1.0


def _is_degenerate_render_span(rendered_span: float, node_diag_mean: float) -> bool:
    """Report whether a rendered layout remains collapsed relative to boxes.

    Parameters
    ----------
    rendered_span : float
        Maximum center-coordinate span after scoring-path unit normalization.
    node_diag_mean : float
        Mean node-box diagonal in rendered point units.

    Returns
    -------
    bool
        ``True`` when the layout remains too small to score as a legible
        drawing after unit conversion.
    """
    if node_diag_mean <= 1e-8:
        return False
    return rendered_span < DEGENERATE_SCALE_RATIO * node_diag_mean


def normalize_position_units_for_scoring(
    graph: Any,
    positions: torch.Tensor,
    engine: str,
) -> ScoringUnitNormalization:
    """Convert stored positions and node boxes to renderer-equivalent units.

    Parameters
    ----------
    graph : Any
        Graph whose measured node sizes define the point-scale boxes.
    positions : torch.Tensor
        Stored position tensor with shape ``[N, 2]`` in engine-native units.
    engine : str
        Engine identifier used to apply the store-unit convention.

    Returns
    -------
    ScoringUnitNormalization
        Positions and node sizes in one rendered coordinate system, plus audit
        telemetry for raw-score rows.
    """
    source_positions = positions.to(dtype=torch.float32)
    node_sizes, node_size_scale = _rendered_node_sizes(graph, source_positions)
    source_span = _position_span(source_positions)
    position_scale = _engine_position_scale(engine)
    rendered_positions = source_positions * position_scale
    rendered_span = source_span * position_scale
    node_diag_mean = float(torch.linalg.vector_norm(node_sizes, dim=1).mean().item())
    flags: Tuple[str, ...] = (
        ("DEGENERATE_SCALE",)
        if _is_degenerate_render_span(rendered_span, node_diag_mean)
        else tuple()
    )
    return ScoringUnitNormalization(
        positions=rendered_positions,
        node_sizes=node_sizes,
        position_scale=position_scale,
        node_size_scale=float(node_size_scale),
        source_span=source_span,
        rendered_span=rendered_span,
        node_diag_mean=node_diag_mean,
        flags=flags,
    )


def scoring_unit_payload(normalization: ScoringUnitNormalization) -> Dict[str, Any]:
    """Serialize scoring unit telemetry for raw score rows.

    Parameters
    ----------
    normalization : ScoringUnitNormalization
        Rendered-unit normalization result.

    Returns
    -------
    Dict[str, Any]
        JSON-compatible audit fields describing the unit conversion.
    """
    return {
        "position_scale": normalization.position_scale,
        "node_size_scale": normalization.node_size_scale,
        "source_span": normalization.source_span,
        "rendered_span": normalization.rendered_span,
        "node_diag_mean": normalization.node_diag_mean,
        "flags": list(normalization.flags),
    }


def merged_row_flags(ruler_flags: Sequence[str], scoring_flags: Sequence[str]) -> List[str]:
    """Merge ruler and scoring-path row flags without duplicates.

    Parameters
    ----------
    ruler_flags : Sequence[str]
        Flags returned by the V3 ruler.
    scoring_flags : Sequence[str]
        Flags raised by scoring-path unit normalization.

    Returns
    -------
    List[str]
        Ordered union of row flags.
    """
    flags: List[str] = []
    for flag in [*ruler_flags, *scoring_flags]:
        if flag not in flags:
            flags.append(flag)
    return flags


def _cluster_labels_from_mapping(
    clusters: Mapping[str, Sequence[int]], num_nodes: int
) -> List[int]:
    """Build per-node labels from declared cluster membership.

    Parameters
    ----------
    clusters : Mapping[str, Sequence[int]]
        Cluster membership keyed by cluster id.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    List[int]
        Per-node cluster labels using ``-1`` for unassigned nodes.
    """
    labels = [-1] * num_nodes
    for cluster_index, cluster_name in enumerate(sorted(clusters)):
        for node_index in clusters[cluster_name]:
            if 0 <= int(node_index) < num_nodes:
                labels[int(node_index)] = cluster_index
    return labels


def _planted_partition_from_declared_row(test_graph: TestGraph) -> Optional[List[int]]:
    """Return planted labels only for pre-registered planted-partition rows.

    Parameters
    ----------
    test_graph : TestGraph
        Corpus row whose input declarations are inspected.

    Returns
    -------
    Optional[List[int]]
        Per-node planted partition labels, or ``None`` when the row is not a
        pre-registered planted-partition case.
    """
    if test_graph.name not in _PLANTED_PARTITION_GRAPH_NAMES:
        return None
    graph = test_graph.graph
    if graph.clusters:
        return _cluster_labels_from_mapping(graph.clusters, graph.num_nodes)
    if test_graph.name == "r79_weighted_community_4x18":
        return [node_index // 18 for node_index in range(graph.num_nodes)]
    if test_graph.name == "weighted_clusters_3x10":
        return [node_index // 10 for node_index in range(graph.num_nodes)]
    return None


def _rooted_forest_roots(test_graph: TestGraph) -> Optional[List[int]]:
    """Return roots when the row declares a rooted tree/forest.

    Parameters
    ----------
    test_graph : TestGraph
        Corpus row whose graph-level tree declaration and topology are checked.

    Returns
    -------
    Optional[List[int]]
        Root node indices for declared tree rows, or ``None`` when no rooted
        tree declaration is present.
    """
    tags = {str(tag).lower() for tag in test_graph.tags}
    if "tree" not in tags:
        return None
    graph = test_graph.graph
    edge_index = graph.edge_index
    parent: Dict[int, Optional[int]] = {node_index: None for node_index in range(graph.num_nodes)}
    child_counts = [0 for _ in range(graph.num_nodes)]
    for source, target in edge_index.t().tolist():
        source_index = int(source)
        target_index = int(target)
        if parent[target_index] is not None:
            return None
        parent[target_index] = source_index
        child_counts[source_index] += 1
    roots = [node_index for node_index, parent_index in parent.items() if parent_index is None]
    if not roots:
        return None
    return roots


def _declared_weight_mode(test_graph: TestGraph) -> str:
    """Return the declared geometric interpretation for edge weights.

    Parameters
    ----------
    test_graph : TestGraph
        Corpus row carrying edge-weight declarations.

    Returns
    -------
    str
        V3 G6 ``weight_mode`` value.
    """
    del test_graph
    return "distance"


def graph_meta_for_v3(test_graph: TestGraph) -> Dict[str, Any]:
    """Extract input-only graph metadata consumed by V3 conditional groups.

    Parameters
    ----------
    test_graph : TestGraph
        Corpus row definition and reconstructed graph.

    Returns
    -------
    Dict[str, Any]
        V3 group-gate metadata. Absent declarations are omitted rather than
        zero-filled so DOC-4 inapplicability remains explicit.
    """
    graph = test_graph.graph
    meta: Dict[str, Any] = {}
    declared_hierarchy = _declares_hierarchy(test_graph)
    if declared_hierarchy:
        meta["declared_hierarchical"] = declared_hierarchy
        meta["flow_direction"] = graph.direction
    if graph.clusters:
        meta["clusters"] = {
            str(cluster_id): [int(node_index) for node_index in members]
            for cluster_id, members in graph.clusters.items()
        }
    if graph.cluster_parents:
        meta["cluster_parents"] = {
            str(cluster_id): (None if parent is None else str(parent))
            for cluster_id, parent in graph.cluster_parents.items()
        }
    tree_roots = _rooted_forest_roots(test_graph)
    if tree_roots is not None:
        meta["declared_tree"] = True
        meta["tree_roots"] = tree_roots
        meta["tree_convention"] = "radial" if "radial" in test_graph.tags else "layered"
    edge_weights = graph.edge_weights
    if edge_weights is not None and int(edge_weights.numel()) > 0:
        meta["edge_weights"] = edge_weights.detach().cpu().to(dtype=torch.float64).tolist()
        meta["weight_mode"] = _declared_weight_mode(test_graph)
    planted_partition = _planted_partition_from_declared_row(test_graph)
    if planted_partition is not None:
        meta["planted_partition"] = planted_partition
    return meta


def _v3_facet_records(result: RulerV3Result) -> Dict[str, Dict[str, Any]]:
    """Serialize V3 facet records for raw score rows.

    Parameters
    ----------
    result : RulerV3Result
        V3 scoring result.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        JSON-compatible facet records keyed by facet code.
    """
    return {
        code: {
            "code": facet.code,
            "name": facet.name,
            "tier": facet.tier,
            "score": facet.score,
            "base_weight": facet.base_weight,
            "effective_weight": facet.effective_weight,
            "applicable": facet.applicable,
            "applicability_reason": facet.applicability_reason,
            "metadata": normalized_metric_value(facet.metadata),
        }
        for code, facet in result.facets.items()
    }


def score_position(
    test_graph: TestGraph,
    path_string: str,
    engine: str,
    signature: str,
    ruler: str = "v2",
) -> Dict[str, Any]:
    """Score one saved position tensor under old and extended rulers.

    Parameters
    ----------
    test_graph : TestGraph
        Reconstructed graph metadata.
    path_string : str
        Saved ``.pt`` position tensor path.
    engine : str
        Engine name.
    signature : str
        Current scoring signature.
    ruler : str, default="v2"
        Ruler version. ``"v2"`` preserves the frozen old/extended scoring path;
        ``"v3"`` scores with the V3 core plus input-gated conditional groups.

    Returns
    -------
    Dict[str, Any]
        Raw candidate score row.

    Raises
    ------
    RuntimeError
        If the graph was built without measured node sizes. Scoring with
        ``node_sizes=None`` silently nulls ``node_occlusion_score`` and
        renormalizes the composite without a facet that is ~always 1.0,
        deflating every score ~0.5-3.5 pts relative to the V2 scale (the
        2026-07-21 S1/M2 tally incident -- 6th instance of the oracle-bug
        class). Legitimate scoring always goes through ``build_graph_map()``,
        which calls ``compute_node_sizes()``; cached-row reuse paths never
        reach this function.
    """
    graph = test_graph.graph
    if graph.node_sizes is None:
        raise RuntimeError(
            f"score_position({test_graph.name!r}): graph.node_sizes is None. "
            "The graph was built without compute_node_sizes() (e.g. bare "
            "get_test_graphs()), which would silently null node_occlusion_score "
            "and deflate the composite. Build graphs via build_graph_map() or "
            "call graph.compute_node_sizes() before scoring."
        )
    positions = torch.load(path_string, map_location="cpu", weights_only=True)
    expected_shape = (graph.num_nodes, 2)
    if not isinstance(positions, torch.Tensor) or tuple(positions.shape) != expected_shape:
        raise ValueError(f"shape {getattr(positions, 'shape', None)} != {expected_shape}")
    unit_normalization = normalize_position_units_for_scoring(graph, positions, engine)
    scoring_units = scoring_unit_payload(unit_normalization)
    normalized_ruler = ruler.lower()
    if normalized_ruler not in {"v2", "v3"}:
        raise ValueError(f"unsupported ruler version: {ruler!r}")
    if normalized_ruler == "v3":
        graph_meta = graph_meta_for_v3(test_graph)
        result = score_core_v3(
            unit_normalization.positions,
            graph.edge_index,
            node_sizes=unit_normalization.node_sizes,
            graph_meta=graph_meta,
        )
        row_flags = merged_row_flags(result.flags, unit_normalization.flags)
        eligibility_key = referee_eligibility_key(result)
        return {
            "graph": test_graph.name,
            "engine": engine,
            "position_path": path_string,
            "position_sha256": sha256_file(Path(path_string)),
            "metrics": {
                "v3_scores": dict(result.scores),
                "v3_flags": list(row_flags),
                "v3_coverage": dict(result.coverage),
            },
            "v3_tiered": float(result.scores["tiered"]),
            "v3_tiered_linear": float(result.scores["tiered_linear"]),
            "v3_tiered_capped": float(result.scores.get("tiered_capped", result.scores["tiered"])),
            "v3_tiered_hold_instrument": float(
                result.scores.get("tiered_hold_instrument", result.scores["tiered"])
            ),
            "v3_equal": float(result.scores["equal"]),
            "v3_tier1_only": float(result.scores["tier1_only"]),
            "v3_referee_eligibility_key": list(eligibility_key),
            "v3_severe_g6_breach": severe_g6_breach(result),
            "v3_coincident_collapse_fraction": float(
                result.metadata.get("coincident_collapse_fraction", 0.0)
            ),
            "v3_facets": _v3_facet_records(result),
            "v3_applicability": dict(result.applicability),
            "v3_row_flags": list(row_flags),
            "graph_meta": normalized_metric_value(graph_meta),
            "scoring_units": scoring_units,
            "scoring_signature": signature,
            "score_origin": "fresh_positions",
            "ruler_version": "v3",
        }
    declared = _declares_hierarchy(test_graph)
    metrics = full(
        unit_normalization.positions,
        graph.edge_index,
        node_sizes=unit_normalization.node_sizes,
        cluster_ids=graph.cluster_ids,
        direction=graph.direction,
        declared_hierarchical=declared,
        clusters=graph.clusters or None,
        cluster_parents=graph.cluster_parents or None,
        cluster_labels=graph.cluster_labels or None,
    )
    metrics["declared_hierarchical"] = declared
    metrics_payload = normalized_metrics(metrics)
    extended = float(composite_auto(metrics_payload, is_semantically_directed(test_graph)))
    old_metrics = dict(metrics_payload)
    for key in CLUSTER_SCORE_KEYS:
        old_metrics[key] = None
    old = float(composite_auto(old_metrics, is_semantically_directed(test_graph)))
    return {
        "graph": test_graph.name,
        "engine": engine,
        "position_path": path_string,
        "position_sha256": sha256_file(Path(path_string)),
        "metrics": metrics_payload,
        "old_composite": old,
        "extended_composite": extended,
        "scoring_units": scoring_units,
        "scoring_signature": signature,
        "score_origin": "fresh_positions",
    }


def score_group(task: Tuple[str, str, List[str]]) -> List[Dict[str, Any]]:
    """Score every stored layout for one graph-engine pair.

    Parameters
    ----------
    task : Tuple[str, str, List[str]]
        Graph name, engine name, and matching tensor paths.

    Returns
    -------
    List[Dict[str, Any]]
        Candidate rows for the pair, plus error rows when all candidates fail.
    """
    graph_name, engine, paths = task
    test_graph = _WORKER_GRAPHS[graph_name]
    rows: List[Dict[str, Any]] = []
    errors: List[str] = []
    for path_string in paths:
        try:
            rows.append(score_position(test_graph, path_string, engine, _WORKER_SIGNATURE))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{path_string}: {type(exc).__name__}: {exc}")
    if rows:
        return rows
    return [
        {
            "graph": graph_name,
            "engine": engine,
            "position_path": None,
            "position_sha256": None,
            "metrics": {},
            "old_composite": None,
            "extended_composite": None,
            "scoring_signature": _WORKER_SIGNATURE,
            "score_origin": "fresh_positions_error",
            "error_count": len(errors),
            "errors": errors[:5],
        }
    ]


def score_tasks(
    tasks: List[Tuple[str, str, List[str]]],
    graphs: Dict[str, TestGraph],
    workers: int,
    signature: str,
) -> List[Dict[str, Any]]:
    """Score all tasks with a worker pool.

    Parameters
    ----------
    tasks : List[Tuple[str, str, List[str]]]
        Scoring tasks.
    graphs : Dict[str, TestGraph]
        Reconstructed corpus graphs.
    workers : int
        Process count.
    signature : str
        Current scoring signature.

    Returns
    -------
    List[Dict[str, Any]]
        Candidate score rows.
    """
    if workers <= 1:
        init_worker(graphs, signature)
        return [row for task in tasks for row in score_group(task)]
    context = mp.get_context("fork")
    with context.Pool(workers, initializer=init_worker, initargs=(graphs, signature)) as pool:
        return [
            row for rows in pool.imap_unordered(score_group, tasks, chunksize=4) for row in rows
        ]


def cached_score_value(row: Mapping[str, Any]) -> Optional[float]:
    """Extract a legacy cache score value.

    Parameters
    ----------
    row : Mapping[str, Any]
        Legacy cache row.

    Returns
    -------
    Optional[float]
        Score value when present.
    """
    value = row.get("best_composite", row.get("composite_score"))
    return None if value is None else float(value)


def cached_path_value(row: Mapping[str, Any]) -> Optional[str]:
    """Extract a legacy cache best-position path.

    Parameters
    ----------
    row : Mapping[str, Any]
        Legacy cache row.

    Returns
    -------
    Optional[str]
        Absolute path string when present.
    """
    value = row.get("best_path", row.get("position_path"))
    return None if value is None else str(Path(str(value)).resolve())


def load_legacy_field_rows(
    legacy_field_cache: Optional[Path],
    field_tasks: Sequence[Tuple[str, str, List[str]]],
    graphs: Mapping[str, TestGraph],
    signature: str,
) -> Tuple[List[Dict[str, Any]], set[Tuple[str, str]], set[Tuple[str, str]]]:
    """Load pinned non-cluster field rows from the old unversioned cache.

    Parameters
    ----------
    legacy_field_cache : Optional[Path]
        Legacy field-cache path.
    field_tasks : Sequence[Tuple[str, str, List[str]]]
        Current field score tasks.
    graphs : Mapping[str, TestGraph]
        Reconstructed graph map.
    signature : str
        Current scoring signature.

    Returns
    -------
    Tuple[List[Dict[str, Any]], set[Tuple[str, str]], set[Tuple[str, str]]]
        Reused raw rows, their ``(graph, engine)`` keys, and unscoreable legacy keys
        excluded from fresh scoring.
    """
    if legacy_field_cache is None:
        return [], set(), set()
    cache_hash = sha256_file(legacy_field_cache)
    if cache_hash != LEGACY_FIELD_CACHE_SHA256:
        raise RuntimeError(
            "legacy field cache SHA mismatch: expected "
            f"{LEGACY_FIELD_CACHE_SHA256}, got {cache_hash}"
        )
    payload = json.loads(legacy_field_cache.read_text())
    rows = payload["rows"] if isinstance(payload, dict) else payload
    cache_by_key = {(str(row["graph"]), str(row["engine"])): row for row in rows}
    if len(cache_by_key) != LEGACY_FIELD_CACHE_PAIR_COUNT:
        raise RuntimeError(
            f"legacy field cache pair count mismatch: expected "
            f"{LEGACY_FIELD_CACHE_PAIR_COUNT}, got {len(cache_by_key)}"
        )
    task_paths = {
        (graph, engine): {str(Path(path).resolve()) for path in paths}
        for graph, engine, paths in field_tasks
    }
    if not set(cache_by_key).issubset(set(task_paths)):
        raise RuntimeError("legacy field cache coverage does not match scoreable field pairs")
    reused: List[Dict[str, Any]] = []
    reused_keys: set[Tuple[str, str]] = set()
    skipped_keys: set[Tuple[str, str]] = set()
    for key, row in cache_by_key.items():
        graph_name, engine = key
        test_graph = graphs[graph_name]
        if test_graph.graph.clusters:
            continue
        position_path = cached_path_value(row)
        if position_path is not None and (
            position_path not in task_paths[key] or not Path(position_path).exists()
        ):
            raise RuntimeError(f"legacy field cache best_path is not scoreable for {key}")
        score = cached_score_value(row)
        if position_path is None or score is None:
            skipped_keys.add(key)
            continue
        reused.append(
            {
                "graph": graph_name,
                "engine": engine,
                "position_path": position_path,
                "position_sha256": sha256_file(Path(position_path)),
                "metrics": dict(row.get("metrics", {})),
                "old_composite": score,
                "extended_composite": score,
                "scoring_signature": signature,
                "score_origin": "legacy_cache_g2_noncluster",
                "legacy_field_cache_sha256": cache_hash,
            }
        )
        reused_keys.add(key)
    return reused, reused_keys, skipped_keys


def read_raw_cache(
    score_cache: Path, expected_header: Mapping[str, Any]
) -> Optional[List[Dict[str, Any]]]:
    """Read a versioned raw cache when every header field matches.

    Parameters
    ----------
    score_cache : Path
        Raw score cache path.
    expected_header : Mapping[str, Any]
        Header expected for the current inputs and ruler.

    Returns
    -------
    Optional[List[Dict[str, Any]]]
        Cached rows when valid, otherwise ``None``.
    """
    if not score_cache.exists():
        return None
    payload = json.loads(score_cache.read_text())
    if payload.get("header") != dict(expected_header):
        print("[score] raw cache stale; re-scoring saved positions", flush=True)
        return None
    return list(payload["rows"])


def write_raw_cache(
    score_cache: Path, header: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> None:
    """Write the versioned raw score cache.

    Parameters
    ----------
    score_cache : Path
        Raw score cache path.
    header : Mapping[str, Any]
        Cache header.
    rows : Sequence[Mapping[str, Any]]
        Raw score rows.

    Returns
    -------
    None
    """
    score_cache.parent.mkdir(parents=True, exist_ok=True)
    score_cache.write_text(json.dumps({"header": dict(header), "rows": list(rows)}, indent=1))


def classify(delta: float) -> str:
    """Classify a native score delta using the frozen tie band.

    Parameters
    ----------
    delta : float
        Native composite minus best field composite.

    Returns
    -------
    str
        ``strictly_best``, ``tied``, or ``behind``.
    """
    if delta > TIE_BAND:
        return "strictly_best"
    if delta >= -TIE_BAND:
        return "tied"
    return "behind"


def score_key_for_table(table_kind: str) -> str:
    """Return the raw-row score key for a table kind.

    Parameters
    ----------
    table_kind : str
        ``"old"`` or ``"extended"``.

    Returns
    -------
    str
        Raw row score key.
    """
    if table_kind == "old":
        return "old_composite"
    if table_kind == "extended":
        return "extended_composite"
    raise ValueError(f"unknown table kind: {table_kind}")


def best_rows_by_graph(
    rows: Iterable[Mapping[str, Any]], score_key: str, engine: Optional[str]
) -> Dict[str, Dict[str, Any]]:
    """Select best raw row per graph for one score column.

    Parameters
    ----------
    rows : Iterable[Mapping[str, Any]]
        Candidate raw rows.
    score_key : str
        Score column to maximize.
    engine : Optional[str]
        Exact engine filter; ``None`` means all non-native field engines.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Best row keyed by graph name.
    """
    best: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if row.get(score_key) is None:
            continue
        row_engine = str(row["engine"])
        if engine is not None and row_engine != engine:
            continue
        if engine is None and row_engine == "dagua":
            continue
        graph_name = str(row["graph"])
        row_key = _selection_key(row, score_key, native_selection=engine == "dagua")
        best_key = (
            None
            if graph_name not in best
            else _selection_key(best[graph_name], score_key, native_selection=engine == "dagua")
        )
        if best_key is None or row_key > best_key:
            best[graph_name] = dict(row)
    return best


def _selection_key(
    row: Mapping[str, Any],
    score_key: str,
    *,
    native_selection: bool,
) -> Tuple[int, Tuple[int, float], float]:
    """Return the score-neutral row-selection key.

    Parameters
    ----------
    row : Mapping[str, Any]
        Raw score row.
    score_key : str
        Score column to maximize.
    native_selection : bool
        Whether severe-G6 eligibility is binding for this comparison.

    Returns
    -------
    Tuple[int, Tuple[int, float], float]
        Lexicographic selection key. Degeneracy eligibility applies to both
        native and field rows; native V3 rows then use the shared severe-G6
        referee eligibility prefix.
    """
    degeneracy_eligible = _row_degeneracy_eligibility_key(row)
    referee_eligibility = _row_referee_eligibility_key(row) if native_selection else (1, -0.0)
    return (degeneracy_eligible, referee_eligibility, float(row[score_key]))


def _row_degeneracy_eligibility_key(row: Mapping[str, Any]) -> int:
    """Return the row-local degeneracy champion eligibility prefix.

    Parameters
    ----------
    row : Mapping[str, Any]
        Raw score row.

    Returns
    -------
    int
        ``1`` when the row can be selected as champion, otherwise ``0``.
    """
    flags = {str(flag) for flag in row.get("v3_row_flags", [])}
    return int(flags.isdisjoint(DEGENERACY_CHAMPION_INELIGIBLE_FLAGS))


def _row_referee_eligibility_key(row: Mapping[str, Any]) -> Tuple[int, float]:
    """Return a persisted referee eligibility key for a raw row.

    Parameters
    ----------
    row : Mapping[str, Any]
        Raw score row.

    Returns
    -------
    Tuple[int, float]
        Persisted severe-G6 eligibility prefix, or the compliant default for
        legacy/non-V3 rows that cannot carry a declared-weight breach.
    """
    raw_key = row.get("v3_referee_eligibility_key")
    if isinstance(raw_key, Sequence) and len(raw_key) == 2:
        return (int(raw_key[0]), float(raw_key[1]))
    if SEVERE_G6_BREACH_FLAG in {str(flag) for flag in row.get("v3_row_flags", [])}:
        return (0, -0.0)
    return (1, -0.0)


def cluster_components(row: Optional[Mapping[str, Any]], prefix: str) -> Dict[str, Any]:
    """Extract prefixed cluster component values from a raw row.

    Parameters
    ----------
    row : Optional[Mapping[str, Any]]
        Raw score row, or ``None``.
    prefix : str
        Output key prefix.

    Returns
    -------
    Dict[str, Any]
        Prefixed cluster metric values.
    """
    metrics = {} if row is None else dict(row.get("metrics", {}))
    return {f"{prefix}_{key}": metrics.get(key) for key in CLUSTER_SCORE_KEYS}


def build_table(
    rows: Sequence[Mapping[str, Any]],
    graphs: Mapping[str, TestGraph],
    names: Sequence[str],
    table_kind: str,
    signature: str,
    raw_cache_path: Path,
) -> Dict[str, Any]:
    """Build one scoreboard table from raw candidate rows.

    Parameters
    ----------
    rows : Sequence[Mapping[str, Any]]
        Raw candidate rows.
    graphs : Mapping[str, TestGraph]
        Reconstructed graph map.
    names : Sequence[str]
        Ordered graph names for the table.
    table_kind : str
        ``"old"`` or ``"extended"``.
    signature : str
        Current scoring signature.
    raw_cache_path : Path
        Raw score cache path.

    Returns
    -------
    Dict[str, Any]
        Scoreboard JSON payload.
    """
    score_key = score_key_for_table(table_kind)
    native_by_graph = best_rows_by_graph(rows, score_key, engine="dagua")
    field_by_graph = best_rows_by_graph(rows, score_key, engine=None)
    per_graph: List[Dict[str, Any]] = []
    tallies = {"strictly_best": 0, "tied": 0, "behind": 0, "missing": 0}
    for graph_name in names:
        native_row = native_by_graph.get(graph_name)
        field_row = field_by_graph.get(graph_name)
        if native_row is None or field_row is None:
            tallies["missing"] += 1
            per_graph.append(
                {
                    "graph": graph_name,
                    "status": "missing",
                    "native": None if native_row is None else native_row[score_key],
                    "field_best": None if field_row is None else field_row[score_key],
                    "field_best_engine": None if field_row is None else field_row["engine"],
                    "native_path": None if native_row is None else native_row.get("position_path"),
                    "field_best_path": (
                        None if field_row is None else field_row.get("position_path")
                    ),
                }
            )
            continue
        delta = float(native_row[score_key]) - float(field_row[score_key])
        status = classify(delta)
        tallies[status] += 1
        test_graph = graphs[graph_name]
        table_row: Dict[str, Any] = {
            "graph": graph_name,
            "family": ",".join(sorted(test_graph.tags)) if test_graph.tags else "",
            "ruler": ("directed" if is_semantically_directed(test_graph) else "undirected"),
            "native": float(native_row[score_key]),
            "field_best": float(field_row[score_key]),
            "field_best_engine": field_row["engine"],
            "delta": delta,
            "status": status,
            "native_path": native_row.get("position_path"),
            "field_best_path": field_row.get("position_path"),
        }
        if test_graph.graph.clusters:
            table_row.update(cluster_components(native_row, "native"))
            table_row.update(cluster_components(field_row, "field_best"))
        per_graph.append(table_row)
    best_or_tied = tallies["strictly_best"] + tallies["tied"]
    return {
        "ruler_schema": RULER_SCHEMA,
        "table_kind": table_kind,
        "corpus_size": len(names),
        "tie_band": TIE_BAND,
        "best_or_tied": best_or_tied,
        "tallies": tallies,
        "scoring_signature": signature,
        "raw_cache_path": str(raw_cache_path),
        "per_graph": per_graph,
    }


def validate_raw_integrity(
    rows: Sequence[Mapping[str, Any]],
    graphs: Mapping[str, TestGraph],
    names: Sequence[str],
    signature: str,
) -> None:
    """Validate mandatory raw-score integrity checks.

    Parameters
    ----------
    rows : Sequence[Mapping[str, Any]]
        Raw candidate rows.
    graphs : Mapping[str, TestGraph]
        Reconstructed graph map.
    names : Sequence[str]
        Expected graph names.
    signature : str
        Current scoring signature.

    Returns
    -------
    None
    """
    raw_by_graph_engine: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        raw_by_graph_engine[(str(row["graph"]), str(row["engine"]))].append(row)
        graph = graphs[str(row["graph"])]
        if graph.graph.clusters:
            if row.get("score_origin") != "fresh_positions":
                raise RuntimeError(
                    f"clustered raw row is not fresh: {row['graph']} {row['engine']}"
                )
            if row.get("scoring_signature") != signature:
                raise RuntimeError(
                    f"clustered raw row has stale signature: {row['graph']} {row['engine']}"
                )
            metrics = dict(row.get("metrics", {}))
            if metrics.get("_cluster_quality_supported") is False:
                raise RuntimeError(
                    f"clustered raw row used unsupported cluster quality: {row['graph']}"
                )
            for key in CLUSTER_SCORE_KEYS:
                if key not in metrics:
                    raise RuntimeError(
                        f"clustered raw row missing {key}: {row['graph']} {row['engine']}"
                    )
                value = metrics[key]
                if value is not None and not math.isfinite(float(value)):
                    raise RuntimeError(f"clustered raw row has non-finite {key}: {row['graph']}")
            extended = float(composite_auto(metrics, is_semantically_directed(graph)))
            old_metrics = dict(metrics)
            for key in CLUSTER_SCORE_KEYS:
                old_metrics[key] = None
            old = float(composite_auto(old_metrics, is_semantically_directed(graph)))
            if not math.isclose(
                extended, float(row["extended_composite"]), rel_tol=0.0, abs_tol=1e-9
            ):
                raise RuntimeError(
                    f"clustered extended score mismatch: {row['graph']} {row['engine']}"
                )
            if not math.isclose(old, float(row["old_composite"]), rel_tol=0.0, abs_tol=1e-9):
                raise RuntimeError(f"clustered old score mismatch: {row['graph']} {row['engine']}")
        else:
            if row.get("old_composite") != row.get("extended_composite"):
                raise RuntimeError(
                    f"non-cluster old/extended scores differ: {row['graph']} {row['engine']}"
                )
            if row.get("score_origin") == "legacy_cache_g2_noncluster" and (
                row.get("legacy_field_cache_sha256") != LEGACY_FIELD_CACHE_SHA256
            ):
                raise RuntimeError(
                    f"legacy cache row lacks pinned hash: {row['graph']} {row['engine']}"
                )
    for graph_name in names:
        native_rows = raw_by_graph_engine.get((graph_name, "dagua"), [])
        field_rows = [
            row
            for (row_graph, row_engine), grouped in raw_by_graph_engine.items()
            if row_graph == graph_name and row_engine != "dagua"
            for row in grouped
        ]
        if not any(row.get("score_origin") == "fresh_positions" for row in native_rows):
            raise RuntimeError(f"missing fresh native score for {graph_name}")
        if not any(row.get("extended_composite") is not None for row in field_rows):
            raise RuntimeError(f"missing usable field score for {graph_name}")


def validate_table_integrity(
    table: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    names: Sequence[str],
) -> None:
    """Validate independent field-best computation for a table.

    Parameters
    ----------
    table : Mapping[str, Any]
        Scoreboard table.
    rows : Sequence[Mapping[str, Any]]
        Raw candidate rows.
    names : Sequence[str]
        Expected graph names.

    Returns
    -------
    None
    """
    table_rows = list(table["per_graph"])
    row_names = [str(row["graph"]) for row in table_rows]
    if len(row_names) != len(names) or set(row_names) != set(names):
        raise RuntimeError("table graph-name coverage mismatch")
    score_key = score_key_for_table(str(table["table_kind"]))
    field_by_graph = best_rows_by_graph(rows, score_key, engine=None)
    for table_row in table_rows:
        if table_row["status"] == "missing":
            continue
        graph_name = str(table_row["graph"])
        winner = field_by_graph[graph_name]
        if not math.isclose(
            float(table_row["field_best"]), float(winner[score_key]), rel_tol=0.0, abs_tol=1e-12
        ):
            raise RuntimeError(f"field_best mismatch for {graph_name}")
        if table_row["field_best_engine"] != winner["engine"]:
            raise RuntimeError(f"field_best_engine mismatch for {graph_name}")
        if table_row["field_best_path"] != winner.get("position_path"):
            raise RuntimeError(f"field_best_path mismatch for {graph_name}")


def assert_native_runtime_referee_tripwire(rows: Sequence[Mapping[str, Any]]) -> None:
    """Assert the deferred runtime G6 referee remains safe.

    Parameters
    ----------
    rows : Sequence[Mapping[str, Any]]
        Raw scorer rows from the mandated re-baseline.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If a native row carries the score-neutral severe-G6 breach flag. That
        result means the in-pipeline runtime referee must be wired before V3
        freeze, even though this patch intentionally leaves it untouched because
        ``composite_auto`` has no G6 reward term.
    """
    offenders = [
        f"{row.get('graph')}:{row.get('position_path')}"
        for row in rows
        if row.get("engine") == "dagua"
        and SEVERE_G6_BREACH_FLAG in {str(flag) for flag in row.get("v3_row_flags", [])}
    ]
    if offenders:
        preview = ", ".join(offenders[:5])
        raise RuntimeError(
            "native severe-G6 breach flag landed during re-baseline; wire the runtime "
            f"referee eligibility key before freeze. Offenders: {preview}"
        )


def validate_outputs(
    old_table: Mapping[str, Any],
    extended_table: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    graphs: Mapping[str, TestGraph],
    old_names: Sequence[str],
    extended_names: Sequence[str],
    signature: str,
) -> None:
    """Run the recipe's mandatory scorer self-fail checks.

    Parameters
    ----------
    old_table : Mapping[str, Any]
        Old-ruler table.
    extended_table : Mapping[str, Any]
        Extended-ruler table.
    rows : Sequence[Mapping[str, Any]]
        Raw candidate rows.
    graphs : Mapping[str, TestGraph]
        Reconstructed graph map.
    old_names : Sequence[str]
        Frozen old-108 names.
    extended_names : Sequence[str]
        Extended table names.
    signature : str
        Current scoring signature.

    Returns
    -------
    None
    """
    if len(set(old_names)) != 108:
        raise RuntimeError(
            f"old table must have 108 unique frozen names, got {len(set(old_names))}"
        )
    if len(set(extended_names)) == 121:
        new_names = tuple(name for name in extended_names if name not in set(old_names))
        if new_names != R8_GRAPH_NAMES:
            raise RuntimeError("extended table R8 names do not match exact recipe tuple")
    elif extended_names != old_names:
        raise RuntimeError(
            f"extended table must have 121 unique names, got {len(set(extended_names))}"
        )
    validate_raw_integrity(rows, graphs, extended_names, signature)
    validate_table_integrity(old_table, rows, old_names)
    validate_table_integrity(extended_table, rows, extended_names)
    assert_native_runtime_referee_tripwire(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the native-vs-field dual-ruler comparison.

    Parameters
    ----------
    argv : Optional[Sequence[str]], optional
        Explicit argument sequence, or ``None`` for ``sys.argv``.

    Returns
    -------
    int
        Process exit code.
    """
    args = parse_args(argv)
    old_names, extended_names = selected_names(args.corpus_positions, args.include_r8)
    graphs = build_graph_map(extended_names)
    signature = scoring_signature()

    field_tasks = load_run_tasks(args.field_dir, extended_names)
    native_tasks = load_run_tasks(args.native_dir, extended_names, engine_filter="dagua")
    all_tasks = [*field_tasks, *native_tasks]
    header = cache_header(
        old_names,
        extended_names,
        graphs,
        args.field_dir,
        args.native_dir,
        all_tasks,
        signature,
    )

    raw_rows = read_raw_cache(args.score_cache, header)
    if raw_rows is None:
        legacy_rows, reused_keys, skipped_keys = load_legacy_field_rows(
            args.legacy_field_cache, field_tasks, graphs, signature
        )
        excluded_keys = reused_keys | skipped_keys
        fresh_field_tasks = [
            task for task in field_tasks if (task[0], task[1]) not in excluded_keys
        ]
        print(
            f"[score] scoring field: {len(fresh_field_tasks)} graph-engine pairs "
            f"({len(legacy_rows)} legacy rows reused, {len(skipped_keys)} unscoreable skipped)",
            flush=True,
        )
        field_rows = [
            *legacy_rows,
            *score_tasks(fresh_field_tasks, graphs, args.workers, signature),
        ]
        print(f"[score] scoring native: {len(native_tasks)} graph-engine pairs", flush=True)
        native_rows = score_tasks(native_tasks, graphs, args.workers, signature)
        raw_rows = [*field_rows, *native_rows]
        write_raw_cache(args.score_cache, header, raw_rows)
    else:
        print(f"[score] raw cache hit: {len(raw_rows)} rows", flush=True)

    old_table = build_table(raw_rows, graphs, old_names, "old", signature, args.score_cache)
    extended_table = build_table(
        raw_rows, graphs, extended_names, "extended", signature, args.score_cache
    )
    validate_outputs(
        old_table, extended_table, raw_rows, graphs, old_names, extended_names, signature
    )

    args.old_output.parent.mkdir(parents=True, exist_ok=True)
    args.extended_output.parent.mkdir(parents=True, exist_ok=True)
    args.old_output.write_text(json.dumps(old_table, indent=1))
    args.extended_output.write_text(json.dumps(extended_table, indent=1))

    old_best = int(old_table["best_or_tied"])
    extended_best = int(extended_table["best_or_tied"])
    print(
        f"[score] old best-or-tied {old_best}/{len(old_names)}; "
        f"extended best-or-tied {extended_best}/{len(extended_names)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
