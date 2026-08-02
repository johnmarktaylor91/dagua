"""Verify ELK compound wrapper fidelity against elkjs across seeds."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dagua.eval import distributional_fidelity as df  # noqa: E402
from dagua.eval.competitors.elk_competitor import (  # noqa: E402
    _build_elk_children,
    _cluster_children,
    _run_elk_layout_json,
)
from dagua.graph import DaguaGraph  # noqa: E402
from dagua.layout.ops.elk_compound import (  # noqa: E402
    _CompoundOptions,
    layout_elk_compound_with_diagnostics,
)
from scripts.verify_elk_distributional import (  # noqa: E402
    POINT_MASS_THRESHOLD,
    PROCRUSTES_BAND_FLOOR,
    VARIANCE_RATIO_MAX,
    VARIANCE_RATIO_MIN,
    LayoutSample,
    _metric_results,
    _reference_layer_indices,
    _reference_order,
    _sample_metrics,
    _seed_range,
)
from scripts.verify_elk_fidelity import _ensure_elkjs_node_path  # noqa: E402


@dataclass(frozen=True)
class CompoundGraphResult:
    """Compound verification result for one fixture.

    Attributes
    ----------
    name : str
        Fixture name.
    num_nodes : int
        Node count.
    num_edges : int
        Edge count.
    structural_exact : bool
        Whether wrapper structural gates match elkjs.
    variance_match : bool
        Whether native and reference cross-seed spreads are comparable.
    dagua_spread : float
        Native within-cloud Procrustes spread.
    reference_spread : float
        Reference within-cloud Procrustes spread.
    procrustes_between : float
        Mean native-vs-reference off-diagonal Procrustes distance.
    procrustes_within_band : float
        Allowed spread band.
    procrustes_pass : bool
        Whether ``procrustes_between`` is inside the spread band.
    metrics : list[Any]
        Scalar TOST metric results from the flat ELK harness.
    first_divergent_stage : str
        Named residual stage when the graph is not fully equivalent.
    """

    name: str
    num_nodes: int
    num_edges: int
    structural_exact: bool
    variance_match: bool
    dagua_spread: float
    reference_spread: float
    procrustes_between: float
    procrustes_within_band: float
    procrustes_pass: bool
    metrics: List[Any]
    first_divergent_stage: str

    @property
    def verdict(self) -> str:
        """Return the compound fidelity verdict.

        Returns
        -------
        str
            ``STRUCTURAL_EXACT_DISTRIBUTIONAL_EQUIVALENT`` when all gates pass,
            otherwise ``not_distributional_equivalent``.
        """
        if (
            self.structural_exact
            and self.variance_match
            and self.procrustes_pass
            and all(metric.passed for metric in self.metrics)
        ):
            return "STRUCTURAL_EXACT_DISTRIBUTIONAL_EQUIVALENT"
        return "not_distributional_equivalent"


def _edge_index(edges: Sequence[Tuple[int, int]]) -> torch.Tensor:
    """Convert edge pairs to an edge-index tensor.

    Parameters
    ----------
    edges : sequence[tuple[int, int]]
        Directed edge pairs.

    Returns
    -------
    torch.Tensor
        Long edge-index tensor with shape ``[2, E]``.
    """
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def _graph_from_edges(name: str, num_nodes: int, edges: Sequence[Tuple[int, int]]) -> DaguaGraph:
    """Build a deterministic labeled graph with measured node sizes.

    Parameters
    ----------
    name : str
        Fixture name used as label prefix.
    num_nodes : int
        Number of nodes.
    edges : sequence[tuple[int, int]]
        Directed edge pairs.

    Returns
    -------
    DaguaGraph
        Graph with measured node sizes.
    """
    graph = DaguaGraph()
    for node in range(num_nodes):
        graph.add_node(node, label=f"{name}_{node}")
    for source, target in edges:
        graph.add_edge(source, target)
    graph.compute_node_sizes()
    return graph


def _compound_graphs() -> List[Tuple[str, DaguaGraph]]:
    """Build compound ELK verification fixtures.

    Returns
    -------
    list[tuple[str, DaguaGraph]]
        Small compound fixtures covering cross-boundary edges, nesting,
        disconnected clusters, and direct leaf mixes.
    """
    bridge = _graph_from_edges("compound_bridge", 7, [(0, 1), (1, 2), (3, 4), (2, 3), (5, 6)])
    bridge.add_cluster("left", [0, 1, 2])
    bridge.add_cluster("right", [3, 4])
    bridge.compute_node_sizes()

    nested = _graph_from_edges("nested_mix", 6, [(0, 1), (1, 2), (3, 4), (2, 5), (4, 5)])
    nested.add_cluster("outer", [0, 1, 2, 3, 4])
    nested.add_cluster("inner_a", [0, 1], parent="outer")
    nested.add_cluster("inner_b", [3, 4], parent="outer")
    nested.compute_node_sizes()

    disconnected = _graph_from_edges("disconnected_cluster", 6, [(0, 1), (2, 3), (4, 5)])
    disconnected.add_cluster("alpha", [0, 1, 2, 3])
    disconnected.compute_node_sizes()

    direct_leaf = _graph_from_edges("direct_leaf_mix", 5, [(0, 1), (2, 3), (1, 4), (3, 4)])
    direct_leaf.add_cluster("group", [0, 1, 2, 3])
    direct_leaf.add_cluster("sub", [0, 1], parent="group")
    direct_leaf.compute_node_sizes()

    return [
        ("compound_bridge", bridge),
        ("nested_mix", nested),
        ("disconnected_cluster", disconnected),
        ("direct_leaf_mix", direct_leaf),
    ]


def _elk_graph_payload(graph: DaguaGraph, seed: int) -> Dict[str, Any]:
    """Build the same nested ELK JSON payload as the competitor adapter.

    Parameters
    ----------
    graph : DaguaGraph
        Source graph.
    seed : int
        ELK random seed.

    Returns
    -------
    dict[str, Any]
        JSON-compatible elkjs graph.
    """
    emitted_nodes: Set[int] = set()
    children = _build_elk_children(graph, None, _cluster_children(graph), emitted_nodes)
    edges = [
        {"id": f"e{edge_index}", "sources": [str(source)], "targets": [str(target)]}
        for edge_index, (source, target) in enumerate(graph.edge_index.t().tolist())
    ]
    return {
        "id": "root",
        "layoutOptions": {
            "elk.algorithm": "layered",
            "elk.direction": "DOWN",
            "elk.spacing.nodeNode": "40",
            "elk.layered.spacing.nodeNodeBetweenLayers": "60",
            "elk.layered.thoroughness": "7",
            "elk.randomSeed": int(seed),
        },
        "children": children,
        "edges": edges,
    }


def _routed_edge_ids(data: Dict[str, Any]) -> Set[int]:
    """Collect routed edge ids recursively from elkjs output.

    Parameters
    ----------
    data : dict[str, Any]
        Parsed elkjs layout graph.

    Returns
    -------
    set[int]
        Edge indices whose output has at least one routing section.
    """
    routed: Set[int] = set()

    def visit(container: Dict[str, Any]) -> None:
        """Traverse nested ELK containers.

        Parameters
        ----------
        container : dict[str, Any]
            ELK graph or child object.

        Returns
        -------
        None
            The enclosing ``routed`` set is mutated.
        """
        for edge in container.get("edges", []) if isinstance(container.get("edges"), list) else []:
            if not isinstance(edge, dict):
                continue
            edge_id = str(edge.get("id", ""))
            if edge_id.startswith("e") and edge_id[1:].isdigit() and edge.get("sections"):
                routed.add(int(edge_id[1:]))
        for child in (
            container.get("children", []) if isinstance(container.get("children"), list) else []
        ):
            if isinstance(child, dict):
                visit(child)

    visit(data)
    return routed


def _reference_sample(
    graph: DaguaGraph,
    seed: int,
    graph_distances: np.ndarray,
    timeout: float,
) -> Tuple[LayoutSample, Set[int]]:
    """Collect one reference compound sample and routed-edge structure.

    Parameters
    ----------
    graph : DaguaGraph
        Source graph.
    seed : int
        ELK random seed.
    graph_distances : numpy.ndarray
        All-pairs graph distances with shape ``[N, N]``.
    timeout : float
        elkjs timeout.

    Returns
    -------
    tuple[LayoutSample, set[int]]
        Metric sample and routed edge indices.

    Raises
    ------
    RuntimeError
        If elkjs fails.
    """
    data, _, error = _run_elk_layout_json(_elk_graph_payload(graph, seed), timeout)
    if data is None:
        raise RuntimeError(f"elkjs failed on seed {seed}: {error}")
    positions = torch.zeros(graph.num_nodes, 2, dtype=torch.float64)

    from dagua.eval.competitors.elk_competitor import _collect_elk_positions

    _collect_elk_positions(data.get("children", []), positions)
    position_array = positions.detach().cpu().numpy().astype(np.float64, copy=False)
    layer_indices = _reference_layer_indices(positions)
    order = _reference_order(positions)
    return (
        _sample_metrics(
            seed,
            position_array,
            graph.edge_index.detach().cpu().to(dtype=torch.long),
            layer_indices,
            order,
            graph_distances,
        ),
        _routed_edge_ids(data),
    )


def _native_sample(
    graph: DaguaGraph,
    seed: int,
    graph_distances: np.ndarray,
) -> Tuple[LayoutSample, Set[int]]:
    """Collect one native compound sample and dropped-edge structure.

    Parameters
    ----------
    graph : DaguaGraph
        Source graph.
    seed : int
        ELK random seed.
    graph_distances : numpy.ndarray
        All-pairs graph distances with shape ``[N, N]``.

    Returns
    -------
    tuple[LayoutSample, set[int]]
        Metric sample and dropped edge indices.
    """
    edge_index = graph.edge_index.detach().cpu().to(dtype=torch.long)
    node_sizes = graph.node_sizes.detach().cpu().to(dtype=torch.float64)
    positions, diagnostics = layout_elk_compound_with_diagnostics(
        edge_index=edge_index,
        num_nodes=graph.num_nodes,
        node_sizes=node_sizes,
        clusters=graph.clusters,
        cluster_parents=graph.cluster_parents,
        root_options=_CompoundOptions(
            direction="DOWN",
            node_node_spacing=40.0,
            between_layers_spacing=60.0,
            cycle_breaking_strategy="greedy",
            layering_strategy="network_simplex",
            crossing_minimization_strategy="layer_sweep",
            node_placement_strategy="brandes_koepf",
            random_seed=seed,
            thoroughness=7,
        ),
    )
    dropped_pairs = set(diagnostics["dropped_edges"])
    dropped_indices = {
        edge_id
        for edge_id, pair in enumerate((tuple(edge) for edge in edge_index.t().tolist()))
        if pair in dropped_pairs
    }
    layer_indices = _reference_layer_indices(positions)
    order = _reference_order(positions)
    return (
        _sample_metrics(
            seed,
            positions.detach().cpu().numpy().astype(np.float64, copy=False),
            edge_index,
            layer_indices,
            order,
            graph_distances,
        ),
        dropped_indices,
    )


def _variance_match(dagua_spread: float, reference_spread: float) -> bool:
    """Return whether layout-cloud spreads match.

    Parameters
    ----------
    dagua_spread : float
        Native mean within-cloud Procrustes distance.
    reference_spread : float
        Reference mean within-cloud Procrustes distance.

    Returns
    -------
    bool
        ``True`` when both are point masses or the spread ratio is bounded.
    """
    dagua_point = dagua_spread < POINT_MASS_THRESHOLD
    reference_point = reference_spread < POINT_MASS_THRESHOLD
    if dagua_point or reference_point:
        return dagua_point and reference_point
    ratio = dagua_spread / reference_spread
    return VARIANCE_RATIO_MIN <= ratio <= VARIANCE_RATIO_MAX


def _analyze_graph(
    name: str,
    graph: DaguaGraph,
    seeds: Sequence[int],
    timeout: float,
) -> CompoundGraphResult:
    """Analyze one compound fixture.

    Parameters
    ----------
    name : str
        Fixture name.
    graph : DaguaGraph
        Source graph.
    seeds : sequence[int]
        ELK seeds.
    timeout : float
        Per-elkjs timeout.

    Returns
    -------
    CompoundGraphResult
        Structural and distributional verdict.
    """
    edge_index = graph.edge_index.detach().cpu().to(dtype=torch.long)
    graph_distances = df.prepare_graph_distances(edge_index.detach().cpu().numpy(), graph.num_nodes)
    reference_samples: List[LayoutSample] = []
    native_samples: List[LayoutSample] = []
    structural_exact = True
    for seed in seeds:
        reference, routed_edges = _reference_sample(graph, seed, graph_distances, timeout)
        native, dropped_edges = _native_sample(graph, seed, graph_distances)
        reference_samples.append(reference)
        native_samples.append(native)
        structural_exact = structural_exact and dropped_edges.isdisjoint(routed_edges)
    rng = np.random.default_rng(9103)
    mode = df.analyze_mode_a(
        [sample.positions for sample in native_samples],
        [sample.positions for sample in reference_samples],
        rng,
    )
    dagua_spread = float(mode["plain_mean_W_D"])
    reference_spread = float(mode["plain_mean_W_R"])
    between = float(mode["mean_B_offdiag"])
    within_band = max(dagua_spread, reference_spread, PROCRUSTES_BAND_FLOOR)
    metrics = _metric_results(native_samples, reference_samples)
    variance_match = _variance_match(dagua_spread, reference_spread)
    procrustes_pass = between <= within_band
    if not structural_exact:
        first_divergent_stage = "edge-redistribution/cross-hierarchy-drop"
    elif not (variance_match and procrustes_pass and all(metric.passed for metric in metrics)):
        first_divergent_stage = "flat-layered-distributional-tier"
    else:
        first_divergent_stage = "none"
    return CompoundGraphResult(
        name=name,
        num_nodes=int(graph.num_nodes),
        num_edges=int(edge_index.shape[1]),
        structural_exact=structural_exact,
        variance_match=variance_match,
        dagua_spread=dagua_spread,
        reference_spread=reference_spread,
        procrustes_between=between,
        procrustes_within_band=within_band,
        procrustes_pass=procrustes_pass,
        metrics=metrics,
        first_divergent_stage=first_divergent_stage,
    )


def _metric_failures(result: CompoundGraphResult) -> str:
    """Format failing scalar metric names.

    Parameters
    ----------
    result : CompoundGraphResult
        Result to inspect.

    Returns
    -------
    str
        Comma-separated failures, or an empty string.
    """
    return ", ".join(metric.name for metric in result.metrics if not metric.passed)


def _print_results(results: Sequence[CompoundGraphResult]) -> None:
    """Print compound verification results.

    Parameters
    ----------
    results : sequence[CompoundGraphResult]
        Results to print.

    Returns
    -------
    None
        Writes JSON lines and a compact summary to stdout.
    """
    for result in results:
        print(
            json.dumps(
                {
                    "graph": result.name,
                    "verdict": result.verdict,
                    "structural_exact": result.structural_exact,
                    "variance_match": result.variance_match,
                    "dagua_spread": result.dagua_spread,
                    "reference_spread": result.reference_spread,
                    "procrustes_between": result.procrustes_between,
                    "procrustes_within_band": result.procrustes_within_band,
                    "procrustes_pass": result.procrustes_pass,
                    "metric_failures": _metric_failures(result),
                    "first_divergent_stage": result.first_divergent_stage,
                },
                sort_keys=True,
            )
        )
    equivalent = sum(
        result.verdict == "STRUCTURAL_EXACT_DISTRIBUTIONAL_EQUIVALENT" for result in results
    )
    structural = sum(result.structural_exact for result in results)
    print(
        f"summary: structural_exact={structural}/{len(results)} "
        f"distributional_equivalent={equivalent}/{len(results)}"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run compound ELK verification.

    Parameters
    ----------
    argv : sequence[str] | None, optional
        Command-line arguments.

    Returns
    -------
    int
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=30, help="Number of one-based seeds to run.")
    parser.add_argument("--timeout", type=float, default=300.0, help="Per-elkjs-layout timeout.")
    args = parser.parse_args(argv)

    _ensure_elkjs_node_path()
    seeds = _seed_range(args.seeds)
    results = [
        _analyze_graph(name, graph, seeds, timeout=float(args.timeout))
        for name, graph in _compound_graphs()
    ]
    _print_results(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
