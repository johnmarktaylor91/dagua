"""Verify CoSE-Bilkent compound distributional fidelity against Cytoscape."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dagua.eval import distributional_fidelity as df  # noqa: E402
from dagua.eval.competitors import get_competitor  # noqa: E402
from dagua.eval.equivalence_metrics import normalized_stress  # noqa: E402
from dagua.graph import DaguaGraph  # noqa: E402
from dagua.layout.ops.cose_base_compound import (  # noqa: E402
    CoSECompoundOptions,
    layout_cose_base_compound,
)
from dagua.layout.ops.pipelines.cose_bilkent import layout_cose_bilkent_pipeline  # noqa: E402
from dagua.metrics import count_crossings, edge_length_cv, neighborhood_preservation  # noqa: E402

PROCRUSTES_BAND_FLOOR = 1.0e-3
POINT_MASS_THRESHOLD = 1.0e-6
TOST_ALPHA = 0.05
VARIANCE_RATIO_MIN = 0.5
VARIANCE_RATIO_MAX = 2.0
CYTOSCAPE_DEFAULT_NODE_WIDTH = 30.0
CYTOSCAPE_DEFAULT_NODE_HEIGHT = 30.0


@dataclass(frozen=True)
class LayoutSample:
    """One seeded layout and its scalar quality metrics."""

    seed: int
    positions: np.ndarray
    crossings: float
    stress: float
    edge_length_cv: float
    ordering_inversions: float
    neighborhood_preservation: float


@dataclass(frozen=True)
class MetricResult:
    """TOST result for one distributional quality metric."""

    name: str
    native_mean: float
    reference_mean: float
    native_sd: float
    reference_sd: float
    margin: float
    p_tost: float
    passed: bool


@dataclass(frozen=True)
class GraphResult:
    """Distributional verdict for one verifier fixture and subject."""

    subject: str
    name: str
    num_nodes: int
    num_edges: int
    variance_match: bool
    native_spread: float
    reference_spread: float
    procrustes_between: float
    procrustes_within_band: float
    procrustes_pass: bool
    split_equivalent: bool
    metrics: List[MetricResult]

    @property
    def verdict(self) -> str:
        """Return the graph-level distributional verdict.

        Returns
        -------
        str
            ``DISTRIBUTIONAL_EQUIVALENT`` when quality metrics, variance, and
            positional spread gates all pass.
        """
        if (
            self.variance_match
            and self.procrustes_pass
            and self.split_equivalent
            and all(metric.passed for metric in self.metrics)
        ):
            return "DISTRIBUTIONAL_EQUIVALENT"
        return "not_distributional_equivalent"


def _add_nodes(graph: DaguaGraph, count: int) -> None:
    """Add numbered nodes to a graph.

    Parameters
    ----------
    graph : DaguaGraph
        Graph to mutate.
    count : int
        Number of nodes to add.

    Returns
    -------
    None
        Mutates ``graph``.
    """
    for node in range(count):
        graph.add_node(str(node))


def _finalize_graph(graph: DaguaGraph) -> DaguaGraph:
    """Finalize fixture graph tensors.

    Parameters
    ----------
    graph : DaguaGraph
        Graph to finalize.

    Returns
    -------
    DaguaGraph
        Same graph with node sizes computed.
    """
    graph.compute_node_sizes()
    return graph


def _compound_micro_fixture() -> DaguaGraph:
    """Build the existing two-cluster Cytoscape micro fixture.

    Returns
    -------
    DaguaGraph
        Six-node compound graph with two sibling clusters.
    """
    graph = DaguaGraph()
    _add_nodes(graph, 6)
    for source, target in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3)]:
        graph.add_edge(str(source), str(target))
    graph.add_cluster("a", ["0", "1", "2"])
    graph.add_cluster("b", ["3", "4", "5"])
    return _finalize_graph(graph)


def _nested_fixture() -> DaguaGraph:
    """Build a nested compound fixture.

    Returns
    -------
    DaguaGraph
        Graph with one nested child cluster under a larger parent.
    """
    graph = DaguaGraph()
    _add_nodes(graph, 8)
    for source, target in [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (2, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (1, 6),
    ]:
        graph.add_edge(str(source), str(target))
    graph.add_cluster("outer", ["0", "1", "2", "3", "4", "5"])
    graph.add_cluster("inner", ["1", "2", "3"], parent="outer")
    graph.add_cluster("right", ["6", "7"])
    return _finalize_graph(graph)


def _tiling_tree_fixture() -> DaguaGraph:
    """Build a fixture exercising zero-degree tiling and tree reduction.

    Returns
    -------
    DaguaGraph
        Compound graph with zero-degree members and leaf-heavy trees.
    """
    graph = DaguaGraph()
    _add_nodes(graph, 10)
    for source, target in [
        (0, 1),
        (1, 2),
        (2, 3),
        (2, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (8, 9),
        (3, 8),
    ]:
        graph.add_edge(str(source), str(target))
    graph.add_cluster("left", ["0", "1", "2", "3", "4"])
    graph.add_cluster("right", ["5", "6", "7", "8", "9"])
    return _finalize_graph(graph)


def _verification_graphs() -> List[Tuple[str, DaguaGraph]]:
    """Return CoSE-Bilkent compound distributional fixtures.

    Returns
    -------
    list[tuple[str, DaguaGraph]]
        Named compound fixture graphs.
    """
    return [
        ("compound_micro", _compound_micro_fixture()),
        ("nested_compound", _nested_fixture()),
        ("tiling_tree", _tiling_tree_fixture()),
    ]


def _edge_index(graph: DaguaGraph) -> torch.Tensor:
    """Return a CPU long edge index for a fixture graph.

    Parameters
    ----------
    graph : DaguaGraph
        Fixture graph.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, E]``.
    """
    return graph.edge_index.detach().cpu().to(dtype=torch.long)


def _node_sizes(graph: DaguaGraph) -> torch.Tensor:
    """Return a CPU float node-size tensor for a fixture graph.

    Parameters
    ----------
    graph : DaguaGraph
        Fixture graph.

    Returns
    -------
    torch.Tensor
        Node sizes with shape ``[N, 2]``.
    """
    if graph.node_sizes is None:
        graph.compute_node_sizes()
    return graph.node_sizes.detach().cpu().to(dtype=torch.float64)


def _matched_cytoscape_node_sizes(graph: DaguaGraph) -> torch.Tensor:
    """Return Cytoscape's unstyled headless node dimensions for native runs.

    Parameters
    ----------
    graph : DaguaGraph
        Fixture graph whose node count determines the output shape.

    Returns
    -------
    torch.Tensor
        Width/height tensor with shape ``[N, 2]`` using Cytoscape's default
        30-by-30 leaf dimensions.
    """
    size = torch.tensor(
        [CYTOSCAPE_DEFAULT_NODE_WIDTH, CYTOSCAPE_DEFAULT_NODE_HEIGHT],
        dtype=torch.float64,
    )
    return size.repeat(graph.num_nodes, 1)


def _ordering_inversion_rate(
    positions: np.ndarray,
    clusters: Mapping[str, Any],
    num_nodes: int,
) -> float:
    """Measure cluster-local x-order inversions against model order.

    Parameters
    ----------
    positions : numpy.ndarray
        Layout positions with shape ``[N, 2]``.
    clusters : Mapping[str, Any]
        Cluster membership map.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    float
        Normalized inversion count in ``[0, 1]``. CoSE has no layered order, so
        this uses the stable model order inside each compound scope.
    """
    groups: List[List[int]] = []
    assigned: set[int] = set()
    for members in clusters.values():
        if isinstance(members, Mapping):
            continue
        group = sorted(int(member) for member in members)
        if len(group) > 1:
            groups.append(group)
            assigned.update(group)
    root_group = [node for node in range(num_nodes) if node not in assigned]
    if len(root_group) > 1:
        groups.append(root_group)

    inversions = 0
    possible = 0
    for group in groups:
        ordered = sorted(group, key=lambda node: (positions[node, 0], positions[node, 1], node))
        possible += len(ordered) * (len(ordered) - 1) // 2
        for left_index, left_node in enumerate(ordered):
            for right_node in ordered[left_index + 1 :]:
                if left_node > right_node:
                    inversions += 1
    if possible == 0:
        return 0.0
    return inversions / possible


def _sample_metrics(
    seed: int,
    positions: np.ndarray,
    graph: DaguaGraph,
    graph_distances: np.ndarray,
) -> LayoutSample:
    """Compute scalar metrics for one seeded layout.

    Parameters
    ----------
    seed : int
        Public random seed.
    positions : numpy.ndarray
        Layout positions with shape ``[N, 2]``.
    graph : DaguaGraph
        Fixture graph.
    graph_distances : numpy.ndarray
        Precomputed graph shortest-path distances.

    Returns
    -------
    LayoutSample
        Sample record with quality metrics.
    """
    edge_index = _edge_index(graph)
    pos_tensor = torch.as_tensor(positions, dtype=torch.float64)
    return LayoutSample(
        seed=seed,
        positions=positions.astype(np.float64, copy=False),
        crossings=float(count_crossings(pos_tensor, edge_index, seed=seed)),
        stress=normalized_stress(
            positions,
            edge_index,
            all_pairs_distances=graph_distances,
            fit_scale=True,
        ),
        edge_length_cv=float(edge_length_cv(pos_tensor, edge_index)["edge_length_cv"]),
        ordering_inversions=_ordering_inversion_rate(
            positions,
            graph.clusters,
            graph.num_nodes,
        ),
        neighborhood_preservation=float(
            neighborhood_preservation(
                pos_tensor,
                edge_index,
                num_nodes=graph.num_nodes,
                n_samples=min(512, graph.num_nodes),
            )["neighborhood_preservation_score"]
        ),
    )


def _reference_samples(
    graph: DaguaGraph,
    seeds: Sequence[int],
    graph_distances: np.ndarray,
    timeout: float,
    steps: int,
    quality: str,
) -> List[LayoutSample]:
    """Collect seeded Cytoscape CoSE-Bilkent reference samples.

    Parameters
    ----------
    graph : DaguaGraph
        Fixture graph.
    seeds : sequence[int]
        Public seed list.
    graph_distances : numpy.ndarray
        Graph distance matrix for stress.
    timeout : float
        Per-layout timeout in seconds.
    steps : int
        CoSE-Bilkent ``numIter`` option.
    quality : str
        CoSE-Bilkent quality tier.

    Returns
    -------
    list[LayoutSample]
        Reference samples.
    """
    competitor = get_competitor("cytoscape")
    if competitor is None:
        raise RuntimeError("Cytoscape competitor is unavailable.")
    samples: List[LayoutSample] = []
    for seed in seeds:
        result = competitor.layout_with_variant(
            graph,
            seed=seed,
            timeout=timeout,
            variant_params={
                "layout": "cose-bilkent",
                "numIter": steps,
                "quality": quality,
                "randomize": True,
                "animate": False,
                "fit": False,
            },
        )
        if result.pos is None:
            raise RuntimeError(f"cytoscape cose-bilkent failed on seed {seed}: {result.error}")
        positions = result.pos.detach().cpu().numpy().astype(np.float64, copy=False)
        samples.append(_sample_metrics(seed, positions, graph, graph_distances))
    return samples


def _core_positions(
    graph: DaguaGraph,
    seed: int,
    steps: int,
    quality: str,
) -> torch.Tensor:
    """Run the shared faithful compound core for one seed.

    Parameters
    ----------
    graph : DaguaGraph
        Fixture graph.
    seed : int
        Public random seed.
    steps : int
        Maximum iteration count.
    quality : str
        CoSE-Bilkent quality tier.

    Returns
    -------
    torch.Tensor
        Positions with shape ``[N, 2]``.
    """
    return layout_cose_base_compound(
        edge_index=_edge_index(graph),
        num_nodes=graph.num_nodes,
        node_sizes=_matched_cytoscape_node_sizes(graph),
        clusters=graph.clusters,
        cluster_parents=graph.cluster_parents,
        options=CoSECompoundOptions(steps=steps, seed=seed, quality=quality, version="1.0.3"),
    )


def _pipeline_positions(
    graph: DaguaGraph,
    seed: int,
    steps: int,
    quality: str,
) -> torch.Tensor:
    """Run the public CoSE-Bilkent pipeline for one seed.

    Parameters
    ----------
    graph : DaguaGraph
        Fixture graph.
    seed : int
        Public random seed.
    steps : int
        Maximum iteration count.
    quality : str
        CoSE-Bilkent quality tier.

    Returns
    -------
    torch.Tensor
        Positions with shape ``[N, 2]``.
    """
    return layout_cose_bilkent_pipeline(
        edge_index=_edge_index(graph),
        num_nodes=graph.num_nodes,
        node_sizes=_matched_cytoscape_node_sizes(graph),
        steps=steps,
        seed=seed,
        quality=quality,
        clusters=graph.clusters,
        cluster_parents=graph.cluster_parents,
    )


def _native_samples(
    graph: DaguaGraph,
    seeds: Sequence[int],
    graph_distances: np.ndarray,
    steps: int,
    quality: str,
    runner: Callable[[DaguaGraph, int, int, str], torch.Tensor],
) -> List[LayoutSample]:
    """Collect seeded native samples.

    Parameters
    ----------
    graph : DaguaGraph
        Fixture graph.
    seeds : sequence[int]
        Public seed list.
    graph_distances : numpy.ndarray
        Graph distance matrix for stress.
    steps : int
        Maximum iteration count.
    quality : str
        CoSE-Bilkent quality tier.
    runner : Callable[[DaguaGraph, int, int, str], torch.Tensor]
        Native runner.

    Returns
    -------
    list[LayoutSample]
        Native samples.
    """
    samples: List[LayoutSample] = []
    for seed in seeds:
        positions = runner(graph, seed, steps, quality).detach().cpu().numpy()
        samples.append(
            _sample_metrics(
                seed,
                positions.astype(np.float64, copy=False),
                graph,
                graph_distances,
            )
        )
    return samples


def _sample_array(samples: Sequence[LayoutSample], name: str) -> np.ndarray:
    """Extract one metric from samples.

    Parameters
    ----------
    samples : sequence[LayoutSample]
        Layout samples.
    name : str
        Metric attribute name.

    Returns
    -------
    numpy.ndarray
        Metric values with shape ``[S]``.
    """
    return np.asarray([float(getattr(sample, name)) for sample in samples], dtype=np.float64)


def _metric_margin(values: np.ndarray, *, relative: float, floor: float) -> float:
    """Return a reference-spread-tied TOST margin.

    Parameters
    ----------
    values : numpy.ndarray
        Reference metric values.
    relative : float
        Relative margin multiplier.
    floor : float
        Absolute margin floor.

    Returns
    -------
    float
        Maximum of relative mean margin, floor, and reference sample spread.
    """
    mean = abs(float(np.mean(values))) if values.size else 0.0
    spread = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    return max(relative * mean, floor, spread)


def _tost_metric(
    name: str,
    native_values: np.ndarray,
    reference_values: np.ndarray,
    margin: float,
) -> MetricResult:
    """Run paired TOST for one quality metric.

    Parameters
    ----------
    name : str
        Metric name.
    native_values : numpy.ndarray
        Native metric values.
    reference_values : numpy.ndarray
        Reference metric values.
    margin : float
        TOST equivalence margin.

    Returns
    -------
    MetricResult
        Metric comparison summary.
    """
    tost = df.paired_tost(native_values - reference_values, margin)
    p_tost = float(tost.get("p_tost", float("nan")))
    direct = bool(tost.get("equivalent_direct", False))
    return MetricResult(
        name=name,
        native_mean=float(np.mean(native_values)),
        reference_mean=float(np.mean(reference_values)),
        native_sd=float(np.std(native_values, ddof=1)) if native_values.size > 1 else 0.0,
        reference_sd=(
            float(np.std(reference_values, ddof=1)) if reference_values.size > 1 else 0.0
        ),
        margin=margin,
        p_tost=p_tost,
        passed=direct or (math.isfinite(p_tost) and p_tost < TOST_ALPHA),
    )


def _metric_results(
    native_samples: Sequence[LayoutSample],
    reference_samples: Sequence[LayoutSample],
) -> List[MetricResult]:
    """Compute all quality-metric TOST results.

    Parameters
    ----------
    native_samples : sequence[LayoutSample]
        Native samples.
    reference_samples : sequence[LayoutSample]
        Reference samples.

    Returns
    -------
    list[MetricResult]
        Results for crossings, stress, edge-length CV, ordering, and
        neighborhood preservation.
    """
    specs = [
        ("crossings", 0.02, 0.5),
        ("stress", 0.02, 1.0e-6),
        ("edge_length_cv", 0.02, 0.02),
        ("ordering_inversions", 0.0, 0.02),
        ("neighborhood_preservation", 0.0, 0.02),
    ]
    results: List[MetricResult] = []
    for name, relative, floor in specs:
        native_values = _sample_array(native_samples, name)
        reference_values = _sample_array(reference_samples, name)
        margin = _metric_margin(reference_values, relative=relative, floor=floor)
        results.append(_tost_metric(name, native_values, reference_values, margin))
    return results


def _variance_match(native_spread: float, reference_spread: float) -> bool:
    """Return whether native and reference clouds have comparable spread.

    Parameters
    ----------
    native_spread : float
        Native mean within-cloud Procrustes distance.
    reference_spread : float
        Reference mean within-cloud Procrustes distance.

    Returns
    -------
    bool
        ``True`` when both are point masses or spread ratio is within bounds.
    """
    native_point = native_spread < POINT_MASS_THRESHOLD
    reference_point = reference_spread < POINT_MASS_THRESHOLD
    if native_point or reference_point:
        return native_point and reference_point
    ratio = native_spread / reference_spread
    return VARIANCE_RATIO_MIN <= ratio <= VARIANCE_RATIO_MAX


def _analyze_subject(
    subject: str,
    graph_name: str,
    graph: DaguaGraph,
    reference_samples: Sequence[LayoutSample],
    native_samples: Sequence[LayoutSample],
) -> GraphResult:
    """Analyze one subject against reference samples.

    Parameters
    ----------
    subject : str
        Native subject label.
    graph_name : str
        Fixture name.
    graph : DaguaGraph
        Fixture graph.
    reference_samples : sequence[LayoutSample]
        Cytoscape reference samples.
    native_samples : sequence[LayoutSample]
        Native samples.

    Returns
    -------
    GraphResult
        Distributional verdict.
    """
    rng = np.random.default_rng(8107)
    mode = df.analyze_mode_a(
        [sample.positions for sample in native_samples],
        [sample.positions for sample in reference_samples],
        rng,
    )
    native_spread = float(mode["plain_mean_W_D"])
    reference_spread = float(mode["plain_mean_W_R"])
    between = float(mode["mean_B_offdiag"])
    within_band = max(reference_spread, PROCRUSTES_BAND_FLOOR)
    return GraphResult(
        subject=subject,
        name=graph_name,
        num_nodes=int(graph.num_nodes),
        num_edges=int(_edge_index(graph).shape[1]),
        variance_match=_variance_match(native_spread, reference_spread),
        native_spread=native_spread,
        reference_spread=reference_spread,
        procrustes_between=between,
        procrustes_within_band=within_band,
        procrustes_pass=between <= within_band,
        split_equivalent=bool(mode["dist_equivalent"]),
        metrics=_metric_results(native_samples, reference_samples),
    )


def _format_metric(metric: MetricResult) -> str:
    """Format one metric result for terminal output.

    Parameters
    ----------
    metric : MetricResult
        Metric result to format.

    Returns
    -------
    str
        Compact status string.
    """
    status = "PASS" if metric.passed else "FAIL"
    return (
        f"{metric.name}={status}(N={metric.native_mean:.6g},R={metric.reference_mean:.6g},"
        f"margin={metric.margin:.6g},p={metric.p_tost:.3g})"
    )


def _print_results(results: Sequence[GraphResult]) -> None:
    """Print distributional result rows.

    Parameters
    ----------
    results : sequence[GraphResult]
        Graph results.

    Returns
    -------
    None
        Writes rows to stdout.
    """
    for result in results:
        metrics = " ".join(_format_metric(metric) for metric in result.metrics)
        print(
            f"{result.subject}/{result.name}: {result.verdict} "
            f"N={result.num_nodes} E={result.num_edges} "
            f"variance={'match' if result.variance_match else 'mismatch'} "
            f"spread_N={result.native_spread:.6g} spread_R={result.reference_spread:.6g} "
            f"procrustes_between={result.procrustes_between:.6g} "
            f"within_ref_band={result.procrustes_within_band:.6g} "
            f"procrustes={'PASS' if result.procrustes_pass else 'FAIL'} "
            f"split={'PASS' if result.split_equivalent else 'FAIL'} "
            f"{metrics}"
        )
    by_subject = sorted({result.subject for result in results})
    for subject in by_subject:
        subject_results = [result for result in results if result.subject == subject]
        passed = sum(result.verdict == "DISTRIBUTIONAL_EQUIVALENT" for result in subject_results)
        print(f"summary/{subject}: {passed}/{len(subject_results)} DISTRIBUTIONAL_EQUIVALENT")


def _seed_range(count: int) -> List[int]:
    """Return one-based seed values.

    Parameters
    ----------
    count : int
        Number of seeds.

    Returns
    -------
    list[int]
        Seed list ``[1, count]``.
    """
    if count < 2:
        raise ValueError("At least two seeds are required for distributional verification.")
    return list(range(1, count + 1))


def _subjects(
    requested: str,
) -> List[Tuple[str, Callable[[DaguaGraph, int, int, str], torch.Tensor]]]:
    """Return requested native subjects.

    Parameters
    ----------
    requested : str
        ``core``, ``pipeline``, or ``both``.

    Returns
    -------
    list[tuple[str, Callable[[DaguaGraph, int, int, str], torch.Tensor]]]
        Subject labels and runners.
    """
    if requested == "core":
        return [("core", _core_positions)]
    if requested == "pipeline":
        return [("cose_bilkent", _pipeline_positions)]
    if requested == "both":
        return [("core", _core_positions), ("cose_bilkent", _pipeline_positions)]
    raise ValueError("subject must be 'core', 'pipeline', or 'both'.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run CoSE-Bilkent distributional verification.

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
    parser.add_argument("--seeds", type=int, default=30, help="Number of one-based seeds.")
    parser.add_argument("--steps", type=int, default=2500, help="CoSE-Bilkent numIter/steps.")
    parser.add_argument("--quality", default="default", choices=("draft", "default", "proof"))
    parser.add_argument("--timeout", type=float, default=60.0, help="Per-reference timeout.")
    parser.add_argument("--subject", default="both", choices=("core", "pipeline", "both"))
    args = parser.parse_args(argv)

    seeds = _seed_range(args.seeds)
    subject_specs = _subjects(str(args.subject))
    results: List[GraphResult] = []
    for graph_name, graph in _verification_graphs():
        edge_array = _edge_index(graph).detach().cpu().numpy()
        graph_distances = df.prepare_graph_distances(edge_array, graph.num_nodes)
        reference_samples = _reference_samples(
            graph,
            seeds,
            graph_distances,
            timeout=float(args.timeout),
            steps=int(args.steps),
            quality=str(args.quality),
        )
        for subject, runner in subject_specs:
            native_samples = _native_samples(
                graph,
                seeds,
                graph_distances,
                steps=int(args.steps),
                quality=str(args.quality),
                runner=runner,
            )
            results.append(
                _analyze_subject(
                    subject,
                    graph_name,
                    graph,
                    reference_samples,
                    native_samples,
                )
            )
    _print_results(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
