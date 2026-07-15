"""Verify ELK Layered distributional fidelity against elkjs across seeds."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dagua.eval import distributional_fidelity as df  # noqa: E402
from dagua.eval.competitors.elk_competitor import ElkLayered  # noqa: E402
from dagua.eval.equivalence_metrics import normalized_stress  # noqa: E402
from dagua.layout.ops.elk import ELK_LAYERS_KEY, ELK_ORDER_KEY  # noqa: E402
from dagua.layout.ops.pipelines.elk import build_elk_pipeline, layout_elk_pipeline  # noqa: E402
from dagua.layout.ops.state import (  # noqa: E402
    ExecutionPlan,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)
from dagua.metrics import count_crossings, edge_length_cv  # noqa: E402
from scripts.verify_elk_fidelity import (  # noqa: E402
    _ensure_elkjs_node_path,
    _reference_layer_indices,
    _reference_order,
    _verification_graphs,
)

DEFAULT_REPORT = ROOT / "docs" / "algorithms" / "elk_fidelity.md"
DISTRIBUTIONAL_MARKER = "## Distributional verification"
PROCRUSTES_BAND_FLOOR = 1.0e-3
VARIANCE_RATIO_MIN = 0.5
VARIANCE_RATIO_MAX = 2.0
POINT_MASS_THRESHOLD = 1.0e-6
TOST_ALPHA = 0.05


@dataclass(frozen=True)
class LayoutSample:
    """One seeded layout and structural metadata."""

    seed: int
    positions: np.ndarray
    layer_indices: List[int]
    order: Dict[int, int]
    crossings: float
    stress: float
    edge_length_cv: float
    ordering_inversions: float


@dataclass(frozen=True)
class MetricResult:
    """TOST result for one scalar distributional metric."""

    name: str
    dagua_mean: float
    reference_mean: float
    dagua_sd: float
    reference_sd: float
    margin: float
    p_tost: float
    passed: bool


@dataclass(frozen=True)
class GraphResult:
    """Distributional verdict for one graph."""

    name: str
    num_nodes: int
    num_edges: int
    layer_exact: bool
    variance_match: bool
    dagua_spread: float
    reference_spread: float
    procrustes_between: float
    procrustes_within_band: float
    procrustes_pass: bool
    dist_equivalent: bool
    metrics: List[MetricResult]

    @property
    def verdict(self) -> str:
        """Return the graph-level distributional verdict.

        Returns
        -------
        str
            ``DISTRIBUTIONAL_EQUIVALENT`` when all conservative gates pass,
            otherwise ``not_distributional_equivalent``.
        """
        metric_pass = all(metric.passed for metric in self.metrics)
        if self.layer_exact and self.variance_match and self.procrustes_pass and metric_pass:
            return "DISTRIBUTIONAL_EQUIVALENT"
        return "not_distributional_equivalent"


def _edge_index(edges: Sequence[Sequence[int]]) -> torch.Tensor:
    """Convert edge pairs to an edge-index tensor.

    Parameters
    ----------
    edges : sequence[sequence[int]]
        Directed edge pairs.

    Returns
    -------
    torch.Tensor
        Long tensor with shape ``[2, E]``.
    """
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def _torch_positions(positions: np.ndarray) -> torch.Tensor:
    """Return positions as a CPU float64 tensor.

    Parameters
    ----------
    positions : numpy.ndarray
        Position array with shape ``[N, 2]``.

    Returns
    -------
    torch.Tensor
        Float64 tensor with shape ``[N, 2]``.
    """
    return torch.as_tensor(positions, dtype=torch.float64)


def _stage_metadata(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: torch.Tensor,
    seed: int,
) -> Tuple[List[int], Dict[int, int]]:
    """Run native ELK stages and return layer/order metadata.

    Parameters
    ----------
    edge_index : torch.Tensor
        Directed graph edges with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    node_sizes : torch.Tensor
        Node sizes with shape ``[N, 2]``.
    seed : int
        Public ELK random seed.

    Returns
    -------
    tuple[list[int], dict[int, int]]
        Layer index per node and within-layer order map.
    """
    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        seed=seed,
    )
    state = build_elk_pipeline(random_seed=seed).apply(
        problem,
        SolveState(),
        RuntimeContext(plan=ExecutionPlan(device="cpu")),
    )
    layer_indices = [0] * num_nodes
    for layer_index, layer in enumerate(state.extras[ELK_LAYERS_KEY]):
        for node in layer:
            layer_indices[int(node)] = layer_index
    return layer_indices, dict(state.extras[ELK_ORDER_KEY])


def _ordering_inversion_rate(layer_indices: Sequence[int], order: Dict[int, int]) -> float:
    """Measure a per-layer permutation as a normalized inversion rate.

    Parameters
    ----------
    layer_indices : sequence[int]
        Layer index per node.
    order : dict[int, int]
        Node id to within-layer order.

    Returns
    -------
    float
        Inversions relative to model order divided by possible same-layer
        pairs. The value is in ``[0, 1]`` when at least one pair exists.
    """
    by_layer: Dict[int, List[int]] = {}
    for node, layer in enumerate(layer_indices):
        by_layer.setdefault(int(layer), []).append(node)
    inversions = 0
    possible = 0
    for nodes in by_layer.values():
        ordered_nodes = sorted(nodes, key=lambda node: order.get(node, 0))
        possible += len(ordered_nodes) * (len(ordered_nodes) - 1) // 2
        for left_index, left_node in enumerate(ordered_nodes):
            for right_node in ordered_nodes[left_index + 1 :]:
                if left_node > right_node:
                    inversions += 1
    if possible == 0:
        return 0.0
    return inversions / possible


def _sample_metrics(
    seed: int,
    positions: np.ndarray,
    edge_index: torch.Tensor,
    layer_indices: List[int],
    order: Dict[int, int],
    graph_distances: np.ndarray,
) -> LayoutSample:
    """Compute scalar metrics for one seeded layout.

    Parameters
    ----------
    seed : int
        Public random seed.
    positions : numpy.ndarray
        Layout coordinates with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge index with shape ``[2, E]``.
    layer_indices : list[int]
        Layer index per node.
    order : dict[int, int]
        Node id to within-layer order.
    graph_distances : numpy.ndarray
        All-pairs graph distances with shape ``[N, N]``.

    Returns
    -------
    LayoutSample
        Layout sample and metric values.
    """
    pos_tensor = _torch_positions(positions)
    cv = edge_length_cv(pos_tensor, edge_index)["edge_length_cv"]
    return LayoutSample(
        seed=seed,
        positions=positions.astype(np.float64, copy=False),
        layer_indices=layer_indices,
        order=order,
        crossings=float(count_crossings(pos_tensor, edge_index, seed=seed)),
        stress=normalized_stress(
            positions,
            edge_index,
            all_pairs_distances=graph_distances,
            fit_scale=True,
        ),
        edge_length_cv=float(cv),
        ordering_inversions=_ordering_inversion_rate(layer_indices, order),
    )


def _elkjs_samples(
    graph: Any,
    seeds: Sequence[int],
    edge_index: torch.Tensor,
    graph_distances: np.ndarray,
    timeout: float,
) -> List[LayoutSample]:
    """Collect seeded elkjs samples for one graph.

    Parameters
    ----------
    graph : Any
        DaguaGraph consumed by the competitor adapter.
    seeds : sequence[int]
        Public random seeds.
    edge_index : torch.Tensor
        Edge index with shape ``[2, E]``.
    graph_distances : numpy.ndarray
        All-pairs graph distances with shape ``[N, N]``.
    timeout : float
        Per-layout timeout in seconds.

    Returns
    -------
    list[LayoutSample]
        Reference layout samples.

    Raises
    ------
    RuntimeError
        If elkjs is unavailable or fails for any seed.
    """
    competitor = ElkLayered()
    if not competitor.available():
        raise RuntimeError("elkjs reference is unavailable; install local npm package 'elkjs'.")
    samples: List[LayoutSample] = []
    for seed in seeds:
        result = competitor.layout(graph, timeout=timeout, seed=seed)
        if result.pos is None:
            raise RuntimeError(f"elkjs failed on seed {seed}: {result.error}")
        positions = result.pos.detach().cpu().numpy().astype(np.float64, copy=False)
        pos_tensor = torch.as_tensor(positions, dtype=torch.float64)
        layer_indices = _reference_layer_indices(pos_tensor)
        order = _reference_order(pos_tensor)
        samples.append(
            _sample_metrics(seed, positions, edge_index, layer_indices, order, graph_distances)
        )
    return samples


def _native_samples(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: torch.Tensor,
    seeds: Sequence[int],
    graph_distances: np.ndarray,
) -> List[LayoutSample]:
    """Collect seeded native ELK samples for one graph.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge index with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    node_sizes : torch.Tensor
        Node sizes with shape ``[N, 2]``.
    seeds : sequence[int]
        Public random seeds.
    graph_distances : numpy.ndarray
        All-pairs graph distances with shape ``[N, N]``.

    Returns
    -------
    list[LayoutSample]
        Native layout samples.
    """
    samples: List[LayoutSample] = []
    for seed in seeds:
        positions = layout_elk_pipeline(
            edge_index,
            num_nodes,
            node_sizes,
            seed=seed,
            random_seed=seed,
        )
        layer_indices, order = _stage_metadata(edge_index, num_nodes, node_sizes, seed)
        samples.append(
            _sample_metrics(
                seed,
                positions.detach().cpu().numpy().astype(np.float64, copy=False),
                edge_index,
                layer_indices,
                order,
                graph_distances,
            )
        )
    return samples


def _sample_array(samples: Sequence[LayoutSample], name: str) -> np.ndarray:
    """Extract one metric array from layout samples.

    Parameters
    ----------
    samples : sequence[LayoutSample]
        Layout samples.
    name : str
        Metric attribute name.

    Returns
    -------
    numpy.ndarray
        Float64 values with shape ``[S]``.
    """
    return np.asarray([float(getattr(sample, name)) for sample in samples], dtype=np.float64)


def _metric_margin(values: np.ndarray, *, relative: float, floor: float) -> float:
    """Return a variance-tied TOST equivalence margin.

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
        Maximum of the relative margin, floor, and reference self-spread.
    """
    mean = abs(float(np.mean(values))) if values.size else 0.0
    spread = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    return max(relative * mean, floor, spread)


def _tost_metric(
    name: str,
    dagua_values: np.ndarray,
    reference_values: np.ndarray,
    margin: float,
) -> MetricResult:
    """Run paired TOST for one metric.

    Parameters
    ----------
    name : str
        Metric label.
    dagua_values : numpy.ndarray
        Native metric values.
    reference_values : numpy.ndarray
        elkjs metric values.
    margin : float
        Equivalence margin.

    Returns
    -------
    MetricResult
        TOST summary.
    """
    tost = df.paired_tost(dagua_values - reference_values, margin)
    p_tost = float(tost.get("p_tost", float("nan")))
    direct = bool(tost.get("equivalent_direct", False))
    passed = direct or (math.isfinite(p_tost) and p_tost < TOST_ALPHA)
    return MetricResult(
        name=name,
        dagua_mean=float(np.mean(dagua_values)),
        reference_mean=float(np.mean(reference_values)),
        dagua_sd=float(np.std(dagua_values, ddof=1)) if dagua_values.size > 1 else 0.0,
        reference_sd=float(np.std(reference_values, ddof=1)) if reference_values.size > 1 else 0.0,
        margin=margin,
        p_tost=p_tost,
        passed=passed,
    )


def _metric_results(
    dagua_samples: Sequence[LayoutSample],
    reference_samples: Sequence[LayoutSample],
) -> List[MetricResult]:
    """Compute all scalar metric TOST results.

    Parameters
    ----------
    dagua_samples : sequence[LayoutSample]
        Native samples.
    reference_samples : sequence[LayoutSample]
        elkjs samples.

    Returns
    -------
    list[MetricResult]
        Results for crossings, stress, edge-length CV, and ordering.
    """
    specs = [
        ("crossings", 0.02, 0.5),
        ("stress", 0.02, 1.0e-6),
        ("edge_length_cv", 0.02, 0.02),
        ("ordering_inversions", 0.0, 0.02),
    ]
    results: List[MetricResult] = []
    for name, relative, floor in specs:
        dagua_values = _sample_array(dagua_samples, name)
        reference_values = _sample_array(reference_samples, name)
        margin = _metric_margin(reference_values, relative=relative, floor=floor)
        results.append(_tost_metric(name, dagua_values, reference_values, margin))
    return results


def _variance_match(dagua_spread: float, reference_spread: float) -> bool:
    """Return whether layout-cloud spreads match without one-sided collapse.

    Parameters
    ----------
    dagua_spread : float
        Native mean within-cloud Procrustes distance.
    reference_spread : float
        elkjs mean within-cloud Procrustes distance.

    Returns
    -------
    bool
        ``True`` when both are point masses or both have comparable spread.
    """
    dagua_point = dagua_spread < POINT_MASS_THRESHOLD
    reference_point = reference_spread < POINT_MASS_THRESHOLD
    if dagua_point or reference_point:
        return dagua_point and reference_point
    ratio = dagua_spread / reference_spread
    return VARIANCE_RATIO_MIN <= ratio <= VARIANCE_RATIO_MAX


def _analyze_graph(
    name: str,
    graph: Any,
    seeds: Sequence[int],
    timeout: float,
) -> GraphResult:
    """Run full distributional analysis for one graph.

    Parameters
    ----------
    name : str
        Graph name.
    graph : Any
        DaguaGraph verification graph.
    seeds : sequence[int]
        Public random seeds.
    timeout : float
        Per-elkjs-layout timeout in seconds.

    Returns
    -------
    GraphResult
        Per-graph verdict and diagnostics.
    """
    edge_index = graph.edge_index.detach().cpu().to(dtype=torch.long)
    node_sizes = graph.node_sizes.detach().cpu().to(dtype=torch.float64)
    edge_array = edge_index.detach().cpu().numpy()
    graph_distances = df.prepare_graph_distances(edge_array, graph.num_nodes)
    reference_samples = _elkjs_samples(graph, seeds, edge_index, graph_distances, timeout)
    dagua_samples = _native_samples(edge_index, graph.num_nodes, node_sizes, seeds, graph_distances)
    layer_exact = all(
        dagua.layer_indices == reference.layer_indices
        for dagua, reference in zip(dagua_samples, reference_samples)
    )
    rng = np.random.default_rng(9103)
    mode = df.analyze_mode_a(
        [sample.positions for sample in dagua_samples],
        [sample.positions for sample in reference_samples],
        rng,
    )
    dagua_spread = float(mode["plain_mean_W_D"])
    reference_spread = float(mode["plain_mean_W_R"])
    between = float(mode["mean_B_offdiag"])
    within_band = max(dagua_spread, reference_spread, PROCRUSTES_BAND_FLOOR)
    return GraphResult(
        name=name,
        num_nodes=int(graph.num_nodes),
        num_edges=int(edge_index.shape[1]),
        layer_exact=layer_exact,
        variance_match=_variance_match(dagua_spread, reference_spread),
        dagua_spread=dagua_spread,
        reference_spread=reference_spread,
        procrustes_between=between,
        procrustes_within_band=within_band,
        procrustes_pass=between <= within_band,
        dist_equivalent=bool(mode["dist_equivalent"]),
        metrics=_metric_results(dagua_samples, reference_samples),
    )


def _format_metric(metric: MetricResult) -> str:
    """Format one metric for terminal output.

    Parameters
    ----------
    metric : MetricResult
        Metric result to format.

    Returns
    -------
    str
        Compact metric status string.
    """
    status = "PASS" if metric.passed else "FAIL"
    return (
        f"{metric.name}={status}(D={metric.dagua_mean:.6g},R={metric.reference_mean:.6g},"
        f"margin={metric.margin:.6g},p={metric.p_tost:.3g})"
    )


def _print_results(results: Sequence[GraphResult]) -> None:
    """Print per-graph distributional verdicts.

    Parameters
    ----------
    results : sequence[GraphResult]
        Graph results to print.

    Returns
    -------
    None
        Writes to stdout.
    """
    for result in results:
        metrics = " ".join(_format_metric(metric) for metric in result.metrics)
        print(
            f"{result.name}: {result.verdict} "
            f"layers={'Y' if result.layer_exact else 'N'} "
            f"variance={'match' if result.variance_match else 'mismatch'} "
            f"spread_D={result.dagua_spread:.6g} spread_R={result.reference_spread:.6g} "
            f"procrustes_between={result.procrustes_between:.6g} "
            f"within_band={result.procrustes_within_band:.6g} "
            f"procrustes={'PASS' if result.procrustes_pass else 'FAIL'} "
            f"split={'PASS' if result.dist_equivalent else 'FAIL'} "
            f"{metrics}"
        )
    passed = sum(result.verdict == "DISTRIBUTIONAL_EQUIVALENT" for result in results)
    print(f"summary: {passed}/{len(results)} DISTRIBUTIONAL_EQUIVALENT")


def _report_table(results: Sequence[GraphResult]) -> List[str]:
    """Build the Markdown distributional result table.

    Parameters
    ----------
    results : sequence[GraphResult]
        Graph results.

    Returns
    -------
    list[str]
        Markdown table lines.
    """
    lines = [
        DISTRIBUTIONAL_MARKER,
        "",
        "Reference: elkjs with `elk.randomSeed=1..30`; native ELK uses the same seed set, "
        "Java-compatible restart shuffles, and `thoroughness=7`.",
        "",
        "| graph | N | E | layers | verdict | variance | procrustes | metric TOST failures |",
        "|---|---:|---:|---|---|---|---|---|",
    ]
    for result in results:
        failures = [metric.name for metric in result.metrics if not metric.passed]
        variance = (
            f"match (D {result.dagua_spread:.4g}, R {result.reference_spread:.4g})"
            if result.variance_match
            else f"mismatch (D {result.dagua_spread:.4g}, R {result.reference_spread:.4g})"
        )
        procrustes = (
            f"PASS (between {result.procrustes_between:.4g} <= band "
            f"{result.procrustes_within_band:.4g})"
            if result.procrustes_pass
            else f"FAIL (between {result.procrustes_between:.4g} > band "
            f"{result.procrustes_within_band:.4g})"
        )
        lines.append(
            "| {name} | {num_nodes} | {num_edges} | {layers} | {verdict} | "
            "{variance} | {procrustes} | {failures} |".format(
                name=result.name,
                num_nodes=result.num_nodes,
                num_edges=result.num_edges,
                layers="Y" if result.layer_exact else "N",
                verdict=result.verdict,
                variance=variance,
                procrustes=procrustes,
                failures=", ".join(failures) if failures else "",
            )
        )
    lines.extend(
        [
            "",
            "Interpretation: remaining per-seed mismatches first diverge at "
            "`AbstractBarycenterPortDistributor.distributePortsWhileSweeping` generated-port "
            "rank/order feedback, after the Java RNG stream has been matched through "
            "`LayerSweepCrossingMinimizer.initialize`, `ISweepPortDistributor.create`, "
            "`LayerSweepCrossingMinimizer.compareDifferentRandomizedLayouts`, and "
            "`BarycenterHeuristic` randomization. "
            "The distributional tier is earned only where layers stay exact, scalar TOSTs pass, "
            "cross-seed variance is comparable, and the elkjs-vs-native Procrustes cloud sits "
            "inside the within-elkjs/native spread band.",
            "",
        ]
    )
    return lines


def _write_report(path: Path, results: Sequence[GraphResult]) -> None:
    """Update the distributional section of the ELK fidelity report.

    Parameters
    ----------
    path : pathlib.Path
        Markdown report path.
    results : sequence[GraphResult]
        Results to write.

    Returns
    -------
    None
        The report is written in place.
    """
    existing = path.read_text() if path.exists() else "# ELK Layered fidelity verification\n"
    prefix = existing.split(DISTRIBUTIONAL_MARKER, 1)[0].rstrip()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(prefix + "\n\n" + "\n".join(_report_table(results)))


def _seed_range(count: int) -> List[int]:
    """Return ELK seed values ``1..count``.

    Parameters
    ----------
    count : int
        Number of seeds.

    Returns
    -------
    list[int]
        One-based seed list.
    """
    if count < 2:
        raise ValueError("At least two seeds are required for distributional verification.")
    return list(range(1, count + 1))


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run ELK distributional verification.

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
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Do not update the Markdown report.",
    )
    args = parser.parse_args(argv)

    _ensure_elkjs_node_path()
    seeds = _seed_range(args.seeds)
    results = [
        _analyze_graph(name, graph, seeds, timeout=float(args.timeout))
        for name, graph in _verification_graphs()
    ]
    _print_results(results)
    if not args.no_report:
        _write_report(args.report, results)
        print(f"report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
