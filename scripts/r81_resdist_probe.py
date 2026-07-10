"""r81 P2 -- Resistance-distance targets in the native stress core (probe).

Swaps every graph-distance source consumed by the r79 native stress pipeline
(exact Stress-SGD terms, Pivot-MDS init rows, SGD2-multi late stress term, and
the SMACOF polish APSP) from Dijkstra/BFS shortest paths to effective
resistance ("resistance distance") targets computed exactly via the Laplacian
pseudoinverse, then measures both variants with the honest benchmark ruler
(``dagua.metrics.evaluate(tier="full")`` + ``composite_auto`` with the default
``aesthetic_profile=None``) against the frozen r79 external baselines.

Method (Omega, arXiv:2512.21901): target distance ``delta_ij = sqrt(R_ij)``
where ``R_ij = Lp_ii + Lp_jj - 2 Lp_ij`` and ``Lp`` is the Moore-Penrose
pseudoinverse of the weighted graph Laplacian (conductance = 1 / edge cost).
``sqrt(R)`` is the Euclidean-embeddable metric (R itself is a squared
distance in the ``Lp^(1/2)`` embedding). Because the pipeline's size-aware
inflation adds node radii in POINTS to adjacent targets, resistance deltas are
calibrated onto the Dijkstra unit scale before entering the pipeline so the
A/B comparison holds everything but the metric constant.

This is a PROBE: it monkeypatches module-level distance functions for the
duration of one layout call and restores them afterwards. The default path is
untouched. No graphviz/reference delegation anywhere: externals are read from
the frozen r79 store.

Usage:
    .venv/bin/python scripts/r81_resdist_probe.py --graphs real_karate_34 sbm_4x30
    .venv/bin/python scripts/r81_resdist_probe.py            # full battery
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from dagua.eval.graphs import get_test_graphs, is_semantically_directed
from dagua.layout.ops import distance as distance_module
from dagua.layout.ops import native_stress as native_stress_module
from dagua.layout.ops import sgd2_multi as sgd2_module
from dagua.layout.ops import stress_sgd as stress_sgd_module
from dagua.layout.ops.graph_utils import _apsp as _true_apsp
from dagua.layout.ops.pipelines import fmmm
from dagua.layout.ops.pipelines import native_undirected as native_undirected_module
from dagua.layout.ops.pipelines.native_stress import (
    NativeStressConfig,
    layout_native_stress_pipeline,
)
from dagua.layout.ops.state import LayoutProblem
from dagua.metrics import composite_auto, evaluate

STORE = Path(__file__).resolve().parents[1] / "eval_output" / "r79_baseline"
ARTIFACTS = Path.home() / ".claude" / "research" / "dagua" / "r81-native" / "p2_artifacts"

EXTERNAL_ENGINES = (
    "graphviz_dot",
    "graphviz_sfdp",
    "graphviz_neato",
    "elk_layered",
    "dagre",
    "nx_spring",
    "igraph_kamada_kawai",
    "igraph_sugiyama",
)

# Smallest-first for early signal; brief's target list.
DEFAULT_GRAPHS = [
    "real_karate_34",
    "weighted_karate_34",
    "r79_weighted_community_4x18",
    "rgg_100",
    "small_world_100",
    "real_football_115",
    "sbm_4x30",
    "chung_lu_150",
    "protein_ppi_200",
    "sbm_5x50",
    "small_world_500",
]


def _laplacian_from_adjacency(adjacency: List[List[Tuple[int, float]]]) -> np.ndarray:
    """Build the weighted graph Laplacian with conductance = 1 / edge cost.

    Parameters
    ----------
    adjacency : list[list[tuple[int, float]]]
        Undirected adjacency list; each entry weight is a traversal COST
        (already weight-transformed upstream by ``BuildAdjacency``), so an
        edge of cost ``c`` contributes conductance ``1 / c`` -- series
        resistances add exactly like path distances.

    Returns
    -------
    numpy.ndarray
        Dense symmetric Laplacian with shape ``[N, N]``.
    """
    num_nodes = len(adjacency)
    laplacian = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    for source, neighbors in enumerate(adjacency):
        for target, cost in neighbors:
            laplacian[source, int(target)] -= 1.0 / max(float(cost), 1.0e-12)
    laplacian = 0.5 * (laplacian + laplacian.T)
    np.fill_diagonal(laplacian, 0.0)
    degrees = -laplacian.sum(axis=1)
    np.fill_diagonal(laplacian, degrees)
    return laplacian


def _edge_pairs(adjacency: List[List[Tuple[int, float]]]) -> List[Tuple[int, int, float]]:
    """Return canonical undirected edges as ``(source, target, cost)``."""
    pairs: Dict[Tuple[int, int], float] = {}
    for source, neighbors in enumerate(adjacency):
        for target, cost in neighbors:
            left, right = (source, int(target)) if source < int(target) else (int(target), source)
            if left == right:
                continue
            key = (left, right)
            if key not in pairs or float(cost) < pairs[key]:
                pairs[key] = float(cost)
    return [(left, right, cost) for (left, right), cost in sorted(pairs.items())]


def resistance_delta_matrix(
    adjacency: List[List[Tuple[int, float]]],
    mode: str,
    calibration: str,
    unit: float = 1.0,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Compute calibrated target-distance matrix for one component.

    Parameters
    ----------
    adjacency : list[list[tuple[int, float]]]
        Connected undirected adjacency list (costs as distances).
    mode : {"sqrt", "raw", "dij"}
        ``"sqrt"`` uses the Euclidean-embeddable ``sqrt(R_ij)`` (Omega);
        ``"raw"`` uses ``R_ij`` directly; ``"dij"`` uses TRUE shortest paths
        (control arm for the unit-scale experiment).
    calibration : {"edge", "apsp", "none", "edgeK"}
        ``"edge"`` rescales so the mean ADJACENT delta equals the mean
        adjacent Dijkstra target (the shipped core's unit); ``"apsp"``
        matches the all-pairs mean instead; ``"none"`` keeps raw units;
        ``"edgeK"`` rescales so the mean ADJACENT delta equals ``unit``
        (POINTS -- fixes the hop-units-vs-point-size mismatch: the shipped
        core asks non-adjacent pairs to sit 2-5 units apart while node boxes
        are ~28pt-radius, so every non-adjacent pair is a forced overlap).
    unit : float, default=1.0
        Target mean adjacent distance in points for ``"edgeK"``.

    Returns
    -------
    tuple[numpy.ndarray, dict]
        Delta matrix with shape ``[N, N]`` plus calibration diagnostics.
    """
    num_nodes = len(adjacency)
    if num_nodes <= 1:
        return np.zeros((num_nodes, num_nodes), dtype=np.float64), {"scale": 1.0}

    if mode == "dij":
        shortest = np.asarray(_true_apsp(adjacency, weighted=True), dtype=np.float64)
        finite = np.isfinite(shortest)
        fill = float(shortest[finite].max()) + 1.0 if finite.any() else 0.0
        delta = np.where(finite, shortest, fill)
    else:
        laplacian = _laplacian_from_adjacency(adjacency)
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
        tolerance = max(float(eigenvalues[-1]), 1.0) * 1.0e-9
        inverse = np.where(eigenvalues > tolerance, 1.0 / np.maximum(eigenvalues, tolerance), 0.0)
        pseudo = (eigenvectors * inverse) @ eigenvectors.T
        diagonal = np.diag(pseudo)
        resistance = np.maximum(diagonal[:, None] + diagonal[None, :] - 2.0 * pseudo, 0.0)
        delta = np.sqrt(resistance) if mode == "sqrt" else resistance.copy()
    np.fill_diagonal(delta, 0.0)

    edges = _edge_pairs(adjacency)
    edge_delta = np.array([delta[left, right] for left, right, _ in edges], dtype=np.float64)
    edge_cost = np.array([cost for _, _, cost in edges], dtype=np.float64)

    if calibration == "edge":
        scale = float(edge_cost.mean() / max(edge_delta.mean(), 1.0e-12))
    elif calibration == "edgeK":
        scale = float(unit / max(edge_delta.mean(), 1.0e-12))
    elif calibration == "apsp":
        shortest = _true_apsp(adjacency, weighted=True)
        shortest = np.asarray(shortest, dtype=np.float64)
        off_mask = ~np.eye(num_nodes, dtype=bool)
        finite = np.isfinite(shortest) & off_mask
        scale = float(shortest[finite].mean() / max(delta[off_mask].mean(), 1.0e-12))
    else:
        scale = 1.0
    delta *= scale

    floor = 1.0e-3 * max(float(edge_delta.mean() * scale), 1.0e-9)
    off_mask = ~np.eye(num_nodes, dtype=bool)
    delta[off_mask] = np.maximum(delta[off_mask], floor)

    diagnostics = {
        "scale": scale,
        "mean_adjacent_delta": float(edge_delta.mean() * scale),
        "mean_all_delta": float(delta[off_mask].mean()),
        "max_delta": float(delta.max()),
    }
    return delta, diagnostics


class ResistancePatch:
    """Context manager swapping every pipeline distance source to resistance.

    Patched module-level names (restored on exit):

    - ``dagua.layout.ops.stress_sgd._graph_distances`` (exact SGD terms)
    - ``dagua.layout.ops.distance._reference_dijkstra_distances`` /
      ``._reference_bfs_distances`` / ``._reference_all_pairs_shortest_paths``
      (Pivot-MDS init rows)
    - ``dagua.layout.ops.sgd2_multi._all_pairs_shortest_paths``
      (late multicriteria stress term)
    - ``dagua.layout.ops.native_stress._shared_all_pairs_shortest_paths``
      (SMACOF polish targets)
    """

    def __init__(self, mode: str, calibration: str, unit: float = 1.0) -> None:
        self.mode = mode
        self.calibration = calibration
        self.unit = unit
        self._cache: Dict[bytes, np.ndarray] = {}
        self.last_diagnostics: Dict[str, float] = {}
        self._saved: List[Tuple[object, str, object]] = []

    def delta(self, adjacency: List[List[Tuple[int, float]]]) -> np.ndarray:
        """Return (cached) calibrated delta matrix for one adjacency list."""
        signature = np.array(
            [
                (source, int(target), float(cost))
                for source, neighbors in enumerate(adjacency)
                for target, cost in neighbors
            ],
            dtype=np.float64,
        ).tobytes() + bytes([len(adjacency) % 251])
        if signature not in self._cache:
            matrix, diagnostics = resistance_delta_matrix(
                adjacency, self.mode, self.calibration, unit=self.unit
            )
            self._cache[signature] = matrix
            self.last_diagnostics = diagnostics
        return self._cache[signature]

    def __enter__(self) -> "ResistancePatch":
        def row(adjacency: list, source: int) -> np.ndarray:
            return self.delta(adjacency)[int(source)].copy()

        def matrix(adjacency: list, weighted: bool = False) -> np.ndarray:
            del weighted
            return self.delta(adjacency).copy()

        def sgd2_matrix(adjacency: list, device: torch.device, weighted: bool) -> torch.Tensor:
            del weighted
            return torch.tensor(self.delta(adjacency), dtype=torch.float64, device=device)

        patches = [
            (
                stress_sgd_module,
                "_graph_distances",
                lambda adjacency, source, weighted: row(adjacency, source),
            ),
            (
                distance_module,
                "_reference_dijkstra_distances",
                lambda adjacency, source: row(adjacency, source),
            ),
            (
                distance_module,
                "_reference_bfs_distances",
                lambda adjacency, source: row(adjacency, source),
            ),
            (
                distance_module,
                "_reference_all_pairs_shortest_paths",
                lambda adjacency, weighted=False: matrix(adjacency),
            ),
            (sgd2_module, "_all_pairs_shortest_paths", sgd2_matrix),
            (
                native_stress_module,
                "_shared_all_pairs_shortest_paths",
                lambda adjacency, weighted=False: matrix(adjacency),
            ),
        ]
        for module, name, replacement in patches:
            self._saved.append((module, name, getattr(module, name)))
            setattr(module, name, replacement)
        return self

    def __exit__(self, *exc_info: object) -> None:
        for module, name, original in reversed(self._saved):
            setattr(module, name, original)
        self._saved.clear()


def prism_equiv(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: Optional[torch.Tensor],
) -> torch.Tensor:
    """Apply dagua's PRISM-equivalent overlap finisher (points in, points out)."""
    inches = positions.detach().to(dtype=torch.float64) / 72.0
    out = fmmm._graphviz_fdp_prism_overlap(
        positions=inches,
        edge_index=edge_index,
        node_sizes=node_sizes,
    )
    return (out * 72.0).to(dtype=torch.float32)


def score(test_graph, positions: torch.Tensor) -> Tuple[float, Dict[str, float]]:
    """Score positions with the honest ruler; return composite + key metrics."""
    metrics = evaluate(
        test_graph.graph, positions.detach().cpu().to(dtype=torch.float32), tier="full"
    )
    composite = float(composite_auto(metrics, is_semantically_directed(test_graph)))
    keys = ("overlap_count", "edge_length_cv", "crossing_rate", "angular_res_mean_deg")
    return composite, {
        key: (float(metrics[key]) if metrics.get(key) is not None else None)
        for key in keys
        if key in metrics
    }


def load_store_rows() -> Dict[Tuple[str, str], dict]:
    """Load the frozen r79 baseline rows keyed by (graph, engine)."""
    rows: Dict[Tuple[str, str], dict] = {}
    with (STORE / "results.rows.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            rows[(row["graph"], row["engine"])] = row
    return rows


def run_variant(
    test_graph,
    label: str,
    patch: Optional[ResistancePatch],
    weight_transform: str,
    use_weights: bool,
    target_unit: str = "hops",
) -> dict:
    """Run one native-stress variant and score raw + prism-finished output."""
    graph = test_graph.graph
    edge_weights = getattr(graph, "edge_weights", None)
    weights = edge_weights.cpu() if (edge_weights is not None and use_weights) else None
    config = NativeStressConfig(seed=42, weight_transform=weight_transform, target_unit=target_unit)
    started = time.perf_counter()
    if patch is not None:
        with patch:
            positions = layout_native_stress_pipeline(
                edge_index=graph.edge_index.cpu(),
                num_nodes=graph.num_nodes,
                node_sizes=graph.node_sizes.cpu(),
                edge_weights=weights,
                seed=42,
                config=config,
            ).cpu()
        diagnostics = dict(patch.last_diagnostics)
    else:
        positions = layout_native_stress_pipeline(
            edge_index=graph.edge_index.cpu(),
            num_nodes=graph.num_nodes,
            node_sizes=graph.node_sizes.cpu(),
            edge_weights=weights,
            seed=42,
            config=config,
        ).cpu()
        diagnostics = {}
    solve_seconds = time.perf_counter() - started

    raw_composite, raw_metrics = score(test_graph, positions)
    finished = prism_equiv(positions, graph.edge_index.cpu(), graph.node_sizes.cpu())
    prism_composite, prism_metrics = score(test_graph, finished)

    # The portfolio contest's other cleanup variant (convergent exact
    # projection) -- read-only call so the binary 20-pt overlap cliff on a
    # 1-residual prism finish does not confound the A/B verdict; the contest
    # itself takes an argmax over cleanup variants.
    problem = LayoutProblem(
        edge_index=graph.edge_index.cpu(),
        num_nodes=graph.num_nodes,
        node_sizes=graph.node_sizes.cpu(),
        edge_weights=edge_weights.cpu() if edge_weights is not None else None,
        seed=42,
    )
    converged = native_undirected_module._project_candidate(positions, problem, convergent=True)
    conv_composite, conv_metrics = score(test_graph, converged)

    best_composite = max(raw_composite, prism_composite, conv_composite)
    return {
        "variant": label,
        "solve_seconds": round(solve_seconds, 2),
        "raw_composite": round(raw_composite, 3),
        "raw_metrics": raw_metrics,
        "prism_composite": round(prism_composite, 3),
        "prism_metrics": prism_metrics,
        "conv_composite": round(conv_composite, 3),
        "conv_metrics": conv_metrics,
        "best_composite": round(best_composite, 3),
        "target_diagnostics": diagnostics,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", nargs="+", default=None)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["base", "res_sqrt_edge", "res_sqrt_apsp", "res_raw_edge"],
        help="Variant labels to run (base, res_{sqrt|raw}_{edge|apsp|none}, "
        "plus _invw / _now weighted-semantics suffixes).",
    )
    parser.add_argument("--out", default=str(ARTIFACTS / "resdist_probe.jsonl"))
    args = parser.parse_args(argv)

    graphs = {t.name: t for t in get_test_graphs(max_nodes=500)}
    names = args.graphs or DEFAULT_GRAPHS
    store = load_store_rows()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for name in names:
        test_graph = graphs[name]
        test_graph.graph.compute_node_sizes()
        edge_weights = getattr(test_graph.graph, "edge_weights", None)
        externals = {
            engine: store.get((name, engine), {}).get("composite") for engine in EXTERNAL_ENGINES
        }
        externals = {k: v for k, v in externals.items() if v is not None}
        best_external = max(externals.items(), key=lambda kv: kv[1]) if externals else (None, None)
        frozen_dagua = store.get((name, "dagua"), {}).get("composite")

        print(
            f"\n=== {name} (n={test_graph.graph.num_nodes}, "
            f"weights={'yes' if edge_weights is not None else 'no'}) "
            f"frozen_dagua={frozen_dagua and round(frozen_dagua, 2)} "
            f"best_ext={best_external[0]}={best_external[1] and round(best_external[1], 2)}",
            flush=True,
        )

        # Unit length K for edgeK calibration: mean adjacent radii sum in
        # points (the smallest center distance at which two mean nodes just
        # avoid overlap), computed on the FULL graph.
        sizes = test_graph.graph.node_sizes.detach().cpu().to(dtype=torch.float64).numpy()
        radii = 0.5 * np.sqrt((sizes**2).sum(axis=1))
        edge_array = test_graph.graph.edge_index.detach().cpu().numpy()
        nonself = edge_array[0] != edge_array[1]
        unit_k = (
            float((radii[edge_array[0][nonself]] + radii[edge_array[1][nonself]]).mean())
            if nonself.any()
            else 1.0
        )
        print(f"  unit K (mean adjacent radii sum) = {unit_k:.1f} pt", flush=True)

        variant_plans: List[Tuple[str, Optional[ResistancePatch], str, bool, str]] = []
        for label in args.variants:
            body = label
            weight_transform = "none"
            use_weights = True
            if body.endswith("_invw"):
                if edge_weights is None:
                    continue
                weight_transform = "inverse"
                body = body[: -len("_invw")]
            elif body.endswith("_now"):
                if edge_weights is None:
                    continue
                use_weights = False
                body = body[: -len("_now")]
            if body == "base":
                variant_plans.append((label, None, weight_transform, use_weights, "hops"))
                continue
            if body == "op_points":
                # Production path: NativeStressConfig(target_unit="points"),
                # no monkeypatching anywhere.
                variant_plans.append((label, None, weight_transform, use_weights, "points"))
                continue
            parts = body.split("_")
            if parts[0] == "res" and len(parts) == 3:
                mode, calibration = parts[1], parts[2]
            elif parts[0] == "dij" and len(parts) == 2:
                mode, calibration = "dij", parts[1]
            else:
                raise SystemExit(f"unknown variant label: {label}")
            variant_plans.append(
                (
                    label,
                    ResistancePatch(mode=mode, calibration=calibration, unit=unit_k),
                    weight_transform,
                    use_weights,
                    "hops",
                )
            )

        for label, patch, weight_transform, use_weights, target_unit in variant_plans:
            result = run_variant(
                test_graph,
                label,
                patch,
                weight_transform,
                use_weights,
                target_unit=target_unit,
            )
            result.update(
                {
                    "graph": name,
                    "nodes": test_graph.graph.num_nodes,
                    "frozen_dagua": frozen_dagua,
                    "best_external_engine": best_external[0],
                    "best_external": best_external[1],
                }
            )
            with out_path.open("a") as handle:
                handle.write(json.dumps(result, sort_keys=True) + "\n")
            delta_vs_ext = (
                result["best_composite"] - best_external[1]
                if best_external[1] is not None
                else float("nan")
            )
            prism_ov = result["prism_metrics"].get("overlap_count")
            prism_cv = result["prism_metrics"].get("edge_length_cv")
            prism_cross = result["prism_metrics"].get("crossing_rate")
            print(
                f"  {label:22s} raw={result['raw_composite']:7.3f} "
                f"prism={result['prism_composite']:7.3f} "
                f"conv={result['conv_composite']:7.3f} "
                f"best={result['best_composite']:7.3f} "
                f"dVSext={delta_vs_ext:+7.2f} "
                f"ov={prism_ov if prism_ov is None else int(prism_ov)} "
                f"cv={prism_cv if prism_cv is None else round(prism_cv, 3)} "
                f"cross={prism_cross if prism_cross is None else round(prism_cross, 3)} "
                f"[{result['solve_seconds']:.1f}s]",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
