"""Round 41 dagua_native self-reference reproducibility smoke harness."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dagua.config import LayoutConfig  # noqa: E402
from dagua.graph import DaguaGraph  # noqa: E402
from dagua.layout import layout  # noqa: E402
from dagua.layout.ops.pipelines.dagua_native import layout_dagua_native_pipeline  # noqa: E402
from scripts.algo_fidelity_cross import fidelity_procrustes  # noqa: E402

SEEDS: tuple[int, ...] = (42, 43, 44)
DEVICE = "cpu"


@dataclass(frozen=True)
class SmokeCase:
    """One topology case for the self-reference smoke.

    Parameters
    ----------
    name : str
        Case name printed in the results table.
    num_nodes : int
        Number of nodes in the graph.
    edges : tuple[tuple[int, int], ...]
        Directed edge list in graph-file order.
    clusters : dict[str, tuple[int, ...]], optional
        Optional cluster metadata for the reference adapter path.
    """

    name: str
    num_nodes: int
    edges: tuple[tuple[int, int], ...]
    clusters: Optional[dict[str, tuple[int, ...]]] = None


@dataclass(frozen=True)
class SmokeRow:
    """One smoke comparison row.

    Parameters
    ----------
    topology : str
        Topology case name.
    seed : int
        Seed forwarded to both layout paths.
    pipeline : str
        Native sub-pipeline selected by config preparation.
    reference_rmsd : float
        Procrustes RMSD between direct pipeline and adapter reference.
    repeat_rmsd : float
        Procrustes RMSD between two direct fixed-seed pipeline runs.
    max_abs_delta : float
        Maximum raw coordinate delta between direct and adapter positions.
    """

    topology: str
    seed: int
    pipeline: str
    reference_rmsd: float
    repeat_rmsd: float
    max_abs_delta: float


def _path_edges(num_nodes: int) -> tuple[tuple[int, int], ...]:
    """Return a directed path edge list.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the path.

    Returns
    -------
    tuple[tuple[int, int], ...]
        Directed path edges in ascending node order.
    """
    return tuple((idx, idx + 1) for idx in range(num_nodes - 1))


def _star_edges(num_nodes: int) -> tuple[tuple[int, int], ...]:
    """Return a directed out-star edge list.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the star.

    Returns
    -------
    tuple[tuple[int, int], ...]
        Edges from node zero to each leaf.
    """
    return tuple((0, idx) for idx in range(1, num_nodes))


def _clustered_edges() -> tuple[tuple[int, int], ...]:
    """Return a two-community clustered DAG edge list.

    Returns
    -------
    tuple[tuple[int, int], ...]
        Dense within-cluster edges plus a small inter-cluster bridge set.
    """
    edges: list[tuple[int, int]] = []
    for base in (0, 5):
        for src in range(base, base + 5):
            for dst in range(src + 1, base + 5):
                edges.append((src, dst))
    edges.extend(((2, 7), (3, 8)))
    return tuple(edges)


def _grid_edges(width: int, height: int) -> tuple[tuple[int, int], ...]:
    """Return a directed right/down grid edge list.

    Parameters
    ----------
    width : int
        Grid width in nodes.
    height : int
        Grid height in nodes.

    Returns
    -------
    tuple[tuple[int, int], ...]
        Directed grid edges in row-major order.
    """
    edges: list[tuple[int, int]] = []
    for row in range(height):
        for col in range(width):
            node = row * width + col
            if col + 1 < width:
                edges.append((node, node + 1))
            if row + 1 < height:
                edges.append((node, node + width))
    return tuple(edges)


def _build_cases() -> tuple[SmokeCase, ...]:
    """Build the four requested smoke topologies.

    Returns
    -------
    tuple[SmokeCase, ...]
        Path, star, clustered, and grid smoke cases.
    """
    return (
        SmokeCase(name="path", num_nodes=12, edges=_path_edges(12)),
        SmokeCase(name="star", num_nodes=12, edges=_star_edges(12)),
        SmokeCase(
            name="clustered",
            num_nodes=10,
            edges=_clustered_edges(),
            clusters={"left": tuple(range(5)), "right": tuple(range(5, 10))},
        ),
        SmokeCase(name="grid", num_nodes=16, edges=_grid_edges(4, 4)),
    )


def _make_graph(case: SmokeCase) -> DaguaGraph:
    """Materialize a ``DaguaGraph`` for one smoke case.

    Parameters
    ----------
    case : SmokeCase
        Topology case to materialize.

    Returns
    -------
    DaguaGraph
        Graph with nodes, edges, clusters, and computed node sizes.
    """
    graph = DaguaGraph()
    for node in range(case.num_nodes):
        graph.add_node(node)
    for src, dst in case.edges:
        graph.add_edge(src, dst)
    if case.clusters is not None:
        graph.clusters.update({name: list(nodes) for name, nodes in case.clusters.items()})
    graph.compute_node_sizes()
    return graph


def _direct_pipeline(
    graph: DaguaGraph,
    seed: int,
    config_mutator: Optional[Callable[[LayoutConfig], None]] = None,
) -> torch.Tensor:
    """Run the direct dagua_native pipeline for one graph.

    Parameters
    ----------
    graph : DaguaGraph
        Prepared graph with computed node sizes.
    seed : int
        Seed forwarded to the pipeline and config.
    config_mutator : Callable[[LayoutConfig], None], optional
        Optional in-place config mutation used by the component diagnosis.

    Returns
    -------
    torch.Tensor
        CPU position tensor with shape ``[N, 2]``.
    """
    config = LayoutConfig(
        algorithm="dagua_native",
        device=DEVICE,
        seed=seed,
    )
    if config_mutator is not None:
        config_mutator(config)
    return (
        layout_dagua_native_pipeline(
            edge_index=graph.edge_index,
            num_nodes=graph.num_nodes,
            node_sizes=graph.node_sizes,
            config=config,
            seed=seed,
            clusters=graph.clusters,
            cluster_parents=graph.cluster_parents,
            fidelity_mode="none",
        )
        .detach()
        .cpu()
    )


def _reference_adapter(graph: DaguaGraph, seed: int) -> torch.Tensor:
    """Run the Dagua adapter reference path for one graph.

    Parameters
    ----------
    graph : DaguaGraph
        Prepared graph with computed node sizes.
    seed : int
        Seed forwarded through ``LayoutConfig``.

    Returns
    -------
    torch.Tensor
        CPU position tensor with shape ``[N, 2]``.
    """
    config = LayoutConfig(algorithm=None, device=DEVICE, seed=seed)
    return layout(graph, config).detach().cpu()


def _infer_pipeline(case: SmokeCase) -> str:
    """Return the expected topology-selected sub-pipeline.

    Parameters
    ----------
    case : SmokeCase
        Smoke case to classify for reporting.

    Returns
    -------
    str
        Conservative diagnosis label for the selected native path.
    """
    if case.name in {"path", "star"}:
        return "tree"
    if case.name == "clustered":
        return "clustered-layered_dag"
    return "layered_dag"


def _compare_positions(left: torch.Tensor, right: torch.Tensor) -> tuple[float, float]:
    """Compare two position tensors with RMSD and raw max delta.

    Parameters
    ----------
    left : torch.Tensor
        First position tensor with shape ``[N, 2]``.
    right : torch.Tensor
        Second position tensor with shape ``[N, 2]``.

    Returns
    -------
    tuple[float, float]
        Procrustes RMSD and maximum absolute coordinate delta.
    """
    rmsd, _ = fidelity_procrustes(left, right)
    max_delta = float(torch.max(torch.abs(left - right)).item()) if left.numel() > 0 else 0.0
    return rmsd, max_delta


def _component_deltas(
    graph: DaguaGraph,
    seed: int,
    reference: torch.Tensor,
    direct: torch.Tensor,
    direct_repeat: torch.Tensor,
) -> dict[str, float]:
    """Quantify candidate residual sources against the adapter reference.

    Parameters
    ----------
    graph : DaguaGraph
        Prepared graph to test.
    seed : int
        Seed forwarded to all direct runs.
    reference : torch.Tensor
        Adapter-reference positions with shape ``[N, 2]``.
    direct : torch.Tensor
        Direct-pipeline positions with shape ``[N, 2]``.
    direct_repeat : torch.Tensor
        Second direct-pipeline fixed-seed run with shape ``[N, 2]``.

    Returns
    -------
    dict[str, float]
        RMSD by candidate sub-component toggle.
    """
    reversed_graph = DaguaGraph()
    for node in range(graph.num_nodes):
        reversed_graph.add_node(node)
    for src, dst in reversed(graph.edge_index.t().tolist()):
        reversed_graph.add_edge(int(src), int(dst))
    reversed_graph.node_sizes = graph.node_sizes.clone()
    if graph.clusters:
        reversed_graph.clusters.update(graph.clusters)

    direct_rmsd, _ = _compare_positions(direct, reference)
    repeat_rmsd, _ = _compare_positions(direct, direct_repeat)
    deltas: dict[str, float] = {
        "initialization": direct_rmsd,
        "rng_repeat": repeat_rmsd,
        "normalization_direct": direct_rmsd,
    }
    toggles: dict[str, Optional[Callable[[LayoutConfig], None]]] = {
        "iteration_order_reversed_edges": None,
        "force_kernel_no_polish": lambda config: setattr(config, "edge_equalize_polish", False),
        "convergence_steps_default": lambda config: setattr(config, "steps", 0),
    }
    for name, mutator in toggles.items():
        candidate_graph = reversed_graph if name == "iteration_order_reversed_edges" else graph
        candidate = _direct_pipeline(candidate_graph, seed, config_mutator=mutator)
        rmsd, _ = _compare_positions(candidate, reference)
        deltas[name] = rmsd
    return deltas


def run_smoke(
    cases: Optional[Iterable[SmokeCase]] = None,
) -> tuple[list[SmokeRow], dict[str, float]]:
    """Run the round 41 dagua_native reproducibility smoke.

    Parameters
    ----------
    cases : Iterable[SmokeCase], optional
        Topology cases to execute.

    Returns
    -------
    tuple[list[SmokeRow], dict[str, float]]
        Per-seed smoke rows and maximum RMSD by diagnosis component.
    """
    rows: list[SmokeRow] = []
    component_maxima: dict[str, float] = {}
    resolved_cases = _build_cases() if cases is None else cases
    for case in resolved_cases:
        for seed in SEEDS:
            graph = _make_graph(case)
            reference = _reference_adapter(graph, seed)
            direct = _direct_pipeline(graph, seed)
            direct_repeat = _direct_pipeline(graph, seed)
            reference_rmsd, max_delta = _compare_positions(direct, reference)
            repeat_rmsd, _ = _compare_positions(direct, direct_repeat)
            rows.append(
                SmokeRow(
                    topology=case.name,
                    seed=seed,
                    pipeline=_infer_pipeline(case),
                    reference_rmsd=reference_rmsd,
                    repeat_rmsd=repeat_rmsd,
                    max_abs_delta=max_delta,
                )
            )
            for name, value in _component_deltas(
                graph,
                seed,
                reference,
                direct,
                direct_repeat,
            ).items():
                component_maxima[name] = max(component_maxima.get(name, 0.0), value)
    return rows, component_maxima


def _print_table(rows: list[SmokeRow], component_maxima: dict[str, float]) -> None:
    """Print smoke rows and diagnosis maxima.

    Parameters
    ----------
    rows : list[SmokeRow]
        Per-seed smoke rows.
    component_maxima : dict[str, float]
        Maximum RMSD observed for each candidate residual source.

    Returns
    -------
    None
        Results are printed to stdout.
    """
    print("topology,seed,pipeline,reference_rmsd,repeat_rmsd,max_abs_delta")
    for row in rows:
        print(
            f"{row.topology},{row.seed},{row.pipeline},"
            f"{row.reference_rmsd:.12g},{row.repeat_rmsd:.12g},{row.max_abs_delta:.12g}"
        )
    overall = sum(row.reference_rmsd for row in rows) / max(len(rows), 1)
    print(f"overall_mean_reference_rmsd,{overall:.12g}")
    print("component,max_rmsd")
    for name in sorted(component_maxima):
        print(f"{name},{component_maxima[name]:.12g}")


def main() -> None:
    """Run the smoke harness from the command line.

    Returns
    -------
    None
        Results are printed to stdout.
    """
    rows, component_maxima = run_smoke()
    _print_table(rows, component_maxima)


if __name__ == "__main__":
    main()
