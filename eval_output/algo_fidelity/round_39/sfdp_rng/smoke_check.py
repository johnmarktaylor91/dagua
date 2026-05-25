"""Round 39 SFDP Graphviz RNG smoke checks."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from dagua.eval.competitors import get_competitor  # noqa: E402
from dagua.eval.competitors.classic_competitor import VariantCompetitor  # noqa: E402
from dagua.eval.variants import AlgorithmVariant, get_variant, original_variant_name  # noqa: E402
from dagua.graph import DaguaGraph  # noqa: E402
from scripts.fidelity_analysis import fidelity_procrustes  # noqa: E402

VARIANT_ID = "classic_sfdp_graphviz_fidelity"
SEEDS = (1, 2, 3)
TOPOLOGY_BUILDERS: dict[str, Callable[[], DaguaGraph]] = {}


def build_path_graph(num_nodes: int = 8) -> DaguaGraph:
    """Build a directed path graph for SFDP smoke checks.

    Parameters
    ----------
    num_nodes : int, default=8
        Number of nodes in the path.

    Returns
    -------
    DaguaGraph
        Path graph with computed node sizes.
    """
    graph = DaguaGraph()
    for index in range(num_nodes):
        graph.add_node(f"n{index}")
    for index in range(num_nodes - 1):
        graph.add_edge(f"n{index}", f"n{index + 1}")
    graph.compute_node_sizes()
    return graph


def build_star_graph(num_leaves: int = 7) -> DaguaGraph:
    """Build a hub-and-spokes graph for RNG-sensitive SFDP smoke checks.

    Parameters
    ----------
    num_leaves : int, default=7
        Number of leaves attached to the central hub.

    Returns
    -------
    DaguaGraph
        Star graph with computed node sizes.
    """
    graph = DaguaGraph()
    graph.add_node("n0")
    for index in range(1, num_leaves + 1):
        graph.add_node(f"n{index}")
        graph.add_edge("n0", f"n{index}")
    graph.compute_node_sizes()
    return graph


def build_clustered_graph() -> DaguaGraph:
    """Build a small clustered graph that still exercises SFDP hierarchy code.

    Returns
    -------
    DaguaGraph
        Clustered graph with two dense groups and bridge edges.
    """
    graph = DaguaGraph()
    for index in range(10):
        graph.add_node(f"n{index}")

    left = range(5)
    right = range(5, 10)
    for group in (left, right):
        members = list(group)
        for offset, source in enumerate(members):
            for target in members[offset + 1 :]:
                graph.add_edge(f"n{source}", f"n{target}")
    graph.add_edge("n1", "n6")
    graph.add_edge("n3", "n8")
    graph.add_cluster("left", [f"n{index}" for index in left])
    graph.add_cluster("right", [f"n{index}" for index in right])
    graph.compute_node_sizes()
    return graph


TOPOLOGY_BUILDERS = {
    "path": build_path_graph,
    "star": build_star_graph,
    "clustered": build_clustered_graph,
}


def require_variant() -> AlgorithmVariant:
    """Resolve the SFDP Graphviz-fidelity variant.

    Returns
    -------
    AlgorithmVariant
        Registered SFDP Graphviz-fidelity variant.

    Raises
    ------
    RuntimeError
        If the variant is not registered.
    """
    variant = get_variant(VARIANT_ID)
    if variant is None:
        raise RuntimeError(f"Missing variant: {VARIANT_ID}")
    return variant


def smoke_rmsd(topology: str, seed: int) -> float:
    """Compute one topology/seed SFDP Graphviz smoke RMSD.

    Parameters
    ----------
    topology : str
        Topology key in :data:`TOPOLOGY_BUILDERS`.
    seed : int
        Seed passed to both the Dagua fidelity implementation and Graphviz.

    Returns
    -------
    float
        Scale-normalized Procrustes RMSD.
    """
    variant = require_variant()
    base_competitor = get_competitor(variant.base_engine)
    if base_competitor is None:
        raise RuntimeError(f"Missing base competitor: {variant.base_engine}")
    if variant.original_engine is None:
        raise RuntimeError(f"Variant has no reference adapter: {VARIANT_ID}")
    reference_competitor = get_competitor(variant.original_engine)
    if reference_competitor is None:
        raise RuntimeError(f"Missing reference competitor: {variant.original_engine}")

    reference_name = original_variant_name(variant)
    if reference_name is None:
        raise RuntimeError(f"Missing synthetic reference name: {VARIANT_ID}")

    reimpl = VariantCompetitor(
        base_competitor=base_competitor,
        variant_params=variant.reimpl_params,
        name=variant.variant_id,
        display_name=variant.display_name,
        is_heavy=variant.is_heavy,
        max_nodes=variant.max_nodes,
    )
    reference = VariantCompetitor(
        base_competitor=reference_competitor,
        variant_params=variant.original_params,
        name=reference_name,
        display_name=reference_name,
        is_heavy=variant.is_heavy,
    )

    graph = TOPOLOGY_BUILDERS[topology]()
    reimpl_result = reimpl.layout(graph, timeout=60.0, seed=seed)
    if reimpl_result.pos is None:
        raise RuntimeError(f"{VARIANT_ID} failed: {reimpl_result.error}")
    reference_result = reference.layout(graph, timeout=60.0, seed=seed)
    if reference_result.pos is None:
        raise RuntimeError(f"{reference_name} failed: {reference_result.error}")

    reimpl_pos = reimpl_result.pos.to(dtype=torch.float64)
    reference_pos = reference_result.pos.to(dtype=torch.float64)
    rmsd, _, _, _ = fidelity_procrustes(reimpl_pos, reference_pos)
    return rmsd


def run_smoke_checks() -> dict[str, dict[int, float]]:
    """Run Round 39 SFDP smoke checks.

    Returns
    -------
    dict[str, dict[int, float]]
        Nested mapping from topology name to seed RMSD values.
    """
    return {
        topology: {seed: smoke_rmsd(topology=topology, seed=seed) for seed in SEEDS}
        for topology in TOPOLOGY_BUILDERS
    }


def main() -> None:
    """Print Round 39 SFDP smoke RMSDs."""
    for topology, seed_values in run_smoke_checks().items():
        values = ", ".join(f"seed {seed}: {rmsd:.9f}" for seed, rmsd in seed_values.items())
        print(f"{topology}: {values}")


if __name__ == "__main__":
    main()
