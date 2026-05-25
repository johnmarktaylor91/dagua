"""Round 37 smoke checks for Graphviz-fidelity benchmark variants."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from dagua.eval.competitors import get_competitor  # noqa: E402
from dagua.eval.competitors.classic_competitor import VariantCompetitor  # noqa: E402
from dagua.eval.variants import AlgorithmVariant, get_variant, original_variant_name  # noqa: E402
from dagua.graph import DaguaGraph  # noqa: E402
from scripts.fidelity_analysis import fidelity_procrustes  # noqa: E402

VARIANT_IDS: tuple[str, ...] = (
    "classic_sugiyama_graphviz_fidelity",
    "classic_sfdp_graphviz_fidelity",
    "classic_fmmm_graphviz_fdp_fidelity",
    "classic_neato_graphviz_fidelity",
)


def build_path_graph(num_nodes: int = 8) -> DaguaGraph:
    """Build the small graph used by all round 37 smoke checks.

    Parameters
    ----------
    num_nodes : int, default=8
        Number of nodes in the directed path graph.

    Returns
    -------
    DaguaGraph
        Path graph with precomputed node sizes for adapters that consult
        dimensions during layout.
    """
    graph = DaguaGraph()
    for index in range(num_nodes):
        graph.add_node(f"n{index}")
    for index in range(num_nodes - 1):
        graph.add_edge(f"n{index}", f"n{index + 1}")
    graph.compute_node_sizes()
    return graph


def build_clustered_path_graph(num_nodes: int = 8) -> DaguaGraph:
    """Build a clustered path graph for Graphviz fdp recursion smoke coverage.

    Parameters
    ----------
    num_nodes : int, default=8
        Number of nodes in the directed path graph.

    Returns
    -------
    DaguaGraph
        Path graph split into two sibling clusters.
    """
    graph = build_path_graph(num_nodes=num_nodes)
    midpoint = num_nodes // 2
    graph.add_cluster("left", [f"n{index}" for index in range(midpoint)])
    graph.add_cluster("right", [f"n{index}" for index in range(midpoint, num_nodes)])
    return graph


def build_smoke_graph(variant_id: str) -> DaguaGraph:
    """Build the smallest graph that exercises one variant's fidelity path.

    Parameters
    ----------
    variant_id : str
        Registry id for the reimplementation-side variant.

    Returns
    -------
    DaguaGraph
        Smoke-test graph for the variant.
    """
    if variant_id == "classic_fmmm_graphviz_fdp_fidelity":
        return build_clustered_path_graph()
    return build_path_graph()


def require_variant(variant_id: str) -> AlgorithmVariant:
    """Resolve one variant or raise a clear error.

    Parameters
    ----------
    variant_id : str
        Registry id for the reimplementation-side variant.

    Returns
    -------
    AlgorithmVariant
        Resolved variant metadata.

    Raises
    ------
    RuntimeError
        If the variant is not registered.
    """
    variant = get_variant(variant_id)
    if variant is None:
        raise RuntimeError(f"Missing variant: {variant_id}")
    return variant


def smoke_rmsd(variant_id: str) -> float:
    """Run one variant and its reference adapter and compute Procrustes RMSD.

    Parameters
    ----------
    variant_id : str
        Registry id for the reimplementation-side variant.

    Returns
    -------
    float
        Scale-normalized Procrustes RMSD between the two layouts.

    Raises
    ------
    RuntimeError
        If any required registry entry or layout result is missing.
    """
    variant = require_variant(variant_id)
    base_competitor = get_competitor(variant.base_engine)
    if base_competitor is None:
        raise RuntimeError(f"Missing base competitor: {variant.base_engine}")
    if variant.original_engine is None:
        raise RuntimeError(f"Variant has no reference adapter: {variant_id}")
    reference_competitor = get_competitor(variant.original_engine)
    if reference_competitor is None:
        raise RuntimeError(f"Missing reference competitor: {variant.original_engine}")

    reimpl = VariantCompetitor(
        base_competitor=base_competitor,
        variant_params=variant.reimpl_params,
        name=variant.variant_id,
        display_name=variant.display_name,
        is_heavy=variant.is_heavy,
        max_nodes=variant.max_nodes,
    )
    reference_name = original_variant_name(variant)
    if reference_name is None:
        raise RuntimeError(f"Missing synthetic reference name: {variant_id}")
    reference = VariantCompetitor(
        base_competitor=reference_competitor,
        variant_params=variant.original_params,
        name=reference_name,
        display_name=reference_name,
        is_heavy=variant.is_heavy,
    )

    graph = build_smoke_graph(variant_id)
    reimpl_result = reimpl.layout(graph, timeout=60.0, seed=1)
    if reimpl_result.pos is None:
        raise RuntimeError(f"{variant_id} failed: {reimpl_result.error}")
    reference_result = reference.layout(graph, timeout=60.0, seed=None)
    if reference_result.pos is None:
        raise RuntimeError(f"{reference_name} failed: {reference_result.error}")

    reimpl_pos = reimpl_result.pos.to(dtype=torch.float64)
    reference_pos = reference_result.pos.to(dtype=torch.float64)
    rmsd, _, _, _ = fidelity_procrustes(reimpl_pos, reference_pos)
    return rmsd


def run_smoke_checks(variant_ids: Iterable[str] = VARIANT_IDS) -> dict[str, float]:
    """Run all requested smoke checks.

    Parameters
    ----------
    variant_ids : Iterable[str], default=VARIANT_IDS
        Variant ids to resolve and compare with their reference adapters.

    Returns
    -------
    dict[str, float]
        Mapping from variant id to Procrustes RMSD.
    """
    return {variant_id: smoke_rmsd(variant_id) for variant_id in variant_ids}


def main() -> None:
    """Print round 37 smoke RMSDs."""
    for variant_id, rmsd in run_smoke_checks().items():
        print(f"{variant_id}: {rmsd:.9f}")


if __name__ == "__main__":
    main()
