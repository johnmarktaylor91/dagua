"""Round 40 smoke and checkpoint diagnostics for Graphviz fdp clusters."""

from __future__ import annotations

import statistics
import sys
from pathlib import Path
from typing import Dict, List

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from dagua.eval.competitors import get_competitor  # noqa: E402
from dagua.graph import DaguaGraph  # noqa: E402
from dagua.layout.ops.cluster_geometry import ClusterTree  # noqa: E402
from dagua.layout.ops.pipelines.fmmm import (  # noqa: E402
    _fdp_recursion_component_offsets,
    _fdp_recursion_component_sizes,
    _fdp_recursion_components,
    _fdp_recursion_derive_graph,
    _fdp_recursion_expand_cluster_ports,
    _fdp_recursion_tlayout_component,
    _fdp_recursion_xlayout_component,
    _graphviz_tile_pack_offsets,
)
from eval_output.algo_fidelity.round_39.fdp_kernels.smoke_check import (  # noqa: E402
    SEEDS,
    build_clustered_path_graph,
    build_multi_cluster_graph,
    build_path_graph,
    smoke_rmsd,
)
from scripts.fidelity_analysis import fidelity_procrustes  # noqa: E402


def _edge_index(edges: list[tuple[int, int]]) -> torch.Tensor:
    """Build a PyTorch edge tensor.

    Parameters
    ----------
    edges : list[tuple[int, int]]
        Source-target pairs.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, E]``.
    """
    sources, targets = zip(*edges)
    return torch.tensor([list(sources), list(targets)], dtype=torch.long)


def build_one_cluster_graph() -> DaguaGraph:
    """Build the one-level clustered recursion diagnostic graph.

    Returns
    -------
    DaguaGraph
        Five-node graph with one three-node cluster and two outside leaves.
    """
    graph = DaguaGraph()
    for index in range(5):
        graph.add_node(f"n{index}")
    for source, target in ((0, 3), (1, 3), (2, 4), (3, 4)):
        graph.add_edge(f"n{source}", f"n{target}")
    graph.add_cluster("core", ["n0", "n1", "n2"])
    graph.compute_node_sizes()
    return graph


def checkpoint_trace() -> Dict[str, object]:
    """Capture one-level fdp recursion checkpoints from the Python port.

    Returns
    -------
    dict[str, object]
        Derived graph, post-tLayout, post-expandCluster, post-xLayout, and
        post-pack data for the one-cluster smoke graph.
    """
    edge_index = _edge_index([(0, 3), (1, 3), (2, 4), (3, 4)])
    node_sizes = torch.full((5, 2), 54.0, dtype=torch.float32)
    tree = ClusterTree.from_flat_membership({"core": [0, 1, 2]}, {"core": None})
    derived = _fdp_recursion_derive_graph(edge_index, 5, tree, None)
    components = _fdp_recursion_components(derived)
    root_component = components[0]
    tlayout, xpms = _fdp_recursion_tlayout_component(derived, root_component, seed=1)
    root_positions = {
        derived_index: tlayout[local_index]
        for local_index, derived_index in enumerate(root_component)
    }
    root_after_tlayout = {index: value.tolist() for index, value in root_positions.items()}
    cluster_index = next(
        index for index, node in enumerate(derived.nodes) if node.kind == "cluster"
    )
    ports = _fdp_recursion_expand_cluster_ports(
        derived=derived,
        derived_positions=root_positions,
        cluster_index=cluster_index,
        edge_index=edge_index,
    )
    child_derived = _fdp_recursion_derive_graph(edge_index, 5, tree, "core", ports)
    child_components = _fdp_recursion_components(child_derived)
    child_tlayout, child_xpms = _fdp_recursion_tlayout_component(
        child_derived,
        child_components[0],
        seed=1,
    )
    child_positions = {
        derived_index: child_tlayout[local_index]
        for local_index, derived_index in enumerate(child_components[0])
    }
    child_positions = _fdp_recursion_xlayout_component(
        derived=child_derived,
        component=child_components[0],
        local_positions=child_positions,
        node_sizes=node_sizes,
        child_layouts={},
        xpms=child_xpms,
    )
    child_sizes = _fdp_recursion_component_sizes(
        child_derived,
        tuple(child_positions),
        node_sizes,
        {},
    )
    child_tensor = torch.stack([child_positions[index] for index in child_positions])
    child_lower = (child_tensor - child_sizes / 2.0).min(dim=0).values
    child_upper = (child_tensor + child_sizes / 2.0).max(dim=0).values
    child_bbox = (
        float(child_lower[0]),
        float(child_lower[1]),
        float(child_upper[0]),
        float(child_upper[1]),
    )
    root_positions = _fdp_recursion_xlayout_component(
        derived=derived,
        component=root_component,
        local_positions=root_positions,
        node_sizes=node_sizes,
        child_layouts={},
        xpms=xpms,
    )
    root_sizes = _fdp_recursion_component_sizes(derived, root_component, node_sizes, {})
    root_tensor = torch.stack([root_positions[index] for index in root_positions])
    root_lower = (root_tensor - root_sizes[: len(root_positions)] / 2.0).min(dim=0).values
    root_upper = (root_tensor + root_sizes[: len(root_positions)] / 2.0).max(dim=0).values
    offsets = _fdp_recursion_component_offsets(
        [(float(root_upper[0] - root_lower[0]), float(root_upper[1] - root_lower[1]))]
    )
    return {
        "derived_nodes": [(node.kind, node.key) for node in derived.nodes],
        "derived_edges": [(edge.source, edge.target, edge.real_edges) for edge in derived.edges],
        "components": components,
        "root_after_tlayout": root_after_tlayout,
        "root_after_xlayout": {index: value.tolist() for index, value in root_positions.items()},
        "ports": [(port.edge_id, port.node, port.alpha) for port in ports],
        "child_after_expand_bbox": child_bbox,
        "pack_offsets": [offset.tolist() for offset in offsets],
        "tile_pack_reference": _graphviz_tile_pack_offsets(
            [(0.0, 0.0, float(root_upper[0] - root_lower[0]), float(root_upper[1] - root_lower[1]))]
        ),
    }


def one_cluster_rmsd(seed: int) -> float:
    """Compute the one-cluster smoke RMSD against Graphviz fdp.

    Parameters
    ----------
    seed : int
        Seed passed to both layout engines.

    Returns
    -------
    float
        Scale-normalized Procrustes RMSD.
    """
    dagua = get_competitor("classic_fmmm")
    graphviz = get_competitor("graphviz_fdp")
    if dagua is None or graphviz is None:
        raise RuntimeError("Required competitors are not registered.")
    graph = build_one_cluster_graph()
    dagua_result = dagua.layout_with_variant(
        graph,
        timeout=60.0,
        seed=seed,
        variant_params={"steps": 200, "fidelity_mode": True},
    )
    graphviz_result = graphviz.layout(graph, timeout=60.0, seed=seed)
    if dagua_result.pos is None or graphviz_result.pos is None:
        raise RuntimeError(f"layout failed: {dagua_result.error} / {graphviz_result.error}")
    rmsd, _, _, _ = fidelity_procrustes(
        dagua_result.pos.to(dtype=torch.float64),
        graphviz_result.pos.to(dtype=torch.float64),
    )
    return float(rmsd)


def run_smoke_checks() -> Dict[str, List[float]]:
    """Run Round 40 FDP cluster smoke checks.

    Returns
    -------
    dict[str, list[float]]
        RMSDs by topology.
    """
    results = {
        "one_cluster": [one_cluster_rmsd(seed) for seed in SEEDS],
        "path": [smoke_rmsd(build_path_graph(), seed) for seed in SEEDS],
        "clustered": [smoke_rmsd(build_clustered_path_graph(), seed) for seed in SEEDS],
        "multi_cluster": [smoke_rmsd(build_multi_cluster_graph(), seed) for seed in SEEDS],
    }
    return results


def main() -> None:
    """Print checkpoints and RMSD smoke results."""
    print("checkpoint_trace:")
    for key, value in checkpoint_trace().items():
        print(f"  {key}: {value}")
    print("smoke:")
    for topology, values in run_smoke_checks().items():
        formatted = ", ".join(f"{value:.9f}" for value in values)
        print(
            f"  {topology}: {formatted} "
            f"(mean={statistics.fmean(values):.9f}, max={max(values):.9f})"
        )


if __name__ == "__main__":
    main()
