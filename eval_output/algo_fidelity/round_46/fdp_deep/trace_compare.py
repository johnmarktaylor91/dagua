"""Round 46 diagnostic trace comparison for Graphviz fdp clusters."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from dagua.graphviz_utils import to_dot  # noqa: E402
from dagua.layout.ops.cluster_geometry import ClusterTree  # noqa: E402
from dagua.layout.ops.pipelines.fmmm import (  # noqa: E402
    _fdp_recursion_component_edges,
    _fdp_recursion_components,
    _fdp_recursion_derive_graph,
    _fdp_recursion_expand_cluster_ports,
    _fdp_recursion_tlayout_component,
    _fdp_recursion_xlayout_component,
)
from eval_output.algo_fidelity.round_39.fdp_kernels.smoke_check import (  # noqa: E402
    build_clustered_path_graph,
)


def _edge_index(edges: List[Tuple[int, int]]) -> torch.Tensor:
    """Build an edge tensor for a trace fixture.

    Parameters
    ----------
    edges : list[tuple[int, int]]
        Directed source-target pairs in node-index space.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, E]``.
    """
    sources, targets = zip(*edges)
    return torch.tensor([list(sources), list(targets)], dtype=torch.long)


def _tensor_dict(values: Dict[int, torch.Tensor]) -> Dict[str, List[float]]:
    """Convert a derived-position mapping into JSON-safe lists.

    Parameters
    ----------
    values : dict[int, torch.Tensor]
        Position tensors keyed by derived-node index.

    Returns
    -------
    dict[str, list[float]]
        Position lists keyed by stringified derived-node index.
    """
    return {
        str(index): [float(position[0].item()), float(position[1].item())]
        for index, position in values.items()
    }


def capture_dagua_trace(seed: int) -> Dict[str, Any]:
    """Capture recursive Dagua FDP checkpoints for the two-cluster fixture.

    Parameters
    ----------
    seed : int
        Seed forwarded to the Graphviz-compatible random initializer.

    Returns
    -------
    dict[str, Any]
        Derived graph, component, port, child layout, and xLayout checkpoints.
    """
    edge_index = _edge_index([(index, index + 1) for index in range(7)])
    tree = ClusterTree.from_flat_membership(
        {"left": [0, 1, 2, 3], "right": [4, 5, 6, 7]},
        {},
    )
    node_sizes = None

    derived = _fdp_recursion_derive_graph(edge_index, 8, tree, None)
    components = _fdp_recursion_components(derived)
    root_component = components[0]
    root_tlayout, root_xpms = _fdp_recursion_tlayout_component(derived, root_component, seed)
    root_positions = {
        derived_index: root_tlayout[local_index]
        for local_index, derived_index in enumerate(root_component)
    }

    child_traces: Dict[str, Any] = {}
    for derived_index in root_component:
        node = derived.nodes[derived_index]
        if node.kind != "cluster":
            continue
        ports = _fdp_recursion_expand_cluster_ports(
            derived=derived,
            derived_positions=root_positions,
            cluster_index=derived_index,
            edge_index=edge_index,
        )
        child_derived = _fdp_recursion_derive_graph(edge_index, 8, tree, str(node.key), ports)
        child_components = _fdp_recursion_components(child_derived)
        child_component = child_components[0]
        child_tlayout, child_xpms = _fdp_recursion_tlayout_component(
            child_derived,
            child_component,
            seed,
        )
        child_positions = {
            child_index: child_tlayout[local_index]
            for local_index, child_index in enumerate(child_component)
        }
        child_xlayout = _fdp_recursion_xlayout_component(
            derived=child_derived,
            component=child_component,
            local_positions=child_positions,
            node_sizes=node_sizes,
            child_layouts={},
            xpms=child_xpms,
        )
        child_traces[str(node.key)] = {
            "ports": [
                {"edge_id": port.edge_id, "node": port.node, "alpha": port.alpha} for port in ports
            ],
            "derived_nodes": [(child.kind, child.key) for child in child_derived.nodes],
            "derived_edges": [
                (edge.source, edge.target, edge.real_edges) for edge in child_derived.edges
            ],
            "component_edges": _fdp_recursion_component_edges(
                child_derived,
                child_component,
            ).tolist(),
            "after_tlayout": _tensor_dict(child_positions),
            "after_xlayout": _tensor_dict(child_xlayout),
        }

    root_xlayout = _fdp_recursion_xlayout_component(
        derived=derived,
        component=root_component,
        local_positions=root_positions,
        node_sizes=node_sizes,
        child_layouts={},
        xpms=root_xpms,
    )
    return {
        "fixture": "clustered_path_2x4",
        "seed": seed,
        "root": {
            "derived_nodes": [(node.kind, node.key) for node in derived.nodes],
            "derived_edges": [
                (edge.source, edge.target, edge.real_edges) for edge in derived.edges
            ],
            "components": components,
            "after_tlayout": _tensor_dict(root_positions),
            "after_xlayout_without_child_sizes": _tensor_dict(root_xlayout),
        },
        "children": child_traces,
    }


def capture_graphviz_verbose(seed: int) -> Dict[str, Any]:
    """Run Graphviz fdp with verbose tracing on the same fixture.

    Parameters
    ----------
    seed : int
        Seed passed as Graphviz ``start`` and ``seed`` graph attributes.

    Returns
    -------
    dict[str, Any]
        Final plain-output positions and verbose phase lines from Graphviz.
    """
    graph = build_clustered_path_graph()
    dot_source = to_dot(graph)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".dot",
        delete=False,
        encoding="utf-8",
    ) as handle:
        handle.write(dot_source)
        dot_path = Path(handle.name)
    try:
        result = subprocess.run(
            [
                "dot",
                "-v",
                "-Tplain",
                "-Kfdp",
                f"-Gseed={seed}",
                f"-Gstart={seed}",
                str(dot_path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=60.0,
        )
    finally:
        dot_path.unlink(missing_ok=True)

    phase_lines = [
        line.strip()
        for line in result.stderr.splitlines()
        if line.strip().startswith(("layout ", "end ", "xLayout", "step size"))
    ]
    positions: Dict[str, List[float]] = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0] == "node" and parts[1].startswith("n"):
            positions[parts[1]] = [float(parts[2]), -float(parts[3])]
    return {"phase_lines": phase_lines, "positions": positions}


def main() -> None:
    """Write the round 46 trace comparison JSON artifact."""
    output_path = Path(__file__).with_name("trace_seed1.json")
    payload = {
        "dagua": capture_dagua_trace(seed=1),
        "graphviz": capture_graphviz_verbose(seed=1),
        "first_confirmed_divergence": {
            "step": "child initPositions before tLayout iteration 0",
            "detail": (
                "Graphviz uses 0.90 for the y coordinate when a child node has exactly "
                "one positioned neighbor; the pre-R46 Dagua port used 0.98 for both axes."
            ),
            "source": "graphviz-7.0.5/lib/fdpgen/tlayout.c:initPositions",
        },
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
