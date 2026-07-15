"""Verify Tutte and HDE deterministic fidelity against pinned local references."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dagua.eval.equivalence_metrics import anisotropic_procrustes, procrustes_rmsd  # noqa: E402
from dagua.layout.ops.pipelines.hde import layout_hde_pipeline  # noqa: E402
from dagua.layout.ops.pipelines.tutte import layout_tutte_pipeline  # noqa: E402

REPORT_PATH = ROOT / "docs" / "algorithms" / "tutte_hde_fidelity.md"
BIT_EXACT_THRESHOLD = 1.0e-9
POSITIONAL_THRESHOLD = 1.0e-6
TWO_PI = 2.0 * math.pi


def _edge_index(edges: Sequence[Tuple[int, int]]) -> torch.Tensor:
    """Convert edge pairs into an edge tensor.

    Parameters
    ----------
    edges : sequence[tuple[int, int]]
        Source-target edge pairs.

    Returns
    -------
    torch.Tensor
        Long tensor with shape ``[2, E]``.
    """
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def _graph_cases() -> List[Tuple[str, int, List[Tuple[int, int]]]]:
    """Return the fixed small graph corpus.

    Returns
    -------
    list[tuple[str, int, list[tuple[int, int]]]]
        Named graph cases.
    """
    grid_edges = [(row * 3 + col, row * 3 + col + 1) for row in range(3) for col in range(2)] + [
        (row * 3 + col, (row + 1) * 3 + col) for row in range(2) for col in range(3)
    ]
    return [
        ("triangle", 3, [(0, 1), (1, 2), (2, 0)]),
        ("wheel_5", 5, [(0, 1), (1, 2), (2, 3), (3, 0), (4, 0), (4, 1), (4, 2), (4, 3)]),
        ("path_5", 5, [(0, 1), (1, 2), (2, 3), (3, 4)]),
        ("disconnected", 6, [(0, 1), (1, 2), (3, 4)]),
        ("grid_3x3", 9, grid_edges),
        ("lollipop", 7, [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (5, 6)]),
    ]


def _adjacency(num_nodes: int, edges: Sequence[Tuple[int, int]]) -> List[List[int]]:
    """Build sorted undirected adjacency.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    edges : sequence[tuple[int, int]]
        Edge pairs.

    Returns
    -------
    list[list[int]]
        Sorted neighbor IDs.
    """
    rows: List[Set[int]] = [set() for _ in range(num_nodes)]
    for source, target in edges:
        if source == target:
            continue
        rows[int(source)].add(int(target))
        rows[int(target)].add(int(source))
    return [sorted(row) for row in rows]


def _canonical_cycle(cycle: Sequence[int]) -> Tuple[int, ...]:
    """Return a stable cycle representative.

    Parameters
    ----------
    cycle : sequence[int]
        Cycle nodes without repeated closure.

    Returns
    -------
    tuple[int, ...]
        Canonical cycle tuple.
    """
    values = list(cycle)
    variants: List[Tuple[int, ...]] = []
    for ordered in (values, list(reversed(values))):
        start = ordered.index(min(ordered))
        variants.append(tuple(ordered[start:] + ordered[:start]))
    return min(variants)


def _is_chordless(cycle: Sequence[int], neighbors: Sequence[Set[int]]) -> bool:
    """Return whether the cycle has no non-consecutive chords.

    Parameters
    ----------
    cycle : sequence[int]
        Cycle node order.
    neighbors : sequence[set[int]]
        Undirected neighbor sets.

    Returns
    -------
    bool
        Whether the cycle is chordless.
    """
    cycle_length = len(cycle)
    for left_index, left_node in enumerate(cycle):
        for right_index in range(left_index + 1, cycle_length):
            if (right_index - left_index) in (1, cycle_length - 1):
                continue
            if cycle[right_index] in neighbors[left_node]:
                return False
    return True


def _reference_boundary(num_nodes: int, edges: Sequence[Tuple[int, int]]) -> Tuple[List[int], str]:
    """Select the reference Tutte boundary.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    edges : sequence[tuple[int, int]]
        Edge pairs.

    Returns
    -------
    tuple[list[int], str]
        Boundary nodes and fallback reason.
    """
    adjacency = _adjacency(num_nodes, edges)
    neighbor_sets = [set(row) for row in adjacency]
    best: Tuple[int, ...] = ()
    for start in range(num_nodes):
        stack: List[Tuple[int, List[int], Set[int]]] = [(start, [start], {start})]
        while stack:
            node, path, seen = stack.pop()
            for neighbor in sorted(adjacency[node], reverse=True):
                if neighbor == start and len(path) >= 3:
                    candidate = _canonical_cycle(path)
                    if _is_chordless(candidate, neighbor_sets) and (
                        len(candidate) > len(best)
                        or (len(candidate) == len(best) and candidate < best)
                    ):
                        best = candidate
                    continue
                if neighbor <= start or neighbor in seen:
                    continue
                stack.append((neighbor, [*path, neighbor], {*seen, neighbor}))
    if len(best) >= 3:
        return list(best), "none"
    return list(range(num_nodes)), "no peripheral cycle; all nodes fixed on convex polygon"


def _reference_tutte(num_nodes: int, edges: Sequence[Tuple[int, int]]) -> Tuple[np.ndarray, str]:
    """Compute the reference Tutte embedding.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    edges : sequence[tuple[int, int]]
        Edge pairs.

    Returns
    -------
    tuple[numpy.ndarray, str]
        Coordinates with shape ``[N, 2]`` and fallback reason.
    """
    boundary, fallback = _reference_boundary(num_nodes, edges)
    positions = np.zeros((num_nodes, 2), dtype=np.float64)
    for index, node in enumerate(boundary):
        positions[node, 0] = math.cos(TWO_PI * index / len(boundary))
        positions[node, 1] = math.sin(TWO_PI * index / len(boundary))
    boundary_set = set(boundary)
    interior = [node for node in range(num_nodes) if node not in boundary_set]
    if not interior:
        return positions, fallback

    row_of = {node: row for row, node in enumerate(interior)}
    lhs = np.zeros((len(interior), len(interior)), dtype=np.float64)
    rhs = np.zeros((len(interior), 2), dtype=np.float64)
    for source, target in edges:
        for row_node, col_node in ((source, target), (target, source)):
            if row_node not in row_of:
                continue
            row = row_of[row_node]
            lhs[row, row] += 1.0
            if col_node in row_of:
                lhs[row, row_of[col_node]] -= 1.0
            elif col_node in boundary_set:
                rhs[row] += positions[col_node]
    try:
        positions[interior] = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        positions[interior] = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
        fallback = "singular interior system; least-squares barycenter solution"
    return positions, fallback


def _bfs(adjacency: Sequence[Sequence[int]], source: int) -> np.ndarray:
    """Compute one unweighted shortest-path row.

    Parameters
    ----------
    adjacency : sequence[sequence[int]]
        Undirected adjacency list.
    source : int
        Source node.

    Returns
    -------
    numpy.ndarray
        Distance row with unreachable nodes filled by ``max + 1``.
    """
    distances = np.full((len(adjacency),), -1.0, dtype=np.float64)
    distances[source] = 0.0
    queue = [source]
    cursor = 0
    while cursor < len(queue):
        node = queue[cursor]
        cursor += 1
        for neighbor in adjacency[node]:
            if distances[neighbor] >= 0.0:
                continue
            distances[neighbor] = distances[node] + 1.0
            queue.append(neighbor)
    reachable = distances >= 0.0
    max_distance = float(distances[reachable].max()) if bool(reachable.any()) else 0.0
    distances[distances < 0.0] = max_distance + 1.0
    return distances


def _reference_hde(
    num_nodes: int,
    edges: Sequence[Tuple[int, int]],
    n_pivots: int = 50,
) -> np.ndarray:
    """Compute the reference HDE embedding.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    edges : sequence[tuple[int, int]]
        Edge pairs.
    n_pivots : int, default=50
        Maximum number of pivots.

    Returns
    -------
    numpy.ndarray
        Coordinates with shape ``[N, 2]``.
    """
    if num_nodes == 0:
        return np.empty((0, 2), dtype=np.float64)
    adjacency = _adjacency(num_nodes, edges)
    pivots = [0]
    selected = np.zeros((num_nodes,), dtype=bool)
    selected[0] = True
    min_distances = _bfs(adjacency, 0)
    while len(pivots) < min(n_pivots, num_nodes):
        scores = min_distances.copy()
        scores[selected] = -1.0
        next_pivot = int(np.argmax(scores))
        if selected[next_pivot]:
            break
        selected[next_pivot] = True
        pivots.append(next_pivot)
        min_distances = np.minimum(min_distances, _bfs(adjacency, next_pivot))

    matrix = np.vstack([_bfs(adjacency, pivot) for pivot in pivots]).T
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    if num_nodes == 1 or np.max(np.abs(centered)) == 0.0:
        return np.zeros((num_nodes, 2), dtype=np.float64)
    u_matrix, singular_values, vh_matrix = np.linalg.svd(centered, full_matrices=False)
    dims = min(2, singular_values.shape[0])
    scores = u_matrix[:, :dims] * singular_values[:dims]
    for component in range(dims):
        anchor = int(np.argmax(np.abs(vh_matrix[component])))
        if vh_matrix[component, anchor] < 0.0:
            scores[:, component] *= -1.0
    if dims < 2:
        scores = np.hstack([scores, np.zeros((num_nodes, 2 - dims), dtype=np.float64)])
    return scores


def _classify(residual: float) -> str:
    """Classify one residual.

    Parameters
    ----------
    residual : float
        Anisotropic residual.

    Returns
    -------
    str
        Fidelity verdict.
    """
    if residual < BIT_EXACT_THRESHOLD:
        return "bit-exact"
    if residual < POSITIONAL_THRESHOLD:
        return "positional"
    return "divergent"


def _row(
    algorithm: str,
    name: str,
    num_nodes: int,
    edges: Sequence[Tuple[int, int]],
) -> Dict[str, object]:
    """Compute one report row.

    Parameters
    ----------
    algorithm : str
        ``"tutte"`` or ``"hde"``.
    name : str
        Graph case name.
    num_nodes : int
        Number of nodes.
    edges : sequence[tuple[int, int]]
        Edge pairs.

    Returns
    -------
    dict[str, object]
        Report row fields.
    """
    edge_index = _edge_index(edges)
    if algorithm == "tutte":
        reference, stage = _reference_tutte(num_nodes, edges)
        observed = layout_tutte_pipeline(
            edge_index=edge_index,
            num_nodes=num_nodes,
            fidelity_dtype=torch.float64,
        )
    else:
        reference = _reference_hde(num_nodes, edges)
        stage = "none"
        observed = layout_hde_pipeline(
            edge_index=edge_index,
            num_nodes=num_nodes,
            fidelity_dtype=torch.float64,
        )
    observed_np = observed.detach().cpu().numpy()
    procrustes = procrustes_rmsd(observed_np, reference)
    anisotropic = anisotropic_procrustes(observed_np, reference)["anisotropic_rmsd"]
    max_diff = float(np.max(np.abs(observed_np - reference))) if reference.size else 0.0
    verdict = _classify(float(anisotropic))
    if verdict != "divergent" and stage == "none":
        stage = "none"
    return {
        "algorithm": algorithm,
        "graph": name,
        "n": num_nodes,
        "e": len(edges),
        "procrustes": float(procrustes),
        "anisotropic": float(anisotropic),
        "max_diff": max_diff,
        "verdict": verdict,
        "stage": stage,
    }


def _format_report(rows: Sequence[Dict[str, object]]) -> str:
    """Format markdown report content.

    Parameters
    ----------
    rows : sequence[dict[str, object]]
        Report rows.

    Returns
    -------
    str
        Markdown document.
    """
    lines = [
        "# Tutte/HDE fidelity verification",
        "",
        "References: pinned local deterministic adapters in "
        "`scripts/verify_tutte_hde_fidelity.py`. "
        "Production pipelines do not invoke external reference engines.",
        "",
    ]
    for algorithm in ("tutte", "hde"):
        subset = [row for row in rows if row["algorithm"] == algorithm]
        bit_exact = sum(1 for row in subset if row["verdict"] == "bit-exact")
        positional = sum(1 for row in subset if row["verdict"] == "positional")
        lines.extend(
            [
                f"## {algorithm}",
                "",
                f"Result: **{bit_exact}/{len(subset)} bit-exact**, **{positional} positional**, "
                f"thresholds `bit < {BIT_EXACT_THRESHOLD:g}`, positional "
                f"`< {POSITIONAL_THRESHOLD:g}`.",
                "",
                "| graph | N | E | Procrustes d_R | anisotropic d_R | max raw diff | "
                "verdict | first divergent/N-A stage |",
                "|---|---:|---:|---:|---:|---:|---|---|",
            ]
        )
        for row in subset:
            lines.append(
                f"| {row['graph']} | {row['n']} | {row['e']} | "
                f"{float(row['procrustes']):.3e} | {float(row['anisotropic']):.3e} | "
                f"{float(row['max_diff']):.3e} | {row['verdict']} | {row['stage']} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Notes",
            "",
            "Tutte uses a chordless peripheral cycle as the fixed convex boundary in the "
            "headless tensor API. Graphs without such a cycle are finite but marked N/A "
            "for theorem fidelity because all nodes are fixed on the fallback polygon.",
            "",
            "HDE is also exposed as the reusable `hde_project_pivot_distances` init op; "
            "the public pipeline composes adjacency, farthest-first pivots, distance "
            "queries, and that init op.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    """Run verification and write the markdown report.

    Returns
    -------
    int
        Process exit status.
    """
    rows: List[Dict[str, object]] = []
    for name, num_nodes, edges in _graph_cases():
        rows.append(_row("tutte", name, num_nodes, edges))
        rows.append(_row("hde", name, num_nodes, edges))

    for algorithm in ("tutte", "hde"):
        subset = [row for row in rows if row["algorithm"] == algorithm]
        bit_exact = sum(1 for row in subset if row["verdict"] == "bit-exact")
        positional = sum(1 for row in subset if row["verdict"] == "positional")
        print(f"{algorithm}: {bit_exact}/{len(subset)} bit-exact, {positional} positional")
        for row in subset:
            print(
                f"  {row['graph']}: d_R={float(row['anisotropic']):.3e} "
                f"procrustes={float(row['procrustes']):.3e} verdict={row['verdict']} "
                f"stage={row['stage']}"
            )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(_format_report(rows))
    print(f"wrote {REPORT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
