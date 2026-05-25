"""Round 41 tsNET smoke checks against sklearn exact t-SNE."""

from __future__ import annotations

import statistics
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from scipy.spatial.distance import squareform
from sklearn.manifold._t_sne import _joint_probabilities

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from dagua.layout.ops.pipelines.tsnet import (  # noqa: E402
    _layout_tsnet_sklearn_reference,
    build_tsnet_pipeline,
    layout_tsnet_pipeline,
)
from dagua.layout.ops.state import (  # noqa: E402
    ExecutionPlan,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)
from dagua.layout.ops.tsnet import (  # noqa: E402
    TsnetInitializePositions,
    TsnetInitializePositionsConfig,
    TsnetPrepareState,
)
from scripts.fidelity_analysis import fidelity_procrustes  # noqa: E402

SEEDS: tuple[int, ...] = (0, 1, 2)
STEPS = 500


def _edge_index(edges: list[tuple[int, int]]) -> torch.Tensor:
    """Build an edge tensor from edge pairs.

    Parameters
    ----------
    edges : list[tuple[int, int]]
        Directed edge pairs.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, E]``.
    """
    sources, targets = zip(*edges)
    return torch.tensor([list(sources), list(targets)], dtype=torch.long)


def build_path() -> tuple[torch.Tensor, int]:
    """Build the path smoke topology.

    Returns
    -------
    tuple[torch.Tensor, int]
        Edge tensor and node count.
    """
    num_nodes = 16
    return _edge_index([(node, node + 1) for node in range(num_nodes - 1)]), num_nodes


def build_star() -> tuple[torch.Tensor, int]:
    """Build the star smoke topology.

    Returns
    -------
    tuple[torch.Tensor, int]
        Edge tensor and node count.
    """
    num_nodes = 16
    return _edge_index([(0, node) for node in range(1, num_nodes)]), num_nodes


def build_clustered() -> tuple[torch.Tensor, int]:
    """Build the clustered smoke topology.

    Returns
    -------
    tuple[torch.Tensor, int]
        Edge tensor and node count.
    """
    edges: list[tuple[int, int]] = []
    for offset in (0, 8):
        edges.extend(
            (offset + source, offset + target)
            for source in range(8)
            for target in range(source + 1, 8)
        )
    edges.extend([(3, 8), (4, 9), (7, 15)])
    return _edge_index(edges), 16


def build_grid() -> tuple[torch.Tensor, int]:
    """Build the grid smoke topology.

    Returns
    -------
    tuple[torch.Tensor, int]
        Edge tensor and node count.
    """
    width = 4
    edges: list[tuple[int, int]] = []
    for row in range(width):
        for col in range(width):
            node = row * width + col
            if col + 1 < width:
                edges.append((node, node + 1))
            if row + 1 < width:
                edges.append((node, node + width))
    return _edge_index(edges), width * width


def _legacy_torch_pipeline(edge_index: torch.Tensor, num_nodes: int, seed: int) -> torch.Tensor:
    """Run the pre-R41 torch fidelity pipeline directly.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    seed : int
        Random seed.

    Returns
    -------
    torch.Tensor
        Position tensor with shape ``[N, 2]``.
    """
    problem = LayoutProblem(edge_index=edge_index, num_nodes=num_nodes, seed=seed)
    state = SolveState(extras={"tsnet_perplexity": 30.0})
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))
    final_state = build_tsnet_pipeline(steps=STEPS, fidelity_mode=True).apply(problem, state, ctx)
    if final_state.pos is None:
        raise RuntimeError("legacy tsNET pipeline produced no positions.")
    return final_state.pos


def _rmsd(pos_a: torch.Tensor, pos_b: torch.Tensor) -> float:
    """Compute scale-normalized Procrustes RMSD.

    Parameters
    ----------
    pos_a : torch.Tensor
        First position tensor with shape ``[N, 2]``.
    pos_b : torch.Tensor
        Second position tensor with shape ``[N, 2]``.

    Returns
    -------
    float
        Procrustes RMSD.
    """
    rmsd, _, _, _ = fidelity_procrustes(
        pos_a.to(dtype=torch.float64),
        pos_b.to(dtype=torch.float64),
    )
    return float(rmsd)


def _probability_delta(edge_index: torch.Tensor, num_nodes: int) -> float:
    """Measure Dagua-vs-sklearn high-dimensional probability drift.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    float
        Maximum absolute probability difference.
    """
    problem = LayoutProblem(edge_index=edge_index, num_nodes=num_nodes, seed=0)
    state = SolveState(extras={"tsnet_perplexity": 30.0})
    prepared = TsnetPrepareState().apply(problem, state, RuntimeContext())
    dagua_probabilities = prepared.extras["tsnet_probabilities"].detach().cpu().numpy()
    distances = prepared.distance_matrix.detach().cpu().numpy()
    sklearn_condensed = _joint_probabilities(
        (distances**2).astype(np.float32, copy=False),
        min(30.0, float(num_nodes - 1)),
        0,
    )
    sklearn_probabilities = squareform(sklearn_condensed)
    return float(np.max(np.abs(dagua_probabilities - sklearn_probabilities)))


def _initialization_delta(num_nodes: int, seed: int) -> float:
    """Measure Dagua-vs-sklearn random initialization drift.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    seed : int
        Random seed.

    Returns
    -------
    float
        Maximum absolute initialization difference.
    """
    problem = LayoutProblem(
        edge_index=torch.empty((2, 0), dtype=torch.long),
        num_nodes=num_nodes,
        seed=seed,
    )
    state = TsnetInitializePositions(TsnetInitializePositionsConfig(fidelity_mode=True)).apply(
        problem,
        SolveState(),
        RuntimeContext(),
    )
    if state.pos is None:
        raise RuntimeError("initializer produced no positions.")
    expected = (1.0e-4 * np.random.RandomState(seed).standard_normal((num_nodes, 2))).astype(
        np.float32,
        copy=False,
    )
    return float(np.max(np.abs(state.pos.detach().cpu().numpy() - expected)))


def run_smoke() -> dict[str, list[tuple[int, float, float]]]:
    """Run the Round 41 smoke matrix.

    Returns
    -------
    dict[str, list[tuple[int, float, float]]]
        Mapping from topology to ``(seed, before_rmsd, after_rmsd)`` rows.
    """
    builders: dict[str, Callable[[], tuple[torch.Tensor, int]]] = {
        "path": build_path,
        "star": build_star,
        "clustered": build_clustered,
        "grid": build_grid,
    }
    results: dict[str, list[tuple[int, float, float]]] = {}
    for name, builder in builders.items():
        edge_index, num_nodes = builder()
        rows: list[tuple[int, float, float]] = []
        for seed in SEEDS:
            reference = _layout_tsnet_sklearn_reference(
                edge_index=edge_index,
                num_nodes=num_nodes,
                node_sizes=None,
                perplexity=30.0,
                steps=STEPS,
                seed=seed,
                edge_weights=None,
            )
            before = _legacy_torch_pipeline(edge_index, num_nodes, seed)
            after = layout_tsnet_pipeline(
                edge_index=edge_index,
                num_nodes=num_nodes,
                perplexity=30.0,
                steps=STEPS,
                seed=seed,
                fidelity_mode=True,
            )
            rows.append((seed, _rmsd(before, reference), _rmsd(after, reference)))
        results[name] = rows
    return results


def main() -> None:
    """Print component diagnostics and smoke RMSD rows."""
    print("component_diagnostics:")
    for name, builder in {
        "path": build_path,
        "star": build_star,
        "clustered": build_clustered,
        "grid": build_grid,
    }.items():
        edge_index, num_nodes = builder()
        init_delta = max(_initialization_delta(num_nodes, seed) for seed in SEEDS)
        probability_delta = _probability_delta(edge_index, num_nodes)
        print(
            f"  {name}: init_max_abs={init_delta:.9g}, probability_max_abs={probability_delta:.9g}"
        )
    print("smoke_rmsd:")
    all_before: list[float] = []
    all_after: list[float] = []
    for name, rows in run_smoke().items():
        before_values = [row[1] for row in rows]
        after_values = [row[2] for row in rows]
        all_before.extend(before_values)
        all_after.extend(after_values)
        formatted = ", ".join(
            f"seed={seed}:before={before:.9f}:after={after:.9f}" for seed, before, after in rows
        )
        print(
            f"  {name}: {formatted} "
            f"(before_mean={statistics.fmean(before_values):.9f}, "
            f"after_mean={statistics.fmean(after_values):.9f})"
        )
    print(
        f"overall: before_mean={statistics.fmean(all_before):.9f}, "
        f"after_mean={statistics.fmean(all_after):.9f}, after_max={max(all_after):.9f}"
    )


if __name__ == "__main__":
    main()
