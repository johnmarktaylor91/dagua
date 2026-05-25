"""tsNET layout pipeline."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from dagua.layout.ops.base import Pipeline, Repeat  # noqa: E402
from dagua.layout.ops.converge import FixedSteps, FixedStepsConfig  # noqa: E402
from dagua.layout.ops.graph_utils import layout_device
from dagua.layout.ops.state import (  # noqa: E402
    ExecutionPlan,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)
from dagua.layout.ops.tsnet import (  # noqa: E402
    TsnetFinalizePositions,
    TsnetGradientStep,
    TsnetInitializeOptimizer,
    TsnetInitializePositions,
    TsnetInitializePositionsConfig,
    TsnetPrepareState,
)


def _layout_tsnet_sklearn_reference(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor],
    perplexity: float,
    steps: int,
    seed: int,
    edge_weights: Optional[torch.Tensor],
    fidelity_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Run the sklearn exact t-SNE reference path for fidelity mode.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]`` used only to choose the
        output device.
    perplexity : float
        Target t-SNE perplexity.
    steps : int
        Maximum sklearn optimization iterations.
    seed : int
        Random seed forwarded to ``sklearn.manifold.TSNE``.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.
    fidelity_dtype : torch.dtype, default=torch.float32
        Internal dtype for the precomputed distance matrix.

    Returns
    -------
    torch.Tensor
        Reference coordinates with shape ``[N, 2]`` on the layout device.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path
    from sklearn.manifold import TSNE

    device = layout_device(edge_index, node_sizes)
    if num_nodes == 0:
        return torch.empty((0, 2), dtype=torch.float32, device=device)
    if num_nodes == 1:
        return torch.zeros((1, 2), dtype=torch.float32, device=device)

    edge_index_cpu = edge_index.detach().to(device="cpu", dtype=torch.long)
    if edge_index_cpu.numel() == 0:
        rows = np.empty(0, dtype=np.int64)
        cols = np.empty(0, dtype=np.int64)
    else:
        edge_index_np = edge_index_cpu.numpy()
        rows = np.concatenate([edge_index_np[0], edge_index_np[1]])
        cols = np.concatenate([edge_index_np[1], edge_index_np[0]])
    if edge_weights is None:
        np_dtype = np.float64 if fidelity_dtype is torch.float64 else np.float32
        data = np.ones(rows.shape[0], dtype=np_dtype)
    else:
        torch_dtype = torch.float64 if fidelity_dtype is torch.float64 else torch.float32
        weights = edge_weights.detach().to(device="cpu", dtype=torch_dtype).numpy()
        data = np.concatenate([weights, weights]).astype(
            np.float64 if fidelity_dtype is torch.float64 else np.float32,
            copy=False,
        )

    adjacency = csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
    distances = shortest_path(adjacency, directed=False)
    finite_mask = np.isfinite(distances)
    max_finite = float(np.max(distances[finite_mask])) if np.any(finite_mask) else 1.0
    fill_value = max(max_finite * 2.0, 1.0)
    dense_distances = np.where(np.isinf(distances), fill_value, distances).astype(
        np.float64 if fidelity_dtype is torch.float64 else np.float32,
        copy=False,
    )

    estimator = TSNE(
        n_components=2,
        metric="precomputed",
        init="random",
        random_state=seed,
        perplexity=min(float(perplexity), float(num_nodes - 1)),
        method="exact",
        max_iter=max(int(steps), 250),
    )
    coordinates = estimator.fit_transform(dense_distances)
    return torch.tensor(coordinates, dtype=torch.float32, device=device)


def build_tsnet_pipeline(
    steps: int = 1000,
    fidelity_mode: bool = False,
    fidelity_dtype: torch.dtype = torch.float32,
) -> Pipeline:
    """Build a tsNET layout pipeline.

    Reference fidelity
    ------------------
    Targets: scikit-learn 1.8.0 t-SNE graph adapter / van der Maaten and
        Hinton (2008), "Visualizing Data using t-SNE".
    Fidelity mode: ``fidelity_mode=True`` in the public wrapper routes through
        sklearn's exact t-SNE implementation; this builder still exposes the
        native torch composition for direct pipeline tests and diagnostics.
    Verified at: round_32 bounded subset median RMSD 0.398822; final
        100-seed report marks TSNET variants partial match at median RMSD
        0.151 to 0.276.
    Known divergences:
        - The native torch composition remains close but not bit-exact because
          sklearn uses SciPy/NumPy condensed-distance probability and gradient
          kernels plus its own two-call optimizer loop.
        - The Round 31 ``c=4`` gradient-scale hypothesis was reverted after
          direct gradient parity checks.

    Parameters
    ----------
    steps : int, default=1000
        Number of optimization updates.
    fidelity_mode : bool, default=False
        Preserve native sklearn-diagnostic settings when this builder is used
        directly.
    fidelity_dtype : torch.dtype, default=torch.float32
        Internal dtype used only when ``fidelity_mode`` is enabled.

    Returns
    -------
    Pipeline
        Pipeline implementing the tsNET algorithm. The pipeline produces final
        node coordinates by initializing positions, preparing t-SNE-style
        affinities, creating the optimizer state, applying repeated
        gains-and-momentum gradient steps, and finalizing the embedding.

    Raises
    ------
    ValueError
        If ``steps`` is negative.
    """
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    if fidelity_dtype not in (torch.float32, torch.float64):
        raise ValueError("fidelity_dtype must be torch.float32 or torch.float64.")
    dtype = fidelity_dtype if fidelity_mode else torch.float32

    return Pipeline(
        [
            FixedSteps(FixedStepsConfig(n=steps)),
            TsnetInitializePositions(
                TsnetInitializePositionsConfig(fidelity_mode=fidelity_mode, dtype=dtype)
            ),
            TsnetPrepareState(),
            TsnetInitializeOptimizer(),
            Repeat(
                n=steps,
                ops=[
                    TsnetGradientStep(),
                ],
            ),
            TsnetFinalizePositions(),
        ],
        name="tsnet_pipeline",
    )


def layout_tsnet_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    perplexity: float = 30,
    steps: int = 1000,
    seed: int = 42,
    edge_weights: Optional[torch.Tensor] = None,
    fidelity_mode: bool = False,
    fidelity_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Run the tsNET pipeline as a drop-in replacement.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes ``N`` in the graph.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]`` used for final
        scaling.
    perplexity : float, default=30
        Target t-SNE perplexity. Currently only the default value of 30
        preserves bit-identity with classic; non-default values require
        extending ``TsnetPrepareState``.
    steps : int, default=1000
        Number of optimization updates.
    seed : int, default=42
        Random seed for the torch generator initialization.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.
    fidelity_mode : bool, default=False
        Route through sklearn's exact t-SNE reference when ``True``.
    fidelity_dtype : torch.dtype, default=torch.float32
        Internal dtype used only when ``fidelity_mode`` is enabled.

    Returns
    -------
    torch.Tensor
        Final position tensor with shape ``[N, 2]``.

    Raises
    ------
    ValueError
        If ``num_nodes``, ``steps``, ``perplexity``, or ``edge_weights``
        are invalid.
    RuntimeError
        If the pipeline fails to populate final positions.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if perplexity <= 0:
        raise ValueError("perplexity must be positive.")
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    if fidelity_dtype not in (torch.float32, torch.float64):
        raise ValueError("fidelity_dtype must be torch.float32 or torch.float64.")
    if edge_weights is not None:
        if edge_weights.ndim != 1:
            raise ValueError("edge_weights must have shape [E].")
        if edge_weights.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"edge_weights length {edge_weights.shape[0]} != edge count {edge_index.shape[1]}"
            )

    if fidelity_mode:
        return _layout_tsnet_sklearn_reference(
            edge_index=edge_index,
            num_nodes=num_nodes,
            node_sizes=node_sizes,
            perplexity=perplexity,
            steps=steps,
            seed=seed,
            edge_weights=edge_weights,
            fidelity_dtype=fidelity_dtype,
        )

    device = layout_device(edge_index, node_sizes)

    # Handle trivial cases exactly like classic.
    if num_nodes == 0:
        return torch.empty((0, 2), dtype=torch.float32, device=device)
    if num_nodes == 1:
        return torch.zeros((1, 2), dtype=torch.float32, device=device)

    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        edge_weights=edge_weights,
        seed=seed,
    )
    state = SolveState()
    state.extras["tsnet_perplexity"] = perplexity
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))
    final_state = build_tsnet_pipeline(
        steps=steps,
        fidelity_mode=fidelity_mode,
        fidelity_dtype=fidelity_dtype,
    ).apply(
        problem,
        state,
        ctx,
    )
    if final_state.pos is None:
        raise RuntimeError("tsNET pipeline did not produce final positions.")
    return final_state.pos.to(dtype=torch.float32)


__all__ = ["build_tsnet_pipeline", "layout_tsnet_pipeline"]
