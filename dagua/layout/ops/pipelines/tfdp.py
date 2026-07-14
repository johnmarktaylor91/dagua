"""t-FDP layout pipeline."""

from __future__ import annotations

from typing import Optional

import torch

from dagua.layout.ops.base import Pipeline
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.tfdp import TFDPConfig, TFDPInitialize, TFDPIteration


def build_tfdp_pipeline(config: Optional[TFDPConfig] = None) -> Pipeline:
    """Build the composable t-FDP operation pipeline.

    Parameters
    ----------
    config : TFDPConfig, optional
        t-FDP parameter bundle. ``None`` uses reference defaults.

    Returns
    -------
    Pipeline
        Pipeline composed from initialization and force-iteration stages.
    """
    resolved = TFDPConfig() if config is None else config
    return Pipeline(
        [
            TFDPInitialize(resolved),
            TFDPIteration(resolved),
        ],
        name="tfdp",
    )


def layout_tfdp_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    seed: Optional[int] = None,
    steps: int = 0,
    init: str = "pmds",
    force_mode: str = "exact",
    alpha: float = 0.1,
    beta: float = 8.0,
    gamma: float = 2.0,
    max_iter: Optional[int] = None,
    pmds_pivots: int = 100,
    combine: bool = True,
    fidelity_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Run t-distributed force-directed placement.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Accepted for pipeline signature consistency; t-FDP ignores sizes.
    seed : int, optional
        Deterministic seed for initialization and iteration jitter.
    steps : int, default=0
        LayoutConfig-forwarded iteration count when ``max_iter`` is omitted.
    init : {"pmds", "random"}, default="pmds"
        Initialization variant.
    force_mode : {"exact", "fft"}, default="exact"
        Force evaluation variant. ``"fft"`` currently falls back to exact.
    alpha : float, default=0.1
        Reference long-range force parameter.
    beta : float, default=8.0
        Reference attraction parameter.
    gamma : float, default=2.0
        Reference t-force exponent.
    max_iter : int, optional
        Explicit iteration count. Defaults to ``steps`` when positive,
        otherwise the reference 300 iterations.
    pmds_pivots : int, default=100
        Pivot count used by PMDS initialization.
    combine : bool, default=True
        Retained for reference ibFFT API compatibility.
    fidelity_dtype : torch.dtype, optional
        Internal coordinate dtype. ``None`` uses reference float32.

    Returns
    -------
    torch.Tensor
        Final positions with shape ``[N, 2]``.

    Raises
    ------
    ValueError
        If an unsupported initialization or force mode is requested.
    """
    del node_sizes
    init_mode = init.lower()
    if init_mode not in {"pmds", "random"}:
        raise ValueError("init must be 'pmds' or 'random'.")
    mode = force_mode.lower()
    if mode not in {"exact", "fft"}:
        raise ValueError("force_mode must be 'exact' or 'fft'.")
    dtype = torch.float32 if fidelity_dtype is None else fidelity_dtype
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("fidelity_dtype must be torch.float32 or torch.float64.")
    resolved_iter = int(max_iter if max_iter is not None else (steps if steps > 0 else 300))
    config = TFDPConfig(
        init=init_mode,  # type: ignore[arg-type]
        force_mode=mode,  # type: ignore[arg-type]
        max_iter=resolved_iter,
        alpha=float(alpha),
        beta=float(beta),
        gamma=float(gamma),
        combine=bool(combine),
        seed=seed,
        pmds_pivots=int(pmds_pivots),
        dtype=dtype,
    )
    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        seed=0 if seed is None else seed,
    )
    state = build_tfdp_pipeline(config=config).apply(
        problem,
        SolveState(),
        RuntimeContext(plan=ExecutionPlan(device=str(edge_index.device))),
    )
    if state.pos is None:
        return torch.zeros((0, 2), dtype=dtype, device=edge_index.device)
    return state.pos.to(dtype=dtype, device=edge_index.device)


def layout_tfdp_exact_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    seed: Optional[int] = None,
    steps: int = 0,
    fidelity_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Run the exact-force t-FDP variant.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Ignored by t-FDP.
    seed : int, optional
        Deterministic seed.
    steps : int, default=0
        Optional iteration count from LayoutConfig.
    fidelity_dtype : torch.dtype, optional
        Internal coordinate dtype.

    Returns
    -------
    torch.Tensor
        Final positions with shape ``[N, 2]``.
    """
    return layout_tfdp_pipeline(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        seed=seed,
        steps=steps,
        force_mode="exact",
        fidelity_dtype=fidelity_dtype,
    )


def layout_tfdp_random_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    seed: Optional[int] = None,
    steps: int = 0,
    fidelity_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Run t-FDP with random normal initialization.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Ignored by t-FDP.
    seed : int, optional
        Deterministic seed.
    steps : int, default=0
        Optional iteration count from LayoutConfig.
    fidelity_dtype : torch.dtype, optional
        Internal coordinate dtype.

    Returns
    -------
    torch.Tensor
        Final positions with shape ``[N, 2]``.
    """
    return layout_tfdp_pipeline(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        seed=seed,
        steps=steps,
        init="random",
        force_mode="exact",
        fidelity_dtype=fidelity_dtype,
    )


__all__ = [
    "build_tfdp_pipeline",
    "layout_tfdp_exact_pipeline",
    "layout_tfdp_pipeline",
    "layout_tfdp_random_pipeline",
]
