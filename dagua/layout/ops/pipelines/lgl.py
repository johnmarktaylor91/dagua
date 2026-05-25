"""Large Graph Layout (LGL) pipeline."""

from __future__ import annotations

from typing import Optional

import torch

from dagua.layout.ops.base import Pipeline
from dagua.layout.ops.graph_utils import layout_device
from dagua.layout.ops.lgl import (
    LGLFinalizePositions,
    LGLInitializePositions,
    LGLInitializePositionsConfig,
    LGLLayeredRefinement,
    LGLLayeredRefinementConfig,
    LGLPrepareState,
    LGLPrepareStateConfig,
)
from dagua.layout.ops.state import (
    ExecutionPlan,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)


def build_lgl_pipeline(
    maxiter: int = 150,
    maxdelta: Optional[float] = None,
    area: Optional[float] = None,
    coolexp: float = 1.5,
    repulserad: Optional[float] = None,
    cellsize: Optional[float] = None,
    root: Optional[int] = None,
    use_edge_weights: bool = False,
    igraph_positive_maxchange: bool = True,
    fidelity_mode: bool = False,
    fidelity_dtype: torch.dtype = torch.float32,
) -> Pipeline:
    """Build a Large Graph Layout pipeline.

    Reference fidelity
    ------------------
    Targets: igraph 1.0.0 LGL / Adai et al. (2004), "LGL: Creating a Map of
        Protein Function with an Algorithm for Visualizing Very Large
        Biological Networks".
    Fidelity mode: ``fidelity_mode=True`` uses igraph's compiled default RNG
        stream, column-major initialization, shell RNG advancement, igraph grid
        constants, future-shell spring activation, and bounded-grid clamping.
    Verified at: round_33 bounded subset median RMSD 0.181326; final
        100-seed report marks LGL variants weak equivalent at median RMSD
        0.132 to 0.146.
    Known divergences:
        - The bounded subset missed the expected 0.05 to 0.08 RMSD range.
        - Default fidelity ignores edge weights; weighted behavior is retained
          as an explicit Dagua option.

    Parameters
    ----------
    maxiter : int, default=150
        Maximum refinement iterations per BFS layer.
    maxdelta : float, optional
        Maximum initial movement magnitude. Defaults to ``num_nodes``.
    area : float, optional
        Layout area. Defaults to ``num_nodes ** 2``.
    coolexp : float, default=1.5
        Power-law cooling exponent.
    repulserad : float, optional
        Modified FR repulsion cutoff. Defaults to ``area * num_nodes``.
    cellsize : float, optional
        Sparse-grid cell size. Defaults to ``area ** 0.25``.
    root : int, optional
        BFS root. When omitted, a seed-controlled random vertex is chosen.
    use_edge_weights : bool, default=False
        Whether edge weights scale attraction. Defaults to false because
        igraph LGL ignores edge weights.
    igraph_positive_maxchange : bool, default=True
        Whether to use igraph's positive-component convergence rule.
    fidelity_mode : bool, default=False
        When ``True``, use igraph's compiled default RNG stream for stochastic
        LGL draws.

    Returns
    -------
    Pipeline
        Pipeline implementing the LGL algorithm. The pipeline produces final
        node coordinates by preparing shell-growth state, initializing shell
        placements from a root breadth-first traversal, refining each BFS
        layer with sparse repulsion, and finalizing the result.

    Raises
    ------
    ValueError
        If an explicit scalar parameter is outside igraph LGL's valid range.
    """
    if maxiter < 0:
        raise ValueError("maxiter must be non-negative.")
    if maxdelta is not None and maxdelta <= 0.0:
        raise ValueError("maxdelta must be positive.")
    if area is not None and area <= 0.0:
        raise ValueError("area must be positive.")
    if coolexp <= 0.0:
        raise ValueError("coolexp must be positive.")
    if repulserad is not None and repulserad <= 0.0:
        raise ValueError("repulserad must be positive.")
    if cellsize is not None and cellsize <= 0.0:
        raise ValueError("cellsize must be positive.")

    return Pipeline(
        [
            LGLPrepareState(
                config=LGLPrepareStateConfig(
                    maxiter=maxiter,
                    maxdelta=maxdelta,
                    area=area,
                    coolexp=coolexp,
                    repulserad=repulserad,
                    cellsize=cellsize,
                    root=root,
                    use_edge_weights=use_edge_weights,
                    fidelity_mode=fidelity_mode,
                )
            ),
            LGLInitializePositions(
                config=LGLInitializePositionsConfig(fidelity_mode=fidelity_mode)
            ),
            LGLLayeredRefinement(
                config=LGLLayeredRefinementConfig(
                    igraph_positive_maxchange=igraph_positive_maxchange,
                    fidelity_mode=fidelity_mode,
                )
            ),
            LGLFinalizePositions(),
        ],
        name="lgl_pipeline",
    )


def layout_lgl_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    seed: int = 42,
    maxiter: int = 150,
    maxdelta: Optional[float] = None,
    area: Optional[float] = None,
    coolexp: float = 1.5,
    repulserad: Optional[float] = None,
    cellsize: Optional[float] = None,
    root: Optional[int] = None,
    edge_weights: Optional[torch.Tensor] = None,
    use_edge_weights: bool = False,
    igraph_positive_maxchange: bool = True,
    fidelity_mode: bool = False,
    fidelity_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Run the Large Graph Layout pipeline as a drop-in replacement.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes ``N`` in the graph.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]`` used only to resolve
        the output device.
    seed : int, default=42
        Random seed controlling root selection and shell placement.
    maxiter : int, default=150
        Maximum refinement iterations per BFS layer.
    maxdelta : float, optional
        Maximum initial movement magnitude.
    area : float, optional
        Layout area.
    coolexp : float, default=1.5
        Power-law cooling exponent.
    repulserad : float, optional
        Modified FR repulsion cutoff.
    cellsize : float, optional
        Sparse-grid cell size.
    root : int, optional
        BFS root. When omitted, a seed-controlled random vertex is chosen.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.
    use_edge_weights : bool, default=False
        Whether edge weights scale attraction. Defaults to false because
        igraph LGL ignores edge weights.
    igraph_positive_maxchange : bool, default=True
        Whether to use igraph's positive-component convergence rule.
    fidelity_mode : bool, default=False
        When ``True``, use igraph's compiled default RNG stream for stochastic
        LGL draws.

    Returns
    -------
    torch.Tensor
        Final position tensor with shape ``[N, 2]``.

    Raises
    ------
    ValueError
        If public inputs are invalid.
    RuntimeError
        If pipeline execution does not produce final positions.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E].")
    if maxiter < 0:
        raise ValueError("maxiter must be non-negative.")
    if maxdelta is not None and maxdelta <= 0.0:
        raise ValueError("maxdelta must be positive.")
    if area is not None and area <= 0.0:
        raise ValueError("area must be positive.")
    if coolexp <= 0.0:
        raise ValueError("coolexp must be positive.")
    if repulserad is not None and repulserad <= 0.0:
        raise ValueError("repulserad must be positive.")
    if cellsize is not None and cellsize <= 0.0:
        raise ValueError("cellsize must be positive.")
    if root is not None and (root < 0 or root >= num_nodes):
        raise ValueError("root must lie in [0, num_nodes).")

    if edge_weights is not None:
        if edge_weights.ndim != 1:
            raise ValueError("edge_weights must have shape [E].")
        if edge_weights.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"edge_weights length {edge_weights.shape[0]} != edge count {edge_index.shape[1]}"
            )

    if edge_index.numel() > 0:
        edge_index_cpu = edge_index.to(device="cpu", dtype=torch.long)
        min_index = int(edge_index_cpu.min().item())
        max_index = int(edge_index_cpu.max().item())
        if min_index < 0:
            raise ValueError("edge_index cannot contain negative node indices.")
        if max_index >= num_nodes:
            raise ValueError("edge_index contains node indices outside [0, num_nodes).")

    if num_nodes == 0:
        output_device = layout_device(edge_index=edge_index, node_sizes=node_sizes)
        return torch.empty((0, 2), dtype=torch.float32, device=output_device)

    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        edge_weights=edge_weights,
        seed=seed,
    )
    state = SolveState()
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))

    state.extras["lgl_root_was_random"] = root is None

    final_state = build_lgl_pipeline(
        maxiter=maxiter,
        maxdelta=maxdelta,
        area=area,
        coolexp=coolexp,
        repulserad=repulserad,
        cellsize=cellsize,
        root=root,
        use_edge_weights=use_edge_weights,
        igraph_positive_maxchange=igraph_positive_maxchange,
        fidelity_mode=fidelity_mode,
        fidelity_dtype=fidelity_dtype,
    ).apply(problem, state, ctx)

    if final_state.pos is None:
        raise RuntimeError("LGL pipeline did not produce final positions.")
    return final_state.pos


__all__ = ["build_lgl_pipeline", "layout_lgl_pipeline"]
