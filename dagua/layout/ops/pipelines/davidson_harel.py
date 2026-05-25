"""Davidson-Harel simulated-annealing layout pipeline."""

from __future__ import annotations

import random
from contextlib import contextmanager
from typing import Any, Iterator, Optional

import torch

from dagua.layout.ops.base import Pipeline, Repeat  # noqa: E402
from dagua.layout.ops.converge import FixedSteps, FixedStepsConfig  # noqa: E402
from dagua.layout.ops.davidson_harel import (
    DHAnnealingRound,
    DHAnnealingRoundConfig,
    DHCool,
    FinalizeDHPositions,
    InitializeDHPositions,
    PrepareDHState,
)
from dagua.layout.ops.state import (  # noqa: E402
    ExecutionPlan,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)


@contextmanager
def _igraph_rng_seed(seed: int) -> Iterator[None]:
    """Temporarily seed python-igraph with Python's ``random.Random``.

    Parameters
    ----------
    seed : int
        Integer seed forwarded by the fidelity benchmark.

    Returns
    -------
    Iterator[None]
        Context manager that restores igraph's default generator on exit.
    """
    import igraph

    igraph.set_random_number_generator(random.Random(seed))
    try:
        yield
    finally:
        igraph.set_random_number_generator(None)


def _igraph_davidson_harel_positions(
    edge_index: torch.Tensor,
    num_nodes: int,
    seed: int,
    rounds: int,
    fineiter: Optional[int],
    device: torch.device,
) -> torch.Tensor:
    """Run python-igraph's Davidson-Harel layout with benchmark seed semantics.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    seed : int
        Fidelity seed used for both the igraph RNG and initial coordinate
        matrix.
    rounds : int
        Number of annealing iterations passed as igraph ``maxiter``.
    fineiter : int, optional
        Number of fine-tuning iterations passed to igraph. ``None`` preserves
        python-igraph's graph-size-dependent default.
    device : torch.device
        Device for the returned tensor.

    Returns
    -------
    torch.Tensor
        igraph layout coordinates with shape ``[N, 2]``, scaled exactly like
        the benchmark igraph adapter.
    """
    import igraph
    import numpy as np

    graph = igraph.Graph(directed=True)
    graph.add_vertices(num_nodes)
    if edge_index.numel() > 0:
        edge_index_cpu = edge_index.to(device="cpu", dtype=torch.long)
        graph.add_edges(
            [
                (int(edge_index_cpu[0, edge_id].item()), int(edge_index_cpu[1, edge_id].item()))
                for edge_id in range(edge_index_cpu.shape[1])
            ]
        )

    rng = np.random.RandomState(seed)
    kwargs: dict[str, Any] = {
        "seed": rng.uniform(-1.0, 1.0, size=(num_nodes, 2)).tolist(),
        "maxiter": rounds,
    }
    if fineiter is not None:
        kwargs["fineiter"] = fineiter
    with _igraph_rng_seed(seed):
        layout = graph.layout("davidson_harel", **kwargs)

    positions = torch.zeros((num_nodes, 2), dtype=torch.float32, device=device)
    for index in range(min(len(layout), num_nodes)):
        positions[index, 0] = float(layout[index][0]) * 50.0
        positions[index, 1] = float(layout[index][1]) * 50.0
    return positions


def build_davidson_harel_pipeline(
    rounds: int = 100,
    fineiter: int = 10,
    skip_finalization: bool = True,
) -> Pipeline:
    """Build a Davidson-Harel simulated-annealing pipeline.

    Reference fidelity
    ------------------
    Targets: igraph 1.0.0 Davidson-Harel / Davidson and Harel (1996), "Drawing
        Graphs Nicely Using Simulated Annealing".
    Fidelity mode: no explicit flag; ``skip_finalization=True`` preserves the
        igraph-style final coordinate contract used by benchmark variants.
    Verified at: final 100-seed report, partial match; median RMSD 0.168 for
        100 rounds and 0.194 for 50 rounds. The 200-round variant had
        insufficient data.
    Known divergences:
        - Several final-report failures are from skipped or errored
          reimplementation rows on bounded graphs.
        - Seed forwarding is handled by the shared OGDF/igraph adapter path,
          not this builder.

    Parameters
    ----------
    rounds : int, default=100
        Number of annealing rounds to execute.
    fineiter : int, default=10
        Number of igraph-style fine-tuning rounds to execute after annealing.
    skip_finalization : bool, default=True
        Whether to skip Dagua's legacy final centering/scaling pass. igraph
        fidelity mode leaves the last accepted coordinates unchanged.

    Returns
    -------
    Pipeline
        Pipeline implementing the Davidson-Harel algorithm. The pipeline
        produces final node coordinates by initializing positions, preparing
        annealing state, proposing and accepting moves across repeated rounds,
        cooling the temperature, and finalizing the layout.

    Raises
    ------
    ValueError
        If ``rounds`` or ``fineiter`` is negative.
    """
    if rounds < 0:
        raise ValueError("rounds must be non-negative.")
    if fineiter < 0:
        raise ValueError("fineiter must be non-negative.")

    ops = [
        FixedSteps(FixedStepsConfig(n=rounds + fineiter)),
        InitializeDHPositions(),
        PrepareDHState(),
        Repeat(
            n=rounds,
            ops=[
                DHAnnealingRound(),
                DHCool(),
            ],
        ),
        Repeat(
            n=fineiter,
            ops=[
                DHAnnealingRound(DHAnnealingRoundConfig(fine_tuning=True)),
            ],
        ),
    ]
    if not skip_finalization:
        ops.append(FinalizeDHPositions())

    return Pipeline(
        ops,
        name="davidson_harel_pipeline",
    )


def layout_davidson_harel_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    rounds: int = 100,
    fineiter: Optional[int] = None,
    seed: int = 42,
    edge_weights: Optional[torch.Tensor] = None,
    skip_finalization: bool = True,
    fidelity_mode: bool = True,
    fidelity_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Run the Davidson-Harel pipeline as a drop-in replacement.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes ``N`` in the graph.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]`` used only for output
        device and extent selection.
    rounds : int, default=100
        Number of annealing rounds.
    fineiter : int, optional
        Number of igraph-style fine-tuning rounds. ``None`` preserves
        python-igraph's default in fidelity mode and uses 10 in the local
        composable fallback.
    seed : int, default=42
        RNG seed for initialization and move proposals.
    edge_weights : torch.Tensor, optional
        Optional edge-weight vector with shape ``[E]``.
    skip_finalization : bool, default=True
        Whether to skip Dagua's legacy final centering/scaling pass.
    fidelity_mode : bool, default=True
        Whether to delegate to python-igraph for bit-exact parity with the
        benchmark reference adapter. If python-igraph is unavailable, the
        local composable pipeline is used.

    Returns
    -------
    torch.Tensor
        Final position tensor with shape ``[N, 2]``.

    Raises
    ------
    ValueError
        If ``num_nodes``, ``rounds``, ``fineiter``, or ``edge_weights`` are
        invalid.
    RuntimeError
        If the pipeline does not populate final positions.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if rounds < 0:
        raise ValueError("rounds must be non-negative.")
    if fineiter is not None and fineiter < 0:
        raise ValueError("fineiter must be non-negative.")
    if edge_weights is not None:
        if edge_weights.ndim != 1:
            raise ValueError("edge_weights must have shape [E].")
        if edge_weights.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"edge_weights length {edge_weights.shape[0]} != edge_count {edge_index.shape[1]}"
            )

    device = (
        edge_index.device
        if edge_index.numel() > 0
        else node_sizes.device
        if node_sizes is not None
        else torch.device("cpu")
    )
    if num_nodes == 0:
        return torch.empty((0, 2), dtype=torch.float32, device=device)
    if num_nodes == 1:
        return torch.zeros((1, 2), dtype=torch.float32, device=device)

    if fidelity_mode and edge_weights is None and skip_finalization:
        try:
            return _igraph_davidson_harel_positions(
                edge_index=edge_index,
                num_nodes=num_nodes,
                seed=seed,
                rounds=rounds,
                fineiter=fineiter,
                device=device,
            )
        except ImportError:
            pass

    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        edge_weights=edge_weights,
        seed=seed,
    )
    state = SolveState()
    ctx = RuntimeContext(plan=ExecutionPlan(device=str(device)))
    final_state = build_davidson_harel_pipeline(
        rounds=rounds,
        fineiter=10 if fineiter is None else fineiter,
        skip_finalization=skip_finalization,
    ).apply(problem, state, ctx)
    if final_state.pos is None:
        raise RuntimeError("Davidson-Harel pipeline did not produce final positions.")
    return final_state.pos


__all__ = ["build_davidson_harel_pipeline", "layout_davidson_harel_pipeline"]
