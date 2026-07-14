"""Op-level regression tests for ELK Layered stages."""

from __future__ import annotations

import pytest
import torch

from dagua.layout.ops.elk import (
    ElkAssignLayers,
    ElkBreakCycles,
    ElkMinimizeCrossings,
    ElkPrepareGraph,
    _count_order_crossings,
    _JavaRandom,
    _restart_sweep_orders,
    _shuffle_layer_orders,
)
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState


def _context() -> RuntimeContext:
    """Return a CPU runtime context for deterministic op tests.

    Returns
    -------
    RuntimeContext
        Runtime context with a CPU execution plan.
    """
    return RuntimeContext(plan=ExecutionPlan(device="cpu"))


def test_elk_cycle_breaking_makes_cycle_layerable() -> None:
    """Break a directed 3-cycle before layer assignment.

    Returns
    -------
    None
        The active edge set must become compatible with increasing layers.
    """
    problem = LayoutProblem(
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
        num_nodes=3,
        node_sizes=torch.tensor([[40.0, 20.0]] * 3),
    )
    state = ElkPrepareGraph().apply(problem, SolveState(), _context())
    state = ElkBreakCycles().apply(problem, state, _context())
    state = ElkAssignLayers().apply(problem, state, _context())

    layers = {
        node: layer for layer, nodes in enumerate(state.extras["elk_layers"]) for node in nodes
    }
    for source, target in state.extras["elk_graph"].active_edges:
        assert layers[source] < layers[target]


def test_java_random_matches_known_next_int_sequence() -> None:
    """Pin Java's 48-bit LCG for ELK seeded tie-breaking.

    Returns
    -------
    None
        The first bounded values for seed 1 must match ``java.util.Random``.
    """
    rng = _JavaRandom(1)

    assert [rng.next_int(4), rng.next_int(10), rng.next_int(10)] == [2, 8, 7]


def test_java_shuffle_matches_collections_shuffle_sequence() -> None:
    """Pin Java ``Collections.shuffle`` semantics for ELK restarts.

    Returns
    -------
    None
        Per-layer shuffles must use Java's backward Fisher-Yates order.
    """
    rng = _JavaRandom(1)

    assert _shuffle_layer_orders([[0, 1, 2, 3], [4, 5, 6]], rng) == [[3, 0, 1, 2], [5, 6, 4]]


def test_elk_restart_sweeps_keep_strictly_better_order() -> None:
    """Keep the earliest order that strictly improves crossing count.

    Returns
    -------
    None
        A crossing two-layer order should be replaced by the first sweep's
        zero-crossing order.
    """
    layers = [[0, 1], [2, 3]]
    edges = [(0, 3), (1, 2)]

    assert _count_order_crossings(layers, edges) == 1
    ordered = _restart_sweep_orders(layers, edges, random_seed=1, thoroughness=7)

    assert ordered == [[0, 1], [3, 2]]
    assert _count_order_crossings(ordered, edges) == 0


def test_elk_restart_sweeps_reorder_same_layer_isolate() -> None:
    """Match ELK's randomized ordering of a first-layer isolate.

    Returns
    -------
    None
        The disconnected graph fixture must place the isolate before the
        incident node under the default verification seed.
    """
    layers = [[0, 4], [1], [2], [3]]
    edges = [(0, 1), (2, 3)]

    ordered = _restart_sweep_orders(layers, edges, random_seed=42, thoroughness=7)

    assert ordered[0] == [4, 0]
    assert _count_order_crossings(ordered, edges) == 0


def test_elk_default_greedy_cycle_breaks_cycle_4_like_elkjs() -> None:
    """Match elkjs default GREEDY cycle-breaking on a four-cycle.

    Returns
    -------
    None
        The resulting layer bands must start at node 2 as in the cached
        elkjs reference.
    """
    problem = LayoutProblem(
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long),
        num_nodes=4,
        node_sizes=torch.tensor([[60.769920349121094, 34.0]] * 4),
    )
    state = ElkPrepareGraph().apply(problem, SolveState(), _context())
    state = ElkBreakCycles().apply(problem, state, _context())
    state = ElkAssignLayers().apply(problem, state, _context())

    layers = {
        node: layer for layer, nodes in enumerate(state.extras["elk_layers"]) for node in nodes
    }

    assert [layers[node] for node in range(4)] == [2, 3, 0, 1]


def test_elk_network_simplex_component_bands_match_elkjs_disconnected() -> None:
    """Stack disconnected non-isolate components in ELK y-band order.

    Returns
    -------
    None
        Two edge components must occupy consecutive layer bands while the
        isolate remains in the first band.
    """
    problem = LayoutProblem(
        edge_index=torch.tensor([[0, 2], [1, 3]], dtype=torch.long),
        num_nodes=5,
        node_sizes=torch.tensor([[79.53633117675781, 34.0]] * 5),
    )
    state = ElkPrepareGraph().apply(problem, SolveState(), _context())
    state = ElkBreakCycles().apply(problem, state, _context())
    state = ElkAssignLayers().apply(problem, state, _context())

    layers = {
        node: layer for layer, nodes in enumerate(state.extras["elk_layers"]) for node in nodes
    }

    assert [layers[node] for node in range(5)] == [0, 1, 2, 3, 0]


def test_elk_network_simplex_balances_random_dag_source_layers() -> None:
    """Pin the random-DAG source/balance layers that longest-path missed.

    Returns
    -------
    None
        Network-simplex balancing must move non-critical sources down to the
        cached elkjs y bands.
    """
    edges = [
        (0, 6),
        (0, 9),
        (0, 15),
        (0, 17),
        (0, 25),
        (0, 39),
        (1, 38),
        (1, 45),
        (2, 10),
        (2, 15),
        (2, 31),
        (2, 33),
        (2, 40),
        (3, 18),
        (3, 26),
        (3, 35),
        (4, 34),
        (5, 6),
        (5, 13),
        (5, 20),
        (5, 22),
        (5, 27),
        (5, 29),
        (5, 30),
        (5, 49),
        (6, 22),
        (6, 25),
        (6, 30),
        (6, 33),
        (6, 40),
        (7, 35),
        (7, 47),
        (8, 37),
        (9, 14),
        (9, 19),
        (9, 28),
        (9, 29),
        (9, 30),
        (9, 38),
        (9, 46),
        (10, 35),
        (10, 48),
        (11, 14),
        (12, 27),
        (12, 32),
        (13, 30),
        (13, 33),
        (13, 35),
        (14, 17),
        (14, 25),
        (15, 37),
        (15, 38),
        (15, 40),
        (15, 48),
        (16, 39),
        (16, 41),
        (16, 43),
        (16, 47),
        (17, 20),
        (17, 21),
        (17, 28),
        (17, 40),
        (17, 49),
        (18, 19),
        (19, 21),
        (19, 30),
        (19, 32),
        (20, 21),
        (20, 23),
        (20, 33),
        (20, 34),
        (21, 27),
        (21, 36),
        (21, 39),
        (23, 34),
        (25, 27),
        (25, 38),
        (28, 41),
        (29, 36),
        (30, 46),
        (31, 43),
        (32, 37),
        (35, 36),
        (37, 42),
        (39, 41),
        (40, 45),
        (42, 45),
        (45, 47),
        (46, 49),
        (47, 49),
    ]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    expected = [
        1,
        5,
        3,
        1,
        6,
        3,
        4,
        4,
        4,
        2,
        4,
        2,
        3,
        4,
        3,
        4,
        6,
        4,
        2,
        3,
        5,
        6,
        5,
        6,
        0,
        5,
        2,
        7,
        5,
        4,
        5,
        6,
        4,
        6,
        7,
        5,
        7,
        5,
        6,
        7,
        5,
        8,
        6,
        7,
        0,
        7,
        6,
        8,
        5,
        9,
    ]
    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=50,
        node_sizes=torch.tensor([[89.96366882324219, 34.0]] * 50),
    )
    state = ElkPrepareGraph().apply(problem, SolveState(), _context())
    state = ElkBreakCycles().apply(problem, state, _context())
    state = ElkAssignLayers().apply(problem, state, _context())
    state = ElkMinimizeCrossings().apply(problem, state, _context())
    layers = {
        node: layer for layer, nodes in enumerate(state.extras["elk_layers"]) for node in nodes
    }

    assert [layers[node] for node in range(50)] == expected


def test_elk_prepare_rejects_invalid_strategy() -> None:
    """Reject unknown ELK strategy names at construction time.

    Returns
    -------
    None
        Invalid public options must raise ``ValueError``.
    """
    with pytest.raises(ValueError, match="layering_strategy"):
        ElkPrepareGraph(layering_strategy="definitely-not-elk")


def test_elk_prepare_accepts_direction_alias() -> None:
    """Accept dagua/dagre-style direction aliases.

    Returns
    -------
    None
        ``TB`` must normalize to ELK ``DOWN``.
    """
    problem = LayoutProblem(
        edge_index=torch.empty((2, 0), dtype=torch.long),
        num_nodes=1,
        node_sizes=torch.tensor([[40.0, 20.0]]),
    )
    state = ElkPrepareGraph(direction="TB").apply(problem, SolveState(), _context())

    assert state.extras["elk_graph"].direction == "DOWN"
