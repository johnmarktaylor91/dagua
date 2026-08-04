"""Wall-clock robustness: certified native output must not depend on load.

The certified native default layout historically consulted measured time
(wall-clock deadlines, per-candidate wall guards, measured step sizing) on
its default path, so concurrent CPU load changed the candidate set and the
finisher work plan, producing different positions. These tests pin the
load-invariance contract: the deterministic default path must produce
byte-identical positions no matter how slowly the clock appears to move.

The starved runs simulate extreme CPU starvation by patching ``time``'s
clocks to jump forward by seconds on every read — a strictly harsher
environment than any real CPU contention. Positions must match the
unstarved run exactly.
"""

from __future__ import annotations

import random
import time
from contextlib import contextmanager
from typing import Callable, Iterator

import pytest
import torch

from dagua.config import LayoutConfig
from dagua.layout.ops.pipelines.native_budget import install_budget_ledger
from dagua.layout.ops.pipelines.native_directed import (
    DIRECTED_ORDERING_DETERMINISTIC_PAIR_BUDGET,
    _ordering_budget_available,
    _OrderingWorkBudget,
)
from dagua.layout.ops.pipelines.native_undirected import (
    POLISH_DETERMINISTIC_EDGE_CAP,
    POLISH_DETERMINISTIC_NODE_CAP,
    _polish_generation_admitted,
)

# Each clock read jumps 30s: one generation span then appears to exceed any
# historical wall guard (the removed per-candidate guard was 25s), so every
# surviving measured-time branch fires under this simulation.
_STARVATION_JUMP_S = 30.0


def _starved_clock(start: float) -> Callable[[], float]:
    """Return a fake clock that jumps forward on every read.

    Parameters
    ----------
    start : float
        Initial clock value.

    Returns
    -------
    Callable[[], float]
        Monotonically increasing clock advancing ``_STARVATION_JUMP_S`` per
        call, simulating a machine so loaded that seconds elapse between any
        two adjacent statements.
    """
    state = {"now": float(start)}

    def read() -> float:
        state["now"] += _STARVATION_JUMP_S
        return state["now"]

    return read


@contextmanager
def _starved_time() -> Iterator[None]:
    """Patch all ``time`` clocks the native path consults to starved clocks.

    Yields
    ------
    None
        Control while ``time.monotonic``, ``time.perf_counter`` and
        ``time.process_time`` race forward by seconds per read.
    """
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(time, "monotonic", _starved_clock(time.monotonic()))
        patch.setattr(time, "perf_counter", _starved_clock(time.perf_counter()))
        patch.setattr(time, "process_time", _starved_clock(time.process_time()))
        yield


def _run_certified_native(graph_name: str) -> torch.Tensor:
    """Run the certified deterministic-native seam on one corpus row.

    This mirrors the certification regen seam: CPU device, seed 42,
    ``--deterministic-native`` ledger semantics with the certified
    ``--timeout 1800`` deterministic work-unit budget.

    Parameters
    ----------
    graph_name : str
        Corpus row name from :func:`dagua.eval.graphs.get_test_graphs`.

    Returns
    -------
    torch.Tensor
        CPU float32 positions with shape ``[N, 2]``.
    """
    from dagua.eval.graphs import get_test_graphs
    from dagua.layout import layout

    graph = next(tg.graph for tg in get_test_graphs() if tg.name == graph_name)
    graph.compute_node_sizes()
    config = LayoutConfig(device="cpu", verbose=False, seed=42)
    setattr(config, "_dagua_native_deterministic_measurement", True)
    install_budget_ledger(config, 1800.0, return_reserve_dwu=5.0)
    random.seed(42)
    torch.manual_seed(42)
    pos = layout(graph, config)
    return pos.detach().to(device="cpu", dtype=torch.float32)


@pytest.mark.slow
def test_native_default_output_is_load_invariant() -> None:
    """Certified native positions are byte-identical under extreme starvation.

    The baseline run uses real clocks; the starved run sees every clock jump
    by seconds per read. Any surviving wall-clock branch on the default path
    would starve the finisher or drop candidates and change the output, so
    byte equality here is the load-invariance regression contract from the
    118/121 false regression scare. The pinned row ``wide_3_50_3`` is the
    empirically bug-reproducing fixture: at certified-HEAD ``16727c51`` this
    test FAILS on it (max position delta ~= 11 under the starved clock,
    driven by the ordering arm's 1.5s/2.5s wall caps and the 25s polish
    candidate wall guards); after the deterministic-budget migration it must
    pass forever.
    """
    baseline = _run_certified_native("wide_3_50_3")
    with _starved_time():
        starved_pos = _run_certified_native("wide_3_50_3")
    assert starved_pos.shape == baseline.shape
    assert starved_pos.numpy().tobytes() == baseline.numpy().tobytes()


def test_ordering_budget_is_clock_free(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ordering-arm budget ignores elapsed time entirely."""
    monkeypatch.setattr(time, "perf_counter", _starved_clock(time.perf_counter()))
    monkeypatch.setattr(time, "monotonic", _starved_clock(time.monotonic()))
    budget = _OrderingWorkBudget(DIRECTED_ORDERING_DETERMINISTIC_PAIR_BUDGET)
    for _ in range(64):
        assert _ordering_budget_available(None, budget)
    budget.spend(DIRECTED_ORDERING_DETERMINISTIC_PAIR_BUDGET)
    assert budget.exhausted()
    assert not _ordering_budget_available(None, budget)


def test_polish_admission_is_size_only() -> None:
    """Polish candidate admission is a pure function of graph size."""
    assert _polish_generation_admitted(40, 64)
    assert _polish_generation_admitted(POLISH_DETERMINISTIC_NODE_CAP, POLISH_DETERMINISTIC_EDGE_CAP)
    assert not _polish_generation_admitted(POLISH_DETERMINISTIC_NODE_CAP + 1, 0)
    assert not _polish_generation_admitted(0, POLISH_DETERMINISTIC_EDGE_CAP + 1)
