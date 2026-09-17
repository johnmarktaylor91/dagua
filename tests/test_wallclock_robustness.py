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


@pytest.mark.slow
def test_native_undirected_output_is_load_invariant() -> None:
    """Undirected-route positions are byte-identical under extreme starvation.

    WP13-F02: the only full-pipeline starved-vs-idle byte-equality row was
    the directed wide DAG ``wide_3_50_3``, so a surviving or future
    measured-time branch specific to the undirected marketplace (the
    fCoSE/tsNET/FR/geodesic/mesh/community/arm-S contest in
    ``native_undirected.py``) would have passed this file unnoticed.
    ``real_football_115`` is an undirected corpus row empirically verified
    to enter the undirected marketplace (contest log shows
    incumbent/sfdp/neato_prism/geodesic_stress_prism/tsnet candidates and
    the W5 telemetry reports ``is_semantically_directed: false``); its
    positions must be byte-identical between real clocks and the starved
    simulation, mirroring the directed ``wide_3_50_3`` pin.
    """
    baseline = _run_certified_native("real_football_115")
    with _starved_time():
        starved_pos = _run_certified_native("real_football_115")
    assert starved_pos.shape == baseline.shape
    assert starved_pos.numpy().tobytes() == baseline.numpy().tobytes()


def test_deterministic_measurement_flag_contract() -> None:
    """The seam's deterministic-measurement declaration is explicit and enforced.

    WP13-F01: determinism is carried by two config facts -- a DWU ledger is
    installed AND the wall-deadline attribute ``_dagua_native_deadline_s``
    is absent. ``_dagua_native_deterministic_measurement`` declares that
    intent at the seam. This pins (a) the certified seam shape has no wall
    deadline to consult (``remaining_wall_s`` is None), and (b) the native
    pipeline entry refuses a config that declares the flag while also
    carrying a wall deadline, so the flag can no longer silently drift from
    the mechanism it documents.
    """
    from dagua.layout.ops.pipelines.dagua_native import layout_dagua_native_pipeline
    from dagua.layout.ops.pipelines.native_budget import remaining_wall_s

    config = LayoutConfig(device="cpu", verbose=False, seed=42)
    setattr(config, "_dagua_native_deterministic_measurement", True)
    install_budget_ledger(config, 1800.0, return_reserve_dwu=5.0)
    assert remaining_wall_s(config) is None

    setattr(config, "_dagua_native_deadline_s", 1e9)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    with pytest.raises(ValueError, match="deterministic"):
        layout_dagua_native_pipeline(
            edge_index=edge_index,
            num_nodes=3,
            node_sizes=torch.ones((3, 2), dtype=torch.float32),
            config=config,
            device="cpu",
            seed=42,
        )


def test_scale_anytime_wrapper_stale_flag_does_not_disarm_native() -> None:
    """Drywell R1 F-1 regression: a stale seam flag must not disarm the scale arm.

    The FROZEN scale anytime-native wrapper (``_run_budgeted_native`` in
    ``dagua/layout/scale/coarsest.py``) shallow-copies the user config --
    which on the deterministic seam carries
    ``_dagua_native_deterministic_measurement=True`` inherited through the
    engine's shallow copies -- then installs a wall deadline plus an anytime
    ledger. The original WP-21 guard raised on that shape and the wrapper's
    blanket ``except Exception`` swallowed the error, silently disarming the
    anytime-native coarsest arm above the scale gate (stress fallback always
    used). The narrowed guard must let this exact end-to-end shape through:
    the wrapper call WITH the flag returns positions, and the caller's
    config keeps its flag untouched.
    """
    from dagua.layout.graph_classify import classify_graph
    from dagua.layout.ops.state import LayoutProblem
    from dagua.layout.scale.coarsest import _run_budgeted_native, _score_v3_position

    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    n = 5
    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=n,
        node_sizes=torch.full((n, 2), 20.0),
        seed=42,
        structure=classify_graph(edge_index, n),
    )
    fallback_pos = torch.stack(
        (torch.arange(n, dtype=torch.float32) * 40.0, torch.zeros(n, dtype=torch.float32)),
        dim=1,
    )
    fallback_score = _score_v3_position(problem, fallback_pos)

    flagged = LayoutConfig(device="cpu", verbose=False, seed=42)
    setattr(flagged, "_dagua_native_deterministic_measurement", True)
    pos = _run_budgeted_native(
        problem,
        flagged,
        deadline_s=time.perf_counter() + 30.0,
        seed=42,
        fallback_pos=fallback_pos,
        fallback_score=fallback_score,
    )

    assert pos is not None, "anytime-native coarsest arm was disarmed by the stale flag"
    assert pos.shape == (n, 2)
    assert bool(torch.isfinite(pos).all())
    assert getattr(flagged, "_dagua_native_deterministic_measurement") is True


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


def _budget_binding_ordering_input() -> tuple[torch.Tensor, torch.Tensor]:
    """Build a narrow-band incumbent whose ordering search exceeds the ledger.

    A single wide rank of shuffled nodes over many edges gives the arm a
    natural trial volume whose pair-check work exceeds the 5M deterministic
    budget, so the ledger truncates mid-search. The construction is a pure
    function of constants.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Incumbent positions with shape ``[N, 2]`` and edge tensor ``[2, E]``.
    """
    top, bottom = 30, 60
    num_nodes = top + bottom
    edges: list[tuple[int, int]] = []
    for src in range(top):
        for spread in range(5):
            dst = top + ((src * 17 + spread * 23 + 7) % bottom)
            edges.append((src, dst))
    edge_index = torch.tensor(sorted(set(edges)), dtype=torch.long).t().contiguous()
    pos = torch.zeros((num_nodes, 2), dtype=torch.float32)
    for src in range(top):
        pos[src, 0] = float((src * 13 + 5) % top) * 10.0
        pos[src, 1] = 0.0
    for offset in range(bottom):
        pos[top + offset, 0] = float((offset * 7 + 3) % bottom) * 5.0
        pos[top + offset, 1] = -40.0
    return pos, edge_index


def test_ordering_ledger_truncation_is_reproducible() -> None:
    """A ledger-truncated ordering search is byte-identical across runs.

    NOTE on the criterion: the deterministic pair-check ledger CAN truncate
    the narrow-band ordering search at a different logical trial than
    certified-HEAD's old 1.5s/2.5s wall caps did. This was audited
    corpus-wide during the wall-clock-robustness re-cert: the ledger binds
    on exactly three 121-corpus rows (dependency_graph_100,
    r79_weighted_skew_dag_6x10, random_dag_50), and ALL corpus rows --
    including those three and every other arm-entering row -- were
    empirically confirmed byte-identical to certified-HEAD 16727c51 when
    idle (the truncated arm candidate never changed the contest winner), so
    idle byte-parity to certified-HEAD is the CONFIRMED baseline criterion.
    Because HEAD artifacts are not available at test time, this test pins
    the property that keeps that confirmation stable: the ledger must
    truncate at the SAME logical trial on every run, on every machine,
    under any load.
    """
    from dagua.layout.ops.pipelines.native_directed import (
        _rank_local_zero_crossing_swap_candidate,
    )

    pos, edge_index = _budget_binding_ordering_input()
    first = _rank_local_zero_crossing_swap_candidate(pos, edge_index, max_passes=3, config=None)
    with _starved_time():
        second = _rank_local_zero_crossing_swap_candidate(
            pos, edge_index, max_passes=3, config=None
        )
    third = _rank_local_zero_crossing_swap_candidate(pos, edge_index, max_passes=3, config=None)
    assert first.numpy().tobytes() == second.numpy().tobytes()
    assert first.numpy().tobytes() == third.numpy().tobytes()
    assert not torch.equal(first, pos)


@pytest.mark.slow
def test_ledger_binding_row_output_is_repeatable() -> None:
    """A ledger-binding corpus row produces byte-identical repeated idle runs.

    ``dependency_graph_100`` is one of the exactly three 121-corpus rows on
    which the deterministic ordering ledger binds mid-search during an
    ordinary certified-seam layout (audit: enumeration over all directed
    corpus rows, 2026-08-04). Its final layout was confirmed byte-identical
    to certified-HEAD 16727c51; this test pins the end-to-end determinism of
    the full native default pipeline on such a row: repeated idle runs must
    produce identical bytes.
    """
    first = _run_certified_native("dependency_graph_100")
    second = _run_certified_native("dependency_graph_100")
    assert first.numpy().tobytes() == second.numpy().tobytes()
