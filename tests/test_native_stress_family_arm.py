"""Tests for the stress-family contest arms (sprint2 W2-2)."""

from __future__ import annotations

import hashlib
import importlib
import json
import time
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from dagua.config import LayoutConfig
from dagua.layout.graph_classify import classify_graph
from dagua.layout.ops.pipelines.native_stress_family_arm import (
    LOW_LAYERING_MIN_AVG_LAYER_WIDTH,
    MAXENT_SEED_BANK,
    STRESS_FAMILY_MAX_NODES,
    STRESS_SGD_SEED_BANK,
    _is_connected,
    build_stress_family_candidates,
    stress_family_arm_admitted,
    stress_family_candidate_prefix,
    stress_family_directed_admitted,
    stress_family_parity_floor,
)

# The F2/F3 fixup APIs are imported inside their tests so this file still
# collects at a5e8fff5, letting the per-finding regressions fail there
# individually (the fail-on-old proof) instead of erroring the whole module.
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState


def _problem(edges: list[tuple[int, int]], num_nodes: int) -> LayoutProblem:
    """Return a LayoutProblem with real classifier output for the edges."""
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    structure = classify_graph(edge_index, num_nodes)
    return LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=torch.full((num_nodes, 2), 20.0),
        structure=cast(Any, structure),
        seed=42,
    )


def _chords_problem() -> LayoutProblem:
    """Return a small connected undirected-style row (gate open)."""
    edges = [(i, i + 1) for i in range(11)] + [(0, 5), (3, 9), (2, 7)]
    return _problem(edges, 12)


def _wide_dag_problem() -> LayoutProblem:
    """Return a shallow two-layer DAG (K3,3 orientation: avg width 3.0)."""
    return _problem([(u, v) for u in range(3) for v in range(3, 6)], 6)


def _chain_dag_problem() -> LayoutProblem:
    """Return a deep chain DAG (avg layer width 1.0: directed gate closed)."""
    return _problem([(i, i + 1) for i in range(7)], 8)


def _disjoint_k5s_problem() -> LayoutProblem:
    """Return two disjoint K5s: the classifier component fast path lies here."""
    edges = [(u, v) for u in range(5) for v in range(u + 1, 5)]
    edges += [(5 + u, 5 + v) for u in range(5) for v in range(u + 1, 5)]
    return _problem(edges, 10)


def _k5_tournament_problem() -> LayoutProblem:
    """Return a K5 tournament: connected but deep chain-like layering."""
    return _problem([(u, v) for u in range(5) for v in range(u + 1, 5)], 5)


def _disjoint_wide_dags_problem() -> LayoutProblem:
    """Return two disjoint K3,3 orientations: wide layering, disconnected.

    Dense enough (18 edges > 11) that the classifier's component fast path
    reports connected; the exact union-find gate must still fail closed.
    Also non-planar (K3,3), so the row is closed for the W1-B arm too.
    """
    edges = [(u, v) for u in range(3) for v in range(3, 6)]
    edges += [(6 + u, 6 + v) for u in range(3) for v in range(3, 6)]
    return _problem(edges, 12)


def _sha256(tensor: torch.Tensor) -> str:
    """Return the SHA-256 of a tensor's float32 CPU bytes."""
    return hashlib.sha256(
        tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy().tobytes()
    ).hexdigest()


def test_is_connected_exact_on_dense_disconnected_inputs() -> None:
    """Union-find sees through the classifier's E > N-1 component fast path."""
    assert _is_connected(_chords_problem().edge_index, 12)
    assert not _is_connected(_disjoint_k5s_problem().edge_index, 10)
    assert not _is_connected(torch.tensor([(0, 1)], dtype=torch.long).t(), 3)
    assert not _is_connected(torch.zeros((2, 0), dtype=torch.long), 2)


def test_gate_opens_on_small_connected_rows() -> None:
    """The shared gate admits small connected rows inside the band."""
    assert stress_family_arm_admitted(_chords_problem())
    assert stress_family_arm_admitted(_wide_dag_problem())


def test_gate_closes_on_disconnected_missing_structure_and_size() -> None:
    """Disconnected, unclassified, tiny, and oversized rows all fail closed."""
    assert not stress_family_arm_admitted(_disjoint_k5s_problem())
    sparse_disconnected = _problem([(0, 1), (1, 2)], 5)
    assert not stress_family_arm_admitted(sparse_disconnected)
    triangle = _problem([(0, 1), (1, 2), (0, 2)], 3)
    assert not stress_family_arm_admitted(triangle)
    unclassified = _chords_problem()
    unclassified.structure = None
    assert not stress_family_arm_admitted(unclassified)
    band_break = STRESS_FAMILY_MAX_NODES + 1
    long_path = _problem([(i, i + 1) for i in range(band_break - 1)], band_break)
    assert not stress_family_arm_admitted(long_path)


def test_directed_gate_fitted_semantics() -> None:
    """The cousin-fitted gate admits every measured layering, fails closed
    on unmeasured width, and bypasses the width check for cyclic digraphs.

    The training-cousin fit (tests/data/w22_layering_fit.json) found
    competitive stress rows down to avg width 1.0 -- deep chains included --
    so chains and tournaments are ADMITTED, not closed (review F4: the
    earlier 2.0 cut rejected cousin-supported competitive turf).
    """
    wide = _wide_dag_problem()
    assert float(getattr(wide.structure, "avg_layer_width", 0.0)) >= (
        LOW_LAYERING_MIN_AVG_LAYER_WIDTH
    )
    assert stress_family_directed_admitted(wide)
    chain = _chain_dag_problem()
    assert float(getattr(chain.structure, "avg_layer_width", 0.0)) == 1.0
    assert stress_family_directed_admitted(chain)
    assert stress_family_directed_admitted(_k5_tournament_problem())
    from types import SimpleNamespace

    unmeasured = _wide_dag_problem()
    unmeasured.structure = cast(
        Any,
        SimpleNamespace(is_directed_acyclic=True, avg_layer_width=0.0),
    )
    assert not stress_family_directed_admitted(unmeasured)
    cyclic = _problem([(0, 1), (1, 2), (2, 3), (3, 0)], 4)
    assert not bool(getattr(cyclic.structure, "is_directed_acyclic", True))
    assert stress_family_directed_admitted(cyclic)


def test_directed_gate_boundary_at_frozen_layering_cut() -> None:
    """Boundary behavior around the frozen cousin-fitted cut (review F4).

    The gate admits exactly at the cut (``>=``), rejects epsilon below it,
    and bypasses the width check for cyclic digraphs (no faithful layering
    exists). The cut itself is established from the training-cousin table
    (scripts/w22_cousin_layering_fit.py, artifact
    tests/data/w22_layering_fit.json); dev63 is confirmation only.
    """
    from types import SimpleNamespace

    problem = _wide_dag_problem()
    problem.structure = cast(
        Any,
        SimpleNamespace(
            is_directed_acyclic=True,
            avg_layer_width=LOW_LAYERING_MIN_AVG_LAYER_WIDTH,
        ),
    )
    assert stress_family_directed_admitted(problem)
    problem.structure = cast(
        Any,
        SimpleNamespace(
            is_directed_acyclic=True,
            avg_layer_width=LOW_LAYERING_MIN_AVG_LAYER_WIDTH - 1e-9,
        ),
    )
    assert not stress_family_directed_admitted(problem)
    problem.structure = cast(
        Any,
        SimpleNamespace(is_directed_acyclic=False, avg_layer_width=0.0),
    )
    assert stress_family_directed_admitted(problem)


def test_builder_emits_expected_candidates_deterministically() -> None:
    """The builder emits the frozen candidate set, byte-identical across runs."""
    problem = _chords_problem()
    first = build_stress_family_candidates(problem, node_sep=40.0)
    second = build_stress_family_candidates(problem, node_sep=40.0)
    expected = {f"stress_sgd_k_seed{seed}" for seed in STRESS_SGD_SEED_BANK}
    expected |= {f"maxent_stress_seed{seed}" for seed in MAXENT_SEED_BANK}
    expected.add("elk_stress_arm")
    assert set(first) == expected
    assert set(second) == expected
    for name, pos in first.items():
        assert pos.shape == (12, 2)
        assert bool(torch.isfinite(pos).all().item())
        assert torch.equal(pos, second[name]), name


def test_candidate_names_never_claim_sgd2() -> None:
    """D2 staged call: the arm is stress_sgd_k, never conflated with sgd2."""
    problem = _chords_problem()
    candidates = build_stress_family_candidates(problem, node_sep=40.0)
    for name in candidates:
        assert "sgd2" not in name, name
        assert stress_family_candidate_prefix(name) is not None, name
    parity = [name for name in candidates if stress_family_parity_floor(name)]
    assert sorted(parity) == [
        "elk_stress_arm",
        f"maxent_stress_seed{MAXENT_SEED_BANK[0]}",
        f"stress_sgd_k_seed{STRESS_SGD_SEED_BANK[0]}",
    ]


def test_candidates_rescaled_into_node_units() -> None:
    """Unit-scale pipeline output is similarity-rescaled for the contest.

    stress-SGD natively emits ~unit-length edges, which the shared degeneracy
    guard would reject against 20-point node boxes (the W1-A t-FDP lesson).
    After the rescale the median edge must sit at the node-box diagonal plus
    node_sep, far above the raw unit scale.
    """
    problem = _chords_problem()
    candidates = build_stress_family_candidates(problem, node_sep=40.0)
    pos = candidates[f"stress_sgd_k_seed{STRESS_SGD_SEED_BANK[0]}"]
    src, dst = problem.edge_index[0], problem.edge_index[1]
    median_edge = float((pos[src] - pos[dst]).norm(dim=1).median().item())
    expected = float(torch.full((2,), 20.0).norm().item()) + 40.0
    assert median_edge == pytest.approx(expected, rel=0.05)


def test_marketplace_family_labels() -> None:
    """Registered variants roll up to the three declared telemetry families."""
    from dagua.layout.ops.pipelines.native_undirected import _marketplace_family

    assert _marketplace_family("stress_sgd_k_seed42") == "stress_sgd_k"
    assert _marketplace_family("stress_sgd_k_seed1379_raw") == "stress_sgd_k"
    assert _marketplace_family("maxent_stress_seed7_prism") == "maxent_stress"
    assert _marketplace_family("elk_stress_arm_convergent") == "elk_stress_arm"


def test_gate_closed_undirected_row_never_builds(monkeypatch: pytest.MonkeyPatch) -> None:
    """On a disconnected row the arm code never runs (byte-inert contract)."""
    from dagua.layout.ops.pipelines.native_undirected import (
        layout_native_undirected_portfolio,
    )

    arm_module = importlib.import_module("dagua.layout.ops.pipelines.native_stress_family_arm")

    def _must_not_run(*args: object, **kwargs: object) -> dict[str, torch.Tensor]:
        raise AssertionError("stress-family arm built candidates on a gate-closed row")

    monkeypatch.setattr(arm_module, "build_stress_family_candidates", _must_not_run)
    result = layout_native_undirected_portfolio(
        _disjoint_k5s_problem(),
        SolveState(),
        RuntimeContext(),
        LayoutConfig(seed=42),
    )
    assert bool(torch.isfinite(result).all().item())


def test_gate_closed_directed_row_never_builds(monkeypatch: pytest.MonkeyPatch) -> None:
    """On a disconnected directed row the contest never runs the arm.

    Disconnection is the shared-gate condition that still closes directed
    rows under the cousin-fitted width cut (every measured acyclic layering
    is admitted after review F4).
    """
    from dagua.layout.ops.pipelines.native_directed import (
        layout_native_directed_portfolio,
    )

    arm_module = importlib.import_module("dagua.layout.ops.pipelines.native_stress_family_arm")

    def _must_not_run(*args: object, **kwargs: object) -> dict[str, torch.Tensor]:
        raise AssertionError("stress-family arm built candidates on a gate-closed row")

    monkeypatch.setattr(arm_module, "build_stress_family_candidates", _must_not_run)
    result = layout_native_directed_portfolio(
        _disjoint_wide_dags_problem(),
        SolveState(),
        RuntimeContext(),
        LayoutConfig(seed=42),
    )
    assert bool(torch.isfinite(result).all().item())


def test_arm_fires_inside_undirected_contest(monkeypatch: pytest.MonkeyPatch) -> None:
    """On a gated undirected row the contest builds and registers the arm."""
    from dagua.layout.ops.pipelines.native_undirected import (
        layout_native_undirected_portfolio,
    )

    arm_module = importlib.import_module("dagua.layout.ops.pipelines.native_stress_family_arm")
    real_builder = arm_module.build_stress_family_candidates
    seen: list[str] = []

    def _recording_builder(*args: object, **kwargs: object) -> dict[str, torch.Tensor]:
        candidates = real_builder(*args, **kwargs)
        seen.extend(sorted(candidates))
        return candidates

    monkeypatch.setattr(arm_module, "build_stress_family_candidates", _recording_builder)
    result = layout_native_undirected_portfolio(
        _chords_problem(),
        SolveState(),
        RuntimeContext(),
        LayoutConfig(seed=42),
    )
    assert bool(torch.isfinite(result).all().item())
    assert f"stress_sgd_k_seed{STRESS_SGD_SEED_BANK[0]}" in seen
    assert "elk_stress_arm" in seen


def test_arm_fires_inside_directed_contest(monkeypatch: pytest.MonkeyPatch) -> None:
    """On a low-layering DAG the directed contest builds and registers the arm."""
    from dagua.layout.ops.pipelines.native_directed import (
        layout_native_directed_portfolio,
    )

    arm_module = importlib.import_module("dagua.layout.ops.pipelines.native_stress_family_arm")
    real_builder = arm_module.build_stress_family_candidates
    seen: list[str] = []

    def _recording_builder(*args: object, **kwargs: object) -> dict[str, torch.Tensor]:
        candidates = real_builder(*args, **kwargs)
        seen.extend(sorted(candidates))
        return candidates

    monkeypatch.setattr(arm_module, "build_stress_family_candidates", _recording_builder)
    problem = _wide_dag_problem()
    assert stress_family_directed_admitted(problem)
    result = layout_native_directed_portfolio(
        problem,
        SolveState(),
        RuntimeContext(),
        LayoutConfig(seed=42),
    )
    assert bool(torch.isfinite(result).all().item())
    assert f"maxent_stress_seed{MAXENT_SEED_BANK[0]}" in seen


# The full frozen-bank candidate inventory: what the arm must generate on a
# gated row whenever the ledger admits every family package, independent of
# machine load (review F2).
_FULL_BANK_INVENTORY = sorted(
    [f"stress_sgd_k_seed{seed}" for seed in STRESS_SGD_SEED_BANK]
    + [f"maxent_stress_seed{seed}" for seed in MAXENT_SEED_BANK]
    + ["elk_stress_arm"]
)


def _recording_builder_events(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[dict[str, torch.Tensor]], list[str]]:
    """Record builder invocations and a shared ordered event log."""
    arm_module = importlib.import_module("dagua.layout.ops.pipelines.native_stress_family_arm")
    real_builder = arm_module.build_stress_family_candidates
    calls: list[dict[str, torch.Tensor]] = []
    events: list[str] = []

    def _recording(*args: object, **kwargs: object) -> dict[str, torch.Tensor]:
        candidates = real_builder(*args, **kwargs)
        calls.append(candidates)
        events.append("builder_called")
        return candidates

    monkeypatch.setattr(arm_module, "build_stress_family_candidates", _recording)
    return calls, events


def test_undirected_stress_admission_ignores_exhausted_wall_reserve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review F2 regression: admission is DWU-ledger-only, never load state.

    The wall reserve expires mid-portfolio (injected through an earlier
    arm's pipeline, i.e. after the top-level marketplace guard but before
    Candidate S), which is exactly the load-dependent state the a5e8fff5
    seam consulted through ``_portfolio_has_budget``: there the arm was
    silently skipped, so this test fails on that commit. A REAL ledger is
    installed so ``admit_native_work`` actually reaches its wall-veto branch
    (re-review F2: without one it returns ``True`` before the veto and the
    test proves nothing). Fixed admission consults only the ledger, so the
    full frozen-bank inventory is still generated.
    """
    from dagua.layout.ops.pipelines import sfdp as sfdp_module
    from dagua.layout.ops.pipelines.native_budget import (
        WALL_DEADLINE_ATTR,
        install_budget_ledger,
    )
    from dagua.layout.ops.pipelines.native_undirected import (
        layout_native_undirected_portfolio,
    )

    calls, events = _recording_builder_events(monkeypatch)
    config = LayoutConfig(seed=42)
    install_budget_ledger(config, timeout_s=10_000.0)
    real_sfdp = sfdp_module.layout_sfdp_pipeline

    def _expiring_sfdp(*args: object, **kwargs: object) -> object:
        if not hasattr(config, WALL_DEADLINE_ATTR):
            setattr(config, WALL_DEADLINE_ATTR, time.perf_counter() - 1000.0)
            events.append("wall_reserve_expired")
        return real_sfdp(*args, **kwargs)

    monkeypatch.setattr(sfdp_module, "layout_sfdp_pipeline", _expiring_sfdp)
    result = layout_native_undirected_portfolio(
        _chords_problem(),
        SolveState(),
        RuntimeContext(),
        config,
    )
    assert bool(torch.isfinite(result).all().item())
    assert "wall_reserve_expired" in events, "injection hook never ran (fixture drifted)"
    assert events.index("wall_reserve_expired") < events.index("builder_called"), (
        "premise broken: the wall reserve must expire before Candidate S runs"
    )
    assert len(calls) == 1, "stress arm must fire despite the exhausted wall reserve"
    assert sorted(calls[0]) == _FULL_BANK_INVENTORY


def test_directed_stress_admission_ignores_exhausted_wall_reserve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review F2 regression (directed seam): no wall/predicted-time gates.

    The a5e8fff5 directed seam consulted ``_portfolio_has_budget`` and
    ``_predicted_arm_budget_available`` (both live wall-deadline state); an
    exhausted reserve injected before its block skipped the arm entirely,
    so this test fails there. A REAL ledger is installed so the wall-veto
    branch inside ``admit_native_work`` is genuinely reachable (re-review
    F2). Fixed admission is ledger-only.
    """
    from dagua.layout.ops.pipelines.native_budget import (
        WALL_DEADLINE_ATTR,
        install_budget_ledger,
    )
    from dagua.layout.ops.pipelines.native_directed import (
        layout_native_directed_portfolio,
    )

    directed_module = importlib.import_module("dagua.layout.ops.pipelines.native_directed")
    calls, events = _recording_builder_events(monkeypatch)
    config = LayoutConfig(seed=42)
    install_budget_ledger(config, timeout_s=10_000.0)
    real_register = directed_module._register_challenger_variants

    def _expiring_register(*args: object, **kwargs: object) -> object:
        if not hasattr(config, WALL_DEADLINE_ATTR):
            setattr(config, WALL_DEADLINE_ATTR, time.perf_counter() - 1000.0)
            events.append("wall_reserve_expired")
        return real_register(*args, **kwargs)

    monkeypatch.setattr(directed_module, "_register_challenger_variants", _expiring_register)
    result = layout_native_directed_portfolio(
        _wide_dag_problem(),
        SolveState(),
        RuntimeContext(),
        config,
    )
    assert bool(torch.isfinite(result).all().item())
    assert "wall_reserve_expired" in events, "injection hook never ran (fixture drifted)"
    assert events.index("wall_reserve_expired") < events.index("builder_called"), (
        "premise broken: the wall reserve must expire before the stress block runs"
    )
    assert len(calls) == 1, "directed stress arm must fire despite the exhausted wall reserve"
    assert sorted(calls[0]) == _FULL_BANK_INVENTORY


def test_exhausted_ledger_vetoes_stress_packages_deterministically() -> None:
    """The ledger, not elapsed time, is the sole admission authority.

    A spent ledger deterministically vetoes every family package (all-or-
    nothing pricing charged before generation, review F2).
    """
    from dagua.layout.ops.pipelines.native_budget import (
        LEDGER_ATTR,
        install_budget_ledger,
    )
    from dagua.layout.ops.pipelines.native_stress_family_arm import (
        admit_stress_family_packages,
    )

    problem = _chords_problem()
    config = LayoutConfig(seed=42)
    install_budget_ledger(config, timeout_s=1.0)
    ledger = getattr(config, LEDGER_ATTR)
    ledger.spent_dwu = ledger.capacity_dwu() + 1.0
    admission = admit_stress_family_packages(problem, config, "cpu")
    assert admission.stress_sgd_seeds == ()
    assert admission.maxent_seeds == ()
    assert not admission.elk_admitted
    assert not admission.any_admitted


def test_admission_charges_aggregate_packages_before_generation() -> None:
    """Every admitted family is one aggregate ledger package (review F2).

    The stress-SGD package charges all three trajectories plus the two
    reserved referee seats as one all-or-nothing decision; maxent charges
    two; ELK one. No per-trajectory drip pricing.
    """
    from dagua.layout.ops.pipelines.native_budget import (
        LEDGER_ATTR,
        install_budget_ledger,
    )
    from dagua.layout.ops.pipelines.native_stress_family_arm import (
        admit_stress_family_packages,
    )

    problem = _chords_problem()
    config = LayoutConfig(seed=42)
    install_budget_ledger(config, timeout_s=10_000.0)
    admission = admit_stress_family_packages(problem, config, "cpu")
    assert admission.stress_sgd_seeds == STRESS_SGD_SEED_BANK
    assert admission.maxent_seeds == MAXENT_SEED_BANK
    assert admission.elk_admitted
    events = [
        (record["event"], str(record["reason"]))
        for record in getattr(config, LEDGER_ATTR).event_log
    ]
    admit_reasons = [reason for event, reason in events if event == "admit"]
    assert admit_reasons == [
        "optional_seed_family_stress_sgd_k",
        "optional_seed_family_maxent_stress",
        "optional_stress_family_elk",
    ]


def test_stress_admission_is_pure_function_of_ledger_state() -> None:
    """Re-review F2: skewed wall deadlines cannot change stress admission.

    The re-review's exact reproduction: identical input, equal-state REAL
    ledgers, and wall deadlines skewed from long-expired to distant-future.
    Admission, the ledger decision log, the charged DWU, and the built
    candidate inventory/output must all be identical. At 03fda545 the
    expired-wall config admitted nothing (five ``wall_reserve_exhausted``
    vetoes through ``admit_native_work``), so this test fails there.
    """
    from dagua.layout.ops.pipelines.native_budget import (
        LEDGER_ATTR,
        WALL_DEADLINE_ATTR,
        install_budget_ledger,
    )
    from dagua.layout.ops.pipelines.native_stress_family_arm import (
        admit_stress_family_packages,
    )

    problem = _chords_problem()
    admissions = []
    ledger_states = []
    inventories = []
    for wall_skew_s in (-1_000.0, 1_000_000.0):
        config = LayoutConfig(seed=42)
        install_budget_ledger(config, timeout_s=10_000.0)
        setattr(config, WALL_DEADLINE_ATTR, time.perf_counter() + wall_skew_s)
        admission = admit_stress_family_packages(problem, config, "cpu")
        admissions.append(admission)
        ledger = getattr(config, LEDGER_ATTR)
        ledger_states.append(
            (
                ledger.spent_dwu,
                [(record["event"], str(record["reason"])) for record in ledger.event_log],
            )
        )
        inventories.append(
            build_stress_family_candidates(
                problem,
                node_sep=40.0,
                stress_sgd_seeds=admission.stress_sgd_seeds,
                maxent_seeds=admission.maxent_seeds,
                include_elk=admission.elk_admitted,
            )
        )
    assert admissions[0] == admissions[1]
    assert admissions[0].any_admitted, "premise broken: a 10k-DWU ledger must admit packages"
    assert ledger_states[0] == ledger_states[1]
    assert sorted(inventories[0]) == _FULL_BANK_INVENTORY
    assert sorted(inventories[1]) == _FULL_BANK_INVENTORY
    for name, pos in inventories[0].items():
        assert torch.equal(pos, inventories[1][name]), name


def test_directed_quota_reserves_every_stress_family_under_proxy_cut() -> None:
    """Review F3 regression: one quota seat per normalized stress family.

    Reproduces the review's synthetic bounded-finalist scenario: eight
    higher-proxy legacy families fill the legacy cut, all three stress
    families sit below the generic proxy cut, and one pre-existing
    non-stress quota entry is present. All four reserved candidates must
    survive ``select_finalists`` (the a5e8fff5 seam reserved at most one
    stress entry and dropped maxent and ELK entirely).
    """
    from dagua.layout.ops.pipelines.native_contest_cascade import select_finalists
    from dagua.layout.ops.pipelines.native_stress_family_arm import (
        stress_family_quota_entries,
    )

    legacy_names = [f"legacy_arm_{index}" for index in range(8)]
    stress_names = [
        "stress_sgd_k_seed42_raw",
        "stress_sgd_k_seed7",
        "maxent_stress_seed42_raw",
        "elk_stress_arm_raw",
    ]
    cluster_name = "cluster_scaffold"
    proxy_scores: dict[str, float] = {"incumbent": 99.0}
    proxy_scores.update({name: 90.0 - index for index, name in enumerate(legacy_names)})
    proxy_scores[cluster_name] = 60.0
    proxy_scores.update({name: 50.0 - index for index, name in enumerate(stress_names)})
    generator = torch.Generator().manual_seed(0)
    candidates = {
        name: torch.rand((6, 2), generator=generator) * (1.0 + index)
        for index, name in enumerate(proxy_scores)
    }
    challenger_names = sorted(
        (name for name in candidates if name != "incumbent"),
        key=lambda name: (-proxy_scores[name], name),
    )
    legacy_finalists = list(legacy_names)
    quota_families = {cluster_name: cluster_name}
    entries = stress_family_quota_entries(challenger_names, legacy_finalists, quota_families)
    assert entries == {
        "stress_sgd_k_seed42_raw": "stress_sgd_k_seed",
        "maxent_stress_seed42_raw": "maxent_stress_seed",
        "elk_stress_arm_raw": "elk_stress_arm",
    }
    quota_families.update(entries)
    finalists = select_finalists(
        candidates,
        proxy_scores,
        quota_families,
        4,
        ["incumbent"],
    )
    assert cluster_name in finalists
    assert "stress_sgd_k_seed42_raw" in finalists
    assert "maxent_stress_seed42_raw" in finalists
    assert "elk_stress_arm_raw" in finalists


def test_quota_entries_skip_families_already_represented() -> None:
    """A stress family inside the legacy cut or quotas gets no second seat."""
    from dagua.layout.ops.pipelines.native_stress_family_arm import (
        stress_family_quota_entries,
    )

    challenger_names = [
        "stress_sgd_k_seed42",
        "stress_sgd_k_seed7",
        "maxent_stress_seed42",
        "elk_stress_arm",
    ]
    entries = stress_family_quota_entries(
        challenger_names,
        ["stress_sgd_k_seed7_convergent"],
        {"maxent_stress_seed42": "maxent_stress_seed"},
    )
    assert entries == {"elk_stress_arm": "elk_stress_arm"}


# Gate-closed golden bytes: rows where the stress-family gate is closed must
# stay byte-identical to the pre-packet parent. The undirected golden was
# captured on 0ee2db36; the directed golden was recaptured on the rebased
# base 014db18b when the cousin-fitted width cut (review F4) opened the old
# deep-layered fixture -- disconnection is the remaining directed closure.
# Both fixtures are also closed for the W1-A/W1-B arms, so the goldens pin
# the whole gate-closed path, not a lucky overlap.
_GOLDEN_UNDIRECTED_DISJOINT_K5S_SHA256 = (
    "5d7078d63636835b3d866a4dc6613c2936aa821737515f222fc9a343ed5c190f"  # pragma: allowlist secret
)
_GOLDEN_DIRECTED_DISJOINT_WIDE_DAGS_SHA256 = (
    "7ed90644bae20bf32346cac98ec94170fac4449109087075a9f316d570132563"  # pragma: allowlist secret
)


def test_gate_closed_undirected_row_byte_identical_golden() -> None:
    """Disconnected undirected row reproduces the pre-packet bytes exactly."""
    from dagua.layout.ops.pipelines.native_undirected import (
        layout_native_undirected_portfolio,
    )

    result = layout_native_undirected_portfolio(
        _disjoint_k5s_problem(),
        SolveState(),
        RuntimeContext(),
        LayoutConfig(seed=42),
    )
    assert _sha256(result) == _GOLDEN_UNDIRECTED_DISJOINT_K5S_SHA256


def test_gate_closed_directed_row_byte_identical_golden() -> None:
    """Disconnected directed row reproduces the pre-packet bytes exactly."""
    from dagua.layout.ops.pipelines.native_directed import (
        layout_native_directed_portfolio,
    )

    result = layout_native_directed_portfolio(
        _disjoint_wide_dags_problem(),
        SolveState(),
        RuntimeContext(),
        LayoutConfig(seed=42),
    )
    assert _sha256(result) == _GOLDEN_DIRECTED_DISJOINT_WIDE_DAGS_SHA256


def test_layering_cut_traces_to_cousin_fit_artifact() -> None:
    """Review F4: the shipped gate constant EQUALS the checked-in fit output.

    The artifact is produced by ``scripts/w22_cousin_layering_fit.py fit``
    (25 directed training cousins, native baseline regenerated at the
    corrected-W2-1 base 014db18b). This pin fails whenever either side
    drifts, so the runtime boundary always traces to cousin evidence.
    """
    artifact = Path(__file__).parent / "data" / "w22_layering_fit.json"
    payload = json.loads(artifact.read_text())
    fitted = payload["fit"]["fitted_min_avg_layer_width"]
    assert fitted == LOW_LAYERING_MIN_AVG_LAYER_WIDTH
    assert payload["fit"]["separating_cut_exists"] is False, (
        "the fit artifact claims a separating width cut exists; the gate "
        "semantics comment in native_stress_family_arm.py is now stale"
    )
    assert len(payload["rows"]) >= 20, "fit artifact lost its cousin rows"
