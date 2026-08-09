"""Tests for the shadow-champion legacy-track contest (sprint2 W2-4a, review F1-F4).

Every displacement test drives the REAL selection seams: candidates are
injected or biased at the generation/selection boundary only, and everything
downstream -- displacement recording, the legacy-track shadow re-run, the
final referee contest, and the emission -- is production code end-to-end.

Top-level imports are restricted to APIs that already exist on the pre-fix
commit so the per-finding regression tests fail BEHAVIORALLY there (a lost
legacy drawing), not at collection.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, cast

import pytest
import torch

from dagua.config import LayoutConfig
from dagua.layout.graph_classify import classify_graph
from dagua.layout.ops.pipelines import (
    dagua_native,
    native_directed,
    native_planar_arm,
    native_sparse_infrastructure,
    native_undirected,
)
from dagua.layout.ops.pipelines.dagua_native import layout_dagua_native_pipeline
from dagua.layout.ops.pipelines.native_budget import (
    PROCESS_DEADLINE_ATTR,
    WALL_DEADLINE_ATTR,
    NativeBudgetLedger,
    install_budget_ledger,
)
from dagua.layout.ops.pipelines.native_shadow_champion import is_new_arm_candidate
from dagua.layout.ops.state import LayoutProblem

_CONFIG_KWARGS: Dict[str, Any] = {"seed": 42}
_LEDGER_DWU = 600.0


def _fresh_config(**overrides: Any) -> LayoutConfig:
    """Return a fresh test config with a fresh deterministic ledger."""
    config = LayoutConfig(**{**_CONFIG_KWARGS, **overrides})
    install_budget_ledger(config, _LEDGER_DWU)
    return config


def _wheel_graph(spokes: int = 8) -> tuple[torch.Tensor, int, torch.Tensor]:
    """Return a wheel graph (planar, 3-connected: the planar arm admits it)."""
    edges = []
    for spoke in range(1, spokes + 1):
        edges.append((0, spoke))
        edges.append((spoke, 1 + spoke % spokes))
    edge_index = torch.tensor(sorted(set(edges)), dtype=torch.long).t().contiguous()
    num_nodes = spokes + 1
    return edge_index, num_nodes, torch.full((num_nodes, 2), 20.0)


def _k5_graph() -> tuple[torch.Tensor, int, torch.Tensor]:
    """Return K5 (non-planar: every planar gate stays closed)."""
    edges = [(u, v) for u in range(5) for v in range(u + 1, 5)]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index, 5, torch.full((5, 2), 20.0)


def _k5_tail_graph(tail: int = 15) -> tuple[torch.Tensor, int, torch.Tensor]:
    """Return K5 plus a path tail (non-planar, n large enough for band tests)."""
    edges = [(u, v) for u in range(5) for v in range(u + 1, 5)]
    for i in range(tail):
        edges.append((4 + i, 5 + i))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    num_nodes = 5 + tail
    return edge_index, num_nodes, torch.full((num_nodes, 2), 20.0)


def _grid_dag_graph(width: int = 3, height: int = 3) -> tuple[torch.Tensor, int, torch.Tensor]:
    """Return a planar acyclic grid (edges right/down: directed route)."""
    edges = []
    for y in range(height):
        for x in range(width):
            node = y * width + x
            if x + 1 < width:
                edges.append((node, node + 1))
            if y + 1 < height:
                edges.append((node, node + width))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    num_nodes = width * height
    return edge_index, num_nodes, torch.full((num_nodes, 2), 20.0)


def _bad_positions(num_nodes: int) -> torch.Tensor:
    """Return a terrible near-collinear drawing (every node overlapping)."""
    line = torch.arange(num_nodes, dtype=torch.float32) * 1.0e-2
    return torch.stack((line, line), dim=1)


def _final_referee_key(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: torch.Tensor,
    edge_weights: Optional[torch.Tensor] = None,
    clusters: Optional[dict[str, Any]] = None,
    cluster_parents: Optional[dict[str, Optional[str]]] = None,
) -> tuple[tuple[int, float], float]:
    """Score one final drawing with the runtime referee (pre-existing APIs)."""
    from dagua.eval.ruler_v3 import referee_eligibility_key
    from dagua.layout.ops.pipelines.native_v3_referee import score_v3_runtime_result

    cpu_edge_index = edge_index.detach().to(device="cpu", dtype=torch.long)
    problem = LayoutProblem(
        edge_index=cpu_edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes.detach().to(device="cpu", dtype=torch.float32),
        direction="TB",
        clusters=clusters,
        cluster_parents=cluster_parents,
        structure=cast(Any, classify_graph(cpu_edge_index, num_nodes)),
        edge_weights=None if edge_weights is None else edge_weights.detach().cpu(),
    )
    result = score_v3_runtime_result(pos.detach().to(device="cpu", dtype=torch.float32), problem)
    return (referee_eligibility_key(result), float(result.scores["tiered"]))


def _inject_bad_planar_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the planar arm emit one terrible certificate-exempt candidate."""
    monkeypatch.setattr(
        native_planar_arm,
        "build_planar_arm_candidates",
        lambda problem, node_sep: {"planar_seeded_stress": _bad_positions(problem.num_nodes)},
    )


def _prefer_new_arm_undirected(
    monkeypatch: pytest.MonkeyPatch,
    prefix: tuple[str, ...] = ("planar_", "tfdp"),
) -> None:
    """Bias the undirected winner selection toward new-arm candidates.

    Simulates an honest new-arm contest win (the grafo1000.14 class) while
    keeping every downstream seam real. The wrapper falls through to the
    real selector when no new-arm candidate is present, so the legacy-track
    re-run selects exactly as an unbiased legacy contest would.
    """
    real_select = native_undirected._select_undirected_winner

    def prefer(scores: Dict[str, float], telemetry: Dict[str, Any], *args: Any) -> str:
        new_arm = sorted(name for name in scores if name.startswith(prefix))
        if new_arm:
            return new_arm[0]
        return real_select(scores, telemetry, *args)

    monkeypatch.setattr(native_undirected, "_select_undirected_winner", prefer)


def _prefer_new_arm_directed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bias the directed winner selection toward new-arm candidates."""
    real_select = native_directed._select_directed_winner

    def prefer(scores: Dict[str, float], telemetry: Dict[str, Any], *args: Any) -> str:
        new_arm = sorted(name for name in scores if is_new_arm_candidate(name))
        if new_arm:
            return new_arm[0]
        return real_select(scores, telemetry, *args)

    monkeypatch.setattr(native_directed, "_select_directed_winner", prefer)


def _veto_legacy_shadow_admission(monkeypatch: pytest.MonkeyPatch) -> None:
    """Veto the pre-fix in-terminal shadow admission (review F1 failure class).

    The pre-fix design asked ``admit_native_work`` for permission to run the
    safety track at the terminal seam and emitted the (weaker) primary on a
    veto. Denying exactly that reason reproduces the budget/wall veto
    deterministically on the pre-fix commit; the fixed design never requests
    it (the legacy track runs unconditionally on its own fresh ledger), so
    this patch is inert after the fix.
    """
    from dagua.layout.ops.pipelines import native_budget

    real_admit = native_budget.admit_native_work

    def admit(config: Any, cost: Any, reserve_reason: str) -> bool:
        if reserve_reason == "shadow_champion_terminal_contest":
            return False
        return real_admit(config, cost, reserve_reason)

    monkeypatch.setattr(native_budget, "admit_native_work", admit)


def _run_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: torch.Tensor,
    config: LayoutConfig,
    **kwargs: Any,
) -> torch.Tensor:
    """Run the public native pipeline entry on one graph."""
    return layout_dagua_native_pipeline(
        edge_index,
        num_nodes,
        node_sizes,
        config=config,
        seed=42,
        **kwargs,
    )


def _legacy_reference(
    monkeypatch: pytest.MonkeyPatch,
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: torch.Tensor,
    **kwargs: Any,
) -> torch.Tensor:
    """Run the true no-new-arm pipeline via pre-existing gates.

    Closes the planar-arm structural gate, the t-FDP normal-contest
    admission, and the band eligibility at their sources (all exist on the
    pre-fix commit), so this reference is computable identically before and
    after the fix.
    """
    with monkeypatch.context() as ctx:
        ctx.setattr(native_planar_arm, "planar_arm_admitted", lambda problem: False)
        ctx.setattr(native_undirected, "_sparse_contest_arm_admitted", lambda *a, **k: False)
        ctx.setattr(native_sparse_infrastructure, "sparse_band_contest_eligible", lambda p: False)
        return _run_pipeline(edge_index, num_nodes, node_sizes, _fresh_config(), **kwargs)


# ---------------------------------------------------------------------------
# F2: the new-arm family registry must cover W1-A's t-FDP names.
# ---------------------------------------------------------------------------


def test_new_arm_family_covers_tfdp_and_planar() -> None:
    """W1-A t-FDP candidate names are new-arm candidates (review F2)."""
    for name in (
        "tfdp",
        "tfdp_raw",
        "tfdp_g2",
        "tfdp_g0.5_raw",
        "planar_fpp",
        "planar_seeded_stress",
    ):
        assert is_new_arm_candidate(name), name
    for name in ("incumbent", "stress", "fcose_seed0", "geodesic_stress", "sfdp_prism"):
        assert not is_new_arm_candidate(name), name


# ---------------------------------------------------------------------------
# F1 + F3: undirected planar displacement through the real selection seam.
# A budget veto must retain the legacy track, never the weaker primary.
# ---------------------------------------------------------------------------


def test_undirected_planar_displacement_recovers_legacy_track(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A vetoed shadow must still emit the legacy final drawing (F1/F3)."""
    edge_index, num_nodes, node_sizes = _wheel_graph()
    legacy_pos = _legacy_reference(monkeypatch, edge_index, num_nodes, node_sizes)

    _inject_bad_planar_candidate(monkeypatch)
    _prefer_new_arm_undirected(monkeypatch)
    _veto_legacy_shadow_admission(monkeypatch)
    emitted = _run_pipeline(edge_index, num_nodes, node_sizes, _fresh_config())

    emitted_key = _final_referee_key(emitted, edge_index, num_nodes, node_sizes)
    legacy_key = _final_referee_key(legacy_pos, edge_index, num_nodes, node_sizes)
    assert emitted_key >= legacy_key, (emitted_key, legacy_key)
    assert torch.equal(emitted.cpu(), legacy_pos.cpu())


# ---------------------------------------------------------------------------
# F2: t-FDP displacement in the normal undirected contest.
# ---------------------------------------------------------------------------


def _admit_tfdp_into_normal_contest(monkeypatch: pytest.MonkeyPatch) -> None:
    """Open the t-FDP normal-contest gates on a small non-planar row."""
    real_shortlist = dagua_native._undirected_route_shortlist

    def shortlist_with_tfdp(*args: Any, **kwargs: Any) -> Any:
        shortlist = real_shortlist(*args, **kwargs)
        if "tfdp_sparse" in shortlist.candidates:
            return shortlist
        return dagua_native.NativeShortlist(
            classes=shortlist.classes,
            candidates=(*shortlist.candidates, "tfdp_sparse"),
        )

    monkeypatch.setattr(dagua_native, "_undirected_route_shortlist", shortlist_with_tfdp)
    monkeypatch.setattr(native_undirected, "_sparse_contest_arm_admitted", lambda *a, **k: True)
    monkeypatch.setattr(
        native_sparse_infrastructure,
        "tfdp_sparse_positions",
        lambda problem, gamma, node_sep: _bad_positions(problem.num_nodes),
    )


def test_undirected_tfdp_displacement_recovers_legacy_track(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A t-FDP normal-contest win is a protected displacement (F2)."""
    edge_index, num_nodes, node_sizes = _k5_graph()
    legacy_pos = _legacy_reference(monkeypatch, edge_index, num_nodes, node_sizes)

    _admit_tfdp_into_normal_contest(monkeypatch)
    _prefer_new_arm_undirected(monkeypatch, prefix=("tfdp",))
    emitted = _run_pipeline(edge_index, num_nodes, node_sizes, _fresh_config())

    emitted_key = _final_referee_key(emitted, edge_index, num_nodes, node_sizes)
    legacy_key = _final_referee_key(legacy_pos, edge_index, num_nodes, node_sizes)
    assert emitted_key >= legacy_key, (emitted_key, legacy_key)
    assert torch.equal(emitted.cpu(), legacy_pos.cpu())


# ---------------------------------------------------------------------------
# F2: t-FDP displacement in the W1-A sparse-band mini-contest.
# ---------------------------------------------------------------------------


def test_sparse_band_tfdp_displacement_recovers_legacy_track(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A t-FDP band win must keep the legacy early-return reachable (F2)."""
    edge_index, num_nodes, node_sizes = _k5_tail_graph()
    # Shrink the contest ceiling so this row takes the band early-return
    # path; the legacy reference closes band eligibility, exactly the
    # pre-W1-A bare-incumbent behavior the invariant protects.
    monkeypatch.setattr(native_undirected, "MAX_CONTEST_NODES", 10)
    legacy_pos = _legacy_reference(monkeypatch, edge_index, num_nodes, node_sizes)

    monkeypatch.setattr(
        native_sparse_infrastructure,
        "sparse_band_contest_eligible",
        lambda problem: True,
    )
    monkeypatch.setattr(
        native_sparse_infrastructure,
        "tfdp_sparse_positions",
        lambda problem, gamma, node_sep: _bad_positions(problem.num_nodes),
    )
    _prefer_new_arm_undirected(monkeypatch, prefix=("tfdp",))
    emitted = _run_pipeline(edge_index, num_nodes, node_sizes, _fresh_config())

    emitted_key = _final_referee_key(emitted, edge_index, num_nodes, node_sizes)
    legacy_key = _final_referee_key(legacy_pos, edge_index, num_nodes, node_sizes)
    assert emitted_key >= legacy_key, (emitted_key, legacy_key)
    assert torch.equal(emitted.cpu(), legacy_pos.cpu())


# ---------------------------------------------------------------------------
# F3: directed displacement with the post-argmax late stages live.
# ---------------------------------------------------------------------------


def test_directed_planar_displacement_recovers_legacy_track(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The directed legacy track replays late ordering/nested-stress seams (F3)."""
    edge_index, num_nodes, node_sizes = _grid_dag_graph()
    legacy_pos = _legacy_reference(monkeypatch, edge_index, num_nodes, node_sizes)

    _inject_bad_planar_candidate(monkeypatch)
    _prefer_new_arm_directed(monkeypatch)
    _veto_legacy_shadow_admission(monkeypatch)
    emitted = _run_pipeline(edge_index, num_nodes, node_sizes, _fresh_config())

    emitted_key = _final_referee_key(emitted, edge_index, num_nodes, node_sizes)
    legacy_key = _final_referee_key(legacy_pos, edge_index, num_nodes, node_sizes)
    assert emitted_key >= legacy_key, (emitted_key, legacy_key)
    assert torch.equal(emitted.cpu(), legacy_pos.cpu())


# ---------------------------------------------------------------------------
# F4.5: weighted and clustered planar displacement rows.
# ---------------------------------------------------------------------------


def test_weighted_planar_displacement_recovers_legacy_track(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Declared-weighted rows traverse the weighted terminal pre-stages (F4.5)."""
    edge_index, num_nodes, node_sizes = _wheel_graph()
    edge_weights = torch.full((edge_index.shape[1],), 3.0)
    legacy_pos = _legacy_reference(
        monkeypatch,
        edge_index,
        num_nodes,
        node_sizes,
        edge_weights=edge_weights,
    )

    _inject_bad_planar_candidate(monkeypatch)
    _prefer_new_arm_undirected(monkeypatch)
    emitted = _run_pipeline(
        edge_index,
        num_nodes,
        node_sizes,
        _fresh_config(),
        edge_weights=edge_weights,
    )

    emitted_key = _final_referee_key(
        emitted, edge_index, num_nodes, node_sizes, edge_weights=edge_weights
    )
    legacy_key = _final_referee_key(
        legacy_pos, edge_index, num_nodes, node_sizes, edge_weights=edge_weights
    )
    assert emitted_key >= legacy_key, (emitted_key, legacy_key)
    assert torch.equal(emitted.cpu(), legacy_pos.cpu())


def test_clustered_planar_displacement_recovers_legacy_track(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clustered rows traverse cluster tightening in both tracks (F4.5)."""
    edge_index, num_nodes, node_sizes = _wheel_graph()
    clusters = {
        "hub": [0, 1, 2, 3, 4],
        "rim": [5, 6, 7, 8],
    }
    legacy_pos = _legacy_reference(
        monkeypatch,
        edge_index,
        num_nodes,
        node_sizes,
        clusters=clusters,
    )

    _inject_bad_planar_candidate(monkeypatch)
    _prefer_new_arm_undirected(monkeypatch)
    emitted = _run_pipeline(
        edge_index,
        num_nodes,
        node_sizes,
        _fresh_config(),
        clusters=clusters,
    )

    emitted_key = _final_referee_key(emitted, edge_index, num_nodes, node_sizes, clusters=clusters)
    legacy_key = _final_referee_key(
        legacy_pos, edge_index, num_nodes, node_sizes, clusters=clusters
    )
    assert emitted_key >= legacy_key, (emitted_key, legacy_key)


# ---------------------------------------------------------------------------
# F3 + F4.6 + F4.7: byte-equality oracle against a true new-arms-disabled run,
# plus a shortlist-composition change the old finalist-mapping restriction
# could not represent.
# ---------------------------------------------------------------------------


def test_shadow_track_byte_equals_true_disabled_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The legacy shadow track IS a new-arms-disabled run, byte-for-byte (F4.7)."""
    from dagua.layout.ops.pipelines.native_shadow_champion import (
        DISABLE_NEW_ARMS_ATTR,
        SHADOW_CONTEST_TELEMETRY_ATTR,
    )

    edge_index, num_nodes, node_sizes = _wheel_graph()
    _inject_bad_planar_candidate(monkeypatch)
    _prefer_new_arm_undirected(monkeypatch)

    config = _fresh_config()
    emitted = _run_pipeline(edge_index, num_nodes, node_sizes, config)
    telemetry = getattr(config, SHADOW_CONTEST_TELEMETRY_ATTR, None)
    assert telemetry is not None
    assert telemetry["displacements"], telemetry
    assert telemetry["emitted"] == "shadow"
    assert torch.equal(emitted.cpu(), telemetry["shadow_pos"])

    disabled_config = _fresh_config()
    setattr(disabled_config, DISABLE_NEW_ARMS_ATTR, True)
    disabled_pos = _run_pipeline(edge_index, num_nodes, node_sizes, disabled_config)
    assert torch.equal(telemetry["shadow_pos"], disabled_pos.cpu())


def test_shortlist_composition_change_still_recovers_legacy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The legacy track survives new-arm finalist-shortlist pressure (F4.6).

    The finalist cascade is patched so that whenever a new-arm candidate is
    present, NO legacy challenger reaches the referee: the with-arms scored
    mapping contains no legacy champion at all (the case the old
    finalist-mapping restriction could not represent). The legacy re-run
    performs its own untampered finalist selection and must still recover
    the true legacy final drawing.
    """
    from dagua.layout.ops.pipelines import native_contest_cascade

    edge_index, num_nodes, node_sizes = _wheel_graph()
    legacy_pos = _legacy_reference(monkeypatch, edge_index, num_nodes, node_sizes)

    _inject_bad_planar_candidate(monkeypatch)
    _prefer_new_arm_undirected(monkeypatch)
    real_select_finalists = native_contest_cascade.select_finalists

    def squeeze_legacy_out(
        positions: Dict[str, torch.Tensor],
        proxy_scores: Dict[str, float],
        quota_families: Dict[str, str],
        finalist_limit: int,
        mandatory: Any,
    ) -> Any:
        new_arm = sorted(name for name in positions if is_new_arm_candidate(name))
        if new_arm:
            return ["incumbent", *new_arm]
        return real_select_finalists(
            positions, proxy_scores, quota_families, finalist_limit, mandatory
        )

    monkeypatch.setattr(native_contest_cascade, "select_finalists", squeeze_legacy_out)
    emitted = _run_pipeline(edge_index, num_nodes, node_sizes, _fresh_config())

    emitted_key = _final_referee_key(emitted, edge_index, num_nodes, node_sizes)
    legacy_key = _final_referee_key(legacy_pos, edge_index, num_nodes, node_sizes)
    assert emitted_key >= legacy_key, (emitted_key, legacy_key)
    assert torch.equal(emitted.cpu(), legacy_pos.cpu())


# ---------------------------------------------------------------------------
# F1: reservation wiring -- the legacy track's budget plan is fixed at entry,
# never the primary's remainder, and carries no wall-clock deadline.
# ---------------------------------------------------------------------------


def test_legacy_shadow_config_is_isolated_and_ledger_only() -> None:
    """The shadow config gets the full entry plan and no wall deadline (F1)."""
    from dagua.layout.ops.pipelines.native_budget import LEDGER_ATTR
    from dagua.layout.ops.pipelines.native_shadow_champion import (
        DISABLE_NEW_ARMS_ATTR,
        build_legacy_shadow_config,
        snapshot_ledger_plan,
    )

    config = LayoutConfig(**_CONFIG_KWARGS)
    install_budget_ledger(config, 50.0, reserved_tail_dwu=2.0, return_reserve_dwu=1.0)
    plan = snapshot_ledger_plan(config)
    assert plan is not None

    # Simulate the primary run: the shared ledger is spent, the wall/process
    # deadlines are installed and long expired.
    ledger = getattr(config, LEDGER_ATTR)
    ledger.spent_dwu = 44.0
    setattr(config, WALL_DEADLINE_ATTR, 0.0)
    setattr(config, PROCESS_DEADLINE_ATTR, 0.0)

    shadow = build_legacy_shadow_config(config, plan)
    assert getattr(shadow, DISABLE_NEW_ARMS_ATTR) is True
    assert getattr(shadow, WALL_DEADLINE_ATTR, None) is None
    assert getattr(shadow, PROCESS_DEADLINE_ATTR, None) is None
    shadow_ledger = getattr(shadow, LEDGER_ATTR)
    assert isinstance(shadow_ledger, NativeBudgetLedger)
    assert shadow_ledger is not ledger
    assert shadow_ledger.spent_dwu == 0.0
    assert shadow_ledger.total_dwu == pytest.approx(50.0)
    assert shadow_ledger.reserved_tail_dwu == pytest.approx(2.0)
    assert shadow_ledger.return_reserve_dwu == pytest.approx(1.0)
    # The primary's ledger and deadlines are untouched.
    assert ledger.spent_dwu == pytest.approx(44.0)
    assert getattr(config, WALL_DEADLINE_ATTR) == 0.0


def test_exhausted_primary_ledger_cannot_starve_the_legacy_track(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even a fully spent primary ledger never skips the safety track (F1)."""
    from dagua.layout.ops.pipelines.native_shadow_champion import (
        SHADOW_CONTEST_TELEMETRY_ATTR,
    )

    edge_index, num_nodes, node_sizes = _wheel_graph()
    _inject_bad_planar_candidate(monkeypatch)
    _prefer_new_arm_undirected(monkeypatch)

    # A tiny primary ledger: optional admissions starve, but the planar arm
    # admission is patched open so the displacement still occurs.
    monkeypatch.setattr(
        native_undirected,
        "admit_native_work",
        lambda config, cost, reason: True,
    )
    config = LayoutConfig(**_CONFIG_KWARGS)
    install_budget_ledger(config, 0.5)
    emitted = _run_pipeline(edge_index, num_nodes, node_sizes, config)

    telemetry = getattr(config, SHADOW_CONTEST_TELEMETRY_ATTR, None)
    assert telemetry is not None, "the legacy-track shadow contest must have run"
    assert "error" not in telemetry, telemetry
    assert telemetry["shadow_key"] >= telemetry["primary_key"]
    assert telemetry["emitted"] == "shadow"
    assert torch.equal(emitted.cpu(), telemetry["shadow_pos"])


# ---------------------------------------------------------------------------
# Trigger-only byte-inertness: no new-arm win, no orchestration.
# ---------------------------------------------------------------------------


def test_no_new_arm_win_is_byte_inert() -> None:
    """Rows without a new-arm contest win never reach the shadow branch."""
    from dagua.layout.ops.pipelines.dagua_native import _layout_dagua_native_single_track
    from dagua.layout.ops.pipelines.native_shadow_champion import (
        SHADOW_CONTEST_TELEMETRY_ATTR,
    )

    edge_index, num_nodes, node_sizes = _k5_graph()
    config = _fresh_config()
    emitted = _run_pipeline(edge_index, num_nodes, node_sizes, config)
    assert getattr(config, SHADOW_CONTEST_TELEMETRY_ATTR, None) is None

    single = _layout_dagua_native_single_track(
        edge_index,
        num_nodes,
        node_sizes,
        config=_fresh_config(),
        seed=42,
    )
    assert torch.equal(emitted.cpu(), single.cpu())
