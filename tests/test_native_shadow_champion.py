"""Tests for the shadow-champion legacy-track contest (sprint2 W2-4a, review F1-F4).

Every displacement test drives the REAL selection seams: candidates are
injected or biased at the generation/selection boundary only, and everything
downstream -- the shadow-package reservation, displacement recording, the
legacy-track shadow re-run, the final referee contest, and the emission -- is
production code end-to-end.

Contract under test (the packet spec):

- A new arm may win a contest only when the complete legacy-track shadow
  package (re-run + final referee) reserves all-or-nothing on the entry
  ledger; a veto fails closed to the legacy-family champion (F1).
- On a displacement the emitted drawing is the max of the two FINAL referee
  keys, ties to the legacy track (F4).
- The shadow track byte-equals the source-gated no-new-arm pipeline at the
  reserved budget (the legacy-track oracle, F3).
- After a displacement an ordinary shadow failure propagates; the displaced
  primary is never silently emitted (F1).

Top-level imports are restricted to APIs that already exist on the pre-fix
commit so the per-finding regression tests fail BEHAVIORALLY there, not at
collection; the reservation APIs are imported inside test bodies.
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
    LEDGER_ATTR,
    PROCESS_DEADLINE_ATTR,
    WALL_DEADLINE_ATTR,
    NativeBudgetLedger,
    install_budget_ledger,
)
from dagua.layout.ops.pipelines.native_shadow_champion import (
    DISABLE_NEW_ARMS_ATTR,
    SHADOW_CONTEST_TELEMETRY_ATTR,
    is_new_arm_candidate,
    new_arms_disabled,
)
from dagua.layout.ops.state import LayoutProblem

_CONFIG_KWARGS: Dict[str, Any] = {"seed": 42}
_LEDGER_DWU = 600.0
# Small enough that the priced shadow package is non-viable (the shadow track
# cannot even afford its own final referee pass), so every new-arm win must
# fail closed to the legacy champion.
_UNAFFORDABLE_LEDGER_DWU = 0.1


class _ShadowTrackFailure(RuntimeError):
    """Synthetic ordinary shadow-track failure (an OOM-like non-timeout)."""


def _fresh_config(ledger_dwu: float = _LEDGER_DWU, **overrides: Any) -> LayoutConfig:
    """Return a fresh test config with a fresh deterministic ledger."""
    config = LayoutConfig(**{**_CONFIG_KWARGS, **overrides})
    install_budget_ledger(config, ledger_dwu)
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


def _k5_ring_graph(tail: int = 15) -> tuple[torch.Tensor, int, torch.Tensor]:
    """Return K5 plus a cycle tail (non-planar, big enough for the t-FDP routes)."""
    edges = [(u, v) for u in range(5) for v in range(u + 1, 5)]
    for i in range(tail):
        edges.append((4 + i, 5 + i))
    edges.append((4 + tail, 0))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    num_nodes = 5 + tail
    return edge_index, num_nodes, torch.full((num_nodes, 2), 20.0)


def _declared_undirected_structure(edge_index: torch.Tensor, num_nodes: int) -> Any:
    """Classify a graph and declare it undirected (the benchmark corpus shape).

    The undirected-portfolio router (and with it the t-FDP contest seams)
    requires high-confidence undirectedness: a declaration or reciprocal
    edge storage. Declaring on the pre-classified structure reproduces how
    corpus undirected rows enter the pipeline.
    """
    import dataclasses

    structure = classify_graph(edge_index, num_nodes)
    return dataclasses.replace(
        structure,
        is_semantically_directed=False,
        direction_is_declared=True,
    )


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
    re-run and the veto fallback both select exactly as an unbiased legacy
    contest would.
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
    config: Optional[LayoutConfig] = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Run the true no-new-arm pipeline via pre-existing SOURCE gates.

    Closes the planar-arm structural gate, the t-FDP normal-contest
    admission, and the band eligibility at their sources (all exist on the
    pre-fix commit and are independent of the shadow mechanism's disable
    flag), so this reference is computable identically before and after the
    fix and remains an independent oracle for the shadow track.
    """
    with monkeypatch.context() as ctx:
        ctx.setattr(native_planar_arm, "planar_arm_admitted", lambda problem: False)
        ctx.setattr(
            native_undirected,
            "_sparse_contest_arm_admitted",
            lambda problem, config, seeds: (),
        )
        ctx.setattr(native_sparse_infrastructure, "sparse_band_contest_eligible", lambda p: False)
        return _run_pipeline(
            edge_index,
            num_nodes,
            node_sizes,
            config if config is not None else _fresh_config(),
            **kwargs,
        )


def _shadow_reference_config(
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    has_clusters: bool = False,
    has_weights: bool = False,
) -> LayoutConfig:
    """Build the shadow-budget legacy config exactly as production does.

    Prices the shadow package from an identical fresh entry ledger and hands
    it to the production shadow-config builder, so the reference run carries
    the same reserved ledger, the same entry sizing envelope, and no
    deadlines -- the budget the emitted shadow track actually had.
    """
    from dagua.layout.ops.pipelines.native_shadow_champion import (
        build_legacy_shadow_config,
        price_shadow_package,
        snapshot_ledger_plan,
    )

    entry = _fresh_config()
    plan = price_shadow_package(
        entry,
        num_nodes=num_nodes,
        num_edges=int(edge_index.shape[1]),
        has_clusters=has_clusters,
        has_weights=has_weights,
    )
    assert plan is not None and plan.viable
    return build_legacy_shadow_config(entry, snapshot_ledger_plan(entry), plan)


def _assert_displacement_contract(
    monkeypatch: pytest.MonkeyPatch,
    config: LayoutConfig,
    emitted: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: torch.Tensor,
    *,
    edge_weights: Optional[torch.Tensor] = None,
    clusters: Optional[dict[str, Any]] = None,
    graph_structure: Any = None,
) -> Dict[str, Any]:
    """Assert the full displacement contract on one solved row.

    (1) A displacement was recorded and both final referee keys exist.
    (2) The emitted drawing is the max of the two FINAL referee keys, ties
        to the legacy track (the spec's final-choice semantics, review F4).
    (3) The shadow track byte-equals the source-gated no-new-arm pipeline at
        the reserved shadow budget (the legacy-track oracle, review F3).
    """
    telemetry = getattr(config, SHADOW_CONTEST_TELEMETRY_ATTR, None)
    assert telemetry is not None, "a displacement row must record contest telemetry"
    assert telemetry["displacements"], telemetry
    if telemetry["shadow_key"] >= telemetry["primary_key"]:
        assert telemetry["emitted"] == "shadow", telemetry
        assert torch.equal(emitted.cpu(), telemetry["shadow_pos"])
    else:
        assert telemetry["emitted"] == "primary", telemetry
        assert torch.equal(emitted.cpu(), telemetry["primary_pos"])

    reference_config = _shadow_reference_config(
        edge_index,
        num_nodes,
        has_clusters=clusters is not None and bool(clusters),
        has_weights=edge_weights is not None,
    )
    run_kwargs: Dict[str, Any] = {}
    if edge_weights is not None:
        run_kwargs["edge_weights"] = edge_weights
    if clusters is not None:
        run_kwargs["clusters"] = clusters
    if graph_structure is not None:
        run_kwargs["graph_structure"] = graph_structure
    legacy_pos = _legacy_reference(
        monkeypatch,
        edge_index,
        num_nodes,
        node_sizes,
        config=reference_config,
        **run_kwargs,
    )
    assert torch.equal(telemetry["shadow_pos"], legacy_pos.cpu())
    return dict(telemetry)


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
# ---------------------------------------------------------------------------


def test_undirected_planar_displacement_recovers_legacy_track(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A planar displacement runs the funded shadow and emits the max (F1/F3)."""
    edge_index, num_nodes, node_sizes = _wheel_graph()
    _inject_bad_planar_candidate(monkeypatch)
    _prefer_new_arm_undirected(monkeypatch)

    config = _fresh_config()
    emitted = _run_pipeline(edge_index, num_nodes, node_sizes, config)
    _assert_displacement_contract(monkeypatch, config, emitted, edge_index, num_nodes, node_sizes)


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
    monkeypatch.setattr(
        native_undirected,
        "_sparse_contest_arm_admitted",
        lambda problem, config, seeds: seeds,
    )
    monkeypatch.setattr(
        native_sparse_infrastructure,
        "tfdp_sparse_positions",
        lambda problem, *, gamma, seed=None, node_sep=0.0: _bad_positions(problem.num_nodes),
    )


def test_undirected_tfdp_displacement_recovers_legacy_track(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A t-FDP normal-contest win is a protected displacement (F2)."""
    edge_index, num_nodes, node_sizes = _k5_ring_graph()
    structure = _declared_undirected_structure(edge_index, num_nodes)
    _admit_tfdp_into_normal_contest(monkeypatch)
    _prefer_new_arm_undirected(monkeypatch, prefix=("tfdp",))

    config = _fresh_config()
    emitted = _run_pipeline(edge_index, num_nodes, node_sizes, config, graph_structure=structure)
    _assert_displacement_contract(
        monkeypatch,
        config,
        emitted,
        edge_index,
        num_nodes,
        node_sizes,
        graph_structure=structure,
    )


# ---------------------------------------------------------------------------
# F2: t-FDP displacement in the W1-A sparse-band mini-contest.
# ---------------------------------------------------------------------------


def test_sparse_band_tfdp_displacement_recovers_legacy_track(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A t-FDP band win must keep the legacy early-return reachable (F2)."""
    edge_index, num_nodes, node_sizes = _k5_ring_graph()
    structure = _declared_undirected_structure(edge_index, num_nodes)
    # Shrink the contest ceiling so this row takes the band early-return
    # path; the legacy reference closes band eligibility, exactly the
    # pre-W1-A bare-incumbent behavior the invariant protects.
    monkeypatch.setattr(native_undirected, "MAX_CONTEST_NODES", 10)
    monkeypatch.setattr(
        native_sparse_infrastructure,
        "sparse_band_contest_eligible",
        lambda problem: True,
    )
    monkeypatch.setattr(
        native_sparse_infrastructure,
        "tfdp_sparse_positions",
        lambda problem, *, gamma, seed=None, node_sep=0.0: _bad_positions(problem.num_nodes),
    )
    _prefer_new_arm_undirected(monkeypatch, prefix=("tfdp",))

    config = _fresh_config()
    emitted = _run_pipeline(edge_index, num_nodes, node_sizes, config, graph_structure=structure)
    _assert_displacement_contract(
        monkeypatch,
        config,
        emitted,
        edge_index,
        num_nodes,
        node_sizes,
        graph_structure=structure,
    )


# ---------------------------------------------------------------------------
# F3 + F4: directed displacement with the post-argmax late stages live.
# The final choice is the max of the two FINAL referee keys, ties to legacy;
# a stronger primary is correctly kept (the reviewed af2ee897 measurement).
# ---------------------------------------------------------------------------


def test_directed_planar_displacement_recovers_legacy_track(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The directed legacy track replays late ordering/nested-stress seams (F3/F4)."""
    edge_index, num_nodes, node_sizes = _grid_dag_graph()
    _inject_bad_planar_candidate(monkeypatch)
    _prefer_new_arm_directed(monkeypatch)

    config = _fresh_config()
    emitted = _run_pipeline(edge_index, num_nodes, node_sizes, config)
    _assert_displacement_contract(monkeypatch, config, emitted, edge_index, num_nodes, node_sizes)


# ---------------------------------------------------------------------------
# F4.5: weighted and clustered planar displacement rows.
# ---------------------------------------------------------------------------


def test_weighted_planar_displacement_recovers_legacy_track(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Declared-weighted rows traverse the weighted terminal pre-stages (F4.5)."""
    edge_index, num_nodes, node_sizes = _wheel_graph()
    edge_weights = torch.full((edge_index.shape[1],), 3.0)
    _inject_bad_planar_candidate(monkeypatch)
    _prefer_new_arm_undirected(monkeypatch)

    config = _fresh_config()
    emitted = _run_pipeline(
        edge_index,
        num_nodes,
        node_sizes,
        config,
        edge_weights=edge_weights,
    )
    _assert_displacement_contract(
        monkeypatch,
        config,
        emitted,
        edge_index,
        num_nodes,
        node_sizes,
        edge_weights=edge_weights,
    )


def test_clustered_planar_displacement_recovers_legacy_track(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clustered rows traverse cluster tightening in both tracks (F4.5)."""
    edge_index, num_nodes, node_sizes = _wheel_graph()
    clusters = {
        "hub": [0, 1, 2, 3, 4],
        "rim": [5, 6, 7, 8],
    }
    _inject_bad_planar_candidate(monkeypatch)
    _prefer_new_arm_undirected(monkeypatch)

    config = _fresh_config()
    emitted = _run_pipeline(
        edge_index,
        num_nodes,
        node_sizes,
        config,
        clusters=clusters,
    )
    _assert_displacement_contract(
        monkeypatch,
        config,
        emitted,
        edge_index,
        num_nodes,
        node_sizes,
        clusters=clusters,
    )


# ---------------------------------------------------------------------------
# F4.6: a shortlist-composition change the old finalist-mapping restriction
# could not represent.
# ---------------------------------------------------------------------------


def test_shortlist_composition_change_still_recovers_legacy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The legacy track survives new-arm finalist-shortlist pressure (F4.6).

    The finalist cascade is patched so that whenever a new-arm candidate is
    present, NO legacy challenger reaches the referee: the with-arms scored
    mapping contains no legacy champion at all. The legacy re-run performs
    its own untampered finalist selection and must still recover the true
    legacy final drawing.
    """
    from dagua.layout.ops.pipelines import native_contest_cascade

    edge_index, num_nodes, node_sizes = _wheel_graph()
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

    config = _fresh_config()
    emitted = _run_pipeline(edge_index, num_nodes, node_sizes, config)
    _assert_displacement_contract(monkeypatch, config, emitted, edge_index, num_nodes, node_sizes)


# ---------------------------------------------------------------------------
# F3 + F4.7: byte-equality oracle against a true disabled run at the shadow
# budget, plus honest reservation accounting on the entry ledger.
# ---------------------------------------------------------------------------


def test_shadow_track_byte_equals_true_disabled_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The legacy shadow track IS a disabled run at the reserved budget (F4.7)."""
    edge_index, num_nodes, node_sizes = _wheel_graph()
    _inject_bad_planar_candidate(monkeypatch)
    _prefer_new_arm_undirected(monkeypatch)

    config = _fresh_config()
    emitted = _run_pipeline(edge_index, num_nodes, node_sizes, config)
    telemetry = _assert_displacement_contract(
        monkeypatch, config, emitted, edge_index, num_nodes, node_sizes
    )

    # Flag-oracle: a fresh disabled-flag run on the production shadow config
    # (reserved ledger, entry sizing envelope) reproduces the shadow bytes.
    disabled_config = _shadow_reference_config(edge_index, num_nodes)
    assert getattr(disabled_config, DISABLE_NEW_ARMS_ATTR) is True
    disabled_pos = _run_pipeline(edge_index, num_nodes, node_sizes, disabled_config)
    assert torch.equal(telemetry["shadow_pos"], disabled_pos.cpu())

    # Honest accounting on the entry ledger: the package was reserved at the
    # argmax, released to fund the shadow, and the final referee was charged.
    ledger = getattr(config, LEDGER_ATTR)
    assert isinstance(ledger, NativeBudgetLedger)
    reasons = [event["reason"] for event in ledger.event_log]
    assert "shadow_champion_package" in reasons
    assert "shadow_champion_fund_shadow_track" in reasons
    assert "shadow_champion_final_referee" in reasons
    plan = telemetry["shadow_plan"]
    assert plan is not None and plan.viable
    assert ledger.reserved_tail_dwu == pytest.approx(0.0)
    assert ledger.spent_dwu >= plan.referee_dwu


# ---------------------------------------------------------------------------
# F1: reservation mechanics -- priced from entry state, reserved
# all-or-nothing on the live ledger, idempotent, honest vetoes.
# ---------------------------------------------------------------------------


def test_shadow_package_reservation_is_all_or_nothing() -> None:
    """The package reserves in full on the entry ledger or not at all (F1)."""
    from dagua.layout.ops.pipelines.native_shadow_champion import (
        install_shadow_reservation_state,
        price_shadow_package,
        try_reserve_shadow_package,
    )

    config = LayoutConfig(**_CONFIG_KWARGS)
    install_budget_ledger(config, 50.0, reserved_tail_dwu=2.0, return_reserve_dwu=1.0)
    plan = price_shadow_package(
        config, num_nodes=9, num_edges=16, has_clusters=False, has_weights=False
    )
    assert plan is not None and plan.viable
    # Half the free entry capacity plus the two-track final referee pass.
    free_entry = 0.9 * 50.0 - 3.0
    assert plan.shadow_track_dwu == pytest.approx(0.5 * (free_entry - plan.referee_dwu))
    assert plan.package_dwu == pytest.approx(plan.shadow_track_dwu + plan.referee_dwu)

    state = install_shadow_reservation_state(config, plan)
    ledger = getattr(config, LEDGER_ATTR)
    assert try_reserve_shadow_package(config, route="undirected", winner_name="planar_x")
    assert state["reserved"] is True
    assert ledger.reserved_tail_dwu == pytest.approx(2.0 + plan.package_dwu)
    # Idempotent: one package funds the single whole-solve shadow re-run.
    assert try_reserve_shadow_package(config, route="directed", winner_name="tfdp_g2")
    assert ledger.reserved_tail_dwu == pytest.approx(2.0 + plan.package_dwu)

    # All-or-nothing: a ledger too spent at the argmax vetoes the package.
    spent_config = LayoutConfig(**_CONFIG_KWARGS)
    install_budget_ledger(spent_config, 50.0)
    spent_plan = price_shadow_package(
        spent_config, num_nodes=9, num_edges=16, has_clusters=False, has_weights=False
    )
    spent_state = install_shadow_reservation_state(spent_config, spent_plan)
    spent_ledger = getattr(spent_config, LEDGER_ATTR)
    spent_ledger.spent_dwu = 44.0
    assert not try_reserve_shadow_package(spent_config, route="undirected", winner_name="planar_x")
    assert spent_state["reserved"] is False
    assert spent_ledger.reserved_tail_dwu == pytest.approx(0.0)
    assert spent_state["vetoes"][0]["reason"] == "package_does_not_fit"

    # Non-viable package (the shadow track cannot afford its own referee).
    tiny_config = LayoutConfig(**_CONFIG_KWARGS)
    install_budget_ledger(tiny_config, _UNAFFORDABLE_LEDGER_DWU)
    tiny_plan = price_shadow_package(
        tiny_config, num_nodes=9, num_edges=16, has_clusters=False, has_weights=False
    )
    assert tiny_plan is not None and not tiny_plan.viable
    tiny_state = install_shadow_reservation_state(tiny_config, tiny_plan)
    assert not try_reserve_shadow_package(tiny_config, route="undirected", winner_name="planar_x")
    assert tiny_state["vetoes"][0]["reason"] == "package_not_viable"


def test_legacy_shadow_config_is_isolated_and_ledger_only() -> None:
    """The shadow config gets the reserved budget and no wall deadline (F1)."""
    from dagua.layout.ops.pipelines.native_budget import (
        DETERMINISTIC_BUDGET_ATTR,
        TOTAL_BUDGET_ATTR,
    )
    from dagua.layout.ops.pipelines.native_shadow_champion import (
        SHADOW_RESERVATION_STATE_ATTR,
        build_legacy_shadow_config,
        install_shadow_reservation_state,
        price_shadow_package,
        snapshot_ledger_plan,
    )

    config = LayoutConfig(**_CONFIG_KWARGS)
    install_budget_ledger(config, 50.0, reserved_tail_dwu=2.0, return_reserve_dwu=1.0)
    ledger_plan = snapshot_ledger_plan(config)
    package_plan = price_shadow_package(
        config, num_nodes=9, num_edges=16, has_clusters=False, has_weights=False
    )
    assert ledger_plan is not None and package_plan is not None
    install_shadow_reservation_state(config, package_plan)

    # Simulate the primary run: the shared ledger is spent, the wall/process
    # deadlines are installed and long expired.
    ledger = getattr(config, LEDGER_ATTR)
    ledger.spent_dwu = 30.0
    setattr(config, WALL_DEADLINE_ATTR, 0.0)
    setattr(config, PROCESS_DEADLINE_ATTR, 0.0)

    shadow = build_legacy_shadow_config(config, ledger_plan, package_plan)
    assert getattr(shadow, DISABLE_NEW_ARMS_ATTR) is True
    assert getattr(shadow, WALL_DEADLINE_ATTR, None) is None
    assert getattr(shadow, PROCESS_DEADLINE_ATTR, None) is None
    assert getattr(shadow, SHADOW_RESERVATION_STATE_ATTR, None) is None
    shadow_ledger = getattr(shadow, LEDGER_ATTR)
    assert isinstance(shadow_ledger, NativeBudgetLedger)
    assert shadow_ledger is not ledger
    assert shadow_ledger.spent_dwu == 0.0
    # The fresh ledger is the RESERVED track budget, never the entry plan.
    assert shadow_ledger.total_dwu == pytest.approx(package_plan.shadow_track_dwu)
    assert shadow_ledger.safety == pytest.approx(ledger_plan.safety)
    assert shadow_ledger.reserved_tail_dwu == pytest.approx(0.0)
    assert shadow_ledger.return_reserve_dwu == pytest.approx(0.0)
    # Sizing heuristics keyed to the row's total envelope see the entry plan.
    assert getattr(shadow, TOTAL_BUDGET_ATTR) == pytest.approx(50.0)
    assert getattr(shadow, DETERMINISTIC_BUDGET_ATTR) == pytest.approx(50.0)
    # The primary's ledger and deadlines are untouched.
    assert ledger.spent_dwu == pytest.approx(30.0)
    assert getattr(config, WALL_DEADLINE_ATTR) == 0.0


# ---------------------------------------------------------------------------
# F1 regression: an unaffordable shadow package fails closed to the legacy
# champion (no displacement, no shadow, never the weaker new-arm drawing).
# ---------------------------------------------------------------------------


def test_unaffordable_shadow_package_fails_closed_to_legacy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Budget pressure disarms the new arm, not the safety track (F1)."""
    edge_index, num_nodes, node_sizes = _wheel_graph()
    _inject_bad_planar_candidate(monkeypatch)
    _prefer_new_arm_undirected(monkeypatch)
    # Generation admission is opened so the arm is generated and would win;
    # the reservation gate itself stays real and must veto the win.
    monkeypatch.setattr(
        native_undirected,
        "admit_native_work",
        lambda config, cost, reason: True,
    )
    monkeypatch.setattr(native_undirected, "_portfolio_has_budget", lambda *a, **k: True)

    config = LayoutConfig(**_CONFIG_KWARGS)
    install_budget_ledger(config, _UNAFFORDABLE_LEDGER_DWU)
    emitted = _run_pipeline(edge_index, num_nodes, node_sizes, config)

    telemetry = getattr(config, SHADOW_CONTEST_TELEMETRY_ATTR, None)
    assert telemetry is not None, "a vetoed displacement must record telemetry"
    assert telemetry["emitted"] == "primary"
    assert telemetry["displacements"] == []
    assert telemetry["reservation_vetoes"], telemetry
    assert "shadow_key" not in telemetry, "no shadow may run on a vetoed row"

    # The emitted drawing IS the legacy track: byte-equal to the source-gated
    # no-new-arm pipeline on an identical entry ledger.
    reference = LayoutConfig(**_CONFIG_KWARGS)
    install_budget_ledger(reference, _UNAFFORDABLE_LEDGER_DWU)
    legacy_pos = _legacy_reference(monkeypatch, edge_index, num_nodes, node_sizes, config=reference)
    assert torch.equal(emitted.cpu(), legacy_pos.cpu())


# ---------------------------------------------------------------------------
# F1 regression: after a displacement, an ordinary shadow failure propagates;
# the displaced primary is never silently emitted.
# ---------------------------------------------------------------------------


def test_shadow_failure_raises_never_emits_displaced_primary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An OOM-like shadow-track failure must not fall back to the primary (F1)."""
    edge_index, num_nodes, node_sizes = _wheel_graph()
    _inject_bad_planar_candidate(monkeypatch)
    _prefer_new_arm_undirected(monkeypatch)

    real_single_track = dagua_native._layout_dagua_native_single_track

    def failing_shadow_track(*args: Any, **kwargs: Any) -> torch.Tensor:
        if new_arms_disabled(kwargs.get("config")):
            raise _ShadowTrackFailure("synthetic shadow-track resource failure")
        return real_single_track(*args, **kwargs)

    monkeypatch.setattr(dagua_native, "_layout_dagua_native_single_track", failing_shadow_track)

    config = _fresh_config()
    with pytest.raises(_ShadowTrackFailure):
        _run_pipeline(edge_index, num_nodes, node_sizes, config)
    telemetry = getattr(config, SHADOW_CONTEST_TELEMETRY_ATTR, None)
    assert telemetry is not None
    assert telemetry["emitted"] == "error"
    assert telemetry["displacements"], telemetry


# ---------------------------------------------------------------------------
# F1 regression: the scale-anytime caller (layout_native_at_coarsest_scale
# installs `_dagua_scale_anytime_native` and skips orchestration) must never
# see a new-arm contest win -- fail-closed to the legacy track.
# ---------------------------------------------------------------------------


def test_scale_anytime_caller_never_runs_new_arms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The scale-anytime path gate-closes every new-arm family (F1)."""
    from dagua.layout.ops.pipelines.native_shadow_champion import (
        NEW_ARM_DISPLACEMENTS_ATTR,
    )

    edge_index, num_nodes, node_sizes = _wheel_graph()
    _inject_bad_planar_candidate(monkeypatch)
    _prefer_new_arm_undirected(monkeypatch)

    config = _fresh_config()
    setattr(config, "_dagua_scale_anytime_native", True)
    emitted = _run_pipeline(edge_index, num_nodes, node_sizes, config)

    assert getattr(config, SHADOW_CONTEST_TELEMETRY_ATTR, None) is None
    assert not getattr(config, NEW_ARM_DISPLACEMENTS_ATTR, [])

    reference = _fresh_config()
    setattr(reference, "_dagua_scale_anytime_native", True)
    setattr(reference, DISABLE_NEW_ARMS_ATTR, True)
    disabled_pos = _run_pipeline(edge_index, num_nodes, node_sizes, reference)
    assert torch.equal(emitted.cpu(), disabled_pos.cpu())


# ---------------------------------------------------------------------------
# Trigger-only byte-inertness: no new-arm win, no orchestration.
# ---------------------------------------------------------------------------


def test_no_new_arm_win_is_byte_inert() -> None:
    """Rows without a new-arm contest win never reach the shadow branch."""
    from dagua.layout.ops.pipelines.dagua_native import _layout_dagua_native_single_track

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
