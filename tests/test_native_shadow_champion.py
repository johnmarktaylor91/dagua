"""Tests for the shadow-champion terminal contest (sprint2 W2-4a, C10 core).

The rome/grafo1000.14 regression (DIAG_GRAFO1000.md): the W1-B planar arm can
honestly win the marketplace contest from a poor terminal-anneal basin, while
the displaced legacy winner's basin polishes far higher. The fix carries BOTH
champions through the deterministic terminal chain and emits the drawing the
runtime referee scores higher, ties to the legacy track.
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest
import torch

from dagua.config import LayoutConfig
from dagua.layout.ops.pipelines.dagua_native import _terminal_w5_polish
from dagua.layout.ops.pipelines.native_shadow_champion import (
    ShadowChampion,
    is_new_arm_candidate,
    legacy_shadow_name,
    pop_shadow_champion,
    stash_shadow_champion,
)


def _shadow_config() -> LayoutConfig:
    """Build a terminal-owner config for shadow-contest unit tests.

    Returns
    -------
    LayoutConfig
        Native config that reaches the terminal chain directly.
    """
    config = LayoutConfig(
        steps=1,
        edge_equalize_polish=True,
        decompose_components=False,
        route_flat_to_stress=False,
        force_pipeline="hybrid",
    )
    config._dagua_native_terminal_w5_owner = True
    return config


def _shadow_fixture_graph() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return an 8-node chorded cycle with a clean and a jumbled drawing.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        Edge index, node sizes, a poor-basin drawing (contest-winner shaped),
        and a clean drawing (displaced legacy champion shaped).
    """
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 0, 2], [1, 2, 3, 4, 5, 6, 7, 0, 4, 6]],
        dtype=torch.long,
    )
    node_sizes = torch.full((8, 2), 4.0, dtype=torch.float32)
    angles = torch.arange(8, dtype=torch.float32) * (2.0 * torch.pi / 8.0)
    clean_pos = torch.stack((torch.cos(angles), torch.sin(angles)), dim=1) * 100.0
    jumbled_pos = torch.stack(
        (
            torch.arange(8, dtype=torch.float32) * 3.0,
            (torch.arange(8, dtype=torch.float32) % 2.0) * 2.0,
        ),
        dim=1,
    )
    return edge_index, node_sizes, jumbled_pos, clean_pos


def _install_chain_stage_stubs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    anneal_improves: dict[int, torch.Tensor],
) -> None:
    """Stub the terminal chain stages with a basin-dependent anneal.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Active monkeypatch fixture.
    anneal_improves : dict[int, torch.Tensor]
        Map from an incumbent tensor's ``id`` to the annealed winner it
        reaches. Incumbents not in the map sit in a poor basin (no strict
        V3-improving perturbation found).
    """
    native_finisher = importlib.import_module("dagua.layout.ops.pipelines.native_finisher")
    from dagua.layout.ops.pipelines.native_finisher import (
        W5ContinuousFacetPolishResult,
        W5GlobalScaleSweepResult,
        W5SMACOFStressResult,
        W5SmallNAnnealResult,
        make_w5_skip_result,
    )

    monkeypatch.setattr(native_finisher, "w5_predicted_skip_reason", lambda *args: None)
    monkeypatch.setattr(native_finisher, "_finisher_slice_s", lambda config: 1.0)

    def fake_run_w5_finisher(**kwargs: Any) -> Any:
        """Return a no-accept W5 result for the track incumbent."""
        return make_w5_skip_result(
            incumbent_pos=kwargs["incumbent_pos"],
            incumbent_score_pair=kwargs["incumbent_score_pair"],
            reason="shadow_test_stub",
            edge_index=kwargs.get("edge_index"),
            config=kwargs.get("config"),
        )

    def fake_scale_sweep(**kwargs: Any) -> Any:
        """Return a not-selected terminal scale sweep."""
        return W5GlobalScaleSweepResult(
            winner_pos=kwargs["incumbent_pos"],
            winner_score_pair=kwargs["incumbent_score_pair"],
            winner_scale=1.0,
            selected=False,
            candidates=(),
        )

    def fake_smacof(**kwargs: Any) -> Any:
        """Return a not-selected terminal SMACOF stress polish."""
        return W5SMACOFStressResult(
            winner_pos=kwargs["incumbent_pos"],
            winner_score_pair=kwargs["incumbent_score_pair"],
            selected=False,
            skipped_reason="shadow_test_stub",
            candidates=(),
        )

    def fake_facet_polish(**kwargs: Any) -> Any:
        """Return a not-selected terminal continuous facet polish."""
        return W5ContinuousFacetPolishResult(
            winner_pos=kwargs["incumbent_pos"],
            winner_score_pair=kwargs["incumbent_score_pair"],
            selected=False,
            skipped_reason="shadow_test_stub",
            gate_reason="shadow_test",
            passes_completed=0,
            evaluations=0,
            accepted=(),
        )

    def fake_small_n_anneal(**kwargs: Any) -> Any:
        """Return a basin-dependent terminal anneal result."""
        incumbent_pos = kwargs["incumbent_pos"]
        annealed = anneal_improves.get(id(incumbent_pos))
        if annealed is None:
            return W5SmallNAnnealResult(
                winner_pos=incumbent_pos,
                winner_score_pair=kwargs["incumbent_score_pair"],
                selected=False,
                trials_completed=1,
                accepted_count=0,
                skipped_reason=None,
                candidates=(),
            )
        return W5SmallNAnnealResult(
            winner_pos=annealed,
            winner_score_pair=kwargs["incumbent_score_pair"],
            selected=True,
            trials_completed=1,
            accepted_count=1,
            skipped_reason=None,
            candidates=(),
        )

    monkeypatch.setattr(native_finisher, "run_w5_finisher", fake_run_w5_finisher)
    monkeypatch.setattr(native_finisher, "run_w5_terminal_global_scale_sweep", fake_scale_sweep)
    monkeypatch.setattr(
        native_finisher,
        "run_w5_terminal_smacof_stress_polish",
        fake_smacof,
    )
    monkeypatch.setattr(
        native_finisher,
        "run_w5_terminal_continuous_facet_polish",
        fake_facet_polish,
    )
    monkeypatch.setattr(native_finisher, "run_w5_terminal_small_n_anneal", fake_small_n_anneal)


def test_is_new_arm_candidate_matches_planar_family_only() -> None:
    """Only planar-arm candidate names arm the shadow contest."""
    assert is_new_arm_candidate("planar_schnyder_f2_polished_convergent")
    assert is_new_arm_candidate("planar_fpp_polished")
    assert is_new_arm_candidate("planar_seeded_stress")
    assert not is_new_arm_candidate("fcose_seed2_raw")
    assert not is_new_arm_candidate("incumbent")
    assert not is_new_arm_candidate("neato")


def test_legacy_shadow_name_detects_undirected_displacement() -> None:
    """A planar contest win over a legacy field yields the legacy argmax."""
    from dagua.layout.ops.pipelines.native_undirected import (
        _ClusterScoreTelemetry,
        _select_undirected_winner,
    )

    def telemetry(v3: float, extended: float) -> _ClusterScoreTelemetry:
        """Build minimal contest telemetry for one candidate."""
        return _ClusterScoreTelemetry(
            extended_score=extended,
            old_score=extended,
            metrics={},
            v3_tiered=v3,
        )

    scores = {
        "incumbent": 47.9,
        "fcose_seed2_raw": 75.8,
        "planar_schnyder_f2_polished_convergent": 78.2,
    }
    telemetries = {
        "incumbent": telemetry(47.9, 47.9),
        "fcose_seed2_raw": telemetry(75.8, 75.8),
        "planar_schnyder_f2_polished_convergent": telemetry(78.2, 78.2),
    }
    best_name = _select_undirected_winner(scores, telemetries)
    assert best_name == "planar_schnyder_f2_polished_convergent"
    assert (
        legacy_shadow_name(best_name, scores, telemetries, _select_undirected_winner)
        == "fcose_seed2_raw"
    )


def test_legacy_shadow_name_gate_stays_closed_without_displacement() -> None:
    """A legacy contest winner never arms the shadow contest."""
    from dagua.layout.ops.pipelines.native_undirected import (
        _ClusterScoreTelemetry,
        _select_undirected_winner,
    )

    def telemetry(v3: float) -> _ClusterScoreTelemetry:
        """Build minimal contest telemetry for one candidate."""
        return _ClusterScoreTelemetry(
            extended_score=v3,
            old_score=v3,
            metrics={},
            v3_tiered=v3,
        )

    scores = {
        "incumbent": 47.9,
        "fcose_seed2_raw": 79.1,
        "planar_schnyder_f2_polished_convergent": 78.2,
    }
    telemetries = {name: telemetry(score) for name, score in scores.items()}
    best_name = _select_undirected_winner(scores, telemetries)
    assert best_name == "fcose_seed2_raw"
    assert legacy_shadow_name(best_name, scores, telemetries, _select_undirected_winner) is None
    # A field with no legacy candidate at all cannot arm the gate either.
    planar_only = {"planar_fpp_polished": 78.2}
    assert (
        legacy_shadow_name(
            "planar_fpp_polished",
            planar_only,
            {},
            _select_undirected_winner,
        )
        is None
    )


def test_legacy_shadow_name_detects_directed_displacement() -> None:
    """The directed contest's selection semantics arm the shadow identically."""
    from dagua.layout.ops.pipelines.native_directed import (
        _DirectedClusterScoreTelemetry,
        _select_directed_winner,
    )

    def telemetry(v3: float) -> _DirectedClusterScoreTelemetry:
        """Build minimal directed contest telemetry for one candidate."""
        return _DirectedClusterScoreTelemetry(
            extended_score=v3,
            old_score=v3,
            metrics={},
            v3_tiered=v3,
        )

    scores = {
        "incumbent": 60.0,
        "dot_order": 71.5,
        "planar_fpp_f1_polished": 74.0,
    }
    telemetries = {name: telemetry(score) for name, score in scores.items()}
    best_name = _select_directed_winner(scores, telemetries)
    assert best_name == "planar_fpp_f1_polished"
    assert (
        legacy_shadow_name(best_name, scores, telemetries, _select_directed_winner) == "dot_order"
    )


def test_pop_shadow_champion_consumes_and_validates_shape() -> None:
    """The stash is consumed once and dropped on component-shape mismatch."""
    config = _shadow_config()
    champion = ShadowChampion(
        route="undirected",
        winner_name="planar_fpp_polished",
        shadow_name="fcose_seed2_raw",
        pos=torch.zeros((8, 2), dtype=torch.float32),
    )
    stash_shadow_champion(config, champion)
    assert pop_shadow_champion(config, expected_nodes=8) is champion
    # Consumed: a second pop finds nothing.
    assert pop_shadow_champion(config, expected_nodes=8) is None
    # A component-local stash whose shape does not match the terminal tensor
    # is discarded (byte-inert drop).
    stash_shadow_champion(config, champion)
    assert pop_shadow_champion(config, expected_nodes=14) is None
    assert getattr(config, "_dagua_native_shadow_champion", None) is None


def test_terminal_contest_emits_legacy_polished_drawing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """grafo1000.14-shaped scenario: the legacy basin's polish wins the final.

    The planar-shaped contest winner enters the terminal chain from a poor
    basin (anneal finds nothing); the displaced legacy champion's basin
    polishes higher. The emitted drawing must be the legacy-polished one.
    """
    edge_index, node_sizes, jumbled_pos, clean_pos = _shadow_fixture_graph()
    legacy_polished = clean_pos * 1.05
    _install_chain_stage_stubs(
        monkeypatch,
        anneal_improves={id(clean_pos): legacy_polished},
    )
    config = _shadow_config()
    stash_shadow_champion(
        config,
        ShadowChampion(
            route="undirected",
            winner_name="planar_schnyder_f2_polished_convergent",
            shadow_name="fcose_seed2_raw",
            pos=clean_pos,
        ),
    )

    actual = _terminal_w5_polish(
        jumbled_pos,
        edge_index=edge_index,
        node_sizes=node_sizes,
        config=config,
        structure=None,
        direction="TB",
    )

    assert torch.equal(actual, legacy_polished)
    # The stash is consumed by the contest.
    assert getattr(config, "_dagua_native_shadow_champion", None) is None


def test_terminal_contest_preserves_stronger_primary_track(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shadow max can only help: a stronger planar final is preserved.

    This is the unit-level guard for the seven W1-B planar wins -- when the
    primary (new-arm) track finishes higher, the emitted drawing is
    byte-identical to the no-shadow run.
    """
    edge_index, node_sizes, jumbled_pos, clean_pos = _shadow_fixture_graph()
    primary_polished = clean_pos * 1.05
    baseline_incumbent = clean_pos.clone()

    _install_chain_stage_stubs(
        monkeypatch,
        anneal_improves={
            id(clean_pos): primary_polished,
            id(baseline_incumbent): primary_polished,
        },
    )
    baseline_config = _shadow_config()
    baseline = _terminal_w5_polish(
        baseline_incumbent,
        edge_index=edge_index,
        node_sizes=node_sizes,
        config=baseline_config,
        structure=None,
        direction="TB",
    )

    config = _shadow_config()
    stash_shadow_champion(
        config,
        ShadowChampion(
            route="undirected",
            winner_name="planar_schnyder_f2_polished_convergent",
            shadow_name="fcose_seed2_raw",
            pos=jumbled_pos,
        ),
    )
    actual = _terminal_w5_polish(
        clean_pos,
        edge_index=edge_index,
        node_sizes=node_sizes,
        config=config,
        structure=None,
        direction="TB",
    )

    assert torch.equal(baseline, primary_polished)
    assert torch.equal(actual, baseline)


def test_terminal_chain_is_byte_inert_without_displacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a stash the terminal chain emits the primary track unchanged."""
    edge_index, node_sizes, jumbled_pos, clean_pos = _shadow_fixture_graph()
    del jumbled_pos
    _install_chain_stage_stubs(monkeypatch, anneal_improves={})
    config = _shadow_config()

    actual = _terminal_w5_polish(
        clean_pos,
        edge_index=edge_index,
        node_sizes=node_sizes,
        config=config,
        structure=None,
        direction="TB",
    )

    assert torch.equal(actual, clean_pos)


def test_terminal_contest_tie_goes_to_the_legacy_track(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Identical final referee keys emit the legacy-track drawing.

    The two tracks finish with byte-identical drawings held in distinct
    tensors, so both referee keys are exactly equal; the tie must emit the
    legacy/incumbent-family track's tensor.
    """
    edge_index, node_sizes, _jumbled_pos, clean_pos = _shadow_fixture_graph()
    primary_pos = clean_pos.clone()
    _install_chain_stage_stubs(monkeypatch, anneal_improves={})
    config = _shadow_config()
    stash_shadow_champion(
        config,
        ShadowChampion(
            route="undirected",
            winner_name="planar_schnyder_f2_polished_convergent",
            shadow_name="fcose_seed2_raw",
            pos=clean_pos,
        ),
    )

    actual = _terminal_w5_polish(
        primary_pos,
        edge_index=edge_index,
        node_sizes=node_sizes,
        config=config,
        structure=None,
        direction="TB",
    )

    assert torch.equal(actual, clean_pos)
    # Same drawing in both tracks: identity proves the tie went to the
    # legacy-track tensor, not the primary clone.
    assert actual is clean_pos


def test_shadow_contest_respects_ledger_veto(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A vetoed ledger admission skips the shadow track deterministically."""
    edge_index, node_sizes, jumbled_pos, clean_pos = _shadow_fixture_graph()
    _install_chain_stage_stubs(monkeypatch, anneal_improves={})
    native_budget = importlib.import_module("dagua.layout.ops.pipelines.native_budget")
    monkeypatch.setattr(native_budget, "admit_native_work", lambda *args: False)
    config = _shadow_config()
    stash_shadow_champion(
        config,
        ShadowChampion(
            route="undirected",
            winner_name="planar_schnyder_f2_polished_convergent",
            shadow_name="fcose_seed2_raw",
            pos=clean_pos,
        ),
    )

    actual = _terminal_w5_polish(
        jumbled_pos,
        edge_index=edge_index,
        node_sizes=node_sizes,
        config=config,
        structure=None,
        direction="TB",
    )

    assert torch.equal(actual, jumbled_pos)
