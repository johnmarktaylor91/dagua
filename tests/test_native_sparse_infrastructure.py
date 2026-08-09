"""Tests for the sparse-infrastructure detector, t-FDP arm, and band contest.

Sprint2 W1-A. Probe graphs are GENERATED in-test (fresh sizes/seeds not
present in any corpus) so the detector tests double as out-of-corpus routing
probes: thresholds must react to structure only, never to names or ids.
"""

from __future__ import annotations

import dataclasses
import importlib
import time
from typing import Any, cast

import pytest
import torch

from dagua.config import LayoutConfig
from dagua.layout.graph_classify import GraphFamily, GraphStructure, classify_graph
from dagua.layout.ops.pipelines.dagua_native import _undirected_route_shortlist
from dagua.layout.ops.pipelines.native_sparse_infrastructure import (
    SPARSE_INFRA,
    sparse_band_contest_eligible,
    sparse_band_mini_contest,
    sparse_infrastructure_gate,
    tfdp_iteration_schedule,
    tfdp_pivot_schedule,
    tfdp_sparse_positions,
)
from dagua.layout.ops.pipelines.native_undirected import _ClusterScoreTelemetry
from dagua.layout.ops.state import LayoutProblem


def _sparse_structure(num_nodes: int = 1612, **overrides: object) -> GraphStructure:
    """Return a hand-built structure matching the sparse-infrastructure class.

    Parameters
    ----------
    num_nodes : int, default=1612
        Node count the diameter default is sized for.
    **overrides : object
        Field overrides applied over the firing baseline.

    Returns
    -------
    GraphStructure
        Structure whose defaults fire the detector at ``num_nodes``.
    """
    base = GraphStructure(
        family=GraphFamily.GENERAL,
        num_components=1,
        max_degree=6,
        num_layers=0,
        avg_layer_width=0.0,
        is_planar_hint=True,
        edge_to_node_ratio=1.3,
        is_semantically_directed=False,
        has_dominant_component=True,
        degree_uniformity=0.4,
        hub_edge_fraction=0.2,
        diameter_estimate=int(2.0 * float(num_nodes) ** 0.5),
    )
    return dataclasses.replace(base, **overrides)  # type: ignore[arg-type]


def _chained_ring_edges(num_nodes: int, chord_every: int = 40) -> torch.Tensor:
    """Return a ring with sparse short chords (meshed-tree-like, hub-free).

    Parameters
    ----------
    num_nodes : int
        Ring size.
    chord_every : int, default=40
        One local chord is added every ``chord_every`` nodes.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, E]``.
    """
    edges = [(i, (i + 1) % num_nodes) for i in range(num_nodes)]
    edges.extend((i, (i + 3) % num_nodes) for i in range(0, num_nodes, chord_every))
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def _star_edges(num_nodes: int) -> torch.Tensor:
    """Return a star graph (maximal hub concentration).

    Parameters
    ----------
    num_nodes : int
        Total node count including the hub.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, N - 1]``.
    """
    return torch.tensor([[0] * (num_nodes - 1), list(range(1, num_nodes))], dtype=torch.long)


class TestDetector:
    def test_fires_on_sparse_infrastructure_profile(self) -> None:
        assert sparse_infrastructure_gate(_sparse_structure(), 1612)

    def test_closed_on_hub_concentration(self) -> None:
        structure = _sparse_structure(hub_edge_fraction=0.6)
        assert not sparse_infrastructure_gate(structure, 1612)

    def test_closed_on_high_max_degree(self) -> None:
        structure = _sparse_structure(max_degree=25)
        assert not sparse_infrastructure_gate(structure, 1612)

    def test_density_escape_only_at_band_ratio(self) -> None:
        structure = _sparse_structure(edge_to_node_ratio=2.5)
        assert not sparse_infrastructure_gate(structure, 1612)
        assert sparse_infrastructure_gate(structure, 1612, ratio_max=SPARSE_INFRA.band_ratio_max)

    def test_closed_on_short_diameter(self) -> None:
        # log-N-diameter small-world profile: 20 < 0.65 * sqrt(1612) ~ 26.1.
        structure = _sparse_structure(diameter_estimate=20)
        assert not sparse_infrastructure_gate(structure, 1612)

    def test_closed_on_unmeasured_diameter(self) -> None:
        structure = _sparse_structure(diameter_estimate=0)
        assert not sparse_infrastructure_gate(structure, 1612)

    def test_closed_below_min_nodes(self) -> None:
        assert not sparse_infrastructure_gate(_sparse_structure(num_nodes=40), 40)

    def test_class_floor_is_the_cousin_justified_200(self) -> None:
        # Review F1 regression: the class floor is the spec's n >= 200. The
        # dev-fitted 50-node floor opened the gate on small non-class rows
        # (random_bipartite_60 spread ratio regressed 1.82 -> 5.84); lowering
        # it again is a training-cousin fit decision, never a dev-row one.
        assert not sparse_infrastructure_gate(_sparse_structure(num_nodes=97), 97)
        assert not sparse_infrastructure_gate(_sparse_structure(num_nodes=199), 199)
        assert sparse_infrastructure_gate(_sparse_structure(num_nodes=200), 200)

    def test_closed_on_declared_directed(self) -> None:
        structure = _sparse_structure(is_semantically_directed=True)
        assert not sparse_infrastructure_gate(structure, 1612)

    def test_closed_on_unknown_semantic_direction(self) -> None:
        # Review F5 regression: ``None`` (unmeasured) fails CLOSED like every
        # other unmeasured feature; only an explicit ``False`` opens the gate.
        structure = _sparse_structure(is_semantically_directed=None)
        assert not sparse_infrastructure_gate(structure, 1612)

    def test_closed_without_dominant_component(self) -> None:
        structure = _sparse_structure(has_dominant_component=False)
        assert not sparse_infrastructure_gate(structure, 1612)

    def test_closed_on_clusters_and_none_structure(self) -> None:
        assert not sparse_infrastructure_gate(_sparse_structure(), 1612, has_clusters=True)
        assert not sparse_infrastructure_gate(None, 1612)

    def test_fires_on_classified_synthetic_mesh(self) -> None:
        """Real classify_graph output on a fresh hub-free long ring fires."""
        num_nodes = 640
        structure = classify_graph(_chained_ring_edges(num_nodes), num_nodes)
        # Pin only the semantic-direction inference (test isolates the
        # structural thresholds, not the direction heuristic).
        structure = dataclasses.replace(structure, is_semantically_directed=False)
        assert sparse_infrastructure_gate(structure, num_nodes)

    def test_closed_on_classified_star(self) -> None:
        num_nodes = 300
        structure = classify_graph(_star_edges(num_nodes), num_nodes)
        structure = dataclasses.replace(structure, is_semantically_directed=False)
        assert not sparse_infrastructure_gate(structure, num_nodes)


class TestSchedules:
    def test_iteration_schedule_reference_below_band(self) -> None:
        assert tfdp_iteration_schedule(1138) == 300
        assert tfdp_iteration_schedule(1612) == 300

    def test_iteration_schedule_tapers_and_floors(self) -> None:
        assert tfdp_iteration_schedule(3000) == 200
        assert tfdp_iteration_schedule(10_000) == 150
        sizes = [200, 800, 1500, 2000, 2500, 3000]
        schedule = [tfdp_iteration_schedule(size) for size in sizes]
        assert schedule == sorted(schedule, reverse=True)

    def test_pivot_schedule_reference_floor_and_sqrt_growth(self) -> None:
        assert tfdp_pivot_schedule(1138) == 100
        assert tfdp_pivot_schedule(3000) == 137
        assert tfdp_pivot_schedule(50) == 50  # never exceeds n


class TestShortlistSeam:
    def test_gated_row_admits_tfdp_sparse(self) -> None:
        shortlist = _undirected_route_shortlist(
            _sparse_structure(num_nodes=1200), 1200, has_edge_weights=False
        )
        assert "sparse_infrastructure" in shortlist.classes
        assert "tfdp_sparse" in shortlist.candidates

    def test_hub_row_stays_closed(self) -> None:
        shortlist = _undirected_route_shortlist(
            _sparse_structure(num_nodes=1200, hub_edge_fraction=0.6),
            1200,
            has_edge_weights=False,
        )
        assert "tfdp_sparse" not in shortlist.candidates

    def test_above_gate_nodes_shortlist_stays_empty(self) -> None:
        # Band rows (> geodesic_gate_nodes) are owned by the band contest,
        # never by the shortlist -- the early return must stay intact.
        shortlist = _undirected_route_shortlist(
            _sparse_structure(num_nodes=1700), 1700, has_edge_weights=False
        )
        assert shortlist.candidates == ()


class TestBandEligibility:
    def _band_problem(self, num_nodes: int, **structure_overrides: object) -> LayoutProblem:
        return LayoutProblem(
            edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            num_nodes=num_nodes,
            seed=42,
            # The classify-side GraphStructure rides in the state-side slot
            # exactly as the production seam stores it (duck-typed fields).
            structure=cast(Any, _sparse_structure(num_nodes=num_nodes, **structure_overrides)),
        )

    def test_band_row_is_eligible(self) -> None:
        assert sparse_band_contest_eligible(self._band_problem(1612))

    def test_below_and_above_band_are_closed(self) -> None:
        assert not sparse_band_contest_eligible(self._band_problem(1400))
        assert not sparse_band_contest_eligible(self._band_problem(3100))

    def test_clustered_row_is_closed(self) -> None:
        problem = dataclasses.replace(self._band_problem(1612), clusters={"c0": ["0", "1"]})
        assert not sparse_band_contest_eligible(problem)

    def test_band_density_escape(self) -> None:
        assert sparse_band_contest_eligible(self._band_problem(1612, edge_to_node_ratio=2.5))
        assert not sparse_band_contest_eligible(self._band_problem(1612, edge_to_node_ratio=3.5))


class TestBandContest:
    def _contest_fixture(
        self, monkeypatch: pytest.MonkeyPatch, winner_marker: float
    ) -> tuple[torch.Tensor, LayoutProblem, LayoutConfig]:
        """Wire a three-node band contest with a controllable referee.

        Parameters
        ----------
        monkeypatch : pytest.MonkeyPatch
            Active monkeypatch fixture.
        winner_marker : float
            ``pos[0, 0]`` value the fake referee scores as the winner.

        Returns
        -------
        tuple[torch.Tensor, LayoutProblem, LayoutConfig]
            Incumbent positions, problem, and config for the contest call.
        """
        sparse_module = importlib.import_module(
            "dagua.layout.ops.pipelines.native_sparse_infrastructure"
        )
        native_undirected = importlib.import_module("dagua.layout.ops.pipelines.native_undirected")
        incumbent_pos = torch.tensor([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]], dtype=torch.float32)
        challenger_pos = torch.tensor([[1.0, 0.0], [11.0, 0.0], [21.0, 0.0]], dtype=torch.float32)
        problem = LayoutProblem(
            edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            num_nodes=3,
            node_sizes=torch.full((3, 2), 1.0),
            seed=42,
        )
        config = LayoutConfig()

        def score_payload(
            pos: torch.Tensor,
            problem: LayoutProblem,
            cluster_ids: torch.Tensor | None,
            aesthetic_profile: object | None = None,
            all_pairs_dist: object | None = None,
        ) -> tuple[float, _ClusterScoreTelemetry]:
            del problem, cluster_ids, aesthetic_profile, all_pairs_dist
            marker = float(pos[0, 0].item())
            v3_score = 1.0 if marker == winner_marker else 0.0
            return v3_score, _ClusterScoreTelemetry(
                extended_score=v3_score,
                old_score=v3_score,
                metrics={},
                v3_referee_eligibility_key=(1, -0.0),
                v3_tiered=v3_score,
            )

        monkeypatch.setattr(
            sparse_module,
            "tfdp_sparse_positions",
            lambda _problem, *, gamma, seed=None, node_sep=0.0: challenger_pos.clone(),
        )
        monkeypatch.setattr(
            native_undirected, "_large_prism_shortlist_candidate", lambda *_args: None
        )
        monkeypatch.setattr(native_undirected, "_repair_flung_isolates", lambda pos, *_args: pos)
        monkeypatch.setattr(
            native_undirected, "_candidate_is_degenerate", lambda *_args: (False, "ok")
        )
        monkeypatch.setattr(native_undirected, "_project_candidate_prism", lambda *_args: None)
        monkeypatch.setattr(native_undirected, "_log_marketplace_telemetry", lambda **_kwargs: None)
        monkeypatch.setattr(
            native_undirected, "_admit_v3_referee_score", lambda *_args, **_kwargs: True
        )
        monkeypatch.setattr(native_undirected, "_score_undirected_candidate_payload", score_payload)
        monkeypatch.setattr(
            native_undirected,
            "_never_nan_winner",
            lambda winner, *_args: winner,
        )
        return incumbent_pos, problem, config

    def test_challenger_wins_when_referee_says_so(self, monkeypatch: pytest.MonkeyPatch) -> None:
        incumbent_pos, problem, config = self._contest_fixture(monkeypatch, 1.0)
        winner = sparse_band_mini_contest(incumbent_pos, problem, config)
        assert float(winner[0, 0].item()) == 1.0

    def test_incumbent_win_returns_exact_tensor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Byte-inertness regression: a no-win contest must return the very
        # incumbent tensor object, not a copy or a re-projected variant.
        incumbent_pos, problem, config = self._contest_fixture(monkeypatch, 0.0)
        winner = sparse_band_mini_contest(incumbent_pos, problem, config)
        assert winner is incumbent_pos

    def test_all_challengers_failing_returns_incumbent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        incumbent_pos, problem, config = self._contest_fixture(monkeypatch, 1.0)
        sparse_module = importlib.import_module(
            "dagua.layout.ops.pipelines.native_sparse_infrastructure"
        )

        def raise_tfdp(*_args: object, **_kwargs: object) -> torch.Tensor:
            raise RuntimeError("challenger down")

        monkeypatch.setattr(sparse_module, "tfdp_sparse_positions", raise_tfdp)
        winner = sparse_band_mini_contest(incumbent_pos, problem, config)
        assert winner is incumbent_pos

    def test_admission_ignores_exhausted_wall_reserve(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Review F2 regression: arm admission is DWU-ledger-only. An expired
        # wall deadline (load-dependent state) must not change the admitted
        # challenger set -- the winning challenger still competes and wins.
        from dagua.layout.ops.pipelines.native_budget import WALL_DEADLINE_ATTR

        incumbent_pos, problem, config = self._contest_fixture(monkeypatch, 1.0)
        setattr(config, WALL_DEADLINE_ATTR, time.perf_counter() - 1000.0)
        winner = sparse_band_mini_contest(incumbent_pos, problem, config)
        assert float(winner[0, 0].item()) == 1.0

    def test_exhausted_ledger_vetoes_challengers_deterministically(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The ledger, not elapsed time, is the sole admission authority: a
        # spent ledger deterministically vetoes every arm, incumbent holds.
        from dagua.layout.ops.pipelines.native_budget import LEDGER_ATTR, install_budget_ledger

        incumbent_pos, problem, config = self._contest_fixture(monkeypatch, 1.0)
        install_budget_ledger(config, timeout_s=1.0)
        ledger = getattr(config, LEDGER_ATTR)
        ledger.spent_dwu = ledger.capacity_dwu() + 1.0
        winner = sparse_band_mini_contest(incumbent_pos, problem, config)
        assert winner is incumbent_pos

    def test_referee_tail_reserved_before_optional_generation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Review F3 regression (budget half): the honest-referee tail for the
        # guaranteed seats is reserved BEFORE any optional challenger
        # generation is admitted, and released when scoring starts.
        from dagua.layout.ops.pipelines.native_budget import LEDGER_ATTR, install_budget_ledger

        incumbent_pos, problem, config = self._contest_fixture(monkeypatch, 1.0)
        install_budget_ledger(config, timeout_s=10_000.0)
        sparse_band_mini_contest(incumbent_pos, problem, config)
        events = [
            (record["event"], record["reason"]) for record in getattr(config, LEDGER_ATTR).event_log
        ]
        reserve_index = events.index(("reserve_tail", "sparse_band_referee_tail"))
        first_arm_admit_index = next(
            index
            for index, (event, reason) in enumerate(events)
            if event == "admit" and str(reason).startswith("sparse_band_") and "apsp" not in reason
        )
        assert reserve_index < first_arm_admit_index
        assert ("release_tail", "sparse_band_referee_tail_entered_scoring") in events

    def test_raw_tfdp_quota_seat_is_mandatory_under_ledger_veto(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Review F3 regression: when the proxy under-ranks the raw t-FDP
        # drawing (the measured bcspwr07 misranking) AND the ledger vetoes
        # every non-mandatory score, the raw representative still reaches the
        # honest referee -- its quota seat is mandatory, not best-effort.
        incumbent_pos, problem, config = self._contest_fixture(monkeypatch, 1.0)
        native_undirected = importlib.import_module("dagua.layout.ops.pipelines.native_undirected")
        prism_pos = torch.tensor([[5.0, 0.0], [15.0, 0.0], [25.0, 0.0]], dtype=torch.float32)
        monkeypatch.setattr(
            native_undirected,
            "_project_candidate_prism",
            lambda _pos, _problem: prism_pos.clone(),
        )

        def proxy_by_marker(
            pos: torch.Tensor,
            _problem: LayoutProblem,
            _cluster_ids: object,
            _all_pairs_dist: object,
        ) -> float:
            marker = float(pos[0, 0].item())
            if marker == 5.0:  # prism variants: proxy-overranked
                return 90.0
            if marker == 1.0:  # raw t-FDP: proxy-underranked
                return 10.0
            return 98.0  # incumbent

        monkeypatch.setattr(native_undirected, "_proxy_undirected_candidate", proxy_by_marker)
        monkeypatch.setattr(
            native_undirected,
            "_admit_v3_referee_score",
            lambda _problem, _config, *, mandatory_floor: mandatory_floor,
        )
        winner = sparse_band_mini_contest(incumbent_pos, problem, config)
        # The referee scores the raw drawing (marker 1.0) as the sole winner;
        # it can only win if its quota seat survived the ledger veto.
        assert float(winner[0, 0].item()) == 1.0


class TestTfdpWiring:
    def test_real_tfdp_candidate_is_finite(self) -> None:
        """The arm calls the real reimpl pipeline and yields finite [N, 2]."""
        num_nodes = 250
        problem = LayoutProblem(
            edge_index=_chained_ring_edges(num_nodes),
            num_nodes=num_nodes,
            seed=7,
        )
        pos = tfdp_sparse_positions(problem, gamma=2.0)
        assert pos.shape == (num_nodes, 2)
        assert bool(torch.isfinite(pos).all().item())
        # Determinism: the same problem seed reproduces the same layout.
        assert torch.equal(pos, tfdp_sparse_positions(problem, gamma=2.0))


class TestScaleToNodeUnits:
    def test_median_edge_matches_box_plus_sep_target(self) -> None:
        from dagua.layout.ops.pipelines.native_sparse_infrastructure import (
            _scale_to_node_units,
        )

        problem = LayoutProblem(
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
            num_nodes=4,
            node_sizes=torch.full((4, 2), 30.0),
            seed=1,
        )
        # Reference-unit-scale drawing: median edge length 0.01.
        pos = torch.tensor([[0.0, 0.0], [0.01, 0.0], [0.02, 0.0], [0.03, 0.0]], dtype=torch.float32)
        scaled = _scale_to_node_units(pos, problem, node_sep=36.0)
        src, dst = problem.edge_index
        median_edge = float((scaled[src] - scaled[dst]).norm(dim=1).median().item())
        assert problem.node_sizes is not None
        box_diag = float(problem.node_sizes.norm(dim=1).median().item())
        assert median_edge == pytest.approx(box_diag + 36.0, rel=1e-5)
        # Similarity transform only: relative geometry is preserved.
        ratio = (scaled[3] - scaled[0]).norm() / (scaled[1] - scaled[0]).norm()
        assert float(ratio.item()) == pytest.approx(3.0, rel=1e-5)


class TestDegree2Fraction:
    def test_path_graph_measures_interior_fraction(self) -> None:
        num_nodes = 50
        edge_index = torch.tensor(
            [list(range(num_nodes - 1)), list(range(1, num_nodes))], dtype=torch.long
        )
        structure = classify_graph(edge_index, num_nodes)
        assert structure.degree2_fraction == pytest.approx((num_nodes - 2) / num_nodes)

    def test_star_graph_measures_zero(self) -> None:
        structure = classify_graph(_star_edges(40), 40)
        assert structure.degree2_fraction == 0.0
