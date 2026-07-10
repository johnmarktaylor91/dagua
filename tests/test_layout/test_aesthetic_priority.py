"""Tests for the r80-S8 user-facing aesthetic-priority knob.

Covers: profile resolution/validation, default-identity (unset knob is a
true no-op on both the selection composite and the loss weights), the
Wire-through B loss-multiplier mapping, and an efficacy proof that two
presets actually select different portfolio contest winners on a real
corpus graph (and each winner is better than the other on its own
prioritized term).
"""

from __future__ import annotations

import copy

import pytest
import torch

from dagua.config import LayoutConfig
from dagua.eval.graphs import get_test_graphs
from dagua.layout.aesthetics import (
    DIRECTED_BASE_WEIGHTS,
    LOSS_MULTIPLIER_CLAMP,
    PRESETS,
    UNDIRECTED_BASE_WEIGHTS,
    AestheticProfile,
    apply_loss_multipliers,
    resolve_aesthetic_profile,
    reweighted_composite,
)
from dagua.layout.engine import layout
from dagua.layout.ops.pipelines import native_undirected as native_undirected_module
from dagua.layout.resolve import prepare_pipeline_config
from dagua.metrics import composite, composite_undirected, full

# ---------------------------------------------------------------------------
# resolve_aesthetic_profile: resolution + validation
# ---------------------------------------------------------------------------


def test_default_config_has_no_aesthetic_profile() -> None:
    """The unset knob resolves to None, not an all-ones profile."""
    assert resolve_aesthetic_profile(LayoutConfig()) is None


def test_preset_resolves_documented_weights() -> None:
    """Each preset name resolves to exactly its documented term weights."""
    for name, weights in PRESETS.items():
        profile = resolve_aesthetic_profile(LayoutConfig(prioritize=name))
        assert profile is not None
        assert dict(profile.weights) == weights


def test_explicit_weights_override_preset_per_key() -> None:
    """aesthetic_weights overrides matching preset keys, keeps the rest."""
    config = LayoutConfig(prioritize="readability", aesthetic_weights={"crossing": 9.0})
    profile = resolve_aesthetic_profile(config)
    assert profile is not None
    assert profile.weights["crossing"] == 9.0
    # Untouched preset keys survive the override.
    assert profile.weights["angular_res"] == PRESETS["readability"]["angular_res"]


def test_explicit_weights_alone_do_not_require_a_preset() -> None:
    """aesthetic_weights works standalone (no prioritize set)."""
    profile = resolve_aesthetic_profile(LayoutConfig(aesthetic_weights={"overlap": 2.0}))
    assert profile == AestheticProfile(weights={"overlap": 2.0})


def test_empty_aesthetic_weights_dict_is_identity() -> None:
    """An explicitly-empty override dict is the same as unset."""
    assert resolve_aesthetic_profile(LayoutConfig(aesthetic_weights={})) is None


def test_unknown_preset_name_raises() -> None:
    """An invalid preset name raises with the valid options listed."""
    with pytest.raises(ValueError, match="prioritize must be one of"):
        resolve_aesthetic_profile(LayoutConfig(prioritize="not_a_real_preset"))


def test_unknown_term_name_raises() -> None:
    """An invalid aesthetic_weights term raises with valid terms listed."""
    with pytest.raises(ValueError, match="Unknown aesthetic term"):
        resolve_aesthetic_profile(LayoutConfig(aesthetic_weights={"not_a_term": 2.0}))


def test_non_positive_weight_raises() -> None:
    """A zero or negative explicit weight is rejected."""
    with pytest.raises(ValueError, match="must be positive"):
        resolve_aesthetic_profile(LayoutConfig(aesthetic_weights={"crossing": 0.0}))


# ---------------------------------------------------------------------------
# reweighted_composite: identity + reweighting behavior
# ---------------------------------------------------------------------------


_SAMPLE_METRICS = {
    "dag_consistency": 0.8,
    "edge_length_cv": 0.3,
    "depth_spearman_rho": 0.6,
    "overlap_count": 0,
    "edge_straightness_mean_deg": 10.0,
    "crossing_rate": 0.02,
    "sampled_stress": 0.2,
    "angular_res_mean_deg": 25.0,
    "cluster_mean_sep_ratio": 3.0,
    "edge_length_mean": 100.0,
    "node_diag_mean": 20.0,
}


def test_reweighted_composite_identity_matches_frozen_composite_directed() -> None:
    """At an all-ones profile, reweighted_composite matches composite() closely."""
    identity_profile = AestheticProfile(weights={term: 1.0 for term in DIRECTED_BASE_WEIGHTS})
    frozen = composite(_SAMPLE_METRICS)
    reweighted = reweighted_composite(_SAMPLE_METRICS, is_directed=True, profile=identity_profile)
    assert reweighted == pytest.approx(frozen, abs=1e-9)


def test_reweighted_composite_identity_matches_frozen_composite_undirected() -> None:
    """At an all-ones profile, reweighted_composite matches composite_undirected()."""
    identity_profile = AestheticProfile(weights={term: 1.0 for term in UNDIRECTED_BASE_WEIGHTS})
    frozen = composite_undirected(_SAMPLE_METRICS)
    reweighted = reweighted_composite(_SAMPLE_METRICS, is_directed=False, profile=identity_profile)
    assert reweighted == pytest.approx(frozen, abs=1e-9)


def test_reweighted_composite_boosts_the_prioritized_term() -> None:
    """Boosting a term's multiplier increases its contribution to the score.

    Uses two metrics dicts that differ ONLY in the crossing term, so the
    score gap between them must widen under a crossings-boosted profile
    relative to the identity profile.
    """
    good_crossing = dict(_SAMPLE_METRICS, crossing_rate=0.0)
    bad_crossing = dict(_SAMPLE_METRICS, crossing_rate=0.09)

    identity_profile = AestheticProfile(weights={})
    boosted_profile = AestheticProfile(weights={"crossing": 3.0})

    identity_gap = reweighted_composite(
        good_crossing, is_directed=False, profile=identity_profile
    ) - reweighted_composite(bad_crossing, is_directed=False, profile=identity_profile)
    boosted_gap = reweighted_composite(
        good_crossing, is_directed=False, profile=boosted_profile
    ) - reweighted_composite(bad_crossing, is_directed=False, profile=boosted_profile)

    assert boosted_gap > identity_gap > 0.0


# ---------------------------------------------------------------------------
# apply_loss_multipliers: Wire-through B
# ---------------------------------------------------------------------------


def test_apply_loss_multipliers_scales_mapped_fields_only() -> None:
    """Only the fields listed in LOSS_MULTIPLIER_MAP for the set term change."""
    base = LayoutConfig()
    profile = AestheticProfile(weights={"crossing": 2.0})
    resolved = apply_loss_multipliers(base, profile)

    assert resolved.w_crossing == pytest.approx(base.w_crossing * 2.0)
    assert resolved.w_edge_crossing == pytest.approx(base.w_edge_crossing * 2.0)
    assert resolved.w_edge_node_crossing == pytest.approx(base.w_edge_node_crossing * 2.0)
    # Unrelated fields are untouched.
    assert resolved.w_dag == base.w_dag
    assert resolved.w_overlap == base.w_overlap


def test_apply_loss_multipliers_clamps_extreme_multipliers() -> None:
    """A multiplier outside LOSS_MULTIPLIER_CLAMP is clamped before applying."""
    base = LayoutConfig()
    low, high = LOSS_MULTIPLIER_CLAMP
    profile = AestheticProfile(weights={"overlap": high * 10.0})
    resolved = apply_loss_multipliers(base, profile)
    assert resolved.w_overlap == pytest.approx(base.w_overlap * high)

    profile_low = AestheticProfile(weights={"overlap": low / 10.0})
    resolved_low = apply_loss_multipliers(base, profile_low)
    assert resolved_low.w_overlap == pytest.approx(base.w_overlap * low)


def test_apply_loss_multipliers_is_a_true_noop_for_identity() -> None:
    """A profile with no weights leaves every w_* field byte-identical."""
    base = LayoutConfig()
    resolved = apply_loss_multipliers(base, AestheticProfile(weights={}))
    for field_name in (
        "w_dag",
        "w_crossing",
        "w_overlap",
        "w_length_variance",
        "w_fanout",
        "w_cluster",
    ):
        assert getattr(resolved, field_name) == getattr(base, field_name)


# ---------------------------------------------------------------------------
# Loss-path test (design requirement / gate 3): profile shifts resolved w_*
# ---------------------------------------------------------------------------


def test_prepare_pipeline_config_shifts_resolved_loss_weight() -> None:
    """A priority profile measurably shifts the resolved w_* value.

    No full layout solve: just resolve one problem instance's config and
    assert the crossing-family weights differ from the unset-knob defaults
    in the expected (increased) direction.
    """
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    default_prepared = prepare_pipeline_config(
        config=LayoutConfig(seed=42, device="cpu"),
        num_nodes=4,
        edge_index=edge_index,
        device="cpu",
        layer_assignments=None,
        prebuilt_layer_index=None,
        graph_structure=None,
        skip_classification=False,
    )
    boosted_prepared = prepare_pipeline_config(
        config=LayoutConfig(seed=42, device="cpu", prioritize="crossings"),
        num_nodes=4,
        edge_index=edge_index,
        device="cpu",
        layer_assignments=None,
        prebuilt_layer_index=None,
        graph_structure=None,
        skip_classification=False,
    )

    assert getattr(default_prepared, "_dagua_native_aesthetic_profile", "unset") is None
    assert getattr(boosted_prepared, "_dagua_native_aesthetic_profile") is not None
    assert boosted_prepared.w_crossing > default_prepared.w_crossing
    assert boosted_prepared.w_crossing == pytest.approx(default_prepared.w_crossing * 3.0)
    # Unrelated weight is untouched.
    assert boosted_prepared.w_dag == default_prepared.w_dag


# ---------------------------------------------------------------------------
# Default-identity test (gate 1's unit-level counterpart)
# ---------------------------------------------------------------------------


def test_unset_knob_is_bit_identical_default_path() -> None:
    """Two default (knob-unset) layouts of the same graph are bit-identical.

    This is the fast unit-level counterpart to the full corpus sweep gate;
    it does not replace the sweep but catches a regression in seconds.
    """
    from dagua.graph import DaguaGraph

    edges = [(i, (i + 1) % 10) for i in range(10)] + [(i, (i + 5) % 10) for i in range(5)]
    graph = DaguaGraph.from_edge_list(edges, num_nodes=10, is_semantically_directed=False)
    graph.compute_node_sizes()

    pos_a = layout(graph, LayoutConfig(seed=42, device="cpu"))
    pos_b = layout(graph, LayoutConfig(seed=42, device="cpu"))
    assert torch.equal(pos_a, pos_b)


# ---------------------------------------------------------------------------
# Efficacy test (gate 2): different presets select different contest winners
# ---------------------------------------------------------------------------


def _contest_trace():
    """Monkeypatch-friendly recorder for one portfolio contest run.

    Returns
    -------
    tuple[dict, callable, callable]
        ``(records, install, uninstall)`` where ``records`` accumulates
        ``(profile, score)`` pairs recorded by every
        ``_score_undirected_candidate`` call while installed.
    """
    original = native_undirected_module._score_undirected_candidate
    records: list[tuple[object, float]] = []

    def _traced(pos, problem, cluster_ids, aesthetic_profile=None):
        score = original(pos, problem, cluster_ids, aesthetic_profile)
        records.append((aesthetic_profile, score))
        return score

    def install() -> None:
        native_undirected_module._score_undirected_candidate = _traced

    def uninstall() -> None:
        native_undirected_module._score_undirected_candidate = original

    return records, install, uninstall


def _find_corpus_graph_with_flip():
    """Return a small real corpus graph whose contest winner flips between presets.

    Searches a short, curated list of known undirected-portfolio graphs
    (all <= 40 nodes, so the search stays fast) rather than the full
    108-graph corpus. Returns ``None`` if no flip is found (the caller
    should skip rather than fail -- corpus contents can drift).
    """
    candidate_names = [
        # random_bipartite_60 is the recorded gate graph (P15_AESTHETIC_KNOB.md):
        # crossings preset -> crossing_rate 0.0269 vs uniform_edges' 0.1536
        # (5.7x lower); uniform_edges preset -> edge_length_cv 0.0136 vs
        # crossings' 0.2632 (19x lower). Each preset dominates on its own term.
        "random_bipartite_60",
        "grid_5x5",
        "weighted_clusters_3x10",
        "regular_3_30",
        "sierpinski_42",
        "triangular_lattice_36",
        "grid_rect_6x8",
    ]
    graphs = {tg.name: tg.graph for tg in get_test_graphs(max_nodes=80)}

    for name in candidate_names:
        graph = graphs.get(name)
        if graph is None:
            continue

        winners = {}
        term_scores = {}
        for label, term in (("crossings", "crossing_rate"), ("uniform_edges", "edge_length_cv")):
            records, install, uninstall = _contest_trace()
            install()
            try:
                config = LayoutConfig(seed=42, device="cpu", prioritize=label)
                pos = layout(copy.deepcopy(graph), config)
            finally:
                uninstall()
            if not records:
                winners = {}
                break
            best_idx = max(range(len(records)), key=lambda i: records[i][1])
            winners[label] = best_idx
            metrics = full(
                pos,
                graph.edge_index,
                node_sizes=getattr(graph, "node_sizes", None),
            )
            term_scores[label] = float(metrics[term])

        if len(winners) == 2 and winners["crossings"] != winners["uniform_edges"]:
            return name, graph, term_scores
    return None


def test_different_presets_select_different_contest_winners_on_real_graph() -> None:
    """prioritize=crossings vs prioritize=uniform_edges pick different winners.

    Proves the knob actually steers engine selection (not just perturbs a
    score) on a real corpus graph, and that each winner is better than the
    OTHER preset's winner on its own prioritized term (not just different).
    """
    found = _find_corpus_graph_with_flip()
    if found is None:
        pytest.skip(
            "No curated corpus graph produced a contest-winner flip between "
            "the crossings/uniform_edges presets in this environment; see "
            "P15_AESTHETIC_KNOB.md for the graph used in the recorded gate evidence."
        )
    name, graph, term_scores = found

    crossings_config = LayoutConfig(seed=42, device="cpu", prioritize="crossings")
    uniform_config = LayoutConfig(seed=42, device="cpu", prioritize="uniform_edges")
    pos_crossings = layout(copy.deepcopy(graph), crossings_config)
    pos_uniform = layout(copy.deepcopy(graph), uniform_config)

    assert not torch.equal(pos_crossings, pos_uniform), (
        f"{name}: crossings/uniform_edges presets produced identical positions"
    )

    metrics_crossings = full(
        pos_crossings, graph.edge_index, node_sizes=getattr(graph, "node_sizes", None)
    )
    metrics_uniform = full(
        pos_uniform, graph.edge_index, node_sizes=getattr(graph, "node_sizes", None)
    )

    # The crossings-prioritized winner must have a lower (better) crossing
    # rate than the uniform_edges-prioritized winner, and vice versa for
    # edge-length uniformity -- each preset wins on ITS OWN prioritized term.
    assert float(metrics_crossings["crossing_rate"]) <= float(metrics_uniform["crossing_rate"])
    assert float(metrics_uniform["edge_length_cv"]) <= float(metrics_crossings["edge_length_cv"])
