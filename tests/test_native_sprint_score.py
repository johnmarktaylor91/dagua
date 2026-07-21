"""Focused tests for the R8 Event-A saved-position scorer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest
import torch

from dagua.eval.graphs import TestGraph
from dagua.graph import DaguaGraph
from scripts import native_sprint_score as scorer


def _make_noncluster_graph(name: str) -> TestGraph:
    """Create a minimal non-cluster test graph.

    Parameters
    ----------
    name : str
        Test graph name.

    Returns
    -------
    TestGraph
        Graph metadata wrapper.
    """
    graph = DaguaGraph()
    graph.add_node("a")
    graph.add_node("b")
    graph.add_edge("a", "b")
    graph.compute_node_sizes()
    return TestGraph(name=name, graph=graph, tags={"directed"})


def _position_file(tmp_path: Path, graph: TestGraph, filename: str = "pos.pt") -> Path:
    """Write a valid position tensor for a test graph.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory.
    graph : TestGraph
        Graph whose node count determines tensor shape.
    filename : str, optional
        Output file name.

    Returns
    -------
    Path
        Position tensor path.
    """
    path = tmp_path / filename
    torch.save(torch.zeros((graph.graph.num_nodes, 2), dtype=torch.float32), path)
    return path


def _cluster_metrics(score: float = 10.0) -> dict[str, Any]:
    """Return a complete cluster metric payload for monkeypatched scoring.

    Parameters
    ----------
    score : float, optional
        Composite score value carried in the payload.

    Returns
    -------
    dict[str, Any]
        Metrics containing all R8 cluster keys.
    """
    metrics: dict[str, Any] = {"score": score}
    metrics.update({key: 1.0 for key in scorer.CLUSTER_SCORE_KEYS})
    return metrics


def _score_from_payload(metrics: Mapping[str, Any], _is_directed: bool) -> float:
    """Return the test score embedded in a metrics payload.

    Parameters
    ----------
    metrics : Mapping[str, Any]
        Metrics payload.
    _is_directed : bool
        Semantic direction flag, unused by this test scorer.

    Returns
    -------
    float
        Embedded score.
    """
    return float(metrics["score"])


def test_frozen_108_plus_r8_is_121_including_1k(monkeypatch: pytest.MonkeyPatch) -> None:
    """The R8 extension appends the exact 13 graph names and reconstructs the 1k graph."""
    old_names = [f"old_{index:03d}" for index in range(108)]
    monkeypatch.setattr(scorer, "corpus_names", lambda _path: old_names)

    frozen, extended = scorer.selected_names(Path("unused"), include_r8=True)
    graphs = scorer.build_graph_map(scorer.R8_GRAPH_NAMES)

    assert frozen == old_names
    assert len(extended) == 121
    assert tuple(extended[-13:]) == scorer.R8_GRAPH_NAMES
    assert "r8_nested_scale_1k_budget" in graphs
    assert graphs["r8_nested_scale_1k_budget"].graph.num_nodes == 1000
    assert all(graphs[name].graph.clusters for name in scorer.R8_GRAPH_NAMES)


def test_stale_cached_clustered_row_ignored_and_six_keys_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Clustered field rows bypass legacy cache and fresh scoring retains all six keys."""
    graph = scorer.build_graph_map(["r8_nested_singleton_deep"])["r8_nested_singleton_deep"]
    pos_path = _position_file(tmp_path, graph)
    cache_path = tmp_path / "legacy.json"
    cache_path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "graph": graph.name,
                        "engine": "dagre",
                        "best_path": str(pos_path),
                        "best_composite": -999.0,
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(scorer, "LEGACY_FIELD_CACHE_SHA256", scorer.sha256_file(cache_path))
    monkeypatch.setattr(scorer, "LEGACY_FIELD_CACHE_PAIR_COUNT", 1)
    monkeypatch.setattr(scorer, "full", lambda *args, **kwargs: _cluster_metrics(12.0))
    monkeypatch.setattr(scorer, "composite_auto", _score_from_payload)

    reused, reused_keys, skipped_keys = scorer.load_legacy_field_rows(
        cache_path, [(graph.name, "dagre", [str(pos_path.resolve())])], {graph.name: graph}, "sig"
    )
    row = scorer.score_position(graph, str(pos_path), "dagre", "sig")

    assert reused == []
    assert reused_keys == set()
    assert skipped_keys == set()
    assert row["score_origin"] == "fresh_positions"
    assert all(key in row["metrics"] for key in scorer.CLUSTER_SCORE_KEYS)


def test_noncluster_legacy_accepted_only_under_pinned_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-cluster legacy row is reused only when the pinned cache guard passes."""
    graph = _make_noncluster_graph("plain")
    pos_path = _position_file(tmp_path, graph)
    cache_path = tmp_path / "legacy.json"
    cache_path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "graph": "plain",
                        "engine": "dagre",
                        "best_path": str(pos_path),
                        "best_composite": 7.5,
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(scorer, "LEGACY_FIELD_CACHE_PAIR_COUNT", 1)
    monkeypatch.setattr(scorer, "LEGACY_FIELD_CACHE_SHA256", "bad")

    with pytest.raises(RuntimeError, match="SHA mismatch"):
        scorer.load_legacy_field_rows(
            cache_path, [("plain", "dagre", [str(pos_path.resolve())])], {"plain": graph}, "sig"
        )

    monkeypatch.setattr(scorer, "LEGACY_FIELD_CACHE_SHA256", scorer.sha256_file(cache_path))
    reused, reused_keys, skipped_keys = scorer.load_legacy_field_rows(
        cache_path, [("plain", "dagre", [str(pos_path.resolve())])], {"plain": graph}, "sig"
    )

    assert reused_keys == {("plain", "dagre")}
    assert skipped_keys == set()
    assert reused[0]["score_origin"] == "legacy_cache_g2_noncluster"
    assert reused[0]["old_composite"] == reused[0]["extended_composite"] == 7.5


def test_unscoreable_legacy_field_row_skipped_and_excluded_from_fresh_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A legacy field error row is absent as a competitor and not fresh-scored."""
    graph = _make_noncluster_graph("plain")
    pos_path = _position_file(tmp_path, graph)
    cache_path = tmp_path / "legacy.json"
    cache_path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "graph": "plain",
                        "engine": "grip_reimpl",
                        "best_path": None,
                        "best_composite": None,
                        "error": "torch.histc: range of [-nan,-nan] is not finite",
                    }
                ]
            }
        )
    )
    field_tasks = [("plain", "grip_reimpl", [str(pos_path.resolve())])]
    monkeypatch.setattr(scorer, "LEGACY_FIELD_CACHE_SHA256", scorer.sha256_file(cache_path))
    monkeypatch.setattr(scorer, "LEGACY_FIELD_CACHE_PAIR_COUNT", 1)

    reused, reused_keys, skipped_keys = scorer.load_legacy_field_rows(
        cache_path, field_tasks, {"plain": graph}, "sig"
    )
    excluded_keys = reused_keys | skipped_keys
    fresh_field_tasks = [task for task in field_tasks if (task[0], task[1]) not in excluded_keys]

    assert reused == []
    assert reused_keys == set()
    assert skipped_keys == {("plain", "grip_reimpl")}
    assert fresh_field_tasks == []


def test_legacy_field_row_with_bad_nonnull_path_still_fails_integrity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A legacy field row with a non-null missing path remains an integrity error."""
    graph = _make_noncluster_graph("plain")
    pos_path = _position_file(tmp_path, graph)
    missing_path = tmp_path / "missing.pt"
    cache_path = tmp_path / "legacy.json"
    cache_path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "graph": "plain",
                        "engine": "dagre",
                        "best_path": str(missing_path),
                        "best_composite": None,
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(scorer, "LEGACY_FIELD_CACHE_SHA256", scorer.sha256_file(cache_path))
    monkeypatch.setattr(scorer, "LEGACY_FIELD_CACHE_PAIR_COUNT", 1)

    with pytest.raises(RuntimeError, match="best_path is not scoreable"):
        scorer.load_legacy_field_rows(
            cache_path, [("plain", "dagre", [str(pos_path.resolve())])], {"plain": graph}, "sig"
        )


def test_scoring_signature_mismatch_causes_rescore(tmp_path: Path) -> None:
    """A raw cache with any header mismatch is not reused."""
    cache_path = tmp_path / "R8_EVENTA_RAW_SCORES_V1.json"
    cache_path.write_text(json.dumps({"header": {"scoring_signature": "old"}, "rows": []}))

    assert scorer.read_raw_cache(cache_path, {"scoring_signature": "new"}) is None


def test_old_and_extended_field_best_recomputed_independently_can_differ(
    tmp_path: Path,
) -> None:
    """Old and extended tables maximize separate columns and can pick different engines."""
    graph = _make_noncluster_graph("plain")
    rows = [
        {
            "graph": "plain",
            "engine": "dagua",
            "position_path": "native.pt",
            "old_composite": 5.0,
            "extended_composite": 5.0,
            "metrics": {},
        },
        {
            "graph": "plain",
            "engine": "dagre",
            "position_path": "dagre.pt",
            "old_composite": 10.0,
            "extended_composite": 1.0,
            "metrics": {},
        },
        {
            "graph": "plain",
            "engine": "elk_layered",
            "position_path": "elk.pt",
            "old_composite": 1.0,
            "extended_composite": 10.0,
            "metrics": {},
        },
    ]

    old_table = scorer.build_table(
        rows, {"plain": graph}, ["plain"], "old", "sig", tmp_path / "raw.json"
    )
    extended_table = scorer.build_table(
        rows, {"plain": graph}, ["plain"], "extended", "sig", tmp_path / "raw.json"
    )

    assert old_table["per_graph"][0]["field_best_engine"] == "dagre"
    assert extended_table["per_graph"][0]["field_best_engine"] == "elk_layered"


def test_every_clustered_raw_row_must_be_fresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raw integrity rejects any clustered row that did not come from saved positions."""
    graph = scorer.build_graph_map(["r8_nested_singleton_deep"])["r8_nested_singleton_deep"]
    metrics = _cluster_metrics(3.0)
    monkeypatch.setattr(scorer, "composite_auto", _score_from_payload)
    rows = [
        {
            "graph": graph.name,
            "engine": "dagua",
            "position_path": "native.pt",
            "metrics": metrics,
            "old_composite": 3.0,
            "extended_composite": 3.0,
            "scoring_signature": "sig",
            "score_origin": "fresh_positions",
        },
        {
            "graph": graph.name,
            "engine": "dagre",
            "position_path": "field.pt",
            "metrics": metrics,
            "old_composite": 3.0,
            "extended_composite": 3.0,
            "scoring_signature": "sig",
            "score_origin": "fresh_positions",
        },
    ]
    scorer.validate_raw_integrity(rows, {graph.name: graph}, [graph.name], "sig")

    bad_rows = [dict(row) for row in rows]
    bad_rows[1]["score_origin"] = "legacy_cache_g2_noncluster"
    with pytest.raises(RuntimeError, match="clustered raw row is not fresh"):
        scorer.validate_raw_integrity(bad_rows, {graph.name: graph}, [graph.name], "sig")


def test_score_position_trips_on_sizeless_graph(tmp_path: Path) -> None:
    """Scoring a graph built without ``compute_node_sizes()`` hard-fails.

    The 2026-07-21 S1/M2 tally incident: bare ``get_test_graphs()`` graphs
    carried ``node_sizes=None``, silently nulling ``node_occlusion_score`` and
    renormalizing the composite ~0.5-3.5 pts low. ``score_position`` must
    refuse to score a nulled-occlusion composite; graphs with measured sizes
    keep working.
    """
    graph = DaguaGraph()
    graph.add_node("a")
    graph.add_node("b")
    graph.add_edge("a", "b")
    test_graph = TestGraph(name="sizeless", graph=graph, tags={"directed"})
    pos_path = _position_file(tmp_path, test_graph)
    assert graph.node_sizes is None

    with pytest.raises(RuntimeError, match="node_sizes is None"):
        scorer.score_position(test_graph, str(pos_path), "dagua", "sig")

    graph.compute_node_sizes()
    row = scorer.score_position(test_graph, str(pos_path), "dagua", "sig")

    assert row["score_origin"] == "fresh_positions"
    assert row["metrics"]["node_occlusion_score"] is not None
