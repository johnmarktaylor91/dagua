from pathlib import Path

import pytest
import torch

from dagua.eval.benchmark import (
    BenchmarkGraph,
    benchmark_run_status,
    get_rare_suite_graphs,
    get_standard_suite_graphs,
    merge_latest_results,
    run_rare_suite,
    run_standard_suite,
)
from dagua.eval.competitors.dagua_competitor import DaguaCompetitor
from dagua.eval.graphs import TestGraph
from dagua.eval.report import generate_report
from dagua.graph import DaguaGraph


@pytest.mark.smoke
def test_standard_suite_contains_expected_cases():
    suite = get_standard_suite_graphs()
    names = {bg.test_graph.name for bg in suite}

    assert "chain_100" in names
    assert "binary_tree_127" in names
    assert "tl_cnn_small" in names
    assert "tl_resnet_2block" in names
    assert "tl_transformer_1layer" in names
    assert "scale_100k" in names
    assert len(suite) >= 15


@pytest.mark.smoke
def test_rare_suite_sizes_present():
    suite = get_rare_suite_graphs()
    names = {bg.test_graph.name for bg in suite}
    assert {
        "scale_500000",
        "scale_1000000",
        "scale_2000000",
        "scale_5000000",
        "scale_10000000",
        "scale_20000000",
        "scale_50000000",
        "scale_100000000",
        "scale_250000000",
        "scale_500000000",
        "scale_1000000000",
    } <= names


@pytest.mark.smoke
def test_merge_latest_results_and_generate_report(tmp_path):
    output_dir = tmp_path / "eval_output"
    standard_run = output_dir / "benchmark_db" / "standard" / "2026-03-12T00:00:00+00:00"
    rare_run = output_dir / "benchmark_db" / "rare" / "2026-03-12T01:00:00+00:00"
    standard_positions = standard_run / "positions"
    rare_positions = rare_run / "positions"
    standard_positions.mkdir(parents=True, exist_ok=True)
    rare_positions.mkdir(parents=True, exist_ok=True)

    standard_suite = {bg.test_graph.name: bg.test_graph for bg in get_standard_suite_graphs()}
    residual = standard_suite["residual_block"]
    residual.graph.compute_node_sizes()
    pos = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 60.0],
            [0.0, 120.0],
            [0.0, 180.0],
            [0.0, 240.0],
            [-80.0, 240.0],
            [-80.0, 300.0],
            [0.0, 300.0],
            [0.0, 360.0],
            [0.0, 420.0],
        ],
        dtype=torch.float32,
    )
    torch.save(pos, standard_positions / "residual_block__dagua.pt")

    standard_payload = {
        "run_id": "2026-03-12T00:00:00+00:00",
        "suite": "standard",
        "system": {"python": "3.11"},
        "graphs": {
            "residual_block": {
                "n_nodes": residual.graph.num_nodes,
                "n_edges": int(residual.graph.edge_index.shape[1]),
                "structural_category": "residual",
                "description": residual.description,
                "expected_challenges": residual.expected_challenges,
                "tags": sorted(residual.tags),
                "source": residual.source,
                "visualize": True,
                "scale_tier": None,
                "competitors": {
                    "dagua": {
                        "status": "OK",
                        "runtime_seconds": 0.12,
                        "metrics": {
                            "dag_consistency": 1.0,
                            "overall_quality": 90.0,
                            "edge_crossings": 0,
                            "node_overlaps": 0,
                            "edge_length_cv": 0.1,
                        },
                        "composite_score": 91.5,
                        "metrics_computed": ["tier1", "tier2", "tier3"],
                        "metrics_skipped": [],
                        "positions_path": "positions/residual_block__dagua.pt",
                    },
                    "graphviz_dot": {
                        "status": "SKIPPED",
                        "reason": "not installed",
                        "runtime_seconds": None,
                        "metrics": {},
                        "composite_score": None,
                        "metrics_computed": [],
                        "metrics_skipped": ["tier1", "tier2", "tier3"],
                        "positions_path": None,
                    },
                },
            }
        },
    }

    rare_payload = {
        "run_id": "2026-03-12T01:00:00+00:00",
        "suite": "rare",
        "system": {"python": "3.11"},
        "graphs": {
            "scale_500000": {
                "n_nodes": 500_000,
                "n_edges": 750_000,
                "structural_category": "scale",
                "description": "Rare scale graph",
                "expected_challenges": "Scale",
                "tags": ["large-sparse"],
                "source": "synthetic",
                "visualize": False,
                "scale_tier": "rare",
                "competitors": {
                    "dagua": {
                        "status": "OK",
                        "runtime_seconds": 12.0,
                        "metrics": {"dag_consistency": 1.0, "overall_quality": 70.0},
                        "composite_score": 74.0,
                        "metrics_computed": ["tier1"],
                        "metrics_skipped": ["tier2", "tier3"],
                        "positions_path": None,
                    }
                },
            }
        },
    }

    (standard_run / "results.json").write_text(__import__("json").dumps(standard_payload), encoding="utf-8")
    (rare_run / "results.json").write_text(__import__("json").dumps(rare_payload), encoding="utf-8")
    (standard_run.parent / "latest").symlink_to(standard_run.name)
    (rare_run.parent / "latest").symlink_to(rare_run.name)

    combined = merge_latest_results(str(output_dir))
    assert "residual_block" in combined["graphs"]
    assert "scale_500000" in combined["graphs"]

    artifacts = generate_report(output_dir=str(output_dir), combined_results=combined, compile_pdf=False)
    assert Path(artifacts["tex"]).exists()
    assert Path(artifacts["scaling_curve"]).exists()
    assert Path(artifacts["benchmark_deltas_json"]).exists()
    assert Path(artifacts["benchmark_deltas_md"]).exists()
    assert Path(artifacts["layout_similarity_json"]).exists()
    assert Path(artifacts["layout_similarity_md"]).exists()
    assert Path(artifacts["placement_summary_json"]).exists()
    assert Path(artifacts["placement_summary_md"]).exists()
    assert Path(artifacts["placement_dashboard_json"]).exists()
    assert Path(artifacts["placement_dashboard_md"]).exists()
    assert (output_dir / "visuals" / "comparisons" / "residual_block_comparison.png").exists()
    assert (output_dir / "report" / "prose_prompt.md").exists()
    assert (output_dir / "report" / "review_round_1.json").exists()


@pytest.mark.smoke
def test_standard_suite_reuses_cached_non_dagua_results(tmp_path, monkeypatch):
    output_dir = tmp_path / "eval_output"
    latest_run = output_dir / "benchmark_db" / "standard" / "2026-03-12T00:00:00+00:00"
    latest_positions = latest_run / "positions"
    latest_positions.mkdir(parents=True, exist_ok=True)

    graph = DaguaGraph.from_edge_list([("a", "b"), ("b", "c")])
    tg = TestGraph(
        name="tiny_chain",
        graph=graph,
        tags={"linear"},
        description="tiny chain",
        source="synthetic",
        expected_challenges="none",
    )
    suite = [BenchmarkGraph(tg, "linear", "standard", True, "small")]

    pos = torch.tensor([[0.0, 0.0], [0.0, 50.0], [0.0, 100.0]], dtype=torch.float32)
    torch.save(pos, latest_positions / "tiny_chain__graphviz_dot.pt")

    latest_payload = {
        "run_id": "2026-03-12T00:00:00+00:00",
        "suite": "standard",
        "system": {"dagua_git_hash": "old", "graphviz": "dot 1.0"},
        "graphs": {
            "tiny_chain": {
                "n_nodes": 3,
                "n_edges": 2,
                "structural_category": "linear",
                "description": "tiny chain",
                "expected_challenges": "none",
                "tags": ["linear"],
                "source": "synthetic",
                "visualize": True,
                "scale_tier": "small",
                "competitors": {
                    "graphviz_dot": {
                        "status": "OK",
                        "runtime_seconds": 0.01,
                        "metrics": {"overall_quality": 80.0},
                        "composite_score": 80.0,
                        "metrics_computed": ["tier1"],
                        "metrics_skipped": ["tier2", "tier3"],
                        "positions_path": "positions/tiny_chain__graphviz_dot.pt",
                    }
                },
            }
        },
    }
    latest_metadata = {
        "graph_signatures": {"tiny_chain": __import__("hashlib").sha256(__import__("json").dumps(graph.to_json(), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()},
        "competitor_signatures": {
            "graphviz_dot": "graphviz_dot:dot 1.0",
            "dagua": "dagua:new",
        },
    }
    (latest_run / "results.json").write_text(__import__("json").dumps(latest_payload), encoding="utf-8")
    (latest_run / "metadata.json").write_text(__import__("json").dumps(latest_metadata), encoding="utf-8")
    (latest_run.parent / "latest").symlink_to(latest_run.name)

    class FakeCompetitor:
        def __init__(self, name):
            self.name = name
            self.max_nodes = 10

        def available(self):
            return True

        def layout(self, graph, timeout=300.0):
            raise AssertionError(f"{self.name} should not have been rerun")

    class FakeDagua(FakeCompetitor):
        def layout(self, graph, timeout=300.0):
            return type("Result", (), {"pos": pos, "runtime_seconds": 0.02, "error": None})()

    monkeypatch.setattr("dagua.eval.benchmark._suite_graphs", lambda suite_name: suite)
    monkeypatch.setattr("dagua.eval.benchmark._competitor_map", lambda names=None: [FakeDagua("dagua"), FakeCompetitor("graphviz_dot")])
    monkeypatch.setattr("dagua.eval.benchmark._system_metadata", lambda: {"dagua_git_hash": "new", "graphviz": "dot 1.0"})
    monkeypatch.setattr("dagua.eval.benchmark.merge_latest_results", lambda output_dir=None: {"graphs": {}})
    monkeypatch.setattr("dagua.eval.report.generate_report", lambda *args, **kwargs: {})

    payload = run_standard_suite(output_dir=str(output_dir), reuse_cached=True)
    result = payload["graphs"]["tiny_chain"]["competitors"]["graphviz_dot"]
    assert result["status"] == "OK"
    assert result["reused_from"] == "2026-03-12T00:00:00+00:00"
    new_run_dir = output_dir / "benchmark_db" / "standard" / payload["run_id"]
    assert (new_run_dir / "positions" / "tiny_chain__graphviz_dot.pt").exists()


@pytest.mark.smoke
def test_standard_suite_can_force_rerun_specific_competitor(tmp_path, monkeypatch):
    output_dir = tmp_path / "eval_output"
    graph = DaguaGraph.from_edge_list([("a", "b")])
    tg = TestGraph(
        name="tiny_force",
        graph=graph,
        tags={"linear"},
        description="tiny force",
        source="synthetic",
        expected_challenges="none",
    )
    suite = [BenchmarkGraph(tg, "linear", "standard", True, "small")]

    calls = {"dot": 0, "dagua": 0}
    pos = torch.tensor([[0.0, 0.0], [0.0, 50.0]], dtype=torch.float32)

    class FakeCompetitor:
        def __init__(self, name):
            self.name = name
            self.max_nodes = 10

        def available(self):
            return True

        def layout(self, graph, timeout=300.0):
            calls[self.name.split("_")[-1] if self.name != "dagua" else "dagua"] += 1
            return type("Result", (), {"pos": pos, "runtime_seconds": 0.01, "error": None})()

    monkeypatch.setattr("dagua.eval.benchmark._suite_graphs", lambda suite_name: suite)
    monkeypatch.setattr("dagua.eval.benchmark._competitor_map", lambda names=None: [FakeCompetitor("dagua"), FakeCompetitor("graphviz_dot")])
    monkeypatch.setattr("dagua.eval.benchmark._system_metadata", lambda: {"dagua_git_hash": "new", "graphviz": "dot 1.0"})
    monkeypatch.setattr("dagua.eval.benchmark.merge_latest_results", lambda output_dir=None: {"graphs": {}})
    monkeypatch.setattr("dagua.eval.report.generate_report", lambda *args, **kwargs: {})

    payload = run_standard_suite(
        output_dir=str(output_dir),
        reuse_cached=True,
        rerun_competitors=["dagua", "graphviz_dot"],
    )
    assert payload["graphs"]["tiny_force"]["competitors"]["graphviz_dot"]["status"] == "OK"
    assert calls["dagua"] == 1
    assert calls["dot"] == 1


@pytest.mark.smoke
def test_rare_suite_resumes_from_partial_results(tmp_path, monkeypatch):
    output_dir = tmp_path / "eval_output"
    run_dir = output_dir / "benchmark_db" / "rare" / "2026-03-12T00:00:00+00:00"
    run_dir.mkdir(parents=True, exist_ok=True)
    graph = DaguaGraph.from_edge_list([("a", "b"), ("b", "c")])
    tg_done = TestGraph(
        name="tiny_done",
        graph=graph,
        tags={"linear"},
        description="done",
        source="synthetic",
        expected_challenges="none",
    )
    tg_todo = TestGraph(
        name="tiny_todo",
        graph=graph,
        tags={"linear"},
        description="todo",
        source="synthetic",
        expected_challenges="none",
    )
    suite = [
        BenchmarkGraph(tg_done, "scale", "rare", False, "rare"),
        BenchmarkGraph(tg_todo, "scale", "rare", False, "rare"),
    ]
    partial = {
        "run_id": run_dir.name,
        "suite": "rare",
        "system": {"dagua_git_hash": "new"},
        "graphs": {
            "tiny_done": {
                "n_nodes": 3,
                "n_edges": 2,
                "structural_category": "scale",
                "description": "done",
                "expected_challenges": "none",
                "tags": ["linear"],
                "source": "synthetic",
                "visualize": False,
                "scale_tier": "rare",
                "competitors": {
                    "dagua": {
                        "status": "OK",
                        "runtime_seconds": 0.1,
                        "metrics": {"overall_quality": 80.0},
                        "composite_score": 80.0,
                        "metrics_computed": ["tier1"],
                        "metrics_skipped": ["tier2", "tier3"],
                        "positions_path": None,
                    }
                },
            }
        },
    }
    (run_dir / "results.partial.json").write_text(__import__("json").dumps(partial), encoding="utf-8")

    calls = {"dagua": 0}
    pos = torch.tensor([[0.0, 0.0], [0.0, 50.0], [0.0, 100.0]], dtype=torch.float32)

    class FakeDagua:
        name = "dagua"
        max_nodes = 100

        def available(self):
            return True

        def layout(self, graph, timeout=300.0):
            calls["dagua"] += 1
            return type("Result", (), {"pos": pos, "runtime_seconds": 0.02, "error": None})()

    monkeypatch.setattr("dagua.eval.benchmark._suite_graphs", lambda suite_name: suite)
    monkeypatch.setattr("dagua.eval.benchmark._competitor_map", lambda names=None: [FakeDagua()])
    monkeypatch.setattr("dagua.eval.benchmark._system_metadata", lambda: {"dagua_git_hash": "new"})
    monkeypatch.setattr("dagua.eval.benchmark.merge_latest_results", lambda output_dir=None: {"graphs": {}})

    payload = run_rare_suite(output_dir=str(output_dir), reuse_cached=False, resume_incomplete=True)
    assert payload["run_id"] == run_dir.name
    assert calls["dagua"] == 1
    assert payload["graphs"]["tiny_done"]["competitors"]["dagua"]["status"] == "OK"
    assert payload["graphs"]["tiny_todo"]["competitors"]["dagua"]["status"] == "OK"
    assert not (run_dir / "results.partial.json").exists()


@pytest.mark.smoke
def test_benchmark_run_status_reports_partial_progress(tmp_path):
    output_dir = tmp_path / "eval_output"
    run_dir = output_dir / "benchmark_db" / "rare" / "2026-03-12T00:00:00+00:00"
    run_dir.mkdir(parents=True, exist_ok=True)
    partial = {
        "run_id": run_dir.name,
        "suite": "rare",
        "graphs": {
            "done_graph": {
                "competitors": {
                    "dagua": {"status": "OK"},
                    "graphviz_sfdp": {"status": "SKIPPED"},
                }
            },
            "todo_graph": {
                "competitors": {
                    "dagua": {"status": "RUNNING"},
                }
            },
        },
    }
    metadata = {
        "graphs": ["done_graph", "todo_graph", "later_graph"],
    }
    progress = {
        "suite": "rare",
        "run_id": run_dir.name,
        "step": "running",
        "current_graph": "todo_graph",
        "current_competitor": "dagua",
        "completed_graphs": 1,
        "total_graphs": 3,
        "completed_pairs": 2,
        "total_pairs": 6,
        "last_artifact": "positions/done_graph__dagua.pt",
        "graphs": {},
    }
    (run_dir / "results.partial.json").write_text(__import__("json").dumps(partial), encoding="utf-8")
    (run_dir / "metadata.json").write_text(__import__("json").dumps(metadata), encoding="utf-8")
    (run_dir / "progress.json").write_text(__import__("json").dumps(progress), encoding="utf-8")

    status = benchmark_run_status(output_dir=str(output_dir), suite="rare")
    assert status["is_partial"] is True
    assert status["completed_graphs"] == 1
    assert status["total_graphs"] == 3
    assert status["remaining_graphs"] == 2
    assert status["completed_pairs"] == 2
    assert status["total_pairs"] == 6
    assert status["current_graph"] == "todo_graph"
    assert status["current_competitor"] == "dagua"
    assert status["graphs"]["done_graph"]["status"] == "complete"
    assert status["graphs"]["todo_graph"]["status"] == "incomplete"


@pytest.mark.smoke
def test_standard_suite_writes_partial_checkpoints(tmp_path, monkeypatch):
    output_dir = tmp_path / "eval_output"
    graph = DaguaGraph.from_edge_list([("a", "b")])
    tg = TestGraph(
        name="tiny_standard",
        graph=graph,
        tags={"linear"},
        description="tiny standard",
        source="synthetic",
        expected_challenges="none",
    )
    suite = [BenchmarkGraph(tg, "linear", "standard", True, "small")]
    pos = torch.tensor([[0.0, 0.0], [0.0, 50.0]], dtype=torch.float32)

    class FakeCompetitor:
        def __init__(self, name):
            self.name = name
            self.max_nodes = 10

        def available(self):
            return True

        def layout(self, graph, timeout=300.0):
            return type("Result", (), {"pos": pos, "runtime_seconds": 0.01, "error": None})()

    monkeypatch.setattr("dagua.eval.benchmark._suite_graphs", lambda suite_name: suite)
    monkeypatch.setattr("dagua.eval.benchmark._competitor_map", lambda names=None: [FakeCompetitor("dagua")])
    monkeypatch.setattr("dagua.eval.benchmark._system_metadata", lambda: {"dagua_git_hash": "new"})
    monkeypatch.setattr("dagua.eval.benchmark.merge_latest_results", lambda output_dir=None: {"graphs": {}})
    monkeypatch.setattr("dagua.eval.report.generate_report", lambda *args, **kwargs: {})

    payload = run_standard_suite(output_dir=str(output_dir), reuse_cached=False)
    run_dir = output_dir / "benchmark_db" / "standard" / payload["run_id"]
    assert not (run_dir / "results.partial.json").exists()
    assert (run_dir / "results.json").exists()
    assert (run_dir / "progress.json").exists()


@pytest.mark.smoke
def test_dagua_competitor_handles_multilevel_path(monkeypatch):
    graph = DaguaGraph.from_edge_list([("a", "b"), ("b", "c")])
    competitor = DaguaCompetitor()

    real_layout_config = __import__("dagua.config", fromlist=["LayoutConfig"]).LayoutConfig

    def tiny_multilevel_config(*args, **kwargs):
        kwargs.setdefault("device", "cpu")
        kwargs.setdefault("verbose", False)
        kwargs.setdefault("steps", 2)
        kwargs.setdefault("multilevel_threshold", 1)
        kwargs.setdefault("multilevel_coarse_steps", 1)
        kwargs.setdefault("multilevel_refine_steps", 1)
        kwargs.setdefault("multilevel_min_nodes", 10)
        return real_layout_config(*args, **kwargs)

    monkeypatch.setattr("dagua.config.LayoutConfig", tiny_multilevel_config)
    result = competitor.layout(graph)

    assert result.error is None
    assert result.pos is not None
    assert result.pos.shape == (graph.num_nodes, 2)
