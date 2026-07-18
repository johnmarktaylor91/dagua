import json
from pathlib import Path

import pytest
import torch

from dagua.eval.benchmark import (
    DEFAULT_COMPETITOR_ORDER,
    BenchmarkGraph,
    BenchmarkResult,
    _build_results_payload,
    _competitor_signature,
    _declares_hierarchy,
    _metric_payload,
    benchmark_run_status,
    get_rare_suite_graphs,
    get_standard_suite_graphs,
    merge_latest_results,
    run_rare_suite,
    run_standard_suite,
)
from dagua.eval.competitors import get_available_competitors
from dagua.eval.competitors.dagua_competitor import DaguaCompetitor
from dagua.eval.graphs import TestGraph, get_test_graphs
from dagua.eval.report import generate_benchmark_markdown, generate_report
from dagua.graph import DaguaGraph
from dagua.metrics import _CLUSTER_WEIGHTS, composite_auto
from dagua.utils import longest_path_layering


def test_production_random_dag_routes_to_directed_ruler() -> None:
    """Corpus DAG metadata reaches the directed composite without an override."""
    test_graph = next(graph for graph in get_test_graphs() if graph.name == "random_dag_200")
    graph = test_graph.graph
    graph.compute_node_sizes()
    ranks = torch.tensor(
        longest_path_layering(graph.edge_index, graph.num_nodes), dtype=torch.float32
    )
    pos = torch.stack((torch.arange(graph.num_nodes, dtype=torch.float32), ranks), dim=1)

    assert _declares_hierarchy(test_graph)
    assert graph.is_semantically_directed is True
    metrics, score, _, _ = _metric_payload(
        graph,
        pos,
        "quick",
        declared_hierarchical=_declares_hierarchy(test_graph),
        semantically_directed=True,
    )

    assert metrics["declared_hierarchical"] is True
    assert score == pytest.approx(composite_auto(metrics, True))


def test_metric_payload_full_forwards_cluster_quality_metadata() -> None:
    """Full benchmark scoring computes finite cluster ruler keys for clustered graphs."""
    graph = DaguaGraph()
    for index in range(5):
        graph.add_node(index)
    graph.add_edge(3, 4)
    graph.node_sizes = torch.full((5, 2), 10.0)
    graph.clusters = {"center": [0, 1], "outer": [0, 1, 2]}
    graph.cluster_parents = {"center": "outer"}
    graph.cluster_labels = {"center": "Center", "outer": "Outer"}
    pos = torch.tensor(
        [
            [0.0, 0.0],
            [5.0, 0.0],
            [2.5, 0.0],
            [-40.0, 0.0],
            [40.0, 0.0],
        ]
    )

    metrics, score, computed, skipped = _metric_payload(graph, pos, "full")
    without_cluster = {**metrics, **{name: None for name in _CLUSTER_WEIGHTS}}

    assert "tier2" in computed
    assert "tier3" in computed
    assert not skipped
    for name in _CLUSTER_WEIGHTS:
        assert metrics[name] is not None
        assert float(metrics[name]) == pytest.approx(float(metrics[name]))
    assert score == pytest.approx(composite_auto(metrics))
    assert score != pytest.approx(composite_auto(without_cluster))


@pytest.mark.smoke
def test_standard_suite_contains_expected_cases():
    suite = get_standard_suite_graphs()
    names = {bg.test_graph.name for bg in suite}

    assert "chain_100" in names
    assert "binary_tree_127" in names
    assert "tl_cnn_small" in names
    assert "tl_resnet_2block" in names
    assert "tl_transformer_1layer" in names
    assert "real_karate_34" in names
    assert "er_2000" in names
    assert "ba_5000" in names
    assert "grid_50x50" in names
    assert "org_chart_deep" in names
    assert "scale_100k" in names
    assert len(suite) >= 40


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
def test_merge_latest_results_and_generate_report(tmp_path: Path) -> None:
    """Generate the report artifacts and keep exclusion notes visible.

    Returns
    -------
    None
        This test asserts that the report pipeline emits the expected files and
        aggregate exclusion note.
    """
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

    (standard_run / "results.json").write_text(
        __import__("json").dumps(standard_payload), encoding="utf-8"
    )
    (rare_run / "results.json").write_text(__import__("json").dumps(rare_payload), encoding="utf-8")
    (standard_run.parent / "latest").symlink_to(standard_run.name)
    (rare_run.parent / "latest").symlink_to(rare_run.name)

    combined = merge_latest_results(str(output_dir))
    assert "residual_block" in combined["graphs"]
    assert "scale_500000" in combined["graphs"]

    artifacts = generate_report(
        output_dir=str(output_dir), combined_results=combined, compile_pdf=False
    )
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
    assert Path(artifacts["artifact_index_json"]).exists()
    assert Path(artifacts["artifact_index_md"]).exists()
    assert (output_dir / "visuals" / "comparisons" / "residual_block_comparison.png").exists()
    assert (output_dir / "report" / "prose_prompt.md").exists()
    assert (output_dir / "report" / "review_round_1.json").exists()
    tex_source = Path(artifacts["tex"]).read_text(encoding="utf-8")
    assert (
        "Averages are computed over successful runs only; 1 failed or skipped runs were excluded."
        in tex_source
    )


@pytest.mark.smoke
def test_generate_benchmark_markdown_includes_failure_analysis(tmp_path: Path) -> None:
    """Benchmark markdown should list excluded runs and their recorded reasons.

    Returns
    -------
    None
        This test asserts that the markdown report includes failure summaries,
        reason buckets, and useful per-cell failure labels.
    """
    output_path = tmp_path / "benchmark.md"
    results = [
        BenchmarkResult(
            graph_name="graph_ok",
            graph_nodes=128,
            graph_edges=256,
            competitor="dagua",
            status="OK",
            runtime_seconds=0.25,
            metrics={"dag_consistency": 1.0},
            composite_score=88.0,
        ),
        BenchmarkResult(
            graph_name="graph_ok",
            graph_nodes=128,
            graph_edges=256,
            competitor="graphviz_dot",
            status="FAILED",
            runtime_seconds=0.4,
            metrics={},
            composite_score=None,
            reason="exception",
            error="RuntimeError: CUDA out of memory",
        ),
        BenchmarkResult(
            graph_name="graph_scale",
            graph_nodes=5_000,
            graph_edges=8_000,
            competitor="graphviz_dot",
            status="SKIPPED",
            runtime_seconds=None,
            metrics={},
            composite_score=None,
            reason="exceeds known limit",
        ),
        BenchmarkResult(
            graph_name="graph_unknown",
            graph_nodes=640,
            graph_edges=900,
            competitor="fruchterman_reingold",
            status="FAILED",
            runtime_seconds=0.7,
            metrics={},
            composite_score=None,
        ),
    ]

    generate_benchmark_markdown(results, str(output_path))

    report_text = output_path.read_text(encoding="utf-8")
    assert (
        "Averages computed over successful runs only. See Failure Analysis for excluded runs."
        in report_text
    )
    assert "## Failure Analysis" in report_text
    assert "3 of 4 total runs failed (75.0%)." in report_text
    assert "**dagua** (0 failures / 1 graphs):" in report_text
    assert (
        "**graphviz_dot** (2 failures / 2 graphs):\n"
        "- graph_ok: exception -- RuntimeError: CUDA out of memory\n"
        "- graph_scale: exceeds known limit (graph has 5,000 nodes)"
    ) in report_text
    assert (
        "**fruchterman_reingold** (1 failures / 1 graphs):\n- graph_unknown: unknown" in report_text
    )
    assert "| exception | 1 | graphviz_dot |" in report_text
    assert "| exceeds known limit | 1 | graphviz_dot |" in report_text
    assert "| unknown | 1 | fruchterman_reingold |" in report_text
    assert "| graph_ok | 128 | 256 | 88.0 (0.25s) | - |" in report_text


@pytest.mark.smoke
def test_dagua_competitor_signature_uses_device_and_source_hash(monkeypatch):
    monkeypatch.setattr(
        "dagua.eval.benchmark._dagua_source_signature",
        lambda: "abc123def4567890",  # pragma: allowlist secret
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert _competitor_signature("dagua", {"dagua_git_hash": "ignored"}) == (
        "dagua:cpu:abc123def4567890"
    )


@pytest.mark.smoke
def test_competitor_signatures_cover_extended_families(monkeypatch):
    """Ensure the benchmark cache key logic covers all supported families."""
    source_signature = "abc123def4567890"  # pragma: allowlist secret
    monkeypatch.setattr(
        "dagua.eval.benchmark._dagua_source_signature",
        lambda: source_signature,  # pragma: allowlist secret
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    system = {
        "graphviz": "dot 12.0",
        "elk": "0.9.0",
        "dagre": "1.1.5",
        "igraph": "0.11.8",
        "networkx": "3.4",
        "sgd2": "1.0.0",
        "pyg": "2.6.1",
        "fa2": "installed",
        "umap": "0.5.7",
        "sklearn": "1.6.1",
        "scipy": "1.15.2",
    }
    names = [
        "dagua",
        "graphviz_dot",
        "graphviz_sfdp",
        "graphviz_neato",
        "graphviz_fdp",
        "elk_layered",
        "dagre",
        "igraph_sugiyama",
        "igraph_fr",
        "igraph_kamada_kawai",
        "igraph_mds",
        "igraph_davidson_harel",
        "igraph_graphopt",
        "igraph_drl",
        "igraph_lgl",
        "igraph_rt",
        "nx_spring",
        "nx_kamada_kawai",
        "nx_spectral",
        "sgd2",
        "sgd2_mds",
        "neulay",
        "fa2_ref",
        "tsne_graph",
        "umap_graph",
        "ogdf_gem",
        "classic_fr",
        "classic_fmmm",
    ]

    signatures = {name: _competitor_signature(name, system) for name in names}

    assert all(":None" not in signature for signature in signatures.values())
    assert signatures["classic_fr"] == f"classic_fr:{source_signature}"
    assert signatures["classic_fmmm"] == f"classic_fmmm:{source_signature}"
    assert signatures["igraph_mds"] == "igraph_mds:0.11.8"
    assert signatures["sgd2_mds"] == "sgd2_mds:1.0.0"
    assert signatures["neulay"] == "neulay:2.6.1"
    assert signatures["tsne_graph"] == "tsne_graph:1.6.1:1.15.2"
    assert signatures["umap_graph"] == "umap_graph:0.5.7:1.15.2"
    assert signatures["ogdf_gem"] in {
        "ogdf_gem:ogdf_available",
        "ogdf_gem:ogdf_unavailable",
    }


@pytest.mark.smoke
def test_build_results_payload_forwards_seed_to_competitor(tmp_path):
    graph = DaguaGraph.from_edge_list([("a", "b"), ("b", "c")])
    tg = TestGraph(
        name="tiny_seeded",
        graph=graph,
        tags={"linear"},
        description="seed forwarding",
        source="synthetic",
        expected_challenges="none",
    )
    suite = [BenchmarkGraph(tg, "linear", "standard", True, "small")]
    observed: dict[str, int | None] = {"seed": None}
    pos = torch.tensor([[0.0, 0.0], [0.0, 50.0], [0.0, 100.0]], dtype=torch.float32)

    class FakeCompetitor:
        name = "fake_seeded"
        max_nodes = 10

        def available(self):
            return True

        def layout(self, graph, timeout=300.0, seed=None):
            del graph, timeout
            observed["seed"] = seed
            return type("Result", (), {"pos": pos, "runtime_seconds": 0.01, "error": None})()

    cached_payload = {
        "run_id": "2026-03-17T00:00:00+00:00",
        "graphs": {
            "tiny_seeded": {
                "competitors": {
                    "fake_seeded": {
                        "status": "OK",
                        "runtime_seconds": 0.5,
                        "metrics": {},
                        "composite_score": None,
                        "metrics_computed": [],
                        "metrics_skipped": ["tier1", "tier2", "tier3"],
                        "positions_path": None,
                    }
                }
            }
        },
    }
    cached_metadata = {
        "graph_signatures": {"tiny_seeded": "graph-v1"},
        "competitor_signatures": {"fake_seeded": "fake-v1"},
    }

    payload = _build_results_payload(
        suite="standard",
        run_id="2026-03-18T00:00:00+00:00",
        graphs=suite,
        competitors=[FakeCompetitor()],
        timeout=30.0,
        output_dir=str(tmp_path),
        seed=99,
        cached_payload=cached_payload,
        cached_metadata=cached_metadata,
        latest_run_dir=tmp_path,
        graph_signatures={"tiny_seeded": "graph-v1"},
        competitor_signatures={"fake_seeded": "fake-v1"},
    )

    assert observed["seed"] == 99
    assert payload["graphs"]["tiny_seeded"]["competitors"]["fake_seeded"]["status"] == "OK"
    assert "reused_from" not in payload["graphs"]["tiny_seeded"]["competitors"]["fake_seeded"]


@pytest.mark.smoke
def test_default_competitor_order_covers_expected_and_available_competitors():
    """Keep the default benchmark roster aligned with registered competitors."""
    expected = {
        "dagua",
        "graphviz_dot",
        "elk_layered",
        "dagre",
        "igraph_sugiyama",
        "graphviz_sfdp",
        "graphviz_neato",
        "graphviz_fdp",
        "nx_spring",
        "nx_kamada_kawai",
        "nx_spectral",
        "nx_spectral_random_walk",
        "igraph_fr",
        "igraph_kamada_kawai",
        "igraph_mds",
        "igraph_davidson_harel",
        "igraph_graphopt",
        "igraph_drl",
        "igraph_lgl",
        "igraph_rt",
        "igraph_rt_horizontal",
        "sgd2",
        "sgd2_mds",
        "sgd2_multi_ref",
        "neulay",
        "fa2_ref",
        "tsne_graph",
        "umap_graph",
        "linlog",
        "cytoscape_fcose",
        "gephi_yifanhu",
        "mulment_reference",
        "nnpnet_reference",
        "ogdf_gem",
        "ogdf_fmmm",
        "ogdf_stress",
        "ogdf_sugiyama",
        "ogdf_davidson_harel",
        "ogdf_pivot_mds",
        "classic_fr",
        "classic_kk",
        "classic_fr_kk",
        "classic_kk_fr",
        "classic_fa2",
        "classic_stress_sgd",
        "classic_sgd2_multi",
        "classic_sugiyama",
        "classic_spectral",
        "classic_classical_mds",
        "classic_stress_maj",
        "classic_pivot_mds",
        "classic_rt",
        "classic_linlog",
        "classic_graphopt",
        "classic_neato",
        "classic_gem",
        "classic_tsnet",
        "classic_umap",
        "classic_neulay",
        "classic_maxent_stress",
        "classic_davidson_harel",
        "classic_fmmm",
        "classic_sfdp",
        "classic_drl",
        "classic_lgl",
        "classic_fcose",
        "mulment_reimpl",
        "nnpnet_reimpl",
    }
    order = set(DEFAULT_COMPETITOR_ORDER)
    available = {competitor.name for competitor in get_available_competitors()}

    assert expected <= order
    assert available <= order


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
        "graph_signatures": {
            "tiny_chain": __import__("hashlib")
            .sha256(
                __import__("json")
                .dumps(graph.to_json(), sort_keys=True, separators=(",", ":"))
                .encode("utf-8")
            )
            .hexdigest()
        },
        "competitor_signatures": {
            "graphviz_dot": "graphviz_dot:dot 1.0",
            "dagua": "dagua:cpu:newhash",
        },
    }
    (latest_run / "results.json").write_text(
        __import__("json").dumps(latest_payload), encoding="utf-8"
    )
    (latest_run / "metadata.json").write_text(
        __import__("json").dumps(latest_metadata), encoding="utf-8"
    )
    (latest_run.parent / "latest").symlink_to(latest_run.name)

    class FakeCompetitor:
        def __init__(self, name):
            self.name = name
            self.max_nodes = 10

        def available(self):
            return True

        def layout(self, graph, timeout=300.0, seed=None):
            del graph, timeout, seed
            raise AssertionError(f"{self.name} should not have been rerun")

    class FakeDagua(FakeCompetitor):
        def layout(self, graph, timeout=300.0, seed=None):
            del graph, timeout, seed
            return type("Result", (), {"pos": pos, "runtime_seconds": 0.02, "error": None})()

    monkeypatch.setattr("dagua.eval.benchmark._suite_graphs", lambda suite_name: suite)
    monkeypatch.setattr(
        "dagua.eval.benchmark._competitor_map",
        lambda names=None: [FakeDagua("dagua"), FakeCompetitor("graphviz_dot")],
    )
    monkeypatch.setattr(
        "dagua.eval.benchmark._system_metadata",
        lambda: {"dagua_git_hash": "new", "graphviz": "dot 1.0"},
    )
    monkeypatch.setattr(
        "dagua.eval.benchmark.merge_latest_results", lambda output_dir=None: {"graphs": {}}
    )
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

        def layout(self, graph, timeout=300.0, seed=None):
            del graph, timeout, seed
            calls[self.name.split("_")[-1] if self.name != "dagua" else "dagua"] += 1
            return type("Result", (), {"pos": pos, "runtime_seconds": 0.01, "error": None})()

    monkeypatch.setattr("dagua.eval.benchmark._suite_graphs", lambda suite_name: suite)
    monkeypatch.setattr(
        "dagua.eval.benchmark._competitor_map",
        lambda names=None: [FakeCompetitor("dagua"), FakeCompetitor("graphviz_dot")],
    )
    monkeypatch.setattr(
        "dagua.eval.benchmark._system_metadata",
        lambda: {"dagua_git_hash": "new", "graphviz": "dot 1.0"},
    )
    monkeypatch.setattr(
        "dagua.eval.benchmark.merge_latest_results", lambda output_dir=None: {"graphs": {}}
    )
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
    (run_dir / "results.partial.json").write_text(
        __import__("json").dumps(partial), encoding="utf-8"
    )

    calls = {"dagua": 0}
    pos = torch.tensor([[0.0, 0.0], [0.0, 50.0], [0.0, 100.0]], dtype=torch.float32)

    class FakeDagua:
        name = "dagua"
        max_nodes = 100

        def available(self):
            return True

        def layout(self, graph, timeout=300.0, seed=None):
            del graph, timeout, seed
            calls["dagua"] += 1
            return type("Result", (), {"pos": pos, "runtime_seconds": 0.02, "error": None})()

    monkeypatch.setattr("dagua.eval.benchmark._suite_graphs", lambda suite_name: suite)
    monkeypatch.setattr("dagua.eval.benchmark._competitor_map", lambda names=None: [FakeDagua()])
    monkeypatch.setattr("dagua.eval.benchmark._system_metadata", lambda: {"dagua_git_hash": "new"})
    monkeypatch.setattr(
        "dagua.eval.benchmark.merge_latest_results", lambda output_dir=None: {"graphs": {}}
    )

    payload = run_rare_suite(output_dir=str(output_dir), reuse_cached=False, resume_incomplete=True)
    assert payload["run_id"] == run_dir.name
    assert calls["dagua"] == 1
    assert payload["graphs"]["tiny_done"]["competitors"]["dagua"]["status"] == "OK"
    assert payload["graphs"]["tiny_todo"]["competitors"]["dagua"]["status"] == "OK"
    assert not (run_dir / "results.partial.json").exists()


@pytest.mark.smoke
def test_standard_suite_retry_failed_resumes_only_failed_results(tmp_path, monkeypatch):
    output_dir = tmp_path / "eval_output"
    run_dir = output_dir / "benchmark_db" / "standard" / "2026-03-12T00:00:00+00:00"
    run_dir.mkdir(parents=True, exist_ok=True)
    graph = DaguaGraph.from_edge_list([("a", "b"), ("b", "c")])
    tg = TestGraph(
        name="tiny_retry",
        graph=graph,
        tags={"linear"},
        description="retry only failures",
        source="synthetic",
        expected_challenges="none",
    )
    suite = [BenchmarkGraph(tg, "linear", "standard", True, "small")]
    partial = {
        "run_id": run_dir.name,
        "suite": "standard",
        "system": {"dagua_git_hash": "new", "graphviz": "dot 1.0"},
        "graphs": {
            "tiny_retry": {
                "n_nodes": 3,
                "n_edges": 2,
                "structural_category": "linear",
                "description": "retry only failures",
                "expected_challenges": "none",
                "tags": ["linear"],
                "source": "synthetic",
                "visualize": True,
                "scale_tier": "small",
                "competitors": {
                    "dagua": {
                        "status": "FAILED",
                        "reason": "exception",
                        "runtime_seconds": None,
                        "metrics": {},
                        "composite_score": None,
                        "metrics_computed": [],
                        "metrics_skipped": ["tier1", "tier2", "tier3"],
                        "positions_path": None,
                    },
                    "graphviz_dot": {
                        "status": "OK",
                        "runtime_seconds": 0.01,
                        "metrics": {"overall_quality": 80.0},
                        "composite_score": 80.0,
                        "metrics_computed": ["tier1"],
                        "metrics_skipped": ["tier2", "tier3"],
                        "positions_path": None,
                    },
                },
            }
        },
    }
    (run_dir / "results.partial.json").write_text(json.dumps(partial), encoding="utf-8")

    calls = {"dagua": 0, "graphviz_dot": 0}
    pos = torch.tensor([[0.0, 0.0], [0.0, 50.0], [0.0, 100.0]], dtype=torch.float32)

    class FakeCompetitor:
        def __init__(self, name):
            self.name = name
            self.max_nodes = 10

        def available(self):
            return True

        def layout(self, graph, timeout=300.0, seed=None):
            del graph, timeout, seed
            calls[self.name] += 1
            return type("Result", (), {"pos": pos, "runtime_seconds": 0.02, "error": None})()

    monkeypatch.setattr("dagua.eval.benchmark._suite_graphs", lambda suite_name: suite)
    monkeypatch.setattr(
        "dagua.eval.benchmark._competitor_map",
        lambda names=None: [FakeCompetitor("dagua"), FakeCompetitor("graphviz_dot")],
    )
    monkeypatch.setattr(
        "dagua.eval.benchmark._system_metadata",
        lambda: {"dagua_git_hash": "new", "graphviz": "dot 1.0"},
    )
    monkeypatch.setattr(
        "dagua.eval.benchmark.merge_latest_results", lambda output_dir=None: {"graphs": {}}
    )
    monkeypatch.setattr("dagua.eval.report.generate_report", lambda *args, **kwargs: {})

    payload = run_standard_suite(
        output_dir=str(output_dir),
        reuse_cached=False,
        resume_incomplete=True,
        retry_failed=True,
    )

    assert payload["run_id"] == run_dir.name
    assert calls["dagua"] == 1
    assert calls["graphviz_dot"] == 0
    assert payload["graphs"]["tiny_retry"]["competitors"]["dagua"]["status"] == "OK"
    assert payload["graphs"]["tiny_retry"]["competitors"]["graphviz_dot"]["status"] == "OK"


@pytest.mark.smoke
def test_standard_suite_retry_failed_reruns_failed_cached_results(tmp_path, monkeypatch):
    output_dir = tmp_path / "eval_output"
    latest_run = output_dir / "benchmark_db" / "standard" / "2026-03-12T00:00:00+00:00"
    latest_run.mkdir(parents=True, exist_ok=True)

    graph = DaguaGraph.from_edge_list([("a", "b"), ("b", "c")])
    tg = TestGraph(
        name="tiny_cached_retry",
        graph=graph,
        tags={"linear"},
        description="retry cached failures",
        source="synthetic",
        expected_challenges="none",
    )
    suite = [BenchmarkGraph(tg, "linear", "standard", True, "small")]

    latest_payload = {
        "run_id": latest_run.name,
        "suite": "standard",
        "system": {"graphviz": "dot 1.0"},
        "graphs": {
            "tiny_cached_retry": {
                "n_nodes": 3,
                "n_edges": 2,
                "structural_category": "linear",
                "description": "retry cached failures",
                "expected_challenges": "none",
                "tags": ["linear"],
                "source": "synthetic",
                "visualize": True,
                "scale_tier": "small",
                "competitors": {
                    "graphviz_dot": {
                        "status": "FAILED",
                        "reason": "exception",
                        "runtime_seconds": None,
                        "metrics": {},
                        "composite_score": None,
                        "metrics_computed": [],
                        "metrics_skipped": ["tier1", "tier2", "tier3"],
                        "positions_path": None,
                    }
                },
            }
        },
    }
    latest_metadata = {
        "graph_signatures": {
            "tiny_cached_retry": __import__("hashlib")
            .sha256(
                json.dumps(graph.to_json(), sort_keys=True, separators=(",", ":")).encode("utf-8")
            )
            .hexdigest()
        },
        "competitor_signatures": {
            "graphviz_dot": "graphviz_dot:dot 1.0",
            "dagua": "dagua:cpu:newhash",
        },
    }
    (latest_run / "results.json").write_text(json.dumps(latest_payload), encoding="utf-8")
    (latest_run / "metadata.json").write_text(json.dumps(latest_metadata), encoding="utf-8")
    (latest_run.parent / "latest").symlink_to(latest_run.name)

    calls = {"graphviz_dot": 0}
    pos = torch.tensor([[0.0, 0.0], [0.0, 50.0], [0.0, 100.0]], dtype=torch.float32)

    class FakeCompetitor:
        def __init__(self, name):
            self.name = name
            self.max_nodes = 10

        def available(self):
            return True

        def layout(self, graph, timeout=300.0, seed=None):
            del graph, timeout, seed
            calls[self.name] += 1
            return type("Result", (), {"pos": pos, "runtime_seconds": 0.02, "error": None})()

    monkeypatch.setattr("dagua.eval.benchmark._suite_graphs", lambda suite_name: suite)
    monkeypatch.setattr(
        "dagua.eval.benchmark._competitor_map", lambda names=None: [FakeCompetitor("graphviz_dot")]
    )
    monkeypatch.setattr("dagua.eval.benchmark._system_metadata", lambda: {"graphviz": "dot 1.0"})
    monkeypatch.setattr(
        "dagua.eval.benchmark.merge_latest_results", lambda output_dir=None: {"graphs": {}}
    )
    monkeypatch.setattr("dagua.eval.report.generate_report", lambda *args, **kwargs: {})

    payload = run_standard_suite(
        output_dir=str(output_dir),
        reuse_cached=True,
        resume_incomplete=False,
        retry_failed=True,
    )

    result = payload["graphs"]["tiny_cached_retry"]["competitors"]["graphviz_dot"]
    assert calls["graphviz_dot"] == 1
    assert result["status"] == "OK"
    assert "reused_from" not in result


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
    (run_dir / "results.partial.json").write_text(
        __import__("json").dumps(partial), encoding="utf-8"
    )
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
def test_standard_suite_checkpoints_after_each_competitor(tmp_path, monkeypatch):
    output_dir = tmp_path / "eval_output"
    graph = DaguaGraph.from_edge_list([("a", "b"), ("b", "c")])
    tg = TestGraph(
        name="tiny_checkpoint",
        graph=graph,
        tags={"linear"},
        description="checkpoint after each competitor",
        source="synthetic",
        expected_challenges="none",
    )
    suite = [BenchmarkGraph(tg, "linear", "standard", True, "small")]
    pos = torch.tensor([[0.0, 0.0], [0.0, 50.0], [0.0, 100.0]], dtype=torch.float32)
    observed = {"checkpoint_seen": False}

    class FakeCompetitor:
        def __init__(self, name):
            self.name = name
            self.max_nodes = 10

        def available(self):
            return True

        def layout(self, graph, timeout=300.0, seed=None):
            del graph, timeout, seed
            if self.name == "dagua":
                return type("Result", (), {"pos": pos, "runtime_seconds": 0.01, "error": None})()

            partial_paths = sorted(
                {
                    path.resolve()
                    for path in (output_dir / "benchmark_db" / "standard").glob(
                        "*/results.partial.json"
                    )
                }
            )
            if len(partial_paths) == 1:
                partial_payload = json.loads(partial_paths[0].read_text(encoding="utf-8"))
                competitors = partial_payload["graphs"]["tiny_checkpoint"]["competitors"]
                observed["checkpoint_seen"] = (
                    competitors.get("dagua", {}).get("status") == "OK"
                    and "graphviz_dot" not in competitors
                )
            raise KeyboardInterrupt("stop after first competitor")

    monkeypatch.setattr("dagua.eval.benchmark._suite_graphs", lambda suite_name: suite)
    monkeypatch.setattr(
        "dagua.eval.benchmark._competitor_map",
        lambda names=None: [FakeCompetitor("dagua"), FakeCompetitor("graphviz_dot")],
    )
    monkeypatch.setattr(
        "dagua.eval.benchmark._system_metadata",
        lambda: {"dagua_git_hash": "new", "graphviz": "dot 1.0"},
    )

    with pytest.raises(KeyboardInterrupt, match="stop after first competitor"):
        run_standard_suite(output_dir=str(output_dir), reuse_cached=False)

    partial_paths = sorted(
        {
            path.resolve()
            for path in (output_dir / "benchmark_db" / "standard").glob("*/results.partial.json")
        }
    )
    assert len(partial_paths) == 1
    partial_payload = json.loads(partial_paths[0].read_text(encoding="utf-8"))
    competitors = partial_payload["graphs"]["tiny_checkpoint"]["competitors"]
    assert observed["checkpoint_seen"] is True
    assert competitors["dagua"]["status"] == "OK"
    assert "graphviz_dot" not in competitors


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

        def layout(self, graph, timeout=300.0, seed=None):
            del graph, timeout, seed
            return type("Result", (), {"pos": pos, "runtime_seconds": 0.01, "error": None})()

    monkeypatch.setattr("dagua.eval.benchmark._suite_graphs", lambda suite_name: suite)
    monkeypatch.setattr(
        "dagua.eval.benchmark._competitor_map", lambda names=None: [FakeCompetitor("dagua")]
    )
    monkeypatch.setattr("dagua.eval.benchmark._system_metadata", lambda: {"dagua_git_hash": "new"})
    monkeypatch.setattr(
        "dagua.eval.benchmark.merge_latest_results", lambda output_dir=None: {"graphs": {}}
    )
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
