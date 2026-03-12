from __future__ import annotations

from pathlib import Path

import pytest
import torch

from dagua import DaguaGraph, LayoutConfig
from dagua.eval import sweep as sweep_mod
from dagua.eval.graphs import TestGraph


def test_get_placement_tuning_suites_include_scale_validation() -> None:
    search = sweep_mod.get_placement_search_suite()
    validation = sweep_mod.get_placement_validation_suite()

    assert search
    assert validation
    assert all(tg.graph.num_nodes <= 1000 for tg in search)
    assert all(tg.graph.num_nodes >= 2000 for tg in validation)


@pytest.mark.smoke
def test_run_placement_tuning_writes_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    graph_small = DaguaGraph.from_edge_list([("a", "b"), ("b", "c")])
    graph_large = DaguaGraph.from_edge_list([("a", "b"), ("b", "c"), ("c", "d")])
    graph_large.add_node("e")

    search_graphs = [TestGraph(name="small_case", graph=graph_small)]
    validation_graphs = [TestGraph(name="large_case", graph=graph_large)]

    monkeypatch.setattr(sweep_mod, "get_placement_search_suite", lambda: search_graphs)
    monkeypatch.setattr(sweep_mod, "get_placement_validation_suite", lambda: validation_graphs)

    def fake_layout(graph: DaguaGraph, config: LayoutConfig) -> torch.Tensor:
        n = graph.num_nodes
        x = torch.arange(n, dtype=torch.float32)
        return torch.stack([x * float(config.node_sep), x * float(config.rank_sep)], dim=1)

    def fake_metrics(pos: torch.Tensor, edge_index: torch.Tensor, node_sizes: torch.Tensor | None) -> dict[str, float]:
        return {
            "dag_consistency": 1.0,
            "edge_crossings": float(pos.shape[0] % 3),
            "overlap_count": 0.0,
            "edge_length_cv": 1.0 / max(float(pos.shape[0]), 1.0),
        }

    monkeypatch.setattr(sweep_mod, "layout", fake_layout)
    monkeypatch.setattr(sweep_mod, "compute_all_metrics", fake_metrics)

    result = sweep_mod.run_placement_tuning(output_dir=str(tmp_path))

    assert result.search_suite_graphs == ["small_case"]
    assert result.validation_suite_graphs == ["large_case"]
    assert result.all_candidates
    assert result.pareto_frontier
    assert (tmp_path / "placement_tuning.json").exists()
    assert (tmp_path / "placement_tuning.md").exists()

