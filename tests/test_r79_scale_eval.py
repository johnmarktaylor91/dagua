"""Regression tests for the R79 scale ladder helper script."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

import scripts.r79_scale_eval as scale_eval


def test_parse_sfdp_plain_fills_missing_nodes_from_neighbor_centroid() -> None:
    """Missing SFDP node rows should be filled without raising."""
    edge_index = torch.tensor([[0, 2], [1, 1]], dtype=torch.long)
    output = "\n".join(
        [
            "graph 1 1 1",
            "node 0 0.0 0.0 0.1 0.1 0 solid ellipse black lightgrey",
            "node 2 4.0 0.0 0.1 0.1 2 solid ellipse black lightgrey",
            "node 99 9.0 9.0 0.1 0.1 99 solid ellipse black lightgrey",
            "node not_an_int 1.0 1.0 0.1 0.1 bad solid ellipse black lightgrey",
        ]
    )

    pos, warnings = scale_eval._parse_sfdp_plain(output, 3, edge_index)

    assert torch.allclose(pos[1], torch.tensor([2.0, 0.0]))
    assert any("omitted 1 node positions" in warning for warning in warnings)
    assert any("out-of-range" in warning for warning in warnings)
    assert any("malformed" in warning for warning in warnings)


def test_write_sfdp_dot_includes_isolated_trailing_nodes(tmp_path: Path) -> None:
    """DOT output should declare all nodes, even when the highest ids are isolated."""
    dot_path = tmp_path / "graph.dot"
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)

    scale_eval._write_sfdp_dot(edge_index, 4, dot_path)

    dot_text = dot_path.read_text(encoding="ascii")
    assert "  3;\n" in dot_text


def test_run_sfdp_reference_records_error_row_on_exception(monkeypatch: Any) -> None:
    """Reference failures should produce ERROR rows instead of escaping."""

    def fake_run(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
        """Return a successful fake subprocess result."""
        return SimpleNamespace(stdout="", stderr="sfdp fake version", returncode=0)

    def raise_generate_graph(_graph_type: str, _num_nodes: int, _seed: int) -> torch.Tensor:
        """Raise from graph generation to exercise the row-level error handler."""
        raise RuntimeError("boom")

    monkeypatch.setattr(scale_eval.shutil, "which", lambda _name: "/usr/bin/sfdp")
    monkeypatch.setattr(scale_eval.subprocess, "run", fake_run)
    monkeypatch.setattr(scale_eval, "generate_graph", raise_generate_graph)

    result = scale_eval.run_sfdp_reference("sparse_er", 20_000, 79)

    assert result["ok"] is False
    assert result["status"] == "ERROR"
    assert "RuntimeError" in result["error"]
