"""Regression tests for the Cytoscape AVSDF pipeline."""

from __future__ import annotations

from pathlib import Path

import torch

from dagua.layout.ops.pipelines import PIPELINE_REGISTRY, get_pipeline_function
from dagua.layout.ops.pipelines.avsdf import build_avsdf_pipeline, layout_avsdf_pipeline
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState


def _cycle_with_chords() -> torch.Tensor:
    """Return a small AVSDF test graph.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, E]``.
    """
    return torch.tensor(
        [[0, 1, 2, 3, 4, 5, 0, 1], [1, 2, 3, 4, 5, 0, 3, 4]],
        dtype=torch.long,
    )


def test_avsdf_pipeline_is_registered() -> None:
    """The dynamic registry should resolve ``avsdf``.

    Returns
    -------
    None
        Registry lookup must return the public entrypoint.
    """
    assert PIPELINE_REGISTRY["avsdf"] == (
        "dagua.layout.ops.pipelines.avsdf",
        "layout_avsdf_pipeline",
    )
    assert get_pipeline_function("AVSDF") is layout_avsdf_pipeline


def test_avsdf_is_deterministic_and_records_order() -> None:
    """AVSDF should be deterministic and expose its circular order.

    Returns
    -------
    None
        Repeated outputs must match bit-for-bit for the same graph.
    """
    edge_index = _cycle_with_chords()
    first = layout_avsdf_pipeline(edge_index=edge_index, num_nodes=6)
    second = layout_avsdf_pipeline(edge_index=edge_index, num_nodes=6)
    state = build_avsdf_pipeline().apply(
        LayoutProblem(edge_index=edge_index, num_nodes=6),
        SolveState(),
        RuntimeContext(plan=ExecutionPlan(device="cpu")),
    )

    assert torch.equal(first, second)
    assert first.shape == (6, 2)
    assert torch.isfinite(first).all()
    assert sorted(state.extras["avsdf_order"]) == list(range(6))


def test_cytoscape_family_production_pipelines_have_no_runtime_delegation() -> None:
    """Production Cytoscape-family code must not call reference adapters.

    Returns
    -------
    None
        Source text should contain no subprocess or competitor hooks.
    """
    root = Path(__file__).parents[1]
    for path in [
        root / "dagua" / "layout" / "ops" / "cytoscape.py",
        root / "dagua" / "layout" / "ops" / "pipelines" / "avsdf.py",
        root / "dagua" / "layout" / "ops" / "pipelines" / "cose.py",
        root / "dagua" / "layout" / "ops" / "pipelines" / "cose_bilkent.py",
        root / "dagua" / "layout" / "ops" / "pipelines" / "cise.py",
    ]:
        source = path.read_text()
        assert "subprocess" not in source
        assert "require(" not in source
        assert "dagua.eval.competitors" not in source
        assert "cytoscape_competitor" not in source
        assert "layout_with_variant" not in source
