"""Regression tests for maxent-stress fidelity wiring."""

from __future__ import annotations

import importlib
import json
import subprocess
from typing import Any

import torch

from dagua.eval.competitors import ogdf_competitor
from dagua.eval.competitors.classic_competitor import ClassicMaxentStress
from dagua.eval.competitors.ogdf_competitor import OGDFStress
from dagua.eval.variants import VARIANT_REGISTRY
from dagua.graph import DaguaGraph
from dagua.layout.ops.maxent_stress import MaxentInitializePositions, MaxentPrepareState
from dagua.layout.ops.pipelines.maxent_stress import build_maxent_stress_pipeline
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState


def _pipeline_step_names(steps: int, use_entropy: bool) -> list[str]:
    """Return op names for one maxent-stress dispatch pipeline.

    Parameters
    ----------
    steps : int
        Number of requested optimization steps.
    use_entropy : bool
        Whether the pipeline should include entropy repulsion.

    Returns
    -------
    list[str]
        Ordered operation names from the dispatched pipeline.
    """
    pipeline = build_maxent_stress_pipeline(
        steps=steps,
        use_entropy=use_entropy,
        use_majorization=True,
        num_nodes=8,
    )
    names: list[str] = []
    for op in pipeline.ops:
        names.append(op.name)
        inner = getattr(op, "inner", None)
        child_ops = getattr(inner, "ops", ())
        for child_op in child_ops:
            names.append(child_op.name)
    return names


def test_maxent_non_entropy_step_variants_use_majorization() -> None:
    """Non-entropy step variants should route through OGDF stress majorization."""
    for steps in (50, 400):
        op_names = _pipeline_step_names(steps=steps, use_entropy=False)

        assert "sm_ogdf_prepare_state" in op_names
        assert "sm_ogdf_initialize_positions" in op_names
        assert "sm_smacof_step" in op_names
        assert "maxent_gradient_step" not in op_names


def test_maxent_entropy_variant_uses_ogdf_stress_fidelity_branch() -> None:
    """Entropy variants paired to OGDF stress should use the fidelity branch."""
    op_names = _pipeline_step_names(steps=200, use_entropy=True)

    assert "sm_ogdf_prepare_state" in op_names
    assert "sm_ogdf_initialize_positions" in op_names
    assert "sm_smacof_step" in op_names
    assert "maxent_gradient_step" not in op_names


def test_maxent_majorization_distances_stay_float64() -> None:
    """Majorization graph distances should not round through float32."""
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_weights = torch.tensor([0.1, 0.2], dtype=torch.float64)
    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=3,
        edge_weights=edge_weights,
        seed=42,
    )

    state = MaxentPrepareState(for_majorization=True).apply(
        problem,
        SolveState(),
        RuntimeContext(),
    )

    assert state.distance_matrix is not None
    assert state.distance_matrix.dtype == torch.float64


def test_maxent_majorization_unweighted_distances_use_ogdf_edge_cost() -> None:
    """Unweighted majorization distances should match OGDF's edge cost scale."""
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    problem = LayoutProblem(edge_index=edge_index, num_nodes=3, seed=42)

    state = MaxentPrepareState(for_majorization=True).apply(
        problem,
        SolveState(),
        RuntimeContext(),
    )

    assert state.distance_matrix is not None
    assert float(state.distance_matrix[0, 1].item()) == 100.0
    assert float(state.distance_matrix[0, 2].item()) == 200.0


def test_maxent_majorization_init_uses_runner_glibc_rand() -> None:
    """Majorization warm start should match the OGDF runner-owned layout."""
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    problem = LayoutProblem(edge_index=edge_index, num_nodes=4, seed=123)

    state = MaxentInitializePositions(for_majorization=True).apply(
        problem,
        SolveState(),
        RuntimeContext(),
    )

    assert state.pos is not None
    expected = torch.tensor(
        [[39.3, 1.3], [87.3, 63.0], [27.9, 33.1], [19.5, 2.2]],
        dtype=torch.float64,
    )
    assert torch.allclose(state.pos, expected)


def test_maxent_step_variants_forward_ogdf_iterations() -> None:
    """Maxent step variants should align the OGDF stress reference budget."""
    original_params = {
        variant.variant_id: variant.original_params
        for variant in VARIANT_REGISTRY
        if variant.variant_id.startswith("classic_maxent_stress_steps")
    }

    assert original_params["classic_maxent_stress_steps50"] == {"iterations": 50}
    assert original_params["classic_maxent_stress_steps400"] == {"iterations": 400}


def test_ogdf_stress_variant_iterations_enter_runner_payload(monkeypatch: Any) -> None:
    """The OGDF stress adapter should pass variant iterations to the runner."""
    captured_payload: dict[str, Any] = {}

    def fake_resolve_runner() -> str:
        """Return a dummy runner path for the subprocess call.

        Returns
        -------
        str
            Placeholder executable path.
        """
        return "/tmp/ogdf_runner"

    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        """Capture the runner payload and return one valid coordinate row.

        Parameters
        ----------
        *args : Any
            Positional arguments forwarded to ``subprocess.run``.
        **kwargs : Any
            Keyword arguments forwarded to ``subprocess.run``.

        Returns
        -------
        subprocess.CompletedProcess[str]
            Successful subprocess result with one node position.
        """
        del args
        captured_payload.update(json.loads(str(kwargs["input"])))
        return subprocess.CompletedProcess(
            args=["/tmp/ogdf_runner"],
            returncode=0,
            stdout='{"positions":[[0.0,0.0]]}',
            stderr="",
        )

    monkeypatch.setattr(ogdf_competitor, "_resolve_ogdf_runner", fake_resolve_runner)
    monkeypatch.setattr(ogdf_competitor.subprocess, "run", fake_run)

    graph = DaguaGraph.from_edge_index(torch.empty((2, 0), dtype=torch.long), num_nodes=1)
    result = OGDFStress().layout_with_variant(
        graph,
        variant_params={"iterations": 50},
    )

    assert result.error is None
    assert captured_payload["algorithm"] == "stress"
    assert captured_payload["iterations"] == 50


def test_direct_classic_maxent_wrapper_forwards_edge_weights(monkeypatch: Any) -> None:
    """The registered direct wrapper should not drop weighted stress inputs."""
    captured_kwargs: dict[str, Any] = {}

    def fake_layout_maxent_stress(*args: Any, **kwargs: Any) -> torch.Tensor:
        """Capture layout kwargs and return valid coordinates.

        Parameters
        ----------
        *args : Any
            Positional arguments forwarded to the layout function.
        **kwargs : Any
            Keyword arguments forwarded to the layout function.

        Returns
        -------
        torch.Tensor
            Dummy coordinates with shape ``[2, 2]``.
        """
        del args
        captured_kwargs.update(kwargs)
        return torch.zeros((2, 2), dtype=torch.float32)

    maxent_pipeline = importlib.import_module("dagua.layout.ops.pipelines.maxent_stress")
    monkeypatch.setattr(maxent_pipeline, "layout_maxent_stress_pipeline", fake_layout_maxent_stress)

    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    edge_weights = torch.tensor([3.0], dtype=torch.float32)
    graph = DaguaGraph.from_edge_index(edge_index, num_nodes=2, edge_weights=edge_weights)
    result = ClassicMaxentStress().layout(graph)

    assert result.error is None
    assert captured_kwargs["edge_weights"] is edge_weights
