"""Regression tests for maxent-stress fidelity wiring."""

from __future__ import annotations

import json
import subprocess
from typing import Any

import torch

from dagua.eval.competitors import ogdf_competitor
from dagua.eval.competitors.ogdf_competitor import OGDFStress
from dagua.eval.variants import VARIANT_REGISTRY
from dagua.graph import DaguaGraph
from dagua.layout.ops.maxent_stress import MaxentPrepareState
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
    """Non-entropy step variants should remain stress-majorization layouts."""
    for steps in (50, 400):
        op_names = _pipeline_step_names(steps=steps, use_entropy=False)

        assert "maxent_majorization_step" in op_names
        assert "maxent_gradient_step" not in op_names


def test_maxent_entropy_variant_still_uses_gradient_branch() -> None:
    """Entropy has no OGDF stress equivalent and should keep the Adam branch."""
    op_names = _pipeline_step_names(steps=200, use_entropy=True)

    assert "maxent_gradient_step" in op_names
    assert "maxent_majorization_step" not in op_names


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
