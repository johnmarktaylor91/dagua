"""Shared helpers for topology-dispatched native sub-pipelines."""

from __future__ import annotations

from typing import Optional

import torch

from dagua.layout.ops.pipelines import dagua_native_legacy
from dagua.layout.ops.pipelines.dagua_native_legacy import (
    _build_coarse_init_pipeline_factory,
    _build_refine_pipeline_factory,
    _prepare_native_config,
    _should_apply_brandes_koepf_refine,
    _should_decompose_components,
    _should_use_native_dummy_nodes,
    _should_use_native_median_transpose,
    _stress_pivot_prep,
    _tile_component_positions,
    build_gradient_core,
)
from dagua.layout.ops.state import LayoutProblem, SolveState


def _extract_component_problem(
    parent_problem: LayoutProblem,
    parent_state: SolveState,
    component_nodes: torch.Tensor,
    layer_assignments: Optional[torch.Tensor] = None,
) -> tuple[LayoutProblem, SolveState, torch.Tensor, Optional[torch.Tensor]]:
    """Build one relabeled child problem for a weak component.

    Parameters
    ----------
    parent_problem : LayoutProblem
        Prepared parent graph problem.
    parent_state : SolveState
        Parent solve state to project into the component.
    component_nodes : torch.Tensor
        Parent node ids in the component with shape ``[K]``.
    layer_assignments : torch.Tensor, optional
        Optional parent layer assignments with shape ``[N_parent]``.

    Returns
    -------
    tuple[LayoutProblem, SolveState, torch.Tensor, torch.Tensor | None]
        Child problem, child state, parent node indices, and child layer
        assignments when supplied.
    """
    return dagua_native_legacy._extract_component_problem(
        parent_problem,
        parent_state,
        component_nodes,
        layer_assignments,
    )


__all__ = [
    "_build_coarse_init_pipeline_factory",
    "_build_refine_pipeline_factory",
    "_extract_component_problem",
    "_prepare_native_config",
    "_should_apply_brandes_koepf_refine",
    "_should_decompose_components",
    "_should_use_native_dummy_nodes",
    "_should_use_native_median_transpose",
    "_stress_pivot_prep",
    "_tile_component_positions",
    "build_gradient_core",
]
