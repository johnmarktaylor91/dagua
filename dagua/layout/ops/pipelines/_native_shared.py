"""Shared helpers for topology-dispatched native sub-pipelines."""

from __future__ import annotations

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

__all__ = [
    "_build_coarse_init_pipeline_factory",
    "_build_refine_pipeline_factory",
    "_prepare_native_config",
    "_should_apply_brandes_koepf_refine",
    "_should_decompose_components",
    "_should_use_native_dummy_nodes",
    "_should_use_native_median_transpose",
    "_stress_pivot_prep",
    "_tile_component_positions",
    "build_gradient_core",
]
