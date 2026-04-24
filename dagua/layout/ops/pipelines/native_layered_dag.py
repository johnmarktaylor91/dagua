"""Layered-DAG native sub-pipeline."""

from __future__ import annotations

import copy

from dagua.config import LayoutConfig
from dagua.layout.ops.base import Pipeline
from dagua.layout.ops.pipelines import dagua_native_legacy


def build_native_layered_dag_pipeline(config: LayoutConfig) -> Pipeline:
    """Build the native layered-DAG pipeline.

    Parameters
    ----------
    config : LayoutConfig
        Prepared native configuration with ``_dagua_native_*`` metadata.

    Returns
    -------
    Pipeline
        Pipeline using the sprint-19 layered DAG polish stack.
    """
    layered_config = copy.copy(config)
    layered_config.insert_dummy_nodes = bool(getattr(config, "insert_dummy_nodes", True))
    layered_config.use_native_median_transpose = bool(
        getattr(config, "use_native_median_transpose", True)
    )
    layered_config.brandes_koepf_refine = bool(getattr(config, "brandes_koepf_refine", True))
    return dagua_native_legacy.build_dagua_pipeline(layered_config)


__all__ = ["build_native_layered_dag_pipeline"]
