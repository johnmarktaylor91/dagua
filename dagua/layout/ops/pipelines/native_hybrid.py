"""Hybrid native sub-pipeline for mostly hierarchical cyclic graphs."""

from __future__ import annotations

import copy

from dagua.config import LayoutConfig
from dagua.layout.ops.base import Pipeline
from dagua.layout.ops.pipelines import dagua_native_legacy


def build_native_hybrid_pipeline(config: LayoutConfig) -> Pipeline:
    """Build the native hybrid pipeline.

    Parameters
    ----------
    config : LayoutConfig
        Prepared native configuration with ``_dagua_native_*`` metadata.

    Returns
    -------
    Pipeline
        DAG-skeleton-style native pipeline with back-edge compactness enabled
        and strict DAG-only dummy/BK stages disabled.
    """
    hybrid_config = copy.copy(config)
    hybrid_config.insert_dummy_nodes = False
    hybrid_config.brandes_koepf_refine = False
    hybrid_config.use_native_median_transpose = bool(
        getattr(config, "use_native_median_transpose", True)
    )
    hybrid_config.w_back_edge = max(float(getattr(config, "w_back_edge", 0.0)), 0.3)
    hybrid_config.w_stress = max(float(getattr(config, "w_stress", 0.0)), 0.05)
    return dagua_native_legacy.build_dagua_pipeline(hybrid_config)


__all__ = ["build_native_hybrid_pipeline"]
