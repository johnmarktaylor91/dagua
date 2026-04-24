"""Tree native sub-pipeline based on Reingold-Tilford placement."""

from __future__ import annotations

from dagua.config import LayoutConfig
from dagua.layout.ops.base import Pipeline
from dagua.layout.ops.coordinate import ReingoldTilfordTree, ReingoldTilfordTreeConfig
from dagua.layout.ops.postprocess import AspectRatioFit, AspectRatioFitConfig


def build_native_tree_pipeline(config: LayoutConfig) -> Pipeline:
    """Build the native tree fast-path pipeline.

    Parameters
    ----------
    config : LayoutConfig
        Prepared native layout configuration.

    Returns
    -------
    Pipeline
        Reingold-Tilford tree placement followed by aspect-ratio fitting.
    """
    return Pipeline(
        [
            ReingoldTilfordTree(ReingoldTilfordTreeConfig()),
            AspectRatioFit(
                AspectRatioFitConfig(
                    target_aspect=getattr(config, "_dagua_native_target_aspect", None),
                )
            ),
        ],
        name="native_tree_pipeline",
    )


__all__ = ["build_native_tree_pipeline"]
