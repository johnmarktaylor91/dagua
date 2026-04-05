"""Named layout pipelines built from composable ops."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module

import torch

PipelineSpec = tuple[str, str]

PIPELINE_REGISTRY: dict[str, PipelineSpec] = {
    "classical_mds": (
        "dagua.layout.ops.pipelines.classical_mds",
        "layout_classical_mds_pipeline",
    ),
    "davidson_harel": (
        "dagua.layout.ops.pipelines.davidson_harel",
        "layout_davidson_harel_pipeline",
    ),
    "drl": ("dagua.layout.ops.pipelines.drl", "layout_drl_pipeline"),
    "dagua_native": (
        "dagua.layout.ops.pipelines.dagua_native",
        "layout_dagua_native_pipeline",
    ),
    "fa2": ("dagua.layout.ops.pipelines.fa2", "layout_fa2_pipeline"),
    "fmmm": ("dagua.layout.ops.pipelines.fmmm", "layout_fmmm_pipeline"),
    "fr": ("dagua.layout.ops.pipelines.fr", "layout_fr_pipeline"),
    "gem": ("dagua.layout.ops.pipelines.gem", "layout_gem_pipeline"),
    "graphopt": ("dagua.layout.ops.pipelines.graphopt", "layout_graphopt_pipeline"),
    "kk": ("dagua.layout.ops.pipelines.kk", "layout_kk_pipeline"),
    "lgl": ("dagua.layout.ops.pipelines.lgl", "layout_lgl_pipeline"),
    "linlog": ("dagua.layout.ops.pipelines.linlog", "layout_linlog_pipeline"),
    "maxent_stress": (
        "dagua.layout.ops.pipelines.maxent_stress",
        "layout_maxent_stress_pipeline",
    ),
    "neulay": ("dagua.layout.ops.pipelines.neulay", "layout_neulay_pipeline"),
    "pivot_mds": (
        "dagua.layout.ops.pipelines.pivot_mds",
        "layout_pivot_mds_pipeline",
    ),
    "reingold_tilford": (
        "dagua.layout.ops.pipelines.reingold_tilford",
        "layout_reingold_tilford_pipeline",
    ),
    "sfdp": ("dagua.layout.ops.pipelines.sfdp", "layout_sfdp_pipeline"),
    "sgd2_multi": (
        "dagua.layout.ops.pipelines.sgd2_multi",
        "layout_sgd2_multi_pipeline",
    ),
    "spectral": ("dagua.layout.ops.pipelines.spectral", "layout_spectral_pipeline"),
    "stress_majorization": (
        "dagua.layout.ops.pipelines.stress_majorization",
        "layout_stress_majorization_pipeline",
    ),
    "stress_sgd": (
        "dagua.layout.ops.pipelines.stress_sgd",
        "layout_stress_sgd_pipeline",
    ),
    "sugiyama": ("dagua.layout.ops.pipelines.sugiyama", "layout_sugiyama_pipeline"),
    "tsnet": ("dagua.layout.ops.pipelines.tsnet", "layout_tsnet_pipeline"),
    "umap": ("dagua.layout.ops.pipelines.umap_layout", "layout_umap_layout_pipeline"),
}


def get_pipeline_function(name: str) -> Callable[..., torch.Tensor]:
    """Resolve a pipeline name to a callable layout function.

    Parameters
    ----------
    name : str
        Short algorithm name, for example ``"fr"`` or ``"umap"``.

    Returns
    -------
    Callable[..., torch.Tensor]
        The pipeline entrypoint function for the requested algorithm.

    Raises
    ------
    KeyError
        If the requested algorithm is not registered.
    AttributeError
        If the registered module does not define the expected function.
    ImportError
        If the implementation module cannot be imported.
    """
    normalized_name = name.lower()
    module_name, function_name = PIPELINE_REGISTRY[normalized_name]
    module = import_module(module_name)
    return getattr(module, function_name)


__all__ = ["PIPELINE_REGISTRY", "get_pipeline_function"]
