"""Named layout pipelines built from composable ops."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any, Optional

import torch

PipelineSpec = tuple[str, str]

PIPELINE_REGISTRY: dict[str, PipelineSpec] = {
    "arf": ("dagua.layout.ops.pipelines.arf", "layout_arf_pipeline"),
    "arc": ("dagua.layout.ops.pipelines.arc", "layout_arc_pipeline"),
    "bfs": ("dagua.layout.ops.pipelines.bfs", "layout_bfs_pipeline"),
    "bipartite": (
        "dagua.layout.ops.pipelines.bipartite",
        "layout_bipartite_pipeline",
    ),
    "classical_mds": (
        "dagua.layout.ops.pipelines.classical_mds",
        "layout_classical_mds_pipeline",
    ),
    "circular": ("dagua.layout.ops.pipelines.circular", "layout_circular_pipeline"),
    "circlepack": ("dagua.layout.ops.pipelines.circlepack", "layout_circlepack_pipeline"),
    "circo": ("dagua.layout.ops.pipelines.circo", "layout_circo_pipeline"),
    "avsdf": ("dagua.layout.ops.pipelines.avsdf", "layout_avsdf_pipeline"),
    "concentric": ("dagua.layout.ops.pipelines.concentric", "layout_concentric_pipeline"),
    "cise": ("dagua.layout.ops.pipelines.cise", "layout_cise_pipeline"),
    "cose": ("dagua.layout.ops.pipelines.cose", "layout_cose_pipeline"),
    "cose_bilkent": (
        "dagua.layout.ops.pipelines.cose_bilkent",
        "layout_cose_bilkent_pipeline",
    ),
    "davidson_harel": (
        "dagua.layout.ops.pipelines.davidson_harel",
        "layout_davidson_harel_pipeline",
    ),
    "drl": ("dagua.layout.ops.pipelines.drl", "layout_drl_pipeline"),
    "dagre": ("dagua.layout.ops.pipelines.dagre", "layout_dagre_pipeline"),
    "d3dag": ("dagua.layout.ops.pipelines.d3dag", "layout_d3dag_pipeline"),
    "d3dag_simplex": (
        "dagua.layout.ops.pipelines.d3dag",
        "layout_d3dag_simplex_pipeline",
    ),
    "d3dag_longestpath": (
        "dagua.layout.ops.pipelines.d3dag",
        "layout_d3dag_longestpath_pipeline",
    ),
    "d3dag_opt": ("dagua.layout.ops.pipelines.d3dag", "layout_d3dag_opt_pipeline"),
    "d3dag_greedy": (
        "dagua.layout.ops.pipelines.d3dag",
        "layout_d3dag_greedy_pipeline",
    ),
    "d3force": ("dagua.layout.ops.pipelines.d3force", "layout_d3force_pipeline"),
    "d3force_default": (
        "dagua.layout.ops.pipelines.d3force",
        "layout_d3force_default_pipeline",
    ),
    "d3force_strong_repulsion": (
        "dagua.layout.ops.pipelines.d3force",
        "layout_d3force_strong_repulsion_pipeline",
    ),
    "dagua_flat": (
        "dagua.layout.ops.pipelines.dagua_flat",
        "layout_dagua_flat_pipeline",
    ),
    "dagua_native": (
        "dagua.layout.ops.pipelines.dagua_native",
        "layout_dagua_native_pipeline",
    ),
    "dot": ("dagua.layout.ops.pipelines.dot", "layout_dot_pipeline"),
    "fa2": ("dagua.layout.ops.pipelines.fa2", "layout_fa2_pipeline"),
    "fdp": ("dagua.layout.ops.pipelines.fdp", "layout_fdp_pipeline"),
    "fmmm": ("dagua.layout.ops.pipelines.fmmm", "layout_fmmm_pipeline"),
    "fcose": ("dagua.layout.ops.pipelines.fcose", "layout_fcose_pipeline"),
    "fr": ("dagua.layout.ops.pipelines.fr", "layout_fr_pipeline"),
    "gem": ("dagua.layout.ops.pipelines.gem", "layout_gem_pipeline"),
    "graphopt": ("dagua.layout.ops.pipelines.graphopt", "layout_graphopt_pipeline"),
    "hde": ("dagua.layout.ops.pipelines.hde", "layout_hde_pipeline"),
    "kk": ("dagua.layout.ops.pipelines.kk", "layout_kk_pipeline"),
    "lgl": ("dagua.layout.ops.pipelines.lgl", "layout_lgl_pipeline"),
    "linlog": ("dagua.layout.ops.pipelines.linlog", "layout_linlog_pipeline"),
    "maxent_stress": (
        "dagua.layout.ops.pipelines.maxent_stress",
        "layout_maxent_stress_pipeline",
    ),
    "multipartite": (
        "dagua.layout.ops.pipelines.multipartite",
        "layout_multipartite_pipeline",
    ),
    "native_force_directed": (
        "dagua.layout.ops.pipelines.native_force_directed",
        "layout_native_force_directed_pipeline",
    ),
    "native_hybrid": (
        "dagua.layout.ops.pipelines.native_hybrid",
        "layout_native_hybrid_pipeline",
    ),
    "native_hybrid_v2": (
        "dagua.layout.ops.pipelines.native_hybrid_v2",
        "layout_native_hybrid_v2_pipeline",
    ),
    "native_layered_dag": (
        "dagua.layout.ops.pipelines.native_layered_dag",
        "layout_native_layered_dag_pipeline",
    ),
    "native_planar": (
        "dagua.layout.ops.pipelines.native_planar",
        "layout_native_planar_pipeline",
    ),
    "native_stress": (
        "dagua.layout.ops.pipelines.native_stress",
        "layout_native_stress_pipeline",
    ),
    "native_stress_ml": (
        "dagua.layout.ops.pipelines.native_stress_ml",
        "layout_native_stress_ml_pipeline",
    ),
    "native_tree": (
        "dagua.layout.ops.pipelines.native_tree",
        "layout_native_tree_pipeline",
    ),
    "neato": ("dagua.layout.ops.pipelines.neato", "layout_neato_pipeline"),
    "neulay": ("dagua.layout.ops.pipelines.neulay", "layout_neulay_pipeline"),
    "osage": ("dagua.layout.ops.pipelines.osage", "layout_osage_pipeline"),
    "planar": ("dagua.layout.ops.pipelines.planar", "layout_planar_pipeline"),
    "pivot_mds": (
        "dagua.layout.ops.pipelines.pivot_mds",
        "layout_pivot_mds_pipeline",
    ),
    "reingold_tilford": (
        "dagua.layout.ops.pipelines.reingold_tilford",
        "layout_reingold_tilford_pipeline",
    ),
    "radial_tree": ("dagua.layout.ops.pipelines.radial_tree", "layout_radial_tree_pipeline"),
    "sfdp": ("dagua.layout.ops.pipelines.sfdp", "layout_sfdp_pipeline"),
    "sgd2_multi": (
        "dagua.layout.ops.pipelines.sgd2_multi",
        "layout_sgd2_multi_pipeline",
    ),
    "shell": ("dagua.layout.ops.pipelines.shell", "layout_shell_pipeline"),
    "spectral": ("dagua.layout.ops.pipelines.spectral", "layout_spectral_pipeline"),
    "spiral": ("dagua.layout.ops.pipelines.spiral", "layout_spiral_pipeline"),
    "star": ("dagua.layout.ops.pipelines.star", "layout_star_pipeline"),
    "stress_majorization": (
        "dagua.layout.ops.pipelines.stress_majorization",
        "layout_stress_majorization_pipeline",
    ),
    "smacof_nonmetric": (
        "dagua.layout.ops.pipelines.smacof_nonmetric",
        "layout_smacof_nonmetric_pipeline",
    ),
    "stress_sgd": (
        "dagua.layout.ops.pipelines.stress_sgd",
        "layout_stress_sgd_pipeline",
    ),
    "sugiyama": ("dagua.layout.ops.pipelines.sugiyama", "layout_sugiyama_pipeline"),
    "tsne": ("dagua.layout.ops.pipelines.tsne_graph", "layout_tsne_pipeline"),
    "tsne_graph": (
        "dagua.layout.ops.pipelines.tsne_graph",
        "layout_tsne_graph_pipeline",
    ),
    "tsnet": ("dagua.layout.ops.pipelines.tsnet", "layout_tsnet_pipeline"),
    "tutte": ("dagua.layout.ops.pipelines.tutte", "layout_tutte_pipeline"),
    "twopi": ("dagua.layout.ops.pipelines.twopi", "layout_twopi_pipeline"),
    "umap": ("dagua.layout.ops.pipelines.umap_layout", "layout_umap_layout_pipeline"),
    "webcola": ("dagua.layout.ops.pipelines.webcola", "layout_webcola_pipeline"),
    "webcola_constrained": (
        "dagua.layout.ops.pipelines.webcola",
        "layout_webcola_constrained_pipeline",
    ),
    "yifanhu": ("dagua.layout.ops.pipelines.yifanhu", "layout_yifanhu_pipeline"),
}


def resolve_fidelity_dtype(
    fidelity_mode: Any,
    fidelity_dtype: Optional[torch.dtype],
) -> torch.dtype:
    """Resolve fidelity-mode internal dtype.

    Parameters
    ----------
    fidelity_mode : Any
        Pipeline-specific fidelity selector.
    fidelity_dtype : torch.dtype or None
        Explicit dtype requested by the caller. ``None`` defaults to
        ``torch.float64`` when fidelity mode is truthy and ``torch.float32``
        otherwise.

    Returns
    -------
    torch.dtype
        ``torch.float32`` or ``torch.float64``.

    Raises
    ------
    ValueError
        If an unsupported dtype is requested.
    """
    resolved = torch.float64 if fidelity_dtype is None and bool(fidelity_mode) else fidelity_dtype
    if resolved is None:
        resolved = torch.float32
    if resolved not in (torch.float32, torch.float64):
        raise ValueError("fidelity_dtype must be torch.float32 or torch.float64.")
    return resolved


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


__all__ = ["PIPELINE_REGISTRY", "get_pipeline_function", "resolve_fidelity_dtype"]
