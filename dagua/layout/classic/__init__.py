"""Classic educational graph layout algorithms."""

from __future__ import annotations

from typing import Any

__all__ = [
    "layout_drl",
    "layout_fa2",
    "layout_fr",
    "layout_graphopt",
    "layout_kk",
    "layout_lgl",
    "layout_neulay",
    "layout_sgd2_multi",
    "layout_sfdp",
    "layout_stress_sgd",
    "layout_sugiyama",
    "layout_umap",
]


def __getattr__(name: str) -> Any:
    """Resolve classic layout exports lazily.

    Parameters
    ----------
    name : str
        Requested attribute name.

    Returns
    -------
    Any
        The requested exported layout function.

    Raises
    ------
    AttributeError
        If ``name`` is not a known export.
    """
    if name == "layout_fa2":
        from dagua.layout.classic.fa2 import layout_fa2

        return layout_fa2
    if name == "layout_drl":
        from dagua.layout.classic.drl import layout_drl

        return layout_drl
    if name == "layout_fr":
        from dagua.layout.classic.fr import layout_fr

        return layout_fr
    if name == "layout_graphopt":
        from dagua.layout.classic.graphopt import layout_graphopt

        return layout_graphopt
    if name == "layout_kk":
        from dagua.layout.classic.kk import layout_kk

        return layout_kk
    if name == "layout_lgl":
        from dagua.layout.classic.lgl import layout_lgl

        return layout_lgl
    if name == "layout_neulay":
        from dagua.layout.classic.neulay import layout_neulay

        return layout_neulay
    if name == "layout_sgd2_multi":
        from dagua.layout.classic.sgd2_multi import layout_sgd2_multi

        return layout_sgd2_multi
    if name == "layout_sfdp":
        from dagua.layout.classic.sfdp import layout_sfdp

        return layout_sfdp
    if name == "layout_stress_sgd":
        from dagua.layout.classic.stress_sgd import layout_stress_sgd

        return layout_stress_sgd
    if name == "layout_sugiyama":
        from dagua.layout.classic.sugiyama import layout_sugiyama

        return layout_sugiyama
    if name == "layout_umap":
        from dagua.layout.classic.umap_layout import layout_umap

        return layout_umap
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
