"""Classic educational graph layout algorithms."""

from __future__ import annotations

from typing import Any

__all__ = ["layout_fa2", "layout_fr", "layout_kk", "layout_stress_sgd", "layout_sugiyama"]


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
    if name == "layout_fr":
        from dagua.layout.classic.fr import layout_fr

        return layout_fr
    if name == "layout_kk":
        from dagua.layout.classic.kk import layout_kk

        return layout_kk
    if name == "layout_stress_sgd":
        from dagua.layout.classic.stress_sgd import layout_stress_sgd

        return layout_stress_sgd
    if name == "layout_sugiyama":
        from dagua.layout.classic.sugiyama import layout_sugiyama

        return layout_sugiyama
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
