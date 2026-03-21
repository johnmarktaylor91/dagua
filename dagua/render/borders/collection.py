"""Batched rendering helpers for node and cluster fill/border paths."""

from __future__ import annotations

from typing import Any, List, Sequence

from matplotlib.collections import PatchCollection
from matplotlib.patches import PathPatch
from matplotlib.path import Path


def make_clip_proxy(path: Path, transform: Any) -> PathPatch:
    """Build an unattached fill proxy patch for clipping.

    Parameters
    ----------
    path : matplotlib.path.Path
        Fill path in data coordinates.
    transform : Any
        Axes data transform.

    Returns
    -------
    matplotlib.patches.PathPatch
        Unattached proxy patch that can be used with ``set_clip_path``.
    """

    return PathPatch(path, facecolor="none", edgecolor="none", linewidth=0.0, transform=transform)


def add_filled_collections(
    ax: Any,
    fill_paths: Sequence[Path],
    fill_colors: Sequence[Any],
    border_paths: Sequence[Path],
    border_colors: Sequence[Any],
    fill_zorder: float,
    border_zorder: float,
) -> List[Any]:
    """Add batched fill and border collections to the axes.

    Parameters
    ----------
    ax : Any
        Matplotlib axes.
    fill_paths : sequence[matplotlib.path.Path]
        Filled interior paths.
    fill_colors : sequence[Any]
        Face colors for ``fill_paths``.
    border_paths : sequence[matplotlib.path.Path]
        Filled border-ring or dash paths.
    border_colors : sequence[Any]
        Face colors for ``border_paths``.
    fill_zorder : float
        Collection z-order for fills.
    border_zorder : float
        Collection z-order for borders.

    Returns
    -------
    list[Any]
        Added matplotlib collection artists.
    """

    artists: List[Any] = []
    if fill_paths:
        fill_collection = PatchCollection(
            [PathPatch(path) for path in fill_paths],
            match_original=False,
            facecolors=list(fill_colors),
            edgecolors="none",
            linewidths=0.0,
            zorder=fill_zorder,
        )
        ax.add_collection(fill_collection)
        artists.append(fill_collection)
    if border_paths:
        border_collection = PatchCollection(
            [PathPatch(path) for path in border_paths],
            match_original=False,
            facecolors=list(border_colors),
            edgecolors="none",
            linewidths=0.0,
            zorder=border_zorder,
        )
        ax.add_collection(border_collection)
        artists.append(border_collection)
    return artists
