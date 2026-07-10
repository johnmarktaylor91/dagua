"""Size-awareness toggle for external layout-engine adapters (r80-P6).

External benchmark competitors (Graphviz dot/sfdp/neato, ELK layered, dagre)
were historically laid out "size-blind": every node was submitted to the
external engine as a fixed placeholder box (or with no size at all), while
dagua's own composite score always penalizes overlaps against the graph's
REAL label-measured node boxes (``graph.node_sizes``). That mismatch is a
systematic scoring bias FOR dagua -- an external engine can rack up overlap
penalties purely because it was never told how big the nodes actually are,
not because its placement is worse.

This module holds a small process-global toggle that size-capable adapters
consult before deciding whether to pass real per-node width/height through
to the underlying engine. The default is size-aware (the honest, apples-to-
apples comparison). ``--size-blind-externals`` on the r79 baseline CLI flips
it back to the old behavior, kept ONLY for store-compatibility experiments
against pre-r80-P6 frozen data (which was produced size-blind throughout).

Adapters without native size support (igraph, nx_spring, and any future
size-blind adapter) do not consult this flag at all; they stay size-blind
regardless of its value, and each such adapter documents that fact in its
own module docstring.
"""

from __future__ import annotations

_SIZE_AWARE_EXTERNALS = True


def set_size_aware_externals(enabled: bool) -> None:
    """Set whether size-capable external adapters receive real node sizes.

    Parameters
    ----------
    enabled : bool
        ``True`` (default) passes real per-node width/height through to
        size-capable adapters. ``False`` restores the old size-blind
        behavior for store-compatibility experiments.

    Returns
    -------
    None
    """
    global _SIZE_AWARE_EXTERNALS
    _SIZE_AWARE_EXTERNALS = bool(enabled)


def size_aware_externals() -> bool:
    """Return whether size-capable external adapters should use real sizes.

    Returns
    -------
    bool
        Current process-global size-awareness setting.
    """
    return _SIZE_AWARE_EXTERNALS
