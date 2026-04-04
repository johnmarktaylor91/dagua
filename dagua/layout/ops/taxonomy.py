"""Operation taxonomy: categories, registry, and discovery.

The taxonomy defines the vocabulary of layout operations. Each operation
belongs to exactly one ``OpCategory``. The registry maps operation names
to their classes for programmatic discovery and pipeline construction.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Type

if TYPE_CHECKING:
    from dagua.layout.ops.base import Op


class OpCategory(Enum):
    """Categories for layout operations.

    Every Op declares exactly one category via its ``category`` class
    attribute. Categories group related operations and enable filtering
    in the registry.

    Members
    -------
    INIT
        Position initialization (random, spectral, MDS, circular, etc.).
    PREPROCESS
        Graph structure preparation (cycle detection, classification,
        adjacency building, component detection).
    DISTANCE
        Graph-theoretic distance computation (BFS, Dijkstra, all-pairs,
        pivot approximation).
    LAYERING
        Layer and rank assignment for hierarchical layouts (longest-path,
        promotion, dummy nodes, layer index building).
    ORDERING
        Within-layer node ordering (barycenter sweep, median, transpose
        heuristic, spectral ordering).
    COORDINATE
        Coordinate assignment from layer ordering (Brandes-Kopf 4-pass,
        Buchheim-Walker tree layout).
    COARSEN
        Multilevel hierarchy building (heavy-edge matching, solar-system,
        layer-aware coarsening, streaming coarsening).
    PROLONG
        Multilevel prolongation and refinement setup (direct mapping,
        lambda interpolation, neighbor smoothing).
    FORCE
        Per-step force computation for non-differentiable force-directed
        layouts (Coulomb repulsion, spring attraction, gravity, Barnes-Hut).
        Force ops accumulate into ``SolveState.forces``.
    LOSS
        Differentiable loss evaluation for gradient-based layouts.
        Loss ops return a scalar tensor via ``LossOp.evaluate()``.
    EMBED
        Spectral and dimensionality-reduction operations (eigendecomposition,
        SVD, perplexity matching, UMAP fuzzy sets, GCN forward).
    OPTIMIZE
        Parameter update mechanics (Adam, SGD, L-BFGS-B, gradient clipping,
        optimizer creation).
    PROJECT
        Hard constraint enforcement -- non-differentiable projections
        (overlap removal, hard pins, boundary clamping, movement clamping,
        monotone safeguard).
    ANNEAL
        Weight, temperature, and schedule evolution (linear cooling,
        exponential cooling, per-node temperature, phase schedules,
        learning rate decay, early exaggeration).
    CONTEXT
        Per-step shared state building (edge batch context, sampled node
        context, quadtree, density grid, KDTree pair refresh).
    CONVERGE
        Convergence and stopping criteria (fixed steps, displacement
        threshold, temperature threshold, stall count, LR threshold).
    POSTPROCESS
        Post-layout transforms (centering, scaling, normalization,
        direction rotation, dummy node removal).
    EDGE_ROUTE
        Post-layout edge curve optimization (Bezier control points,
        edge route reconstruction from dummy nodes).
    UTILITY
        Infrastructure operations (checkpoint, disk offload/reload,
        garbage collection, VRAM guard, progress reporting, snapshots).
    CONTROL
        Composition and control flow (Pipeline, Repeat, Conditional,
        LossGroup, MultilevelVCycle, EarlyBreak).
    UNKNOWN
        Default for ops that have not declared a category.
    """

    INIT = "init"
    PREPROCESS = "preprocess"
    DISTANCE = "distance"
    LAYERING = "layering"
    ORDERING = "ordering"
    COORDINATE = "coordinate"
    COARSEN = "coarsen"
    PROLONG = "prolong"
    FORCE = "force"
    LOSS = "loss"
    EMBED = "embed"
    OPTIMIZE = "optimize"
    PROJECT = "project"
    ANNEAL = "anneal"
    CONTEXT = "context"
    CONVERGE = "converge"
    POSTPROCESS = "postprocess"
    EDGE_ROUTE = "edge_route"
    UTILITY = "utility"
    CONTROL = "control"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Operation registry
# ---------------------------------------------------------------------------

_OP_REGISTRY: Dict[str, Type["Op"]] = {}
OP_REGISTRY: Dict[str, Type["Op"]] = _OP_REGISTRY


def register_op(cls: Type["Op"]) -> Type["Op"]:
    """Class decorator that registers an ``Op`` by its ``name`` attribute.

    Raises ``ValueError`` on name collision to prevent silent overwrites
    from user-defined ops or plugin conflicts.

    Parameters
    ----------
    cls : type[Op]
        The operation class to register.

    Returns
    -------
    type[Op]
        The same class, unmodified.

    Raises
    ------
    ValueError
        If an op with the same name is already registered.
    """
    name = getattr(cls, "name", None)
    if name is None or name == "unnamed_op":
        return cls
    if name in _OP_REGISTRY:
        existing = _OP_REGISTRY[name]
        if existing is not cls:
            raise ValueError(
                f"Op name collision: '{name}' is already registered by "
                f"{existing.__module__}.{existing.__qualname__}. "
                f"Cannot register {cls.__module__}.{cls.__qualname__}."
            )
    _OP_REGISTRY[name] = cls
    return cls


def unregister_op(name: str) -> None:
    """Remove an op from the registry.

    Useful in tests and plugin unload scenarios.

    Parameters
    ----------
    name : str
        The operation name to remove.

    Raises
    ------
    KeyError
        If the name is not registered.
    """
    del _OP_REGISTRY[name]


def get_op_class(name: str) -> Type["Op"]:
    """Look up a registered ``Op`` class by name.

    Parameters
    ----------
    name : str
        Registered operation name.

    Returns
    -------
    type[Op]
        The corresponding operation class.

    Raises
    ------
    KeyError
        If the name is not in the registry.
    """
    return _OP_REGISTRY[name]


def list_ops(category: Optional[OpCategory] = None) -> List[str]:
    """List registered operation names, optionally filtered by category.

    Parameters
    ----------
    category : OpCategory or None
        If provided, only ops with this category are returned.

    Returns
    -------
    list[str]
        Sorted list of registered operation names.
    """
    if category is None:
        return sorted(_OP_REGISTRY)
    return sorted(
        name for name, cls in _OP_REGISTRY.items() if getattr(cls, "category", None) == category
    )


def list_categories() -> List[OpCategory]:
    """Return all ``OpCategory`` values.

    Returns
    -------
    list[OpCategory]
        Every member of the OpCategory enum.
    """
    return list(OpCategory)
