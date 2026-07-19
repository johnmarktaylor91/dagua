"""Public user-intent constraint API for Dagua layouts.

This module is intentionally both the public namespace and the ``C`` alias
exported from :mod:`dagua`. Constraint objects are immutable descriptions of
layout intent; graph verbs append these objects to a :class:`ConstraintSet`,
and the layout engine lowers them to tensor losses and projectors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, replace
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import torch

StrengthLike = Any
Selection = Any
LossFn = Callable[[torch.Tensor, "ConstraintContext"], torch.Tensor]
ProjectFn = Callable[[torch.Tensor, "ConstraintContext"], None]

STRENGTHS: Dict[str, float] = {
    "soft": 0.5,
    "firm": 2.0,
    "rigid": 10.0,
    "hard": float("inf"),
}


class ConstraintError(Exception):
    """Base class for constraint API errors."""


class ConstraintTypeError(ConstraintError, TypeError):
    """Raised when a constraint is not meaningful for an element or strength."""


class ConstraintReferenceError(ConstraintError, KeyError):
    """Raised when an eager selector references an unknown graph element."""


class ConstraintStagedError(ConstraintTypeError):
    """Raised when a fixed API surface has intentionally staged lowering."""


class ConstraintConflictError(ConstraintError, RuntimeError):
    """Raised when hard constraints are jointly infeasible under strict policy."""


def strength_weight(strength: StrengthLike) -> float:
    """Return the numeric weight represented by a strength value.

    Parameters
    ----------
    strength : Any
        One of ``"soft"``, ``"firm"``, ``"rigid"``, ``"hard"``, or a
        numeric value. ``math.inf`` is treated as hard.

    Returns
    -------
    float
        Numeric weight for the shared Dagua/Flex strength ladder.

    Raises
    ------
    ValueError
        If the strength name is unknown or the numeric weight is invalid.
    """
    if isinstance(strength, str):
        key = strength.lower()
        if key not in STRENGTHS:
            names = ", ".join(STRENGTHS)
            raise ValueError(f"Unknown constraint strength {strength!r}; expected one of {names}.")
        return STRENGTHS[key]
    weight = float(strength)
    if math.isnan(weight) or weight < 0.0:
        raise ValueError("Constraint strength must be non-negative and not NaN.")
    return weight


def is_hard_strength(strength: StrengthLike) -> bool:
    """Return whether ``strength`` represents an exact hard constraint.

    Parameters
    ----------
    strength : Any
        Strength value accepted by :func:`strength_weight`.

    Returns
    -------
    bool
        ``True`` when the resolved weight is infinite.
    """
    return math.isinf(strength_weight(strength))


@dataclass(frozen=True)
class Selector:
    """Lazy or typed selection expression used by constraints.

    Parameters
    ----------
    kind : str
        Selector family name such as ``"cluster"``, ``"edges"``, or
        ``"canvas"``.
    args : tuple[Any, ...]
        Positional selector payload.
    kwargs : dict[str, Any]
        Keyword selector payload.
    allow_empty : bool, default=False
        Whether lowering may accept an empty result.
    staged : bool, default=False
        Whether the selector API is fixed but lowering is staged.
    """

    kind: str
    args: Tuple[Any, ...] = ()
    kwargs: Dict[str, Any] = field(default_factory=dict)
    allow_empty: bool = False
    staged: bool = False

    def with_allow_empty(self, allow_empty: bool = True) -> "Selector":
        """Return a copy with the lazy empty-result policy changed.

        Parameters
        ----------
        allow_empty : bool, default=True
            Whether an empty lazy resolution should be accepted.

        Returns
        -------
        Selector
            Updated selector copy.
        """
        return replace(self, allow_empty=allow_empty)


@dataclass(frozen=True)
class CanvasSelector:
    """Typed canvas pseudo-element selector.

    Parameters
    ----------
    target : str
        Canvas primitive name.
    values : tuple[Any, ...]
        Primitive parameters, for example a side name or fractional location.
    """

    target: str = "canvas"
    values: Tuple[Any, ...] = ()

    @property
    def center(self) -> "CanvasSelector":
        """Return the canvas center pseudo-element.

        Returns
        -------
        CanvasSelector
            Canvas center selector.
        """
        return CanvasSelector("center")

    def edge(self, side: str) -> "CanvasSelector":
        """Return a canvas edge pseudo-element.

        Parameters
        ----------
        side : str
            Edge side name such as ``"top"`` or ``"bottom"``.

        Returns
        -------
        CanvasSelector
            Canvas edge selector.
        """
        return CanvasSelector("edge", (side,))

    def corner(self, corner: str) -> "CanvasSelector":
        """Return a canvas corner pseudo-element.

        Parameters
        ----------
        corner : str
            Corner name such as ``"ne"`` or ``"sw"``.

        Returns
        -------
        CanvasSelector
            Canvas corner selector.
        """
        return CanvasSelector("corner", (corner,))

    def fraction(self, fx: float, fy: float) -> "CanvasSelector":
        """Return a fractional canvas point pseudo-element.

        Parameters
        ----------
        fx : float
            Horizontal fraction in view coordinates.
        fy : float
            Vertical fraction in view coordinates.

        Returns
        -------
        CanvasSelector
            Canvas fractional point selector.
        """
        return CanvasSelector("fraction", (float(fx), float(fy)))


canvas = CanvasSelector()


def cluster(name: str, *, allow_empty: bool = False) -> Selector:
    """Select members of a named cluster lazily.

    Parameters
    ----------
    name : str
        Cluster name to resolve at lowering time.
    allow_empty : bool, default=False
        Whether an empty cluster resolution is accepted.

    Returns
    -------
    Selector
        Cluster selector.
    """
    return Selector("cluster", (name,), allow_empty=allow_empty)


def where(predicate: Callable[[Any], bool], *, allow_empty: bool = False) -> Selector:
    """Select nodes by predicate lazily.

    Parameters
    ----------
    predicate : callable
        Function receiving a node id and returning whether it is selected.
    allow_empty : bool, default=False
        Whether an empty predicate result is accepted.

    Returns
    -------
    Selector
        Predicate selector.
    """
    return Selector("where", (predicate,), allow_empty=allow_empty)


def edges(*pairs: Any, allow_empty: bool = False, **filters: Any) -> Selector:
    """Select graph edges by explicit pairs or lazy filters.

    Parameters
    ----------
    *pairs : Any
        Explicit ``(source, target)`` edge identifiers.
    allow_empty : bool, default=False
        Whether a lazy edge selector may resolve empty.
    **filters : Any
        Edge filters such as ``between=``, ``where=``, ``source=``, or
        attribute-based sugar.

    Returns
    -------
    Selector
        Edge selector.
    """
    return Selector("edges", tuple(pairs), dict(filters), allow_empty=allow_empty)


def path(*nodes: Any) -> Selector:
    """Select an ordered node path eagerly.

    Parameters
    ----------
    *nodes : Any
        Ordered node identifiers.

    Returns
    -------
    Selector
        Path selector.
    """
    return Selector("path", tuple(nodes))


def label(owner: Any) -> Selector:
    """Select the label associated with a node or edge owner.

    Parameters
    ----------
    owner : Any
        Node id or edge selector owner.

    Returns
    -------
    Selector
        Label selector.
    """
    return Selector("label", (owner,))


def labels(sel: Any = None, *, edges: bool = False, allow_empty: bool = False) -> Selector:
    """Select labels associated with a node/edge selection.

    Parameters
    ----------
    sel : Any, optional
        Owner selection for labels.
    edges : bool, default=False
        Whether to select edge labels.
    allow_empty : bool, default=False
        Whether empty lazy resolution is accepted.

    Returns
    -------
    Selector
        Labels selector.
    """
    return Selector("labels", (sel,), {"edges": edges}, allow_empty=allow_empty)


def port(node: Any, side_or_name: str) -> Selector:
    """Select one named or sided port on a node.

    Parameters
    ----------
    node : Any
        Node identifier.
    side_or_name : str
        Port side or user-defined name.

    Returns
    -------
    Selector
        Staged port selector.
    """
    return Selector("port", (node, side_or_name), staged=True)


def ports(sel: Any, side: Optional[str] = None, *, allow_empty: bool = False) -> Selector:
    """Select a set of ports on selected nodes.

    Parameters
    ----------
    sel : Any
        Node selection.
    side : str, optional
        Optional side filter.
    allow_empty : bool, default=False
        Whether empty lazy resolution is accepted.

    Returns
    -------
    Selector
        Staged ports selector.
    """
    return Selector("ports", (sel,), {"side": side}, allow_empty=allow_empty, staged=True)


@dataclass(frozen=True, kw_only=True)
class Constraint:
    """Immutable base class for user-intent constraints.

    Parameters
    ----------
    strength : Any, default="firm"
        Shared strength ladder value.
    name : str, optional
        User-visible handle name for reports and replacement.
    system : bool, default=False
        Whether this constraint was generated by Dagua.
    tags : tuple[str, ...]
        Report and serialization tags.
    """

    strength: StrengthLike = "firm"
    name: Optional[str] = None
    system: bool = False
    tags: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate the shared strength ladder after construction.

        Returns
        -------
        None
            Validation only.
        """
        strength_weight(self.strength)
        self._validate_hard()

    @property
    def weight(self) -> float:
        """Return the numeric weight for this constraint.

        Returns
        -------
        float
            Resolved strength weight.
        """
        return strength_weight(self.strength)

    @property
    def is_hard(self) -> bool:
        """Return whether this constraint is exact/hard.

        Returns
        -------
        bool
            ``True`` when the constraint uses infinite strength.
        """
        return math.isinf(self.weight)

    def _validate_hard(self) -> None:
        """Validate type-level hard support for the concrete constraint.

        Returns
        -------
        None
            Subclasses override to raise :class:`ConstraintTypeError`.
        """

    def soft(self) -> "Constraint":
        """Return a soft copy of this constraint.

        Returns
        -------
        Constraint
            Updated immutable copy.
        """
        return self.with_strength("soft")

    def firm(self) -> "Constraint":
        """Return a firm copy of this constraint.

        Returns
        -------
        Constraint
            Updated immutable copy.
        """
        return self.with_strength("firm")

    def rigid(self) -> "Constraint":
        """Return a rigid copy of this constraint.

        Returns
        -------
        Constraint
            Updated immutable copy.
        """
        return self.with_strength("rigid")

    def hard(self) -> "Constraint":
        """Return a hard copy of this constraint.

        Returns
        -------
        Constraint
            Updated immutable copy.
        """
        return self.with_strength("hard")

    def with_strength(self, strength: StrengthLike) -> "Constraint":
        """Return a copy using ``strength``.

        Parameters
        ----------
        strength : Any
            Shared strength ladder value.

        Returns
        -------
        Constraint
            Updated immutable copy.
        """
        return _copy_constraint(self, strength=strength)

    def named(self, name: str) -> "Constraint":
        """Return a copy with a report name.

        Parameters
        ----------
        name : str
            User-visible constraint name.

        Returns
        -------
        Constraint
            Updated immutable copy.
        """
        return _copy_constraint(self, name=name)

    def residual(self, pos: torch.Tensor, ctx: "ConstraintContext") -> torch.Tensor:
        """Return a normalized residual for reporting.

        Parameters
        ----------
        pos : torch.Tensor
            Position tensor with shape ``[N, 2]``.
        ctx : ConstraintContext
            Read-only layout context.

        Returns
        -------
        torch.Tensor
            Scalar residual in layout units.
        """
        return torch.zeros((), device=pos.device, dtype=pos.dtype)


def _init_constraint(
    instance: Constraint,
    *,
    strength: StrengthLike,
    name: Optional[str],
    system: bool,
    tags: Tuple[str, ...],
) -> None:
    """Initialize inherited frozen constraint fields.

    Parameters
    ----------
    instance : Constraint
        Constraint instance being constructed.
    strength : Any
        Shared strength value.
    name : str, optional
        Report name.
    system : bool
        Whether Dagua generated the constraint.
    tags : tuple[str, ...]
        Constraint tags.

    Returns
    -------
    None
        Mutates fields during construction only.
    """
    object.__setattr__(instance, "strength", strength)
    object.__setattr__(instance, "name", name)
    object.__setattr__(instance, "system", system)
    object.__setattr__(instance, "tags", tags)


def _copy_constraint(instance: Constraint, **updates: Any) -> Constraint:
    """Return a shallow frozen-field copy with updates applied.

    Parameters
    ----------
    instance : Constraint
        Existing immutable constraint.
    **updates : Any
        Field updates.

    Returns
    -------
    Constraint
        Updated constraint copy.
    """
    clone = object.__new__(type(instance))
    for constraint_field in fields(instance):
        object.__setattr__(clone, constraint_field.name, getattr(instance, constraint_field.name))
    for key, value in updates.items():
        object.__setattr__(clone, key, value)
    clone.__post_init__()
    return clone


@dataclass(frozen=True)
class Pin(Constraint):
    """Fix selected elements on one or both axes."""

    sel: Selection = None
    x: Optional[float] = None
    y: Optional[float] = None
    at: Optional[Any] = None
    frame: str = "view"
    strength: StrengthLike = field(default="hard", kw_only=True)


@dataclass(frozen=True, init=False)
class Align(Constraint):
    """Align selected elements along an axis."""

    sels: Tuple[Selection, ...] = ()
    axis: str = "x"
    at: Optional[Any] = None
    spacing: Optional[float] = None
    strength: StrengthLike = field(default="rigid", kw_only=True)

    def __init__(
        self,
        *sels: Selection,
        axis: str = "x",
        at: Optional[Any] = None,
        spacing: Optional[float] = None,
        strength: StrengthLike = "rigid",
        name: Optional[str] = None,
        system: bool = False,
        tags: Tuple[str, ...] = (),
    ) -> None:
        """Create an alignment constraint from positional selections."""
        _init_constraint(self, strength=strength, name=name, system=system, tags=tags)
        object.__setattr__(self, "sels", tuple(sels))
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "at", at)
        object.__setattr__(self, "spacing", spacing)
        self.__post_init__()


@dataclass(frozen=True, init=False)
class Order(Constraint):
    """Place items in directed order along an axis."""

    items: Tuple[Selection, ...] = ()
    axis: str = "flow"
    gap: Optional[float] = None
    strength: StrengthLike = field(default="hard", kw_only=True)

    def __init__(
        self,
        *items: Selection,
        axis: str = "flow",
        gap: Optional[float] = None,
        strength: StrengthLike = "hard",
        name: Optional[str] = None,
        system: bool = False,
        tags: Tuple[str, ...] = (),
    ) -> None:
        """Create an ordering constraint from positional selections."""
        _init_constraint(self, strength=strength, name=name, system=system, tags=tags)
        object.__setattr__(self, "items", tuple(items))
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "gap", gap)
        self.__post_init__()


@dataclass(frozen=True, init=False)
class Group(Constraint):
    """Keep selected members compact as a group."""

    sels: Tuple[Selection, ...] = ()
    padding: Optional[float] = None
    strength: StrengthLike = field(default="firm", kw_only=True)

    def __init__(
        self,
        *sels: Selection,
        padding: Optional[float] = None,
        strength: StrengthLike = "firm",
        name: Optional[str] = None,
        system: bool = False,
        tags: Tuple[str, ...] = (),
    ) -> None:
        """Create a grouping constraint from positional selections."""
        _init_constraint(self, strength=strength, name=name, system=system, tags=tags)
        object.__setattr__(self, "sels", tuple(sels))
        object.__setattr__(self, "padding", padding)
        self.__post_init__()


@dataclass(frozen=True)
class Separate(Constraint):
    """Keep two selected units apart."""

    a: Selection = None
    b: Selection = None
    gap: Optional[float] = None
    axis: Optional[str] = None
    side: Optional[str] = None
    strength: StrengthLike = field(default="firm", kw_only=True)

    def _validate_hard(self) -> None:
        """Refuse unprojectable symmetric hard separation.

        Returns
        -------
        None
            Validation only.

        Raises
        ------
        ConstraintTypeError
            If hard separation has no committed axis/side projector.
        """
        if self.is_hard and (self.axis is None or self.side is None):
            raise ConstraintTypeError(
                "hard separate requires a committed side; use order()/left_of()/right_of()/"
                'above()/below(), pass side="auto" with an axis, or use strength="rigid".'
            )


@dataclass(frozen=True)
class Anchor(Constraint):
    """Anchor selected elements to external coordinates."""

    mapping: Mapping[Any, Tuple[float, float]] = field(default_factory=dict)
    fit: str = "similarity"
    projection: Optional[Any] = None
    strength: StrengthLike = field(default="firm", kw_only=True)

    def _validate_hard(self) -> None:
        """Refuse hard anchors unless the transform is fixed.

        Returns
        -------
        None
            Validation only.

        Raises
        ------
        ConstraintTypeError
            If a free fitted transform is marked hard.
        """
        if self.is_hard and self.fit != "fixed":
            raise ConstraintTypeError(
                'hard anchor requires fit="fixed"; use fit="fixed" or strength="rigid".'
            )


@dataclass(frozen=True, init=False)
class Emphasize(Constraint):
    """Emphasize a path or edge set in the layout."""

    path_or_edges: Tuple[Selection, ...] = ()
    lane: Optional[Any] = None
    strength: StrengthLike = field(default="firm", kw_only=True)

    def __init__(
        self,
        *path_or_edges: Selection,
        lane: Optional[Any] = None,
        strength: StrengthLike = "firm",
        name: Optional[str] = None,
        system: bool = False,
        tags: Tuple[str, ...] = (),
    ) -> None:
        """Create an emphasis constraint from path or edge selections."""
        _init_constraint(self, strength=strength, name=name, system=system, tags=tags)
        object.__setattr__(self, "path_or_edges", tuple(path_or_edges))
        object.__setattr__(self, "lane", lane)
        self.__post_init__()

    def _validate_hard(self) -> None:
        """Refuse unprojectable hard emphasis.

        Returns
        -------
        None
            Validation only.
        """
        if self.is_hard:
            raise ConstraintTypeError('hard emphasize is not projectable; use strength="rigid".')


@dataclass(frozen=True)
class Focus(Constraint):
    """Pull selected members toward a focus region."""

    sel: Selection = None
    zoom: float = 1.5
    radius: int = 2
    at: Optional[Any] = None
    strength: StrengthLike = field(default="soft", kw_only=True)

    def _validate_hard(self) -> None:
        """Refuse unprojectable hard focus.

        Returns
        -------
        None
            Validation only.
        """
        if self.is_hard:
            raise ConstraintTypeError('hard focus is not projectable; use strength="rigid".')


@dataclass(frozen=True)
class Contain(Constraint):
    """Keep selected elements inside a containing box or pseudo-element."""

    sel: Selection = None
    within: Any = canvas
    padding: float = 0.0
    strength: StrengthLike = field(default="firm", kw_only=True)


@dataclass(frozen=True)
class Custom(Constraint):
    """Custom constraint implemented by user-supplied callables."""

    loss_fn: Optional[LossFn] = None
    project_fn: Optional[ProjectFn] = None
    access: str = "global"
    over: Optional[Selection] = None

    def _validate_hard(self) -> None:
        """Refuse hard custom constraints without a projector.

        Returns
        -------
        None
            Validation only.
        """
        if self.is_hard and self.project_fn is None:
            raise ConstraintTypeError("hard custom constraints require project().")

    def loss(self, pos: torch.Tensor, ctx: "ConstraintContext") -> torch.Tensor:
        """Evaluate the custom differentiable loss.

        Parameters
        ----------
        pos : torch.Tensor
            Position tensor with shape ``[N, 2]``.
        ctx : ConstraintContext
            Read-only layout context.

        Returns
        -------
        torch.Tensor
            Scalar loss tensor.
        """
        if self.loss_fn is None:
            return torch.zeros((), device=pos.device, dtype=pos.dtype)
        return self.loss_fn(pos, ctx)

    def project(self, pos: torch.Tensor, ctx: "ConstraintContext") -> None:
        """Run the custom exact projector.

        Parameters
        ----------
        pos : torch.Tensor
            Position tensor with shape ``[N, 2]``.
        ctx : ConstraintContext
            Read-only layout context.

        Returns
        -------
        None
            Projectors mutate ``pos`` in place under ``torch.no_grad``.
        """
        if self.project_fn is not None:
            self.project_fn(pos, ctx)


def loss(
    fn: LossFn,
    *,
    strength: StrengthLike = "firm",
    name: Optional[str] = None,
    access: str = "global",
    over: Optional[Selection] = None,
) -> Custom:
    """Construct a custom differentiable loss constraint.

    Parameters
    ----------
    fn : callable
        Function ``(pos, ctx) -> scalar tensor``.
    strength : Any, default="firm"
        Shared strength ladder value.
    name : str, optional
        Report name.
    access : str, default="global"
        Memory access hint: ``"global"``, ``"edge"``, or ``"sampled"``.
    over : Any, optional
        Optional scoped selection.

    Returns
    -------
    Custom
        Custom loss constraint.
    """
    return Custom(loss_fn=fn, strength=strength, name=name, access=access, over=over)


def project(fn: ProjectFn, *, name: Optional[str] = None) -> Custom:
    """Construct a custom exact projection constraint.

    Parameters
    ----------
    fn : callable
        Function ``(pos, ctx) -> None`` mutating positions in place.
    name : str, optional
        Report name.

    Returns
    -------
    Custom
        Hard custom projector constraint.
    """
    return Custom(project_fn=fn, strength="hard", name=name)


@dataclass
class ConstraintContext:
    """Read-only adapter exposed to custom constraints and reports."""

    graph: Any = None
    node_id_to_index: Dict[Any, int] = field(default_factory=dict)
    edge_index: Optional[torch.Tensor] = None
    node_sizes: Optional[torch.Tensor] = None
    direction: str = "TB"

    def idx(self, node_id: Any) -> int:
        """Return the integer index for ``node_id``.

        Parameters
        ----------
        node_id : Any
            Public node identifier or integer index.

        Returns
        -------
        int
            Internal node index.
        """
        if isinstance(node_id, int):
            return node_id
        return self.node_id_to_index[node_id]

    def indices(self, selection: Any) -> torch.Tensor:
        """Resolve a node selection to an index tensor.

        Parameters
        ----------
        selection : Any
            Bare id, list, tuple, or selector.

        Returns
        -------
        torch.Tensor
            Long tensor of selected node indices.
        """
        resolved = resolve_node_selection(selection, self.graph)
        return torch.tensor(resolved, dtype=torch.long)

    def edges(self, selection: Any) -> torch.Tensor:
        """Resolve an edge selection to edge indices.

        Parameters
        ----------
        selection : Any
            Edge selector.

        Returns
        -------
        torch.Tensor
            Long tensor of selected edge indices.
        """
        resolved = resolve_edge_selection(selection, self.graph)
        return torch.tensor(resolved, dtype=torch.long)

    def label_indices(self, selection: Any) -> torch.Tensor:
        """Resolve label selections.

        Parameters
        ----------
        selection : Any
            Label selection.

        Returns
        -------
        torch.Tensor
            Empty tensor until label lowering is promoted into the engine.
        """
        resolved = resolve_label_selection(selection, self.graph)
        return torch.tensor(resolved, dtype=torch.long)

    @property
    def extent(self) -> Optional[Tuple[float, float, float, float]]:
        """Return the current view extent when available.

        Returns
        -------
        tuple[float, float, float, float] | None
            Current extent as ``(xmin, ymin, xmax, ymax)`` or ``None``.
        """
        return None

    def distances(self) -> torch.Tensor:
        """Return lazy graph distances for custom constraints.

        Returns
        -------
        torch.Tensor
            Empty tensor placeholder until SolveState-backed APSP is provided.
        """
        return torch.zeros(0)


@dataclass(frozen=True)
class ConstraintResidual:
    """One normalized residual row in a constraint report."""

    constraint: Constraint
    residual: float
    hard_satisfied: bool
    resolved_count: int = 0


@dataclass
class ConstraintReport:
    """Report of constraint residuals and hard satisfaction."""

    residuals: List[ConstraintResidual] = field(default_factory=list)
    policy: str = "warn"
    outcome: str = "ok"

    @property
    def violations(self) -> List[ConstraintResidual]:
        """Return residual rows that violate their expected tolerance.

        Returns
        -------
        list[ConstraintResidual]
            Violated constraints.
        """
        return [row for row in self.residuals if not row.hard_satisfied]

    def summary(self) -> str:
        """Return a compact human-readable report summary.

        Returns
        -------
        str
            Summary string.
        """
        hard = sum(1 for row in self.residuals if row.constraint.is_hard)
        failed = len(self.violations)
        return f"{len(self.residuals)} constraints, {hard} hard, {failed} violations"


@dataclass
class ConstraintSet:
    """Mutable container for immutable constraints."""

    _items: List[Constraint] = field(default_factory=list)
    _last_report: ConstraintReport = field(default_factory=ConstraintReport)
    _owner_graph: Any = None

    def __iter__(self) -> Iterator[Constraint]:
        """Iterate over constraints in insertion order.

        Returns
        -------
        Iterator[Constraint]
            Constraint iterator.
        """
        return iter(self._items)

    def __len__(self) -> int:
        """Return the number of constraints.

        Returns
        -------
        int
            Constraint count.
        """
        return len(self._items)

    def __getitem__(self, index: int) -> Constraint:
        """Return a constraint by index.

        Parameters
        ----------
        index : int
            Zero-based constraint index.

        Returns
        -------
        Constraint
            Stored constraint.
        """
        return self._items[index]

    def append(self, constraint: Constraint) -> Constraint:
        """Append and return ``constraint``.

        Parameters
        ----------
        constraint : Constraint
            Constraint to store.

        Returns
        -------
        Constraint
            The same immutable handle.
        """
        self._items.append(constraint)
        return constraint

    def extend(self, constraints: Iterable[Constraint]) -> None:
        """Append multiple constraints.

        Parameters
        ----------
        constraints : iterable[Constraint]
            Constraints to append.

        Returns
        -------
        None
        """
        self._items.extend(constraints)

    def remove(self, constraint: Constraint) -> None:
        """Remove a stored constraint.

        Parameters
        ----------
        constraint : Constraint
            Constraint handle to remove.

        Returns
        -------
        None
        """
        self._items.remove(constraint)

    def replace(self, old: Constraint, new: Constraint) -> Constraint:
        """Replace one stored constraint with another.

        Parameters
        ----------
        old : Constraint
            Existing immutable handle.
        new : Constraint
            Replacement immutable handle.

        Returns
        -------
        Constraint
            Replacement handle.
        """
        index = self._items.index(old)
        self._items[index] = new
        return new

    def clear(self) -> None:
        """Remove all stored constraints.

        Returns
        -------
        None
        """
        self._items.clear()

    def tags(self, *tags: str) -> List[Constraint]:
        """Return constraints carrying any requested tag.

        Parameters
        ----------
        *tags : str
            Tags to match.

        Returns
        -------
        list[Constraint]
            Matching constraints.
        """
        wanted = set(tags)
        return [constraint for constraint in self._items if wanted.intersection(constraint.tags)]

    def report(
        self,
        pos: Optional[torch.Tensor] = None,
        *,
        policy: str = "report",
    ) -> ConstraintReport:
        """Return the most recent constraint report.

        Parameters
        ----------
        pos : torch.Tensor, optional
            Position tensor with shape ``[N, 2]``. When provided, a fresh
            report is computed against the owning graph.
        policy : str, default="report"
            Policy label stored on freshly computed reports.

        Returns
        -------
        ConstraintReport
            Last report object.
        """
        if pos is not None:
            self._last_report = build_constraint_report(
                self._items,
                pos,
                self._owner_graph,
                policy=policy,
            )
        return self._last_report

    def set_report(self, report: ConstraintReport) -> None:
        """Store a new report for later user inspection.

        Parameters
        ----------
        report : ConstraintReport
            Report to retain.

        Returns
        -------
        None
        """
        self._last_report = report

    def to_list(self) -> List[Constraint]:
        """Return a shallow copy of stored constraints.

        Returns
        -------
        list[Constraint]
            Constraint handles.
        """
        return list(self._items)


def build_constraint_report(
    constraints: Sequence[Constraint],
    pos: torch.Tensor,
    graph: Any,
    *,
    policy: str = "report",
) -> ConstraintReport:
    """Build normalized residual rows for constraints at ``pos``.

    Parameters
    ----------
    constraints : sequence[Constraint]
        Constraint objects to evaluate.
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    graph : Any
        Graph used to resolve lazy selectors. May be ``None`` when selections
        are already lowered to integer indices.
    policy : str, default="report"
        Conflict policy label.

    Returns
    -------
    ConstraintReport
        Residual report with hard/rigid violations marked.
    """
    rows = []
    for constraint in constraints:
        residual, resolved_count = _constraint_residual(constraint, pos, graph)
        tolerance = 1e-5 if constraint.is_hard else 1e-3
        hard_satisfied = True
        if constraint.is_hard or constraint.weight >= STRENGTHS["rigid"]:
            hard_satisfied = residual <= tolerance
        rows.append(
            ConstraintResidual(
                constraint=constraint,
                residual=residual,
                hard_satisfied=hard_satisfied,
                resolved_count=resolved_count,
            )
        )
    return ConstraintReport(
        residuals=rows,
        policy=policy,
        outcome="ok" if all(row.hard_satisfied for row in rows) else "violated",
    )


def _constraint_residual(
    constraint: Constraint,
    pos: torch.Tensor,
    graph: Any,
) -> Tuple[float, int]:
    """Return one normalized residual and resolved element count.

    Parameters
    ----------
    constraint : Constraint
        Constraint object.
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    graph : Any
        Optional graph for lazy selector resolution.

    Returns
    -------
    tuple[float, int]
        Residual in layout units and resolved element count.
    """
    try:
        if isinstance(constraint, Pin):
            return _pin_residual(constraint, pos, graph)
        if isinstance(constraint, Align):
            return _align_residual(constraint, pos, graph)
        if isinstance(constraint, Order):
            return _order_residual(constraint, pos, graph)
        if isinstance(constraint, Separate):
            return _separate_residual(constraint, pos, graph)
        if isinstance(constraint, Group):
            return _group_residual(constraint, pos, graph)
        if isinstance(constraint, Anchor):
            return _anchor_residual(constraint, pos, graph)
        if isinstance(constraint, Focus):
            indices = resolve_node_selection(constraint.sel, graph, allow_empty=True)
            return (0.0, len(indices))
        if isinstance(constraint, Contain):
            return _contain_residual(constraint, pos, graph)
        if isinstance(constraint, Emphasize):
            count = sum(
                len(resolve_node_selection(item, graph, allow_empty=True))
                for item in constraint.path_or_edges
                if not (isinstance(item, Selector) and item.kind == "edges")
            )
            return (0.0, count)
    except ConstraintError:
        raise
    except Exception:
        return (0.0, 0)
    return (0.0, 0)


def _pin_residual(constraint: Pin, pos: torch.Tensor, graph: Any) -> Tuple[float, int]:
    """Return normalized pin residual.

    Parameters
    ----------
    constraint : Pin
        Pin constraint.
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    graph : Any
        Graph for selector resolution.

    Returns
    -------
    tuple[float, int]
        Max axis residual and selected count.
    """
    indices = resolve_node_selection(constraint.sel, graph, allow_empty=True)
    at_x, at_y = resolve_canvas_point(constraint.at, pos)
    residuals: List[float] = []
    for index in indices:
        target_x = constraint.x if constraint.x is not None else at_x
        target_y = constraint.y if constraint.y is not None else at_y
        if target_x is not None:
            residuals.append(abs(float(pos[index, 0].item()) - float(target_x)))
        if target_y is not None:
            residuals.append(abs(float(pos[index, 1].item()) - float(target_y)))
    return (max(residuals, default=0.0), len(indices))


def _align_residual(constraint: Align, pos: torch.Tensor, graph: Any) -> Tuple[float, int]:
    """Return normalized alignment residual.

    Parameters
    ----------
    constraint : Align
        Alignment constraint.
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    graph : Any
        Graph for selector resolution.

    Returns
    -------
    tuple[float, int]
        Coordinate spread or target residual and selected count.
    """
    indices = _flatten_resolved_nodes(constraint.sels, graph)
    if len(indices) < 2:
        return (0.0, len(indices))
    axis = 0 if constraint.axis in {"x", "cross"} else 1
    coords = pos[indices, axis]
    spread = float((coords.max() - coords.min()).abs().item())
    at_x, at_y = resolve_canvas_point(constraint.at, pos)
    at_value = at_x if axis == 0 else at_y
    if at_value is not None:
        spread = max(spread, float((coords - float(at_value)).abs().max().item()))
    return (spread, len(indices))


def _order_residual(constraint: Order, pos: torch.Tensor, graph: Any) -> Tuple[float, int]:
    """Return normalized ordering residual.

    Parameters
    ----------
    constraint : Order
        Order constraint.
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    graph : Any
        Graph for selector resolution.

    Returns
    -------
    tuple[float, int]
        Max hinge violation and resolved count.
    """
    items = [resolve_node_selection(item, graph, allow_empty=True) for item in constraint.items]
    axis = 1 if constraint.axis in {"flow", "y"} else 0
    gap = float(constraint.gap or 0.0)
    residuals = []
    for left, right in zip(items, items[1:]):
        if not left or not right:
            continue
        left_coord = float(pos[left, axis].mean().item())
        right_coord = float(pos[right, axis].mean().item())
        residuals.append(max(0.0, left_coord + gap - right_coord))
    return (max(residuals, default=0.0), sum(len(item) for item in items))


def _separate_residual(constraint: Separate, pos: torch.Tensor, graph: Any) -> Tuple[float, int]:
    """Return normalized separation residual.

    Parameters
    ----------
    constraint : Separate
        Separation constraint.
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    graph : Any
        Graph for selector resolution.

    Returns
    -------
    tuple[float, int]
        Distance shortfall and resolved count.
    """
    left = resolve_node_selection(constraint.a, graph, allow_empty=True)
    right = resolve_node_selection(constraint.b, graph, allow_empty=True)
    if not left or not right:
        return (0.0, len(left) + len(right))
    a = pos[left].mean(dim=0)
    b = pos[right].mean(dim=0)
    if constraint.axis is None:
        dist = float((a - b).square().sum().sqrt().item())
    else:
        axis = 0 if constraint.axis in {"x", "cross"} else 1
        dist = abs(float(a[axis].item()) - float(b[axis].item()))
    return (max(0.0, float(constraint.gap or 0.0) - dist), len(left) + len(right))


def _group_residual(constraint: Group, pos: torch.Tensor, graph: Any) -> Tuple[float, int]:
    """Return normalized group spread residual.

    Parameters
    ----------
    constraint : Group
        Group constraint.
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    graph : Any
        Graph for selector resolution.

    Returns
    -------
    tuple[float, int]
        Mean distance from centroid and resolved count.
    """
    indices = _flatten_resolved_nodes(constraint.sels, graph)
    if len(indices) < 2:
        return (0.0, len(indices))
    points = pos[indices]
    centroid = points.mean(dim=0, keepdim=True)
    residual = float((points - centroid).square().sum(dim=1).sqrt().mean().item())
    return (residual, len(indices))


def _anchor_residual(constraint: Anchor, pos: torch.Tensor, graph: Any) -> Tuple[float, int]:
    """Return normalized fixed-anchor residual.

    Parameters
    ----------
    constraint : Anchor
        Anchor constraint.
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    graph : Any
        Graph for selector resolution.

    Returns
    -------
    tuple[float, int]
        Mean target distance and resolved count.
    """
    residuals = []
    for node_id, target in constraint.mapping.items():
        indices = resolve_node_selection(node_id, graph, allow_empty=True)
        if not indices:
            continue
        delta = pos[indices[0]] - torch.tensor(target, dtype=pos.dtype, device=pos.device)
        residuals.append(float(delta.square().sum().sqrt().item()))
    return (max(residuals, default=0.0), len(residuals))


def _contain_residual(constraint: Contain, pos: torch.Tensor, graph: Any) -> Tuple[float, int]:
    """Return normalized containment residual.

    Parameters
    ----------
    constraint : Contain
        Containment constraint.
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    graph : Any
        Graph for selector resolution.

    Returns
    -------
    tuple[float, int]
        Max outside distance and selected count.
    """
    indices = resolve_node_selection(constraint.sel, graph, allow_empty=True)
    if not indices:
        return (0.0, 0)
    if isinstance(constraint.within, CanvasSelector):
        left = resolve_canvas_point(constraint.within.edge("left"), pos)[0]
        top = resolve_canvas_point(constraint.within.edge("top"), pos)[1]
        right = resolve_canvas_point(constraint.within.edge("right"), pos)[0]
        bottom = resolve_canvas_point(constraint.within.edge("bottom"), pos)[1]
        if left is None or top is None or right is None or bottom is None:
            return (0.0, len(indices))
        bounds = (
            float(left) + constraint.padding,
            float(top) + constraint.padding,
            float(right) - constraint.padding,
            float(bottom) - constraint.padding,
        )
    else:
        within = resolve_node_selection(constraint.within, graph, allow_empty=True)
        if not within:
            return (0.0, len(indices))
        container = pos[within]
        bounds = (
            float(container[:, 0].min().item()) + constraint.padding,
            float(container[:, 1].min().item()) + constraint.padding,
            float(container[:, 0].max().item()) - constraint.padding,
            float(container[:, 1].max().item()) - constraint.padding,
        )
    xmin, ymin, xmax, ymax = bounds
    residuals = []
    for index in indices:
        x = float(pos[index, 0].item())
        y = float(pos[index, 1].item())
        residuals.append(max(0.0, xmin - x, x - xmax, ymin - y, y - ymax))
    return (max(residuals, default=0.0), len(indices))


def _flatten_resolved_nodes(selections: Sequence[Any], graph: Any) -> List[int]:
    """Resolve and flatten node selections.

    Parameters
    ----------
    selections : sequence[Any]
        Selections to flatten.
    graph : Any
        Graph for selector resolution.

    Returns
    -------
    list[int]
        Unique node indices in first-seen order.
    """
    indices: List[int] = []
    for selection in selections:
        indices.extend(resolve_node_selection(selection, graph, allow_empty=True))
    return _unique_preserving(indices)


def resolve_node_selection(selection: Any, graph: Any, *, allow_empty: bool = False) -> List[int]:
    """Resolve a node selection against a graph object.

    Parameters
    ----------
    selection : Any
        Bare node id, list of ids, path selector, cluster selector, predicate
        selector, or a group constraint handle.
    graph : Any
        DaguaGraph-like object with ``_id_to_index`` and cluster metadata.
    allow_empty : bool, default=False
        Whether an empty result is accepted.

    Returns
    -------
    list[int]
        Internal node indices.

    Raises
    ------
    ConstraintReferenceError
        If an eager id is unknown or a lazy selector resolves empty.
    ConstraintTypeError
        If the selector kind cannot resolve to nodes.
    """
    if graph is None:
        return [int(item) for item in _as_sequence(selection)]
    if isinstance(selection, Group):
        out: List[int] = []
        for sel in selection.sels:
            out.extend(resolve_node_selection(sel, graph, allow_empty=allow_empty))
        return _unique_preserving(out)
    if isinstance(selection, Selector):
        if selection.staged or selection.kind in {"port", "ports"}:
            raise ConstraintStagedError(
                "Port constraint lowering is staged to the edge-routing sprint; "
                "the C.port/C.ports API is fixed, but it cannot be lowered yet."
            )
        if selection.kind == "cluster":
            name = selection.args[0]
            if name not in graph.clusters:
                raise ConstraintReferenceError(f"Unknown cluster: {name!r}")
            result = list(graph.clusters[name])
            return _checked_nonempty(
                result,
                selection.allow_empty or allow_empty,
                f"cluster {name!r}",
            )
        if selection.kind == "where":
            predicate = selection.args[0]
            ids = list(getattr(graph, "_index_to_id", []))
            result = [graph._id_to_index[node_id] for node_id in ids if predicate(node_id)]
            return _checked_nonempty(
                result,
                selection.allow_empty or allow_empty,
                "where() selector",
            )
        if selection.kind == "path":
            result = [resolve_single_node(node_id, graph) for node_id in selection.args]
            return _checked_nonempty(result, selection.allow_empty or allow_empty, "path selector")
        if selection.kind == "label":
            result = resolve_label_selection(selection, graph)
            return _checked_nonempty(result, selection.allow_empty or allow_empty, "label selector")
        if selection.kind == "labels":
            result = resolve_label_selection(selection, graph)
            return _checked_nonempty(
                result,
                selection.allow_empty or allow_empty,
                "labels selector",
            )
        if selection.kind == "edges":
            raise ConstraintTypeError(
                "edge selectors cannot be used where node-like points are required; "
                "use emphasize(), separate(), or C.loss(..., over=C.edges(...))."
            )
        raise ConstraintTypeError(f"{selection.kind!r} selectors cannot be resolved as nodes.")
    if isinstance(selection, (list, tuple)) and not _is_graph_node_id(selection, graph):
        result = [resolve_single_node(item, graph) for item in selection]
        return _checked_nonempty(result, allow_empty, "node list")
    return [resolve_single_node(selection, graph)]


def resolve_single_node(node_id: Any, graph: Any) -> int:
    """Resolve one node id to an index.

    Parameters
    ----------
    node_id : Any
        Public node id or integer index.
    graph : Any
        DaguaGraph-like object.

    Returns
    -------
    int
        Internal node index.
    """
    if isinstance(node_id, int) and 0 <= node_id < graph.num_nodes:
        return int(node_id)
    if node_id in graph._id_to_index:
        return int(graph._id_to_index[node_id])
    raise ConstraintReferenceError(f"Unknown node ID: {node_id!r}")


def validate_eager_selection(selection: Any, graph: Any) -> None:
    """Validate eager selections at graph-verb call time.

    Parameters
    ----------
    selection : Any
        User selection.
    graph : Any
        DaguaGraph-like object.

    Returns
    -------
    None
        Validation only.
    """
    if isinstance(selection, Selector):
        if selection.staged or selection.kind in {"port", "ports"}:
            return
        if selection.kind in {"path"}:
            resolve_node_selection(selection, graph)
        elif selection.kind == "edges" and selection.args:
            resolve_edge_selection(selection, graph)
        elif selection.kind == "label":
            resolve_label_selection(selection, graph)
        return
    if isinstance(selection, CanvasSelector):
        return
    if isinstance(selection, (list, tuple)) and not _is_graph_node_id(selection, graph):
        for item in selection:
            validate_eager_selection(item, graph)
        return
    resolve_single_node(selection, graph)


def resolve_edge_selection(selection: Any, graph: Any, *, allow_empty: bool = False) -> List[int]:
    """Resolve an edge selection against a graph object.

    Parameters
    ----------
    selection : Any
        Edge selector, explicit edge pair, or ``None`` for all edges.
    graph : Any
        DaguaGraph-like object with ``edge_index`` and edge metadata.
    allow_empty : bool, default=False
        Whether an empty result is accepted.

    Returns
    -------
    list[int]
        Packed edge indices.
    """
    if graph is None:
        return []
    edge_index = getattr(graph, "edge_index", None)
    if edge_index is None:
        return []
    edge_pairs = [
        (int(edge_index[0, edge_idx].item()), int(edge_index[1, edge_idx].item()))
        for edge_idx in range(int(edge_index.shape[1]))
    ]
    if selection is None:
        return list(range(len(edge_pairs)))
    if isinstance(selection, Selector):
        if selection.kind != "edges":
            raise ConstraintTypeError(f"{selection.kind!r} selectors cannot be resolved as edges.")
        result = _resolve_edges_selector(selection, graph, edge_pairs)
        return _checked_nonempty(
            result,
            selection.allow_empty or allow_empty,
            "edges() selector",
        )
    if isinstance(selection, tuple) and len(selection) == 2:
        src = resolve_single_node(selection[0], graph)
        tgt = resolve_single_node(selection[1], graph)
        result = [idx for idx, pair in enumerate(edge_pairs) if pair == (src, tgt)]
        return _checked_nonempty(result, allow_empty, f"edge {(selection[0], selection[1])!r}")
    raise ConstraintTypeError("Edges must be selected with C.edges(...).")


def _resolve_edges_selector(
    selection: Selector,
    graph: Any,
    edge_pairs: List[Tuple[int, int]],
) -> List[int]:
    """Resolve a ``C.edges`` selector payload.

    Parameters
    ----------
    selection : Selector
        Edge selector.
    graph : Any
        DaguaGraph-like object.
    edge_pairs : list[tuple[int, int]]
        Packed source/target index pairs.

    Returns
    -------
    list[int]
        Matching edge indices.
    """
    result: List[int] = []
    for pair in selection.args:
        result.extend(resolve_edge_selection(pair, graph, allow_empty=selection.allow_empty))
    filters = selection.kwargs
    if "between" in filters:
        left_sel, right_sel = filters["between"]
        left = set(resolve_node_selection(left_sel, graph))
        right = set(resolve_node_selection(right_sel, graph))
        result.extend(
            edge_idx
            for edge_idx, (src, tgt) in enumerate(edge_pairs)
            if (src in left and tgt in right) or (src in right and tgt in left)
        )
    if "where" in filters:
        predicate = filters["where"]
        result.extend(
            edge_idx
            for edge_idx, (src, tgt) in enumerate(edge_pairs)
            if predicate((graph._index_to_id[src], graph._index_to_id[tgt]))
        )
    source_filter = filters.get("source")
    if source_filter is not None:
        source_idx = resolve_single_node(source_filter, graph)
        result.extend(
            edge_idx for edge_idx, (src, _tgt) in enumerate(edge_pairs) if src == source_idx
        )
    target_filter = filters.get("target")
    if target_filter is not None:
        target_idx = resolve_single_node(target_filter, graph)
        result.extend(
            edge_idx for edge_idx, (_src, tgt) in enumerate(edge_pairs) if tgt == target_idx
        )
    edge_types = getattr(graph, "edge_types", [])
    type_filter = filters.get("type")
    if type_filter is not None:
        result.extend(
            edge_idx for edge_idx, edge_type in enumerate(edge_types) if edge_type == type_filter
        )
    if not selection.args and not filters:
        result.extend(range(len(edge_pairs)))
    return _unique_preserving(result)


def resolve_label_selection(selection: Any, graph: Any, *, allow_empty: bool = False) -> List[int]:
    """Resolve label selectors to owner node indices.

    Parameters
    ----------
    selection : Any
        ``C.label`` or ``C.labels`` selector.
    graph : Any
        DaguaGraph-like object.
    allow_empty : bool, default=False
        Whether an empty result is accepted.

    Returns
    -------
    list[int]
        Owner node indices for label satellites.
    """
    if not isinstance(selection, Selector):
        return resolve_node_selection(selection, graph, allow_empty=allow_empty)
    if selection.kind == "label":
        owner = selection.args[0]
        if isinstance(owner, Selector) and owner.kind == "edges":
            edge_ids = resolve_edge_selection(owner, graph, allow_empty=allow_empty)
            edge_index = graph.edge_index
            return _unique_preserving(
                [int(edge_index[0, edge_idx].item()) for edge_idx in edge_ids]
            )
        return resolve_node_selection(owner, graph, allow_empty=allow_empty)
    if selection.kind == "labels":
        owner = selection.args[0]
        if selection.kwargs.get("edges", False):
            edge_sel = owner if owner is not None else edges(allow_empty=allow_empty)
            edge_ids = resolve_edge_selection(edge_sel, graph, allow_empty=allow_empty)
            edge_index = graph.edge_index
            result = [int(edge_index[0, edge_idx].item()) for edge_idx in edge_ids]
            return _checked_nonempty(
                _unique_preserving(result),
                selection.allow_empty or allow_empty,
                "edge labels selector",
            )
        if owner is None:
            labelled = [
                idx
                for idx, label_text in enumerate(getattr(graph, "node_labels", []))
                if label_text is not None
            ]
            return _checked_nonempty(
                labelled,
                selection.allow_empty or allow_empty,
                "labels selector",
            )
        return resolve_node_selection(
            owner,
            graph,
            allow_empty=selection.allow_empty or allow_empty,
        )
    raise ConstraintTypeError(f"{selection.kind!r} selectors cannot be resolved as labels.")


def resolve_canvas_point(
    target: Any,
    pos: Optional[torch.Tensor] = None,
) -> Tuple[Optional[float], Optional[float]]:
    """Resolve a canvas pseudo-element to a deterministic view-frame point.

    Parameters
    ----------
    target : Any
        Canvas selector or raw ``(x, y)`` tuple.
    pos : torch.Tensor, optional
        Current positions used to derive an extent when available.

    Returns
    -------
    tuple[float | None, float | None]
        Point coordinates. ``None`` denotes an axis-specific canvas line.
    """
    if isinstance(target, CanvasSelector):
        xmin, ymin, xmax, ymax = _canvas_extent(pos)
        if target.target in {"canvas", "center"}:
            return ((xmin + xmax) * 0.5, (ymin + ymax) * 0.5)
        if target.target == "fraction":
            fx, fy = target.values
            return (xmin + (xmax - xmin) * float(fx), ymin + (ymax - ymin) * float(fy))
        if target.target == "corner":
            corner_name = str(target.values[0]).lower()
            x = xmax if "e" in corner_name else xmin
            y = ymax if "s" in corner_name else ymin
            return (x, y)
        if target.target == "edge":
            side = str(target.values[0]).lower()
            if side in {"top", "north", "n"}:
                return (None, ymin)
            if side in {"bottom", "south", "s"}:
                return (None, ymax)
            if side in {"left", "west", "w"}:
                return (xmin, None)
            if side in {"right", "east", "e"}:
                return (xmax, None)
    if isinstance(target, tuple) and len(target) == 2:
        return (float(target[0]), float(target[1]))
    return (None, None)


def _canvas_extent(pos: Optional[torch.Tensor]) -> Tuple[float, float, float, float]:
    """Return a stable canvas extent derived from positions when possible.

    Parameters
    ----------
    pos : torch.Tensor, optional
        Position tensor with shape ``[N, 2]``.

    Returns
    -------
    tuple[float, float, float, float]
        ``(xmin, ymin, xmax, ymax)`` with a minimum useful size.
    """
    if pos is None or pos.numel() == 0:
        return (-500.0, -500.0, 500.0, 500.0)
    detached = pos.detach()
    xmin = float(detached[:, 0].min().item())
    xmax = float(detached[:, 0].max().item())
    ymin = float(detached[:, 1].min().item())
    ymax = float(detached[:, 1].max().item())
    pad = max(50.0, 0.1 * max(xmax - xmin, ymax - ymin, 1.0))
    return (xmin - pad, ymin - pad, xmax + pad, ymax + pad)


def _as_sequence(value: Any) -> Sequence[Any]:
    """Return ``value`` as a simple sequence.

    Parameters
    ----------
    value : Any
        Value to normalize.

    Returns
    -------
    sequence[Any]
        Sequence view.
    """
    if isinstance(value, (list, tuple)):
        return value
    return (value,)


def _is_graph_node_id(value: Any, graph: Any) -> bool:
    """Return whether a tuple/list value is itself a graph node id.

    Parameters
    ----------
    value : Any
        Candidate node identifier.
    graph : Any
        DaguaGraph-like object.

    Returns
    -------
    bool
        ``True`` when the value is a literal node id.
    """
    try:
        return value in graph._id_to_index
    except TypeError:
        return False


def _checked_nonempty(values: List[int], allow_empty: bool, label: str) -> List[int]:
    """Raise for empty lazy selections unless explicitly allowed.

    Parameters
    ----------
    values : list[int]
        Resolved indices.
    allow_empty : bool
        Empty-result policy.
    label : str
        Human-readable selector label.

    Returns
    -------
    list[int]
        The input values.
    """
    if not values and not allow_empty:
        raise ConstraintReferenceError(f"{label} resolved to no nodes.")
    return values


def _unique_preserving(values: Iterable[int]) -> List[int]:
    """Return unique integers preserving first-seen order.

    Parameters
    ----------
    values : iterable[int]
        Values to deduplicate.

    Returns
    -------
    list[int]
        Deduplicated values.
    """
    seen: set[int] = set()
    out: List[int] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out
