"""Independent dispatch registry for the 45 phase-1 V4 facets."""

from __future__ import annotations

from typing import Callable, Mapping, Optional, Union

from dagua.eval.ruler_v4.clusters import U25, U26, U27, U28, U29, U30
from dagua.eval.ruler_v4.contracts import CONTRACTS, SCORED_SUBTERM_COUNT
from dagua.eval.ruler_v4.directed import U31, U32, U33, U34, U39, U40
from dagua.eval.ruler_v4.edges import U07, U08, U10, U11, U12, U13, U15, U16
from dagua.eval.ruler_v4.legibility import U17, U18, U19, U21, U20a, U20b
from dagua.eval.ruler_v4.packing import U38, U41, U42
from dagua.eval.ruler_v4.scene import FacetResult, Scene, TemporalScene
from dagua.eval.ruler_v4.structure import (
    U01,
    U02,
    U03,
    U05,
    U06,
    U09,
    U14,
    U22,
    U23,
    U24,
    U01b,
    U04a,
    U04b,
)
from dagua.eval.ruler_v4.weights import U35, U36, U37

FacetFunction = Callable[..., FacetResult]

_ALPHA_GRID_FACETS = frozenset({"U17", "U18", "U27", "U28", "U30"})

FACET_FUNCTIONS: Mapping[str, FacetFunction] = {
    function.__name__: function
    for function in (
        U01,
        U01b,
        U02,
        U03,
        U04a,
        U04b,
        U05,
        U06,
        U07,
        U08,
        U09,
        U10,
        U11,
        U12,
        U13,
        U14,
        U15,
        U16,
        U17,
        U18,
        U19,
        U20a,
        U20b,
        U21,
        U22,
        U23,
        U24,
        U25,
        U26,
        U27,
        U28,
        U29,
        U30,
        U31,
        U32,
        U33,
        U34,
        U35,
        U36,
        U37,
        U38,
        U39,
        U40,
        U41,
        U42,
    )
}


def evaluate_facet(
    facet_id: str,
    scene: Union[Scene, TemporalScene],
    *,
    alpha_grid_index: Optional[int] = None,
    gamma: Optional[float] = None,
    lambda_T: Optional[float] = None,
) -> FacetResult:
    """Evaluate one independent contract facet.

    Parameters
    ----------
    facet_id : str
        Frozen contract id.
    scene : Scene or TemporalScene
        Validated canonical static scene, or the temporal scene required by U40.
    alpha_grid_index : int or None
        Shared U17-family grid row. ``None`` explicitly requests the frozen
        pre-selection envelope for grid-consuming facets.
    gamma : float or None
        U07 fitted crossing-severity scale. ``None`` uses its documented phase-4
        worked-example default.
    lambda_T : float or None
        U07 fitted tail weight. ``None`` uses its documented phase-4
        worked-example default.

    Returns
    -------
    FacetResult
        Independent facet result without cross-facet composition.

    Raises
    ------
    KeyError
        If ``facet_id`` is not one of the 45 frozen contracts.
    """

    function = FACET_FUNCTIONS[facet_id]
    if isinstance(scene, TemporalScene):
        if facet_id != "U40":
            raise TypeError(f"{facet_id} requires a static Scene")
        return U40(scene)
    if facet_id in _ALPHA_GRID_FACETS:
        return function(scene, alpha_grid_index)
    if facet_id == "U07":
        parameters = {}
        if gamma is not None:
            parameters["gamma"] = gamma
        if lambda_T is not None:
            parameters["lambda_T"] = lambda_T
        return function(scene, **parameters)
    if gamma is not None or lambda_T is not None:
        raise TypeError("gamma and lambda_T are only valid for U07")
    return function(scene)


def validate_registry() -> None:
    """Validate dispatch, names, hashes, docstrings, and scored row count.

    Raises
    ------
    RuntimeError
        If production metadata diverges from the frozen manifest inventory.
    """

    if set(FACET_FUNCTIONS) != set(CONTRACTS):
        missing = sorted(set(CONTRACTS) - set(FACET_FUNCTIONS))
        extra = sorted(set(FACET_FUNCTIONS) - set(CONTRACTS))
        raise RuntimeError(f"facet dispatch mismatch; missing={missing}, extra={extra}")
    if SCORED_SUBTERM_COUNT != 91:
        raise RuntimeError(f"expected 91 scored subterms, found {SCORED_SUBTERM_COUNT}")
    for facet_id, function in FACET_FUNCTIONS.items():
        contract = CONTRACTS[facet_id]
        docstring = function.__doc__ or ""
        if contract.title not in docstring or contract.sha256 not in docstring:
            raise RuntimeError(f"contract identity missing from {facet_id} docstring")
