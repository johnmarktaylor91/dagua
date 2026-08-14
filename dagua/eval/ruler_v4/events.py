"""Frozen event-manifold registry loading and graph-local jump bounds."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple


@dataclass(frozen=True)
class JumpBound:
    """One event's frozen jump-bound declaration.

    Parameters
    ----------
    kind : str
        ``numeric`` or ``closed_form``.
    value : float or None
        Fixed numeric bound.
    formula : str or None
        Frozen human-readable formula.
    """

    kind: str
    value: Optional[float] = None
    formula: Optional[str] = None


@dataclass(frozen=True)
class EventManifold:
    """One registered combinatorial event manifold.

    Parameters
    ----------
    facet_id : str
        Owning facet contract id.
    event_id : str
        Globally stable event identifier.
    predicate : str
        Frozen combinatorial predicate.
    combinatorial : bool
        Registry admission bit.
    jump_bound : JumpBound
        Published event-local bound declaration.
    links : tuple[str, ...]
        Related event ids.
    """

    facet_id: str
    event_id: str
    predicate: str
    combinatorial: bool
    jump_bound: JumpBound
    links: Tuple[str, ...] = ()


@dataclass(frozen=True)
class EventRegistry:
    """Validated immutable event registry.

    Parameters
    ----------
    schema_version : str
        Frozen registry schema version.
    entries : tuple[EventManifold, ...]
        Registered manifolds.
    facets_declaring_no_events : tuple[str, ...]
        Facets explicitly declaring an empty event set.
    """

    schema_version: str
    entries: Tuple[EventManifold, ...]
    facets_declaring_no_events: Tuple[str, ...]

    def by_facet(self, facet_id: str) -> Tuple[EventManifold, ...]:
        """Return all event manifolds owned by a facet.

        Parameters
        ----------
        facet_id : str
            Facet contract id.

        Returns
        -------
        tuple[EventManifold, ...]
            Stable registry-order entries.
        """

        return tuple(entry for entry in self.entries if entry.facet_id == facet_id)


_DEFAULT_DATA: Mapping[str, Any] = {
    "schema_version": "A5-event-registry-1.0",
    "entries": (
        ("U07", "U7.crossing-parity", "proper inter-edge crossing parity change", "u07"),
        ("U10", "U10.penetration", "route-obstacle intersection count change", "zero"),
        (
            "U11",
            "U11.i.self_intersection",
            "transversal route self-intersection count change",
            "u11",
        ),
        ("U17", "U17.contact_i_j", "node OBB disjoint/intersecting transition", "zero"),
        ("U18", "U18.contact", "label OBB and primitive contact transition", "zero"),
        ("U20a", "U20a.coincidence", "node coincidence or OBB contact", "zero"),
        ("U27", "U27.region_contact", "primitive and declared cluster region contact", "zero"),
        ("U28", "U28.region_contact", "declared hierarchy region contact", "zero"),
        ("U30", "U30.contact", "cluster label and primitive or region contact", "zero"),
        ("U38", "U38_COMPONENT_CONTACT", "component primitive-union contact", "zero"),
        ("U41", "U41_FACE_SPLIT", "one proper crossing creates or removes one face", "u41"),
        ("U42", "U42.backdrop_change", "topmost backdrop identity change", "zero"),
    ),
    "facets_declaring_no_events": (
        "U01",
        "U01b",
        "U02",
        "U03",
        "U04a",
        "U04b",
        "U05",
        "U06",
        "U08",
        "U09",
        "U12",
        "U13",
        "U14",
        "U15",
        "U16",
        "U19",
        "U20b",
        "U21",
        "U22",
        "U23",
        "U24",
        "U25",
        "U26",
        "U29",
        "U31",
        "U32",
        "U33",
        "U34",
        "U35",
        "U36",
        "U37",
        "U39",
        "U40",
    ),
}


def _default_registry() -> EventRegistry:
    """Construct the embedded frozen registry.

    Returns
    -------
    EventRegistry
        Immutable default registry.
    """

    formulas = {
        "zero": JumpBound("numeric", value=0.0),
        "u07": JumpBound(
            "closed_form",
            formula="min(1, (1+lambda_T)*(1+gamma+gamma*r_r/2+gamma*r_d*Dtilde/D0)/(Z_prime*x0))",
        ),
        "u11": JumpBound("closed_form", formula="w_U11_i*pi_e*0.75*b_max"),
        "u41": JumpBound(
            "closed_form", formula="min(1, 0.60*min(1,3/max(1,F0))+0.40*min(1,3/max(1,F0)))"
        ),
    }
    entries = tuple(
        EventManifold(facet, event, predicate, True, formulas[bound])
        for facet, event, predicate, bound in _DEFAULT_DATA["entries"]
    )
    return EventRegistry(
        str(_DEFAULT_DATA["schema_version"]),
        entries,
        tuple(_DEFAULT_DATA["facets_declaring_no_events"]),
    )


def _parse_registry(data: Mapping[str, Any]) -> EventRegistry:
    """Parse and validate one frozen registry mapping.

    Parameters
    ----------
    data : mapping[str, Any]
        Decoded registry JSON.

    Returns
    -------
    EventRegistry
        Immutable registry.

    Raises
    ------
    ValueError
        If required fields or admission invariants are invalid.
    """

    entries = []
    seen = set()
    for row in data.get("entries", ()):
        event_id = str(row["event_id"])
        if event_id in seen or not bool(row.get("combinatorial", False)):
            raise ValueError("event ids must be unique and combinatorial")
        seen.add(event_id)
        raw_bound = row["jump_bound"]
        bound = JumpBound(
            str(raw_bound["kind"]),
            float(raw_bound["value"]) if "value" in raw_bound else None,
            str(raw_bound["formula"]) if "formula" in raw_bound else None,
        )
        if bound.kind == "numeric" and (
            bound.value is None or not math.isfinite(bound.value) or bound.value < 0.0
        ):
            raise ValueError("numeric jump bounds must be finite and nonnegative")
        entries.append(
            EventManifold(
                str(row["facet_id"]),
                event_id,
                str(row["predicate"]),
                True,
                bound,
                tuple(str(item) for item in row.get("links", ())),
            )
        )
    return EventRegistry(
        str(data["schema_version"]),
        tuple(entries),
        tuple(str(item) for item in data.get("facets_declaring_no_events", ())),
    )


def load_event_registry(path: Optional[Path] = None) -> EventRegistry:
    """Load the embedded registry or validate an external frozen JSON copy.

    Parameters
    ----------
    path : pathlib.Path or None
        Optional EVENT_REGISTRY.json path. ``None`` uses the production-embedded
        contract data and has no workspace dependency.

    Returns
    -------
    EventRegistry
        Immutable validated registry.
    """

    if path is None:
        return _default_registry()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("event registry root must be an object")
    return _parse_registry(data)


def evaluate_jump_bound(
    event: EventManifold, context: Optional[Mapping[str, float]] = None
) -> float:
    """Evaluate a registered graph-local per-facet jump bound.

    Parameters
    ----------
    event : EventManifold
        Registered event entry.
    context : mapping[str, float] or None
        Formula inputs for nonconstant bounds.

    Returns
    -------
    float
        Nonnegative bound in facet-defect units.

    Raises
    ------
    ValueError
        If a required graph-local input is missing or invalid.
    """

    if event.jump_bound.kind == "numeric":
        assert event.jump_bound.value is not None
        return event.jump_bound.value
    if event.jump_bound.formula is not None and event.jump_bound.formula.strip() == "0.0":
        return 0.0
    values: Dict[str, float] = {key: float(value) for key, value in (context or {}).items()}
    if event.facet_id == "U41":
        f0 = max(1.0, values.get("F0", 1.0))
        return min(1.0, 0.60 * min(1.0, 3.0 / f0) + 0.40 * min(1.0, 3.0 / f0))
    if event.facet_id == "U11":
        required = ("w_U11_i", "pi_e", "b_max")
        if any(key not in values for key in required):
            raise ValueError("U11 jump bound requires w_U11_i, pi_e, and b_max")
        return values["w_U11_i"] * values["pi_e"] * 0.75 * values["b_max"]
    if event.facet_id == "U07":
        required = ("lambda_T", "gamma", "r_r", "r_d", "Dtilde", "D0", "Z_prime", "x0")
        if any(key not in values for key in required):
            raise ValueError("U07 jump bound context is incomplete")
        denominator = values["Z_prime"] * values["x0"]
        if denominator <= 0.0 or values["D0"] <= 0.0:
            raise ValueError("U07 jump-bound denominators must be positive")
        raw = (1.0 + values["lambda_T"]) * (
            1.0
            + values["gamma"]
            + values["gamma"] * values["r_r"] / 2.0
            + values["gamma"] * values["r_d"] * values["Dtilde"] / values["D0"]
        )
        return min(1.0, raw / denominator)
    raise ValueError(f"unsupported closed-form jump bound for {event.facet_id}")
