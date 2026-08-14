"""RULER V4 facet-layer primitives.

This package is intentionally independent of earlier ruler implementations.  Phase 1
contains scene ingestion, measurement frames, event metadata, and individual facets;
cross-facet composition is deliberately absent.
"""

from dagua.eval.ruler_v4.ingestion import ingest, ingest_record
from dagua.eval.ruler_v4.scene import (
    DrawingScene,
    FacetResult,
    GraphSemantics,
    InvalidScene,
    ObservationProfile,
    Route,
    Scene,
    StyleContract,
    ValidAbsence,
    ValidScene,
)

__all__ = [
    "DrawingScene",
    "FacetResult",
    "GraphSemantics",
    "InvalidScene",
    "ObservationProfile",
    "Route",
    "Scene",
    "StyleContract",
    "ValidAbsence",
    "ValidScene",
    "ingest",
    "ingest_record",
]
