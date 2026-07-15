"""Shared visual parity v2 schema types.

This module is the frozen cross-lane interface for the visual parity v2
program.  Dataclasses mirror the JSON records described in the sprint design
and intentionally avoid behavior beyond typed storage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

COVERAGE_MATRIX_SCHEMA_VERSION = 2
LEDGER_SCHEMA_VERSION = 2
CARD_MANIFEST_SCHEMA_VERSION = 2


class SupportStatus(str, Enum):
    """Coverage support states for a feature cell."""

    SUPPORTED = "supported"
    PARTIAL = "partial"
    MISSING = "missing"
    WAIVED = "waived"
    REFERENCE_UNAVAILABLE = "reference_unavailable"


class ParityStatus(str, Enum):
    """Parity measurement states for coverage and ledger rows."""

    UNTESTED = "untested"
    NEEDS_FIXTURE = "needs_fixture"
    MEASURING = "measuring"
    IN_TOLERANCE = "in_tolerance"
    MATCHED = "matched"
    DIVERGENT = "divergent"
    BLOCKED_UPSTREAM = "blocked_upstream"
    METRIC_UNTRUSTED = "metric_untrusted"
    WAIVED_IMPROVEMENT = "waived_improvement"
    WAIVED_OUT_OF_SCOPE = "waived_out_of_scope"


class Priority(str, Enum):
    """Visual parity priority bands."""

    P0 = "P0"
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"


class TargetKind(str, Enum):
    """Reference target lane used for a measurement."""

    SVG_DECLARED = "svg_declared"
    PNG_RASTER = "png_raster"
    TOOL_NATIVE = "tool_native"
    HEURISTIC = "heuristic"


class GeometryMode(str, Enum):
    """Geometry source used when rendering a parity case."""

    INJECTED = "injected"
    NATIVE = "native"


@dataclass(frozen=True)
class Waiver:
    """Document an accepted departure from strict visual parity.

    Parameters
    ----------
    reason
        Short reason the row is waived.
    why_not
        Explanation of why the implementation will not match the reference.
    user_visible_impact
        User-facing impact of the departure.
    future_work
        Follow-up path if the waiver is revisited.
    decided_by
        Decision owner, expected to be ``"orchestrator"`` or ``"JMT"``.
    date
        ISO-8601 decision date.
    evidence
        Path to supporting evidence, usually a comparison image.
    """

    reason: str
    why_not: str
    user_visible_impact: str
    future_work: str
    decided_by: str
    date: str
    evidence: str


@dataclass(frozen=True)
class SourceSnapshot:
    """Provenance for an external denominator source.

    Parameters
    ----------
    id
        Stable source identifier such as ``"gv-arrows"``.
    source_url
        Upstream URL for the source.
    source_kind
        Source category, for example ``"official_docs"``.
    retrieved_at
        ISO-8601 retrieval timestamp.
    version
        Pinned tool or document version.
    cache_path
        Repository-relative cached snapshot path.
    """

    id: str
    source_url: str
    source_kind: str
    retrieved_at: str
    version: str
    cache_path: str


@dataclass(frozen=True)
class AdapterCapability:
    """Machine-readable capability record for a competitor adapter.

    Parameters
    ----------
    adapter
        Adapter name.
    fixed_positions
        Whether fixed positions are honored.
    per_element_styles
        Whether per-element style variation is honored.
    deterministic
        Whether repeated runs are deterministic.
    gate_eligible
        Whether this adapter can gate parity rows.
    evidence
        Human-readable evidence supporting the capability classification.
    """

    adapter: str
    fixed_positions: bool
    per_element_styles: bool
    deterministic: bool
    gate_eligible: bool
    evidence: str


@dataclass(frozen=True)
class CoverageCell:
    """Flat coverage matrix cell record.

    Parameters
    ----------
    cell_id
        Stable coverage cell identifier.
    tool
        Reference tool, such as ``"graphviz"``.
    object
        Styled object kind, such as ``"edge"`` or ``"node"``.
    attribute
        Reference attribute name.
    value
        Reference attribute value.
    value_group
        Group for related values, for example ``"arrow_primitive"``.
    source
        Source snapshot id.
    dagua_field
        Dagua style field mapped to this cell.
    dagua_value
        Dagua value mapped to this cell.
    support_status
        Dagua support state.
    parity_status
        Current parity state.
    priority
        Priority band.
    target_kind
        Measurement target lane.
    geometry_mode
        Geometry source used by the fixture.
    fixture_ids
        Fixture ids that exercise this cell.
    metric_ids
        Metric ids used to decide this cell.
    tolerance
        Metric-specific tolerance dictionary.
    last_checked_round
        Last round id that updated this cell, if any.
    locked
        Whether a generated lock test protects this cell.
    lock_test
        Generated lock test path or node id, if locked.
    residual_class
        Residual class id for accepted non-matches, if any.
    waiver
        Waiver object when status is waived.
    notes
        Free-form notes for human triage.
    """

    cell_id: str
    tool: str
    object: str
    attribute: str
    value: str
    value_group: str
    source: str
    dagua_field: str
    dagua_value: str
    support_status: SupportStatus
    parity_status: ParityStatus
    priority: Priority
    target_kind: TargetKind
    geometry_mode: GeometryMode
    fixture_ids: List[str] = field(default_factory=list)
    metric_ids: List[str] = field(default_factory=list)
    tolerance: Dict[str, float] = field(default_factory=dict)
    last_checked_round: Optional[str] = None
    locked: bool = False
    lock_test: Optional[str] = None
    residual_class: Optional[str] = None
    waiver: Optional[Waiver] = None
    notes: str = ""


@dataclass(frozen=True)
class MetricRecord:
    """Metric result embedded in a ledger row.

    Parameters
    ----------
    metric_id
        Stable metric identifier.
    target
        Target value for the metric.
    current
        Current observed value for the metric.
    tolerance
        Passing tolerance for the metric.
    unit
        Metric unit, such as ``"iou"`` or ``"pt"``.
    status
        Metric status, typically ``"pass"`` or ``"fail"``.
    validated_tripwire
        Whether the paired tripwire has validated this metric.
    """

    metric_id: str
    target: Optional[float]
    current: Optional[float]
    tolerance: Optional[float]
    unit: str
    status: str
    validated_tripwire: bool


@dataclass(frozen=True)
class KnobEntry:
    """Ledger record for a tunable parity dial.

    Parameters
    ----------
    knob_id
        Stable knob identifier.
    dial
        Human-readable dial description.
    code_ref
        Code reference for the dial.
    status
        Current knob status.
    linked_rows
        Ledger row ids affected by this knob.
    consecutive_no_improve
        Number of consecutive rounds without improvement.
    """

    knob_id: str
    dial: str
    code_ref: str
    status: str
    linked_rows: List[str] = field(default_factory=list)
    consecutive_no_improve: int = 0


@dataclass(frozen=True)
class RoundEntry:
    """Ledger record for one parity loop round.

    Parameters
    ----------
    round_id
        Stable round identifier.
    track
        Track label, such as ``"G"`` or ``"D"``.
    subsprint
        Active sub-sprint id.
    commit
        Commit sha for the round.
    gates_summary
        Gate summary payload.
    tripwires
        Tripwire result summary.
    audit
        Audit artifact id or path, if any.
    rebase_label
        Rebase label when a round changes the target lane.
    """

    round_id: str
    track: str
    subsprint: str
    commit: str
    gates_summary: Dict[str, Any] = field(default_factory=dict)
    tripwires: str = "unknown"
    audit: Optional[str] = None
    rebase_label: Optional[str] = None


@dataclass(frozen=True)
class ResidualEntry:
    """Ledger record for an accepted residual.

    Parameters
    ----------
    residual_id
        Stable residual identifier.
    class
        Residual class, for example ``"rendering_stack_residual"``.
    lane
        Target lane affected by the residual.
    affected_rows
        Ledger rows affected by the residual.
    evidence
        Evidence artifact paths.
    accepted_at
        ISO-8601 acceptance timestamp.
    accepted_by
        Acceptance authority.
    rationale
        Human-readable rationale.
    """

    residual_id: str
    class_: str
    lane: str
    affected_rows: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    accepted_at: str = ""
    accepted_by: str = ""
    rationale: str = ""


@dataclass(frozen=True)
class LedgerRow:
    """Ledger row tying one coverage cell to references, metrics, and history.

    Parameters
    ----------
    row_id
        Stable ledger row identifier.
    coverage_cell_id
        Coverage cell id represented by this row.
    priority
        Priority band.
    feature_group
        Human-readable feature group.
    target_kind
        Measurement target lane.
    geometry_mode
        Geometry source used for this row.
    reference
        Reference provenance payload.
    dagua
        Dagua mapping payload.
    metrics
        Metric records for this row.
    parity_status
        Current parity state.
    support_status
        Dagua support state.
    locked
        Whether a generated lock test protects this row.
    lock_test
        Generated lock test path or node id, if locked.
    residual_class
        Residual class id, if any.
    waiver
        Waiver object, if any.
    history
        Round history payloads.
    last_updated
        ISO-8601 update timestamp.
    """

    row_id: str
    coverage_cell_id: str
    priority: Priority
    feature_group: str
    target_kind: TargetKind
    geometry_mode: GeometryMode
    reference: Dict[str, Any]
    dagua: Dict[str, Any]
    metrics: List[MetricRecord] = field(default_factory=list)
    parity_status: ParityStatus = ParityStatus.UNTESTED
    support_status: SupportStatus = SupportStatus.MISSING
    locked: bool = False
    lock_test: Optional[str] = None
    residual_class: Optional[str] = None
    waiver: Optional[Waiver] = None
    history: List[Dict[str, Any]] = field(default_factory=list)
    last_updated: str = ""


@dataclass(frozen=True)
class SplineSegment:
    """One cubic spline segment parsed from Graphviz xdot output.

    Parameters
    ----------
    edge_id
        Stable edge identifier.
    segment_index
        Zero-based segment index within the edge spline.
    start
        Segment start point in points as ``(x, y)``.
    control_1
        First cubic control point in points as ``(x, y)``.
    control_2
        Second cubic control point in points as ``(x, y)``.
    end
        Segment end point in points as ``(x, y)``.
    endpoint
        Optional arrow endpoint from the xdot ``e,`` field.
    """

    edge_id: str
    segment_index: int
    start: Tuple[float, float]
    control_1: Tuple[float, float]
    control_2: Tuple[float, float]
    end: Tuple[float, float]
    endpoint: Optional[Tuple[float, float]] = None


@dataclass(frozen=True)
class InjectedGeometry:
    """Graphviz geometry injected into a Dagua render pass.

    Parameters
    ----------
    case_id
        Case id associated with this geometry snapshot.
    tool
        Layout tool, normally ``"graphviz"``.
    tool_version
        Tool version string.
    dot_source_hash
        Content hash for the DOT source used by the reference cache.
    engine
        Graphviz engine used for geometry production.
    canvas_pt
        Declared canvas size in points as ``(width, height)``.
    node_positions
        Node center positions in points keyed by node id.
    node_sizes
        Node sizes in points keyed by node id.
    edge_splines
        Cubic spline segments keyed by edge id.
    cluster_rects
        Cluster rectangles in points keyed by cluster id.
    graph_attrs
        Parsed graph-level attributes.
    """

    case_id: str
    tool: str
    tool_version: str
    dot_source_hash: str
    engine: str
    canvas_pt: Tuple[float, float]
    node_positions: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    node_sizes: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    edge_splines: Dict[str, List[SplineSegment]] = field(default_factory=dict)
    cluster_rects: Dict[str, Tuple[float, float, float, float]] = field(default_factory=dict)
    graph_attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AlignmentManifest:
    """Manifest recording render alignment and composition operations.

    Parameters
    ----------
    canvas_pt
        Declared canvas size in points as ``(width, height)``.
    dpi
        Raster DPI used for both panels.
    pixel_size
        Raster size in pixels as ``(width, height)``.
    crop_box_px
        Crop box in pixels as ``(left, top, right, bottom)``.
    crop_reason
        Reason for cropping, if any.
    metric_uses_crop
        Whether metrics used the cropped image.
    pad_or_crop_applied
        Whether any pad or crop was applied.
    """

    canvas_pt: Tuple[float, float]
    dpi: float
    pixel_size: Tuple[int, int]
    crop_box_px: Optional[Tuple[int, int, int, int]]
    crop_reason: Optional[str]
    metric_uses_crop: bool
    pad_or_crop_applied: bool


@dataclass(frozen=True)
class TripwireSpec:
    """Declarative specification for one metric validation tripwire.

    Parameters
    ----------
    tripwire_id
        Stable tripwire identifier.
    injected_defect
        Defect injected by the tripwire.
    metric_id
        Metric expected to fire.
    min_effect_size
        Required minimum movement or condition.
    target_kind
        Target lane validated by this tripwire.
    notes
        Free-form implementation notes.
    """

    tripwire_id: str
    injected_defect: str
    metric_id: str
    min_effect_size: str
    target_kind: TargetKind
    notes: str = ""


@dataclass(frozen=True)
class TripwireResult:
    """Observed result for one tripwire run.

    Parameters
    ----------
    tripwire_id
        Stable tripwire identifier.
    metric_id
        Metric validated by this result.
    status
        Result status, usually ``"pass"`` or ``"fail"``.
    observed_effect
        Observed effect payload.
    min_effect_size
        Required minimum movement or condition.
    failed_metric_ids
        Metric ids made untrusted by a failure.
    evidence
        Evidence artifact paths.
    run_at
        ISO-8601 run timestamp.
    notes
        Free-form notes.
    """

    tripwire_id: str
    metric_id: str
    status: str
    observed_effect: Dict[str, Any] = field(default_factory=dict)
    min_effect_size: str = ""
    failed_metric_ids: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    run_at: str = ""
    notes: str = ""
