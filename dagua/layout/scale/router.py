"""Explicit scale-router decisions for large default layouts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from dagua.config import LayoutConfig
from dagua.layout.scale.sketch import TopologySketch

THRESHOLDS_VERSION = "scale-router-thresholds-v1"
DEFAULT_SCALE_NODE_GATE = 20_000
DEFAULT_SCALE_EDGE_GATE = 200_000
DEFAULT_DEPTH_CAP = 256
GIANT_SCC_MIN_FRACTION = 0.05
NONTRIVIAL_SCC_MIN_SIZE = 2
DEFAULT_REDUCTION_STALL_DEGREE = 4_096
DEFAULT_REDUCTION_STALL_FRACTION = 0.20
DECLARED_TOPOLOGY_VALUES = {"directed_cyclic", "directed_acyclic", "undirected"}


class ScaleStrategy(str, Enum):
    """Scale strategy selected by the explicit router."""

    NATIVE = "NATIVE"
    LAYERS = "LAYERS"
    FIELD = "FIELD"


@dataclass(frozen=True)
class RouteDecision:
    """Scale routing decision and the structural reasons behind it.

    Parameters
    ----------
    strategy : ScaleStrategy
        Selected large-graph strategy.
    reason_codes : list[str]
        Stable reason codes for regression-locked routing tests.
    sketch_fingerprint : str
        Fingerprint of the sketch used to make the decision.
    thresholds_version : str
        Version tag for threshold semantics.
    """

    strategy: ScaleStrategy
    reason_codes: List[str]
    sketch_fingerprint: str
    thresholds_version: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the decision for graph/result metadata.

        Returns
        -------
        dict[str, object]
            JSON-friendly route decision.
        """
        payload = asdict(self)
        payload["strategy"] = self.strategy.value
        return payload


@dataclass(frozen=True)
class DeclaredTopology:
    """Caller-declared topology facts for the scale-sketch bypass.

    Parameters
    ----------
    topology : str
        One of ``"directed_cyclic"``, ``"directed_acyclic"``, or
        ``"undirected"``.
    depth : int or None
        Optional known DAG depth.
    depth_cap_tripped : bool or None
        Optional known depth-cap result for declared DAGs.
    """

    topology: str
    depth: Optional[int]
    depth_cap_tripped: Optional[bool]


def route(sketch: TopologySketch, config: LayoutConfig) -> RouteDecision:
    """Select the large-graph layout strategy for a topology sketch.

    Parameters
    ----------
    sketch : TopologySketch
        Exact scale sketch for the input graph.
    config : LayoutConfig
        Layout configuration. ``algorithm_params["scale_strategy"]`` may force
        ``NATIVE``, ``LAYERS``, or ``FIELD`` for tests and diagnostics.

    Returns
    -------
    RouteDecision
        Explicit strategy and stable reason codes.
    """
    override = str(config.algorithm_params.get("scale_strategy", "")).upper()
    if override:
        strategy = ScaleStrategy(override)
        return RouteDecision(
            strategy=strategy,
            reason_codes=[f"override_{strategy.value.lower()}"],
            sketch_fingerprint=sketch.fingerprint,
            thresholds_version=THRESHOLDS_VERSION,
        )

    if _has_nontrivial_giant_scc(sketch):
        return RouteDecision(
            strategy=ScaleStrategy.FIELD,
            reason_codes=["nontrivial_giant_scc", "cyclic_field_required"],
            sketch_fingerprint=sketch.fingerprint,
            thresholds_version=THRESHOLDS_VERSION,
        )
    if not sketch.is_acyclic:
        return RouteDecision(
            strategy=ScaleStrategy.FIELD,
            reason_codes=["cyclic_scc", "cyclic_field_required"],
            sketch_fingerprint=sketch.fingerprint,
            thresholds_version=THRESHOLDS_VERSION,
        )
    if sketch.depth_cap_tripped:
        return RouteDecision(
            strategy=ScaleStrategy.FIELD,
            reason_codes=["depth_cap_tripped", "acyclic_hostile_field"],
            sketch_fingerprint=sketch.fingerprint,
            thresholds_version=THRESHOLDS_VERSION,
        )
    if _has_reduction_stall_risk(sketch, config):
        return RouteDecision(
            strategy=ScaleStrategy.FIELD,
            reason_codes=["reduction_stall_risk", "acyclic_hostile_field"],
            sketch_fingerprint=sketch.fingerprint,
            thresholds_version=THRESHOLDS_VERSION,
        )
    return RouteDecision(
        strategy=ScaleStrategy.LAYERS,
        reason_codes=["acyclic", "depth_cap_passed"],
        sketch_fingerprint=sketch.fingerprint,
        thresholds_version=THRESHOLDS_VERSION,
    )


def should_enter_scale_gate(num_nodes: int, num_edges: int, config: LayoutConfig) -> bool:
    """Return whether the default path must sketch and route.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    num_edges : int
        Number of directed edge entries.
    config : LayoutConfig
        Layout configuration that may carry private test threshold overrides in
        ``algorithm_params``.

    Returns
    -------
    bool
        ``True`` when node or edge count crosses the scale gate.
    """
    node_gate = int(config.algorithm_params.get("scale_node_gate", DEFAULT_SCALE_NODE_GATE))
    edge_gate = int(config.algorithm_params.get("scale_edge_gate", DEFAULT_SCALE_EDGE_GATE))
    return int(num_nodes) > node_gate or int(num_edges) > edge_gate


def depth_cap_from_config(config: LayoutConfig) -> int:
    """Return the configured scale-sketch depth cap.

    Parameters
    ----------
    config : LayoutConfig
        Layout configuration that may carry ``scale_depth_cap`` in
        ``algorithm_params``.

    Returns
    -------
    int
        Positive depth cap used by :class:`TopologySketch`.
    """
    return max(1, int(config.algorithm_params.get("scale_depth_cap", DEFAULT_DEPTH_CAP)))


def declared_topology_from_config(config: LayoutConfig) -> Optional[DeclaredTopology]:
    """Return caller-declared topology facts from ``algorithm_params``.

    Parameters
    ----------
    config : LayoutConfig
        Layout configuration that may carry ``scale_declared_topology`` and
        optional declared depth metadata.

    Returns
    -------
    DeclaredTopology or None
        Normalized declaration, or ``None`` when no bypass was requested.
    """
    value = config.algorithm_params.get("scale_declared_topology", None)
    if value is None or str(value).strip() == "":
        return None
    topology = str(value).strip().lower()
    if topology not in DECLARED_TOPOLOGY_VALUES:
        allowed = ", ".join(sorted(DECLARED_TOPOLOGY_VALUES))
        raise ValueError(f"scale_declared_topology must be one of: {allowed}")
    depth = config.algorithm_params.get("scale_declared_depth", None)
    tripped = config.algorithm_params.get("scale_declared_depth_cap_tripped", None)
    return DeclaredTopology(
        topology=topology,
        depth=None if depth is None else int(depth),
        depth_cap_tripped=None if tripped is None else bool(tripped),
    )


def _has_nontrivial_giant_scc(sketch: TopologySketch) -> bool:
    """Return whether the sketch contains a nontrivial giant SCC.

    Parameters
    ----------
    sketch : TopologySketch
        Topology sketch with exact SCC sizes.

    Returns
    -------
    bool
        ``True`` when the largest SCC is cyclic and large enough to dominate
        routing.
    """
    if sketch.largest_scc_size < NONTRIVIAL_SCC_MIN_SIZE:
        return False
    giant_min = max(NONTRIVIAL_SCC_MIN_SIZE, int(sketch.num_nodes * GIANT_SCC_MIN_FRACTION))
    return sketch.largest_scc_size >= giant_min


def _has_reduction_stall_risk(sketch: TopologySketch, config: LayoutConfig) -> bool:
    """Return whether hub degree makes layered coarsening likely to stall.

    Parameters
    ----------
    sketch : TopologySketch
        Topology sketch with degree percentiles.
    config : LayoutConfig
        Layout configuration with optional private test threshold overrides.

    Returns
    -------
    bool
        ``True`` when a dominant hub is large enough to threaten matching-based
        layer coarsening.
    """
    absolute = int(
        config.algorithm_params.get(
            "scale_reduction_stall_degree",
            DEFAULT_REDUCTION_STALL_DEGREE,
        )
    )
    fraction = float(
        config.algorithm_params.get(
            "scale_reduction_stall_fraction",
            DEFAULT_REDUCTION_STALL_FRACTION,
        )
    )
    fractional = max(1, int(max(1, sketch.num_nodes) * fraction))
    threshold = max(1, min(absolute, fractional))
    return sketch.max_degree >= threshold and sketch.max_degree >= 2 * max(1.0, sketch.degree_p90)
