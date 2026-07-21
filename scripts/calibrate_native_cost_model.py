"""Fit frozen native modeled-cost constants from existing telemetry files.

This script is intentionally offline-only: it reads harvested idle corpus logs
and JSONL telemetry, then emits a candidate constants table and provenance
document. It does not run benchmarks or measure the current process.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import platform
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

RUNTIME_RE = re.compile(
    r"(?P<scope>Undirected|Directed)\s+candidate\s+runtime\s+"
    r"family=(?P<family>[A-Za-z0-9_.:-]+).*?"
    r"(?:cpu_seconds|seconds)=(?P<seconds>[0-9.]+)"
)
DEFAULT_ENVELOPE_FACTORS: dict[str, float] = {
    "fcose": 2.0,
    "stress": 1.8,
    "arm_s": 1.8,
    "apsp": 1.5,
    "ruler": 1.5,
    "w5": 1.2,
    "opaque": 2.0,
}
# Mirrors FCoSEConfig.max_exact_repulsion_nodes: at or below this node count the
# fCoSE spring embedder uses exact vectorized pairwise repulsion (cheap,
# overhead-dominated); above it the per-step Barnes-Hut tree pass dominates and
# real cost jumps by 3-4 orders of magnitude (C2 telemetry, 2026-07-21).
FCOSE_EXACT_REPULSION_NODE_CAP = 512
# Per-term fit modes. "intercept_slope" subtracts the raw P10 overhead before
# computing rates, so full-arm reference-step samples with heterogeneous
# volumes do not double-count the flat per-arm overhead into alpha (the C1
# "p90_rate" fit anchored alpha on tiny-row rates, over-pricing large exact
# rows ~200x). "rate" is a pure zero-intercept fit for short-step samples that
# extrapolate linearly in planned steps (Barnes-Hut regime). Everything else
# keeps the C1 "p90_rate" methodology.
TERM_FIT_MODES: dict[str, str] = {
    "force": "intercept_slope",
    "force_bh": "rate",
}


@dataclass(frozen=True)
class TelemetryRecord:
    """One observed native runtime sample.

    Parameters
    ----------
    family : str
        Native family label.
    term : str
        Cost-model term label.
    seconds : float
        Observed idle wall seconds.
    volume : float
        Deterministic unit volume for the term.
    source : str
        Source file path or logical source.
    device_class : str
        Device class axis, usually ``"cpu"`` or ``"cuda"``.
    """

    family: str
    term: str
    seconds: float
    volume: float
    source: str
    device_class: str = "cpu"


def _positive_float(value: object, default: float = 0.0) -> float:
    """Return ``value`` as a non-negative finite float.

    Parameters
    ----------
    value : object
        Candidate numeric value.
    default : float, default=0.0
        Fallback when conversion fails.

    Returns
    -------
    float
        Non-negative finite value.
    """
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return max(0.0, float(default))
    if not math.isfinite(result):
        return max(0.0, float(default))
    return max(0.0, result)


def _family_from_payload(payload: Mapping[str, Any]) -> str:
    """Return the best native family label in a telemetry payload.

    Parameters
    ----------
    payload : Mapping[str, Any]
        JSON telemetry record.

    Returns
    -------
    str
        Normalized family label.
    """
    for key in ("family", "candidate_family", "winner_family", "arm_family", "mode"):
        value = payload.get(key)
        if value:
            return str(value).lower()
    name = str(payload.get("name", payload.get("candidate", "opaque"))).lower()
    if "w5" in name:
        return "w5"
    if "arm_s" in name or "stress_k" in name:
        return "arm_s"
    if "fcose" in name:
        return "fcose"
    return "opaque"


def _term_from_payload(payload: Mapping[str, Any], family: str) -> str:
    """Return a cost-model term label for a telemetry payload.

    Parameters
    ----------
    payload : Mapping[str, Any]
        JSON telemetry record.
    family : str
        Normalized family label.

    Returns
    -------
    str
        Term label compatible with ``native_cost_model.FROZEN_COST_TABLE``.
    """
    event = str(payload.get("event", payload.get("phase", ""))).lower()
    if family == "w5":
        if "referee" in event or "score" in event:
            return "referee"
        return "step"
    if "score" in event or "referee" in event:
        return "score"
    if family == "fcose":
        num_nodes = int(_positive_float(payload.get("num_nodes", payload.get("n", 0)), 0.0))
        if num_nodes > FCOSE_EXACT_REPULSION_NODE_CAP:
            return "force_bh"
        return "force"
    if family == "arm_s":
        return "stress_pairs"
    if family == "apsp":
        return "apsp"
    if family == "ruler":
        return "samples"
    return "full_arm"


def _volume_from_payload(payload: Mapping[str, Any], family: str, term: str) -> float:
    """Return deterministic volume encoded by a telemetry payload.

    Parameters
    ----------
    payload : Mapping[str, Any]
        JSON telemetry record.
    family : str
        Normalized family label.
    term : str
        Cost-model term label.

    Returns
    -------
    float
        Positive unit volume. Missing historical logs default to one so the
        fitted term becomes a conservative flat prior.
    """
    for key in ("volume", "work_volume", "unit_volume", f"{term}_volume"):
        if key in payload:
            return max(1.0, _positive_float(payload[key], 1.0))
    steps = int(_positive_float(payload.get("steps", payload.get("planned_steps", 1)), 1.0))
    seeds = int(_positive_float(payload.get("seeds", payload.get("seed_count", 1)), 1.0))
    checkpoints = int(
        _positive_float(payload.get("checkpoints", payload.get("checkpoint_count", 1)), 1.0)
    )
    num_nodes = int(_positive_float(payload.get("num_nodes", payload.get("n", 0)), 0.0))
    num_edges = int(_positive_float(payload.get("num_edges", payload.get("e", 0)), 0.0))
    if family == "w5":
        return float(max(steps * seeds if term == "step" else checkpoints * seeds, 1))
    if term in {"force", "force_bh", "samples"}:
        base = max(num_edges * num_edges, num_nodes) if term == "samples" else num_nodes + num_edges
        return float(max(base * max(steps, 1), 1))
    if term in {"pairs", "stress_pairs"}:
        pairs = num_nodes * max(num_nodes - 1, 0) // 2
        return float(max(pairs * max(steps, 1), 1))
    if term == "apsp":
        return float(max(num_nodes * (num_nodes + num_edges), 1))
    return 1.0


def records_from_text(path: Path) -> list[TelemetryRecord]:
    """Parse native runtime records from a text log.

    Parameters
    ----------
    path : Path
        Log file containing existing native runtime telemetry lines.

    Returns
    -------
    list[TelemetryRecord]
        Parsed runtime records.
    """
    records: list[TelemetryRecord] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = RUNTIME_RE.search(line)
        if match is None:
            continue
        family = match.group("family").lower()
        term = "force" if family == "fcose" else "full_arm"
        records.append(
            TelemetryRecord(
                family=family,
                term=term,
                seconds=_positive_float(match.group("seconds")),
                volume=1.0,
                source=str(path),
            )
        )
    return records


def records_from_jsonl(path: Path) -> list[TelemetryRecord]:
    """Parse native marketplace and W5 JSONL telemetry records.

    Parameters
    ----------
    path : Path
        JSONL telemetry file from idle corpus runs.

    Returns
    -------
    list[TelemetryRecord]
        Parsed runtime records.
    """
    records: list[TelemetryRecord] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        seconds = _positive_float(
            payload.get("seconds", payload.get("wall_s", payload.get("runtime_seconds", 0.0)))
        )
        if seconds <= 0.0:
            continue
        family = _family_from_payload(payload)
        term = _term_from_payload(payload, family)
        records.append(
            TelemetryRecord(
                family=family,
                term=term,
                seconds=seconds,
                volume=_volume_from_payload(payload, family, term),
                source=str(path),
                device_class=str(payload.get("device_class", payload.get("device", "cpu"))).lower(),
            )
        )
    return records


def load_records(paths: Sequence[Path]) -> list[TelemetryRecord]:
    """Load telemetry records from JSONL and text paths.

    Parameters
    ----------
    paths : Sequence[Path]
        Input telemetry files.

    Returns
    -------
    list[TelemetryRecord]
        Combined telemetry records.
    """
    records: list[TelemetryRecord] = []
    for path in paths:
        if path.suffix.lower() == ".jsonl":
            records.extend(records_from_jsonl(path))
        else:
            records.extend(records_from_text(path))
    return records


def percentile(values: Sequence[float], q: float) -> float:
    """Return a nearest-rank percentile.

    Parameters
    ----------
    values : Sequence[float]
        Numeric sample values.
    q : float
        Quantile in ``[0, 1]``.

    Returns
    -------
    float
        Percentile value, or zero for an empty sample.
    """
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, math.ceil(float(q) * len(ordered)) - 1))
    return ordered[index]


def fit_term(
    records: Sequence[TelemetryRecord],
    envelope_factor: float,
    mode: str = "p90_rate",
) -> tuple[float, float]:
    """Fit a conservative ``alpha, beta`` pair for one term.

    Parameters
    ----------
    records : Sequence[TelemetryRecord]
        Runtime observations for one family/device/term.
    envelope_factor : float
        Per-family load-envelope multiplier applied after the idle fit.
    mode : str, default="p90_rate"
        Fit shape. ``"p90_rate"`` is the C1 methodology (P90 raw rate plus P10
        overhead; both carry the overhead, which is conservative for
        homogeneous-volume samples). ``"intercept_slope"`` subtracts the raw
        P10 overhead before computing rates so heterogeneous full-arm samples
        do not fold the flat per-arm overhead into ``alpha``.
        ``"rate"`` is a zero-intercept P90-rate fit for short-step samples
        whose cost extrapolates linearly in planned steps.

    Returns
    -------
    tuple[float, float]
        ``alpha`` and ``beta`` constants for ``alpha * volume + beta``.
    """
    if not records:
        return (0.0, 0.0)
    envelope = max(float(envelope_factor), 0.0)
    seconds = [record.seconds for record in records]
    if mode == "rate":
        rates = [record.seconds / max(record.volume, 1.0) for record in records]
        return (percentile(rates, 0.90) * envelope, 0.0)
    if mode == "intercept_slope":
        raw_beta = percentile(seconds, 0.10)
        residual_rates = [
            max(0.0, record.seconds - raw_beta) / max(record.volume, 1.0) for record in records
        ]
        return (percentile(residual_rates, 0.90) * envelope, raw_beta * envelope)
    rates = [record.seconds / max(record.volume, 1.0) for record in records]
    alpha = percentile(rates, 0.90) * envelope
    beta = percentile(seconds, 0.10) * envelope
    return (alpha, beta)


def fit_table(
    records: Sequence[TelemetryRecord],
    envelope_factors: Mapping[str, float],
) -> dict[tuple[str, str], dict[str, tuple[float, float]]]:
    """Fit a frozen constants table from telemetry records.

    Parameters
    ----------
    records : Sequence[TelemetryRecord]
        Runtime observations.
    envelope_factors : Mapping[str, float]
        Per-family load-envelope multipliers.

    Returns
    -------
    dict[tuple[str, str], dict[str, tuple[float, float]]]
        Frozen table candidate keyed by ``(family, device_class)``.
    """
    grouped: defaultdict[tuple[str, str, str], list[TelemetryRecord]] = defaultdict(list)
    for record in records:
        device = "cuda" if "cuda" in record.device_class else "cpu"
        grouped[(record.family, device, record.term)].append(record)
    table: dict[tuple[str, str], dict[str, tuple[float, float]]] = defaultdict(dict)
    for (family, device, term), term_records in grouped.items():
        factor = envelope_factors.get(family, envelope_factors.get("opaque", 2.0))
        mode = TERM_FIT_MODES.get(term, "p90_rate")
        table[(family, device)][term] = fit_term(term_records, factor, mode)
    return dict(table)


def render_table(table: Mapping[tuple[str, str], Mapping[str, tuple[float, float]]]) -> str:
    """Render a Python literal for the frozen cost table.

    Parameters
    ----------
    table : Mapping[tuple[str, str], Mapping[str, tuple[float, float]]]
        Fitted table candidate.

    Returns
    -------
    str
        Python assignment text.
    """
    lines = ["FROZEN_COST_TABLE = {"]
    for family_device in sorted(table):
        family, device = family_device
        lines.append(f'    ("{family}", "{device}"): {{')
        for term, (alpha, beta) in sorted(table[family_device].items()):
            lines.append(f'        "{term}": ({alpha:.12g}, {beta:.12g}),')
        lines.append("    },")
    lines.append("}")
    return "\n".join(lines) + "\n"


def write_provenance(
    path: Path,
    records: Sequence[TelemetryRecord],
    envelope_factors: Mapping[str, float],
) -> None:
    """Write a provenance document for a fitted constants table.

    Parameters
    ----------
    path : Path
        Output Markdown path.
    records : Sequence[TelemetryRecord]
        Runtime observations used in the fit.
    envelope_factors : Mapping[str, float]
        Per-family load-envelope multipliers.

    Returns
    -------
    None
        The document is written to ``path``.
    """
    sources = sorted({record.source for record in records})
    lines = [
        "# Native Cost Model Calibration Provenance",
        "",
        f"- generated_at_utc: {dt.datetime.now(dt.timezone.utc).isoformat()}",
        f"- host: {platform.node()}",
        f"- platform: {platform.platform()}",
        "- envelope_definition: P90 idle runtime rates multiplied by per-family "
        "half-cores sibling-load envelope factors",
        f"- records: {len(records)}",
        "- input_sources:",
    ]
    lines.extend(f"  - {source}" for source in sources)
    lines.append("- envelope_factors:")
    lines.extend(f"  - {family}: {factor}" for family, factor in sorted(envelope_factors.items()))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_envelope_factor(values: Optional[Sequence[str]]) -> dict[str, float]:
    """Parse ``family=factor`` CLI overrides.

    Parameters
    ----------
    values : Sequence[str], optional
        CLI override strings.

    Returns
    -------
    dict[str, float]
        Envelope factors merged over defaults.
    """
    factors = dict(DEFAULT_ENVELOPE_FACTORS)
    for value in values or ():
        family, separator, factor = value.partition("=")
        if not separator:
            raise ValueError(f"invalid envelope factor {value!r}; expected family=factor")
        factors[family.lower()] = _positive_float(factor)
    return factors


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser for offline calibration inputs and outputs.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Existing telemetry log/JSONL files")
    parser.add_argument("--output-table", required=True, type=Path, help="Python table output path")
    parser.add_argument(
        "--output-provenance",
        required=True,
        type=Path,
        help="Markdown provenance output path",
    )
    parser.add_argument(
        "--envelope-factor",
        action="append",
        default=None,
        help="Override per-family envelope multiplier, e.g. fcose=2.0",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run offline calibration table fitting.

    Parameters
    ----------
    argv : Sequence[str], optional
        CLI arguments excluding the program name.

    Returns
    -------
    int
        Process exit code.
    """
    args = build_parser().parse_args(argv)
    envelope_factors = parse_envelope_factor(args.envelope_factor)
    records = load_records(args.inputs)
    table = fit_table(records, envelope_factors)
    args.output_table.write_text(render_table(table), encoding="utf-8")
    write_provenance(args.output_provenance, records, envelope_factors)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
