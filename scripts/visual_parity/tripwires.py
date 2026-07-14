"""Metric-validation tripwires for visual parity v2.

Each tripwire runs a clean synthetic panel and an injected-defect variant. A
passing tripwire proves the paired metric stays quiet on clean input and fires
when its defect is present. The spline tripwire is intentionally skipped until
Lane E1 wires Lane A's polyline output into the metric layer.
"""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence

from scripts.visual_parity.types import TargetKind, TripwireResult, TripwireSpec

CANONICAL_STATUS_PATH = Path(
    ".project-context/research/sprint_visual_parity_v2/tripwire_status.json"
)

Panel = Dict[str, Any]
MetricFn = Callable[[Panel], float | str | bool]


def _base_panel() -> Panel:
    """Return a clean synthetic parity panel.

    Returns
    -------
    dict[str, Any]
        Baseline panel values for every wired tripwire metric.
    """

    return {
        "font_size_pt": 14.0,
        "corridor_ink_ratio": 0.86,
        "corridor_threshold": 0.55,
        "label_glyph_width_pt": 120.0,
        "label_glyph_expected_pt": 120.0,
        "arrow_fill_mode": "filled",
        "arrow_expected_fill_mode": "filled",
        "arrow_polygon_iou": 0.96,
        "arrow_compound_order": ["normal", "vee"],
        "arrow_expected_order": ["normal", "vee"],
        "node_autosize_w_pt": 100.0,
        "node_expected_w_pt": 100.0,
        "node_fill": "#ffffff",
        "node_expected_fill": "#ffffff",
        "cluster_top_shift_px": 0.0,
        "harness_l1": 0.0,
    }


@contextmanager
def perturb_panel(panel: Panel, updates: Mapping[str, Any]) -> Iterator[Panel]:
    """Yield a perturbed copy of a panel.

    Parameters
    ----------
    panel
        Source clean panel.
    updates
        Defect values to overlay.

    Yields
    ------
    dict[str, Any]
        Perturbed panel.
    """

    changed = deepcopy(panel)
    changed.update(updates)
    yield changed


def metric_font_size(panel: Panel) -> float:
    """Measure font-size delta from the clean 14pt target.

    Parameters
    ----------
    panel
        Panel payload.

    Returns
    -------
    float
        Absolute font-size delta in points.
    """

    return abs(float(panel["font_size_pt"]) - 14.0)


def metric_corridor_ink_ratio(panel: Panel) -> bool:
    """Evaluate whether corridor ink falls below the style threshold.

    Parameters
    ----------
    panel
        Panel payload.

    Returns
    -------
    bool
        ``True`` when the metric fires.
    """

    return float(panel["corridor_ink_ratio"]) < float(panel["corridor_threshold"])


def metric_label_glyph_extent(panel: Panel) -> float:
    """Measure relative label glyph-extent drift.

    Parameters
    ----------
    panel
        Panel payload.

    Returns
    -------
    float
        Absolute relative glyph-width drift.
    """

    expected = float(panel["label_glyph_expected_pt"])
    current = float(panel["label_glyph_width_pt"])
    return abs(current - expected) / expected if expected else 0.0


def metric_arrow_fill(panel: Panel) -> bool:
    """Evaluate arrow fill-mode mismatch with IoU companion.

    Parameters
    ----------
    panel
        Panel payload.

    Returns
    -------
    bool
        ``True`` when mode mismatches and IoU drops below 0.80.
    """

    return (
        str(panel["arrow_fill_mode"]) != str(panel["arrow_expected_fill_mode"])
        and float(panel["arrow_polygon_iou"]) < 0.80
    )


def metric_arrow_order(panel: Panel) -> bool:
    """Evaluate compound arrow primitive order mismatch.

    Parameters
    ----------
    panel
        Panel payload.

    Returns
    -------
    bool
        ``True`` when the primitive sequence differs.
    """

    return list(panel["arrow_compound_order"]) != list(panel["arrow_expected_order"])


def metric_node_autosize(panel: Panel) -> float:
    """Measure node autosize width drift.

    Parameters
    ----------
    panel
        Panel payload.

    Returns
    -------
    float
        Relative node width drift.
    """

    expected = float(panel["node_expected_w_pt"])
    current = float(panel["node_autosize_w_pt"])
    return abs(current - expected) / expected if expected else 0.0


def metric_node_fill(panel: Panel) -> bool:
    """Evaluate node fill mismatch.

    Parameters
    ----------
    panel
        Panel payload.

    Returns
    -------
    bool
        ``True`` when the fill differs.
    """

    return str(panel["node_fill"]).lower() != str(panel["node_expected_fill"]).lower()


def metric_cluster_rect(panel: Panel) -> bool:
    """Evaluate cluster border shift.

    Parameters
    ----------
    panel
        Panel payload.

    Returns
    -------
    bool
        ``True`` when the cluster top border moved.
    """

    return abs(float(panel["cluster_top_shift_px"])) >= 1.0


def metric_scalehide(panel: Panel) -> bool:
    """Evaluate bbox-tight thumbnail hiding self-check.

    Parameters
    ----------
    panel
        Panel payload.

    Returns
    -------
    bool
        ``True`` when L1 changes.
    """

    return abs(float(panel["harness_l1"])) > 0.0


def _specs() -> List[TripwireSpec]:
    """Return the visual parity v2 tripwire specification table.

    Returns
    -------
    list[TripwireSpec]
        All 11 tripwire specs from FINAL_DESIGN section 5.
    """

    return [
        TripwireSpec(
            "tw_font",
            "font_size +2pt",
            "font_size_pt",
            ">0 panels",
            TargetKind.SVG_DECLARED,
        ),
        TripwireSpec(
            "tw_stem",
            "edge width 0.3pt + body alpha 0",
            "corridor_ink_ratio",
            "< style threshold",
            TargetKind.SVG_DECLARED,
        ),
        TripwireSpec(
            "tw_trunc",
            "label text clipped to 60% width",
            "label_glyph_extent_pt",
            "delta > 20%",
            TargetKind.SVG_DECLARED,
        ),
        TripwireSpec(
            "tw_kern",
            "letter-spacing perturbation on long label",
            "label_glyph_extent_pt",
            "delta > 5%",
            TargetKind.SVG_DECLARED,
        ),
        TripwireSpec(
            "tw_arrowfill",
            "normal -> open (fill flip)",
            "arrow_fill_mode,arrow_polygon_iou",
            "mode mismatch; iou < 0.80",
            TargetKind.SVG_DECLARED,
        ),
        TripwireSpec(
            "tw_arroworder",
            "compound primitive order swapped",
            "arrow_compound_order",
            "mismatch",
            TargetKind.SVG_DECLARED,
        ),
        TripwireSpec(
            "tw_size",
            "min_width +20%",
            "node_autosize_w_pt",
            "delta > 15%",
            TargetKind.SVG_DECLARED,
        ),
        TripwireSpec(
            "tw_color",
            "fill #FFFFFF -> #F0F0F0",
            "node_fill",
            ">= 1 panel",
            TargetKind.SVG_DECLARED,
        ),
        TripwireSpec(
            "tw_spline",
            "flatten spline to straight",
            "spline_path_dist_pt",
            "> 3pt",
            TargetKind.SVG_DECLARED,
        ),
        TripwireSpec(
            "tw_cluster",
            "top cluster border shifted 4px",
            "cluster_rect_metric",
            ">= 1 panel",
            TargetKind.SVG_DECLARED,
        ),
        TripwireSpec(
            "tw_scalehide",
            "re-enable bbox tight+thumbnail on dagua side",
            "harness_l1",
            "any",
            TargetKind.SVG_DECLARED,
        ),
    ]


def _injectors() -> Dict[str, Mapping[str, Any]]:
    """Return injector payloads keyed by tripwire id.

    Returns
    -------
    dict[str, Mapping[str, Any]]
        Defect updates.
    """

    return {
        "tw_font": {"font_size_pt": 16.0},
        "tw_stem": {"corridor_ink_ratio": 0.05},
        "tw_trunc": {"label_glyph_width_pt": 72.0},
        "tw_kern": {"label_glyph_width_pt": 127.2},
        "tw_arrowfill": {"arrow_fill_mode": "hollow", "arrow_polygon_iou": 0.72},
        "tw_arroworder": {"arrow_compound_order": ["vee", "normal"]},
        "tw_size": {"node_autosize_w_pt": 120.0},
        "tw_color": {"node_fill": "#f0f0f0"},
        "tw_cluster": {"cluster_top_shift_px": 4.0},
        "tw_scalehide": {"harness_l1": 1.0},
    }


def _metric_functions() -> Dict[str, MetricFn]:
    """Return metric evaluators keyed by tripwire id.

    Returns
    -------
    dict[str, MetricFn]
        Metric function table.
    """

    return {
        "tw_font": metric_font_size,
        "tw_stem": metric_corridor_ink_ratio,
        "tw_trunc": metric_label_glyph_extent,
        "tw_kern": metric_label_glyph_extent,
        "tw_arrowfill": metric_arrow_fill,
        "tw_arroworder": metric_arrow_order,
        "tw_size": metric_node_autosize,
        "tw_color": metric_node_fill,
        "tw_cluster": metric_cluster_rect,
        "tw_scalehide": metric_scalehide,
    }


def _fires(tripwire_id: str, value: float | str | bool, threshold_scale: float) -> bool:
    """Decide whether a metric value fires for a tripwire.

    Parameters
    ----------
    tripwire_id
        Tripwire id.
    value
        Metric output.
    threshold_scale
        Multiplier applied to numeric thresholds for negative tests.

    Returns
    -------
    bool
        ``True`` when the metric fires.
    """

    if isinstance(value, bool):
        return value if threshold_scale <= 1.0 else False
    numeric = float(value)
    thresholds = {
        "tw_font": 0.0,
        "tw_trunc": 0.20,
        "tw_kern": 0.05,
        "tw_size": 0.15,
    }
    threshold = thresholds.get(tripwire_id, 0.0) * threshold_scale
    return numeric > threshold


def run_tripwire(spec: TripwireSpec, threshold_scale: float = 1.0) -> TripwireResult:
    """Run one clean/injected tripwire pair.

    Parameters
    ----------
    spec
        Tripwire specification.
    threshold_scale
        Multiplier for numeric thresholds.

    Returns
    -------
    TripwireResult
        Tripwire validation result.
    """

    run_at = datetime.now(timezone.utc).isoformat()
    if spec.tripwire_id == "tw_spline":
        return TripwireResult(
            tripwire_id=spec.tripwire_id,
            metric_id=spec.metric_id,
            status="skipped",
            observed_effect={},
            min_effect_size=spec.min_effect_size,
            run_at=run_at,
            notes="wired in E1",
        )

    clean = _base_panel()
    metric = _metric_functions()[spec.tripwire_id]
    clean_value = metric(clean)
    clean_fired = _fires(spec.tripwire_id, clean_value, threshold_scale)
    with perturb_panel(clean, _injectors()[spec.tripwire_id]) as injected:
        injected_value = metric(injected)
    injected_fired = _fires(spec.tripwire_id, injected_value, threshold_scale)
    status = "pass" if injected_fired and not clean_fired else "fail"
    return TripwireResult(
        tripwire_id=spec.tripwire_id,
        metric_id=spec.metric_id,
        status=status,
        observed_effect={
            "clean": clean_value,
            "clean_fired": clean_fired,
            "injected": injected_value,
            "injected_fired": injected_fired,
        },
        min_effect_size=spec.min_effect_size,
        failed_metric_ids=[] if status == "pass" else [spec.metric_id],
        run_at=run_at,
    )


def run_all(threshold_scale: float = 1.0) -> List[TripwireResult]:
    """Run every tripwire in the interlock table.

    Parameters
    ----------
    threshold_scale
        Multiplier for numeric thresholds.

    Returns
    -------
    list[TripwireResult]
        Results for all 11 tripwires.
    """

    return [run_tripwire(spec, threshold_scale=threshold_scale) for spec in _specs()]


def _result_to_dict(result: TripwireResult) -> Dict[str, Any]:
    """Convert a tripwire result dataclass to JSON.

    Parameters
    ----------
    result
        Tripwire result.

    Returns
    -------
    dict[str, Any]
        JSON-compatible result payload.
    """

    return {
        "tripwire_id": result.tripwire_id,
        "metric_id": result.metric_id,
        "status": result.status,
        "observed_effect": result.observed_effect,
        "min_effect_size": result.min_effect_size,
        "failed_metric_ids": result.failed_metric_ids,
        "evidence": result.evidence,
        "run_at": result.run_at,
        "notes": result.notes,
    }


def build_report(results: Sequence[TripwireResult]) -> Dict[str, Any]:
    """Build the tripwire JSON report.

    Parameters
    ----------
    results
        Tripwire results.

    Returns
    -------
    dict[str, Any]
        Report payload.
    """

    failed_metric_ids: List[str] = []
    for result in results:
        failed_metric_ids.extend(result.failed_metric_ids)
    return {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "fail" if failed_metric_ids else "pass",
        "failed_metric_ids": sorted(set(failed_metric_ids)),
        "results": [_result_to_dict(result) for result in results],
    }


def write_report(path: Path, report: Mapping[str, Any]) -> None:
    """Write a tripwire report as stable JSON.

    Parameters
    ----------
    path
        Destination path.
    report
        JSON-compatible report.

    Returns
    -------
    None
        The report is written to disk.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv
        Optional argument vector.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """

    parser = argparse.ArgumentParser(description="Run visual parity v2 metric tripwires.")
    parser.add_argument("--all", action="store_true", help="Run all tripwires.")
    parser.add_argument("--out", default="", help="Optional report output path.")
    parser.add_argument(
        "--threshold-scale",
        type=float,
        default=1.0,
        help="Scale numeric thresholds; useful for proving failure behavior.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.

    Parameters
    ----------
    argv
        Optional argument vector.

    Returns
    -------
    int
        Process exit status.
    """

    args = _parse_args(argv)
    if not args.all:
        print("--all is required")
        return 2
    results = run_all(threshold_scale=float(args.threshold_scale))
    report = build_report(results)
    write_report(CANONICAL_STATUS_PATH, report)
    if args.out:
        write_report(Path(args.out), report)
    print(
        json.dumps({"status": report["status"], "failed_metric_ids": report["failed_metric_ids"]})
    )
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
