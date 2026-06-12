#!/usr/bin/env python3
"""Assemble the r70 definitive fidelity report and four-tier categorization."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import torch
from scipy import stats

from dagua.eval import distributional_fidelity as df
from dagua.eval.equivalence_metrics import compute_equivalence_metrics
from dagua.eval.graphs import get_test_graphs
from dagua.eval.variants import get_variant, original_variant_name

SPEC_VERSION = "r70-v6"
DEFAULT_PER_COMBO = Path("eval_output/fidelity_definitive/per_combo.jsonl")
DEFAULT_OUTPUT_DIR = Path("eval_output/fidelity_definitive")
DEFAULT_FAILING_MAP = Path(".project-context/research/sprint_rng_matching/failing_map_final.json")
DEFAULT_TRIAGE = Path("eval_output/fidelity_report_final/triage_final.md")
DEFAULT_FIVE_SEED = Path("eval_output/benchmark_5seed_final/results.json")
DEFAULT_DATA_DIR = Path("eval_output/benchmark_100seed_escalation_final")
DEFAULT_REFRESH = Path("eval_output/benchmark_5seed_deterministic_refresh")
DEFAULT_CONTROLS = Path("eval_output/fidelity_definitive/controls")
REPORT_NAME = "DEFINITIVE_FIDELITY_REPORT.md"
TIERS_NAME = "FOUR_TIER_CATEGORIZATION.md"
INVARIANCE_THRESHOLD = 1.0e-3
SPOTCHECK_MAX_COMBOS = 200
SPOTCHECK_MAX_SEED_PAIRS = 25
SPOTCHECK_MAX_AUTOMORPHISMS = 5_000
SPOTCHECK_PAIR_TIMEOUT_SECONDS = 10
SIZE_BINS = (
    ("N <= 50", 0, 50),
    ("51-200", 51, 200),
    ("201-1000", 201, 1000),
    (">1000", 1001, None),
)
PASSING_RUNGS = {"0", "1", "2", "2'", "3"}
MODE_A_PASS_RUNGS = {"0", "1", "2"}
MODE_B_PASS_RUNGS = {"0", "2'"}
TIMEOUT_RE = re.compile("timeout", re.IGNORECASE)
HIERARCHY_ENGINE_PREFIXES = (
    "classic_sugiyama",
    "classic_reingold_tilford",
    "classic_rt",
)
TRIAGE_CATEGORIES = {
    "BIT_IDENTICAL",
    "ESCALATE_STOCHASTIC",
    "DETERMINISTIC_DIFFERENT",
    "TIMEOUT",
    "NO_REFERENCE",
    "UNVERDICTED_OTHER",
}


@dataclass
class ReportContext:
    """Mutable state shared across report assembly.

    Parameters
    ----------
    strict : bool
        Whether failed assertions should terminate report generation.
    warnings : list[str]
        Non-fatal assertion and data-quality messages.
    """

    strict: bool
    warnings: list[str] = field(default_factory=list)

    def check(self, condition: bool, message: str) -> None:
        """Apply a report assertion.

        Parameters
        ----------
        condition : bool
            Assertion predicate.
        message : str
            Human-readable assertion failure.

        Returns
        -------
        None
            Raises in strict mode or records a warning in no-strict mode.
        """
        if condition:
            return
        if self.strict:
            raise AssertionError(message)
        self.warnings.append(message)


@dataclass(frozen=True)
class TriageInventory:
    """Variant inventory parsed from the final 5-seed triage report.

    Parameters
    ----------
    categories : dict[str, list[str]]
        Mapping from triage heading to variant identifiers.
    variant_to_category : dict[str, str]
        Reverse mapping from variant identifier to exactly one heading.
    """

    categories: dict[str, list[str]]
    variant_to_category: dict[str, str]


@dataclass(frozen=True)
class GraphInfo:
    """Graph metadata needed for domain and size reporting.

    Parameters
    ----------
    name : str
        Graph identifier.
    num_nodes : Optional[int]
        Node count, if available without executing layout engines.
    has_hierarchy : bool
        Whether the source generator semantically produces a DAG/tree/pipeline.
    source : str
        Graph source metadata.
    tags : list[str]
        Graph tag metadata.
    reason : str
        Short classification reason.
    disconnected_hint : bool
        Whether metadata or graph structure indicates multiple components.
    """

    name: str
    num_nodes: Optional[int]
    has_hierarchy: bool
    source: str
    tags: list[str]
    reason: str
    disconnected_hint: bool


@dataclass(frozen=True)
class PositionRow:
    """Compact benchmark row used for spot-check position resolution.

    Parameters
    ----------
    key : str
        Original ``results.json`` key.
    graph : str
        Graph name.
    engine : str
        Engine name.
    seed : Optional[int]
        Integer seed for seeded layouts, otherwise ``None``.
    status : str
        Benchmark status.
    positions_file : Optional[str]
        Stored position path from the benchmark row.
    num_nodes : Optional[int]
        Node count reported by the benchmark row.
    """

    key: str
    graph: str
    engine: str
    seed: Optional[int]
    status: str
    positions_file: Optional[str]
    num_nodes: Optional[int]


@dataclass(frozen=True)
class AccountingEntry:
    """One graph-level accounting disposition for an engine.

    Parameters
    ----------
    engine : str
        Variant identifier.
    graph : str
        Graph identifier.
    bucket : str
        First-match accounting bucket.
    final_rung : Optional[str]
        Rung when the bucket maps to a rung.
    reason : str
        Supporting reason for the bucket.
    on_domain : bool
        Whether the engine is on-domain for the graph.
    mode : Optional[str]
        Final Mode A/B classification when known.
    informative : bool
        Whether the combo is informative for Mode B REF_COMPATIBLE.
    seed_na : bool
        Whether seed-faithfulness denominator excludes this combo.
    num_nodes : Optional[int]
        Node count used for size-bin aggregation.
    runtime_ratio : Optional[float]
        Median reimpl/reference runtime ratio.
    annotations : list[str]
        Report-stage annotations.
    """

    engine: str
    graph: str
    bucket: str
    final_rung: Optional[str]
    reason: str
    on_domain: bool
    mode: Optional[str]
    informative: bool
    seed_na: bool
    num_nodes: Optional[int]
    runtime_ratio: Optional[float]
    annotations: list[str]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed CLI options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-combo", type=Path, default=DEFAULT_PER_COMBO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--failing-map", type=Path, default=DEFAULT_FAILING_MAP)
    parser.add_argument("--triage", type=Path, default=DEFAULT_TRIAGE)
    parser.add_argument("--five-seed-results", type=Path, default=DEFAULT_FIVE_SEED)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--refresh-dir", type=Path, default=DEFAULT_REFRESH)
    parser.add_argument("--controls-dir", type=Path, default=DEFAULT_CONTROLS)
    parser.add_argument("--spec-version", default=SPEC_VERSION)
    parser.add_argument("--no-strict", action="store_true")
    parser.add_argument("--controls", action="store_true", help="Evaluate control gates only.")
    return parser.parse_args()


def main() -> int:
    """Run report generation or controls gate evaluation.

    Returns
    -------
    int
        Process exit status.
    """
    args = parse_args()
    context = ReportContext(strict=not args.no_strict)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.controls:
        results = evaluate_controls(args.controls_dir, args.output_dir, context)
        print(json.dumps(results["summary"], indent=2, sort_keys=True))
        return 0 if results["all_passed"] else 1

    state = build_report_state(args, context)
    write_outputs(state, args.output_dir)
    headers = extract_headers(args.output_dir / REPORT_NAME) + extract_headers(
        args.output_dir / TIERS_NAME
    )
    print("Rendered section headers:")
    for header in headers:
        print(header)
    if context.warnings:
        print("\nNon-strict assertion warnings:")
        for warning in context.warnings:
            print(f"- {warning}")
    return 0


def build_report_state(args: argparse.Namespace, context: ReportContext) -> dict[str, Any]:
    """Build all report tables and serialized outputs.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI options.
    context : ReportContext
        Assertion and warning state.

    Returns
    -------
    dict[str, Any]
        Complete report state.
    """
    rows = read_jsonl(args.per_combo)
    failing_map = load_failing_map(args.failing_map)
    triage = parse_triage(args.triage)
    five_seed = load_json(args.five_seed_results) if args.five_seed_results.exists() else {}
    graph_info = load_graph_info(five_seed, rows)
    finalized_rows = finalize_rows(rows, args.spec_version)
    check_completeness(finalized_rows, failing_map, args.spec_version, context)
    check_no_mixed_modes(finalized_rows, context)
    deterministic_rows = read_jsonl(args.refresh_dir / "deterministic.jsonl")
    rung0_reverify_rows = read_jsonl(args.refresh_dir / "rung0-reverify.jsonl")
    accounting = build_accounting(
        finalized_rows,
        deterministic_rows,
        rung0_reverify_rows,
        failing_map,
        triage,
        five_seed,
        graph_info,
        context,
    )
    aggregation = aggregate_accounting(accounting, finalized_rows, triage, graph_info)
    headlines = assign_headlines(accounting, triage)
    spotcheck = run_invariance_spotcheck(finalized_rows, graph_info, args.data_dir, context)
    controls = load_existing_controls_summary(args.output_dir)
    oc_path = args.output_dir / "oc_simulation.json"
    oc_simulation = ensure_oc_simulation(oc_path, context.strict)
    per_combo_json = args.output_dir / "per_combo.json"
    per_combo_json.write_text(json.dumps(finalized_rows, indent=2, sort_keys=True))
    check_variant_partition(triage, context)
    return {
        "args": args,
        "rows": finalized_rows,
        "failing_map": failing_map,
        "triage": triage,
        "five_seed": five_seed,
        "graph_info": graph_info,
        "accounting": accounting,
        "aggregation": aggregation,
        "headlines": headlines,
        "spotcheck": spotcheck,
        "controls": controls,
        "oc_simulation": oc_simulation,
        "warnings": context.warnings,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL rows if a file exists.

    Parameters
    ----------
    path : pathlib.Path
        JSONL path.

    Returns
    -------
    list[dict[str, Any]]
        Parsed rows, or an empty list for absent files.
    """
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open() as file_obj:
        for line_no, line in enumerate(file_obj, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def load_json(path: Path) -> Any:
    """Load a JSON file.

    Parameters
    ----------
    path : pathlib.Path
        JSON path.

    Returns
    -------
    Any
        Parsed JSON payload.
    """
    with path.open() as file_obj:
        return json.load(file_obj)


def load_failing_map(path: Path) -> dict[str, dict[str, Any]]:
    """Load the escalation failing map.

    Parameters
    ----------
    path : pathlib.Path
        Failing-map JSON path.

    Returns
    -------
    dict[str, dict[str, Any]]
        Mapping from engine to reference and graph list.
    """
    if not path.exists():
        return {}
    payload = load_json(path)
    return {str(engine): dict(value) for engine, value in payload.items()}


def finalize_rows(
    rows: list[dict[str, Any]], spec_version: str, include_controls: bool = False
) -> list[dict[str, Any]]:
    """Apply full-run FDR families and final rung assignment.

    Parameters
    ----------
    rows : list[dict[str, Any]]
        Raw Task B rows.
    spec_version : str
        Required spec version.

    Returns
    -------
    list[dict[str, Any]]
        Rows with q-values, final rungs, and annotations.
    """
    eligible = [
        dict(row) for row in rows if include_controls or row.get("control_kind") in (None, "")
    ]
    for row in eligible:
        row.setdefault("q_track", None)
        row.setdefault("q_tost", None)
    apply_bh_family(eligible, "p_track", "q_track")
    apply_bh_family(eligible, "stress_p_tost", "q_tost")
    typ_count = 0
    typ_raw_fail = 0
    for row in eligible:
        if row.get("spec_version") != spec_version:
            row["final_rung"] = "INSUFFICIENT_DATA"
            row["final_annotations"] = ["spec_version_mismatch"]
            continue
        p_typ = as_float(row.get("p_typ"))
        if p_typ is not None:
            typ_count += 1
            typ_raw_fail += int(p_typ <= 0.05)
            row["not_typical_raw"] = bool(p_typ <= 0.05)
        rung, annotations = df.assign_rung(row)
        row["final_rung"] = rung
        row["final_annotations"] = annotations
    expected_false = 0.05 * typ_count
    for row in eligible:
        row["typicality_family_size"] = typ_count
        row["typicality_raw_fail_count"] = typ_raw_fail
        row["expected_false_atypicality"] = expected_false
    return sorted(eligible, key=lambda item: str(item.get("combo_id", "")))


def apply_bh_family(rows: list[dict[str, Any]], p_key: str, q_key: str) -> None:
    """Apply Benjamini-Hochberg q-values to a row family.

    Parameters
    ----------
    rows : list[dict[str, Any]]
        Rows to update in place.
    p_key : str
        Raw p-value field.
    q_key : str
        Output q-value field.

    Returns
    -------
    None
        Rows are updated in place.
    """
    indexed: list[tuple[int, float]] = []
    for idx, row in enumerate(rows):
        value = as_float(row.get(p_key))
        if value is not None:
            indexed.append((idx, value))
    q_values = df.bh_fdr([value for _idx, value in indexed])
    for (idx, _value), q_value in zip(indexed, q_values):
        rows[idx][q_key] = float(q_value)


def check_completeness(
    rows: list[dict[str, Any]],
    failing_map: dict[str, dict[str, Any]],
    spec_version: str,
    context: ReportContext,
) -> None:
    """Refuse full-run FDR unless every escalation combo is present once.

    Parameters
    ----------
    rows : list[dict[str, Any]]
        Finalized non-control rows.
    failing_map : dict[str, dict[str, Any]]
        Required escalation combo map.
    spec_version : str
        Required row spec version.
    context : ReportContext
        Assertion and warning state.

    Returns
    -------
    None
        Raises or records assertion failures.
    """
    required = {
        f"{graph}::{engine}"
        for engine, payload in failing_map.items()
        for graph in payload.get("graphs", [])
    }
    counts = Counter(
        str(row.get("combo_id")) for row in rows if row.get("spec_version") == spec_version
    )
    missing = sorted(combo for combo in required if counts.get(combo, 0) != 1)
    extras = sorted(combo for combo, count in counts.items() if count > 1)
    context.check(not missing, f"FDR completeness failed: {len(missing)} missing/duplicate combos.")
    context.check(not extras, f"FDR completeness failed: duplicate rows {extras[:10]}.")


def check_no_mixed_modes(rows: list[dict[str, Any]], context: ReportContext) -> None:
    """Assert that no engine has mixed Mode A/Mode B escalation rows.

    Parameters
    ----------
    rows : list[dict[str, Any]]
        Finalized rows.
    context : ReportContext
        Assertion and warning state.

    Returns
    -------
    None
        Raises or records assertion failures.
    """
    modes: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        mode = row.get("mode")
        if mode in {"A", "B"} and not row.get("insufficient_data", False):
            modes[str(row.get("engine"))].add(str(mode))
    mixed = {engine: sorted(values) for engine, values in modes.items() if len(values) > 1}
    context.check(not mixed, f"Mixed reference modes detected: {mixed}")


def parse_triage(path: Path) -> TriageInventory:
    """Parse the final triage Markdown inventory.

    Parameters
    ----------
    path : pathlib.Path
        Triage Markdown path.

    Returns
    -------
    TriageInventory
        Parsed category and reverse maps.
    """
    categories: dict[str, list[str]] = defaultdict(list)
    if not path.exists():
        return TriageInventory({}, {})
    current: Optional[str] = None
    heading_re = re.compile(r"^## ([A-Z_]+)(?:\s+\(\d+\))?\s*$")
    variant_re = re.compile(r"^- `([^`]+)`")
    for line in path.read_text().splitlines():
        heading = heading_re.match(line)
        if heading:
            candidate = heading.group(1)
            current = candidate if candidate in TRIAGE_CATEGORIES else None
            if current is not None:
                categories.setdefault(current, [])
            continue
        variant = variant_re.match(line)
        if current and variant:
            categories[current].append(variant.group(1))
    reverse: dict[str, str] = {}
    for category, engines in categories.items():
        for engine in engines:
            if engine in reverse:
                raise ValueError(f"Variant {engine} appears in multiple triage categories.")
            reverse[engine] = category
    return TriageInventory(dict(categories), reverse)


def check_variant_partition(triage: TriageInventory, context: ReportContext) -> None:
    """Assert that the triage inventory contains all 118 variants once.

    Parameters
    ----------
    triage : TriageInventory
        Parsed triage inventory.
    context : ReportContext
        Assertion and warning state.

    Returns
    -------
    None
        Raises or records assertion failures.
    """
    count = len(triage.variant_to_category)
    context.check(count == 118, f"Triage inventory has {count} variants, expected 118.")


def load_graph_info(
    five_seed: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, GraphInfo]:
    """Load graph metadata for domain and size aggregation.

    Parameters
    ----------
    five_seed : dict[str, Any]
        5-seed benchmark result rows.
    rows : list[dict[str, Any]]
        Per-combo rows.

    Returns
    -------
    dict[str, GraphInfo]
        Metadata keyed by graph name.
    """
    info: dict[str, GraphInfo] = {}
    try:
        test_graphs = get_test_graphs()
    except Exception:
        test_graphs = []
    for test_graph in test_graphs:
        graph = test_graph.graph
        edge_index = getattr(graph, "edge_index", None)
        disconnected = "disconnected" in test_graph.name or "disconnected" in test_graph.tags
        info[test_graph.name] = GraphInfo(
            name=test_graph.name,
            num_nodes=int(getattr(graph, "num_nodes", 0)),
            has_hierarchy=classify_graph_hierarchy(
                test_graph.name, test_graph.source, test_graph.tags
            ),
            source=str(test_graph.source),
            tags=sorted(str(tag) for tag in test_graph.tags),
            reason=hierarchy_reason(test_graph.name, test_graph.source, test_graph.tags),
            disconnected_hint=bool(disconnected or edge_index is None),
        )
    for row in five_seed.values():
        graph = str(row.get("graph_name") or row.get("graph") or "")
        if not graph or graph in info:
            continue
        n_nodes = as_int(row.get("num_nodes"))
        info[graph] = GraphInfo(
            name=graph,
            num_nodes=n_nodes,
            has_hierarchy=classify_graph_hierarchy(graph, "", set()),
            source="benchmark_5seed_final",
            tags=[],
            reason=hierarchy_reason(graph, "", set()),
            disconnected_hint="disconnected" in graph,
        )
    for row in rows:
        graph = str(row.get("graph") or "")
        if not graph or graph in info:
            continue
        info[graph] = GraphInfo(
            name=graph,
            num_nodes=as_int(row.get("num_nodes")),
            has_hierarchy=classify_graph_hierarchy(graph, "", set()),
            source="per_combo",
            tags=[],
            reason=hierarchy_reason(graph, "", set()),
            disconnected_hint=bool(row.get("disconnected", False)),
        )
    return info


def classify_graph_hierarchy(name: str, source: str, tags: Iterable[str]) -> bool:
    """Classify graph-domain hierarchy from generator-family semantics.

    Parameters
    ----------
    name : str
        Graph identifier.
    source : str
        Graph source metadata.
    tags : Iterable[str]
        Graph tag metadata.

    Returns
    -------
    bool
        ``True`` when the source family is semantically hierarchical.
    """
    tokens = " ".join([name, source, *list(tags)]).lower()
    # SPEC-INTERPRETATION: names/tags carrying dag, tree, dependency, pipeline,
    # org-chart, or model-block families are treated as source-hierarchical;
    # undirected generators such as er/ba/sbm/rgg remain non-hierarchical even
    # though graphs.py later orients them through _undirected_to_dag.
    undirected_prefixes = (
        "er_",
        "ba_",
        "sbm_",
        "rgg_",
        "grid_",
        "regular_",
        "powerlaw_",
        "small_world_",
        "chung_lu_",
        "real_",
        "weighted_karate",
        "protein_ppi",
    )
    if any(name.startswith(prefix) for prefix in undirected_prefixes):
        return False
    hierarchy_markers = (
        "dag",
        "tree",
        "dependency",
        "pipeline",
        "org_chart",
        "transformer",
        "mlp",
        "resnet",
        "densenet",
        "unet",
        "inception",
        "residual",
    )
    return any(marker in tokens for marker in hierarchy_markers)


def hierarchy_reason(name: str, source: str, tags: Iterable[str]) -> str:
    """Return the graph hierarchy classification reason.

    Parameters
    ----------
    name : str
        Graph identifier.
    source : str
        Graph source metadata.
    tags : Iterable[str]
        Graph tag metadata.

    Returns
    -------
    str
        Human-readable reason.
    """
    return (
        "generator-family hierarchical"
        if classify_graph_hierarchy(name, source, tags)
        else ("undirected/procedural source or no hierarchy marker")
    )


def is_hierarchy_requiring(engine: str) -> bool:
    """Return whether an engine requires hierarchical input.

    Parameters
    ----------
    engine : str
        Engine identifier.

    Returns
    -------
    bool
        ``True`` for sugiyama, reingold_tilford, and rt variants.
    """
    return engine.startswith(HIERARCHY_ENGINE_PREFIXES)


def is_on_domain(engine: str, graph: str, graph_info: dict[str, GraphInfo]) -> bool:
    """Return whether an engine/graph combo is on-domain.

    Parameters
    ----------
    engine : str
        Engine identifier.
    graph : str
        Graph identifier.
    graph_info : dict[str, GraphInfo]
        Graph metadata.

    Returns
    -------
    bool
        ``False`` only for hierarchy-requiring engines on non-hierarchical graphs.
    """
    if not is_hierarchy_requiring(engine):
        return True
    return bool(
        graph_info.get(graph, GraphInfo(graph, None, False, "", [], "", False)).has_hierarchy
    )


def build_accounting(
    rows: list[dict[str, Any]],
    deterministic_rows: list[dict[str, Any]],
    rung0_reverify_rows: list[dict[str, Any]],
    failing_map: dict[str, dict[str, Any]],
    triage: TriageInventory,
    five_seed: dict[str, Any],
    graph_info: dict[str, GraphInfo],
    context: ReportContext,
) -> list[AccountingEntry]:
    """Build the exact first-match accounting partition.

    Parameters
    ----------
    rows : list[dict[str, Any]]
        Finalized escalation rows.
    deterministic_rows : list[dict[str, Any]]
        Refresh verdict rows for deterministic-different engines.
    rung0_reverify_rows : list[dict[str, Any]]
        Sugiyama rung-0 reverify rows.
    failing_map : dict[str, dict[str, Any]]
        Escalation failing map.
    triage : TriageInventory
        Parsed 118-variant inventory.
    five_seed : dict[str, Any]
        5-seed benchmark result rows.
    graph_info : dict[str, GraphInfo]
        Graph metadata.
    context : ReportContext
        Assertion and warning state.

    Returns
    -------
    list[AccountingEntry]
        Accounting entries for 39 bit-identical, 64 escalation, and 8 deterministic variants.
    """
    row_by_combo = {str(row.get("combo_id")): row for row in rows}
    deterministic_by_combo = {str(row.get("combo_id")): row for row in deterministic_rows}
    reverify_by_combo = {str(row.get("combo_id")): row for row in rung0_reverify_rows}
    index = index_five_seed(five_seed)
    scope = (
        triage.categories.get("BIT_IDENTICAL", [])
        + triage.categories.get("ESCALATE_STOCHASTIC", [])
        + triage.categories.get("DETERMINISTIC_DIFFERENT", [])
    )
    entries: list[AccountingEntry] = []
    for engine in scope:
        universe = sorted(graph for graph, _rows in index.get(engine, {}).items())
        for graph in universe:
            bucket, reason = classify_accounting_bucket(
                engine, graph, index, failing_map, deterministic_by_combo
            )
            combo_id = f"{graph}::{engine}"
            row = (
                deterministic_by_combo.get(combo_id)
                if engine in triage.categories.get("DETERMINISTIC_DIFFERENT", [])
                else row_by_combo.get(combo_id)
            )
            final_rung = final_rung_for_bucket(bucket, row)
            annotations = list(row.get("final_annotations", [])) if row else []
            if combo_id in reverify_by_combo and not bool(
                reverify_by_combo[combo_id].get("still_bit_exact", True)
            ):
                annotations.append("stale_rung0_failed_reverify")
            on_domain = is_on_domain(engine, graph, graph_info)
            if not on_domain:
                reason = f"{reason}; DOMAIN_MISMATCH"
            entries.append(
                AccountingEntry(
                    engine=engine,
                    graph=graph,
                    bucket=bucket,
                    final_rung=final_rung,
                    reason=reason,
                    on_domain=on_domain,
                    mode=str(row.get("mode")) if row and row.get("mode") else None,
                    informative=is_informative(row, final_rung),
                    seed_na=bool(row.get("seed_na", False)) if row else False,
                    num_nodes=node_count(graph, graph_info, row),
                    runtime_ratio=as_float(row.get("runtime_ratio")) if row else None,
                    annotations=annotations,
                )
            )
        actual = len([entry for entry in entries if entry.engine == engine])
        context.check(actual == len(universe), f"Accounting partition size mismatch for {engine}.")
    return entries


def index_five_seed(five_seed: dict[str, Any]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Index 5-seed rows by engine and graph.

    Parameters
    ----------
    five_seed : dict[str, Any]
        Raw benchmark results.

    Returns
    -------
    dict[str, dict[str, list[dict[str, Any]]]]
        ``engine -> graph -> rows`` index.
    """
    index: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for key, row in five_seed.items():
        engine = str(row.get("engine_name") or parse_result_key(key)[1])
        graph = str(row.get("graph_name") or parse_result_key(key)[0])
        if engine and graph:
            index[engine][graph].append(dict(row))
    return {engine: dict(graphs) for engine, graphs in index.items()}


def parse_result_key(key: str) -> tuple[str, str]:
    """Parse a benchmark result key fallback.

    Parameters
    ----------
    key : str
        Result key shaped ``graph::engine::seed``.

    Returns
    -------
    tuple[str, str]
        Graph and engine names.
    """
    parts = key.split("::")
    if len(parts) < 2:
        return "", ""
    return parts[0], parts[1]


def classify_accounting_bucket(
    engine: str,
    graph: str,
    index: dict[str, dict[str, list[dict[str, Any]]]],
    failing_map: dict[str, dict[str, Any]],
    deterministic_by_combo: dict[str, dict[str, Any]],
) -> tuple[str, str]:
    """Classify one engine/graph by the registered bucket order.

    Parameters
    ----------
    engine : str
        Engine identifier.
    graph : str
        Graph identifier.
    index : dict[str, dict[str, list[dict[str, Any]]]]
        5-seed row index.
    failing_map : dict[str, dict[str, Any]]
        Escalation map.
    deterministic_by_combo : dict[str, dict[str, Any]]
        Refresh rows keyed by combo id.

    Returns
    -------
    tuple[str, str]
        Bucket and reason.
    """
    rows = index.get(engine, {}).get(graph, [])
    if rows and majority_timeout(rows):
        return "TIMEOUT", "timeout-majority in 5-seed rows"
    if not any(row.get("status") == "ok" for row in rows):
        return "ERROR_NO_DATA", "zero ok reimplementation 5-seed rows"
    combo_id = f"{graph}::{engine}"
    if combo_id in deterministic_by_combo:
        return "ESCALATION", "deterministic refresh verdict"
    if graph in set(failing_map.get(engine, {}).get("graphs", [])):
        return "ESCALATION", "listed in failing_map_final"
    reference = reference_for_engine(engine, failing_map)
    if reference:
        ref_rows = index.get(reference, {}).get(graph, [])
        if not any(row.get("status") == "ok" for row in ref_rows):
            return "REF_NO_DATA", "reference has zero ok 5-seed rows"
    return "RUNG_0", "5-seed bit-identical complement"


def majority_timeout(rows: list[dict[str, Any]]) -> bool:
    """Return whether rows are timeout-dominated.

    Parameters
    ----------
    rows : list[dict[str, Any]]
        Benchmark rows for one engine/graph.

    Returns
    -------
    bool
        ``True`` when timeout statuses/errors are a strict majority.
    """
    if not rows:
        return False
    timeouts = sum(
        1
        for row in rows
        if row.get("status") == "timeout" or TIMEOUT_RE.search(str(row.get("error") or ""))
    )
    return timeouts > len(rows) / 2


def reference_for_engine(engine: str, failing_map: dict[str, dict[str, Any]]) -> Optional[str]:
    """Resolve the original-side reference name for an engine.

    Parameters
    ----------
    engine : str
        Variant identifier.
    failing_map : dict[str, dict[str, Any]]
        Escalation failing map.

    Returns
    -------
    Optional[str]
        Reference identifier when known.
    """
    ref = failing_map.get(engine, {}).get("ref")
    if ref:
        return str(ref)
    variant = get_variant(engine)
    if variant is None:
        return None
    return original_variant_name(variant)


def final_rung_for_bucket(bucket: str, row: Optional[dict[str, Any]]) -> Optional[str]:
    """Return final rung for an accounting bucket.

    Parameters
    ----------
    bucket : str
        Accounting bucket.
    row : Optional[dict[str, Any]]
        Finalized escalation row.

    Returns
    -------
    Optional[str]
        Rung label or ``None`` for non-rung buckets.
    """
    if bucket == "RUNG_0":
        return "0"
    if bucket != "ESCALATION":
        return None
    if row is None:
        return "INSUFFICIENT_DATA"
    if row.get("mode") == "deterministic":
        return deterministic_rung(row)
    return str(row.get("final_rung", "INSUFFICIENT_DATA"))


def deterministic_rung(row: dict[str, Any]) -> str:
    """Map deterministic refresh rows to rung-like labels.

    Parameters
    ----------
    row : dict[str, Any]
        Deterministic refresh row.

    Returns
    -------
    str
        ``INVARIANCE_EQUIVALENT``, ``3``, ``4``, or ``INSUFFICIENT_DATA``.
    """
    if row.get("insufficient_data", False):
        return "INSUFFICIENT_DATA"
    verdict = str(row.get("deterministic_verdict") or row.get("verdict") or "")
    if "INVARIANCE" in verdict:
        return "INVARIANCE_EQUIVALENT"
    if row.get("invariance_equivalent", False):
        return "INVARIANCE_EQUIVALENT"
    if row.get("quality_equivalent", False):
        return "3"
    return "4"


def is_informative(row: Optional[dict[str, Any]], final_rung: Optional[str]) -> bool:
    """Return whether a combo is informative for Mode B headline accounting.

    Parameters
    ----------
    row : Optional[dict[str, Any]]
        Finalized row.
    final_rung : Optional[str]
        Final rung.

    Returns
    -------
    bool
        ``True`` when the combo can enter REF_COMPATIBLE denominator.
    """
    if final_rung == "0":
        return True
    if row is None:
        return False
    if row.get("mode") != "B":
        return False
    return not bool(row.get("typicality_uninformative", False))


def node_count(
    graph: str, graph_info: dict[str, GraphInfo], row: Optional[dict[str, Any]]
) -> Optional[int]:
    """Resolve node count for a combo.

    Parameters
    ----------
    graph : str
        Graph identifier.
    graph_info : dict[str, GraphInfo]
        Graph metadata.
    row : Optional[dict[str, Any]]
        Optional row with direct node count.

    Returns
    -------
    Optional[int]
        Node count if known.
    """
    if row is not None:
        value = as_int(row.get("num_nodes"))
        if value is not None:
            return value
    return graph_info.get(graph).num_nodes if graph in graph_info else None


def aggregate_accounting(
    accounting: list[AccountingEntry],
    rows: list[dict[str, Any]],
    triage: TriageInventory,
    graph_info: dict[str, GraphInfo],
) -> dict[str, Any]:
    """Aggregate report diagnostics and percentages.

    Parameters
    ----------
    accounting : list[AccountingEntry]
        Accounting entries.
    rows : list[dict[str, Any]]
        Finalized escalation rows.
    triage : TriageInventory
        Parsed triage inventory.
    graph_info : dict[str, GraphInfo]
        Graph metadata.

    Returns
    -------
    dict[str, Any]
        Aggregated tables.
    """
    del triage, graph_info
    by_engine: dict[str, list[AccountingEntry]] = defaultdict(list)
    for entry in accounting:
        by_engine[entry.engine].append(entry)
    engine_tables = {
        engine: summarize_entries(entries) for engine, entries in sorted(by_engine.items())
    }
    family_tables: dict[str, Counter[str]] = defaultdict(Counter)
    for entry in accounting:
        family_tables[engine_family(entry.engine)][entry.final_rung or entry.bucket] += 1
    size_curves = build_size_curves(accounting)
    diagnostics = build_diagnostics(rows)
    mode_b = build_mode_b_diagnostics(rows)
    insufficient = [
        row
        for row in rows
        if row.get("final_rung") == "INSUFFICIENT_DATA" or row.get("insufficient_data", False)
    ]
    return {
        "engine_tables": engine_tables,
        "family_tables": {family: dict(counts) for family, counts in sorted(family_tables.items())},
        "size_curves": size_curves,
        "diagnostics": diagnostics,
        "mode_b": mode_b,
        "insufficient": insufficient,
    }


def summarize_entries(entries: list[AccountingEntry]) -> dict[str, Any]:
    """Summarize accounting entries for one engine.

    Parameters
    ----------
    entries : list[AccountingEntry]
        Entries for one engine.

    Returns
    -------
    dict[str, Any]
        Count and percentage summary.
    """
    counts = Counter(entry.final_rung or entry.bucket for entry in entries)
    total = len(entries)
    escalation = [entry for entry in entries if entry.bucket == "ESCALATION"]
    escalation_counts = Counter(entry.final_rung or entry.bucket for entry in escalation)
    runtime_values = [entry.runtime_ratio for entry in entries if entry.runtime_ratio is not None]
    return {
        "total": total,
        "counts": dict(counts),
        "percentages": {key: safe_percent(value, total) for key, value in counts.items()},
        "escalation_total": len(escalation),
        "escalation_counts": dict(escalation_counts),
        "escalation_percentages": {
            key: safe_percent(value, len(escalation)) for key, value in escalation_counts.items()
        },
        "median_runtime_ratio": median_or_none(runtime_values),
    }


def build_size_curves(accounting: list[AccountingEntry]) -> dict[str, dict[str, Any]]:
    """Build size-bin degradation curves.

    Parameters
    ----------
    accounting : list[AccountingEntry]
        Accounting entries.

    Returns
    -------
    dict[str, dict[str, Any]]
        Rung-count summaries by size bin.
    """
    curves: dict[str, dict[str, Any]] = {}
    for label, low, high in SIZE_BINS:
        entries = [
            entry
            for entry in accounting
            if entry.num_nodes is not None
            and entry.num_nodes >= low
            and (high is None or entry.num_nodes <= high)
        ]
        counts = Counter(entry.final_rung or entry.bucket for entry in entries)
        curves[label] = {
            "total": len(entries),
            "counts": dict(counts),
            "pass_percent": safe_percent(
                sum(value for key, value in counts.items() if key in PASSING_RUNGS),
                len(entries),
            ),
        }
    return curves


def build_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate dispersion, tracking, and twin-rank diagnostics.

    Parameters
    ----------
    rows : list[dict[str, Any]]
        Finalized rows.

    Returns
    -------
    dict[str, Any]
        Diagnostic aggregate values.
    """
    disp = values_for(rows, "disp")
    track = values_for(rows, "track_ratio")
    twin = values_for(rows, "twin_rank_median")
    return {
        "disp_median": median_or_none(disp),
        "track_ratio_median": median_or_none(track),
        "twin_rank_deciles": deciles(twin),
        "p_diff_annotation_count": len(values_for(rows, "p_diff")),
    }


def build_mode_b_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build Mode B mean(W_D)/sqrt(2) diagnostics.

    Parameters
    ----------
    rows : list[dict[str, Any]]
        Finalized rows.

    Returns
    -------
    dict[str, Any]
        Per-engine Mode B diagnostic table.
    """
    by_engine: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("mode") != "B":
            continue
        value = as_float(row.get("plain_mean_W_D"))
        if value is not None:
            by_engine[str(row.get("engine"))].append(value / math.sqrt(2.0))
    table: dict[str, Any] = {}
    for engine, values in sorted(by_engine.items()):
        array = np.asarray(values, dtype=np.float64)
        table[engine] = {
            "n": int(array.size),
            "median": float(np.median(array)) if array.size else None,
            "over_guard": int(np.count_nonzero(array > 0.85)),
            "band_0_70_0_85": int(np.count_nonzero((array >= 0.70) & (array <= 0.85))),
        }
    return table


def assign_headlines(
    accounting: list[AccountingEntry],
    triage: TriageInventory,
) -> dict[str, dict[str, Any]]:
    """Assign pre-registered engine headline labels.

    Parameters
    ----------
    accounting : list[AccountingEntry]
        Accounting entries.
    triage : TriageInventory
        Parsed triage inventory.

    Returns
    -------
    dict[str, dict[str, Any]]
        Per-engine headline records.
    """
    by_engine: dict[str, list[AccountingEntry]] = defaultdict(list)
    for entry in accounting:
        by_engine[entry.engine].append(entry)
    headlines: dict[str, dict[str, Any]] = {}
    for engine, entries in sorted(by_engine.items()):
        category = triage.variant_to_category.get(engine, "")
        usable = [
            entry
            for entry in entries
            if entry.on_domain
            and entry.final_rung in {"0", "1", "2", "2'", "3", "4", "INVARIANCE_EQUIVALENT"}
        ]
        escalation_entries = [entry for entry in entries if entry.bucket == "ESCALATION"]
        escalation_usable = [
            entry
            for entry in escalation_entries
            if entry.on_domain and entry.final_rung not in {None, "INSUFFICIENT_DATA"}
        ]
        coverage_ok = not escalation_entries or len(escalation_usable) >= 0.5 * len(
            escalation_entries
        )
        if len(usable) < 10 or not coverage_ok:
            headlines[engine] = {
                "headline": f"UNDETERMINED(n={len(usable)})",
                "usable": len(usable),
                "coverage_ok": coverage_ok,
            }
            continue
        if category == "BIT_IDENTICAL":
            headlines[engine] = {
                "headline": "BIT_EXACT",
                "usable": len(usable),
                "coverage_ok": True,
            }
            continue
        if category == "DETERMINISTIC_DIFFERENT":
            headlines[engine] = {
                "headline": "DETERMINISTIC_REFRESH_VERDICT",
                "usable": len(usable),
                "coverage_ok": coverage_ok,
            }
            continue
        mode = dominant_mode(usable)
        if mode == "A":
            dist_rate = safe_percent(
                sum(1 for entry in usable if entry.final_rung in MODE_A_PASS_RUNGS), len(usable)
            )
            seed_denom = [entry for entry in usable if not entry.seed_na or entry.final_rung == "1"]
            seed_rate = safe_percent(
                sum(1 for entry in seed_denom if entry.final_rung == "1"), len(seed_denom)
            )
            headline = (
                "DISTRIBUTIONALLY_MATCHED"
                if dist_rate >= 90.0
                else "UNDETERMINED(n={})".format(len(usable))
            )
            headlines[engine] = {
                "headline": headline,
                "secondary": "SEED_FAITHFUL" if seed_rate >= 90.0 and seed_denom else None,
                "usable": len(usable),
                "distributional_percent": dist_rate,
                "seed_faithful_percent": seed_rate,
                "coverage_ok": coverage_ok,
            }
        elif mode == "B":
            informative = [entry for entry in usable if entry.informative]
            if len(informative) < 10:
                headline = f"UNDETERMINED(n_informative={len(informative)})"
                ref_rate = None
            else:
                ref_rate = safe_percent(
                    sum(1 for entry in informative if entry.final_rung in MODE_B_PASS_RUNGS),
                    len(informative),
                )
                headline = (
                    "REF_COMPATIBLE"
                    if ref_rate >= 90.0
                    else "UNDETERMINED(n={})".format(len(informative))
                )
            headlines[engine] = {
                "headline": headline,
                "usable": len(usable),
                "informative": len(informative),
                "ref_compatible_percent": ref_rate,
                "coverage_ok": coverage_ok,
            }
        else:
            headlines[engine] = {
                "headline": f"UNDETERMINED(n={len(usable)})",
                "usable": len(usable),
                "coverage_ok": coverage_ok,
            }
    return headlines


def dominant_mode(entries: list[AccountingEntry]) -> Optional[str]:
    """Resolve the engine mode from usable entries.

    Parameters
    ----------
    entries : list[AccountingEntry]
        Usable accounting entries.

    Returns
    -------
    Optional[str]
        ``A`` or ``B`` when available.
    """
    counts = Counter(entry.mode for entry in entries if entry.mode in {"A", "B"})
    if not counts:
        return None
    return str(counts.most_common(1)[0][0])


def run_invariance_spotcheck(
    rows: list[dict[str, Any]],
    graph_info: dict[str, GraphInfo],
    data_dir: Path,
    context: ReportContext,
) -> dict[str, Any]:
    """Select, re-score, and summarize the invariance spot-check sample.

    Parameters
    ----------
    rows : list[dict[str, Any]]
        Finalized rows.
    graph_info : dict[str, GraphInfo]
        Graph metadata.
    data_dir : pathlib.Path
        Benchmark directory containing ``results.json`` and position artifacts.
    context : ReportContext
        Assertion and warning state.

    Returns
    -------
    dict[str, Any]
        Sample metadata and report-only would-flip count.
    """
    candidates = []
    for row in rows:
        graph = str(row.get("graph"))
        tracking_fail = row.get("mode") == "A" and not bool(row.get("seed_tracking", False))
        rung4 = row.get("final_rung") == "4"
        disconnected = (
            bool(row.get("disconnected", False))
            or graph_info.get(
                graph, GraphInfo(graph, None, False, "", [], "", False)
            ).disconnected_hint
        )
        # SPEC-INTERPRETATION: without re-loading positions in Task C, symmetry is
        # conservatively approximated by disconnected metadata plus graph names with
        # obvious automorphism families; the report marks actual re-score as pending
        # when position artifacts are not part of the report input.
        symmetric_hint = any(
            token in graph for token in ("grid", "cycle", "ring", "tree", "complete", "bipartite")
        )
        if (rung4 or tracking_fail) and (disconnected or symmetric_hint):
            candidates.append(str(row.get("combo_id")))
    selected = deterministic_sample(sorted(set(candidates)), SPOTCHECK_MAX_COMBOS, "r70::spotcheck")
    row_by_combo = {str(row.get("combo_id")): row for row in rows}
    if not (data_dir / "results.json").exists():
        message = (
            f"Invariance spot-check skipped re-score because {data_dir / 'results.json'} "
            "does not exist."
        )
        context.check(False, message)
        return {
            "candidate_count": len(set(candidates)),
            "sampled_count": len(selected),
            "qualified_count": 0,
            "sampled_combo_ids": selected,
            "would_flip_count": 0,
            "would_flip_percent": 0.0,
            "threshold": INVARIANCE_THRESHOLD,
            "note": "0 candidates qualified for re-score; data-dir results.json is missing.",
        }

    index = index_spotcheck_results(load_json(data_dir / "results.json"))
    graph_edges = load_spotcheck_graph_edges()
    scores = [
        score_invariance_spotcheck_combo(row_by_combo[combo_id], index, graph_edges, data_dir)
        for combo_id in selected
        if combo_id in row_by_combo
    ]
    qualified = [score for score in scores if score.get("qualified")]
    would_flip_count = sum(1 for score in qualified if score.get("would_flip"))
    would_flip_percent = safe_percent(would_flip_count, len(qualified))
    return {
        "candidate_count": len(set(candidates)),
        "sampled_count": len(selected),
        "qualified_count": len(qualified),
        "sampled_combo_ids": selected,
        "would_flip_count": would_flip_count,
        "would_flip_percent": would_flip_percent,
        "threshold": INVARIANCE_THRESHOLD,
        "max_seed_pairs_per_combo": SPOTCHECK_MAX_SEED_PAIRS,
        "scores": scores,
        "note": (
            "Distances are sampled over matched-seed diagonal pairs for Mode A or the "
            "deterministic reference column for Mode B. Would-flip means the sampled mean "
            "toolkit invariance distance fell below the registered 1e-3 threshold."
        ),
    }


def deterministic_sample(keys: list[str], limit: int, purpose: str) -> list[str]:
    """Sample sorted keys deterministically with SHA-256.

    Parameters
    ----------
    keys : list[str]
        Sorted candidate keys.
    limit : int
        Maximum sample size.
    purpose : str
        Purpose string for deterministic ranking.

    Returns
    -------
    list[str]
        Selected keys.
    """
    ranked = sorted(
        keys,
        key=lambda key: hashlib.sha256(f"{purpose}::{key}".encode()).hexdigest(),
    )
    return ranked[:limit]


def index_spotcheck_results(results: dict[str, Any]) -> dict[tuple[str, str], list[PositionRow]]:
    """Index benchmark rows for spot-check position recovery.

    Parameters
    ----------
    results : dict[str, Any]
        Raw ``results.json`` mapping.

    Returns
    -------
    dict[tuple[str, str], list[PositionRow]]
        Rows keyed by ``(graph, engine)`` in the same order used by the analysis runner.
    """
    index: dict[tuple[str, str], list[PositionRow]] = defaultdict(list)
    for key, value in results.items():
        graph, engine, key_seed = split_spotcheck_key(key)
        graph = str(value.get("graph_name") or graph)
        engine = str(value.get("engine_name") or engine)
        seed = normalize_spotcheck_seed(value.get("seed", key_seed))
        nodes_raw = value.get("num_nodes")
        index[(graph, engine)].append(
            PositionRow(
                key=key,
                graph=graph,
                engine=engine,
                seed=seed,
                status=str(value.get("status", "")),
                positions_file=value.get("positions_file"),
                num_nodes=None if nodes_raw is None else int(nodes_raw),
            )
        )
    for indexed_rows in index.values():
        indexed_rows.sort(
            key=lambda item: (-int(item.status == "ok"), spotcheck_seed_sort(item.seed), item.key)
        )
    return dict(index)


def split_spotcheck_key(key: str) -> tuple[str, str, str]:
    """Split a benchmark result key into graph, engine, and seed label.

    Parameters
    ----------
    key : str
        Result key shaped like ``graph::engine::seedN``.

    Returns
    -------
    tuple[str, str, str]
        Parsed graph, engine, and seed-label values.
    """
    parts = key.split("::")
    if len(parts) < 3:
        return key, "", ""
    return parts[0], parts[1], "::".join(parts[2:])


def normalize_spotcheck_seed(value: Any) -> Optional[int]:
    """Normalize a benchmark seed value.

    Parameters
    ----------
    value : Any
        Raw JSON seed or seed-label value.

    Returns
    -------
    Optional[int]
        Integer seed, or ``None`` for deterministic rows.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped in {"", "None", "deterministic"}:
            return None
        if stripped.startswith("seed"):
            stripped = stripped[4:]
        if stripped.lstrip("-").isdigit():
            return int(stripped)
    return None


def spotcheck_seed_sort(seed: Optional[int]) -> int:
    """Return a stable sort key for optional seeds.

    Parameters
    ----------
    seed : Optional[int]
        Seed value.

    Returns
    -------
    int
        Sort value with deterministic rows after seeded rows.
    """
    return 10**12 if seed is None else int(seed)


def load_spotcheck_graph_edges() -> dict[str, tuple[tuple[int, int], ...]]:
    """Load graph edge lists for toolkit invariance metrics.

    Returns
    -------
    dict[str, tuple[tuple[int, int], ...]]
        Graph-name to edge-list mapping.
    """
    graph_edges: dict[str, tuple[tuple[int, int], ...]] = {}
    for item in get_test_graphs():
        edge_index = item.graph.edge_index
        edge_array = (
            edge_index.detach().cpu().numpy()
            if hasattr(edge_index, "detach")
            else np.asarray(edge_index)
        )
        if edge_array.ndim == 2 and edge_array.shape[0] == 2:
            edge_array = edge_array.T
        graph_edges[item.name] = tuple(
            (int(source), int(target)) for source, target in np.asarray(edge_array).reshape(-1, 2)
        )
    return graph_edges


def score_invariance_spotcheck_combo(
    row: dict[str, Any],
    index: dict[tuple[str, str], list[PositionRow]],
    graph_edges: dict[str, tuple[tuple[int, int], ...]],
    data_dir: Path,
) -> dict[str, Any]:
    """Re-score one selected combo with toolkit invariance distance.

    Parameters
    ----------
    row : dict[str, Any]
        Finalized per-combo report row.
    index : dict[tuple[str, str], list[PositionRow]]
        Indexed benchmark rows from ``results.json``.
    graph_edges : dict[str, tuple[tuple[int, int], ...]]
        Graph edge lists keyed by graph name.
    data_dir : pathlib.Path
        Benchmark root used to resolve position paths.

    Returns
    -------
    dict[str, Any]
        Per-combo score summary with qualification status and optional warning.
    """
    combo_id = str(row.get("combo_id"))
    graph = str(row.get("graph"))
    engine = str(row.get("engine"))
    reference = str(row.get("reference"))
    mode = str(row.get("mode"))
    base: dict[str, Any] = {
        "combo_id": combo_id,
        "graph": graph,
        "engine": engine,
        "reference": reference,
        "mode": mode,
        "qualified": False,
    }
    try:
        pairs = spotcheck_position_pairs(row, index, data_dir)
        if not pairs:
            return {**base, "warning": "0 candidates qualified: no loadable position pairs"}
        edge_array = spotcheck_edge_index(graph_edges.get(graph, ()), pairs[0][0].shape[0])
        distances = [
            toolkit_invariance_distance(dagua_pos, ref_pos, edge_array, engine)
            for dagua_pos, ref_pos in pairs
        ]
        finite_distances = [distance for distance in distances if math.isfinite(distance)]
        if not finite_distances:
            return {**base, "warning": "0 candidates qualified: no finite toolkit distances"}
        mean_distance = float(np.mean(finite_distances))
        return {
            **base,
            "qualified": True,
            "pair_count": len(finite_distances),
            "mean_distance": mean_distance,
            "min_distance": float(np.min(finite_distances)),
            "max_distance": float(np.max(finite_distances)),
            "would_flip": mean_distance < INVARIANCE_THRESHOLD,
        }
    except Exception as exc:
        return {**base, "warning": f"{type(exc).__name__}: {exc}"}


def spotcheck_position_pairs(
    row: dict[str, Any],
    index: dict[tuple[str, str], list[PositionRow]],
    data_dir: Path,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Load bounded Mode A diagonal or Mode B column position pairs.

    Parameters
    ----------
    row : dict[str, Any]
        Finalized per-combo row.
    index : dict[tuple[str, str], list[PositionRow]]
        Indexed benchmark rows.
    data_dir : pathlib.Path
        Benchmark root used to resolve position paths.

    Returns
    -------
    list[tuple[numpy.ndarray, numpy.ndarray]]
        Loaded position pairs, each with shape ``[N, 2]`` on both sides.
    """
    graph = str(row.get("graph"))
    engine = str(row.get("engine"))
    reference = str(row.get("reference"))
    mode = str(row.get("mode"))
    if mode == "A":
        seeds = row.get("matched_seeds")
        if not isinstance(seeds, list):
            seeds = sorted(
                set(ok_spotcheck_seeds(index.get((graph, engine), [])))
                & set(ok_spotcheck_seeds(index.get((graph, reference), [])))
            )
        selected_seeds = deterministic_sample(
            [str(seed) for seed in seeds],
            SPOTCHECK_MAX_SEED_PAIRS,
            f"r70::spotcheck-pairs::{graph}::{engine}",
        )
        pairs = []
        for seed_text in selected_seeds:
            seed = int(seed_text)
            dagua_row = first_loadable_row(index, graph, engine, seed)
            ref_row = first_loadable_row(index, graph, reference, seed)
            if dagua_row is not None and ref_row is not None:
                pairs.append(
                    (
                        load_spotcheck_position(data_dir, dagua_row.positions_file or ""),
                        load_spotcheck_position(data_dir, ref_row.positions_file or ""),
                    )
                )
        return pairs
    if mode == "B":
        ref_row = first_deterministic_row(index, graph, reference)
        if ref_row is None:
            return []
        seeds = row.get("reimpl_seeds")
        if not isinstance(seeds, list):
            seeds = ok_spotcheck_seeds(index.get((graph, engine), []))
        selected_seeds = deterministic_sample(
            [str(seed) for seed in seeds],
            SPOTCHECK_MAX_SEED_PAIRS,
            f"r70::spotcheck-pairs::{graph}::{engine}",
        )
        ref_position = load_spotcheck_position(data_dir, ref_row.positions_file or "")
        pairs = []
        for seed_text in selected_seeds:
            dagua_row = first_loadable_row(index, graph, engine, int(seed_text))
            if dagua_row is not None:
                dagua_position = load_spotcheck_position(data_dir, dagua_row.positions_file or "")
                pairs.append((dagua_position, ref_position))
        return pairs
    return []


def ok_spotcheck_seeds(rows: Iterable[PositionRow]) -> list[int]:
    """Return sorted ok seeded row labels.

    Parameters
    ----------
    rows : Iterable[PositionRow]
        Candidate benchmark rows.

    Returns
    -------
    list[int]
        Sorted integer seeds with ok rows and position paths.
    """
    return sorted(
        row.seed
        for row in rows
        if row.status == "ok" and row.seed is not None and row.positions_file is not None
    )


def first_loadable_row(
    index: dict[tuple[str, str], list[PositionRow]],
    graph: str,
    engine: str,
    seed: Optional[int],
) -> Optional[PositionRow]:
    """Resolve a row by the analysis runner's seed-then-deterministic fallback.

    Parameters
    ----------
    index : dict[tuple[str, str], list[PositionRow]]
        Indexed benchmark rows.
    graph : str
        Graph name.
    engine : str
        Engine name.
    seed : Optional[int]
        Preferred seed.

    Returns
    -------
    Optional[PositionRow]
        First ok row with a position path, if available.
    """
    rows = index.get((graph, engine), [])
    for candidate in (seed, None):
        for row in rows:
            if row.seed == candidate and row.status == "ok" and row.positions_file is not None:
                return row
    return None


def first_deterministic_row(
    index: dict[tuple[str, str], list[PositionRow]],
    graph: str,
    engine: str,
) -> Optional[PositionRow]:
    """Return the deterministic ok row for a graph/engine pair.

    Parameters
    ----------
    index : dict[tuple[str, str], list[PositionRow]]
        Indexed benchmark rows.
    graph : str
        Graph name.
    engine : str
        Engine name.

    Returns
    -------
    Optional[PositionRow]
        First deterministic ok row with a position path, if available.
    """
    for row in index.get((graph, engine), []):
        if row.seed is None and row.status == "ok" and row.positions_file is not None:
            return row
    return None


def load_spotcheck_position(data_dir: Path, positions_file: str) -> np.ndarray:
    """Load one benchmark position artifact.

    Parameters
    ----------
    data_dir : pathlib.Path
        Benchmark root.
    positions_file : str
        Relative or absolute position path from ``results.json``.

    Returns
    -------
    numpy.ndarray
        Finite float64 positions with shape ``[N, 2]``.
    """
    path = Path(positions_file)
    if not path.is_absolute():
        path = data_dir / path
    tensor = torch.load(path, map_location="cpu", weights_only=False)
    array = np.asarray(
        tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else tensor,
        dtype=np.float64,
    )
    if array.ndim != 2 or array.shape[1] != 2 or array.size == 0:
        raise ValueError(f"Invalid position shape {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Position array contains non-finite values.")
    return array


def spotcheck_edge_index(edges: tuple[tuple[int, int], ...], num_nodes: int) -> np.ndarray:
    """Build an edge-index array for equivalence metrics.

    Parameters
    ----------
    edges : tuple[tuple[int, int], ...]
        Graph edge list.
    num_nodes : int
        Number of nodes in the position arrays.

    Returns
    -------
    numpy.ndarray
        Edge index with shape ``[2, E]``.
    """
    if not edges:
        return np.empty((2, 0), dtype=np.int64)
    valid_edges = [
        (source, target) for source, target in edges if source < num_nodes and target < num_nodes
    ]
    if not valid_edges:
        return np.empty((2, 0), dtype=np.int64)
    return np.asarray(valid_edges, dtype=np.int64).T


def toolkit_invariance_distance(
    positions_dagua: np.ndarray,
    positions_reference: np.ndarray,
    edge_index: np.ndarray,
    engine: str,
) -> float:
    """Compute the registered toolkit invariance distance.

    Parameters
    ----------
    positions_dagua : numpy.ndarray
        Reimplementation coordinates with shape ``[N, 2]``.
    positions_reference : numpy.ndarray
        Reference coordinates with shape ``[N, 2]``.
    edge_index : numpy.ndarray
        Graph edge index with shape ``[2, E]``.
    engine : str
        Engine name used for the free-aspect anisotropic exception.

    Returns
    -------
    float
        Minimum of automorphism-aligned, component-aligned, and optional
        anisotropic RMSD. Spectrum and quality branches are intentionally not
        consulted.
    """
    # SIGALRM cannot preempt the single igraph/BLISS C call (measured >490s on twin-heavy
    # graphs) -- compute in a hard-killed child process instead. inf on timeout/failure;
    # the caller's finite-distance filter handles it.
    import multiprocessing as mp

    ctx = mp.get_context("fork")
    parent_conn, child_conn = ctx.Pipe(duplex=False)

    proc = ctx.Process(
        target=_spotcheck_toolkit_child,
        args=(child_conn, positions_dagua, positions_reference, edge_index, engine),
    )
    proc.start()
    child_conn.close()
    result: Optional[float] = None
    if parent_conn.poll(SPOTCHECK_PAIR_TIMEOUT_SECONDS):
        try:
            result = parent_conn.recv()
        except EOFError:
            result = None
    proc.join(timeout=1.0)
    if proc.is_alive():
        proc.kill()
        proc.join()
    parent_conn.close()
    return float("inf") if result is None else float(result)


def _spotcheck_toolkit_child(
    conn: Any,
    positions_dagua: np.ndarray,
    positions_reference: np.ndarray,
    edge_index: np.ndarray,
    engine: str,
) -> None:
    """Killable child computing the toolkit invariance distance."""
    try:
        metrics = compute_equivalence_metrics(
            positions_dagua,
            positions_reference,
            edge_index,
            max_automorphisms=SPOTCHECK_MAX_AUTOMORPHISMS,
            engine_name=engine,
        )
        distances = [metrics.aut_procrustes_rmsd, metrics.component_aligned_rmsd]
        if metrics.anisotropic_rmsd is not None:
            distances.append(metrics.anisotropic_rmsd)
        conn.send(float(min(distances)))
    except Exception:  # pragma: no cover - defensive
        conn.send(None)
    finally:
        conn.close()


def raise_spotcheck_timeout(signum: int, frame: Any) -> None:
    """Raise a timeout error for one toolkit spot-check pair.

    Parameters
    ----------
    signum : int
        Delivered signal number.
    frame : Any
        Interrupted frame object supplied by ``signal``.

    Returns
    -------
    None
        This function always raises ``TimeoutError``.
    """
    del signum, frame
    raise TimeoutError(
        f"Toolkit invariance pair exceeded {SPOTCHECK_PAIR_TIMEOUT_SECONDS} seconds."
    )


def evaluate_controls(
    controls_dir: Path, output_dir: Path, context: ReportContext
) -> dict[str, Any]:
    """Evaluate registered control gates with local FDR families.

    Parameters
    ----------
    controls_dir : pathlib.Path
        Directory containing control JSONL rows.
    output_dir : pathlib.Path
        Base output directory.
    context : ReportContext
        Assertion and warning state.

    Returns
    -------
    dict[str, Any]
        Gate result payload written to ``controls/gate_results.json``.
    """
    del context
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(controls_dir.glob("*.jsonl")):
        rows = read_jsonl(path)
        for row in rows:
            kind = str(row.get("control_kind") or path.stem)
            grouped[kind].append(row)
    finalized = {
        kind: finalize_rows(rows, SPEC_VERSION, include_controls=True)
        for kind, rows in grouped.items()
    }
    gates = {
        "gate_1_positive_mode_a": evaluate_gate_positive_mode_a(finalized),
        "gate_2_positive_mode_b": evaluate_gate_positive_mode_b(finalized),
        "gate_3_negative": evaluate_gate_negative(finalized),
        "gate_4_chance": evaluate_gate_chance(finalized),
    }
    all_passed = all(gate["passed"] for gate in gates.values())
    payload = {"all_passed": all_passed, "gates": gates, "summary": {"all_passed": all_passed}}
    out_path = output_dir / "controls" / "gate_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def control_rows(
    finalized: dict[str, list[dict[str, Any]]], needles: tuple[str, ...]
) -> list[dict[str, Any]]:
    """Return finalized control rows matching any kind substring.

    Parameters
    ----------
    finalized : dict[str, list[dict[str, Any]]]
        Rows grouped by control kind.
    needles : tuple[str, ...]
        Lowercase substrings to match.

    Returns
    -------
    list[dict[str, Any]]
        Matching rows.
    """
    rows: list[dict[str, Any]] = []
    for kind, values in finalized.items():
        lower = kind.lower()
        if any(needle in lower for needle in needles):
            rows.extend(values)
    return rows


def evaluate_gate_positive_mode_a(finalized: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Evaluate control gate 1.

    Parameters
    ----------
    finalized : dict[str, list[dict[str, Any]]]
        Finalized control rows by kind.

    Returns
    -------
    dict[str, Any]
        PASS/FAIL and measured values.
    """
    rows = [
        row
        for row in control_rows(finalized, ("positive", "mode-a", "full"))
        if row.get("mode") == "A"
    ]
    scored = [row for row in rows if row.get("final_rung") != "INSUFFICIENT_DATA"]
    pass_count = sum(1 for row in scored if str(row.get("final_rung")) in {"0", "1", "2"})
    rate = safe_percent(pass_count, len(scored))
    passed = len(scored) >= 30 and rate >= 95.0
    return {"passed": passed, "scored": len(scored), "pass_count": pass_count, "pass_percent": rate}


def evaluate_gate_positive_mode_b(finalized: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Evaluate control gate 2.

    Parameters
    ----------
    finalized : dict[str, list[dict[str, Any]]]
        Finalized control rows by kind.

    Returns
    -------
    dict[str, Any]
        PASS/FAIL and measured values.
    """
    rows = [row for row in control_rows(finalized, ("modeb", "mode-b")) if row.get("mode") == "B"]
    informative = [row for row in rows if not bool(row.get("typicality_uninformative", False))]
    pass_count = sum(1 for row in informative if str(row.get("final_rung")) in {"2'", "0"})
    rate = safe_percent(pass_count, len(informative))
    passed = len(informative) >= 30 and rate >= 95.0
    return {
        "passed": passed,
        "informative": len(informative),
        "pass_count": pass_count,
        "pass_percent": rate,
    }


def evaluate_gate_negative(finalized: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Evaluate control gate 3.

    Parameters
    ----------
    finalized : dict[str, list[dict[str, Any]]]
        Finalized control rows by kind.

    Returns
    -------
    dict[str, Any]
        PASS/FAIL and measured values.
    """
    rows = control_rows(finalized, ("negative", "negref"))
    scored = [row for row in rows if row.get("final_rung") != "INSUFFICIENT_DATA"]
    non_primary = sum(
        1 for row in scored if str(row.get("final_rung")) not in {"0", "1", "2", "2'"}
    )
    rate = safe_percent(non_primary, len(scored))
    mode_a = [row for row in scored if row.get("mode") == "A"]
    raw_tracking = sum(1 for row in mode_a if (as_float(row.get("p_track")) or 1.0) < 0.05)
    tracking_limit = math.ceil(0.05 * len(mode_a)) + 2
    passed = bool(scored) and rate >= 95.0 and raw_tracking <= tracking_limit
    return {
        "passed": passed,
        "scored": len(scored),
        "non_primary_percent": rate,
        "raw_tracking_count": raw_tracking,
        "raw_tracking_limit": tracking_limit,
    }


def evaluate_gate_chance(finalized: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Evaluate control gate 4.

    Parameters
    ----------
    finalized : dict[str, list[dict[str, Any]]]
        Finalized control rows by kind.

    Returns
    -------
    dict[str, Any]
        PASS/FAIL and measured values.
    """
    rows = control_rows(finalized, ("chance",))
    p_values = values_for(rows, "p_track")
    ks_p = float(stats.kstest(p_values, "uniform").pvalue) if p_values else 0.0
    recovery_count = int(
        round(
            sum(
                (as_float(row.get("recovery_at_1")) or 0.0) * (as_float(row.get("n")) or 0.0)
                for row in rows
            )
        )
    )
    passed = bool(rows) and ks_p > 0.01 and 8 <= recovery_count <= 33
    return {"passed": passed, "n": len(rows), "ks_p": ks_p, "recovery_count": recovery_count}


def write_outputs(state: dict[str, Any], output_dir: Path) -> None:
    """Write the definitive report and four-tier categorization.

    Parameters
    ----------
    state : dict[str, Any]
        Complete report state.
    output_dir : pathlib.Path
        Output directory.

    Returns
    -------
    None
        Files are written in place.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / REPORT_NAME).write_text(render_report(state))
    (output_dir / TIERS_NAME).write_text(render_tiers(state))


def render_report(state: dict[str, Any]) -> str:
    """Render the definitive Markdown report.

    Parameters
    ----------
    state : dict[str, Any]
        Complete report state.

    Returns
    -------
    str
        Markdown report.
    """
    rows = state["rows"]
    aggregation = state["aggregation"]
    lines = [
        "# Definitive Fidelity Report (r70)",
        "",
        "## Methodology Summary",
        (
            "This report consumes Task B rows, applies full-run report-stage multiplicity, "
            "and assigns final rungs with `dagua.eval.distributional_fidelity.assign_rung`."
        ),
        (
            "Mode A supports distributional matching plus seed tracking. Mode B is "
            "single-draw reference typicality and quality only, so it cannot earn a "
            "distributional-match headline."
        ),
        "",
        "## Decisions Log",
        "- Calibration margin: q95 self-reference split rule.",
        "- Headline threshold: 90% of usable on-domain combos.",
        (
            "- FDR: BH q=0.05 for `p_track` and stress `p_tost`; raw alpha=0.05 "
            "for conformal `p_typ` because the conformal p-value floor would create a "
            "BH cliff."
        ),
        "- Stress margins: max(5% of reference stress, 1e-6).",
        (
            "- Mode B design: REF_TYPICAL is a non-refutation against a single "
            "deterministic reference draw."
        ),
        (
            "- Distance: plain Procrustes by default; free-aspect sugiyama variants "
            "use the registered anisotropic exception."
        ),
        "",
        "## Controls Results",
        render_controls_summary(state["controls"]),
        "",
        "## FDR And Expected False Atypicality",
        render_fdr_summary(rows),
        "",
        "## Headlines",
        render_headlines(state["headlines"]),
        "",
        "## Per-Engine Tables",
        render_engine_tables(aggregation["engine_tables"]),
        "",
        "## Per-Family Tables",
        render_family_tables(aggregation["family_tables"]),
        "",
        "## Degradation Curves",
        render_degradation_curves(aggregation["size_curves"]),
        "",
        "## Runtime Ratios",
        render_runtime_ratios(aggregation["engine_tables"]),
        "",
        "## Dispersion And Tracking Diagnostics",
        render_json_block(aggregation["diagnostics"]),
        "",
        "## Twin-Rank Deciles",
        render_json_block(
            {"twin_rank_deciles": aggregation["diagnostics"].get("twin_rank_deciles")}
        ),
        "",
        "## Mode B Disclosure",
        (
            "Mode B reference rows are single deterministic draws. Seed-capable native "
            "references should receive a seeded-reference follow-up sweep before any "
            "stronger distributional claim."
        ),
        render_json_block(aggregation["mode_b"]),
        "",
        "## Domain Map",
        render_domain_table(state["graph_info"]),
        "",
        "## Domain Mismatch Combos",
        render_domain_mismatches(state["accounting"]),
        "",
        "## Invariance Spot-Check",
        render_json_block(state["spotcheck"]),
        "",
        "## Insufficient Data",
        render_insufficient(aggregation["insufficient"]),
        "",
        "## Deterministic Sub-Verdicts",
        (
            "Deterministic-different variants use refresh-dir invariance verdicts. "
            "INVARIANCE_EQUIVALENT is Tier-3 with a Tier-1-adjacent note, not silently "
            "upgraded."
        ),
        render_deterministic_rows(state["accounting"]),
        "",
        "## Rung0 Reverify Results",
        render_rung0_reverify(state["accounting"]),
        "",
        "## Appendices",
        (
            "NO_REFERENCE: classic_fr_kk_default, classic_fr_kk_long, "
            "classic_kk_fr_default, classic_kk_fr_long. No-port: classic_fcose_default, "
            "classic_fcose_proof. UNVERDICTED_OTHER includes "
            "classic_neato_graphviz_fidelity."
        ),
        "",
        "## Supersession Statement",
        (
            "This r70 report supersedes WHERE_WE_STAND.md group tables, "
            "ALLGRAPHS_SUMMARY.md, fidelity_report_r69, and fidelity_report_final "
            "per-variant verdicts. The final triage remains scoping input only."
        ),
        "",
        "## Assertion Warnings",
        render_warning_list(state["warnings"]),
    ]
    return "\n".join(lines) + "\n"


def render_tiers(state: dict[str, Any]) -> str:
    """Render the four-tier Markdown categorization.

    Parameters
    ----------
    state : dict[str, Any]
        Complete report state.

    Returns
    -------
    str
        Markdown categorization.
    """
    accounting: list[AccountingEntry] = state["accounting"]
    triage: TriageInventory = state["triage"]
    lines = [
        "# Four-Tier Categorization (r70)",
        "",
        "## Tier Definitions",
        "- Tier 1: rung 0 bit-exact.",
        "- Tier 2: TIMEOUT only.",
        "- Tier 3: rungs 1, 2, 2', 3, and deterministic INVARIANCE_EQUIVALENT sub-verdicts.",
        "- Tier 4: rung 4 DIFFERENT.",
        "- NO_DATA: ERROR_NO_DATA, REF_NO_DATA, and INSUFFICIENT_DATA.",
        "",
        "## All 118 Variants",
        render_variant_inventory(triage),
        "",
        "## Per-Combo Tables",
        render_combo_table(accounting),
        "",
        "## Deterministic Sub-Verdicts",
        render_deterministic_rows(accounting),
        "",
        "## Insufficient Data Appendix",
        render_accounting_by_bucket(
            accounting, {"ERROR_NO_DATA", "REF_NO_DATA", "INSUFFICIENT_DATA"}
        ),
        "",
        "## No-Reference Appendix",
        "- `classic_fr_kk_default` -- fr_kk chain has no paired reference.",
        "- `classic_fr_kk_long` -- fr_kk chain has no paired reference.",
        "- `classic_kk_fr_default` -- kk_fr chain has no paired reference.",
        "- `classic_kk_fr_long` -- kk_fr chain has no paired reference.",
        "",
        "## No-Port Appendix",
        "- `classic_fcose_default` -- no Python port.",
        "- `classic_fcose_proof` -- no Python port.",
        "",
        "## Unverdicted Appendix",
        "- `classic_neato_graphviz_fidelity` -- triage UNVERDICTED_OTHER.",
        "",
        "## Supersession Statement",
        (
            "This four-tier categorization supersedes WHERE_WE_STAND.md group tables, "
            "ALLGRAPHS_SUMMARY.md, fidelity_report_r69, and fidelity_report_final "
            "per-variant verdicts."
        ),
    ]
    return "\n".join(lines) + "\n"


def render_controls_summary(payload: Optional[dict[str, Any]]) -> str:
    """Render controls gate summary.

    Parameters
    ----------
    payload : Optional[dict[str, Any]]
        Controls payload.

    Returns
    -------
    str
        Markdown summary.
    """
    if not payload:
        return "No controls/gate_results.json was present when the report was rendered."
    return render_json_block(payload)


def render_fdr_summary(rows: list[dict[str, Any]]) -> str:
    """Render FDR and p_typ accounting summary.

    Parameters
    ----------
    rows : list[dict[str, Any]]
        Finalized rows.

    Returns
    -------
    str
        Markdown summary.
    """
    typ_family = int(rows[0].get("typicality_family_size", 0)) if rows else 0
    typ_fail = int(rows[0].get("typicality_raw_fail_count", 0)) if rows else 0
    expected = float(rows[0].get("expected_false_atypicality", 0.0)) if rows else 0.0
    q_track = len(values_for(rows, "q_track"))
    q_tost = len(values_for(rows, "q_tost"))
    return "\n".join(
        [
            f"- p_track BH family size: {q_track}",
            f"- p_tost BH family size: {q_tost}",
            f"- p_typ raw family size: {typ_family}",
            f"- p_typ raw fail count: {typ_fail}",
            f"- expected false-atypicality count at alpha=0.05: {expected:.2f}",
            "- p_diff is annotation-only.",
        ]
    )


def render_headlines(headlines: dict[str, dict[str, Any]]) -> str:
    """Render headline table.

    Parameters
    ----------
    headlines : dict[str, dict[str, Any]]
        Per-engine headline records.

    Returns
    -------
    str
        Markdown table.
    """
    lines = ["| Engine | Headline | Usable | Detail |", "|---|---:|---:|---|"]
    for engine, record in sorted(headlines.items()):
        detail = {key: value for key, value in record.items() if key not in {"headline", "usable"}}
        detail_text = json.dumps(detail, sort_keys=True)
        lines.append(
            f"| `{engine}` | {record.get('headline')} | {record.get('usable', 0)} | "
            f"`{detail_text}` |"
        )
    return "\n".join(lines)


def render_engine_tables(engine_tables: dict[str, Any]) -> str:
    """Render per-engine rung percentages.

    Parameters
    ----------
    engine_tables : dict[str, Any]
        Per-engine summaries.

    Returns
    -------
    str
        Markdown table.
    """
    lines = [
        "| Engine | Total | Counts | Full % | Escalation Counts | Escalation % |",
        "|---|---:|---|---|---|---|",
    ]
    for engine, record in sorted(engine_tables.items()):
        counts = json.dumps(record["counts"], sort_keys=True)
        percentages = json.dumps(record["percentages"], sort_keys=True)
        escalation_counts = json.dumps(record["escalation_counts"], sort_keys=True)
        escalation_percentages = json.dumps(record["escalation_percentages"], sort_keys=True)
        lines.append(
            f"| `{engine}` | {record['total']} | `{counts}` | `{percentages}` | "
            f"`{escalation_counts}` | `{escalation_percentages}` |"
        )
    return "\n".join(lines)


def render_family_tables(family_tables: dict[str, Any]) -> str:
    """Render per-family rung counts.

    Parameters
    ----------
    family_tables : dict[str, Any]
        Family summaries.

    Returns
    -------
    str
        Markdown table.
    """
    lines = ["| Family | Counts |", "|---|---|"]
    for family, counts in sorted(family_tables.items()):
        lines.append(f"| `{family}` | `{json.dumps(counts, sort_keys=True)}` |")
    return "\n".join(lines)


def render_degradation_curves(size_curves: dict[str, Any]) -> str:
    """Render size-bin degradation curves.

    Parameters
    ----------
    size_curves : dict[str, Any]
        Size-bin summaries.

    Returns
    -------
    str
        Markdown table.
    """
    lines = ["| Size Bin | Total | Pass % | Counts |", "|---|---:|---:|---|"]
    for label, record in size_curves.items():
        counts = json.dumps(record["counts"], sort_keys=True)
        lines.append(f"| {label} | {record['total']} | {record['pass_percent']:.2f} | `{counts}` |")
    return "\n".join(lines)


def render_runtime_ratios(engine_tables: dict[str, Any]) -> str:
    """Render runtime-ratio table.

    Parameters
    ----------
    engine_tables : dict[str, Any]
        Per-engine summaries.

    Returns
    -------
    str
        Markdown table.
    """
    lines = ["| Engine | Median Runtime Ratio |", "|---|---:|"]
    for engine, record in sorted(engine_tables.items()):
        value = record.get("median_runtime_ratio")
        rendered = "NA" if value is None else f"{value:.4g}"
        lines.append(f"| `{engine}` | {rendered} |")
    return "\n".join(lines)


def render_domain_table(graph_info: dict[str, GraphInfo]) -> str:
    """Render full per-graph domain classification table.

    Parameters
    ----------
    graph_info : dict[str, GraphInfo]
        Graph metadata.

    Returns
    -------
    str
        Markdown table.
    """
    lines = [
        "| Graph | N | Has Hierarchy | Source | Tags | Reason |",
        "|---|---:|---:|---|---|---|",
    ]
    for graph, info in sorted(graph_info.items()):
        lines.append(
            f"| `{graph}` | {info.num_nodes if info.num_nodes is not None else 'NA'} | "
            f"{info.has_hierarchy} | {info.source} | `{','.join(info.tags)}` | {info.reason} |"
        )
    return "\n".join(lines)


def render_domain_mismatches(accounting: list[AccountingEntry]) -> str:
    """Render domain-mismatch combos.

    Parameters
    ----------
    accounting : list[AccountingEntry]
        Accounting entries.

    Returns
    -------
    str
        Markdown list or placeholder.
    """
    mismatches = [entry for entry in accounting if not entry.on_domain]
    if not mismatches:
        return "No DOMAIN_MISMATCH combos."
    return "\n".join(f"- `{entry.graph}::{entry.engine}` -- {entry.reason}" for entry in mismatches)


def render_insufficient(rows: list[dict[str, Any]]) -> str:
    """Render insufficient-data rows.

    Parameters
    ----------
    rows : list[dict[str, Any]]
        Insufficient data rows.

    Returns
    -------
    str
        Markdown list.
    """
    if not rows:
        return "No INSUFFICIENT_DATA rows."
    return "\n".join(
        f"- `{row.get('combo_id')}` -- "
        f"{row.get('insufficient_reason') or row.get('final_annotations')}"
        for row in rows
    )


def render_deterministic_rows(accounting: list[AccountingEntry]) -> str:
    """Render deterministic sub-verdict entries.

    Parameters
    ----------
    accounting : list[AccountingEntry]
        Accounting entries.

    Returns
    -------
    str
        Markdown list.
    """
    rows = [entry for entry in accounting if entry.final_rung == "INVARIANCE_EQUIVALENT"]
    if not rows:
        return "No deterministic refresh sub-verdict rows were available."
    return "\n".join(f"- `{entry.graph}::{entry.engine}` -- {entry.final_rung}" for entry in rows)


def render_rung0_reverify(accounting: list[AccountingEntry]) -> str:
    """Render stale rung-0 reverify flags.

    Parameters
    ----------
    accounting : list[AccountingEntry]
        Accounting entries.

    Returns
    -------
    str
        Markdown list.
    """
    rows = [entry for entry in accounting if "stale_rung0_failed_reverify" in entry.annotations]
    if not rows:
        return "No stale_rung0_failed_reverify flags."
    return "\n".join(f"- `{entry.graph}::{entry.engine}`" for entry in rows)


def render_variant_inventory(triage: TriageInventory) -> str:
    """Render all triage categories.

    Parameters
    ----------
    triage : TriageInventory
        Parsed inventory.

    Returns
    -------
    str
        Markdown list.
    """
    lines: list[str] = []
    for category, engines in sorted(triage.categories.items()):
        lines.append(f"### {category}")
        lines.extend(f"- `{engine}`" for engine in engines)
        lines.append("")
    return "\n".join(lines).strip()


def render_combo_table(accounting: list[AccountingEntry]) -> str:
    """Render per-combo accounting rows.

    Parameters
    ----------
    accounting : list[AccountingEntry]
        Accounting entries.

    Returns
    -------
    str
        Markdown table.
    """
    lines = [
        "| Engine | Graph | Bucket | Rung | On Domain | Reason |",
        "|---|---|---|---|---:|---|",
    ]
    for entry in sorted(accounting, key=lambda item: (item.engine, item.graph)):
        lines.append(
            f"| `{entry.engine}` | `{entry.graph}` | {entry.bucket} | {entry.final_rung or 'NA'} | "
            f"{entry.on_domain} | {entry.reason} |"
        )
    return "\n".join(lines)


def render_accounting_by_bucket(accounting: list[AccountingEntry], buckets: set[str]) -> str:
    """Render accounting entries in selected buckets.

    Parameters
    ----------
    accounting : list[AccountingEntry]
        Accounting entries.
    buckets : set[str]
        Buckets or rung labels to include.

    Returns
    -------
    str
        Markdown list.
    """
    rows = [
        entry
        for entry in accounting
        if entry.bucket in buckets or (entry.final_rung or "") in buckets
    ]
    if not rows:
        return "No rows."
    return "\n".join(
        f"- `{entry.graph}::{entry.engine}` -- {entry.bucket}/{entry.final_rung}: {entry.reason}"
        for entry in rows
    )


def render_json_block(payload: Any) -> str:
    """Render JSON as a Markdown code block.

    Parameters
    ----------
    payload : Any
        JSON-serializable payload.

    Returns
    -------
    str
        Markdown fenced JSON.
    """
    return "```json\n" + json.dumps(payload, indent=2, sort_keys=True) + "\n```"


def render_warning_list(warnings: list[str]) -> str:
    """Render assertion warnings.

    Parameters
    ----------
    warnings : list[str]
        Warning messages.

    Returns
    -------
    str
        Markdown list.
    """
    if not warnings:
        return "No assertion warnings."
    return "\n".join(f"- {warning}" for warning in warnings)


def load_existing_controls_summary(output_dir: Path) -> Optional[dict[str, Any]]:
    """Load controls gate results if present.

    Parameters
    ----------
    output_dir : pathlib.Path
        Output directory.

    Returns
    -------
    Optional[dict[str, Any]]
        Parsed controls payload.
    """
    path = output_dir / "controls" / "gate_results.json"
    return load_json(path) if path.exists() else None


def ensure_oc_simulation(path: Path, strict: bool) -> dict[str, Any]:
    """Write or load the operating-characteristics annex.

    Parameters
    ----------
    path : pathlib.Path
        OC simulation output path.
    strict : bool
        Whether to run the full Task A helper.  No-strict smoke renders record a
        deferred payload because the helper is intentionally expensive.

    Returns
    -------
    dict[str, Any]
        OC simulation payload.
    """
    if path.exists():
        cached = load_json(path)
        if not strict or not bool(cached.get("deferred", False)):
            return cached
    if not strict:
        payload = {
            "deferred": True,
            "reason": "no-strict smoke render skipped expensive OC helper",
            "required_helper": "dagua.eval.distributional_fidelity.run_oc_simulation",
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        return payload
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(b"r70::oc").digest()[:8], "little"))
    payload = df.run_oc_simulation(rng)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def extract_headers(path: Path) -> list[str]:
    """Extract Markdown section headers from a rendered file.

    Parameters
    ----------
    path : pathlib.Path
        Markdown path.

    Returns
    -------
    list[str]
        Header lines.
    """
    return [line for line in path.read_text().splitlines() if line.startswith("#")]


def engine_family(engine: str) -> str:
    """Return a coarse family label for a variant.

    Parameters
    ----------
    engine : str
        Engine identifier.

    Returns
    -------
    str
        Family label.
    """
    stripped = engine.removeprefix("classic_")
    for token in (
        "classical_mds",
        "davidson_harel",
        "maxent_stress",
        "pivot_mds",
        "reingold_tilford",
        "stress_maj",
        "stress_sgd",
        "sgd2_multi",
        "sugiyama",
        "spectral",
        "graphopt",
        "linlog",
        "neulay",
        "tsnet",
        "fcose",
        "fmmm",
        "gem",
        "drl",
        "fa2",
        "kk",
        "lgl",
        "fr",
        "rt",
    ):
        if stripped.startswith(token):
            return token
    return stripped.split("_", 1)[0]


def values_for(rows: list[dict[str, Any]], key: str) -> list[float]:
    """Extract finite float values from row dictionaries.

    Parameters
    ----------
    rows : list[dict[str, Any]]
        Input rows.
    key : str
        Field name.

    Returns
    -------
    list[float]
        Finite values.
    """
    values: list[float] = []
    for row in rows:
        value = as_float(row.get(key))
        if value is not None:
            values.append(value)
    return values


def as_float(value: Any) -> Optional[float]:
    """Convert a value to a finite float.

    Parameters
    ----------
    value : Any
        Candidate value.

    Returns
    -------
    Optional[float]
        Finite float or ``None``.
    """
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def as_int(value: Any) -> Optional[int]:
    """Convert a value to an integer.

    Parameters
    ----------
    value : Any
        Candidate value.

    Returns
    -------
    Optional[int]
        Integer or ``None``.
    """
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def safe_percent(numerator: int, denominator: int) -> float:
    """Compute a percentage with zero-denominator protection.

    Parameters
    ----------
    numerator : int
        Numerator.
    denominator : int
        Denominator.

    Returns
    -------
    float
        Percentage, or 0 for an empty denominator.
    """
    return 0.0 if denominator == 0 else 100.0 * numerator / denominator


def median_or_none(values: list[float]) -> Optional[float]:
    """Return median or ``None`` for empty values.

    Parameters
    ----------
    values : list[float]
        Numeric values.

    Returns
    -------
    Optional[float]
        Median value.
    """
    return float(np.median(np.asarray(values, dtype=np.float64))) if values else None


def deciles(values: list[float]) -> list[float]:
    """Return deciles for diagnostic values.

    Parameters
    ----------
    values : list[float]
        Numeric values.

    Returns
    -------
    list[float]
        Deciles at 10, 20, ..., 90 percent.
    """
    if not values:
        return []
    array = np.asarray(values, dtype=np.float64)
    return [float(np.quantile(array, q)) for q in np.linspace(0.1, 0.9, 9)]


if __name__ == "__main__":
    sys.exit(main())
