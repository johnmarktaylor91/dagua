#!/usr/bin/env python3
"""Matched-seed bit-exact harness for classic layout fidelity variants."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import sys
import warnings
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dagua.eval.competitors import (  # noqa: E402
    CompetitorBase,
    CompetitorResult,
    get_competitor,
)
from dagua.eval.competitors.classic_competitor import VariantCompetitor  # noqa: E402
from dagua.eval.variants import (  # noqa: E402
    VARIANT_REGISTRY,
    AlgorithmVariant,
    engine_is_heavy,
    get_variant,
    get_variant_for_original_name,
    original_variant_name,
    variants_for_base_engine,
)
from dagua.graph import DaguaGraph  # noqa: E402
from scripts.fast_fidelity_report import procrustes_rmsd  # noqa: E402
from scripts.rng_match.small_fixtures import small_fixtures  # noqa: E402

STATUS_DIR = REPO_ROOT / ".project-context" / "research" / "sprint_rng_matching"
STATUS_MD = STATUS_DIR / "STATUS.md"
STATUS_JSON = STATUS_DIR / "status.json"
BIT_EXACT_THRESHOLD = 1e-7
CLOSE_THRESHOLD = 1e-3
DEFAULT_SEEDS = (1, 2, 3)


@dataclass(frozen=True)
class MatchRow:
    """One matched-seed comparison result.

    Parameters
    ----------
    engine : str
        Reimplementation-side variant name.
    reference : str
        Synthetic original-side variant name, base original name, or status marker.
    graph : str
        Fixture graph name.
    seed : int | None
        Matched seed used for both engines, or the first requested seed for a
        deterministic one-run comparison.
    rmsd : float | None
        Procrustes RMSD, or ``None`` when the comparison did not run.
    exact_match : bool
        Whether raw positions matched at ``atol=1e-9, rtol=0``.
    n_nodes : int
        Number of fixture nodes.
    status : str
        Row status: ``ok``, ``no_reference``, ``unavailable``, or ``error``.
    note : str
        Additional detail for deterministic runs or failures.
    """

    engine: str
    reference: str
    graph: str
    seed: Optional[int]
    rmsd: Optional[float]
    exact_match: bool
    n_nodes: int
    status: str
    note: str = ""


@dataclass(frozen=True)
class EngineSummary:
    """Summary row for one reimplementation variant.

    Parameters
    ----------
    engine : str
        Reimplementation-side variant name.
    reference : str
        Reference engine name or status marker.
    max_rmsd : float | None
        Maximum row RMSD across successful comparisons.
    worst_fixture : str
        Fixture name associated with ``max_rmsd``.
    verdict : str
        ``BIT_EXACT``, ``CLOSE``, ``DIVERGENT``, ``NO_REFERENCE``,
        ``UNAVAILABLE``, or ``NO_DATA``.
    exact_match_count : int
        Count of successful rows with bit-exact raw positions.
    total : int
        Count of successful comparison rows.
    timestamp : str
        UTC ISO-8601 timestamp for this summary update.
    """

    engine: str
    reference: str
    max_rmsd: Optional[float]
    worst_fixture: str
    verdict: str
    exact_match_count: int
    total: int
    timestamp: str


def clone_graph(graph: DaguaGraph) -> DaguaGraph:
    """Return a fresh graph copy for a competitor run.

    Parameters
    ----------
    graph : DaguaGraph
        Fixture graph to clone.

    Returns
    -------
    DaguaGraph
        Independent graph copy preserving topology and labels.
    """
    return DaguaGraph.from_json(graph.to_json())


def resolve_competitor(engine_name: str) -> Optional[CompetitorBase]:
    """Resolve a base, reimplementation variant, or original variant competitor.

    Parameters
    ----------
    engine_name : str
        Engine name supplied to the harness.

    Returns
    -------
    CompetitorBase | None
        Runnable competitor instance, or ``None`` when the registry cannot
        resolve the name.
    """
    base_competitor = get_competitor(engine_name)
    if base_competitor is not None:
        return base_competitor

    variant = get_variant(engine_name)
    if variant is not None:
        variant_base = get_competitor(variant.base_engine)
        if variant_base is None:
            return None
        return VariantCompetitor(
            base_competitor=variant_base,
            variant_params=variant.reimpl_params,
            name=variant.variant_id,
            display_name=variant.display_name,
            is_heavy=variant.is_heavy,
            max_nodes=variant.max_nodes,
        )

    original_variant = get_variant_for_original_name(engine_name)
    if original_variant is None or original_variant.original_engine is None:
        return None
    original_base = get_competitor(original_variant.original_engine)
    if original_base is None:
        return None
    return VariantCompetitor(
        base_competitor=original_base,
        variant_params=original_variant.original_params,
        name=engine_name,
        display_name=f"{original_variant.display_name} [original]",
        is_heavy=engine_is_heavy(engine_name),
        max_nodes=original_variant.max_nodes,
    )


def parse_seeds(raw_seeds: str) -> list[int]:
    """Parse a comma-delimited seed list.

    Parameters
    ----------
    raw_seeds : str
        Comma-delimited seed values.

    Returns
    -------
    list[int]
        Parsed integer seeds in input order.
    """
    seeds = [int(seed.strip()) for seed in raw_seeds.split(",") if seed.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def expand_engine_names(raw_engines: Optional[str]) -> list[str]:
    """Resolve requested engines to reimplementation variant IDs.

    Parameters
    ----------
    raw_engines : str | None
        Comma-delimited engine filter, or ``None`` for all variants.

    Returns
    -------
    list[str]
        Variant IDs to run in registry order where possible.
    """
    if raw_engines is None:
        return [variant.variant_id for variant in VARIANT_REGISTRY]

    requested = [engine.strip() for engine in raw_engines.split(",") if engine.strip()]
    expanded: list[str] = []
    for engine_name in requested:
        variant = get_variant(engine_name)
        if variant is not None:
            expanded.append(variant.variant_id)
            continue
        base_variants = variants_for_base_engine(engine_name)
        if base_variants:
            expanded.extend(variant.variant_id for variant in base_variants)
            continue
        expanded.append(engine_name)
    return _unique(expanded)


def _unique(values: Sequence[str]) -> list[str]:
    """Return values with duplicates removed while preserving order.

    Parameters
    ----------
    values : Sequence[str]
        Values to deduplicate.

    Returns
    -------
    list[str]
        Deduplicated values.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def seeds_for_variant(variant: AlgorithmVariant, seeds: Sequence[int]) -> list[int]:
    """Return comparison seeds for a variant.

    Parameters
    ----------
    variant : AlgorithmVariant
        Reimplementation-side variant metadata.
    seeds : Sequence[int]
        Requested matched seeds.

    Returns
    -------
    list[int]
        All requested seeds for stochastic reimplementations, or one seed for
        deterministic reimplementations.
    """
    if variant.is_stochastic:
        return list(seeds)
    return [int(seeds[0])]


def _position_tensor(result_pos: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Normalize a competitor position payload to a CPU float64 tensor.

    Parameters
    ----------
    result_pos : torch.Tensor | None
        Competitor position tensor.

    Returns
    -------
    torch.Tensor | None
        CPU tensor with shape ``[N, 2]`` and dtype ``float64`` when valid.
    """
    if result_pos is None:
        return None
    pos = result_pos.detach().cpu()
    if pos.ndim != 2 or pos.shape[1] != 2:
        return None
    return pos.to(dtype=torch.float64)


def compare_positions(reference_pos: torch.Tensor, dagua_pos: torch.Tensor) -> tuple[float, bool]:
    """Compute Procrustes RMSD and raw-position exactness.

    Parameters
    ----------
    reference_pos : torch.Tensor
        Reference positions with shape ``[N, 2]``.
    dagua_pos : torch.Tensor
        Reimplementation positions with shape ``[N, 2]``.

    Returns
    -------
    tuple[float, bool]
        Procrustes RMSD and exact-match flag.
    """
    exact = bool(torch.allclose(reference_pos, dagua_pos, atol=1e-9, rtol=0.0))
    rmsd = procrustes_rmsd(
        reference_pos.numpy().astype(np.float64, copy=False),
        dagua_pos.numpy().astype(np.float64, copy=False),
    )
    return rmsd, exact


def _layout_worker(
    queue: mp.Queue,
    competitor: CompetitorBase,
    graph: DaguaGraph,
    timeout: float,
    seed: int,
) -> None:
    """Run one competitor layout inside a killable worker process.

    Parameters
    ----------
    queue : multiprocessing.Queue
        Queue used to return the competitor result to the parent process.
    competitor : CompetitorBase
        Competitor adapter to run.
    graph : DaguaGraph
        Fixture graph clone.
    timeout : float
        Adapter-level timeout forwarded to the competitor.
    seed : int
        Matched seed for this run.

    Returns
    -------
    None
        Places a picklable result payload on ``queue``.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = competitor.layout(graph, timeout=timeout, seed=seed)
        pos_payload = None if result.pos is None else result.pos.detach().cpu().numpy()
        queue.put((result.name, pos_payload, result.runtime_seconds, result.error))
    except Exception as exc:
        queue.put((competitor.name, None, 0.0, str(exc)))


def run_competitor_with_timeout(
    competitor: CompetitorBase,
    graph: DaguaGraph,
    timeout: float,
    seed: int,
) -> CompetitorResult:
    """Run a competitor layout with parent-enforced timeout.

    Parameters
    ----------
    competitor : CompetitorBase
        Competitor adapter to run.
    graph : DaguaGraph
        Fixture graph clone.
    timeout : float
        Maximum wall-clock seconds for the child process.
    seed : int
        Matched seed for this run.

    Returns
    -------
    CompetitorResult
        Competitor result, or an error result when the child process times out.
    """
    if timeout <= 0:
        return competitor.layout(graph, timeout=timeout, seed=seed)

    context = mp.get_context("fork")
    queue: mp.Queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_layout_worker,
        args=(queue, competitor, graph, timeout, seed),
    )
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join(5.0)
        if process.is_alive():
            process.kill()
            process.join()
        return CompetitorResult(
            name=competitor.name,
            pos=None,
            runtime_seconds=timeout,
            error=f"process timeout after {timeout:.1f}s",
        )
    if not queue.empty():
        name, pos_payload, runtime_seconds, error = queue.get()
        pos = None if pos_payload is None else torch.as_tensor(pos_payload)
        return CompetitorResult(
            name=name,
            pos=pos,
            runtime_seconds=float(runtime_seconds),
            error=error,
        )
    return CompetitorResult(
        name=competitor.name,
        pos=None,
        runtime_seconds=0.0,
        error=f"process exited with code {process.exitcode}",
    )


def run_engine(engine_name: str, seeds: Sequence[int], timeout: float) -> list[MatchRow]:
    """Run matched-seed comparisons for one variant engine.

    Parameters
    ----------
    engine_name : str
        Reimplementation-side variant ID.
    seeds : Sequence[int]
        Matched seeds to use for stochastic variants.
    timeout : float
        Per-run competitor timeout in seconds.

    Returns
    -------
    list[MatchRow]
        Per-fixture, per-seed result rows.
    """
    variant = get_variant(engine_name)
    fixtures = small_fixtures()
    if variant is None:
        return [
            MatchRow(
                engine=engine_name,
                reference="no_reference",
                graph=graph_name,
                seed=int(seeds[0]),
                rmsd=None,
                exact_match=False,
                n_nodes=graph.num_nodes,
                status="no_reference",
                note="engine is not a registered reimplementation variant",
            )
            for graph_name, graph in fixtures.items()
        ]

    reference_name = original_variant_name(variant)
    if reference_name is None:
        return [
            MatchRow(
                engine=engine_name,
                reference="no_reference",
                graph=graph_name,
                seed=int(seeds[0]),
                rmsd=None,
                exact_match=False,
                n_nodes=graph.num_nodes,
                status="no_reference",
                note="variant registry has no original_engine pairing",
            )
            for graph_name, graph in fixtures.items()
        ]

    dagua_competitor = resolve_competitor(engine_name)
    reference_competitor = resolve_competitor(reference_name)
    if dagua_competitor is None or reference_competitor is None:
        missing = engine_name if dagua_competitor is None else reference_name
        return [
            MatchRow(
                engine=engine_name,
                reference=reference_name,
                graph=graph_name,
                seed=int(seeds[0]),
                rmsd=None,
                exact_match=False,
                n_nodes=graph.num_nodes,
                status="unavailable",
                note=f"competitor unavailable: {missing}",
            )
            for graph_name, graph in fixtures.items()
        ]

    if not reference_competitor.available():
        return _availability_rows(
            engine_name,
            reference_name,
            fixtures,
            seeds,
            "reference unavailable",
        )
    if not dagua_competitor.available():
        return _availability_rows(
            engine_name,
            reference_name,
            fixtures,
            seeds,
            "dagua competitor unavailable",
        )

    rows: list[MatchRow] = []
    run_seeds = seeds_for_variant(variant, seeds)
    deterministic_note = "" if variant.is_stochastic else "deterministic reimplementation: ran once"
    timeout_note = ""
    for graph_name, fixture in fixtures.items():
        for seed in run_seeds:
            if timeout_note:
                rows.append(
                    MatchRow(
                        engine=engine_name,
                        reference=reference_name,
                        graph=graph_name,
                        seed=seed,
                        rmsd=None,
                        exact_match=False,
                        n_nodes=fixture.num_nodes,
                        status="error",
                        note=timeout_note,
                    )
                )
                continue
            reference_result = run_competitor_with_timeout(
                reference_competitor,
                clone_graph(fixture),
                timeout,
                seed,
            )
            dagua_result = run_competitor_with_timeout(
                dagua_competitor,
                clone_graph(fixture),
                timeout,
                seed,
            )
            reference_pos = _position_tensor(reference_result.pos)
            dagua_pos = _position_tensor(dagua_result.pos)
            if reference_pos is None or dagua_pos is None:
                note = _error_note(reference_result.error, dagua_result.error)
                if _is_timeout_error(reference_result.error) or _is_timeout_error(
                    dagua_result.error
                ):
                    timeout_note = f"skipped after first process timeout: {note}"
                rows.append(
                    MatchRow(
                        engine=engine_name,
                        reference=reference_name,
                        graph=graph_name,
                        seed=seed,
                        rmsd=None,
                        exact_match=False,
                        n_nodes=fixture.num_nodes,
                        status="error",
                        note=note,
                    )
                )
                continue
            if reference_pos.shape != dagua_pos.shape:
                shape_note = (
                    f"shape mismatch: reference={tuple(reference_pos.shape)} "
                    f"dagua={tuple(dagua_pos.shape)}"
                )
                rows.append(
                    MatchRow(
                        engine=engine_name,
                        reference=reference_name,
                        graph=graph_name,
                        seed=seed,
                        rmsd=None,
                        exact_match=False,
                        n_nodes=fixture.num_nodes,
                        status="error",
                        note=shape_note,
                    )
                )
                continue
            rmsd, exact = compare_positions(reference_pos, dagua_pos)
            rows.append(
                MatchRow(
                    engine=engine_name,
                    reference=reference_name,
                    graph=graph_name,
                    seed=seed,
                    rmsd=rmsd,
                    exact_match=exact,
                    n_nodes=fixture.num_nodes,
                    status="ok",
                    note=deterministic_note,
                )
            )
    return rows


def _is_timeout_error(error: Optional[str]) -> bool:
    """Return whether an error message is a process timeout.

    Parameters
    ----------
    error : str | None
        Competitor error text.

    Returns
    -------
    bool
        ``True`` when the parent-enforced timeout fired.
    """
    return error is not None and "process timeout" in error


def _availability_rows(
    engine_name: str,
    reference_name: str,
    fixtures: dict[str, DaguaGraph],
    seeds: Sequence[int],
    note: str,
) -> list[MatchRow]:
    """Build unavailable rows for all fixtures.

    Parameters
    ----------
    engine_name : str
        Reimplementation-side variant name.
    reference_name : str
        Reference-side variant name.
    fixtures : dict[str, DaguaGraph]
        Fixture graphs keyed by name.
    seeds : Sequence[int]
        Requested matched seeds.
    note : str
        Availability note to record.

    Returns
    -------
    list[MatchRow]
        Unavailable status rows.
    """
    return [
        MatchRow(
            engine=engine_name,
            reference=reference_name,
            graph=graph_name,
            seed=int(seeds[0]),
            rmsd=None,
            exact_match=False,
            n_nodes=graph.num_nodes,
            status="unavailable",
            note=note,
        )
        for graph_name, graph in fixtures.items()
    ]


def _error_note(reference_error: Optional[str], dagua_error: Optional[str]) -> str:
    """Format competitor errors into a compact row note.

    Parameters
    ----------
    reference_error : str | None
        Reference competitor error message.
    dagua_error : str | None
        Reimplementation competitor error message.

    Returns
    -------
    str
        Combined error note.
    """
    parts = []
    if reference_error:
        parts.append(f"reference: {reference_error}")
    if dagua_error:
        parts.append(f"dagua: {dagua_error}")
    return "; ".join(parts) if parts else "position missing or invalid"


def summarize_engine(engine_name: str, rows: Sequence[MatchRow], timestamp: str) -> EngineSummary:
    """Build one engine summary from detailed rows.

    Parameters
    ----------
    engine_name : str
        Reimplementation-side variant name.
    rows : Sequence[MatchRow]
        Detailed rows for the engine.
    timestamp : str
        UTC ISO-8601 timestamp.

    Returns
    -------
    EngineSummary
        Aggregated status for the engine.
    """
    reference = rows[0].reference if rows else "unknown"
    ok_rows = [row for row in rows if row.status == "ok" and row.rmsd is not None]
    if ok_rows:
        worst = max(ok_rows, key=lambda row: float(row.rmsd))
        max_rmsd = float(worst.rmsd) if worst.rmsd is not None else math.nan
        if max_rmsd < BIT_EXACT_THRESHOLD:
            verdict = "BIT_EXACT"
        elif max_rmsd < CLOSE_THRESHOLD:
            verdict = "CLOSE"
        else:
            verdict = "DIVERGENT"
        return EngineSummary(
            engine=engine_name,
            reference=reference,
            max_rmsd=max_rmsd,
            worst_fixture=worst.graph,
            verdict=verdict,
            exact_match_count=sum(1 for row in ok_rows if row.exact_match),
            total=len(ok_rows),
            timestamp=timestamp,
        )

    statuses = {row.status for row in rows}
    if "no_reference" in statuses:
        verdict = "NO_REFERENCE"
    elif "unavailable" in statuses:
        verdict = "UNAVAILABLE"
    elif "error" in statuses:
        verdict = "ERROR"
    else:
        verdict = "NO_DATA"
    return EngineSummary(
        engine=engine_name,
        reference=reference,
        max_rmsd=None,
        worst_fixture="",
        verdict=verdict,
        exact_match_count=0,
        total=0,
        timestamp=timestamp,
    )


def load_status_json() -> dict[str, Any]:
    """Load the existing machine-readable status file.

    Returns
    -------
    dict[str, Any]
        Existing status payload, or an empty payload when absent.
    """
    if not STATUS_JSON.exists():
        return {"rows": [], "summary": []}
    with STATUS_JSON.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return {
        "rows": list(data.get("rows", [])),
        "summary": list(data.get("summary", [])),
    }


def write_status(engine_names: Sequence[str], rows: Sequence[MatchRow]) -> list[EngineSummary]:
    """Merge rows into ``status.json`` and regenerate ``STATUS.md``.

    Parameters
    ----------
    engine_names : Sequence[str]
        Engines updated by this harness invocation.
    rows : Sequence[MatchRow]
        Detailed rows for the updated engines.

    Returns
    -------
    list[EngineSummary]
        Summaries for engines updated by this invocation.
    """
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    updated = set(engine_names)
    existing = load_status_json()
    existing_rows = [row for row in existing["rows"] if row.get("engine") not in updated]
    existing_summary = [
        summary for summary in existing["summary"] if summary.get("engine") not in updated
    ]

    row_payloads = [asdict(row) for row in rows]
    summaries = [
        summarize_engine(engine_name, [row for row in rows if row.engine == engine_name], timestamp)
        for engine_name in engine_names
    ]
    summary_payloads = [asdict(summary) for summary in summaries]

    all_rows = existing_rows + row_payloads
    all_summaries = existing_summary + summary_payloads
    payload = {
        "updated_at": timestamp,
        "thresholds": {
            "bit_exact": BIT_EXACT_THRESHOLD,
            "close": CLOSE_THRESHOLD,
        },
        "rows": all_rows,
        "summary": all_summaries,
    }
    with STATUS_JSON.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    write_status_markdown(all_summaries)
    return summaries


def write_status_markdown(summary_payloads: Sequence[dict[str, Any]]) -> None:
    """Write the markdown source-of-truth status table.

    Parameters
    ----------
    summary_payloads : Sequence[dict[str, Any]]
        Summary dictionaries for all known engines.

    Returns
    -------
    None
        Writes ``STATUS.md``.
    """
    sorted_summaries = sorted(summary_payloads, key=_summary_sort_key, reverse=True)
    lines = [
        "# RNG Matching Status",
        "",
        "Single source of truth for matched-seed small-graph bit-exact checks.",
        "",
        (
            "| engine | reference | best(max) RMSD over fixtures&seeds | worst fixture | "
            "verdict | exact_match_count/total | timestamp |"
        ),
        "|---|---|---:|---|---|---:|---|",
    ]
    for summary in sorted_summaries:
        max_rmsd = summary.get("max_rmsd")
        max_text = "--" if max_rmsd is None else f"{float(max_rmsd):.9e}"
        exact_text = f"{int(summary.get('exact_match_count', 0))}/{int(summary.get('total', 0))}"
        row_template = (
            "| {engine} | {reference} | {max_rmsd} | {worst_fixture} | "
            "{verdict} | {exact_text} | {timestamp} |"
        )
        lines.append(
            row_template.format(
                engine=summary.get("engine", ""),
                reference=summary.get("reference", ""),
                max_rmsd=max_text,
                worst_fixture=summary.get("worst_fixture", ""),
                verdict=summary.get("verdict", ""),
                exact_text=exact_text,
                timestamp=summary.get("timestamp", ""),
            )
        )
    STATUS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summary_sort_key(summary: dict[str, Any]) -> tuple[float, str]:
    """Return a worst-first sort key for status summaries.

    Parameters
    ----------
    summary : dict[str, Any]
        Engine summary payload.

    Returns
    -------
    tuple[float, str]
        Numeric RMSD sort key and engine name.
    """
    max_rmsd = summary.get("max_rmsd")
    if max_rmsd is None:
        verdict_rank = {
            "ERROR": -1.0,
            "UNAVAILABLE": -2.0,
            "NO_REFERENCE": -3.0,
            "NO_DATA": -4.0,
        }
        return verdict_rank.get(str(summary.get("verdict")), -5.0), str(summary.get("engine", ""))
    return float(max_rmsd), str(summary.get("engine", ""))


def run_harness(
    engine_names: Sequence[str],
    seeds: Sequence[int],
    timeout: float,
) -> list[MatchRow]:
    """Run the requested engines and return detailed rows.

    Parameters
    ----------
    engine_names : Sequence[str]
        Reimplementation variant names to run.
    seeds : Sequence[int]
        Matched seeds for stochastic variants.
    timeout : float
        Per-run competitor timeout in seconds.

    Returns
    -------
    list[MatchRow]
        Detailed comparison rows.
    """
    rows: list[MatchRow] = []
    for engine_name in engine_names:
        print(f"Running {engine_name}...", flush=True)
        engine_rows = run_engine(engine_name, seeds, timeout)
        rows.extend(engine_rows)
        summary = summarize_engine(
            engine_name,
            engine_rows,
            datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        )
        max_text = "--" if summary.max_rmsd is None else f"{summary.max_rmsd:.9e}"
        exact_text = f"{summary.exact_match_count}/{summary.total}"
        print(f"  {summary.verdict}: max={max_text} exact={exact_text}", flush=True)
    return rows


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engines",
        default=None,
        help="Comma-delimited variant IDs or base engines. Default: all registered variants.",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="Comma-delimited matched seeds. Default: 1,2,3.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Per-competitor run timeout in seconds.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the matched-seed bit-exact harness.

    Parameters
    ----------
    argv : Sequence[str] | None, default=None
        Optional argument vector for tests or helper scripts.

    Returns
    -------
    int
        Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    seeds = parse_seeds(args.seeds)
    engine_names = expand_engine_names(args.engines)
    rows: list[MatchRow] = []
    for engine_name in engine_names:
        engine_rows = run_harness([engine_name], seeds, args.timeout)
        write_status([engine_name], engine_rows)
        rows.extend(engine_rows)
    print(f"Wrote {STATUS_MD.relative_to(REPO_ROOT)}")
    print(f"Wrote {STATUS_JSON.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
