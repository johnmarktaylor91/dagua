#!/usr/bin/env python3
"""Validate that results.json and positions.h5 are in sync.

Every 'ok' record in results.json must have a corresponding key in
positions.h5. Every key in positions.h5 must have a corresponding
record in results.json. Exits nonzero if ANY desync found.

Written as enforcement code from retro 2026-03-30 after 2 days of
wasted compute due to results.json/positions.h5 desync.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

REFERENCE_VARIANT_SEPARATOR = "__for__"
PARAM_SENSITIVITY_SAMPLE_SEEDS = 3
MATCHED_SEED_WARN_THRESHOLD = 100
CLAMP_EQUIVALENT_WHITELIST = {
    (
        "umap_graph",
        "parallel_multiedge_bundle",
    ): (
        "r76_final_sprint_STATE.md 2026-07-03: UMAP n_neighbors clamps on the tiny "
        "parallel_multiedge_bundle graph, making default/nn5/nn30 identical."
    ),
    (
        "graphviz_sfdp",
        "*",
    ): (
        "r76_final_sprint_STATE.md 2026-07-03: installed Graphviz 7.0.5 ignores the "
        "theta, maxiter/steps, and repulsiveforce p_neg2 attributes for these "
        "synthetic reference variants."
    ),
}


@dataclass(frozen=True)
class ResultRow:
    """Benchmark row annotated with its source directory.

    Parameters
    ----------
    key : str
        Flat results.json key.
    data_dir : Path
        Directory that contains the row's results.json and positions.
    payload : dict[str, Any]
        JSON row payload.
    """

    key: str
    data_dir: Path
    payload: dict[str, Any]


def validate_sync(
    results_path: Path,
    h5_path: Path,
    engines: set[str] | None = None,
) -> list[str]:
    """Check results.json and positions.h5 are in sync.

    Parameters
    ----------
    results_path : Path
        Path to results.json.
    h5_path : Path
        Path to positions.h5.
    engines : set[str] | None
        If provided, only validate these engine names.

    Returns
    -------
    list[str]
        Error messages. Empty list means sync is valid.
    """
    import h5py

    errors: list[str] = []

    with open(results_path) as f:
        results = json.load(f)

    # Build set of keys that should have positions (status=ok, not --no-positions)
    rj_ok_keys: set[str] = set()
    for key, record in results.items():
        eng = record.get("engine_name", record.get("engine", ""))
        if engines and eng not in engines:
            continue
        status = record.get("status", "")
        if status == "ok":
            rj_ok_keys.add(key)

    # Build set of H5 keys
    if not h5_path.exists():
        if rj_ok_keys:
            errors.append(
                f"positions.h5 does not exist but results.json has {len(rj_ok_keys)} ok records"
            )
        return errors

    with h5py.File(h5_path, "r") as h5f:
        h5_keys: set[str] = set(h5f.keys())
        if engines:
            h5_keys = {k for k in h5_keys if "::" in k and k.split("::")[1] in engines}

    # Check for results without positions
    missing_positions = rj_ok_keys - h5_keys
    if missing_positions:
        # Group by engine for readable output
        by_engine: dict[str, int] = {}
        for key in missing_positions:
            parts = key.split("::")
            eng = parts[1] if len(parts) >= 2 else "unknown"
            by_engine[eng] = by_engine.get(eng, 0) + 1
        for eng, count in sorted(by_engine.items(), key=lambda x: -x[1]):
            errors.append(f"DESYNC: {eng} has {count} ok results but missing positions")

    # Check for orphaned positions (H5 keys without results)
    orphaned = h5_keys - rj_ok_keys
    if orphaned:
        by_engine_orphan: dict[str, int] = {}
        for key in orphaned:
            parts = key.split("::")
            eng = parts[1] if len(parts) >= 2 else "unknown"
            by_engine_orphan[eng] = by_engine_orphan.get(eng, 0) + 1
        for eng, count in sorted(by_engine_orphan.items(), key=lambda x: -x[1]):
            errors.append(f"ORPHAN: {eng} has {count} positions but no ok result")

    return errors


def load_result_rows(data_dirs: list[Path]) -> list[ResultRow]:
    """Load benchmark rows from one or more results directories.

    Parameters
    ----------
    data_dirs : list[Path]
        Benchmark output directories. Later directories override earlier rows
        with the same flat key, matching definitive-analysis overlay rules.

    Returns
    -------
    list[ResultRow]
        Overlayed rows with source directory provenance.
    """
    rows_by_key: dict[str, ResultRow] = {}
    for data_dir in data_dirs:
        results_path = data_dir / "results.json"
        if not results_path.exists():
            continue
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        for key, row_payload in payload.items():
            if isinstance(row_payload, dict):
                rows_by_key[str(key)] = ResultRow(str(key), data_dir, row_payload)
    return list(rows_by_key.values())


def row_graph(row: ResultRow) -> str:
    """Return the graph name for a result row.

    Parameters
    ----------
    row : ResultRow
        Benchmark row.

    Returns
    -------
    str
        Stable graph name or an empty string when absent.
    """
    return str(row.payload.get("graph_name", row.payload.get("graph", "")))


def row_engine(row: ResultRow) -> str:
    """Return the engine name for a result row.

    Parameters
    ----------
    row : ResultRow
        Benchmark row.

    Returns
    -------
    str
        Stable engine name or an empty string when absent.
    """
    return str(row.payload.get("engine_name", row.payload.get("engine", "")))


def row_seed(row: ResultRow) -> int | None:
    """Return the stochastic seed for a result row.

    Parameters
    ----------
    row : ResultRow
        Benchmark row.

    Returns
    -------
    int | None
        Integer seed or ``None`` for deterministic rows.
    """
    seed = row.payload.get("seed")
    return None if seed is None else int(seed)


def row_status(row: ResultRow) -> str:
    """Return the execution status for a result row.

    Parameters
    ----------
    row : ResultRow
        Benchmark row.

    Returns
    -------
    str
        Status string.
    """
    return str(row.payload.get("status", ""))


def row_positions_path(row: ResultRow) -> Path | None:
    """Resolve the position tensor path for a result row.

    Parameters
    ----------
    row : ResultRow
        Benchmark row.

    Returns
    -------
    Path | None
        Absolute path to the saved position tensor, or ``None`` when the row
        has no positions_file field.
    """
    positions_file = row.payload.get("positions_file")
    if positions_file is None:
        return None
    path = Path(str(positions_file))
    return path if path.is_absolute() else row.data_dir / path


def reference_family(engine_name: str) -> str | None:
    """Return the synthetic reference family for a ``__for__`` engine.

    Parameters
    ----------
    engine_name : str
        Engine name from results.json.

    Returns
    -------
    str | None
        Prefix before ``__for__`` when the row is a parameterized reference
        variant, otherwise ``None``.
    """
    if REFERENCE_VARIANT_SEPARATOR not in engine_name:
        return None
    return engine_name.split(REFERENCE_VARIANT_SEPARATOR, 1)[0]


def whitelist_reason(reference_engine: str, graph_name: str) -> str | None:
    """Return the documented whitelist reason for an engine/graph pair.

    Parameters
    ----------
    reference_engine : str
        Reference family engine name.
    graph_name : str
        Graph name.

    Returns
    -------
    str | None
        Evidence string when the pair is explicitly exempt.
    """
    return CLAMP_EQUIVALENT_WHITELIST.get(
        (reference_engine, graph_name)
    ) or CLAMP_EQUIVALENT_WHITELIST.get((reference_engine, "*"))


def load_position_tensor(path: Path) -> torch.Tensor:
    """Load a saved position tensor from disk.

    Parameters
    ----------
    path : Path
        Tensor path.

    Returns
    -------
    torch.Tensor
        CPU tensor payload.
    """
    tensor = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)
    return tensor.detach().cpu()


def validate_param_sensitivity(rows: list[ResultRow]) -> list[str]:
    """Detect reference parameter variants that produce identical tensors.

    Parameters
    ----------
    rows : list[ResultRow]
        Overlayed benchmark rows.

    Returns
    -------
    list[str]
        Error messages for non-whitelisted param-insensitive families.
    """
    by_family: dict[tuple[str, str], dict[str, dict[int, ResultRow]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in rows:
        if row_status(row) != "ok":
            continue
        seed = row_seed(row)
        if seed is None:
            continue
        engine = row_engine(row)
        family = reference_family(engine)
        if family is None:
            continue
        by_family[(row_graph(row), family)][engine][seed] = row

    errors: list[str] = []
    for (graph_name, family), variants in sorted(by_family.items()):
        if len(variants) < 2:
            continue
        common_seeds = set.intersection(*(set(seed_rows.keys()) for seed_rows in variants.values()))
        if not common_seeds:
            continue
        sampled_seeds = sorted(common_seeds)[:PARAM_SENSITIVITY_SAMPLE_SEEDS]
        all_sampled_identical = True
        for seed in sampled_seeds:
            tensors = []
            for variant_name in sorted(variants):
                path = row_positions_path(variants[variant_name][seed])
                if path is None or not path.exists():
                    all_sampled_identical = False
                    break
                tensors.append(load_position_tensor(path))
            if not tensors or any(not torch.equal(tensors[0], tensor) for tensor in tensors[1:]):
                all_sampled_identical = False
                break
        if not all_sampled_identical:
            continue
        reason = whitelist_reason(family, graph_name)
        if reason is not None:
            print(
                "WHITELIST param-equivalent: "
                f"{family} on {graph_name}; variants={', '.join(sorted(variants))}; {reason}"
            )
            continue
        errors.append(
            "PARAM-SENSITIVITY FAIL: "
            f"{family} on {graph_name} produced bit-identical positions for seeds "
            f"{sampled_seeds} across variants: {', '.join(sorted(variants))}"
        )
    return errors


def format_seed_range(seeds: set[int]) -> str:
    """Format a seed set as a compact range string.

    Parameters
    ----------
    seeds : set[int]
        Seed values.

    Returns
    -------
    str
        Human-readable seed range and count.
    """
    if not seeds:
        return "none (n=0)"
    return f"{min(seeds)}..{max(seeds)} (n={len(seeds)})"


def validate_seed_era(rows: list[ResultRow]) -> list[str]:
    """Warn when reimplementation/reference rows have weak seed overlap.

    Parameters
    ----------
    rows : list[ResultRow]
        Overlayed benchmark rows.

    Returns
    -------
    list[str]
        Warning messages for combos below the matched-seed threshold.
    """
    seeds_by_pair: dict[tuple[str, str], set[int]] = defaultdict(set)
    references_by_pair: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        if row_status(row) != "ok":
            continue
        seed = row_seed(row)
        if seed is not None:
            seeds_by_pair[(row_graph(row), row_engine(row))].add(seed)
        for reference in row.payload.get("reimpl_of", []) or []:
            references_by_pair[(row_graph(row), row_engine(row))].add(str(reference))

    warnings: list[str] = []
    for (graph_name, engine_name), references in sorted(references_by_pair.items()):
        reimpl_seeds = seeds_by_pair.get((graph_name, engine_name), set())
        if not reimpl_seeds:
            continue
        for reference in sorted(references):
            ref_seeds = seeds_by_pair.get((graph_name, reference), set())
            if not ref_seeds:
                continue
            matched = reimpl_seeds & ref_seeds
            if len(matched) < MATCHED_SEED_WARN_THRESHOLD:
                warnings.append(
                    "SEED-ERA WARN: "
                    f"{graph_name}::{engine_name} vs {reference} matched {len(matched)} seeds; "
                    f"dagua/reimpl seeds {format_seed_range(reimpl_seeds)}; "
                    f"reference seeds {format_seed_range(ref_seeds)}"
                )
    return warnings


def validate_for_row_counts(data_dirs: list[Path]) -> list[str]:
    """Ensure seeded-reference benchmark runs emitted each requested variant.

    Parameters
    ----------
    data_dirs : list[Path]
        Benchmark output directories.

    Returns
    -------
    list[str]
        Error messages for manifests with zero rows for a requested ``__for__``
        engine variant.
    """
    errors: list[str] = []
    for data_dir in data_dirs:
        manifest_path = data_dir / "manifest.json"
        results_path = data_dir / "results.json"
        if not manifest_path.exists() or not results_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        seed_refs = manifest.get("config", {}).get("seed_refs", [])
        if not seed_refs:
            continue
        requested_for_engines = [
            str(engine.get("name"))
            for engine in manifest.get("engines", [])
            if REFERENCE_VARIANT_SEPARATOR in str(engine.get("name"))
        ]
        if not requested_for_engines:
            continue
        results = json.loads(results_path.read_text(encoding="utf-8"))
        counts = {engine: 0 for engine in requested_for_engines}
        for row_payload in results.values():
            if not isinstance(row_payload, dict):
                continue
            engine = str(row_payload.get("engine_name", row_payload.get("engine", "")))
            if engine in counts:
                counts[engine] += 1
        missing = [engine for engine, count in counts.items() if count == 0]
        if missing:
            errors.append(
                f"__for__ ROW-COUNT FAIL in {data_dir}: zero rows for {', '.join(missing)}"
            )
    return errors


def main() -> None:
    """Run sync validation from CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        action="append",
        default=None,
        help="Directory containing results.json and positions.h5",
    )
    parser.add_argument(
        "--engines",
        type=str,
        default=None,
        help="Comma-separated engine names to validate (default: all)",
    )
    args = parser.parse_args()

    data_dirs = args.data_dir or [Path("eval_output/variant_bench_full")]
    engine_set = set(args.engines.split(",")) if args.engines else None

    errors: list[str] = []
    for data_dir in data_dirs:
        results_path = data_dir / "results.json"
        h5_path = data_dir / "positions.h5"
        if not results_path.exists():
            print(f"ERROR: {results_path} not found", file=sys.stderr)
            sys.exit(1)
        if h5_path.exists():
            errors.extend(validate_sync(results_path, h5_path, engine_set))

    rows = load_result_rows(data_dirs)
    errors.extend(validate_param_sensitivity(rows))
    errors.extend(validate_for_row_counts(data_dirs))
    warnings = validate_seed_era(rows)
    for warning in warnings:
        print(warning, file=sys.stderr)

    if errors:
        print(
            f"VALIDATION FAILED: {len(errors)} integrity errors",
            file=sys.stderr,
        )
        for err in errors:
            print(f"  {err}", file=sys.stderr)
        sys.exit(1)
    else:
        print("Benchmark integrity OK")


if __name__ == "__main__":
    main()
