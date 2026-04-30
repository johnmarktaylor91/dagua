#!/usr/bin/env python3
"""Regenerate seeded OGDF position cache for bounded fidelity analysis."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dagua.eval.competitors import get_competitor  # noqa: E402
from dagua.eval.graphs import TestGraph, get_test_graphs  # noqa: E402
from dagua.graph import DaguaGraph  # noqa: E402

DEFAULT_OUTPUT_DIR = Path("eval_output/algo_fidelity/round_28/ogdf_seeded_cache_30")
DEFAULT_ENGINES = ("ogdf_fmmm", "ogdf_gem", "ogdf_pivot_mds", "ogdf_stress")
DEFAULT_GRAPHS = (
    "linear_3layer_mlp",
    "parallel_multiedge_bundle",
    "nested_shallow_enc_dec",
    "tl_mlp_3layer",
    "mixed_width_labels",
)
DEFAULT_SEED_START = 42
DEFAULT_SEED_STOP = 71


def parse_csv_filter(raw_values: Optional[str]) -> Optional[set[str]]:
    """Parse an optional comma-delimited filter.

    Parameters
    ----------
    raw_values : str | None
        Comma-delimited values supplied on the command line.

    Returns
    -------
    set[str] | None
        Parsed values, or ``None`` when no filter was supplied.
    """
    if raw_values is None:
        return None
    values = {value.strip() for value in raw_values.split(",") if value.strip()}
    return values or None


def seed_range(seed_start: int, seed_stop: int) -> list[int]:
    """Build an inclusive seed sequence.

    Parameters
    ----------
    seed_start : int
        First seed to generate.
    seed_stop : int
        Last seed to generate.

    Returns
    -------
    list[int]
        Inclusive integer seed sequence.

    Raises
    ------
    ValueError
        If ``seed_stop`` is smaller than ``seed_start``.
    """
    if seed_stop < seed_start:
        raise ValueError("--seed-stop must be >= --seed-start")
    return list(range(seed_start, seed_stop + 1))


def load_cached_graph(cache_path: Path) -> Optional[DaguaGraph]:
    """Load one cached graph when optional generators are unavailable.

    Parameters
    ----------
    cache_path : Path
        Local cache file that may contain a serialized ``DaguaGraph``.

    Returns
    -------
    DaguaGraph | None
        Cached graph instance, or ``None`` when the payload is not a graph.
    """
    try:
        try:
            graph = torch.load(cache_path, map_location="cpu", weights_only=False)
        except TypeError:
            graph = torch.load(cache_path, map_location="cpu")
    except Exception:
        return None
    if not isinstance(graph, DaguaGraph):
        return None
    return graph


def graph_registry() -> dict[str, TestGraph]:
    """Return evaluation test graphs keyed by name.

    Returns
    -------
    dict[str, TestGraph]
        Test graph registry including serialized TorchLens cache fallbacks.
    """
    registry = {test_graph.name: test_graph for test_graph in get_test_graphs()}
    cache_dir = REPO_ROOT / "dagua" / "eval" / "_graph_cache"
    for cache_path in sorted(cache_dir.glob("*.pt")):
        if cache_path.stem in registry:
            continue
        graph = load_cached_graph(cache_path)
        if graph is None:
            continue
        registry[cache_path.stem] = TestGraph(
            name=cache_path.stem,
            graph=graph,
            source="torchlens-cache",
            description="Cached TorchLens benchmark graph",
        )
    return registry


def selected_graphs(raw_graphs: Optional[str]) -> list[TestGraph]:
    """Return graphs selected for OGDF cache regeneration.

    Parameters
    ----------
    raw_graphs : str | None
        Optional comma-delimited graph-name filter.

    Returns
    -------
    list[TestGraph]
        Matching test graph payloads.

    Raises
    ------
    ValueError
        If a requested graph is unknown.
    """
    requested_graphs = parse_csv_filter(raw_graphs) or set(DEFAULT_GRAPHS)
    registry = graph_registry()
    missing = sorted(graph for graph in requested_graphs if graph not in registry)
    if missing:
        raise ValueError(f"Unknown graph names: {', '.join(missing)}")
    return [registry[name] for name in sorted(requested_graphs)]


def parse_engines(raw_engines: str) -> list[str]:
    """Parse a comma-delimited engine list.

    Parameters
    ----------
    raw_engines : str
        Comma-delimited OGDF competitor names.

    Returns
    -------
    list[str]
        Engine names in requested order.

    Raises
    ------
    ValueError
        If no engine names were supplied.
    """
    engines = [engine.strip() for engine in raw_engines.split(",") if engine.strip()]
    if not engines:
        raise ValueError("--engines must name at least one competitor")
    return engines


def output_path(output_dir: Path, graph_name: str, engine: str, seed: int) -> Path:
    """Return the flat cache path for one seeded OGDF layout.

    Parameters
    ----------
    output_dir : Path
        Output cache directory.
    graph_name : str
        Test graph name.
    engine : str
        OGDF competitor name.
    seed : int
        Seed value.

    Returns
    -------
    Path
        Destination tensor path.
    """
    return output_dir / f"{graph_name}__{engine}__seed{seed}.pt"


def regenerate_cache(
    graphs: Sequence[TestGraph],
    engines: Sequence[str],
    seeds: Sequence[int],
    output_dir: Path,
    timeout: float,
    overwrite: bool,
) -> dict[str, object]:
    """Run seeded OGDF competitors and write tensor cache files.

    Parameters
    ----------
    graphs : sequence[TestGraph]
        Test graphs to lay out.
    engines : sequence[str]
        OGDF competitors to run.
    seeds : sequence[int]
        Seeds to pass to each competitor.
    output_dir : Path
        Flat output directory for ``.pt`` tensors.
    timeout : float
        Maximum runtime for each OGDF subprocess.
    overwrite : bool
        Whether to overwrite existing cache files.

    Returns
    -------
    dict[str, object]
        Regeneration manifest with counts and failure records.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    competitors = {}
    for engine in engines:
        competitor = get_competitor(engine)
        if competitor is None:
            raise ValueError(f"Unknown competitor: {engine}")
        competitors[engine] = competitor

    written = 0
    skipped = 0
    failures: list[dict[str, object]] = []
    started_at = time.perf_counter()
    for test_graph in graphs:
        test_graph.graph.compute_node_sizes()
        for engine, competitor in competitors.items():
            for seed in seeds:
                path = output_path(output_dir, test_graph.name, engine, int(seed))
                if path.exists() and not overwrite:
                    skipped += 1
                    continue

                result = competitor.layout(test_graph.graph, timeout=timeout, seed=int(seed))
                if result.error is not None or result.pos is None:
                    failures.append(
                        {
                            "graph": test_graph.name,
                            "engine": engine,
                            "seed": int(seed),
                            "error": result.error or "missing_positions",
                        }
                    )
                    continue
                positions = result.pos.detach().to(device="cpu", dtype=torch.float32)
                torch.save(positions, path)
                written += 1

    manifest: dict[str, object] = {
        "output_dir": str(output_dir),
        "graphs": [test_graph.name for test_graph in graphs],
        "engines": list(engines),
        "seeds": [int(seed) for seed in seeds],
        "written": written,
        "skipped": skipped,
        "failures": failures,
        "expected_entries": len(graphs) * len(engines) * len(seeds),
        "runtime_seconds": time.perf_counter() - started_at,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if failures:
        raise RuntimeError(f"OGDF seed regeneration failed for {len(failures)} layouts")
    return manifest


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv : Sequence[str] | None
        Optional argument vector for tests.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--graphs", default=",".join(DEFAULT_GRAPHS))
    parser.add_argument("--engines", default=",".join(DEFAULT_ENGINES))
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--seed-stop", type=int, default=DEFAULT_SEED_STOP)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run OGDF seed cache regeneration.

    Parameters
    ----------
    argv : Sequence[str] | None
        Optional argument vector for tests.

    Returns
    -------
    int
        Process exit code.
    """
    args = parse_args(argv)
    manifest = regenerate_cache(
        graphs=selected_graphs(args.graphs),
        engines=parse_engines(args.engines),
        seeds=seed_range(args.seed_start, args.seed_stop),
        output_dir=args.output_dir,
        timeout=args.timeout,
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
