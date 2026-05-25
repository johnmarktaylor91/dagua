#!/usr/bin/env python3
"""Live-compare one dagua competitor against cached benchmark positions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
from statsmodels.stats.weightstats import ttost_ind

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dagua.eval.competitors import CompetitorBase, get_competitor  # noqa: E402
from dagua.eval.competitors.classic_competitor import VariantCompetitor  # noqa: E402
from dagua.eval.graphs import TestGraph, get_test_graphs  # noqa: E402
from dagua.eval.pipeline_io import load_position_tensor, validate_positions  # noqa: E402
from dagua.eval.variants import get_variant, get_variant_for_original_name  # noqa: E402
from dagua.graph import DaguaGraph  # noqa: E402
from scripts.algo_fidelity_cross import (  # noqa: E402
    fidelity_procrustes,
    load_results,
    load_valid_positions,
    percentile,
    select_ok_record,
)
from scripts.algo_fidelity_panel import draw_layout, normalize_for_panel  # noqa: E402

TOST_ALPHA = 0.05
TOST_MARGIN_FACTORS: tuple[float, ...] = (0.25, 0.5, 1.0, 1.5, 2.0)
TOST_MARGIN_LABELS: dict[float, str] = {
    0.25: "0.25x",
    0.5: "0.5x",
    1.0: "1x",
    1.5: "1.5x",
    2.0: "2x",
}
DEFAULT_SEED_START = 42
CACHED_SEED_STOP = 50


def resolve_competitor(engine_name: str) -> Optional[CompetitorBase]:
    """Return a base, reimplementation-variant, or original-variant competitor.

    Parameters
    ----------
    engine_name : str
        Registered base competitor name, reimplementation variant name, or
        synthetic original-side variant name.

    Returns
    -------
    CompetitorBase | None
        Runnable competitor instance, or ``None`` when the name cannot be
        resolved in the local registry.
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
        max_nodes=original_variant.max_nodes,
    )


def parse_graph_filter(raw_graphs: Optional[str]) -> Optional[set[str]]:
    """Parse the optional comma-delimited graph filter.

    Parameters
    ----------
    raw_graphs : str, optional
        Comma-delimited graph names supplied on the command line.

    Returns
    -------
    set of str, optional
        Requested graph names, or ``None`` when no filter was provided.
    """
    if raw_graphs is None:
        return None
    graph_names = {graph.strip() for graph in raw_graphs.split(",") if graph.strip()}
    return graph_names or None


def graph_registry() -> dict[str, TestGraph]:
    """Return evaluation test graphs keyed by name.

    Returns
    -------
    dict[str, TestGraph]
        Test graph registry from :func:`dagua.eval.graphs.get_test_graphs`.
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


def load_cached_graph(cache_path: Path) -> Optional[DaguaGraph]:
    """Load one cached graph tensor when optional generators are unavailable.

    Parameters
    ----------
    cache_path : Path
        Local cache file that may contain a serialized ``DaguaGraph``.

    Returns
    -------
    DaguaGraph, optional
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


def target_graphs(
    indexed: Mapping[str, list[Any]],
    dagua_engine: str,
    target_engine: str,
    requested_graphs: Optional[set[str]],
) -> list[str]:
    """Select graph names with successful cached target positions.

    Parameters
    ----------
    indexed : Mapping[str, list[Any]]
        Benchmark records returned by ``load_results``.
    dagua_engine : str
        Dagua engine name. The default graph set is constrained to benchmark
        graphs where this side was successful so live numbers can be compared
        directly with Round 1 cached pairwise rows.
    target_engine : str
        Cached target engine name.
    requested_graphs : set of str, optional
        Optional explicit graph-name filter.

    Returns
    -------
    list of str
        Sorted graph names that have successful cached target positions.
    """
    if requested_graphs is not None:
        return [
            graph
            for graph in sorted(requested_graphs)
            if select_ok_record(indexed=indexed, graph=graph, engine=target_engine) is not None
        ]

    graph_names = sorted(
        key.rsplit("::", 1)[0] for key in indexed if key.endswith(f"::{target_engine}")
    )
    return [
        graph
        for graph in graph_names
        if select_ok_record(indexed=indexed, graph=graph, engine=target_engine) is not None
        and select_ok_record(indexed=indexed, graph=graph, engine=dagua_engine) is not None
    ]


def load_cached_target(
    indexed: Mapping[str, list[Any]],
    graph: str,
    target_engine: str,
    input_dir: Path,
) -> torch.Tensor:
    """Load one cached target position tensor.

    Parameters
    ----------
    indexed : Mapping[str, list[Any]]
        Benchmark records returned by ``load_results``.
    graph : str
        Graph name.
    target_engine : str
        Target engine name.
    input_dir : Path
        Benchmark root containing ``results.json`` and position tensors.

    Returns
    -------
    torch.Tensor
        Cached target positions with shape ``[N, 2]``.

    Raises
    ------
    ValueError
        If no successful cached target positions can be loaded.
    """
    target_record = select_ok_record(indexed=indexed, graph=graph, engine=target_engine)
    if target_record is None:
        raise ValueError(f"No successful cached target record for {graph}/{target_engine}")
    target_positions, error = load_valid_positions(record=target_record, input_dir=input_dir)
    if error or target_positions is None:
        raise ValueError(f"Could not load cached target positions for {graph}: {error}")
    return target_positions


def seed_sequence(seed_count: int) -> list[int]:
    """Return the canonical Round 8 seed sequence.

    Parameters
    ----------
    seed_count : int
        Requested number of seeds.

    Returns
    -------
    list of int
        Seeds starting at 42.
    """
    return list(range(DEFAULT_SEED_START, DEFAULT_SEED_START + seed_count))


def load_cached_target_seeds(
    graph: str,
    target_engine: str,
    input_dir: Path,
    expected_nodes: int,
    max_seeds: int,
    graphviz_cache_dir: Optional[Path] = None,
) -> dict[int, torch.Tensor]:
    """Load up to ``max_seeds`` cached target position tensors.

    Parameters
    ----------
    graph : str
        Benchmark graph name.
    target_engine : str
        Cached target engine name.
    input_dir : Path
        Benchmark root containing ``positions/`` tensors.
    expected_nodes : int
        Expected node count for position validation.
    max_seeds : int
        Maximum number of cached seeds to load.
    graphviz_cache_dir : Path, optional
        Optional flat directory containing graphviz seed tensors named
        ``<graph>__<engine>__seed<S>.pt``. When omitted, tensors are loaded
        from ``input_dir / "positions"`` using the historical benchmark cache
        layout and the original 42..50 seed window.

    Returns
    -------
    dict[int, torch.Tensor]
        Loaded target positions keyed by seed.
    """
    positions_by_seed: dict[int, torch.Tensor] = {}
    if graphviz_cache_dir is not None:
        target_seeds = seed_sequence(max_seeds)
    else:
        historical_seed_count = CACHED_SEED_STOP - DEFAULT_SEED_START + 1
        target_seeds = seed_sequence(min(max_seeds, historical_seed_count))

    for seed in target_seeds:
        basename = f"{graph}__{target_engine}__seed{seed}.pt"
        if graphviz_cache_dir is not None:
            positions_path = graphviz_cache_dir / basename
            if not positions_path.exists():
                continue
            try:
                loaded = torch.load(positions_path, map_location="cpu")
            except Exception:
                continue
            if not isinstance(loaded, torch.Tensor):
                continue
            positions = loaded.detach().to(dtype=torch.float32, device="cpu")
            error = None
        else:
            positions_file = f"positions/{basename}"
            positions_path = input_dir / positions_file
            if not positions_path.exists():
                continue
            positions, error = load_position_tensor(
                record_key=None,
                positions_file=positions_file,
                input_dir=input_dir,
            )
        if error is not None or positions is None:
            continue
        validation_error = validate_positions(positions, expected_nodes=expected_nodes)
        if validation_error is not None:
            continue
        positions_by_seed[seed] = positions.detach().to(device="cpu", dtype=torch.float32)
        if len(positions_by_seed) >= max_seeds:
            break
    return positions_by_seed


def run_dagua_seeded_layouts(
    dagua_engine: str,
    test_graph: TestGraph,
    seeds: Sequence[int],
) -> dict[int, torch.Tensor]:
    """Run live layouts for each requested seed.

    Parameters
    ----------
    dagua_engine : str
        Registered base competitor name or variant name to run.
    test_graph : TestGraph
        Evaluation graph payload.
    seeds : sequence of int
        Seeds to pass to the competitor.

    Returns
    -------
    dict[int, torch.Tensor]
        Live position tensors keyed by seed.

    Raises
    ------
    ValueError
        If the competitor is unavailable or a layout run fails.
    """
    competitor = resolve_competitor(dagua_engine)
    if competitor is None:
        raise ValueError(f"Unknown competitor: {dagua_engine}")

    positions_by_seed: dict[int, torch.Tensor] = {}
    for seed in seeds:
        result = competitor.layout(test_graph.graph, seed=seed)
        if result.error is not None or result.pos is None:
            raise ValueError(
                f"Live layout failed for {test_graph.name}/{dagua_engine}/seed{seed}: "
                f"{result.error}"
            )
        positions_by_seed[int(seed)] = result.pos.detach().to(device="cpu", dtype=torch.float32)
    return positions_by_seed


def summarize_values(values: Sequence[float]) -> dict[str, Optional[float]]:
    """Summarize a finite RMSD distribution.

    Parameters
    ----------
    values : sequence of float
        RMSD values.

    Returns
    -------
    dict[str, float | None]
        Mean, median, and p95 values, or ``None`` values for an empty sample.
    """
    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    if not finite_values:
        return {"mean": None, "median": None, "p95": None}
    return {
        "mean": float(statistics.fmean(finite_values)),
        "median": float(statistics.median(finite_values)),
        "p95": float(percentile(finite_values, 0.95)),
    }


def tost_equivalence_verdicts(
    between_values: Sequence[float],
    within_values: Sequence[float],
) -> dict[str, dict[str, Optional[float] | str | bool]]:
    """Run TOST tests against within-target stochastic floor margins.

    Parameters
    ----------
    between_values : sequence of float
        Dagua-vs-target pairwise RMSDs.
    within_values : sequence of float
        Target-vs-target pairwise RMSDs.

    Returns
    -------
    dict[str, dict[str, float | str | bool | None]]
        Per-margin TOST metadata and verdict labels.
    """
    verdicts: dict[str, dict[str, Optional[float] | str | bool]] = {}
    finite_between = np.asarray(
        [float(value) for value in between_values if math.isfinite(float(value))],
        dtype=np.float64,
    )
    finite_within = np.asarray(
        [float(value) for value in within_values if math.isfinite(float(value))],
        dtype=np.float64,
    )
    if finite_between.size < 2 or finite_within.size < 2:
        for factor in TOST_MARGIN_FACTORS:
            label = TOST_MARGIN_LABELS[factor]
            verdicts[label] = {
                "margin": None,
                "pvalue": None,
                "equivalent": False,
                "verdict": "not_tested",
            }
        return verdicts

    floor_mean = float(finite_within.mean())
    for factor in TOST_MARGIN_FACTORS:
        label = TOST_MARGIN_LABELS[factor]
        margin = max(factor * floor_mean, 1e-6)
        try:
            pvalue, _, _ = ttost_ind(
                finite_between,
                finite_within,
                low=-margin,
                upp=margin,
                usevar="unequal",
            )
            pvalue_float = float(pvalue)
        except Exception:
            pvalue_float = math.nan
        equivalent = math.isfinite(pvalue_float) and pvalue_float <= TOST_ALPHA
        verdicts[label] = {
            "margin": float(margin),
            "pvalue": pvalue_float if math.isfinite(pvalue_float) else None,
            "equivalent": bool(equivalent),
            "verdict": f"equivalent_at_{label}" if equivalent else "not_equivalent",
        }
    return verdicts


def pairwise_rmsd_rows(
    graph: str,
    side: str,
    positions_a: Mapping[int, torch.Tensor],
    positions_b: Mapping[int, torch.Tensor],
    *,
    unique_pairs: bool,
) -> list[dict[str, Any]]:
    """Compute pairwise Procrustes RMSDs for one comparison side.

    Parameters
    ----------
    graph : str
        Benchmark graph name.
    side : str
        Output side label.
    positions_a : Mapping[int, torch.Tensor]
        First seed-position mapping.
    positions_b : Mapping[int, torch.Tensor]
        Second seed-position mapping.
    unique_pairs : bool
        When true, compare only unique seed pairs within one mapping.

    Returns
    -------
    list of dict[str, Any]
        CSV-ready rows containing graph, side, seeds, and RMSD.
    """
    rows: list[dict[str, Any]] = []
    seeds_a = sorted(positions_a)
    seeds_b = sorted(positions_b)
    if unique_pairs:
        for index, seed_a in enumerate(seeds_a):
            for seed_b in seeds_a[index + 1 :]:
                rmsd, _ = fidelity_procrustes(positions_a[seed_a], positions_a[seed_b])
                rows.append(
                    {"graph": graph, "side": side, "seed_a": seed_a, "seed_b": seed_b, "rmsd": rmsd}
                )
        return rows

    for seed_a in seeds_a:
        for seed_b in seeds_b:
            rmsd, _ = fidelity_procrustes(positions_a[seed_a], positions_b[seed_b])
            rows.append(
                {"graph": graph, "side": side, "seed_a": seed_a, "seed_b": seed_b, "rmsd": rmsd}
            )
    return rows


def render_live_panel(
    graph_name: str,
    dagua_engine: str,
    target_engine: str,
    live_positions: torch.Tensor,
    target_positions: torch.Tensor,
    edge_index: torch.Tensor,
    output_path: Path,
) -> None:
    """Render a side-by-side panel for a live dagua run and cached target.

    Parameters
    ----------
    graph_name : str
        Benchmark graph name.
    dagua_engine : str
        Live dagua competitor name.
    target_engine : str
        Cached target engine name.
    live_positions : torch.Tensor
        Live dagua positions with shape ``[N, 2]``.
    target_positions : torch.Tensor
        Cached target positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Graph edge index with shape ``[2, E]``.
    output_path : Path
        Destination PNG path.
    """
    import matplotlib.pyplot as plt

    aligned_live, aligned_target, rmsd = normalize_for_panel(live_positions, target_positions)
    all_positions = torch.cat([aligned_live, aligned_target], dim=0)
    min_xy = all_positions.min(dim=0).values
    max_xy = all_positions.max(dim=0).values
    span = torch.clamp(max_xy - min_xy, min=1e-6)
    margin = 0.08 * float(span.max().item())

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=160)
    draw_layout(axes[0], aligned_live, edge_index.detach().cpu(), dagua_engine)
    draw_layout(axes[1], aligned_target, edge_index.detach().cpu(), target_engine)
    for axis in axes:
        axis.set_xlim(float(min_xy[0]) - margin, float(max_xy[0]) + margin)
        axis.set_ylim(float(min_xy[1]) - margin, float(max_xy[1]) + margin)
    fig.suptitle(f"{graph_name} | live RMSD {rmsd:.4f}", fontsize=12)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def compare_live(
    dagua_engine: str,
    target_engine: str,
    graphs: Sequence[str],
    test_graphs: Mapping[str, TestGraph],
    indexed: Mapping[str, list[Any]],
    input_dir: Path,
    output_dir: Path,
    render_panels: bool,
) -> list[dict[str, Any]]:
    """Run live dagua layouts and compare them with cached targets.

    Parameters
    ----------
    dagua_engine : str
        Registered dagua competitor name to run live.
    target_engine : str
        Cached target engine name.
    graphs : sequence of str
        Graph names to compare.
    test_graphs : Mapping[str, TestGraph]
        Test graph registry keyed by graph name.
    indexed : Mapping[str, list[Any]]
        Benchmark records returned by ``load_results``.
    input_dir : Path
        Benchmark root containing cached target positions.
    output_dir : Path
        Output directory for CSV and optional panels.
    render_panels : bool
        Whether to render panel PNGs.

    Returns
    -------
    list of dict
        CSV-ready comparison rows.

    Raises
    ------
    ValueError
        If the competitor or requested graph data is unavailable.
    """
    competitor = resolve_competitor(dagua_engine)
    if competitor is None:
        raise ValueError(f"Unknown dagua competitor: {dagua_engine}")

    rows: list[dict[str, Any]] = []
    for graph_name in graphs:
        test_graph = test_graphs.get(graph_name)
        if test_graph is None:
            raise ValueError(f"Target graph is absent from get_test_graphs(): {graph_name}")

        test_graph.graph.compute_node_sizes()
        target_positions: Optional[torch.Tensor]
        try:
            target_positions = load_cached_target(
                indexed=indexed,
                graph=graph_name,
                target_engine=target_engine,
                input_dir=input_dir,
            )
        except ValueError:
            target_positions_by_seed = run_dagua_seeded_layouts(
                dagua_engine=target_engine,
                test_graph=test_graph,
                seeds=[DEFAULT_SEED_START],
            )
            target_positions = target_positions_by_seed[DEFAULT_SEED_START]
        result = competitor.layout(test_graph.graph, seed=42)
        if result.error is not None or result.pos is None:
            raise ValueError(f"Live layout failed for {graph_name}/{dagua_engine}: {result.error}")
        live_positions = result.pos.detach().to(device="cpu", dtype=torch.float32)
        target_positions = target_positions.detach().to(device="cpu", dtype=torch.float32)
        if live_positions.shape[0] != target_positions.shape[0]:
            raise ValueError(
                f"Node count mismatch for {graph_name}: "
                f"live={live_positions.shape[0]} target={target_positions.shape[0]}"
            )

        rmsd, _ = fidelity_procrustes(live_positions, target_positions)
        rows.append(
            {
                "graph": graph_name,
                "n_nodes": int(test_graph.graph.num_nodes),
                "dagua_engine": dagua_engine,
                "target_engine": target_engine,
                "rmsd": rmsd,
            }
        )
        if render_panels:
            panel_name = f"{graph_name}__{dagua_engine}__{target_engine}.png"
            panel_path = output_dir / "panels" / panel_name
            render_live_panel(
                graph_name=graph_name,
                dagua_engine=dagua_engine,
                target_engine=target_engine,
                live_positions=live_positions,
                target_positions=target_positions,
                edge_index=test_graph.graph.edge_index,
                output_path=panel_path,
            )
    return rows


def compare_live_multi_seed(
    dagua_engine: str,
    target_engine: str,
    graphs: Sequence[str],
    test_graphs: Mapping[str, TestGraph],
    indexed: Mapping[str, list[Any]],
    input_dir: Path,
    seed_count: int,
    graphviz_cache_dir: Optional[Path] = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run multi-seed live comparison against cached target seeds.

    Parameters
    ----------
    dagua_engine : str
        Registered dagua competitor name to run live.
    target_engine : str
        Cached target engine name.
    graphs : sequence of str
        Graph names to compare.
    test_graphs : Mapping[str, TestGraph]
        Test graph registry keyed by graph name.
    indexed : Mapping[str, list[Any]]
        Benchmark records returned by ``load_results``.
    input_dir : Path
        Benchmark root containing cached target positions.
    seed_count : int
        Requested seed count. Deterministic dagua engines force this to one.
    graphviz_cache_dir : Path, optional
        Optional flat directory containing seeded graphviz target positions.

    Returns
    -------
    tuple[list[dict[str, Any]], dict[str, Any]]
        Pairwise RMSD CSV rows and per-graph JSON summary payload.

    Raises
    ------
    ValueError
        If requested graph data is unavailable.
    """
    dagua_seed_count = seed_count
    dagua_seeds = seed_sequence(dagua_seed_count)
    target_max_seeds = max(
        1,
        seed_count
        if graphviz_cache_dir is not None
        else min(
            seed_count,
            CACHED_SEED_STOP - DEFAULT_SEED_START + 1,
        ),
    )

    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "dagua_engine": dagua_engine,
        "target_engine": target_engine,
        "requested_seeds": int(seed_count),
        "effective_dagua_seeds": int(dagua_seed_count),
        "graphs": {},
    }
    for graph_name in graphs:
        test_graph = test_graphs.get(graph_name)
        if test_graph is None:
            raise ValueError(f"Target graph is absent from get_test_graphs(): {graph_name}")

        test_graph.graph.compute_node_sizes()
        target_positions_by_seed = load_cached_target_seeds(
            graph=graph_name,
            target_engine=target_engine,
            input_dir=input_dir,
            expected_nodes=int(test_graph.graph.num_nodes),
            max_seeds=target_max_seeds,
            graphviz_cache_dir=graphviz_cache_dir,
        )
        if not target_positions_by_seed:
            if resolve_competitor(target_engine) is not None:
                target_positions_by_seed = run_dagua_seeded_layouts(
                    dagua_engine=target_engine,
                    test_graph=test_graph,
                    seeds=dagua_seeds,
                )
            else:
                target_positions = load_cached_target(
                    indexed=indexed,
                    graph=graph_name,
                    target_engine=target_engine,
                    input_dir=input_dir,
                )
                target_positions_by_seed = {
                    DEFAULT_SEED_START: target_positions.detach().to(
                        device="cpu",
                        dtype=torch.float32,
                    )
                }

        dagua_positions_by_seed = run_dagua_seeded_layouts(
            dagua_engine=dagua_engine,
            test_graph=test_graph,
            seeds=dagua_seeds,
        )
        for seed, positions in dagua_positions_by_seed.items():
            if positions.shape[0] != int(test_graph.graph.num_nodes):
                raise ValueError(
                    f"Node count mismatch for {graph_name}/seed{seed}: "
                    f"live={positions.shape[0]} graph={test_graph.graph.num_nodes}"
                )
        for seed, positions in target_positions_by_seed.items():
            if positions.shape[0] != int(test_graph.graph.num_nodes):
                raise ValueError(
                    f"Node count mismatch for {graph_name}/target_seed{seed}: "
                    f"target={positions.shape[0]} graph={test_graph.graph.num_nodes}"
                )

        between_rows = pairwise_rmsd_rows(
            graph=graph_name,
            side="dagua_vs_graphviz",
            positions_a=dagua_positions_by_seed,
            positions_b=target_positions_by_seed,
            unique_pairs=False,
        )
        within_target_rows = pairwise_rmsd_rows(
            graph=graph_name,
            side="within_graphviz",
            positions_a=target_positions_by_seed,
            positions_b=target_positions_by_seed,
            unique_pairs=True,
        )
        within_dagua_rows = pairwise_rmsd_rows(
            graph=graph_name,
            side="within_dagua",
            positions_a=dagua_positions_by_seed,
            positions_b=dagua_positions_by_seed,
            unique_pairs=True,
        )
        rows.extend(between_rows)
        rows.extend(within_target_rows)
        rows.extend(within_dagua_rows)

        between_values = [float(row["rmsd"]) for row in between_rows]
        within_target_values = [float(row["rmsd"]) for row in within_target_rows]
        within_dagua_values = [float(row["rmsd"]) for row in within_dagua_rows]
        deterministic = dagua_seed_count == 1 and len(target_positions_by_seed) == 1
        graph_summary = {
            "n_dagua_seeds": len(dagua_positions_by_seed),
            "n_graphviz_seeds": len(target_positions_by_seed),
            "dagua_vs_graphviz": summarize_values(between_values),
            "within_graphviz": summarize_values(within_target_values),
            "within_dagua": summarize_values(within_dagua_values),
            "deterministic_degenerate": deterministic,
            "tost": {},
        }
        if not deterministic:
            graph_summary["tost"] = tost_equivalence_verdicts(
                between_values=between_values,
                within_values=within_target_values,
            )
        summary["graphs"][graph_name] = graph_summary
    return rows, summary


def write_live_csv(output_dir: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    """Write live comparison rows to ``live_rmsd.csv``.

    Parameters
    ----------
    output_dir : Path
        Destination directory.
    rows : sequence of Mapping[str, Any]
        CSV rows.

    Returns
    -------
    Path
        Written CSV path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "live_rmsd.csv"
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["graph", "n_nodes", "dagua_engine", "target_engine", "rmsd"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def write_multi_seed_outputs(
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> tuple[Path, Path]:
    """Write multi-seed CSV rows and JSON summary.

    Parameters
    ----------
    output_dir : Path
        Destination directory.
    rows : sequence of Mapping[str, Any]
        Pairwise RMSD rows.
    summary : Mapping[str, Any]
        Per-graph summary payload.

    Returns
    -------
    tuple[Path, Path]
        Written CSV and JSON paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "multi_seed_rmsd.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["graph", "side", "seed_a", "seed_b", "rmsd"])
        writer.writeheader()
        writer.writerows(rows)

    summary_path = output_dir / "multi_seed_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return csv_path, summary_path


def print_summary(rows: Sequence[Mapping[str, Any]]) -> None:
    """Print percentile and worst-graph summary for live RMSDs.

    Parameters
    ----------
    rows : sequence of Mapping[str, Any]
        Live comparison rows.
    """
    rmsd_values = [
        float(row["rmsd"])
        for row in rows
        if row.get("rmsd") is not None and math.isfinite(float(row["rmsd"]))
    ]
    if not rmsd_values:
        print("No finite RMSD values produced.")
        return
    worst_row = max(rows, key=lambda row: float(row["rmsd"]))
    print(f"graphs: {len(rmsd_values)}")
    print(f"median: {statistics.median(rmsd_values):.6f}")
    print(f"p25: {percentile(rmsd_values, 0.25):.6f}")
    print(f"p75: {percentile(rmsd_values, 0.75):.6f}")
    print(f"p95: {percentile(rmsd_values, 0.95):.6f}")
    print(f"worst: {worst_row['graph']} {float(worst_row['rmsd']):.6f}")


def print_multi_seed_summary(summary: Mapping[str, Any]) -> None:
    """Print graph-level summary for multi-seed dagua-vs-target RMSDs.

    Parameters
    ----------
    summary : Mapping[str, Any]
        Multi-seed JSON summary payload.
    """
    graph_payloads = summary.get("graphs", {})
    if not isinstance(graph_payloads, Mapping):
        print("No finite RMSD values produced.")
        return

    graph_rows: list[tuple[str, float]] = []
    for graph_name, payload in graph_payloads.items():
        if not isinstance(payload, Mapping):
            continue
        between = payload.get("dagua_vs_graphviz", {})
        if not isinstance(between, Mapping):
            continue
        median = between.get("median")
        if median is None:
            continue
        median_float = float(median)
        if math.isfinite(median_float):
            graph_rows.append((str(graph_name), median_float))

    if not graph_rows:
        print("No finite RMSD values produced.")
        return
    rmsd_values = [value for _, value in graph_rows]
    worst_graph, worst_value = max(graph_rows, key=lambda item: item[1])
    print(f"graphs: {len(graph_rows)}")
    print(f"median: {statistics.median(rmsd_values):.6f}")
    print(f"p25: {percentile(rmsd_values, 0.25):.6f}")
    print(f"p75: {percentile(rmsd_values, 0.75):.6f}")
    print(f"p95: {percentile(rmsd_values, 0.95):.6f}")
    print(f"worst: {worst_graph} {worst_value:.6f}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Parameters
    ----------
    argv : Sequence[str], optional
        Optional argument vector for tests.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dagua_engine")
    parser.add_argument("target_engine")
    parser.add_argument("--graphs", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("eval_output/algo_fidelity/live"))
    parser.add_argument("--input-dir", type=Path, default=Path("eval_output/benchmark_full"))
    parser.add_argument(
        "--graphviz-cache-dir",
        type=Path,
        default=None,
        help="Flat directory of graphviz seed tensors to use for multi-seed target positions.",
    )
    parser.add_argument("--render-panels", action="store_true")
    parser.add_argument(
        "--seeds",
        type=int,
        default=1,
        help="Run multi-seed comparison with N seeds on both sides.",
    )
    return parser.parse_args(argv)


def cache_backed_target_graphs(
    requested_graphs: set[str],
    test_graphs: Mapping[str, TestGraph],
) -> list[str]:
    """Select explicitly requested graphs for cache-backed target comparison.

    Parameters
    ----------
    requested_graphs : set of str
        Graph names supplied on the command line.
    test_graphs : Mapping[str, TestGraph]
        Test graph registry keyed by graph name.

    Returns
    -------
    list of str
        Requested graph names present in the local graph registry.

    Raises
    ------
    ValueError
        If any requested graph is absent from the local graph registry.
    """
    missing = sorted(graph for graph in requested_graphs if graph not in test_graphs)
    if missing:
        raise ValueError(f"Target graph is absent from graph registry: {', '.join(missing)}")
    return sorted(requested_graphs)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the live comparator CLI.

    Parameters
    ----------
    argv : Sequence[str], optional
        Optional argument vector for tests.

    Returns
    -------
    int
        Process exit code.
    """
    args = parse_args(argv)
    indexed = load_results(args.input_dir)
    test_graphs = graph_registry()
    requested_graphs = parse_graph_filter(args.graphs)
    if requested_graphs is not None and (
        args.graphviz_cache_dir is not None or resolve_competitor(args.target_engine) is not None
    ):
        selected_graphs = cache_backed_target_graphs(
            requested_graphs=requested_graphs,
            test_graphs=test_graphs,
        )
    else:
        selected_graphs = target_graphs(
            indexed=indexed,
            dagua_engine=args.dagua_engine,
            target_engine=args.target_engine,
            requested_graphs=requested_graphs,
        )
    if args.seeds < 1:
        raise ValueError("--seeds must be >= 1")
    if args.seeds > 1:
        rows, summary = compare_live_multi_seed(
            dagua_engine=args.dagua_engine,
            target_engine=args.target_engine,
            graphs=selected_graphs,
            test_graphs=test_graphs,
            indexed=indexed,
            input_dir=args.input_dir,
            seed_count=args.seeds,
            graphviz_cache_dir=args.graphviz_cache_dir,
        )
        csv_path, summary_path = write_multi_seed_outputs(args.output_dir, rows, summary)
        print(f"Wrote {len(rows)} rows to {csv_path}")
        print(f"Wrote summary to {summary_path}")
        print_multi_seed_summary(summary)
        return 0

    rows = compare_live(
        dagua_engine=args.dagua_engine,
        target_engine=args.target_engine,
        graphs=selected_graphs,
        test_graphs=test_graphs,
        indexed=indexed,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        render_panels=args.render_panels,
    )
    output_path = write_live_csv(args.output_dir, rows)
    print(f"Wrote {len(rows)} rows to {output_path}")
    print_summary(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
