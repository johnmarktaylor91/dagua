"""Verify LargeVis and DRGraph against built C++ references."""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dagua.eval import distributional_fidelity as df  # noqa: E402
from dagua.eval.equivalence_metrics import procrustes_rmsd  # noqa: E402
from dagua.layout.ops.largevis import (  # noqa: E402
    build_geodesic_knn_graph,
    symmetrize_largevis_similarity,
)
from dagua.layout.ops.pipelines.drgraph import layout_drgraph_pipeline  # noqa: E402
from dagua.layout.ops.pipelines.largevis import layout_largevis_pipeline  # noqa: E402
from dagua.metrics import sampled_stress  # noqa: E402

DEFAULT_REPORT = ROOT / "docs" / "algorithms" / "drgraph_largevis_fidelity.md"
DEFAULT_LARGEVIS_BINARY = Path.home() / "tools" / "dagua-refs" / "largevis" / "Linux" / "LargeVis"
DEFAULT_DRGRAPH_BINARY = Path.home() / "tools" / "dagua-refs" / "drgraph" / "Vis"
REFERENCE_SEED = 314159265
DEFAULT_SEEDS = (
    REFERENCE_SEED,
    1,
    2,
    3,
    5,
    8,
    13,
    21,
    34,
    55,
    89,
    144,
    233,
    377,
    610,
    987,
    1597,
    2584,
    4181,
    6765,
)
LARGEVIS_REFERENCE_SAMPLES = 3
TOST_ALPHA = 0.05
PROCRUSTES_BAND_FLOOR = 1.0e-6


@dataclass(frozen=True)
class VerificationGraph:
    """Small graph fixture for C++ reference verification.

    Parameters
    ----------
    name : str
        Graph case name.
    num_nodes : int
        Number of graph nodes.
    edges : list[tuple[int, int]]
        Undirected edge list.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    """

    name: str
    num_nodes: int
    edges: list[tuple[int, int]]
    edge_index: torch.Tensor


@dataclass(frozen=True)
class ReferenceRun:
    """Reference coordinates and repeat determinism metadata.

    Parameters
    ----------
    positions : numpy.ndarray
        Reference position array with shape ``[N, 2]``.
    repeat_residual : float
        Procrustes RMSD between two reference runs.
    command : str
        Reference command used for the first run.
    seed : int
        Reference RNG seed.
    """

    positions: np.ndarray
    repeat_residual: float
    command: str
    seed: int


@dataclass(frozen=True)
class StressTost:
    """Sampled-stress TOST result.

    Parameters
    ----------
    passed : bool
        Whether the TOST accepted stress equivalence.
    margin : float
        Symmetric stress equivalence margin.
    pvalue : float
        Maximum one-sided t-test p-value.
    dagua_mean : float
        Mean dagua sampled stress.
    reference_mean : float
        Mean reference sampled stress.
    """

    passed: bool
    margin: float
    pvalue: float
    dagua_mean: float
    reference_mean: float


def _edge_index(edges: list[tuple[int, int]]) -> torch.Tensor:
    """Build an edge tensor from edge pairs.

    Parameters
    ----------
    edges : list[tuple[int, int]]
        Edge pairs.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, E]``.
    """
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def _verification_graphs() -> list[VerificationGraph]:
    """Return the small-first verification graph corpus.

    Returns
    -------
    list[VerificationGraph]
        Named graph cases with edge tensors.
    """
    grid = [(row * 3 + col, row * 3 + col + 1) for row in range(3) for col in range(2)] + [
        (row * 3 + col, (row + 1) * 3 + col) for row in range(2) for col in range(3)
    ]
    cases = [
        ("chain_5", 5, [(0, 1), (1, 2), (2, 3), (3, 4)]),
        ("cycle_4", 4, [(0, 1), (1, 2), (2, 3), (3, 0)]),
        ("diamond", 4, [(0, 1), (0, 2), (1, 3), (2, 3)]),
        ("grid_3x3", 9, grid),
    ]
    return [
        VerificationGraph(name, num_nodes, edges, _edge_index(edges))
        for name, num_nodes, edges in cases
    ]


def _runtime_env() -> dict[str, str]:
    """Build the runtime environment for conda-linked references.

    Returns
    -------
    dict[str, str]
        Environment with ``LD_LIBRARY_PATH`` pointing at conda libraries.
    """
    env = os.environ.copy()
    conda_prefix = env.get("CONDA_PREFIX")
    if conda_prefix:
        lib_path = str(Path(conda_prefix) / "lib")
        env["LD_LIBRARY_PATH"] = f"{lib_path}:{env.get('LD_LIBRARY_PATH', '')}"
    return env


def _read_positions(path: Path) -> np.ndarray:
    """Read a LargeVis/DRGraph coordinate file.

    Parameters
    ----------
    path : pathlib.Path
        Reference output path.

    Returns
    -------
    numpy.ndarray
        Position array with shape ``[N, 2]``.
    """
    lines = path.read_text().strip().splitlines()
    if not lines:
        return np.empty((0, 2), dtype=np.float64)
    num_nodes, _ = (int(value) for value in lines[0].split()[:2])
    positions = np.empty((num_nodes, 2), dtype=np.float64)
    for index, line in enumerate(lines[1:]):
        parts = line.split()
        positions[index] = [float(parts[-2]), float(parts[-1])]
    return positions


def _run_command(command: list[str], env: dict[str, str]) -> None:
    """Run a reference command and raise with captured output on failure.

    Parameters
    ----------
    command : list[str]
        Command vector.
    env : dict[str, str]
        Runtime environment.

    Returns
    -------
    None
        The command completed successfully.
    """
    subprocess.run(command, check=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _write_largevis_input(path: Path, graph: VerificationGraph) -> None:
    """Write the source-weighted LargeVis graph-mode input.

    Parameters
    ----------
    path : pathlib.Path
        Destination input file.
    graph : VerificationGraph
        Graph fixture.

    Returns
    -------
    None
        Input file is written.
    """
    similarity = symmetrize_largevis_similarity(
        build_geodesic_knn_graph(graph.edge_index, graph.num_nodes, 150),
        perplexity=50.0,
    )
    with path.open("w") as handle:
        for source, target, weight in zip(
            similarity.source.tolist(),
            similarity.target.tolist(),
            similarity.weight.tolist(),
        ):
            handle.write(f"{source} {target} {float(weight):.9g}\n")


def _write_drgraph_input(path: Path, graph: VerificationGraph) -> None:
    """Write the DRGraph graph-layout input.

    Parameters
    ----------
    path : pathlib.Path
        Destination input file.
    graph : VerificationGraph
        Graph fixture.

    Returns
    -------
    None
        Input file is written.
    """
    with path.open("w") as handle:
        handle.write(f"{graph.num_nodes} {len(graph.edges)}\n")
        for source, target in graph.edges:
            handle.write(f"{source} {target} 1\n")


def _run_reference_twice(
    command_factory: Callable[[Path, Path, int], list[str]],
    input_path: Path,
    output_stem: Path,
    env: dict[str, str],
    seed: int,
) -> ReferenceRun:
    """Run one reference twice and collect deterministic coordinates.

    Parameters
    ----------
    command_factory : Callable[[pathlib.Path, pathlib.Path, int], list[str]]
        Factory receiving input path, output path, and seed.
    input_path : pathlib.Path
        Reference input file.
    output_stem : pathlib.Path
        Prefix for output files.
    env : dict[str, str]
        Runtime environment.
    seed : int
        Reference RNG seed.

    Returns
    -------
    ReferenceRun
        First-run positions and repeat residual.
    """
    first_output = output_stem.with_suffix(f".seed{seed}.out1")
    second_output = output_stem.with_suffix(f".seed{seed}.out2")
    first_command = command_factory(input_path, first_output, seed)
    second_command = command_factory(input_path, second_output, seed)
    _run_command(first_command, env)
    _run_command(second_command, env)
    first = _read_positions(first_output)
    second = _read_positions(second_output)
    return ReferenceRun(first, float(procrustes_rmsd(first, second)), " ".join(first_command), seed)


def _largevis_reference(
    binary: Path,
    graph: VerificationGraph,
    workdir: Path,
    env: dict[str, str],
    seed: int,
) -> ReferenceRun:
    """Run the LargeVis C++ graph-mode reference.

    Parameters
    ----------
    binary : pathlib.Path
        LargeVis executable.
    graph : VerificationGraph
        Graph fixture.
    workdir : pathlib.Path
        Temporary working directory.
    env : dict[str, str]
        Runtime environment.
    seed : int
        Reference RNG seed.

    Returns
    -------
    ReferenceRun
        Reference run metadata.
    """
    input_path = workdir / f"{graph.name}.largevis.in"
    _write_largevis_input(input_path, graph)

    def command_factory(source: Path, output: Path, run_seed: int) -> list[str]:
        """Build the LargeVis command.

        Parameters
        ----------
        source : pathlib.Path
            Input path.
        output : pathlib.Path
            Output path.
        run_seed : int
            Reference RNG seed.

        Returns
        -------
        list[str]
            Command vector.
        """
        return [
            str(binary),
            "-fea",
            "0",
            "-input",
            str(source),
            "-output",
            str(output),
            "-threads",
            "1",
            "-outdim",
            "2",
            "-samples",
            "0",
            "-neg",
            "5",
            "-alpha",
            "1",
            "-gamma",
            "7",
            "-perp",
            "50",
            "--seed",
            str(run_seed),
        ]

    return _run_reference_twice(command_factory, input_path, workdir / graph.name, env, seed)


def _drgraph_reference(
    binary: Path,
    graph: VerificationGraph,
    workdir: Path,
    env: dict[str, str],
    seed: int,
) -> ReferenceRun:
    """Run the DRGraph C++ graph-layout reference.

    Parameters
    ----------
    binary : pathlib.Path
        DRGraph ``Vis`` executable.
    graph : VerificationGraph
        Graph fixture.
    workdir : pathlib.Path
        Temporary working directory.
    env : dict[str, str]
        Runtime environment.
    seed : int
        Reference RNG seed.

    Returns
    -------
    ReferenceRun
        Reference run metadata.
    """
    input_path = workdir / f"{graph.name}.drgraph.in"
    _write_drgraph_input(input_path, graph)

    def command_factory(source: Path, output: Path, run_seed: int) -> list[str]:
        """Build the DRGraph command.

        Parameters
        ----------
        source : pathlib.Path
            Input path.
        output : pathlib.Path
            Output path.
        run_seed : int
            Reference RNG seed.

        Returns
        -------
        list[str]
            Command vector.
        """
        return [
            str(binary),
            "-input",
            str(source),
            "-output",
            str(output),
            "-threads",
            "1",
            "-mode",
            "1",
            "-samples",
            "1",
            "-neg",
            "5",
            "-alpha",
            "1",
            "-gamma",
            "0.01",
            "-multilevel",
            "0",
            "-A",
            "-1",
            "-B",
            "-1",
            "--seed",
            str(run_seed),
        ]

    return _run_reference_twice(command_factory, input_path, workdir / graph.name, env, seed)


def _quality_label(stress_value: float) -> str:
    """Map sampled stress to a compact quality label.

    Parameters
    ----------
    stress_value : float
        Sampled graph stress.

    Returns
    -------
    str
        Quality label for reporting.
    """
    if stress_value <= 0.2:
        return "GOOD"
    if stress_value <= 0.6:
        return "ACCEPTABLE"
    return "WEAK"


def _tier(reference_residual: float) -> str:
    """Classify a reference residual into a fidelity tier.

    Parameters
    ----------
    reference_residual : float
        Procrustes RMSD against the C++ reference.

    Returns
    -------
    str
        Fidelity tier label.
    """
    if reference_residual <= 1.0e-6:
        return "BIT_EXACT"
    if reference_residual <= 5.0e-2:
        return "POSITIONAL"
    return "DISTRIBUTIONAL"


def _format_float(value: float) -> str:
    """Format a float for console and markdown output.

    Parameters
    ----------
    value : float
        Numeric value.

    Returns
    -------
    str
        Compact formatted value.
    """
    return f"{value:.6g}"


def _parse_seeds(value: str) -> list[int]:
    """Parse a comma-separated seed list.

    Parameters
    ----------
    value : str
        Comma-separated integer seeds.

    Returns
    -------
    list[int]
        Parsed seed values.
    """
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if len(seeds) < 2:
        raise argparse.ArgumentTypeError("At least two seeds are required.")
    return seeds


def _sampled_stress_value(positions: np.ndarray, graph: VerificationGraph) -> float:
    """Compute sampled stress for one coordinate array.

    Parameters
    ----------
    positions : numpy.ndarray
        Position array with shape ``[N, 2]``.
    graph : VerificationGraph
        Graph fixture.

    Returns
    -------
    float
        Sampled stress value.
    """
    tensor = torch.as_tensor(positions, dtype=torch.float32)
    return float(
        sampled_stress(
            tensor,
            graph.edge_index,
            graph.num_nodes,
            n_sources=20,
            n_targets=50,
        )["sampled_stress"]
    )


def _stress_tost(dagua_values: list[float], reference_values: list[float]) -> StressTost:
    """Run a paired TOST over sampled-stress distributions.

    Parameters
    ----------
    dagua_values : list[float]
        Per-seed dagua sampled-stress values.
    reference_values : list[float]
        Per-seed reference sampled-stress values.

    Returns
    -------
    StressTost
        TOST summary with reference-spread-tied margin.
    """
    dagua = np.asarray(dagua_values, dtype=np.float64)
    reference = np.asarray(reference_values, dtype=np.float64)
    diff = dagua - reference
    reference_mean = float(np.mean(reference))
    dagua_mean = float(np.mean(dagua))
    reference_spread = float(np.std(reference, ddof=1)) if reference.size > 1 else 0.0
    margin = max(0.25 * abs(reference_mean), reference_spread, 1.0e-6)
    sd = float(np.std(diff, ddof=1)) if diff.size > 1 else 0.0
    mean_diff = float(np.mean(diff))
    if sd <= 1.0e-12:
        pvalue = 0.0 if abs(mean_diff) <= margin else 1.0
    else:
        se = sd / math.sqrt(diff.size)
        p_low = float(stats.t.sf((mean_diff + margin) / se, df=diff.size - 1))
        p_high = float(stats.t.cdf((mean_diff - margin) / se, df=diff.size - 1))
        pvalue = max(p_low, p_high)
    return StressTost(
        passed=bool(pvalue <= TOST_ALPHA),
        margin=float(margin),
        pvalue=float(pvalue),
        dagua_mean=dagua_mean,
        reference_mean=reference_mean,
    )


def _distributional_row(
    algorithm: str,
    graph: VerificationGraph,
    references: list[ReferenceRun],
    dagua_positions: list[np.ndarray],
    cause: str,
) -> dict[str, float | str | bool]:
    """Build one matched-seed distributional result row.

    Parameters
    ----------
    algorithm : str
        Algorithm label.
    graph : VerificationGraph
        Graph fixture.
    references : list[ReferenceRun]
        Reference runs for matched seeds.
    dagua_positions : list[numpy.ndarray]
        Dagua positions for the same seeds as ``references``.
    cause : str
        Residual cause summary.

    Returns
    -------
    dict[str, float | str | bool]
        Distributional result row.
    """
    rng = np.random.default_rng(REFERENCE_SEED)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        mode = df.analyze_mode_a(
            dagua_positions,
            [reference.positions for reference in references],
            rng,
        )
    dagua_spread = float(mode["plain_mean_W_D"])
    reference_spread = float(mode["plain_mean_W_R"])
    between = float(mode["mean_B_offdiag"])
    within_band = max(dagua_spread, reference_spread, PROCRUSTES_BAND_FLOOR)
    procrustes_pass = between <= within_band
    dagua_stress = [_sampled_stress_value(positions, graph) for positions in dagua_positions]
    reference_stress = [
        _sampled_stress_value(reference.positions, graph) for reference in references
    ]
    stress = _stress_tost(dagua_stress, reference_stress)
    equivalent = bool(mode["dist_equivalent"]) and procrustes_pass and stress.passed
    return {
        "algorithm": algorithm,
        "graph": graph.name,
        "seed_count": len(references),
        "distributional_tier": (
            "DISTRIBUTIONAL_EQUIVALENT" if equivalent else "not_distributional_equivalent"
        ),
        "within_dagua": dagua_spread,
        "within_reference": reference_spread,
        "between": between,
        "within_band": within_band,
        "procrustes_pass": procrustes_pass,
        "split_pass": bool(mode["dist_equivalent"]),
        "stress_tost_pass": stress.passed,
        "stress_tost_pvalue": stress.pvalue,
        "stress_tost_margin": stress.margin,
        "dagua_stress_mean": stress.dagua_mean,
        "reference_stress_mean": stress.reference_mean,
        "repeat_max": max(float(reference.repeat_residual) for reference in references),
        "cause": cause,
    }


def _run_rows(args: argparse.Namespace) -> list[dict[str, float | str | bool]]:
    """Run all verification rows.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    list[dict[str, float | str | bool]]
        Per-graph verification rows.
    """
    if not args.largevis_binary.exists():
        raise FileNotFoundError(f"LargeVis reference binary not found: {args.largevis_binary}")
    if not args.drgraph_binary.exists():
        raise FileNotFoundError(f"DRGraph reference binary not found: {args.drgraph_binary}")

    rows: list[dict[str, float | str | bool]] = []
    env = _runtime_env()
    with tempfile.TemporaryDirectory(prefix="drgraph_largevis_verify.") as tmp:
        workdir = Path(tmp)
        for graph in _verification_graphs():
            largevis_refs = [
                _largevis_reference(args.largevis_binary, graph, workdir, env, seed)
                for seed in args.seeds
            ]
            largevis_positions = [
                layout_largevis_pipeline(
                    graph.edge_index,
                    graph.num_nodes,
                    samples=LARGEVIS_REFERENCE_SAMPLES,
                    seed=seed,
                )
                .detach()
                .cpu()
                .numpy()
                for seed in args.seeds
            ]
            rows.append(
                _row(
                    "largevis",
                    graph,
                    largevis_refs[0],
                    torch.as_tensor(largevis_positions[0], dtype=torch.float32),
                    "GSL rand48 now matched; residual remains from source graph/input ordering "
                    "and stochastic negative-sampling trajectory divergence.",
                )
            )
            rows.append(
                _distributional_row(
                    "largevis",
                    graph,
                    largevis_refs,
                    largevis_positions,
                    "Seed patch works for the single-thread optimizer; residual is the "
                    "Hogwild-style sampled negative-trajectory/input-order divergence "
                    "between implementations.",
                )
            )

            drgraph_samples = graph.num_nodes + 3
            drgraph_refs = [
                _drgraph_reference(args.drgraph_binary, graph, workdir, env, seed)
                for seed in args.seeds
            ]
            drgraph_positions = [
                layout_drgraph_pipeline(
                    graph.edge_index,
                    graph.num_nodes,
                    samples=drgraph_samples,
                    seed=seed,
                    multilevel=False,
                )
                .detach()
                .cpu()
                .numpy()
                for seed in args.seeds
            ]
            rows.append(
                _row(
                    "drgraph",
                    graph,
                    drgraph_refs[0],
                    torch.as_tensor(drgraph_positions[0], dtype=torch.float32),
                    "GSL rand48 now matched; residual remains from DRGraph multilevel/input "
                    "ordering and stochastic negative-sampling trajectory divergence.",
                )
            )
            rows.append(
                _distributional_row(
                    "drgraph",
                    graph,
                    drgraph_refs,
                    drgraph_positions,
                    "Seed patch covers the CPU graph-layout path; residual is the Hogwild-style "
                    "sampled negative-trajectory plus DRGraph input/multilevel ordering "
                    "divergence.",
                )
            )
    return rows


def _row(
    algorithm: str,
    graph: VerificationGraph,
    reference: ReferenceRun,
    positions: torch.Tensor,
    cause: str,
) -> dict[str, float | str]:
    """Build one measured result row.

    Parameters
    ----------
    algorithm : str
        Algorithm label.
    graph : VerificationGraph
        Graph fixture.
    reference : ReferenceRun
        Reference coordinates.
    positions : torch.Tensor
        Dagua position tensor with shape ``[N, 2]``.
    cause : str
        Residual cause summary.

    Returns
    -------
    dict[str, float | str]
        Result row.
    """
    actual = positions.detach().cpu().numpy()
    reference_residual = float(procrustes_rmsd(reference.positions, actual))
    stress = sampled_stress(
        positions,
        graph.edge_index,
        graph.num_nodes,
        n_sources=20,
        n_targets=50,
    )["sampled_stress"]
    return {
        "algorithm": algorithm,
        "graph": graph.name,
        "tier": _tier(reference_residual),
        "reference_residual": reference_residual,
        "reference_repeat_residual": reference.repeat_residual,
        "sampled_stress": float(stress),
        "quality": _quality_label(float(stress)),
        "cause": cause,
        "command": reference.command,
    }


def _write_report(path: Path, rows: list[dict[str, float | str | bool]]) -> None:
    """Write the fidelity markdown report.

    Parameters
    ----------
    path : pathlib.Path
        Destination report path.
    rows : list[dict[str, float | str | bool]]
        Per-graph quality rows.

    Returns
    -------
    None
        Report is written to disk.
    """
    lines = [
        "# DRGraph + LargeVis fidelity",
        "",
        "Implementation: native Python/PyTorch-compatible port of the LargeVis and "
        "DRGraph graph-layout source loops. Shared code covers geodesic KNN "
        "similarity construction, alias-table edge sampling, degree^0.75 negative "
        "sampling, GSL `rand48` RNG emulation, and sampled SGD updates.",
        "",
        "Named residual stage: `reference_runtime_rng`. GSL is available from conda "
        "and both C++ references build and run single-threaded with patched `--seed` "
        "support. The default seed list keeps the historical fixed source seed "
        "`314159265` as the first matched seed.",
        "",
        "## Reference build/run",
        "",
        "- LargeVis clone: `~/tools/dagua-refs/largevis`; built with "
        "`g++ LargeVis.cpp main.cpp -o LargeVis -I$CONDA_PREFIX/include "
        "-L$CONDA_PREFIX/lib -lm -pthread -lgsl -lgslcblas -Ofast -march=native "
        "-ffast-math`.",
        "- DRGraph clone: `~/tools/dagua-refs/drgraph`; built with CMake using conda Boost and "
        "`-I$CONDA_PREFIX/include -L$CONDA_PREFIX/lib`, linking `gsl gslcblas`.",
        "- Runtime uses `LD_LIBRARY_PATH=$CONDA_PREFIX/lib`.",
        "- Both CLIs are patched locally to accept `--seed`/`-seed`; omitted seeds "
        "preserve the upstream default `314159265`.",
        "- LargeVis CLI `-samples` is in millions; `-samples 0` executes the "
        "three-sample single-thread reference smoke path. DRGraph `-samples 1` "
        "executes `N + 3` single-thread samples on these graphs.",
        "",
        "## DRGraph license text found in repository",
        "",
        "No top-level `LICENSE` or `COPYING` file exists in the cloned "
        "`ZJUVAG/DRGraph` snapshot. Source files include mixed third-party notices:",
        "",
        "- `src/algorithm/maxheap.h` and `src/algorithm/fastcommunity_mh.cc`: "
        "GPL-2.0-or-later text.",
        "- `src/algorithm/kmeans.h`: MIT-style permission notice.",
        "- `src/ANNOY/annoylib.h`: Apache License, Version 2.0 notice.",
        "",
        "## Results",
        "",
        "| algorithm | graph | seed n | distributional tier | within D | within R | "
        "between | band | proc | split | stress TOST | repeat max | residual cause |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | --- |",
    ]
    for row in rows:
        if "distributional_tier" not in row:
            continue
        lines.append(
            "| {algorithm} | {graph} | {seed_count} | {tier} | {within_dagua} | "
            "{within_reference} | {between} | {within_band} | {procrustes} | {split} | "
            "{stress} (p={stress_p}) | {repeat_max} | {cause} |".format(
                algorithm=row["algorithm"],
                graph=row["graph"],
                seed_count=row["seed_count"],
                tier=row["distributional_tier"],
                within_dagua=_format_float(float(row["within_dagua"])),
                within_reference=_format_float(float(row["within_reference"])),
                between=_format_float(float(row["between"])),
                within_band=_format_float(float(row["within_band"])),
                procrustes="PASS" if bool(row["procrustes_pass"]) else "FAIL",
                split="PASS" if bool(row["split_pass"]) else "FAIL",
                stress="PASS" if bool(row["stress_tost_pass"]) else "FAIL",
                stress_p=_format_float(float(row["stress_tost_pvalue"])),
                repeat_max=_format_float(float(row["repeat_max"])),
                cause=row["cause"],
            )
        )
    lines.extend(
        [
            "",
            "## Fixed-seed residuals",
            "",
            "The first seed in the distributional run is the historical source seed "
            "`314159265`; these rows preserve the previous single-seed diagnostics.",
            "",
            "| algorithm | graph | tier | ref residual | ref repeat | sampled stress | "
            "quality | residual cause |",
            "| --- | --- | --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in rows:
        if "distributional_tier" in row:
            continue
        lines.append(
            "| {algorithm} | {graph} | {tier} | {reference_residual} | "
            "{reference_repeat_residual} | {stress} | {quality} | {cause} |".format(
                algorithm=row["algorithm"],
                graph=row["graph"],
                tier=row["tier"],
                reference_residual=_format_float(float(row["reference_residual"])),
                reference_repeat_residual=_format_float(float(row["reference_repeat_residual"])),
                stress=_format_float(float(row["sampled_stress"])),
                quality=row["quality"],
                cause=row["cause"],
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Production pipelines do not call adapters, subprocesses, or reference clones.",
            "- The patched C++ references are repeat-deterministic in this single-thread setup "
            "when the same seed is passed.",
            "- The Python optimizer now uses a GSL `rand48` emulator and the source "
            "negative-sampling skip rules. Remaining residuals are therefore reported "
            "against the matched-seed distribution, not as positional or bit-exact.",
            "- DRGraph still contains unpatched CUDA "
            "`curandSetPseudoRandomGeneratorSeed(time(NULL))` in the optional GPU visualizer "
            "path, but this verification uses the CPU visualizer with `-threads 1`; the "
            "production dagua pipeline never delegates to that path.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> int:
    """Run DRGraph and LargeVis fidelity verification.

    Returns
    -------
    int
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--largevis-binary", type=Path, default=DEFAULT_LARGEVIS_BINARY)
    parser.add_argument("--drgraph-binary", type=Path, default=DEFAULT_DRGRAPH_BINARY)
    parser.add_argument(
        "--seeds",
        type=_parse_seeds,
        default=list(DEFAULT_SEEDS),
        help="Comma-separated matched seed list for distributional verification.",
    )
    args = parser.parse_args()

    rows = _run_rows(args)
    _write_report(args.report, rows)

    for row in rows:
        if "distributional_tier" in row:
            print(
                "{algorithm} {graph}: distributional_tier={tier} seeds={seed_count} "
                "within_dagua={within_dagua} within_reference={within_reference} "
                "between={between} within_band={within_band} procrustes={procrustes} "
                "split={split} stress_tost={stress} stress_p={stress_p} repeat_max={repeat}".format(
                    algorithm=row["algorithm"],
                    graph=row["graph"],
                    tier=row["distributional_tier"],
                    seed_count=row["seed_count"],
                    within_dagua=_format_float(float(row["within_dagua"])),
                    within_reference=_format_float(float(row["within_reference"])),
                    between=_format_float(float(row["between"])),
                    within_band=_format_float(float(row["within_band"])),
                    procrustes="PASS" if bool(row["procrustes_pass"]) else "FAIL",
                    split="PASS" if bool(row["split_pass"]) else "FAIL",
                    stress="PASS" if bool(row["stress_tost_pass"]) else "FAIL",
                    stress_p=_format_float(float(row["stress_tost_pvalue"])),
                    repeat=_format_float(float(row["repeat_max"])),
                )
            )
            continue
        print(
            "{algorithm} {graph}: tier={tier} ref_residual={reference_residual} "
            "ref_repeat={reference_repeat_residual} quality={quality} "
            "sampled_stress={stress}".format(
                algorithm=row["algorithm"],
                graph=row["graph"],
                tier=row["tier"],
                reference_residual=_format_float(float(row["reference_residual"])),
                reference_repeat_residual=_format_float(float(row["reference_repeat_residual"])),
                quality=row["quality"],
                stress=_format_float(float(row["sampled_stress"])),
            )
        )
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
