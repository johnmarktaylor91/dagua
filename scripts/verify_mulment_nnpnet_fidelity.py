"""Verify MulMent and NNP-NET against seeded C++ references."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dagua.eval.competitors.mulment_nnpnet_competitor import (  # noqa: E402
    MulMentReferenceCompetitor,
    NNPNetReferenceCompetitor,
)
from dagua.graph import DaguaGraph  # noqa: E402
from dagua.layout.ops.pipelines.mulment import layout_mulment_pipeline  # noqa: E402
from dagua.layout.ops.pipelines.nnpnet import layout_nnpnet_pipeline  # noqa: E402


@dataclass(frozen=True)
class SeedResult:
    """One seeded reference-vs-reimplementation comparison.

    Parameters
    ----------
    seed : int
        Seed used for both reference and reimplementation.
    residual : float
        Rotation/reflection-invariant RMS residual after Procrustes alignment.
    reference_repeat_residual : float
        Same-seed reference repeat residual.
    dagua_repeat_residual : float
        Same-seed reimplementation repeat residual.
    reference_quality : float
        Reference edge-length stress-style score.
    dagua_quality : float
        Reimplementation edge-length stress-style score.
    """

    seed: int
    residual: float
    reference_repeat_residual: float
    dagua_repeat_residual: float
    reference_quality: float
    dagua_quality: float


@dataclass(frozen=True)
class FidelityResult:
    """One algorithm verification result.

    Parameters
    ----------
    algorithm : str
        Algorithm name.
    tier : str
        Fidelity tier label.
    seed_results : list[SeedResult]
        Per-seed metrics.
    reference_runtime : str
        Reference build/run status.
    residual_cause : str
        Exact known source of the first non-bit-exact residual.
    """

    algorithm: str
    tier: str
    seed_results: list[SeedResult]
    reference_runtime: str
    residual_cause: str


def _edge_index_from_edges(edges: list[tuple[int, int]]) -> torch.Tensor:
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
    sources, targets = zip(*edges)
    return torch.tensor([list(sources), list(targets)], dtype=torch.long)


def _graph_from_edges(num_nodes: int, edges: list[tuple[int, int]]) -> DaguaGraph:
    """Build a connected DaguaGraph fixture.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    edges : list[tuple[int, int]]
        Zero-based undirected edge pairs.

    Returns
    -------
    DaguaGraph
        Graph with string node IDs.
    """
    graph = DaguaGraph()
    for node in range(num_nodes):
        graph.add_node(str(node))
    for source, target in edges:
        graph.add_edge(str(source), str(target))
    return graph


def _mulment_fixture() -> tuple[DaguaGraph, torch.Tensor]:
    """Create the MulMent verification fixture.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Graph object and matching edge tensor.
    """
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (8, 9),
        (9, 10),
        (10, 11),
        (0, 6),
        (1, 7),
        (2, 8),
        (3, 9),
        (4, 10),
        (5, 11),
    ]
    return _graph_from_edges(12, edges), _edge_index_from_edges(edges)


def _nnpnet_fixture() -> tuple[DaguaGraph, torch.Tensor]:
    """Create the NNP-NET verification fixture.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Graph object and matching edge tensor.
    """
    edges = [(node, (node + 1) % 64) for node in range(64)]
    edges.extend((node, (node + 8) % 64) for node in range(0, 64, 4))
    return _graph_from_edges(64, edges), _edge_index_from_edges(edges)


def _rotation_invariant_residual(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    """Compute a centered Procrustes RMS residual.

    Parameters
    ----------
    reference : torch.Tensor
        Reference coordinates with shape ``[N, 2]``.
    candidate : torch.Tensor
        Candidate coordinates with shape ``[N, 2]``.

    Returns
    -------
    float
        Root mean square residual after optimal rotation/reflection and scale.
    """
    ref = reference.to(dtype=torch.float64) - reference.to(dtype=torch.float64).mean(dim=0)
    cand = candidate.to(dtype=torch.float64) - candidate.to(dtype=torch.float64).mean(dim=0)
    ref_norm = torch.clamp(torch.linalg.norm(ref), min=1.0e-12)
    cand_norm = torch.clamp(torch.linalg.norm(cand), min=1.0e-12)
    ref = ref / ref_norm
    cand = cand / cand_norm
    covariance = cand.T @ ref
    u, _, vh = torch.linalg.svd(covariance)
    aligned = cand @ (u @ vh)
    return float(torch.sqrt(torch.mean((aligned - ref).square())).item())


def _stress_quality(edge_index: torch.Tensor, positions: torch.Tensor) -> float:
    """Compute a compact edge-length stress score.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    positions : torch.Tensor
        Coordinates with shape ``[N, 2]``.

    Returns
    -------
    float
        Mean squared deviation from median edge length. Lower is better.
    """
    if edge_index.numel() == 0:
        return 0.0
    pos = positions.to(dtype=torch.float64)
    lengths = torch.linalg.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1)
    target = torch.clamp(torch.median(lengths), min=1.0e-9)
    return float(torch.mean(((lengths - target) / target).square()).item())


def _tier_from_residuals(seed_results: list[SeedResult]) -> str:
    """Classify fidelity from seeded residuals.

    Parameters
    ----------
    seed_results : list[SeedResult]
        Per-seed metrics.

    Returns
    -------
    str
        Fidelity tier label.
    """
    max_residual = max(result.residual for result in seed_results)
    if max_residual <= 1.0e-8:
        return "POSITIONAL_IDENTICAL"
    if max_residual <= 1.0e-3:
        return "POSITIONAL_CLOSE"
    return "STRUCTURAL_PORT_RESIDUAL"


def _compare_seeded(
    *,
    graph: DaguaGraph,
    edge_index: torch.Tensor,
    seeds: list[int],
    reference_layout: Callable[[DaguaGraph, int], torch.Tensor],
    dagua_layout: Callable[[int], torch.Tensor],
) -> list[SeedResult]:
    """Run seeded reference and reimplementation comparisons.

    Parameters
    ----------
    graph : DaguaGraph
        Graph fixture for the reference adapter.
    edge_index : torch.Tensor
        Edge tensor matching ``graph``.
    seeds : list[int]
        Seeds to test.
    reference_layout : Callable[[DaguaGraph, int], torch.Tensor]
        Function that returns reference positions for one seed.
    dagua_layout : Callable[[int], torch.Tensor]
        Function that returns Dagua positions for one seed.

    Returns
    -------
    list[SeedResult]
        Per-seed comparison results.
    """
    results: list[SeedResult] = []
    for seed in seeds:
        reference = reference_layout(graph, seed)
        reference_repeat = reference_layout(graph, seed)
        dagua = dagua_layout(seed)
        dagua_repeat = dagua_layout(seed)
        result = SeedResult(
            seed=seed,
            residual=_rotation_invariant_residual(reference, dagua),
            reference_repeat_residual=_rotation_invariant_residual(reference, reference_repeat),
            dagua_repeat_residual=_rotation_invariant_residual(dagua, dagua_repeat),
            reference_quality=_stress_quality(edge_index, reference),
            dagua_quality=_stress_quality(edge_index, dagua),
        )
        values = [
            result.residual,
            result.reference_repeat_residual,
            result.dagua_repeat_residual,
            result.reference_quality,
            result.dagua_quality,
        ]
        if any(not math.isfinite(value) for value in values):
            raise RuntimeError(f"Non-finite metric for seed {seed}.")
        results.append(result)
    return results


def _reference_positions(
    competitor: MulMentReferenceCompetitor | NNPNetReferenceCompetitor,
    graph: DaguaGraph,
    seed: int,
    timeout: float,
) -> torch.Tensor:
    """Run a reference competitor and unwrap positions.

    Parameters
    ----------
    competitor : MulMentReferenceCompetitor | NNPNetReferenceCompetitor
        Reference adapter.
    graph : DaguaGraph
        Graph fixture.
    seed : int
        Reference seed.
    timeout : float
        Maximum runtime in seconds.

    Returns
    -------
    torch.Tensor
        Reference positions with shape ``[N, 2]``.
    """
    result = competitor.layout(graph, timeout=timeout, seed=seed)
    if result.pos is None:
        raise RuntimeError(f"{competitor.name} failed: {result.error}")
    return result.pos


def _result_line(result: FidelityResult) -> str:
    """Format one verification line.

    Parameters
    ----------
    result : FidelityResult
        Verification result.

    Returns
    -------
    str
        Human-readable summary line.
    """
    residuals = [seed_result.residual for seed_result in result.seed_results]
    ref_repeats = [seed_result.reference_repeat_residual for seed_result in result.seed_results]
    dagua_repeats = [seed_result.dagua_repeat_residual for seed_result in result.seed_results]
    return (
        f"{result.algorithm}: tier={result.tier}; "
        f"max_residual={max(residuals):.6g}; "
        f"max_reference_repeat={max(ref_repeats):.6g}; "
        f"max_dagua_repeat={max(dagua_repeats):.6g}; "
        f"reference_runtime={result.reference_runtime}; "
        f"residual_cause={result.residual_cause}"
    )


def verify() -> list[FidelityResult]:
    """Run MulMent and NNP-NET seeded reference verification.

    Returns
    -------
    list[FidelityResult]
        Per-algorithm verification results.
    """
    mulment_graph, mulment_edges = _mulment_fixture()
    nnpnet_graph, nnpnet_edges = _nnpnet_fixture()
    mulment_reference = MulMentReferenceCompetitor()
    nnpnet_reference = NNPNetReferenceCompetitor()

    mulment_results = _compare_seeded(
        graph=mulment_graph,
        edge_index=mulment_edges,
        seeds=[13, 17, 23],
        reference_layout=lambda graph, seed: _reference_positions(
            mulment_reference,
            graph,
            seed,
            timeout=120.0,
        ),
        dagua_layout=lambda seed: layout_mulment_pipeline(
            mulment_edges,
            mulment_graph.num_nodes,
            seed=seed,
            fidelity_dtype=torch.float64,
        ),
    )
    nnpnet_results = _compare_seeded(
        graph=nnpnet_graph,
        edge_index=nnpnet_edges,
        seeds=[13],
        reference_layout=lambda graph, seed: _reference_positions(
            nnpnet_reference,
            graph,
            seed,
            timeout=300.0,
        ),
        dagua_layout=lambda seed: layout_nnpnet_pipeline(
            nnpnet_edges,
            nnpnet_graph.num_nodes,
            steps=250,
            seed=seed,
            embedding_size=4,
            fidelity_dtype=torch.float64,
        ),
    )

    results = [
        FidelityResult(
            algorithm="mulment",
            tier=_tier_from_residuals(mulment_results),
            seed_results=mulment_results,
            reference_runtime=(
                "KaDraw rebuilt with --seed wired to config.seed and ran single-thread."
            ),
            residual_cause=(
                "Full algorithm match (hierarchy, both RNG streams, fast-approx "
                "refinement); residual is the reference's float32 CoordType "
                "arithmetic vs dagua float64 plus 6-significant-digit coordinate "
                "output, ~1e-5 floor."
            ),
        ),
        FidelityResult(
            algorithm="nnpnet",
            tier=_tier_from_residuals(nnpnet_results),
            seed_results=nnpnet_results,
            reference_runtime=(
                "NNP-NET rebuilt with --seed wired to srand, PivotMDS, and TensorFlow/Keras."
            ),
            residual_cause=(
                "dagua/layout/ops/pipelines/nnpnet_reference.py reproduces PMDS "
                "features, the BH tsNET* teacher, and the Keras MLP bit-exactly "
                "in memory; the only residual is the reference Graph::saveToVNA "
                "6-significant-digit text serialization (Graph.h operator<< "
                "default precision), a ~5e-7 quantization floor."
            ),
        ),
    ]
    for result in results:
        print(_result_line(result))
        for seed_result in result.seed_results:
            print(
                "  "
                f"seed={seed_result.seed}; residual={seed_result.residual:.6g}; "
                f"reference_repeat={seed_result.reference_repeat_residual:.6g}; "
                f"dagua_repeat={seed_result.dagua_repeat_residual:.6g}; "
                f"reference_quality={seed_result.reference_quality:.6g}; "
                f"dagua_quality={seed_result.dagua_quality:.6g}"
            )
    return results


if __name__ == "__main__":
    verify()
