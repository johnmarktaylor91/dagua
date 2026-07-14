"""Verify MulMent and NNP-NET fidelity/quality smoke metrics."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dagua.layout.ops.pipelines.mulment import layout_mulment_pipeline  # noqa: E402
from dagua.layout.ops.pipelines.nnpnet import layout_nnpnet_pipeline  # noqa: E402


@dataclass(frozen=True)
class FidelityResult:
    """One algorithm verification result.

    Parameters
    ----------
    algorithm : str
        Algorithm name.
    tier : str
        Fidelity tier label.
    quality : float
        Stress-style quality score.
    residual : float
        Rotation-invariant residual between seeded repeats.
    reference_runtime : str
        Reference build/run status.
    rng_matched : bool
        Whether repeated seeded runs matched exactly.
    """

    algorithm: str
    tier: str
    quality: float
    residual: float
    reference_runtime: str
    rng_matched: bool


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
        Root mean square residual after optimal rotation/reflection.
    """
    ref = reference.to(dtype=torch.float64) - reference.to(dtype=torch.float64).mean(dim=0)
    cand = candidate.to(dtype=torch.float64) - candidate.to(dtype=torch.float64).mean(dim=0)
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
    lengths = torch.linalg.norm(positions[edge_index[0]] - positions[edge_index[1]], dim=1)
    target = torch.clamp(torch.median(lengths), min=1.0e-9)
    return float(torch.mean(((lengths - target) / target).square()).item())


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
    return (
        f"{result.algorithm}: tier={result.tier}; quality={result.quality:.6g}; "
        f"rotation_residual={result.residual:.6g}; rng_matched={result.rng_matched}; "
        f"reference_runtime={result.reference_runtime}"
    )


def verify() -> list[FidelityResult]:
    """Run MulMent and NNP-NET verification cases.

    Returns
    -------
    list[FidelityResult]
        Per-algorithm verification results.
    """
    edge_index = _edge_index_from_edges(
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3), (1, 4)]
    )
    mulment_a = layout_mulment_pipeline(
        edge_index,
        6,
        steps=4,
        seed=13,
        fidelity_dtype=torch.float64,
    )
    mulment_b = layout_mulment_pipeline(
        edge_index,
        6,
        steps=4,
        seed=13,
        fidelity_dtype=torch.float64,
    )
    nnpnet_a = layout_nnpnet_pipeline(
        edge_index,
        6,
        steps=250,
        seed=13,
        embedding_size=4,
        fidelity_dtype=torch.float64,
    )
    nnpnet_b = layout_nnpnet_pipeline(
        edge_index,
        6,
        steps=250,
        seed=13,
        embedding_size=4,
        fidelity_dtype=torch.float64,
    )

    results = [
        FidelityResult(
            algorithm="mulment",
            tier="coarsener-port",
            quality=_stress_quality(edge_index, mulment_a),
            residual=_rotation_invariant_residual(mulment_a, mulment_b),
            reference_runtime=(
                "KaDraw kadraw target built and ran single-thread with --seed; "
                "label-propagation probe first diverges at level 3 (12->7 vs 12->6)"
            ),
            rng_matched=torch.equal(mulment_a, mulment_b),
        ),
        FidelityResult(
            algorithm="nnpnet",
            tier="structural-port",
            quality=_stress_quality(edge_index, nnpnet_a),
            residual=_rotation_invariant_residual(nnpnet_a, nnpnet_b),
            reference_runtime=(
                "NNP-NET built with explicit pthread flags and ran single-thread; "
                "reference CLI exposes no seed for Keras training"
            ),
            rng_matched=torch.equal(nnpnet_a, nnpnet_b),
        ),
    ]
    for result in results:
        if not math.isfinite(result.quality) or not math.isfinite(result.residual):
            raise RuntimeError(f"{result.algorithm} produced a non-finite verification metric.")
        print(_result_line(result))
    return results


if __name__ == "__main__":
    verify()
