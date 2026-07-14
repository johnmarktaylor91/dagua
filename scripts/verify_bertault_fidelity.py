"""Verify Bertault pipeline fidelity against the standalone OGDF runner."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dagua.eval.equivalence_metrics import procrustes_rmsd  # noqa: E402
from dagua.layout.ops.pipelines.bertault import layout_bertault_pipeline  # noqa: E402

_RUNNER = _REPO_ROOT / "scripts" / "ogdf_runner"
_NUMERIC_TIER_MAX_ABS = 1.0e-6


@dataclass(frozen=True)
class _GraphCase:
    """Small graph fixture for Bertault verification.

    Parameters
    ----------
    name : str
        Fixture name.
    num_nodes : int
        Number of nodes.
    edges : list[tuple[int, int]]
        Undirected edge list.
    """

    name: str
    num_nodes: int
    edges: list[tuple[int, int]]


def _edge_index(case: _GraphCase) -> torch.Tensor:
    """Return a Dagua edge tensor for a fixture.

    Parameters
    ----------
    case : _GraphCase
        Graph fixture.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, E]``.
    """
    if not case.edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(case.edges, dtype=torch.long).t().contiguous()


def _runner_positions(case: _GraphCase, seed: int) -> torch.Tensor:
    """Run OGDF Bertault for one fixture.

    Parameters
    ----------
    case : _GraphCase
        Graph fixture.
    seed : int
        Seed forwarded to the OGDF runner.

    Returns
    -------
    torch.Tensor
        Reference positions with shape ``[N, 2]``.
    """
    payload = json.dumps(
        {
            "nodes": case.num_nodes,
            "edges": [[source, target] for source, target in case.edges],
            "algorithm": "bertault",
            "seed": seed,
        }
    )
    result = subprocess.run(
        [str(_RUNNER)],
        input=payload,
        capture_output=True,
        text=True,
        timeout=20.0,
        check=True,
    )
    output = json.loads(result.stdout)
    return torch.tensor(output["positions"], dtype=torch.float64)


def _cases() -> list[_GraphCase]:
    """Return the small graph verification set.

    Returns
    -------
    list[_GraphCase]
        Small connected fixtures used for Bertault fidelity checks.
    """
    return [
        _GraphCase("path3", 3, [(0, 1), (1, 2)]),
        _GraphCase("path4", 4, [(0, 1), (1, 2), (2, 3)]),
        _GraphCase("cycle4", 4, [(0, 1), (1, 2), (2, 3), (3, 0)]),
        _GraphCase("k4", 4, [(i, j) for i in range(4) for j in range(i + 1, 4)]),
        _GraphCase(
            "house",
            5,
            [(0, 1), (1, 2), (2, 3), (3, 0), (1, 4), (2, 4)],
        ),
    ]


def _tier(max_abs: float) -> str:
    """Return the fidelity tier for a max-absolute residual.

    Parameters
    ----------
    max_abs : float
        Maximum absolute coordinate residual.

    Returns
    -------
    str
        ``BIT_EXACT``, ``NUMERIC``, or ``RESIDUAL``.
    """
    if max_abs == 0.0:
        return "BIT_EXACT"
    if max_abs <= _NUMERIC_TIER_MAX_ABS:
        return "NUMERIC"
    return "RESIDUAL"


def main() -> int:
    """Run Bertault fidelity verification.

    Returns
    -------
    int
        ``0`` when all fixtures reach the numeric tier, otherwise ``1``.
    """
    failures = 0
    for case in _cases():
        edge_index = _edge_index(case)
        actual = layout_bertault_pipeline(
            edge_index,
            case.num_nodes,
            seed=1,
            fidelity_dtype=torch.float64,
        )
        expected = _runner_positions(case, seed=1)
        residual = actual.cpu() - expected
        max_abs = float(torch.max(torch.abs(residual)).item()) if residual.numel() else 0.0
        rmsd = procrustes_rmsd(actual.cpu(), expected)
        tier = _tier(max_abs)
        if tier == "RESIDUAL":
            failures += 1
        print(f"{case.name}: tier={tier} max_abs={max_abs:.12g} procrustes_rmsd={rmsd:.12g}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
