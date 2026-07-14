"""Verify OGDF batch pipeline fidelity for balloon, fpp, and schnyder."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dagua.layout.ops.pipelines.balloon import layout_balloon_pipeline  # noqa: E402
from dagua.layout.ops.pipelines.fpp import layout_fpp_pipeline  # noqa: E402
from dagua.layout.ops.pipelines.planar import check_planarity  # noqa: E402
from dagua.layout.ops.pipelines.schnyder import layout_schnyder_pipeline  # noqa: E402

_RUNNER = _REPO_ROOT / "scripts" / "ogdf_runner"


@dataclass(frozen=True)
class _GraphCase:
    """Small graph fixture for batch fidelity verification.

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


@dataclass
class _AlgoCounts:
    """Per-algorithm verification counters.

    Parameters
    ----------
    bit_exact : int
        Number of exact tensor matches.
    residual : int
        Number of planar finite residuals.
    not_applicable : int
        Number of graphs skipped as N/A.
    failures : list[str]
        Human-readable residual/failure descriptions.
    """

    bit_exact: int = 0
    residual: int = 0
    not_applicable: int = 0
    failures: list[str] | None = None

    def __post_init__(self) -> None:
        """Initialize mutable failure storage.

        Returns
        -------
        None
            The ``failures`` field is populated when omitted.
        """
        if self.failures is None:
            self.failures = []


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


def _runner_positions(case: _GraphCase, algorithm: str) -> torch.Tensor:
    """Run the OGDF reference runner for one planar-safe fixture.

    Parameters
    ----------
    case : _GraphCase
        Graph fixture.
    algorithm : str
        OGDF runner algorithm name.

    Returns
    -------
    torch.Tensor
        Reference positions with shape ``[N, 2]``.
    """
    payload = json.dumps(
        {
            "nodes": case.num_nodes,
            "edges": [[source, target] for source, target in case.edges],
            "algorithm": algorithm,
            "seed": 1,
        }
    )
    result = subprocess.run(
        [str(_RUNNER)],
        input=payload,
        capture_output=True,
        text=True,
        timeout=10.0,
        check=True,
    )
    output = json.loads(result.stdout)
    return torch.tensor(output["positions"], dtype=torch.float64)


def _is_planar(case: _GraphCase) -> bool:
    """Return whether the local Python planarity gate accepts a fixture.

    Parameters
    ----------
    case : _GraphCase
        Graph fixture.

    Returns
    -------
    bool
        ``True`` for planar fixtures.
    """
    is_planar, _embedding = check_planarity(_edge_index(case), case.num_nodes)
    return bool(is_planar)


def _verify_algorithm(
    algorithm: str,
    pipeline: Callable[[torch.Tensor, int], torch.Tensor],
    cases: list[_GraphCase],
    planar_only: bool,
) -> _AlgoCounts:
    """Verify one algorithm against the OGDF runner.

    Parameters
    ----------
    algorithm : str
        OGDF runner and pipeline algorithm name.
    pipeline : Callable[[torch.Tensor, int], torch.Tensor]
        Local pipeline callable.
    cases : list[_GraphCase]
        Fixtures to verify.
    planar_only : bool
        Whether non-planar fixtures should be marked N/A before runner use.

    Returns
    -------
    _AlgoCounts
        Verification counters and residual details.
    """
    counts = _AlgoCounts()
    assert counts.failures is not None
    for case in cases:
        edge_index = _edge_index(case)
        if planar_only and not _is_planar(case):
            counts.not_applicable += 1
            continue
        actual = pipeline(edge_index, case.num_nodes).to(dtype=torch.float64)
        expected = _runner_positions(case, algorithm)
        if torch.equal(actual.cpu(), expected):
            counts.bit_exact += 1
            continue
        max_abs = float(torch.max(torch.abs(actual.cpu() - expected)).item())
        counts.residual += 1
        counts.failures.append(f"{case.name}: max_abs={max_abs:.12g}")
    return counts


def _cases() -> list[_GraphCase]:
    """Return the small graph verification set.

    Returns
    -------
    list[_GraphCase]
        Small tree, planar, and non-planar fixtures.
    """
    k5_edges = [(i, j) for i in range(5) for j in range(i + 1, 5)]
    k33_edges = [(source, target) for source in range(3) for target in range(3, 6)]
    return [
        _GraphCase("path3", 3, [(0, 1), (1, 2)]),
        _GraphCase("path4", 4, [(0, 1), (1, 2), (2, 3)]),
        _GraphCase("cycle4", 4, [(0, 1), (1, 2), (2, 3), (3, 0)]),
        _GraphCase("k4", 4, [(i, j) for i in range(4) for j in range(i + 1, 4)]),
        _GraphCase("k5", 5, k5_edges),
        _GraphCase("k3_3", 6, k33_edges),
    ]


def main() -> int:
    """Run OGDF batch fidelity verification.

    Returns
    -------
    int
        Process exit status.
    """
    algorithms: list[tuple[str, Callable[[torch.Tensor, int], torch.Tensor], bool]] = [
        ("balloon", layout_balloon_pipeline, False),
        ("fpp", layout_fpp_pipeline, True),
        ("schnyder", layout_schnyder_pipeline, True),
    ]
    for algorithm, pipeline, planar_only in algorithms:
        counts = _verify_algorithm(algorithm, pipeline, _cases(), planar_only)
        details = ""
        if counts.failures:
            details = " residuals=[" + "; ".join(counts.failures) + "]"
        print(
            f"{algorithm}: bit-exact={counts.bit_exact} "
            f"residual={counts.residual} N/A={counts.not_applicable}{details}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
