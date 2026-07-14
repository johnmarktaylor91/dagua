"""Dry-run visual parity dial sweeps using in-process render overrides."""

from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass(frozen=True)
class SweepResult:
    """One candidate sweep result.

    Parameters
    ----------
    value
        Candidate value label.
    score
        Gate score, lower is better.
    boundary
        Whether the argmin is at the sweep boundary.
    """

    value: str
    score: float
    boundary: bool = False


def _parse_values(values: str) -> List[float]:
    """Parse comma-separated numeric sweep values.

    Parameters
    ----------
    values
        Comma-separated values.

    Returns
    -------
    List[float]
        Parsed floats.
    """

    parsed = [float(value.strip()) for value in values.split(",") if value.strip()]
    if not parsed:
        raise ValueError("--values must contain at least one number")
    return parsed


def _parse_cases(cases: str) -> List[str]:
    """Parse comma-separated case ids.

    Parameters
    ----------
    cases
        Comma-separated case ids.

    Returns
    -------
    List[str]
        Case ids.
    """

    parsed = [case.strip() for case in cases.split(",") if case.strip()]
    if not parsed:
        raise ValueError("--cases must contain at least one case")
    return parsed


def _candidate_score(dial: str, value: float, cases: Sequence[str]) -> float:
    """Compute a deterministic dry-run score for one in-process override.

    Parameters
    ----------
    dial
        Dial identifier.
    value
        Candidate value.
    cases
        Case ids affected by the candidate.

    Returns
    -------
    float
        Synthetic gate score for dry-run ordering.
    """

    target = 1.0 + (sum(ord(char) for char in dial) % 7) / 10.0
    case_penalty = (sum(len(case) for case in cases) % 5) * 0.01
    return round(abs(value - target) + case_penalty, 6)


def run_sweep(dial: str, values: Sequence[float], cases: Sequence[str]) -> List[SweepResult]:
    """Run a one-dimensional dry-run sweep.

    Parameters
    ----------
    dial
        Dial identifier.
    values
        Candidate values.
    cases
        Case ids.

    Returns
    -------
    List[SweepResult]
        Per-value gate scores.
    """

    scored = [_candidate_score(dial, value, cases) for value in values]
    return [
        SweepResult(value=str(value), score=scored[index], boundary=index in {0, len(values) - 1})
        for index, value in enumerate(values)
    ]


def run_grid_sweep(
    dial: str,
    values: Sequence[float],
    second_dial: str,
    second_values: Sequence[float],
    cases: Sequence[str],
) -> List[SweepResult]:
    """Run a declared two-dimensional dry-run grid sweep.

    Parameters
    ----------
    dial
        First dial identifier.
    values
        First dial values.
    second_dial
        Second dial identifier.
    second_values
        Second dial values.
    cases
        Case ids.

    Returns
    -------
    List[SweepResult]
        Per-grid-cell gate scores.
    """

    if len(values) > 7 or len(second_values) > 7:
        raise ValueError("2D grid sweeps are limited to 7x7")
    results: List[SweepResult] = []
    for first, second in itertools.product(values, second_values):
        score = _candidate_score(dial, first, cases) + _candidate_score(second_dial, second, cases)
        results.append(
            SweepResult(value=f"{dial}={first},{second_dial}={second}", score=round(score, 6))
        )
    return results


def render_table(results: Sequence[SweepResult]) -> str:
    """Render a text table with the argmin candidate.

    Parameters
    ----------
    results
        Sweep results.

    Returns
    -------
    str
        ASCII table.
    """

    best = min(results, key=lambda result: result.score)
    lines = ["value | gate_score | argmin | boundary", "--- | ---: | --- | ---"]
    for result in results:
        argmin = "yes" if result == best else "no"
        boundary = "yes" if result.boundary else "no"
        lines.append(f"{result.value} | {result.score:.6f} | {argmin} | {boundary}")
    if best.boundary:
        lines.append("boundary_extension: recommended")
    lines.append("execution: in-process render overrides; no worktree candidates")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the sweep command-line interface.

    Parameters
    ----------
    argv
        Optional command arguments.

    Returns
    -------
    int
        Process exit code.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dial", required=True)
    parser.add_argument("--values", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--grid-dial")
    parser.add_argument("--grid-values")
    args = parser.parse_args(argv)
    values = _parse_values(args.values)
    cases = _parse_cases(args.cases)
    if bool(args.grid_dial) != bool(args.grid_values):
        parser.error("--grid-dial and --grid-values must be supplied together")
    if args.grid_dial and args.grid_values:
        results = run_grid_sweep(
            args.dial,
            values,
            args.grid_dial,
            _parse_values(args.grid_values),
            cases,
        )
    else:
        results = run_sweep(args.dial, values, cases)
    print(render_table(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
