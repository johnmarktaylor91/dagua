"""Verify LargeVis and DRGraph native layout quality."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Optional

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dagua.eval.equivalence_metrics import procrustes_rmsd  # noqa: E402
from dagua.layout.ops.pipelines.drgraph import layout_drgraph_pipeline  # noqa: E402
from dagua.layout.ops.pipelines.largevis import layout_largevis_pipeline  # noqa: E402
from dagua.metrics import sampled_stress  # noqa: E402

DEFAULT_REPORT = ROOT / "docs" / "algorithms" / "drgraph_largevis_fidelity.md"


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


def _verification_graphs() -> list[tuple[str, int, torch.Tensor]]:
    """Return the small-first verification graph corpus.

    Returns
    -------
    list[tuple[str, int, torch.Tensor]]
        Named graph cases with node counts and edge tensors.
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
    return [(name, num_nodes, _edge_index(edges)) for name, num_nodes, edges in cases]


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


def _tier(reference_residual: Optional[float]) -> str:
    """Return the fidelity tier under the current reference blocker.

    Parameters
    ----------
    reference_residual : float or None
        Procrustes residual against a built reference, or ``None`` when the
        reference runtime is blocked.

    Returns
    -------
    str
        Fidelity tier label.
    """
    if reference_residual is None:
        return "SOURCE_PORTED_REFERENCE_RUNTIME_BLOCKED"
    if reference_residual <= 1.0e-6:
        return "BIT/SIMILARITY_EXACT"
    if reference_residual <= 5.0e-2:
        return "POSITIONAL"
    return "DISTRIBUTIONAL"


def _run_algorithm(
    name: str,
    layout_fn: Callable[..., torch.Tensor],
    *,
    samples: int,
) -> list[dict[str, float | str]]:
    """Run one algorithm over the verification corpus.

    Parameters
    ----------
    name : str
        Algorithm label.
    layout_fn : Callable[..., torch.Tensor]
        Pipeline function to execute.
    samples : int
        Positive-edge SGD sample count.

    Returns
    -------
    list[dict[str, float | str]]
        Per-graph quality rows.
    """
    rows: list[dict[str, float | str]] = []
    for graph_name, num_nodes, edge_index in _verification_graphs():
        first = layout_fn(edge_index, num_nodes, samples=samples, seed=314159265)
        second = layout_fn(edge_index, num_nodes, samples=samples, seed=314159265)
        residual = procrustes_rmsd(first.cpu().numpy(), second.cpu().numpy())
        stress = sampled_stress(first, edge_index, num_nodes, n_sources=20, n_targets=50)[
            "sampled_stress"
        ]
        rows.append(
            {
                "algorithm": name,
                "graph": graph_name,
                "self_residual": float(residual),
                "reference_residual": "blocked",
                "tier": _tier(None),
                "sampled_stress": float(stress),
                "quality": _quality_label(float(stress)),
            }
        )
    return rows


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


def _write_report(path: Path, rows: list[dict[str, float | str]]) -> None:
    """Write the fidelity markdown report.

    Parameters
    ----------
    path : pathlib.Path
        Destination report path.
    rows : list[dict[str, float | str]]
        Per-graph quality rows.

    Returns
    -------
    None
        Report is written to disk.
    """
    lines = [
        "# DRGraph + LargeVis fidelity",
        "",
        "Implementation: native Python/PyTorch-compatible port of the LargeVis "
        "and DRGraph graph-layout source loops. Shared code covers geodesic KNN "
        "similarity construction, alias-table edge sampling, degree^0.75 "
        "negative sampling, and sampled SGD updates.",
        "",
        "Named residual stage: `reference_runtime_rng`. Both references use GSL "
        "`rand48` seeded with `314159265`, but this environment could not link "
        "GSL, so no reference coordinates were available for runtime residuals.",
        "",
        "## Reference build/run",
        "",
        "- LargeVis clone: `/tmp/LargeVis`; documented compile command failed: "
        "`fatal error: gsl/gsl_rng.h: No such file or directory`.",
        "- DRGraph clone: `/tmp/DRGraph`; documented build first required changing "
        "`Boost_USE_STATIC_LIBS` to `OFF` for the local shared Boost install, then "
        "failed at link: `cannot find -lgsl` and `cannot find -lgslcblas`.",
        "- Single-thread reference runs were therefore blocked before execution.",
        "",
        "## DRGraph license text found in repository",
        "",
        "No top-level `LICENSE` or `COPYING` file exists in the cloned "
        "`ZJUVAG/DRGraph` snapshot. Source files include mixed third-party notices:",
        "",
        "- `src/algorithm/maxheap.h` and `src/algorithm/fastcommunity_mh.cc`: "
        '"This program is free software; you can redistribute it and/or modify '
        "it under the terms of the GNU General Public License as published by "
        "the Free Software Foundation; either version 2 of the License, or "
        '(at your option) any later version."',
        "- `src/algorithm/kmeans.h`: MIT-style permission notice beginning "
        '"Permission is hereby granted, free of charge, to any person obtaining '
        'a copy of this software...".',
        "- `src/ANNOY/annoylib.h`: Apache License, Version 2.0 notice.",
        "",
        "## Results",
        "",
        "| algorithm | graph | tier | self residual | sampled stress | quality |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {algorithm} | {graph} | {tier} | {self_residual} | {stress} | {quality} |".format(
                algorithm=row["algorithm"],
                graph=row["graph"],
                tier=row["tier"],
                self_residual=_format_float(float(row["self_residual"])),
                stress=_format_float(float(row["sampled_stress"])),
                quality=row["quality"],
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Production pipelines do not call adapters, subprocesses, or reference clones.",
            "- LargeVis graph mode uses graph geodesics as the high-dimensional distance "
            "space, matching the project t-SNE/UMAP graph-layout adaptation pattern.",
            "- DRGraph multilevel scaffolding is exposed as an API parameter, but this "
            "native port runs the deterministic single-level optimizer; this is the "
            "main remaining fidelity gap after the blocked reference runtime.",
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
    parser.add_argument("--samples", type=int, default=80)
    args = parser.parse_args()

    rows = []
    rows.extend(_run_algorithm("largevis", layout_largevis_pipeline, samples=args.samples))
    rows.extend(_run_algorithm("drgraph", layout_drgraph_pipeline, samples=args.samples))
    _write_report(args.report, rows)

    for row in rows:
        print(
            "{algorithm} {graph}: tier={tier} quality={quality} "
            "self_residual={self_residual} sampled_stress={stress}".format(
                algorithm=row["algorithm"],
                graph=row["graph"],
                tier=row["tier"],
                quality=row["quality"],
                self_residual=_format_float(float(row["self_residual"])),
                stress=_format_float(float(row["sampled_stress"])),
            )
        )
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
