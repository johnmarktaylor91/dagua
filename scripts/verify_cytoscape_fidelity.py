"""Verify Cytoscape-family layout fidelity against Node reference adapters."""

# ruff: noqa: E402

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dagua.eval.competitors import get_competitor
from dagua.eval.equivalence_metrics import procrustes_rmsd
from dagua.graph import DaguaGraph
from dagua.layout.ops.pipelines.avsdf import layout_avsdf_pipeline
from dagua.layout.ops.pipelines.cise import layout_cise_pipeline
from dagua.layout.ops.pipelines.cose import layout_cose_pipeline
from dagua.layout.ops.pipelines.cose_bilkent import layout_cose_bilkent_pipeline


@dataclass(frozen=True)
class FidelityCase:
    """One Cytoscape fidelity verification case.

    Parameters
    ----------
    algorithm : str
        Dagua algorithm name.
    reference_layout : str
        Cytoscape layout name.
    runner : Callable[..., torch.Tensor]
        Native pipeline entrypoint.
    options : dict[str, object]
        Shared option payload for native and reference calls when names match.
    residual_note : str
        Named residual stage for non-exact layouts.
    """

    algorithm: str
    reference_layout: str
    runner: Callable[..., torch.Tensor]
    options: Dict[str, object]
    residual_note: str


def _build_graph(compound: bool = False) -> DaguaGraph:
    """Build the small verification graph.

    Parameters
    ----------
    compound : bool, default=False
        Whether to add two cluster memberships.

    Returns
    -------
    DaguaGraph
        Test graph with six nodes.
    """
    graph = DaguaGraph()
    for node in range(6):
        graph.add_node(str(node))
    for source, target in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3)]:
        graph.add_edge(str(source), str(target))
    if compound:
        graph.add_cluster("a", ["0", "1", "2"])
        graph.add_cluster("b", ["3", "4", "5"])
    graph.compute_node_sizes()
    return graph


def _tier(algorithm: str, residual: float) -> str:
    """Map a residual into the requested fidelity tier labels.

    Parameters
    ----------
    algorithm : str
        Algorithm name.
    residual : float
        Procrustes residual.

    Returns
    -------
    str
        Fidelity tier.
    """
    if algorithm == "avsdf" and residual < 1.0e-7:
        return "bit-exact"
    if residual < 0.05:
        return "positional"
    return "distributional"


def _native_options(case: FidelityCase) -> Dict[str, object]:
    """Return options accepted by the native pipeline.

    Parameters
    ----------
    case : FidelityCase
        Case to filter.

    Returns
    -------
    dict[str, object]
        Native option payload.
    """
    ignored = {"layout", "numIter", "animate", "fit"}
    options = {key: value for key, value in case.options.items() if key not in ignored}
    if "numIter" in case.options:
        options["steps"] = case.options["numIter"]
    return options


def _run_case(case: FidelityCase) -> str:
    """Run one verification case and format the report line.

    Parameters
    ----------
    case : FidelityCase
        Verification case.

    Returns
    -------
    str
        Human-readable result line.
    """
    compound = case.algorithm in {"cose_bilkent", "cise"}
    graph = _build_graph(compound=compound)
    competitor = get_competitor("cytoscape")
    if competitor is None:
        return f"{case.algorithm}: reference unavailable (cytoscape competitor not registered)"
    reference_options = {"layout": case.reference_layout, **case.options}
    reference = competitor.layout_with_variant(
        graph,
        seed=7,
        timeout=30.0,
        variant_params=reference_options,
    )
    if reference.pos is None:
        return f"{case.algorithm}: reference failed: {reference.error}"
    native_options = _native_options(case)
    native_kwargs: Dict[str, object] = {
        "edge_index": graph.edge_index,
        "num_nodes": graph.num_nodes,
        "node_sizes": graph.node_sizes,
        "seed": 7,
        **native_options,
    }
    if compound:
        native_kwargs["clusters"] = getattr(graph, "clusters", None)
        native_kwargs["cluster_parents"] = getattr(graph, "cluster_parents", None)
    native = case.runner(**native_kwargs)
    residual = procrustes_rmsd(native, reference.pos)
    tier = _tier(case.algorithm, residual)
    return (
        f"{case.algorithm}: {tier} d_R={residual:.6g} "
        f"reference={case.reference_layout} residual={case.residual_note}"
    )


def main() -> None:
    """Run Cytoscape fidelity verification.

    Returns
    -------
    None
        Prints one result line per layout.
    """
    cases: List[FidelityCase] = [
        FidelityCase(
            algorithm="avsdf",
            reference_layout="avsdf",
            runner=layout_avsdf_pipeline,
            options={"nodeSeparation": 60.0},
            residual_note="none expected; deterministic order+circle placement",
        ),
        FidelityCase(
            algorithm="cose",
            reference_layout="cose",
            runner=layout_cose_pipeline,
            options={"numIter": 5, "randomize": False, "animate": False, "fit": False},
            residual_note="core init/first-force closed; residual multi-step cached-bounds drift",
        ),
        FidelityCase(
            algorithm="cose_bilkent",
            reference_layout="cose-bilkent",
            runner=layout_cose_bilkent_pipeline,
            options={"numIter": 5, "randomize": True, "quality": "default"},
            residual_note="compound gravity/tiling stage",
        ),
        FidelityCase(
            algorithm="cise",
            reference_layout="cise",
            runner=layout_cise_pipeline,
            options={"randomize": False},
            residual_note="inter-cluster force relaxation stage",
        ),
    ]
    for case in cases:
        print(_run_case(case))


if __name__ == "__main__":
    main()
