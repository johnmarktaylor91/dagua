"""fCoSE force-directed layout pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from dagua.layout.ops.base import Pipeline, Repeat
from dagua.layout.ops.fcose import (
    FCoSEApplyConstraints,
    FCoSECool,
    FCoSECoolConfig,
    FCoSEFinalize,
    FCoSEFinalizeConfig,
    FCoSEInitialPlacement,
    FCoSEInitialPlacementConfig,
    FCoSEPrepareState,
    FCoSEPrepareStateConfig,
    FCoSESpringEmbedderStep,
    FCoSESpringEmbedderStepConfig,
    FCoSEValidateInputs,
    FCoSEValidateInputsConfig,
)
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState


@dataclass(frozen=True)
class FCoSEConfig:
    """Configuration for the fCoSE pipeline.

    Attributes
    ----------
    quality : str
        fCoSE quality mode, one of ``"draft"``, ``"default"``, or ``"proof"``.
    randomize : bool
        Whether to compute a fresh spectral-style initialization.
    steps : int
        Maximum number of CoSE-like refinement iterations.
    node_separation : float
        Spectral initialization separation in Cytoscape coordinate units.
    ideal_edge_length : float
        Desired non-nested edge length.
    node_repulsion : float
        Node repulsion coefficient.
    edge_elasticity : float
        Spring elasticity coefficient.
    gravity : float
        Barycenter gravity coefficient.
    gravity_range : float
        Gravity range multiplier.
    barnes_hut_theta : float
        Barnes-Hut opening threshold.
    max_exact_repulsion_nodes : int
        Exact pairwise repulsion threshold.
    output_extent : float | None
        Optional final normalization half-width.
    """

    quality: str = "default"
    randomize: bool = True
    steps: int = 2500
    node_separation: float = 75.0
    ideal_edge_length: float = 50.0
    node_repulsion: float = 4500.0
    edge_elasticity: float = 0.45
    gravity: float = 0.25
    gravity_range: float = 3.8
    barnes_hut_theta: float = 0.8
    max_exact_repulsion_nodes: int = 512
    output_extent: Optional[float] = None


def build_fcose_pipeline(config: Optional[FCoSEConfig] = None) -> Pipeline:
    """Build an fCoSE-style layout pipeline.

    Reference fidelity
    ------------------
    Targets: Cytoscape fCoSE / Dogrusoz et al. (2009) CoSE lineage and fCoSE
        compound spring-embedder implementation.
    Fidelity mode: no dedicated flag; ``FCoSEConfig.quality`` selects the
        draft/default/proof benchmark variants.
    Verified at: round_33 bounded subset, non-compound subset only; median
        RMSD ranged from 0.001429 to 0.228957 by graph, with max bounded RMSD
        0.428249.
    Known divergences:
        - Compound-node handling is not implemented yet.
        - Initialization matches fCoSE's BFS-distance start in spirit, not as a
          bit-exact Cytoscape port.

    Parameters
    ----------
    config : FCoSEConfig, optional
        Pipeline configuration. Defaults mirror Cytoscape fCoSE's public
        defaults for the non-compound subset.

    Returns
    -------
    Pipeline
        Pipeline implementing spectral-style initialization and CoSE-like
        spring embedding with gravity and fixed-node projection.

    Raises
    ------
    ValueError
        If ``quality`` is unsupported or ``steps`` is negative.
    """
    resolved = config or FCoSEConfig()
    if resolved.quality not in {"draft", "default", "proof"}:
        raise ValueError("quality must be one of 'draft', 'default', or 'proof'.")
    if resolved.steps < 0:
        raise ValueError("steps must be non-negative.")
    refinement_steps = 0 if resolved.quality == "draft" else resolved.steps
    return Pipeline(
        [
            FCoSEValidateInputs(
                FCoSEValidateInputsConfig(
                    steps=refinement_steps,
                    node_separation=resolved.node_separation,
                    ideal_edge_length=resolved.ideal_edge_length,
                    node_repulsion=resolved.node_repulsion,
                    edge_elasticity=resolved.edge_elasticity,
                    gravity=resolved.gravity,
                )
            ),
            FCoSEInitialPlacement(
                FCoSEInitialPlacementConfig(
                    node_separation=resolved.node_separation,
                    randomize=resolved.randomize,
                )
            ),
            FCoSEPrepareState(
                FCoSEPrepareStateConfig(
                    ideal_edge_length=resolved.ideal_edge_length,
                    edge_elasticity=resolved.edge_elasticity,
                    quality=resolved.quality,
                )
            ),
            Repeat(
                n=refinement_steps,
                ops=[
                    FCoSESpringEmbedderStep(
                        FCoSESpringEmbedderStepConfig(
                            node_repulsion=resolved.node_repulsion,
                            gravity=resolved.gravity,
                            gravity_range=resolved.gravity_range,
                            barnes_hut_theta=resolved.barnes_hut_theta,
                            max_exact_nodes=resolved.max_exact_repulsion_nodes,
                        )
                    ),
                    FCoSEApplyConstraints(),
                    FCoSECool(FCoSECoolConfig(quality=resolved.quality)),
                ],
            ),
            FCoSEFinalize(FCoSEFinalizeConfig(extent=resolved.output_extent)),
        ],
        name="fcose_pipeline",
    )


def layout_fcose_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    steps: int = 2500,
    seed: int = 42,
    edge_weights: Optional[torch.Tensor] = None,
    quality: str = "default",
    randomize: bool = True,
    node_separation: float = 75.0,
    nodeRepulsion: Optional[float] = None,
    idealEdgeLength: Optional[float] = None,
    edgeElasticity: Optional[float] = None,
    gravity: float = 0.25,
    gravity_range: float = 3.8,
    barnes_hut_theta: float = 0.8,
    max_exact_repulsion_nodes: int = 512,
    output_extent: Optional[float] = None,
    pos: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the fCoSE pipeline as a drop-in layout function.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes ``N``.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]``. Retained for API
        compatibility; the current non-compound subset does not consume it.
    steps : int, default=2500
        Maximum number of CoSE-like refinement iterations.
    seed : int, default=42
        Random seed for scalable fallback initialization.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``.
    quality : {"draft", "default", "proof"}, default="default"
        fCoSE quality mode. ``"draft"`` returns initialization only.
    randomize : bool, default=True
        Whether to compute a fresh initialization. If ``False`` and ``pos`` is
        provided, refinement starts from ``pos``.
    node_separation : float, default=75.0
        Spectral initialization separation.
    nodeRepulsion : float, optional
        Cytoscape-style node repulsion override.
    idealEdgeLength : float, optional
        Cytoscape-style ideal edge length override.
    edgeElasticity : float, optional
        Cytoscape-style edge elasticity override.
    gravity : float, default=0.25
        Gravity coefficient.
    gravity_range : float, default=3.8
        Gravity range multiplier.
    barnes_hut_theta : float, default=0.8
        Barnes-Hut opening threshold.
    max_exact_repulsion_nodes : int, default=512
        Exact pairwise repulsion threshold.
    output_extent : float, optional
        Optional final normalization half-width.
    pos : torch.Tensor, optional
        Optional warm-start position tensor with shape ``[N, 2]``.

    Returns
    -------
    torch.Tensor
        Final position tensor with shape ``[N, 2]``.

    Raises
    ------
    RuntimeError
        If the pipeline fails to produce final positions.
    """
    del node_sizes
    config = FCoSEConfig(
        quality=quality,
        randomize=randomize,
        steps=steps,
        node_separation=node_separation,
        ideal_edge_length=50.0 if idealEdgeLength is None else float(idealEdgeLength),
        node_repulsion=4500.0 if nodeRepulsion is None else float(nodeRepulsion),
        edge_elasticity=0.45 if edgeElasticity is None else float(edgeElasticity),
        gravity=gravity,
        gravity_range=gravity_range,
        barnes_hut_theta=barnes_hut_theta,
        max_exact_repulsion_nodes=max_exact_repulsion_nodes,
        output_extent=output_extent,
    )
    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        edge_weights=edge_weights,
        seed=seed,
    )
    state = SolveState(pos=pos)
    ctx = RuntimeContext(plan=ExecutionPlan(device=str(edge_index.device)))
    final_state = build_fcose_pipeline(config=config).apply(problem, state, ctx)
    if final_state.pos is None:
        raise RuntimeError("fCoSE pipeline did not produce final positions.")
    return final_state.pos


__all__ = ["FCoSEConfig", "build_fcose_pipeline", "layout_fcose_pipeline"]
