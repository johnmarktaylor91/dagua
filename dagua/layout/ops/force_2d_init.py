"""Sprint 17: 2D init fallback for cyclic / non-DAG graphs.

NativeEngineInit places nodes by layer-y (longest-path layering). For
cyclic graphs that fail acyclicity, layering returns a single layer
(all nodes y=0). The downstream gradient pipeline can't recover
because the dag_ordering loss compresses all nodes to the same y AND
the spring/repulsion losses operate against a 1D-collapsed initial
state.

Symptom: small_world holdout family stuck at composite 22.50 across
every sprint — the layout is a horizontal stripe (bbox 68466 x 34
on n=900).

Fix: after NativeEngineInit, if state.layer_index reports <= 1
layers AND the graph has more than a few nodes, randomize y in
proportion to the existing x extent. This gives the spring + repulsion
losses 2D space to work with from step 1.

Also: dag_ordering_loss is destructive on cyclic graphs (every back
edge is a violation). Sprint 17 also adds a config flag to skip
dag_ordering when the graph is detected as non-DAG; but that's
plumbed in dagua_native_pipeline, not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Tuple

import torch

from dagua.layout.ops.base import Op
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op


@dataclass(frozen=True)
class Force2DInitIfFlatConfig:
    """Configuration for :class:`Force2DInitIfFlat`.

    Parameters
    ----------
    min_nodes : int, default=10
        Don't bother randomizing on very small graphs (the gradient
        pipeline can find 2D positions with few-node bouncing).
    min_layers : int, default=2
        Trigger threshold. When ``state.layer_index.num_layers`` is
        below this, the graph has no useful DAG structure and we
        randomize.
    extent_factor : float, default=1.0
        Y-extent of the randomized initial layout, as a multiple of
        the existing X-extent. 1.0 produces a square-ish layout.
    enabled : bool, default=True
        Master opt-out.
    """

    min_nodes: int = 10
    min_layers: int = 2
    extent_factor: float = 1.0
    enabled: bool = True


@register_op
@dataclass
class Force2DInitIfFlat(Op):
    """Randomize y-coordinates when the layered init produced a 1-layer
    layout (i.e., the graph is cyclic / not a DAG).

    Runs after NativeEngineInit and before the optimizer loop. Reads
    state.layer_index + state.pos; on the trigger condition, replaces
    state.pos[:, 1] with uniform random values spanning the existing
    x-range. Does NOT touch state.layers / state.layer_index — those
    are still valid (1 layer) and downstream layer-aware ops behave
    correctly (they're just no-ops on a single-layer graph).
    """

    name: ClassVar[str] = "force_2d_init_if_flat"
    category: ClassVar[OpCategory] = OpCategory.INIT
    reads: ClassVar[Tuple[str, ...]] = ("pos", "layer_index")
    writes: ClassVar[Tuple[str, ...]] = ("pos",)
    requires: ClassVar[Tuple[str, ...]] = ("pos",)

    config: Force2DInitIfFlatConfig = field(default_factory=Force2DInitIfFlatConfig)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        del ctx
        if not self.config.enabled or state.pos is None:
            return state
        if state.pos.shape[0] < self.config.min_nodes:
            return state
        layer_index = state.layer_index
        num_layers = int(layer_index.num_layers) if layer_index is not None else 1
        if num_layers >= self.config.min_layers:
            return state

        pos = state.pos.detach()
        y = pos[:, 1]
        if int(torch.unique(y).numel()) >= self.config.min_layers:
            return state

        x = pos[:, 0]
        x_extent = float((x.max() - x.min()).item())
        if x_extent <= 0.0:
            x_extent = 1.0
        y_extent = x_extent * self.config.extent_factor

        # Deterministic-via-seed random y-coords, centered.
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(problem.seed))
        new_y = (torch.rand(pos.shape[0], generator=gen) - 0.5).to(
            device=pos.device, dtype=pos.dtype
        ) * y_extent
        new_pos = pos.clone()
        new_pos[:, 1] = new_y
        state.pos = new_pos.requires_grad_(state.pos.requires_grad)
        return state


__all__ = ["Force2DInitIfFlat", "Force2DInitIfFlatConfig"]
