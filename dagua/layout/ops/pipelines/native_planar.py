"""Native planar sub-pipeline using NetworkX embeddings and stress refinement."""

from __future__ import annotations

import copy
from typing import Any, ClassVar, Optional, Sequence, Tuple

import torch

from dagua.config import LayoutConfig
from dagua.layout.graph_classify import GraphStructure, classify_graph
from dagua.layout.ops.base import LossOp, Op, Pipeline
from dagua.layout.ops.pipelines import dagua_native_legacy
from dagua.layout.ops.pipelines.dagua_flat import layout_dagua_flat_pipeline
from dagua.layout.ops.postprocess import AspectRatioFit, AspectRatioFitConfig
from dagua.layout.ops.project import OverlapProjection, OverlapProjectionConfig
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory

_PLANAR_EMBEDDING_KEY = "native_planar_embedding"
_PLANAR_FACES_KEY = "native_planar_faces"
_PLANAR_FACE_SIGNS_KEY = "native_planar_face_signs"
_PLANAR_INITIAL_POS_KEY = "native_planar_initial_pos"
_FACE_EPS = 1.0e-6


class PlanarityFailure(RuntimeError):
    """Raised when the planar sub-pipeline receives a non-planar graph."""


def _face_signed_area(pos: torch.Tensor, face: torch.Tensor) -> torch.Tensor:
    """Return signed polygon area for one embedding face.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    face : torch.Tensor
        Face vertex indices with shape ``[F]``.

    Returns
    -------
    torch.Tensor
        Scalar signed area.
    """
    vertices = pos[face]
    shifted = torch.roll(vertices, shifts=-1, dims=0)
    return 0.5 * (vertices[:, 0] * shifted[:, 1] - vertices[:, 1] * shifted[:, 0]).sum()


def _embedding_faces(embedding: Any) -> list[list[int]]:
    """Extract face walks from a NetworkX planar embedding.

    Parameters
    ----------
    embedding : Any
        NetworkX ``PlanarEmbedding`` returned by ``nx.check_planarity``.

    Returns
    -------
    list[list[int]]
        Face vertex walks in embedding order.
    """
    marked: set[tuple[int, int]] = set()
    faces: list[list[int]] = []
    for source in embedding:
        for target in embedding.neighbors_cw_order(source):
            half_edge = (int(source), int(target))
            if half_edge in marked:
                continue
            face = [int(node) for node in embedding.traverse_face(source, target, marked)]
            if len(face) >= 3:
                faces.append(face)
    return faces


def _faces_to_tensors(faces: Sequence[Sequence[int]], device: torch.device) -> list[torch.Tensor]:
    """Convert face walks into tensors.

    Parameters
    ----------
    faces : sequence[sequence[int]]
        Face vertex ids.
    device : torch.device
        Target device.

    Returns
    -------
    list[torch.Tensor]
        Face tensors with at least three unique vertices.
    """
    return [
        torch.tensor(face, dtype=torch.long, device=device)
        for face in faces
        if len(set(int(node) for node in face)) >= 3
    ]


def _reference_face_signs(pos: torch.Tensor, faces: Sequence[torch.Tensor]) -> torch.Tensor:
    """Return reference winding signs for faces.

    Parameters
    ----------
    pos : torch.Tensor
        Reference positions with shape ``[N, 2]``.
    faces : sequence[torch.Tensor]
        Face index tensors.

    Returns
    -------
    torch.Tensor
        Sign tensor with shape ``[F]``.
    """
    signs = [
        torch.where(
            _face_signed_area(pos, face) < 0.0,
            pos.new_tensor(-1.0),
            pos.new_tensor(1.0),
        )
        for face in faces
    ]
    if not signs:
        return torch.empty(0, dtype=pos.dtype, device=pos.device)
    return torch.stack(signs)


def _count_crossings_if_small(pos: torch.Tensor, edge_index: torch.Tensor) -> int:
    """Return exact crossings for small graph candidates.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.

    Returns
    -------
    int
        Exact crossing count for small graphs, otherwise ``0``.
    """
    if edge_index.shape[1] > 500:
        return 0
    from dagua.metrics import count_crossings

    return int(count_crossings(pos.detach().cpu(), edge_index.detach().cpu()))


def _embedding_positions(embedding: Any, problem: LayoutProblem, spacing: float) -> torch.Tensor:
    """Return centered coordinates from a NetworkX combinatorial embedding.

    Parameters
    ----------
    embedding : Any
        NetworkX planar embedding.
    problem : LayoutProblem
        Layout inputs.
    spacing : float
        Target median edge length.

    Returns
    -------
    torch.Tensor
        Position tensor with shape ``[N, 2]``.
    """
    import networkx as nx

    raw_pos = nx.combinatorial_embedding_to_pos(embedding)
    device = problem.edge_index.device
    pos = torch.zeros((problem.num_nodes, 2), dtype=torch.float32, device=device)
    for node in range(problem.num_nodes):
        if node not in raw_pos:
            continue
        x_coord, y_coord = raw_pos[node]
        pos[node] = torch.tensor((float(x_coord), float(y_coord)), device=device)
    pos = pos - pos.mean(dim=0, keepdim=True)
    if problem.edge_index.numel() == 0:
        return pos
    source = problem.edge_index[0].to(device=device, dtype=torch.long)
    target = problem.edge_index[1].to(device=device, dtype=torch.long)
    lengths = torch.linalg.norm(pos[source] - pos[target], dim=1)
    positive = lengths[lengths > _FACE_EPS]
    if positive.numel() > 0:
        pos = pos * (max(spacing, 1.0) / float(torch.median(positive).item()))
    return pos


class _ValidatePlanar(Op):
    """Validate that the problem is planar and cache its embedding."""

    name: ClassVar[str] = "validate_planar"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Store planar embedding metadata in ``state.extras``.

        Parameters
        ----------
        problem : LayoutProblem
            Layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Runtime context, unused.

        Returns
        -------
        SolveState
            State with embedding metadata.
        """
        del ctx

        structure = problem.structure
        if structure is None:
            structure = classify_graph(problem.edge_index, problem.num_nodes)
            problem.structure = structure
        embedding = getattr(structure, "planar_embedding", None)
        if not bool(getattr(structure, "is_planar", False)) or embedding is None:
            raise PlanarityFailure("native_planar requires an exactly planar graph.")
        state.extras[_PLANAR_EMBEDDING_KEY] = embedding
        state.extras[_PLANAR_FACES_KEY] = _embedding_faces(embedding)
        return state


class _SchnyderInit(Op):
    """Initialize positions from the combinatorial embedding drawing."""

    name: ClassVar[str] = "schnyder_init"
    category: ClassVar[OpCategory] = OpCategory.INIT
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("pos", "extras")

    def __init__(self, spacing: float) -> None:
        """Store target spacing.

        Parameters
        ----------
        spacing : float
            Target median edge length.
        """
        self.spacing = spacing

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Populate ``state.pos`` from the cached embedding.

        Parameters
        ----------
        problem : LayoutProblem
            Layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Runtime context, unused.

        Returns
        -------
        SolveState
            State with planar warm-start coordinates.
        """
        del ctx

        embedding = state.extras.get(_PLANAR_EMBEDDING_KEY)
        if embedding is None:
            raise PlanarityFailure("native_planar missing planar embedding.")
        pos = _embedding_positions(embedding, problem, self.spacing)
        faces = _faces_to_tensors(state.extras.get(_PLANAR_FACES_KEY, []), pos.device)
        state.pos = pos
        state.extras[_PLANAR_INITIAL_POS_KEY] = pos.detach().clone()
        state.extras[_PLANAR_FACES_KEY] = faces
        state.extras[_PLANAR_FACE_SIGNS_KEY] = _reference_face_signs(pos, faces)
        return state


class _StressRefine(Op):
    """Refine planar warm starts through the existing flat stress pipeline."""

    name: ClassVar[str] = "planar_stress_refine"
    category: ClassVar[OpCategory] = OpCategory.OPTIMIZE
    reads: ClassVar[Tuple[str, ...]] = ("pos",)
    writes: ClassVar[Tuple[str, ...]] = ("pos",)

    def __init__(self, steps: int, n_pivots: int) -> None:
        """Store refinement settings.

        Parameters
        ----------
        steps : int
            Stress-SGD epoch count.
        n_pivots : int
            Pivot count for the existing flat stress pipeline.
        """
        self.steps = steps
        self.n_pivots = n_pivots

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run stress refinement and keep the result only when planar-safe.

        Parameters
        ----------
        problem : LayoutProblem
            Layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Runtime context.

        Returns
        -------
        SolveState
            Refined state, or the initial embedding if refinement crosses.
        """
        del ctx

        if self.steps <= 0 or state.pos is None:
            return state
        refined = layout_dagua_flat_pipeline(
            edge_index=problem.edge_index,
            num_nodes=problem.num_nodes,
            node_sizes=problem.node_sizes,
            steps=self.steps,
            n_pivots=max(self.n_pivots, 1),
            target_aspect=1.0,
        )
        initial = state.extras.get(_PLANAR_INITIAL_POS_KEY)
        if _count_crossings_if_small(refined, problem.edge_index) == 0:
            state.pos = refined
        elif isinstance(initial, torch.Tensor):
            state.pos = initial.to(device=refined.device, dtype=refined.dtype)
        return state


class _FacePreservingConstraint(LossOp):
    """Penalize face winding inversions from the reference embedding."""

    name: ClassVar[str] = "face_preserving_constraint"
    category: ClassVar[OpCategory] = OpCategory.LOSS
    reads: ClassVar[Tuple[str, ...]] = ("pos", "extras")
    writes: ClassVar[Tuple[str, ...]] = ("pos", "prev_loss")

    def __init__(
        self,
        faces: Optional[Sequence[Sequence[int]]] = None,
        reference_pos: Optional[torch.Tensor] = None,
        weight: float = 1.0,
    ) -> None:
        """Store optional standalone face metadata.

        Parameters
        ----------
        faces : sequence[sequence[int]], optional
            Face vertex ids. Defaults to ``state.extras``.
        reference_pos : torch.Tensor, optional
            Reference positions with shape ``[N, 2]``.
        weight : float, default=1.0
            Penalty multiplier.
        """
        self.faces = [tuple(int(node) for node in face) for face in faces] if faces else None
        self.reference_pos = reference_pos.detach().clone() if reference_pos is not None else None
        self.weight = weight

    def _faces_and_signs(
        self,
        state: SolveState,
        device: torch.device,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Resolve face tensors and reference signs.

        Parameters
        ----------
        state : SolveState
            Mutable solve state.
        device : torch.device
            Target device.

        Returns
        -------
        tuple[list[torch.Tensor], torch.Tensor]
            Face tensors and reference signs.
        """
        if self.faces is not None:
            faces = _faces_to_tensors(self.faces, device)
            if self.reference_pos is None:
                return faces, torch.ones(len(faces), dtype=torch.float32, device=device)
            ref = self.reference_pos.to(device=device, dtype=torch.float32)
            return faces, _reference_face_signs(ref, faces)
        faces = [
            face.to(device=device, dtype=torch.long)
            if isinstance(face, torch.Tensor)
            else torch.tensor(face, dtype=torch.long, device=device)
            for face in state.extras.get(_PLANAR_FACES_KEY, [])
        ]
        signs = state.extras.get(_PLANAR_FACE_SIGNS_KEY)
        if isinstance(signs, torch.Tensor):
            return faces, signs.to(device=device, dtype=torch.float32)
        return faces, torch.ones(len(faces), dtype=torch.float32, device=device)

    def evaluate(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> torch.Tensor:
        """Compute the scalar face inversion penalty.

        Parameters
        ----------
        problem : LayoutProblem
            Layout inputs.
        state : SolveState
            State containing positions with shape ``[N, 2]``.
        ctx : RuntimeContext
            Runtime context, unused.

        Returns
        -------
        torch.Tensor
            Zero when faces keep their reference winding, positive otherwise.
        """
        del ctx

        if state.pos is None:
            return torch.zeros((), dtype=torch.float32, device=problem.edge_index.device)
        faces, signs = self._faces_and_signs(state, state.pos.device)
        penalties = [
            torch.relu(-_face_signed_area(state.pos, face) * sign.to(dtype=state.pos.dtype))
            for face, sign in zip(faces, signs)
            if face.numel() >= 3
        ]
        if not penalties:
            return state.pos.new_zeros(())
        return torch.stack(penalties).sum() * float(self.weight)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Restore the embedding warm start if faces invert or cross.

        Parameters
        ----------
        problem : LayoutProblem
            Layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Runtime context.

        Returns
        -------
        SolveState
            State after the face-preservation guardrail.
        """
        loss = self.evaluate(problem, state, ctx)
        state.prev_loss = float(loss.detach().item())
        initial = state.extras.get(_PLANAR_INITIAL_POS_KEY)
        if state.pos is None or not isinstance(initial, torch.Tensor):
            return state
        if (
            state.prev_loss <= _FACE_EPS
            and _count_crossings_if_small(state.pos, problem.edge_index) == 0
        ):
            return state
        state.pos = initial.to(device=state.pos.device, dtype=state.pos.dtype).detach().clone()
        state.prev_loss = float(self.evaluate(problem, state, ctx).detach().item())
        return state


class _OverlapProjection(Op):
    """Apply overlap projection and recheck face preservation."""

    name: ClassVar[str] = "planar_overlap_projection"
    category: ClassVar[OpCategory] = OpCategory.PROJECT

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run existing overlap projection and preserve embedding faces.

        Parameters
        ----------
        problem : LayoutProblem
            Layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Runtime context.

        Returns
        -------
        SolveState
            Projected state when planar-safe.
        """
        state = OverlapProjection(OverlapProjectionConfig(padding=2.0, iterations=10)).apply(
            problem,
            state,
            ctx,
        )
        return _FacePreservingConstraint().apply(problem, state, ctx)


class _AspectRatioFit(Op):
    """Apply Dagua's existing aspect-ratio fitting op."""

    name: ClassVar[str] = "planar_aspect_ratio_fit"
    category: ClassVar[OpCategory] = OpCategory.POSTPROCESS

    def __init__(self, target_aspect: Optional[float]) -> None:
        """Store target aspect.

        Parameters
        ----------
        target_aspect : float, optional
            Desired width/height ratio.
        """
        self.target_aspect = target_aspect

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run the existing aspect-ratio fit.

        Parameters
        ----------
        problem : LayoutProblem
            Layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Runtime context.

        Returns
        -------
        SolveState
            Aspect-fitted state.
        """
        return AspectRatioFit(AspectRatioFitConfig(target_aspect=self.target_aspect)).apply(
            problem,
            state,
            ctx,
        )


def build_planar_pipeline(config: Optional[LayoutConfig] = None) -> Pipeline:
    """Build the native planar pipeline.

    Parameters
    ----------
    config : LayoutConfig, optional
        Layout configuration.

    Returns
    -------
    Pipeline
        Validation, embedding init, stress refinement, face guardrail,
        overlap projection, and aspect fit.
    """
    effective_config = config if config is not None else LayoutConfig()
    steps = int(effective_config.steps) if int(getattr(effective_config, "steps", 0)) > 0 else 30
    spacing = max(
        float(getattr(effective_config, "_dagua_native_node_sep", effective_config.node_sep)),
        1.0,
    )
    return Pipeline(
        [
            _ValidatePlanar(),
            _SchnyderInit(spacing=spacing),
            _StressRefine(
                steps=steps,
                n_pivots=int(getattr(effective_config, "w_stress_n_pivots", 50)),
            ),
            _FacePreservingConstraint(),
            _OverlapProjection(),
            _AspectRatioFit(getattr(effective_config, "_dagua_native_target_aspect", None)),
        ],
        name="native_planar",
    )


def build_native_planar_pipeline(config: LayoutConfig) -> Pipeline:
    """Build the native topology-dispatched planar pipeline.

    Parameters
    ----------
    config : LayoutConfig
        Prepared native configuration.

    Returns
    -------
    Pipeline
        Native planar pipeline.
    """
    return build_planar_pipeline(config)


def layout_native_planar_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    config: Optional[LayoutConfig] = None,
    steps: Optional[int] = None,
    seed: Optional[int] = None,
    edge_weights: Optional[torch.Tensor] = None,
    graph_structure: Optional[GraphStructure] = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Run the native planar pipeline directly.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    node_sizes : torch.Tensor, optional
        Node sizes with shape ``[N, 2]``.
    config : LayoutConfig, optional
        Layout configuration.
    steps : int, optional
        Stress epoch override.
    seed : int, optional
        Random seed.
    edge_weights : torch.Tensor, optional
        Edge weights with shape ``[E]``.
    graph_structure : GraphStructure, optional
        Precomputed graph structure.
    **kwargs : Any
        Ignored compatibility keywords.

    Returns
    -------
    torch.Tensor
        Position tensor with shape ``[N, 2]``.
    """
    del kwargs

    effective_config = copy.copy(config) if config is not None else LayoutConfig()
    if steps is not None:
        effective_config.steps = int(steps)
    resolved_seed = seed if seed is not None else effective_config.seed
    if resolved_seed is not None:
        torch.manual_seed(int(resolved_seed))
    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        structure=graph_structure,
        edge_weights=edge_weights,
        seed=int(resolved_seed if resolved_seed is not None else 42),
    )
    output_device = (
        edge_index.device
        if edge_index.numel() > 0
        else node_sizes.device
        if node_sizes is not None
        else torch.device("cpu")
    )
    ctx = RuntimeContext(plan=ExecutionPlan(device=str(output_device)))
    final_state = build_planar_pipeline(effective_config).apply(problem, SolveState(), ctx)
    if final_state.pos is None:
        raise RuntimeError("native_planar pipeline did not produce final positions.")
    result = final_state.pos
    if node_sizes is None:
        return result

    fallback_config = copy.copy(effective_config)
    fallback_config.try_planar_first = False
    fallback = dagua_native_legacy.layout_dagua_native_pipeline(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        config=fallback_config,
        seed=seed,
        edge_weights=edge_weights,
    )
    result_score = dagua_native_legacy._score_native_result(result, edge_index, node_sizes)
    fallback_score = dagua_native_legacy._score_native_result(fallback, edge_index, node_sizes)
    fallback_crossings = _count_crossings_if_small(fallback, edge_index)
    result_crossings = _count_crossings_if_small(result, edge_index)
    if fallback_crossings <= result_crossings and fallback_score > result_score:
        return fallback
    return result


__all__ = [
    "PlanarityFailure",
    "_FacePreservingConstraint",
    "_SchnyderInit",
    "_ValidatePlanar",
    "build_native_planar_pipeline",
    "build_planar_pipeline",
    "layout_native_planar_pipeline",
]
