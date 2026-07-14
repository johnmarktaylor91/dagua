"""NNP-NET neural-neighborhood-projection layout pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from dagua.layout.ops.base import Op, Pipeline
from dagua.layout.ops.graph_utils import (
    all_pairs_shortest_paths,
    build_undirected_adjacency,
    layout_device,
)
from dagua.layout.ops.pipelines.tsnet import layout_tsnet_pipeline
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op

_DEFAULT_PERPLEXITY = 40.0
_DEFAULT_EMBEDDING_SIZE = 50
_DEFAULT_SUBGRAPH_SIZE = 10_000
_DEFAULT_RIDGE = 1.0e-3
_DEFAULT_OUTPUT_SCALE = 50.0
_EPSILON = 1.0e-9
_REFERENCE_BATCH_SIZE = 64
_REFERENCE_BASE_STEPS_PER_EPOCH = 250


@dataclass(frozen=True)
class NNPNetConfig:
    """Configuration for the NNP-NET projection pipeline.

    Parameters
    ----------
    steps : int
        Optimization iterations for the tsNET teacher stage.
    perplexity : float
        Perplexity used by the tsNET stage.
    embedding_size : int
        Number of pivot-distance embedding dimensions.
    subgraph_size : int
        Maximum sampled teacher subgraph size.
    ridge : float
        Ridge penalty for the deterministic projection model.
    output_scale : float
        Final normalization scale.
    fidelity_dtype : torch.dtype
        Internal dtype used by distance, teacher, and projection stages.
    """

    steps: int = 300
    perplexity: float = _DEFAULT_PERPLEXITY
    embedding_size: int = _DEFAULT_EMBEDDING_SIZE
    subgraph_size: int = _DEFAULT_SUBGRAPH_SIZE
    ridge: float = _DEFAULT_RIDGE
    output_scale: float = _DEFAULT_OUTPUT_SCALE
    fidelity_dtype: torch.dtype = torch.float32


def _validate_config(config: NNPNetConfig) -> None:
    """Validate NNP-NET configuration values.

    Parameters
    ----------
    config : NNPNetConfig
        Configuration to validate.

    Returns
    -------
    None
        The function raises on invalid values.
    """
    if config.steps < 0:
        raise ValueError("steps must be non-negative.")
    if config.perplexity <= 0.0:
        raise ValueError("perplexity must be positive.")
    if config.embedding_size < 1:
        raise ValueError("embedding_size must be positive.")
    if config.subgraph_size < 2:
        raise ValueError("subgraph_size must be at least 2.")
    if config.ridge < 0.0:
        raise ValueError("ridge must be non-negative.")
    if config.output_scale <= 0.0:
        raise ValueError("output_scale must be positive.")
    if config.fidelity_dtype not in (torch.float32, torch.float64):
        raise ValueError("fidelity_dtype must be torch.float32 or torch.float64.")


def _distance_matrix(
    edge_index: torch.Tensor,
    num_nodes: int,
    dtype: torch.dtype,
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute finite all-pairs graph distances.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    dtype : torch.dtype
        Output dtype.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``.

    Returns
    -------
    torch.Tensor
        Dense finite distance matrix with shape ``[N, N]``.
    """
    adjacency = build_undirected_adjacency(
        edge_index=edge_index,
        num_nodes=num_nodes,
        edge_weights=edge_weights,
    )
    distances = all_pairs_shortest_paths(
        adjacency,
        weighted=edge_weights is not None,
    ).to(dtype=dtype)
    finite = torch.isfinite(distances)
    fill = torch.tensor(1.0, dtype=dtype)
    if finite.any():
        fill = torch.clamp(2.0 * distances[finite].max(), min=1.0)
    return torch.where(finite, distances, fill)


def _pivot_distance_embedding(distances: torch.Tensor, embedding_size: int) -> torch.Tensor:
    """Create NNP-NET's farthest-pivot graph-distance embedding.

    Parameters
    ----------
    distances : torch.Tensor
        Dense graph distances with shape ``[N, N]``.
    embedding_size : int
        Number of pivot dimensions.

    Returns
    -------
    torch.Tensor
        Pivot-distance features with shape ``[N, P]``.
    """
    num_nodes = int(distances.shape[0])
    pivots = min(num_nodes, embedding_size)
    features = torch.empty((num_nodes, pivots), dtype=distances.dtype)
    lowest = torch.full((num_nodes,), float("inf"), dtype=distances.dtype)
    pivot = 0
    for dim in range(pivots):
        column = distances[pivot]
        features[:, dim] = column
        if dim + 1 < pivots:
            lowest = torch.minimum(lowest, column)
            pivot = int(torch.argmax(lowest).item())
    biggest = torch.clamp(features.max(), min=_EPSILON)
    return features / biggest


def _reference_pmds_embedding(
    distances: torch.Tensor,
    embedding_size: int,
    seed: int,
) -> torch.Tensor:
    """Create the reference PMDS embedding used as NNP-NET features.

    Parameters
    ----------
    distances : torch.Tensor
        Dense graph distances with shape ``[N, N]``.
    embedding_size : int
        Number of output embedding coordinates.
    seed : int
        Seed for the reference PMDS eigensolver initialization.

    Returns
    -------
    torch.Tensor
        PMDS features with shape ``[N, embedding_size]``.
    """
    num_nodes = int(distances.shape[0])
    num_pivots = min(num_nodes, embedding_size)
    pivot_distances = _pivot_distance_embedding(distances, num_pivots).T.to(dtype=torch.float32)
    pivot_distances = pivot_distances * torch.clamp(
        distances.max().to(dtype=torch.float32),
        min=_EPSILON,
    )

    squared = pivot_distances.square()
    normalization = squared.sum() / float(num_nodes * num_pivots)
    column_normalization = squared.sum(dim=1, keepdim=True) / float(num_nodes)
    row_normalization = squared.mean(dim=0, keepdim=True)
    centered = -0.5 * (squared + normalization - column_normalization - row_normalization)

    gram = centered @ centered.T
    eigenvalues, eigenvectors = torch.linalg.eigh(gram.to(dtype=torch.float64))
    order = torch.argsort(eigenvalues, descending=True)
    dims = min(embedding_size, int(order.numel()))
    selected_values = torch.clamp(eigenvalues[order[:dims]], min=0.0)
    selected_vectors = eigenvectors[:, order[:dims]]
    coords = centered.T.to(dtype=torch.float64) @ selected_vectors
    scales = torch.sqrt(torch.sqrt(selected_values).clamp_min(_EPSILON))
    coords = coords / scales.unsqueeze(0)
    coords = coords.to(dtype=torch.float32)

    if dims < embedding_size:
        padding = torch.zeros((num_nodes, embedding_size - dims), dtype=torch.float32)
        coords = torch.cat([coords, padding], dim=1)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    signs = torch.where(
        torch.rand((embedding_size,), generator=generator) >= 0.5,
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(-1.0, dtype=torch.float32),
    )
    return _minmax_normalize(coords * signs)


def _sample_teacher_nodes(features: torch.Tensor, subgraph_size: int) -> torch.Tensor:
    """Select deterministic farthest-first teacher nodes.

    Parameters
    ----------
    features : torch.Tensor
        Pivot-distance features with shape ``[N, P]``.
    subgraph_size : int
        Maximum selected count.

    Returns
    -------
    torch.Tensor
        Selected node indices with shape ``[S]``.
    """
    num_nodes = int(features.shape[0])
    target = min(num_nodes, subgraph_size)
    selected = [0]
    nearest = torch.linalg.norm(features - features[0], dim=1)
    for _ in range(1, target):
        pivot = int(torch.argmax(nearest).item())
        selected.append(pivot)
        nearest = torch.minimum(nearest, torch.linalg.norm(features - features[pivot], dim=1))
    return torch.tensor(selected, dtype=torch.long)


def _induced_edge_index(edge_index: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
    """Build an edge tensor induced by sampled nodes.

    Parameters
    ----------
    edge_index : torch.Tensor
        Original edge tensor with shape ``[2, E]``.
    nodes : torch.Tensor
        Sampled node IDs with shape ``[S]``.

    Returns
    -------
    torch.Tensor
        Relabeled induced edge tensor with shape ``[2, E_s]``.
    """
    mapping = {int(node): index for index, node in enumerate(nodes.tolist())}
    edges: list[tuple[int, int]] = []
    for source, target in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        if int(source) in mapping and int(target) in mapping:
            edges.append((mapping[int(source)], mapping[int(target)]))
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).transpose(0, 1).contiguous()


def _fit_projection(features: torch.Tensor, labels: torch.Tensor, ridge: float) -> torch.Tensor:
    """Fit a deterministic ridge projection from features to labels.

    Parameters
    ----------
    features : torch.Tensor
        Training features with shape ``[S, P]``.
    labels : torch.Tensor
        Teacher coordinates with shape ``[S, 2]``.
    ridge : float
        Diagonal ridge penalty.

    Returns
    -------
    torch.Tensor
        Projection matrix with shape ``[P + 1, 2]`` including intercept.
    """
    ones = torch.ones((features.shape[0], 1), dtype=features.dtype)
    design = torch.cat([features, ones], dim=1)
    gram = design.T @ design
    penalty = ridge * torch.eye(gram.shape[0], dtype=features.dtype)
    penalty[-1, -1] = 0.0
    return torch.linalg.solve(gram + penalty, design.T @ labels.to(dtype=features.dtype))


def _minmax_normalize(positions: torch.Tensor) -> torch.Tensor:
    """Normalize coordinates with the reference graph min/max rule.

    Parameters
    ----------
    positions : torch.Tensor
        Coordinates with shape ``[N, D]``.

    Returns
    -------
    torch.Tensor
        Coordinates scaled to the reference ``[0, 1]`` box convention.
    """
    minimum = positions.min(dim=0, keepdim=True).values
    maximum = positions.max(dim=0, keepdim=True).values
    max_size = torch.clamp((maximum - minimum).max(), min=_EPSILON)
    return (positions - minimum) / max_size


def _keras_train_plus_infer(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    full_features: torch.Tensor,
    seed: int,
    epochs: int,
) -> Optional[torch.Tensor]:
    """Train and apply the reference Keras MLP.

    Parameters
    ----------
    train_features : torch.Tensor
        Training features with shape ``[S, P]``.
    train_labels : torch.Tensor
        Training labels with shape ``[S, 2]``.
    full_features : torch.Tensor
        Inference features with shape ``[N, P]``.
    seed : int
        TensorFlow/Keras random seed.
    epochs : int
        Number of training epochs.

    Returns
    -------
    torch.Tensor or None
        Predicted coordinates with shape ``[N, 2]`` when TensorFlow is
        available and there is a full reference batch; otherwise ``None``.
    """
    batch_aligned = int(train_features.shape[0]) - (
        int(train_features.shape[0]) % _REFERENCE_BATCH_SIZE
    )
    if batch_aligned <= 0:
        return None

    try:
        import tensorflow as tf
    except Exception:
        return None

    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError:
        pass

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(int(seed))

    feature_array = train_features[:batch_aligned].detach().cpu().to(dtype=torch.float32).numpy()
    label_array = train_labels[:batch_aligned].detach().cpu().to(dtype=torch.float32).numpy()
    full_array = full_features.detach().cpu().to(dtype=torch.float32).numpy()

    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(
                input_shape=[int(full_features.shape[1])],
                batch_size=_REFERENCE_BATCH_SIZE,
            ),
            tf.keras.layers.Dense(256, activation="leaky_relu"),
            tf.keras.layers.Dense(512, activation="leaky_relu"),
            tf.keras.layers.Dense(256, activation="leaky_relu"),
            tf.keras.layers.Dense(int(train_labels.shape[1])),
        ]
    )
    model.compile(optimizer="Adam", loss=tf.keras.losses.MeanSquaredError())
    model.fit(
        feature_array,
        label_array,
        epochs=max(1, int(epochs)),
        batch_size=_REFERENCE_BATCH_SIZE,
        verbose=0,
    )
    predictions = model.predict(full_array, batch_size=4096, verbose=0)
    return torch.from_numpy(np.asarray(predictions, dtype=np.float32))


def _normalize_positions(positions: torch.Tensor, output_scale: float) -> torch.Tensor:
    """Center and scale final coordinates.

    Parameters
    ----------
    positions : torch.Tensor
        Coordinates with shape ``[N, 2]``.
    output_scale : float
        Target maximum absolute coordinate.

    Returns
    -------
    torch.Tensor
        Normalized coordinates with shape ``[N, 2]``.
    """
    centered = positions - positions.mean(dim=0, keepdim=True)
    scale = torch.clamp(centered.abs().max(), min=_EPSILON)
    return centered / scale * output_scale


@register_op
class NNPNetProjectNeighborhood(Op):
    """Run NNP-NET-style pivot embedding, tsNET teacher, and projection."""

    name = "nnpnet_project_neighborhood"
    category = OpCategory.EMBED
    writes = ("pos",)

    def __init__(self, config: NNPNetConfig) -> None:
        """Store projection configuration.

        Parameters
        ----------
        config : NNPNetConfig
            Validated NNP-NET configuration.

        Returns
        -------
        None
            The operation stores the configuration.
        """
        self.config = config

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Apply the NNP-NET projection stages.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Runtime context. Present for pipeline compatibility.

        Returns
        -------
        SolveState
            State with final positions in ``state.pos``.
        """
        del ctx

        distances = _distance_matrix(
            problem.edge_index,
            problem.num_nodes,
            self.config.fidelity_dtype,
            problem.edge_weights,
        )
        features = _reference_pmds_embedding(distances, self.config.embedding_size, problem.seed)
        teacher_nodes = _sample_teacher_nodes(features, self.config.subgraph_size)
        teacher_edges = _induced_edge_index(problem.edge_index.detach().cpu(), teacher_nodes)
        teacher = (
            layout_tsnet_pipeline(
                edge_index=teacher_edges,
                num_nodes=int(teacher_nodes.numel()),
                perplexity=min(self.config.perplexity, max(1.0, float(teacher_nodes.numel() - 1))),
                steps=self.config.steps,
                seed=problem.seed,
                fidelity_mode="exact",
                fidelity_dtype=self.config.fidelity_dtype,
            )
            .detach()
            .cpu()
        )
        teacher = _minmax_normalize(teacher.to(dtype=torch.float32))

        epochs = max(1, int(round(float(self.config.steps) / _REFERENCE_BASE_STEPS_PER_EPOCH)))
        neural = _keras_train_plus_infer(
            features[teacher_nodes],
            teacher,
            features,
            problem.seed,
            epochs,
        )
        if neural is not None:
            state.pos = neural.to(dtype=self.config.fidelity_dtype)
            return state

        weights = _fit_projection(features[teacher_nodes], teacher, self.config.ridge)
        design = torch.cat(
            [features, torch.ones((problem.num_nodes, 1), dtype=features.dtype)],
            dim=1,
        )
        state.pos = _normalize_positions(design @ weights, self.config.output_scale)
        return state


def build_nnpnet_pipeline(config: Optional[NNPNetConfig] = None) -> Pipeline:
    """Build the NNP-NET pipeline.

    Parameters
    ----------
    config : NNPNetConfig, optional
        Pipeline configuration.

    Returns
    -------
    Pipeline
        Pipeline containing the projection operation.
    """
    resolved = config or NNPNetConfig()
    _validate_config(resolved)
    return Pipeline([NNPNetProjectNeighborhood(resolved)], name="nnpnet_pipeline")


def layout_nnpnet_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    steps: int = 300,
    seed: int = 42,
    perplexity: float = _DEFAULT_PERPLEXITY,
    embedding_size: int = _DEFAULT_EMBEDDING_SIZE,
    subgraph_size: int = _DEFAULT_SUBGRAPH_SIZE,
    ridge: float = _DEFAULT_RIDGE,
    output_scale: float = _DEFAULT_OUTPUT_SCALE,
    edge_weights: Optional[torch.Tensor] = None,
    fidelity_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Run the NNP-NET pipeline.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Optional node sizes with shape ``[N, 2]``.
    steps : int, default=300
        tsNET teacher iterations.
    seed : int, default=42
        Seed for tsNET teacher initialization.
    perplexity : float, default=40
        Teacher perplexity.
    embedding_size : int, default=50
        Pivot embedding dimensions.
    subgraph_size : int, default=10000
        Maximum teacher subgraph size.
    ridge : float, default=1e-3
        Projection ridge penalty.
    output_scale : float, default=50
        Final coordinate scale.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``.
    fidelity_dtype : torch.dtype, optional
        Internal dtype.

    Returns
    -------
    torch.Tensor
        Final positions with shape ``[N, 2]``.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    resolved_dtype = torch.float32 if fidelity_dtype is None else fidelity_dtype
    device = layout_device(edge_index=edge_index, node_sizes=node_sizes)
    if num_nodes == 0:
        return torch.empty((0, 2), dtype=resolved_dtype, device=device)
    if num_nodes == 1:
        return torch.zeros((1, 2), dtype=resolved_dtype, device=device)

    config = NNPNetConfig(
        steps=steps,
        perplexity=perplexity,
        embedding_size=embedding_size,
        subgraph_size=subgraph_size,
        ridge=ridge,
        output_scale=output_scale,
        fidelity_dtype=resolved_dtype,
    )
    _validate_config(config)
    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        edge_weights=edge_weights,
        seed=seed,
    )
    final_state = build_nnpnet_pipeline(config).apply(
        problem,
        SolveState(),
        RuntimeContext(plan=ExecutionPlan(device="cpu")),
    )
    if final_state.pos is None:
        raise RuntimeError("NNP-NET pipeline did not produce final positions.")
    return final_state.pos.to(device=device)


__all__ = [
    "NNPNetConfig",
    "build_nnpnet_pipeline",
    "layout_nnpnet_pipeline",
]
