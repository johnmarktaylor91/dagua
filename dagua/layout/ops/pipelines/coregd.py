"""CoRe-GD neural scalable stress layout pipeline."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, Union

import numpy as np
import torch
from scipy.spatial import Delaunay, QhullError
from torch_cluster import knn
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, MessagePassing, radius_graph
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix, to_undirected

from dagua.config import LayoutConfig
from dagua.layout.ops.pipelines.native_stress_ml import (
    NativeStressMLConfig,
    layout_native_stress_ml_pipeline,
)

_DEFAULT_HIDDEN_DIMENSION = 64
_DEFAULT_HIDDEN_STATE_FACTOR = 4.0
_DEFAULT_MLP_DEPTH = 2
_DEFAULT_ITERATIONS = 5
_DEFAULT_KNN_K = 8
_DEFAULT_RANDOM_IN_CHANNELS = 1
_DEFAULT_LAPLACE_EIGVECS = 8
_DEFAULT_NUM_BEACONS = 2
_DEFAULT_ENCODING_SIZE_PER_BEACON = 8
_DEFAULT_COARSEN_MIN_SIZE = 100
_DEFAULT_COARSEN_K = 5
_DEFAULT_RADIUS = 0.05


@dataclass(frozen=True)
class CoreGDConfig:
    """Configuration for the CoRe-GD pipeline.

    Parameters
    ----------
    hidden_dimension : int, default=64
        Hidden GNN channel count.
    hidden_state_factor : float, default=4.0
        Multiplier used for MLP hidden widths.
    mlp_depth : int, default=2
        Number of additional hidden layers in each reference MLP.
    dropout : float, default=0.0
        Dropout probability. Inference calls set the module to eval mode.
    conv : str, default="gru"
        Message-passing cell, one of ``"gru"``, ``"gru-mlp"``, ``"gin"``, or
        ``"gat"``.
    aggregation : str, default="add"
        PyG aggregation mode for GIN/GRU message passing.
    normalization : str, default="LayerNorm"
        Normalization layer name matching the reference configs.
    rewiring : str, default="knn"
        Positional overlay edge source: ``"knn"``, ``"delaunay"``,
        ``"radius"``, or ``"none"``.
    knn_k : int, default=8
        Number of KNN overlay neighbors.
    alt_freq : int, default=2
        Number of graph-edge convolutions before each overlay convolution.
    iterations : int, default=5
        Iterative refinement count.
    random_in_channels : int, default=1
        Count of random scalar node features used by the reference model.
    laplace_eigvec : int, default=8
        Number of non-trivial Laplacian eigenvectors appended to features.
    use_beacons : bool, default=True
        Whether to append sinusoidal BFS beacon encodings.
    num_beacons : int, default=2
        Number of deterministic beacon sources.
    encoding_size_per_beacon : int, default=8
        Sinusoidal encoding width per beacon.
    out_dim : int, default=2
        Coordinate dimension. Dagua callers should leave this at ``2``.
    coarsen : bool, default=True
        Whether to run the Dagua-native hierarchy wrapper before final neural
        refinement.
    coarsen_min_size : int, default=100
        Finest node count that still triggers hierarchy construction.
    coarsen_k : int, default=5
        Heavy-edge style grouping target for the lightweight hierarchy.
    checkpoint_path : str | None, default=None
        Optional reference checkpoint path. When absent, randomly initialized
        weights are used.
    config_path : str | None, default=None
        Optional reference JSON config path used to override fields above.
    seed : int, default=42
        Deterministic feature and random-weight seed.
    """

    hidden_dimension: int = _DEFAULT_HIDDEN_DIMENSION
    hidden_state_factor: float = _DEFAULT_HIDDEN_STATE_FACTOR
    mlp_depth: int = _DEFAULT_MLP_DEPTH
    dropout: float = 0.0
    conv: str = "gru"
    skip_input: bool = False
    skip_previous: bool = False
    aggregation: str = "add"
    normalization: str = "LayerNorm"
    rewiring: str = "knn"
    knn_k: int = _DEFAULT_KNN_K
    alt_freq: int = 2
    iterations: int = _DEFAULT_ITERATIONS
    random_in_channels: int = _DEFAULT_RANDOM_IN_CHANNELS
    laplace_eigvec: int = _DEFAULT_LAPLACE_EIGVECS
    use_beacons: bool = True
    num_beacons: int = _DEFAULT_NUM_BEACONS
    encoding_size_per_beacon: int = _DEFAULT_ENCODING_SIZE_PER_BEACON
    out_dim: int = 2
    coarsen: bool = True
    coarsen_min_size: int = _DEFAULT_COARSEN_MIN_SIZE
    coarsen_k: int = _DEFAULT_COARSEN_K
    checkpoint_path: Optional[str] = None
    config_path: Optional[str] = None
    seed: int = 42


class CoreGDGRUEdgeConv(MessagePassing):
    """Reference CoRe-GD GRU edge convolution."""

    def __init__(self, emb_dim: int, mlp_edge: torch.nn.Module, aggr: str) -> None:
        """Initialize the GRU edge convolution.

        Parameters
        ----------
        emb_dim : int
            Hidden channel count.
        mlp_edge : torch.nn.Module
            Edge-message MLP applied to concatenated source/target states.
        aggr : str
            PyG aggregation mode.
        """
        super().__init__(aggr=aggr)
        self.rnn = torch.nn.GRUCell(emb_dim, emb_dim)
        self.mlp_edge = mlp_edge

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Apply one GRU message-passing step.

        Parameters
        ----------
        x : torch.Tensor
            Node state tensor with shape ``[N, H]``.
        edge_index : torch.Tensor
            Directed edge tensor with shape ``[2, E]``.

        Returns
        -------
        torch.Tensor
            Updated node state tensor with shape ``[N, H]``.
        """
        return self.rnn(self.propagate(edge_index, x=x), x)

    def message(self, x_j: torch.Tensor, x_i: torch.Tensor) -> torch.Tensor:
        """Build an edge message from source and target states.

        Parameters
        ----------
        x_j : torch.Tensor
            Source node states with shape ``[E, H]``.
        x_i : torch.Tensor
            Target node states with shape ``[E, H]``.

        Returns
        -------
        torch.Tensor
            Edge messages with shape ``[E, H]``.
        """
        return self.mlp_edge(torch.cat((x_j, x_i), dim=1))


class CoreGDGINEdgeConv(MessagePassing):
    """Reference CoRe-GD GIN edge convolution."""

    def __init__(self, mlp: torch.nn.Module, mlp_edge: torch.nn.Module, aggr: str) -> None:
        """Initialize the GIN edge convolution.

        Parameters
        ----------
        mlp : torch.nn.Module
            Node update MLP.
        mlp_edge : torch.nn.Module
            Edge-message MLP.
        aggr : str
            PyG aggregation mode.
        """
        super().__init__(aggr=aggr)
        self.mlp = mlp
        self.mlp_edge = mlp_edge
        self.eps = torch.nn.Parameter(torch.Tensor([0.0]))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Apply one GIN message-passing step.

        Parameters
        ----------
        x : torch.Tensor
            Node state tensor with shape ``[N, H]``.
        edge_index : torch.Tensor
            Directed edge tensor with shape ``[2, E]``.

        Returns
        -------
        torch.Tensor
            Updated node state tensor with shape ``[N, H]``.
        """
        return self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x))

    def message(self, x_j: torch.Tensor, x_i: torch.Tensor) -> torch.Tensor:
        """Build a GIN edge message from endpoint states.

        Parameters
        ----------
        x_j : torch.Tensor
            Source node states with shape ``[E, H]``.
        x_i : torch.Tensor
            Target node states with shape ``[E, H]``.

        Returns
        -------
        torch.Tensor
            Edge messages with shape ``[E, H]``.
        """
        return self.mlp_edge(torch.cat((x_j, x_i), dim=1))


class CoReGD(torch.nn.Module):
    """Dagua port of the reference CoRe-GD network."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        hidden_state_factor: float,
        dropout: float,
        mlp_depth: int = _DEFAULT_MLP_DEPTH,
        conv: str = "gin",
        skip_input: bool = False,
        skip_prev: bool = False,
        aggregation: str = "add",
        normalization: type[torch.nn.Module] = torch.nn.LayerNorm,
        overlay: str = "knn",
        overlay_freq: int = 1,
        knn_k: int = 4,
    ) -> None:
        """Initialize a CoRe-GD network with reference-compatible names.

        Parameters
        ----------
        in_channels : int
            Input node feature count.
        hidden_channels : int
            Hidden GNN channel count.
        out_channels : int
            Output coordinate dimensions.
        hidden_state_factor : float
            MLP hidden-width multiplier.
        dropout : float
            Dropout probability.
        mlp_depth : int, default=2
            Number of extra hidden layers in each MLP.
        conv : str, default="gin"
            Message-passing cell name.
        skip_input : bool, default=False
            Whether to use the reference input-skip MLP.
        skip_prev : bool, default=False
            Whether to use the reference previous-state skip MLP.
        aggregation : str, default="add"
            PyG aggregation mode.
        normalization : type[torch.nn.Module], default=torch.nn.LayerNorm
            Normalization layer constructor.
        overlay : str, default="knn"
            Positional rewiring mode.
        overlay_freq : int, default=1
            Number of copied main convolutions.
        knn_k : int, default=4
            KNN overlay neighbor count.
        """
        super().__init__()
        self.dropout = float(dropout)
        self.overlay = overlay
        self.overlay_freq = int(overlay_freq)
        self.knn_k = int(knn_k)
        self.encoder = self.get_mlp(
            in_channels,
            hidden_state_factor * hidden_channels,
            mlp_depth,
            hidden_channels,
            normalization,
            last_relu=True,
        )
        main_conv = self._build_main_conv(
            conv=conv,
            hidden_channels=hidden_channels,
            hidden_state_factor=hidden_state_factor,
            mlp_depth=mlp_depth,
            aggregation=aggregation,
            normalization=normalization,
        )
        self.convs = torch.nn.ModuleList(
            [copy.deepcopy(main_conv) for _ in range(int(overlay_freq))]
        )
        self.conv_alt = copy.deepcopy(main_conv)
        self.decoder = self.get_mlp(
            hidden_channels,
            hidden_state_factor * hidden_channels,
            mlp_depth,
            out_channels,
            normalization,
            last_relu=False,
        )
        self.skip_input = (
            self.get_mlp(
                hidden_channels + in_channels,
                hidden_state_factor * hidden_channels,
                mlp_depth,
                hidden_channels,
                normalization,
            )
            if skip_input
            else None
        )
        self.skip_previous = (
            self.get_mlp(
                2 * hidden_channels,
                hidden_state_factor * 2 * hidden_channels,
                mlp_depth,
                hidden_channels,
                normalization,
            )
            if skip_prev
            else None
        )

    def get_mlp(
        self,
        input_dim: int,
        hidden_dim: float,
        mlp_depth: int,
        output_dim: int,
        normalization: type[torch.nn.Module],
        last_relu: bool = True,
    ) -> torch.nn.Sequential:
        """Construct the reference MLP block.

        Parameters
        ----------
        input_dim : int
            Input feature width.
        hidden_dim : float
            Hidden feature width before integer casting.
        mlp_depth : int
            Number of additional hidden layers.
        output_dim : int
            Output feature width.
        normalization : type[torch.nn.Module]
            Normalization layer constructor.
        last_relu : bool, default=True
            Whether to append final normalization and ReLU.

        Returns
        -------
        torch.nn.Sequential
            MLP with reference-compatible module indices.
        """
        relu_layer = torch.nn.ReLU()
        hidden_width = int(hidden_dim)
        modules: list[torch.nn.Module] = [
            torch.nn.Linear(input_dim, hidden_width),
            normalization(hidden_width),
            relu_layer,
            torch.nn.Dropout(self.dropout),
        ]
        for _ in range(int(mlp_depth)):
            modules.extend(
                [
                    torch.nn.Linear(hidden_width, hidden_width),
                    normalization(hidden_width),
                    relu_layer,
                    torch.nn.Dropout(self.dropout),
                ]
            )
        modules.append(torch.nn.Linear(hidden_width, output_dim))
        if last_relu:
            modules.append(normalization(output_dim))
            modules.append(relu_layer)
        return torch.nn.Sequential(*modules)

    def _build_main_conv(
        self,
        conv: str,
        hidden_channels: int,
        hidden_state_factor: float,
        mlp_depth: int,
        aggregation: str,
        normalization: type[torch.nn.Module],
    ) -> torch.nn.Module:
        """Create the configured reference message-passing cell.

        Parameters
        ----------
        conv : str
            Message-passing cell name.
        hidden_channels : int
            Hidden state width.
        hidden_state_factor : float
            MLP hidden-width multiplier.
        mlp_depth : int
            MLP hidden-layer depth.
        aggregation : str
            PyG aggregation mode.
        normalization : type[torch.nn.Module]
            Normalization layer constructor.

        Returns
        -------
        torch.nn.Module
            Message-passing module.
        """
        if conv == "gin":
            return CoreGDGINEdgeConv(
                self.get_mlp(
                    hidden_channels,
                    hidden_state_factor * hidden_channels,
                    mlp_depth,
                    hidden_channels,
                    normalization,
                    last_relu=True,
                ),
                self.get_mlp(
                    2 * hidden_channels,
                    hidden_state_factor * 2 * hidden_channels,
                    mlp_depth,
                    hidden_channels,
                    normalization,
                    last_relu=True,
                ),
                aggr=aggregation,
            )
        if conv in {"gru", "gru-mlp"}:
            return CoreGDGRUEdgeConv(
                hidden_channels,
                self.get_mlp(
                    2 * hidden_channels,
                    hidden_state_factor * hidden_channels,
                    mlp_depth,
                    hidden_channels,
                    normalization,
                ),
                aggr=aggregation,
            )
        if conv == "gat":
            return GATv2Conv(hidden_channels, hidden_channels)
        raise ValueError(f"Unrecognized CoRe-GD convolution: {conv!r}.")

    def encode(self, batched_data: Data) -> torch.Tensor:
        """Encode raw graph features.

        Parameters
        ----------
        batched_data : torch_geometric.data.Data
            Graph batch with ``x`` feature tensor of shape ``[N, F]``.

        Returns
        -------
        torch.Tensor
            Encoded node states with shape ``[N, H]``.
        """
        return self.encoder(batched_data.x)

    def compute_rewiring(self, pos: torch.Tensor, batched_data: Data) -> Optional[torch.Tensor]:
        """Compute positional overlay edges.

        Parameters
        ----------
        pos : torch.Tensor
            Current decoded positions with shape ``[N, 2]``.
        batched_data : torch_geometric.data.Data
            Graph batch with a ``batch`` vector.

        Returns
        -------
        torch.Tensor or None
            Overlay edge tensor with shape ``[2, E_overlay]`` or ``None`` when
            positional rewiring is disabled.
        """
        if self.overlay == "knn" and self.knn_k > 0:
            effective_k = min(self.knn_k, int(pos.shape[0]))
            new_edges = knn(
                x=pos,
                y=pos,
                k=effective_k,
                batch_x=batched_data.batch,
                batch_y=batched_data.batch,
            )
            return torch.flip(new_edges, dims=[0, 1])
        if self.overlay == "delaunay":
            return _delaunay_edges(pos, batched_data.batch)
        if self.overlay == "radius":
            return radius_graph(x=pos, r=_DEFAULT_RADIUS, batch=batched_data.batch, loop=False)
        return None

    def forward(
        self,
        batched_data: Data,
        iterations: int,
        return_layers: bool = False,
        encode: bool = True,
        transform_to_undirected: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, list[torch.Tensor]]]:
        """Run iterative CoRe-GD coordinate refinement.

        Parameters
        ----------
        batched_data : torch_geometric.data.Data
            Graph batch with ``x``, ``x_orig``, ``edge_index``, and ``batch``.
        iterations : int
            Number of refinement iterations.
        return_layers : bool, default=False
            Whether to return intermediate hidden states.
        encode : bool, default=True
            Whether to encode raw input features before refinement.
        transform_to_undirected : bool, default=False
            Whether to symmetrize graph edges before message passing.

        Returns
        -------
        torch.Tensor or tuple[torch.Tensor, list[torch.Tensor]]
            Final positions with shape ``[N, 2]`` and optionally hidden layers.
        """
        x_orig, x, edge_index = batched_data.x_orig, batched_data.x, batched_data.edge_index
        if transform_to_undirected:
            edge_index = to_undirected(edge_index)

        layers: list[torch.Tensor] = []
        if encode:
            batched_data.x_orig = x
            x_orig = x
            x = self.encoder(x)
        else:
            pos = torch.sigmoid(self.decoder(x)).detach()
            if self.skip_input is not None:
                x = self.skip_input(torch.cat([x, x_orig], dim=1))
            new_edges = self.compute_rewiring(pos, batched_data)
            if new_edges is not None:
                x = self.conv_alt(x, new_edges)

        previous = x
        for _ in range(max(int(iterations) - 1, 0)):
            for conv in self.convs:
                x = conv(x, edge_index)
            pos = torch.sigmoid(self.decoder(x)).detach()
            if return_layers:
                layers.append(x)
            if self.skip_input is not None:
                x = self.skip_input(torch.cat([x, x_orig], dim=1))
            new_edges = self.compute_rewiring(pos, batched_data)
            if new_edges is not None:
                x = self.conv_alt(x, new_edges)
            if self.skip_previous is not None:
                x = self.skip_previous(torch.cat([x, x + previous], dim=1))
            previous = x

        for conv in self.convs:
            x = conv(x, edge_index)
        if return_layers:
            layers.append(x)
        x = torch.sigmoid(self.decoder(x))
        return (x, layers) if return_layers else x


def _normalization_from_name(name: str) -> type[torch.nn.Module]:
    """Resolve a reference normalization name.

    Parameters
    ----------
    name : str
        Normalization name from a CoRe-GD config.

    Returns
    -------
    type[torch.nn.Module]
        Normalization layer constructor.
    """
    if name == "LayerNorm":
        return torch.nn.LayerNorm
    if name == "BatchNorm":
        return torch.nn.BatchNorm1d
    if name == "None":
        return torch.nn.Identity
    raise ValueError(f"Unrecognized CoRe-GD normalization: {name!r}.")


def _delaunay_edges(pos: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Build Delaunay overlay edges per graph in a batch.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    batch : torch.Tensor
        Graph id per node with shape ``[N]``.

    Returns
    -------
    torch.Tensor
        Undirected Delaunay overlay edge tensor with shape ``[2, E]``.
    """
    edge_chunks: list[torch.Tensor] = []
    graph_count = int(batch.max().item()) + 1 if batch.numel() else 0
    for graph_id in range(graph_count):
        node_ids = torch.nonzero(batch == graph_id, as_tuple=False).flatten()
        if int(node_ids.numel()) < 3:
            continue
        points = pos[node_ids].detach().cpu().numpy()
        try:
            tri = Delaunay(points)
        except QhullError:
            continue
        starts: list[int] = []
        ends: list[int] = []
        for simplex in tri.simplices:
            a, b, c = [int(node_ids[int(index)].item()) for index in simplex]
            starts.extend([a, b, c])
            ends.extend([b, c, a])
        edge_chunks.append(
            to_undirected(torch.tensor([starts, ends], dtype=torch.long, device=pos.device))
        )
    if not edge_chunks:
        return torch.empty((2, 0), dtype=torch.long, device=pos.device)
    return torch.cat(edge_chunks, dim=1)


def _resolve_coregd_config(
    config: Optional[Union[CoreGDConfig, LayoutConfig, SimpleNamespace]],
    seed: int,
    **overrides: Any,
) -> CoreGDConfig:
    """Resolve public algorithm parameters into ``CoreGDConfig``.

    Parameters
    ----------
    config : CoreGDConfig or LayoutConfig or types.SimpleNamespace or None
        Explicit CoRe-GD config, public layout config, or JSON-like namespace.
    seed : int
        Fallback random seed.
    **overrides : Any
        Direct keyword overrides from public pipeline calls.

    Returns
    -------
    CoreGDConfig
        Resolved configuration.
    """
    values: dict[str, Any] = {"seed": int(seed)}
    if isinstance(config, CoreGDConfig):
        values.update(config.__dict__)
    elif config is not None:
        params = getattr(config, "algorithm_params", None)
        if isinstance(params, dict):
            values.update(params)
        for field in CoreGDConfig.__dataclass_fields__:
            if hasattr(config, field):
                values[field] = getattr(config, field)
        if hasattr(config, "seed"):
            values["seed"] = getattr(config, "seed")
    values.update({key: value for key, value in overrides.items() if value is not None})

    config_path = values.get("config_path")
    if config_path:
        with Path(str(config_path)).open("r", encoding="utf-8") as handle:
            reference_values = json.load(handle)
        for field in CoreGDConfig.__dataclass_fields__:
            if field in reference_values and field not in overrides:
                values[field] = reference_values[field]
        values["config_path"] = str(config_path)

    accepted = set(CoreGDConfig.__dataclass_fields__)
    return CoreGDConfig(**{key: values[key] for key in accepted if key in values})


def build_coregd_model(config: CoreGDConfig) -> CoReGD:
    """Build a CoRe-GD model from a resolved config.

    Parameters
    ----------
    config : CoreGDConfig
        Resolved CoRe-GD configuration.

    Returns
    -------
    CoReGD
        Reference-compatible model instance.
    """
    in_channels = int(config.random_in_channels) + int(config.laplace_eigvec)
    if config.use_beacons:
        in_channels += int(config.num_beacons) * int(config.encoding_size_per_beacon)
    return CoReGD(
        in_channels=in_channels,
        hidden_channels=int(config.hidden_dimension),
        out_channels=int(config.out_dim),
        hidden_state_factor=float(config.hidden_state_factor),
        dropout=float(config.dropout),
        mlp_depth=int(config.mlp_depth),
        conv=str(config.conv),
        skip_input=bool(config.skip_input),
        skip_prev=bool(config.skip_previous),
        aggregation=str(config.aggregation),
        normalization=_normalization_from_name(str(config.normalization)),
        overlay=str(config.rewiring),
        overlay_freq=int(config.alt_freq),
        knn_k=int(config.knn_k),
    )


def _build_batch(num_nodes: int, device: torch.device) -> torch.Tensor:
    """Build a single-graph PyG batch vector.

    Parameters
    ----------
    num_nodes : int
        Node count.
    device : torch.device
        Target device.

    Returns
    -------
    torch.Tensor
        Batch tensor with shape ``[N]``.
    """
    return torch.zeros(int(num_nodes), dtype=torch.long, device=device)


def _laplacian_eigenvectors(
    edge_index: torch.Tensor,
    num_nodes: int,
    k: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute reference-style Laplacian eigenvector features.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    k : int
        Number of non-trivial eigenvectors.
    device : torch.device
        Target device for the returned tensor.

    Returns
    -------
    torch.Tensor
        Eigenvector feature tensor with shape ``[N, k]``.
    """
    if k <= 0:
        return torch.zeros((num_nodes, 0), dtype=torch.float32, device=device)
    lap_edge_index, edge_weight = get_laplacian(
        edge_index.detach().cpu(),
        normalization="sym",
        num_nodes=num_nodes,
    )
    laplacian = to_scipy_sparse_matrix(lap_edge_index, edge_weight, num_nodes).toarray()
    eig_vals, eig_vecs = np.linalg.eigh(laplacian)
    eig_vecs = np.real(eig_vecs[:, eig_vals.argsort()])
    usable = eig_vecs[:, 1 : k + 1]
    if usable.shape[1] < k:
        usable = np.pad(usable, ((0, 0), (0, k - usable.shape[1])), mode="constant")
    return torch.from_numpy(usable).to(device=device, dtype=torch.float32)


def _bfs_distances(edge_index: torch.Tensor, num_nodes: int, sources: list[int]) -> torch.Tensor:
    """Compute unweighted BFS distances from selected sources.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    sources : list[int]
        Source node indices.

    Returns
    -------
    torch.Tensor
        Distance tensor with shape ``[N, S]``.
    """
    adjacency: list[list[int]] = [[] for _ in range(num_nodes)]
    src = edge_index[0].detach().cpu().tolist()
    dst = edge_index[1].detach().cpu().tolist()
    for start, end in zip(src, dst):
        adjacency[int(start)].append(int(end))
        adjacency[int(end)].append(int(start))

    distances = torch.empty((num_nodes, len(sources)), dtype=torch.float32).fill_(float("inf"))
    for column, source in enumerate(sources):
        distances[int(source), column] = 0.0
        queue: list[int] = [int(source)]
        cursor = 0
        while cursor < len(queue):
            node = queue[cursor]
            cursor += 1
            next_distance = float(distances[node, column].item()) + 1.0
            for neighbor in adjacency[node]:
                if math.isinf(float(distances[neighbor, column].item())):
                    distances[neighbor, column] = next_distance
                    queue.append(neighbor)
    distances[torch.isinf(distances)] = float(num_nodes)
    return distances


def _beacon_features(
    edge_index: torch.Tensor,
    num_nodes: int,
    num_beacons: int,
    encoding_size_per_beacon: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute deterministic sinusoidal beacon features.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    num_beacons : int
        Number of beacon sources.
    encoding_size_per_beacon : int
        Sinusoidal width per beacon.
    seed : int
        Seed used to pick beacon sources.
    device : torch.device
        Target device.

    Returns
    -------
    torch.Tensor
        Beacon feature tensor with shape ``[N, num_beacons * width]``.
    """
    if num_beacons <= 0 or encoding_size_per_beacon <= 0:
        return torch.zeros((num_nodes, 0), dtype=torch.float32, device=device)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    sources = torch.randperm(num_nodes, generator=generator)[: min(num_beacons, num_nodes)].tolist()
    while len(sources) < num_beacons:
        sources.append(0)
    bfs_distances = _bfs_distances(edge_index, num_nodes, [int(source) for source in sources])
    div_term = torch.exp(
        torch.arange(0, encoding_size_per_beacon, 2, dtype=torch.float32)
        * (-math.log(10000.0) / float(encoding_size_per_beacon))
    )
    features: list[torch.Tensor] = []
    for beacon_index in range(num_beacons):
        pe = torch.zeros((num_nodes, encoding_size_per_beacon), dtype=torch.float32)
        pe[:, 0::2] = torch.sin(bfs_distances[:, beacon_index].unsqueeze(1) * div_term)
        pe[:, 1::2] = torch.cos(bfs_distances[:, beacon_index].unsqueeze(1) * div_term)
        features.append(pe)
    return torch.cat(features, dim=1).to(device=device)


def prepare_coregd_data(
    edge_index: torch.Tensor,
    num_nodes: int,
    config: CoreGDConfig,
    device: torch.device,
) -> Data:
    """Prepare a single-graph PyG ``Data`` object for CoRe-GD.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    config : CoreGDConfig
        Resolved CoRe-GD configuration.
    device : torch.device
        Target device.

    Returns
    -------
    torch_geometric.data.Data
        Data object with reference-compatible ``x`` and ``x_orig`` fields.
    """
    generator = torch.Generator(device=device).manual_seed(int(config.seed))
    random_features = torch.rand(
        (num_nodes, int(config.random_in_channels)),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    feature_parts = [random_features]
    if config.use_beacons:
        feature_parts.append(
            _beacon_features(
                edge_index=edge_index,
                num_nodes=num_nodes,
                num_beacons=int(config.num_beacons),
                encoding_size_per_beacon=int(config.encoding_size_per_beacon),
                seed=int(config.seed),
                device=device,
            )
        )
    feature_parts.append(
        _laplacian_eigenvectors(
            edge_index=edge_index,
            num_nodes=num_nodes,
            k=int(config.laplace_eigvec),
            device=device,
        )
    )
    x = torch.cat(feature_parts, dim=1)
    data = Data(
        x=x,
        edge_index=edge_index.to(device=device, dtype=torch.long),
        num_nodes=num_nodes,
        batch=_build_batch(num_nodes, device),
    )
    data.x_orig = torch.clone(x)
    return data


def _load_checkpoint(model: CoReGD, checkpoint_path: Optional[str], device: torch.device) -> None:
    """Load optional CoRe-GD weights into a model.

    Parameters
    ----------
    model : CoReGD
        Model to update.
    checkpoint_path : str or None
        Optional checkpoint path.
    device : torch.device
        Map location for loaded tensors.

    Returns
    -------
    None
        The model is modified in place.
    """
    if checkpoint_path is None:
        return
    state = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(state)


def _coarsen_edge_index(
    edge_index: torch.Tensor,
    num_nodes: int,
    target_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Build a lightweight deterministic hierarchy level.

    Parameters
    ----------
    edge_index : torch.Tensor
        Fine edge tensor with shape ``[2, E]``.
    num_nodes : int
        Fine node count.
    target_nodes : int
        Coarse node count target.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, int]
        Coarse edge tensor, fine-to-coarse assignment with shape ``[N]``, and
        actual coarse node count.
    """
    coarse_nodes = max(1, min(int(target_nodes), int(num_nodes)))
    assignment = torch.div(
        torch.arange(num_nodes, dtype=torch.long, device=edge_index.device) * coarse_nodes,
        num_nodes,
        rounding_mode="floor",
    )
    coarse_edges = assignment[edge_index]
    mask = coarse_edges[0] != coarse_edges[1]
    coarse_edges = coarse_edges[:, mask]
    if coarse_edges.numel() == 0:
        return (
            torch.empty((2, 0), dtype=torch.long, device=edge_index.device),
            assignment,
            coarse_nodes,
        )
    coarse_edges = torch.unique(coarse_edges.t(), dim=0).t().contiguous()
    return coarse_edges, assignment, coarse_nodes


def coregd_reference_forward(
    edge_index: torch.Tensor,
    num_nodes: int,
    config: CoreGDConfig,
    *,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """Run one CoRe-GD forward pass without hierarchy wrapping.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    config : CoreGDConfig
        Resolved CoRe-GD configuration.
    device : str or torch.device, optional
        Target device. ``None`` uses the edge tensor's device.

    Returns
    -------
    torch.Tensor
        Position tensor with shape ``[N, 2]``.
    """
    resolved_device = torch.device(device) if device is not None else edge_index.device
    torch.manual_seed(int(config.seed))
    model = build_coregd_model(config).to(resolved_device)
    _load_checkpoint(model, config.checkpoint_path, resolved_device)
    model.eval()
    data = prepare_coregd_data(
        edge_index=edge_index.to(device=resolved_device, dtype=torch.long),
        num_nodes=num_nodes,
        config=config,
        device=resolved_device,
    )
    with torch.no_grad():
        pos = model(data, int(config.iterations), transform_to_undirected=True)
    if isinstance(pos, tuple):
        return pos[0].to(dtype=torch.float32)
    return pos.to(dtype=torch.float32)


def layout_coregd_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    config: Optional[Union[CoreGDConfig, LayoutConfig]] = None,
    seed: int = 42,
    steps: Optional[int] = None,
    checkpoint_path: Optional[str] = None,
    config_path: Optional[str] = None,
    coarsen: Optional[bool] = None,
    iterations: Optional[int] = None,
    fidelity_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Lay out a graph with the CoRe-GD neural stress pipeline.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Node sizes with shape ``[N, 2]``. Used by the hierarchy warm start.
    config : CoreGDConfig or LayoutConfig, optional
        CoRe-GD or public layout configuration.
    seed : int, default=42
        Random seed for model initialization and randomized node features.
    steps : int, optional
        Public layout step budget. When provided, it overrides iterations.
    checkpoint_path : str, optional
        Optional pretrained checkpoint path.
    config_path : str, optional
        Optional reference JSON config path.
    coarsen : bool, optional
        Override hierarchy usage.
    iterations : int, optional
        Override neural refinement count.
    fidelity_dtype : torch.dtype, optional
        Accepted for pipeline API compatibility; CoRe-GD inference uses
        checkpoint dtype.

    Returns
    -------
    torch.Tensor
        Position tensor with shape ``[N, 2]``.
    """
    del fidelity_dtype
    resolved = _resolve_coregd_config(
        config,
        seed=seed,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        coarsen=coarsen,
        iterations=iterations if iterations is not None else steps,
    )
    if num_nodes <= 0:
        return torch.empty((0, 2), dtype=torch.float32, device=edge_index.device)

    pos = coregd_reference_forward(
        edge_index=edge_index,
        num_nodes=num_nodes,
        config=resolved,
        device=edge_index.device,
    )

    # The paper's CoRe-GD hierarchy trains/evaluates coarse graphs then refines
    # finer levels. Dagua keeps this as an inference-time warm-start hierarchy
    # so the native pipeline can use it without depending on the paper's
    # external graph-coarsening package.
    if bool(resolved.coarsen) and num_nodes > int(resolved.coarsen_min_size):
        coarse_target = max(
            int(resolved.coarsen_min_size),
            num_nodes // max(int(resolved.coarsen_k), 2),
        )
        coarse_edges, assignment, coarse_nodes = _coarsen_edge_index(
            edge_index,
            num_nodes,
            coarse_target,
        )
        coarse_pos = coregd_reference_forward(
            edge_index=coarse_edges,
            num_nodes=coarse_nodes,
            config=resolved,
            device=edge_index.device,
        )
        pos = 0.5 * pos + 0.5 * coarse_pos[assignment].to(device=pos.device)
        if node_sizes is not None:
            ml_config = NativeStressMLConfig(
                ml_min_nodes=0,
                ml_min_edges=0,
                coarsest_nodes=max(2, coarse_nodes),
                max_levels=1,
                coarse_steps=0,
                refine_steps=2,
                overlap_max_nodes=0,
                seed=int(resolved.seed),
            )
            pos = layout_native_stress_ml_pipeline(
                edge_index=edge_index,
                num_nodes=num_nodes,
                node_sizes=node_sizes,
                config=ml_config,
                seed=int(resolved.seed),
                pos=pos,
            )

    return pos.to(dtype=torch.float32)
