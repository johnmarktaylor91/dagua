"""NeuLay two-phase graph layout.

This module implements the published NeuLay objective with an optional
PyTorch-Geometric GCN reparameterization phase followed by direct force-driven
refinement.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn

try:
    from torch_geometric.nn import GCNConv as _GCNConv
except Exception:
    _GCNConv = None

_FULL_REPULSION_THRESHOLD = 512
_REPULSION_SAMPLE_COUNT = 8_192
_SHORT_STOP_WINDOW = 32
_LONG_STOP_WINDOW = 1_000
_SHORT_STOP_RATIO = 5.0e-4
_LONG_STOP_RATIO = 1.0e-4
_LATENT_DIM = 10
_GNN_LR = 0.05
_EPS = 1.0e-9


def _layout_device(
    edge_index: torch.Tensor,
    node_sizes: Optional[torch.Tensor],
) -> torch.device:
    """Choose the output device for the layout result.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    node_sizes : torch.Tensor | None
        Optional node-size tensor.

    Returns
    -------
    torch.device
        Device used for optimization and the returned coordinates.
    """
    if edge_index.numel() > 0:
        return edge_index.device
    if node_sizes is not None:
        return node_sizes.device
    return torch.device("cpu")


def _validate_inputs(
    edge_index: torch.Tensor,
    num_nodes: int,
    steps: int,
    gcn_steps: int,
    dim: int,
    lr: float,
    radius: float,
    magnitude: float,
    edge_weights: Optional[torch.Tensor],
) -> None:
    """Validate the public NeuLay inputs.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes in the graph.
    steps : int
        Number of direct-refinement steps.
    gcn_steps : int
        Number of GCN reparameterization steps.
    dim : int
        Embedding dimensionality.
    lr : float
        Adam learning rate for the direct phase.
    radius : float
        Gaussian repulsion radius.
    magnitude : float
        Gaussian repulsion magnitude.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.

    Returns
    -------
    None
        Raises ``ValueError`` when the configuration is invalid.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    if gcn_steps < 0:
        raise ValueError("gcn_steps must be non-negative.")
    if dim <= 0:
        raise ValueError("dim must be positive.")
    if lr <= 0.0:
        raise ValueError("lr must be positive.")
    if radius <= 0.0:
        raise ValueError("radius must be positive.")
    if magnitude < 0.0:
        raise ValueError("magnitude must be non-negative.")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E].")
    if edge_weights is not None and edge_weights.shape[0] != edge_index.shape[1]:
        raise ValueError(
            f"edge_weights length {edge_weights.shape[0]} does not match "
            f"edge count {edge_index.shape[1]}"
        )
    if edge_index.numel() == 0:
        return

    if edge_index.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise ValueError("edge_index must use an integer dtype.")

    min_index = int(edge_index.min().item())
    max_index = int(edge_index.max().item())
    if min_index < 0 or max_index >= num_nodes:
        raise ValueError("edge_index contains node indices outside [0, num_nodes).")


def _set_seed(seed: int) -> None:
    """Seed the PyTorch RNGs used by NeuLay.

    Parameters
    ----------
    seed : int
        Requested random seed.

    Returns
    -------
    None
        The global RNG state is updated in-place.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _clean_edge_index(edge_index: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Remove self-loops and move the edge list to the optimization device.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    device : torch.device
        Device used by the optimization loop.

    Returns
    -------
    torch.Tensor
        Cleaned edge tensor with shape ``[2, E_clean]``.
    """
    cleaned = edge_index.to(device=device, dtype=torch.long)
    if cleaned.numel() == 0:
        return cleaned.reshape(2, 0)
    non_self = cleaned[0] != cleaned[1]
    return cleaned[:, non_self].contiguous()


def _initial_positions(num_nodes: int, dim: int, device: torch.device) -> torch.Tensor:
    """Create the NeuLay random initialization.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    dim : int
        Embedding dimensionality.
    device : torch.device
        Device used for the returned tensor.

    Returns
    -------
    torch.Tensor
        Initial coordinates with shape ``[N, dim]``.
    """
    scale = math.sqrt(float(max(num_nodes, 1)))
    return torch.randn((num_nodes, dim), device=device, dtype=torch.float32) * scale


def _center_positions(pos: torch.Tensor) -> torch.Tensor:
    """Remove translational drift from a coordinate tensor.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, dim]``.

    Returns
    -------
    torch.Tensor
        Centered coordinates with the same shape as ``pos``.
    """
    if pos.numel() == 0:
        return pos
    return pos - pos.mean(dim=0, keepdim=True)


def _elastic_loss(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Evaluate the NeuLay elastic energy.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, dim]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.

    Returns
    -------
    torch.Tensor
        Scalar elastic loss.
    """
    if edge_index.numel() == 0:
        return pos.sum() * 0.0
    src = edge_index[0]
    dst = edge_index[1]
    diff = pos[src] - pos[dst]
    return diff.square().sum() * 0.5


def _repulsion_loss(
    pos: torch.Tensor,
    radius: float,
    magnitude: float,
) -> torch.Tensor:
    """Evaluate exact or sampled Gaussian repulsion.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, dim]``.
    radius : float
        Gaussian radius.
    magnitude : float
        Repulsion magnitude.

    Returns
    -------
    torch.Tensor
        Scalar repulsion loss.
    """
    num_nodes = pos.shape[0]
    if num_nodes == 0 or magnitude == 0.0:
        return pos.sum() * 0.0

    radius_term = 4.0 * radius * radius
    if num_nodes <= _FULL_REPULSION_THRESHOLD:
        pairwise_sq = torch.cdist(pos, pos).square()
        return magnitude * torch.exp(-pairwise_sq / radius_term).sum()

    sample_count = min(_REPULSION_SAMPLE_COUNT, max(num_nodes * 16, 1_024))
    src = torch.randint(0, num_nodes, (sample_count,), device=pos.device)
    dst = torch.randint(0, num_nodes, (sample_count,), device=pos.device)
    diff = pos[src] - pos[dst]
    kernel = torch.exp(-diff.square().sum(dim=1) / radius_term)
    return magnitude * kernel.mean() * float(num_nodes * num_nodes)


def _loss_change_ratio(loss_history: list[float], window: int) -> Optional[float]:
    """Compute the relative change between consecutive loss windows.

    Parameters
    ----------
    loss_history : list[float]
        Scalar loss values accumulated during optimization.
    window : int
        Window size used for the rolling comparison.

    Returns
    -------
    float | None
        Relative mean-loss change, or ``None`` until two full windows are
        available.
    """
    if len(loss_history) < 2 * window:
        return None
    previous = sum(loss_history[-2 * window : -window]) / float(window)
    current = sum(loss_history[-window:]) / float(window)
    return abs(current - previous) / max(abs(previous), _EPS)


def _should_stop(loss_history: list[float]) -> bool:
    """Check the NeuLay dual-window early-stopping condition.

    Parameters
    ----------
    loss_history : list[float]
        Scalar loss values accumulated during optimization.

    Returns
    -------
    bool
        ``True`` when both the short and long windows have stabilized.
    """
    short_ratio = _loss_change_ratio(loss_history, _SHORT_STOP_WINDOW)
    long_ratio = _loss_change_ratio(loss_history, _LONG_STOP_WINDOW)
    if short_ratio is None or long_ratio is None:
        return False
    return short_ratio < _SHORT_STOP_RATIO and long_ratio < _LONG_STOP_RATIO


def _optimize_positions(
    initial_pos: torch.Tensor,
    edge_index: torch.Tensor,
    steps: int,
    lr: float,
    radius: float,
    magnitude: float,
) -> torch.Tensor:
    """Run the direct NeuLay refinement phase.

    Parameters
    ----------
    initial_pos : torch.Tensor
        Initial coordinates with shape ``[N, dim]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    steps : int
        Number of Adam steps.
    lr : float
        Adam learning rate.
    radius : float
        Gaussian repulsion radius.
    magnitude : float
        Gaussian repulsion magnitude.

    Returns
    -------
    torch.Tensor
        Refined coordinates with shape ``[N, dim]``.
    """
    pos = nn.Parameter(initial_pos.clone())
    optimizer = torch.optim.Adam([pos], lr=lr)
    loss_history: list[float] = []

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        centered = _center_positions(pos)
        loss = _elastic_loss(centered, edge_index) + _repulsion_loss(
            centered,
            radius=radius,
            magnitude=magnitude,
        )
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            pos.sub_(pos.mean(dim=0, keepdim=True))
        loss_history.append(float(loss.detach().item()))
        if _should_stop(loss_history):
            break

    return _center_positions(pos.detach())


class _ResGCN(nn.Module):
    """NeuLay residual GCN with cross-layer skip concatenation."""

    def __init__(self, num_nodes: int, dim: int, device: torch.device) -> None:
        """Construct the published NeuLay encoder.

        Parameters
        ----------
        num_nodes : int
            Number of graph nodes.
        dim : int
            Output dimensionality.
        device : torch.device
            Device used for the learnable parameters.
        """
        super().__init__()
        if _GCNConv is None:
            raise RuntimeError("PyTorch Geometric is required for the NeuLay GCN phase.")

        latent_scale = float(max(num_nodes, 1)) ** (1.0 / float(_LATENT_DIM))
        self.latent = nn.Parameter(
            torch.randn((num_nodes, _LATENT_DIM), device=device, dtype=torch.float32) * latent_scale
        )
        self.gcn_layers = nn.ModuleList([_GCNConv(_LATENT_DIM, _LATENT_DIM)])
        self.proj = nn.Linear(_LATENT_DIM * 2, dim)

    def forward(self, edge_index: torch.Tensor) -> torch.Tensor:
        """Project the latent node features into coordinates.

        Parameters
        ----------
        edge_index : torch.Tensor
            Edge tensor with shape ``[2, E]``.

        Returns
        -------
        torch.Tensor
            Predicted coordinates with shape ``[N, dim]``.
        """
        outputs: list[torch.Tensor] = [self.latent]
        features = self.latent
        for layer in self.gcn_layers:
            features = layer(features, edge_index)
            outputs.append(features)
        return self.proj(torch.cat(outputs, dim=1))


def _optimize_gcn_phase(
    edge_index: torch.Tensor,
    num_nodes: int,
    dim: int,
    device: torch.device,
    steps: int,
    radius: float,
    magnitude: float,
) -> torch.Tensor:
    """Run the optional NeuLay GCN reparameterization phase.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    dim : int
        Embedding dimensionality.
    device : torch.device
        Optimization device.
    steps : int
        Number of GCN Adam steps.
    radius : float
        Gaussian repulsion radius.
    magnitude : float
        Gaussian repulsion magnitude.

    Returns
    -------
    torch.Tensor
        Coarse coordinates with shape ``[N, dim]``.
    """
    model = _ResGCN(num_nodes=num_nodes, dim=dim, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=_GNN_LR)
    loss_history: list[float] = []

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        centered = _center_positions(model(edge_index))
        loss = _elastic_loss(centered, edge_index) + _repulsion_loss(
            centered,
            radius=radius,
            magnitude=magnitude,
        )
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.detach().item()))
        if _should_stop(loss_history):
            break

    with torch.no_grad():
        return _center_positions(model(edge_index).detach())


def layout_neulay(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    seed: int = 42,
    steps: int = 20_000,
    gcn_steps: int = 2_000,
    use_gcn: bool = True,
    dim: int = 2,
    lr: float = 0.1,
    radius: float = 0.4,
    magnitude: float = 10.0,
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Lay out a graph with the published NeuLay objective.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor | None, default=None
        Unused placeholder kept for API compatibility with other layouts.
    seed : int, default=42
        Random seed used for the initialization and optimizer trajectory.
    steps : int, default=20000
        Number of direct Adam refinement steps.
    gcn_steps : int, default=2000
        Number of GCN reparameterization steps when PyG is available.
    use_gcn : bool, default=True
        Whether to run the optional GCN phase when PyG is installed.
    dim : int, default=2
        Output dimensionality.
    lr : float, default=0.1
        Adam learning rate for the direct phase.
    radius : float, default=0.4
        Gaussian repulsion radius.
    magnitude : float, default=10.0
        Gaussian repulsion magnitude.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``. Accepted for interface
        consistency; the current NeuLay port does not yet thread them into the
        GCN message-passing or direct force phases.

    Returns
    -------
    torch.Tensor
        Coordinate tensor with shape ``[N, dim]``.
    """
    _validate_inputs(
        edge_index=edge_index,
        num_nodes=num_nodes,
        steps=steps,
        gcn_steps=gcn_steps,
        dim=dim,
        lr=lr,
        radius=radius,
        magnitude=magnitude,
        edge_weights=edge_weights,
    )
    device = _layout_device(edge_index=edge_index, node_sizes=node_sizes)
    if num_nodes == 0:
        return torch.empty((0, dim), dtype=torch.float32, device=device)
    if num_nodes == 1:
        return torch.zeros((1, dim), dtype=torch.float32, device=device)

    _set_seed(seed)
    cleaned_edge_index = _clean_edge_index(edge_index=edge_index, device=device)
    # TODO: integrate edge_weights into GCN message passing and the direct loss.
    if cleaned_edge_index.numel() == 0:
        return _center_positions(_initial_positions(num_nodes=num_nodes, dim=dim, device=device))

    if use_gcn and _GCNConv is not None and gcn_steps > 0:
        coarse = _optimize_gcn_phase(
            edge_index=cleaned_edge_index,
            num_nodes=num_nodes,
            dim=dim,
            device=device,
            steps=gcn_steps,
            radius=radius,
            magnitude=magnitude,
        )
    else:
        coarse = _center_positions(_initial_positions(num_nodes=num_nodes, dim=dim, device=device))

    if steps == 0:
        return coarse

    return _optimize_positions(
        initial_pos=coarse,
        edge_index=cleaned_edge_index,
        steps=steps,
        lr=lr,
        radius=radius,
        magnitude=magnitude,
    )
