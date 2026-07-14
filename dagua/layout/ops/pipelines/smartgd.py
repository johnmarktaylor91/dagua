"""SmartGD neural graph drawing generator pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Optional

import torch
from torch import nn

from dagua.config import LayoutConfig
from dagua.layout.ops.pipelines.native_stress import layout_native_stress_pipeline

_DEFAULT_NUM_BLOCKS = 11
_DEFAULT_BLOCK_DEPTH = 3
_DEFAULT_BLOCK_WIDTH = 8
_DEFAULT_BLOCK_OUTPUT_DIM = 8
_DEFAULT_EDGE_NET_DEPTH = 2
_DEFAULT_EDGE_NET_WIDTH = 16
_DEFAULT_EDGE_ATTR_DIM = 2
_DEFAULT_NODE_ATTR_DIM = 2
_DEFAULT_EPS = 1.0e-7


@dataclass(frozen=True)
class SmartGDConfig:
    """Configuration for the SmartGD generator pipeline.

    Parameters
    ----------
    num_blocks : int, default=11
        Number of residual generator blocks after the input block.
    block_depth : int, default=3
        Number of hidden NNConv layers per residual block.
    block_width : int, default=8
        Hidden channel count inside each residual block.
    block_output_dim : int, default=8
        Output channel count for each residual block.
    edge_net_depth : int, default=2
        Number of hidden layers in each NNConv edge network.
    edge_net_width : int, default=16
        Hidden channel count in each NNConv edge network.
    edge_attr_dim : int, default=2
        Edge feature count. The reference checkpoints use shortest-path
        distance and inverse squared distance.
    node_attr_dim : int, default=2
        Input coordinate feature count.
    checkpoint_path : str, optional
        Optional generator checkpoint path. When omitted, random weights are
        used unless ``objective`` resolves to a bundled reference checkpoint.
    objective : str, default="stress"
        Built-in checkpoint selector, ``"stress"`` or ``"crossings"``.
    use_reference_checkpoint : bool, default=True
        Whether to load the `/tmp/smartgd-ref` checkpoint when available.
    seed : int, default=42
        Deterministic seed for random initialization and fallback positions.
    """

    num_blocks: int = _DEFAULT_NUM_BLOCKS
    block_depth: int = _DEFAULT_BLOCK_DEPTH
    block_width: int = _DEFAULT_BLOCK_WIDTH
    block_output_dim: int = _DEFAULT_BLOCK_OUTPUT_DIM
    edge_net_depth: int = _DEFAULT_EDGE_NET_DEPTH
    edge_net_width: int = _DEFAULT_EDGE_NET_WIDTH
    edge_attr_dim: int = _DEFAULT_EDGE_ATTR_DIM
    node_attr_dim: int = _DEFAULT_NODE_ATTR_DIM
    checkpoint_path: Optional[str] = None
    objective: str = "stress"
    use_reference_checkpoint: bool = True
    seed: int = 42


@dataclass(frozen=True)
class EdgeFeatureExpansions:
    """Edge feature expansion flags matching the SmartGD reference.

    Parameters
    ----------
    src_feat : bool, default=False
        Append source node features.
    dst_feat : bool, default=False
        Append target node features.
    diff_vec : bool, default=False
        Append source-minus-target feature vectors.
    unit_vec : bool, default=False
        Append normalized source-minus-target vectors.
    vec_norm : bool, default=False
        Append Euclidean vector norms.
    vec_norm_inv : bool, default=False
        Append inverse Euclidean vector norms.
    vec_norm_square : bool, default=False
        Append squared Euclidean vector norms.
    vec_norm_inv_square : bool, default=False
        Append squared inverse Euclidean vector norms.
    edge_attr_inv : bool, default=False
        Append inverse edge attributes.
    edge_attr_square : bool, default=False
        Append squared edge attributes.
    edge_attr_inv_square : bool, default=False
        Append squared inverse edge attributes.
    """

    src_feat: bool = False
    dst_feat: bool = False
    diff_vec: bool = False
    unit_vec: bool = False
    vec_norm: bool = False
    vec_norm_inv: bool = False
    vec_norm_square: bool = False
    vec_norm_inv_square: bool = False
    edge_attr_inv: bool = False
    edge_attr_square: bool = False
    edge_attr_inv_square: bool = False


class SkipConnection(nn.Module):
    """Reference residual projection helper."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        """Initialize the residual projection.

        Parameters
        ----------
        in_dim : int
            Input feature width.
        out_dim : int
            Output feature width.
        """
        super().__init__()
        self.same_dim = in_dim == out_dim
        self.proj = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, *, block_input: torch.Tensor, block_output: torch.Tensor) -> torch.Tensor:
        """Add input features to output features.

        Parameters
        ----------
        block_input : torch.Tensor
            Input tensor with shape ``[N, F_in]``.
        block_output : torch.Tensor
            Output tensor with shape ``[N, F_out]``.

        Returns
        -------
        torch.Tensor
            Residual sum with shape ``[N, F_out]``.
        """
        if self.same_dim:
            return block_input + block_output
        return self.proj(block_input) + block_output


class LinearLayer(nn.Module):
    """Reference MLP linear block."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        bn: Optional[str] = "batch_norm",
        act: Optional[str] = "leaky_relu",
        dp: float = 0.0,
        residual: bool = False,
    ) -> None:
        """Initialize a linear block.

        Parameters
        ----------
        in_dim : int
            Input feature width.
        out_dim : int
            Output feature width.
        bn : str, optional
            Normalization selector. Only ``"batch_norm"`` is used by the
            reference checkpoints.
        act : str, optional
            Activation selector.
        dp : float, default=0.0
            Dropout probability.
        residual : bool, default=False
            Whether to add a residual skip.
        """
        super().__init__()
        self.with_bn = bn is not None
        self.with_act = act is not None
        self.with_dp = dp > 0.0
        self.residual = residual
        self.dense = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim) if bn == "batch_norm" else nn.Identity()
        self.act = _activation(act)
        self.dp = nn.Dropout(dp)
        self.skip = SkipConnection(in_dim, out_dim)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """Apply the linear block.

        Parameters
        ----------
        feat : torch.Tensor
            Feature tensor with shape ``[N, F]``.

        Returns
        -------
        torch.Tensor
            Transformed feature tensor.
        """
        inputs = outputs = feat
        outputs = self.dense(outputs)
        if self.with_bn:
            outputs = self.bn(outputs)
        if self.with_act:
            outputs = self.act(outputs)
        if self.with_dp:
            outputs = self.dp(outputs)
        if self.residual:
            outputs = self.skip(block_input=inputs, block_output=outputs)
        return outputs


class MLP(nn.Module):
    """Reference edge-network MLP."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: list[int],
        *,
        hidden_act: str = "leaky_relu",
        out_act: Optional[str] = None,
        bn: Optional[str] = "batch_norm",
        dp: float = 0.0,
        residual: bool = True,
    ) -> None:
        """Initialize the reference MLP.

        Parameters
        ----------
        in_dim : int
            Input feature width.
        out_dim : int
            Output feature width.
        hidden_dims : list[int]
            Hidden layer widths.
        hidden_act : str, default="leaky_relu"
            Hidden activation selector.
        out_act : str, optional
            Output activation selector.
        bn : str, optional
            Hidden normalization selector.
        dp : float, default=0.0
            Hidden dropout probability.
        residual : bool, default=True
            Whether hidden layers use residual skips.
        """
        super().__init__()
        self.linear_seq = nn.Sequential()
        in_dims = [in_dim] + hidden_dims
        out_dims = hidden_dims + [out_dim]
        for index, (source_dim, target_dim) in enumerate(zip(in_dims, out_dims)):
            if index < len(hidden_dims):
                self.linear_seq.append(
                    LinearLayer(
                        source_dim,
                        target_dim,
                        bn=bn,
                        act=hidden_act,
                        dp=dp,
                        residual=residual,
                    )
                )
            else:
                self.linear_seq.append(
                    LinearLayer(
                        source_dim,
                        target_dim,
                        bn=None,
                        act=out_act,
                        dp=0.0,
                        residual=False,
                    )
                )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """Apply all MLP layers.

        Parameters
        ----------
        feat : torch.Tensor
            Feature tensor with shape ``[N, F]``.

        Returns
        -------
        torch.Tensor
            Transformed feature tensor.
        """
        return self.linear_seq(feat)


class EdgeFeatureExpansion(nn.Module):
    """Reference edge-feature expansion module."""

    def __init__(
        self,
        node_feat_dim: int,
        edge_attr_dim: int,
        expansions: EdgeFeatureExpansions,
        eps: float = _DEFAULT_EPS,
    ) -> None:
        """Initialize edge expansion.

        Parameters
        ----------
        node_feat_dim : int
            Node feature width used for relative features.
        edge_attr_dim : int
            Base edge attribute width.
        expansions : EdgeFeatureExpansions
            Expansion flags.
        eps : float, default=1e-7
            Division guard matching the reference constant.
        """
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.edge_attr_dim = edge_attr_dim
        self.expansions = expansions
        self.eps = eps
        self.get_edge_feat = torch.jit.trace_module(
            _TracedEdgeFeatureExpansion(expansions=expansions, eps=eps),
            inputs={
                "forward": {
                    "node_feat": torch.zeros((1, max(node_feat_dim, 0)), dtype=torch.float32),
                    "edge_attr": torch.zeros((1, max(edge_attr_dim, 0)), dtype=torch.float32),
                    "edge_index": torch.zeros((2, 1), dtype=torch.long),
                }
            },
        )

    def get_feature_channels(
        self,
        *,
        node_feat_dim: Optional[int] = None,
        edge_attr_dim: Optional[int] = None,
    ) -> int:
        """Return output channel count for the configured expansion.

        Parameters
        ----------
        node_feat_dim : int, optional
            Override node feature width.
        edge_attr_dim : int, optional
            Override edge attribute width.

        Returns
        -------
        int
            Expanded edge feature width.
        """
        n_dim = int(node_feat_dim if node_feat_dim is not None else self.node_feat_dim)
        e_dim = int(edge_attr_dim if edge_attr_dim is not None else self.edge_attr_dim)
        total = e_dim
        for name, enabled in self.expansions.__dict__.items():
            if not enabled:
                continue
            if name in {"src_feat", "dst_feat", "diff_vec", "unit_vec"}:
                total += n_dim
            else:
                total += 1 if name.startswith("vec_norm") else e_dim
        return total

    def forward(
        self,
        *,
        node_feat: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Expand edge attributes.

        Parameters
        ----------
        node_feat : torch.Tensor
            Node features with shape ``[N, F]``.
        edge_attr : torch.Tensor
            Edge attributes with shape ``[E, A]``.
        edge_index : torch.Tensor
            Directed edge tensor with shape ``[2, E]``.

        Returns
        -------
        torch.Tensor
            Expanded edge attributes with shape ``[E, A2]``.
        """
        return self.get_edge_feat(
            {
                "node_feat": node_feat,
                "edge_attr": edge_attr,
                "edge_index": edge_index,
            }
        )


class _TracedEdgeFeatureExpansion(nn.Module):
    """TorchScript-traced edge expansion matching the reference helper."""

    def __init__(self, expansions: EdgeFeatureExpansions, eps: float) -> None:
        """Initialize the traced expansion body.

        Parameters
        ----------
        expansions : EdgeFeatureExpansions
            Expansion flags.
        eps : float
            Division guard.
        """
        super().__init__()
        self.expansions = expansions
        self.eps = eps

    def forward(self, kwargs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Expand edge features from a traced kwargs dictionary.

        Parameters
        ----------
        kwargs : dict[str, torch.Tensor]
            Dictionary containing ``node_feat``, ``edge_attr``, and
            ``edge_index`` tensors.

        Returns
        -------
        torch.Tensor
            Expanded edge features.
        """
        node_feat = kwargs["node_feat"]
        edge_attr = kwargs["edge_attr"]
        edge_index = kwargs["edge_index"]
        feat_list = [edge_attr]
        src_dst_feat = node_feat[edge_index]
        src_feat = src_dst_feat[1, ...]
        dst_feat = src_dst_feat[0, ...]
        diff_vec = src_feat - dst_feat
        vec_norm = diff_vec.norm(dim=1, p=2, keepdim=True)
        if self.expansions.src_feat:
            feat_list.append(src_feat)
        if self.expansions.dst_feat:
            feat_list.append(dst_feat)
        if self.expansions.diff_vec:
            feat_list.append(diff_vec)
        if self.expansions.unit_vec:
            feat_list.append(diff_vec / (vec_norm + self.eps))
        if self.expansions.vec_norm:
            feat_list.append(vec_norm)
        if self.expansions.vec_norm_inv:
            feat_list.append(1 / (vec_norm + self.eps))
        if self.expansions.vec_norm_square:
            feat_list.append(vec_norm**2)
        if self.expansions.vec_norm_inv_square:
            feat_list.append((1 / (vec_norm + self.eps)) ** 2)
        if self.expansions.edge_attr_inv:
            feat_list.append(1 / (edge_attr + self.eps))
        if self.expansions.edge_attr_square:
            feat_list.append(edge_attr**2)
        if self.expansions.edge_attr_inv_square:
            feat_list.append((1 / (edge_attr + self.eps)) ** 2)
        return torch.cat(feat_list, dim=-1)


class NNConvBasicLayer(nn.Module):
    """Reference NNConv block with optional normalization."""

    def __init__(
        self,
        layer_index: int,
        in_dim: int,
        out_dim: int,
        edge_feat_dim: int,
        edge_net: nn.Module,
        *,
        bn: Optional[str] = "pyg_batch_norm",
        act: Optional[str] = "leaky_relu",
        dp: float = 0.0,
        residual: bool = False,
        aggr: str = "mean",
        root_weight: bool = True,
    ) -> None:
        """Initialize a reference NNConv block.

        Parameters
        ----------
        layer_index : int
            Layer index retained for reference naming and sampled-mode parity.
        in_dim : int
            Input node feature width.
        out_dim : int
            Output node feature width.
        edge_feat_dim : int
            Edge feature width.
        edge_net : torch.nn.Module
            Edge-conditioned weight network.
        bn : str, optional
            Normalization selector.
        act : str, optional
            Activation selector.
        dp : float, default=0.0
            Dropout probability.
        residual : bool, default=False
            Whether to apply a residual skip.
        aggr : str, default="mean"
            PyG aggregation mode.
        root_weight : bool, default=True
            Whether NNConv includes a root transform.
        """
        super().__init__()
        from torch_geometric.nn import BatchNorm, NNConv

        self.layer_index = layer_index
        self.with_bn = bn is not None
        self.with_act = act is not None
        self.with_dp = dp > 0.0
        self.residual = residual
        self.conv = NNConv(in_dim, out_dim, nn=edge_net, aggr=aggr, root_weight=root_weight)
        self.dense = nn.Linear(out_dim, out_dim)
        self.bn = (
            PygBatchNormWrapper(BatchNorm(out_dim)) if bn == "pyg_batch_norm" else nn.Identity()
        )
        self.act = _activation(act)
        self.dp = nn.Dropout(dp)
        self.skip = SkipConnection(in_dim, out_dim)

    def forward(
        self,
        *,
        node_feat: torch.Tensor,
        edge_feat: torch.Tensor,
        edge_index: torch.Tensor,
        batch_index: torch.Tensor,
        num_sampled_nodes_per_hop: Optional[list[int]],
        num_sampled_edges_per_hop: Optional[list[int]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply the NNConv block.

        Parameters
        ----------
        node_feat : torch.Tensor
            Node features with shape ``[N, F]``.
        edge_feat : torch.Tensor
            Edge features with shape ``[E, A]``.
        edge_index : torch.Tensor
            Directed edge tensor with shape ``[2, E]``.
        batch_index : torch.Tensor
            Batch vector with shape ``[N]``.
        num_sampled_nodes_per_hop : list[int], optional
            Unused sampled-neighborhood metadata kept for signature parity.
        num_sampled_edges_per_hop : list[int], optional
            Unused sampled-neighborhood metadata kept for signature parity.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Updated node features, edge index, and edge features.
        """
        del num_sampled_nodes_per_hop, num_sampled_edges_per_hop
        inputs = outputs = node_feat
        outputs = self.conv(x=outputs, edge_index=edge_index, edge_attr=edge_feat)
        if self.with_bn:
            outputs = self.bn(outputs)
        if self.with_act:
            outputs = self.act(outputs)
        if self.with_dp:
            outputs = self.dp(outputs)
        if self.residual:
            outputs = self.skip(block_input=inputs, block_output=outputs)
        return outputs, edge_index, edge_feat


class PygBatchNormWrapper(nn.Module):
    """Reference wrapper that preserves SmartGD checkpoint key names."""

    def __init__(self, bn: nn.Module) -> None:
        """Initialize the wrapper.

        Parameters
        ----------
        bn : torch.nn.Module
            PyG batch normalization module.
        """
        super().__init__()
        self.bn = bn

    def forward(self, node_feat: torch.Tensor) -> torch.Tensor:
        """Apply wrapped PyG batch normalization.

        Parameters
        ----------
        node_feat : torch.Tensor
            Node features with shape ``[N, F]``.

        Returns
        -------
        torch.Tensor
            Normalized node features with shape ``[N, F]``.
        """
        return self.bn(node_feat)


class NNConvLayer(nn.Module):
    """Reference wrapper around ``NNConvBasicLayer``."""

    def __init__(
        self,
        layer_index: int,
        in_dim: int,
        out_dim: int,
        edge_feat_dim: int,
        *,
        edge_net_width: int,
        edge_net_depth: int,
        edge_hidden_act: str = "leaky_relu",
        edge_out_act: Optional[str] = "tanh",
        edge_bn: Optional[str] = "batch_norm",
        edge_dp: float = 0.2,
        edge_residual: bool = False,
        bn: Optional[str] = "pyg_batch_norm",
        act: Optional[str] = "leaky_relu",
        dp: float = 0.1,
        aggr: str = "mean",
        root_weight: bool = True,
    ) -> None:
        """Initialize a reference NNConv wrapper.

        Parameters
        ----------
        layer_index : int
            Layer index.
        in_dim : int
            Input node feature width.
        out_dim : int
            Output node feature width.
        edge_feat_dim : int
            Edge feature width.
        edge_net_width : int
            Edge network hidden width.
        edge_net_depth : int
            Edge network hidden depth.
        edge_hidden_act : str, default="leaky_relu"
            Edge network hidden activation.
        edge_out_act : str, optional
            Edge network output activation.
        edge_bn : str, optional
            Edge network normalization.
        edge_dp : float, default=0.2
            Edge network dropout.
        edge_residual : bool, default=False
            Edge network residual flag.
        bn : str, optional
            GNN normalization.
        act : str, optional
            GNN activation.
        dp : float, default=0.1
            GNN dropout.
        aggr : str, default="mean"
            Aggregation mode.
        root_weight : bool, default=True
            NNConv root transform flag.
        """
        super().__init__()
        self.nnconv_layer = NNConvBasicLayer(
            layer_index,
            in_dim,
            out_dim,
            edge_feat_dim,
            MLP(
                edge_feat_dim,
                in_dim * out_dim,
                [edge_net_width] * edge_net_depth,
                hidden_act=edge_hidden_act,
                out_act=edge_out_act,
                bn=edge_bn,
                dp=edge_dp,
                residual=edge_residual,
            ),
            bn=bn,
            act=act,
            dp=dp,
            residual=False,
            aggr=aggr,
            root_weight=root_weight,
        )

    def forward(
        self,
        *,
        node_feat: torch.Tensor,
        edge_feat: torch.Tensor,
        edge_index: torch.Tensor,
        batch_index: torch.Tensor,
        num_sampled_nodes_per_hop: Optional[list[int]],
        num_sampled_edges_per_hop: Optional[list[int]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply wrapped NNConv layer.

        Parameters
        ----------
        node_feat : torch.Tensor
            Node features with shape ``[N, F]``.
        edge_feat : torch.Tensor
            Edge features with shape ``[E, A]``.
        edge_index : torch.Tensor
            Directed edge tensor with shape ``[2, E]``.
        batch_index : torch.Tensor
            Batch vector with shape ``[N]``.
        num_sampled_nodes_per_hop : list[int], optional
            Unused sampled-neighborhood metadata.
        num_sampled_edges_per_hop : list[int], optional
            Unused sampled-neighborhood metadata.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Updated node features, edge index, and edge features.
        """
        return self.nnconv_layer(
            node_feat=node_feat,
            edge_feat=edge_feat,
            edge_index=edge_index,
            batch_index=batch_index,
            num_sampled_nodes_per_hop=num_sampled_nodes_per_hop,
            num_sampled_edges_per_hop=num_sampled_edges_per_hop,
        )


class GeneratorFeatureRouter(nn.Module):
    """Reference feature router for dynamic edge attributes."""

    def __init__(
        self,
        *,
        input_source: Optional[str],
        block_input_dim: int,
        raw_input_dim: int,
        edge_attr_dim: int,
        edge_feat_expansion: EdgeFeatureExpansions,
        eps: float = _DEFAULT_EPS,
    ) -> None:
        """Initialize the feature router.

        Parameters
        ----------
        input_source : str, optional
            Source selector: ``"block"``, ``"raw"``, ``"null"``, or ``None``.
        block_input_dim : int
            Block input feature width.
        raw_input_dim : int
            Raw coordinate feature width.
        edge_attr_dim : int
            Base edge attribute width.
        edge_feat_expansion : EdgeFeatureExpansions
            Expansion flags.
        eps : float, default=1e-7
            Division guard.
        """
        super().__init__()
        self.input_source = input_source
        node_feat_dim = {"block": block_input_dim, "raw": raw_input_dim, "null": 0}.get(
            input_source, 0
        )
        expansions = (
            edge_feat_expansion if input_source in {"block", "raw"} else EdgeFeatureExpansions()
        )
        self.edge_feature_provider = EdgeFeatureExpansion(
            node_feat_dim=node_feat_dim,
            edge_attr_dim=edge_attr_dim,
            expansions=expansions,
            eps=eps,
        )

    def get_output_channels(self) -> int:
        """Return routed edge feature width.

        Returns
        -------
        int
            Output edge feature width.
        """
        return self.edge_feature_provider.get_feature_channels()

    def forward(
        self,
        *,
        block_input: torch.Tensor,
        raw_input: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Route and expand edge features.

        Parameters
        ----------
        block_input : torch.Tensor
            Block input node features with shape ``[N, F]``.
        raw_input : torch.Tensor
            Raw input node features with shape ``[N, 2]``.
        edge_attr : torch.Tensor
            Edge attributes with shape ``[E, A]``.
        edge_index : torch.Tensor
            Directed edge tensor with shape ``[2, E]``.

        Returns
        -------
        torch.Tensor
            Routed edge features.
        """
        if self.input_source is None:
            return edge_attr
        if self.input_source == "block":
            node_feat = block_input
        elif self.input_source == "raw":
            node_feat = raw_input
        else:
            node_feat = torch.zeros((block_input.shape[0], 0), device=block_input.device)
        return self.edge_feature_provider(
            node_feat=node_feat,
            edge_attr=edge_attr,
            edge_index=edge_index,
        )


class GeneratorLayer(nn.Module):
    """Reference SmartGD generator layer."""

    def __init__(
        self,
        *,
        layer_index: int,
        in_dim: int,
        out_dim: int,
        node_feat_dim: int,
        edge_feat_dim: int,
        edge_feat_expansion: EdgeFeatureExpansions,
        edge_net_width: int,
        edge_net_depth: int,
        gnn_bn: Optional[str] = "pyg_batch_norm",
        gnn_act: Optional[str] = "leaky_relu",
        gnn_dp: float = 0.1,
        root_weight: bool = True,
        eps: float = _DEFAULT_EPS,
    ) -> None:
        """Initialize a generator layer.

        Parameters
        ----------
        layer_index : int
            Layer index.
        in_dim : int
            Input node feature width.
        out_dim : int
            Output node feature width.
        node_feat_dim : int
            Node feature width used for edge expansion.
        edge_feat_dim : int
            Base edge feature width.
        edge_feat_expansion : EdgeFeatureExpansions
            Expansion flags.
        edge_net_width : int
            Edge network hidden width.
        edge_net_depth : int
            Edge network hidden depth.
        gnn_bn : str, optional
            GNN normalization selector.
        gnn_act : str, optional
            GNN activation selector.
        gnn_dp : float, default=0.1
            GNN dropout.
        root_weight : bool, default=True
            NNConv root transform flag.
        eps : float, default=1e-7
            Division guard.
        """
        super().__init__()
        self.edge_feat_provider = EdgeFeatureExpansion(
            node_feat_dim=node_feat_dim,
            edge_attr_dim=edge_feat_dim,
            expansions=edge_feat_expansion,
            eps=eps,
        )
        self.gnn_layer = NNConvLayer(
            layer_index,
            in_dim,
            out_dim,
            self.edge_feat_provider.get_feature_channels(),
            edge_net_width=edge_net_width,
            edge_net_depth=edge_net_depth,
            bn=gnn_bn,
            act=gnn_act,
            dp=gnn_dp,
            root_weight=root_weight,
        )

    def forward(
        self,
        *,
        node_feat: torch.Tensor,
        edge_feat: torch.Tensor,
        edge_index: torch.Tensor,
        batch_index: torch.Tensor,
        num_sampled_nodes_per_hop: Optional[list[int]],
        num_sampled_edges_per_hop: Optional[list[int]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply the generator layer.

        Parameters
        ----------
        node_feat : torch.Tensor
            Node features with shape ``[N, F]``.
        edge_feat : torch.Tensor
            Edge features with shape ``[E, A]``.
        edge_index : torch.Tensor
            Directed edge tensor with shape ``[2, E]``.
        batch_index : torch.Tensor
            Batch vector with shape ``[N]``.
        num_sampled_nodes_per_hop : list[int], optional
            Unused sampled-neighborhood metadata.
        num_sampled_edges_per_hop : list[int], optional
            Unused sampled-neighborhood metadata.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Updated node features, edge index, and edge features.
        """
        return self.gnn_layer(
            node_feat=node_feat,
            edge_feat=self.edge_feat_provider(
                node_feat=node_feat,
                edge_index=edge_index,
                edge_attr=edge_feat,
            ),
            edge_index=edge_index,
            batch_index=batch_index,
            num_sampled_nodes_per_hop=num_sampled_nodes_per_hop,
            num_sampled_edges_per_hop=num_sampled_edges_per_hop,
        )


class GeneratorBlock(nn.Module):
    """Reference SmartGD generator block."""

    def __init__(
        self,
        *,
        start_layer_index: int,
        in_dim: int,
        hidden_dims: list[int],
        out_dim: int,
        edge_attr_dim: int,
        node_attr_dim: int,
        dynamic_edge_feat_mode: Optional[str],
        residual: bool,
        edge_net_width: int,
        edge_net_depth: int,
        edge_feat_expansion: EdgeFeatureExpansions,
        gnn_bn: Optional[str] = "pyg_batch_norm",
        gnn_act: Optional[str] = "leaky_relu",
        gnn_dp: float = 0.1,
        root_weight: bool = True,
        eps: float = _DEFAULT_EPS,
    ) -> None:
        """Initialize a generator block.

        Parameters
        ----------
        start_layer_index : int
            Index of the first contained layer.
        in_dim : int
            Input node feature width.
        hidden_dims : list[int]
            Hidden layer widths.
        out_dim : int
            Output node feature width.
        edge_attr_dim : int
            Base edge attribute width.
        node_attr_dim : int
            Raw node attribute width.
        dynamic_edge_feat_mode : str, optional
            Dynamic edge feature mode from the reference implementation.
        residual : bool
            Whether to apply a block residual.
        edge_net_width : int
            Edge network hidden width.
        edge_net_depth : int
            Edge network hidden depth.
        edge_feat_expansion : EdgeFeatureExpansions
            Edge expansion flags.
        gnn_bn : str, optional
            GNN normalization selector.
        gnn_act : str, optional
            GNN activation selector.
        gnn_dp : float, default=0.1
            GNN dropout.
        root_weight : bool, default=True
            NNConv root transform flag.
        eps : float, default=1e-7
            Division guard.
        """
        super().__init__()
        self.residual = residual
        self.edge_feat_expansion = edge_feat_expansion
        self.skip = SkipConnection(in_dim, out_dim)
        first_source = {
            "simple": "block",
            "jump": "block",
            "chain": "block",
            "raw_simple": "raw",
            "raw_jump": "raw",
        }.get(dynamic_edge_feat_mode, "null")
        rest_source = {
            "simple": "null",
            "jump": "block",
            "chain": None,
            "raw_simple": "null",
            "raw_jump": "raw",
        }.get(dynamic_edge_feat_mode, "null")
        self.first_layer_feature_router = GeneratorFeatureRouter(
            input_source=first_source,
            block_input_dim=in_dim,
            raw_input_dim=node_attr_dim,
            edge_attr_dim=edge_attr_dim,
            edge_feat_expansion=edge_feat_expansion,
            eps=eps,
        )
        self.rest_layers_feature_router = GeneratorFeatureRouter(
            input_source=rest_source,
            block_input_dim=in_dim,
            raw_input_dim=node_attr_dim,
            edge_attr_dim=edge_attr_dim,
            edge_feat_expansion=edge_feat_expansion,
            eps=eps,
        )
        dims = [in_dim] + hidden_dims + [out_dim]
        self.first_layer = self._make_layer(
            start_layer_index,
            dims[0],
            dims[1],
            self.first_layer_feature_router,
            edge_attr_dim,
            edge_net_width,
            edge_net_depth,
            gnn_bn,
            gnn_act,
            gnn_dp,
            root_weight,
            eps,
        )
        self.rest_layers = nn.ModuleList()
        for layer_index, (source_dim, target_dim) in enumerate(
            zip(dims[1:-1], dims[2:]), start=start_layer_index + 1
        ):
            self.rest_layers.append(
                self._make_layer(
                    layer_index,
                    source_dim,
                    target_dim,
                    self.rest_layers_feature_router,
                    edge_attr_dim,
                    edge_net_width,
                    edge_net_depth,
                    gnn_bn,
                    gnn_act,
                    gnn_dp,
                    root_weight,
                    eps,
                )
            )

    def _make_layer(
        self,
        layer_index: int,
        in_dim: int,
        out_dim: int,
        router: GeneratorFeatureRouter,
        edge_attr_dim: int,
        edge_net_width: int,
        edge_net_depth: int,
        gnn_bn: Optional[str],
        gnn_act: Optional[str],
        gnn_dp: float,
        root_weight: bool,
        eps: float,
    ) -> GeneratorLayer:
        """Create a contained generator layer.

        Parameters
        ----------
        layer_index : int
            Layer index.
        in_dim : int
            Input node feature width.
        out_dim : int
            Output node feature width.
        router : GeneratorFeatureRouter
            Feature router for edge attributes.
        edge_attr_dim : int
            Base edge attribute width.
        edge_net_width : int
            Edge network hidden width.
        edge_net_depth : int
            Edge network hidden depth.
        gnn_bn : str, optional
            GNN normalization selector.
        gnn_act : str, optional
            GNN activation selector.
        gnn_dp : float
            GNN dropout.
        root_weight : bool
            NNConv root transform flag.
        eps : float
            Division guard.

        Returns
        -------
        GeneratorLayer
            Configured generator layer.
        """
        return GeneratorLayer(
            layer_index=layer_index,
            in_dim=in_dim,
            out_dim=out_dim,
            node_feat_dim=in_dim,
            edge_feat_dim=router.get_output_channels() if router.input_source else edge_attr_dim,
            edge_feat_expansion=EdgeFeatureExpansions()
            if router.input_source
            else self.edge_feat_expansion,
            edge_net_width=edge_net_width,
            edge_net_depth=edge_net_depth,
            gnn_bn=gnn_bn,
            gnn_act=gnn_act,
            gnn_dp=gnn_dp,
            root_weight=root_weight,
            eps=eps,
        )

    def forward(
        self,
        *,
        node_feat: torch.Tensor,
        node_attr: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch_index: torch.Tensor,
        num_sampled_nodes_per_hop: Optional[list[int]],
        num_sampled_edges_per_hop: Optional[list[int]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply a generator block.

        Parameters
        ----------
        node_feat : torch.Tensor
            Block input features with shape ``[N, F]``.
        node_attr : torch.Tensor
            Raw node attributes with shape ``[N, 2]``.
        edge_index : torch.Tensor
            Directed edge tensor with shape ``[2, E]``.
        edge_attr : torch.Tensor
            Edge attributes with shape ``[E, A]``.
        batch_index : torch.Tensor
            Batch vector with shape ``[N]``.
        num_sampled_nodes_per_hop : list[int], optional
            Unused sampled-neighborhood metadata.
        num_sampled_edges_per_hop : list[int], optional
            Unused sampled-neighborhood metadata.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            Output features, raw attributes, edge index, and edge attributes.
        """
        init_edge_index = edge_index
        first_layer_edge_feat = self.first_layer_feature_router(
            block_input=node_feat,
            raw_input=node_attr,
            edge_attr=edge_attr,
            edge_index=init_edge_index,
        )
        rest_layers_edge_feat = self.rest_layers_feature_router(
            block_input=node_feat,
            raw_input=node_attr,
            edge_attr=edge_attr,
            edge_index=init_edge_index,
        )
        outputs, edge_index, _ = self.first_layer(
            node_feat=node_feat,
            edge_feat=first_layer_edge_feat,
            edge_index=init_edge_index,
            batch_index=batch_index,
            num_sampled_nodes_per_hop=num_sampled_nodes_per_hop,
            num_sampled_edges_per_hop=num_sampled_edges_per_hop,
        )
        for layer in self.rest_layers:
            outputs, edge_index, rest_layers_edge_feat = layer(
                node_feat=outputs,
                edge_feat=rest_layers_edge_feat,
                edge_index=edge_index,
                batch_index=batch_index,
                num_sampled_nodes_per_hop=num_sampled_nodes_per_hop,
                num_sampled_edges_per_hop=num_sampled_edges_per_hop,
            )
        if self.residual:
            outputs = self.skip(block_input=node_feat, block_output=outputs)
        return outputs, node_attr, init_edge_index, edge_attr

    def __len__(self) -> int:
        """Return contained layer count.

        Returns
        -------
        int
            Number of generator layers.
        """
        return 1 + len(self.rest_layers)

    @property
    def next_layer_index(self) -> int:
        """Return the next global layer index.

        Returns
        -------
        int
            Next layer index after this block.
        """
        return self.first_layer.gnn_layer.nnconv_layer.layer_index + len(self)


class SmartGDGenerator(nn.Module):
    """Dagua port of the SmartGD/DeepGD generator architecture."""

    def __init__(self, config: SmartGDConfig) -> None:
        """Initialize the generator.

        Parameters
        ----------
        config : SmartGDConfig
            Generator architecture configuration.
        """
        super().__init__()
        self.block_list = nn.ModuleList()
        expansion = EdgeFeatureExpansions(unit_vec=True, vec_norm=True)
        self.block_list.append(
            GeneratorBlock(
                start_layer_index=0,
                in_dim=config.node_attr_dim,
                hidden_dims=[],
                out_dim=config.block_output_dim,
                edge_attr_dim=config.edge_attr_dim,
                node_attr_dim=config.node_attr_dim,
                dynamic_edge_feat_mode=None,
                residual=False,
                edge_net_width=config.edge_net_width,
                edge_net_depth=config.edge_net_depth,
                edge_feat_expansion=EdgeFeatureExpansions(),
            )
        )
        for _ in range(config.num_blocks):
            self.block_list.append(
                GeneratorBlock(
                    start_layer_index=self.block_list[-1].next_layer_index,
                    in_dim=config.block_output_dim,
                    hidden_dims=[config.block_width] * config.block_depth,
                    out_dim=config.block_output_dim,
                    edge_attr_dim=config.edge_attr_dim,
                    node_attr_dim=config.node_attr_dim,
                    dynamic_edge_feat_mode="jump",
                    residual=True,
                    edge_net_width=config.edge_net_width,
                    edge_net_depth=config.edge_net_depth,
                    edge_feat_expansion=expansion,
                )
            )
        self.block_list.append(
            GeneratorBlock(
                start_layer_index=self.block_list[-1].next_layer_index,
                in_dim=config.block_output_dim,
                hidden_dims=[],
                out_dim=2,
                edge_attr_dim=config.edge_attr_dim,
                node_attr_dim=config.node_attr_dim,
                dynamic_edge_feat_mode=None,
                residual=False,
                edge_net_width=config.edge_net_width,
                edge_net_depth=config.edge_net_depth,
                edge_feat_expansion=EdgeFeatureExpansions(),
                gnn_bn=None,
                gnn_act=None,
                gnn_dp=0.0,
                root_weight=True,
            )
        )

    def forward(
        self,
        init_pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch_index: torch.Tensor,
        num_sampled_nodes_per_hop: Optional[list[int]] = None,
        num_sampled_edges_per_hop: Optional[list[int]] = None,
    ) -> torch.Tensor:
        """Run the generator forward pass.

        Parameters
        ----------
        init_pos : torch.Tensor
            Initial positions with shape ``[N, 2]``.
        edge_index : torch.Tensor
            Directed all-pairs edge tensor with shape ``[2, E]``.
        edge_attr : torch.Tensor
            Edge attributes with shape ``[E, 2]``.
        batch_index : torch.Tensor
            Batch vector with shape ``[N]``.
        num_sampled_nodes_per_hop : list[int], optional
            Unused sampled-neighborhood metadata.
        num_sampled_edges_per_hop : list[int], optional
            Unused sampled-neighborhood metadata.

        Returns
        -------
        torch.Tensor
            Predicted positions with shape ``[N, 2]``.
        """
        inputs = outputs = init_pos
        for block in self.block_list:
            outputs, inputs, edge_index, edge_attr = block(
                node_feat=outputs,
                node_attr=inputs,
                edge_index=edge_index,
                edge_attr=edge_attr,
                batch_index=batch_index,
                num_sampled_nodes_per_hop=num_sampled_nodes_per_hop,
                num_sampled_edges_per_hop=num_sampled_edges_per_hop,
            )
        return outputs

    @property
    def total_layers(self) -> int:
        """Return total contained GNN layer count.

        Returns
        -------
        int
            Total layer count.
        """
        return self.block_list[-1].next_layer_index


def _activation(name: Optional[str]) -> nn.Module:
    """Resolve a reference activation module.

    Parameters
    ----------
    name : str, optional
        Activation name.

    Returns
    -------
    torch.nn.Module
        Activation module.
    """
    if name == "leaky_relu":
        return nn.LeakyReLU()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    return nn.Identity()


def _resolve_smartgd_config(
    config: Optional[LayoutConfig],
    *,
    seed: Optional[int] = None,
    **kwargs: object,
) -> SmartGDConfig:
    """Resolve public algorithm parameters into ``SmartGDConfig``.

    Parameters
    ----------
    config : LayoutConfig, optional
        Public layout configuration.
    seed : int, optional
        Explicit seed override.
    **kwargs : object
        Algorithm parameter overrides.

    Returns
    -------
    SmartGDConfig
        Resolved configuration.
    """
    params: dict[str, object] = {}
    if config is not None:
        params.update(getattr(config, "algorithm_params", {}) or {})
        if getattr(config, "seed", None) is not None:
            params["seed"] = config.seed
    params.update(kwargs)
    if seed is not None:
        params["seed"] = seed
    valid = SmartGDConfig.__dataclass_fields__
    filtered = {key: value for key, value in params.items() if key in valid}
    return SmartGDConfig(**filtered)


def _checkpoint_for_config(config: SmartGDConfig) -> Optional[Path]:
    """Resolve the local SmartGD reference checkpoint path.

    Parameters
    ----------
    config : SmartGDConfig
        Pipeline configuration.

    Returns
    -------
    pathlib.Path or None
        Checkpoint path when configured and available.
    """
    if config.checkpoint_path:
        return Path(config.checkpoint_path)
    if not config.use_reference_checkpoint:
        return None
    name = (
        "generator_xing_only.pt" if config.objective == "crossings" else "generator_stress_only.pt"
    )
    candidate = Path("/tmp/smartgd-ref") / name
    return candidate if candidate.exists() else None


def build_smartgd_model(config: SmartGDConfig) -> SmartGDGenerator:
    """Build a SmartGD generator model.

    Parameters
    ----------
    config : SmartGDConfig
        Model configuration.

    Returns
    -------
    SmartGDGenerator
        Generator model.
    """
    return SmartGDGenerator(config)


def _all_pairs_shortest_paths(
    edge_index: torch.Tensor,
    num_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute unweighted all-pairs shortest-path distances.

    Parameters
    ----------
    edge_index : torch.Tensor
        Input graph edges with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    device : torch.device
        Target device for the result.

    Returns
    -------
    torch.Tensor
        Dense distance matrix with shape ``[N, N]``.
    """
    dist = torch.full((num_nodes, num_nodes), float(num_nodes + 1), device=device)
    dist.fill_diagonal_(0.0)
    if edge_index.numel() > 0:
        rows = edge_index[0].to(device=device, dtype=torch.long)
        cols = edge_index[1].to(device=device, dtype=torch.long)
        dist[rows, cols] = 1.0
        dist[cols, rows] = 1.0
    for pivot in range(num_nodes):
        dist = torch.minimum(dist, dist[:, pivot : pivot + 1] + dist[pivot : pivot + 1, :])
    return dist


def prepare_smartgd_data(
    edge_index: torch.Tensor,
    num_nodes: int,
    init_pos: Optional[torch.Tensor],
    *,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare reference-compatible SmartGD generator inputs.

    Parameters
    ----------
    edge_index : torch.Tensor
        Input graph edges with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    init_pos : torch.Tensor, optional
        Optional warm-start positions with shape ``[N, 2]``.
    device : torch.device
        Target device.
    seed : int
        Deterministic seed for fallback initialization.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        Initial positions, all-pairs directed edge index, edge attributes, and
        batch vector.
    """
    if num_nodes <= 0:
        empty_pos = torch.empty((0, 2), device=device)
        empty_edges = torch.empty((2, 0), dtype=torch.long, device=device)
        empty_attr = torch.empty((0, 2), device=device)
        empty_batch = torch.empty((0,), dtype=torch.long, device=device)
        return empty_pos, empty_edges, empty_attr, empty_batch
    if init_pos is None:
        stress_config = LayoutConfig(algorithm="native_stress", seed=seed, steps=100)
        init_pos = layout_native_stress_pipeline(
            edge_index=edge_index,
            num_nodes=num_nodes,
            config=stress_config,
            seed=seed,
            steps=100,
        )
    pos = init_pos.to(device=device, dtype=torch.float32)
    pair_list = list(permutations(range(num_nodes), 2))
    perm_index = torch.tensor(pair_list, dtype=torch.long, device=device).t().contiguous()
    dist = _all_pairs_shortest_paths(edge_index, num_nodes, device)
    apsp = dist[perm_index[0], perm_index[1]].clamp_min(1.0)
    edge_attr = torch.cat((apsp[:, None], 1.0 / apsp[:, None].square()), dim=1)
    batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
    return _rescale_by_stress(pos, apsp, perm_index, batch), perm_index, edge_attr, batch


def _rescale_by_stress(
    pos: torch.Tensor,
    apsp: torch.Tensor,
    edge_index: torch.Tensor,
    batch_index: torch.Tensor,
) -> torch.Tensor:
    """Apply the reference stress-based coordinate scaling.

    Parameters
    ----------
    pos : torch.Tensor
        Positions with shape ``[N, 2]``.
    apsp : torch.Tensor
        Pairwise shortest path distances with shape ``[E]``.
    edge_index : torch.Tensor
        Directed all-pairs edge tensor with shape ``[2, E]``.
    batch_index : torch.Tensor
        Batch vector with shape ``[N]``.

    Returns
    -------
    torch.Tensor
        Rescaled positions with shape ``[N, 2]``.
    """
    from torch_geometric.utils import scatter

    if pos.numel() == 0 or edge_index.numel() == 0:
        return pos
    src_pos, dst_pos = pos[edge_index[0]], pos[edge_index[1]]
    dist = (dst_pos - src_pos).norm(dim=1)
    u_over_d = dist / apsp.clamp_min(_DEFAULT_EPS)
    scatter_sq = scatter(u_over_d**2, batch_index[edge_index[0]], dim=0)
    scatter_linear = scatter(u_over_d, batch_index[edge_index[0]], dim=0).clamp_min(_DEFAULT_EPS)
    scale = scatter_sq / scatter_linear
    return pos / scale[batch_index][:, None].clamp_min(_DEFAULT_EPS)


def smartgd_reference_forward(
    edge_index: torch.Tensor,
    num_nodes: int,
    config: SmartGDConfig,
    *,
    init_pos: Optional[torch.Tensor] = None,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Run the Dagua SmartGD generator forward pass.

    Parameters
    ----------
    edge_index : torch.Tensor
        Input graph edges with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    config : SmartGDConfig
        Generator configuration.
    init_pos : torch.Tensor, optional
        Optional warm-start positions with shape ``[N, 2]``.
    device : str or torch.device, default="cpu"
        Target device.

    Returns
    -------
    torch.Tensor
        Predicted positions with shape ``[N, 2]``.
    """
    resolved_device = torch.device(device)
    torch.manual_seed(int(config.seed))
    model = build_smartgd_model(config).to(resolved_device)
    checkpoint = _checkpoint_for_config(config)
    if checkpoint is not None and checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, map_location=resolved_device))
    model.eval()
    prepared = prepare_smartgd_data(
        edge_index=edge_index,
        num_nodes=num_nodes,
        init_pos=init_pos,
        device=resolved_device,
        seed=int(config.seed),
    )
    with torch.no_grad():
        pred = model(*prepared)
    return _rescale_by_stress(pred, prepared[2][:, 0], prepared[1], prepared[3])


def layout_smartgd_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    *,
    config: Optional[LayoutConfig] = None,
    seed: Optional[int] = None,
    init_pos: Optional[torch.Tensor] = None,
    device: str | torch.device = "cpu",
    **kwargs: object,
) -> torch.Tensor:
    """Compute a layout with the SmartGD generator.

    Parameters
    ----------
    edge_index : torch.Tensor
        Input graph edges with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    node_sizes : torch.Tensor, optional
        Node sizes with shape ``[N, 2]``. Unused by SmartGD.
    config : LayoutConfig, optional
        Public layout configuration.
    seed : int, optional
        Seed override.
    init_pos : torch.Tensor, optional
        Optional warm-start positions with shape ``[N, 2]``.
    device : str or torch.device, default="cpu"
        Target device.
    **kwargs : object
        Algorithm parameter overrides.

    Returns
    -------
    torch.Tensor
        Position tensor with shape ``[N, 2]``.
    """
    del node_sizes
    resolved = _resolve_smartgd_config(config, seed=seed, **kwargs)
    return smartgd_reference_forward(
        edge_index=edge_index,
        num_nodes=num_nodes,
        config=resolved,
        init_pos=init_pos,
        device=device,
    )
