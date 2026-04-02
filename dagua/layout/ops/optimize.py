"""Optimization operations for composable layout pipelines.

This module provides generic optimizer lifecycle ops plus classic-algorithm
update rules that do not map cleanly onto ``torch.optim``. Algorithm-specific
transient state lives in ``SolveState.extras`` using documented keys.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, cast

import numpy as np
import torch

from dagua.layout.classic import kk as kk_classic
from dagua.layout.ops.base import Op
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op

_DEFAULT_TSNE_EARLY_STEPS = 250
_OPTIMIZER_KEY_PREFIX = "optimizer_"
_UMAP_DEFAULT_GAMMA = 1.0


def _optimizer_storage_key(key: str) -> str:
    """Build the extras key used for a named optimizer.

    Parameters
    ----------
    key : str
        User-facing optimizer key.

    Returns
    -------
    str
        Extras dictionary key used for non-default optimizers.
    """
    return f"{_OPTIMIZER_KEY_PREFIX}{key}"


def _store_target_tensor(state: SolveState, target: str, tensor: torch.Tensor) -> None:
    """Write an optimized tensor back to the solve state.

    Parameters
    ----------
    state : SolveState
        Mutable solve state.
    target : str
        Target specifier. Supported values are ``"pos"`` and ``"extras.X"``.
    tensor : torch.Tensor
        Tensor to store.

    Returns
    -------
    None
        The state is updated in place.

    Raises
    ------
    ValueError
        If ``target`` is not supported.
    """
    if target == "pos":
        state.pos = tensor
        return
    if target.startswith("extras.") and len(target) > len("extras."):
        state.extras[target.split(".", 1)[1]] = tensor
        return
    raise ValueError(f"Unsupported optimizer target '{target}'.")


def _resolve_target_tensor(state: SolveState, target: str) -> torch.Tensor:
    """Resolve an optimizer target from the solve state.

    Parameters
    ----------
    state : SolveState
        Mutable solve state.
    target : str
        Target specifier. Supported values are ``"pos"`` and ``"extras.X"``.

    Returns
    -------
    torch.Tensor
        Target tensor.

    Raises
    ------
    ValueError
        If the target is malformed or missing.
    TypeError
        If the resolved extras value is not a tensor.
    """
    if target == "pos":
        if state.pos is None:
            raise ValueError("CreateOptimizer target 'pos' requires state.pos to be set.")
        return state.pos
    if not target.startswith("extras.") or len(target) <= len("extras."):
        raise ValueError(f"Unsupported optimizer target '{target}'.")

    extras_key = target.split(".", 1)[1]
    if extras_key not in state.extras:
        raise ValueError(f"CreateOptimizer target '{target}' is missing from state.extras.")
    tensor = state.extras[extras_key]
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"CreateOptimizer target '{target}' must resolve to a torch.Tensor.")
    return tensor


def _prepare_optimizable_target(state: SolveState, target: str) -> torch.Tensor:
    """Ensure the optimizer target is a leaf floating tensor with gradients.

    Parameters
    ----------
    state : SolveState
        Mutable solve state.
    target : str
        Target specifier. Supported values are ``"pos"`` and ``"extras.X"``.

    Returns
    -------
    torch.Tensor
        Tensor safe to pass to ``torch.optim``.

    Raises
    ------
    TypeError
        If the resolved tensor does not use a floating-point dtype.
    """
    tensor = _resolve_target_tensor(state, target)
    if not tensor.is_floating_point():
        raise TypeError(f"Optimizer target '{target}' must use a floating-point dtype.")
    if tensor.is_leaf and tensor.requires_grad:
        return tensor

    prepared = tensor.detach().clone().requires_grad_(True)
    _store_target_tensor(state, target, prepared)
    return prepared


def _create_torch_optimizer(
    optimizer_type: str,
    parameter: torch.Tensor,
    lr: float,
) -> torch.optim.Optimizer:
    """Instantiate a supported PyTorch optimizer.

    Parameters
    ----------
    optimizer_type : str
        Supported optimizer name.
    parameter : torch.Tensor
        Learnable tensor to optimize.
    lr : float
        Optimizer learning rate.

    Returns
    -------
    torch.optim.Optimizer
        Configured optimizer.

    Raises
    ------
    ValueError
        If the optimizer name is unsupported or the learning rate is invalid.
    """
    if lr <= 0.0:
        raise ValueError("Optimizer learning rate must be positive.")
    if optimizer_type == "adam":
        return torch.optim.Adam([parameter], lr=lr)
    if optimizer_type == "sgd":
        return torch.optim.SGD([parameter], lr=lr)
    if optimizer_type == "sgd_nesterov":
        return torch.optim.SGD([parameter], lr=lr, momentum=0.9, nesterov=True)
    if optimizer_type == "rmsprop":
        return torch.optim.RMSprop([parameter], lr=lr)
    raise ValueError(f"Unsupported optimizer type '{optimizer_type}'.")


def _load_optimizer(state: SolveState, key: str) -> torch.optim.Optimizer:
    """Load an optimizer from the solve state.

    Parameters
    ----------
    state : SolveState
        Mutable solve state.
    key : str
        User-facing optimizer key.

    Returns
    -------
    torch.optim.Optimizer
        Stored optimizer instance.

    Raises
    ------
    ValueError
        If the optimizer is missing.
    TypeError
        If the stored value is not a PyTorch optimizer.
    """
    raw_optimizer: Any
    if key == "default":
        raw_optimizer = state.optimizer
    else:
        raw_optimizer = state.extras.get(_optimizer_storage_key(key))
    if raw_optimizer is None:
        raise ValueError(f"Optimizer '{key}' has not been created.")
    if not isinstance(raw_optimizer, torch.optim.Optimizer):
        raise TypeError(f"Stored optimizer '{key}' is not a torch.optim.Optimizer.")
    return raw_optimizer


def _parameters_for_clipping(state: SolveState) -> Sequence[torch.Tensor]:
    """Resolve the parameter sequence used by gradient-clipping ops.

    Parameters
    ----------
    state : SolveState
        Mutable solve state.

    Returns
    -------
    sequence[torch.Tensor]
        Parameters to clip. The default optimizer takes priority because it
        carries the authoritative parameter set when present.
    """
    if isinstance(state.optimizer, torch.optim.Optimizer):
        return [
            cast(torch.Tensor, parameter)
            for group in state.optimizer.param_groups
            for parameter in group["params"]
        ]
    if state.pos is not None:
        return [state.pos]
    return []


def _resolve_tsne_learning_rate(rule: str, num_nodes: int) -> float:
    """Resolve the t-SNE learning-rate rule used by the classic update.

    Parameters
    ----------
    rule : str
        Supported rule string. ``"N/48"`` matches the classic tsNET port.
        Numeric strings are interpreted as explicit learning rates.
    num_nodes : int
        Number of points in the embedding.

    Returns
    -------
    float
        Learning rate for the current step.

    Raises
    ------
    ValueError
        If the rule is unsupported.
    """
    if rule == "N/48":
        return max(float(max(num_nodes, 1)) / 48.0, 50.0)
    try:
        learning_rate = float(rule)
    except ValueError as error:
        raise ValueError(f"Unsupported t-SNE learning-rate rule '{rule}'.") from error
    if learning_rate <= 0.0:
        raise ValueError("t-SNE learning rate must be positive.")
    return learning_rate


def _get_or_create_torch_generator(
    state: SolveState,
    ctx: RuntimeContext,
    seed: int,
) -> torch.Generator:
    """Return the torch RNG backend used by UMAP negative sampling.

    Parameters
    ----------
    state : SolveState
        Mutable solve state.
    ctx : RuntimeContext
        Runtime context that may already provide a generator.
    seed : int
        Fallback seed used when neither ``ctx.generator`` nor an existing
        stored generator is available.

    Returns
    -------
    torch.Generator
        CPU generator whose state persists across op calls.
    """
    if ctx.generator is not None:
        return ctx.generator

    stored = state.extras.get("umap_generator")
    if isinstance(stored, torch.Generator):
        return stored

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    state.extras["umap_generator"] = generator
    return generator


def _umap_positive_gradient(
    diff: torch.Tensor,
    distance_sq: float,
    a: float,
    b: float,
    clip: float,
) -> torch.Tensor:
    """Compute the clipped attractive UMAP gradient for one positive edge.

    Parameters
    ----------
    diff : torch.Tensor
        Coordinate difference ``y_i - y_j`` with shape ``[2]``.
    distance_sq : float
        Squared Euclidean distance between the endpoints.
    a : float
        UMAP curve parameter.
    b : float
        UMAP curve parameter.
    clip : float
        Symmetric gradient clamp applied per component.

    Returns
    -------
    torch.Tensor
        Gradient contribution with shape ``[2]``.
    """
    if distance_sq <= 0.0:
        return torch.zeros_like(diff)
    grad_coeff = -2.0 * a * b * (distance_sq ** (b - 1.0)) / ((a * (distance_sq**b)) + 1.0)
    return torch.clamp(grad_coeff * diff, min=-clip, max=clip)


def _umap_negative_gradient(
    diff: torch.Tensor,
    distance_sq: float,
    a: float,
    b: float,
    gamma: float,
    clip: float,
) -> torch.Tensor:
    """Compute the clipped repulsive UMAP gradient for one negative sample.

    Parameters
    ----------
    diff : torch.Tensor
        Coordinate difference ``y_i - y_k`` with shape ``[2]``.
    distance_sq : float
        Squared Euclidean distance between the sampled pair.
    a : float
        UMAP curve parameter.
    b : float
        UMAP curve parameter.
    gamma : float
        Negative-sample repulsion strength.
    clip : float
        Symmetric gradient clamp applied per component.

    Returns
    -------
    torch.Tensor
        Gradient contribution with shape ``[2]``.
    """
    if distance_sq <= 0.0:
        return torch.zeros_like(diff)
    grad_coeff = 2.0 * gamma * b / ((0.001 + distance_sq) * ((a * (distance_sq**b)) + 1.0))
    return torch.clamp(grad_coeff * diff, min=-clip, max=clip)


@dataclass(frozen=True)
class CreateOptimizerConfig:
    """Configuration for :class:`CreateOptimizer`.

    Attributes
    ----------
    optimizer_type : str, default="adam"
        Optimizer name. Supported values are ``"adam"``, ``"sgd"``,
        ``"sgd_nesterov"``, and ``"rmsprop"``.
    lr : float, default=0.05
        Learning rate passed to the optimizer constructor.
    target : str, default="pos"
        Tensor target. Supported values are ``"pos"`` and ``"extras.X"``.
    key : str, default="default"
        Storage key. ``"default"`` writes to ``state.optimizer``; any other
        key writes to ``state.extras["optimizer_<key>"]``.
    """

    optimizer_type: str = "adam"
    lr: float = 0.05
    target: str = "pos"
    key: str = "default"


@register_op
class CreateOptimizer(Op):
    """Create and store a PyTorch optimizer.

    Randomness
    ----------
    This op does not use randomness.
    """

    name = "CreateOptimizer"
    category = OpCategory.OPTIMIZE
    reads = ("pos", "extras", "optimizer")
    writes = ("pos", "extras", "optimizer")
    requires = ()

    def __init__(self, config: Optional[CreateOptimizerConfig] = None) -> None:
        """Store the frozen configuration for this op.

        Parameters
        ----------
        config : CreateOptimizerConfig, optional
            Optimizer configuration. When omitted, defaults are used.

        Returns
        -------
        None
            This constructor stores configuration only.
        """
        self.config = config or CreateOptimizerConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Create the requested optimizer and store it on the state.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused beyond signature compatibility.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution context. Unused by this op.

        Returns
        -------
        SolveState
            State with the optimizer attached.
        """
        del problem, ctx

        target_tensor = _prepare_optimizable_target(state, self.config.target)
        optimizer = _create_torch_optimizer(
            optimizer_type=self.config.optimizer_type,
            parameter=target_tensor,
            lr=self.config.lr,
        )
        if self.config.key == "default":
            state.optimizer = optimizer
        else:
            state.extras[_optimizer_storage_key(self.config.key)] = optimizer
        return state


@dataclass(frozen=True)
class OptimizerStepConfig:
    """Configuration for :class:`OptimizerStep`.

    Attributes
    ----------
    key : str, default="default"
        Optimizer storage key to step.
    """

    key: str = "default"


@register_op
class OptimizerStep(Op):
    """Apply one ``optimizer.step()`` call to a stored optimizer.

    Randomness
    ----------
    This op does not use randomness.
    """

    name = "OptimizerStep"
    category = OpCategory.OPTIMIZE
    reads = ("optimizer", "extras")
    writes = ("pos", "extras", "optimizer")
    requires = ("optimizer",)

    def __init__(self, config: Optional[OptimizerStepConfig] = None) -> None:
        """Store the frozen configuration for this op.

        Parameters
        ----------
        config : OptimizerStepConfig, optional
            Step configuration. When omitted, defaults are used.

        Returns
        -------
        None
            This constructor stores configuration only.
        """
        self.config = config or OptimizerStepConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Step the named optimizer.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused by this op.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution context. Unused by this op.

        Returns
        -------
        SolveState
            State after the optimizer step.
        """
        del problem, ctx

        optimizer = _load_optimizer(state, self.config.key)
        optimizer.step()
        return state


@dataclass(frozen=True)
class ClipGradNormConfig:
    """Configuration for :class:`ClipGradNorm`.

    Attributes
    ----------
    max_norm : float, default=100.0
        Maximum allowed total gradient norm.
    """

    max_norm: float = 100.0


@register_op
class ClipGradNorm(Op):
    """Clip gradient norms on the default optimizer parameter set.

    Randomness
    ----------
    This op does not use randomness.
    """

    name = "ClipGradNorm"
    category = OpCategory.OPTIMIZE
    reads = ("optimizer", "pos")
    writes = ("pos",)
    requires = ()

    def __init__(self, config: Optional[ClipGradNormConfig] = None) -> None:
        """Store the frozen configuration for this op.

        Parameters
        ----------
        config : ClipGradNormConfig, optional
            Clipping configuration. When omitted, defaults are used.

        Returns
        -------
        None
            This constructor stores configuration only.
        """
        self.config = config or ClipGradNormConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Clip the gradient norm of the active parameter set.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused by this op.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution context. Unused by this op.

        Returns
        -------
        SolveState
            State with clipped gradients.
        """
        del problem, ctx

        if self.config.max_norm <= 0.0:
            raise ValueError("ClipGradNorm max_norm must be positive.")
        parameters = _parameters_for_clipping(state)
        if not parameters:
            return state
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=self.config.max_norm)
        return state


@dataclass(frozen=True)
class ClipGradValueConfig:
    """Configuration for :class:`ClipGradValue`.

    Attributes
    ----------
    max_value : float, default=4.0
        Symmetric gradient clamp applied element-wise.
    """

    max_value: float = 4.0


@register_op
class ClipGradValue(Op):
    """Clamp gradients element-wise on the default optimizer parameter set.

    Randomness
    ----------
    This op does not use randomness.
    """

    name = "ClipGradValue"
    category = OpCategory.OPTIMIZE
    reads = ("optimizer", "pos")
    writes = ("pos",)
    requires = ()

    def __init__(self, config: Optional[ClipGradValueConfig] = None) -> None:
        """Store the frozen configuration for this op.

        Parameters
        ----------
        config : ClipGradValueConfig, optional
            Clipping configuration. When omitted, defaults are used.

        Returns
        -------
        None
            This constructor stores configuration only.
        """
        self.config = config or ClipGradValueConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Clamp gradient values on the active parameter set.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused by this op.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution context. Unused by this op.

        Returns
        -------
        SolveState
            State with clipped gradients.
        """
        del problem, ctx

        if self.config.max_value <= 0.0:
            raise ValueError("ClipGradValue max_value must be positive.")
        parameters = _parameters_for_clipping(state)
        if not parameters:
            return state
        torch.nn.utils.clip_grad_value_(parameters, clip_value=self.config.max_value)
        return state


@dataclass(frozen=True)
class LBFGSStepConfig:
    """Configuration for :class:`LBFGSStep`.

    Attributes
    ----------
    maxiter : int or None, default=None
        Maximum SciPy L-BFGS-B iterations. ``None`` and ``0`` leave SciPy's
        default solve budget unchanged, matching the classic KK port.
    """

    maxiter: Optional[int] = None


@register_op
class LBFGSStep(Op):
    """Run SciPy's L-BFGS-B update for Kamada-Kawai refinement.

    Notes
    -----
    This op mirrors the classic KK solver path in
    :mod:`dagua.layout.classic.kk`. It requires ``state.distance_matrix`` and
    ``state.pos``. SciPy's internal line search is not externally seeded.
    """

    name = "LBFGSStep"
    category = OpCategory.OPTIMIZE
    reads = ("pos", "distance_matrix")
    writes = ("pos", "prev_loss")
    requires = ("pos", "distance_matrix")

    def __init__(self, config: Optional[LBFGSStepConfig] = None) -> None:
        """Store the frozen configuration for this op.

        Parameters
        ----------
        config : LBFGSStepConfig, optional
            Solver configuration. When omitted, defaults are used.

        Returns
        -------
        None
            This constructor stores configuration only.
        """
        self.config = config or LBFGSStepConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run one full SciPy L-BFGS-B solve on the current position tensor.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused by this op.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution context. Unused by this op.

        Returns
        -------
        SolveState
            State with optimized positions and the latest objective value.
        """
        del problem, ctx

        if state.pos is None:
            raise ValueError("LBFGSStep requires state.pos to be set.")
        if state.distance_matrix is None:
            raise ValueError("LBFGSStep requires state.distance_matrix to be set.")
        if self.config.maxiter is not None and self.config.maxiter < 0:
            raise ValueError("LBFGSStep maxiter must be non-negative or None.")
        if state.pos.numel() == 0:
            state.prev_loss = 0.0
            return state

        try:
            import scipy as sp
        except ImportError as error:
            raise ImportError("LBFGSStep requires scipy.") from error

        original_pos = state.pos
        dim = int(original_pos.shape[1]) if original_pos.ndim == 2 else 2
        initial_positions = (
            original_pos.detach().to(device="cpu", dtype=torch.float64).numpy().copy()
        )
        distance_matrix = (
            state.distance_matrix.detach().to(device="cpu", dtype=torch.float64).numpy()
        )
        inverse_distances = 1.0 / (
            distance_matrix
            + np.eye(distance_matrix.shape[0], dtype=np.float64) * kk_classic.DISTANCE_EPSILON
        )

        minimize_kwargs: dict[str, Any] = {
            "method": "L-BFGS-B",
            "args": (np, inverse_distances, kk_classic.CENTERING_WEIGHT, dim),
            "jac": True,
        }
        if self.config.maxiter not in {None, 0}:
            minimize_kwargs["options"] = {"maxiter": self.config.maxiter}

        result = sp.optimize.minimize(
            kk_classic._kamada_kawai_costfn,
            initial_positions.ravel(),
            **minimize_kwargs,
        )
        optimized = torch.from_numpy(result.x.reshape((-1, dim))).to(
            device=original_pos.device,
            dtype=original_pos.dtype,
        )
        if original_pos.requires_grad:
            optimized = optimized.detach().clone().requires_grad_(True)
        state.pos = optimized
        state.prev_loss = float(result.fun)
        return state


@dataclass(frozen=True)
class TSNEGainsMomentumStepConfig:
    """Configuration for :class:`TSNEGainsMomentumStep`.

    Attributes
    ----------
    lr_rule : str, default="N/48"
        Learning-rate rule. ``"N/48"`` matches the classic tsNET port and
        resolves to ``max(N / 48, 50)``.
    min_gain : float, default=0.01
        Minimum per-parameter gain.
    momentum_early : float, default=0.5
        Momentum used before the early-exaggeration phase ends.
    momentum_late : float, default=0.8
        Momentum used afterward.
    """

    lr_rule: str = "N/48"
    min_gain: float = 0.01
    momentum_early: float = 0.5
    momentum_late: float = 0.8


@register_op
class TSNEGainsMomentumStep(Op):
    """Apply the classic t-SNE gains-plus-momentum update to ``state.pos``.

    Notes
    -----
    The op uses ``state.pos.grad`` as the current gradient and persists the
    classic auxiliary tensors in ``state.extras["tsne_update"]`` and
    ``state.extras["tsne_gains"]``. ``state.extras["tsne_early_exaggeration_steps"]``
    may override the default phase boundary of 250 steps.
    """

    name = "TSNEGainsMomentumStep"
    category = OpCategory.OPTIMIZE
    reads = ("pos", "step", "extras")
    writes = ("pos", "extras")
    requires = ("pos",)

    def __init__(self, config: Optional[TSNEGainsMomentumStepConfig] = None) -> None:
        """Store the frozen configuration for this op.

        Parameters
        ----------
        config : TSNEGainsMomentumStepConfig, optional
            Update configuration. When omitted, defaults are used.

        Returns
        -------
        None
            This constructor stores configuration only.
        """
        self.config = config or TSNEGainsMomentumStepConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Apply one t-SNE custom optimizer step.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused by this op.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution context. Unused by this op.

        Returns
        -------
        SolveState
            State with updated positions and persisted gains/momentum buffers.
        """
        del problem, ctx

        if state.pos is None:
            raise ValueError("TSNEGainsMomentumStep requires state.pos to be set.")
        if state.pos.grad is None:
            return state
        if self.config.min_gain <= 0.0:
            raise ValueError("TSNEGainsMomentumStep min_gain must be positive.")

        grad = state.pos.grad.detach().clone()
        update = state.extras.get("tsne_update")
        if not isinstance(update, torch.Tensor) or tuple(update.shape) != tuple(state.pos.shape):
            update = torch.zeros_like(state.pos)
        gains = state.extras.get("tsne_gains")
        if not isinstance(gains, torch.Tensor) or tuple(gains.shape) != tuple(state.pos.shape):
            gains = torch.ones_like(state.pos)

        learning_rate = _resolve_tsne_learning_rate(self.config.lr_rule, state.pos.shape[0])
        early_steps = int(
            state.extras.get("tsne_early_exaggeration_steps", _DEFAULT_TSNE_EARLY_STEPS)
        )
        momentum = (
            self.config.momentum_early if state.step < early_steps else self.config.momentum_late
        )

        with torch.no_grad():
            increasing = (update * grad) < 0.0
            decreasing = ~increasing
            gains[increasing] += 0.2
            gains[decreasing] *= 0.8
            gains.clamp_(min=self.config.min_gain)
            grad = grad * gains
            update = momentum * update - learning_rate * grad
            state.pos.add_(update)
            state.pos.grad.zero_()

        state.extras["tsne_update"] = update
        state.extras["tsne_gains"] = gains
        return state


@dataclass(frozen=True)
class UMAPPairSGDConfig:
    """Configuration for :class:`UMAPPairSGD`.

    Attributes
    ----------
    neg_rate : int, default=5
        Number of negative samples per positive edge update.
    clip : float, default=4.0
        Symmetric gradient clip applied to each coordinate component.
    learning_rate : float, default=1.0
        Initial UMAP SGD learning rate.
    """

    neg_rate: int = 5
    clip: float = 4.0
    learning_rate: float = 1.0


@register_op
class UMAPPairSGD(Op):
    """Run one epoch of UMAP pairwise SGD with negative sampling.

    Notes
    -----
    Required extras keys:

    ``"umap_head"`` : torch.Tensor
        Positive edge heads with shape ``[E]``.
    ``"umap_tail"`` : torch.Tensor
        Positive edge tails with shape ``[E]``.
    ``"umap_epochs_per_sample"`` : torch.Tensor
        Sampling interval per positive edge with shape ``[E]``.
    ``"umap_a"`` / ``"umap_b"`` : float
        UMAP curve parameters.

    Optional extras keys:

    ``"umap_gamma"`` : float
        Negative-sample repulsion strength. Defaults to ``1.0``.
    ``"umap_n_epochs"`` : int
        Total number of epochs. Defaults to ``state.total_steps``.
    ``"umap_seed"`` : int
        Seed used when a local CPU generator must be created.

    Randomness
    ----------
    Negative sampling uses ``ctx.generator`` when provided. Otherwise the op
    creates and persists a private CPU ``torch.Generator`` in
    ``state.extras["umap_generator"]`` seeded from ``problem.seed`` or
    ``state.extras["umap_seed"]``.
    """

    name = "UMAPPairSGD"
    category = OpCategory.OPTIMIZE
    reads = ("pos", "step", "total_steps", "extras")
    writes = ("pos", "extras")
    requires = ("pos", "extras")

    def __init__(self, config: Optional[UMAPPairSGDConfig] = None) -> None:
        """Store the frozen configuration for this op.

        Parameters
        ----------
        config : UMAPPairSGDConfig, optional
            Update configuration. When omitted, defaults are used.

        Returns
        -------
        None
            This constructor stores configuration only.
        """
        self.config = config or UMAPPairSGDConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Apply one epoch of the classic UMAP SGD update.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution context supplying the optional RNG generator.

        Returns
        -------
        SolveState
            State with updated positions and persisted epoch counters.
        """
        if state.pos is None:
            raise ValueError("UMAPPairSGD requires state.pos to be set.")
        if self.config.neg_rate < 0:
            raise ValueError("UMAPPairSGD neg_rate must be non-negative.")
        if self.config.clip <= 0.0:
            raise ValueError("UMAPPairSGD clip must be positive.")
        if self.config.learning_rate <= 0.0:
            raise ValueError("UMAPPairSGD learning_rate must be positive.")

        head = state.extras.get("umap_head")
        tail = state.extras.get("umap_tail")
        epochs_per_sample = state.extras.get("umap_epochs_per_sample")
        if not isinstance(head, torch.Tensor) or not isinstance(tail, torch.Tensor):
            raise ValueError("UMAPPairSGD requires extras['umap_head'] and extras['umap_tail'].")
        if not isinstance(epochs_per_sample, torch.Tensor):
            raise ValueError("UMAPPairSGD requires extras['umap_epochs_per_sample'].")
        if head.numel() == 0:
            return state

        a = float(state.extras.get("umap_a"))
        b = float(state.extras.get("umap_b"))
        gamma = float(state.extras.get("umap_gamma", _UMAP_DEFAULT_GAMMA))
        n_epochs = int(state.extras.get("umap_n_epochs", state.total_steps))
        if n_epochs <= 0:
            return state

        epoch = int(state.step)
        alpha = self.config.learning_rate * (1.0 - (float(epoch) / float(max(n_epochs, 1))))
        if alpha <= 0.0:
            return state

        next_sample_epoch = state.extras.get("umap_next_sample_epoch")
        sample_epoch_matches = isinstance(next_sample_epoch, torch.Tensor) and (
            tuple(next_sample_epoch.shape) == tuple(epochs_per_sample.shape)
        )
        if not sample_epoch_matches:
            next_sample_epoch = torch.zeros_like(epochs_per_sample, dtype=torch.float32)
        epochs_per_negative_sample = epochs_per_sample / float(max(self.config.neg_rate, 1))
        next_negative_epoch = state.extras.get("umap_next_negative_epoch")
        negative_epoch_matches = isinstance(next_negative_epoch, torch.Tensor) and (
            tuple(next_negative_epoch.shape) == tuple(epochs_per_sample.shape)
        )
        if not negative_epoch_matches:
            next_negative_epoch = torch.zeros_like(epochs_per_negative_sample, dtype=torch.float32)

        generator = _get_or_create_torch_generator(
            state=state,
            ctx=ctx,
            seed=int(state.extras.get("umap_seed", problem.seed)),
        )
        num_nodes = int(state.pos.shape[0])

        with torch.no_grad():
            for edge_id in range(int(head.shape[0])):
                if float(next_sample_epoch[edge_id].item()) > float(epoch):
                    continue

                source = int(head[edge_id].item())
                target = int(tail[edge_id].item())
                diff = state.pos[source] - state.pos[target]
                distance_sq = float(torch.dot(diff, diff).item())
                grad = _umap_positive_gradient(diff, distance_sq, a, b, self.config.clip)
                state.pos[source] = state.pos[source] + (alpha * grad)
                state.pos[target] = state.pos[target] - (alpha * grad)
                next_sample_epoch[edge_id] = next_sample_epoch[edge_id] + epochs_per_sample[edge_id]

                if self.config.neg_rate <= 0:
                    continue

                negatives = 0
                while float(next_negative_epoch[edge_id].item()) <= float(epoch):
                    negative = int(torch.randint(0, num_nodes, (1,), generator=generator).item())
                    negative_diff = state.pos[source] - state.pos[negative]
                    negative_distance_sq = float(torch.dot(negative_diff, negative_diff).item())
                    negative_grad = _umap_negative_gradient(
                        negative_diff,
                        negative_distance_sq,
                        a,
                        b,
                        gamma,
                        self.config.clip,
                    )
                    state.pos[source] = state.pos[source] + (alpha * negative_grad)
                    next_negative_epoch[edge_id] = (
                        next_negative_epoch[edge_id] + epochs_per_negative_sample[edge_id]
                    )
                    negatives += 1
                    if negatives >= self.config.neg_rate:
                        break

        state.extras["umap_next_sample_epoch"] = next_sample_epoch
        state.extras["umap_next_negative_epoch"] = next_negative_epoch
        return state
