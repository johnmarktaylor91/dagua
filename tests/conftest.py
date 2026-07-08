"""Shared test fixtures — sample graphs, common assertions."""

from collections.abc import Generator
from typing import Any, Callable, Optional

import pytest
import torch

from dagua.config import LayoutConfig
from dagua.graph import DaguaGraph

try:
    from torchlens import unwrap_torch

    _HAS_TORCHLENS = True
except Exception:  # pragma: no cover - torchlens is optional and may be a broken checkout
    _HAS_TORCHLENS = False


_TorchFactory = Callable[..., torch.Tensor]
_ORIGINAL_TORCH_TENSOR = torch.tensor
_ORIGINAL_TORCH_AS_TENSOR = torch.as_tensor
_ORIGINAL_TORCH_STACK = torch.stack


def _stack_tensor_sequence(data: Any, kwargs: dict[str, Any]) -> Optional[torch.Tensor]:
    """Convert tensor sequences without relying on ``torch.tensor`` recursion.

    Parameters
    ----------
    data : Any
        Candidate tensor sequence passed to a tensor factory.
    kwargs : dict[str, Any]
        Keyword arguments forwarded to the tensor factory.

    Returns
    -------
    Optional[torch.Tensor]
        Stacked tensor when ``data`` is a flat tensor sequence, otherwise ``None``.
    """
    if not isinstance(data, (list, tuple)) or not data:
        return None
    if not all(isinstance(item, torch.Tensor) for item in data):
        return None

    result = _ORIGINAL_TORCH_STACK([item.detach() for item in data])
    dtype = kwargs.get("dtype")
    device = kwargs.get("device")
    if dtype is not None or device is not None:
        result = result.to(device=device, dtype=dtype)
    if kwargs.get("pin_memory", False):
        result = result.pin_memory()
    if kwargs.get("requires_grad", False):
        result = result.requires_grad_(True)
    return result


def _safe_tensor_factory(
    factory: _TorchFactory,
    data: Any,
    *args: Any,
    **kwargs: Any,
) -> torch.Tensor:
    """Dispatch to a safe tensor factory for tensor sequences.

    Parameters
    ----------
    factory : Callable[..., torch.Tensor]
        Original torch factory to call when no workaround is needed.
    data : Any
        Data passed to the tensor factory.
    *args : Any
        Positional arguments for the tensor factory.
    **kwargs : Any
        Keyword arguments for the tensor factory.

    Returns
    -------
    torch.Tensor
        Tensor constructed from ``data``.
    """
    if not args:
        stacked = _stack_tensor_sequence(data, kwargs)
        if stacked is not None:
            return stacked
    return factory(data, *args, **kwargs)


def _safe_torch_tensor(data: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
    """Safe wrapper for ``torch.tensor`` after torchlens decoration.

    Parameters
    ----------
    data : Any
        Data passed to ``torch.tensor``.
    *args : Any
        Positional arguments for ``torch.tensor``.
    **kwargs : Any
        Keyword arguments for ``torch.tensor``.

    Returns
    -------
    torch.Tensor
        Tensor constructed from ``data``.
    """
    return _safe_tensor_factory(_ORIGINAL_TORCH_TENSOR, data, *args, **kwargs)


def _safe_torch_as_tensor(data: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
    """Safe wrapper for ``torch.as_tensor`` after torchlens decoration.

    Parameters
    ----------
    data : Any
        Data passed to ``torch.as_tensor``.
    *args : Any
        Positional arguments for ``torch.as_tensor``.
    **kwargs : Any
        Keyword arguments for ``torch.as_tensor``.

    Returns
    -------
    torch.Tensor
        Tensor constructed from ``data``.
    """
    return _safe_tensor_factory(_ORIGINAL_TORCH_AS_TENSOR, data, *args, **kwargs)


def _restore_torch_tensor_factories() -> None:
    """Restore test-safe tensor factories after torchlens mutates torch globals.

    Returns
    -------
    None
        Torch tensor constructors are rebound in place.
    """
    torch.tensor = _safe_torch_tensor
    torch.as_tensor = _safe_torch_as_tensor


@pytest.fixture(autouse=True)
def _unwrap_torchlens_after_test() -> Generator[None, None, None]:
    """Ensure torchlens decorations don't leak between tests.

    Yields
    ------
    None
        Control returns to the test before torchlens teardown runs.
    """
    yield
    if _HAS_TORCHLENS:
        unwrap_torch()
    _restore_torch_tensor_factories()


@pytest.fixture
def simple_chain():
    """Simple 5-node chain: a→b→c→d→e."""
    return DaguaGraph.from_edge_list(
        [
            ("a", "b"),
            ("b", "c"),
            ("c", "d"),
            ("d", "e"),
        ]
    )


@pytest.fixture
def diamond_graph():
    """Diamond: a→b, a→c, b→d, c→d."""
    return DaguaGraph.from_edge_list(
        [
            ("a", "b"),
            ("a", "c"),
            ("b", "d"),
            ("c", "d"),
        ]
    )


@pytest.fixture
def skip_graph():
    """Chain with skip connection: a→b→c→d, a→d."""
    return DaguaGraph.from_edge_list(
        [
            ("a", "b"),
            ("b", "c"),
            ("c", "d"),
            ("a", "d"),
        ]
    )


@pytest.fixture
def wide_graph():
    """Wide parallel: input→b1,b2,b3,b4→output."""
    return DaguaGraph.from_edge_list(
        [
            ("input", "b1"),
            ("input", "b2"),
            ("input", "b3"),
            ("input", "b4"),
            ("b1", "output"),
            ("b2", "output"),
            ("b3", "output"),
            ("b4", "output"),
        ]
    )


@pytest.fixture
def clustered_graph():
    """Graph with clusters."""
    g = DaguaGraph.from_edge_list(
        [
            ("input", "enc1"),
            ("enc1", "enc2"),
            ("enc2", "dec1"),
            ("dec1", "dec2"),
            ("dec2", "output"),
        ]
    )
    g.add_cluster("encoder", [1, 2], label="Encoder")
    g.add_cluster("decoder", [3, 4], label="Decoder")
    return g


@pytest.fixture
def empty_graph():
    """Graph with no nodes or edges."""
    return DaguaGraph()


@pytest.fixture
def single_node_graph():
    """Graph with one node, no edges."""
    g = DaguaGraph()
    g.add_node("alone")
    return g


@pytest.fixture
def default_config():
    """Default layout config."""
    return LayoutConfig()


@pytest.fixture
def fast_config():
    """Fast layout config for quick tests."""
    return LayoutConfig(steps=50, lr=0.1)
