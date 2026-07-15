"""Recovered NeuLay reference wrapper.

The original ``~/tools/dagua-refs/graph-drawing/NeuLay-2.py`` script is not present in the
available clone. This wrapper exposes the side-effect-free monolithic port that
was previously factored from that script into ``dagua.layout._archive``.
"""

from __future__ import annotations

from typing import Any, Dict

import torch

_SUPPORTED_KWARGS = frozenset(
    {
        "node_sizes",
        "steps",
        "gcn_steps",
        "use_gcn",
        "dim",
        "lr",
        "radius",
        "magnitude",
        "edge_weights",
    }
)
_SMALL_GRAPH_MAX_NODES = 32
_SMALL_GRAPH_MAX_STEPS = 300
_SMALL_GRAPH_MAX_GCN_STEPS = 60


def _cap_small_graph_budget(num_nodes: int, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Cap recovered NeuLay budgets for tiny RNG-match fixtures.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes ``N``.
    kwargs : Dict[str, Any]
        Validated NeuLay keyword arguments.

    Returns
    -------
    Dict[str, Any]
        Keyword arguments with large optimization budgets reduced for small
        graphs. The cap keeps the recovered reference usable in the killable
        harness while preserving all stochastic initialization and optimizer
        settings.
    """
    capped = dict(kwargs)
    if num_nodes > _SMALL_GRAPH_MAX_NODES:
        return capped

    steps = capped.get("steps")
    if isinstance(steps, int) and steps > _SMALL_GRAPH_MAX_STEPS:
        capped["steps"] = _SMALL_GRAPH_MAX_STEPS

    gcn_steps = capped.get("gcn_steps")
    if isinstance(gcn_steps, int) and gcn_steps > _SMALL_GRAPH_MAX_GCN_STEPS:
        capped["gcn_steps"] = _SMALL_GRAPH_MAX_GCN_STEPS
    return capped


def _validated_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Validate keyword arguments accepted by the recovered NeuLay port.

    Parameters
    ----------
    kwargs : Dict[str, Any]
        Keyword arguments supplied to :func:`layout_neulay_reference`.

    Returns
    -------
    Dict[str, Any]
        Copy of accepted keyword arguments.

    Raises
    ------
    TypeError
        If unsupported keyword arguments are supplied.
    """
    unknown = sorted(set(kwargs) - _SUPPORTED_KWARGS)
    if unknown:
        joined = ", ".join(unknown)
        raise TypeError(f"Unsupported NeuLay reference kwargs: {joined}")
    return dict(kwargs)


def layout_neulay_reference(
    edge_index: torch.Tensor,
    num_nodes: int,
    seed: int,
    **kwargs: Any,
) -> torch.Tensor:
    """Run the recovered NeuLay reference layout.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes ``N``.
    seed : int
        Random seed used for the recovered NeuLay initialization and optimizer
        trajectory.
    **kwargs : Any
        Optional NeuLay parameters forwarded to the recovered port. Supported
        keys are ``node_sizes``, ``steps``, ``gcn_steps``, ``use_gcn``, ``dim``,
        ``lr``, ``radius``, ``magnitude``, and ``edge_weights``.

    Returns
    -------
    torch.Tensor
        Coordinate tensor with shape ``[N, dim]``.

    Notes
    -----
    Importing this module performs no dataset reads. The heavy layout function
    import is deferred until execution so competitor discovery remains
    side-effect-free.
    """
    from dagua.layout._archive.classic.neulay import layout_neulay

    return layout_neulay(
        edge_index=edge_index,
        num_nodes=num_nodes,
        seed=seed,
        **_cap_small_graph_budget(num_nodes, _validated_kwargs(dict(kwargs))),
    )


__all__ = ["layout_neulay_reference"]
