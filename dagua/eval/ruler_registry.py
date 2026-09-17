"""Lazy opt-in registry for versioned evaluation rulers."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping, Tuple

RULER_CONFIG_KEY = "ruler"
DEFAULT_RULER_KEY = "ruler_v3"


@dataclass(frozen=True)
class RulerRegistration:
    """Describe one lazily imported evaluation ruler.

    Parameters
    ----------
    key : str
        Opt-in configuration value.
    module : str
        Import path containing the scorer.
    symbol : str
        Callable scorer symbol.
    experimental : bool
        Whether probation prevents this ruler from becoming the default.
    required_arguments : tuple[str, ...]
        High-level input contract advertised to registry consumers.
    """

    key: str
    module: str
    symbol: str
    experimental: bool
    required_arguments: Tuple[str, ...]


RULER_REGISTRY: Mapping[str, RulerRegistration] = MappingProxyType(
    {
        "ruler_v3": RulerRegistration(
            key="ruler_v3",
            module="dagua.eval.ruler_v3",
            symbol="score_core_v3",
            experimental=False,
            required_arguments=("pos", "edge_index"),
        ),
        "ruler_v4": RulerRegistration(
            key="ruler_v4",
            module="dagua.eval.ruler_v4.score",
            symbol="score",
            experimental=True,
            required_arguments=("scene", "weight_table", "profiles"),
        ),
    }
)


def get_ruler_registration(config: Mapping[str, Any]) -> RulerRegistration:
    """Resolve a ruler registration from additive evaluation configuration.

    Absence of the new ``ruler`` key preserves the V3 default. V4 is selected
    only by ``{"ruler": "ruler_v4"}`` while it remains on probation.

    Parameters
    ----------
    config : mapping[str, Any]
        Evaluation configuration with an optional ``ruler`` key.

    Returns
    -------
    RulerRegistration
        Selected lazy registration.

    Raises
    ------
    ValueError
        If the key is non-string or unregistered.
    """

    key = config.get(RULER_CONFIG_KEY, DEFAULT_RULER_KEY)
    if not isinstance(key, str) or key not in RULER_REGISTRY:
        raise ValueError(
            f"unknown evaluation ruler {key!r}; expected one of {sorted(RULER_REGISTRY)}"
        )
    return RULER_REGISTRY[key]


def get_ruler_scorer(config: Mapping[str, Any]) -> Callable[..., Any]:
    """Load the scorer selected by evaluation configuration.

    Parameters
    ----------
    config : mapping[str, Any]
        Evaluation configuration consumed by :func:`get_ruler_registration`.

    Returns
    -------
    callable
        Registered scoring function.

    Raises
    ------
    TypeError
        If the registered symbol is unexpectedly non-callable.
    """

    registration = get_ruler_registration(config)
    module = importlib.import_module(registration.module)
    scorer = getattr(module, registration.symbol)
    if not callable(scorer):
        raise TypeError(f"registered ruler symbol is not callable: {registration.key}")
    return scorer


def score_with_registered_ruler(
    config: Mapping[str, Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Dispatch one score call through the versioned ruler registry.

    Parameters
    ----------
    config : mapping[str, Any]
        Evaluation configuration with optional ``ruler`` opt-in.
    *args : Any
        Positional arguments for the selected version's input contract.
    **kwargs : Any
        Keyword arguments for the selected version's input contract.

    Returns
    -------
    Any
        Selected ruler's native result type.
    """

    return get_ruler_scorer(config)(*args, **kwargs)
