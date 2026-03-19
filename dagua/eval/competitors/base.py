"""Base class and registry for competitor layout adapters."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

import torch

if TYPE_CHECKING:
    from dagua.graph import DaguaGraph


@dataclass
class CompetitorResult:
    """Result of running a competitor layout engine on a single graph."""

    name: str  # e.g. "graphviz_dot"
    pos: Optional[torch.Tensor]  # [N, 2] or None if failed/timeout
    runtime_seconds: float
    error: Optional[str] = None


class CompetitorBase(ABC):
    """Base class for competitor layout adapters."""

    name: str = ""
    max_nodes: int = 0
    supports_clusters: bool = False

    @abstractmethod
    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run layout and return result with timing.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Maximum runtime in seconds.
        seed : int | None, default=None
            Random seed for stochastic algorithms. ``None`` means use the
            adapter default, typically ``42`` for reproducibility.

        Returns
        -------
        CompetitorResult
            Layout outcome, runtime, and optional error details.
        """
        ...

    def available(self) -> bool:
        """Check if this competitor's tool is installed and usable."""
        return True


# ── Registry ──────────────────────────────────────────────────────────────────

_COMPETITORS: Dict[str, CompetitorBase] = {}
_RUNTIME_SEED_ENV_VAR = "DAGUA_COMPETITOR_SEED"


def get_runtime_seed(default: Optional[int] = 42) -> Optional[int]:
    """Return the benchmark runtime seed when one has been configured.

    Parameters
    ----------
    default : int | None, default=42
        Fallback seed used when no benchmark-specific override is present.

    Returns
    -------
    int | None
        Seed value for stochastic competitor adapters.
    """
    raw_seed = os.environ.get(_RUNTIME_SEED_ENV_VAR)
    if raw_seed is None:
        return default
    try:
        return int(raw_seed)
    except ValueError:
        return default


def register(cls):
    """Class decorator that instantiates and registers a competitor."""
    instance = cls()
    _COMPETITORS[instance.name] = instance
    return cls


def get_competitors() -> List[CompetitorBase]:
    """Return all registered competitors (installed or not)."""
    return list(_COMPETITORS.values())


def get_available_competitors() -> List[CompetitorBase]:
    """Return only competitors whose tools are installed."""
    return [c for c in _COMPETITORS.values() if c.available()]
