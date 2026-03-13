"""Base class and registry for competitor layout adapters."""

from __future__ import annotations

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

    @abstractmethod
    def layout(self, graph: DaguaGraph, timeout: float = 300.0) -> CompetitorResult:
        """Run layout and return result with timing."""
        ...

    def available(self) -> bool:
        """Check if this competitor's tool is installed and usable."""
        return True


# ── Registry ──────────────────────────────────────────────────────────────────

_COMPETITORS: Dict[str, CompetitorBase] = {}


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
