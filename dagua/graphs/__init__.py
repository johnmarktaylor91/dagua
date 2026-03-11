"""Bundled graph definitions for examples and testing."""

from pathlib import Path

_GRAPHS_DIR = Path(__file__).parent


def load(name: str) -> "DaguaGraph":  # noqa: F821
    """Load a bundled graph by name (e.g. 'diamond').

    Args:
        name: Graph name (without extension). Must match a .yaml file in this directory.

    Returns:
        A fully constructed DaguaGraph.
    """
    from dagua.io import load as io_load

    path = _GRAPHS_DIR / f"{name}.yaml"
    if not path.exists():
        available = list_graphs()
        raise ValueError(f"Unknown graph: {name!r}. Available: {available}")
    return io_load(path)


def list_graphs() -> list:
    """List available bundled graph names."""
    return sorted(p.stem for p in _GRAPHS_DIR.glob("*.yaml"))
