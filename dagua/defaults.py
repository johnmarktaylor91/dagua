"""Thread-safe global defaults with flat configure() API and context manager.

Usage:
    import dagua

    # One-liner theme switch
    dagua.set_theme('dark')
    dagua.set_device('cuda')

    # Flat namespace — routes kwargs to the right style/config fields
    dagua.configure(font_size=10, node_sep=40, background_color='#1A1E24')

    # Scoped override (re-entrant)
    with dagua.defaults(theme='minimal', node_sep=60):
        dagua.draw(g)  # uses minimal theme + 60px spacing

    # Inspect / export
    dagua.get_defaults()
    dagua.export_config('my_settings.yaml')
    dagua.reset()
"""

from __future__ import annotations

import copy
import dataclasses
import threading
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

import torch

from dagua.styles import (
    ClusterStyle,
    EdgeStyle,
    GraphStyle,
    NodeStyle,
    Theme,
    get_theme,
    DEFAULT_THEME_OBJ,
)


# ─── Valid field names per target (for routing and did-you-mean) ──────────

def _field_names(cls) -> set:
    return {f.name for f in dataclasses.fields(cls)}


_META_FIELDS = {"theme", "device", "index_dtype", "size_dtype"}
_NODE_STYLE_FIELDS = _field_names(NodeStyle)
_EDGE_STYLE_FIELDS = _field_names(EdgeStyle)
_GRAPH_STYLE_FIELDS = _field_names(GraphStyle)
_CLUSTER_STYLE_FIELDS = _field_names(ClusterStyle)

# LayoutConfig fields — import lazily to avoid circular import at module level
_LAYOUT_CONFIG_FIELDS: Optional[set] = None


def _get_layout_config_fields() -> set:
    global _LAYOUT_CONFIG_FIELDS
    if _LAYOUT_CONFIG_FIELDS is None:
        from dagua.config import LayoutConfig
        _LAYOUT_CONFIG_FIELDS = _field_names(LayoutConfig)
    return _LAYOUT_CONFIG_FIELDS


# Prefixed fields route to their style class after stripping prefix
_EDGE_PREFIX = "edge_"
_GRAPH_PREFIX = "graph_"

_DTYPE_NAME_TO_TORCH = {
    "int32": torch.int32,
    "int64": torch.int64,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
}


def _dtype_name(dtype: torch.dtype) -> str:
    for name, torch_dtype in _DTYPE_NAME_TO_TORCH.items():
        if dtype == torch_dtype:
            return name
    raise TypeError(f"Unsupported dtype: {dtype}")


def _normalize_index_dtype(value: Any) -> torch.dtype:
    if isinstance(value, str):
        value = _DTYPE_NAME_TO_TORCH.get(value)
    if value not in (torch.int32, torch.int64):
        raise TypeError("index_dtype must be one of: torch.int32, torch.int64, 'int32', 'int64'")
    return value


def _normalize_size_dtype(value: Any) -> torch.dtype:
    if isinstance(value, str):
        value = _DTYPE_NAME_TO_TORCH.get(value)
    if value not in (torch.float16, torch.float32, torch.float64):
        raise TypeError(
            "size_dtype must be one of: torch.float16, torch.float32, torch.float64, "
            "'float16', 'float32', 'float64'"
        )
    return value

# All valid kwarg names (union across all targets)
def _all_valid_names() -> set:
    names = set(_META_FIELDS)
    names |= _NODE_STYLE_FIELDS
    names |= {f"edge_{f}" for f in _EDGE_STYLE_FIELDS}
    names |= _GRAPH_STYLE_FIELDS
    names |= _get_layout_config_fields()
    return names


def _did_you_mean(name: str) -> str:
    """Suggest the closest valid kwarg name using Levenshtein distance."""
    valid = _all_valid_names()
    # Simple edit distance (good enough for typo detection)
    def _dist(a: str, b: str) -> int:
        if len(a) > len(b):
            a, b = b, a
        prev = list(range(len(a) + 1))
        for j in range(1, len(b) + 1):
            curr = [j] + [0] * len(a)
            for i in range(1, len(a) + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                curr[i] = min(curr[i - 1] + 1, prev[i] + 1, prev[i - 1] + cost)
            prev = curr
        return prev[-1]

    scored = [(n, _dist(name, n)) for n in valid]
    scored.sort(key=lambda x: x[1])
    if scored and scored[0][1] <= 3:
        return f" Did you mean {scored[0][0]!r}?"
    return ""


# ─── Thread-local state ──────────────────────────────────────────────────

class _Defaults:
    """Container for one level of defaults."""

    def __init__(self):
        self.theme_name: str = "default"
        self.theme: Theme = copy.deepcopy(DEFAULT_THEME_OBJ)
        self.device: str = "cpu"
        self.index_dtype: torch.dtype = torch.int64
        self.size_dtype: torch.dtype = torch.float32
        self.node_style_overrides: Dict[str, Any] = {}
        self.edge_style_overrides: Dict[str, Any] = {}
        self.graph_style_overrides: Dict[str, Any] = {}
        self.layout_overrides: Dict[str, Any] = {}

    def copy(self) -> _Defaults:
        d = _Defaults()
        d.theme_name = self.theme_name
        d.theme = copy.deepcopy(self.theme)
        d.device = self.device
        d.index_dtype = self.index_dtype
        d.size_dtype = self.size_dtype
        d.node_style_overrides = dict(self.node_style_overrides)
        d.edge_style_overrides = dict(self.edge_style_overrides)
        d.graph_style_overrides = dict(self.graph_style_overrides)
        d.layout_overrides = dict(self.layout_overrides)
        return d


_local = threading.local()


def _get_stack() -> list:
    if not hasattr(_local, "stack"):
        _local.stack = [_Defaults()]
    return _local.stack


def _current() -> _Defaults:
    return _get_stack()[-1]


# ─── Public API ──────────────────────────────────────────────────────────


def set_theme(name: str) -> None:
    """Set the global default theme by name."""
    cur = _current()
    cur.theme_name = name
    cur.theme = get_theme(name)


def set_device(device: str) -> None:
    """Set the global default device ('cpu' or 'cuda')."""
    _current().device = device


def configure(**kwargs: Any) -> None:
    """Set global defaults via flat namespace.

    Routes kwargs to the correct target:
    - theme, device → meta fields
    - node_sep, rank_sep, steps, lr, w_* → LayoutConfig
    - font_size, shape, fill, base_color, ... → NodeStyle
    - edge_color, edge_width, edge_style, ... → EdgeStyle (strip edge_ prefix)
    - background_color, margin, ... → GraphStyle

    Raises TypeError for unknown kwargs with did-you-mean suggestions.
    """
    cur = _current()
    layout_fields = _get_layout_config_fields()

    for key, value in kwargs.items():
        if key == "theme":
            if isinstance(value, str):
                cur.theme_name = value
                cur.theme = get_theme(value)
            elif isinstance(value, Theme):
                cur.theme = copy.deepcopy(value)
                cur.theme_name = value.name
            else:
                raise TypeError(f"theme must be a string or Theme, got {type(value).__name__}")
        elif key == "device":
            cur.device = value
        elif key == "index_dtype":
            cur.index_dtype = _normalize_index_dtype(value)
        elif key == "size_dtype":
            cur.size_dtype = _normalize_size_dtype(value)
        elif key in layout_fields:
            cur.layout_overrides[key] = value
        elif key in _NODE_STYLE_FIELDS:
            cur.node_style_overrides[key] = value
        elif key.startswith(_EDGE_PREFIX) and key[len(_EDGE_PREFIX):] in _EDGE_STYLE_FIELDS:
            cur.edge_style_overrides[key[len(_EDGE_PREFIX):]] = value
        elif key in _GRAPH_STYLE_FIELDS:
            cur.graph_style_overrides[key] = value
        else:
            hint = _did_you_mean(key)
            raise TypeError(f"Unknown configure() option: {key!r}.{hint}")


@contextmanager
def defaults(**kwargs: Any) -> Iterator[None]:
    """Context manager for scoped default overrides.

    Pushes a copy of current defaults, applies kwargs, yields,
    then pops back to previous state. Re-entrant safe.

    Usage:
        with dagua.defaults(theme='dark', node_sep=60):
            dagua.draw(g)
    """
    stack = _get_stack()
    new = _current().copy()
    stack.append(new)
    try:
        if kwargs:
            configure(**kwargs)
        yield
    finally:
        stack.pop()


def get_defaults() -> Dict[str, Any]:
    """Return the current effective defaults as a flat dict."""
    cur = _current()
    result: Dict[str, Any] = {
        "theme": cur.theme_name,
        "device": cur.device,
        "index_dtype": _dtype_name(cur.index_dtype),
        "size_dtype": _dtype_name(cur.size_dtype),
    }
    for k, v in cur.layout_overrides.items():
        result[k] = v
    for k, v in cur.node_style_overrides.items():
        result[k] = v
    for k, v in cur.edge_style_overrides.items():
        result[f"edge_{k}"] = v
    for k, v in cur.graph_style_overrides.items():
        result[k] = v
    return result


def get_default_theme() -> Theme:
    """Return a deep copy of the current default theme."""
    return copy.deepcopy(_current().theme)


def get_default_device() -> str:
    """Return the current default device."""
    return _current().device


def get_default_index_dtype() -> torch.dtype:
    """Return the current default storage dtype for graph indices."""
    return _current().index_dtype


def get_default_size_dtype() -> torch.dtype:
    """Return the current default storage dtype for computed node sizes."""
    return _current().size_dtype


def get_default_node_style_overrides() -> Dict[str, Any]:
    """Return current global node style overrides (field_name -> value)."""
    return dict(_current().node_style_overrides)


def get_default_edge_style_overrides() -> Dict[str, Any]:
    """Return current global edge style overrides (field_name -> value)."""
    return dict(_current().edge_style_overrides)


def get_default_graph_style_overrides() -> Dict[str, Any]:
    """Return current global graph style overrides (field_name -> value)."""
    return dict(_current().graph_style_overrides)


def get_default_layout_overrides() -> Dict[str, Any]:
    """Return current global layout config overrides."""
    return dict(_current().layout_overrides)


def export_config(path: str) -> None:
    """Dump current defaults to a YAML or JSON file.

    Format auto-detected from extension (.yaml/.yml → YAML, .json → JSON).
    """
    import json as _json
    from pathlib import Path

    data = get_defaults()
    p = Path(path)

    if p.suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError("PyYAML required for YAML export: pip install pyyaml")
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    else:
        with open(path, "w") as f:
            _json.dump(data, f, indent=2)


def reset() -> None:
    """Restore all defaults to library initial state."""
    stack = _get_stack()
    stack.clear()
    stack.append(_Defaults())
