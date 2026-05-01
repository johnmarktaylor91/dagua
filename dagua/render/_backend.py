"""Matplotlib canvas backend resolution for Dagua rendering."""

from __future__ import annotations

import ctypes
from typing import Literal, Optional, Tuple, Type

BackendName = Literal["agg", "cairo"]

_CAIRO_INSTALL_MESSAGE = """Cairo backend requires mplcairo. Install with:
    pip install 'dagua[cairo]'

On Linux/Mac, mplcairo also requires the libcairo system library:
    apt install libcairo2-dev   # Ubuntu/Debian
    brew install cairo          # macOS
"""

_DEFAULT_BACKEND_OVERRIDE: Optional[BackendName] = None
_CAIRO_STROKE_WIDTH_SCALE = 0.86
_AGG_STROKE_WIDTH_SCALE = 1.0


def _preload_cairo_library() -> None:
    """Load the system cairo library into the global symbol table.

    Returns
    -------
    None
        Best-effort preload; missing libraries are ignored so auto-detect can
        still fall back to Agg.
    """
    candidates = (
        "/usr/lib/x86_64-linux-gnu/libcairo.so.2",
        "/usr/lib/aarch64-linux-gnu/libcairo.so.2",
        "/usr/lib/libcairo.so.2",
        "/opt/homebrew/lib/libcairo.2.dylib",
        "/usr/local/lib/libcairo.2.dylib",
    )
    for candidate in candidates:
        try:
            ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue
        return


def _cairo_available() -> bool:
    """Return True if mplcairo can be imported.

    Returns
    -------
    bool
        True when the non-GUI mplcairo backend imports cleanly, otherwise False.
    """
    try:
        # pycairo can require cairo symbols to be globally visible before the
        # extension module loads, especially in mixed system/conda installs.
        _preload_cairo_library()
        import mplcairo.base  # noqa: F401
    except Exception:
        return False
    return True


def _agg_canvas_cls() -> Type:
    """Return Matplotlib's built-in Agg figure canvas class.

    Returns
    -------
    type
        Matplotlib Agg ``FigureCanvas`` class.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    return FigureCanvasAgg


def _cairo_canvas_cls() -> Type:
    """Return mplcairo's non-GUI figure canvas class.

    Returns
    -------
    type
        mplcairo ``FigureCanvas`` class.

    Raises
    ------
    ImportError
        If mplcairo is unavailable or cannot import cleanly.
    """
    try:
        _preload_cairo_library()
        from mplcairo.base import FigureCanvasCairo
    except Exception as exc:
        raise ImportError(_CAIRO_INSTALL_MESSAGE) from exc
    return FigureCanvasCairo


def _coerce_backend_name(name: str) -> BackendName:
    """Validate a backend name string.

    Parameters
    ----------
    name : str
        Candidate backend name.

    Returns
    -------
    BackendName
        Validated backend name.

    Raises
    ------
    ValueError
        If ``name`` is not one of Dagua's supported Matplotlib backends.
    """
    normalized = name.lower()
    if normalized == "agg":
        return "agg"
    if normalized == "cairo":
        return "cairo"
    raise ValueError(f"Unknown render backend {name!r}; expected 'agg', 'cairo', or None.")


def stroke_width_scale_for(backend_name: str) -> float:
    """Return the per-backend stroke-width calibration factor.

    Parameters
    ----------
    backend_name : str
        Resolved Matplotlib backend name.

    Returns
    -------
    float
        Multiplicative stroke-width calibration factor for render-time
        data-coordinate ribbons.

    Notes
    -----
    Cairo distributes stroke ink differently from Agg at the same nominal
    filled-ribbon width. Multiplying the data-coordinate ribbon width by this
    constant under cairo restores effective ink density parity with Agg and the
    graphviz reference. Empirically calibrated to 0.86 during Sprint B Round 3
    on 2026-04-30 after the proposed 1.15 nudge moved
    ``nodes_shapes_rect`` / ``nodes_shapes_tab`` farther from graphviz; the
    lower value closes the L1 regression on ``nodes_shapes_rect`` and
    ``nodes_shapes_tab`` without regressing cairo's wins on dashed strokes,
    curve anti-aliasing, or text hinting. The optimizer still sees the user's
    ``style.stroke_width`` value unchanged.
    """
    if backend_name == "cairo":
        return _CAIRO_STROKE_WIDTH_SCALE
    return _AGG_STROKE_WIDTH_SCALE


def _resolve_backend(name: Optional[str]) -> Tuple[Type, str]:
    """Resolve a backend name to a canvas class and resolved name.

    Parameters
    ----------
    name : str, optional
        Backend selector. ``None`` uses the global default override when set,
        otherwise auto-detects cairo and falls back to Agg.

    Returns
    -------
    tuple[type, str]
        Matplotlib ``FigureCanvas`` subclass and resolved backend name.

    Raises
    ------
    ImportError
        If cairo is requested explicitly but mplcairo is unavailable.
    ValueError
        If an unsupported backend name is requested.
    """
    requested = _DEFAULT_BACKEND_OVERRIDE if name is None else _coerce_backend_name(name)

    if requested == "agg":
        return _agg_canvas_cls(), "agg"
    if requested == "cairo":
        return _cairo_canvas_cls(), "cairo"

    if _cairo_available():
        return _cairo_canvas_cls(), "cairo"
    return _agg_canvas_cls(), "agg"


def get_default_backend() -> str:
    """Return the current default render backend.

    Returns
    -------
    str
        Resolved backend name. Auto-detection is evaluated on each call when no
        global override is set.
    """
    _, resolved_name = _resolve_backend(None)
    return resolved_name


def set_default_backend(name: Optional[str]) -> None:
    """Override the global default render backend.

    Parameters
    ----------
    name : str, optional
        ``"agg"`` or ``"cairo"`` to set a persistent default. ``None`` resets
        Dagua to runtime auto-detection.

    Returns
    -------
    None
        This function mutates module-level backend state.

    Raises
    ------
    ValueError
        If ``name`` is not one of Dagua's supported Matplotlib backends.
    """
    global _DEFAULT_BACKEND_OVERRIDE

    _DEFAULT_BACKEND_OVERRIDE = None if name is None else _coerce_backend_name(name)
