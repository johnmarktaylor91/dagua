"""Optional runtime environment helpers for evaluation workflows."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator


@contextmanager
def suspend_torchlens_decoration() -> Iterator[None]:
    """Temporarily strip TorchLens global wrappers if TorchLens is imported.

    Dagua's benchmark and tuning workflows sometimes import TorchLens-derived
    graph fixtures in the same Python process. TorchLens globally decorates
    torch callables at import time, which is valuable for tracing but pure
    overhead for Dagua layout benchmarking. This helper removes that overhead
    when the override API is available and restores the wrappers afterward.
    """
    try:
        import torchlens  # type: ignore
    except Exception:
        yield
        return

    undecorate = getattr(torchlens, "undecorate_all_globally", None)
    redecorate = getattr(torchlens, "redecorate_all_globally", None)
    if undecorate is None or redecorate is None:
        yield
        return

    undecorate()
    try:
        yield
    finally:
        redecorate()
