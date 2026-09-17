"""Traced-execution seam for the 6.5 differentiable surrogate.

The exact scorer casts every score-visible quantity to a Python float at
the moment it leaves tensor arithmetic. The 6.5 surrogate is the SAME
frozen closed forms evaluated without those casts, so position tensors
carrying ``requires_grad`` flow through facet arithmetic to ``l_total``
(DISCREPANCIES entry 38). This module provides the two mechanisms that
make one shared implementation serve both paths:

- a context-scoped trace: inside :func:`trace_subterms`, :func:`keep`
  returns the live tensor instead of casting, and ``value_result``
  records each tensor-valued subterm into the active buffer;
- type-polymorphic scalar helpers (:func:`p_sqrt`, :func:`p_min`, ...)
  whose float branches execute byte-for-byte the same operations the
  exact scorer always used, and whose tensor branches are the autograd
  equivalents. Exact-path outputs are therefore bit-identical to the
  pre-surrogate implementation; traced outputs may differ from exact by
  accumulation order only, and that gap is measured, never assumed zero.

No smoothing is introduced here: relaxations live in the facet closed
forms themselves where the contracts pin them (entry 38 citations).
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Dict, Iterator, Optional, Sequence, Union

import torch

Scalar = Union[float, torch.Tensor]

# One-sided tolerance for traced subterm values that drift past the [0, 1]
# contract bound by accumulation-order noise. The traced-vs-exact gap is
# measured at 1-3 ULP (~1e-16 near 1.0); 1e-12 gives four orders of margin
# while still refusing anything that could be a real facet defect.
TRACED_BOUND_TOLERANCE = 1e-12


class SurrogateTraceError(ValueError):
    """A traced (tensor-valued) subterm violated the facet value contract.

    Raised instead of the exact path's bare ``ValueError`` so callers can
    distinguish a surrogate-side numeric failure (NaN in a traced tensor,
    or a beyond-tolerance bound violation) from a facet-contract violation
    on the frozen float path.
    """


_TRACE_BUFFER: ContextVar[Optional[Dict[str, torch.Tensor]]] = ContextVar(
    "ruler_v4_trace_buffer", default=None
)


def tracing_active() -> bool:
    """Return whether a traced execution is in progress.

    Returns
    -------
    bool
        True inside a :func:`trace_subterms` context.
    """

    return _TRACE_BUFFER.get() is not None


@contextmanager
def trace_subterms() -> Iterator[Dict[str, torch.Tensor]]:
    """Collect tensor-valued score-visible subterms from facet execution.

    Yields
    ------
    dict[str, torch.Tensor]
        Buffer filled by ``value_result`` with one live scalar tensor per
        tensor-valued subterm id evaluated inside the context.
    """

    buffer: Dict[str, torch.Tensor] = {}
    token = _TRACE_BUFFER.set(buffer)
    try:
        yield buffer
    finally:
        _TRACE_BUFFER.reset(token)


def record_subterm(subterm_id: str, value: torch.Tensor) -> None:
    """Record one live subterm tensor into the active trace, if any.

    Parameters
    ----------
    subterm_id : str
        Frozen manifest subterm id.
    value : torch.Tensor
        Scalar defect tensor.
    """

    buffer = _TRACE_BUFFER.get()
    if buffer is not None:
        buffer[subterm_id] = value.reshape(())


def keep(value: torch.Tensor) -> Scalar:
    """Leave tensor arithmetic: cast exactly as before, or keep the graph.

    Parameters
    ----------
    value : torch.Tensor
        Scalar tensor produced by facet arithmetic.

    Returns
    -------
    float or torch.Tensor
        ``float(value.item())`` on the exact path (the historical cast,
        bit-identical), or the live scalar tensor inside a trace.
    """

    if tracing_active():
        return value.reshape(())
    return float(value.item())


def p_sqrt(value: Scalar) -> Scalar:
    """Square root; ``math.sqrt`` on floats, ``torch.sqrt`` on tensors."""

    if isinstance(value, torch.Tensor):
        return torch.sqrt(value)
    return math.sqrt(value)


def p_exp(value: Scalar) -> Scalar:
    """Exponential; ``math.exp`` on floats, ``torch.exp`` on tensors."""

    if isinstance(value, torch.Tensor):
        return torch.exp(value)
    return math.exp(value)


def p_log(value: Scalar) -> Scalar:
    """Natural log; ``math.log`` on floats, ``torch.log`` on tensors."""

    if isinstance(value, torch.Tensor):
        return torch.log(value)
    return math.log(value)


def p_log1p(value: Scalar) -> Scalar:
    """``log(1 + x)``; ``math.log1p`` on floats, ``torch.log1p`` on tensors."""

    if isinstance(value, torch.Tensor):
        return torch.log1p(value)
    return math.log1p(value)


def p_abs(value: Scalar) -> Scalar:
    """Absolute value; ``abs`` on floats, ``torch.abs`` on tensors."""

    if isinstance(value, torch.Tensor):
        return torch.abs(value)
    return abs(value)


def _promote(value: Scalar) -> torch.Tensor:
    """Return the scalar as a float64 tensor, preserving any graph."""

    if isinstance(value, torch.Tensor):
        return value
    return torch.tensor(value, dtype=torch.float64)


def p_min(first: Scalar, second: Scalar) -> Scalar:
    """Two-argument minimum; ``min`` on floats, ``torch.minimum`` on tensors."""

    if isinstance(first, torch.Tensor) or isinstance(second, torch.Tensor):
        return torch.minimum(_promote(first), _promote(second))
    return min(first, second)


def p_max(first: Scalar, second: Scalar) -> Scalar:
    """Two-argument maximum; ``max`` on floats, ``torch.maximum`` on tensors."""

    if isinstance(first, torch.Tensor) or isinstance(second, torch.Tensor):
        return torch.maximum(_promote(first), _promote(second))
    return max(first, second)


def p_fsum(values: Sequence[Scalar]) -> Scalar:
    """Sum a population; ``math.fsum`` on floats, stacked sum with tensors.

    Parameters
    ----------
    values : sequence[float or torch.Tensor]
        Population to sum. The float branch keeps ``math.fsum``'s exact
        rounding, so exact-path behavior is unchanged.

    Returns
    -------
    float or torch.Tensor
        Population sum.
    """

    items = list(values)
    if any(isinstance(item, torch.Tensor) for item in items):
        return torch.stack([_promote(item) for item in items]).sum()
    return math.fsum(items)


def p_sum(values: Sequence[Scalar]) -> Scalar:
    """Sum a population; builtin ``sum`` on floats, stacked sum with tensors.

    Use where the exact scorer historically used builtin ``sum`` (left
    to right accumulation) rather than ``math.fsum``.
    """

    items = list(values)
    if any(isinstance(item, torch.Tensor) for item in items):
        return torch.stack([_promote(item) for item in items]).sum()
    return sum(items)


def as_float(value: Scalar) -> float:
    """Read a scalar for control flow or diagnostics, never for scoring.

    Parameters
    ----------
    value : float or torch.Tensor
        Scalar whose numeric value drives a branch or a published raw
        statistic. Detached: gradients never flow through this read.

    Returns
    -------
    float
        Plain float value.
    """

    if isinstance(value, torch.Tensor):
        return float(value.detach().item())
    return float(value)
