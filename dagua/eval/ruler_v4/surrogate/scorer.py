"""Evaluate the manifest-compiled differentiable defect surrogate."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Mapping, Optional, Tuple

import torch

from dagua.eval.ruler_v4.composition import (
    CompositionFamily,
    CompositionProfile,
    CompositionResult,
    compose,
)
from dagua.eval.ruler_v4.scene import FacetResult
from dagua.eval.ruler_v4.surrogate.manifest import (
    CompiledSurrogateManifest,
    SurrogateTermTrace,
    compile_surrogate_manifest,
)
from dagua.eval.ruler_v4.weight_table import WeightTable


@dataclass(frozen=True)
class SoftTermResult:
    """Publish one active differentiable surrogate term.

    Parameters
    ----------
    trace : SurrogateTermTrace
        Contract and compilation-rule trace.
    value : torch.Tensor
        Scalar differentiable defect coordinate.
    exact_value : float
        Exact facet value used when no differentiable binding is supplied.
    normalized_weight : float
        Applicable headline mass after NA exclusion.
    group : str
        Frozen-table reporting group.
    """

    trace: SurrogateTermTrace
    value: torch.Tensor
    exact_value: float
    normalized_weight: float
    group: str


@dataclass(frozen=True)
class SoftScoreResult:
    """Return differentiable loss-space surrogate outputs.

    Parameters
    ----------
    l_mean : torch.Tensor
        Applicable-mass arithmetic defect.
    l_bottleneck : torch.Tensor or None
        Exact-family soft bottleneck arm when selected.
    l_total : torch.Tensor
        Differentiable pre-map ordering loss.
    exact_composition : CompositionResult
        Closed point composition evaluated from the same facet rows.
    terms : tuple[SoftTermResult, ...]
        Active compiled terms and their provenance.
    manifest_digest : str
        Digest of the compiled facet graph.
    """

    l_mean: torch.Tensor
    l_bottleneck: Optional[torch.Tensor]
    l_total: torch.Tensor
    exact_composition: CompositionResult
    terms: Tuple[SoftTermResult, ...]
    manifest_digest: str


def _validate_bound_tensor(subterm_id: str, value: torch.Tensor) -> None:
    """Validate one caller-supplied differentiable defect coordinate.

    Parameters
    ----------
    subterm_id : str
        Frozen subterm id used in error messages.
    value : torch.Tensor
        Candidate scalar defect tensor.

    Raises
    ------
    ValueError
        If the tensor is not scalar, floating, finite, or in ``[0, 1]``.
    """

    if value.numel() != 1 or not value.is_floating_point():
        raise ValueError(f"surrogate binding {subterm_id} must be one floating scalar")
    detached = value.detach()
    if not bool(torch.isfinite(detached).item()) or not 0.0 <= float(detached) <= 1.0:
        raise ValueError(f"surrogate binding {subterm_id} must be finite and in [0, 1]")


def _active_terms(
    exact: CompositionResult,
    compiled: CompiledSurrogateManifest,
    term_tensors: Mapping[str, torch.Tensor],
) -> Tuple[SoftTermResult, ...]:
    """Bind active exact rows to compiled identity terms.

    Parameters
    ----------
    exact : CompositionResult
        Closed composition whose NA handling defines the active set.
    compiled : CompiledSurrogateManifest
        Complete compiled term graph.
    term_tensors : mapping[str, torch.Tensor]
        Optional differentiable replacements keyed by subterm id.

    Returns
    -------
    tuple[SoftTermResult, ...]
        Applicable positive-mass terms in table order.

    Raises
    ------
    ValueError
        If a binding is unknown or targets an inactive row.
    """

    active_ids = {
        row.subterm_id
        for row in exact.subterms
        if row.value is not None and not row.diagnostic and row.normalized_weight > 0.0
    }
    unknown = sorted(set(term_tensors) - set(compiled.by_subterm))
    if unknown:
        raise ValueError(f"unknown surrogate bindings: {unknown}")
    inactive = sorted(set(term_tensors) - active_ids)
    if inactive:
        raise ValueError(f"surrogate bindings target inactive terms: {inactive}")

    terms: List[SoftTermResult] = []
    for row in exact.subterms:
        if row.subterm_id not in active_ids:
            continue
        assert row.value is not None
        bound = term_tensors.get(row.subterm_id)
        if bound is None:
            bound = torch.tensor(row.value, dtype=torch.float64)
        _validate_bound_tensor(row.subterm_id, bound)
        terms.append(
            SoftTermResult(
                trace=compiled.by_subterm[row.subterm_id],
                value=bound.reshape(()),
                exact_value=row.value,
                normalized_weight=row.normalized_weight,
                group=row.group,
            )
        )
    return tuple(terms)


def _weighted_sum(terms: Tuple[SoftTermResult, ...]) -> torch.Tensor:
    """Compute the applicable-mass arithmetic defect.

    Parameters
    ----------
    terms : tuple[SoftTermResult, ...]
        Active compiled terms.

    Returns
    -------
    torch.Tensor
        Scalar weighted mean.
    """

    return torch.stack([term.value * term.normalized_weight for term in terms]).sum()


def _smooth_positive_tensor(value: torch.Tensor, temperature: float) -> torch.Tensor:
    """Apply composition.py's exact C1 positive-part formula in tensor form.

    Parameters
    ----------
    value : torch.Tensor
        Signed group excess.
    temperature : float
        Positive onset scale.

    Returns
    -------
    torch.Tensor
        Zero below the allowance and C1 increasing above it.
    """

    positive = torch.clamp(value, min=0.0)
    return positive.square() / (positive + temperature)


def _compose_soft(
    terms: Tuple[SoftTermResult, ...],
    profile: CompositionProfile,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """Compile the selected frozen composition family over bound terms.

    Parameters
    ----------
    terms : tuple[SoftTermResult, ...]
        Active compiled defect coordinates.
    profile : CompositionProfile
        Exact composition family and parameters.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor or None, torch.Tensor]
        ``l_mean``, optional bottleneck arm, and ``l_total``.
    """

    l_mean = _weighted_sum(terms)
    if profile.family is CompositionFamily.P_MEAN:
        assert profile.power is not None
        power = profile.power
        powered = torch.stack(
            [term.normalized_weight * term.value.pow(power) for term in terms]
        ).sum()
        if float(powered.detach()) == 0.0:
            # composition.py's origin kink rule: at the p-mean's zero-defect
            # origin (p > 1) the exact side publishes the one-sided
            # sensitivity normalized_mass ** (1/p). pow(1/p) back-propagates
            # NaN at exactly zero, so the differentiable branch here is the
            # matching linearization: value 0 at the origin, gradient
            # normalized_weight ** (1/p) per term, exactly the frozen
            # sibling's published derivative.
            return (
                l_mean,
                None,
                torch.stack(
                    [term.value * term.normalized_weight ** (1.0 / power) for term in terms]
                ).sum(),
            )
        return l_mean, None, powered.pow(1.0 / power)

    assert profile.bottleneck_mix is not None
    assert profile.bottleneck_temperature is not None
    group_terms = {}
    for term in terms:
        group_terms.setdefault(term.group, []).append(term)
    debts = []
    for group in sorted(group_terms):
        allowance = profile.group_allowances.get(group)
        if allowance is None:
            raise ValueError(f"missing explicit soft-bottleneck allowance for {group}")
        rows = group_terms[group]
        group_mass = math.fsum(term.normalized_weight for term in rows)
        group_loss = torch.stack(
            [term.value * (term.normalized_weight / group_mass) for term in rows]
        ).sum()
        debts.append(
            _smooth_positive_tensor(
                group_loss - allowance,
                profile.bottleneck_temperature,
            )
        )
    l_bottleneck = torch.stack(debts).sum()
    l_total = (1.0 - profile.bottleneck_mix) * l_mean + profile.bottleneck_mix * l_bottleneck
    return l_mean, l_bottleneck, l_total


def score_v4_soft(
    facet_results: Mapping[str, FacetResult],
    weight_table: WeightTable,
    profile: CompositionProfile,
    *,
    term_tensors: Optional[Mapping[str, torch.Tensor]] = None,
    compiled: Optional[CompiledSurrogateManifest] = None,
) -> SoftScoreResult:
    """Evaluate the analytic surrogate compiled from exact facet rows.

    Exact facet results determine applicability and provide the default
    forward values. A caller may bind scalar tensors for active subterms;
    these tensors must come from contract-authorized differentiable geometry
    if gradients with respect to positions are claimed. No learned component,
    finite-difference estimator, or unregistered smoothing is introduced.

    Parameters
    ----------
    facet_results : mapping[str, FacetResult]
        Closed facet outputs used by exact composition.
    weight_table : WeightTable
        Explicit subterm mass and grouping table.
    profile : CompositionProfile
        Frozen exact composition family.
    term_tensors : mapping[str, torch.Tensor] or None
        Optional differentiable exact-defect coordinates.
    compiled : CompiledSurrogateManifest or None
        Precompiled graph, or ``None`` to compile the production manifest.

    Returns
    -------
    SoftScoreResult
        Differentiable pre-map loss plus exact forward reference and traces.
    """

    program = compiled if compiled is not None else compile_surrogate_manifest()
    exact = compose(facet_results, weight_table, profile)
    terms = _active_terms(exact, program, term_tensors or {})
    l_mean, l_bottleneck, l_total = _compose_soft(terms, profile)
    return SoftScoreResult(
        l_mean=l_mean,
        l_bottleneck=l_bottleneck,
        l_total=l_total,
        exact_composition=exact,
        terms=terms,
        manifest_digest=program.source_digest,
    )
