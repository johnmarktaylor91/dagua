"""Compile the frozen facet manifest into traceable surrogate terms."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Tuple

from dagua.eval.ruler_v4.contracts import CONTRACTS, ContractMetadata


class CompilationRule(str, Enum):
    """Rules admitted by the analytic surrogate compiler."""

    EXACT_DEFECT_IDENTITY = "exact_defect_identity"


@dataclass(frozen=True)
class SurrogateTermTrace:
    """Trace one compiled term to its frozen facet contract.

    Parameters
    ----------
    facet_id : str
        Frozen facet contract id.
    subterm_id : str
        Frozen score-visible subterm id.
    contract_sha256 : str
        Frozen contract byte digest.
    compilation_rule : CompilationRule
        Analytic rule applied to the exact defect coordinate.
    smoothing : str or None
        Spec-named smoothing. ``None`` means no relaxation was authorized.
    """

    facet_id: str
    subterm_id: str
    contract_sha256: str
    compilation_rule: CompilationRule
    smoothing: None = None


@dataclass(frozen=True)
class CompiledSurrogateManifest:
    """Immutable surrogate program compiled from the facet graph manifest.

    Parameters
    ----------
    source_digest : str
        Canonical digest of the consumed contract/subterm graph.
    terms : tuple[SurrogateTermTrace, ...]
        Contract-ordered compiled terms.
    by_subterm : mapping[str, SurrogateTermTrace]
        Read-only subterm lookup.
    """

    source_digest: str
    terms: Tuple[SurrogateTermTrace, ...]
    by_subterm: Mapping[str, SurrogateTermTrace]


def _source_digest(contracts: Mapping[str, ContractMetadata]) -> str:
    """Hash the manifest fields that determine surrogate term identity.

    Parameters
    ----------
    contracts : mapping[str, ContractMetadata]
        Frozen facet graph metadata.

    Returns
    -------
    str
        SHA-256 of canonical contract/subterm rows.
    """

    rows = [
        {
            "contract_sha256": contract.sha256,
            "facet_id": facet_id,
            "subterms": list(contract.scored_subterms),
        }
        for facet_id, contract in contracts.items()
    ]
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def compile_surrogate_manifest(
    contracts: Mapping[str, ContractMetadata] = CONTRACTS,
) -> CompiledSurrogateManifest:
    """Compile every manifest subterm without a parallel hand-written inventory.

    The r4 spec names no position-level smoothing for any facet. Consequently
    the compiler admits only the analytic identity map over an exact defect
    coordinate. Callers may bind differentiable tensors produced by future
    contract-authorized geometry rules, but this compiler never invents one.

    Parameters
    ----------
    contracts : mapping[str, ContractMetadata]
        Facet graph manifest. Production uses :data:`CONTRACTS`, which is
        validated against the frozen generated inventory by ``registry.py``.

    Returns
    -------
    CompiledSurrogateManifest
        Complete traceable surrogate term program.

    Raises
    ------
    ValueError
        If a contract or subterm identity is empty or duplicated.
    """

    terms = []
    seen = set()
    for facet_id, contract in contracts.items():
        if not facet_id or not contract.sha256:
            raise ValueError("surrogate contracts require nonempty identities")
        for subterm_id in contract.scored_subterms:
            if not subterm_id or subterm_id in seen:
                raise ValueError(f"duplicate or empty surrogate subterm: {subterm_id!r}")
            seen.add(subterm_id)
            terms.append(
                SurrogateTermTrace(
                    facet_id=facet_id,
                    subterm_id=subterm_id,
                    contract_sha256=contract.sha256,
                    compilation_rule=CompilationRule.EXACT_DEFECT_IDENTITY,
                )
            )
    frozen_terms = tuple(terms)
    return CompiledSurrogateManifest(
        source_digest=_source_digest(contracts),
        terms=frozen_terms,
        by_subterm=MappingProxyType({term.subterm_id: term for term in frozen_terms}),
    )
