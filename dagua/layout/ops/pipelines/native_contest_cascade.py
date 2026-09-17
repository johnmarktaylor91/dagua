"""Deterministic proxy cascade for native candidate contests."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch

_FINGERPRINT_PAIR_COUNT = 64
_FINGERPRINT_BASIN_DISTANCE = 0.02


def _sampled_pair_indices(num_nodes: int, pair_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return deterministic node-pair samples for a geometric fingerprint.

    Parameters
    ----------
    num_nodes : int
        Number of candidate positions.
    pair_count : int
        Maximum number of node pairs to return.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Left and right node indices, each with shape ``[S]``.
    """
    if num_nodes < 2 or pair_count <= 0:
        empty = torch.empty(0, dtype=torch.long)
        return empty, empty
    left: list[int] = []
    right: list[int] = []
    # Coprime strides spread samples across the pair space without an RNG or
    # allocating the quadratic set of all node pairs.
    stride = max(1, num_nodes // 2 + 1)
    cursor = 0
    seen: set[tuple[int, int]] = set()
    pair_limit = min(pair_count, num_nodes * (num_nodes - 1) // 2)
    while len(left) < pair_limit:
        first = cursor % num_nodes
        second = (cursor * stride + 1) % num_nodes
        cursor += 1
        if first == second:
            continue
        pair = (min(first, second), max(first, second))
        if pair in seen:
            # The deterministic walk can cycle on small composite node counts.
            # Fall back to lexicographic completion while preserving determinism.
            if cursor > num_nodes * num_nodes:
                for lex_left in range(num_nodes):
                    for lex_right in range(lex_left + 1, num_nodes):
                        lex_pair = (lex_left, lex_right)
                        if lex_pair not in seen:
                            seen.add(lex_pair)
                            left.append(lex_left)
                            right.append(lex_right)
                            if len(left) >= pair_limit:
                                break
                    if len(left) >= pair_limit:
                        break
                break
            continue
        seen.add(pair)
        left.append(pair[0])
        right.append(pair[1])
    return torch.tensor(left, dtype=torch.long), torch.tensor(right, dtype=torch.long)


def geometric_fingerprint(pos: torch.Tensor) -> torch.Tensor:
    """Return a translation- and scale-normalized pair-distance fingerprint.

    Parameters
    ----------
    pos : torch.Tensor
        Candidate positions with shape ``[N, 2]``.

    Returns
    -------
    torch.Tensor
        Normalized sampled pair distances with shape ``[S]``.
    """
    cpu_pos = pos.detach().to(device="cpu", dtype=torch.float64)
    left, right = _sampled_pair_indices(int(cpu_pos.shape[0]), _FINGERPRINT_PAIR_COUNT)
    if left.numel() == 0:
        return torch.zeros(1, dtype=torch.float64)
    distances = torch.linalg.vector_norm(cpu_pos[left] - cpu_pos[right], dim=1)
    finite = distances[torch.isfinite(distances)]
    scale = torch.median(finite) if finite.numel() else torch.tensor(0.0, dtype=torch.float64)
    if not math.isfinite(float(scale.item())) or float(scale.item()) <= 1.0e-12:
        return torch.zeros_like(distances)
    return torch.nan_to_num(distances / scale, nan=0.0, posinf=0.0, neginf=0.0)


def _same_geometric_basin(first: torch.Tensor, second: torch.Tensor) -> bool:
    """Return whether two fingerprints represent the same geometric basin.

    Parameters
    ----------
    first : torch.Tensor
        First normalized fingerprint with shape ``[S]``.
    second : torch.Tensor
        Second normalized fingerprint with shape ``[S]``.

    Returns
    -------
    bool
        ``True`` when the mean sampled distance delta is within the basin cut.
    """
    if first.shape != second.shape:
        return False
    return float(torch.mean(torch.abs(first - second)).item()) <= _FINGERPRINT_BASIN_DISTANCE


def select_finalists(
    candidates: Mapping[str, torch.Tensor],
    proxy_scores: Mapping[str, float],
    families: Mapping[str, str],
    m: int,
    mandatory: Sequence[str],
) -> list[str]:
    """Select full-referee finalists using quotas, proxy rank, and diversity.

    Parameters
    ----------
    candidates : Mapping[str, torch.Tensor]
        Candidate positions keyed by stable arm name; each tensor has shape
        ``[N, 2]``.
    proxy_scores : Mapping[str, float]
        Cheap higher-is-better score for every candidate.
    families : Mapping[str, str]
        Gate-admitted family labels. One best-proxy representative per distinct
        label is mandatory. Candidates absent from this mapping receive no quota.
    m : int
        Target finalist count. Mandatory candidates may make the result exceed it.
    mandatory : Sequence[str]
        Explicit mandatory candidates, normally beginning with ``"incumbent"``.

    Returns
    -------
    list[str]
        Deterministic finalist names in scoring order.

    Raises
    ------
    ValueError
        If inputs are incomplete or ``m`` is not positive.
    """
    if m <= 0:
        raise ValueError("m must be positive")
    names = set(candidates)
    if names != set(proxy_scores):
        raise ValueError("proxy_scores must contain exactly the candidate names")
    unknown = (set(mandatory) | set(families)) - names
    if unknown:
        raise ValueError(f"unknown candidate names: {sorted(unknown)}")
    ranked = sorted(names, key=lambda name: (-float(proxy_scores[name]), name))
    selected: list[str] = []
    slot_selected: set[str] = set()

    def append_once(name: str) -> None:
        """Append one candidate if it has not already been selected."""
        if name not in selected:
            selected.append(name)

    for name in mandatory:
        append_once(name)
    if "incumbent" in selected:
        slot_selected.add("incumbent")
    for family in sorted(set(families.values())):
        representative = next(name for name in ranked if families.get(name) == family)
        append_once(representative)
    if ranked:
        append_once(ranked[0])
        slot_selected.add(ranked[0])
    # When the full-referee budget is not binding, retain every arm. This is
    # the cascade's byte-inert path because winner selection sees the exact
    # same candidate set as the pre-cascade contest.
    if m >= len(candidates):
        for name in ranked:
            append_once(name)
        return [
            *(name for name in ("incumbent",) if name in selected),
            *(name for name in ranked if name != "incumbent" and name in selected),
        ]

    fingerprints = {name: geometric_fingerprint(candidates[name]) for name in names}
    for name in ranked:
        if len(slot_selected) >= m:
            break
        if name in selected:
            continue
        if any(
            _same_geometric_basin(fingerprints[name], fingerprints[chosen])
            for chosen in slot_selected
        ):
            continue
        selected.append(name)
        slot_selected.add(name)
    return [
        *(name for name in ("incumbent",) if name in selected),
        *(name for name in ranked if name != "incumbent" and name in selected),
    ]


__all__ = ["geometric_fingerprint", "select_finalists"]
