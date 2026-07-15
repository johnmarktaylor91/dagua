"""Rotating family-stratified validation protocol for router changes.

The native router-v2 shortlist (``dagua.layout.ops.pipelines.dagua_native``)
uses only STRUCTURAL features -- degrees, diameters, modularity, planarity --
never graph names or corpus identity. This module supplies the validation
side of that honesty contract:

**Promotion rule** (from the r2 charter): a routing-rule or threshold change
lands only when it improves best-or-tied on at least ``min_improving_folds``
of the training folds AND does not regress the held-out fold. The held fold
ROTATES each round (``rotation`` parameter), so no fixed subset of the corpus
can be silently tuned against.

**Why family-stratified**: routing errors are family-shaped (a threshold that
helps meshes can hurt scale-free graphs). Stratifying every fold by family
means each fold is a miniature corpus and a family cliff shows up in every
fold rather than hiding in one.

The fold assignment is used only for VALIDATION bookkeeping by measurement
scripts; nothing in the layout path ever reads it. Assignments are
deterministic (sorted names, round-robin) so every agent computes identical
folds without shared state.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping

DEFAULT_NUM_FOLDS = 5
DEFAULT_MIN_IMPROVING_FOLDS = 4


def family_stratified_folds(
    families: Mapping[str, str],
    num_folds: int = DEFAULT_NUM_FOLDS,
    rotation: int = 0,
) -> dict[str, int]:
    """Assign each graph to a fold, stratified by family, deterministically.

    Graphs are grouped by family, sorted by name within each family, and
    dealt round-robin into folds with a ``rotation`` offset. Rotating shifts
    every assignment, so across ``num_folds`` rounds each graph spends one
    round in the held-out fold.

    Parameters
    ----------
    families : Mapping[str, str]
        Graph name to family name.
    num_folds : int, default=5
        Number of folds.
    rotation : int, default=0
        Round index; changes which fold each graph lands in.

    Returns
    -------
    dict[str, int]
        Graph name to fold index in ``[0, num_folds)``.
    """
    if num_folds < 2:
        raise ValueError("num_folds must be at least 2 for held-out validation.")
    by_family: dict[str, list[str]] = defaultdict(list)
    for name in sorted(families):
        by_family[families[name]].append(name)
    assignment: dict[str, int] = {}
    for family in sorted(by_family):
        for position, name in enumerate(by_family[family]):
            assignment[name] = (position + rotation) % num_folds
    return assignment


def held_out_fold(rotation: int, num_folds: int = DEFAULT_NUM_FOLDS) -> int:
    """Return the held-out fold index for a validation round.

    Parameters
    ----------
    rotation : int
        Round index.
    num_folds : int, default=5
        Number of folds.

    Returns
    -------
    int
        Fold index excluded from tuning this round.
    """
    return rotation % num_folds


def routing_change_accepted(
    per_fold_best_or_tied_delta: Mapping[int, int],
    rotation: int,
    num_folds: int = DEFAULT_NUM_FOLDS,
    min_improving_folds: int = DEFAULT_MIN_IMPROVING_FOLDS,
) -> bool:
    """Apply the promotion rule to measured per-fold best-or-tied deltas.

    Parameters
    ----------
    per_fold_best_or_tied_delta : Mapping[int, int]
        Fold index to best-or-tied delta (candidate minus baseline).
    rotation : int
        Round index (selects the held-out fold).
    num_folds : int, default=5
        Number of folds.
    min_improving_folds : int, default=4
        Training folds that must strictly improve or stay equal, with at
        least one strict improvement overall.

    Returns
    -------
    bool
        ``True`` when the change may land.
    """
    held = held_out_fold(rotation, num_folds)
    if per_fold_best_or_tied_delta.get(held, 0) < 0:
        return False
    training_deltas = [delta for fold, delta in per_fold_best_or_tied_delta.items() if fold != held]
    non_regressing = sum(1 for delta in training_deltas if delta >= 0)
    strictly_improving = any(delta > 0 for delta in training_deltas)
    return non_regressing >= min_improving_folds and strictly_improving


__all__ = [
    "DEFAULT_MIN_IMPROVING_FOLDS",
    "DEFAULT_NUM_FOLDS",
    "family_stratified_folds",
    "held_out_fold",
    "routing_change_accepted",
]
