"""Unit tests for native contest finalist economics."""

from __future__ import annotations

import pytest
import torch

from dagua.layout.ops.pipelines.native_contest_cascade import select_finalists


def _candidates(count: int = 8) -> dict[str, torch.Tensor]:
    """Return deterministic, geometrically distinct candidate layouts.

    Parameters
    ----------
    count : int, default=8
        Number of candidates including the incumbent.

    Returns
    -------
    dict[str, torch.Tensor]
        Candidate positions keyed by stable name, each with shape ``[5, 2]``.
    """
    base = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [3.0, 1.0], [2.0, 4.0]],
        dtype=torch.float64,
    )
    result = {"incumbent": base}
    for index in range(1, count):
        candidate = base.clone()
        candidate[index % 5, index % 2] += float(index) * 0.7
        result[f"arm_{index}"] = candidate
    return result


@pytest.mark.parametrize("family_count", [0, 1, 3])
def test_select_finalists_preserves_each_admitted_family(family_count: int) -> None:
    """Assert zero, one, or three firing families retain a representative.

    Parameters
    ----------
    family_count : int
        Number of gate-admitted families to reserve.
    """
    candidates = _candidates()
    proxy_scores = {name: float(20 - index) for index, name in enumerate(candidates)}
    families = {f"arm_{index + 4}": f"family_{index}" for index in range(family_count)}
    finalists = select_finalists(candidates, proxy_scores, families, 3, ["incumbent"])
    assert finalists[0] == "incumbent"
    assert set(families).issubset(finalists)


def test_select_finalists_preserves_explicit_mandatory_and_proxy_argmax() -> None:
    """Assert explicit mandatory arms and the proxy winner survive a tight cut."""
    candidates = _candidates()
    proxy_scores = {name: float(index) for index, name in enumerate(candidates)}
    finalists = select_finalists(
        candidates,
        proxy_scores,
        {},
        2,
        ["incumbent", "arm_2"],
    )
    assert finalists == ["incumbent", "arm_2", "arm_7"]


def test_select_finalists_deduplicates_scale_equivalent_basin() -> None:
    """Assert pair-distance dedup keeps one representative per proxy basin."""
    candidates = _candidates(4)
    candidates["arm_2"] = candidates["arm_1"] * 3.0 + 7.0
    proxy_scores = {"arm_1": 10.0, "incumbent": 9.0, "arm_2": 8.0, "arm_3": 7.0}
    finalists = select_finalists(candidates, proxy_scores, {}, 3, ["incumbent"])
    assert "arm_1" in finalists
    assert "arm_2" not in finalists
    assert len(finalists) == 3


def test_select_finalists_is_inert_when_budget_is_not_binding() -> None:
    """Assert an unbounded cascade retains every pre-cascade candidate."""
    candidates = _candidates(4)
    candidates["arm_1"] = candidates["incumbent"] * 2.0
    proxy_scores = {name: float(index) for index, name in enumerate(candidates)}
    finalists = select_finalists(candidates, proxy_scores, {}, len(candidates), ["incumbent"])
    assert set(finalists) == set(candidates)
