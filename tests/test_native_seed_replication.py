"""Tests for deterministic stochastic-arm seed replication."""

from __future__ import annotations

from pathlib import Path

import torch

from dagua.config import LayoutConfig
from dagua.layout.ops.pipelines.native_budget import install_budget_ledger
from dagua.layout.ops.pipelines.native_cost_model import NativeWorkCost
from dagua.layout.ops.pipelines.native_seed_replication import (
    FROZEN_CONTEST_SEED_OFFSETS,
    SEED_COUNT_OVERRIDE_ATTR,
    admit_seed_family,
    frozen_seed_bank,
    replicated_work_cost,
    retained_seed_replicas,
)


def _replica_candidates(count: int) -> dict[str, torch.Tensor]:
    """Return geometrically distinct candidate tensors for one seed family.

    Parameters
    ----------
    count : int
        Number of seed candidates to create.

    Returns
    -------
    dict[str, torch.Tensor]
        Candidate positions keyed by seed name, each shaped ``[4, 2]``.
    """
    base = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [3.0, 1.0]],
        dtype=torch.float64,
    )
    candidates: dict[str, torch.Tensor] = {}
    for index in range(count):
        candidate = base.clone()
        candidate[index % 4, index % 2] += 0.5 * float(index + 1)
        candidates[f"arm_seed{index}"] = candidate
    return candidates


def test_frozen_seed_bank_preserves_baseline_and_supports_k1() -> None:
    """The bank is frozen and its forced-k1 prefix disables replication."""
    config = LayoutConfig()
    assert frozen_seed_bank(config, 42) == tuple(
        42 + offset for offset in FROZEN_CONTEST_SEED_OFFSETS
    )
    setattr(config, SEED_COUNT_OVERRIDE_ATTR, 1)
    assert frozen_seed_bank(config, 42) == (42,)


def test_replicated_work_cost_prices_all_generation_but_two_scores() -> None:
    """The family package reserves generation for k and referee work for two."""
    base = NativeWorkCost("probe", 3.0, 2.0, metadata={"source": "test"})
    package = replicated_work_cost(base, 5)
    assert package.generation_dwu == 15.0
    assert package.reserved_score_dwu == 4.0
    assert package.metadata == {"source": "test", "seed_count": 5, "seed_finalist_count": 2}


def test_seed_family_admission_is_all_or_nothing() -> None:
    """One ledger event admits every replica when the complete package fits."""
    base = NativeWorkCost("probe", 1.0, 0.5)
    seeds = (42, 43, 44, 59, 85)

    admitted_config = LayoutConfig()
    install_budget_ledger(admitted_config, timeout_s=10.0, safety=0.9)
    assert admit_seed_family(admitted_config, base, "probe", seeds) == seeds
    admitted_ledger = getattr(admitted_config, "_dagua_native_budget_ledger")
    assert admitted_ledger.spent_dwu == 6.0
    assert [event["event"] for event in admitted_ledger.event_log] == ["admit"]


def test_seed_family_admission_falls_back_to_affordable_base_arm() -> None:
    """An unaffordable replication increment preserves the pre-packet base arm."""
    base = NativeWorkCost("probe", 1.0, 0.5)
    seeds = (42, 43, 44, 59, 85)
    config = LayoutConfig()
    install_budget_ledger(config, timeout_s=2.0, safety=0.9)

    assert admit_seed_family(config, base, "probe", seeds) == (42,)
    ledger = getattr(config, "_dagua_native_budget_ledger")
    assert ledger.spent_dwu == 1.5
    assert [event["event"] for event in ledger.event_log] == ["skip", "skip", "admit"]
    assert [event["metadata"]["seed_count"] for event in ledger.event_log] == [5, 3, 1]


def test_seed_family_admission_uses_largest_affordable_frozen_prefix() -> None:
    """A three-seed package wins over the base arm when it fits the ledger."""
    base = NativeWorkCost("probe", 1.0, 0.5)
    seeds = (42, 43, 44, 59, 85)
    config = LayoutConfig()
    install_budget_ledger(config, timeout_s=5.0, safety=0.9)

    assert admit_seed_family(config, base, "probe", seeds) == seeds[:3]
    ledger = getattr(config, "_dagua_native_budget_ledger")
    assert ledger.spent_dwu == 4.0
    assert [event["event"] for event in ledger.event_log] == ["skip", "admit"]
    assert [event["metadata"]["seed_count"] for event in ledger.event_log] == [5, 3]


def test_proxy_cull_keeps_two_replicas_and_every_nonreplicated_arm() -> None:
    """Within-family W1-C pruning keeps two representatives and all other arms."""
    candidates = _replica_candidates(5)
    candidates["incumbent"] = torch.zeros((4, 2), dtype=torch.float64)
    candidates["deterministic_planar"] = torch.ones((4, 2), dtype=torch.float64)
    proxy_scores = {name: float(index) for index, name in enumerate(candidates)}
    families = {f"arm_seed{index}": "stochastic_arm" for index in range(5)}
    retained = retained_seed_replicas(candidates, proxy_scores, families)
    assert "incumbent" in retained
    assert "deterministic_planar" in retained
    assert len(retained & set(families)) == 2


def test_proxy_cull_keeps_best_raw_replica_within_family_quota() -> None:
    """The proxy cull cannot evict a replicated family's raw fidelity floor."""
    candidates = _replica_candidates(5)
    candidates.update(
        {
            "arm_seed0_raw": candidates["arm_seed0"].clone(),
            "arm_seed1_raw": candidates["arm_seed1"].clone(),
        }
    )
    proxy_scores = {name: float(index + 10) for index, name in enumerate(_replica_candidates(5))}
    proxy_scores.update({"arm_seed0_raw": -2.0, "arm_seed1_raw": -1.0})
    families = {name: "stochastic_arm" for name in candidates}

    retained = retained_seed_replicas(candidates, proxy_scores, families)

    assert len(retained) == 2
    assert "arm_seed1_raw" in retained


def test_proxy_cull_is_inert_for_single_seed() -> None:
    """A k=1 family preserves the complete candidate set without reordering."""
    candidates = _replica_candidates(1)
    candidates["incumbent"] = torch.zeros((4, 2), dtype=torch.float64)
    proxy_scores = {name: float(index) for index, name in enumerate(candidates)}
    retained = retained_seed_replicas(
        candidates,
        proxy_scores,
        {"arm_seed0": "stochastic_arm"},
    )
    assert retained == set(candidates)


def test_done_note_labels_k1_as_replication_off_not_prepacket_parity() -> None:
    """The completion note must not claim that forced k=1 is the old engine."""
    done_note = Path(__file__).parents[1] / "W2_1_DONE.md"
    text = done_note.read_text(encoding="utf-8")
    assert "replication-machinery-off" in text
    assert "byte-inert `k=1`" not in text
    assert "baseline-byte-inert" not in text
