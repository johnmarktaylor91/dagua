"""Stress-family contest arms: stress-SGD at k seeds, maxent-stress, ELK-stress.

Sprint2 W2-2 (plan cluster C9 plus the D2 staged call). The dev tie band is
dominated by three in-house stress-family engines -- ``classic_maxent_stress``,
``elk_stress(_reimpl)``, and ``classic_stress_sgd`` -- plus the external
``sgd2`` adapter. This module puts exactly those basins INSIDE the native
contest as ordinary refereed candidates:

- ``stress_sgd_k_seed{i}`` -- ``layout_stress_sgd_pipeline`` at the frozen
  seed bank (seed 42 first: the classic adapter's historical default, so the
  first member is the field parity floor). Deliberately labeled
  ``stress_sgd_k``, NEVER ``sgd2``: per the D2 call this arm attacks the
  stress-SGD FAMILY basin and must not be conflated with a faithful port of
  the external ``sgd2`` schedule (that port is parked as P8).
- ``maxent_stress_seed{i}`` -- ``layout_maxent_stress_pipeline`` with the
  classic adapter's winning parameters (steps=200, alpha=1.0).
- ``elk_stress_arm`` -- ``layout_elk_stress_pipeline`` at pipeline defaults,
  matching ``elk_stress_reimpl``; the solve is deterministic, one candidate.

All three pipelines are shared with reimplemented field competitors, so they
are CALLED with parameters and never modified (fidelity invariant). Outputs
are similarity-rescaled into node-box units (center + uniform scale, the
W1-A/circo-calibration precedent: stress-SGD emits unit-scale coordinates the
shared degeneracy guard would otherwise reject before the referee saw them).

Everything is a REFEREED CANDIDATE: no route switching, the incumbent wins
ties, and on any row where the gate is closed no code in this module runs
(byte-inert gate-closed path).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Collection, Dict, Mapping, Optional, Sequence

import torch

from dagua.layout.ops.pipelines.native_sparse_infrastructure import _scale_to_node_units
from dagua.layout.ops.state import LayoutProblem

if TYPE_CHECKING:  # pragma: no cover - typing only
    from dagua.config import LayoutConfig

_LOGGER = logging.getLogger(__name__)

# Structural gate bounds. The tie-band evidence sits on small connected
# rows and every arm needs the exact APSP metric, so the ceiling keeps the
# dense O(n^2) distance work trivially cheap; 800 is the plan's declared
# band, far above every targeted row (n <= ~130 across all 14 ties).
STRESS_FAMILY_MIN_NODES = 4
STRESS_FAMILY_MAX_NODES = 800
# Low-layering admission for the DIRECTED contest, COUSIN-FITTED (review
# F4): scripts/w22_cousin_layering_fit.py on the 25 directed training
# cousins (artifact: tests/data/w22_layering_fit.json, regenerated at the
# packet head against the corrected-W2-1 native baseline) found competitive
# stress rows at avg widths {1.0, 1.43, 3.67} interleaved with hopeless
# rows -- NO width cut separates competitive from hopeless turf. The fitted
# threshold is therefore the maximal cut that loses zero competitive
# cousins: 1.0, the arithmetic floor of any measured layering (n/num_layers
# >= 1). Width thus provides no exclusion beyond the classifier's 0.0
# "unmeasured" default, which still fails closed. Cyclic digraphs have no
# faithful layering at all, so they bypass the width check by construction.
# tests/test_native_stress_family_arm.py pins this constant to the artifact.
LOW_LAYERING_MIN_AVG_LAYER_WIDTH = 1.0
# Frozen seed banks (constants, never RNG-derived). 42 first = the classic
# adapters' historical default seed, making candidate 0 the parity floor
# with the field engines; the rest are arbitrary distinct constants.
STRESS_SGD_SEED_BANK = (42, 7, 1379)
MAXENT_SEED_BANK = (42, 7)
# Classic-adapter parity parameters (classic_competitor.py): the exact
# configuration the winning field engines ran with.
STRESS_SGD_STEPS = 300
MAXENT_STEPS = 200
MAXENT_ALPHA = 1.0
# ELK stress iterates to an epsilon rather than a step count; this frozen
# prior charges the deterministic solve as the same order of stress work as
# its family siblings in the calibrated cost table.
ELK_COST_PRIOR_STEPS = 300

# Candidate-name prefixes owned by this arm, used by the contest seams to
# attach family-quota labels to every registered variant.
_STRESS_FAMILY_PREFIXES = ("stress_sgd_k_seed", "maxent_stress_seed", "elk_stress_arm")


def _is_connected(edge_index: torch.Tensor, num_nodes: int) -> bool:
    """Return whether the undirected graph is a single connected component.

    The classifier's ``num_components``/``has_dominant_component`` fields are
    fast-pathed for ``E > N - 1`` inputs (both report connected regardless),
    so the gate computes exact union-find connectivity itself -- trivial at
    the gate's size ceiling.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    bool
        ``True`` when every node is reachable from every other.
    """
    if num_nodes <= 1:
        return True
    if edge_index.numel() == 0:
        return False
    parents = list(range(num_nodes))

    def _root(node: int) -> int:
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node

    merged = 0
    for src, dst in edge_index.detach().to(device="cpu", dtype=torch.long).t().tolist():
        if src == dst:
            continue
        src_root = _root(int(src))
        dst_root = _root(int(dst))
        if src_root != dst_root:
            parents[src_root] = dst_root
            merged += 1
            if merged == num_nodes - 1:
                return True
    return merged == num_nodes - 1


def stress_family_arm_admitted(problem: LayoutProblem) -> bool:
    """Return whether the stress-family arm may build candidates.

    The gate is input-only structure: classifier output present, a single
    connected component (every arm optimizes the exact APSP metric, which
    does not exist across components; checked exactly, see
    :func:`_is_connected`), and the size band that keeps the dense distance
    work trivially cheap.

    Parameters
    ----------
    problem : LayoutProblem
        Prepared layout problem carrying the classifier output.

    Returns
    -------
    bool
        ``True`` when every structural condition holds.
    """
    structure = problem.structure
    if structure is None:
        return False
    num_nodes = int(problem.num_nodes)
    if num_nodes < STRESS_FAMILY_MIN_NODES or num_nodes > STRESS_FAMILY_MAX_NODES:
        return False
    if problem.edge_index.numel() == 0:
        return False
    if not _is_connected(problem.edge_index, num_nodes):
        return False
    return True


def stress_family_directed_admitted(problem: LayoutProblem) -> bool:
    """Return whether the DIRECTED contest admits the stress-family arm.

    Directed admission adds the cousin-fitted low-layering condition on top
    of :func:`stress_family_arm_admitted`. The training-cousin fit found
    competitive stress rows down to avg width 1.0 (no separating width cut
    exists), so the fitted threshold admits every MEASURED acyclic layering
    and the condition only excludes the classifier's 0.0 "unmeasured"
    default, which fails closed. Cyclic digraphs have no faithful layering
    and are admitted by construction.

    Parameters
    ----------
    problem : LayoutProblem
        Prepared directed layout problem.

    Returns
    -------
    bool
        ``True`` when the shared gate holds and layering is weak.
    """
    if not stress_family_arm_admitted(problem):
        return False
    structure = problem.structure
    if not bool(getattr(structure, "is_directed_acyclic", True)):
        return True
    avg_layer_width = float(getattr(structure, "avg_layer_width", 0.0))
    return avg_layer_width >= LOW_LAYERING_MIN_AVG_LAYER_WIDTH


@dataclass(frozen=True)
class StressFamilyAdmission:
    """Ledger-admitted work for the three stress-family arms.

    Attributes
    ----------
    stress_sgd_seeds : tuple[int, ...]
        Admitted frozen prefix of :data:`STRESS_SGD_SEED_BANK` (empty when
        even the single-seed base package was rejected).
    maxent_seeds : tuple[int, ...]
        Admitted frozen prefix of :data:`MAXENT_SEED_BANK`.
    elk_admitted : bool
        Whether the deterministic ELK-stress package was admitted.
    """

    stress_sgd_seeds: tuple[int, ...]
    maxent_seeds: tuple[int, ...]
    elk_admitted: bool

    @property
    def any_admitted(self) -> bool:
        """Return whether any family package was admitted."""
        return bool(self.stress_sgd_seeds) or bool(self.maxent_seeds) or self.elk_admitted


def admit_stress_family_packages(
    problem: LayoutProblem,
    config: Optional["LayoutConfig"],
    device_class: str,
) -> StressFamilyAdmission:
    """Admit the stress-family packages through the DWU ledger ONLY.

    Never wall/process-time conditional (review F2): identical input plus
    ledger state admits identical work regardless of elapsed time or machine
    load. Every underlying ledger call passes ``ledger_only=True``, which
    skips :func:`native_budget.admit_native_work`'s live wall-reserve veto
    (the residual live-wall path the re-review found); admission is a pure
    function of the ledger. Each family is priced as one all-or-nothing
    aggregate package --
    every frozen trajectory's generation cost plus the reserved referee
    seats -- through W2-1's seed-family plumbing, charged before any
    generation happens. Multi-seed families fall back through deterministic
    frozen prefixes (``k -> 1``), never partial mid-generation stops.

    Parameters
    ----------
    problem : LayoutProblem
        Prepared layout problem (gate already checked by the caller).
    config : LayoutConfig, optional
        Native configuration carrying the deterministic ledger.
    device_class : str
        Frozen cost-table device axis (``"cpu"`` or ``"cuda"``).

    Returns
    -------
    StressFamilyAdmission
        Admitted per-family work. Gate-closed callers never reach this, so
        the ledger stays untouched on gate-closed rows (byte-inert).
    """
    from dagua.layout.ops.pipelines.native_budget import admit_native_work
    from dagua.layout.ops.pipelines.native_cost_model import estimate_native_work_cost
    from dagua.layout.ops.pipelines.native_seed_replication import (
        admit_seed_family,
        replicated_work_cost,
    )

    stress_sgd_cost = estimate_native_work_cost(
        problem,
        "stress",
        {"steps": STRESS_SGD_STEPS, "samples": None},
        device_class,
    )
    stress_sgd_seeds = admit_seed_family(
        config,
        stress_sgd_cost,
        "stress_sgd_k",
        STRESS_SGD_SEED_BANK,
        ledger_only=True,
    )
    maxent_cost = estimate_native_work_cost(
        problem,
        "stress",
        {"steps": MAXENT_STEPS, "samples": None},
        device_class,
    )
    maxent_seeds = admit_seed_family(
        config,
        maxent_cost,
        "maxent_stress",
        MAXENT_SEED_BANK,
        ledger_only=True,
    )
    elk_cost = estimate_native_work_cost(
        problem,
        "stress",
        {"steps": ELK_COST_PRIOR_STEPS, "samples": None},
        device_class,
    )
    elk_admitted = admit_native_work(
        config,
        replicated_work_cost(elk_cost, 1),
        "optional_stress_family_elk",
        ledger_only=True,
    )
    return StressFamilyAdmission(
        stress_sgd_seeds=tuple(stress_sgd_seeds),
        maxent_seeds=tuple(maxent_seeds),
        elk_admitted=bool(elk_admitted),
    )


def stress_family_replicated_family(
    candidate_name: str,
    stress_sgd_seeds: tuple[int, ...],
    maxent_seeds: tuple[int, ...],
) -> Optional[str]:
    """Return the replication-map label for one base candidate name.

    Only multi-seed families register in the contest's replicated-candidate
    map (the W2-1 policy): the within-family proxy cull then retains the
    best-proxy raw parity floor plus one best variant per family, bounding
    the referee load the aggregate package reserved. The deterministic
    single-shot ELK candidate stays out of the map like every other
    single-shot arm.

    Parameters
    ----------
    candidate_name : str
        Base candidate name from :func:`build_stress_family_candidates`.
    stress_sgd_seeds : tuple[int, ...]
        Admitted stress-SGD seed prefix.
    maxent_seeds : tuple[int, ...]
        Admitted maxent seed prefix.

    Returns
    -------
    str | None
        Telemetry-family label for replicated candidates, else ``None``.
    """
    if candidate_name.startswith("stress_sgd_k_seed") and len(stress_sgd_seeds) > 1:
        return "stress_sgd_k"
    if candidate_name.startswith("maxent_stress_seed") and len(maxent_seeds) > 1:
        return "maxent_stress"
    return None


def stress_family_candidate_prefix(candidate_name: str) -> Optional[str]:
    """Return the owning arm prefix for one candidate/variant name.

    Parameters
    ----------
    candidate_name : str
        Registered candidate name, possibly carrying a cleanup-variant
        suffix (``_raw``, ``_prism``, ``_convergent``).

    Returns
    -------
    str | None
        The matching prefix from this module, or ``None`` for names this
        arm does not own.
    """
    for prefix in _STRESS_FAMILY_PREFIXES:
        if candidate_name.startswith(prefix):
            return prefix
    return None


def stress_family_quota_entries(
    challenger_names: "Sequence[str]",
    legacy_finalists: "Collection[str]",
    existing_quotas: "Mapping[str, str]",
) -> Dict[str, str]:
    """Return one best-proxy quota entry per unrepresented stress family.

    Review F3: every admitted stress family gets exactly one honest-referee
    representative, composed with (never evicting) existing quota entries.
    Families normalize by arm prefix, so seed and cleanup variants share one
    family identity; a family already represented among the legacy finalists
    or existing quotas needs no additional seat. Mandatory entries may push
    the finalist count past the target, which ``select_finalists`` permits.

    Parameters
    ----------
    challenger_names : Sequence[str]
        Proxy-ranked candidate names, best first.
    legacy_finalists : Collection[str]
        Names already admitted by the legacy family cut.
    existing_quotas : Mapping[str, str]
        Quota entries reserved by earlier arms; never overwritten.

    Returns
    -------
    dict[str, str]
        New entries keyed by candidate name, labeled with the normalized
        stress-family prefix.
    """
    represented = {
        prefix
        for name in (*legacy_finalists, *existing_quotas)
        if (prefix := stress_family_candidate_prefix(name)) is not None
    }
    entries: Dict[str, str] = {}
    for name in challenger_names:
        prefix = stress_family_candidate_prefix(name)
        if prefix is None or prefix in represented or name in existing_quotas:
            continue
        represented.add(prefix)
        entries[name] = prefix
    return entries


def build_stress_family_candidates(
    problem: LayoutProblem,
    node_sep: float,
    stress_sgd_seeds: tuple[int, ...] = STRESS_SGD_SEED_BANK,
    maxent_seeds: tuple[int, ...] = MAXENT_SEED_BANK,
    include_elk: bool = True,
) -> Dict[str, torch.Tensor]:
    """Build the stress-family candidate drawings.

    Every pipeline is called UNCHANGED with the classic-adapter parity
    parameters, then similarity-rescaled into node-box units. A failing
    family is skipped (logged) without sinking the others; the enclosing
    contest seams additionally guard the whole block.

    Parameters
    ----------
    problem : LayoutProblem
        Prepared layout problem (gate already checked by the caller).
    node_sep : float
        Configured node separation in point units, used by the rescale.
    stress_sgd_seeds : tuple[int, ...], default=STRESS_SGD_SEED_BANK
        Ledger-admitted stress-SGD frozen seed prefix; only admitted
        trajectories are generated (review F2: work is charged, then built).
    maxent_seeds : tuple[int, ...], default=MAXENT_SEED_BANK
        Ledger-admitted maxent frozen seed prefix.
    include_elk : bool, default=True
        Whether the deterministic ELK-stress package was admitted.

    Returns
    -------
    dict[str, torch.Tensor]
        Candidate positions keyed by stable arm name, each ``[N, 2]``.
    """
    cpu_edges = problem.edge_index.detach().to(device="cpu", dtype=torch.long)
    num_nodes = int(problem.num_nodes)
    cpu_sizes = (
        problem.node_sizes.detach().to(device="cpu") if problem.node_sizes is not None else None
    )
    cpu_weights = (
        problem.edge_weights.detach().to(device="cpu") if problem.edge_weights is not None else None
    )
    candidates: Dict[str, torch.Tensor] = {}

    try:
        from dagua.layout.ops.pipelines.stress_sgd import layout_stress_sgd_pipeline

        for seed in stress_sgd_seeds:
            result = layout_stress_sgd_pipeline(
                cpu_edges,
                num_nodes,
                node_sizes=cpu_sizes,
                edge_weights=cpu_weights,
                steps=STRESS_SGD_STEPS,
                seed=seed,
                fidelity_mode=True,
            )
            pos = result[0] if isinstance(result, tuple) else result
            candidates[f"stress_sgd_k_seed{seed}"] = _scale_to_node_units(pos, problem, node_sep)
    except Exception:  # noqa: BLE001 -- one failed family never sinks the arm
        _LOGGER.warning("stress_sgd_k candidates failed", exc_info=True)

    try:
        from dagua.layout.ops.pipelines.maxent_stress import layout_maxent_stress_pipeline

        for seed in maxent_seeds:
            pos = layout_maxent_stress_pipeline(
                cpu_edges,
                num_nodes,
                node_sizes=cpu_sizes,
                edge_weights=cpu_weights,
                steps=MAXENT_STEPS,
                alpha=MAXENT_ALPHA,
                seed=seed,
            )
            candidates[f"maxent_stress_seed{seed}"] = _scale_to_node_units(pos, problem, node_sep)
    except Exception:  # noqa: BLE001 -- one failed family never sinks the arm
        _LOGGER.warning("maxent_stress candidates failed", exc_info=True)

    if include_elk:
        try:
            from dagua.layout.ops.pipelines.elk_stress import layout_elk_stress_pipeline

            pos = layout_elk_stress_pipeline(
                cpu_edges,
                num_nodes,
                node_sizes=cpu_sizes,
                edge_weights=cpu_weights,
            )
            candidates["elk_stress_arm"] = _scale_to_node_units(pos, problem, node_sep)
        except Exception:  # noqa: BLE001 -- one failed family never sinks the arm
            _LOGGER.warning("elk_stress candidate failed", exc_info=True)

    return candidates


def stress_family_parity_floor(candidate_name: str) -> bool:
    """Return whether a candidate is a family parity floor.

    Parity floors are the exact classic-adapter configurations (seed 42 /
    deterministic ELK solve). Their RAW drawings must reach the honest
    referee so the field engine's own drawing is always represented
    (the W1-A raw-t-FDP-seat precedent).

    Parameters
    ----------
    candidate_name : str
        Base candidate name from :func:`build_stress_family_candidates`.

    Returns
    -------
    bool
        ``True`` for the three parity-floor candidates.
    """
    return candidate_name in {
        f"stress_sgd_k_seed{STRESS_SGD_SEED_BANK[0]}",
        f"maxent_stress_seed{MAXENT_SEED_BANK[0]}",
        "elk_stress_arm",
    }
