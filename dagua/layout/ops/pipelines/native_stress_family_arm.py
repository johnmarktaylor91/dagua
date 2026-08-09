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
from typing import Dict, Optional

import torch

from dagua.layout.ops.pipelines.native_sparse_infrastructure import _scale_to_node_units
from dagua.layout.ops.state import LayoutProblem

_LOGGER = logging.getLogger(__name__)

# Structural gate bounds. The tie-band evidence sits on small connected
# rows and every arm needs the exact APSP metric, so the ceiling keeps the
# dense O(n^2) distance work trivially cheap; 800 is the plan's declared
# band, far above every targeted row (n <= ~130 across all 14 ties).
STRESS_FAMILY_MIN_NODES = 4
STRESS_FAMILY_MAX_NODES = 800
# Low-layering admission for the DIRECTED contest: symmetric-distance
# engines are competitive with rank-based drawing only when the precedence
# structure is shallow relative to size. avg_layer_width >= 2.0 is exactly
# num_layers <= n/2 (each rank of the longest-path layering packs on
# average two-plus nodes); deep chain/tree rows stay closed, and the
# classifier's 0.0 "unmeasured" default fails closed. Cyclic digraphs have
# no faithful layering at all, so they are low-layering by construction.
LOW_LAYERING_MIN_AVG_LAYER_WIDTH = 2.0
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
# Legacy structural prior seconds for the directed opaque-arm cost table.
STRESS_FAMILY_PRIOR_S = 2.0

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

    Directed admission adds the low-layering condition on top of
    :func:`stress_family_arm_admitted`: the tie-band evidence is stress
    engines beating rank-based drawing on directed rows whose layering is
    shallow (wide) relative to size. Deep hierarchies keep the gate closed
    so the layered incumbent's turf stays byte-inert.

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


def build_stress_family_candidates(
    problem: LayoutProblem,
    node_sep: float,
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

        for seed in STRESS_SGD_SEED_BANK:
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

        for seed in MAXENT_SEED_BANK:
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
