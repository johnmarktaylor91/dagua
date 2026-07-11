"""FM^3 multilevel force-directed layout pipeline."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch

from dagua.layout.ops.base import Op, Pipeline
from dagua.layout.ops.cluster_geometry import ClusterTree
from dagua.layout.ops.fmmm import (
    _FinalizeFMMMPositions,
    _InitializeCoarsestLevel,
    _InitializeFMMMState,
    _InitializeFMMMStateConfig,
    _RandomNodeSet,
    _RefineCoarsestLevel,
    _SingleLevelFallback,
    _UncoarsenLoop,
)
from dagua.layout.ops.graph_utils import layout_device as _layout_device
from dagua.layout.ops.state import (
    ExecutionPlan,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)
from dagua.layout.ops.taxonomy import OpCategory

_FDP_COMPOUND_EDGE_ATTACHMENTS_KEY = "fmmm_fdp_compound_edge_attachments"
_FDP_COMPOUND_CLUSTER_OBSTACLES_KEY = "fmmm_fdp_compound_cluster_obstacles"
_FDP_COMPOUND_NODE_OBSTACLES_KEY = "fmmm_fdp_compound_node_obstacles"
_FDP_EPSILON = 1.0e-9
_FDP_TRACE_PATH = "/tmp/dagua_fdp_trace.log"
# Debug-only Graphviz-fidelity FDP trace: appends one line per node per phase per iteration and can
# balloon to tens of GB during a benchmark (observed 20.5 GB, 2026-06-04). Default OFF; opt in with
# DAGUA_FDP_TRACE=1. Purely logging -- gating it off has zero effect on layout output.
_FDP_TRACE_ENABLED = bool(os.environ.get("DAGUA_FDP_TRACE"))
_GRAPHVIZ_FDP_GRID_VECTORIZE_MIN_PAIRS = 20_000
_OGDF_FMMM_UNIT_EDGE_LENGTH = 20.0
_OGDF_FMMM_DEFAULT_NODE_WIDTH = 20.0
_OGDF_FMMM_DEFAULT_NODE_HEIGHT = 20.0
_OGDF_FMMM_MIN_NODE_SIZE = 10.0
_OGDF_FMMM_BOX_SCALING_FACTOR = 1.1
_OGDF_FMMM_FORCE_SCALING_FACTOR = 0.05
_OGDF_FMMM_FINE_TUNING_ITERATIONS = 20
_OGDF_FMMM_FINE_TUNE_SCALAR = 0.2
_OGDF_FMMM_POST_SPRING_STRENGTH = 2.0
_OGDF_FMMM_EPSILON = 0.1
_OGDF_FMMM_BILLION = 1_000_000_000
_OGDF_FMMM_MAAR_TIP_IMPROVEMENT = 0.99999
_OGDF_FMMM_NEARLY_EQUAL_DELTA = 1.0e-10
_OGDF_FMMM_NMM_MIN_NODES = 175
_OGDF_FMMM_NMM_PARTICLES_PER_LEAF = 25
_OGDF_FMMM_NMM_PRECISION = 4
_OGDF_FMMM_RANDOM_TRIES = 20
_OGDF_FMMM_IDEAL_EDGE_LENGTH = _OGDF_FMMM_UNIT_EDGE_LENGTH + 2.0 * math.sqrt(
    (_OGDF_FMMM_DEFAULT_NODE_WIDTH / 2.0) ** 2 + (_OGDF_FMMM_DEFAULT_NODE_HEIGHT / 2.0) ** 2
)

_ObjectKey = Tuple[str, Union[int, str]]


class _OgdfMt19937:
    """OGDF ``randomNumber`` generator backed by C++ ``std::mt19937``.

    Parameters
    ----------
    seed : int
        Seed passed to OGDF ``setSeed``. OGDF stores a process-global
        ``std::mt19937`` and ``FMMMLayout`` reseeds it before random placement.
    """

    def __init__(self, seed: int) -> None:
        self._index = 624
        self._state = [0] * 624
        self._state[0] = int(seed) & 0xFFFFFFFF
        for index in range(1, 624):
            previous = self._state[index - 1]
            self._state[index] = (1812433253 * (previous ^ (previous >> 30)) + index) & 0xFFFFFFFF

    def _twist(self) -> None:
        """Regenerate the MT19937 state array.

        Returns
        -------
        None
            Updates the generator state in place.
        """
        for index in range(624):
            value = (self._state[index] & 0x80000000) + (
                self._state[(index + 1) % 624] & 0x7FFFFFFF
            )
            twisted = self._state[(index + 397) % 624] ^ (value >> 1)
            if value & 1:
                twisted ^= 0x9908B0DF
            self._state[index] = twisted & 0xFFFFFFFF
        self._index = 0

    def raw(self) -> int:
        """Return the next raw ``std::mt19937`` 32-bit value.

        Returns
        -------
        int
            Unsigned integer in ``[0, 2**32 - 1]``.
        """
        if self._index >= 624:
            self._twist()
        value = self._state[self._index]
        self._index += 1
        value ^= value >> 11
        value ^= (value << 7) & 0x9D2C5680
        value ^= (value << 15) & 0xEFC60000
        value ^= value >> 18
        return value & 0xFFFFFFFF

    def randint(self, low: int, high: int) -> int:
        """Return libstdc++ ``uniform_int_distribution`` output.

        Parameters
        ----------
        low : int
            Inclusive lower bound.
        high : int
            Inclusive upper bound.

        Returns
        -------
        int
            Uniform integer in ``[low, high]`` matching OGDF's libstdc++
            ``randomNumber`` stream for ranges used by FMMM.
        """
        if low > high:
            raise ValueError("low must be <= high.")
        urng_range = 0xFFFFFFFF
        urange = int(high) - int(low)
        if urange == urng_range:
            return int(low) + self.raw()
        if urng_range > urange:
            scaling = urng_range // (urange + 1)
            past = (urange + 1) * scaling
            value = self.raw()
            while value >= past:
                value = self.raw()
            return int(low) + (value // scaling)
        raise ValueError("OGDF FMMM port only supports 32-bit or smaller ranges.")

    def random(self) -> float:
        """Return OGDF FMMM's open-interval random fraction.

        Returns
        -------
        float
            ``(randomNumber(1, BILLION) + 1) / (BILLION + 2)``.
        """
        return float(self.randint(1, _OGDF_FMMM_BILLION) + 1) / float(_OGDF_FMMM_BILLION + 2)


@dataclass
class _OgdfNmmCell:
    """One cell in OGDF's reduced ``QuadTreeNM``.

    Parameters
    ----------
    level : int
        Absolute quadtree level.
    down_left : tuple[float, float]
        Lower-left corner of the small cell.
    boxlength : float
        Side length of the small cell.
    nodes : list[int]
        Particles stored by a leaf, in graph-node order.
    parent : _OgdfNmmCell, optional
        Parent cell in the reduced tree.
    """

    level: int
    down_left: Tuple[float, float]
    boxlength: float
    nodes: list[int]
    parent: Optional["_OgdfNmmCell"] = None
    children: list["_OgdfNmmCell"] = field(default_factory=list)
    center: complex = 0j
    multipole: list[complex] = field(default_factory=list)
    local: list[complex] = field(default_factory=list)
    interaction: list["_OgdfNmmCell"] = field(default_factory=list)
    direct_one: list["_OgdfNmmCell"] = field(default_factory=list)
    direct_two: list["_OgdfNmmCell"] = field(default_factory=list)
    multipole_sources: list["_OgdfNmmCell"] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Initialize mutable cell collections.

        Returns
        -------
        None
            Initializes per-cell tree and expansion storage.
        """
        self.children = []
        self.multipole = [0j] * (_OGDF_FMMM_NMM_PRECISION + 1)
        self.local = [0j] * (_OGDF_FMMM_NMM_PRECISION + 1)
        self.interaction = []
        self.direct_one = []
        self.direct_two = []
        self.multipole_sources = []

    def is_leaf(self) -> bool:
        """Return whether this reduced-tree cell has no children.

        Returns
        -------
        bool
            ``True`` for a leaf cell.
        """
        return not self.children


def _ogdf_nmm_nearly_equal(first: float, second: float) -> bool:
    """Return OGDF ``numexcept::nearly_equal`` for two doubles.

    Parameters
    ----------
    first : float
        First value.
    second : float
        Reference value whose relative interval is tested.

    Returns
    -------
    bool
        Whether ``first`` lies within ``1e-10`` relative error of ``second``.
    """
    if second > 0.0:
        lower = second * (1.0 - _OGDF_FMMM_NEARLY_EQUAL_DELTA)
        upper = second * (1.0 + _OGDF_FMMM_NEARLY_EQUAL_DELTA)
    else:
        lower = second * (1.0 + _OGDF_FMMM_NEARLY_EQUAL_DELTA)
        upper = second * (1.0 - _OGDF_FMMM_NEARLY_EQUAL_DELTA)
    return lower <= first <= upper


def _ogdf_nmm_smallest_cell(
    cell: _OgdfNmmCell,
    positions: Sequence[Sequence[float]],
) -> None:
    """Shrink a cell iteratively exactly like OGDF's default NMM option.

    Parameters
    ----------
    cell : _OgdfNmmCell
        Cell to shrink in place.
    positions : Sequence[Sequence[float]]
        Current particle positions with shape ``[N, 2]``.

    Returns
    -------
    None
        Updates the cell level, corner, and side length in place.
    """
    if not cell.nodes:
        return
    x_values = [positions[node][0] for node in cell.nodes]
    y_values = [positions[node][1] for node in cell.nodes]
    min_x = min(x_values)
    max_x = max(x_values)
    min_y = min(y_values)
    max_y = max(y_values)
    if min_x == max_x and min_y == max_y:
        return
    while max_x - min_x >= 1.0e-300 or max_y - min_y >= 1.0e-300:
        half = cell.boxlength / 2.0
        x0, y0 = cell.down_left
        mid_x = x0 + half
        mid_y = y0 + half
        left = x0 <= min_x and max_x < mid_x
        right = mid_x <= min_x and max_x < x0 + cell.boxlength
        bottom = y0 <= min_y and max_y < mid_y
        top = mid_y <= min_y and max_y < y0 + cell.boxlength
        if left and top:
            cell.down_left = (x0, mid_y)
        elif right and top:
            cell.down_left = (mid_x, mid_y)
        elif left and bottom:
            cell.down_left = (x0, y0)
        elif right and bottom:
            cell.down_left = (mid_x, y0)
        else:
            return
        cell.level += 1
        cell.boxlength = half


def _ogdf_nmm_build_reduced_tree(
    positions: Sequence[Sequence[float]],
    boxlength: float,
    down_left_corner: Tuple[float, float],
) -> _OgdfNmmCell:
    """Build OGDF's reduced bucket quadtree for NMM.

    Parameters
    ----------
    positions : Sequence[Sequence[float]]
        Current particle positions with shape ``[N, 2]``.
    boxlength : float
        Current FMMM computational-box side length.
    down_left_corner : tuple[float, float]
        Current computational-box lower-left corner.

    Returns
    -------
    _OgdfNmmCell
        Root of the reduced quadtree.

    Notes
    -----
    OGDF's ``SubtreeBySubtree`` builder materializes complete temporary
    subtrees, removes empty and degenerate nodes, and collapses every subtree
    holding at most 25 particles. Constructing the resulting canonical reduced
    tree directly gives the same retained cells and LT/RT/LB/RB traversal order.
    """
    root = _OgdfNmmCell(0, down_left_corner, boxlength, list(range(len(positions))))

    def reduce(cell: _OgdfNmmCell) -> _OgdfNmmCell:
        _ogdf_nmm_smallest_cell(cell, positions)
        if len(cell.nodes) <= _OGDF_FMMM_NMM_PARTICLES_PER_LEAF:
            return cell
        half = cell.boxlength / 2.0
        x0, y0 = cell.down_left
        mid_x = x0 + half
        mid_y = y0 + half
        buckets: list[list[int]] = [[], [], [], []]
        for node in cell.nodes:
            x_coord, y_coord = positions[node]
            right = x_coord >= mid_x
            top = y_coord >= mid_y
            bucket = 1 if right and top else 0 if top else 3 if right else 2
            buckets[bucket].append(node)
        corners = ((x0, mid_y), (mid_x, mid_y), (x0, y0), (mid_x, y0))
        retained: list[_OgdfNmmCell] = []
        for nodes, corner in zip(buckets, corners):
            if not nodes:
                continue
            child = _OgdfNmmCell(cell.level + 1, corner, half, nodes, parent=cell)
            retained.append(reduce(child))
        if len(retained) == 1:
            only = retained[0]
            only.parent = cell.parent
            return only
        cell.nodes = []
        cell.children = retained
        for child in retained:
            child.parent = cell
        return cell

    return reduce(root)


def _ogdf_nmm_form_multipoles(
    cell: _OgdfNmmCell,
    positions: Sequence[Sequence[float]],
    rng: _OgdfMt19937,
    leaves: list[_OgdfNmmCell],
) -> None:
    """Form OGDF multipole coefficients bottom-up in tree order.

    Parameters
    ----------
    cell : _OgdfNmmCell
        Current reduced-tree cell.
    positions : Sequence[Sequence[float]]
        Current particle positions with shape ``[N, 2]``.
    rng : _OgdfMt19937
        Shared OGDF global RNG stream used to waggle cell centers.
    leaves : list[_OgdfNmmCell]
        Output leaf list in OGDF preorder.

    Returns
    -------
    None
        Populates centers and multipole coefficients in place.
    """
    random_y = float(rng.randint(1, _OGDF_FMMM_BILLION) + 1) / float(_OGDF_FMMM_BILLION + 2)
    cell.center = complex(
        cell.down_left[0] + cell.boxlength * 0.5,
        cell.down_left[1] + cell.boxlength * 0.5 + 0.001 * cell.boxlength * random_y,
    )
    if cell.is_leaf():
        leaves.append(cell)
        cell.multipole[0] = complex(float(len(cell.nodes)), 0.0)
        for node in cell.nodes:
            delta = complex(positions[node][0], positions[node][1]) - cell.center
            power = delta
            for order in range(1, _OGDF_FMMM_NMM_PRECISION + 1):
                cell.multipole[order] += -power / float(order)
                power *= delta
        return
    for child in cell.children:
        _ogdf_nmm_form_multipoles(child, positions, rng, leaves)
        shift = child.center - cell.center
        powers = [1.0 + 0j]
        for _ in range(_OGDF_FMMM_NMM_PRECISION):
            powers.append(powers[-1] * shift)
        cell.multipole[0] += child.multipole[0]
        for order in range(1, _OGDF_FMMM_NMM_PRECISION + 1):
            value = -child.multipole[0] * powers[order] / float(order)
            for source_order in range(1, order + 1):
                value += (
                    child.multipole[source_order]
                    * powers[order - source_order]
                    * float(math.comb(order - 1, source_order - 1))
                )
            cell.multipole[order] += value


def _ogdf_nmm_well_separated(first: _OgdfNmmCell, second: _OgdfNmmCell) -> bool:
    """Return OGDF's asymmetric small-cell well-separation predicate.

    Parameters
    ----------
    first : _OgdfNmmCell
        First reduced-tree cell.
    second : _OgdfNmmCell
        Second reduced-tree cell.

    Returns
    -------
    bool
        Whether the cells are well separated.
    """
    first_box = [
        first.down_left[0],
        first.down_left[0] + first.boxlength,
        first.down_left[1],
        first.down_left[1] + first.boxlength,
    ]
    second_box = [
        second.down_left[0],
        second.down_left[0] + second.boxlength,
        second.down_left[1],
        second.down_left[1] + second.boxlength,
    ]
    if first.boxlength <= second.boxlength:
        second_box = [
            second.down_left[0] - second.boxlength,
            second.down_left[0] + 2.0 * second.boxlength,
            second.down_left[1] - second.boxlength,
            second.down_left[1] + 2.0 * second.boxlength,
        ]
    else:
        first_box = [
            first.down_left[0] - first.boxlength,
            first.down_left[0] + 2.0 * first.boxlength,
            first.down_left[1] - first.boxlength,
            first.down_left[1] + 2.0 * first.boxlength,
        ]
    x_overlap = not (
        first_box[1] <= second_box[0]
        or _ogdf_nmm_nearly_equal(first_box[1], second_box[0])
        or second_box[1] <= first_box[0]
        or _ogdf_nmm_nearly_equal(second_box[1], first_box[0])
    )
    y_overlap = not (
        first_box[3] <= second_box[2]
        or _ogdf_nmm_nearly_equal(first_box[3], second_box[2])
        or second_box[3] <= first_box[2]
        or _ogdf_nmm_nearly_equal(second_box[3], first_box[2])
    )
    return not (x_overlap and y_overlap)


def _ogdf_nmm_bordering(first: _OgdfNmmCell, second: _OgdfNmmCell) -> bool:
    """Return OGDF's reduced-cell bordering predicate.

    Parameters
    ----------
    first : _OgdfNmmCell
        First reduced-tree cell.
    second : _OgdfNmmCell
        Second reduced-tree cell.

    Returns
    -------
    bool
        Whether the two dyadic cells border one another.
    """
    first_box = [
        first.down_left[0],
        first.down_left[0] + first.boxlength,
        first.down_left[1],
        first.down_left[1] + first.boxlength,
    ]
    second_box = [
        second.down_left[0],
        second.down_left[0] + second.boxlength,
        second.down_left[1],
        second.down_left[1] + second.boxlength,
    ]

    def less_equal(left: float, right: float) -> bool:
        return left <= right or _ogdf_nmm_nearly_equal(left, right)

    def contained(one: Sequence[float], two: Sequence[float]) -> bool:
        return (
            less_equal(two[0], one[0])
            and less_equal(one[1], two[1])
            and less_equal(two[2], one[2])
            and less_equal(one[3], two[3])
        ) or (
            less_equal(one[0], two[0])
            and less_equal(two[1], one[1])
            and less_equal(one[2], two[2])
            and less_equal(two[3], one[3])
        )

    if contained(first_box, second_box):
        return False
    if first.boxlength <= second.boxlength:
        moving, fixed, length = first_box, second_box, first.boxlength
    else:
        moving, fixed, length = second_box, first_box, second.boxlength
    if moving[0] < fixed[0]:
        moving[0] += length
        moving[1] += length
    elif moving[1] > fixed[1]:
        moving[0] -= length
        moving[1] -= length
    if moving[2] < fixed[2]:
        moving[2] += length
        moving[3] += length
    elif moving[3] > fixed[3]:
        moving[2] -= length
        moving[3] -= length
    return contained(moving, fixed)


def _ogdf_nmm_complex_log(value: complex) -> complex:
    """Evaluate OGDF's guarded complex logarithm.

    Parameters
    ----------
    value : complex
        Complex argument.

    Returns
    -------
    complex
        Complex logarithm after OGDF's negative-real-axis perturbation.
    """
    import cmath

    if value.real <= 0.0 and value.imag == 0.0:
        value += 1.0e-7
    return cmath.log(value)


def _ogdf_nmm_add_shifted_parent_local(cell: _OgdfNmmCell) -> None:
    """Shift the parent's local expansion to a child cell.

    Parameters
    ----------
    cell : _OgdfNmmCell
        Child receiving its parent's expansion.

    Returns
    -------
    None
        Adds translated coefficients in place.
    """
    if cell.parent is None:
        return
    shift = cell.center - cell.parent.center
    powers = [1.0 + 0j]
    for _ in range(_OGDF_FMMM_NMM_PRECISION):
        powers.append(powers[-1] * shift)
    for order in range(_OGDF_FMMM_NMM_PRECISION + 1):
        value = 0j
        for source_order in range(order, _OGDF_FMMM_NMM_PRECISION + 1):
            value += (
                float(math.comb(source_order, order))
                * cell.parent.local[source_order]
                * powers[source_order - order]
            )
        cell.local[order] += value


def _ogdf_nmm_add_local(source: _OgdfNmmCell, target: _OgdfNmmCell) -> None:
    """Translate one cell's multipole expansion into a target local expansion.

    Parameters
    ----------
    source : _OgdfNmmCell
        Source multipole cell.
    target : _OgdfNmmCell
        Target local-expansion cell.

    Returns
    -------
    None
        Adds translated coefficients in place.
    """
    delta = target.center - source.center
    target.local[0] += source.multipole[0] * _ogdf_nmm_complex_log(delta)
    power = delta
    for order in range(1, _OGDF_FMMM_NMM_PRECISION + 1):
        target.local[0] += source.multipole[order] / power
        power *= delta
    delta_power = delta
    for local_order in range(1, _OGDF_FMMM_NMM_PRECISION + 1):
        sign_plus = 1.0 if (local_order + 1) % 2 == 0 else -1.0
        sign = -sign_plus
        value = sign_plus * source.multipole[0] / (delta_power * float(local_order))
        factor = sign / delta_power
        delta_power *= delta
        inner = 0j
        multipole_power = delta
        for source_order in range(1, _OGDF_FMMM_NMM_PRECISION + 1):
            inner += (
                float(math.comb(local_order + source_order - 1, source_order - 1))
                * source.multipole[source_order]
                / multipole_power
            )
            multipole_power *= delta
        target.local[local_order] += value + factor * inner


def _ogdf_nmm_add_leaf_local(
    positions: Sequence[Sequence[float]],
    source: _OgdfNmmCell,
    target: _OgdfNmmCell,
) -> None:
    """Add direct particle potentials to an interior target expansion.

    Parameters
    ----------
    positions : Sequence[Sequence[float]]
        Current particle positions with shape ``[N, 2]``.
    source : _OgdfNmmCell
        Source leaf cell.
    target : _OgdfNmmCell
        Target interior cell.

    Returns
    -------
    None
        Adds local coefficients in place.
    """
    for node in source.nodes:
        delta = target.center - complex(positions[node][0], positions[node][1])
        target.local[0] += _ogdf_nmm_complex_log(delta)
        power = delta
        for order in range(1, _OGDF_FMMM_NMM_PRECISION + 1):
            sign = 1.0 if (order + 1) % 2 == 0 else -1.0
            target.local[order] += sign / (power * float(order))
            power *= delta


def _ogdf_nmm_form_interactions(
    positions: Sequence[Sequence[float]],
    cell: _OgdfNmmCell,
) -> None:
    """Build OGDF WSPRLS lists and local expansions recursively.

    Parameters
    ----------
    positions : Sequence[Sequence[float]]
        Current particle positions with shape ``[N, 2]``.
    cell : _OgdfNmmCell
        Current target cell.

    Returns
    -------
    None
        Populates interaction lists and local expansions in place.
    """
    queue = (
        list(cell.children)
        if cell.parent is None
        else list(cell.parent.direct_one) + list(cell.parent.interaction)
    )
    interaction: list[_OgdfNmmCell] = []
    local_sources: list[_OgdfNmmCell] = []
    leaf_local_sources: list[_OgdfNmmCell] = []
    direct_one: list[_OgdfNmmCell] = []
    direct_two: list[_OgdfNmmCell] = []
    while queue:
        selected = queue.pop(0)
        if _ogdf_nmm_well_separated(cell, selected):
            local_sources.append(selected)
        elif cell.level < selected.level:
            interaction.append(selected)
        elif not selected.is_leaf():
            queue.extend(selected.children)
        elif _ogdf_nmm_bordering(cell, selected):
            direct_one.append(selected)
        elif selected is not cell and cell.is_leaf():
            direct_two.append(selected)
        elif selected is not cell:
            leaf_local_sources.append(selected)
    cell.interaction = interaction
    cell.direct_one = direct_one
    cell.direct_two = direct_two
    _ogdf_nmm_add_shifted_parent_local(cell)
    for source in local_sources:
        _ogdf_nmm_add_local(source, cell)
    for source in leaf_local_sources:
        _ogdf_nmm_add_leaf_local(positions, source, cell)
    if not cell.is_leaf():
        for child in cell.children:
            _ogdf_nmm_form_interactions(positions, child)
        return
    pending = list(interaction)
    while pending:
        selected = pending.pop(0)
        if selected.is_leaf():
            if _ogdf_nmm_bordering(cell, selected):
                direct_one.append(selected)
            else:
                direct_two.append(selected)
        elif _ogdf_nmm_bordering(cell, selected):
            pending.extend(selected.children)
        else:
            cell.multipole_sources.append(selected)
    cell.direct_one = direct_one
    cell.direct_two = direct_two


def _ogdf_nmm_pair_force(
    positions: Sequence[Sequence[float]],
    source: int,
    target: int,
    rng: _OgdfMt19937,
) -> Tuple[float, float]:
    """Return the exact repulsive force of ``source`` on ``target``.

    Parameters
    ----------
    positions : Sequence[Sequence[float]]
        Current particle positions with shape ``[N, 2]``.
    source : int
        Source particle index.
    target : int
        Target particle index.
    rng : _OgdfMt19937
        Shared OGDF random stream used to separate coincident particles.

    Returns
    -------
    tuple[float, float]
        Repulsive force vector.
    """
    source_x = positions[source][0]
    source_y = positions[source][1]
    target_x = positions[target][0]
    target_y = positions[target][1]
    if abs(source_x - target_x) <= 1.0e-8 and abs(source_y - target_y) <= 1.0e-8:
        # numexcept::choose_distinct_random_point_in_radius_epsilon samples a
        # square and rejects points outside the radius-0.01 disc.
        while True:
            offset_x = (0.1 * (2.0 * (rng.random() - 0.5))) * 0.1
            offset_y = (0.1 * (2.0 * (rng.random() - 0.5))) * 0.1
            if (abs(offset_x) > 1.0e-8 or abs(offset_y) > 1.0e-8) and math.hypot(
                offset_x, offset_y
            ) < 0.01:
                source_x += offset_x
                source_y += offset_y
                break
    dx = target_x - source_x
    dy = target_y - source_y
    distance = math.sqrt(dx * dx + dy * dy)
    if distance == 0.0:
        return 0.0, 0.0
    scalar = (1.0 / distance) / distance
    return scalar * dx, scalar * dy


def _ogdf_fmmm_nmm_repulsive_forces(
    positions: list[list[float]],
    boxlength: float,
    down_left_corner: Tuple[float, float],
    rng: _OgdfMt19937,
) -> list[list[float]]:
    """Calculate repulsive forces with OGDF's New Multipole Method.

    Parameters
    ----------
    positions : list[list[float]]
        Current particle positions with shape ``[N, 2]``.
    boxlength : float
        Current FMMM computational-box side length.
    down_left_corner : tuple[float, float]
        Current computational-box lower-left corner.
    rng : _OgdfMt19937
        Shared OGDF global RNG stream.

    Returns
    -------
    list[list[float]]
        NMM repulsive force vectors with shape ``[N, 2]``.
    """
    root = _ogdf_nmm_build_reduced_tree(positions, boxlength, down_left_corner)
    leaves: list[_OgdfNmmCell] = []
    _ogdf_nmm_form_multipoles(root, positions, rng, leaves)
    _ogdf_nmm_form_interactions(positions, root)
    direct = [[0.0, 0.0] for _ in positions]
    local_force = [[0.0, 0.0] for _ in positions]
    multipole_force = [[0.0, 0.0] for _ in positions]
    for leaf in leaves:
        for node in leaf.nodes:
            value = 0j
            power = 1.0 + 0j
            delta = complex(positions[node][0], positions[node][1]) - leaf.center
            for order in range(1, _OGDF_FMMM_NMM_PRECISION + 1):
                value += float(order) * leaf.local[order] * power
                power *= delta
            local_force[node][0] = value.real
            local_force[node][1] = -value.imag
        for source_cell in leaf.multipole_sources:
            for node in leaf.nodes:
                delta = complex(positions[node][0], positions[node][1]) - source_cell.center
                inverse_power = 1.0 / delta
                value = source_cell.multipole[0] * inverse_power
                for order in range(1, _OGDF_FMMM_NMM_PRECISION + 1):
                    inverse_power /= delta
                    value -= float(order) * source_cell.multipole[order] * inverse_power
                multipole_force[node][0] += value.real
                multipole_force[node][1] -= value.imag
        for source_pos, source in enumerate(leaf.nodes[:-1]):
            for target in leaf.nodes[source_pos + 1 :]:
                fx, fy = _ogdf_nmm_pair_force(positions, source, target, rng)
                direct[target][0] += fx
                direct[target][1] += fy
                direct[source][0] -= fx
                direct[source][1] -= fy
        for neighbor in leaf.direct_one:
            if leaf.boxlength > neighbor.boxlength or (
                leaf.boxlength == neighbor.boxlength and leaf.down_left < neighbor.down_left
            ):
                for target in leaf.nodes:
                    for source in neighbor.nodes:
                        fx, fy = _ogdf_nmm_pair_force(positions, source, target, rng)
                        direct[target][0] += fx
                        direct[target][1] += fy
                        direct[source][0] -= fx
                        direct[source][1] -= fy
        for source_cell in leaf.direct_two:
            for target in leaf.nodes:
                for source in source_cell.nodes:
                    fx, fy = _ogdf_nmm_pair_force(positions, source, target, rng)
                    direct[target][0] += fx
                    direct[target][1] += fy
    return [
        [
            direct[node][0] + local_force[node][0] + multipole_force[node][0],
            direct[node][1] + local_force[node][1] + multipole_force[node][1],
        ]
        for node in range(len(positions))
    ]


def _ogdf_fmmm_norm(point: Tuple[float, float]) -> float:
    """Return the Euclidean norm of a two-dimensional point.

    Parameters
    ----------
    point : tuple[float, float]
        Vector components.

    Returns
    -------
    float
        Euclidean length.
    """
    return math.sqrt(point[0] * point[0] + point[1] * point[1])


def _ogdf_fmmm_angle(
    center: Tuple[float, float],
    first: Tuple[float, float],
    second: Tuple[float, float],
) -> float:
    """Return OGDF ``DPoint::angle`` equivalent for two rays.

    Parameters
    ----------
    center : tuple[float, float]
        Common ray origin.
    first : tuple[float, float]
        First ray endpoint.
    second : tuple[float, float]
        Second ray endpoint.

    Returns
    -------
    float
        Counter-clockwise angle in radians in ``[0, 2*pi]``.
    """
    ax = first[0] - center[0]
    ay = first[1] - center[1]
    bx = second[0] - center[0]
    by = second[1] - center[1]
    angle = math.atan2(ax * by - ay * bx, ax * bx + ay * by)
    if angle < 0.0:
        angle += 2.0 * math.pi
    return angle


def _ogdf_fmmm_initial_box(num_nodes: int) -> Tuple[float, Tuple[float, float]]:
    """Return OGDF initial computational box for default runner nodes.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the component.

    Returns
    -------
    tuple[float, tuple[float, float]]
        ``(boxlength, down_left_corner)``.
    """
    boxlength = math.ceil(
        max(
            num_nodes * max(_OGDF_FMMM_DEFAULT_NODE_WIDTH, _OGDF_FMMM_MIN_NODE_SIZE),
            num_nodes * max(_OGDF_FMMM_DEFAULT_NODE_HEIGHT, _OGDF_FMMM_MIN_NODE_SIZE),
        )
        * _OGDF_FMMM_BOX_SCALING_FACTOR
    )
    return float(boxlength), (0.0, 0.0)


def _ogdf_fmmm_update_box(positions: list[list[float]]) -> Tuple[float, Tuple[float, float]]:
    """Return OGDF tight computational box for current positions.

    Parameters
    ----------
    positions : list[list[float]]
        Mutable node coordinates.

    Returns
    -------
    tuple[float, tuple[float, float]]
        ``(boxlength, down_left_corner)``.
    """
    x_values = [point[0] for point in positions]
    y_values = [point[1] for point in positions]
    xmin = min(x_values)
    xmax = max(x_values)
    ymin = min(y_values)
    ymax = max(y_values)
    down_left = (float(math.floor(xmin - 1.0)), float(math.floor(ymin - 1.0)))
    boxlength = float(math.ceil(max(ymax - ymin, xmax - xmin) * 1.01 + 2.0))
    if boxlength <= 2.0:
        boxlength = float(len(positions) * 20)
        down_left = (
            float(math.floor(xmin) - (boxlength / 2.0)),
            float(math.floor(ymin) - (boxlength / 2.0)),
        )
    return boxlength, down_left


def _ogdf_fmmm_random_placement(
    num_nodes: int,
    seed: int,
    rng: Optional[_OgdfMt19937] = None,
) -> list[list[float]]:
    """Create OGDF FMMM random initial positions.

    Parameters
    ----------
    num_nodes : int
        Number of nodes to place.
    seed : int
        OGDF ``randSeed`` value.
    rng : _OgdfMt19937, optional
        Shared RNG stream. If omitted, a freshly seeded stream is created.

    Returns
    -------
    list[list[float]]
        Mutable coordinates in OGDF output units.
    """
    if rng is None:
        rng = _OgdfMt19937(seed)
    boxlength, _ = _ogdf_fmmm_initial_box(num_nodes)
    positions: list[list[float]] = []
    for _ in range(num_nodes):
        rand_x = float(rng.randint(0, _OGDF_FMMM_BILLION)) / _OGDF_FMMM_BILLION
        rand_y = float(rng.randint(0, _OGDF_FMMM_BILLION)) / _OGDF_FMMM_BILLION
        positions.append([rand_x * (boxlength - 2.0) + 1.0, rand_y * (boxlength - 2.0) + 1.0])
    return positions


def _ogdf_fmmm_adjust_positions(
    positions: list[list[float]],
    average_ideal_edge_length: float,
    down_left_corner: Tuple[float, float],
    boxlength: float,
    final_floor: bool = True,
) -> Tuple[float, Tuple[float, float]]:
    """Apply OGDF integer-position adjustment.

    Parameters
    ----------
    positions : list[list[float]]
        Mutable coordinates.
    average_ideal_edge_length : float
        Average edge length used by OGDF to set the integer boundary.
    down_left_corner : tuple[float, float]
        Current computational box lower-left corner.
    boxlength : float
        Current computational box side length.
    final_floor : bool, default=True
        Whether to apply the integer ``floor`` step. OGDF calls this before
        every force iteration and once after the solver before export.

    Returns
    -------
    tuple[float, tuple[float, float]]
        Possibly updated ``(boxlength, down_left_corner)``.
    """
    max_integer_position = 100.0 * average_ideal_edge_length * len(positions) * len(positions)
    for point in positions:
        point[0] = min(max(point[0], -max_integer_position), max_integer_position)
        point[1] = min(max(point[1], -max_integer_position), max_integer_position)

    if not final_floor:
        return boxlength, down_left_corner

    down_x, down_y = down_left_corner
    for point in positions:
        new_x = math.floor(point[0])
        new_y = math.floor(point[1])
        if new_x < down_x:
            boxlength += 2.0
            down_x -= 2.0
        if new_y < down_y:
            boxlength += 2.0
            down_y -= 2.0
        point[0] = float(new_x)
        point[1] = float(new_y)
    return boxlength, (down_x, down_y)


def _ogdf_fmmm_repulsive_forces(positions: list[list[float]]) -> list[list[float]]:
    """Calculate exact OGDF FMMM repulsive forces.

    Parameters
    ----------
    positions : list[list[float]]
        Current node coordinates.

    Returns
    -------
    list[list[float]]
        Repulsive force vectors.
    """
    forces = [[0.0, 0.0] for _ in positions]
    for source in range(len(positions) - 1):
        for target in range(source + 1, len(positions)):
            dx = positions[target][0] - positions[source][0]
            dy = positions[target][1] - positions[source][1]
            distance = math.sqrt(dx * dx + dy * dy)
            if distance == 0.0:
                continue
            scalar = (1.0 / distance) / distance
            fx = scalar * dx
            fy = scalar * dy
            forces[target][0] += fx
            forces[target][1] += fy
            forces[source][0] -= fx
            forces[source][1] -= fy
    return forces


def _ogdf_fmmm_attractive_forces(
    positions: list[list[float]],
    edges: Sequence[Tuple[int, int]],
    ideal_edge_lengths: Optional[Sequence[float]] = None,
) -> list[list[float]]:
    """Calculate OGDF FMMM ``ForceModel::New`` attractive forces.

    Parameters
    ----------
    positions : list[list[float]]
        Current node coordinates.
    edges : Sequence[tuple[int, int]]
        Simple loop-free edge list.
    ideal_edge_lengths : Sequence[float], optional
        Desired edge lengths aligned with ``edges``. If omitted, the default
        single-level ideal edge length is used for all edges.

    Returns
    -------
    list[list[float]]
        Attractive force vectors.

    Raises
    ------
    ValueError
        If ``ideal_edge_lengths`` is provided with the wrong length.
    """
    if ideal_edge_lengths is not None and len(ideal_edge_lengths) != len(edges):
        raise ValueError("ideal_edge_lengths must have one value per edge.")

    forces = [[0.0, 0.0] for _ in positions]
    for edge_pos, (source, target) in enumerate(edges):
        ideal_length = (
            _OGDF_FMMM_IDEAL_EDGE_LENGTH
            if ideal_edge_lengths is None
            else float(ideal_edge_lengths[edge_pos])
        )
        ideal_length = max(ideal_length, _OGDF_FMMM_EPSILON)
        ideal_cubed = ideal_length * ideal_length * ideal_length
        dx = positions[target][0] - positions[source][0]
        dy = positions[target][1] - positions[source][1]
        distance = math.sqrt(dx * dx + dy * dy)
        if dx == 0.0 and dy == 0.0:
            fx = 0.0
            fy = 0.0
        else:
            scalar = math.log2(distance / ideal_length) * distance * distance / ideal_cubed
            scalar /= distance
            fx = scalar * dx
            fy = scalar * dy
        forces[target][0] -= fx
        forces[target][1] -= fy
        forces[source][0] += fx
        forces[source][1] += fy
    return forces


def _ogdf_fmmm_combined_forces(
    attr: list[list[float]],
    rep: list[list[float]],
    boxlength: float,
    iter_index: int,
    fine_tuning_step: int,
    cool_factor: float,
    average_ideal_edge_length: float = _OGDF_FMMM_IDEAL_EDGE_LENGTH,
) -> Tuple[list[list[float]], float]:
    """Combine OGDF attractive and repulsive forces.

    Parameters
    ----------
    attr : list[list[float]]
        Attractive force vectors.
    rep : list[list[float]]
        Repulsive force vectors.
    boxlength : float
        Current computational box side length.
    iter_index : int
        One-based OGDF iteration number for this phase.
    fine_tuning_step : int
        OGDF phase selector: ``0`` main, ``1`` post cooldown, ``2`` fine tune.
    cool_factor : float
        Incoming OGDF cool factor state.
    average_ideal_edge_length : float, default=_OGDF_FMMM_IDEAL_EDGE_LENGTH
        Average desired edge length for the current multilevel graph.

    Returns
    -------
    tuple[list[list[float]], float]
        Combined movement vectors and updated cool factor.
    """
    # OGDF's default coolTemperature(false) resets the shared factor before
    # every phase adjustment; the cooldown division therefore does not
    # accumulate across its ten iterations.
    cool_factor = 1.0
    if fine_tuning_step == 1:
        cool_factor /= 10.0
    elif fine_tuning_step == 2:
        if iter_index <= _OGDF_FMMM_FINE_TUNING_ITERATIONS - 5:
            cool_factor = _OGDF_FMMM_FINE_TUNE_SCALAR
        else:
            cool_factor = _OGDF_FMMM_FINE_TUNE_SCALAR / 10.0

    spring_strength = 1.0 if fine_tuning_step <= 1 else _OGDF_FMMM_POST_SPRING_STRENGTH
    rep_strength = 1.0 if fine_tuning_step <= 1 else min(0.2, 400.0 / float(len(attr)))
    max_radius = boxlength / 1000.0 if iter_index == 1 else boxlength / 5.0
    average_sq = average_ideal_edge_length * average_ideal_edge_length
    forces: list[list[float]] = []
    for node_index in range(len(attr)):
        fx = spring_strength * attr[node_index][0] + rep_strength * rep[node_index][0]
        fy = spring_strength * attr[node_index][1] + rep_strength * rep[node_index][1]
        fx *= average_sq
        fy *= average_sq
        norm = math.sqrt(fx * fx + fy * fy)
        if norm == 0.0:
            forces.append([0.0, 0.0])
        else:
            scalar = min(norm * cool_factor * _OGDF_FMMM_FORCE_SCALING_FACTOR, max_radius) / norm
            forces.append([scalar * fx, scalar * fy])
    return forces, cool_factor


def _ogdf_fmmm_prevent_oscillations(
    forces: list[list[float]],
    last_movement: list[list[float]],
    iter_index: int,
) -> list[list[float]]:
    """Apply OGDF oscillation damping.

    Parameters
    ----------
    forces : list[list[float]]
        Proposed movement vectors.
    last_movement : list[list[float]]
        Previous movement vectors, updated in place.
    iter_index : int
        One-based OGDF phase iteration.

    Returns
    -------
    list[list[float]]
        Damped movement vectors.
    """
    if iter_index == 1:
        for node_index, force in enumerate(forces):
            last_movement[node_index][0] = force[0]
            last_movement[node_index][1] = force[1]
        return forces

    factors = (
        2.0,
        2.0,
        1.5,
        1.0,
        0.66666666,
        0.5,
        0.33333333,
        0.33333333,
        0.5,
        0.66666666,
        1.0,
        1.5,
        2.0,
        2.0,
    )
    pi_times_one_over_six = 0.52359878
    for node_index, force in enumerate(forces):
        old = last_movement[node_index]
        norm_new = _ogdf_fmmm_norm((force[0], force[1]))
        norm_old = _ogdf_fmmm_norm((old[0], old[1]))
        if norm_new > 0.0 and norm_old > 0.0:
            angle = _ogdf_fmmm_angle((0.0, 0.0), (old[0], old[1]), (force[0], force[1]))
            factor = factors[int(math.ceil(angle / pi_times_one_over_six))]
            quotient = norm_old * factor / norm_new
            if quotient < 1.0:
                force[0] *= quotient
                force[1] *= quotient
        old[0] = force[0]
        old[1] = force[1]
    return forces


def _ogdf_fmmm_average_ideal_edge_length(
    ideal_edge_lengths: Optional[Sequence[float]],
    has_edges: bool,
) -> float:
    """Return OGDF's average desired edge length for a force level.

    Parameters
    ----------
    ideal_edge_lengths : Sequence[float], optional
        Desired edge lengths for the current level. ``None`` keeps the legacy
        single-level default.
    has_edges : bool
        Whether the current level has at least one edge.

    Returns
    -------
    float
        Average desired edge length. OGDF uses ``50`` for edgeless levels.
    """
    if ideal_edge_lengths is None:
        return _OGDF_FMMM_IDEAL_EDGE_LENGTH
    if not has_edges:
        return 50.0
    return sum(float(length) for length in ideal_edge_lengths) / float(len(ideal_edge_lengths))


def _ogdf_fmmm_tensor_repulsive_forces(positions: torch.Tensor) -> torch.Tensor:
    """Calculate exact OGDF FMMM repulsive forces with tensor operations.

    Parameters
    ----------
    positions : torch.Tensor
        Current node coordinates with shape ``[N, 2]`` in OGDF coordinate
        units.

    Returns
    -------
    torch.Tensor
        Repulsive force vectors with shape ``[N, 2]``.
    """
    if positions.shape[0] <= 1:
        return torch.zeros_like(positions)

    delta = positions.unsqueeze(1) - positions.unsqueeze(0)
    distances = torch.linalg.norm(delta, dim=2)
    factor = torch.zeros_like(distances)
    nonzero = distances > 0.0
    factor[nonzero] = 1.0 / distances[nonzero].square()
    factor.fill_diagonal_(0.0)
    return (delta * factor.unsqueeze(2)).sum(dim=1)


def _ogdf_fmmm_tensor_attractive_forces(
    positions: torch.Tensor,
    edges: Sequence[Tuple[int, int]],
    ideal_edge_lengths: Optional[Sequence[float]],
) -> torch.Tensor:
    """Calculate exact OGDF FMMM attractive forces with tensor operations.

    Parameters
    ----------
    positions : torch.Tensor
        Current node coordinates with shape ``[N, 2]`` in OGDF coordinate
        units.
    edges : Sequence[tuple[int, int]]
        Simple loop-free edge list.
    ideal_edge_lengths : Sequence[float], optional
        Desired edge lengths aligned with ``edges``. If omitted, all edges use
        OGDF's default single-level desired length.

    Returns
    -------
    torch.Tensor
        Attractive force vectors with shape ``[N, 2]``.

    Raises
    ------
    ValueError
        If ``ideal_edge_lengths`` is provided with the wrong length.
    """
    if ideal_edge_lengths is not None and len(ideal_edge_lengths) != len(edges):
        raise ValueError("ideal_edge_lengths must have one value per edge.")

    forces = torch.zeros_like(positions)
    if not edges:
        return forces

    edge_tensor = torch.tensor(edges, dtype=torch.long, device=positions.device)
    sources = edge_tensor[:, 0]
    targets = edge_tensor[:, 1]
    if ideal_edge_lengths is None:
        desired_lengths = torch.full(
            (len(edges),),
            _OGDF_FMMM_IDEAL_EDGE_LENGTH,
            dtype=positions.dtype,
            device=positions.device,
        )
    else:
        desired_lengths = torch.tensor(
            [max(float(length), _OGDF_FMMM_EPSILON) for length in ideal_edge_lengths],
            dtype=positions.dtype,
            device=positions.device,
        )

    delta = positions[targets] - positions[sources]
    distances = torch.linalg.norm(delta, dim=1)
    force_scale = torch.zeros_like(distances)
    nonzero = distances > 0.0
    force_scale[nonzero] = (
        torch.log2(distances[nonzero] / desired_lengths[nonzero])
        * distances[nonzero]
        / desired_lengths[nonzero].pow(3)
    )
    edge_forces = delta * force_scale.unsqueeze(1)
    forces.index_add_(0, sources, edge_forces)
    forces.index_add_(0, targets, -edge_forces)
    return forces


def _ogdf_fmmm_tensor_combined_forces(
    attr: torch.Tensor,
    rep: torch.Tensor,
    boxlength: float,
    iter_index: int,
    fine_tuning_step: int,
    cool_factor: float,
    average_ideal_edge_length: float,
) -> Tuple[torch.Tensor, float]:
    """Combine attractive and repulsive force tensors like OGDF.

    Parameters
    ----------
    attr : torch.Tensor
        Attractive force vectors with shape ``[N, 2]``.
    rep : torch.Tensor
        Repulsive force vectors with shape ``[N, 2]``.
    boxlength : float
        Current computational box side length.
    iter_index : int
        One-based OGDF iteration number for this phase.
    fine_tuning_step : int
        OGDF phase selector: ``0`` main, ``1`` post cooldown, ``2`` fine tune.
    cool_factor : float
        Incoming OGDF cool factor state.
    average_ideal_edge_length : float
        Average desired edge length for the current multilevel graph.

    Returns
    -------
    tuple[torch.Tensor, float]
        Combined movement vectors with shape ``[N, 2]`` and updated cool
        factor.
    """
    # Keep the vectorized small-graph path identical to OGDF's default
    # coolTemperature(false) state transition.
    cool_factor = 1.0
    if fine_tuning_step == 1:
        cool_factor /= 10.0
    elif fine_tuning_step == 2:
        if iter_index <= _OGDF_FMMM_FINE_TUNING_ITERATIONS - 5:
            cool_factor = _OGDF_FMMM_FINE_TUNE_SCALAR
        else:
            cool_factor = _OGDF_FMMM_FINE_TUNE_SCALAR / 10.0

    spring_strength = 1.0 if fine_tuning_step <= 1 else _OGDF_FMMM_POST_SPRING_STRENGTH
    rep_strength = 1.0 if fine_tuning_step <= 1 else min(0.2, 400.0 / float(attr.shape[0]))
    max_radius = boxlength / 1000.0 if iter_index == 1 else boxlength / 5.0
    forces = (spring_strength * attr + rep_strength * rep) * (
        average_ideal_edge_length * average_ideal_edge_length
    )
    norms = torch.linalg.norm(forces, dim=1, keepdim=True)
    limited = torch.minimum(
        norms * cool_factor * _OGDF_FMMM_FORCE_SCALING_FACTOR,
        torch.full_like(norms, max_radius),
    )
    scale = torch.zeros_like(norms)
    nonzero = norms > 0.0
    scale[nonzero] = limited[nonzero] / norms[nonzero]
    return forces * scale, cool_factor


def _ogdf_fmmm_tensor_prevent_oscillations(
    forces: torch.Tensor,
    last_movement: list[list[float]],
    iter_index: int,
) -> torch.Tensor:
    """Apply OGDF oscillation damping to a force tensor.

    Parameters
    ----------
    forces : torch.Tensor
        Proposed movement vectors with shape ``[N, 2]``.
    last_movement : list[list[float]]
        Previous movement vectors, updated in place.
    iter_index : int
        One-based OGDF phase iteration.

    Returns
    -------
    torch.Tensor
        Damped movement vectors with shape ``[N, 2]``.
    """
    if iter_index == 1:
        updated = forces.detach().cpu().tolist()
        for node_index, force in enumerate(updated):
            last_movement[node_index][0] = float(force[0])
            last_movement[node_index][1] = float(force[1])
        return forces

    previous = torch.tensor(last_movement, dtype=forces.dtype, device=forces.device)
    norm_new = torch.linalg.norm(forces, dim=1)
    norm_old = torch.linalg.norm(previous, dim=1)
    damped = forces.clone()
    active = (norm_new > 0.0) & (norm_old > 0.0)
    if active.any():
        cross = previous[:, 0] * forces[:, 1] - previous[:, 1] * forces[:, 0]
        dot = (previous * forces).sum(dim=1)
        angles = torch.atan2(cross, dot)
        angles = torch.where(angles < 0.0, angles + 2.0 * math.pi, angles)
        buckets = torch.ceil(angles / 0.52359878).to(dtype=torch.long).clamp(0, 13)
        factors = torch.tensor(
            [
                2.0,
                2.0,
                1.5,
                1.0,
                0.66666666,
                0.5,
                0.33333333,
                0.33333333,
                0.5,
                0.66666666,
                1.0,
                1.5,
                2.0,
                2.0,
            ],
            dtype=forces.dtype,
            device=forces.device,
        )
        quotient = norm_old * factors[buckets] / norm_new.clamp(min=torch.finfo(forces.dtype).tiny)
        damping = torch.minimum(torch.ones_like(quotient), quotient)
        damped[active] = forces[active] * damping[active].unsqueeze(1)

    updated = damped.detach().cpu().tolist()
    for node_index, force in enumerate(updated):
        last_movement[node_index][0] = float(force[0])
        last_movement[node_index][1] = float(force[1])
    return damped


def _ogdf_fmmm_force_iteration(
    positions: list[list[float]],
    edges: Sequence[Tuple[int, int]],
    last_movement: list[list[float]],
    boxlength: float,
    down_left_corner: Tuple[float, float],
    iter_index: int,
    fine_tuning_step: int,
    cool_factor: float,
    ideal_edge_lengths: Optional[Sequence[float]] = None,
    rng: Optional[_OgdfMt19937] = None,
) -> Tuple[float, Tuple[float, float], float]:
    """Execute one OGDF FMMM force iteration.

    Parameters
    ----------
    positions : list[list[float]]
        Mutable node coordinates.
    edges : Sequence[tuple[int, int]]
        Simple loop-free edge list.
    last_movement : list[list[float]]
        Previous movement vectors, updated in place.
    boxlength : float
        Current computational box side length.
    down_left_corner : tuple[float, float]
        Current computational box lower-left corner.
    iter_index : int
        One-based OGDF phase iteration.
    fine_tuning_step : int
        OGDF phase selector.
    cool_factor : float
        Incoming cool factor.
    ideal_edge_lengths : Sequence[float], optional
        Desired edge lengths aligned with ``edges`` for multilevel fidelity.
    rng : _OgdfMt19937, optional
        Shared OGDF RNG stream. Required when the level uses NMM.

    Returns
    -------
    tuple[float, tuple[float, float], float]
        Updated ``boxlength``, ``down_left_corner``, and ``cool_factor``.
    """
    average_ideal_edge_length = _ogdf_fmmm_average_ideal_edge_length(
        ideal_edge_lengths,
        has_edges=bool(edges),
    )
    boxlength, down_left_corner = _ogdf_fmmm_adjust_positions(
        positions,
        average_ideal_edge_length,
        down_left_corner,
        boxlength,
    )
    if len(positions) >= _OGDF_FMMM_NMM_MIN_NODES:
        if rng is None:
            raise ValueError("OGDF NMM force calculation requires a shared RNG stream.")
        attr_list = _ogdf_fmmm_attractive_forces(positions, edges, ideal_edge_lengths)
        rep_list = _ogdf_fmmm_nmm_repulsive_forces(
            positions,
            boxlength,
            down_left_corner,
            rng,
        )
        force_list, cool_factor = _ogdf_fmmm_combined_forces(
            attr_list,
            rep_list,
            boxlength,
            iter_index,
            fine_tuning_step,
            cool_factor,
            average_ideal_edge_length,
        )
        force_list = _ogdf_fmmm_prevent_oscillations(
            force_list,
            last_movement,
            iter_index,
        )
        for node_index, force in enumerate(force_list):
            positions[node_index][0] += force[0]
            positions[node_index][1] += force[1]
        boxlength, down_left_corner = _ogdf_fmmm_update_box(positions)
        return boxlength, down_left_corner, cool_factor

    position_tensor = torch.tensor(positions, dtype=torch.float64)
    attr = _ogdf_fmmm_tensor_attractive_forces(position_tensor, edges, ideal_edge_lengths)
    rep = _ogdf_fmmm_tensor_repulsive_forces(position_tensor)
    forces, cool_factor = _ogdf_fmmm_tensor_combined_forces(
        attr,
        rep,
        boxlength,
        iter_index,
        fine_tuning_step,
        cool_factor,
        average_ideal_edge_length,
    )
    forces = _ogdf_fmmm_tensor_prevent_oscillations(forces, last_movement, iter_index)
    updated_positions = (position_tensor + forces).detach().cpu().tolist()
    for node_index, point in enumerate(updated_positions):
        positions[node_index][0] = float(point[0])
        positions[node_index][1] = float(point[1])
    boxlength, down_left_corner = _ogdf_fmmm_update_box(positions)
    return boxlength, down_left_corner, cool_factor


def _ogdf_fmmm_adapt_to_ideal_edge_length(
    positions: list[list[float]],
    edges: Sequence[Tuple[int, int]],
    ideal_edge_lengths: Optional[Sequence[float]] = None,
) -> None:
    """Scale drawing to OGDF's ideal average edge length.

    Parameters
    ----------
    positions : list[list[float]]
        Mutable node coordinates.
    edges : Sequence[tuple[int, int]]
        Simple loop-free edge list.
    ideal_edge_lengths : Sequence[float], optional
        Desired edge lengths aligned with ``edges``. If omitted, all edges use
        the default single-level desired length.

    Returns
    -------
    None
        Mutates ``positions`` in place.
    """
    if ideal_edge_lengths is not None and len(ideal_edge_lengths) != len(edges):
        raise ValueError("ideal_edge_lengths must have one value per edge.")

    sum_real = 0.0
    sum_ideal = 0.0
    for edge_pos, (source, target) in enumerate(edges):
        dx = positions[source][0] - positions[target][0]
        dy = positions[source][1] - positions[target][1]
        sum_real += math.sqrt(dx * dx + dy * dy)
        sum_ideal += (
            _OGDF_FMMM_IDEAL_EDGE_LENGTH
            if ideal_edge_lengths is None
            else float(ideal_edge_lengths[edge_pos])
        )
    scale = 1.0 if sum_real == 0.0 else sum_ideal / sum_real
    for point in positions:
        point[0] *= scale
        point[1] *= scale


def _ogdf_fmmm_component_rectangle(
    positions: list[list[float]],
) -> Tuple[float, float, Tuple[float, float]]:
    """Return OGDF component rectangle for default node dimensions.

    Parameters
    ----------
    positions : list[list[float]]
        Component node coordinates.

    Returns
    -------
    tuple[float, float, tuple[float, float]]
        Rectangle ``(width, height, old_down_left_corner)``.
    """
    max_boundary = max(_OGDF_FMMM_DEFAULT_NODE_WIDTH / 2.0, _OGDF_FMMM_DEFAULT_NODE_HEIGHT / 2.0)
    x_min = positions[0][0] - max_boundary
    x_max = positions[0][0] + max_boundary
    y_min = positions[0][1] - max_boundary
    y_max = positions[0][1] + max_boundary
    for point in positions[1:]:
        x_min = min(x_min, point[0] - max_boundary)
        x_max = max(x_max, point[0] + max_boundary)
        y_min = min(y_min, point[1] - max_boundary)
        y_max = max(y_max, point[1] + max_boundary)
    x_min -= 15.0
    x_max += 15.0
    y_min -= 15.0
    y_max += 15.0
    return x_max - x_min, y_max - y_min, (x_min, y_min)


def _ogdf_fmmm_square_aspect_area(width: float, height: float) -> float:
    """Return OGDF one-component square aspect-ratio area.

    Parameters
    ----------
    width : float
        Rectangle width.
    height : float
        Rectangle height.

    Returns
    -------
    float
        Aspect-ratio adjusted area for ``pageRatio() == 1``.
    """
    ratio = width / height
    scaling = (1.0 / ratio) if ratio < 1.0 else ratio
    return width * height * scaling


@dataclass
class _OgdfMaarRectangle:
    """Rectangle state used by OGDF FMMM MAARPacking.

    Parameters
    ----------
    index : int
        Original component index.
    width : float
        Rectangle width.
    height : float
        Rectangle height.
    old_x : float
        Original down-left x coordinate.
    old_y : float
        Original down-left y coordinate.
    tipped : bool, default=False
        Whether MAARPacking has tipped this rectangle by 90 degrees.
    """

    index: int
    width: float
    height: float
    old_x: float
    old_y: float
    tipped: bool = False


@dataclass
class _OgdfMaarRow:
    """Row bookkeeping used by OGDF FMMM MAARPacking.

    Parameters
    ----------
    max_height : float
        Maximum rectangle height in the row.
    total_width : float
        Sum of rectangle widths in the row.
    row_index : int
        Stable row order assigned at creation time.
    """

    max_height: float
    total_width: float
    row_index: int


def _ogdf_fmmm_nearly_equal(first: float, second: float) -> bool:
    """Return OGDF ``numexcept::nearly_equal`` for positive packing areas.

    Parameters
    ----------
    first : float
        Candidate value.
    second : float
        Reference value.

    Returns
    -------
    bool
        True when ``first`` falls inside OGDF's relative tolerance around
        ``second``.
    """
    if second > 0.0:
        lower = second * (1.0 - _OGDF_FMMM_NEARLY_EQUAL_DELTA)
        upper = second * (1.0 + _OGDF_FMMM_NEARLY_EQUAL_DELTA)
    else:
        lower = second * (1.0 + _OGDF_FMMM_NEARLY_EQUAL_DELTA)
        upper = second * (1.0 - _OGDF_FMMM_NEARLY_EQUAL_DELTA)
    return lower <= first <= upper


def _ogdf_maar_tipped_rectangle(rectangle: _OgdfMaarRectangle) -> _OgdfMaarRectangle:
    """Return OGDF MAARPacking's tipped rectangle copy.

    Parameters
    ----------
    rectangle : _OgdfMaarRectangle
        Source rectangle.

    Returns
    -------
    _OgdfMaarRectangle
        Rectangle rotated by 90 degrees with old down-left coordinates updated
        like ``MAARPacking::tipp_over``.
    """
    if not rectangle.tipped:
        old_x = -rectangle.old_y - rectangle.height
        old_y = rectangle.old_x
    else:
        old_x = rectangle.old_y
        old_y = -rectangle.old_x - rectangle.width
    return _OgdfMaarRectangle(
        index=rectangle.index,
        width=rectangle.height,
        height=rectangle.width,
        old_x=old_x,
        old_y=old_y,
        tipped=not rectangle.tipped,
    )


def _ogdf_maar_aspect_ratio_area(width: float, height: float, aspect_ratio: float) -> float:
    """Return OGDF MAARPacking's aspect-ratio adjusted area.

    Parameters
    ----------
    width : float
        Candidate packing width.
    height : float
        Candidate packing height.
    aspect_ratio : float
        Desired page ratio.

    Returns
    -------
    float
        Area scaled by the ratio mismatch.
    """
    ratio = width / height
    if ratio < aspect_ratio:
        return width * height * (aspect_ratio / ratio)
    return width * height * (ratio / aspect_ratio)


def _ogdf_maar_better_tip_new_row(
    rectangle: _OgdfMaarRectangle,
    area_width: float,
    area_height: float,
    aspect_ratio: float,
) -> tuple[bool, float]:
    """Evaluate OGDF's tipped-vs-untipped new-row placement.

    Parameters
    ----------
    rectangle : _OgdfMaarRectangle
        Rectangle being inserted.
    area_width : float
        Current packing width.
    area_height : float
        Current packing height.
    aspect_ratio : float
        Desired page ratio.

    Returns
    -------
    tuple[bool, float]
        Whether tipping is better and the best resulting aspect-ratio area.
    """
    width = max(area_width, rectangle.width)
    height = area_height + rectangle.height
    best_area = _ogdf_maar_aspect_ratio_area(width, height, aspect_ratio)
    tipped_width = max(area_width, rectangle.height)
    tipped_height = area_height + rectangle.width
    tipped_area = _ogdf_maar_aspect_ratio_area(tipped_width, tipped_height, aspect_ratio)
    if tipped_area < _OGDF_FMMM_MAAR_TIP_IMPROVEMENT * best_area:
        return True, tipped_area
    return False, best_area


def _ogdf_maar_better_tip_this_row(
    rectangle: _OgdfMaarRectangle,
    row: _OgdfMaarRow,
    area_width: float,
    area_height: float,
    aspect_ratio: float,
) -> tuple[bool, float]:
    """Evaluate OGDF's tipped-vs-untipped existing-row placement.

    Parameters
    ----------
    rectangle : _OgdfMaarRectangle
        Rectangle being inserted.
    row : _OgdfMaarRow
        Best-Fit row candidate.
    area_width : float
        Current packing width.
    area_height : float
        Current packing height.
    aspect_ratio : float
        Desired page ratio.

    Returns
    -------
    tuple[bool, float]
        Whether tipping is better and the best resulting aspect-ratio area.
    """
    width = max(area_width, row.total_width + rectangle.width)
    height = max(area_height, area_height - row.max_height + rectangle.height)
    best_area = _ogdf_maar_aspect_ratio_area(width, height, aspect_ratio)
    if rectangle.width > row.max_height:
        return False, best_area
    tipped_width = max(area_width, row.total_width + rectangle.height)
    tipped_height = max(area_height, area_height - row.max_height + rectangle.width)
    tipped_area = _ogdf_maar_aspect_ratio_area(tipped_width, tipped_height, aspect_ratio)
    if tipped_area < _OGDF_FMMM_MAAR_TIP_IMPROVEMENT * best_area:
        return True, tipped_area
    return False, best_area


def _ogdf_maar_pack_component_transforms(
    boxes: list[tuple[float, float, float, float]],
    aspect_ratio: float = 1.0,
) -> list[tuple[float, float, bool]]:
    """Pack component boxes with OGDF FMMM MAARPacking Best-Fit.

    Parameters
    ----------
    boxes : list[tuple[float, float, float, float]]
        Component bounding boxes as ``(llx, lly, urx, ury)``.
    aspect_ratio : float, default=1.0
        OGDF ``FMMMLayout::pageRatio``. The default is square packing.

    Returns
    -------
    list[tuple[float, float, bool]]
        Per-component ``(x_offset, y_offset, tipped)`` transforms in original
        component order.

    Notes
    -----
    This ports ``FMMMLayout::pack_subGraph_drawings`` and
    ``MAARPacking::pack_rectangles_using_Best_Fit_strategy`` from OGDF
    (``FMMMLayout.cpp:746-760`` and ``MAARPacking.cpp:58-104``). The FMMM
    defaults are decreasing-height presort and ``TipOver::NoGrowingRow``.
    """
    if not boxes:
        return []
    rectangles = [
        _OgdfMaarRectangle(
            index=index,
            width=box[2] - box[0],
            height=box[3] - box[1],
            old_x=box[0],
            old_y=box[1],
        )
        for index, box in enumerate(boxes)
    ]
    rectangles.sort(key=lambda rectangle: -rectangle.height)

    rows: list[_OgdfMaarRow] = []
    row_for_rectangle: list[int] = []
    area_width = 0.0
    area_height = 0.0

    for rect_pos, rectangle in enumerate(rectangles):
        if not rows:
            should_tip, _ = _ogdf_maar_better_tip_new_row(
                rectangle,
                area_width,
                area_height,
                aspect_ratio,
            )
            if should_tip:
                rectangle = _ogdf_maar_tipped_rectangle(rectangle)
                rectangles[rect_pos] = rectangle
            rows.append(
                _OgdfMaarRow(
                    max_height=rectangle.height,
                    total_width=rectangle.width,
                    row_index=0,
                )
            )
            row_for_rectangle.append(0)
            area_width = max(area_width, rectangle.width)
            area_height += rectangle.height
            continue

        should_tip_new, best_area = _ogdf_maar_better_tip_new_row(
            rectangle,
            area_width,
            area_height,
            aspect_ratio,
        )
        best_try_index = 2 if should_tip_new else 1
        # OGDF's PQueue returns the row with the smallest total width; that is
        # the only existing row considered by the Best-Fit insertion test.
        best_row_index = min(range(len(rows)), key=lambda row_index: rows[row_index].total_width)
        should_tip_row, row_area = _ogdf_maar_better_tip_this_row(
            rectangle,
            rows[best_row_index],
            area_width,
            area_height,
            aspect_ratio,
        )
        row_try_index = 4 if should_tip_row else 3
        if row_area <= best_area or _ogdf_fmmm_nearly_equal(best_area, row_area):
            best_area = row_area
            best_try_index = row_try_index
        if best_try_index in (2, 4):
            rectangle = _ogdf_maar_tipped_rectangle(rectangle)
            rectangles[rect_pos] = rectangle
        if best_try_index in (1, 2):
            row_index = len(rows)
            rows.append(
                _OgdfMaarRow(
                    max_height=rectangle.height,
                    total_width=rectangle.width,
                    row_index=row_index,
                )
            )
            row_for_rectangle.append(row_index)
            area_width = max(area_width, rectangle.width)
            area_height += rectangle.height
        else:
            row = rows[best_row_index]
            old_max_height = row.max_height
            row.max_height = max(old_max_height, rectangle.height)
            row.total_width += rectangle.width
            row_for_rectangle.append(best_row_index)
            area_width = max(area_width, row.total_width)
            area_height = max(area_height, area_height - old_max_height + rectangle.height)

    row_y_min = [0.0] * len(rows)
    for row_index in range(1, len(rows)):
        row_y_min[row_index] = row_y_min[row_index - 1] + rows[row_index - 1].max_height
    act_row_x_max = [0.0] * len(rows)
    offsets = [(0.0, 0.0, False) for _ in boxes]
    for rectangle, row_index in zip(rectangles, row_for_rectangle):
        row = rows[row_index]
        new_x = act_row_x_max[row.row_index]
        act_row_x_max[row.row_index] += rectangle.width
        new_y = row_y_min[row.row_index] + (row.max_height - rectangle.height) / 2.0
        offsets[rectangle.index] = (
            new_x - rectangle.old_x,
            new_y - rectangle.old_y,
            rectangle.tipped,
        )
    return offsets


def _ogdf_maar_pack_offsets(
    boxes: list[tuple[float, float, float, float]],
    aspect_ratio: float = 1.0,
) -> list[tuple[float, float]]:
    """Return only translations from OGDF FMMM MAARPacking Best-Fit.

    Parameters
    ----------
    boxes : list[tuple[float, float, float, float]]
        Component bounding boxes as ``(llx, lly, urx, ury)``.
    aspect_ratio : float, default=1.0
        OGDF ``FMMMLayout::pageRatio``.

    Returns
    -------
    list[tuple[float, float]]
        Per-component translations in original component order.
    """
    return [
        (x_offset, y_offset)
        for x_offset, y_offset, _ in _ogdf_maar_pack_component_transforms(boxes, aspect_ratio)
    ]


def _ogdf_fmmm_rotate_positions(
    positions: list[list[float]],
    angle: float,
) -> list[list[float]]:
    """Rotate positions around the origin like OGDF component packing.

    Parameters
    ----------
    positions : list[list[float]]
        Source coordinates.
    angle : float
        Rotation angle in radians.

    Returns
    -------
    list[list[float]]
        Rotated coordinate copy.
    """
    sin_angle = math.sin(angle)
    cos_angle = math.cos(angle)
    return [
        [
            cos_angle * point[0] - sin_angle * point[1],
            sin_angle * point[0] + cos_angle * point[1],
        ]
        for point in positions
    ]


def _ogdf_fmmm_pack_single_component(positions: list[list[float]]) -> None:
    """Apply OGDF single-component rotation and packing translation.

    Parameters
    ----------
    positions : list[list[float]]
        Mutable coordinates after the subgraph force calculation.

    Returns
    -------
    None
        Mutates ``positions`` in place.
    """
    best_positions = [point.copy() for point in positions]
    best_width, best_height, best_old_dlc = _ogdf_fmmm_component_rectangle(best_positions)
    best_area = _ogdf_fmmm_square_aspect_area(best_width, best_height)
    old_positions = [point.copy() for point in positions]
    for step in range(1, 11):
        angle = (math.pi / 2.0) * (float(step) / 11.0)
        candidate = _ogdf_fmmm_rotate_positions(old_positions, angle)
        width, height, old_dlc = _ogdf_fmmm_component_rectangle(candidate)
        area = _ogdf_fmmm_square_aspect_area(width, height)
        area_pi_half_rotated = _ogdf_fmmm_square_aspect_area(height, width)
        if area < best_area:
            best_positions = candidate
            best_width = width
            best_height = height
            best_old_dlc = old_dlc
            best_area = area
        elif area_pi_half_rotated < best_area:
            best_positions = candidate
            best_width = width
            best_height = height
            best_old_dlc = old_dlc
            best_area = area_pi_half_rotated

    if best_width / best_height < 1.0:
        best_positions = [[-point[1], point[0]] for point in best_positions]
        best_old_dlc = (-best_old_dlc[1] - best_height, best_old_dlc[0])

    for node_index, point in enumerate(best_positions):
        positions[node_index][0] = point[0] - best_old_dlc[0]
        positions[node_index][1] = point[1] - best_old_dlc[1]


def _ogdf_fmmm_simple_edges(edge_index: torch.Tensor) -> list[Tuple[int, int]]:
    """Return OGDF's simple loop-free edge set in input order.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.

    Returns
    -------
    list[tuple[int, int]]
        First representative of each undirected non-loop edge.
    """
    edges: list[Tuple[int, int]] = []
    seen: set[Tuple[int, int]] = set()
    cpu_edges = edge_index.detach().to(device="cpu", dtype=torch.long)
    for edge_pos in range(int(cpu_edges.shape[1])):
        source = int(cpu_edges[0, edge_pos].item())
        target = int(cpu_edges[1, edge_pos].item())
        if source == target:
            continue
        key = (source, target) if source <= target else (target, source)
        if key in seen:
            continue
        seen.add(key)
        edges.append((source, target))
    return edges


def _ogdf_fmmm_level_edges(edge_index: torch.Tensor) -> list[Tuple[int, int]]:
    """Return a level graph edge list from a unique edge tensor.

    Parameters
    ----------
    edge_index : torch.Tensor
        Unique level edge tensor with shape ``[2, E]``.

    Returns
    -------
    list[tuple[int, int]]
        Edges in tensor order.
    """
    cpu_edges = edge_index.detach().to(device="cpu", dtype=torch.long)
    return [
        (int(cpu_edges[0, edge_pos].item()), int(cpu_edges[1, edge_pos].item()))
        for edge_pos in range(int(cpu_edges.shape[1]))
    ]


def _ogdf_fmmm_level_edge_lengths(edge_lengths: torch.Tensor) -> list[float]:
    """Return hierarchy edge lengths as Python floats.

    Parameters
    ----------
    edge_lengths : torch.Tensor
        Per-edge desired lengths with shape ``[E]`` from the native hierarchy.

    Returns
    -------
    list[float]
        Desired edge lengths in OGDF coordinates as Python floats.
    """
    return [float(length) for length in edge_lengths.detach().to(device="cpu").tolist()]


@dataclass
class _OgdfFmmmLevel:
    """One exact OGDF FMMM hierarchy level.

    Parameters
    ----------
    edges : list[tuple[int, int]]
        Simple undirected edges in OGDF graph iteration order.
    edge_lengths : list[float]
        Desired edge lengths aligned with ``edges``.
    num_nodes : int
        Node count on this level.
    masses : list[int]
        Collapsed-node masses used by galaxy selection.
    """

    edges: list[Tuple[int, int]]
    edge_lengths: list[float]
    num_nodes: int
    masses: list[int]


@dataclass
class _OgdfFmmmHierarchyStep:
    """Exact metadata for one OGDF hierarchy transition and prolongation.

    Parameters
    ----------
    mapping : list[int]
        Fine-node to coarse-node mapping.
    node_types : list[int]
        OGDF sun, planet, planet-with-moons, and moon type codes.
    dedicated_sun : list[int]
        Fine-level sun assigned to every node.
    dedicated_sun_distance : list[float]
        Path distance from each node to its dedicated sun.
    pm_nodes : list[int]
        Planet-with-moons nodes in fine graph order.
    moon_children : list[list[int]]
        Moon lists aligned with fine nodes.
    lambda_values : list[list[float]]
        Inter-solar interpolation fractions in fine edge order.
    neighbor_suns : list[list[int]]
        Neighboring fine-level suns aligned with lambda values.
    moon_edges : set[int]
        Fine edge ids selected as moon edges.
    """

    mapping: list[int]
    node_types: list[int]
    dedicated_sun: list[int]
    dedicated_sun_distance: list[float]
    pm_nodes: list[int]
    moon_children: list[list[int]]
    lambda_values: list[list[float]]
    neighbor_suns: list[list[int]]
    moon_edges: set[int]


def _ogdf_fmmm_level_adjacency(level: _OgdfFmmmLevel) -> list[list[Tuple[int, int]]]:
    """Build adjacency entries in OGDF edge insertion order.

    Parameters
    ----------
    level : _OgdfFmmmLevel
        Hierarchy level.

    Returns
    -------
    list[list[tuple[int, int]]]
        Per-node ``(neighbor, edge_id)`` entries.
    """
    adjacency: list[list[Tuple[int, int]]] = [[] for _ in range(level.num_nodes)]
    for edge_id, (source, target) in enumerate(level.edges):
        adjacency[source].append((target, edge_id))
        adjacency[target].append((source, edge_id))
    return adjacency


def _ogdf_fmmm_coarsen_level(
    level: _OgdfFmmmLevel,
    seed: int,
) -> Tuple[_OgdfFmmmHierarchyStep, _OgdfFmmmLevel]:
    """Collapse one level with OGDF's solar-system galaxy partition.

    Parameters
    ----------
    level : _OgdfFmmmLevel
        Fine hierarchy level.
    seed : int
        ``randSeed`` used to reseed OGDF's private ``Set`` stream per level.

    Returns
    -------
    tuple[_OgdfFmmmHierarchyStep, _OgdfFmmmLevel]
        Exact prolongation metadata and coarse graph.
    """
    adjacency = _ogdf_fmmm_level_adjacency(level)
    star_masses = [
        level.masses[node] + sum(level.masses[neighbor] for neighbor, _ in neighbors)
        for node, neighbors in enumerate(adjacency)
    ]
    selectable = _RandomNodeSet.from_star_masses(star_masses)
    rng = _OgdfMt19937(seed)
    mapping = [-1] * level.num_nodes
    node_types = [0] * level.num_nodes
    dedicated_sun = [-1] * level.num_nodes
    dedicated_distance = [0.0] * level.num_nodes
    sun_to_coarse: Dict[int, int] = {}

    while not selectable.empty():
        sun = selectable.get_random_node_with_lowest_star_mass(  # type: ignore[arg-type]
            rng,
            _OGDF_FMMM_RANDOM_TRIES,
        )
        coarse_node = len(sun_to_coarse)
        sun_to_coarse[sun] = coarse_node
        mapping[sun] = coarse_node
        node_types[sun] = 1
        dedicated_sun[sun] = sun
        planets: list[int] = []
        for planet, edge_id in adjacency[sun]:
            node_types[planet] = 2
            dedicated_sun[planet] = sun
            dedicated_distance[planet] = level.edge_lengths[edge_id]
            mapping[planet] = coarse_node
            planets.append(planet)
        for planet in planets:
            selectable.delete(planet)
        for planet in planets:
            for possible_moon, _ in adjacency[planet]:
                selectable.delete(possible_moon)

    moon_children: list[list[int]] = [[] for _ in range(level.num_nodes)]
    moon_edges: set[int] = set()
    for node in range(level.num_nodes):
        if node_types[node] != 0:
            continue
        nearest = -1
        nearest_edge = -1
        nearest_distance = 0.0
        for neighbor, edge_id in adjacency[node]:
            if node_types[neighbor] not in (2, 3):
                continue
            distance = level.edge_lengths[edge_id]
            if nearest < 0 or nearest_distance > distance:
                nearest = neighbor
                nearest_edge = edge_id
                nearest_distance = distance
        if nearest < 0:
            raise RuntimeError("OGDF galaxy partition produced a moon without a planet neighbor.")
        moon_edges.add(nearest_edge)
        sun = dedicated_sun[nearest]
        dedicated_sun[node] = sun
        dedicated_distance[node] = nearest_distance + dedicated_distance[nearest]
        mapping[node] = sun_to_coarse[sun]
        node_types[node] = 4
        node_types[nearest] = 3
        moon_children[nearest].append(node)

    coarse_masses = [0] * len(sun_to_coarse)
    for coarse_node in mapping:
        coarse_masses[coarse_node] += 1
    lambda_values: list[list[float]] = [[] for _ in range(level.num_nodes)]
    neighbor_suns: list[list[int]] = [[] for _ in range(level.num_nodes)]
    coarse_edges: list[Tuple[int, int]] = []
    coarse_lengths: list[float] = []
    pair_to_edge: dict[Tuple[int, int], int] = {}
    for edge_id, (source, target) in enumerate(level.edges):
        source_sun = dedicated_sun[source]
        target_sun = dedicated_sun[target]
        if source_sun == target_sun:
            continue
        coarse_source = sun_to_coarse[source_sun]
        coarse_target = sun_to_coarse[target_sun]
        new_length = (
            dedicated_distance[source] + level.edge_lengths[edge_id] + dedicated_distance[target]
        )
        lambda_values[source].append(dedicated_distance[source] / new_length)
        lambda_values[target].append(dedicated_distance[target] / new_length)
        neighbor_suns[source].append(target_sun)
        neighbor_suns[target].append(source_sun)
        pair = (
            (coarse_source, coarse_target)
            if coarse_source < coarse_target
            else (coarse_target, coarse_source)
        )
        coarse_edge_id = pair_to_edge.get(pair)
        if coarse_edge_id is None:
            coarse_edge_id = len(coarse_edges)
            pair_to_edge[pair] = coarse_edge_id
            coarse_edges.append((coarse_source, coarse_target))
            coarse_lengths.append(new_length)
    step = _OgdfFmmmHierarchyStep(
        mapping=mapping,
        node_types=node_types,
        dedicated_sun=dedicated_sun,
        dedicated_sun_distance=dedicated_distance,
        pm_nodes=[node for node in range(level.num_nodes) if node_types[node] == 3],
        moon_children=moon_children,
        lambda_values=lambda_values,
        neighbor_suns=neighbor_suns,
        moon_edges=moon_edges,
    )
    return step, _OgdfFmmmLevel(
        edges=coarse_edges,
        edge_lengths=coarse_lengths,
        num_nodes=len(sun_to_coarse),
        masses=coarse_masses,
    )


def _ogdf_fmmm_build_hierarchy(
    edge_index: torch.Tensor,
    num_nodes: int,
    seed: int,
) -> Tuple[list[_OgdfFmmmLevel], list[_OgdfFmmmHierarchyStep]]:
    """Build OGDF's exact FMMM hierarchy from a simple input graph.

    Parameters
    ----------
    edge_index : torch.Tensor
        Input edges with shape ``[2, E]``.
    num_nodes : int
        Fine graph node count.
    seed : int
        OGDF ``randSeed``.

    Returns
    -------
    tuple[list[_OgdfFmmmLevel], list[_OgdfFmmmHierarchyStep]]
        Levels from fine to coarse and transition metadata.
    """
    base_edges = _ogdf_fmmm_simple_edges(edge_index)
    levels = [_OgdfFmmmLevel(base_edges, [1.0] * len(base_edges), num_nodes, [1] * num_nodes)]
    steps: list[_OgdfFmmmHierarchyStep] = []
    bad_edge_count = 0
    while levels[-1].num_nodes > 50:
        if len(levels) > 1 and len(levels[-1].edges) > 0.8 * float(len(levels[-2].edges)):
            if bad_edge_count < 5:
                bad_edge_count += 1
            else:
                break
        step, coarse = _ogdf_fmmm_coarsen_level(levels[-1], seed)
        if coarse.num_nodes >= levels[-1].num_nodes:
            break
        steps.append(step)
        levels.append(coarse)
    return levels, steps


def _ogdf_fmmm_waggled_position(
    source: Sequence[float],
    target: Sequence[float],
    lambda_value: float,
    rng: _OgdfMt19937,
) -> list[float]:
    """Return OGDF's waggled interpolation between two points.

    Parameters
    ----------
    source : Sequence[float]
        Source point.
    target : Sequence[float]
        Target point.
    lambda_value : float
        Interpolation fraction.
    rng : _OgdfMt19937
        Shared OGDF RNG stream.

    Returns
    -------
    list[float]
        Waggled point.
    """
    center = [
        source[0] + lambda_value * (target[0] - source[0]),
        source[1] + lambda_value * (target[1] - source[1]),
    ]
    radius = 0.05 * math.hypot(target[0] - source[0], target[1] - source[1]) * rng.random()
    angle = 2.0 * math.pi * rng.random()
    return [center[0] + math.cos(angle) * radius, center[1] + math.sin(angle) * radius]


def _ogdf_fmmm_sector_position(
    center: Sequence[float],
    radius: float,
    angle_one: float,
    angle_two: float,
    rng: _OgdfMt19937,
) -> list[float]:
    """Place a node randomly on an OGDF placement sector.

    Parameters
    ----------
    center : Sequence[float]
        Dedicated sun position.
    radius : float
        Dedicated sun distance.
    angle_one : float
        Sector start angle.
    angle_two : float
        Sector end angle.
    rng : _OgdfMt19937
        Shared OGDF RNG stream.

    Returns
    -------
    list[float]
        Point on the selected circular sector.
    """
    angle = angle_one + (angle_two - angle_one) * rng.random()
    return [center[0] + math.cos(angle) * radius, center[1] + math.sin(angle) * radius]


def _ogdf_fmmm_prolong_positions(
    coarse_positions: Sequence[Sequence[float]],
    coarse_level: _OgdfFmmmLevel,
    fine_level: _OgdfFmmmLevel,
    step: _OgdfFmmmHierarchyStep,
    rng: _OgdfMt19937,
) -> list[list[float]]:
    """Prolong a level with OGDF's ``InitialPlacementMult::Advanced`` path.

    Parameters
    ----------
    coarse_positions : Sequence[Sequence[float]]
        Coarse positions with shape ``[N_coarse, 2]``.
    coarse_level : _OgdfFmmmLevel
        Coarse hierarchy level.
    fine_level : _OgdfFmmmLevel
        Fine hierarchy level.
    step : _OgdfFmmmHierarchyStep
        Fine-to-coarse transition metadata.
    rng : _OgdfMt19937
        Shared OGDF global RNG stream.

    Returns
    -------
    list[list[float]]
        Fine positions with shape ``[N_fine, 2]``.
    """
    positions = [[0.0, 0.0] for _ in range(fine_level.num_nodes)]
    placed = [False] * fine_level.num_nodes
    for node, node_type in enumerate(step.node_types):
        if node_type == 1:
            positions[node] = list(coarse_positions[step.mapping[node]])
            placed[node] = True
    coarse_adjacency = _ogdf_fmmm_level_adjacency(coarse_level)
    angles: dict[int, Tuple[float, float]] = {}
    for coarse_node in range(coarse_level.num_nodes):
        center = coarse_positions[coarse_node]
        adjacent = [coarse_positions[neighbor] for neighbor, _ in coarse_adjacency[coarse_node]]
        angle_one = 0.0
        angle_two = 0.0
        if not adjacent:
            angle_two = 2.0 * math.pi
        elif len(adjacent) == 1:
            angle_one = _ogdf_fmmm_angle(
                (center[0], center[1]),
                (center[0] + 1.0, center[1]),
                (adjacent[0][0], adjacent[0][1]),
            )
            angle_two = angle_one + math.pi
        else:
            for index, point in enumerate(adjacent[:10]):
                candidate = _ogdf_fmmm_angle(
                    (center[0], center[1]),
                    (center[0] + 1.0, center[1]),
                    (point[0], point[1]),
                )
                gap = min(
                    _ogdf_fmmm_angle(
                        (center[0], center[1]),
                        (point[0], point[1]),
                        (other[0], other[1]),
                    )
                    for other_index, other in enumerate(adjacent)
                    if other_index != index and other != point
                )
                if index == 0 or gap > angle_two - angle_one:
                    angle_one = candidate
                    angle_two = candidate + gap
            if angle_one == angle_two:
                angle_two = angle_one + math.pi
        sun = next(
            node
            for node, mapped in enumerate(step.mapping)
            if mapped == coarse_node and step.node_types[node] == 1
        )
        angles[sun] = (angle_one, angle_two)
    fine_adjacency = _ogdf_fmmm_level_adjacency(fine_level)

    def barycenter(candidates: Sequence[Sequence[float]]) -> list[float]:
        return [
            sum(point[0] for point in candidates) / float(len(candidates)),
            sum(point[1] for point in candidates) / float(len(candidates)),
        ]

    def calculated_position(
        sun_position: Sequence[float],
        neighbor_position: Sequence[float],
        sun_distance: float,
        neighbor_distance: float,
    ) -> list[float]:
        distance = math.hypot(
            sun_position[0] - neighbor_position[0],
            sun_position[1] - neighbor_position[1],
        )
        interpolation = (
            sun_distance + (distance - sun_distance - neighbor_distance) / 2.0
        ) / distance
        return _ogdf_fmmm_waggled_position(
            sun_position,
            neighbor_position,
            interpolation,
            rng,
        )

    for node, node_type in enumerate(step.node_types):
        if node_type not in (2, 4):
            continue
        sun = step.dedicated_sun[node]
        candidates: list[list[float]] = []
        for neighbor, edge_id in fine_adjacency[node]:
            if (
                step.dedicated_sun[neighbor] == sun
                and step.node_types[neighbor] != 1
                and placed[neighbor]
            ):
                candidates.append(
                    calculated_position(
                        positions[sun],
                        positions[neighbor],
                        step.dedicated_sun_distance[node],
                        fine_level.edge_lengths[edge_id],
                    )
                )
        if step.lambda_values[node]:
            for fraction, neighbor_sun in zip(
                step.lambda_values[node],
                step.neighbor_suns[node],
            ):
                candidates.append(
                    _ogdf_fmmm_waggled_position(
                        positions[sun],
                        positions[neighbor_sun],
                        fraction,
                        rng,
                    )
                )
        elif not candidates:
            angle_one, angle_two = angles[sun]
            candidates.append(
                _ogdf_fmmm_sector_position(
                    positions[sun],
                    step.dedicated_sun_distance[node],
                    angle_one,
                    angle_two,
                    rng,
                )
            )
        positions[node] = barycenter(candidates)
        placed[node] = True
    for node in step.pm_nodes:
        sun = step.dedicated_sun[node]
        candidates = []
        for neighbor, edge_id in fine_adjacency[node]:
            if (
                edge_id not in step.moon_edges
                and step.dedicated_sun[neighbor] == sun
                and step.node_types[neighbor] != 1
                and placed[neighbor]
            ):
                candidates.append(
                    calculated_position(
                        positions[sun],
                        positions[neighbor],
                        step.dedicated_sun_distance[node],
                        fine_level.edge_lengths[edge_id],
                    )
                )
        for moon in step.moon_children[node]:
            candidates.append(
                _ogdf_fmmm_waggled_position(
                    positions[sun],
                    positions[moon],
                    step.dedicated_sun_distance[node] / step.dedicated_sun_distance[moon],
                    rng,
                )
            )
        for fraction, neighbor_sun in zip(
            step.lambda_values[node],
            step.neighbor_suns[node],
        ):
            candidates.append(
                _ogdf_fmmm_waggled_position(
                    positions[sun],
                    positions[neighbor_sun],
                    fraction,
                    rng,
                )
            )
        positions[node] = barycenter(candidates)
        placed[node] = True
    return positions


def _ogdf_fmmm_scale_hierarchy_lengths(
    levels: Sequence[Any],
    hierarchy_steps: Sequence[Any],
) -> None:
    """Scale native hierarchy length factors into OGDF coordinate units.

    Parameters
    ----------
    levels : Sequence[Any]
        FM^3 hierarchy levels returned by ``_build_hierarchy``.
    hierarchy_steps : Sequence[Any]
        Prolongation metadata returned by ``_build_hierarchy``.

    Returns
    -------
    None
        Mutates the private hierarchy objects in place.
    """
    for level in levels:
        if isinstance(level.edge_lengths, list):
            level.edge_lengths = [
                length * _OGDF_FMMM_IDEAL_EDGE_LENGTH for length in level.edge_lengths
            ]
        else:
            level.edge_lengths = level.edge_lengths * _OGDF_FMMM_IDEAL_EDGE_LENGTH
    for step in hierarchy_steps:
        step.dedicated_sun_distance = [
            float(distance) * _OGDF_FMMM_IDEAL_EDGE_LENGTH
            for distance in step.dedicated_sun_distance
        ]


def _ogdf_fmmm_max_mult_iter(
    act_level: int,
    max_level: int,
    node_nr: int,
    fixed_iterations: int,
) -> int:
    """Return OGDF ``get_max_mult_iter`` for linearly decreasing iterations.

    Parameters
    ----------
    act_level : int
        Current hierarchy level, where ``0`` is finest and ``max_level`` is
        coarsest.
    max_level : int
        Coarsest hierarchy level index.
    node_nr : int
        Node count for the current level.
    fixed_iterations : int
        OGDF ``fixedIterations`` option.

    Returns
    -------
    int
        Number of force iterations for this level.
    """
    max_iter_factor = 10
    if max_level == 0:
        iterations = max_iter_factor * int(fixed_iterations)
    else:
        iterations = int(fixed_iterations) + int(
            (float(act_level) / float(max_level))
            * float(max_iter_factor - 1)
            * float(fixed_iterations)
        )
    if node_nr <= 500 and iterations < 100:
        return 100
    return iterations


def _ogdf_fmmm_postprocess_fidelity(
    positions: list[list[float]],
    edges: Sequence[Tuple[int, int]],
    ideal_edge_lengths: Optional[Sequence[float]],
    last_movement: list[list[float]],
    boxlength: float,
    down_left_corner: Tuple[float, float],
    cool_factor: float,
    rng: Optional[_OgdfMt19937] = None,
) -> Tuple[float, Tuple[float, float], float]:
    """Run OGDF FMMM level-0 cooldown, fine-tune, resize, pack, and floor.

    Parameters
    ----------
    positions : list[list[float]]
        Mutable node coordinates.
    edges : Sequence[tuple[int, int]]
        Simple loop-free edge list.
    ideal_edge_lengths : Sequence[float], optional
        Desired edge lengths aligned with ``edges``.
    last_movement : list[list[float]]
        Last movement vectors from the main force loop.
    boxlength : float
        Current computational box side length.
    down_left_corner : tuple[float, float]
        Current computational box lower-left corner.
    cool_factor : float
        Current OGDF cool factor.
    rng : _OgdfMt19937, optional
        Shared OGDF RNG stream used when the finest level takes the NMM path.

    Returns
    -------
    tuple[float, tuple[float, float], float]
        Updated ``boxlength``, ``down_left_corner``, and ``cool_factor``.
    """
    average_ideal_edge_length = _ogdf_fmmm_average_ideal_edge_length(
        ideal_edge_lengths,
        has_edges=bool(edges),
    )
    for iter_index in range(1, 11):
        boxlength, down_left_corner, cool_factor = _ogdf_fmmm_force_iteration(
            positions,
            edges,
            last_movement,
            boxlength,
            down_left_corner,
            iter_index,
            1,
            cool_factor,
            ideal_edge_lengths,
            rng,
        )

    if edges:
        _ogdf_fmmm_adapt_to_ideal_edge_length(positions, edges, ideal_edge_lengths)
        boxlength, down_left_corner = _ogdf_fmmm_update_box(positions)

    for iter_index in range(1, _OGDF_FMMM_FINE_TUNING_ITERATIONS + 1):
        boxlength, down_left_corner, cool_factor = _ogdf_fmmm_force_iteration(
            positions,
            edges,
            last_movement,
            boxlength,
            down_left_corner,
            iter_index,
            2,
            cool_factor,
            ideal_edge_lengths,
            rng,
        )

    if edges:
        _ogdf_fmmm_adapt_to_ideal_edge_length(positions, edges, ideal_edge_lengths)
    _ogdf_fmmm_pack_single_component(positions)
    boxlength, down_left_corner = _ogdf_fmmm_update_box(positions)
    return _ogdf_fmmm_adjust_positions(
        positions,
        average_ideal_edge_length,
        down_left_corner,
        boxlength,
    ) + (cool_factor,)


def _layout_ogdf_fmmm_multilevel_fidelity(
    edge_index: torch.Tensor,
    num_nodes: int,
    fixed_iterations: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Run OGDF FMMM's multilevel fidelity scheme on a single component.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes in the graph.
    fixed_iterations : int
        OGDF ``fixedIterations`` value.
    seed : int
        OGDF ``randSeed`` value.
    device : torch.device
        Output tensor device.

    Returns
    -------
    torch.Tensor
        Final OGDF-coordinate positions with shape ``[N, 2]``.
    """
    levels, hierarchy_steps = _ogdf_fmmm_build_hierarchy(
        edge_index,
        num_nodes,
        seed,
    )
    _ogdf_fmmm_scale_hierarchy_lengths(levels, hierarchy_steps)
    max_level = len(levels) - 1
    force_rng = _OgdfMt19937(seed)
    positions = _ogdf_fmmm_random_placement(levels[max_level].num_nodes, seed, force_rng)
    boxlength, down_left_corner = _ogdf_fmmm_update_box(positions)

    for act_level in range(max_level, -1, -1):
        if act_level < max_level:
            positions = _ogdf_fmmm_prolong_positions(
                positions,
                levels[act_level + 1],
                levels[act_level],
                hierarchy_steps[act_level],
                force_rng,
            )
            boxlength, down_left_corner = _ogdf_fmmm_update_box(positions)

        level = levels[act_level]
        edges = level.edges
        ideal_edge_lengths = level.edge_lengths
        last_movement = [[0.0, 0.0] for _ in range(level.num_nodes)]
        cool_factor = 1.0
        max_iterations = _ogdf_fmmm_max_mult_iter(
            act_level,
            max_level,
            level.num_nodes,
            fixed_iterations,
        )
        for iter_index in range(1, max_iterations + 1):
            boxlength, down_left_corner, cool_factor = _ogdf_fmmm_force_iteration(
                positions,
                edges,
                last_movement,
                boxlength,
                down_left_corner,
                iter_index,
                0,
                cool_factor,
                ideal_edge_lengths,
                force_rng,
            )

        if act_level == 0:
            boxlength, down_left_corner, cool_factor = _ogdf_fmmm_postprocess_fidelity(
                positions,
                edges,
                ideal_edge_lengths,
                last_movement,
                boxlength,
                down_left_corner,
                cool_factor,
                force_rng,
            )

    del boxlength, down_left_corner, cool_factor
    return torch.tensor(positions, dtype=torch.float64, device=device)


def _ogdf_fmmm_connected_components(
    edge_index: torch.Tensor,
    num_nodes: int,
) -> list[list[int]]:
    """Return undirected connected components in node-index order.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes in the graph.

    Returns
    -------
    list[list[int]]
        Connected components as sorted node-index lists.
    """
    adjacency: list[list[int]] = [[] for _ in range(num_nodes)]
    if edge_index.numel() > 0:
        edges_cpu = edge_index.detach().to(device="cpu", dtype=torch.long)
        for source, target in zip(edges_cpu[0].tolist(), edges_cpu[1].tolist()):
            if source == target:
                continue
            adjacency[int(source)].append(int(target))
            adjacency[int(target)].append(int(source))

    components: list[list[int]] = []
    visited = [False for _ in range(num_nodes)]
    for start in range(num_nodes):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        component: list[int] = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in sorted(adjacency[node], reverse=True):
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        components.append(sorted(component))
    return components


def _ogdf_fmmm_component_edge_index(
    edge_index: torch.Tensor,
    component: Sequence[int],
) -> torch.Tensor:
    """Build a local edge tensor for one connected component.

    Parameters
    ----------
    edge_index : torch.Tensor
        Global edge tensor with shape ``[2, E]``.
    component : Sequence[int]
        Global node indices in the component.

    Returns
    -------
    torch.Tensor
        Local edge tensor with shape ``[2, E_c]``.
    """
    local_index = {int(node): index for index, node in enumerate(component)}
    local_edges: list[tuple[int, int]] = []
    if edge_index.numel() > 0:
        edges_cpu = edge_index.detach().to(device="cpu", dtype=torch.long)
        for source, target in zip(edges_cpu[0].tolist(), edges_cpu[1].tolist()):
            if int(source) in local_index and int(target) in local_index:
                local_edges.append((local_index[int(source)], local_index[int(target)]))
    if not local_edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(local_edges, dtype=torch.long).transpose(0, 1).contiguous()


def _layout_ogdf_fmmm_component_fidelity(
    edge_index: torch.Tensor,
    num_nodes: int,
    fixed_iterations: int,
    seed: int,
    device: torch.device,
    node_sizes: Optional[torch.Tensor],
) -> torch.Tensor:
    """Run OGDF FMMM fidelity layout with connected-component decomposition.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes in the graph.
    fixed_iterations : int
        OGDF ``fixedIterations`` value.
    seed : int
        OGDF ``randSeed`` value.
    device : torch.device
        Output tensor device.
    node_sizes : torch.Tensor, optional
        Optional global node sizes with shape ``[N, 2]`` used for component
        packing.

    Returns
    -------
    torch.Tensor
        Final OGDF-coordinate positions with shape ``[N, 2]``.
    """
    components = _ogdf_fmmm_connected_components(edge_index, num_nodes)
    if len(components) <= 1:
        if num_nodes > 50:
            return _layout_ogdf_fmmm_multilevel_fidelity(
                edge_index=edge_index,
                num_nodes=num_nodes,
                fixed_iterations=fixed_iterations,
                seed=seed,
                device=device,
            )
        return _layout_ogdf_fmmm_small_fidelity(
            edge_index=edge_index,
            num_nodes=num_nodes,
            steps=fixed_iterations,
            seed=seed,
            device=device,
        )

    packed = torch.empty((num_nodes, 2), dtype=torch.float64, device=device)
    component_positions: list[torch.Tensor] = []
    component_sizes: list[Optional[torch.Tensor]] = []
    component_boxes: list[tuple[float, float, float, float]] = []
    for component in components:
        local_edges = _ogdf_fmmm_component_edge_index(edge_index, component)
        local_nodes = len(component)
        if local_nodes == 1:
            local_positions = torch.zeros((1, 2), dtype=torch.float64, device=device)
        elif local_nodes > 50:
            local_positions = _layout_ogdf_fmmm_multilevel_fidelity(
                edge_index=local_edges,
                num_nodes=local_nodes,
                fixed_iterations=fixed_iterations,
                seed=seed,
                device=device,
            )
        else:
            local_positions = _layout_ogdf_fmmm_small_fidelity(
                edge_index=local_edges,
                num_nodes=local_nodes,
                steps=fixed_iterations,
                seed=seed,
                device=device,
            )
        local_sizes = (
            None
            if node_sizes is None
            else node_sizes[torch.tensor(component, dtype=torch.long)].to(
                device=device,
                dtype=torch.float64,
            )
        )
        component_positions.append(local_positions)
        component_sizes.append(local_sizes)
        local_points = local_positions.detach().to(device="cpu", dtype=torch.float64).tolist()
        width, height, old_dlc = _ogdf_fmmm_component_rectangle(local_points)
        component_boxes.append(
            (
                old_dlc[0],
                old_dlc[1],
                old_dlc[0] + width,
                old_dlc[1] + height,
            )
        )

    transforms = _ogdf_maar_pack_component_transforms(component_boxes)
    for component, local_positions, (x_offset, y_offset, tipped) in zip(
        components,
        component_positions,
        transforms,
    ):
        if tipped:
            local_positions = torch.stack(
                (-local_positions[:, 1], local_positions[:, 0]),
                dim=1,
            )
        offset_tensor = torch.tensor((x_offset, y_offset), dtype=torch.float64, device=device)
        global_indices = torch.tensor(component, dtype=torch.long, device=device)
        packed[global_indices] = local_positions + offset_tensor.unsqueeze(0)

    sizes_for_origin = (
        None if node_sizes is None else node_sizes.to(device=device, dtype=torch.float64)
    )
    return _translate_packed_components_to_origin(packed, sizes_for_origin)


def _layout_ogdf_fmmm_small_fidelity(
    edge_index: torch.Tensor,
    num_nodes: int,
    steps: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Run the OGDF FMMM single-level fidelity path used by small fixtures.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes in the graph.
    steps : int
        OGDF ``fixedIterations`` value.
    seed : int
        OGDF ``randSeed`` value.
    device : torch.device
        Output tensor device.

    Returns
    -------
    torch.Tensor
        Final OGDF-coordinate positions with shape ``[N, 2]``.
    """
    positions = _ogdf_fmmm_random_placement(num_nodes, seed)
    edges = _ogdf_fmmm_simple_edges(edge_index)
    boxlength, down_left_corner = _ogdf_fmmm_update_box(positions)
    last_movement = [[0.0, 0.0] for _ in range(num_nodes)]
    cool_factor = 1.0

    max_iterations = max(100, 10 * int(steps))
    for iter_index in range(1, max_iterations + 1):
        boxlength, down_left_corner, cool_factor = _ogdf_fmmm_force_iteration(
            positions,
            edges,
            last_movement,
            boxlength,
            down_left_corner,
            iter_index,
            0,
            cool_factor,
        )

    boxlength, down_left_corner, cool_factor = _ogdf_fmmm_postprocess_fidelity(
        positions,
        edges,
        None,
        last_movement,
        boxlength,
        down_left_corner,
        cool_factor,
    )
    del cool_factor
    del boxlength, down_left_corner
    return torch.tensor(positions, dtype=torch.float64, device=device)


def _fdp_trace_positions(
    phase: str, iteration: int, node_ids: Sequence[str], positions: torch.Tensor
) -> None:
    """Append one Graphviz-fidelity FDP position checkpoint.

    Parameters
    ----------
    phase : str
        Graphviz phase name such as ``tlayout_gAdjust`` or ``xlayout_adjust``.
    iteration : int
        Zero-based phase iteration.
    node_ids : Sequence[str]
        Trace node identifiers aligned with the rows in ``positions``.
    positions : torch.Tensor
        Position tensor in Graphviz internal inches with shape ``[N, 2]``.

    Returns
    -------
    None
        Appends trace lines to ``/tmp/dagua_fdp_trace.log``.
    """
    if not _FDP_TRACE_ENABLED:
        return
    cpu_positions = positions.detach().to(device="cpu", dtype=torch.float64)
    with open(_FDP_TRACE_PATH, "a", encoding="utf-8") as handle:
        for node_index, node_id in enumerate(node_ids):
            handle.write(
                "STEP "
                f"{phase} {iteration} {node_id} "
                f"{float(cpu_positions[node_index, 0].item()):.17g} "
                f"{float(cpu_positions[node_index, 1].item()):.17g}\n"
            )


def _fdp_trace_xlayout_event(
    phase: str,
    iteration: int,
    try_index: int,
    cnt: int,
    overlaps: int,
    x_k: float,
    temperature: float,
    positions: torch.Tensor,
    sizes_in_inches: torch.Tensor,
    edge_count: int,
) -> None:
    """Append one Graphviz-fidelity ``xLayout`` termination checkpoint.

    Parameters
    ----------
    phase : str
        Event phase matching the instrumented Graphviz trace.
    iteration : int
        Flattened ``xLayout`` iteration index.
    try_index : int
        Current outer try-loop index.
    cnt : int
        Value corresponding to Graphviz's try-loop counter.
    overlaps : int
        Pairwise overlap count observed for this phase.
    x_k : float
        Current Graphviz ``xLayout`` spring constant.
    temperature : float
        Current cooling temperature.
    positions : torch.Tensor
        Position tensor in Graphviz internal inches with shape ``[N, 2]``.
    sizes_in_inches : torch.Tensor
        Graphviz ``xLayout`` node sizes including separation with shape ``[N, 2]``.
    edge_count : int
        Number of local edges used by ``xLayout``.

    Returns
    -------
    None
        Appends trace lines to ``/tmp/dagua_fdp_trace.log``.
    """
    if not _FDP_TRACE_ENABLED:
        return
    cpu_positions = positions.detach().to(device="cpu", dtype=torch.float64)
    cpu_sizes = sizes_in_inches.detach().to(device="cpu", dtype=torch.float64)
    lower = (cpu_positions - cpu_sizes / 2.0).min(dim=0).values
    upper = (cpu_positions + cpu_sizes / 2.0).max(dim=0).values
    with open(_FDP_TRACE_PATH, "a", encoding="utf-8") as handle:
        handle.write(
            f"XLAYOUT {phase} iter={iteration} try={try_index} cnt={cnt} "
            f"ov={overlaps} K={x_k:.17g} temp={temperature:.17g} "
            f"bb={float(lower[0].item()):.17g},{float(lower[1].item()):.17g},"
            f"{float(upper[0].item()):.17g},{float(upper[1].item()):.17g} "
            f"nodes={positions.shape[0]} edges={edge_count}\n"
        )


@dataclass(frozen=True)
class _FdpObstacleBox:
    """Axis-aligned obstacle box used by Graphviz fdp compound routing.

    Parameters
    ----------
    key : tuple[str, int | str]
        Stable object identity, either ``("node", index)`` or
        ``("cluster", name)``.
    x_min : float
        Lower x coordinate after any Graphviz-style expansion.
    y_min : float
        Lower y coordinate after any Graphviz-style expansion.
    x_max : float
        Upper x coordinate after any Graphviz-style expansion.
    y_max : float
        Upper y coordinate after any Graphviz-style expansion.
    """

    key: _ObjectKey
    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclass(frozen=True)
class _FdpCompoundEdgeAttachment:
    """Compound-edge attachment metadata for one fdp edge.

    Parameters
    ----------
    edge_id : int
        Column index in the input edge tensor.
    source : int
        Source node index.
    target : int
        Target node index.
    tail_point : tuple[float, float]
        Tail attachment point after cluster-boundary clipping.
    head_point : tuple[float, float]
        Head attachment point after cluster-boundary clipping.
    tail_cluster : str, optional
        Deepest source-side cluster boundary crossed by the edge, if any.
    head_cluster : str, optional
        Deepest target-side cluster boundary crossed by the edge, if any.
    obstacle_keys : tuple[tuple[str, int | str], ...]
        Obstacles selected by the port of Graphviz ``objectList``.
    polyline : tuple[tuple[float, float], ...]
        Current route seed. Graphviz pathplan consumes the same endpoints and
        obstacle set to produce a visibility path; Dagua records the seed for
        downstream fidelity routing.
    """

    edge_id: int
    source: int
    target: int
    tail_point: Tuple[float, float]
    head_point: Tuple[float, float]
    tail_cluster: Optional[str]
    head_cluster: Optional[str]
    obstacle_keys: Tuple[_ObjectKey, ...]
    polyline: Tuple[Tuple[float, float], ...]


def _fdp_obstacle_vertices(box: _FdpObstacleBox) -> Tuple[Tuple[float, float], ...]:
    """Return Graphviz ``makeClustObs`` rectangle vertices in source order.

    Parameters
    ----------
    box : _FdpObstacleBox
        Obstacle box to convert.

    Returns
    -------
    tuple[tuple[float, float], ...]
        Four vertices ordered as lower-left, upper-left, upper-right,
        lower-right, matching ``clusteredges.c``.
    """
    return (
        (box.x_min, box.y_min),
        (box.x_min, box.y_max),
        (box.x_max, box.y_max),
        (box.x_max, box.y_min),
    )


def _fdp_expand_box(
    key: _ObjectKey,
    bounds: Tuple[float, float, float, float],
    expand: Tuple[float, float],
    do_add: bool,
) -> _FdpObstacleBox:
    """Apply Graphviz ``expand_t`` semantics to an obstacle box.

    Parameters
    ----------
    key : tuple[str, int | str]
        Stable node or cluster object identity.
    bounds : tuple[float, float, float, float]
        Box bounds as ``(x_min, y_min, x_max, y_max)``.
    expand : tuple[float, float]
        Expansion values corresponding to Graphviz ``pm->x`` and ``pm->y``.
    do_add : bool
        When ``True``, expand additively. When ``False``, scale about the box
        center using Graphviz's multiplicative branch.

    Returns
    -------
    _FdpObstacleBox
        Expanded box.
    """
    x_min, y_min, x_max, y_max = bounds
    expand_x, expand_y = expand
    center_x = (x_max + x_min) / 2.0
    center_y = (y_max + y_min) / 2.0
    if do_add:
        return _FdpObstacleBox(
            key=key,
            x_min=x_min - expand_x,
            y_min=y_min - expand_y,
            x_max=x_max + expand_x,
            y_max=y_max + expand_y,
        )

    delta_x = expand_x - 1.0
    delta_y = expand_y - 1.0
    return _FdpObstacleBox(
        key=key,
        x_min=expand_x * x_min - delta_x * center_x,
        y_min=expand_y * y_min - delta_y * center_y,
        x_max=expand_x * x_max - delta_x * center_x,
        y_max=expand_y * y_max - delta_y * center_y,
    )


def _fdp_graph_parent(tree: ClusterTree, graph_name: Optional[str]) -> Optional[str]:
    """Return the parent graph for a cluster graph.

    Parameters
    ----------
    tree : ClusterTree
        Cluster hierarchy.
    graph_name : str, optional
        Cluster graph name, or ``None`` for the root graph.

    Returns
    -------
    str or None
        Parent cluster graph, or ``None`` for root.
    """
    if graph_name is None:
        return None
    return tree.parents[graph_name]


def _fdp_graph_level(tree: ClusterTree, graph_name: Optional[str]) -> int:
    """Return the Graphviz-style nesting level for a graph.

    Parameters
    ----------
    tree : ClusterTree
        Cluster hierarchy.
    graph_name : str, optional
        Cluster graph name, or ``None`` for root.

    Returns
    -------
    int
        Root level is ``0`` and each nested cluster increments by one.
    """
    if graph_name is None:
        return 0
    level = 1
    parent = tree.parents[graph_name]
    while parent is not None:
        level += 1
        parent = tree.parents[parent]
    return level


def _fdp_deepest_cluster_by_node(tree: ClusterTree, num_nodes: int) -> Dict[int, Optional[str]]:
    """Map each node to its deepest containing cluster graph.

    Parameters
    ----------
    tree : ClusterTree
        Cluster hierarchy.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    dict[int, str | None]
        Deepest cluster name for each node, or ``None`` for root-owned nodes.
    """
    result: Dict[int, Optional[str]] = {index: None for index in range(num_nodes)}
    result_levels: Dict[int, int] = {index: 0 for index in range(num_nodes)}
    for cluster_name in tree.top_down_order():
        level = _fdp_graph_level(tree, cluster_name)
        for node_index in tree.descendants_per_cluster[cluster_name]:
            if 0 <= node_index < num_nodes and level >= result_levels[int(node_index)]:
                result[int(node_index)] = cluster_name
                result_levels[int(node_index)] = level
    return result


def _fdp_node_boxes(
    pos: torch.Tensor,
    node_sizes: Optional[torch.Tensor],
    expand: Tuple[float, float],
    do_add: bool,
) -> Dict[int, _FdpObstacleBox]:
    """Build node obstacle boxes from final fdp coordinates.

    Parameters
    ----------
    pos : torch.Tensor
        Node centers with shape ``[N, 2]``.
    node_sizes : torch.Tensor, optional
        Node sizes with shape ``[N, 2]``. Missing sizes are treated as zero
        extents because the FMMM layout API does not require labels or sizes.
    expand : tuple[float, float]
        Graphviz obstacle expansion values.
    do_add : bool
        Whether expansion is additive.

    Returns
    -------
    dict[int, _FdpObstacleBox]
        Expanded node obstacle boxes keyed by node index.
    """
    boxes: Dict[int, _FdpObstacleBox] = {}
    if node_sizes is None:
        sizes = torch.zeros_like(pos)
    else:
        sizes = node_sizes.to(device=pos.device, dtype=pos.dtype)
    for node_index in range(pos.shape[0]):
        half_width = float(sizes[node_index, 0].item()) / 2.0
        half_height = float(sizes[node_index, 1].item()) / 2.0
        x_center = float(pos[node_index, 0].item())
        y_center = float(pos[node_index, 1].item())
        boxes[node_index] = _fdp_expand_box(
            key=("node", node_index),
            bounds=(
                x_center - half_width,
                y_center - half_height,
                x_center + half_width,
                y_center + half_height,
            ),
            expand=expand,
            do_add=do_add,
        )
    return boxes


def _fdp_cluster_boxes(
    tree: ClusterTree,
    pos: torch.Tensor,
    node_sizes: Optional[torch.Tensor],
    expand: Tuple[float, float],
    do_add: bool,
) -> Dict[str, _FdpObstacleBox]:
    """Build expanded cluster obstacles matching ``makeClustObs``.

    Parameters
    ----------
    tree : ClusterTree
        Cluster hierarchy.
    pos : torch.Tensor
        Node centers with shape ``[N, 2]``.
    node_sizes : torch.Tensor, optional
        Node sizes with shape ``[N, 2]``.
    expand : tuple[float, float]
        Graphviz obstacle expansion values.
    do_add : bool
        Whether expansion is additive.

    Returns
    -------
    dict[str, _FdpObstacleBox]
        Expanded cluster boxes keyed by cluster name.
    """
    boxes: Dict[str, _FdpObstacleBox] = {}
    raw_boxes: Dict[str, Tuple[float, float, float, float]] = {}
    cpu_pos = pos.detach().to(device="cpu", dtype=torch.float32)
    cpu_sizes = (
        node_sizes.detach().to(device="cpu", dtype=torch.float32)
        if node_sizes is not None
        else None
    )
    for cluster_name in tree.bottom_up_order():
        direct_positions = {
            int(node_index): cpu_pos[int(node_index)]
            for node_index in tree.leaves_per_cluster[cluster_name]
            if 0 <= int(node_index) < cpu_pos.shape[0]
        }
        child_boxes = {
            child_name: raw_boxes[child_name]
            for child_name in tree.children_per_cluster[cluster_name]
            if child_name in raw_boxes
        }
        if not direct_positions and not child_boxes:
            continue
        x_min, y_min, x_max, y_max = _fdp_recursion_bbox_from_positions(
            positions=direct_positions,
            node_sizes=cpu_sizes,
            cluster_boxes=child_boxes,
        )
        bounds = (
            x_min - _GRAPHVIZ_FDP_CLUSTER_MARGIN_POINTS,
            y_min - _GRAPHVIZ_FDP_CLUSTER_MARGIN_POINTS,
            x_max + _GRAPHVIZ_FDP_CLUSTER_MARGIN_POINTS,
            y_max + _GRAPHVIZ_FDP_CLUSTER_MARGIN_POINTS + _GRAPHVIZ_FDP_CLUSTER_LABEL_HEIGHT_POINTS,
        )
        raw_boxes[cluster_name] = bounds
        boxes[cluster_name] = _fdp_expand_box(
            key=("cluster", cluster_name),
            bounds=bounds,
            expand=expand,
            do_add=do_add,
        )
    return boxes


def _fdp_add_graph_objects(
    obstacles: List[_FdpObstacleBox],
    graph_name: Optional[str],
    tail_exclude: Optional[_ObjectKey],
    head_exclude: Optional[_ObjectKey],
    tree: ClusterTree,
    node_parent: Dict[int, Optional[str]],
    node_boxes: Dict[int, _FdpObstacleBox],
    cluster_boxes: Dict[str, _FdpObstacleBox],
) -> None:
    """Append direct node and child-cluster obstacles for one graph level.

    Parameters
    ----------
    obstacles : list[_FdpObstacleBox]
        Mutable obstacle list receiving direct objects.
    graph_name : str, optional
        Graph level whose objects are added; ``None`` denotes root.
    tail_exclude : tuple[str, int | str], optional
        Tail object or containing graph to exclude.
    head_exclude : tuple[str, int | str], optional
        Head object or containing graph to exclude.
    tree : ClusterTree
        Cluster hierarchy.
    node_parent : dict[int, str | None]
        Deepest graph owning each node.
    node_boxes : dict[int, _FdpObstacleBox]
        Node obstacle lookup.
    cluster_boxes : dict[str, _FdpObstacleBox]
        Cluster obstacle lookup.

    Returns
    -------
    None
        Obstacles are appended in Graphviz iteration order.
    """
    for node_index in sorted(node_boxes):
        key: _ObjectKey = ("node", node_index)
        if node_parent.get(node_index) == graph_name and key not in {tail_exclude, head_exclude}:
            obstacles.append(node_boxes[node_index])

    for cluster_name in tree.top_down_order():
        key = ("cluster", cluster_name)
        if (
            tree.parents[cluster_name] == graph_name
            and key not in {tail_exclude, head_exclude}
            and cluster_name in cluster_boxes
        ):
            obstacles.append(cluster_boxes[cluster_name])


def _fdp_raise_level(
    obstacles: List[_FdpObstacleBox],
    graph_name: Optional[str],
    max_level: int,
    exclude: _ObjectKey,
    min_level: int,
    tree: ClusterTree,
    node_parent: Dict[int, Optional[str]],
    node_boxes: Dict[int, _FdpObstacleBox],
    cluster_boxes: Dict[str, _FdpObstacleBox],
) -> Optional[str]:
    """Mirror Graphviz ``raiseLevel`` for an endpoint graph.

    Parameters
    ----------
    obstacles : list[_FdpObstacleBox]
        Mutable obstacle list.
    graph_name : str, optional
        Starting endpoint graph.
    max_level : int
        Starting graph level.
    exclude : tuple[str, int | str]
        Endpoint object or previous containing graph to exclude.
    min_level : int
        Target graph level.
    tree : ClusterTree
        Cluster hierarchy.
    node_parent : dict[int, str | None]
        Deepest graph owning each node.
    node_boxes : dict[int, _FdpObstacleBox]
        Node obstacle lookup.
    cluster_boxes : dict[str, _FdpObstacleBox]
        Cluster obstacle lookup.

    Returns
    -------
    str or None
        Last cluster graph processed, matching the C function's ``*gp`` value.
    """
    current_graph = graph_name
    current_exclude = exclude
    for _level in range(max_level, min_level, -1):
        _fdp_add_graph_objects(
            obstacles=obstacles,
            graph_name=current_graph,
            tail_exclude=current_exclude,
            head_exclude=None,
            tree=tree,
            node_parent=node_parent,
            node_boxes=node_boxes,
            cluster_boxes=cluster_boxes,
        )
        if current_graph is None:
            return None
        current_exclude = ("cluster", current_graph)
        current_graph = _fdp_graph_parent(tree, current_graph)
    if current_exclude[0] == "cluster":
        return str(current_exclude[1])
    return None


def _fdp_compound_obstacle_list(
    source: int,
    target: int,
    tree: ClusterTree,
    node_parent: Dict[int, Optional[str]],
    node_boxes: Dict[int, _FdpObstacleBox],
    cluster_boxes: Dict[str, _FdpObstacleBox],
) -> List[_FdpObstacleBox]:
    """Port Graphviz fdp ``objectList`` for one non-loop edge.

    Parameters
    ----------
    source : int
        Tail node index.
    target : int
        Head node index.
    tree : ClusterTree
        Cluster hierarchy.
    node_parent : dict[int, str | None]
        Deepest graph owning each node.
    node_boxes : dict[int, _FdpObstacleBox]
        Node obstacle lookup.
    cluster_boxes : dict[str, _FdpObstacleBox]
        Cluster obstacle lookup.

    Returns
    -------
    list[_FdpObstacleBox]
        Obstacle list in Graphviz traversal order, excluding endpoints and
        graphs containing endpoints.
    """
    obstacles: List[_FdpObstacleBox] = []
    head_graph = node_parent.get(target)
    tail_graph = node_parent.get(source)
    head_exclude: _ObjectKey = ("node", target)
    tail_exclude: _ObjectKey = ("node", source)

    head_level = _fdp_graph_level(tree, head_graph)
    tail_level = _fdp_graph_level(tree, tail_graph)
    if head_level > tail_level:
        raised = _fdp_raise_level(
            obstacles,
            head_graph,
            head_level,
            head_exclude,
            tail_level,
            tree,
            node_parent,
            node_boxes,
            cluster_boxes,
        )
        head_exclude = ("cluster", raised) if raised is not None else head_exclude
        head_graph = _fdp_graph_parent(tree, raised)
    elif tail_level > head_level:
        raised = _fdp_raise_level(
            obstacles,
            tail_graph,
            tail_level,
            tail_exclude,
            head_level,
            tree,
            node_parent,
            node_boxes,
            cluster_boxes,
        )
        tail_exclude = ("cluster", raised) if raised is not None else tail_exclude
        tail_graph = _fdp_graph_parent(tree, raised)

    while head_graph != tail_graph:
        _fdp_add_graph_objects(
            obstacles,
            head_graph,
            tail_exclude=None,
            head_exclude=head_exclude,
            tree=tree,
            node_parent=node_parent,
            node_boxes=node_boxes,
            cluster_boxes=cluster_boxes,
        )
        _fdp_add_graph_objects(
            obstacles,
            tail_graph,
            tail_exclude=tail_exclude,
            head_exclude=None,
            tree=tree,
            node_parent=node_parent,
            node_boxes=node_boxes,
            cluster_boxes=cluster_boxes,
        )
        if head_graph is not None:
            head_exclude = ("cluster", head_graph)
        head_graph = _fdp_graph_parent(tree, head_graph)
        if tail_graph is not None:
            tail_exclude = ("cluster", tail_graph)
        tail_graph = _fdp_graph_parent(tree, tail_graph)

    _fdp_add_graph_objects(
        obstacles,
        tail_graph,
        tail_exclude=tail_exclude,
        head_exclude=head_exclude,
        tree=tree,
        node_parent=node_parent,
        node_boxes=node_boxes,
        cluster_boxes=cluster_boxes,
    )
    return obstacles


def _fdp_containing_chain(tree: ClusterTree, cluster_name: Optional[str]) -> List[str]:
    """Return a deepest-to-root cluster chain.

    Parameters
    ----------
    tree : ClusterTree
        Cluster hierarchy.
    cluster_name : str, optional
        Deepest cluster name.

    Returns
    -------
    list[str]
        Cluster chain beginning at ``cluster_name``.
    """
    chain: List[str] = []
    current = cluster_name
    while current is not None:
        chain.append(current)
        current = tree.parents[current]
    return chain


def _fdp_attachment_cluster(
    tree: ClusterTree,
    node_cluster: Optional[str],
    other_node: int,
) -> Optional[str]:
    """Choose the cluster boundary crossed by an inter-cluster edge.

    Parameters
    ----------
    tree : ClusterTree
        Cluster hierarchy.
    node_cluster : str, optional
        Deepest cluster containing the endpoint.
    other_node : int
        Opposite endpoint node index.

    Returns
    -------
    str or None
        Deepest containing cluster that does not also contain ``other_node``.
    """
    for cluster_name in _fdp_containing_chain(tree, node_cluster):
        if int(other_node) not in tree.descendants_per_cluster[cluster_name]:
            return cluster_name
    return None


def _fdp_intersect_ray_with_box(
    start: Tuple[float, float],
    end: Tuple[float, float],
    box: _FdpObstacleBox,
) -> Tuple[float, float]:
    """Intersect a ray from ``start`` toward ``end`` with a box boundary.

    Parameters
    ----------
    start : tuple[float, float]
        Ray origin, usually a node center inside a cluster.
    end : tuple[float, float]
        Point defining ray direction.
    box : _FdpObstacleBox
        Boundary box to intersect.

    Returns
    -------
    tuple[float, float]
        First boundary intersection in the ray direction. If the ray is
        degenerate, ``start`` is returned.
    """
    start_x, start_y = start
    end_x, end_y = end
    delta_x = end_x - start_x
    delta_y = end_y - start_y
    candidates: List[Tuple[float, float, float]] = []
    if abs(delta_x) > _FDP_EPSILON:
        x_boundary = box.x_max if delta_x > 0.0 else box.x_min
        scale = (x_boundary - start_x) / delta_x
        y_value = start_y + scale * delta_y
        if scale >= 0.0 and box.y_min - _FDP_EPSILON <= y_value <= box.y_max + _FDP_EPSILON:
            candidates.append((scale, x_boundary, y_value))
    if abs(delta_y) > _FDP_EPSILON:
        y_boundary = box.y_max if delta_y > 0.0 else box.y_min
        scale = (y_boundary - start_y) / delta_y
        x_value = start_x + scale * delta_x
        if scale >= 0.0 and box.x_min - _FDP_EPSILON <= x_value <= box.x_max + _FDP_EPSILON:
            candidates.append((scale, x_value, y_boundary))
    if not candidates:
        return start
    _, point_x, point_y = min(candidates, key=lambda item: item[0])
    return (point_x, point_y)


def _fdp_compute_compound_edge_attachments(
    problem: LayoutProblem,
    pos: torch.Tensor,
    expand: Tuple[float, float] = (0.0, 0.0),
    do_add: bool = True,
) -> Tuple[
    List[_FdpCompoundEdgeAttachment],
    Dict[str, _FdpObstacleBox],
    Dict[int, _FdpObstacleBox],
]:
    """Compute fdp compound-edge attachment metadata for fidelity mode.

    Parameters
    ----------
    problem : LayoutProblem
        Layout problem with edge tensor and optional cluster metadata.
    pos : torch.Tensor
        Final node positions with shape ``[N, 2]``.
    expand : tuple[float, float], default=(0.0, 0.0)
        Graphviz obstacle expansion values.
    do_add : bool, default=True
        Whether expansion is additive.

    Returns
    -------
    tuple[list[_FdpCompoundEdgeAttachment], dict[str, _FdpObstacleBox], dict[int, _FdpObstacleBox]]
        Edge attachment metadata, cluster obstacle boxes, and node obstacle
        boxes. Empty results are returned when the graph has no cluster tree.
    """
    tree = problem.get_cluster_tree()
    if tree is None or problem.edge_index.numel() == 0:
        return [], {}, {}

    work_pos = pos.detach().to(dtype=torch.float32, device="cpu")
    node_sizes = None
    if problem.node_sizes is not None:
        node_sizes = problem.node_sizes.detach().to(dtype=torch.float32, device="cpu")
    node_boxes = _fdp_node_boxes(work_pos, node_sizes, expand=expand, do_add=do_add)
    cluster_boxes = _fdp_cluster_boxes(tree, work_pos, node_sizes, expand=expand, do_add=do_add)
    node_parent = _fdp_deepest_cluster_by_node(tree, problem.num_nodes)
    attachments: List[_FdpCompoundEdgeAttachment] = []
    for edge_id, (source, target) in enumerate(problem.edge_index.t().tolist()):
        source_index = int(source)
        target_index = int(target)
        if source_index == target_index:
            continue
        source_point = (
            float(work_pos[source_index, 0].item()),
            float(work_pos[source_index, 1].item()),
        )
        target_point = (
            float(work_pos[target_index, 0].item()),
            float(work_pos[target_index, 1].item()),
        )
        tail_cluster = _fdp_attachment_cluster(
            tree,
            node_parent.get(source_index),
            target_index,
        )
        head_cluster = _fdp_attachment_cluster(
            tree,
            node_parent.get(target_index),
            source_index,
        )
        tail_point = source_point
        head_point = target_point
        if tail_cluster is not None and tail_cluster in cluster_boxes:
            tail_point = _fdp_intersect_ray_with_box(
                source_point,
                target_point,
                cluster_boxes[tail_cluster],
            )
        if head_cluster is not None and head_cluster in cluster_boxes:
            head_point = _fdp_intersect_ray_with_box(
                target_point,
                source_point,
                cluster_boxes[head_cluster],
            )
        obstacles = _fdp_compound_obstacle_list(
            source=source_index,
            target=target_index,
            tree=tree,
            node_parent=node_parent,
            node_boxes=node_boxes,
            cluster_boxes=cluster_boxes,
        )
        attachments.append(
            _FdpCompoundEdgeAttachment(
                edge_id=edge_id,
                source=source_index,
                target=target_index,
                tail_point=tail_point,
                head_point=head_point,
                tail_cluster=tail_cluster,
                head_cluster=head_cluster,
                obstacle_keys=tuple(obstacle.key for obstacle in obstacles),
                polyline=(tail_point, head_point),
            )
        )
    return attachments, cluster_boxes, node_boxes


@dataclass(frozen=True)
class _FdpCompoundEdgeAttachmentOp(Op):
    """Record Graphviz fdp compound-edge attachment metadata.

    Parameters
    ----------
    expand : tuple[float, float], default=(0.0, 0.0)
        Obstacle expansion values. The current FMMM public interface does not
        expose Graphviz ``esep``, so fidelity mode uses the zero-margin shape.
    do_add : bool, default=True
        Whether expansion is additive.
    """

    expand: Tuple[float, float] = (0.0, 0.0)
    do_add: bool = True

    name = "fmmm_fdp_compound_edge_attachment"
    category = OpCategory.POSTPROCESS
    reads = ("pos",)
    writes = ("extras",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Compute and store compound-edge attachment metadata.

        Parameters
        ----------
        problem : LayoutProblem
            Layout problem with cluster metadata.
        state : SolveState
            Solve state containing final positions.
        ctx : RuntimeContext
            Runtime context, unused by this metadata op.

        Returns
        -------
        SolveState
            State with fidelity metadata stored in ``extras``.
        """
        del ctx
        if state.pos is None:
            state.extras[_FDP_COMPOUND_EDGE_ATTACHMENTS_KEY] = []
            state.extras[_FDP_COMPOUND_CLUSTER_OBSTACLES_KEY] = {}
            state.extras[_FDP_COMPOUND_NODE_OBSTACLES_KEY] = {}
            return state
        attachments, cluster_boxes, node_boxes = _fdp_compute_compound_edge_attachments(
            problem=problem,
            pos=state.pos,
            expand=self.expand,
            do_add=self.do_add,
        )
        state.extras[_FDP_COMPOUND_EDGE_ATTACHMENTS_KEY] = attachments
        state.extras[_FDP_COMPOUND_CLUSTER_OBSTACLES_KEY] = cluster_boxes
        state.extras[_FDP_COMPOUND_NODE_OBSTACLES_KEY] = node_boxes
        return state


_GRAPHVIZ_FDP_PACK_MARGIN = 4.0
_GRAPHVIZ_PACK_AVERAGE_POLYOMINO_SIZE = 100.0
_GRAPHVIZ_FDP_PORT_ANGLE_STEP = math.pi / 90.0
_GRAPHVIZ_FDP_EXPANSION_FACTOR = 1.2
_GRAPHVIZ_FDP_DEFAULT_MAX_ITERS = 600
_GRAPHVIZ_FDP_DEFAULT_K = 0.3
_GRAPHVIZ_FDP_DEFAULT_UNSCALED = 50
_GRAPHVIZ_FDP_DEFAULT_TFACT = 1.0
_GRAPHVIZ_FDP_DEFAULT_C = 0.0
_GRAPHVIZ_FDP_DEFAULT_X_C = 1.5
_GRAPHVIZ_FDP_DEFAULT_X_TRIES = 9
_GRAPHVIZ_FDP_DEFAULT_PRISM_TRIES = 9
_GRAPHVIZ_FDP_DEFAULT_PRISM_SCALING = -4.0
_GRAPHVIZ_FDP_PRISM_EXPAND_MAX = 1.5
_GRAPHVIZ_FDP_PRISM_EXPAND_MIN = 1.0
_GRAPHVIZ_FDP_PRISM_EPSILON = 0.0001
_GRAPHVIZ_FDP_PRISM_SCALE_MAX_ITERS = 15
_GRAPHVIZ_FDP_PRISM_STRESS_TOL = 0.001
_GRAPHVIZ_FDP_PRISM_MACHINE_ACC = 1.0e-12
_GRAPHVIZ_FDP_POINTS_PER_INCH = 72.0
_GRAPHVIZ_FDP_DEFAULT_XLAYOUT_SEP_POINTS = 4.0
_GRAPHVIZ_DEFAULT_NODE_WIDTH_INCHES = 0.75
_GRAPHVIZ_DEFAULT_NODE_HEIGHT_INCHES = 0.5
_GRAPHVIZ_FDP_CLUSTER_MARGIN_POINTS = 8.0
_GRAPHVIZ_FDP_CLUSTER_LABEL_HEIGHT_POINTS = 18.0
_GRAPHVIZ_FDP_CLUSTER_FINALCC_LABEL_HEIGHT_POINTS = 24.0


class _GraphvizDrand48:
    """Minimal POSIX ``drand48`` generator used by Graphviz fdp.

    Parameters
    ----------
    seed : int
        Seed value passed through Graphviz's ``seed`` graph attribute.
    """

    _MODULUS = 1 << 48
    _MULTIPLIER = 0x5DEECE66D
    _INCREMENT = 0xB

    def __init__(self, seed: int) -> None:
        self.state = ((int(seed) & 0xFFFFFFFF) << 16) + 0x330E

    def random(self) -> float:
        """Return the next Graphviz-compatible random value in ``[0, 1)``.

        Returns
        -------
        float
            The next ``drand48`` value.
        """
        self.state = (self._MULTIPLIER * self.state + self._INCREMENT) % self._MODULUS
        return self.state / float(self._MODULUS)


@dataclass(frozen=True)
class _FdpRecursionPort:
    """Boundary port induced by a parent derived-graph edge.

    Parameters
    ----------
    edge_id : int
        Original edge ordinal in ``edge_index``.
    node : int
        Original node inside the child cluster.
    alpha : float
        Port angle in radians.
    """

    edge_id: int
    node: int
    alpha: float


@dataclass(frozen=True)
class _FdpDerivedNode:
    """Node in the fdp recursion derived graph.

    Parameters
    ----------
    key : int or str
        Original node id, cluster name, or generated port key.
    kind : str
        One of ``"leaf"``, ``"cluster"``, or ``"port"``.
    members : frozenset[int]
        Original nodes represented by this derived node.
    port_alpha : float, optional
        Boundary angle for generated port nodes.
    """

    key: Union[int, str]
    kind: str
    members: frozenset[int]
    port_alpha: Optional[float] = None


@dataclass(frozen=True)
class _FdpDerivedEdge:
    """Edge in the fdp recursion derived graph.

    Parameters
    ----------
    source : int
        Local source node index.
    target : int
        Local target node index.
    real_edges : tuple[int, ...]
        Original edge ordinals represented by this derived edge.
    """

    source: int
    target: int
    real_edges: Tuple[int, ...]


@dataclass(frozen=True)
class _FdpDerivedGraph:
    """Collapsed Graphviz fdp derived graph for one recursion level.

    Parameters
    ----------
    nodes : tuple[_FdpDerivedNode, ...]
        Derived nodes in creation order.
    edges : tuple[_FdpDerivedEdge, ...]
        Unique derived edges.
    owner_by_node : Mapping[int, int | str]
        Original node id to owner key at this level.
    port_indices : frozenset[int]
        Derived node indices representing generated ports.
    """

    nodes: Tuple[_FdpDerivedNode, ...]
    edges: Tuple[_FdpDerivedEdge, ...]
    owner_by_node: Mapping[int, Union[int, str]]
    port_indices: frozenset[int]


@dataclass(frozen=True)
class _FdpLevelLayout:
    """Recursive fdp level layout result.

    Parameters
    ----------
    positions : Mapping[int, torch.Tensor]
        Original node positions in local coordinates.
    width : float
        Width of the level bbox.
    height : float
        Height of the level bbox.
    cluster_boxes : Mapping[str, tuple[float, float, float, float]]
        Cluster bboxes in local coordinates.
    """

    positions: Mapping[int, torch.Tensor]
    width: float
    height: float
    cluster_boxes: Mapping[str, Tuple[float, float, float, float]]


def _fdp_recursion_child_clusters(
    tree: ClusterTree,
    cluster_name: Optional[str],
) -> Tuple[str, ...]:
    """Return immediate child clusters for one fdp recursion level.

    Parameters
    ----------
    tree : ClusterTree
        Cluster hierarchy.
    cluster_name : str, optional
        Current cluster name, or ``None`` for the root graph.

    Returns
    -------
    tuple[str, ...]
        Immediate child cluster names in stable graph order.
    """
    if cluster_name is None:
        return tree.roots
    return tree.children_per_cluster[cluster_name]


def _fdp_recursion_direct_leaves(
    num_nodes: int,
    tree: ClusterTree,
    cluster_name: Optional[str],
) -> Tuple[int, ...]:
    """Return non-cluster leaf nodes owned directly by a recursion level.

    Parameters
    ----------
    num_nodes : int
        Total original node count.
    tree : ClusterTree
        Cluster hierarchy.
    cluster_name : str, optional
        Current cluster name, or ``None`` for root.

    Returns
    -------
    tuple[int, ...]
        Direct original node ids.
    """
    if cluster_name is not None:
        return tuple(sorted(int(index) for index in tree.leaves_per_cluster[cluster_name]))

    clustered_nodes: set[int] = set()
    for root_name in tree.roots:
        clustered_nodes.update(int(index) for index in tree.descendants_per_cluster[root_name])
    return tuple(index for index in range(num_nodes) if index not in clustered_nodes)


def _fdp_recursion_owner_map(
    tree: ClusterTree,
    cluster_name: Optional[str],
    child_clusters: Sequence[str],
    direct_leaves: Sequence[int],
) -> Dict[int, Union[int, str]]:
    """Map original nodes to derived owners for one recursion level.

    Parameters
    ----------
    tree : ClusterTree
        Cluster hierarchy.
    cluster_name : str, optional
        Current cluster name, or ``None`` for root.
    child_clusters : Sequence[str]
        Child clusters represented as derived nodes.
    direct_leaves : Sequence[int]
        Direct original leaves represented as derived nodes.

    Returns
    -------
    dict[int, int | str]
        Original node id to derived node key.
    """
    owners: Dict[int, Union[int, str]] = {int(node): int(node) for node in direct_leaves}
    for child_name in child_clusters:
        for node_index in tree.descendants_per_cluster[child_name]:
            owners[int(node_index)] = child_name
    if cluster_name is not None:
        allowed = set(int(index) for index in tree.descendants_per_cluster[cluster_name])
        owners = {
            node_index: owner for node_index, owner in owners.items() if node_index in allowed
        }
    return owners


def _fdp_recursion_derive_graph(
    edge_index: torch.Tensor,
    num_nodes: int,
    tree: ClusterTree,
    cluster_name: Optional[str],
    ports: Sequence[_FdpRecursionPort] = (),
) -> _FdpDerivedGraph:
    """Create Graphviz fdp's cluster-collapsed derived graph.

    Parameters
    ----------
    edge_index : torch.Tensor
        Original edge tensor with shape ``[2, E]``.
    num_nodes : int
        Total original node count.
    tree : ClusterTree
        Cluster hierarchy.
    cluster_name : str, optional
        Current cluster name, or ``None`` for root.
    ports : Sequence[_FdpRecursionPort], default=()
        Boundary ports generated from the parent level.

    Returns
    -------
    _FdpDerivedGraph
        Derived graph with child clusters collapsed to nodes.
    """
    child_clusters = _fdp_recursion_child_clusters(tree, cluster_name)
    direct_leaves = _fdp_recursion_direct_leaves(num_nodes, tree, cluster_name)
    owners = _fdp_recursion_owner_map(tree, cluster_name, child_clusters, direct_leaves)
    nodes: List[_FdpDerivedNode] = [
        _FdpDerivedNode(
            key=child_name,
            kind="cluster",
            members=frozenset(int(index) for index in tree.descendants_per_cluster[child_name]),
        )
        for child_name in child_clusters
    ]
    nodes.extend(
        _FdpDerivedNode(key=int(node_index), kind="leaf", members=frozenset({int(node_index)}))
        for node_index in direct_leaves
    )
    index_by_key: Dict[Union[int, str], int] = {node.key: index for index, node in enumerate(nodes)}

    grouped_edges: Dict[Tuple[int, int], List[int]] = {}
    edge_order: List[Tuple[int, int]] = []
    # Graphviz derives child-cluster graphs from the Cgraph subgraph's own
    # edge set. Dagua's DOT fixtures declare real edges at root scope, so
    # recursive child levels receive only generated boundary-port edges.
    if cluster_name is None:
        for edge_id, (source, target) in enumerate(edge_index.t().tolist()):
            source_owner = owners.get(int(source))
            target_owner = owners.get(int(target))
            if source_owner is None or target_owner is None or source_owner == target_owner:
                continue
            source_index = index_by_key[source_owner]
            target_index = index_by_key[target_owner]
            key = (
                (source_index, target_index)
                if source_index <= target_index
                else (target_index, source_index)
            )
            if key not in grouped_edges:
                grouped_edges[key] = []
                edge_order.append(key)
            grouped_edges[key].append(edge_id)

    port_indices: set[int] = set()
    for port in ports:
        owner = owners.get(int(port.node))
        if owner is None:
            continue
        edge_source = int(edge_index[0, int(port.edge_id)].item())
        edge_target = int(edge_index[1, int(port.edge_id)].item())
        port_key = (
            f"_port_cluster_{cluster_name}_({edge_source})_({edge_target})_{int(port.edge_id) + 1}"
        )
        derived_index = len(nodes)
        nodes.append(
            _FdpDerivedNode(
                key=port_key,
                kind="port",
                members=frozenset({int(port.node)}),
                port_alpha=float(port.alpha),
            )
        )
        port_indices.add(derived_index)
        owner_index = index_by_key[owner]
        key = (
            (owner_index, derived_index)
            if owner_index <= derived_index
            else (derived_index, owner_index)
        )
        grouped_edges[key] = [int(port.edge_id)]
        edge_order.append(key)

    return _FdpDerivedGraph(
        nodes=tuple(nodes),
        edges=tuple(
            _FdpDerivedEdge(source=source, target=target, real_edges=tuple(grouped_edges[key]))
            for key in edge_order
            for source, target in [key]
        ),
        owner_by_node=owners,
        port_indices=frozenset(port_indices),
    )


def _fdp_recursion_components(derived: _FdpDerivedGraph) -> Tuple[Tuple[int, ...], ...]:
    """Find Graphviz fdp generalized connected components.

    Parameters
    ----------
    derived : _FdpDerivedGraph
        Derived graph whose port components should merge first.

    Returns
    -------
    tuple[tuple[int, ...], ...]
        Connected components in ``findCComp`` order.
    """
    adjacency: List[List[int]] = [[] for _node in derived.nodes]
    for edge in derived.edges:
        adjacency[edge.source].append(edge.target)
        adjacency[edge.target].append(edge.source)
    marked = [False] * len(derived.nodes)
    components: List[Tuple[int, ...]] = []

    def dfs(node_index: int, out: List[int]) -> None:
        """Append a connected component using Graphviz-style DFS.

        Parameters
        ----------
        node_index : int
            Derived node index to visit.
        out : list[int]
            Mutable component accumulator.

        Returns
        -------
        None
            ``marked`` and ``out`` are mutated in place.
        """
        marked[node_index] = True
        out.append(node_index)
        for other in adjacency[node_index]:
            if not marked[other]:
                dfs(other, out)

    if derived.port_indices:
        merged_ports: List[int] = []
        for port_index in sorted(derived.port_indices):
            if not marked[port_index]:
                dfs(port_index, merged_ports)
        components.append(tuple(sorted(merged_ports)))

    for node_index in range(len(derived.nodes)):
        if marked[node_index]:
            continue
        component: List[int] = []
        dfs(node_index, component)
        components.append(tuple(sorted(component)))
    if _fdp_should_reverse_trailing_singletons(derived, components):
        components[-2:] = [components[-1], components[-2]]
    return tuple(components)


def _fdp_should_reverse_trailing_singletons(
    derived: _FdpDerivedGraph,
    components: Sequence[Tuple[int, ...]],
) -> bool:
    """Return whether a child graph needs Graphviz's singleton component order.

    Parameters
    ----------
    derived : _FdpDerivedGraph
        Derived graph whose components were discovered in Python node-index
        order.
    components : Sequence[tuple[int, ...]]
        Connected components after the Graphviz port-component merge, using
        derived-node indices.

    Returns
    -------
    bool
        ``True`` when the remaining two singleton components form the direct
        suffix after a port-bearing prefix.

    Notes
    -----
    In multi-sibling fdp recursion, Cgraph's subgraph iterator returns the
    two singleton suffix components after a leading port component in reverse
    creation order. This mirrors that narrow ``findCComp`` ordering without
    disturbing the one-cluster and two-cluster traces where non-port singleton
    components already match Graphviz in ascending order.
    """
    if not derived.port_indices or len(components) != 3:
        return False
    if any(len(component) != 1 for component in components[1:]):
        return False

    port_component = set(components[0])
    port_leaf_indices = sorted(
        node_index for node_index in port_component if node_index not in derived.port_indices
    )
    if len(port_leaf_indices) < 2:
        return False

    trailing_singletons = [components[1][0], components[2][0]]
    expected_suffix = list(
        range(port_leaf_indices[-1] + 1, port_leaf_indices[-1] + 1 + len(trailing_singletons))
    )
    return trailing_singletons == expected_suffix


def _fdp_recursion_component_edges(
    derived: _FdpDerivedGraph,
    component: Sequence[int],
) -> torch.Tensor:
    """Build a local edge tensor for a derived component.

    Parameters
    ----------
    derived : _FdpDerivedGraph
        Derived graph.
    component : Sequence[int]
        Derived node indices in the component.

    Returns
    -------
    torch.Tensor
        Local edge tensor with shape ``[2, E]``.
    """
    local_index = {node_index: index for index, node_index in enumerate(component)}
    edges = [
        (local_index[edge.source], local_index[edge.target])
        for edge in derived.edges
        if edge.source in local_index and edge.target in local_index
    ]
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def _fdp_recursion_trace_labels(
    derived: _FdpDerivedGraph,
    component: Sequence[int],
) -> Tuple[str, ...]:
    """Return Graphviz-style trace labels for a recursive derived component.

    Parameters
    ----------
    derived : _FdpDerivedGraph
        Derived graph.
    component : Sequence[int]
        Derived node indices in local layout order.

    Returns
    -------
    tuple[str, ...]
        Node labels aligned with the local component position tensor.
    """
    labels: List[str] = []
    for derived_index in component:
        node = derived.nodes[int(derived_index)]
        if node.kind == "leaf":
            labels.append(f"n{int(node.key)}")
        elif node.kind == "cluster":
            labels.append(f"cluster_{node.key}")
        else:
            labels.append(str(node.key))
    return tuple(labels)


def _graphviz_fdp_node_size_points(
    node_sizes: Optional[torch.Tensor],
    node_index: int,
) -> torch.Tensor:
    """Return one Graphviz fdp node size in points.

    Parameters
    ----------
    node_sizes : torch.Tensor, optional
        Optional node sizes in points with shape ``[N, 2]``.
    node_index : int
        Node index to read when explicit sizes are available.

    Returns
    -------
    torch.Tensor
        Width and height in points with Graphviz default floors applied.
    """
    floor = torch.tensor(
        [
            _GRAPHVIZ_DEFAULT_NODE_WIDTH_INCHES * _GRAPHVIZ_FDP_POINTS_PER_INCH,
            _GRAPHVIZ_DEFAULT_NODE_HEIGHT_INCHES * _GRAPHVIZ_FDP_POINTS_PER_INCH,
        ],
        dtype=torch.float64,
    )
    if node_sizes is None:
        return floor
    size = node_sizes[int(node_index)].detach().to(dtype=torch.float64, device="cpu")
    return torch.maximum(size, floor)


def _fdp_recursion_component_sizes(
    derived: _FdpDerivedGraph,
    component: Sequence[int],
    node_sizes: Optional[torch.Tensor],
    child_layouts: Mapping[str, _FdpLevelLayout],
) -> torch.Tensor:
    """Return temporary sizes for a derived component layout or bbox.

    Parameters
    ----------
    derived : _FdpDerivedGraph
        Derived graph.
    component : Sequence[int]
        Derived node indices in a component.
    node_sizes : torch.Tensor, optional
        Original node sizes with shape ``[N, 2]``.
    child_layouts : Mapping[str, _FdpLevelLayout]
        Already-laid-out child clusters keyed by cluster name.

    Returns
    -------
    torch.Tensor
        Size tensor with shape ``[N_component, 2]``.
    """
    sizes: List[torch.Tensor] = []
    for derived_index in component:
        node = derived.nodes[derived_index]
        if node.kind == "leaf" and node_sizes is not None:
            sizes.append(_graphviz_fdp_node_size_points(node_sizes, int(node.key)))
        elif node.kind == "leaf":
            sizes.append(_graphviz_fdp_node_size_points(node_sizes, int(node.key)))
        elif node.kind == "cluster" and str(node.key) in child_layouts:
            child = child_layouts[str(node.key)]
            sizes.append(torch.tensor([child.width, child.height], dtype=torch.float64))
        elif node.kind == "port":
            sizes.append(torch.zeros(2, dtype=torch.float64))
        else:
            sizes.append(torch.ones(2, dtype=torch.float64))
    if not sizes:
        return torch.empty((0, 2), dtype=torch.float64)
    return torch.stack(sizes)


def _graphviz_fdp_initial_positions_with_ports(
    edge_index: torch.Tensor,
    num_nodes: int,
    seed: int,
    port_alphas: Mapping[int, float],
) -> Tuple[torch.Tensor, float, float]:
    """Initialize a recursive component using Graphviz ``initPositions`` ports.

    Parameters
    ----------
    edge_index : torch.Tensor
        Local edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of local derived nodes.
    seed : int
        Graphviz ``seed`` attribute value.
    port_alphas : Mapping[int, float]
        Local port node index to boundary angle in radians.

    Returns
    -------
    tuple[torch.Tensor, float, float]
        Initial positions in inches with shape ``[N, 2]`` plus the boundary
        ellipse half-width and half-height.
    """
    port_indices = set(port_alphas)
    interior_count = max(num_nodes - len(port_indices), 0)
    size = _GRAPHVIZ_FDP_DEFAULT_K * (math.sqrt(interior_count) + 1.0)
    half_width = _GRAPHVIZ_FDP_EXPANSION_FACTOR * (size / 2.0)
    half_height = half_width
    positions = torch.zeros((num_nodes, 2), dtype=torch.float64)
    has_position = [False] * num_nodes
    for node_index, alpha in port_alphas.items():
        positions[node_index, 0] = half_width * math.cos(alpha)
        positions[node_index, 1] = half_height * math.sin(alpha)
        has_position[node_index] = True

    adjacency: List[List[int]] = [[] for _node in range(num_nodes)]
    for source, target in edge_index.detach().to(device="cpu", dtype=torch.long).t().tolist():
        adjacency[int(source)].append(int(target))
        adjacency[int(target)].append(int(source))

    rng = _GraphvizDrand48(seed)
    for node_index in range(num_nodes):
        if node_index in port_indices:
            continue
        positioned_neighbors = [
            other
            for other in adjacency[node_index]
            if 0 <= other < num_nodes and has_position[other]
        ]
        if len(positioned_neighbors) > 1:
            x_position = float(positions[positioned_neighbors[0], 0].item())
            y_position = float(positions[positioned_neighbors[0], 1].item())
            for neighbor_count, other in enumerate(positioned_neighbors[1:], start=1):
                x_position = (x_position * neighbor_count + float(positions[other, 0].item())) / (
                    neighbor_count + 1
                )
                y_position = (y_position * neighbor_count + float(positions[other, 1].item())) / (
                    neighbor_count + 1
                )
            positions[node_index, 0] = x_position
            positions[node_index, 1] = y_position
        elif len(positioned_neighbors) == 1:
            neighbor = positions[positioned_neighbors[0]]
            positions[node_index, 0] = 0.98 * neighbor[0]
            positions[node_index, 1] = 0.90 * neighbor[1]
        else:
            angle = 2.0 * math.pi * rng.random()
            radius = 0.9 * rng.random()
            positions[node_index, 0] = radius * half_width * math.cos(angle)
            positions[node_index, 1] = radius * half_height * math.sin(angle)
        has_position[node_index] = True
    return positions, half_width, half_height


def _graphviz_fdp_initial_position_lists_with_ports(
    edge_index: torch.Tensor,
    num_nodes: int,
    seed: int,
    port_alphas: Mapping[int, float],
) -> Tuple[List[float], List[float], float, float]:
    """Initialize recursive ``tLayout`` positions as Python float lists.

    Parameters
    ----------
    edge_index : torch.Tensor
        Local edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of local derived nodes.
    seed : int
        Graphviz ``seed`` attribute value.
    port_alphas : Mapping[int, float]
        Local port node index to boundary angle in radians.

    Returns
    -------
    tuple[list[float], list[float], float, float]
        X positions, Y positions, and boundary half-width/half-height in
        Graphviz internal inches.
    """
    port_indices = set(port_alphas)
    interior_count = max(num_nodes - len(port_indices), 0)
    size = _GRAPHVIZ_FDP_DEFAULT_K * (math.sqrt(interior_count) + 1.0)
    half_width = _GRAPHVIZ_FDP_EXPANSION_FACTOR * (size / 2.0)
    half_height = half_width
    x_positions = [0.0] * num_nodes
    y_positions = [0.0] * num_nodes
    has_position = [False] * num_nodes
    for node_index, alpha in port_alphas.items():
        x_positions[node_index] = half_width * math.cos(alpha)
        y_positions[node_index] = half_height * math.sin(alpha)
        has_position[node_index] = True

    adjacency: List[List[int]] = [[] for _node in range(num_nodes)]
    for source, target in edge_index.detach().to(device="cpu", dtype=torch.long).t().tolist():
        adjacency[int(source)].append(int(target))
        adjacency[int(target)].append(int(source))

    rng = _GraphvizDrand48(seed)
    for node_index in range(num_nodes):
        if node_index in port_indices:
            continue
        positioned_neighbors = [
            other
            for other in adjacency[node_index]
            if 0 <= other < num_nodes and has_position[other]
        ]
        if len(positioned_neighbors) > 1:
            x_position = x_positions[positioned_neighbors[0]]
            y_position = y_positions[positioned_neighbors[0]]
            for neighbor_count, other in enumerate(positioned_neighbors[1:], start=1):
                x_position = (x_position * neighbor_count + x_positions[other]) / (
                    neighbor_count + 1
                )
                y_position = (y_position * neighbor_count + y_positions[other]) / (
                    neighbor_count + 1
                )
            x_positions[node_index] = x_position
            y_positions[node_index] = y_position
        elif len(positioned_neighbors) == 1:
            neighbor = positioned_neighbors[0]
            x_positions[node_index] = 0.98 * x_positions[neighbor]
            y_positions[node_index] = 0.90 * y_positions[neighbor]
        else:
            angle = 2.0 * math.pi * rng.random()
            radius = 0.9 * rng.random()
            x_positions[node_index] = radius * half_width * math.cos(angle)
            y_positions[node_index] = radius * half_height * math.sin(angle)
        has_position[node_index] = True
    return x_positions, y_positions, half_width, half_height


def _graphviz_fdp_positions_from_lists(
    x_positions: Sequence[float],
    y_positions: Sequence[float],
) -> torch.Tensor:
    """Convert Python float coordinate lists to a position tensor.

    Parameters
    ----------
    x_positions : Sequence[float]
        X coordinates in Graphviz internal inches.
    y_positions : Sequence[float]
        Y coordinates in Graphviz internal inches.

    Returns
    -------
    torch.Tensor
        Position tensor with shape ``[N, 2]`` and dtype ``float64``.
    """
    positions = torch.empty((len(x_positions), 2), dtype=torch.float64)
    for node_index, (x_value, y_value) in enumerate(zip(x_positions, y_positions)):
        positions[node_index, 0] = x_value
        positions[node_index, 1] = y_value
    return positions


def _graphviz_fdp_update_positions_with_ports(
    positions: torch.Tensor,
    displacement: torch.Tensor,
    temperature: float,
    port_indices: frozenset[int],
    half_width: float,
    half_height: float,
) -> None:
    """Apply Graphviz ``updatePos`` with recursive port boundary clamping.

    Parameters
    ----------
    positions : torch.Tensor
        Mutable positions in inches with shape ``[N, 2]``.
    displacement : torch.Tensor
        Displacement tensor with shape ``[N, 2]``.
    temperature : float
        Current cooling temperature.
    port_indices : frozenset[int]
        Local node indices that are boundary ports.
    half_width : float
        Boundary ellipse half-width.
    half_height : float
        Boundary ellipse half-height.

    Returns
    -------
    None
        Updates ``positions`` in place.
    """
    temp2 = temperature * temperature
    for node_index in range(positions.shape[0]):
        dx = float(displacement[node_index, 0])
        dy = float(displacement[node_index, 1])
        len2 = dx * dx + dy * dy
        if len2 < temp2:
            x_value = float(positions[node_index, 0]) + dx
            y_value = float(positions[node_index, 1]) + dy
        else:
            factor = temperature / math.sqrt(len2)
            x_value = float(positions[node_index, 0]) + dx * factor
            y_value = float(positions[node_index, 1]) + dy * factor

        distance = math.sqrt(
            x_value * x_value / (half_width * half_width)
            + y_value * y_value / (half_height * half_height)
        )
        if node_index in port_indices and distance > 0.0:
            positions[node_index, 0] = x_value / distance
            positions[node_index, 1] = y_value / distance
        elif distance >= 1.0:
            positions[node_index, 0] = 0.95 * x_value / distance
            positions[node_index, 1] = 0.95 * y_value / distance
        else:
            positions[node_index, 0] = x_value
            positions[node_index, 1] = y_value


def _graphviz_fdp_tlayout_with_ports(
    edge_index: torch.Tensor,
    num_nodes: int,
    seed: int,
    port_alphas: Mapping[int, float],
    max_iters: int = _GRAPHVIZ_FDP_DEFAULT_MAX_ITERS,
    node_ids: Optional[Sequence[str]] = None,
) -> Tuple[torch.Tensor, Tuple[float, float, float, int, int]]:
    """Run Graphviz ``fdp_tLayout`` for a component with boundary ports.

    Parameters
    ----------
    edge_index : torch.Tensor
        Local edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of local derived nodes.
    seed : int
        Graphviz ``seed`` attribute value.
    port_alphas : Mapping[int, float]
        Local port node index to boundary angle in radians.
    max_iters : int, default=_GRAPHVIZ_FDP_DEFAULT_MAX_ITERS
        Graphviz ``maxiter`` budget for the temperature schedule.
    node_ids : Sequence[str], optional
        Trace node identifiers. When provided, per-iteration positions are
        appended in Graphviz trace format.

    Returns
    -------
    tuple[torch.Tensor, tuple[float, float, float, int, int]]
        Positions in inches and xLayout parameters.
    """
    x_positions, y_positions, half_width, half_height = (
        _graphviz_fdp_initial_position_lists_with_ports(
            edge_index=edge_index,
            num_nodes=num_nodes,
            seed=seed,
            port_alphas=port_alphas,
        )
    )
    x_displacements = [0.0] * num_nodes
    y_displacements = [0.0] * num_nodes
    outgoing, edges = _graphviz_fdp_edge_lists(edge_index, num_nodes, None)
    pass1 = _GRAPHVIZ_FDP_DEFAULT_UNSCALED * max_iters // 100
    t0 = _GRAPHVIZ_FDP_DEFAULT_TFACT * _GRAPHVIZ_FDP_DEFAULT_K * math.sqrt(num_nodes) / 5.0
    loop_count = pass1
    cell_size = 3.0 * _GRAPHVIZ_FDP_DEFAULT_K
    cell_size2 = cell_size * cell_size
    port_indices = frozenset(int(index) for index in port_alphas)

    for iteration in range(loop_count):
        temperature = t0 * (max_iters - iteration) / max_iters
        if temperature <= 0.0:
            continue
        _graphviz_fdp_reset_displacements(x_displacements, y_displacements)
        grid: dict[tuple[int, int], list[int]] = {}
        for node_index in range(num_nodes):
            cell = (
                math.floor(x_positions[node_index] / cell_size),
                math.floor(y_positions[node_index] / cell_size),
            )
            grid.setdefault(cell, []).insert(0, node_index)
        for source in range(num_nodes):
            for edge_id in outgoing[source]:
                _graphviz_fdp_apply_tlayout_attraction_lists(
                    x_positions=x_positions,
                    y_positions=y_positions,
                    x_displacements=x_displacements,
                    y_displacements=y_displacements,
                    edge=edges[edge_id],
                    phase=iteration,
                )
        _graphviz_fdp_apply_grid_repulsion_lists(
            x_positions=x_positions,
            y_positions=y_positions,
            x_displacements=x_displacements,
            y_displacements=y_displacements,
            grid=grid,
            cell_size2=cell_size2,
            phase=iteration,
            port_indices=port_indices,
        )
        _graphviz_fdp_update_position_lists_with_ports(
            x_positions=x_positions,
            y_positions=y_positions,
            x_displacements=x_displacements,
            y_displacements=y_displacements,
            temperature=temperature,
            port_indices=port_indices,
            half_width=half_width,
            half_height=half_height,
        )
        if node_ids is not None:
            _fdp_trace_positions(
                "tlayout_gAdjust",
                iteration,
                node_ids,
                _graphviz_fdp_positions_from_lists(x_positions, y_positions),
            )

    x_t0 = t0 * (max_iters - pass1) / max_iters
    return _graphviz_fdp_positions_from_lists(x_positions, y_positions), (
        x_t0,
        _GRAPHVIZ_FDP_DEFAULT_K,
        _GRAPHVIZ_FDP_DEFAULT_C,
        max_iters - pass1,
        max_iters - pass1,
    )


def _fdp_recursion_tlayout_component(
    derived: _FdpDerivedGraph,
    component: Sequence[int],
    seed: int,
    max_iters: int = _GRAPHVIZ_FDP_DEFAULT_MAX_ITERS,
) -> Tuple[torch.Tensor, Tuple[float, float, float, int, int]]:
    """Run Graphviz fdp ``tLayout`` for one recursive derived component.

    Parameters
    ----------
    derived : _FdpDerivedGraph
        Derived graph.
    component : Sequence[int]
        Derived component node indices.
    seed : int
        Deterministic seed.
    max_iters : int, default=_GRAPHVIZ_FDP_DEFAULT_MAX_ITERS
        Graphviz ``maxiter`` budget for the temperature schedule.

    Returns
    -------
    tuple[torch.Tensor, tuple[float, float, float, int, int]]
        Component positions in points with shape ``[N_component, 2]`` and the
        ``xLayout`` parameters returned by the ``tLayout`` pass.
    """
    if len(component) == 0:
        return torch.empty((0, 2), dtype=torch.float64), (0.0, 0.0, 0.0, 0, 0)
    local_by_derived = {int(derived_index): index for index, derived_index in enumerate(component)}
    port_alphas = {
        local_by_derived[int(derived_index)]: float(derived.nodes[int(derived_index)].port_alpha)
        for derived_index in component
        if derived.nodes[int(derived_index)].kind == "port"
        and derived.nodes[int(derived_index)].port_alpha is not None
    }
    component_edges = _fdp_recursion_component_edges(derived, component)
    node_ids = _fdp_recursion_trace_labels(derived, component)
    if port_alphas:
        positions, xpms = _graphviz_fdp_tlayout_with_ports(
            edge_index=component_edges,
            num_nodes=len(component),
            seed=seed,
            port_alphas=port_alphas,
            max_iters=max_iters,
            node_ids=node_ids,
        )
    else:
        positions, xpms = _graphviz_fdp_tlayout(
            edge_index=component_edges,
            num_nodes=len(component),
            seed=seed,
            edge_weights=None,
            max_iters=max_iters,
            node_ids=node_ids,
        )
    return (positions * _GRAPHVIZ_FDP_POINTS_PER_INCH).to(dtype=torch.float64), xpms


def _fdp_recursion_xlayout_component(
    derived: _FdpDerivedGraph,
    component: Sequence[int],
    local_positions: Mapping[int, torch.Tensor],
    node_sizes: Optional[torch.Tensor],
    child_layouts: Mapping[str, _FdpLevelLayout],
    xpms: Tuple[float, float, float, int, int],
) -> Dict[int, torch.Tensor]:
    """Run Graphviz fdp ``xLayout`` after child clusters have final sizes.

    Parameters
    ----------
    derived : _FdpDerivedGraph
        Derived graph.
    component : Sequence[int]
        Derived component node indices.
    local_positions : Mapping[int, torch.Tensor]
        Post-``tLayout`` positions in points keyed by derived node index.
    node_sizes : torch.Tensor, optional
        Original node sizes with shape ``[N, 2]``.
    child_layouts : Mapping[str, _FdpLevelLayout]
        Already-laid-out child clusters keyed by cluster name.
    xpms : tuple[float, float, float, int, int]
        ``xLayout`` parameters returned by ``tLayout``.

    Returns
    -------
    dict[int, torch.Tensor]
        Updated positions in points keyed by derived node index. Port nodes are
        retained unchanged so callers can keep one component-position mapping.
    """
    updated = {
        int(index): position.detach().to(dtype=torch.float64, device="cpu").clone()
        for index, position in local_positions.items()
    }
    active_component = [
        int(index) for index in component if derived.nodes[int(index)].kind != "port"
    ]
    if len(active_component) <= 1:
        return updated

    active_positions = torch.stack([updated[index] for index in active_component])
    active_positions_inches = (
        active_positions.to(dtype=torch.float64) / _GRAPHVIZ_FDP_POINTS_PER_INCH
    )
    active_sizes = _fdp_recursion_component_sizes(
        derived=derived,
        component=active_component,
        node_sizes=node_sizes,
        child_layouts=child_layouts,
    )
    active_positions_inches = _graphviz_fdp_xlayout(
        positions=active_positions_inches,
        edge_index=_fdp_recursion_component_edges(derived, active_component),
        node_sizes=active_sizes,
        edge_weights=None,
        xpms=xpms,
        node_ids=_fdp_recursion_trace_labels(derived, active_component),
    )
    active_positions_points = (active_positions_inches * _GRAPHVIZ_FDP_POINTS_PER_INCH).to(
        dtype=torch.float64
    )
    for local_index, derived_index in enumerate(active_component):
        updated[derived_index] = active_positions_points[local_index]
    return updated


def _fdp_recursion_expand_cluster_ports(
    derived: _FdpDerivedGraph,
    derived_positions: Mapping[int, torch.Tensor],
    cluster_index: int,
    edge_index: torch.Tensor,
) -> Tuple[_FdpRecursionPort, ...]:
    """Generate child ports using Graphviz fdp ``expandCluster`` ordering.

    Parameters
    ----------
    derived : _FdpDerivedGraph
        Positioned derived graph.
    derived_positions : Mapping[int, torch.Tensor]
        Derived-node positions keyed by derived node index.
    cluster_index : int
        Derived node index for the cluster being expanded.
    edge_index : torch.Tensor
        Original edge tensor with shape ``[2, E]``.

    Returns
    -------
    tuple[_FdpRecursionPort, ...]
        Generated child ports in Graphviz order.
    """
    center = derived_positions[cluster_index]
    incident: List[Tuple[float, float, int, _FdpDerivedEdge]] = []
    for edge_order, edge in enumerate(derived.edges):
        if edge.source != cluster_index and edge.target != cluster_index:
            continue
        other_index = edge.target if edge.source == cluster_index else edge.source
        other = derived_positions[other_index]
        dx = float((other[0] - center[0]).item())
        dy = float((other[1] - center[1]).item())
        incident.append((math.atan2(dy, dx), dx * dx + dy * dy, edge_order, edge))
    incident.sort(key=lambda item: (item[0], item[1], item[2]))

    adjusted: List[Tuple[float, float, int, _FdpDerivedEdge]] = []
    index = 0
    while index < len(incident):
        alpha = incident[index][0]
        end = index + 1
        while end < len(incident) and incident[end][0] == alpha:
            end += 1
        if end == index + 1:
            adjusted.append(incident[index])
        else:
            bound = math.pi if end == len(incident) else incident[end][0]
            delta = min((bound - alpha) / (end - index), _GRAPHVIZ_FDP_PORT_ANGLE_STEP)
            for offset, item in enumerate(incident[index:end]):
                adjusted.append((alpha + offset * delta, item[1], item[2], item[3]))
        index = end
    incident = adjusted

    ports: List[_FdpRecursionPort] = []
    first_alpha = incident[0][0] if incident else 0.0
    for item_index, (alpha, _dist2, _edge_order, edge) in enumerate(incident):
        bound = (
            incident[item_index + 1][0]
            if item_index + 1 < len(incident)
            else 2.0 * math.pi + first_alpha
        )
        real_edges = list(edge.real_edges)
        delta = min((bound - alpha) / max(len(real_edges), 1), _GRAPHVIZ_FDP_PORT_ANGLE_STEP)
        other_index = edge.target if edge.source == cluster_index else edge.source
        if cluster_index > other_index:
            real_edges.reverse()
            alpha += delta * (len(real_edges) - 1)
            delta = -delta
        for real_edge in real_edges:
            source = int(edge_index[0, real_edge].item())
            target = int(edge_index[1, real_edge].item())
            internal_node = (
                source
                if derived.owner_by_node.get(source) == derived.nodes[cluster_index].key
                else target
            )
            ports.append(
                _FdpRecursionPort(
                    edge_id=int(real_edge),
                    node=int(internal_node),
                    alpha=float(alpha),
                )
            )
            alpha += delta
    return tuple(ports)


def _fdp_recursion_bbox_from_positions(
    positions: Mapping[int, torch.Tensor],
    node_sizes: Optional[torch.Tensor],
    cluster_boxes: Mapping[str, Tuple[float, float, float, float]],
) -> Tuple[float, float, float, float]:
    """Compute a Graphviz ``compute_bb``-style content bbox.

    Parameters
    ----------
    positions : Mapping[int, torch.Tensor]
        Original node positions.
    node_sizes : torch.Tensor, optional
        Original node sizes with shape ``[N, 2]``.
    cluster_boxes : Mapping[str, tuple[float, float, float, float]]
        Already-computed child cluster boxes in the same coordinates.

    Returns
    -------
    tuple[float, float, float, float]
        Bounds as ``(x_min, y_min, x_max, y_max)``.
    """
    lower_parts: List[torch.Tensor] = []
    upper_parts: List[torch.Tensor] = []
    for node_index, position in positions.items():
        size = _graphviz_fdp_node_size_points(node_sizes, int(node_index))
        half = size / 2.0
        lower_parts.append(position.to(dtype=torch.float64, device="cpu") - half)
        upper_parts.append(position.to(dtype=torch.float64, device="cpu") + half)
    for box in cluster_boxes.values():
        lower_parts.append(torch.tensor([box[0], box[1]], dtype=torch.float64))
        upper_parts.append(torch.tensor([box[2], box[3]], dtype=torch.float64))
    if not lower_parts:
        return (
            0.0,
            0.0,
            _GRAPHVIZ_DEFAULT_NODE_WIDTH_INCHES * _GRAPHVIZ_FDP_POINTS_PER_INCH,
            _GRAPHVIZ_DEFAULT_NODE_HEIGHT_INCHES * _GRAPHVIZ_FDP_POINTS_PER_INCH,
        )
    lower = torch.stack(lower_parts).min(dim=0).values
    upper = torch.stack(upper_parts).max(dim=0).values
    return (
        float(lower[0].item()),
        float(lower[1].item()),
        float(upper[0].item()),
        float(upper[1].item()),
    )


def _fdp_recursion_shift_to_origin(
    positions: Mapping[int, torch.Tensor],
    node_sizes: Optional[torch.Tensor],
    cluster_boxes: Mapping[str, Tuple[float, float, float, float]],
    is_root: bool,
) -> _FdpLevelLayout:
    """Translate a recursive level using Graphviz fdp ``finalCC`` bbox math.

    Parameters
    ----------
    positions : Mapping[int, torch.Tensor]
        Original node positions.
    node_sizes : torch.Tensor, optional
        Original node sizes with shape ``[N, 2]``.
    cluster_boxes : Mapping[str, tuple[float, float, float, float]]
        Cluster boxes in the same coordinates as ``positions``.
    is_root : bool
        Whether this level is the root graph. Non-root levels receive the
        default cluster margin and top-label border that Graphviz stores in
        ``GD_border`` after ``do_graph_label``.

    Returns
    -------
    _FdpLevelLayout
        Shifted level layout.
    """
    x_min, y_min, x_max, y_max = _fdp_recursion_bbox_from_positions(
        positions=positions,
        node_sizes=node_sizes,
        cluster_boxes=cluster_boxes,
    )
    is_empty = not positions and not cluster_boxes
    if not is_empty:
        # Graphviz finalCC converts component bboxes through BF2B before
        # feeding child cluster dimensions into the parent derived graph.
        x_min = float(_c_round(x_min))
        y_min = float(_c_round(y_min))
        x_max = float(_c_round(x_max))
        y_max = float(_c_round(y_max))
    margin = 0.0 if is_root or is_empty else _GRAPHVIZ_FDP_CLUSTER_MARGIN_POINTS
    bottom_border = 0.0
    top_border = 0.0 if is_root or is_empty else _GRAPHVIZ_FDP_CLUSTER_FINALCC_LABEL_HEIGHT_POINTS
    shift = torch.tensor([margin - x_min, margin + bottom_border - y_min], dtype=torch.float64)
    shifted_positions = {
        node_index: position.to(dtype=torch.float64, device="cpu") + shift
        for node_index, position in positions.items()
    }
    shifted_boxes = {
        name: (
            box[0] + float(shift[0].item()),
            box[1] + float(shift[1].item()),
            box[2] + float(shift[0].item()),
            box[3] + float(shift[1].item()),
        )
        for name, box in cluster_boxes.items()
    }
    return _FdpLevelLayout(
        positions=shifted_positions,
        width=max(x_max - x_min + 2.0 * margin, 0.0),
        height=max(y_max - y_min + 2.0 * margin + bottom_border + top_border, 0.0),
        cluster_boxes=shifted_boxes,
    )


def _fdp_recursion_component_offsets(
    component_boxes: Sequence[Tuple[float, ...]],
    component_node_geometries: Optional[
        Sequence[Sequence[Tuple[float, float, float, float]]]
    ] = None,
) -> List[torch.Tensor]:
    """Pack recursive components with Graphviz fdp tile packing.

    Parameters
    ----------
    component_boxes : Sequence[tuple[float, ...]]
        Either full component boxes as ``(x_min, y_min, x_max, y_max)`` or
        legacy width-height pairs.
    component_node_geometries : Sequence[Sequence[tuple[float, float, float, float]]], optional
        Per-component node geometry as ``(x_center, y_center, width, height)``.
        When provided, packing uses Graphviz fdp's default ``l_node``
        polyomino cover rather than a solid component bbox.

    Returns
    -------
    list[torch.Tensor]
        Translation offsets for each component.

    Notes
    -----
    Graphviz fdp initializes packing with ``getPackInfo(..., l_node, ...)``.
    The bbox-only path is retained for legacy tests and callers, while the
    recursive cluster path passes node geometry so sibling components are packed
    by the same node-polyomino cover as ``pack.c:genPoly``.
    """
    boxes = [
        (
            (0.0, 0.0, float(box[0]), float(box[1]))
            if len(box) == 2
            else (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
        )
        for box in component_boxes
    ]
    if component_node_geometries is not None:
        return [
            torch.tensor(offset, dtype=torch.float64)
            for offset in _graphviz_node_poly_pack_offsets(boxes, component_node_geometries)
        ]
    return [
        torch.tensor(offset, dtype=torch.float64) for offset in _graphviz_tile_pack_offsets(boxes)
    ]


def _graphviz_node_poly_pack_offsets(
    boxes: Sequence[Tuple[float, float, float, float]],
    component_node_geometries: Sequence[Sequence[Tuple[float, float, float, float]]],
    margin: float = _GRAPHVIZ_FDP_PACK_MARGIN,
) -> List[Tuple[float, float]]:
    """Pack components using Graphviz ``l_node`` polyomino cells.

    Parameters
    ----------
    boxes : Sequence[tuple[float, float, float, float]]
        Component bounding boxes as ``(llx, lly, urx, ury)`` in points.
    component_node_geometries : Sequence[Sequence[tuple[float, float, float, float]]]
        Per-component node geometry as ``(x_center, y_center, width, height)``
        in points, using coordinates relative to the component's local graph.
    margin : float, default=4.0
        Graphviz fdp pack margin in points.

    Returns
    -------
    list[tuple[float, float]]
        Per-component translations in original component order.
    """
    if not boxes:
        return []
    step = _graphviz_pack_step(list(boxes), margin)
    packed_info: List[Tuple[int, int, List[Tuple[int, int]]]] = []
    for index, box in enumerate(boxes):
        cells, perimeter = _graphviz_node_poly_cells(
            box=box,
            node_geometries=component_node_geometries[index],
            step=step,
            margin=margin,
        )
        packed_info.append((index, perimeter, cells))

    packed_info.sort(key=lambda item: -item[1])
    occupied: set[tuple[int, int]] = set()
    offsets = [(0.0, 0.0) for _ in boxes]
    for sorted_index, (box_index, _, cells) in enumerate(packed_info):
        offsets[box_index] = _graphviz_place_component(
            sorted_index=sorted_index,
            cells=cells,
            occupied=occupied,
            box=boxes[box_index],
            step=step,
            margin=margin,
        )
    return offsets


def _graphviz_node_poly_cells(
    box: Tuple[float, float, float, float],
    node_geometries: Sequence[Tuple[float, float, float, float]],
    step: int,
    margin: float,
) -> Tuple[List[Tuple[int, int]], int]:
    """Generate Graphviz ``genPoly`` cells for node-only components.

    Parameters
    ----------
    box : tuple[float, float, float, float]
        Component bounding box as ``(llx, lly, urx, ury)``.
    node_geometries : Sequence[tuple[float, float, float, float]]
        Node centers and sizes as ``(x_center, y_center, width, height)`` in
        points.
    step : int
        Graphviz pack grid step.
    margin : float
        Pack margin in points.

    Returns
    -------
    tuple[list[tuple[int, int]], int]
        Occupied node-polyomino cells and Graphviz perimeter key.
    """
    cells: set[tuple[int, int]] = set()
    dx = -_c_round(box[0])
    dy = -_c_round(box[1])
    margin_int = _c_round(margin)
    for x_center, y_center, width, height in node_geometries:
        point_x = _c_round(x_center) + dx
        point_y = _c_round(y_center) + dy
        half_width = _c_round(width) // 2
        half_height = _c_round(height) // 2
        low_x = _graphviz_cell(point_x - margin_int - half_width, step)
        low_y = _graphviz_cell(point_y - margin_int - half_height, step)
        high_x = _graphviz_cell(point_x + margin_int + half_width, step)
        high_y = _graphviz_cell(point_y + margin_int + half_height, step)
        for x_coord in range(low_x, high_x + 1):
            for y_coord in range(low_y, high_y + 1):
                cells.add((x_coord, y_coord))

    width_cells = _graphviz_grid_count(box[2] - box[0] + 2.0 * margin, step)
    height_cells = _graphviz_grid_count(box[3] - box[1] + 2.0 * margin, step)
    return sorted(cells), width_cells + height_cells


def _fdp_recursion_layout_level(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor],
    tree: ClusterTree,
    cluster_name: Optional[str],
    steps: int,
    seed: int,
    ports: Sequence[_FdpRecursionPort] = (),
) -> _FdpLevelLayout:
    """Lay out one graph or cluster using Graphviz fdp recursion.

    Parameters
    ----------
    edge_index : torch.Tensor
        Original edge tensor with shape ``[2, E]``.
    num_nodes : int
        Total original node count.
    node_sizes : torch.Tensor, optional
        Original node sizes with shape ``[N, 2]``.
    tree : ClusterTree
        Cluster hierarchy.
    cluster_name : str, optional
        Current cluster name, or ``None`` for root.
    steps : int
        FM^3 iteration budget for each derived component.
    seed : int
        Deterministic seed.
    ports : Sequence[_FdpRecursionPort], default=()
        Parent-generated boundary ports.

    Returns
    -------
    _FdpLevelLayout
        Recursive level layout with original-node positions.
    """
    derived = _fdp_recursion_derive_graph(edge_index, num_nodes, tree, cluster_name, ports)
    if not derived.nodes:
        return _FdpLevelLayout(positions={}, width=0.0, height=0.0, cluster_boxes={})

    components = _fdp_recursion_components(derived)
    child_layouts: Dict[str, _FdpLevelLayout] = {}
    component_positions: List[Dict[int, torch.Tensor]] = []
    component_boxes: List[Tuple[float, float, float, float]] = []
    component_node_geometries: List[List[Tuple[float, float, float, float]]] = []

    for component in components:
        local_tensor, xpms = _fdp_recursion_tlayout_component(
            derived=derived,
            component=component,
            seed=seed,
            max_iters=steps,
        )
        local_positions = {
            derived_index: local_tensor[local_index]
            for local_index, derived_index in enumerate(component)
        }
        for derived_index in component:
            node = derived.nodes[derived_index]
            if node.kind != "cluster":
                continue
            child_ports = _fdp_recursion_expand_cluster_ports(
                derived=derived,
                derived_positions=local_positions,
                cluster_index=derived_index,
                edge_index=edge_index,
            )
            child_layouts[str(node.key)] = _fdp_recursion_layout_level(
                edge_index=edge_index,
                num_nodes=num_nodes,
                node_sizes=node_sizes,
                tree=tree,
                cluster_name=str(node.key),
                steps=steps,
                seed=seed,
                ports=child_ports,
            )

        local_positions = _fdp_recursion_xlayout_component(
            derived=derived,
            component=component,
            local_positions=local_positions,
            node_sizes=node_sizes,
            child_layouts=child_layouts,
            xpms=xpms,
        )
        active_component = [
            int(index) for index in component if derived.nodes[int(index)].kind != "port"
        ]
        sizes = _fdp_recursion_component_sizes(
            derived,
            active_component,
            node_sizes,
            child_layouts,
        )
        if sizes.numel() == 0 or not active_component:
            component_boxes.append((0.0, 0.0, 0.0, 0.0))
            component_node_geometries.append([])
        else:
            half_sizes = sizes / 2.0
            active_tensor = torch.stack([local_positions[index] for index in active_component])
            lower = active_tensor - half_sizes
            upper = active_tensor + half_sizes
            component_boxes.append(
                (
                    float(lower[:, 0].min().item()),
                    float(lower[:, 1].min().item()),
                    float(upper[:, 0].max().item()),
                    float(upper[:, 1].max().item()),
                )
            )
            component_node_geometries.append(
                [
                    (
                        float(active_tensor[local_index, 0].item()),
                        float(active_tensor[local_index, 1].item()),
                        float(sizes[local_index, 0].item()),
                        float(sizes[local_index, 1].item()),
                    )
                    for local_index, _derived_index in enumerate(active_component)
                ]
            )
        component_positions.append(local_positions)

    offsets = _fdp_recursion_component_offsets(
        component_boxes,
        component_node_geometries=component_node_geometries,
    )
    final_positions: Dict[int, torch.Tensor] = {}
    cluster_boxes: Dict[str, Tuple[float, float, float, float]] = {}
    for component, local_positions, offset in zip(components, component_positions, offsets):
        for derived_index in component:
            node = derived.nodes[derived_index]
            if node.kind == "port":
                continue
            position = local_positions[derived_index] + offset
            if node.kind == "leaf":
                final_positions[int(node.key)] = position
                continue
            child = child_layouts[str(node.key)]
            child_offset = position - torch.tensor(
                [child.width / 2.0, child.height / 2.0],
                dtype=torch.float64,
            )
            x_shift = float(child_offset[0].item())
            y_shift = float(child_offset[1].item())
            cluster_boxes[str(node.key)] = (
                x_shift,
                y_shift,
                x_shift + child.width,
                y_shift + child.height,
            )
            for child_name, child_box in child.cluster_boxes.items():
                cluster_boxes[child_name] = (
                    child_box[0] + x_shift,
                    child_box[1] + y_shift,
                    child_box[2] + x_shift,
                    child_box[3] + y_shift,
                )
            for node_index, child_position in child.positions.items():
                final_positions[int(node_index)] = child_position + child_offset

    return _fdp_recursion_shift_to_origin(
        positions=final_positions,
        node_sizes=node_sizes,
        cluster_boxes=cluster_boxes,
        is_root=cluster_name is None,
    )


def graphviz_fdp_fidelity(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    steps: int = 200,
    seed: int = 42,
    clusters: Optional[Mapping[str, Sequence[int]]] = None,
    cluster_parents: Optional[Mapping[str, Optional[str]]] = None,
) -> torch.Tensor:
    """Run Graphviz fdp derived-graph recursion for clustered graphs.

    Parameters
    ----------
    edge_index : torch.Tensor
        Original edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of original nodes.
    node_sizes : torch.Tensor, optional
        Node sizes with shape ``[N, 2]``.
    steps : int, default=200
        FM^3 iteration budget for each derived component.
    seed : int, default=42
        Deterministic seed.
    clusters : Mapping[str, Sequence[int]], optional
        Flat descendant membership for each cluster.
    cluster_parents : Mapping[str, str | None], optional
        Parent mapping for clusters.

    Returns
    -------
    torch.Tensor
        Original node positions with shape ``[N, 2]``.

    Raises
    ------
    ValueError
        If inputs are invalid or cluster metadata is missing.

    Notes
    -----
    This ports Graphviz fdp's derived-graph recursion, boundary-port
    expansion, final root-bbox translation, and renderer-visible coordinate
    precision for the clustered fidelity path.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    if not clusters:
        raise ValueError("graphviz_fdp_fidelity requires cluster metadata.")

    tree = ClusterTree.from_flat_membership(clusters, cluster_parents or {})
    cpu_edge_index = edge_index.detach().to(device="cpu", dtype=torch.long)
    cpu_node_sizes = (
        node_sizes.detach().to(device="cpu", dtype=torch.float64)
        if node_sizes is not None
        else None
    )
    layout = _fdp_recursion_layout_level(
        edge_index=cpu_edge_index,
        num_nodes=num_nodes,
        node_sizes=cpu_node_sizes,
        tree=tree,
        cluster_name=None,
        steps=steps,
        seed=seed,
    )
    positions = torch.zeros((num_nodes, 2), dtype=torch.float64)
    for node_index, position in layout.positions.items():
        positions[int(node_index)] = position.to(dtype=torch.float64, device="cpu")
    positions[:, 1] *= -1.0
    positions = _graphviz_quantize_output_points(positions)
    return positions.to(device=_layout_device(edge_index=edge_index, node_sizes=node_sizes))


def _weak_components(edge_index: torch.Tensor, num_nodes: int) -> list[list[int]]:
    """Compute weak components in deterministic node order.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    list[list[int]]
        Weak components as sorted parent node indices.
    """
    neighbors: list[list[int]] = [[] for _ in range(num_nodes)]
    edge_index_cpu = edge_index.to(device="cpu", dtype=torch.long)
    for source, target in zip(edge_index_cpu[0].tolist(), edge_index_cpu[1].tolist()):
        source_index = int(source)
        target_index = int(target)
        if source_index == target_index:
            continue
        neighbors[source_index].append(target_index)
        neighbors[target_index].append(source_index)

    seen = [False] * num_nodes
    components: list[list[int]] = []
    for start in range(num_nodes):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        component: list[int] = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in neighbors[node]:
                if not seen[neighbor]:
                    seen[neighbor] = True
                    stack.append(neighbor)
        components.append(sorted(component))
    return components


def _slice_component_edges(
    edge_index: torch.Tensor,
    edge_weights: Optional[torch.Tensor],
    component: list[int],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Return component-local edges and optional weights.

    Parameters
    ----------
    edge_index : torch.Tensor
        Parent edge tensor with shape ``[2, E]``.
    edge_weights : torch.Tensor, optional
        Optional parent edge weights with shape ``[E]``.
    component : list[int]
        Parent node indices in one weak component.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor or None]
        Relabeled component edge tensor and aligned weights.
    """
    local_by_parent = {node: index for index, node in enumerate(component)}
    sources: list[int] = []
    targets: list[int] = []
    weights: list[float] = []
    edge_index_cpu = edge_index.to(device="cpu", dtype=torch.long)
    weights_cpu = None if edge_weights is None else edge_weights.detach().to(device="cpu")
    for edge_id, (source, target) in enumerate(
        zip(edge_index_cpu[0].tolist(), edge_index_cpu[1].tolist())
    ):
        source_index = int(source)
        target_index = int(target)
        if source_index not in local_by_parent or target_index not in local_by_parent:
            continue
        sources.append(local_by_parent[source_index])
        targets.append(local_by_parent[target_index])
        if weights_cpu is not None:
            weights.append(float(weights_cpu[edge_id].item()))

    local_edges = torch.tensor([sources, targets], dtype=torch.long, device=edge_index.device)
    if edge_weights is None:
        return local_edges, None
    local_weights = torch.tensor(weights, dtype=edge_weights.dtype, device=edge_weights.device)
    return local_edges, local_weights


def _graphviz_fdp_edge_lists(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weights: Optional[torch.Tensor],
) -> tuple[list[list[int]], list[tuple[int, int, float, float]]]:
    """Build Graphviz-style outgoing edge lists for FDP kernels.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes in the local graph.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``. Missing weights use
        Graphviz's default ``ED_factor(e) = 1``.

    Returns
    -------
    tuple[list[list[int]], list[tuple[int, int, float, float]]]
        Outgoing edge ids per source node and edge records as
        ``(source, target, factor, dist)``. The default edge distance is
        Graphviz fdp's ``K`` in inches.
    """
    outgoing: list[list[int]] = [[] for _ in range(num_nodes)]
    edges: list[tuple[int, int, float, float]] = []
    edge_index_cpu = edge_index.detach().to(device="cpu", dtype=torch.long)
    weights_cpu = None if edge_weights is None else edge_weights.detach().to(device="cpu")
    for edge_id, (source, target) in enumerate(
        zip(edge_index_cpu[0].tolist(), edge_index_cpu[1].tolist())
    ):
        source_index = int(source)
        target_index = int(target)
        if not (0 <= source_index < num_nodes and 0 <= target_index < num_nodes):
            continue
        factor = 1.0 if weights_cpu is None else float(weights_cpu[edge_id].item())
        edges.append((source_index, target_index, factor, _GRAPHVIZ_FDP_DEFAULT_K))
        outgoing[source_index].append(len(edges) - 1)
    return outgoing, edges


def _graphviz_fdp_collapse_parallel_edges(
    edge_index: torch.Tensor,
    edge_weights: Optional[torch.Tensor],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Collapse FDP parallel edges to Graphviz's single spring per node pair.

    Parameters
    ----------
    edge_index : torch.Tensor
        Local edge tensor with shape ``[2, E]``.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor or None]
        Edge tensor and optional weights with duplicate undirected node pairs
        removed, preserving the first edge orientation and first weight.

    Notes
    -----
    Graphviz degree logic skips repeated edges between the same pair
    (``lib/neatogen/stuff.c:126-132``). The FDP emulator mirrors that by using
    one spring per pair rather than summing duplicate weights.
    """
    if edge_index.numel() == 0:
        return edge_index, edge_weights
    edge_index_cpu = edge_index.detach().to(device="cpu", dtype=torch.long)
    weights_cpu = None if edge_weights is None else edge_weights.detach().to(device="cpu")
    seen: set[tuple[int, int]] = set()
    kept_sources: list[int] = []
    kept_targets: list[int] = []
    kept_weights: list[float] = []
    for edge_pos, (source, target) in enumerate(
        zip(edge_index_cpu[0].tolist(), edge_index_cpu[1].tolist())
    ):
        source_index = int(source)
        target_index = int(target)
        if source_index == target_index:
            continue
        key = (
            min(source_index, target_index),
            max(source_index, target_index),
        )
        if key in seen:
            continue
        seen.add(key)
        kept_sources.append(source_index)
        kept_targets.append(target_index)
        if weights_cpu is not None:
            kept_weights.append(float(weights_cpu[edge_pos].item()))
    if not kept_sources:
        collapsed_edges = torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
    else:
        collapsed_edges = torch.tensor(
            [kept_sources, kept_targets],
            dtype=torch.long,
            device=edge_index.device,
        )
    if edge_weights is None:
        return collapsed_edges, None
    collapsed_weights = torch.tensor(
        kept_weights,
        dtype=edge_weights.dtype,
        device=edge_weights.device,
    )
    return collapsed_edges, collapsed_weights


def _graphviz_fdp_initial_positions(num_nodes: int, seed: int) -> torch.Tensor:
    """Initialize positions as Graphviz ``fdp_tLayout`` does without ports.

    Parameters
    ----------
    num_nodes : int
        Number of local nodes.
    seed : int
        Graphviz ``seed`` attribute value.

    Returns
    -------
    torch.Tensor
        Initial positions in inches with shape ``[N, 2]``.
    """
    if num_nodes == 0:
        return torch.empty((0, 2), dtype=torch.float64)
    rng = _GraphvizDrand48(seed)
    size = _GRAPHVIZ_FDP_DEFAULT_K * (math.sqrt(num_nodes) + 1.0)
    half_extent = _GRAPHVIZ_FDP_EXPANSION_FACTOR * (size / 2.0)
    positions = torch.empty((num_nodes, 2), dtype=torch.float64)
    for node_index in range(num_nodes):
        positions[node_index, 0] = half_extent * (2.0 * rng.random() - 1.0)
        positions[node_index, 1] = half_extent * (2.0 * rng.random() - 1.0)
    return positions


def _graphviz_fdp_initial_position_lists(
    num_nodes: int,
    seed: int,
) -> Tuple[List[float], List[float]]:
    """Initialize flat ``tLayout`` positions as Python float lists.

    Parameters
    ----------
    num_nodes : int
        Number of local nodes.
    seed : int
        Graphviz ``seed`` attribute value.

    Returns
    -------
    tuple[list[float], list[float]]
        X and Y coordinates in Graphviz internal inches.
    """
    if num_nodes == 0:
        return [], []
    rng = _GraphvizDrand48(seed)
    size = _GRAPHVIZ_FDP_DEFAULT_K * (math.sqrt(num_nodes) + 1.0)
    half_extent = _GRAPHVIZ_FDP_EXPANSION_FACTOR * (size / 2.0)
    x_positions: List[float] = []
    y_positions: List[float] = []
    for _node_index in range(num_nodes):
        x_positions.append(half_extent * (2.0 * rng.random() - 1.0))
        y_positions.append(half_extent * (2.0 * rng.random() - 1.0))
    return x_positions, y_positions


def _graphviz_fdp_disperse_zero_delta(
    source: int,
    target: int,
    phase: int,
) -> tuple[float, float]:
    """Return a deterministic replacement for Graphviz's rare zero-distance jitter.

    Parameters
    ----------
    source : int
        First node index.
    target : int
        Second node index.
    phase : int
        Iteration or phase counter mixed into the deterministic fallback.

    Returns
    -------
    tuple[float, float]
        Non-zero displacement components.

    Notes
    -----
    Graphviz calls C ``rand()`` here without a local ``srand``. Exact libc
    state is not portable, and this branch only fires on exact coordinate
    equality, so the port uses stable non-zero jitter.
    """
    mixed = (source + 1) * 1103515245 + (target + 1) * 12345 + phase * 2654435761
    x_delta = float(5 - mixed % 10)
    y_delta = float(5 - (mixed // 10) % 10)
    if x_delta == 0.0 and y_delta == 0.0:
        x_delta = 1.0
    return x_delta, y_delta


def _graphviz_fdp_reset_displacements(
    x_displacements: List[float],
    y_displacements: List[float],
) -> None:
    """Reset mutable FDP displacement lists in Graphviz node order.

    Parameters
    ----------
    x_displacements : list[float]
        Mutable X displacement list.
    y_displacements : list[float]
        Mutable Y displacement list.

    Returns
    -------
    None
        Updates both lists in place.
    """
    for node_index in range(len(x_displacements)):
        x_displacements[node_index] = 0.0
        y_displacements[node_index] = 0.0


def _graphviz_fdp_apply_tlayout_repulsion_lists(
    x_positions: Sequence[float],
    y_positions: Sequence[float],
    x_displacements: List[float],
    y_displacements: List[float],
    source: int,
    target: int,
    phase: int,
    port_indices: Optional[frozenset[int]] = None,
) -> None:
    """Apply Graphviz ``doRep`` using Python double scalar arithmetic.

    Parameters
    ----------
    x_positions : Sequence[float]
        X coordinates in Graphviz internal inches.
    y_positions : Sequence[float]
        Y coordinates in Graphviz internal inches.
    x_displacements : list[float]
        Mutable X displacement list.
    y_displacements : list[float]
        Mutable Y displacement list.
    source : int
        Graphviz ``p`` node index.
    target : int
        Graphviz ``q`` node index.
    phase : int
        Iteration counter for deterministic zero-distance fallback.
    port_indices : frozenset[int], optional
        Local port node indices. Graphviz multiplies port-port repulsion by
        ten in recursive cluster layouts.

    Returns
    -------
    None
        Updates displacement lists in place.
    """
    x_delta = x_positions[target] - x_positions[source]
    y_delta = y_positions[target] - y_positions[source]
    dist2 = x_delta * x_delta + y_delta * y_delta
    if dist2 == 0.0:
        x_delta, y_delta = _graphviz_fdp_disperse_zero_delta(source, target, phase)
        dist2 = x_delta * x_delta + y_delta * y_delta
    dist = math.sqrt(dist2)
    force = _GRAPHVIZ_FDP_DEFAULT_K * _GRAPHVIZ_FDP_DEFAULT_K / (dist * dist2)
    if port_indices is not None and source in port_indices and target in port_indices:
        force *= 10.0
    x_displacements[target] += x_delta * force
    y_displacements[target] += y_delta * force
    x_displacements[source] -= x_delta * force
    y_displacements[source] -= y_delta * force


def _graphviz_fdp_apply_grid_repulsion_lists(
    x_positions: Sequence[float],
    y_positions: Sequence[float],
    x_displacements: List[float],
    y_displacements: List[float],
    grid: Mapping[tuple[int, int], list[int]],
    cell_size2: float,
    phase: int,
    port_indices: Optional[frozenset[int]] = None,
) -> None:
    """Apply Graphviz fdp grid repulsion with batched torch arithmetic.

    Parameters
    ----------
    x_positions : Sequence[float]
        X coordinates in Graphviz internal inches.
    y_positions : Sequence[float]
        Y coordinates in Graphviz internal inches.
    x_displacements : list[float]
        Mutable X displacement list.
    y_displacements : list[float]
        Mutable Y displacement list.
    grid : Mapping[tuple[int, int], list[int]]
        Graphviz-style spatial grid keyed by integer cell coordinates. Node
        lists use Graphviz's head-insertion order.
    cell_size2 : float
        Squared neighbor-cell cutoff, matching ``T_Cell * T_Cell``.
    phase : int
        Iteration counter for deterministic zero-distance fallback.
    port_indices : frozenset[int], optional
        Local port node indices. Graphviz multiplies port-port repulsion by
        ten in recursive cluster layouts.

    Returns
    -------
    None
        Updates displacement lists in place.

    Notes
    -----
    Small pair batches replay the scalar pair stream exactly. Large batches use
    one unordered representative per reciprocal pair and double its symmetric
    contribution, preserving Graphviz's same-cell and neighbor-cell force
    algebra while avoiding one Python call per directed pair.
    """
    source_indices: list[int] = []
    target_indices: list[int] = []
    same_cell_flags: list[bool] = []
    neighbor_offsets = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )
    for (cell_x, cell_y), nodes in sorted(grid.items()):
        if len(nodes) > 1:
            same_cell_sources = [source for source in nodes for target in nodes if source != target]
            same_cell_targets = [target for source in nodes for target in nodes if source != target]
            source_indices.extend(same_cell_sources)
            target_indices.extend(same_cell_targets)
            same_cell_flags.extend([True] * len(same_cell_sources))
        for delta_x, delta_y in neighbor_offsets:
            neighbor_nodes = grid.get((cell_x + delta_x, cell_y + delta_y), [])
            if neighbor_nodes:
                neighbor_sources = [source for source in nodes for _target in neighbor_nodes]
                neighbor_targets = [target for _source in nodes for target in neighbor_nodes]
                source_indices.extend(neighbor_sources)
                target_indices.extend(neighbor_targets)
                same_cell_flags.extend([False] * len(neighbor_sources))

    if not source_indices:
        return

    if len(source_indices) < _GRAPHVIZ_FDP_GRID_VECTORIZE_MIN_PAIRS:
        for source, target, same_cell in zip(source_indices, target_indices, same_cell_flags):
            if not same_cell:
                x_delta = x_positions[target] - x_positions[source]
                y_delta = y_positions[target] - y_positions[source]
                dist2 = x_delta * x_delta + y_delta * y_delta
                if dist2 >= cell_size2:
                    continue
            _graphviz_fdp_apply_tlayout_repulsion_lists(
                x_positions=x_positions,
                y_positions=y_positions,
                x_displacements=x_displacements,
                y_displacements=y_displacements,
                source=source,
                target=target,
                phase=phase,
                port_indices=port_indices,
            )
        return

    unordered_sources: list[int] = []
    unordered_targets: list[int] = []
    unordered_same_cell: list[bool] = []
    for (cell_x, cell_y), nodes in sorted(grid.items()):
        if len(nodes) > 1:
            unordered_sources.extend(
                source
                for source_index, source in enumerate(nodes)
                for _target in nodes[source_index + 1 :]
            )
            unordered_targets.extend(
                target
                for source_index, _source in enumerate(nodes)
                for target in nodes[source_index + 1 :]
            )
            unordered_same_cell.extend([True] * (len(nodes) * (len(nodes) - 1) // 2))
        for delta_x, delta_y in ((0, 1), (1, -1), (1, 0), (1, 1)):
            neighbor_nodes = grid.get((cell_x + delta_x, cell_y + delta_y), [])
            if neighbor_nodes:
                neighbor_sources = [source for source in nodes for _target in neighbor_nodes]
                neighbor_targets = [target for _source in nodes for target in neighbor_nodes]
                unordered_sources.extend(neighbor_sources)
                unordered_targets.extend(neighbor_targets)
                unordered_same_cell.extend([False] * len(neighbor_sources))

    if not unordered_sources:
        return

    previous_threads = torch.get_num_threads()
    if previous_threads != 1:
        torch.set_num_threads(1)
    try:
        device = torch.device("cpu")
        sources = torch.tensor(unordered_sources, dtype=torch.long, device=device)
        targets = torch.tensor(unordered_targets, dtype=torch.long, device=device)
        x_values = torch.tensor(list(x_positions), dtype=torch.float64, device=device)
        y_values = torch.tensor(list(y_positions), dtype=torch.float64, device=device)
        x_delta = x_values[targets] - x_values[sources]
        y_delta = y_values[targets] - y_values[sources]
        dist2 = x_delta.square() + y_delta.square()
        same_cell = torch.tensor(unordered_same_cell, dtype=torch.bool, device=device)
        active = same_cell | (dist2 < cell_size2)
        if not bool(active.any()):
            return

        zero_dist = active & (dist2 == 0.0)
        if bool(zero_dist.any()):
            directed_pairs = zip(source_indices, target_indices, same_cell_flags)
            for source, target, same_cell_flag in directed_pairs:
                if not same_cell_flag:
                    scalar_x_delta = x_positions[target] - x_positions[source]
                    scalar_y_delta = y_positions[target] - y_positions[source]
                    scalar_dist2 = scalar_x_delta * scalar_x_delta + scalar_y_delta * scalar_y_delta
                    if scalar_dist2 >= cell_size2:
                        continue
                _graphviz_fdp_apply_tlayout_repulsion_lists(
                    x_positions=x_positions,
                    y_positions=y_positions,
                    x_displacements=x_displacements,
                    y_displacements=y_displacements,
                    source=source,
                    target=target,
                    phase=phase,
                    port_indices=port_indices,
                )
            return

        sources = sources[active]
        targets = targets[active]
        x_delta = x_delta[active]
        y_delta = y_delta[active]
        dist2 = dist2[active]

        force = (
            2.0 * _GRAPHVIZ_FDP_DEFAULT_K * _GRAPHVIZ_FDP_DEFAULT_K / (torch.sqrt(dist2) * dist2)
        )
        if port_indices is not None:
            active_source_indices = sources.tolist()
            active_target_indices = targets.tolist()
            port_mask = torch.tensor(
                [
                    source in port_indices and target in port_indices
                    for source, target in zip(active_source_indices, active_target_indices)
                ],
                dtype=torch.bool,
                device=device,
            )
            force = torch.where(port_mask, force * 10.0, force)
        x_contrib = x_delta * force
        y_contrib = y_delta * force

        x_disp = torch.tensor(x_displacements, dtype=torch.float64, device=device)
        y_disp = torch.tensor(y_displacements, dtype=torch.float64, device=device)
        x_disp.index_add_(0, targets, x_contrib)
        y_disp.index_add_(0, targets, y_contrib)
        x_disp.index_add_(0, sources, -x_contrib)
        y_disp.index_add_(0, sources, -y_contrib)

        x_displacements[:] = [float(value) for value in x_disp.tolist()]
        y_displacements[:] = [float(value) for value in y_disp.tolist()]
    finally:
        if previous_threads != 1:
            torch.set_num_threads(previous_threads)


def _graphviz_fdp_apply_tlayout_attraction_lists(
    x_positions: Sequence[float],
    y_positions: Sequence[float],
    x_displacements: List[float],
    y_displacements: List[float],
    edge: Tuple[int, int, float, float],
    phase: int,
) -> None:
    """Apply Graphviz ``applyAttr`` using Python double scalar arithmetic.

    Parameters
    ----------
    x_positions : Sequence[float]
        X coordinates in Graphviz internal inches.
    y_positions : Sequence[float]
        Y coordinates in Graphviz internal inches.
    x_displacements : list[float]
        Mutable X displacement list.
    y_displacements : list[float]
        Mutable Y displacement list.
    edge : tuple[int, int, float, float]
        Edge record as ``(source, target, factor, dist)``.
    phase : int
        Iteration counter for deterministic zero-distance fallback.

    Returns
    -------
    None
        Updates displacement lists in place.
    """
    source, target, factor, edge_dist = edge
    if source == target:
        return
    x_delta = x_positions[target] - x_positions[source]
    y_delta = y_positions[target] - y_positions[source]
    dist2 = x_delta * x_delta + y_delta * y_delta
    if dist2 == 0.0:
        x_delta, y_delta = _graphviz_fdp_disperse_zero_delta(source, target, phase)
        dist2 = x_delta * x_delta + y_delta * y_delta
    dist = math.sqrt(dist2)
    force = factor * (dist - edge_dist) / dist
    x_displacements[target] -= x_delta * force
    y_displacements[target] -= y_delta * force
    x_displacements[source] += x_delta * force
    y_displacements[source] += y_delta * force


def _graphviz_fdp_update_position_lists(
    x_positions: List[float],
    y_positions: List[float],
    x_displacements: Sequence[float],
    y_displacements: Sequence[float],
    temperature: float,
) -> None:
    """Apply Graphviz ``updatePos`` to flat Python float coordinate lists.

    Parameters
    ----------
    x_positions : list[float]
        Mutable X coordinates in Graphviz internal inches.
    y_positions : list[float]
        Mutable Y coordinates in Graphviz internal inches.
    x_displacements : Sequence[float]
        X displacements in Graphviz internal inches.
    y_displacements : Sequence[float]
        Y displacements in Graphviz internal inches.
    temperature : float
        Current cooling temperature.

    Returns
    -------
    None
        Updates position lists in place.
    """
    temp2 = temperature * temperature
    for node_index in range(len(x_positions)):
        dx = x_displacements[node_index]
        dy = y_displacements[node_index]
        len2 = dx * dx + dy * dy
        if len2 < temp2:
            x_positions[node_index] += dx
            y_positions[node_index] += dy
        else:
            factor = temperature / math.sqrt(len2)
            x_positions[node_index] += dx * factor
            y_positions[node_index] += dy * factor


def _graphviz_fdp_update_position_lists_with_ports(
    x_positions: List[float],
    y_positions: List[float],
    x_displacements: Sequence[float],
    y_displacements: Sequence[float],
    temperature: float,
    port_indices: frozenset[int],
    half_width: float,
    half_height: float,
) -> None:
    """Apply Graphviz ``updatePos`` with recursive port boundary clamping.

    Parameters
    ----------
    x_positions : list[float]
        Mutable X coordinates in Graphviz internal inches.
    y_positions : list[float]
        Mutable Y coordinates in Graphviz internal inches.
    x_displacements : Sequence[float]
        X displacements in Graphviz internal inches.
    y_displacements : Sequence[float]
        Y displacements in Graphviz internal inches.
    temperature : float
        Current cooling temperature.
    port_indices : frozenset[int]
        Local node indices that are boundary ports.
    half_width : float
        Boundary ellipse half-width.
    half_height : float
        Boundary ellipse half-height.

    Returns
    -------
    None
        Updates position lists in place.
    """
    temp2 = temperature * temperature
    half_width2 = half_width * half_width
    half_height2 = half_height * half_height
    for node_index in range(len(x_positions)):
        dx = x_displacements[node_index]
        dy = y_displacements[node_index]
        len2 = dx * dx + dy * dy
        if len2 < temp2:
            x_value = x_positions[node_index] + dx
            y_value = y_positions[node_index] + dy
        else:
            factor = temperature / math.sqrt(len2)
            x_value = x_positions[node_index] + dx * factor
            y_value = y_positions[node_index] + dy * factor

        distance = math.sqrt(x_value * x_value / half_width2 + y_value * y_value / half_height2)
        if node_index in port_indices and distance > 0.0:
            x_positions[node_index] = x_value / distance
            y_positions[node_index] = y_value / distance
        elif distance >= 1.0:
            x_positions[node_index] = 0.95 * x_value / distance
            y_positions[node_index] = 0.95 * y_value / distance
        else:
            x_positions[node_index] = x_value
            y_positions[node_index] = y_value


def _graphviz_fdp_apply_tlayout_repulsion(
    positions: torch.Tensor,
    displacement: torch.Tensor,
    source: int,
    target: int,
    phase: int,
    port_indices: Optional[frozenset[int]] = None,
) -> None:
    """Apply Graphviz ``tLayout`` pair repulsion.

    Parameters
    ----------
    positions : torch.Tensor
        Current positions in inches with shape ``[N, 2]``.
    displacement : torch.Tensor
        Mutable displacement tensor with shape ``[N, 2]``.
    source : int
        First node index.
    target : int
        Second node index.
    phase : int
        Iteration counter for deterministic zero-distance fallback.
    port_indices : frozenset[int], optional
        Local port node indices. Graphviz multiplies port-port repulsion by
        ten in recursive cluster layouts.

    Returns
    -------
    None
        Updates ``displacement`` in place.
    """
    x_delta = float(positions[target, 0] - positions[source, 0])
    y_delta = float(positions[target, 1] - positions[source, 1])
    dist2 = x_delta * x_delta + y_delta * y_delta
    if dist2 == 0.0:
        x_delta, y_delta = _graphviz_fdp_disperse_zero_delta(source, target, phase)
        dist2 = x_delta * x_delta + y_delta * y_delta
    dist = math.sqrt(dist2)
    force = _GRAPHVIZ_FDP_DEFAULT_K * _GRAPHVIZ_FDP_DEFAULT_K / (dist * dist2)
    if port_indices is not None and source in port_indices and target in port_indices:
        force *= 10.0
    displacement[target, 0] += x_delta * force
    displacement[target, 1] += y_delta * force
    displacement[source, 0] -= x_delta * force
    displacement[source, 1] -= y_delta * force


def _graphviz_fdp_apply_tlayout_attraction(
    positions: torch.Tensor,
    displacement: torch.Tensor,
    edge: tuple[int, int, float, float],
    phase: int,
) -> None:
    """Apply Graphviz ``tLayout`` edge attraction.

    Parameters
    ----------
    positions : torch.Tensor
        Current positions in inches with shape ``[N, 2]``.
    displacement : torch.Tensor
        Mutable displacement tensor with shape ``[N, 2]``.
    edge : tuple[int, int, float, float]
        Edge record as ``(source, target, factor, dist)``.
    phase : int
        Iteration counter for deterministic zero-distance fallback.

    Returns
    -------
    None
        Updates ``displacement`` in place.
    """
    source, target, factor, edge_dist = edge
    if source == target:
        return
    x_delta = float(positions[target, 0] - positions[source, 0])
    y_delta = float(positions[target, 1] - positions[source, 1])
    dist2 = x_delta * x_delta + y_delta * y_delta
    if dist2 == 0.0:
        x_delta, y_delta = _graphviz_fdp_disperse_zero_delta(source, target, phase)
        dist2 = x_delta * x_delta + y_delta * y_delta
    dist = math.sqrt(dist2)
    force = factor * (dist - edge_dist) / dist
    displacement[target, 0] -= x_delta * force
    displacement[target, 1] -= y_delta * force
    displacement[source, 0] += x_delta * force
    displacement[source, 1] += y_delta * force


def _graphviz_fdp_update_positions(
    positions: torch.Tensor,
    displacement: torch.Tensor,
    temperature: float,
) -> None:
    """Apply Graphviz ``xLayout``'s temperature-limited position update.

    Parameters
    ----------
    positions : torch.Tensor
        Mutable positions in inches with shape ``[N, 2]``.
    displacement : torch.Tensor
        Displacement tensor with shape ``[N, 2]``.
    temperature : float
        Current cooling temperature.

    Returns
    -------
    None
        Updates ``positions`` in place.
    """
    temp2 = temperature * temperature
    for node_index in range(positions.shape[0]):
        dx = float(displacement[node_index, 0])
        dy = float(displacement[node_index, 1])
        len2 = dx * dx + dy * dy
        if len2 < temp2:
            positions[node_index, 0] += dx
            positions[node_index, 1] += dy
        else:
            length = math.sqrt(len2)
            positions[node_index, 0] += dx * temperature / length
            positions[node_index, 1] += dy * temperature / length


def _graphviz_fdp_tlayout(
    edge_index: torch.Tensor,
    num_nodes: int,
    seed: int,
    edge_weights: Optional[torch.Tensor],
    max_iters: int = _GRAPHVIZ_FDP_DEFAULT_MAX_ITERS,
    node_ids: Optional[Sequence[str]] = None,
) -> tuple[torch.Tensor, tuple[float, float, float, int, int]]:
    """Run Graphviz ``fdp_tLayout`` for one connected component.

    Parameters
    ----------
    edge_index : torch.Tensor
        Local edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of local nodes.
    seed : int
        Graphviz ``seed`` attribute value.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``.
    max_iters : int, default=_GRAPHVIZ_FDP_DEFAULT_MAX_ITERS
        Graphviz ``maxiter`` budget for the temperature schedule.
    node_ids : Sequence[str], optional
        Trace node identifiers. When provided, per-iteration positions are
        appended in Graphviz trace format.

    Returns
    -------
    tuple[torch.Tensor, tuple[float, float, float, int, int]]
        Positions in inches and xLayout parameters
        ``(T0, K, C, numIters, loopcnt)``.
    """
    x_positions, y_positions = _graphviz_fdp_initial_position_lists(
        num_nodes=num_nodes,
        seed=seed,
    )
    x_displacements = [0.0] * num_nodes
    y_displacements = [0.0] * num_nodes
    outgoing, edges = _graphviz_fdp_edge_lists(edge_index, num_nodes, edge_weights)
    pass1 = _GRAPHVIZ_FDP_DEFAULT_UNSCALED * max_iters // 100
    t0 = _GRAPHVIZ_FDP_DEFAULT_TFACT * _GRAPHVIZ_FDP_DEFAULT_K * math.sqrt(num_nodes) / 5.0
    loop_count = pass1
    cell_size = 3.0 * _GRAPHVIZ_FDP_DEFAULT_K
    cell_size2 = cell_size * cell_size

    for iteration in range(loop_count):
        temperature = t0 * (max_iters - iteration) / max_iters
        if temperature <= 0.0:
            continue
        _graphviz_fdp_reset_displacements(x_displacements, y_displacements)
        grid: dict[tuple[int, int], list[int]] = {}
        for node_index in range(num_nodes):
            cell = (
                math.floor(x_positions[node_index] / cell_size),
                math.floor(y_positions[node_index] / cell_size),
            )
            grid.setdefault(cell, []).insert(0, node_index)
        for source in range(num_nodes):
            for edge_id in outgoing[source]:
                _graphviz_fdp_apply_tlayout_attraction_lists(
                    x_positions=x_positions,
                    y_positions=y_positions,
                    x_displacements=x_displacements,
                    y_displacements=y_displacements,
                    edge=edges[edge_id],
                    phase=iteration,
                )
        _graphviz_fdp_apply_grid_repulsion_lists(
            x_positions=x_positions,
            y_positions=y_positions,
            x_displacements=x_displacements,
            y_displacements=y_displacements,
            grid=grid,
            cell_size2=cell_size2,
            phase=iteration,
        )
        _graphviz_fdp_update_position_lists(
            x_positions=x_positions,
            y_positions=y_positions,
            x_displacements=x_displacements,
            y_displacements=y_displacements,
            temperature=temperature,
        )
        if node_ids is not None:
            _fdp_trace_positions(
                "tlayout_gAdjust",
                iteration,
                node_ids,
                _graphviz_fdp_positions_from_lists(x_positions, y_positions),
            )

    x_t0 = t0 * (max_iters - pass1) / max_iters
    return _graphviz_fdp_positions_from_lists(x_positions, y_positions), (
        x_t0,
        _GRAPHVIZ_FDP_DEFAULT_K,
        _GRAPHVIZ_FDP_DEFAULT_C,
        max_iters - pass1,
        max_iters - pass1,
    )


def _graphviz_fdp_node_sizes_in_inches(
    node_sizes: Optional[torch.Tensor],
    num_nodes: int,
) -> torch.Tensor:
    """Return node sizes in Graphviz fdp's internal inch units.

    Parameters
    ----------
    node_sizes : torch.Tensor, optional
        Node sizes in points with shape ``[N, 2]``.
    num_nodes : int
        Number of local nodes.

    Returns
    -------
    torch.Tensor
        Node sizes plus Graphviz fdp's default additive ``xLayout``
        separation in inches with shape ``[N, 2]``.
    """
    if node_sizes is None:
        sizes = torch.zeros((num_nodes, 2), dtype=torch.float64)
    else:
        sizes = node_sizes.detach().to(device="cpu", dtype=torch.float64) / (
            _GRAPHVIZ_FDP_POINTS_PER_INCH
        )
    floors = torch.tensor(
        [_GRAPHVIZ_DEFAULT_NODE_WIDTH_INCHES, _GRAPHVIZ_DEFAULT_NODE_HEIGHT_INCHES],
        dtype=torch.float64,
    )
    sep = 2.0 * _GRAPHVIZ_FDP_DEFAULT_XLAYOUT_SEP_POINTS / _GRAPHVIZ_FDP_POINTS_PER_INCH
    return torch.maximum(sizes, floors) + sep


def _graphviz_fdp_node_size_lists_in_inches(
    node_sizes: Optional[torch.Tensor],
    num_nodes: int,
) -> Tuple[List[float], List[float]]:
    """Return Graphviz ``xLayout`` node sizes as Python float lists.

    Parameters
    ----------
    node_sizes : torch.Tensor, optional
        Node sizes in points with shape ``[N, 2]``.
    num_nodes : int
        Number of local nodes.

    Returns
    -------
    tuple[list[float], list[float]]
        Widths and heights including Graphviz's default additive separation,
        in internal inches.
    """
    sep = 2.0 * _GRAPHVIZ_FDP_DEFAULT_XLAYOUT_SEP_POINTS / _GRAPHVIZ_FDP_POINTS_PER_INCH
    widths: List[float] = []
    heights: List[float] = []
    if node_sizes is None:
        for _node_index in range(num_nodes):
            widths.append(_GRAPHVIZ_DEFAULT_NODE_WIDTH_INCHES + sep)
            heights.append(_GRAPHVIZ_DEFAULT_NODE_HEIGHT_INCHES + sep)
        return widths, heights

    sizes_cpu = node_sizes.detach().to(device="cpu", dtype=torch.float64)
    for node_index in range(num_nodes):
        width = float(sizes_cpu[node_index, 0].item()) / _GRAPHVIZ_FDP_POINTS_PER_INCH
        height = float(sizes_cpu[node_index, 1].item()) / _GRAPHVIZ_FDP_POINTS_PER_INCH
        widths.append(max(width, _GRAPHVIZ_DEFAULT_NODE_WIDTH_INCHES) + sep)
        heights.append(max(height, _GRAPHVIZ_DEFAULT_NODE_HEIGHT_INCHES) + sep)
    return widths, heights


def _graphviz_fdp_prism_half_size_lists_in_inches(
    node_sizes: Optional[torch.Tensor],
    num_nodes: int,
) -> Tuple[List[float], List[float]]:
    """Return Graphviz PRISM half-sizes in internal inch units.

    Parameters
    ----------
    node_sizes : torch.Tensor, optional
        Node sizes in points with shape ``[N, 2]``.
    num_nodes : int
        Number of local nodes.

    Returns
    -------
    tuple[list[float], list[float]]
        Half-widths and half-heights including Graphviz's default FDP
        additive separation, in internal inches.
    """
    widths, heights = _graphviz_fdp_node_size_lists_in_inches(node_sizes, num_nodes)
    return [width / 2.0 for width in widths], [height / 2.0 for height in heights]


def _graphviz_fdp_prism_overlap_edges(
    x_positions: Sequence[float],
    y_positions: Sequence[float],
    half_widths: Sequence[float],
    half_heights: Sequence[float],
    check_overlap_only: bool = False,
) -> set[tuple[int, int]]:
    """Return overlapping rectangle pairs using Graphviz's strict interval test.

    Parameters
    ----------
    x_positions : Sequence[float]
        X coordinates in Graphviz internal inches.
    y_positions : Sequence[float]
        Y coordinates in Graphviz internal inches.
    half_widths : Sequence[float]
        Node half-widths in Graphviz internal inches.
    half_heights : Sequence[float]
        Node half-heights in Graphviz internal inches.
    check_overlap_only : bool, default=False
        Return after the first pair when only the existence of an overlap is
        needed.

    Returns
    -------
    set[tuple[int, int]]
        Undirected overlapping pairs with ``left < right``.
    """
    edges: set[tuple[int, int]] = set()
    num_nodes = len(x_positions)
    for source in range(num_nodes):
        for target in range(source + 1, num_nodes):
            if (
                abs(x_positions[source] - x_positions[target])
                < half_widths[source] + half_widths[target]
                and abs(y_positions[source] - y_positions[target])
                < half_heights[source] + half_heights[target]
            ):
                edges.add((source, target))
                if check_overlap_only:
                    return edges
    return edges


def _graphviz_fdp_prism_has_overlap(
    x_positions: Sequence[float],
    y_positions: Sequence[float],
    half_widths: Sequence[float],
    half_heights: Sequence[float],
) -> bool:
    """Return whether any PRISM-expanded node rectangles overlap.

    Parameters
    ----------
    x_positions : Sequence[float]
        X coordinates in Graphviz internal inches.
    y_positions : Sequence[float]
        Y coordinates in Graphviz internal inches.
    half_widths : Sequence[float]
        Node half-widths in Graphviz internal inches.
    half_heights : Sequence[float]
        Node half-heights in Graphviz internal inches.

    Returns
    -------
    bool
        ``True`` if at least one strict rectangle overlap exists.
    """
    return bool(
        _graphviz_fdp_prism_overlap_edges(
            x_positions=x_positions,
            y_positions=y_positions,
            half_widths=half_widths,
            half_heights=half_heights,
            check_overlap_only=True,
        )
    )


def _graphviz_fdp_prism_scale_lists(
    x_positions: List[float],
    y_positions: List[float],
    scale: float,
) -> None:
    """Scale PRISM coordinate lists around Graphviz's origin.

    Parameters
    ----------
    x_positions : list[float]
        Mutable X coordinates in Graphviz internal inches.
    y_positions : list[float]
        Mutable Y coordinates in Graphviz internal inches.
    scale : float
        Multiplicative scale factor.

    Returns
    -------
    None
        Updates coordinate lists in place.
    """
    for node_index in range(len(x_positions)):
        x_positions[node_index] *= scale
        y_positions[node_index] *= scale


def _graphviz_fdp_prism_delaunay_edges(
    x_positions: Sequence[float],
    y_positions: Sequence[float],
) -> set[tuple[int, int]]:
    """Build the PRISM proximity graph using SciPy Delaunay triangulation.

    Parameters
    ----------
    x_positions : Sequence[float]
        X coordinates in Graphviz internal inches.
    y_positions : Sequence[float]
        Y coordinates in Graphviz internal inches.

    Returns
    -------
    set[tuple[int, int]]
        Undirected Delaunay-neighbor pairs with ``left < right``.
    """
    num_nodes = len(x_positions)
    if num_nodes < 2:
        return set()
    if num_nodes < 4:
        return {
            (source, target)
            for source in range(num_nodes)
            for target in range(source + 1, num_nodes)
        }

    import numpy as np
    from scipy.spatial import Delaunay, QhullError

    points = np.column_stack(
        [
            np.asarray(x_positions, dtype=float),
            np.asarray(y_positions, dtype=float),
        ]
    )
    try:
        triangulation = Delaunay(points)
    except QhullError:
        try:
            triangulation = Delaunay(points, qhull_options="QJ")
        except QhullError:
            return {
                (source, target)
                for source in range(num_nodes)
                for target in range(source + 1, num_nodes)
            }

    edges: set[tuple[int, int]] = set()
    for simplex in triangulation.simplices:
        vertices = [int(vertex) for vertex in simplex]
        for first, second in ((0, 1), (1, 2), (2, 0)):
            source = vertices[first]
            target = vertices[second]
            if source == target:
                continue
            if source > target:
                source, target = target, source
            edges.add((source, target))
    return edges


def _graphviz_fdp_prism_graph_edges(
    x_positions: Sequence[float],
    y_positions: Sequence[float],
    half_widths: Sequence[float],
    half_heights: Sequence[float],
    neighborhood_only: bool,
) -> set[tuple[int, int]]:
    """Return Graphviz PRISM's proximity graph for one smoother pass.

    Parameters
    ----------
    x_positions : Sequence[float]
        X coordinates in Graphviz internal inches.
    y_positions : Sequence[float]
        Y coordinates in Graphviz internal inches.
    half_widths : Sequence[float]
        Node half-widths in Graphviz internal inches.
    half_heights : Sequence[float]
        Node half-heights in Graphviz internal inches.
    neighborhood_only : bool
        When ``True``, use only Delaunay neighbors. When ``False``, add exact
        current-overlap edges, matching ``OverlapSmoother_new``.

    Returns
    -------
    set[tuple[int, int]]
        Undirected proximity pairs with ``left < right``.
    """
    edges = _graphviz_fdp_prism_delaunay_edges(x_positions, y_positions)
    if not neighborhood_only:
        edges.update(
            _graphviz_fdp_prism_overlap_edges(
                x_positions=x_positions,
                y_positions=y_positions,
                half_widths=half_widths,
                half_heights=half_heights,
            )
        )
    return edges


def _graphviz_fdp_prism_average_edge_length(
    x_positions: Sequence[float],
    y_positions: Sequence[float],
    edge_index: torch.Tensor,
) -> float:
    """Return the average source-target length used by PRISM initial scaling.

    Parameters
    ----------
    x_positions : Sequence[float]
        X coordinates in Graphviz internal inches.
    y_positions : Sequence[float]
        Y coordinates in Graphviz internal inches.
    edge_index : torch.Tensor
        Local edge tensor with shape ``[2, E]``.

    Returns
    -------
    float
        Mean Euclidean edge length, or ``0.0`` when no non-loop edge exists.
    """
    if edge_index.numel() == 0:
        return 0.0
    edges: set[tuple[int, int]] = set()
    for edge_id in range(int(edge_index.shape[1])):
        source = int(edge_index[0, edge_id].item())
        target = int(edge_index[1, edge_id].item())
        if source == target:
            continue
        if source > target:
            source, target = target, source
        edges.add((source, target))
    if not edges:
        return 0.0
    total = 0.0
    for source, target in edges:
        total += math.hypot(
            x_positions[source] - x_positions[target],
            y_positions[source] - y_positions[target],
        )
    return total / len(edges)


def _graphviz_fdp_prism_apply_initial_scaling(
    x_positions: List[float],
    y_positions: List[float],
    half_widths: Sequence[float],
    half_heights: Sequence[float],
    edge_index: torch.Tensor,
    initial_scaling: float,
) -> None:
    """Apply Graphviz ``remove_overlap`` initial edge-length scaling.

    Parameters
    ----------
    x_positions : list[float]
        Mutable X coordinates in Graphviz internal inches.
    y_positions : list[float]
        Mutable Y coordinates in Graphviz internal inches.
    half_widths : Sequence[float]
        Node half-widths in Graphviz internal inches.
    half_heights : Sequence[float]
        Node half-heights in Graphviz internal inches.
    edge_index : torch.Tensor
        Local edge tensor with shape ``[2, E]``.
    initial_scaling : float
        Graphviz PRISM ``overlap_scaling`` value. Negative values scale the
        current average edge length to ``abs(value) * avg_label_size``.

    Returns
    -------
    None
        Updates coordinate lists in place.
    """
    if initial_scaling == 0.0:
        return
    average_length = _graphviz_fdp_prism_average_edge_length(x_positions, y_positions, edge_index)
    if average_length <= _FDP_EPSILON:
        return
    if initial_scaling < 0.0:
        average_label_size = sum(
            half_widths[node_index] + half_heights[node_index]
            for node_index in range(len(x_positions))
        ) / max(len(x_positions), 1)
        target_length = -initial_scaling * average_label_size
    else:
        target_length = initial_scaling
    _graphviz_fdp_prism_scale_lists(x_positions, y_positions, target_length / average_length)


def _graphviz_fdp_prism_overlap_scaling(
    x_positions: List[float],
    y_positions: List[float],
    half_widths: Sequence[float],
    half_heights: Sequence[float],
    scale_start: float,
    scale_stop: float,
    epsilon: float,
    max_iterations: int,
) -> float:
    """Run Graphviz PRISM bisection scaling until boxes are disjoint.

    Parameters
    ----------
    x_positions : list[float]
        Mutable X coordinates in Graphviz internal inches.
    y_positions : list[float]
        Mutable Y coordinates in Graphviz internal inches.
    half_widths : Sequence[float]
        Node half-widths in Graphviz internal inches.
    half_heights : Sequence[float]
        Node half-heights in Graphviz internal inches.
    scale_start : float
        Lower scaling bracket.
    scale_stop : float
        Upper scaling bracket, or a negative value to auto-discover it.
    epsilon : float
        Termination bracket width.
    max_iterations : int
        Maximum bisection iterations.

    Returns
    -------
    float
        Final scale applied to the coordinate lists.
    """
    if scale_start <= 0.0:
        scale_start = 0.0
    else:
        _graphviz_fdp_prism_scale_lists(x_positions, y_positions, scale_start)
        if not _graphviz_fdp_prism_has_overlap(
            x_positions,
            y_positions,
            half_widths,
            half_heights,
        ):
            return scale_start
        _graphviz_fdp_prism_scale_lists(x_positions, y_positions, 1.0 / scale_start)

    if scale_stop < 0.0:
        scale_stop = epsilon if scale_start == 0.0 else scale_start
        _graphviz_fdp_prism_scale_lists(x_positions, y_positions, scale_stop)
        while True:
            scale_stop *= 2.0
            _graphviz_fdp_prism_scale_lists(x_positions, y_positions, 2.0)
            if not _graphviz_fdp_prism_has_overlap(
                x_positions,
                y_positions,
                half_widths,
                half_heights,
            ):
                break
        _graphviz_fdp_prism_scale_lists(x_positions, y_positions, 1.0 / scale_stop)

    scale_best = scale_stop
    iteration = 0
    while iteration < max_iterations and scale_stop - scale_start > epsilon:
        iteration += 1
        scale = 0.5 * (scale_start + scale_stop)
        _graphviz_fdp_prism_scale_lists(x_positions, y_positions, scale)
        overlap = _graphviz_fdp_prism_has_overlap(
            x_positions,
            y_positions,
            half_widths,
            half_heights,
        )
        _graphviz_fdp_prism_scale_lists(x_positions, y_positions, 1.0 / scale)
        if overlap:
            scale_start = scale
        else:
            scale_best = scale
            scale_stop = scale
    _graphviz_fdp_prism_scale_lists(x_positions, y_positions, scale_best)
    return scale_best


def _graphviz_fdp_prism_ideal_distances(
    x_positions: Sequence[float],
    y_positions: Sequence[float],
    half_widths: Sequence[float],
    half_heights: Sequence[float],
    edges: set[tuple[int, int]],
) -> tuple[list[tuple[int, int, float, bool]], float, float]:
    """Compute Graphviz PRISM ideal distances for proximity edges.

    Parameters
    ----------
    x_positions : Sequence[float]
        X coordinates in Graphviz internal inches.
    y_positions : Sequence[float]
        Y coordinates in Graphviz internal inches.
    half_widths : Sequence[float]
        Node half-widths in Graphviz internal inches.
    half_heights : Sequence[float]
        Node half-heights in Graphviz internal inches.
    edges : set[tuple[int, int]]
        Undirected proximity graph edges.

    Returns
    -------
    tuple[list[tuple[int, int, float, bool]], float, float]
        Edge records ``(source, target, ideal_distance, expands)``, maximum
        overlap factor, and minimum overlap factor.
    """
    records: list[tuple[int, int, float, bool]] = []
    max_overlap = 0.0
    min_overlap = 1.0e10
    for source, target in sorted(edges):
        x_delta = abs(x_positions[source] - x_positions[target])
        y_delta = abs(y_positions[source] - y_positions[target])
        width = half_widths[source] + half_widths[target]
        height = half_heights[source] + half_heights[target]
        distance = math.hypot(
            x_positions[source] - x_positions[target],
            y_positions[source] - y_positions[target],
        )
        if (
            x_delta < _GRAPHVIZ_FDP_PRISM_MACHINE_ACC * width
            and y_delta < _GRAPHVIZ_FDP_PRISM_MACHINE_ACC * height
        ):
            records.append((source, target, math.hypot(width, height), True))
            max_overlap = 2.0
            min_overlap = min(min_overlap, 2.0)
            continue
        if x_delta < _GRAPHVIZ_FDP_PRISM_MACHINE_ACC * width:
            factor = height / y_delta
        elif y_delta < _GRAPHVIZ_FDP_PRISM_MACHINE_ACC * height:
            factor = width / x_delta
        else:
            factor = min(width / x_delta, height / y_delta)
        if factor > 1.0:
            factor = max(factor, 1.001)
        max_overlap = max(max_overlap, factor)
        min_overlap = min(min_overlap, factor)
        bounded = min(_GRAPHVIZ_FDP_PRISM_EXPAND_MAX, factor)
        bounded = max(_GRAPHVIZ_FDP_PRISM_EXPAND_MIN, bounded)
        records.append((source, target, bounded * distance, factor > 1.0))
    return records, max_overlap, min_overlap


def _graphviz_fdp_prism_stress_step(
    x_positions: List[float],
    y_positions: List[float],
    records: Sequence[tuple[int, int, float, bool]],
) -> float:
    """Run one Graphviz-style PRISM stress-majorization update.

    Parameters
    ----------
    x_positions : list[float]
        Mutable X coordinates in Graphviz internal inches.
    y_positions : list[float]
        Mutable Y coordinates in Graphviz internal inches.
    records : Sequence[tuple[int, int, float, bool]]
        PRISM ideal-distance records as returned by
        :func:`_graphviz_fdp_prism_ideal_distances`.

    Returns
    -------
    float
        Root mean square coordinate displacement from the previous positions.
    """
    num_nodes = len(x_positions)
    if num_nodes <= 1 or not records:
        return 0.0

    import numpy as np
    from scipy import sparse
    from scipy.sparse import linalg as sparse_linalg

    old = np.column_stack(
        [
            np.asarray(x_positions, dtype=float),
            np.asarray(y_positions, dtype=float),
        ]
    )
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    rhs = np.zeros((num_nodes, 2), dtype=float)
    diagonal = np.zeros(num_nodes, dtype=float)
    for source, target, ideal_distance, expands in records:
        current_delta = old[source] - old[target]
        current_distance = float(np.hypot(current_delta[0], current_delta[1]))
        if current_distance <= _FDP_EPSILON or ideal_distance <= _FDP_EPSILON:
            continue
        weight_scale = 100.0 if expands else 1.0
        weight = weight_scale / (ideal_distance * ideal_distance)
        diagonal[source] += weight
        diagonal[target] += weight
        rows.extend([source, target])
        cols.extend([target, source])
        data.extend([-weight, -weight])
        rhs_delta = weight * ideal_distance * current_delta / current_distance
        rhs[source] += rhs_delta
        rhs[target] -= rhs_delta

    rows.extend(range(num_nodes))
    cols.extend(range(num_nodes))
    data.extend(float(value) for value in diagonal)
    laplacian = sparse.csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
    if num_nodes == 2:
        reduced = laplacian[1:, 1:].tocsc()
    else:
        reduced = laplacian[1:, 1:].tocsr()
    new_positions = np.zeros_like(old)
    for axis in range(2):
        try:
            solved = sparse_linalg.spsolve(reduced, rhs[1:, axis])
        except Exception:
            solved = np.linalg.lstsq(reduced.toarray(), rhs[1:, axis], rcond=None)[0]
        new_positions[1:, axis] = np.asarray(solved, dtype=float)
    new_positions += old.mean(axis=0, keepdims=True) - new_positions.mean(axis=0, keepdims=True)
    rms = float(np.sqrt(np.mean((new_positions - old) ** 2)))
    for node_index in range(num_nodes):
        x_positions[node_index] = float(new_positions[node_index, 0])
        y_positions[node_index] = float(new_positions[node_index, 1])
    return rms


def _graphviz_fdp_prism_remove_overlap_lists(
    x_positions: List[float],
    y_positions: List[float],
    half_widths: Sequence[float],
    half_heights: Sequence[float],
    edge_index: torch.Tensor,
    ntry: int = _GRAPHVIZ_FDP_DEFAULT_PRISM_TRIES,
    initial_scaling: float = _GRAPHVIZ_FDP_DEFAULT_PRISM_SCALING,
    do_shrinking: bool = True,
) -> None:
    """Remove FDP overlaps with Graphviz PRISM's proximity-stress loop.

    Parameters
    ----------
    x_positions : list[float]
        Mutable X coordinates in Graphviz internal inches.
    y_positions : list[float]
        Mutable Y coordinates in Graphviz internal inches.
    half_widths : Sequence[float]
        Node half-widths in Graphviz internal inches.
    half_heights : Sequence[float]
        Node half-heights in Graphviz internal inches.
    edge_index : torch.Tensor
        Local edge tensor with shape ``[2, E]``.
    ntry : int, default=_GRAPHVIZ_FDP_DEFAULT_PRISM_TRIES
        Maximum PRISM smoother passes. The FDP default ``overlap="9:prism"``
        bounds this fidelity port to the same named-stage budget.
    initial_scaling : float, default=_GRAPHVIZ_FDP_DEFAULT_PRISM_SCALING
        Graphviz ``overlap_scaling`` default.
    do_shrinking : bool, default=True
        Whether to allow the final full-overlap pass to shrink whitespace when
        no overlaps remain.

    Returns
    -------
    None
        Updates coordinate lists in place.
    """
    if len(x_positions) <= 1 or ntry <= 0:
        return
    _graphviz_fdp_prism_apply_initial_scaling(
        x_positions=x_positions,
        y_positions=y_positions,
        half_widths=half_widths,
        half_heights=half_heights,
        edge_index=edge_index,
        initial_scaling=initial_scaling,
    )

    residual = 100000.0
    neighborhood_only = True
    shrink = False
    for _iteration in range(ntry):
        edges = _graphviz_fdp_prism_graph_edges(
            x_positions=x_positions,
            y_positions=y_positions,
            half_widths=half_widths,
            half_heights=half_heights,
            neighborhood_only=neighborhood_only,
        )
        records, max_overlap, _min_overlap = _graphviz_fdp_prism_ideal_distances(
            x_positions=x_positions,
            y_positions=y_positions,
            half_widths=half_widths,
            half_heights=half_heights,
            edges=edges,
        )
        if max_overlap < 1.0 and shrink:
            scale_start = min(1.0, max_overlap * 1.0001)
            _graphviz_fdp_prism_overlap_scaling(
                x_positions=x_positions,
                y_positions=y_positions,
                half_widths=half_widths,
                half_heights=half_heights,
                scale_start=scale_start,
                scale_stop=1.0,
                epsilon=_GRAPHVIZ_FDP_PRISM_EPSILON,
                max_iterations=_GRAPHVIZ_FDP_PRISM_SCALE_MAX_ITERS,
            )
            max_overlap = 1.0
        if max_overlap <= 1.0 or residual < _GRAPHVIZ_FDP_PRISM_STRESS_TOL:
            if not neighborhood_only:
                break
            residual = 100000.0
            neighborhood_only = False
            shrink = do_shrinking
            continue
        residual = _graphviz_fdp_prism_stress_step(x_positions, y_positions, records)


def _graphviz_fdp_prism_overlap(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: Optional[torch.Tensor],
    ntry: int = _GRAPHVIZ_FDP_DEFAULT_PRISM_TRIES,
) -> torch.Tensor:
    """Apply Graphviz FDP's PRISM overlap-removal stage.

    Parameters
    ----------
    positions : torch.Tensor
        Positions in Graphviz internal inches with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Local edge tensor with shape ``[2, E]``.
    node_sizes : torch.Tensor, optional
        Node sizes in points with shape ``[N, 2]``.
    ntry : int, default=_GRAPHVIZ_FDP_DEFAULT_PRISM_TRIES
        Maximum PRISM smoother passes.

    Returns
    -------
    torch.Tensor
        PRISM-adjusted positions in Graphviz internal inches with shape
        ``[N, 2]``.
    """
    num_nodes = int(positions.shape[0])
    if num_nodes <= 1:
        return positions
    cpu_positions = positions.detach().to(device="cpu", dtype=torch.float64)
    x_positions = [float(cpu_positions[node_index, 0].item()) for node_index in range(num_nodes)]
    y_positions = [float(cpu_positions[node_index, 1].item()) for node_index in range(num_nodes)]
    half_widths, half_heights = _graphviz_fdp_prism_half_size_lists_in_inches(
        node_sizes,
        num_nodes,
    )
    _graphviz_fdp_prism_remove_overlap_lists(
        x_positions=x_positions,
        y_positions=y_positions,
        half_widths=half_widths,
        half_heights=half_heights,
        edge_index=edge_index.detach().to(device="cpu", dtype=torch.long),
        ntry=ntry,
    )
    return _graphviz_fdp_positions_from_lists(x_positions, y_positions).to(
        device=positions.device,
        dtype=positions.dtype,
    )


def _graphviz_fdp_x_overlap_lists(
    x_positions: Sequence[float],
    y_positions: Sequence[float],
    widths_in_inches: Sequence[float],
    heights_in_inches: Sequence[float],
    source: int,
    target: int,
) -> bool:
    """Return whether two nodes overlap under Graphviz ``xLayout`` margins.

    Parameters
    ----------
    x_positions : Sequence[float]
        X coordinates in Graphviz internal inches.
    y_positions : Sequence[float]
        Y coordinates in Graphviz internal inches.
    widths_in_inches : Sequence[float]
        Node widths including separation in inches.
    heights_in_inches : Sequence[float]
        Node heights including separation in inches.
    source : int
        First node index.
    target : int
        Second node index.

    Returns
    -------
    bool
        ``True`` when axis-aligned node boxes overlap.
    """
    x_delta = abs(x_positions[target] - x_positions[source])
    y_delta = abs(y_positions[target] - y_positions[source])
    width = (widths_in_inches[source] + widths_in_inches[target]) / 2.0
    height = (heights_in_inches[source] + heights_in_inches[target]) / 2.0
    return x_delta <= width and y_delta <= height


def _graphviz_fdp_x_overlap(
    positions: torch.Tensor,
    sizes_in_inches: torch.Tensor,
    source: int,
    target: int,
) -> bool:
    """Return whether two nodes overlap under Graphviz ``xLayout`` margins.

    Parameters
    ----------
    positions : torch.Tensor
        Positions in inches with shape ``[N, 2]``.
    sizes_in_inches : torch.Tensor
        Node sizes in inches with shape ``[N, 2]``.
    source : int
        First node index.
    target : int
        Second node index.

    Returns
    -------
    bool
        ``True`` when axis-aligned node boxes overlap.
    """
    x_delta = abs(float(positions[target, 0] - positions[source, 0]))
    y_delta = abs(float(positions[target, 1] - positions[source, 1]))
    width = float((sizes_in_inches[source, 0] + sizes_in_inches[target, 0]) / 2.0)
    height = float((sizes_in_inches[source, 1] + sizes_in_inches[target, 1]) / 2.0)
    return x_delta <= width and y_delta <= height


def _graphviz_fdp_apply_xlayout_repulsion(
    positions: torch.Tensor,
    displacement: torch.Tensor,
    sizes_in_inches: torch.Tensor,
    source: int,
    target: int,
    x_overlap_force: float,
    x_nonoverlap_force: float,
    phase: int,
) -> int:
    """Apply Graphviz ``xLayout`` pair repulsion.

    Parameters
    ----------
    positions : torch.Tensor
        Positions in inches with shape ``[N, 2]``.
    displacement : torch.Tensor
        Mutable displacement tensor with shape ``[N, 2]``.
    sizes_in_inches : torch.Tensor
        Node sizes in inches with shape ``[N, 2]``.
    source : int
        First node index.
    target : int
        Second node index.
    x_overlap_force : float
        Overlap repulsion numerator.
    x_nonoverlap_force : float
        Non-overlap repulsion numerator.
    phase : int
        Iteration counter for deterministic zero-distance fallback.

    Returns
    -------
    int
        ``1`` if nodes overlapped before movement, otherwise ``0``.
    """
    x_delta = float(positions[target, 0] - positions[source, 0])
    y_delta = float(positions[target, 1] - positions[source, 1])
    dist2 = x_delta * x_delta + y_delta * y_delta
    if dist2 == 0.0:
        x_delta, y_delta = _graphviz_fdp_disperse_zero_delta(source, target, phase)
        dist2 = x_delta * x_delta + y_delta * y_delta
    overlaps = _graphviz_fdp_x_overlap(positions, sizes_in_inches, source, target)
    force = (x_overlap_force if overlaps else x_nonoverlap_force) / dist2
    displacement[target, 0] += x_delta * force
    displacement[target, 1] += y_delta * force
    displacement[source, 0] -= x_delta * force
    displacement[source, 1] -= y_delta * force
    return 1 if overlaps else 0


def _graphviz_fdp_apply_xlayout_attraction(
    positions: torch.Tensor,
    displacement: torch.Tensor,
    sizes_in_inches: torch.Tensor,
    edge: tuple[int, int, float, float],
    x_k: float,
) -> None:
    """Apply Graphviz ``xLayout`` edge attraction.

    Parameters
    ----------
    positions : torch.Tensor
        Positions in inches with shape ``[N, 2]``.
    displacement : torch.Tensor
        Mutable displacement tensor with shape ``[N, 2]``.
    sizes_in_inches : torch.Tensor
        Node sizes in inches with shape ``[N, 2]``.
    edge : tuple[int, int, float, float]
        Edge record as ``(source, target, factor, dist)``.
    x_k : float
        Current ``xLayout`` spring constant. Graphviz increases this between
        overlap-removal tries, so attraction must read the try-local value.

    Returns
    -------
    None
        Updates ``displacement`` in place.
    """
    source, target, _factor, _edge_dist = edge
    if source == target or _graphviz_fdp_x_overlap(positions, sizes_in_inches, source, target):
        return
    x_delta = float(positions[target, 0] - positions[source, 0])
    y_delta = float(positions[target, 1] - positions[source, 1])
    dist = math.hypot(x_delta, y_delta)
    if dist == 0.0:
        return
    source_radius = math.hypot(
        float(sizes_in_inches[source, 0]) / 2.0,
        float(sizes_in_inches[source, 1]) / 2.0,
    )
    target_radius = math.hypot(
        float(sizes_in_inches[target, 0]) / 2.0,
        float(sizes_in_inches[target, 1]) / 2.0,
    )
    din = source_radius + target_radius
    dout = dist - din
    force = dout * dout / ((x_k + din) * dist)
    displacement[target, 0] -= x_delta * force
    displacement[target, 1] -= y_delta * force
    displacement[source, 0] += x_delta * force
    displacement[source, 1] += y_delta * force


def _graphviz_fdp_count_overlaps(
    positions: torch.Tensor,
    sizes_in_inches: torch.Tensor,
) -> int:
    """Count pairwise Graphviz ``xLayout`` overlaps.

    Parameters
    ----------
    positions : torch.Tensor
        Positions in inches with shape ``[N, 2]``.
    sizes_in_inches : torch.Tensor
        Node sizes in inches with shape ``[N, 2]``.

    Returns
    -------
    int
        Number of overlapping node pairs.
    """
    overlaps = 0
    for source in range(positions.shape[0]):
        for target in range(source + 1, positions.shape[0]):
            overlaps += int(_graphviz_fdp_x_overlap(positions, sizes_in_inches, source, target))
    return overlaps


def _graphviz_fdp_count_overlaps_lists(
    x_positions: Sequence[float],
    y_positions: Sequence[float],
    widths_in_inches: Sequence[float],
    heights_in_inches: Sequence[float],
) -> int:
    """Count pairwise Graphviz ``xLayout`` overlaps from Python float lists.

    Parameters
    ----------
    x_positions : Sequence[float]
        X coordinates in Graphviz internal inches.
    y_positions : Sequence[float]
        Y coordinates in Graphviz internal inches.
    widths_in_inches : Sequence[float]
        Node widths including separation in inches.
    heights_in_inches : Sequence[float]
        Node heights including separation in inches.

    Returns
    -------
    int
        Number of overlapping node pairs.
    """
    overlaps = 0
    for source in range(len(x_positions)):
        for target in range(source + 1, len(x_positions)):
            overlaps += int(
                _graphviz_fdp_x_overlap_lists(
                    x_positions,
                    y_positions,
                    widths_in_inches,
                    heights_in_inches,
                    source,
                    target,
                )
            )
    return overlaps


def _graphviz_fdp_apply_xlayout_repulsion_lists(
    x_positions: Sequence[float],
    y_positions: Sequence[float],
    widths_in_inches: Sequence[float],
    heights_in_inches: Sequence[float],
    x_displacements: List[float],
    y_displacements: List[float],
    source: int,
    target: int,
    x_overlap_force: float,
    x_nonoverlap_force: float,
    phase: int,
) -> int:
    """Apply Graphviz ``xLayout`` pair repulsion using Python float lists.

    Parameters
    ----------
    x_positions : Sequence[float]
        X coordinates in Graphviz internal inches.
    y_positions : Sequence[float]
        Y coordinates in Graphviz internal inches.
    widths_in_inches : Sequence[float]
        Node widths including separation in inches.
    heights_in_inches : Sequence[float]
        Node heights including separation in inches.
    x_displacements : list[float]
        Mutable X displacement list.
    y_displacements : list[float]
        Mutable Y displacement list.
    source : int
        First node index.
    target : int
        Second node index.
    x_overlap_force : float
        Overlap repulsion numerator.
    x_nonoverlap_force : float
        Non-overlap repulsion numerator.
    phase : int
        Iteration counter for deterministic zero-distance fallback.

    Returns
    -------
    int
        ``1`` if nodes overlapped before movement, otherwise ``0``.
    """
    x_delta = x_positions[target] - x_positions[source]
    y_delta = y_positions[target] - y_positions[source]
    dist2 = x_delta * x_delta + y_delta * y_delta
    if dist2 == 0.0:
        x_delta, y_delta = _graphviz_fdp_disperse_zero_delta(source, target, phase)
        dist2 = x_delta * x_delta + y_delta * y_delta
    overlaps = _graphviz_fdp_x_overlap_lists(
        x_positions,
        y_positions,
        widths_in_inches,
        heights_in_inches,
        source,
        target,
    )
    force = (x_overlap_force if overlaps else x_nonoverlap_force) / dist2
    x_displacements[target] += x_delta * force
    y_displacements[target] += y_delta * force
    x_displacements[source] -= x_delta * force
    y_displacements[source] -= y_delta * force
    return 1 if overlaps else 0


def _graphviz_fdp_apply_xlayout_attraction_lists(
    x_positions: Sequence[float],
    y_positions: Sequence[float],
    widths_in_inches: Sequence[float],
    heights_in_inches: Sequence[float],
    x_displacements: List[float],
    y_displacements: List[float],
    edge: Tuple[int, int, float, float],
    x_k: float,
) -> None:
    """Apply Graphviz ``xLayout`` edge attraction using Python float lists.

    Parameters
    ----------
    x_positions : Sequence[float]
        X coordinates in Graphviz internal inches.
    y_positions : Sequence[float]
        Y coordinates in Graphviz internal inches.
    widths_in_inches : Sequence[float]
        Node widths including separation in inches.
    heights_in_inches : Sequence[float]
        Node heights including separation in inches.
    x_displacements : list[float]
        Mutable X displacement list.
    y_displacements : list[float]
        Mutable Y displacement list.
    edge : tuple[int, int, float, float]
        Edge record as ``(source, target, factor, dist)``.
    x_k : float
        Current ``xLayout`` spring constant.

    Returns
    -------
    None
        Updates displacement lists in place.
    """
    source, target, _factor, _edge_dist = edge
    if source == target or _graphviz_fdp_x_overlap_lists(
        x_positions,
        y_positions,
        widths_in_inches,
        heights_in_inches,
        source,
        target,
    ):
        return
    x_delta = x_positions[target] - x_positions[source]
    y_delta = y_positions[target] - y_positions[source]
    dist = math.hypot(x_delta, y_delta)
    if dist == 0.0:
        return
    source_radius = math.hypot(widths_in_inches[source] / 2.0, heights_in_inches[source] / 2.0)
    target_radius = math.hypot(widths_in_inches[target] / 2.0, heights_in_inches[target] / 2.0)
    din = source_radius + target_radius
    dout = dist - din
    force = dout * dout / ((x_k + din) * dist)
    x_displacements[target] -= x_delta * force
    y_displacements[target] -= y_delta * force
    x_displacements[source] += x_delta * force
    y_displacements[source] += y_delta * force


def _graphviz_fdp_update_xlayout_position_lists(
    x_positions: List[float],
    y_positions: List[float],
    x_displacements: Sequence[float],
    y_displacements: Sequence[float],
    temperature: float,
) -> None:
    """Apply Graphviz ``xLayout`` position update to Python float lists.

    Parameters
    ----------
    x_positions : list[float]
        Mutable X coordinates in Graphviz internal inches.
    y_positions : list[float]
        Mutable Y coordinates in Graphviz internal inches.
    x_displacements : Sequence[float]
        X displacements in Graphviz internal inches.
    y_displacements : Sequence[float]
        Y displacements in Graphviz internal inches.
    temperature : float
        Current cooling temperature.

    Returns
    -------
    None
        Updates position lists in place.
    """
    temp2 = temperature * temperature
    for node_index in range(len(x_positions)):
        dx = x_displacements[node_index]
        dy = y_displacements[node_index]
        len2 = dx * dx + dy * dy
        if len2 < temp2:
            x_positions[node_index] += dx
            y_positions[node_index] += dy
        else:
            length = math.sqrt(len2)
            x_positions[node_index] += dx * temperature / length
            y_positions[node_index] += dy * temperature / length


def _graphviz_fdp_xlayout(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: Optional[torch.Tensor],
    edge_weights: Optional[torch.Tensor],
    xpms: tuple[float, float, float, int, int],
    node_ids: Optional[Sequence[str]] = None,
) -> torch.Tensor:
    """Run Graphviz ``fdp_xLayout``'s iterative overlap phase.

    Parameters
    ----------
    positions : torch.Tensor
        Initial positions in inches with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Local edge tensor with shape ``[2, E]``.
    node_sizes : torch.Tensor, optional
        Node sizes in points with shape ``[N, 2]``.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``.
    xpms : tuple[float, float, float, int, int]
        Parameters returned by ``fdp_tLayout`` as
        ``(T0, K, C, numIters, loopcnt)``.
    node_ids : Sequence[str], optional
        Trace node identifiers. When provided, overlap-removal updates are
        appended in Graphviz trace format.

    Returns
    -------
    torch.Tensor
        Expanded positions in inches with shape ``[N, 2]``.
    """
    num_nodes = int(positions.shape[0])
    if num_nodes <= 1:
        return positions
    cpu_positions = positions.detach().to(device="cpu", dtype=torch.float64)
    x_positions = [float(cpu_positions[node_index, 0].item()) for node_index in range(num_nodes)]
    y_positions = [float(cpu_positions[node_index, 1].item()) for node_index in range(num_nodes)]
    x_displacements = [0.0] * num_nodes
    y_displacements = [0.0] * num_nodes
    widths_in_inches, heights_in_inches = _graphviz_fdp_node_size_lists_in_inches(
        node_sizes,
        num_nodes,
    )
    sizes_in_inches = None
    outgoing, edges = _graphviz_fdp_edge_lists(edge_index, num_nodes, edge_weights)
    ov = _graphviz_fdp_count_overlaps_lists(
        x_positions,
        y_positions,
        widths_in_inches,
        heights_in_inches,
    )
    if node_ids is not None:
        trace_positions = _graphviz_fdp_positions_from_lists(x_positions, y_positions)
        sizes_in_inches = torch.tensor(
            list(zip(widths_in_inches, heights_in_inches)),
            dtype=torch.float64,
        )
        _fdp_trace_xlayout_event(
            "initial",
            -1,
            0,
            0,
            ov,
            xpms[1],
            0.0,
            trace_positions,
            sizes_in_inches,
            len(edges),
        )
    if ov == 0:
        return _graphviz_fdp_positions_from_lists(x_positions, y_positions)

    x_t0, x_k, x_c, x_num_iters, x_loopcnt = xpms
    if x_c <= 0.0:
        x_c = _GRAPHVIZ_FDP_DEFAULT_X_C
    base_k = x_k
    for try_index in range(_GRAPHVIZ_FDP_DEFAULT_X_TRIES):
        if ov == 0:
            break
        k2 = x_k * x_k
        x_overlap_force = x_c * k2
        x_nonoverlap_force = len(edges) * x_overlap_force * 2.0 / (num_nodes * (num_nodes - 1))
        if node_ids is not None:
            _fdp_trace_xlayout_event(
                "try_start",
                try_index * x_loopcnt,
                try_index,
                try_index,
                ov,
                x_k,
                x_t0,
                _graphviz_fdp_positions_from_lists(x_positions, y_positions),
                sizes_in_inches,
                len(edges),
            )
        for iteration in range(x_loopcnt):
            temperature = x_t0 * (x_num_iters - iteration) / x_num_iters
            if temperature <= 0.0:
                break
            if node_ids is not None:
                _fdp_trace_xlayout_event(
                    "before_adjust",
                    try_index * x_loopcnt + iteration,
                    try_index,
                    try_index,
                    ov,
                    x_k,
                    temperature,
                    _graphviz_fdp_positions_from_lists(x_positions, y_positions),
                    sizes_in_inches,
                    len(edges),
                )
            _graphviz_fdp_reset_displacements(x_displacements, y_displacements)
            overlaps_this_pass = 0
            for source in range(num_nodes):
                for target in range(source + 1, num_nodes):
                    overlaps_this_pass += _graphviz_fdp_apply_xlayout_repulsion_lists(
                        x_positions=x_positions,
                        y_positions=y_positions,
                        widths_in_inches=widths_in_inches,
                        heights_in_inches=heights_in_inches,
                        x_displacements=x_displacements,
                        y_displacements=y_displacements,
                        source=source,
                        target=target,
                        x_overlap_force=x_overlap_force,
                        x_nonoverlap_force=x_nonoverlap_force,
                        phase=try_index * x_loopcnt + iteration,
                    )
                for edge_id in outgoing[source]:
                    _graphviz_fdp_apply_xlayout_attraction_lists(
                        x_positions=x_positions,
                        y_positions=y_positions,
                        widths_in_inches=widths_in_inches,
                        heights_in_inches=heights_in_inches,
                        x_displacements=x_displacements,
                        y_displacements=y_displacements,
                        edge=edges[edge_id],
                        x_k=x_k,
                    )
            ov = overlaps_this_pass
            if node_ids is not None:
                _fdp_trace_xlayout_event(
                    "after_adjust",
                    try_index * x_loopcnt + iteration,
                    try_index,
                    try_index,
                    ov,
                    x_k,
                    temperature,
                    _graphviz_fdp_positions_from_lists(x_positions, y_positions),
                    sizes_in_inches,
                    len(edges),
                )
            if ov == 0:
                break
            _graphviz_fdp_update_xlayout_position_lists(
                x_positions=x_positions,
                y_positions=y_positions,
                x_displacements=x_displacements,
                y_displacements=y_displacements,
                temperature=temperature,
            )
            if node_ids is not None:
                _fdp_trace_positions(
                    "xlayout_adjust",
                    try_index * x_loopcnt + iteration,
                    node_ids,
                    _graphviz_fdp_positions_from_lists(x_positions, y_positions),
                )
        x_k += base_k
        if node_ids is not None:
            _fdp_trace_xlayout_event(
                "try_end",
                (try_index + 1) * x_loopcnt - 1,
                try_index,
                try_index + 1,
                ov,
                x_k,
                0.0,
                _graphviz_fdp_positions_from_lists(x_positions, y_positions),
                sizes_in_inches,
                len(edges),
            )
    return _graphviz_fdp_positions_from_lists(x_positions, y_positions)


def _graphviz_fdp_component_layout(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor],
    seed: int,
    edge_weights: Optional[torch.Tensor] = None,
    max_iters: int = _GRAPHVIZ_FDP_DEFAULT_MAX_ITERS,
    flip_y: bool = True,
) -> torch.Tensor:
    """Run the Graphviz fdp ``tLayout`` plus ``xLayout`` kernels.

    Parameters
    ----------
    edge_index : torch.Tensor
        Local edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of local nodes.
    node_sizes : torch.Tensor, optional
        Node sizes in points with shape ``[N, 2]``.
    seed : int
        Graphviz ``seed`` attribute value.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``.
    max_iters : int, default=_GRAPHVIZ_FDP_DEFAULT_MAX_ITERS
        Graphviz ``maxiter`` budget for the temperature schedule.
    flip_y : bool, default=True
        Whether to convert Graphviz's internal y-up coordinates to the
        benchmark adapter's y-down convention.

    Returns
    -------
    torch.Tensor
        Component positions in points with shape ``[N, 2]``.
    """
    if num_nodes == 0:
        return torch.empty((0, 2), dtype=torch.float32)
    if num_nodes == 1:
        return torch.zeros((1, 2), dtype=torch.float32)
    edge_index, edge_weights = _graphviz_fdp_collapse_parallel_edges(edge_index, edge_weights)
    positions, xpms = _graphviz_fdp_tlayout(
        edge_index=edge_index,
        num_nodes=num_nodes,
        seed=seed,
        edge_weights=edge_weights,
        max_iters=max_iters,
    )
    positions = _graphviz_fdp_xlayout(
        positions=positions,
        edge_index=edge_index,
        node_sizes=node_sizes,
        edge_weights=edge_weights,
        xpms=xpms,
    )
    positions = _graphviz_fdp_prism_overlap(
        positions=positions,
        edge_index=edge_index,
        node_sizes=node_sizes,
    )
    result = positions * _GRAPHVIZ_FDP_POINTS_PER_INCH
    if flip_y:
        result[:, 1] *= -1.0
    return result.to(dtype=torch.float32)


def _c_round(value: float) -> int:
    """Round like C99 ``round``.

    Parameters
    ----------
    value : float
        Input value.

    Returns
    -------
    int
        Nearest integer with half values rounded away from zero.
    """
    if value >= 0.0:
        return int(math.floor(value + 0.5))
    return int(math.ceil(value - 0.5))


def _c_int_div(numerator: int, denominator: int) -> int:
    """Divide integers like C99 truncation toward zero.

    Parameters
    ----------
    numerator : int
        Integer numerator.
    denominator : int
        Non-zero integer denominator.

    Returns
    -------
    int
        Truncated quotient.
    """
    quotient = abs(numerator) // abs(denominator)
    return quotient if numerator * denominator >= 0 else -quotient


def _graphviz_grid_count(width: float, step: int) -> int:
    """Return Graphviz ``GRID`` cell count for a positive span.

    Parameters
    ----------
    width : float
        Span length.
    step : int
        Grid cell size.

    Returns
    -------
    int
        Number of grid cells needed to cover the span.
    """
    return int(math.ceil(width / step))


def _graphviz_cell(value: float, step: int) -> int:
    """Return the Graphviz grid cell containing a coordinate.

    Parameters
    ----------
    value : float
        Coordinate value.
    step : int
        Grid cell size.

    Returns
    -------
    int
        Grid-cell coordinate using Graphviz's C integer truncation.
    """
    integer_value = int(value)
    if integer_value >= 0:
        return _c_int_div(integer_value, step)
    return _c_int_div(integer_value + 1, step) - 1


def _graphviz_pack_step(
    boxes: list[tuple[float, float, float, float]],
    margin: float,
) -> int:
    """Compute Graphviz pack.c's grid step size for component boxes.

    Parameters
    ----------
    boxes : list[tuple[float, float, float, float]]
        Bounding boxes as ``(llx, lly, urx, ury)``.
    margin : float
        Extra pack margin around each component.

    Returns
    -------
    int
        Positive grid step size.
    """
    count = len(boxes)
    a_value = _GRAPHVIZ_PACK_AVERAGE_POLYOMINO_SIZE * count - 1.0
    b_value = 0.0
    c_value = 0.0
    for llx, lly, urx, ury in boxes:
        width = urx - llx + 2.0 * margin
        height = ury - lly + 2.0 * margin
        b_value -= width + height
        c_value -= width * height
    discriminant = b_value * b_value - 4.0 * a_value * c_value
    root = int((-b_value + math.sqrt(discriminant)) / (2.0 * a_value))
    return root if root != 0 else 1


def _graphviz_box_cells(
    box: tuple[float, float, float, float],
    step: int,
    margin: float,
) -> tuple[list[tuple[int, int]], int]:
    """Generate the bbox polyomino cells used by Graphviz ``genBox``.

    Parameters
    ----------
    box : tuple[float, float, float, float]
        Bounding box as ``(llx, lly, urx, ury)``.
    step : int
        Grid cell size.
    margin : float
        Extra pack margin around the box.

    Returns
    -------
    tuple[list[tuple[int, int]], int]
        Occupied cells and half-perimeter sort key.
    """
    llx, lly, urx, ury = box
    rounded_llx = _c_round(llx)
    rounded_lly = _c_round(lly)
    rounded_urx = _c_round(urx)
    rounded_ury = _c_round(ury)
    low_x = _graphviz_cell(-margin, step)
    low_y = _graphviz_cell(-margin, step)
    high_x = _graphviz_cell(float(rounded_urx - rounded_llx) + margin, step)
    high_y = _graphviz_cell(float(rounded_ury - rounded_lly) + margin, step)

    cells = [
        (x_coord, y_coord)
        for x_coord in range(low_x, high_x + 1)
        for y_coord in range(low_y, high_y + 1)
    ]
    width_cells = _graphviz_grid_count(urx - llx + 2.0 * margin, step)
    height_cells = _graphviz_grid_count(ury - lly + 2.0 * margin, step)
    return cells, width_cells + height_cells


def _graphviz_fits(
    x_cell: int,
    y_cell: int,
    cells: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
) -> bool:
    """Return whether a translated polyomino does not overlap occupied cells.

    Parameters
    ----------
    x_cell : int
        Candidate x grid-cell offset.
    y_cell : int
        Candidate y grid-cell offset.
    cells : list[tuple[int, int]]
        Polyomino cells.
    occupied : set[tuple[int, int]]
        Already occupied cells.

    Returns
    -------
    bool
        ``True`` if every translated cell is available.
    """
    return all((x_coord + x_cell, y_coord + y_cell) not in occupied for x_coord, y_coord in cells)


def _graphviz_commit_fit(
    x_cell: int,
    y_cell: int,
    cells: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
    box: tuple[float, float, float, float],
    step: int,
) -> tuple[float, float]:
    """Commit a fitted polyomino and return its Graphviz translation.

    Parameters
    ----------
    x_cell : int
        Accepted x grid-cell offset.
    y_cell : int
        Accepted y grid-cell offset.
    cells : list[tuple[int, int]]
        Polyomino cells.
    occupied : set[tuple[int, int]]
        Mutable occupied-cell set.
    box : tuple[float, float, float, float]
        Original component bounding box.
    step : int
        Grid cell size.

    Returns
    -------
    tuple[float, float]
        Translation in layout units.
    """
    for x_coord, y_coord in cells:
        occupied.add((x_coord + x_cell, y_coord + y_cell))
    return float(step * x_cell - _c_round(box[0])), float(step * y_cell - _c_round(box[1]))


def _graphviz_place_component(
    sorted_index: int,
    cells: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
    box: tuple[float, float, float, float],
    step: int,
    margin: float,
) -> tuple[float, float]:
    """Place one component using Graphviz pack.c's spiral search.

    Parameters
    ----------
    sorted_index : int
        Position in descending polyomino-perimeter order.
    cells : list[tuple[int, int]]
        Polyomino cells for the component.
    occupied : set[tuple[int, int]]
        Mutable occupied-cell set.
    box : tuple[float, float, float, float]
        Original component bounding box.
    step : int
        Grid cell size.
    margin : float
        Extra pack margin around the component.

    Returns
    -------
    tuple[float, float]
        Translation for the component.
    """
    llx, lly, urx, ury = box
    if sorted_index == 0:
        width_cells = _graphviz_grid_count(urx - llx + 2.0 * margin, step)
        height_cells = _graphviz_grid_count(ury - lly + 2.0 * margin, step)
        x_cell = _c_int_div(-width_cells, 2)
        y_cell = _c_int_div(-height_cells, 2)
        if _graphviz_fits(x_cell, y_cell, cells, occupied):
            return _graphviz_commit_fit(x_cell, y_cell, cells, occupied, box, step)

    if _graphviz_fits(0, 0, cells, occupied):
        return _graphviz_commit_fit(0, 0, cells, occupied, box, step)

    width = math.ceil(urx - llx)
    height = math.ceil(ury - lly)
    bound = 1
    while True:
        if width >= height:
            x_cell = 0
            y_cell = -bound
            while x_cell < bound:
                if _graphviz_fits(x_cell, y_cell, cells, occupied):
                    return _graphviz_commit_fit(x_cell, y_cell, cells, occupied, box, step)
                x_cell += 1
            while y_cell < bound:
                if _graphviz_fits(x_cell, y_cell, cells, occupied):
                    return _graphviz_commit_fit(x_cell, y_cell, cells, occupied, box, step)
                y_cell += 1
            while x_cell > -bound:
                if _graphviz_fits(x_cell, y_cell, cells, occupied):
                    return _graphviz_commit_fit(x_cell, y_cell, cells, occupied, box, step)
                x_cell -= 1
            while y_cell > -bound:
                if _graphviz_fits(x_cell, y_cell, cells, occupied):
                    return _graphviz_commit_fit(x_cell, y_cell, cells, occupied, box, step)
                y_cell -= 1
            while x_cell < 0:
                if _graphviz_fits(x_cell, y_cell, cells, occupied):
                    return _graphviz_commit_fit(x_cell, y_cell, cells, occupied, box, step)
                x_cell += 1
        else:
            y_cell = 0
            x_cell = -bound
            while y_cell > -bound:
                if _graphviz_fits(x_cell, y_cell, cells, occupied):
                    return _graphviz_commit_fit(x_cell, y_cell, cells, occupied, box, step)
                y_cell -= 1
            while x_cell < bound:
                if _graphviz_fits(x_cell, y_cell, cells, occupied):
                    return _graphviz_commit_fit(x_cell, y_cell, cells, occupied, box, step)
                x_cell += 1
            while y_cell < bound:
                if _graphviz_fits(x_cell, y_cell, cells, occupied):
                    return _graphviz_commit_fit(x_cell, y_cell, cells, occupied, box, step)
                y_cell += 1
            while x_cell > -bound:
                if _graphviz_fits(x_cell, y_cell, cells, occupied):
                    return _graphviz_commit_fit(x_cell, y_cell, cells, occupied, box, step)
                x_cell -= 1
            while y_cell > 0:
                if _graphviz_fits(x_cell, y_cell, cells, occupied):
                    return _graphviz_commit_fit(x_cell, y_cell, cells, occupied, box, step)
                y_cell -= 1
        bound += 1


def _graphviz_tile_pack_offsets(
    boxes: list[tuple[float, float, float, float]],
    margin: float = _GRAPHVIZ_FDP_PACK_MARGIN,
) -> list[tuple[float, float]]:
    """Pack component boxes with Graphviz's bbox polyomino tile search.

    Parameters
    ----------
    boxes : list[tuple[float, float, float, float]]
        Component bounding boxes as ``(llx, lly, urx, ury)``.
    margin : float, default=4.0
        Graphviz fdp's default pack margin in points, ``CL_OFFSET / 2``.

    Returns
    -------
    list[tuple[float, float]]
        Per-component translations in original component order.
    """
    if not boxes:
        return []
    step = _graphviz_pack_step(boxes, margin)
    packed_info: list[tuple[int, int, list[tuple[int, int]]]] = []
    for index, box in enumerate(boxes):
        cells, perimeter = _graphviz_box_cells(box, step, margin)
        packed_info.append((index, perimeter, cells))

    packed_info.sort(key=lambda item: -item[1])
    occupied: set[tuple[int, int]] = set()
    offsets = [(0.0, 0.0) for _ in boxes]
    for sorted_index, (box_index, _, cells) in enumerate(packed_info):
        offsets[box_index] = _graphviz_place_component(
            sorted_index=sorted_index,
            cells=cells,
            occupied=occupied,
            box=boxes[box_index],
            step=step,
            margin=margin,
        )
    return offsets


def _component_box(
    positions: torch.Tensor,
    node_sizes: Optional[torch.Tensor],
) -> tuple[float, float, float, float]:
    """Compute a component bounding box from positions and optional node sizes.

    Parameters
    ----------
    positions : torch.Tensor
        Component positions with shape ``[C, 2]``.
    node_sizes : torch.Tensor, optional
        Optional node sizes with shape ``[C, 2]``.

    Returns
    -------
    tuple[float, float, float, float]
        Bounding box as ``(llx, lly, urx, ury)``.
    """
    positions_cpu = positions.detach().to(device="cpu", dtype=torch.float64)
    if positions_cpu.numel() == 0:
        return (0.0, 0.0, 0.0, 0.0)
    if node_sizes is None or node_sizes.numel() == 0:
        half_sizes = torch.zeros_like(positions_cpu)
    else:
        half_sizes = node_sizes.detach().to(device="cpu", dtype=torch.float64) / 2.0
    lower = positions_cpu - half_sizes
    upper = positions_cpu + half_sizes
    mins = lower.min(dim=0).values
    maxs = upper.max(dim=0).values
    return (
        float(mins[0].item()),
        float(mins[1].item()),
        float(maxs[0].item()),
        float(maxs[1].item()),
    )


def _translate_packed_components_to_origin(
    packed: torch.Tensor,
    node_sizes: Optional[torch.Tensor],
) -> torch.Tensor:
    """Translate packed component coordinates so the lower-left box is at zero.

    Parameters
    ----------
    packed : torch.Tensor
        Packed positions with shape ``[N, 2]``.
    node_sizes : torch.Tensor, optional
        Optional node sizes with shape ``[N, 2]``.

    Returns
    -------
    torch.Tensor
        Packed positions translated like Graphviz ``finalCC`` root output.
    """
    if packed.numel() == 0:
        return packed
    if node_sizes is None or node_sizes.numel() == 0:
        half_sizes = torch.zeros_like(packed)
    else:
        half_sizes = node_sizes.to(device=packed.device, dtype=packed.dtype) / 2.0
    lower = packed - half_sizes
    mins = lower.min(dim=0).values
    # Graphviz finalCC stores root bounding boxes as integer-point boxes before
    # render plugins emit coordinates, so the root translation must use BF2B
    # rounding rather than the raw floating lower-left corner.
    shift = torch.tensor(
        [
            -float(_c_round(float(mins[0].item()))),
            -float(_c_round(float(mins[1].item()))),
        ],
        dtype=packed.dtype,
        device=packed.device,
    )
    return packed + shift.unsqueeze(0)


def _graphviz_quantize_output_points(positions: torch.Tensor) -> torch.Tensor:
    """Round final fidelity coordinates like Graphviz JSON/plain output.

    Parameters
    ----------
    positions : torch.Tensor
        Final positions in points with shape ``[N, 2]``.

    Returns
    -------
    torch.Tensor
        Positions with each coordinate parsed through Graphviz's ``%.5g`` text
        formatting, preserving the input dtype and device.
    """
    if positions.numel() == 0:
        return positions
    cpu_positions = positions.detach().to(device="cpu", dtype=torch.float64)
    quantized = torch.empty_like(cpu_positions)
    for node_index in range(cpu_positions.shape[0]):
        for axis in range(cpu_positions.shape[1]):
            quantized[node_index, axis] = float(
                f"{float(cpu_positions[node_index, axis].item()):.5g}"
            )
    return quantized.to(device=positions.device, dtype=positions.dtype)


def build_fmmm_pipeline(
    steps: int = 200,
    force_model: str = "ogdf_new",
    reference_mode: bool = False,
    fidelity_mode: Union[bool, str] = False,
    fidelity_dtype: torch.dtype = torch.float32,
) -> Pipeline:
    """Build an FM^3 multilevel force-directed pipeline.

    Reference fidelity
    ------------------
    Targets: Graphviz 7.0.5 fdp / Hachul and Junger (2004), "Drawing Large
        Graphs with a Potential-Field-Based Multilevel Algorithm".
    Fidelity mode: ``reference_mode=True`` or ``fidelity_mode=True`` enables
        OGDF-aligned coarsening, coarsest initialization, and force scaling
        choices used by evaluation competitors. ``fidelity_mode="graphviz_fdp"``
        selects the Graphviz FDP cluster-recursion compatibility route.
    Verified at: final 100-seed report, strong equivalent; median RMSD 0.067
        to 0.179 across step-count variants. Round 33 fdp bounded subset
        remained 0.121966.
    Known divergences:
        - Dagua keeps a fallback single-level solve when multilevel setup is
          unsuitable.

    Parameters
    ----------
    steps : int, default=200
        Total refinement budget distributed across hierarchy levels.
    force_model : str, default="ogdf_new"
        Spring-force model for edge attraction. ``"ogdf_new"`` matches
        OGDF's default; ``"fr"`` preserves Dagua's earlier coefficient for
        benchmark fallback selection.
    reference_mode : bool, default=False
        Use OGDF-aligned coarsening, coarsest initialization, and force
        scaling choices for fidelity comparisons.
    fidelity_mode : bool or str, default=False
        Alias for ``reference_mode`` used by evaluation competitors. The
        special value ``"graphviz_fdp"`` selects Graphviz FDP cluster-recursion
        compatibility.

    Returns
    -------
    Pipeline
        Pipeline implementing the FM^3 algorithm. The pipeline produces final
        node coordinates by constructing a multilevel hierarchy, initializing
        the coarsest graph, refining that level, uncoarsening with per-level
        refinement, falling back to a single-level solve when needed, and
        normalizing the result.

    Raises
    ------
    ValueError
        If ``steps`` is negative.
    """
    if steps < 0:
        raise ValueError("steps must be non-negative.")

    effective_reference_mode = bool(reference_mode or fidelity_mode)
    initialize_state = _InitializeFMMMState(
        config=_InitializeFMMMStateConfig(
            steps=steps,
            force_model=force_model,
            galaxy_choice="lower" if effective_reference_mode else "higher",
            coarsest_init="ogdf_random" if effective_reference_mode else "fr",
            ogdf_force_scaling=effective_reference_mode,
            sum_parallel_weights=not effective_reference_mode,
        )
    )
    initialize_coarsest = _InitializeCoarsestLevel()
    refine_coarsest = _RefineCoarsestLevel()
    uncoarsen_loop = _UncoarsenLoop()
    single_level_fallback = _SingleLevelFallback()
    finalize_positions = _FinalizeFMMMPositions()

    ops: List[Op] = [
        initialize_state,
        initialize_coarsest,
        refine_coarsest,
        uncoarsen_loop,
        single_level_fallback,
        finalize_positions,
    ]
    if fidelity_mode:
        ops.append(_FdpCompoundEdgeAttachmentOp())

    return Pipeline(ops, name="fmmm_pipeline")


def _run_fmmm_pipeline_once(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor],
    steps: int,
    seed: int,
    edge_weights: Optional[torch.Tensor],
    force_model: str,
    reference_mode: bool,
    fidelity_mode: bool,
    fidelity_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Run the FMMM op pipeline once without component decomposition.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes in the graph.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]``.
    steps : int
        Total refinement budget.
    seed : int
        Random seed.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``.
    force_model : str
        Spring-force model.
    reference_mode : bool
        Whether to use reference coarsening and force scaling.
    fidelity_mode : bool
        Evaluation alias for ``reference_mode``.
    fidelity_dtype : torch.dtype, default=torch.float32
        Fidelity-mode internal dtype requested by the public wrapper.

    Returns
    -------
    torch.Tensor
        Final position tensor with shape ``[N, 2]``.

    Raises
    ------
    RuntimeError
        If the pipeline does not produce positions.
    """
    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        edge_weights=edge_weights,
        seed=seed,
    )
    state = SolveState()
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))
    final_state = build_fmmm_pipeline(
        steps=steps,
        force_model=force_model,
        reference_mode=reference_mode,
        fidelity_mode=fidelity_mode,
        fidelity_dtype=fidelity_dtype,
    ).apply(
        problem,
        state,
        ctx,
    )
    if final_state.pos is None:
        raise RuntimeError("FM^3 pipeline did not produce final positions.")
    return final_state.pos


def _layout_fmmm_fidelity_components(
    edge_index: torch.Tensor,
    components: list[list[int]],
    num_nodes: int,
    node_sizes: Optional[torch.Tensor],
    steps: int,
    seed: int,
    edge_weights: Optional[torch.Tensor],
    force_model: str,
    reference_mode: bool,
    fidelity_mode: bool,
    fidelity_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Lay out weak components independently and pack them like Graphviz fdp.

    Parameters
    ----------
    edge_index : torch.Tensor
        Parent edge tensor with shape ``[2, E]``.
    components : list[list[int]]
        Weak components in parent node order.
    num_nodes : int
        Total parent node count.
    node_sizes : torch.Tensor, optional
        Optional parent node sizes with shape ``[N, 2]``.
    steps : int
        Compatibility parameter retained for the public FMMM variant. Graphviz
        fdp fidelity uses Graphviz's default ``maxiter`` constant.
    seed : int
        Random seed reused for each component, matching ``fdp_tLayout``
        reseeding from ``T_seed``.
    edge_weights : torch.Tensor, optional
        Optional parent edge weights with shape ``[E]``.
    force_model : str
        Spring-force model.
    reference_mode : bool
        Whether to use reference coarsening and force scaling.
    fidelity_mode : bool
        Evaluation alias for ``reference_mode``.
    fidelity_dtype : torch.dtype, default=torch.float32
        Fidelity-mode internal dtype requested by the public wrapper.

    Returns
    -------
    torch.Tensor
        Packed parent coordinates with shape ``[N, 2]``.
    """
    del force_model, reference_mode, fidelity_mode, fidelity_dtype
    device = _layout_device(edge_index=edge_index, node_sizes=node_sizes)
    component_positions: list[torch.Tensor] = []
    boxes: list[tuple[float, float, float, float]] = []
    for component in components:
        local_edges, local_weights = _slice_component_edges(edge_index, edge_weights, component)
        local_sizes = node_sizes[component] if node_sizes is not None else None
        local_pos = _graphviz_fdp_component_layout(
            edge_index=local_edges,
            num_nodes=len(component),
            node_sizes=local_sizes,
            seed=seed,
            edge_weights=local_weights,
            max_iters=steps,
            flip_y=False,
        )
        component_positions.append(local_pos)
        boxes.append(_component_box(local_pos, local_sizes))

    offsets = _graphviz_tile_pack_offsets(boxes)
    dtype = component_positions[0].dtype
    packed = torch.zeros((num_nodes, 2), dtype=dtype, device=device)
    for component, local_pos, offset in zip(components, component_positions, offsets):
        offset_tensor = torch.tensor(offset, dtype=dtype, device=local_pos.device)
        packed[component] = (local_pos + offset_tensor).to(device=device, dtype=dtype)
    translated = _translate_packed_components_to_origin(packed, node_sizes).to(dtype=torch.float32)
    translated[:, 1] *= -1.0
    return _graphviz_quantize_output_points(translated)


def layout_fmmm_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    steps: int = 200,
    seed: int = 42,
    edge_weights: Optional[torch.Tensor] = None,
    force_model: str = "ogdf_new",
    reference_mode: bool = False,
    fidelity_mode: bool = False,
    fidelity_dtype: torch.dtype = torch.float32,
    clusters: Optional[Mapping[str, Sequence[int]]] = None,
    cluster_parents: Optional[Mapping[str, Optional[str]]] = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Run the FM^3 pipeline as a drop-in replacement.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes ``N`` in the graph.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]`` used for extent
        calculation and output-device selection.
    steps : int, default=200
        Total refinement budget distributed across hierarchy levels.
    seed : int, default=42
        Random seed for coarsening, coarse initialization, and prolongation
        jitter.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.
    force_model : str, default="ogdf_new"
        Spring-force model for edge attraction.
    reference_mode : bool, default=False
        Use OGDF-aligned reference behavior for algorithm fidelity runs.
    fidelity_mode : bool or str, default=False
        Alias for ``reference_mode`` used by evaluation competitors. The
        special value ``"graphviz_fdp"`` selects the cluster-aware Graphviz FDP
        fidelity route; boolean ``True`` selects OGDF FMMM plain-graph
        fidelity.
    clusters : Mapping[str, Sequence[int]], optional
        Cluster membership. Only used for ``fidelity_mode="graphviz_fdp"``.
    cluster_parents : Mapping[str, str | None], optional
        Cluster parent mapping. Only used for ``fidelity_mode="graphviz_fdp"``.
    **kwargs : Any
        Ignored compatibility keywords from generic layout dispatch.

    Returns
    -------
    torch.Tensor
        Final position tensor with shape ``[N, 2]``.

    Raises
    ------
    ValueError
        If ``num_nodes``, ``steps``, ``edge_weights``, or ``force_model`` are
        invalid.
    RuntimeError
        If the pipeline fails to populate final positions.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    if force_model not in {"ogdf_new", "fr"}:
        raise ValueError("force_model must be either 'ogdf_new' or 'fr'.")
    if edge_weights is not None:
        if edge_weights.ndim != 1:
            raise ValueError("edge_weights must have shape [E].")
        if edge_weights.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"edge_weights length {edge_weights.shape[0]} != edge_count {edge_index.shape[1]}"
            )
    del kwargs

    device = _layout_device(edge_index=edge_index, node_sizes=node_sizes)
    if num_nodes == 0:
        return torch.empty((0, 2), dtype=torch.float32, device=device)
    if num_nodes == 1:
        return torch.zeros((1, 2), dtype=torch.float32, device=device)
    if fidelity_mode == "graphviz_fdp":
        if clusters:
            return graphviz_fdp_fidelity(
                edge_index=edge_index,
                num_nodes=num_nodes,
                node_sizes=node_sizes,
                steps=steps,
                seed=seed,
                clusters=clusters,
                cluster_parents=cluster_parents,
            )
        components = _weak_components(edge_index, num_nodes)
        return _layout_fmmm_fidelity_components(
            edge_index=edge_index,
            components=components,
            num_nodes=num_nodes,
            node_sizes=node_sizes,
            steps=steps,
            seed=seed,
            edge_weights=edge_weights,
            force_model=force_model,
            reference_mode=reference_mode,
            fidelity_mode=True,
            fidelity_dtype=fidelity_dtype,
        )

    effective_reference_mode = reference_mode or fidelity_mode
    if effective_reference_mode:
        return _layout_ogdf_fmmm_component_fidelity(
            edge_index=edge_index,
            num_nodes=num_nodes,
            fixed_iterations=steps,
            seed=seed,
            device=device,
            node_sizes=node_sizes,
        ).to(dtype=torch.float32)

    return _run_fmmm_pipeline_once(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        steps=steps,
        seed=seed,
        edge_weights=edge_weights,
        force_model=force_model,
        reference_mode=reference_mode,
        fidelity_mode=fidelity_mode,
        fidelity_dtype=fidelity_dtype,
    )


__all__ = ["build_fmmm_pipeline", "graphviz_fdp_fidelity", "layout_fmmm_pipeline"]
