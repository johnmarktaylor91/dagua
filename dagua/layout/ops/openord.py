"""Registered operations implementing the OpenOrd layout schedule."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Mapping, Optional, Protocol, Union, cast

import torch

from dagua.layout.ops.base import Op
from dagua.layout.ops.drl import (
    DRLDensityGridConfig,
    _build_undirected_adjacency,
    _DrlParameters,
    _PhaseParameters,
    _run_reference_drl,
)
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op


class OpenOrdOptionObject(Protocol):
    """Attribute-style OpenOrd option container."""

    def __getattr__(self, name: str) -> object:
        """Return an option value by attribute name.

        Parameters
        ----------
        name : str
            Attribute name to resolve.

        Returns
        -------
        object
            Option value.
        """


OpenOrdOptions = Union[str, Mapping[str, object], OpenOrdOptionObject]


@dataclass(frozen=True)
class OpenOrdPrepareStateConfig:
    """Configuration for :class:`OpenOrdPrepareState`.

    Parameters
    ----------
    options : str or Mapping[str, object] or OpenOrdOptionObject, default="default"
        Preset name or override provider.
    edge_cut : float, optional
        Edge-cutting ratio in ``[0, 1]``. ``None`` uses the preset default.
    """

    options: OpenOrdOptions = "default"
    edge_cut: Optional[float] = None


_OPENORD_PRESETS: dict[str, _DrlParameters] = {
    "default": _DrlParameters(
        edge_cut=32.0 / 40.0,
        init=_PhaseParameters(0, 2000.0, 10.0, 1.0),
        liquid=_PhaseParameters(200, 2000.0, 2.0, 1.0),
        expansion=_PhaseParameters(200, 2000.0, 10.0, 1.0),
        cooldown=_PhaseParameters(200, 2000.0, 1.0, 0.1),
        crunch=_PhaseParameters(50, 250.0, 1.0, 0.25),
        simmer=_PhaseParameters(100, 250.0, 0.5, 0.0),
    ),
    "coarsen": _DrlParameters(
        edge_cut=32.0 / 40.0,
        init=_PhaseParameters(0, 2000.0, 10.0, 1.0),
        liquid=_PhaseParameters(200, 2000.0, 2.0, 1.0),
        expansion=_PhaseParameters(200, 2000.0, 10.0, 1.0),
        cooldown=_PhaseParameters(200, 2000.0, 1.0, 0.1),
        crunch=_PhaseParameters(50, 250.0, 1.0, 0.25),
        simmer=_PhaseParameters(100, 250.0, 0.5, 0.0),
    ),
    "coarsest": _DrlParameters(
        edge_cut=32.0 / 40.0,
        init=_PhaseParameters(0, 2000.0, 10.0, 1.0),
        liquid=_PhaseParameters(200, 2000.0, 2.0, 1.0),
        expansion=_PhaseParameters(200, 2000.0, 10.0, 1.0),
        cooldown=_PhaseParameters(200, 2000.0, 1.0, 0.1),
        crunch=_PhaseParameters(200, 250.0, 1.0, 0.25),
        simmer=_PhaseParameters(100, 250.0, 0.5, 0.0),
    ),
    "refine": _DrlParameters(
        edge_cut=0.5,
        init=_PhaseParameters(0, 50.0, 0.5, 0.0),
        liquid=_PhaseParameters(0, 2000.0, 2.0, 1.0),
        expansion=_PhaseParameters(50, 500.0, 0.1, 0.25),
        cooldown=_PhaseParameters(50, 200.0, 1.0, 0.1),
        crunch=_PhaseParameters(50, 250.0, 1.0, 0.25),
        simmer=_PhaseParameters(0, 250.0, 0.5, 0.0),
    ),
    "final": _DrlParameters(
        edge_cut=0.5,
        init=_PhaseParameters(0, 50.0, 0.5, 0.0),
        liquid=_PhaseParameters(0, 2000.0, 2.0, 1.0),
        expansion=_PhaseParameters(50, 50.0, 0.1, 0.25),
        cooldown=_PhaseParameters(50, 200.0, 1.0, 0.1),
        crunch=_PhaseParameters(50, 250.0, 1.0, 0.25),
        simmer=_PhaseParameters(25, 250.0, 0.5, 0.0),
    ),
}


def _lookup_option(options: OpenOrdOptions, name: str) -> Optional[object]:
    """Read one OpenOrd option from a mapping or attribute object.

    Parameters
    ----------
    options : str or Mapping[str, object] or OpenOrdOptionObject
        Raw OpenOrd options.
    name : str
        Option key.

    Returns
    -------
    object or None
        Returned option value if present.
    """
    if isinstance(options, str):
        return None
    if isinstance(options, Mapping):
        return options.get(name)
    return getattr(options, name, None)


def _resolve_openord_parameters(
    options: OpenOrdOptions,
    edge_cut: Optional[float] = None,
) -> _DrlParameters:
    """Resolve OpenOrd parameters against source-preserving presets.

    Parameters
    ----------
    options : str or Mapping[str, object] or OpenOrdOptionObject
        Preset name or override container.
    edge_cut : float, optional
        Explicit edge-cutting ratio overriding the preset.

    Returns
    -------
    _DrlParameters
        Resolved phase schedule and edge-cut parameter.

    Raises
    ------
    ValueError
        If a preset name is unknown or edge cutting is outside ``[0, 1]``.
    """
    preset_name = (
        options if isinstance(options, str) else str(_lookup_option(options, "preset") or "default")
    )
    try:
        default = _OPENORD_PRESETS[preset_name]
    except KeyError as exc:
        available = ", ".join(sorted(_OPENORD_PRESETS))
        raise ValueError(
            f"unknown OpenOrd preset {preset_name!r}; expected one of {available}."
        ) from exc

    values: dict[str, float] = {
        "edge_cut": default.edge_cut,
        "init_iterations": float(default.init.iterations),
        "init_temperature": default.init.temperature,
        "init_attraction": default.init.attraction,
        "init_damping_mult": default.init.damping_mult,
        "liquid_iterations": float(default.liquid.iterations),
        "liquid_temperature": default.liquid.temperature,
        "liquid_attraction": default.liquid.attraction,
        "liquid_damping_mult": default.liquid.damping_mult,
        "expansion_iterations": float(default.expansion.iterations),
        "expansion_temperature": default.expansion.temperature,
        "expansion_attraction": default.expansion.attraction,
        "expansion_damping_mult": default.expansion.damping_mult,
        "cooldown_iterations": float(default.cooldown.iterations),
        "cooldown_temperature": default.cooldown.temperature,
        "cooldown_attraction": default.cooldown.attraction,
        "cooldown_damping_mult": default.cooldown.damping_mult,
        "crunch_iterations": float(default.crunch.iterations),
        "crunch_temperature": default.crunch.temperature,
        "crunch_attraction": default.crunch.attraction,
        "crunch_damping_mult": default.crunch.damping_mult,
        "simmer_iterations": float(default.simmer.iterations),
        "simmer_temperature": default.simmer.temperature,
        "simmer_attraction": default.simmer.attraction,
        "simmer_damping_mult": default.simmer.damping_mult,
    }
    for key in tuple(values):
        override = _lookup_option(options=options, name=key)
        if override is not None:
            values[key] = float(cast(float, override))
    if edge_cut is not None:
        values["edge_cut"] = float(edge_cut)
    if not 0.0 <= values["edge_cut"] <= 1.0:
        raise ValueError("edge_cut must be between 0 and 1.")

    return _DrlParameters(
        edge_cut=values["edge_cut"],
        init=_PhaseParameters(
            int(values["init_iterations"]),
            values["init_temperature"],
            values["init_attraction"],
            values["init_damping_mult"],
        ),
        liquid=_PhaseParameters(
            int(values["liquid_iterations"]),
            values["liquid_temperature"],
            values["liquid_attraction"],
            values["liquid_damping_mult"],
        ),
        expansion=_PhaseParameters(
            int(values["expansion_iterations"]),
            values["expansion_temperature"],
            values["expansion_attraction"],
            values["expansion_damping_mult"],
        ),
        cooldown=_PhaseParameters(
            int(values["cooldown_iterations"]),
            values["cooldown_temperature"],
            values["cooldown_attraction"],
            values["cooldown_damping_mult"],
        ),
        crunch=_PhaseParameters(
            int(values["crunch_iterations"]),
            values["crunch_temperature"],
            values["crunch_attraction"],
            values["crunch_damping_mult"],
        ),
        simmer=_PhaseParameters(
            int(values["simmer_iterations"]),
            values["simmer_temperature"],
            values["simmer_attraction"],
            values["simmer_damping_mult"],
        ),
    )


def _initialize_openord_positions(num_nodes: int) -> torch.Tensor:
    """Create OpenOrd's default initial coordinates.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    torch.Tensor
        Initial positions with shape ``[N, 2]`` and dtype ``float64``. The
        reference ``Node`` constructor sets both coordinates to zero unless a
        ``.real`` file provides fixed coordinates.

    Raises
    ------
    ValueError
        If ``num_nodes`` is negative.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    return torch.zeros((num_nodes, 2), dtype=torch.float64)


@register_op
@dataclass(frozen=True)
class OpenOrdPrepareState(Op):
    """Resolve OpenOrd parameters and build mutable weighted adjacency."""

    name: ClassVar[str] = "openord_prepare_state"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    reads: ClassVar[tuple[str, ...]] = ()
    writes: ClassVar[tuple[str, ...]] = ("extras",)
    requires: ClassVar[tuple[str, ...]] = ()
    config: OpenOrdPrepareStateConfig = field(default_factory=OpenOrdPrepareStateConfig)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Populate OpenOrd phase parameters and prunable adjacency.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs.
        state : SolveState
            Mutable solve state receiving OpenOrd extras.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with resolved OpenOrd parameters and adjacency.
        """
        del ctx
        state.extras["openord_params"] = _resolve_openord_parameters(
            options=self.config.options,
            edge_cut=self.config.edge_cut,
        )
        state.extras["openord_adjacency"] = _build_undirected_adjacency(
            edge_index=problem.edge_index,
            num_nodes=problem.num_nodes,
            edge_weights=problem.edge_weights,
        )
        return state


@register_op
@dataclass(frozen=True)
class OpenOrdInitializePositions(Op):
    """Seed the serial OpenOrd starting coordinates."""

    name: ClassVar[str] = "openord_initialize_positions"
    category: ClassVar[OpCategory] = OpCategory.INIT
    reads: ClassVar[tuple[str, ...]] = ()
    writes: ClassVar[tuple[str, ...]] = ("pos",)
    requires: ClassVar[tuple[str, ...]] = ()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Create deterministic OpenOrd initial coordinates.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs carrying node count and seed.
        state : SolveState
            Mutable solve state receiving coordinates.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with initial ``float64`` positions.
        """
        del ctx
        state.pos = _initialize_openord_positions(num_nodes=problem.num_nodes)
        return state


@register_op
@dataclass(frozen=True)
class OpenOrdPhaseSolve(Op):
    """Run OpenOrd's five-phase annealing and edge-cut loop."""

    name: ClassVar[str] = "openord_phase_solve"
    category: ClassVar[OpCategory] = OpCategory.FORCE
    reads: ClassVar[tuple[str, ...]] = ("pos", "extras")
    writes: ClassVar[tuple[str, ...]] = ("pos", "extras")
    requires: ClassVar[tuple[str, ...]] = ("pos", "extras")
    density_grid: DRLDensityGridConfig = field(default_factory=DRLDensityGridConfig)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Execute OpenOrd's source-matched serial state machine.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs.
        state : SolveState
            Mutable solve state containing initialized positions and extras.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with final unscaled OpenOrd coordinates.
        """
        del ctx
        if state.pos is None:
            raise ValueError("OpenOrdPhaseSolve requires state.pos to be set.")
        state.pos = _run_reference_drl(
            initial_positions=state.pos,
            adjacency=state.extras["openord_adjacency"],
            params=state.extras["openord_params"],
            seed=problem.seed,
            density_config=self.density_grid,
            rng_kind="libc",
        )
        return state


@register_op
@dataclass(frozen=True)
class OpenOrdFinalizePositions(Op):
    """Cast final OpenOrd coordinates to Dagua's output dtype and device."""

    name: ClassVar[str] = "openord_finalize_positions"
    category: ClassVar[OpCategory] = OpCategory.POSTPROCESS
    reads: ClassVar[tuple[str, ...]] = ("pos",)
    writes: ClassVar[tuple[str, ...]] = ("pos",)
    requires: ClassVar[tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Move final coordinates to the requested output device.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs used to resolve the output device.
        state : SolveState
            Mutable solve state containing final coordinates.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with final ``float32`` coordinates on the input device.
        """
        del ctx
        if state.pos is None:
            raise ValueError("OpenOrdFinalizePositions requires state.pos to be set.")
        state.pos = state.pos.to(device=problem.edge_index.device, dtype=torch.float32)
        return state


__all__ = [
    "OpenOrdFinalizePositions",
    "OpenOrdInitializePositions",
    "OpenOrdOptions",
    "OpenOrdPhaseSolve",
    "OpenOrdPrepareState",
    "OpenOrdPrepareStateConfig",
    "_OPENORD_PRESETS",
    "_initialize_openord_positions",
    "_resolve_openord_parameters",
]
