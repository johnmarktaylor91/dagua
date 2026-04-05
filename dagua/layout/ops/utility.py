"""Utility operations for checkpointing, memory hygiene, and progress."""

from __future__ import annotations

import ctypes
import gc
import json
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Tuple, Union

import torch

from dagua.layout.ops.base import Op
from dagua.layout.ops.state import HierarchyLevel, LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op

_HIERARCHY_TENSOR_FIELDS: Tuple[str, ...] = (
    "edge_index",
    "node_sizes",
    "fine_to_coarse",
    "fine_layer_assignments",
    "coarse_layer_assignments",
    "cluster_ids",
)


def _atomic_torch_save(path: Path, payload: object) -> None:
    """Write a torch-serializable payload atomically.

    Parameters
    ----------
    path : Path
        Destination file path.
    payload : object
        Pickle-serializable object for ``torch.save``.

    Returns
    -------
    None
        The payload is written to disk atomically.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.stem}_",
        suffix=".tmp",
        delete=False,
    ) as handle:
        tmp_path = Path(handle.name)
    try:
        torch.save(payload, tmp_path)
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _atomic_json_write(path: Path, payload: Dict[str, object]) -> None:
    """Write a JSON payload atomically.

    Parameters
    ----------
    path : Path
        Destination JSON path.
    payload : dict[str, object]
        Serializable progress payload.

    Returns
    -------
    None
        The payload is written to disk atomically.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


def _resolve_checkpoint_path(
    configured_path: Optional[Union[str, Path]],
    ctx: RuntimeContext,
    state: SolveState,
) -> Path:
    """Resolve the checkpoint file path for a state snapshot.

    Parameters
    ----------
    configured_path : str or Path, optional
        Explicit checkpoint path.
    ctx : RuntimeContext
        Execution infrastructure that may define a checkpoint directory.
    state : SolveState
        Mutable solve state whose step is used in auto-generated names.

    Returns
    -------
    Path
        Absolute checkpoint path.
    """
    if configured_path is not None:
        return Path(configured_path).expanduser().resolve()

    base_dir = ctx.memory.checkpoint_dir
    if base_dir is None:
        base_dir = Path.cwd() / "checkpoints"
    return Path(base_dir).expanduser().resolve() / f"solve_state_step_{state.step:06d}.pt"


def _resolve_offload_dir(ctx: RuntimeContext) -> Path:
    """Return the directory used for hierarchy tensor offload.

    Parameters
    ----------
    ctx : RuntimeContext
        Execution infrastructure carrying the memory policy.

    Returns
    -------
    Path
        Absolute directory where hierarchy levels are serialized.
    """
    if ctx.memory.offload_dir is None:
        ctx.memory.offload_dir = Path(
            tempfile.mkdtemp(prefix="dagua_hierarchy_", dir=str(Path.cwd()))
        ).resolve()
    else:
        ctx.memory.offload_dir = Path(ctx.memory.offload_dir).expanduser().resolve()
        ctx.memory.offload_dir.mkdir(parents=True, exist_ok=True)
    return ctx.memory.offload_dir


def _hierarchy_payload(level: HierarchyLevel) -> Dict[str, torch.Tensor]:
    """Collect the resident tensor payload for one hierarchy level.

    Parameters
    ----------
    level : HierarchyLevel
        Hierarchy level that may hold resident tensors.

    Returns
    -------
    dict[str, torch.Tensor]
        Tensor payload keyed by field name.
    """
    payload: Dict[str, torch.Tensor] = {}
    for field_name in _HIERARCHY_TENSOR_FIELDS:
        value = getattr(level, field_name)
        if isinstance(value, torch.Tensor):
            payload[field_name] = value.detach().cpu()
    return payload


def _restore_hierarchy_payload(level: HierarchyLevel, payload: Dict[str, Any]) -> None:
    """Restore serialized tensor payload into a hierarchy level.

    Parameters
    ----------
    level : HierarchyLevel
        Hierarchy level that should be repopulated.
    payload : dict[str, Any]
        Tensor payload loaded from disk.

    Returns
    -------
    None
        The level is updated in place.
    """
    for field_name in level.payload_fields or tuple(payload.keys()):
        if field_name in payload:
            setattr(level, field_name, payload[field_name])


def _try_malloc_trim() -> bool:
    """Attempt to return freed heap pages to the OS.

    Returns
    -------
    bool
        ``True`` when ``malloc_trim(0)`` was called successfully.
    """
    try:
        return bool(ctypes.CDLL("libc.so.6").malloc_trim(0))
    except (AttributeError, OSError):
        return False


def _current_vram_usage_mb() -> float:
    """Return the current process CUDA allocation in megabytes.

    Returns
    -------
    float
        Current CUDA allocation in megabytes, or ``0.0`` on CPU-only runs.
    """
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.memory_allocated() / 1024**2)


@dataclass(frozen=True)
class CheckpointConfig:
    """Configuration for :class:`Checkpoint`.

    Parameters
    ----------
    path : str or Path, optional
        Explicit checkpoint path. When omitted, the op uses
        ``ctx.memory.checkpoint_dir`` or ``./checkpoints``.
    """

    path: Optional[Union[str, Path]] = None


@register_op
@dataclass(frozen=True)
class Checkpoint(Op):
    """Serialize the current solve state to disk."""

    config: CheckpointConfig = CheckpointConfig()

    name: ClassVar[str] = "checkpoint"
    category: ClassVar[OpCategory] = OpCategory.UTILITY
    reads: ClassVar[Tuple[str, ...]] = ("*",)
    writes: ClassVar[Tuple[str, ...]] = ("extras.checkpoint_path",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Persist the current solve state to a checkpoint file.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state to serialize.
        ctx : RuntimeContext
            Execution infrastructure carrying checkpoint policy.

        Returns
        -------
        SolveState
            Unmodified state with ``extras['checkpoint_path']`` recorded.
        """
        del problem
        path = _resolve_checkpoint_path(self.config.path, ctx, state)
        _atomic_torch_save(path, state)
        state.extras["checkpoint_path"] = str(path)
        ctx.trace_sink.log(f"{ctx.log_prefix} checkpoint saved to {path}")
        return state


@register_op
class DiskOffload(Op):
    """Offload resident hierarchy tensors to disk and free host memory."""

    name: ClassVar[str] = "disk_offload"
    category: ClassVar[OpCategory] = OpCategory.UTILITY
    reads: ClassVar[Tuple[str, ...]] = ("hierarchy",)
    writes: ClassVar[Tuple[str, ...]] = ("hierarchy", "extras.offload_dir")

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Serialize resident hierarchy tensors to disk.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state that may carry a hierarchy.
        ctx : RuntimeContext
            Execution infrastructure carrying offload policy.

        Returns
        -------
        SolveState
            State with hierarchy tensors replaced by on-disk payloads.
        """
        del problem
        if not state.hierarchy:
            return state

        offload_dir = _resolve_offload_dir(ctx)
        state.extras["offload_dir"] = str(offload_dir)
        for level_index, level in enumerate(state.hierarchy):
            payload = _hierarchy_payload(level)
            if not payload:
                continue
            path = offload_dir / f"level_{level_index:02d}.pt"
            _atomic_torch_save(path, payload)
            # Clear only tensors we just serialized so non-tensor bookkeeping
            # remains resident on the hierarchy objects.
            for field_name in payload:
                setattr(level, field_name, None)
            level.offload_path = path
            level.offload_dir = offload_dir
            level.payload_fields = tuple(payload.keys())

        ctx.trace_sink.log(f"{ctx.log_prefix} hierarchy offloaded to {offload_dir}")
        return state


@register_op
class DiskReload(Op):
    """Reload offloaded hierarchy tensors back into memory."""

    name: ClassVar[str] = "disk_reload"
    category: ClassVar[OpCategory] = OpCategory.UTILITY
    reads: ClassVar[Tuple[str, ...]] = ("hierarchy",)
    writes: ClassVar[Tuple[str, ...]] = ("hierarchy",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Reload offloaded hierarchy level payloads from disk.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state that may carry offloaded hierarchy levels.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        SolveState
            State with resident hierarchy tensors restored where possible.
        """
        del problem
        if not state.hierarchy:
            return state

        for level in state.hierarchy:
            if level.offload_path is None:
                continue
            payload = torch.load(level.offload_path, map_location="cpu")
            _restore_hierarchy_payload(level, payload)

        ctx.trace_sink.log(f"{ctx.log_prefix} hierarchy reloaded from disk")
        return state


@register_op
class GarbageCollect(Op):
    """Run Python, CUDA, and libc memory cleanup hooks."""

    name: ClassVar[str] = "garbage_collect"
    category: ClassVar[OpCategory] = OpCategory.UTILITY
    reads: ClassVar[Tuple[str, ...]] = ()
    writes: ClassVar[Tuple[str, ...]] = ("extras.gc_stats",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Trigger host and device memory cleanup best-effort hooks.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        SolveState
            State with cleanup statistics stored in ``extras``.
        """
        del problem
        collected = gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        trimmed = _try_malloc_trim()
        state.extras["gc_stats"] = {"collected": collected, "malloc_trim": trimmed}
        ctx.trace_sink.log(f"{ctx.log_prefix} garbage collection collected {collected} objects")
        return state


@dataclass(frozen=True)
class VRAMGuardConfig:
    """Configuration for :class:`VRAMGuard`.

    Parameters
    ----------
    budget_fraction : float, default=0.85
        Maximum allowed fraction of total VRAM already in use before the op
        raises ``MemoryError``.
    """

    budget_fraction: float = 0.85


@register_op
@dataclass(frozen=True)
class VRAMGuard(Op):
    """Abort GPU execution early when the active VRAM budget is exhausted."""

    config: VRAMGuardConfig = VRAMGuardConfig()

    name: ClassVar[str] = "vram_guard"
    category: ClassVar[OpCategory] = OpCategory.UTILITY
    reads: ClassVar[Tuple[str, ...]] = ()
    writes: ClassVar[Tuple[str, ...]] = ("extras.vram_guard",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Check the current VRAM budget and raise on unsafe GPU pressure.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure describing the active device plan.

        Returns
        -------
        SolveState
            Unmodified state when the guard passes.

        Raises
        ------
        ValueError
            If ``budget_fraction`` is outside ``(0, 1]``.
        MemoryError
            If current GPU usage already exceeds the configured budget.
        """
        del problem
        budget_fraction = float(self.config.budget_fraction)
        if budget_fraction <= 0.0 or budget_fraction > 1.0:
            raise ValueError("VRAMGuard budget_fraction must be in the interval (0, 1].")

        use_cuda = ctx.plan.device == "cuda" and torch.cuda.is_available()
        state.extras["vram_guard"] = {"checked": use_cuda, "budget_fraction": budget_fraction}
        if not use_cuda:
            return state

        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            used_bytes = int(total_bytes - free_bytes)
        except RuntimeError:
            # Older drivers occasionally fail `mem_get_info`; reserved/allocated
            # memory is the closest stable fallback for a guardrail check.
            total_bytes = int(
                torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
            )
            used_bytes = int(max(torch.cuda.memory_reserved(), torch.cuda.memory_allocated()))

        budget_bytes = int(total_bytes * budget_fraction)
        state.extras["vram_guard"].update(
            {
                "used_bytes": used_bytes,
                "total_bytes": int(total_bytes),
                "budget_bytes": budget_bytes,
            }
        )
        if used_bytes > budget_bytes:
            raise MemoryError(
                f"VRAMGuard blocked execution: {used_bytes} bytes in use exceeds "
                f"{budget_bytes} byte budget."
            )
        return state


@dataclass(frozen=True)
class ProgressReportConfig:
    """Configuration for :class:`ProgressReport`.

    Parameters
    ----------
    file : str or Path, optional
        Explicit output path for ``progress.json``.
    interval : int, default=10
        Write cadence in solver steps. Final or converged states always write.
    """

    file: Optional[Union[str, Path]] = None
    interval: int = 10


@register_op
@dataclass(frozen=True)
class ProgressReport(Op):
    """Write a compact progress snapshot for external polling."""

    config: ProgressReportConfig = ProgressReportConfig()

    name: ClassVar[str] = "progress_report"
    category: ClassVar[OpCategory] = OpCategory.UTILITY
    reads: ClassVar[Tuple[str, ...]] = ("step", "total_steps", "prev_loss", "converged")
    writes: ClassVar[Tuple[str, ...]] = ()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Write ``progress.json`` when the configured cadence is reached.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state carrying progress counters.
        ctx : RuntimeContext
            Execution infrastructure that may provide a default progress path.

        Returns
        -------
        SolveState
            Unmodified state after any side-effectful progress write.

        Raises
        ------
        ValueError
            If ``interval`` is not positive.
        """
        interval = int(self.config.interval)
        if interval <= 0:
            raise ValueError("ProgressReport interval must be positive.")

        should_write = (
            state.step % interval == 0
            or state.converged
            or (state.total_steps > 0 and state.step >= state.total_steps)
        )
        if not should_write:
            return state

        path = self.config.file or ctx.progress_file or (Path.cwd() / "progress.json")
        resolved_path = Path(path).expanduser().resolve()
        ctx.progress_file = resolved_path
        payload: Dict[str, object] = {
            "step": state.step,
            "total_steps": state.total_steps,
            "step_pct": state.step / max(state.total_steps, 1),
            "loss": float(state.prev_loss),
            "converged": bool(state.converged),
            "num_nodes": int(problem.num_nodes),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "vram_mb": round(_current_vram_usage_mb(), 1),
        }
        _atomic_json_write(resolved_path, payload)
        ctx.trace_sink.log(f"{ctx.log_prefix} progress written to {resolved_path}")
        return state


@register_op
class Timer(Op):
    """Measure and log the execution time of an optional wrapped operation."""

    name: ClassVar[str] = "timer"
    category: ClassVar[OpCategory] = OpCategory.UTILITY
    reads: ClassVar[Tuple[str, ...]] = ()
    writes: ClassVar[Tuple[str, ...]] = ("extras.op_timings", "extras.last_timing_seconds")

    def __init__(self, op: Optional[Op] = None, label: Optional[str] = None) -> None:
        """Store the wrapped op and optional timing label.

        Parameters
        ----------
        op : Op, optional
            Inner op to execute and time. When omitted, the timer measures only
            its own no-op bookkeeping overhead.
        label : str, optional
            Explicit label used for logging and metrics storage.

        Returns
        -------
        None
            The timer stores the supplied op and label.
        """
        self.op = op
        self.label = label or (op.name if op is not None else self.name)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Execute the wrapped op, record elapsed time, and log it.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure whose trace sink receives the timing log.

        Returns
        -------
        SolveState
            Resulting solve state after the wrapped op, if any, plus timing
            metadata in ``extras``.
        """
        start = time.perf_counter()
        result = self.op.apply(problem, state, ctx) if self.op is not None else state
        elapsed = time.perf_counter() - start
        timings = result.extras.setdefault("op_timings", {})
        entries = timings.setdefault(self.label, [])
        if not isinstance(entries, list):
            entries = timings[self.label] = [entries]
        entries.append(elapsed)
        result.extras["last_timing_seconds"] = elapsed
        ctx.trace_sink.log(f"{ctx.log_prefix} {self.label} took {elapsed:.6f}s")
        return result
