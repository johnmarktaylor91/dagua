"""Checkpoint spine for above-gate scale layout strategies."""

from __future__ import annotations

import ctypes
import gc
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch

from dagua.config import LayoutConfig
from dagua.layout.scale.sketch import TopologySketch

CHECKPOINT_SCHEMA_VERSION = "scale-checkpoint-v1"
CHECKPOINT_ARM_NODE_THRESHOLD = 10_000_000
MANIFEST_FILENAME = "manifest.json"
RNG_STATE_FILENAME = "rng_state.pt"


@dataclass(frozen=True)
class CheckpointTensorRef:
    """A tensor payload recorded in the checkpoint manifest.

    Parameters
    ----------
    name : str
        Logical tensor name within a phase/level payload.
    path : str
        Path relative to the checkpoint root.
    shape : list[int]
        Tensor shape at write time.
    dtype : str
        String representation of the tensor dtype.
    device : str
        Source device at write time. Tensors are persisted on CPU.
    bytes : int
        Estimated uncompressed tensor bytes.
    """

    name: str
    path: str
    shape: list[int]
    dtype: str
    device: str
    bytes: int


@dataclass(frozen=True)
class CheckpointRecord:
    """One completed checkpoint phase/level entry.

    Parameters
    ----------
    phase : str
        Strategy phase, for example ``"hierarchy"`` or ``"refine"``.
    level : int
        Strategy level index. FIELD uses fine-side transition indices.
    completed : bool
        Whether the record is complete and eligible for resume.
    tensor_refs : list[CheckpointTensorRef]
        Tensor payloads belonging to this record.
    telemetry : dict[str, object]
        JSON-friendly phase telemetry.
    rss_before_release : int
        RSS before the caller released offloaded tensors.
    rss_after_release : int
        RSS after garbage collection and ``malloc_trim``.
    released_bytes : int
        Measured RSS decrease. Negative means RSS grew.
    written_at : float
        UNIX timestamp at manifest update.
    """

    phase: str
    level: int
    completed: bool
    tensor_refs: list[CheckpointTensorRef]
    telemetry: dict[str, object] = field(default_factory=dict)
    rss_before_release: int = 0
    rss_after_release: int = 0
    released_bytes: int = 0
    written_at: float = 0.0


@dataclass(frozen=True)
class CheckpointManifest:
    """Manifest-as-code for scale checkpoint compatibility.

    Parameters
    ----------
    schema_version : str
        Checkpoint schema version.
    strategy : str
        Scale strategy that owns the checkpoint.
    graph_fingerprint : str
        Fingerprint of the graph topology and shape.
    config_fingerprint : str
        Fingerprint of resume-relevant layout configuration.
    num_nodes : int
        Finest graph node count.
    num_edges : int
        Finest graph directed edge count.
    checkpoint_root : str
        Absolute checkpoint directory.
    phase : str
        Last completed phase.
    last_completed_level : int
        Last completed strategy level, or ``-1`` when no level is complete.
    records : list[CheckpointRecord]
        Completed checkpoint records.
    rng_state_file : str
        Relative file storing CPU/CUDA RNG streams.
    oom_degrade_intent : dict[str, object] or None
        Write-ahead intent for a planned OOM-safe degraded mode.
    telemetry : dict[str, object]
        Run telemetry and structural invariants.
    created_at : float
        UNIX timestamp when the manifest was created.
    updated_at : float
        UNIX timestamp when the manifest was last updated.
    """

    schema_version: str
    strategy: str
    graph_fingerprint: str
    config_fingerprint: str
    num_nodes: int
    num_edges: int
    checkpoint_root: str
    phase: str
    last_completed_level: int
    records: list[CheckpointRecord]
    rng_state_file: str
    oom_degrade_intent: Optional[dict[str, object]]
    telemetry: dict[str, object]
    created_at: float
    updated_at: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable manifest dictionary.

        Returns
        -------
        dict[str, object]
            Serialized manifest payload.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CheckpointManifest":
        """Build a validated manifest from JSON data.

        Parameters
        ----------
        payload : dict[str, object]
            JSON payload read from ``manifest.json``.

        Returns
        -------
        CheckpointManifest
            Validated manifest instance.
        """
        required = set(cls.__dataclass_fields__)
        missing = required - set(payload)
        if missing:
            raise ValueError(f"checkpoint manifest missing fields: {sorted(missing)}")
        records = [
            CheckpointRecord(
                phase=str(record["phase"]),
                level=int(record["level"]),
                completed=bool(record["completed"]),
                tensor_refs=[
                    CheckpointTensorRef(
                        name=str(ref["name"]),
                        path=str(ref["path"]),
                        shape=[int(value) for value in ref["shape"]],
                        dtype=str(ref["dtype"]),
                        device=str(ref["device"]),
                        bytes=int(ref["bytes"]),
                    )
                    for ref in record["tensor_refs"]
                ],
                telemetry=dict(record.get("telemetry", {})),
                rss_before_release=int(record.get("rss_before_release", 0)),
                rss_after_release=int(record.get("rss_after_release", 0)),
                released_bytes=int(record.get("released_bytes", 0)),
                written_at=float(record.get("written_at", 0.0)),
            )
            for record in payload["records"]
        ]
        manifest = cls(
            schema_version=str(payload["schema_version"]),
            strategy=str(payload["strategy"]),
            graph_fingerprint=str(payload["graph_fingerprint"]),
            config_fingerprint=str(payload["config_fingerprint"]),
            num_nodes=int(payload["num_nodes"]),
            num_edges=int(payload["num_edges"]),
            checkpoint_root=str(payload["checkpoint_root"]),
            phase=str(payload["phase"]),
            last_completed_level=int(payload["last_completed_level"]),
            records=records,
            rng_state_file=str(payload["rng_state_file"]),
            oom_degrade_intent=payload.get("oom_degrade_intent"),
            telemetry=dict(payload.get("telemetry", {})),
            created_at=float(payload["created_at"]),
            updated_at=float(payload["updated_at"]),
        )
        if manifest.schema_version != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported checkpoint schema {manifest.schema_version!r}; "
                f"expected {CHECKPOINT_SCHEMA_VERSION!r}"
            )
        return manifest


class ScaleCheckpointManager:
    """Read/write manager for one scale checkpoint root."""

    def __init__(
        self,
        *,
        root: Path,
        strategy: str,
        graph_fingerprint: str,
        config_fingerprint: str,
        num_nodes: int,
        num_edges: int,
        armed: bool,
    ) -> None:
        """Create a checkpoint manager.

        Parameters
        ----------
        root : pathlib.Path
            Checkpoint root for this graph/config/strategy tuple.
        strategy : str
            Owning scale strategy.
        graph_fingerprint : str
            Topology fingerprint expected on resume.
        config_fingerprint : str
            Resume-relevant config fingerprint.
        num_nodes : int
            Finest node count.
        num_edges : int
            Finest directed edge count.
        armed : bool
            Whether writes and resume are enabled.

        Returns
        -------
        None
            Initializes the manager and loads a matching manifest when present.
        """
        self.root = root.expanduser().resolve()
        self.strategy = str(strategy)
        self.graph_fingerprint = str(graph_fingerprint)
        self.config_fingerprint = str(config_fingerprint)
        self.num_nodes = int(num_nodes)
        self.num_edges = int(num_edges)
        self.armed = bool(armed)
        self.manifest_path = self.root / MANIFEST_FILENAME
        self._manifest: Optional[CheckpointManifest] = None
        if self.armed and self.manifest_path.exists():
            manifest = CheckpointManifest.from_dict(
                json.loads(self.manifest_path.read_text(encoding="utf-8"))
            )
            self._validate_match(manifest)
            self._manifest = manifest

    @classmethod
    def from_config(
        cls,
        config: LayoutConfig,
        sketch: TopologySketch,
        *,
        strategy: str,
    ) -> "ScaleCheckpointManager":
        """Build a manager from layout config and topology sketch.

        Parameters
        ----------
        config : LayoutConfig
            Layout configuration with optional checkpoint parameters.
        sketch : TopologySketch
            Scale topology sketch.
        strategy : str
            Owning strategy name.

        Returns
        -------
        ScaleCheckpointManager
            Armed manager at ``N >= 10M`` or when explicitly requested.
        """
        armed = checkpoint_is_armed(config, int(sketch.num_nodes))
        graph_fp = str(sketch.fingerprint)
        config_fp = config_fingerprint(config, strategy=strategy)
        base_root = checkpoint_base_root(config)
        run_dir = f"{strategy.lower()}-{graph_fp[:16]}-{config_fp[:16]}"
        return cls(
            root=base_root / run_dir,
            strategy=strategy,
            graph_fingerprint=graph_fp,
            config_fingerprint=config_fp,
            num_nodes=int(sketch.num_nodes),
            num_edges=int(sketch.num_edges),
            armed=armed,
        )

    @property
    def manifest(self) -> Optional[CheckpointManifest]:
        """Return the loaded manifest when checkpointing is armed.

        Returns
        -------
        CheckpointManifest or None
            Current manifest state.
        """
        return self._manifest

    def ensure_manifest(self, telemetry: Optional[dict[str, object]] = None) -> CheckpointManifest:
        """Create and persist an empty manifest if one is not loaded.

        Parameters
        ----------
        telemetry : dict[str, object], optional
            Initial run telemetry.

        Returns
        -------
        CheckpointManifest
            Current manifest.
        """
        if not self.armed:
            raise RuntimeError("checkpoint manager is not armed")
        if self._manifest is not None:
            return self._manifest
        now = time.time()
        self.root.mkdir(parents=True, exist_ok=True)
        self._manifest = CheckpointManifest(
            schema_version=CHECKPOINT_SCHEMA_VERSION,
            strategy=self.strategy,
            graph_fingerprint=self.graph_fingerprint,
            config_fingerprint=self.config_fingerprint,
            num_nodes=self.num_nodes,
            num_edges=self.num_edges,
            checkpoint_root=str(self.root),
            phase="created",
            last_completed_level=-1,
            records=[],
            rng_state_file=RNG_STATE_FILENAME,
            oom_degrade_intent=None,
            telemetry=dict(telemetry or {}),
            created_at=now,
            updated_at=now,
        )
        self._write_manifest()
        self._write_rng_state()
        return self._manifest

    def record(
        self,
        *,
        phase: str,
        level: int,
        tensors: dict[str, torch.Tensor],
        telemetry: Optional[dict[str, object]] = None,
        rss_before_release: int = 0,
        rss_after_release: int = 0,
    ) -> None:
        """Persist one completed phase/level tensor group.

        Parameters
        ----------
        phase : str
            Phase name.
        level : int
            Level index.
        tensors : dict[str, torch.Tensor]
            Tensors to save. They are detached and moved to CPU one at a time.
        telemetry : dict[str, object], optional
            Phase telemetry.
        rss_before_release : int, default=0
            RSS before caller-side offload/release.
        rss_after_release : int, default=0
            RSS after caller-side offload/release.

        Returns
        -------
        None
            Tensor files and manifest are written atomically enough for
            kill-9 resume: incomplete records are ignored because the manifest
            is replaced only after all tensors land.
        """
        if not self.armed:
            return
        manifest = self.ensure_manifest()
        tensor_refs: list[CheckpointTensorRef] = []
        phase_dir = self.root / str(phase)
        phase_dir.mkdir(parents=True, exist_ok=True)
        for name, tensor in tensors.items():
            rel_path = Path(str(phase)) / f"level_{int(level):04d}_{name}.pt"
            path = self.root / rel_path
            cpu_tensor = tensor.detach().to(device="cpu").contiguous()
            torch.save(cpu_tensor, path)
            tensor_refs.append(
                CheckpointTensorRef(
                    name=str(name),
                    path=str(rel_path),
                    shape=[int(value) for value in cpu_tensor.shape],
                    dtype=str(cpu_tensor.dtype),
                    device=str(tensor.device),
                    bytes=int(cpu_tensor.numel() * cpu_tensor.element_size()),
                )
            )
            del cpu_tensor
        released = int(rss_before_release) - int(rss_after_release)
        record = CheckpointRecord(
            phase=str(phase),
            level=int(level),
            completed=True,
            tensor_refs=tensor_refs,
            telemetry=dict(telemetry or {}),
            rss_before_release=int(rss_before_release),
            rss_after_release=int(rss_after_release),
            released_bytes=released,
            written_at=time.time(),
        )
        retained = [
            item
            for item in manifest.records
            if not (item.phase == str(phase) and item.level == int(level))
        ]
        self._manifest = CheckpointManifest(
            schema_version=manifest.schema_version,
            strategy=manifest.strategy,
            graph_fingerprint=manifest.graph_fingerprint,
            config_fingerprint=manifest.config_fingerprint,
            num_nodes=manifest.num_nodes,
            num_edges=manifest.num_edges,
            checkpoint_root=manifest.checkpoint_root,
            phase=str(phase),
            last_completed_level=int(level),
            records=[*retained, record],
            rng_state_file=manifest.rng_state_file,
            oom_degrade_intent=manifest.oom_degrade_intent,
            telemetry=manifest.telemetry,
            created_at=manifest.created_at,
            updated_at=time.time(),
        )
        self._write_rng_state()
        self._write_manifest()

    def record_oom_degrade_intent(self, intent: dict[str, object]) -> None:
        """Write a planned OOM-degrade intent before executing it.

        Parameters
        ----------
        intent : dict[str, object]
            JSON-friendly degraded-mode intent.

        Returns
        -------
        None
            Manifest is updated when checkpointing is armed.
        """
        if not self.armed:
            return
        manifest = self.ensure_manifest()
        self._manifest = CheckpointManifest(
            schema_version=manifest.schema_version,
            strategy=manifest.strategy,
            graph_fingerprint=manifest.graph_fingerprint,
            config_fingerprint=manifest.config_fingerprint,
            num_nodes=manifest.num_nodes,
            num_edges=manifest.num_edges,
            checkpoint_root=manifest.checkpoint_root,
            phase=manifest.phase,
            last_completed_level=manifest.last_completed_level,
            records=manifest.records,
            rng_state_file=manifest.rng_state_file,
            oom_degrade_intent=dict(intent),
            telemetry=manifest.telemetry,
            created_at=manifest.created_at,
            updated_at=time.time(),
        )
        self._write_manifest()

    def latest_record(self, phase: str) -> Optional[CheckpointRecord]:
        """Return the latest completed record for a phase.

        Parameters
        ----------
        phase : str
            Phase name to search.

        Returns
        -------
        CheckpointRecord or None
            Highest-level completed record for the phase.
        """
        if self._manifest is None:
            return None
        records = [record for record in self._manifest.records if record.phase == str(phase)]
        if not records:
            return None
        return max(records, key=lambda record: (record.level, record.written_at))

    def record_for(self, phase: str, level: int) -> Optional[CheckpointRecord]:
        """Return a completed record by phase and level.

        Parameters
        ----------
        phase : str
            Phase name.
        level : int
            Level index.

        Returns
        -------
        CheckpointRecord or None
            Matching record when present.
        """
        if self._manifest is None:
            return None
        for record in self._manifest.records:
            if record.phase == str(phase) and record.level == int(level) and record.completed:
                return record
        return None

    def load_tensors(
        self,
        phase: str,
        level: int,
        *,
        map_location: str | torch.device = "cpu",
    ) -> dict[str, torch.Tensor]:
        """Load one phase/level payload lazily.

        Parameters
        ----------
        phase : str
            Phase name.
        level : int
            Level index.
        map_location : str or torch.device, default="cpu"
            Device for loaded tensors.

        Returns
        -------
        dict[str, torch.Tensor]
            Named tensors for exactly one record.
        """
        record = self.record_for(phase, int(level))
        if record is None:
            raise FileNotFoundError(f"no checkpoint record for {phase} level {int(level)}")
        tensors: dict[str, torch.Tensor] = {}
        for ref in record.tensor_refs:
            tensor = torch.load(self.root / ref.path, map_location=map_location, weights_only=True)
            if [int(value) for value in tensor.shape] != ref.shape:
                raise ValueError(f"checkpoint tensor {ref.name!r} shape mismatch")
            tensors[ref.name] = tensor
        return tensors

    def _validate_match(self, manifest: CheckpointManifest) -> None:
        """Validate that an on-disk manifest matches this run.

        Parameters
        ----------
        manifest : CheckpointManifest
            Manifest loaded from disk.

        Returns
        -------
        None
            Raises if the manifest cannot be resumed by this manager.
        """
        if manifest.strategy != self.strategy:
            raise ValueError("checkpoint strategy mismatch")
        if manifest.graph_fingerprint != self.graph_fingerprint:
            raise ValueError("checkpoint graph fingerprint mismatch")
        if manifest.config_fingerprint != self.config_fingerprint:
            raise ValueError("checkpoint config fingerprint mismatch")
        if manifest.num_nodes != self.num_nodes or manifest.num_edges != self.num_edges:
            raise ValueError("checkpoint graph shape mismatch")

    def _write_manifest(self) -> None:
        """Replace ``manifest.json`` with the current manifest.

        Returns
        -------
        None
            Manifest JSON is written with sorted keys.
        """
        if self._manifest is None:
            return
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(self._manifest.to_dict(), sort_keys=True, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, self.manifest_path)

    def _write_rng_state(self) -> None:
        """Persist CPU and CUDA RNG streams.

        Returns
        -------
        None
            RNG state is saved beside the manifest.
        """
        payload: dict[str, object] = {"torch_cpu": torch.get_rng_state()}
        if torch.cuda.is_available():
            payload["torch_cuda_all"] = torch.cuda.get_rng_state_all()
        torch.save(payload, self.root / RNG_STATE_FILENAME)


def checkpoint_is_armed(config: LayoutConfig, num_nodes: int) -> bool:
    """Return whether scale checkpointing is armed for this run.

    Parameters
    ----------
    config : LayoutConfig
        Layout config with optional ``algorithm_params["scale_checkpoint"]``.
    num_nodes : int
        Finest node count.

    Returns
    -------
    bool
        ``True`` at ``N >= 10M`` or when explicitly enabled.
    """
    value = config.algorithm_params.get("scale_checkpoint", None)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on", "force", "explicit"}:
            return True
        if normalized in {"0", "false", "no", "off", "none"}:
            return False
    if value is not None:
        return bool(value)
    return int(num_nodes) >= CHECKPOINT_ARM_NODE_THRESHOLD


def checkpoint_base_root(config: LayoutConfig) -> Path:
    """Return the configured checkpoint base root.

    Parameters
    ----------
    config : LayoutConfig
        Layout config with optional ``algorithm_params["scale_checkpoint_root"]``.

    Returns
    -------
    pathlib.Path
        Base directory used for checkpoint run folders.
    """
    configured = config.algorithm_params.get("scale_checkpoint_root", None)
    if configured is None:
        return Path.cwd() / ".dagua_checkpoints"
    root = Path(str(configured)).expanduser()
    if str(root).startswith("/mnt/locker"):
        raise ValueError("scale checkpoint root must not use /mnt/locker")
    return root


def config_fingerprint(config: LayoutConfig, *, strategy: str) -> str:
    """Fingerprint resume-relevant config fields.

    Parameters
    ----------
    config : LayoutConfig
        Layout configuration.
    strategy : str
        Strategy name folded into the fingerprint.

    Returns
    -------
    str
        SHA-256 hex digest.
    """
    ignored_params = {
        "scale_checkpoint",
        "scale_checkpoint_root",
        "scale_checkpoint_exit_after_level",
    }
    params = {
        str(key): _json_safe(value)
        for key, value in sorted(config.algorithm_params.items())
        if str(key) not in ignored_params
    }
    payload = {
        "strategy": str(strategy),
        "seed": config.seed,
        "device": str(config.device),
        "direction": str(config.direction),
        "node_sep": float(config.node_sep),
        "rank_sep": float(config.rank_sep),
        "params": params,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def rss_bytes() -> int:
    """Return current resident set size in bytes.

    Returns
    -------
    int
        Process RSS, or ``0`` when unavailable.
    """
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        with open("/proc/self/statm", encoding="utf-8") as handle:
            pages = int(handle.read().split()[1])
        return int(pages) * int(page_size)
    except (FileNotFoundError, IndexError, OSError, ValueError):
        return 0


def release_memory() -> tuple[int, int, int]:
    """Force Python/glibc/CUDA release and measure RSS delta.

    Returns
    -------
    tuple[int, int, int]
        ``(before_rss, after_rss, released_bytes)``.
    """
    before = rss_bytes()
    gc.collect()
    _malloc_trim()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    after = rss_bytes()
    return before, after, before - after


def restore_rng_state(manager: ScaleCheckpointManager) -> None:
    """Restore RNG streams from a checkpoint manager when present.

    Parameters
    ----------
    manager : ScaleCheckpointManager
        Checkpoint manager whose root may contain ``rng_state.pt``.

    Returns
    -------
    None
        Missing RNG state is ignored because older partial checkpoints may not
        have reached the first manifest update.
    """
    if not manager.armed:
        return
    path = manager.root / RNG_STATE_FILENAME
    if not path.exists():
        return
    payload = torch.load(path, map_location="cpu", weights_only=True)
    cpu_state = payload.get("torch_cpu")
    if isinstance(cpu_state, torch.Tensor):
        torch.set_rng_state(cpu_state)
    cuda_states = payload.get("torch_cuda_all")
    if torch.cuda.is_available() and isinstance(cuda_states, list):
        torch.cuda.set_rng_state_all(cuda_states)


def _json_safe(value: object) -> object:
    """Convert common config values into JSON-safe primitives.

    Parameters
    ----------
    value : object
        Arbitrary config value.

    Returns
    -------
    object
        JSON-serializable representation.
    """
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in sorted(value.items())}
    return repr(value)


def _malloc_trim() -> None:
    """Ask glibc to return free arenas to the operating system.

    Returns
    -------
    None
        Unsupported platforms are ignored.
    """
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except OSError:
        return


__all__ = [
    "CHECKPOINT_ARM_NODE_THRESHOLD",
    "CHECKPOINT_SCHEMA_VERSION",
    "CheckpointManifest",
    "CheckpointRecord",
    "CheckpointTensorRef",
    "ScaleCheckpointManager",
    "checkpoint_base_root",
    "checkpoint_is_armed",
    "config_fingerprint",
    "release_memory",
    "restore_rng_state",
    "rss_bytes",
]
