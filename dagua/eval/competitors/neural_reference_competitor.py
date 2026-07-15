"""Reference SmartGD and DeepGD competitor adapters."""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Optional

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

from dagua.eval.competitors.base import CompetitorBase, CompetitorResult, register  # noqa: E402
from dagua.layout.ops.pipelines.smartgd import prepare_smartgd_data  # noqa: E402

if TYPE_CHECKING:
    from dagua.graph import DaguaGraph

_REFERENCE_ROOT = Path.home() / "tools" / "dagua-refs"


@dataclass(frozen=True)
class NeuralReferenceSpec:
    """Metadata for one neural reference layout adapter.

    Parameters
    ----------
    name : str
        Registered competitor name.
    package_name : str
        Upstream package import name, ``"smartgd"`` or ``"deepgd"``.
    checkpoint_name : str
        Checkpoint filename under the upstream clone root.
    max_nodes : int
        Graph-size cap for benchmark scheduling.
    """

    name: str
    package_name: str
    checkpoint_name: str
    max_nodes: int

    @property
    def root(self) -> Path:
        """Return the durable upstream clone root.

        Returns
        -------
        pathlib.Path
            Local reference source directory.
        """
        return _REFERENCE_ROOT / self.package_name

    @property
    def checkpoint(self) -> Path:
        """Return the released checkpoint path.

        Returns
        -------
        pathlib.Path
            Local checkpoint path.
        """
        return self.root / self.checkpoint_name


@contextmanager
def _deterministic_torch(seed: int) -> Iterator[None]:
    """Run a block with deterministic PyTorch settings.

    Parameters
    ----------
    seed : int
        Seed for CPU and CUDA random generators.

    Yields
    ------
    None
        Control while deterministic algorithm mode is enabled.
    """
    previous_enabled = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    previous_cudnn_deterministic = torch.backends.cudnn.deterministic
    previous_cudnn_benchmark = torch.backends.cudnn.benchmark

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous_enabled, warn_only=previous_warn_only)
        torch.backends.cudnn.deterministic = previous_cudnn_deterministic
        torch.backends.cudnn.benchmark = previous_cudnn_benchmark


def _reference_device() -> torch.device:
    """Return the preferred deterministic inference device.

    Returns
    -------
    torch.device
        CUDA when available, otherwise CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _NeuralReferenceCompetitor(CompetitorBase):
    """Benchmark adapter for released SmartGD/DeepGD reference checkpoints."""

    spec: NeuralReferenceSpec
    supports_clusters = False

    def __init__(self) -> None:
        """Initialize registration metadata from the class spec."""
        self.name = self.spec.name
        self.max_nodes = self.spec.max_nodes

    def available(self) -> bool:
        """Check whether the reference clone and checkpoint are available.

        Returns
        -------
        bool
            ``True`` when the upstream package and released checkpoint exist.
        """
        return (
            self.spec.root / self.spec.package_name / "model" / "__init__.py"
        ).exists() and self.spec.checkpoint.exists()

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run released reference-network inference for one graph.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Compatibility timeout. The in-process adapter does not enforce it.
        seed : int | None, default=None
            Deterministic preprocessing and inference seed.

        Returns
        -------
        CompetitorResult
            Layout positions or an error payload.
        """
        del timeout
        start = time.perf_counter()
        if not self.available():
            return CompetitorResult(
                name=self.name,
                pos=None,
                runtime_seconds=time.perf_counter() - start,
                error=f"{self.spec.package_name} reference missing at {self.spec.root}.",
            )

        resolved_seed = 42 if seed is None else int(seed)
        device = _reference_device()
        try:
            if str(self.spec.root) not in sys.path:
                sys.path.insert(0, str(self.spec.root))
            module = __import__(f"{self.spec.package_name}.model", fromlist=["Generator"])
            generator_cls = module.Generator
            params_cls = generator_cls.Params

            with _deterministic_torch(resolved_seed):
                model = generator_cls(
                    params=params_cls(
                        num_blocks=11,
                        block_depth=3,
                        block_width=8,
                        block_output_dim=8,
                        edge_net_depth=2,
                        edge_net_width=16,
                        edge_attr_dim=2,
                        node_attr_dim=2,
                    )
                ).to(device)
                state_dict = torch.load(self.spec.checkpoint, map_location=device)
                model.load_state_dict(state_dict)
                model.eval()
                prepared = prepare_smartgd_data(
                    edge_index=graph.edge_index.to(dtype=torch.long),
                    num_nodes=graph.num_nodes,
                    init_pos=None,
                    device=device,
                    seed=resolved_seed,
                )
                with torch.no_grad():
                    pos = model(*prepared)
            return CompetitorResult(
                name=self.name,
                pos=pos.detach().cpu(),
                runtime_seconds=time.perf_counter() - start,
            )
        except Exception as exc:
            return CompetitorResult(
                name=self.name,
                pos=None,
                runtime_seconds=time.perf_counter() - start,
                error=str(exc),
            )


@register
class SmartGDReference(_NeuralReferenceCompetitor):
    """Benchmark adapter for the released SmartGD stress generator."""

    spec = NeuralReferenceSpec(
        name="smartgd_reference",
        package_name="smartgd",
        checkpoint_name="generator_stress_only.pt",
        max_nodes=5_000,
    )


@register
class DeepGDReference(_NeuralReferenceCompetitor):
    """Benchmark adapter for the released DeepGD stress generator."""

    spec = NeuralReferenceSpec(
        name="deepgd_reference",
        package_name="deepgd",
        checkpoint_name="model_stress_only.pt",
        max_nodes=5_000,
    )
