"""Reference CoRe-GD competitor adapter."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Optional

import torch

from dagua.eval.competitors.base import CompetitorBase, CompetitorResult, register

if TYPE_CHECKING:
    from dagua.graph import DaguaGraph

_DEFAULT_REFERENCE_ROOT = Path.home() / "tools" / "dagua-refs" / "coregd"
_DEFAULT_CONFIG = _DEFAULT_REFERENCE_ROOT / "configs" / "config_rome.json"
_DEFAULT_CHECKPOINT = _DEFAULT_REFERENCE_ROOT / "checkpoints" / "core_rome.pt"


@register
class CoreGDCompetitor(CompetitorBase):
    """Benchmark adapter for the cloned reference CoRe-GD implementation."""

    name = "coregd_reference"
    max_nodes = 100_000

    def available(self) -> bool:
        """Check whether the reference clone and checkpoint are available.

        Returns
        -------
        bool
            ``True`` when the reference repo, config, and checkpoint exist.
        """
        return (
            (_DEFAULT_REFERENCE_ROOT / "neuraldrawer" / "network" / "model.py").exists()
            and _DEFAULT_CONFIG.exists()
            and _DEFAULT_CHECKPOINT.exists()
        )

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run reference CoRe-GD inference for benchmarking.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Compatibility timeout. The in-process adapter does not enforce it.
        seed : int | None, default=None
            Random seed for Dagua's reference-compatible preprocessing.

        Returns
        -------
        CompetitorResult
            Layout positions or an error.
        """
        del timeout
        start = time.perf_counter()
        if not self.available():
            return CompetitorResult(
                name=self.name,
                pos=None,
                runtime_seconds=time.perf_counter() - start,
                error=f"CoRe-GD reference clone/checkpoint missing at {_DEFAULT_REFERENCE_ROOT}.",
            )
        try:
            from dagua.layout.ops.pipelines.coregd import (
                _resolve_coregd_config,
                prepare_coregd_data,
            )

            config = _resolve_coregd_config(
                None,
                seed=seed if seed is not None else 42,
                config_path=str(_DEFAULT_CONFIG),
                checkpoint_path=str(_DEFAULT_CHECKPOINT),
                coarsen=False,
            )
            if str(_DEFAULT_REFERENCE_ROOT) not in sys.path:
                sys.path.insert(0, str(_DEFAULT_REFERENCE_ROOT))
            from neuraldrawer.network.model import get_model

            with _DEFAULT_CONFIG.open("r", encoding="utf-8") as handle:
                reference_config = SimpleNamespace(**json.load(handle))
            model = get_model(reference_config)
            model.load_state_dict(torch.load(_DEFAULT_CHECKPOINT, map_location=torch.device("cpu")))
            model.eval()
            data = prepare_coregd_data(
                graph.edge_index.to(dtype=torch.long),
                graph.num_nodes,
                config,
                torch.device("cpu"),
            )
            with torch.no_grad():
                pos = model(data, int(config.iterations), transform_to_undirected=True)
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
