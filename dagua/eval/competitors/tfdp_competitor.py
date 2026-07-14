"""t-FDP reference competitor adapter."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch

from dagua.eval.competitors.base import CompetitorBase, CompetitorResult, get_runtime_seed, register

if TYPE_CHECKING:
    from dagua.graph import DaguaGraph

_DEFAULT_SEED = 1
_DEFAULT_ITERATIONS = 300
_REFERENCE_ROOT = Path.home() / "tools" / "dagua-refs" / "tfdp"

_TFDP_REFERENCE_SCRIPT = r"""
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, tril

payload = json.loads(sys.stdin.read())
root = Path(payload["reference_root"])
sys.path.insert(0, str(root / "source_code"))
from optimize.Exact import Exact  # noqa: E402
from utils import pivotMDS, scaleByEdge  # noqa: E402

num_nodes = int(payload["num_nodes"])
edges = payload["edges"]
rows = []
cols = []
for source, target in edges:
    source = int(source)
    target = int(target)
    if source == target:
        continue
    rows.extend([source, target])
    cols.extend([target, source])
data = np.ones(len(rows), dtype=np.float32)
full_graph = csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
graph = tril(full_graph)
seed = payload["seed"]
if payload["algo"] != "Exact":
    raise RuntimeError("The adapter currently supports the reference Exact path only.")
if payload["init"] == "pmds":
    if seed:
        np.random.seed(int(seed))
    noise_pos = 0.01 * np.random.randn(num_nodes, 2)
    init = pivotMDS(graph, full_graph.indptr, full_graph.indices, NP=100, hidden_size=2)
    init *= 2 * scaleByEdge(init, full_graph.indptr, full_graph.indices)
    init = 1.0 * init.copy() + noise_pos
elif payload["init"] == "random":
    rng = np.random.default_rng(seed)
    init = rng.standard_normal((num_nodes, 2)).astype(np.float32)
else:
    raise RuntimeError(f"Unsupported t-FDP init: {payload['init']}")
start = time.perf_counter()
with contextlib.redirect_stdout(io.StringIO()):
    pos, _ = Exact(
        init,
        full_graph.indptr.astype(np.int32),
        full_graph.indices.astype(np.int32),
        alpha=float(payload["alpha"]),
        beta=float(payload["beta"]),
        gamma=float(payload["gamma"]),
        max_iter=int(payload["max_iter"]),
        seed=seed,
    )
elapsed = time.perf_counter() - start
sys.stdout.write(
    json.dumps({"positions": np.asarray(pos, dtype=float).tolist(), "runtime": elapsed})
)
"""


def _build_tfdp_input(
    graph: DaguaGraph,
    seed: Optional[int],
    variant_params: Optional[dict[str, Any]],
) -> Dict[str, Any]:
    """Build the JSON payload for the t-FDP reference subprocess.

    Parameters
    ----------
    graph : DaguaGraph
        Source graph.
    seed : int or None
        Runtime seed.
    variant_params : dict[str, Any], optional
        t-FDP reference option overrides.

    Returns
    -------
    dict[str, Any]
        JSON-serializable reference input.
    """
    params = {} if variant_params is None else dict(variant_params)
    edges: List[List[int]] = []
    if graph.edge_index.numel() > 0:
        edges = [[int(source), int(target)] for source, target in graph.edge_index.t().tolist()]
    return {
        "reference_root": str(params.get("reference_root", _REFERENCE_ROOT)),
        "num_nodes": int(graph.num_nodes),
        "edges": edges,
        "seed": seed,
        "init": str(params.get("init", "pmds")),
        "algo": str(params.get("algo", params.get("force_mode", "Exact"))),
        "max_iter": int(params.get("max_iter", params.get("steps", _DEFAULT_ITERATIONS))),
        "alpha": float(params.get("alpha", 0.1)),
        "beta": float(params.get("beta", 8.0)),
        "gamma": float(params.get("gamma", 2.0)),
        "combine": bool(params.get("combine", True)),
    }


@register
class TFDPCompetitor(CompetitorBase):
    """Run the cloned t-FDP reference implementation in a subprocess."""

    name = "tfdp"
    max_nodes = 2_000
    supports_clusters = False
    variant_param_names = frozenset(
        {
            "algo",
            "alpha",
            "beta",
            "combine",
            "force_mode",
            "gamma",
            "init",
            "max_iter",
            "reference_root",
            "steps",
        }
    )

    def available(self) -> bool:
        """Return whether the reference checkout is importable.

        Returns
        -------
        bool
            ``True`` when the durable t-FDP checkout contains ``source_code/tfdp.py``.
        """
        return (_REFERENCE_ROOT / "source_code" / "tfdp.py").exists()

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run the exact t-FDP reference layout.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Maximum subprocess runtime in seconds.
        seed : int or None, default=None
            Deterministic reference seed.

        Returns
        -------
        CompetitorResult
            Position tensor and runtime, or an error.
        """
        return self.layout_with_variant(graph, timeout=timeout, seed=seed, variant_params=None)

    def layout_with_variant(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
        variant_params: Optional[dict[str, Any]] = None,
    ) -> CompetitorResult:
        """Run t-FDP with optional reference parameter overrides.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Maximum subprocess runtime in seconds.
        seed : int or None, default=None
            Deterministic reference seed.
        variant_params : dict[str, Any], optional
            Reference option overrides.

        Returns
        -------
        CompetitorResult
            Position tensor and runtime, or an error.
        """
        resolved_seed = get_runtime_seed(_DEFAULT_SEED) if seed is None else seed
        input_data = json.dumps(_build_tfdp_input(graph, resolved_seed, variant_params))
        start = time.perf_counter()
        try:
            result = subprocess.run(
                [sys.executable, "-c", _TFDP_REFERENCE_SCRIPT],
                input=input_data,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            elapsed = time.perf_counter() - start
            if result.returncode != 0:
                error = (result.stderr or result.stdout or "unknown t-FDP reference failure")[:1000]
                return CompetitorResult(self.name, None, elapsed, error=error)
            data = json.loads(result.stdout)
            pos = torch.tensor(data["positions"], dtype=torch.float64)
            return CompetitorResult(
                name=self.name,
                pos=pos,
                runtime_seconds=float(data.get("runtime", elapsed)),
            )
        except subprocess.TimeoutExpired:
            return CompetitorResult(self.name, None, timeout, error="timeout")
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
            elapsed = time.perf_counter() - start
            return CompetitorResult(self.name, None, elapsed, error=str(exc))
