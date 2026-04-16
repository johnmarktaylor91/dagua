"""(SGD)^2 multicriteria reference competitor adapter.

Runs the original (SGD)^2 code from github.com/tiga1231/graph-drawing
(cloned to /tmp/graph-drawing) as a reference implementation.
"""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from inspect import signature
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional

import torch

from dagua.eval.competitors.base import CompetitorBase, CompetitorResult, register

if TYPE_CHECKING:
    from dagua.graph import DaguaGraph

_SGD2_REPO = Path("/tmp/graph-drawing")


def _sgd2_multi_available() -> bool:
    """Check if the upstream (SGD)^2 code is available."""
    return (_SGD2_REPO / "gd2.py").exists()


@contextmanager
def _compat_reduce_lr_on_plateau() -> Any:
    """Temporarily drop unsupported ``verbose=`` kwargs from ReduceLROnPlateau.

    Yields
    ------
    Any
        Context manager used for compatibility with older upstream GD2 code.
    """
    import torch.optim.lr_scheduler as lr_scheduler

    reduce_lr_cls = lr_scheduler.ReduceLROnPlateau
    if "verbose" in signature(reduce_lr_cls.__init__).parameters:
        yield
        return

    class _ReduceLROnPlateauCompat(reduce_lr_cls):
        """Compatibility wrapper that ignores ``verbose`` when unsupported."""

        def __init__(
            self,
            optimizer: Any,
            *args: Any,
            verbose: bool = False,
            **kwargs: Any,
        ) -> None:
            del verbose
            super().__init__(optimizer, *args, **kwargs)

    lr_scheduler.ReduceLROnPlateau = _ReduceLROnPlateauCompat
    try:
        yield
    finally:
        lr_scheduler.ReduceLROnPlateau = reduce_lr_cls


@contextmanager
def _compat_criteria_patches() -> Any:
    """Patch upstream GD2 criteria functions for modern NetworkX and PyTorch.

    Fixes three classes of bugs in the reference ``criteria.py``:

    1. ``ideal_edge_length`` calls ``random.sample(G.edges, sampleSize)`` but
       modern NetworkX returns an EdgeView which ``random.sample`` rejects
       (requires a sequence).
    2. ``stress`` (and ``ideal_edge_length``) use
       ``torch.tensor([D[i,j] for ...])`` to build tensors from lists of 0-d
       tensors.  This breaks when TorchLens decorates torch functions (the
       wrapped ``__len__`` path rejects 0-d tensors).  Fixed by using
       ``torch.stack`` / ``.item()`` instead.
    3. ``aspect_ratio`` indexes ``singular_values[1]`` unconditionally, which
       crashes when the SVD sample has fewer than 2 points.  Upstream GD2's
       DataLoader leaves a trailing size-1 batch whenever
       ``num_nodes % batch_size == 1`` (e.g. 257 nodes / batch 128 -> [128,
       128, 1]); SVD of a 1x2 matrix returns only 1 singular value.  Fixed by
       short-circuiting to zero loss when the sample has fewer than 2 points.

    Yields
    ------
    Any
        Context manager that patches and restores the criteria functions.
    """
    import importlib
    import random as _random

    import numpy as _np

    repo_str = str(_SGD2_REPO)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    criteria = importlib.import_module("criteria")
    _orig_ideal_edge_length = criteria.ideal_edge_length
    _orig_stress = criteria.stress
    _orig_aspect_ratio = criteria.aspect_ratio

    # -- patched ideal_edge_length -----------------------------------------

    def _patched_ideal_edge_length(
        pos: Any,
        G: Any,
        k2i: Any,
        targetLengths: Any = None,
        sampleSize: Any = None,
        sample: Any = None,
        reduce: str = "mean",
    ) -> Any:
        """ideal_edge_length with EdgeView and 0-d tensor fixes."""
        import torch as _torch

        if targetLengths is None:
            targetLengths = {e: 1 for e in G.edges}

        if sample is not None:
            edges = sample
        elif sampleSize is not None:
            edges = _random.sample(list(G.edges), min(sampleSize, max(len(G.edges), 1)))
        else:
            edges = list(G.edges)
        if not edges:
            return _torch.zeros((), dtype=pos.dtype, device=pos.device)

        sourceIndices, targetIndices = zip(*[[k2i[e0], k2i[e1]] for e0, e1 in edges])
        source = pos[list(sourceIndices), :]
        target = pos[list(targetIndices), :]
        edgeLengths = (source - target).norm(dim=1)
        tl = _torch.tensor([float(targetLengths[e]) for e in edges])

        eu = ((edgeLengths - tl) / tl).pow(2)

        if reduce == "sum":
            return eu.sum()
        return eu.mean()

    # -- patched stress ----------------------------------------------------

    def _patched_stress(
        pos: Any,
        D: Any,
        W: Any,
        sampleSize: Any = None,
        sample: Any = None,
        reduce: str = "mean",
    ) -> Any:
        """stress with torch.stack instead of torch.tensor on 0-d tensors."""
        import torch as _torch
        from torch import nn as _nn

        if sample is None:
            n, m = pos.shape[0], pos.shape[1]
            if sampleSize is not None:
                i0 = _np.random.choice(n, sampleSize)
                i1 = _np.random.choice(n, sampleSize)
                x0 = pos[i0, :]
                x1 = pos[i1, :]
                # Use advanced indexing instead of torch.tensor on 0-d list
                D = D[_torch.from_numpy(i0).long(), _torch.from_numpy(i1).long()]
                W = W[_torch.from_numpy(i0).long(), _torch.from_numpy(i1).long()]
            else:
                x0 = pos.repeat(1, n).view(-1, m)
                x1 = pos.repeat(n, 1)
                D = D.view(-1)
                W = W.view(-1)
        else:
            x0 = pos[sample[:, 0], :]
            x1 = pos[sample[:, 1], :]
            # Use advanced indexing instead of torch.tensor on 0-d list
            D = D[sample[:, 0], sample[:, 1]]
            W = W[sample[:, 0], sample[:, 1]]

        pdist = _nn.PairwiseDistance()(x0, x1)
        res = W * (pdist - D) ** 2

        if reduce == "sum":
            return res.sum()
        return res.mean()

    # -- patched aspect_ratio -----------------------------------------------

    def _patched_aspect_ratio(
        pos: Any,
        target: Any = (1, 1),
        sampleSize: Any = None,
        sample: Any = None,
    ) -> Any:
        """aspect_ratio with guard against degenerate (<2-point) SVD samples."""
        import torch as _torch
        from torch import nn as _nn

        target = list(target)
        if target[0] < target[1]:
            target = target[::-1]

        if sample is not None:
            sample = pos[sample, :]
        elif sampleSize is None or sampleSize == "full":
            sample = pos
        else:
            n = pos.shape[0]
            i = _np.random.choice(n, min(n, sampleSize), replace=False)
            sample = pos[i, :]

        # SVD of an Nx2 matrix returns min(N, 2) singular values. When the
        # DataLoader produces a trailing size-1 batch the upstream code crashes
        # on ``singular_values[1]``; skip the term in that case.
        if sample.shape[0] < 2:
            return _torch.zeros((), dtype=pos.dtype, device=pos.device)

        bce = _nn.BCELoss(reduction="sum")
        singular_values = _torch.svd(sample - sample.mean(dim=0)).S
        return bce(
            singular_values[1] / singular_values[0],
            _torch.tensor(target[1] / target[0]),
        )

    criteria.ideal_edge_length = _patched_ideal_edge_length
    criteria.stress = _patched_stress
    criteria.aspect_ratio = _patched_aspect_ratio
    try:
        yield
    finally:
        criteria.ideal_edge_length = _orig_ideal_edge_length
        criteria.stress = _orig_stress
        criteria.aspect_ratio = _orig_aspect_ratio


@register
class SGD2MultiRef(CompetitorBase):
    """Reference adapter for the (SGD)^2 multicriteria layout engine."""

    name = "sgd2_multi_ref"
    max_nodes = 5_000
    variant_param_names = frozenset(
        {"criteria_weights", "grad_clamp", "max_iter", "optimizer_kwargs", "sample_sizes"}
    )

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run the upstream ``(SGD)^2`` multicriteria engine.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float
            Unused.
        seed : int | None
            Random seed.

        Returns
        -------
        CompetitorResult
            Layout result.
        """
        return self.layout_with_variant(graph, timeout=timeout, seed=seed, variant_params=None)

    def layout_with_variant(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
        variant_params: Optional[Mapping[str, Any]] = None,
    ) -> CompetitorResult:
        """Run the upstream (SGD)^2 multicriteria engine.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float
            Unused.
        seed : int | None
            Random seed.

        Returns
        -------
        CompetitorResult
            Layout result.
        """
        del timeout

        start = time.perf_counter()
        try:
            import numpy as np
            from scipy.sparse import csr_matrix
            from scipy.sparse.csgraph import shortest_path

            # Add repo to path temporarily
            repo_str = str(_SGD2_REPO)
            if repo_str not in sys.path:
                sys.path.insert(0, repo_str)

            from gd2 import GD2  # type: ignore[import-untyped]

            n = graph.num_nodes
            ei = graph.edge_index.cpu().numpy()

            # Build symmetric adjacency
            rows = np.concatenate([ei[0], ei[1]])
            cols = np.concatenate([ei[1], ei[0]])
            data = np.ones(len(rows))
            adj = csr_matrix((data, (rows, cols)), shape=(n, n))

            # Shortest paths
            dist = shortest_path(adj, directed=False)
            if not np.all(np.isfinite(dist)):
                return CompetitorResult(
                    name=self.name,
                    pos=None,
                    runtime_seconds=time.perf_counter() - start,
                    error="graph is disconnected",
                )

            # Build networkx graph for GD2
            import networkx as nx

            G_nx = nx.Graph()
            G_nx.add_nodes_from(range(n))
            for i in range(ei.shape[1]):
                s, t = int(ei[0, i]), int(ei[1, i])
                # networkx.Graph is undirected and dedups (s, t) / (t, s)
                # automatically; we only need to skip self-loops.
                if s != t:
                    G_nx.add_edge(s, t)

            # Run GD2 with stress criterion
            if seed is not None:
                torch.manual_seed(seed)
                np.random.seed(seed)

            optimize_kwargs: dict[str, Any] = {
                "criteria_weights": {"stress": 1.0},
                "max_iter": 2000,
                "optimizer_kwargs": {"lr": 0.01},
            }
            if variant_params is not None:
                for key, value in dict(variant_params).items():
                    if key == "optimizer_kwargs":
                        merged_optimizer_kwargs = dict(optimize_kwargs["optimizer_kwargs"])
                        merged_optimizer_kwargs.update(dict(value))
                        optimize_kwargs["optimizer_kwargs"] = merged_optimizer_kwargs
                    else:
                        optimize_kwargs[key] = value

            criteria_weights = dict(optimize_kwargs["criteria_weights"])
            optimize_kwargs["criteria_weights"] = criteria_weights
            optimize_kwargs.setdefault(
                "sample_sizes",
                {criterion_name: 128 for criterion_name in criteria_weights},
            )

            # Disable visualization -- the upstream optimize() has a broken
            # V.plot() call that is missing required positional args, and we
            # never want interactive plots during benchmarks anyway.
            optimize_kwargs["vis_interval"] = 0
            # Evaluate once at the very end so get_result_dict() can access
            # qualities_by_time[-1] without crashing.
            max_iter = int(optimize_kwargs.get("max_iter", 2000))
            optimize_kwargs.setdefault("evaluate_interval", max_iter)
            optimize_kwargs.setdefault("evaluate", {"stress"})
            optimize_kwargs.setdefault("grad_clamp", 5.0)

            gd2 = GD2(G_nx)
            # Strip crossings criteria when there are no non-incident edge
            # pairs to sample. Upstream GD2 builds a DataLoader over this list
            # for crossing-based criteria, and PyTorch rejects num_samples=0.
            if not gd2.non_incident_edge_pairs:
                crossings_keys = {"crossings", "crossing_angle_maximization"}
                criteria_weights = {
                    key: value
                    for key, value in optimize_kwargs["criteria_weights"].items()
                    if key not in crossings_keys
                }
                sample_sizes = {
                    key: value
                    for key, value in optimize_kwargs["sample_sizes"].items()
                    if key not in crossings_keys
                }
                if not criteria_weights:
                    # Fall back to stress-only so the wrapper still returns a layout.
                    criteria_weights = {"stress": 1.0}
                    sample_sizes = {"stress": 128}
                optimize_kwargs["criteria_weights"] = criteria_weights
                optimize_kwargs["sample_sizes"] = sample_sizes
            with _compat_reduce_lr_on_plateau(), _compat_criteria_patches():
                gd2.optimize(**optimize_kwargs)
            if torch.isnan(gd2.pos).any():
                return CompetitorResult(
                    name=self.name,
                    pos=None,
                    runtime_seconds=time.perf_counter() - start,
                    error="optimization diverged (NaN positions)",
                )
            positions = gd2.pos.detach().cpu()

            elapsed = time.perf_counter() - start
            return CompetitorResult(name=self.name, pos=positions, runtime_seconds=elapsed)
        except Exception as exc:
            elapsed = time.perf_counter() - start
            # MemoryError and some C-level exceptions have empty str(); fall back
            # to the exception class name so the benchmark doesn't lose the signal.
            message = str(exc) or type(exc).__name__
            return CompetitorResult(
                name=self.name,
                pos=None,
                runtime_seconds=elapsed,
                error=message,
            )

    def available(self) -> bool:
        """Check if the upstream code is cloned."""
        return _sgd2_multi_available()
