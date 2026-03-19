"""Competitor adapters for dagua's classic algorithm reimplementations.

These wrap the educational implementations in ``dagua/layout/classic/`` so
they can be benchmarked alongside the original reference implementations.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

from dagua.eval.competitors.base import (
    CompetitorBase,
    CompetitorResult,
    register,
)

if TYPE_CHECKING:
    from dagua.graph import DaguaGraph


class _ClassicBase(CompetitorBase):
    """Base for classic reimplementation adapters."""

    def _layout_seed(self, seed: Optional[int]) -> int:
        """Resolve the seed for stochastic classic layouts.

        Parameters
        ----------
        seed : int | None
            Explicit benchmark seed override.

        Returns
        -------
        int
            Explicit benchmark seed, or ``42`` when no override is provided.
        """
        return 42 if seed is None else seed

    def available(self) -> bool:
        """Report whether the adapter can run in the current environment.

        Returns
        -------
        bool
            ``True`` because the classic implementations are pure PyTorch.
        """
        return True


@register
class ClassicFR(_ClassicBase):
    """Competitor wrapper for the classic Fruchterman-Reingold reimplementation."""

    name = "classic_fr"
    max_nodes = 50_000

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run the classic Fruchterman-Reingold layout.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Unused compatibility parameter for the competitor interface.
        seed : int | None, default=None
            Random seed for the stochastic solver. ``None`` preserves the
            historical default of ``42``.

        Returns
        -------
        CompetitorResult
            Layout result and runtime information.
        """
        del timeout

        from dagua.layout.classic.fr import layout_fr

        start = time.perf_counter()
        try:
            pos = layout_fr(
                graph.edge_index,
                graph.num_nodes,
                node_sizes=graph.node_sizes,
                steps=200,
                seed=self._layout_seed(seed),
            )
            elapsed = time.perf_counter() - start
            return CompetitorResult(name=self.name, pos=pos, runtime_seconds=elapsed)
        except Exception as exc:
            elapsed = time.perf_counter() - start
            return CompetitorResult(
                name=self.name,
                pos=None,
                runtime_seconds=elapsed,
                error=str(exc),
            )


@register
class ClassicKK(_ClassicBase):
    """Competitor wrapper for the classic Kamada-Kawai reimplementation."""

    name = "classic_kk"
    max_nodes = 5_000

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run the classic Kamada-Kawai layout.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Unused compatibility parameter for the competitor interface.
        seed : int | None, default=None
            Random seed for the stochastic solver. ``None`` preserves the
            historical default of ``42``.

        Returns
        -------
        CompetitorResult
            Layout result and runtime information.
        """
        del timeout

        from dagua.layout.classic.kk import layout_kk

        start = time.perf_counter()
        try:
            pos = layout_kk(
                graph.edge_index,
                graph.num_nodes,
                node_sizes=graph.node_sizes,
                steps=300,
                seed=self._layout_seed(seed),
            )
            elapsed = time.perf_counter() - start
            return CompetitorResult(name=self.name, pos=pos, runtime_seconds=elapsed)
        except Exception as exc:
            elapsed = time.perf_counter() - start
            return CompetitorResult(
                name=self.name,
                pos=None,
                runtime_seconds=elapsed,
                error=str(exc),
            )


@register
class ClassicFA2(_ClassicBase):
    """Competitor wrapper for the classic ForceAtlas2 reimplementation."""

    name = "classic_fa2"
    max_nodes = 50_000

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run the classic ForceAtlas2 layout.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Unused compatibility parameter for the competitor interface.
        seed : int | None, default=None
            Random seed for the stochastic solver. ``None`` preserves the
            historical default of ``42``.

        Returns
        -------
        CompetitorResult
            Layout result and runtime information.
        """
        del timeout

        from dagua.layout.classic.fa2 import layout_fa2

        start = time.perf_counter()
        try:
            pos = layout_fa2(
                graph.edge_index,
                graph.num_nodes,
                node_sizes=graph.node_sizes,
                steps=200,
                seed=self._layout_seed(seed),
            )
            elapsed = time.perf_counter() - start
            return CompetitorResult(name=self.name, pos=pos, runtime_seconds=elapsed)
        except Exception as exc:
            elapsed = time.perf_counter() - start
            return CompetitorResult(
                name=self.name,
                pos=None,
                runtime_seconds=elapsed,
                error=str(exc),
            )


@register
class ClassicStressSGD(_ClassicBase):
    """Competitor wrapper for the classic Stress-SGD reimplementation."""

    name = "classic_stress_sgd"
    max_nodes = 50_000

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run the classic Stress-SGD layout.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Unused compatibility parameter for the competitor interface.
        seed : int | None, default=None
            Random seed for the stochastic solver. ``None`` preserves the
            historical default of ``42``.

        Returns
        -------
        CompetitorResult
            Layout result and runtime information.
        """
        del timeout

        from dagua.layout.classic.stress_sgd import layout_stress_sgd

        start = time.perf_counter()
        try:
            pos = layout_stress_sgd(
                graph.edge_index,
                graph.num_nodes,
                node_sizes=graph.node_sizes,
                steps=300,
                seed=self._layout_seed(seed),
            )
            elapsed = time.perf_counter() - start
            return CompetitorResult(name=self.name, pos=pos, runtime_seconds=elapsed)
        except Exception as exc:
            elapsed = time.perf_counter() - start
            return CompetitorResult(
                name=self.name,
                pos=None,
                runtime_seconds=elapsed,
                error=str(exc),
            )


@register
class ClassicSugiyama(_ClassicBase):
    """Competitor wrapper for the classic Sugiyama reimplementation."""

    name = "classic_sugiyama"
    max_nodes = 50_000

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run the classic Sugiyama layout.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Unused compatibility parameter for the competitor interface.
        seed : int | None, default=None
            Random seed for barycenter tie-breaking. ``None`` preserves the
            historical default of ``42``.

        Returns
        -------
        CompetitorResult
            Layout result and runtime information.
        """
        del timeout

        from dagua.layout.classic.sugiyama import layout_sugiyama

        start = time.perf_counter()
        try:
            pos = layout_sugiyama(
                graph.edge_index,
                graph.num_nodes,
                node_sizes=graph.node_sizes,
                seed=self._layout_seed(seed),
            )
            elapsed = time.perf_counter() - start
            return CompetitorResult(name=self.name, pos=pos, runtime_seconds=elapsed)
        except Exception as exc:
            elapsed = time.perf_counter() - start
            return CompetitorResult(
                name=self.name,
                pos=None,
                runtime_seconds=elapsed,
                error=str(exc),
            )


@register
class ClassicSpectral(_ClassicBase):
    """Competitor wrapper for the classic spectral layout reimplementation."""

    name = "classic_spectral"
    max_nodes = 100_000

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run the classic spectral layout.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Unused compatibility parameter for the competitor interface.
        seed : int | None, default=None
            Random seed for the stochastic solver. ``None`` preserves the
            historical default of ``42``.

        Returns
        -------
        CompetitorResult
            Layout result and runtime information.
        """
        del timeout

        from dagua.layout.classic.spectral import layout_spectral

        start = time.perf_counter()
        try:
            pos = layout_spectral(
                graph.edge_index,
                graph.num_nodes,
                node_sizes=graph.node_sizes,
                seed=self._layout_seed(seed),
            )
            elapsed = time.perf_counter() - start
            return CompetitorResult(name=self.name, pos=pos, runtime_seconds=elapsed)
        except Exception as exc:
            elapsed = time.perf_counter() - start
            return CompetitorResult(
                name=self.name,
                pos=None,
                runtime_seconds=elapsed,
                error=str(exc),
            )


@register
class ClassicPivotMDS(_ClassicBase):
    """Competitor wrapper for the classic pivot-MDS reimplementation."""

    name = "classic_pivot_mds"
    max_nodes = 500_000

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run the classic pivot-MDS layout.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Unused compatibility parameter for the competitor interface.
        seed : int | None, default=None
            Random seed for pivot sampling. ``None`` preserves the historical
            default of ``42``.

        Returns
        -------
        CompetitorResult
            Layout result and runtime information.
        """
        del timeout

        from dagua.layout.classic.pivot_mds import layout_pivot_mds

        start = time.perf_counter()
        try:
            pos = layout_pivot_mds(
                graph.edge_index,
                graph.num_nodes,
                node_sizes=graph.node_sizes,
                n_pivots=50,
                seed=self._layout_seed(seed),
            )
            elapsed = time.perf_counter() - start
            return CompetitorResult(name=self.name, pos=pos, runtime_seconds=elapsed)
        except Exception as exc:
            elapsed = time.perf_counter() - start
            return CompetitorResult(
                name=self.name,
                pos=None,
                runtime_seconds=elapsed,
                error=str(exc),
            )


@register
class ClassicLinLog(_ClassicBase):
    """Competitor wrapper for the classic LinLog reimplementation."""

    name = "classic_linlog"
    max_nodes = 50_000

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run the classic LinLog layout.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Unused compatibility parameter for the competitor interface.
        seed : int | None, default=None
            Random seed for the stochastic solver. ``None`` preserves the
            historical default of ``42``.

        Returns
        -------
        CompetitorResult
            Layout result and runtime information.
        """
        del timeout

        from dagua.layout.classic.linlog import layout_linlog

        start = time.perf_counter()
        try:
            pos = layout_linlog(
                graph.edge_index,
                graph.num_nodes,
                node_sizes=graph.node_sizes,
                steps=300,
                seed=self._layout_seed(seed),
            )
            elapsed = time.perf_counter() - start
            return CompetitorResult(name=self.name, pos=pos, runtime_seconds=elapsed)
        except Exception as exc:
            elapsed = time.perf_counter() - start
            return CompetitorResult(
                name=self.name,
                pos=None,
                runtime_seconds=elapsed,
                error=str(exc),
            )


@register
class ClassicGEM(_ClassicBase):
    """Competitor wrapper for the classic GEM reimplementation."""

    name = "classic_gem"
    max_nodes = 50_000

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run the classic GEM layout.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Unused compatibility parameter for the competitor interface.
        seed : int | None, default=None
            Random seed for the stochastic solver. ``None`` preserves the
            historical default of ``42``.

        Returns
        -------
        CompetitorResult
            Layout result and runtime information.
        """
        del timeout

        from dagua.layout.classic.gem import layout_gem

        start = time.perf_counter()
        try:
            pos = layout_gem(
                graph.edge_index,
                graph.num_nodes,
                node_sizes=graph.node_sizes,
                max_iters=500,
                seed=self._layout_seed(seed),
            )
            elapsed = time.perf_counter() - start
            return CompetitorResult(name=self.name, pos=pos, runtime_seconds=elapsed)
        except Exception as exc:
            elapsed = time.perf_counter() - start
            return CompetitorResult(
                name=self.name,
                pos=None,
                runtime_seconds=elapsed,
                error=str(exc),
            )


@register
class ClassicTsNET(_ClassicBase):
    """Competitor wrapper for the classic tsNET reimplementation."""

    name = "classic_tsnet"
    max_nodes = 10_000

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run the classic tsNET layout.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Unused compatibility parameter for the competitor interface.
        seed : int | None, default=None
            Random seed for the stochastic solver. ``None`` preserves the
            historical default of ``42``.

        Returns
        -------
        CompetitorResult
            Layout result and runtime information.
        """
        del timeout

        from dagua.layout.classic.tsnet import layout_tsnet

        start = time.perf_counter()
        try:
            pos = layout_tsnet(
                graph.edge_index,
                graph.num_nodes,
                node_sizes=graph.node_sizes,
                perplexity=30,
                steps=500,
                seed=self._layout_seed(seed),
            )
            elapsed = time.perf_counter() - start
            return CompetitorResult(name=self.name, pos=pos, runtime_seconds=elapsed)
        except Exception as exc:
            elapsed = time.perf_counter() - start
            return CompetitorResult(
                name=self.name,
                pos=None,
                runtime_seconds=elapsed,
                error=str(exc),
            )


@register
class ClassicMaxentStress(_ClassicBase):
    """Competitor wrapper for the classic maxent-stress reimplementation."""

    name = "classic_maxent_stress"
    max_nodes = 100_000

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run the classic maxent-stress layout.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Unused compatibility parameter for the competitor interface.
        seed : int | None, default=None
            Random seed for the stochastic solver. ``None`` preserves the
            historical default of ``42``.

        Returns
        -------
        CompetitorResult
            Layout result and runtime information.
        """
        del timeout

        from dagua.layout.classic.maxent_stress import layout_maxent_stress

        start = time.perf_counter()
        try:
            pos = layout_maxent_stress(
                graph.edge_index,
                graph.num_nodes,
                node_sizes=graph.node_sizes,
                steps=200,
                alpha=1.0,
                seed=self._layout_seed(seed),
            )
            elapsed = time.perf_counter() - start
            return CompetitorResult(name=self.name, pos=pos, runtime_seconds=elapsed)
        except Exception as exc:
            elapsed = time.perf_counter() - start
            return CompetitorResult(
                name=self.name,
                pos=None,
                runtime_seconds=elapsed,
                error=str(exc),
            )


@register
class ClassicDavidsonHarel(_ClassicBase):
    """Competitor wrapper for the classic Davidson-Harel reimplementation."""

    name = "classic_davidson_harel"
    max_nodes = 50

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run the classic Davidson-Harel layout.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Unused compatibility parameter for the competitor interface.
        seed : int | None, default=None
            Random seed for the stochastic solver. ``None`` preserves the
            historical default of ``42``.

        Returns
        -------
        CompetitorResult
            Layout result and runtime information.
        """
        del timeout

        from dagua.layout.classic.davidson_harel import layout_davidson_harel

        start = time.perf_counter()
        try:
            pos = layout_davidson_harel(
                graph.edge_index,
                graph.num_nodes,
                node_sizes=graph.node_sizes,
                rounds=100,
                seed=self._layout_seed(seed),
            )
            elapsed = time.perf_counter() - start
            return CompetitorResult(name=self.name, pos=pos, runtime_seconds=elapsed)
        except Exception as exc:
            elapsed = time.perf_counter() - start
            return CompetitorResult(
                name=self.name,
                pos=None,
                runtime_seconds=elapsed,
                error=str(exc),
            )


@register
class ClassicFMMM(_ClassicBase):
    """Competitor wrapper for the classic FM^3 reimplementation."""

    name = "classic_fmmm"
    max_nodes = 500_000

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run the classic FM^3 layout.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Unused compatibility parameter for the competitor interface.
        seed : int | None, default=None
            Random seed for the stochastic solver. ``None`` preserves the
            historical default of ``42``.

        Returns
        -------
        CompetitorResult
            Layout result and runtime information.
        """
        del timeout

        from dagua.layout.classic.fmmm import layout_fmmm

        start = time.perf_counter()
        try:
            pos = layout_fmmm(
                graph.edge_index,
                graph.num_nodes,
                node_sizes=graph.node_sizes,
                steps=100,
                seed=self._layout_seed(seed),
            )
            elapsed = time.perf_counter() - start
            return CompetitorResult(name=self.name, pos=pos, runtime_seconds=elapsed)
        except Exception as exc:
            elapsed = time.perf_counter() - start
            return CompetitorResult(
                name=self.name,
                pos=None,
                runtime_seconds=elapsed,
                error=str(exc),
            )
