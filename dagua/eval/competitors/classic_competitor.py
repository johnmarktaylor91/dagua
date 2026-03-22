"""Competitor adapters for dagua's classic algorithm reimplementations.

These wrap the educational implementations in ``dagua/layout/classic/`` so
they can be benchmarked alongside the original reference implementations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Optional

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

    def layout_with_variant(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
        variant_params: Optional[Mapping[str, Any]] = None,
    ) -> CompetitorResult:
        """Run the classic adapter with variant-specific parameters.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Unused compatibility timeout.
        seed : int | None, default=None
            Random seed override for stochastic variants.
        variant_params : Mapping[str, Any] | None, default=None
            Variant-specific parameter overrides applied on top of the
            benchmark defaults for this classic reimplementation.

        Returns
        -------
        CompetitorResult
            Layout result and runtime information.
        """
        del timeout
        spec = _CLASSIC_LAYOUT_SPECS[self.name]
        layout_params = dict(spec.default_params)
        if variant_params is not None:
            layout_params.update(dict(variant_params))
        return _quick_classic(
            self.name,
            spec.import_path,
            spec.function_name,
            graph,
            self._layout_seed(seed),
            **layout_params,
        )


@dataclass(frozen=True)
class _ClassicLayoutSpec:
    """Dispatch metadata for one classic competitor adapter.

    Parameters
    ----------
    import_path : str
        Module path containing the layout function.
    function_name : str
        Function name to call inside ``import_path``.
    default_params : dict[str, Any]
        Benchmark-default keyword arguments for the classic layout.
    """

    import_path: str
    function_name: str
    default_params: dict[str, Any]


_CLASSIC_LAYOUT_SPECS: dict[str, _ClassicLayoutSpec] = {
    "classic_fr": _ClassicLayoutSpec(
        import_path="dagua.layout.classic.fr",
        function_name="layout_fr",
        default_params={"steps": 200},
    ),
    "classic_kk": _ClassicLayoutSpec(
        import_path="dagua.layout.classic.kk",
        function_name="layout_kk",
        default_params={"steps": 300},
    ),
    "classic_fa2": _ClassicLayoutSpec(
        import_path="dagua.layout.classic.fa2",
        function_name="layout_fa2",
        default_params={"steps": 200},
    ),
    "classic_stress_sgd": _ClassicLayoutSpec(
        import_path="dagua.layout.classic.stress_sgd",
        function_name="layout_stress_sgd",
        default_params={"steps": 300},
    ),
    "classic_sugiyama": _ClassicLayoutSpec(
        import_path="dagua.layout.classic.sugiyama",
        function_name="layout_sugiyama",
        default_params={},
    ),
    "classic_spectral": _ClassicLayoutSpec(
        import_path="dagua.layout.classic.spectral",
        function_name="layout_spectral",
        default_params={},
    ),
    "classic_pivot_mds": _ClassicLayoutSpec(
        import_path="dagua.layout.classic.pivot_mds",
        function_name="layout_pivot_mds",
        default_params={"n_pivots": 50},
    ),
    "classic_linlog": _ClassicLayoutSpec(
        import_path="dagua.layout.classic.linlog",
        function_name="layout_linlog",
        default_params={"steps": 300},
    ),
    "classic_gem": _ClassicLayoutSpec(
        import_path="dagua.layout.classic.gem",
        function_name="layout_gem",
        default_params={"max_iters": 500},
    ),
    "classic_tsnet": _ClassicLayoutSpec(
        import_path="dagua.layout.classic.tsnet",
        function_name="layout_tsnet",
        default_params={"perplexity": 30, "steps": 500},
    ),
    "classic_maxent_stress": _ClassicLayoutSpec(
        import_path="dagua.layout.classic.maxent_stress",
        function_name="layout_maxent_stress",
        default_params={"steps": 200, "alpha": 1.0},
    ),
    "classic_davidson_harel": _ClassicLayoutSpec(
        import_path="dagua.layout.classic.davidson_harel",
        function_name="layout_davidson_harel",
        default_params={"rounds": 100},
    ),
    "classic_fmmm": _ClassicLayoutSpec(
        import_path="dagua.layout.classic.fmmm",
        function_name="layout_fmmm",
        default_params={"steps": 100},
    ),
    "classic_graphopt": _ClassicLayoutSpec(
        import_path="dagua.layout.classic.graphopt",
        function_name="layout_graphopt",
        default_params={},
    ),
    "classic_drl": _ClassicLayoutSpec(
        import_path="dagua.layout.classic.drl",
        function_name="layout_drl",
        default_params={},
    ),
    "classic_lgl": _ClassicLayoutSpec(
        import_path="dagua.layout.classic.lgl",
        function_name="layout_lgl",
        default_params={},
    ),
    "classic_sfdp": _ClassicLayoutSpec(
        import_path="dagua.layout.classic.sfdp",
        function_name="layout_sfdp",
        default_params={},
    ),
    "classic_umap": _ClassicLayoutSpec(
        import_path="dagua.layout.classic.umap_layout",
        function_name="layout_umap",
        default_params={},
    ),
    "classic_neulay": _ClassicLayoutSpec(
        import_path="dagua.layout.classic.neulay",
        function_name="layout_neulay",
        default_params={"steps": 20_000, "gcn_steps": 2_000, "use_gcn": True},
    ),
    "classic_sgd2_multi": _ClassicLayoutSpec(
        import_path="dagua.layout.classic.sgd2_multi",
        function_name="layout_sgd2_multi",
        default_params={"criteria": {"stress": 1.0, "ideal_edge_length": 1.0}},
    ),
}


class VariantCompetitor(CompetitorBase):
    """Wrap a base competitor with variant-specific parameters.

    Parameters
    ----------
    base_competitor : CompetitorBase
        Base registered competitor instance to delegate to.
    variant_params : Mapping[str, Any]
        Parameter overrides forwarded to the base competitor's
        ``layout_with_variant`` method.
    name : str
        Synthetic competitor name for this variant.
    display_name : str | None, default=None
        Human-readable label for manifests and reports.
    is_heavy : bool, default=False
        Whether this wrapper belongs to the heavy scheduling lane.
    """

    def __init__(
        self,
        base_competitor: CompetitorBase,
        variant_params: Mapping[str, Any],
        name: str,
        display_name: Optional[str] = None,
        is_heavy: bool = False,
    ) -> None:
        self._base = base_competitor
        self._variant_params = dict(variant_params)
        self.name = name
        self.display_name = name if display_name is None else display_name
        self.max_nodes = base_competitor.max_nodes
        self.supports_clusters = base_competitor.supports_clusters
        self.variant_param_names = base_competitor.variant_param_names
        self.is_heavy = is_heavy

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run the wrapped competitor with this variant's fixed parameters.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Maximum runtime in seconds.
        seed : int | None, default=None
            Optional benchmark seed.

        Returns
        -------
        CompetitorResult
            Layout result from the wrapped competitor.
        """
        return self._base.layout_with_variant(
            graph,
            timeout=timeout,
            seed=seed,
            variant_params=self._variant_params,
        )

    def layout_with_variant(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
        variant_params: Optional[Mapping[str, Any]] = None,
    ) -> CompetitorResult:
        """Run the wrapped competitor with additional overrides.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Maximum runtime in seconds.
        seed : int | None, default=None
            Optional benchmark seed.
        variant_params : Mapping[str, Any] | None, default=None
            Extra overrides merged on top of this wrapper's fixed params.

        Returns
        -------
        CompetitorResult
            Layout result from the wrapped competitor.
        """
        merged_params = dict(self._variant_params)
        if variant_params is not None:
            merged_params.update(dict(variant_params))
        return self._base.layout_with_variant(
            graph,
            timeout=timeout,
            seed=seed,
            variant_params=merged_params,
        )

    def available(self) -> bool:
        """Report whether the wrapped base competitor is available.

        Returns
        -------
        bool
            Availability of the wrapped competitor.
        """
        return self._base.available()


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


# ── New algorithms (March 2026) ──────────────────────────────────────────────


def _quick_classic(
    name: str,
    import_path: str,
    fn_name: str,
    graph: DaguaGraph,
    seed: int,
    **extra_kwargs: Any,
) -> CompetitorResult:
    """Run one classic layout function with shared timing/error handling.

    Parameters
    ----------
    name : str
        Competitor name for the result payload.
    import_path : str
        Module path containing the layout function.
    fn_name : str
        Layout function name inside ``import_path``.
    graph : DaguaGraph
        Graph to lay out.
    seed : int
        Explicit seed forwarded to the classic implementation.
    **extra_kwargs : Any
        Additional layout keyword arguments.

    Returns
    -------
    CompetitorResult
        Layout result and runtime metadata.
    """
    import importlib

    mod = importlib.import_module(import_path)
    fn = getattr(mod, fn_name)
    start = time.perf_counter()
    try:
        pos = fn(
            graph.edge_index,
            graph.num_nodes,
            node_sizes=graph.node_sizes,
            seed=seed,
            **extra_kwargs,
        )
        return CompetitorResult(name=name, pos=pos, runtime_seconds=time.perf_counter() - start)
    except Exception as exc:
        return CompetitorResult(
            name=name, pos=None, runtime_seconds=time.perf_counter() - start, error=str(exc)
        )


@register
class ClassicGraphOpt(_ClassicBase):
    name = "classic_graphopt"
    max_nodes = 20_000

    def layout(
        self, graph: DaguaGraph, timeout: float = 300.0, seed: Optional[int] = None
    ) -> CompetitorResult:
        del timeout
        return _quick_classic(
            self.name,
            "dagua.layout.classic.graphopt",
            "layout_graphopt",
            graph,
            self._layout_seed(seed),
        )


@register
class ClassicDRL(_ClassicBase):
    name = "classic_drl"
    max_nodes = 100_000

    def layout(
        self, graph: DaguaGraph, timeout: float = 300.0, seed: Optional[int] = None
    ) -> CompetitorResult:
        del timeout
        return _quick_classic(
            self.name, "dagua.layout.classic.drl", "layout_drl", graph, self._layout_seed(seed)
        )


@register
class ClassicLGL(_ClassicBase):
    name = "classic_lgl"
    max_nodes = 100_000

    def layout(
        self, graph: DaguaGraph, timeout: float = 300.0, seed: Optional[int] = None
    ) -> CompetitorResult:
        del timeout
        return _quick_classic(
            self.name, "dagua.layout.classic.lgl", "layout_lgl", graph, self._layout_seed(seed)
        )


@register
class ClassicSFDP(_ClassicBase):
    name = "classic_sfdp"
    max_nodes = 100_000

    def layout(
        self, graph: DaguaGraph, timeout: float = 300.0, seed: Optional[int] = None
    ) -> CompetitorResult:
        del timeout
        return _quick_classic(
            self.name, "dagua.layout.classic.sfdp", "layout_sfdp", graph, self._layout_seed(seed)
        )


@register
class ClassicUMAP(_ClassicBase):
    name = "classic_umap"
    max_nodes = 20_000

    def layout(
        self, graph: DaguaGraph, timeout: float = 300.0, seed: Optional[int] = None
    ) -> CompetitorResult:
        del timeout
        return _quick_classic(
            self.name,
            "dagua.layout.classic.umap_layout",
            "layout_umap",
            graph,
            self._layout_seed(seed),
        )


@register
class ClassicNeuLay(_ClassicBase):
    name = "classic_neulay"
    max_nodes = 50_000

    def layout(
        self, graph: DaguaGraph, timeout: float = 300.0, seed: Optional[int] = None
    ) -> CompetitorResult:
        """Run the classic NeuLay reimplementation with the full two-phase setup.

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
        return _quick_classic(
            self.name,
            "dagua.layout.classic.neulay",
            "layout_neulay",
            graph,
            self._layout_seed(seed),
            steps=20_000,
            gcn_steps=2_000,
            use_gcn=True,
        )


@register
class ClassicSGD2Multi(_ClassicBase):
    name = "classic_sgd2_multi"
    max_nodes = 10_000

    def layout(
        self, graph: DaguaGraph, timeout: float = 300.0, seed: Optional[int] = None
    ) -> CompetitorResult:
        """Run the classic multicriteria ``(SGD)^2`` reimplementation.

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
        return _quick_classic(
            self.name,
            "dagua.layout.classic.sgd2_multi",
            "layout_sgd2_multi",
            graph,
            self._layout_seed(seed),
            criteria={"stress": 1.0, "ideal_edge_length": 1.0},
        )
