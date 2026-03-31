"""Competitor adapters for dagua's classic algorithm reimplementations.

These wrap the educational implementations in ``dagua/layout/classic/`` so
they can be benchmarked alongside the original reference implementations.
"""

from __future__ import annotations

import time
import warnings
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
            _warn_on_unrecognized_variant_params(
                competitor_name=self.name,
                variant_params=variant_params,
                variant_param_names=self.variant_param_names,
            )
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


def _warn_on_unrecognized_variant_params(
    competitor_name: str,
    variant_params: Mapping[str, Any],
    variant_param_names: frozenset[str],
) -> None:
    """Warn when variant overrides include unsupported keyword names.

    Parameters
    ----------
    competitor_name : str
        Name of the competitor receiving the override parameters.
    variant_params : Mapping[str, Any]
        Requested variant overrides.
    variant_param_names : frozenset[str]
        Supported override names for the competitor.

    Returns
    -------
    None
        Emits a ``UserWarning`` when unsupported parameter names are present.
    """
    unrecognized_param_names = sorted(set(variant_params) - set(variant_param_names))
    if not unrecognized_param_names:
        return
    warnings.warn(
        (
            f"{competitor_name} received unrecognized variant params: "
            f"{', '.join(unrecognized_param_names)}"
        ),
        UserWarning,
        stacklevel=3,
    )


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
        default_params={"steps": 200, "barnes_hut": True, "barnes_hut_theta": 1.2},
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
    "classic_classical_mds": _ClassicLayoutSpec(
        import_path="dagua.layout.classic.classical_mds",
        function_name="layout_classical_mds",
        default_params={},
    ),
    "classic_stress_maj": _ClassicLayoutSpec(
        import_path="dagua.layout.classic.stress_majorization",
        function_name="layout_stress_majorization",
        default_params={"iterations": 200},
    ),
    "classic_pivot_mds": _ClassicLayoutSpec(
        import_path="dagua.layout.classic.pivot_mds",
        function_name="layout_pivot_mds",
        default_params={"n_pivots": 50},
    ),
    "classic_rt": _ClassicLayoutSpec(
        import_path="dagua.layout.classic.reingold_tilford",
        function_name="layout_reingold_tilford",
        default_params={},
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
        default_params={
            "steps": 20_000,
            "gcn_steps": 2_000,
            "use_gcn": True,
            "lr": 0.1,
            "radius": 0.4,
        },
    ),
    "classic_sgd2_multi": _ClassicLayoutSpec(
        import_path="dagua.layout.classic.sgd2_multi",
        function_name="layout_sgd2_multi",
        default_params={"criteria": {"stress": 1.0, "ideal_edge_length": 1.0}, "lr": 0.01},
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


class ChainCompetitor(CompetitorBase):
    """Run two competitor layouts sequentially with a warm-start handoff.

    The first competitor produces an initial placement. That placement is then
    forwarded to the second competitor as its ``pos`` override so the second
    pass refines rather than reinitializes.
    """

    def __init__(
        self,
        first_competitor: CompetitorBase,
        second_competitor: CompetitorBase,
        name: str,
        first_params: dict[str, Any] | None = None,
        second_params: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the chained competitor.

        Parameters
        ----------
        first_competitor : CompetitorBase
            Competitor used for the warm-start placement.
        second_competitor : CompetitorBase
            Competitor used for the refinement pass.
        name : str
            Registered competitor name for the chained adapter.
        first_params : dict[str, Any] | None, default=None
            Fixed variant-style overrides for the first pass.
        second_params : dict[str, Any] | None, default=None
            Fixed variant-style overrides for the second pass.
        """
        self._first = first_competitor
        self._second = second_competitor
        self.name = name
        self.max_nodes = min(self._first.max_nodes, self._second.max_nodes)
        self.supports_clusters = self._first.supports_clusters and self._second.supports_clusters
        self._first_params = {} if first_params is None else dict(first_params)
        self._second_params = {} if second_params is None else dict(second_params)

    def available(self) -> bool:
        """Return whether both chained competitors are available.

        Returns
        -------
        bool
            ``True`` when both passes can run in the current environment.
        """
        return self._first.available() and self._second.available()

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run both chained layout passes.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Maximum runtime budget shared across both passes.
        seed : int | None, default=None
            Benchmark seed forwarded to both passes.

        Returns
        -------
        CompetitorResult
            Final second-pass result, or an error payload if either pass fails.
        """
        return self._layout_chain(
            graph=graph,
            timeout=timeout,
            seed=seed,
            first_params=self._first_params,
            second_params=self._second_params,
        )

    def layout_with_variant(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
        variant_params: Optional[Mapping[str, Any]] = None,
    ) -> CompetitorResult:
        """Run the chain with extra second-pass parameter overrides.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Maximum runtime budget shared across both passes.
        seed : int | None, default=None
            Benchmark seed forwarded to both passes.
        variant_params : Mapping[str, Any] | None, default=None
            Additional overrides merged onto the second-pass parameters.

        Returns
        -------
        CompetitorResult
            Final second-pass result, or an error payload if either pass fails.
        """
        second_params = dict(self._second_params)
        if variant_params is not None:
            second_params.update(dict(variant_params))
        return self._layout_chain(
            graph=graph,
            timeout=timeout,
            seed=seed,
            first_params=self._first_params,
            second_params=second_params,
        )

    def _layout_chain(
        self,
        graph: DaguaGraph,
        timeout: float,
        seed: Optional[int],
        first_params: Mapping[str, Any],
        second_params: Mapping[str, Any],
    ) -> CompetitorResult:
        """Execute the warm-start chain.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float
            Maximum runtime budget shared across both passes.
        seed : int | None
            Benchmark seed forwarded to both passes.
        first_params : Mapping[str, Any]
            Resolved first-pass parameters.
        second_params : Mapping[str, Any]
            Resolved second-pass parameters before warm-start injection.

        Returns
        -------
        CompetitorResult
            Final chained layout result.
        """
        start = time.perf_counter()
        result1 = self._first.layout_with_variant(
            graph,
            timeout=timeout / 2.0,
            seed=seed,
            variant_params=first_params,
        )
        if result1.pos is None:
            return CompetitorResult(
                name=self.name,
                pos=None,
                runtime_seconds=time.perf_counter() - start,
                error=f"first pass failed: {result1.error}",
            )

        refined_params = dict(second_params)
        refined_params["pos"] = result1.pos
        remaining_timeout = max(10.0, timeout - (time.perf_counter() - start))
        result2 = self._second.layout_with_variant(
            graph,
            timeout=remaining_timeout,
            seed=seed,
            variant_params=refined_params,
        )
        elapsed = time.perf_counter() - start
        if result2.pos is None:
            return CompetitorResult(
                name=self.name,
                pos=None,
                runtime_seconds=elapsed,
                error=f"second pass failed: {result2.error}",
            )
        return CompetitorResult(name=self.name, pos=result2.pos, runtime_seconds=elapsed)


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
class ClassicFrKk(ChainCompetitor):
    """FR warm-start followed by KK refinement."""

    variant_param_names = frozenset({"first_steps", "second_steps"})

    def __init__(self) -> None:
        """Initialize the FR-to-KK warm-start chain."""
        super().__init__(
            first_competitor=ClassicFR(),
            second_competitor=ClassicKK(),
            name="classic_fr_kk",
            first_params={"steps": 50},
            second_params={"steps": 300},
        )

    def layout_with_variant(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
        variant_params: Optional[Mapping[str, Any]] = None,
    ) -> CompetitorResult:
        """Run the FR-to-KK chain with step-count overrides.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Maximum runtime budget shared across both passes.
        seed : int | None, default=None
            Benchmark seed forwarded to both passes.
        variant_params : Mapping[str, Any] | None, default=None
            Optional ``first_steps`` and ``second_steps`` overrides.

        Returns
        -------
        CompetitorResult
            Final chained layout result.
        """
        params = {} if variant_params is None else dict(variant_params)
        first_steps = int(params.pop("first_steps", self._first_params.get("steps", 50)))
        second_steps = int(params.pop("second_steps", self._second_params.get("steps", 300)))
        second_params = {"steps": second_steps, **params}
        return self._layout_chain(
            graph=graph,
            timeout=timeout,
            seed=seed,
            first_params={"steps": first_steps},
            second_params=second_params,
        )


@register
class ClassicKkFr(ChainCompetitor):
    """KK warm-start followed by FR refinement."""

    variant_param_names = frozenset({"first_steps", "second_steps"})

    def __init__(self) -> None:
        """Initialize the KK-to-FR warm-start chain."""
        super().__init__(
            first_competitor=ClassicKK(),
            second_competitor=ClassicFR(),
            name="classic_kk_fr",
            first_params={"steps": 300},
            second_params={"steps": 50},
        )

    def layout_with_variant(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
        variant_params: Optional[Mapping[str, Any]] = None,
    ) -> CompetitorResult:
        """Run the KK-to-FR chain with step-count overrides.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Maximum runtime budget shared across both passes.
        seed : int | None, default=None
            Benchmark seed forwarded to both passes.
        variant_params : Mapping[str, Any] | None, default=None
            Optional ``first_steps`` and ``second_steps`` overrides.

        Returns
        -------
        CompetitorResult
            Final chained layout result.
        """
        params = {} if variant_params is None else dict(variant_params)
        first_steps = int(params.pop("first_steps", self._first_params.get("steps", 300)))
        second_steps = int(params.pop("second_steps", self._second_params.get("steps", 50)))
        second_params = {"steps": second_steps, **params}
        return self._layout_chain(
            graph=graph,
            timeout=timeout,
            seed=seed,
            first_params={"steps": first_steps},
            second_params=second_params,
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
class ClassicClassicalMDS(_ClassicBase):
    """Competitor wrapper for the classical-MDS reimplementation."""

    name = "classic_classical_mds"
    max_nodes = 5_000

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run the classical-MDS layout with benchmark defaults.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Unused compatibility parameter for the competitor interface.
        seed : int | None, default=None
            Accepted for interface compatibility. Classical MDS itself is
            deterministic, but the adapter keeps the standard signature.

        Returns
        -------
        CompetitorResult
            Layout result and runtime information.
        """
        return self.layout_with_variant(graph, timeout=timeout, seed=seed, variant_params=None)


@register
class ClassicStressMajorization(_ClassicBase):
    """Competitor wrapper for the dense stress-majorization reimplementation."""

    name = "classic_stress_maj"
    max_nodes = 500

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run the stress-majorization layout with benchmark defaults.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Unused compatibility parameter for the competitor interface.
        seed : int | None, default=None
            Random seed for the stochastic warm-start jitter.

        Returns
        -------
        CompetitorResult
            Layout result and runtime information.
        """
        return self.layout_with_variant(graph, timeout=timeout, seed=seed, variant_params=None)


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
class ClassicRT(_ClassicBase):
    """Competitor wrapper for the tidy tree reimplementation."""

    name = "classic_rt"
    max_nodes = 500_000

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run the Reingold-Tilford style tree layout.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Unused compatibility parameter for the competitor interface.
        seed : int | None, default=None
            Accepted for interface compatibility. This layout is deterministic.

        Returns
        -------
        CompetitorResult
            Layout result and runtime information.
        """
        return self.layout_with_variant(graph, timeout=timeout, seed=seed, variant_params=None)


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
    variant_param_names = frozenset({"perplexity", "steps"})

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
    edge_index = graph.edge_index
    start = time.perf_counter()
    try:
        if graph.edge_weights is not None:
            extra_kwargs.setdefault("edge_weights", graph.edge_weights)
        pos = fn(
            edge_index,
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
    variant_param_names = frozenset({"n_neighbors", "min_dist", "spread"})

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
    variant_param_names = frozenset({"steps", "gcn_steps", "use_gcn", "lr", "radius"})

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
            lr=0.1,
            radius=0.4,
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
            lr=0.01,
        )
