"""Canonical algorithm-variant registry for benchmark runs.

This module is the single source of truth for variant benchmark metadata:

- the reimplementation competitor name for each variant
- the reimplementation parameters to forward
- the original competitor to compare against, when available
- the original-parameter mapping for that comparison
- stochastic and heavy-engine classification
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

ORIGINAL_VARIANT_SEPARATOR = "__for__"


@dataclass(frozen=True)
class AlgorithmVariant:
    """One benchmarkable algorithm variant.

    Parameters
    ----------
    variant_id : str
        Stable competitor name for the reimplementation variant.
    base_engine : str
        Base registered reimplementation competitor name.
    display_name : str
        Human-readable variant label used in reports.
    reimpl_params : dict[str, Any]
        Exact parameters forwarded to the reimplementation adapter.
    original_engine : str | None
        Canonical original or proxy competitor to compare against.
    original_params : dict[str, Any]
        Parameters forwarded to the original competitor when supported.
    is_true_original : bool
        Whether the configured original is an exact algorithm match for the
        reimplementation variant rather than a proxy baseline.
    is_stochastic : bool
        Whether the reimplementation variant should run across multiple seeds.
    is_heavy : bool
        Whether the variant should be scheduled in the throttled heavy lane.
    """

    variant_id: str
    base_engine: str
    display_name: str
    reimpl_params: dict[str, Any]
    original_engine: Optional[str]
    original_params: dict[str, Any]
    is_true_original: bool
    is_stochastic: bool
    is_heavy: bool


def _variant(
    variant_id: str,
    base_engine: str,
    display_name: str,
    reimpl_params: dict[str, Any],
    original_engine: Optional[str],
    original_params: dict[str, Any],
    is_true_original: bool,
    is_stochastic: bool,
    is_heavy: bool,
) -> AlgorithmVariant:
    """Build one immutable registry entry.

    Parameters
    ----------
    variant_id : str
        Stable reimplementation variant name.
    base_engine : str
        Base classic competitor name.
    display_name : str
        Human-readable report label.
    reimpl_params : dict[str, Any]
        Parameters for the reimplementation adapter.
    original_engine : str | None
        Canonical original or proxy competitor.
    original_params : dict[str, Any]
        Parameters for the original competitor.
    is_true_original : bool
        Whether the original is an exact algorithm match.
    is_stochastic : bool
        Whether the reimplementation variant is stochastic.
    is_heavy : bool
        Whether the variant needs throttled scheduling.

    Returns
    -------
    AlgorithmVariant
        Frozen registry record.
    """
    return AlgorithmVariant(
        variant_id=variant_id,
        base_engine=base_engine,
        display_name=display_name,
        reimpl_params=dict(reimpl_params),
        original_engine=original_engine,
        original_params=dict(original_params),
        is_true_original=is_true_original,
        is_stochastic=is_stochastic,
        is_heavy=is_heavy,
    )


def original_variant_name(variant: AlgorithmVariant) -> Optional[str]:
    """Return the synthetic competitor name for the original-side variant run.

    Parameters
    ----------
    variant : AlgorithmVariant
        Reimplementation-side registry entry.

    Returns
    -------
    str | None
        Synthetic original-variant competitor name, or ``None`` when the
        variant has no original comparison target.
    """
    if variant.original_engine is None:
        return None
    return f"{variant.original_engine}{ORIGINAL_VARIANT_SEPARATOR}{variant.variant_id}"


def algorithm_family(engine_name: str) -> str:
    """Return a report-friendly algorithm family label.

    Parameters
    ----------
    engine_name : str
        Reimplementation base engine name or variant name.

    Returns
    -------
    str
        Family identifier without the ``classic_`` prefix.
    """
    if engine_name.startswith("classic_"):
        return engine_name.removeprefix("classic_")
    variant = get_variant(engine_name)
    if variant is not None:
        return variant.base_engine.removeprefix("classic_")
    original_variant = get_variant_for_original_name(engine_name)
    if original_variant is not None:
        return original_variant.base_engine.removeprefix("classic_")
    return engine_name


def get_variant(variant_id: str) -> Optional[AlgorithmVariant]:
    """Look up one reimplementation variant by name.

    Parameters
    ----------
    variant_id : str
        Reimplementation-side variant competitor name.

    Returns
    -------
    AlgorithmVariant | None
        Matching registry entry, or ``None`` when not found.
    """
    return _VARIANT_BY_ID.get(variant_id)


def get_variant_for_original_name(engine_name: str) -> Optional[AlgorithmVariant]:
    """Resolve an original-side synthetic variant competitor name.

    Parameters
    ----------
    engine_name : str
        Synthetic original competitor name created from one registry entry.

    Returns
    -------
    AlgorithmVariant | None
        Matching registry entry, or ``None`` when the name is not an
        original-side variant competitor.
    """
    return _VARIANT_BY_ORIGINAL_NAME.get(engine_name)


def variants_for_base_engine(base_engine: str) -> list[AlgorithmVariant]:
    """Return all variants for one base reimplementation engine.

    Parameters
    ----------
    base_engine : str
        Base classic competitor name.

    Returns
    -------
    list[AlgorithmVariant]
        Variants in registry order.
    """
    return list(_VARIANTS_BY_BASE_ENGINE.get(base_engine, ()))


def variant_pairings() -> dict[str, list[str]]:
    """Build the variant-mode reimplementation-to-original pairing map.

    Returns
    -------
    dict[str, list[str]]
        Mapping from reimplementation variant names to synthetic original
        variant competitor names.
    """
    pairings: dict[str, list[str]] = {}
    for variant in VARIANT_REGISTRY:
        original_name = original_variant_name(variant)
        pairings[variant.variant_id] = [] if original_name is None else [original_name]
    for original_name, variant in _VARIANT_BY_ORIGINAL_NAME.items():
        pairings[original_name] = [variant.variant_id]
    return pairings


def base_pairings() -> dict[str, list[str]]:
    """Build the canonical base reimplementation-to-original pairing map.

    Returns
    -------
    dict[str, list[str]]
        Mapping from base classic competitors to their canonical original or
        proxy baseline.
    """
    pairings: dict[str, list[str]] = {}
    for variant in VARIANT_REGISTRY:
        pairings.setdefault(variant.base_engine, [])
        if variant.original_engine is None:
            continue
        if variant.original_engine not in pairings[variant.base_engine]:
            pairings[variant.base_engine].append(variant.original_engine)
    return pairings


def engine_is_stochastic(engine_name: str) -> bool:
    """Return whether an engine should run across multiple seeds.

    Parameters
    ----------
    engine_name : str
        Base competitor name, reimplementation variant name, or synthetic
        original-side variant name.

    Returns
    -------
    bool
        ``True`` when the engine is stochastic in the benchmark harness.
    """
    variant = get_variant(engine_name)
    if variant is not None:
        return variant.is_stochastic
    original_variant = get_variant_for_original_name(engine_name)
    if original_variant is not None and original_variant.original_engine is not None:
        return _BASE_ENGINE_STOCHASTICITY.get(original_variant.original_engine, False)
    return _BASE_ENGINE_STOCHASTICITY.get(engine_name, False)


def engine_is_heavy(engine_name: str) -> bool:
    """Return whether an engine belongs to the throttled heavy lane.

    Parameters
    ----------
    engine_name : str
        Base competitor name, reimplementation variant name, or synthetic
        original-side variant name.

    Returns
    -------
    bool
        ``True`` when the engine should run in the heavy scheduling lane.
    """
    variant = get_variant(engine_name)
    if variant is not None:
        return variant.is_heavy
    original_variant = get_variant_for_original_name(engine_name)
    if original_variant is not None and original_variant.original_engine is not None:
        return _BASE_ENGINE_HEAVY.get(original_variant.original_engine, False)
    return _BASE_ENGINE_HEAVY.get(engine_name, False)


def engine_timeout_cap(engine_name: str) -> Optional[int]:
    """Return an optional timeout cap for one engine.

    Parameters
    ----------
    engine_name : str
        Base competitor name, reimplementation variant name, or synthetic
        original-side variant name.

    Returns
    -------
    int | None
        Timeout cap in seconds when one is configured.
    """
    variant = get_variant(engine_name)
    if variant is not None:
        return _BASE_TIMEOUT_CAPS.get(variant.base_engine)
    original_variant = get_variant_for_original_name(engine_name)
    if original_variant is not None and original_variant.original_engine is not None:
        return _BASE_TIMEOUT_CAPS.get(original_variant.original_engine)
    return _BASE_TIMEOUT_CAPS.get(engine_name)


def variant_engine_names() -> list[str]:
    """Return all reimplementation-side variant competitor names.

    Returns
    -------
    list[str]
        Registry-order reimplementation variant names.
    """
    return [variant.variant_id for variant in VARIANT_REGISTRY]


def original_variant_engine_names() -> list[str]:
    """Return all original-side synthetic variant competitor names.

    Returns
    -------
    list[str]
        Registry-order original variant names for entries that define one.
    """
    names: list[str] = []
    for variant in VARIANT_REGISTRY:
        name = original_variant_name(variant)
        if name is not None:
            names.append(name)
    return names


VARIANT_REGISTRY: list[AlgorithmVariant] = [
    _variant(
        "classic_fr_steps50",
        "classic_fr",
        "FR (steps=50)",
        {"steps": 50},
        "nx_spring",
        {"iterations": 50},
        True,
        True,
        False,
    ),
    _variant(
        "classic_fr_steps100",
        "classic_fr",
        "FR (steps=100)",
        {"steps": 100},
        "nx_spring",
        {"iterations": 100},
        True,
        True,
        False,
    ),
    _variant(
        "classic_fr_steps200",
        "classic_fr",
        "FR (steps=200)",
        {"steps": 200},
        "nx_spring",
        {"iterations": 200},
        True,
        True,
        False,
    ),
    _variant(
        "classic_fr_steps500",
        "classic_fr",
        "FR (steps=500)",
        {"steps": 500},
        "nx_spring",
        {"iterations": 500},
        True,
        True,
        False,
    ),
    _variant(
        "classic_kk_steps100",
        "classic_kk",
        "KK (SciPy default)",
        {"steps": None, "orient_to_direction": False},
        "nx_kamada_kawai",
        {},
        False,
        False,
        False,
    ),
    _variant(
        "classic_kk_steps300",
        "classic_kk",
        "KK (SciPy default)",
        {"steps": None, "orient_to_direction": False},
        "nx_kamada_kawai",
        {},
        False,
        False,
        False,
    ),
    _variant(
        "classic_kk_steps1000",
        "classic_kk",
        "KK (SciPy default)",
        {"steps": None, "orient_to_direction": False},
        "nx_kamada_kawai",
        {},
        False,
        False,
        False,
    ),
    _variant(
        "classic_fr_kk_default",
        "classic_fr_kk",
        "FR->KK (50+300)",
        {"first_steps": 50, "second_steps": 300},
        None,
        {},
        False,
        True,
        False,
    ),
    _variant(
        "classic_fr_kk_long",
        "classic_fr_kk",
        "FR->KK (100+300)",
        {"first_steps": 100, "second_steps": 300},
        None,
        {},
        False,
        True,
        False,
    ),
    _variant(
        "classic_kk_fr_default",
        "classic_kk_fr",
        "KK->FR (300+50)",
        {"first_steps": 300, "second_steps": 50},
        None,
        {},
        False,
        True,
        False,
    ),
    _variant(
        "classic_kk_fr_long",
        "classic_kk_fr",
        "KK->FR (300+100)",
        {"first_steps": 300, "second_steps": 100},
        None,
        {},
        False,
        True,
        False,
    ),
    _variant(
        "classic_fa2_default",
        "classic_fa2",
        "FA2 (default)",
        {
            "steps": 200,
            "gravity": 1.0,
            "scaling_ratio": 2.0,
            "strong_gravity": False,
            "outbound_attraction_distribution": True,
            "barnes_hut": True,
            "barnes_hut_theta": 1.2,
        },
        "fa2_ref",
        {
            "iterations": 200,
            "gravity": 1.0,
            "scalingRatio": 2.0,
            "strongGravityMode": False,
            "outboundAttractionDistribution": True,
        },
        True,
        True,
        False,
    ),
    _variant(
        "classic_fa2_gravity0",
        "classic_fa2",
        "FA2 (gravity=0.0)",
        {
            "steps": 200,
            "gravity": 0.0,
            "scaling_ratio": 2.0,
            "strong_gravity": False,
            "outbound_attraction_distribution": True,
            "barnes_hut": True,
            "barnes_hut_theta": 1.2,
        },
        "fa2_ref",
        {
            "iterations": 200,
            "gravity": 0.0,
            "scalingRatio": 2.0,
            "strongGravityMode": False,
            "outboundAttractionDistribution": True,
        },
        True,
        True,
        False,
    ),
    _variant(
        "classic_fa2_gravity2",
        "classic_fa2",
        "FA2 (gravity=2.0)",
        {
            "steps": 200,
            "gravity": 2.0,
            "scaling_ratio": 2.0,
            "strong_gravity": False,
            "outbound_attraction_distribution": True,
            "barnes_hut": True,
            "barnes_hut_theta": 1.2,
        },
        "fa2_ref",
        {
            "iterations": 200,
            "gravity": 2.0,
            "scalingRatio": 2.0,
            "strongGravityMode": False,
            "outboundAttractionDistribution": True,
        },
        True,
        True,
        False,
    ),
    _variant(
        "classic_fa2_scaling1",
        "classic_fa2",
        "FA2 (scaling=1.0)",
        {
            "steps": 200,
            "gravity": 1.0,
            "scaling_ratio": 1.0,
            "strong_gravity": False,
            "outbound_attraction_distribution": True,
            "barnes_hut": True,
            "barnes_hut_theta": 1.2,
        },
        "fa2_ref",
        {
            "iterations": 200,
            "gravity": 1.0,
            "scalingRatio": 1.0,
            "strongGravityMode": False,
            "outboundAttractionDistribution": True,
        },
        True,
        True,
        False,
    ),
    _variant(
        "classic_fa2_scaling4",
        "classic_fa2",
        "FA2 (scaling=4.0)",
        {
            "steps": 200,
            "gravity": 1.0,
            "scaling_ratio": 4.0,
            "strong_gravity": False,
            "outbound_attraction_distribution": True,
            "barnes_hut": True,
            "barnes_hut_theta": 1.2,
        },
        "fa2_ref",
        {
            "iterations": 200,
            "gravity": 1.0,
            "scalingRatio": 4.0,
            "strongGravityMode": False,
            "outboundAttractionDistribution": True,
        },
        True,
        True,
        False,
    ),
    _variant(
        "classic_fa2_strong_gravity",
        "classic_fa2",
        "FA2 (strong gravity)",
        {
            "steps": 200,
            "gravity": 1.0,
            "scaling_ratio": 2.0,
            "strong_gravity": True,
            "outbound_attraction_distribution": True,
            "barnes_hut": True,
            "barnes_hut_theta": 1.2,
        },
        "fa2_ref",
        {
            "iterations": 200,
            "gravity": 1.0,
            "scalingRatio": 2.0,
            "strongGravityMode": True,
            "outboundAttractionDistribution": True,
        },
        True,
        True,
        False,
    ),
    _variant(
        "classic_fa2_no_outbound",
        "classic_fa2",
        "FA2 (no outbound weighting)",
        {
            "steps": 200,
            "gravity": 1.0,
            "scaling_ratio": 2.0,
            "strong_gravity": False,
            "outbound_attraction_distribution": False,
            "barnes_hut": True,
            "barnes_hut_theta": 1.2,
        },
        "fa2_ref",
        {
            "iterations": 200,
            "gravity": 1.0,
            "scalingRatio": 2.0,
            "strongGravityMode": False,
            "outboundAttractionDistribution": False,
        },
        True,
        True,
        False,
    ),
    _variant(
        "classic_fa2_dissuade_hubs",
        "classic_fa2",
        "FA2 (dissuade hubs)",
        {
            "steps": 200,
            "gravity": 1.0,
            "scaling_ratio": 2.0,
            "strong_gravity": False,
            "outbound_attraction_distribution": True,
            "dissuade_hubs": True,
            "barnes_hut": True,
            "barnes_hut_theta": 1.2,
        },
        "fa2_ref",
        {
            "iterations": 200,
            "gravity": 1.0,
            "scalingRatio": 2.0,
            "strongGravityMode": False,
            "outboundAttractionDistribution": True,
        },
        False,
        True,
        False,
    ),
    _variant(
        "classic_fa2_linlog",
        "classic_fa2",
        "FA2 (linlog)",
        {
            "steps": 200,
            "gravity": 1.0,
            "scaling_ratio": 2.0,
            "strong_gravity": False,
            "outbound_attraction_distribution": True,
            "linlog": True,
            "barnes_hut": True,
            "barnes_hut_theta": 1.2,
        },
        "fa2_ref",
        {
            "iterations": 200,
            "gravity": 1.0,
            "scalingRatio": 2.0,
            "strongGravityMode": False,
            "outboundAttractionDistribution": True,
            "linLogMode": True,
        },
        False,
        True,
        False,
    ),
    _variant(
        "classic_fa2_barnes_hut",
        "classic_fa2",
        "FA2 (Barnes-Hut)",
        {
            "steps": 200,
            "gravity": 1.0,
            "scaling_ratio": 2.0,
            "strong_gravity": False,
            "outbound_attraction_distribution": True,
            "barnes_hut": True,
            "barnes_hut_theta": 1.2,
        },
        "fa2_ref",
        {
            "iterations": 200,
            "gravity": 1.0,
            "scalingRatio": 2.0,
            "strongGravityMode": False,
            "outboundAttractionDistribution": True,
            "barnesHutOptimize": True,
            "barnesHutTheta": 1.2,
        },
        True,
        True,
        False,
    ),
    _variant(
        "classic_fa2_exact",
        "classic_fa2",
        "FA2 (exact, no BH)",
        {
            "steps": 200,
            "gravity": 1.0,
            "scaling_ratio": 2.0,
            "strong_gravity": False,
            "outbound_attraction_distribution": True,
            "barnes_hut": False,
        },
        "fa2_ref",
        {
            "iterations": 200,
            "gravity": 1.0,
            "scalingRatio": 2.0,
            "strongGravityMode": False,
            "outboundAttractionDistribution": True,
            "barnesHutOptimize": False,
        },
        True,
        True,
        False,
    ),
    _variant(
        "classic_stress_sgd_steps30",
        "classic_stress_sgd",
        "Stress-SGD (steps=30)",
        {"steps": 30, "eps": 0.01},
        "sgd2",
        {"t_max": 30, "eps": 0.01},
        True,
        True,
        True,
    ),
    _variant(
        "classic_stress_sgd_steps300",
        "classic_stress_sgd",
        "Stress-SGD (steps=300)",
        {"steps": 300, "eps": 0.01},
        "sgd2",
        {"t_max": 300, "eps": 0.01},
        True,
        True,
        True,
    ),
    _variant(
        "classic_stress_sgd_eps001",
        "classic_stress_sgd",
        "Stress-SGD (eps=0.001)",
        {"steps": 300, "eps": 0.001},
        "sgd2",
        {"t_max": 300, "eps": 0.001},
        True,
        True,
        True,
    ),
    _variant(
        "classic_stress_sgd_eps01",
        "classic_stress_sgd",
        "Stress-SGD (eps=0.1)",
        {"steps": 300, "eps": 0.1},
        "sgd2",
        {"t_max": 300, "eps": 0.1},
        True,
        True,
        True,
    ),
    _variant(
        "classic_spectral_default",
        "classic_spectral",
        "Spectral",
        {},
        "nx_spectral",
        {},
        True,
        False,
        True,
    ),
    _variant(
        "classic_spectral_random_walk",
        "classic_spectral",
        "Spectral (random walk)",
        {"normalization": "random_walk"},
        None,
        {},
        False,
        False,
        False,
    ),
    _variant(
        "classic_spectral_unnormalized",
        "classic_spectral",
        "Spectral (unnormalized)",
        {"normalization": "unnormalized"},
        None,
        {},
        False,
        False,
        False,
    ),
    _variant(
        "classic_classical_mds_default",
        "classic_classical_mds",
        "Classical MDS",
        {},
        "igraph_mds",
        {},
        True,
        False,
        True,
    ),
    _variant(
        "classic_stress_maj_default",
        "classic_stress_maj",
        "Stress Majorization (iter=200)",
        {"iterations": 200},
        "ogdf_stress",
        {},
        False,
        True,
        True,
    ),
    _variant(
        "classic_stress_maj_iter50",
        "classic_stress_maj",
        "Stress Majorization (iter=50)",
        {"iterations": 50},
        "ogdf_stress",
        {},
        False,
        True,
        True,
    ),
    _variant(
        "classic_stress_maj_iter500",
        "classic_stress_maj",
        "Stress Majorization (iter=500)",
        {"iterations": 500},
        "ogdf_stress",
        {},
        False,
        True,
        True,
    ),
    _variant(
        "classic_sugiyama_default",
        "classic_sugiyama",
        "Sugiyama (default)",
        {"barycenter_passes": 24, "rank_sep": 1.0, "node_sep": 1.0},
        "igraph_sugiyama",
        {"maxiter": 24, "vgap": 1.0, "hgap": 1.0},
        True,
        True,
        False,
    ),
    _variant(
        "classic_sugiyama_passes4",
        "classic_sugiyama",
        "Sugiyama (passes=4)",
        {"barycenter_passes": 4, "rank_sep": 1.0, "node_sep": 1.0},
        "igraph_sugiyama",
        {"maxiter": 4, "vgap": 1.0, "hgap": 1.0},
        True,
        True,
        False,
    ),
    _variant(
        "classic_sugiyama_passes48",
        "classic_sugiyama",
        "Sugiyama (passes=48)",
        {"barycenter_passes": 48, "rank_sep": 1.0, "node_sep": 1.0},
        "igraph_sugiyama",
        {"maxiter": 48, "vgap": 1.0, "hgap": 1.0},
        True,
        True,
        False,
    ),
    _variant(
        "classic_sugiyama_wide",
        "classic_sugiyama",
        "Sugiyama (wide)",
        {"barycenter_passes": 24, "rank_sep": 2.0, "node_sep": 2.0},
        "igraph_sugiyama",
        {"maxiter": 24, "vgap": 2.0, "hgap": 2.0},
        True,
        True,
        False,
    ),
    _variant(
        "classic_sugiyama_tight",
        "classic_sugiyama",
        "Sugiyama (tight)",
        {"barycenter_passes": 24, "rank_sep": 0.5, "node_sep": 0.5},
        "igraph_sugiyama",
        {"maxiter": 24, "vgap": 0.5, "hgap": 0.5},
        True,
        True,
        False,
    ),
    _variant(
        "classic_tsnet_default",
        "classic_tsnet",
        "tsNET (default)",
        {"perplexity": 30.0, "steps": 500},
        "tsne_graph",
        {"perplexity": 30.0, "max_iter": 500},
        False,
        True,
        True,
    ),
    _variant(
        "classic_tsnet_perp5",
        "classic_tsnet",
        "tsNET (perplexity=5)",
        {"perplexity": 5.0, "steps": 500},
        "tsne_graph",
        {"perplexity": 5.0, "max_iter": 500},
        False,
        True,
        True,
    ),
    _variant(
        "classic_tsnet_perp50",
        "classic_tsnet",
        "tsNET (perplexity=50)",
        {"perplexity": 50.0, "steps": 500},
        "tsne_graph",
        {"perplexity": 50.0, "max_iter": 500},
        False,
        True,
        True,
    ),
    _variant(
        "classic_tsnet_steps200",
        "classic_tsnet",
        "tsNET (steps=200)",
        {"perplexity": 30.0, "steps": 200},
        "tsne_graph",
        {"perplexity": 30.0, "max_iter": 200},
        False,
        True,
        True,
    ),
    _variant(
        "classic_tsnet_steps2000",
        "classic_tsnet",
        "tsNET (steps=2000)",
        {"perplexity": 30.0, "steps": 2000},
        "tsne_graph",
        {"perplexity": 30.0, "max_iter": 2000},
        False,
        True,
        True,
    ),
    _variant(
        "classic_gem_iters100",
        "classic_gem",
        "GEM (iters=100)",
        {"max_iters": 100},
        "ogdf_gem",
        {},
        False,
        True,
        False,
    ),
    _variant(
        "classic_gem_iters500",
        "classic_gem",
        "GEM (iters=500)",
        {"max_iters": 500},
        "ogdf_gem",
        {},
        False,
        True,
        False,
    ),
    _variant(
        "classic_gem_iters2000",
        "classic_gem",
        "GEM (iters=2000)",
        {"max_iters": 2000},
        "ogdf_gem",
        {},
        False,
        True,
        False,
    ),
    _variant(
        "classic_fmmm_steps10",
        "classic_fmmm",
        "FM^3 (steps=10)",
        {"steps": 10},
        "ogdf_fmmm",
        {},
        False,
        True,
        False,
    ),
    _variant(
        "classic_fmmm_steps100",
        "classic_fmmm",
        "FM^3 (steps=100)",
        {"steps": 100},
        "ogdf_fmmm",
        {},
        False,
        True,
        False,
    ),
    _variant(
        "classic_fmmm_steps200",
        "classic_fmmm",
        "FM^3 (steps=200)",
        {"steps": 200},
        "ogdf_fmmm",
        {},
        False,
        True,
        False,
    ),
    _variant(
        "classic_maxent_stress_default",
        "classic_maxent_stress",
        "Maxent-Stress (default)",
        {"steps": 200, "alpha": 1.0, "use_entropy": False},
        "ogdf_stress",
        {},
        False,
        True,
        False,
    ),
    _variant(
        "classic_maxent_stress_entropy",
        "classic_maxent_stress",
        "Maxent-Stress (entropy)",
        {"steps": 200, "alpha": 1.0, "use_entropy": True},
        "ogdf_stress",
        {},
        False,
        True,
        False,
    ),
    _variant(
        "classic_maxent_stress_alpha2",
        "classic_maxent_stress",
        "Maxent-Stress (alpha=2.0)",
        {"steps": 200, "alpha": 2.0, "use_entropy": True},
        "ogdf_stress",
        {},
        False,
        True,
        False,
    ),
    _variant(
        "classic_maxent_stress_steps50",
        "classic_maxent_stress",
        "Maxent-Stress (steps=50)",
        {"steps": 50, "alpha": 1.0, "use_entropy": False},
        "ogdf_stress",
        {},
        False,
        True,
        False,
    ),
    _variant(
        "classic_maxent_stress_steps400",
        "classic_maxent_stress",
        "Maxent-Stress (steps=400)",
        {"steps": 400, "alpha": 1.0, "use_entropy": False},
        "ogdf_stress",
        {},
        False,
        True,
        False,
    ),
    _variant(
        "classic_davidson_harel_rounds50",
        "classic_davidson_harel",
        "Davidson-Harel (rounds=50)",
        {"rounds": 50},
        "igraph_davidson_harel",
        {"maxiter": 50},
        False,
        True,
        False,
    ),
    _variant(
        "classic_davidson_harel_rounds100",
        "classic_davidson_harel",
        "Davidson-Harel (rounds=100)",
        {"rounds": 100},
        "igraph_davidson_harel",
        {"maxiter": 100},
        False,
        True,
        False,
    ),
    _variant(
        "classic_davidson_harel_rounds200",
        "classic_davidson_harel",
        "Davidson-Harel (rounds=200)",
        {"rounds": 200},
        "igraph_davidson_harel",
        {"maxiter": 200},
        False,
        True,
        False,
    ),
    _variant(
        "classic_linlog_default",
        "classic_linlog",
        "LinLog (default)",
        {"a": 1.0, "r": 0.0, "steps": 300},
        None,
        {},
        False,
        True,
        False,
    ),
    _variant(
        "classic_linlog_quadratic",
        "classic_linlog",
        "LinLog (a=2.0)",
        {"a": 2.0, "r": 0.0, "steps": 300},
        None,
        {},
        False,
        True,
        False,
    ),
    _variant(
        "classic_linlog_power",
        "classic_linlog",
        "LinLog (r=0.5)",
        {"a": 1.0, "r": 0.5, "steps": 300},
        None,
        {},
        False,
        True,
        False,
    ),
    _variant(
        "classic_linlog_steps100",
        "classic_linlog",
        "LinLog (steps=100)",
        {"a": 1.0, "r": 0.0, "steps": 100},
        None,
        {},
        False,
        True,
        False,
    ),
    _variant(
        "classic_linlog_steps500",
        "classic_linlog",
        "LinLog (steps=500)",
        {"a": 1.0, "r": 0.0, "steps": 500},
        None,
        {},
        False,
        True,
        False,
    ),
    _variant(
        "classic_pivot_mds_10",
        "classic_pivot_mds",
        "Pivot-MDS (pivots=10)",
        {"n_pivots": 10},
        "ogdf_pivot_mds",
        {},
        False,
        True,
        False,
    ),
    _variant(
        "classic_pivot_mds_50",
        "classic_pivot_mds",
        "Pivot-MDS (pivots=50)",
        {"n_pivots": 50},
        "ogdf_pivot_mds",
        {},
        False,
        True,
        False,
    ),
    _variant(
        "classic_pivot_mds_100",
        "classic_pivot_mds",
        "Pivot-MDS (pivots=100)",
        {"n_pivots": 100},
        "ogdf_pivot_mds",
        {},
        False,
        True,
        False,
    ),
    _variant(
        "classic_pivot_mds_200",
        "classic_pivot_mds",
        "Pivot-MDS (pivots=200)",
        {"n_pivots": 200},
        "ogdf_pivot_mds",
        {},
        False,
        True,
        False,
    ),
    _variant(
        "classic_rt_default",
        "classic_rt",
        "Reingold-Tilford",
        {},
        "igraph_rt",
        {},
        True,
        False,
        False,
    ),
    _variant(
        "classic_rt_horizontal",
        "classic_rt",
        "Reingold-Tilford (horizontal)",
        {"horizontal": True},
        None,
        {},
        False,
        False,
        False,
    ),
    _variant(
        "classic_graphopt_default",
        "classic_graphopt",
        "GraphOpt (default)",
        {"niter": 500, "node_charge": 0.001, "node_mass": 30.0, "spring_constant": 1.0},
        "igraph_graphopt",
        {"niter": 500, "node_charge": 0.001, "node_mass": 30.0, "spring_constant": 1.0},
        True,
        True,
        False,
    ),
    _variant(
        "classic_graphopt_charge_low",
        "classic_graphopt",
        "GraphOpt (charge=0.0005)",
        {"niter": 500, "node_charge": 0.0005, "node_mass": 30.0, "spring_constant": 1.0},
        "igraph_graphopt",
        {"niter": 500, "node_charge": 0.0005, "node_mass": 30.0, "spring_constant": 1.0},
        True,
        True,
        False,
    ),
    _variant(
        "classic_graphopt_charge_high",
        "classic_graphopt",
        "GraphOpt (charge=0.002)",
        {"niter": 500, "node_charge": 0.002, "node_mass": 30.0, "spring_constant": 1.0},
        "igraph_graphopt",
        {"niter": 500, "node_charge": 0.002, "node_mass": 30.0, "spring_constant": 1.0},
        True,
        True,
        False,
    ),
    _variant(
        "classic_graphopt_mass_low",
        "classic_graphopt",
        "GraphOpt (mass=10.0)",
        {"niter": 500, "node_charge": 0.001, "node_mass": 10.0, "spring_constant": 1.0},
        "igraph_graphopt",
        {"niter": 500, "node_charge": 0.001, "node_mass": 10.0, "spring_constant": 1.0},
        True,
        True,
        False,
    ),
    _variant(
        "classic_graphopt_mass_high",
        "classic_graphopt",
        "GraphOpt (mass=50.0)",
        {"niter": 500, "node_charge": 0.001, "node_mass": 50.0, "spring_constant": 1.0},
        "igraph_graphopt",
        {"niter": 500, "node_charge": 0.001, "node_mass": 50.0, "spring_constant": 1.0},
        True,
        True,
        False,
    ),
    _variant(
        "classic_graphopt_spring2",
        "classic_graphopt",
        "GraphOpt (spring=2.0)",
        {"niter": 500, "node_charge": 0.001, "node_mass": 30.0, "spring_constant": 2.0},
        "igraph_graphopt",
        {"niter": 500, "node_charge": 0.001, "node_mass": 30.0, "spring_constant": 2.0},
        True,
        True,
        False,
    ),
    _variant(
        "classic_drl_default",
        "classic_drl",
        "DrL (default)",
        {"options": "default"},
        "igraph_drl",
        {"options": "default"},
        True,
        True,
        False,
    ),
    _variant(
        "classic_drl_coarsen",
        "classic_drl",
        "DrL (coarsen)",
        {"options": "coarsen"},
        "igraph_drl",
        {"options": "coarsen"},
        True,
        True,
        False,
    ),
    _variant(
        "classic_drl_coarsest",
        "classic_drl",
        "DrL (coarsest)",
        {"options": "coarsest"},
        "igraph_drl",
        {"options": "coarsest"},
        True,
        True,
        False,
    ),
    _variant(
        "classic_drl_refine",
        "classic_drl",
        "DrL (refine)",
        {"options": "refine"},
        "igraph_drl",
        {"options": "refine"},
        True,
        True,
        False,
    ),
    _variant(
        "classic_drl_final",
        "classic_drl",
        "DrL (final)",
        {"options": "final"},
        "igraph_drl",
        {"options": "final"},
        True,
        True,
        False,
    ),
    _variant(
        "classic_lgl_default",
        "classic_lgl",
        "LGL (default)",
        {"maxiter": 150, "coolexp": 1.5},
        "igraph_lgl",
        {"maxiter": 150, "coolexp": 1.5},
        True,
        True,
        False,
    ),
    _variant(
        "classic_lgl_iter50",
        "classic_lgl",
        "LGL (iters=50)",
        {"maxiter": 50, "coolexp": 1.5},
        "igraph_lgl",
        {"maxiter": 50, "coolexp": 1.5},
        True,
        True,
        False,
    ),
    _variant(
        "classic_lgl_iter300",
        "classic_lgl",
        "LGL (iters=300)",
        {"maxiter": 300, "coolexp": 1.5},
        "igraph_lgl",
        {"maxiter": 300, "coolexp": 1.5},
        True,
        True,
        False,
    ),
    _variant(
        "classic_lgl_cool1",
        "classic_lgl",
        "LGL (cool=1.0)",
        {"maxiter": 150, "coolexp": 1.0},
        "igraph_lgl",
        {"maxiter": 150, "coolexp": 1.0},
        True,
        True,
        False,
    ),
    _variant(
        "classic_lgl_cool2",
        "classic_lgl",
        "LGL (cool=2.0)",
        {"maxiter": 150, "coolexp": 2.0},
        "igraph_lgl",
        {"maxiter": 150, "coolexp": 2.0},
        True,
        True,
        False,
    ),
    _variant(
        "classic_sfdp_default",
        "classic_sfdp",
        "SFDP (default)",
        {"steps": 500, "theta": 0.6, "repulsive_exponent": -1.0},
        "graphviz_sfdp",
        {},
        False,
        True,
        False,
    ),
    _variant(
        "classic_sfdp_theta04",
        "classic_sfdp",
        "SFDP (theta=0.4)",
        {"steps": 500, "theta": 0.4, "repulsive_exponent": -1.0},
        "graphviz_sfdp",
        {},
        False,
        True,
        False,
    ),
    _variant(
        "classic_sfdp_theta08",
        "classic_sfdp",
        "SFDP (theta=0.8)",
        {"steps": 500, "theta": 0.8, "repulsive_exponent": -1.0},
        "graphviz_sfdp",
        {},
        False,
        True,
        False,
    ),
    _variant(
        "classic_sfdp_p_neg2",
        "classic_sfdp",
        "SFDP (p=-2.0)",
        {"steps": 500, "theta": 0.6, "repulsive_exponent": -2.0},
        "graphviz_sfdp",
        {},
        False,
        True,
        False,
    ),
    _variant(
        "classic_sfdp_steps200",
        "classic_sfdp",
        "SFDP (steps=200)",
        {"steps": 200, "theta": 0.6, "repulsive_exponent": -1.0},
        "graphviz_sfdp",
        {},
        False,
        True,
        False,
    ),
    _variant(
        "classic_umap_default",
        "classic_umap",
        "UMAP (default)",
        {"n_neighbors": 15, "min_dist": 0.1, "spread": 1.0},
        "umap_graph",
        {"n_neighbors": 15, "min_dist": 0.1, "spread": 1.0},
        True,
        True,
        True,
    ),
    _variant(
        "classic_umap_nn5",
        "classic_umap",
        "UMAP (neighbors=5)",
        {"n_neighbors": 5, "min_dist": 0.1, "spread": 1.0},
        "umap_graph",
        {"n_neighbors": 5, "min_dist": 0.1, "spread": 1.0},
        True,
        True,
        True,
    ),
    _variant(
        "classic_umap_nn30",
        "classic_umap",
        "UMAP (neighbors=30)",
        {"n_neighbors": 30, "min_dist": 0.1, "spread": 1.0},
        "umap_graph",
        {"n_neighbors": 30, "min_dist": 0.1, "spread": 1.0},
        True,
        True,
        True,
    ),
    _variant(
        "classic_umap_mindist001",
        "classic_umap",
        "UMAP (min_dist=0.01)",
        {"n_neighbors": 15, "min_dist": 0.01, "spread": 1.0},
        "umap_graph",
        {"n_neighbors": 15, "min_dist": 0.01, "spread": 1.0},
        True,
        True,
        True,
    ),
    _variant(
        "classic_umap_mindist05",
        "classic_umap",
        "UMAP (min_dist=0.5)",
        {"n_neighbors": 15, "min_dist": 0.5, "spread": 1.0},
        "umap_graph",
        {"n_neighbors": 15, "min_dist": 0.5, "spread": 1.0},
        True,
        True,
        True,
    ),
    _variant(
        "classic_umap_spread2",
        "classic_umap",
        "UMAP (spread=2.0)",
        {"n_neighbors": 15, "min_dist": 0.1, "spread": 2.0},
        "umap_graph",
        {"n_neighbors": 15, "min_dist": 0.1, "spread": 2.0},
        True,
        True,
        True,
    ),
    _variant(
        "classic_neulay_default",
        "classic_neulay",
        "NeuLay (default)",
        {"steps": 20000, "gcn_steps": 2000, "use_gcn": True, "lr": 0.1, "radius": 0.4},
        "neulay",
        {"steps": 20000, "gcn_steps": 2000, "use_gcn": True, "lr": 0.1, "radius": 0.4},
        True,
        True,
        True,
    ),
    _variant(
        "classic_neulay_lr001",
        "classic_neulay",
        "NeuLay (lr=0.01)",
        {"steps": 20000, "gcn_steps": 2000, "use_gcn": True, "lr": 0.01, "radius": 0.4},
        "neulay",
        {"steps": 20000, "gcn_steps": 2000, "use_gcn": True, "lr": 0.01, "radius": 0.4},
        True,
        True,
        True,
    ),
    _variant(
        "classic_neulay_lr05",
        "classic_neulay",
        "NeuLay (lr=0.5)",
        {"steps": 20000, "gcn_steps": 2000, "use_gcn": True, "lr": 0.5, "radius": 0.4},
        "neulay",
        {"steps": 20000, "gcn_steps": 2000, "use_gcn": True, "lr": 0.5, "radius": 0.4},
        True,
        True,
        True,
    ),
    _variant(
        "classic_neulay_radius02",
        "classic_neulay",
        "NeuLay (radius=0.2)",
        {"steps": 20000, "gcn_steps": 2000, "use_gcn": True, "lr": 0.1, "radius": 0.2},
        "neulay",
        {"steps": 20000, "gcn_steps": 2000, "use_gcn": True, "lr": 0.1, "radius": 0.2},
        True,
        True,
        True,
    ),
    _variant(
        "classic_neulay_radius08",
        "classic_neulay",
        "NeuLay (radius=0.8)",
        {"steps": 20000, "gcn_steps": 2000, "use_gcn": True, "lr": 0.1, "radius": 0.8},
        "neulay",
        {"steps": 20000, "gcn_steps": 2000, "use_gcn": True, "lr": 0.1, "radius": 0.8},
        True,
        True,
        True,
    ),
    _variant(
        "classic_neulay_no_gcn",
        "classic_neulay",
        "NeuLay (no GCN)",
        {"steps": 20000, "use_gcn": False, "lr": 0.1, "radius": 0.4},
        "neulay",
        {"steps": 20000, "use_gcn": False, "lr": 0.1, "radius": 0.4},
        True,
        True,
        True,
    ),
    _variant(
        "classic_sgd2_multi_default",
        "classic_sgd2_multi",
        "(SGD)^2 (default)",
        {
            "criteria": {"stress": 1.0, "ideal_edge_length": 1.0},
            "lr": 0.01,
            "steps": 2000,
            "grad_clamp": 5.0,
        },
        "sgd2_multi_ref",
        {
            "criteria_weights": {"stress": 1.0, "ideal_edge_length": 1.0},
            "max_iter": 2000,
            "grad_clamp": 5.0,
            "optimizer_kwargs": {"lr": 0.01},
        },
        True,
        True,
        False,
    ),
    _variant(
        "classic_sgd2_multi_stress_only",
        "classic_sgd2_multi",
        "(SGD)^2 (stress only)",
        {"criteria": {"stress": 1.0}, "lr": 0.01, "steps": 2000, "grad_clamp": 5.0},
        "sgd2_multi_ref",
        {
            "criteria_weights": {"stress": 1.0},
            "max_iter": 2000,
            "grad_clamp": 5.0,
            "optimizer_kwargs": {"lr": 0.01},
        },
        True,
        True,
        False,
    ),
    _variant(
        "classic_sgd2_multi_with_crossing",
        "classic_sgd2_multi",
        "(SGD)^2 (with crossings)",
        {
            "criteria": {"stress": 1.0, "crossings": 0.5},
            "lr": 0.01,
            "steps": 2000,
            "grad_clamp": 5.0,
        },
        "sgd2_multi_ref",
        {
            "criteria_weights": {"stress": 1.0, "crossings": 0.5},
            "max_iter": 2000,
            "grad_clamp": 5.0,
            "optimizer_kwargs": {"lr": 0.01},
        },
        True,
        True,
        False,
    ),
    _variant(
        "classic_sgd2_multi_with_aspect",
        "classic_sgd2_multi",
        "(SGD)^2 (with aspect)",
        {
            "criteria": {"stress": 0.8, "aspect_ratio": 0.2},
            "lr": 0.01,
            "steps": 2000,
            "grad_clamp": 5.0,
        },
        "sgd2_multi_ref",
        {
            "criteria_weights": {"stress": 0.8, "aspect_ratio": 0.2},
            "max_iter": 2000,
            "grad_clamp": 5.0,
            "optimizer_kwargs": {"lr": 0.01},
        },
        True,
        True,
        False,
    ),
    _variant(
        "classic_sgd2_multi_lr001",
        "classic_sgd2_multi",
        "(SGD)^2 (lr=0.001)",
        {
            "criteria": {"stress": 1.0, "ideal_edge_length": 1.0},
            "lr": 0.001,
            "steps": 2000,
            "grad_clamp": 5.0,
        },
        "sgd2_multi_ref",
        {
            "criteria_weights": {"stress": 1.0, "ideal_edge_length": 1.0},
            "max_iter": 2000,
            "grad_clamp": 5.0,
            "optimizer_kwargs": {"lr": 0.001},
        },
        True,
        True,
        False,
    ),
    _variant(
        "classic_sgd2_multi_lr01",
        "classic_sgd2_multi",
        "(SGD)^2 (lr=0.1)",
        {
            "criteria": {"stress": 1.0, "ideal_edge_length": 1.0},
            "lr": 0.1,
            "steps": 2000,
            "grad_clamp": 5.0,
        },
        "sgd2_multi_ref",
        {
            "criteria_weights": {"stress": 1.0, "ideal_edge_length": 1.0},
            "max_iter": 2000,
            "grad_clamp": 5.0,
            "optimizer_kwargs": {"lr": 0.1},
        },
        True,
        True,
        False,
    ),
    _variant(
        "classic_sgd2_multi_batch8",
        "classic_sgd2_multi",
        "(SGD)^2 (batch=8)",
        {
            "criteria": {"stress": 1.0, "ideal_edge_length": 1.0},
            "lr": 0.01,
            "steps": 2000,
            "grad_clamp": 5.0,
            "batch_size": 8,
        },
        "sgd2_multi_ref",
        {
            "criteria_weights": {"stress": 1.0, "ideal_edge_length": 1.0},
            "max_iter": 2000,
            "grad_clamp": 5.0,
            "optimizer_kwargs": {"lr": 0.01},
            "sample_sizes": {"stress": 8, "ideal_edge_length": 8},
        },
        True,
        True,
        False,
    ),
    _variant(
        "classic_sgd2_multi_batch128",
        "classic_sgd2_multi",
        "(SGD)^2 (batch=128)",
        {
            "criteria": {"stress": 1.0, "ideal_edge_length": 1.0},
            "lr": 0.01,
            "steps": 2000,
            "grad_clamp": 5.0,
            "batch_size": 128,
        },
        "sgd2_multi_ref",
        {
            "criteria_weights": {"stress": 1.0, "ideal_edge_length": 1.0},
            "max_iter": 2000,
            "grad_clamp": 5.0,
            "optimizer_kwargs": {"lr": 0.01},
            "sample_sizes": {"stress": 128, "ideal_edge_length": 128},
        },
        True,
        True,
        False,
    ),
    _variant(
        "cytoscape_fcose_default",
        "cytoscape_fcose",
        "fcose default",
        {},
        None,
        {},
        False,
        True,
        False,
    ),
    _variant(
        "cytoscape_fcose_quality",
        "cytoscape_fcose",
        "fcose quality=proof",
        {"quality": "proof"},
        None,
        {},
        False,
        True,
        True,
    ),
    _variant(
        "gephi_yifanhu_default",
        "gephi_yifanhu",
        "YifanHu default",
        {},
        None,
        {},
        False,
        True,
        False,
    ),
]

_VARIANT_BY_ID = {variant.variant_id: variant for variant in VARIANT_REGISTRY}
_VARIANTS_BY_BASE_ENGINE: dict[str, list[AlgorithmVariant]] = {}
for _variant_entry in VARIANT_REGISTRY:
    _VARIANTS_BY_BASE_ENGINE.setdefault(_variant_entry.base_engine, []).append(_variant_entry)

_VARIANT_BY_ORIGINAL_NAME = {
    original_name: variant
    for variant in VARIANT_REGISTRY
    for original_name in [original_variant_name(variant)]
    if original_name is not None
}

_BASE_ENGINE_STOCHASTICITY: dict[str, bool] = {
    "classic_fr": True,
    "classic_fr_kk": True,
    "classic_kk": False,
    "classic_kk_fr": True,
    "classic_fa2": True,
    "classic_stress_sgd": True,
    "classic_spectral": False,
    "classic_classical_mds": False,
    "classic_stress_maj": True,
    "classic_sugiyama": True,
    "classic_tsnet": True,
    "classic_gem": True,
    "classic_maxent_stress": True,
    "classic_pivot_mds": True,
    "classic_rt": False,
    "classic_davidson_harel": True,
    "classic_linlog": True,
    "classic_fmmm": True,
    "classic_graphopt": True,
    "classic_drl": True,
    "classic_lgl": True,
    "classic_sfdp": True,
    "classic_umap": True,
    "classic_neulay": True,
    "classic_sgd2_multi": True,
    "nx_spring": True,
    "nx_kamada_kawai": False,
    "nx_spectral": False,
    "igraph_sugiyama": False,
    "igraph_davidson_harel": True,
    "igraph_graphopt": True,
    "igraph_drl": True,
    "igraph_lgl": True,
    "graphviz_dot": False,
    "graphviz_sfdp": False,
    "ogdf_gem": False,
    "ogdf_fmmm": False,
    "ogdf_stress": False,
    "ogdf_pivot_mds": False,
    "fa2_ref": True,
    "sgd2": True,
    "tsne_graph": True,
    "umap_graph": True,
    "neulay": True,
    "sgd2_multi_ref": True,
    "cytoscape_fcose": True,
    "gephi_yifanhu": True,
}

_BASE_ENGINE_HEAVY: dict[str, bool] = {
    "classic_stress_sgd": True,
    "classic_spectral": False,
    "classic_classical_mds": True,
    "classic_stress_maj": True,
    "classic_rt": False,
    "classic_tsnet": True,
    "classic_umap": True,
    "classic_neulay": True,
    "sgd2": True,
    "tsne_graph": True,
    "umap_graph": True,
    "neulay": True,
}

_BASE_TIMEOUT_CAPS: dict[str, int] = {
    "classic_davidson_harel": 60,
    "classic_stress_sgd": 90,
}
